#pragma once

// Template-Method skeleton (CRTP) shared by all transport-equation
// turbulence models with N transported scalars (SA: 1, k-epsilon / SST: 2).
//
// advance():
//   1. velocity kinematics (element gradients, S, Omega; nodal recovery)
//   2. Derived::compute_sources(ctx)        nodal source splits P, D from phi^n
//   3. fused CBS transport of all N scalars (one element pass) with the
//      element diffusivities of phi^n
//   4. point-implicit nodal update (Patankar positivity per scalar if
//      Derived::kPositive[n])
//   5. Dirichlet boundary conditions
//   6. derived fields of phi^{n+1}: Derived::compute_eddy_viscosity(ctx)
//      (nodal nu_t), element means of nu_t, Derived::compute_diffusivity(ctx)
//      and the per-element maximum diffusivity (time step limit)
// Every quantity is computed exactly once per step.
//
// Derived must provide:
//   static constexpr std::array<const char*, N> kNames;   variable names
//   static constexpr std::array<bool, N> kPositive;       positivity enforcement
//   void initialize_fields(const TurbulenceContext&);      phi_, bcs_
//   void compute_sources(const TurbulenceContext&);        production_, destruction_ and, for
//                                                           positive variables, quadratic_ (uses kinematics_)
//   void compute_eddy_viscosity(const TurbulenceContext&); nut_ (nodal)
//   void compute_diffusivity(const TurbulenceContext&);    diffusivity_elem_ (may use nut_)
//   GradientRecovery recovery_method() const;              nodal gradient recovery
// and may shadow the optional hooks (no-ops here):
//   void apply_dynamic_bcs(const TurbulenceContext&);       after the Dirichlet sets
//                                                           (state-dependent BCs, e.g.
//                                                           wall-function epsilon)
//   void adjust_element_viscosity(const TurbulenceContext&); after the element means
//                                                           of nu_t (wall functions)
// The hooks are called statically (inlined), never through virtual dispatch.

#include "geometry/FieldOps.hpp"
#include "numerics/DirichletSet.hpp"
#include "numerics/PointImplicit.hpp"
#include "numerics/ScalarTransport.hpp"
#include "turbulence/TurbulenceModel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace hydrama {

template <class Derived, int N>
class TransportModelBase : public TurbulenceModel {
public:
    static constexpr int kNumEquations = N;

    void initialize(const TurbulenceContext& ctx, const FlowView& flow) final {
        const int npoin = ctx.mesh.npoin;
        const int nelem = ctx.mesh.nelem;
        nodal_mass_ = lumped_nodal_mass(ctx.mesh);
        recovery_ = NodalGradient(ctx.mesh, self().recovery_method()); // static mesh: stencils once
        for (int n = 0; n < N; ++n) {
            phi_[n].assign(npoin, 0.0);
            phi_old_[n].assign(npoin, 0.0);
            rhs_[n].assign(npoin, 0.0);
            production_[n].assign(npoin, 0.0);
            destruction_[n].assign(npoin, 0.0);
            quadratic_[n].assign(npoin, 0.0);
            diffusivity_elem_[n].assign(nelem, 0.0);
            residual_[n] = 0.0;
        }
        nut_.assign(npoin, 0.0);
        nut_elem_.assign(nelem, 0.0);
        max_diffusivity_elem_.assign(nelem, 0.0);

        self().initialize_fields(ctx);
        for (int n = 0; n < N; ++n) bcs_[n].apply(phi_[n]);
        self().apply_dynamic_bcs(ctx);
        kinematics_.compute(ctx.mesh, recovery_, flow.u, flow.v);
        update_derived_fields(ctx);
        initialized_ = true;
    }

    void advance(const TurbulenceContext& ctx, const FlowView& flow) final {
        if (!initialized_) throw std::logic_error(std::string(name()) + ": advance() before initialize()");

        kinematics_.compute(ctx.mesh, recovery_, flow.u, flow.v);
        self().compute_sources(ctx);

        std::array<TransportedScalar, N> scalars;
        for (int n = 0; n < N; ++n) scalars[n] = {&phi_[n], &diffusivity_elem_[n], &rhs_[n]};
        assemble_scalar_transport<N>(ctx.mesh, flow.u, flow.v, ctx.delte, scalars);

        for (int n = 0; n < N; ++n) {
            phi_old_[n] = phi_[n];
            if (Derived::kPositive[n]) {
                point_implicit_update(nodal_mass_, ctx.elcoe2, rhs_[n], production_[n], destruction_[n],
                                      quadratic_[n], true, phi_[n]);
            } else {
                point_implicit_update(nodal_mass_, ctx.elcoe2, rhs_[n], production_[n], destruction_[n], false,
                                      phi_[n]);
            }
            bcs_[n].apply(phi_[n]);
        }
        self().apply_dynamic_bcs(ctx);
        for (int n = 0; n < N; ++n) residual_[n] = relative_change(phi_old_[n], phi_[n]);
        update_derived_fields(ctx);
    }

    const std::vector<double>& nut() const override { return nut_; }
    const std::vector<double>& nut_elem() const override { return nut_elem_; }
    const std::vector<double>& max_diffusivity_elem() const override { return max_diffusivity_elem_; }

    std::vector<FieldRef> fields() override {
        std::vector<FieldRef> f;
        for (int n = 0; n < N; ++n) f.push_back({Derived::kNames[n], &phi_[n]});
        f.push_back({"nut", &nut_});
        return f;
    }

    std::vector<FieldRef> state_fields() override {
        std::vector<FieldRef> f;
        for (int n = 0; n < N; ++n) f.push_back({Derived::kNames[n], &phi_[n]});
        return f;
    }

    void refresh(const TurbulenceContext& ctx) override {
        if (!initialized_) throw std::logic_error(std::string(name()) + ": refresh() before initialize()");
        for (int n = 0; n < N; ++n) {
            if (static_cast<int>(phi_[n].size()) != ctx.mesh.npoin) {
                throw std::invalid_argument(std::string(name()) + ": state size does not match the mesh");
            }
        }
        update_derived_fields(ctx);
    }

    std::vector<EquationResidual> residuals() const override {
        std::vector<EquationResidual> r;
        for (int n = 0; n < N; ++n) r.push_back({Derived::kNames[n], residual_[n]});
        return r;
    }

    // Transported variable n (nodal).
    const std::vector<double>& field(int n) const { return phi_[n]; }

    // Replace variable n (e.g. from a restart file) and recompute nu_t.
    // Boundary conditions are NOT re-applied.
    void set_field(const TurbulenceContext& ctx, int n, const std::vector<double>& values) {
        if (!initialized_) throw std::logic_error(std::string(name()) + ": set_field() before initialize()");
        if (static_cast<int>(values.size()) != ctx.mesh.npoin) throw std::invalid_argument("set_field: size mismatch");
        phi_[n] = values;
        update_derived_fields(ctx);
    }

    const DirichletSet& boundary_conditions(int n) const { return bcs_[n]; }
    const FlowKinematics& kinematics() const { return kinematics_; }
    const NodalGradient& recovery() const { return recovery_; }
    const std::vector<double>& nodal_mass() const { return nodal_mass_; }
    const std::vector<double>& production(int n) const { return production_[n]; }
    const std::vector<double>& destruction(int n) const { return destruction_[n]; }
    const std::vector<double>& quadratic(int n) const { return quadratic_[n]; }
    const std::vector<double>& diffusivity_elem(int n) const { return diffusivity_elem_[n]; }

protected:
    std::array<std::vector<double>, N> phi_, phi_old_, rhs_, production_, destruction_, quadratic_, diffusivity_elem_;
    std::array<DirichletSet, N> bcs_;
    std::vector<double> nodal_mass_, nut_, nut_elem_, max_diffusivity_elem_;
    NodalGradient recovery_;
    FlowKinematics kinematics_;

    // Default (no-op) hooks, shadowed by models that need them.
    void apply_dynamic_bcs(const TurbulenceContext&) {}
    void adjust_element_viscosity(const TurbulenceContext&) {}

private:
    std::array<double, N> residual_{};
    bool initialized_ = false;

    Derived& self() { return static_cast<Derived&>(*this); }

    void update_derived_fields(const TurbulenceContext& ctx) {
        self().compute_eddy_viscosity(ctx);
        element_mean(ctx.mesh, nut_, nut_elem_);
        self().adjust_element_viscosity(ctx);
        self().compute_diffusivity(ctx);
        for (int ie = 0; ie < ctx.mesh.nelem; ++ie) {
            double g = 0.0;
            for (int n = 0; n < N; ++n) g = std::max(g, diffusivity_elem_[n][ie]);
            max_diffusivity_elem_[ie] = g;
        }
    }

    static double relative_change(const std::vector<double>& old, const std::vector<double>& now) {
        double diff = 0.0, norm = 0.0;
        for (std::size_t i = 0; i < now.size(); ++i) {
            const double d = now[i] - old[i];
            diff += d * d;
            norm += now[i] * now[i];
        }
        return norm > 0.0 ? std::sqrt(diff / norm) : std::sqrt(diff);
    }
};

} // namespace hydrama
