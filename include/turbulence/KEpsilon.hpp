#pragma once

// k-epsilon family: KEpsilon<Variant, Damping> (policy-based design).
//
//   Dk/Dt  = P_k - eps - D + div((nu + nu_t/sigma_k) grad k)
//   De/Dt  = C1 f1 (e/k) P_k - C2 f2 e^2/k + E + div((nu + nu_t/sigma_e) grad e)
//   nu_t   = C_mu f_mu k^2 / e,   P_k = min(nu_t S^2, c_lim eps)   (c_lim = 0: no limiter)
//
// Variant (constants; RNG / Realizable are further policies):
//   Standard (Launder & Spalding 1974): C_mu 0.09, C1 1.44, C2 1.92, sigma_k 1.0, sigma_e 1.3
//
// Damping:
//   NoDamping (high-Re, "k-epsilon"): f_mu = f1 = f2 = 1, D = E = 0, e = eps;
//     standard log-law wall functions (WallFunctions.hpp): k has a natural
//     (zero-flux) wall condition; eps at the wall nodes = C_mu^(3/4) k^(3/2) / (kappa y_p);
//     nu_t of the wall-adjacent elements from the log law. The production at
//     every node of an element touching the wall (the wall-function layer;
//     wall nodes use their own faces, the other layer nodes those of the wall
//     nodes they share an element with) is the log-law value
//     P = tau_w u_tau / (kappa y), y = y_p at wall nodes and the wall distance
//     otherwise: the linear no-slip velocity in these
//     elements does not represent the log profile (its gradient overestimates
//     dU/dy by ~ln(E y+)), exactly as in finite-volume wall functions, where
//     the production of the wall-adjacent cells is replaced.
//   LaunderSharma (low-Re, "launder-sharma-k-epsilon", Launder & Sharma 1974):
//     transported e = eps~ (eps = eps~ + D),
//     f_mu = exp(-3.4 / (1 + R_t/50)^2),  f1 = 1,  f2 = 1 - 0.3 exp(-R_t^2),
//     R_t = k^2 / (nu eps~),  D = 2 nu |grad sqrt(k)|^2,
//     E = 2 nu nu_t sum_ijk (d^2 u_i / dx_j dx_k)^2;  walls: k = 0, eps~ = 0.
//     Nodal gradients / second derivatives by recovery (PPR applied twice).
//
// Source limiters / robustness: the production limiter P_k <= c_lim eps
// (production_limiter, default 10); positivity-preserving point-implicit
// update of k and e (Patankar, PointImplicit.hpp); all ratios e/k, k^2/e are
// evaluated with tiny positive floors; nu_t capped at nut_ratio_max * nu.

#include "turbulence/TransportModelBase.hpp"
#include "turbulence/TurbulenceBC.hpp"
#include "turbulence/TurbulenceConfig.hpp"
#include "turbulence/WallFunctions.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace hydrama {

namespace ke {

inline constexpr double kTiny = 1.0e-30;

struct Standard {
    static constexpr double c_mu = 0.09;
    static constexpr double c1 = 1.44;
    static constexpr double c2 = 1.92;
    static constexpr double sigma_k = 1.0;
    static constexpr double sigma_eps = 1.3;
};

struct NoDamping {
    static constexpr bool kLowRe = false;
    static constexpr const char* kModelName = "k-epsilon";
    static constexpr const char* kEpsilonName = "epsilon";
    static double f_mu(double) noexcept { return 1.0; }
    static double f2(double) noexcept { return 1.0; }
};

struct LaunderSharma {
    static constexpr bool kLowRe = true;
    static constexpr const char* kModelName = "launder-sharma-k-epsilon";
    static constexpr const char* kEpsilonName = "epsilon_tilde";
    static double f_mu(double rt) noexcept {
        const double a = 1.0 + rt / 50.0;
        return std::exp(-3.4 / (a * a));
    }
    static double f2(double rt) noexcept { return 1.0 - 0.3 * std::exp(-rt * rt); }
};

// Turbulence Reynolds number k^2 / (nu e) (large for e -> 0 with k > 0, 0 for k = 0).
double turbulence_reynolds(double k, double e, double nu) noexcept;

// nu_t = C_mu f_mu(R_t) k^2 / e  (0 for k <= 0).
template <class Variant, class Damping>
double eddy_viscosity(double k, double e, double nu) noexcept {
    if (!(k > 0.0)) return 0.0;
    const double rt = turbulence_reynolds(k, e, nu);
    return Variant::c_mu * Damping::f_mu(rt) * k * k / std::max(e, kTiny);
}

struct PointState {
    double k = 0.0;
    double e = 0.0;      // eps (NoDamping) or eps~ (LaunderSharma)
    double nu = 0.0;
    double nut = 0.0;    // nu_t of the current state
    double strain = 0.0; // S = sqrt(2 S_ij S_ij)
    double D = 0.0;      // 2 nu |grad sqrt(k)|^2 (low-Re only)
    double E = 0.0;      // 2 nu nu_t |grad grad u|^2 (low-Re only)
};

struct SourceTerms {
    double production = 0.0;        // P_k after the limiter
    double dissipation = 0.0;       // eps = e + D (sink of k)
    double eps_production = 0.0;    // C1 f1 (e/k) P_k + E
    double eps_destruction = 0.0;   // C2 f2 e^2 / k
    bool limited = false;           // production limiter active
    SourceSplit k, e;               // linearisations for the point-implicit update
};

// production_limiter c_lim > 0: P_k <= c_lim eps. production_override >= 0
// replaces nu_t S^2 (wall-function production at the wall nodes).
template <class Variant, class Damping>
SourceTerms source(const PointState& s, double production_limiter, double production_override = -1.0) noexcept {
    SourceTerms t;
    const double k = std::max(s.k, 0.0);
    const double e = std::max(s.e, 0.0);
    const double eps = e + s.D;
    double p = production_override >= 0.0 ? production_override : s.nut * s.strain * s.strain;
    if (production_limiter > 0.0 && p > production_limiter * eps) {
        p = production_limiter * eps;
        t.limited = true;
    }
    const double rt = turbulence_reynolds(k, e, s.nu);
    const double inv_k = 1.0 / std::max(k, kTiny);
    t.production = p;
    t.dissipation = eps;
    t.eps_production = Variant::c1 * e * inv_k * p + s.E;
    t.eps_destruction = Variant::c2 * Damping::f2(rt) * e * e * inv_k;

    t.k.add_value(p, k);
    t.k.add_value(-eps, k);
    // eps: production C1 (e/k) P_k = C1 C_mu f_mu k S^2 does not depend on e
    // for the lagged nu_t ~ 1/e (explicit constant); the Riccati sink
    // C2 f2 e^2 / k is solved exactly by the backward-Euler update.
    t.e.add_value(t.eps_production, e);
    t.e.add_quadratic_sink(Variant::c2 * Damping::f2(rt) * inv_k);
    return t;
}

} // namespace ke

template <class Variant, class Damping>
class KEpsilon final : public TransportModelBase<KEpsilon<Variant, Damping>, 2> {
    using Base = TransportModelBase<KEpsilon<Variant, Damping>, 2>;

public:
    static constexpr std::array<const char*, 2> kNames = {"k", Damping::kEpsilonName};
    static constexpr std::array<bool, 2> kPositive = {true, true};

    explicit KEpsilon(TurbulenceConfig config);

    std::string_view name() const override { return Damping::kModelName; }

    const std::vector<double>& k() const { return this->phi_[0]; }
    const std::vector<double>& epsilon() const { return this->phi_[1]; } // eps~ for Launder-Sharma

    // Inlet / free-stream values (k, eps, omega) for a flag / the global settings.
    TwoEquationInlet inlet_values(int flag, double nu) const;
    TwoEquationInlet freestream_values(double nu) const;

    // Low-Re terms of the last source evaluation (nodal).
    const std::vector<double>& extra_dissipation() const { return D_; }
    const std::vector<double>& extra_production() const { return E_; }
    double y_plus_laminar() const { return y_plus_lam_; }

private:
    friend Base;

    TurbulenceConfig config_;
    wallfn::LogLaw law_;
    double y_plus_lam_ = 11.53;
    std::vector<double> sqrt_k_, D_, E_, diffusivity_node_k_, diffusivity_node_e_;
    std::vector<std::array<double, 2>> grad_;
    std::array<std::vector<std::array<double, 2>>, 4> second_; // gradients of du/dx, du/dy, dv/dx, dv/dy
    // Wall functions: distinct wall-adjacent elements, their wall-face count,
    // the element index of every wall face and an accumulation buffer.
    std::vector<int> wall_elems_, wall_elem_faces_, face_wall_elem_;
    std::vector<double> wall_elem_sum_;
    // Wall-function layer: nodes of elements touching the wall, CSR map to
    // the wall faces of the wall nodes in these elements, and their log-law height.
    std::vector<int> layer_nodes_, layer_offsets_, layer_faces_;
    std::vector<double> layer_height_;

    void initialize_fields(const TurbulenceContext& ctx);
    void compute_sources(const TurbulenceContext& ctx);
    void compute_eddy_viscosity(const TurbulenceContext& ctx);
    void compute_diffusivity(const TurbulenceContext& ctx);
    void apply_dynamic_bcs(const TurbulenceContext& ctx);
    void adjust_element_viscosity(const TurbulenceContext& ctx);
    GradientRecovery recovery_method() const { return config_.gradient_recovery; }
};

using StandardKEpsilon = KEpsilon<ke::Standard, ke::NoDamping>;
using LaunderSharmaKEpsilon = KEpsilon<ke::Standard, ke::LaunderSharma>;

extern template class KEpsilon<ke::Standard, ke::NoDamping>;
extern template class KEpsilon<ke::Standard, ke::LaunderSharma>;

} // namespace hydrama
