#pragma once

// Spalart-Allmaras one-equation model with the negative-SA extension (SA-neg).
//
// References:
//  [SA94] P. R. Spalart, S. R. Allmaras, "A one-equation turbulence model for
//         aerodynamic flows", La Recherche Aerospatiale 1 (1994) 5-21.
//  [AJS12] S. R. Allmaras, F. T. Johnson, P. R. Spalart, "Modifications and
//         clarifications for the implementation of the Spalart-Allmaras
//         turbulence model", ICCFD7-1902 (2012): negative SA, S-tilde limiter.
//  [TMR]  NASA Langley Turbulence Modeling Resource, SA and SA-neg pages,
//         https://turbmodels.larc.nasa.gov
//
// nu_tilde >= 0 (standard SA, trip term f_t1 omitted):
//   D nu~/Dt = cb1 (1 - ft2) S~ nu~ - (cw1 fw - cb1 ft2 / kappa^2) (nu~/d)^2
//              + (1/sigma) [div((nu + nu~) grad nu~) + cb2 |grad nu~|^2]
//   nu_t = nu~ fv1,  chi = nu~/nu,  fv1 = chi^3 / (chi^3 + cv1^3)
//   S~ = Omega + S_bar               if S_bar >= -cv2 Omega
//      = Omega + Omega (cv2^2 Omega + cv3 S_bar) / ((cv3 - 2 cv2) Omega - S_bar)   otherwise
//   S_bar = nu~ fv2 / (kappa^2 d^2),  fv2 = 1 - chi / (1 + chi fv1)
//   r = min(nu~ / (S~ kappa^2 d^2), 10),  g = r + cw2 (r^6 - r),
//   fw = g [(1 + cw3^6) / (g^6 + cw3^6)]^(1/6),  ft2 = ct3 exp(-ct4 chi^2)
//
// nu_tilde < 0 (negative SA):
//   D nu~/Dt = cb1 (1 - ct3) Omega nu~ + cw1 (nu~/d)^2
//              + (1/sigma) [div((nu + fn nu~) grad nu~) + cb2 |grad nu~|^2]
//   fn = (cn1 + chi^3) / (cn1 - chi^3),  nu_t = 0
//
// Discretisation: CBS transport (ScalarTransport) with the element mean of
// the nodal diffusivity (nu + fn nu~)/sigma; sources evaluated at the nodes
// from the recovered nodal vorticity and the recovered nodal gradient of
// nu~ (|grad nu~|^2 >= 0 by construction); point-implicit update (decay terms implicit, growth terms
// explicit, see PointImplicit.hpp). Boundary conditions: nu~ = 0 at walls,
// nu~ = ratio * nu at inlets, natural elsewhere.

#include "numerics/PointImplicit.hpp"
#include "turbulence/TransportModelBase.hpp"
#include "turbulence/TurbulenceConfig.hpp"

#include <array>
#include <vector>

namespace hydrama {

namespace sa {

struct Constants {
    static constexpr double cb1 = 0.1355;
    static constexpr double sigma = 2.0 / 3.0;
    static constexpr double cb2 = 0.622;
    static constexpr double kappa = 0.41;
    static constexpr double cw1 = cb1 / (kappa * kappa) + (1.0 + cb2) / sigma;
    static constexpr double cw2 = 0.3;
    static constexpr double cw3 = 2.0;
    static constexpr double cv1 = 7.1;
    static constexpr double cv2 = 0.7;
    static constexpr double cv3 = 0.9;
    static constexpr double ct3 = 1.2;
    static constexpr double ct4 = 0.5;
    static constexpr double cn1 = 16.0;
    static constexpr double r_max = 10.0;
};

double fv1(double chi) noexcept;
double fv2(double chi, double fv1_value) noexcept;
double fw(double r) noexcept;
double ft2(double chi) noexcept;
double fn(double chi) noexcept;

// Modified vorticity S~ with the [AJS12] limiter (S~ > 0 for Omega > 0).
double s_tilde(double omega, double s_bar) noexcept;

// nu_t = nu~ fv1 for nu~ >= 0, 0 otherwise.
double eddy_viscosity(double nu_tilde, double nu) noexcept;

// Diffusion coefficient (nu + fn nu~) / sigma (fn = 1 for nu~ >= 0); always > 0.
double diffusivity(double nu_tilde, double nu) noexcept;

// nu~/nu such that nu_t/nu = viscosity_ratio (inverse of chi fv1(chi)).
double nu_tilde_ratio_from_viscosity_ratio(double viscosity_ratio);

struct PointState {
    double nu_tilde = 0.0;
    double nu = 0.0;
    double d = 0.0;       // wall distance (may be +inf)
    double omega = 0.0;   // vorticity magnitude
    double grad_sq = 0.0; // |grad nu~|^2
};

// Source terms per unit volume at one point:
//   D nu~/Dt|_source = production - destruction + cross_diffusion,
// and their linearisation Q = P - D nu~ for the point-implicit update.
struct SourceTerms {
    double production = 0.0;      // cb1 (1 - ft2) S~ nu~   | cb1 (1 - ct3) Omega nu~
    double destruction = 0.0;     // (cw1 fw - cb1 ft2/kappa^2) (nu~/d)^2 | -cw1 (nu~/d)^2
    double cross_diffusion = 0.0; // (cb2 / sigma) |grad nu~|^2
    SourceSplit split;
    double total() const noexcept { return production - destruction + cross_diffusion; }
};

// d <= 0 (wall nodes, where nu~ is prescribed) gives zero sources.
SourceTerms source(const PointState& state, bool use_ft2) noexcept;

} // namespace sa

class SpalartAllmaras final : public TransportModelBase<SpalartAllmaras, 1> {
public:
    static constexpr std::array<const char*, 1> kNames = {"nu_tilde"};
    static constexpr std::array<bool, 1> kPositive = {false}; // SA-neg: negative values allowed

    explicit SpalartAllmaras(TurbulenceConfig config);

    std::string_view name() const override { return "spalart-allmaras"; }

    const std::vector<double>& nu_tilde() const { return phi_[0]; }

    // nu~ / nu at inlet flag `flag`: nu_tilde_ratio if given, else converted
    // from viscosity_ratio, else 3 (TMR free-stream recommendation).
    double inlet_nu_tilde_ratio(int flag) const;
    // Initial / free-stream value nu~ = ratio * nu (global inlet settings).
    double freestream_nu_tilde(double nu) const;

private:
    friend class TransportModelBase<SpalartAllmaras, 1>;

    TurbulenceConfig config_;
    std::vector<std::array<double, 2>> grad_node_;
    std::vector<double> diffusivity_node_;

    void initialize_fields(const TurbulenceContext& ctx);
    void compute_sources(const TurbulenceContext& ctx);
    void compute_eddy_viscosity(const TurbulenceContext& ctx);
    void compute_diffusivity(const TurbulenceContext& ctx);
    GradientRecovery recovery_method() const { return config_.gradient_recovery; }
};

} // namespace hydrama
