#pragma once

// Menter k-omega SST model (2003 version).
//
//   Dk/Dt = P~_k - beta* k omega + div((nu + sigma_k nu_t) grad k)
//   Dw/Dt = gamma S^2 - beta omega^2 + div((nu + sigma_w nu_t) grad omega)
//           + 2 (1 - F1) sigma_w2 (1/omega) grad k . grad omega          (cross diffusion)
//   nu_t  = a1 k / max(a1 omega, S F2)                                   (shear stress limiter)
//   P~_k  = min(nu_t S^2, c_lim beta* k omega)                           (production limiter, c_lim = 10)
//
//   F1 = tanh(arg1^4), arg1 = min(max(sqrt(k)/(beta* omega d), 500 nu/(d^2 omega)),
//                                 4 sigma_w2 k / (CD_kw d^2))
//   CD_kw = max(2 sigma_w2 (1/omega) grad k . grad omega, 1e-10)
//   F2 = tanh(arg2^2), arg2 = max(2 sqrt(k)/(beta* omega d), 500 nu/(d^2 omega))
//   phi = F1 phi_1 + (1 - F1) phi_2 for sigma_k, sigma_w, beta, gamma.
//
// Constants: beta* = 0.09, a1 = 0.31; set 1 (k-omega): sigma_k1 = 0.85,
// sigma_w1 = 0.5, beta1 = 0.075, gamma1 = 5/9; set 2 (k-epsilon):
// sigma_k2 = 1.0, sigma_w2 = 0.856, beta2 = 0.0828, gamma2 = 0.44.
// S is the strain-rate magnitude sqrt(2 S_ij S_ij) (2003 version).
//
// Boundary conditions (low-Re, resolved wall): k = 0 and
// omega = 60 nu / (beta1 y_1^2) at the walls (y_1: first-cell height);
// inlets from intensity / length scale / viscosity ratio / direct values.
//
// Logarithmic omega: the transported variable is w~ = ln(omega). Near walls
// omega ~ 1/y^2 varies by orders of magnitude between node layers; the
// explicit Galerkin/CBS convection of such an under-resolved profile drove
// omega to zero next to the wall (observed on the benchmark), after which
// nu_t ~ 1/omega and the cross-diffusion ~ 1/omega blew up. ln(omega) varies
// mildly (~ -2 ln y), omega = exp(w~) > 0 by construction, and the
// production becomes a restoring term ~ 1/omega. The transformed equation
// (identical in the continuum):
//   Dw~/Dt = gamma S^2 e^-w~ - beta e^w~ + div(Gamma_w grad w~) + Gamma_w |grad w~|^2
//            + 2 (1 - F1) sigma_w2 grad k . grad w~ e^-w~
// (grad omega / omega = grad w~). The exponential terms are Newton-
// linearised (SourceSplit::add_newton). References: D. Ilinca, D. Pelletier,
// AIAA J. 36(1) (1998) 44-50 (logarithmic turbulence variables in FE);
// F. Bassi, A. Crivellini, S. Rebay, M. Savini, Comput. Fluids 34 (2005)
// 507-540 (k - ln(omega) in DG).
//
// Discretisation: nodal sources with recovered gradients of k and ln(omega)
// (PPR) and the exact wall distance; F1 = 1 at walls (d = 0), F1 = F2 = 0
// without walls (d = inf); positivity-preserving point-implicit update of k.
//
// References: F. R. Menter, AIAA J. 32(8) (1994) 1598-1605; F. R. Menter,
// M. Kuntz, R. Langtry, Turbulence, Heat and Mass Transfer 4 (2003) 625-632;
// NASA Langley Turbulence Modeling Resource (SST pages).

#include "turbulence/TransportModelBase.hpp"
#include "turbulence/TurbulenceBC.hpp"
#include "turbulence/TurbulenceConfig.hpp"

#include <array>
#include <vector>

namespace hydrama {

namespace sst {

struct Constants {
    static constexpr double beta_star = 0.09;
    static constexpr double a1 = 0.31;
    static constexpr double sigma_k1 = 0.85;
    static constexpr double sigma_w1 = 0.5;
    static constexpr double beta1 = 0.075;
    static constexpr double gamma1 = 5.0 / 9.0;
    static constexpr double sigma_k2 = 1.0;
    static constexpr double sigma_w2 = 0.856;
    static constexpr double beta2 = 0.0828;
    static constexpr double gamma2 = 0.44;
    static constexpr double cd_min = 1.0e-10;
};

inline constexpr double kTiny = 1.0e-30;

inline double blend(double f1, double phi1, double phi2) noexcept { return f1 * phi1 + (1.0 - f1) * phi2; }

// CD_kw = max(2 sigma_w2 (1/omega) grad k . grad omega, 1e-10)
//       = max(2 sigma_w2 grad k . grad ln(omega), 1e-10).
double cross_diffusion(double grad_k_dot_grad_log_omega) noexcept;

// Blending functions; d <= 0 (wall) gives 1, d = inf (no walls) gives 0.
double f1(double k, double omega, double d, double nu, double cd_kw) noexcept;
double f2(double k, double omega, double d, double nu) noexcept;

// nu_t = a1 k / max(a1 omega, S F2).
double eddy_viscosity(double k, double omega, double strain, double f2) noexcept;

// Wall value omega = 60 nu / (beta1 y1^2).
double wall_omega(double nu, double y1) noexcept;

struct PointState {
    double k = 0.0;
    double omega = 0.0;                     // omega = exp(w~)
    double nu = 0.0;
    double nut = 0.0;                       // nu_t of the current state
    double strain = 0.0;                    // S
    double f1 = 0.0;                        // blending function F1
    double grad_k_dot_grad_log_omega = 0.0; // grad k . grad w~
    double grad_log_omega_sq = 0.0;         // |grad w~|^2
    double diffusivity_omega = 0.0;         // Gamma_w = nu + sigma_w nu_t
};

struct SourceTerms {
    // Terms of the k and omega equations (per unit volume).
    double production = 0.0;        // P~_k
    double k_destruction = 0.0;     // beta* k omega
    double omega_production = 0.0;  // gamma S^2
    double omega_destruction = 0.0; // beta omega^2
    double cross_diffusion = 0.0;   // 2 (1 - F1) sigma_w2 (1/omega) grad k . grad omega
    bool limited = false;           // production limiter active
    // Source of the w~ = ln(omega) equation:
    //   (omega_production - omega_destruction + cross_diffusion) / omega + Gamma_w |grad w~|^2
    double log_omega_source = 0.0;
    SourceSplit k, log_omega;       // linearisations (log_omega: Newton)
};

SourceTerms source(const PointState& s, double production_limiter) noexcept;

} // namespace sst

class KOmegaSST final : public TransportModelBase<KOmegaSST, 2> {
public:
    // Transported: k and w~ = ln(omega) (restart state); omega is derived.
    static constexpr std::array<const char*, 2> kNames = {"k", "log_omega"};
    static constexpr std::array<bool, 2> kPositive = {true, false};

    explicit KOmegaSST(TurbulenceConfig config);

    std::string_view name() const override { return "k-omega-sst"; }

    // Output fields: k, omega, nut and the blending function F1 ("sst_f1").
    std::vector<FieldRef> fields() override;

    const std::vector<double>& k() const { return phi_[0]; }
    const std::vector<double>& log_omega() const { return phi_[1]; }
    const std::vector<double>& omega() const { return omega_; }
    const std::vector<double>& f1() const { return f1_; }
    const std::vector<double>& f2() const { return f2_; }

    TwoEquationInlet inlet_values(int flag, double nu) const;
    TwoEquationInlet freestream_values(double nu) const;

private:
    friend class TransportModelBase<KOmegaSST, 2>;

    TurbulenceConfig config_;
    std::vector<std::array<double, 2>> grad_k_, grad_lw_;
    std::vector<double> omega_, f1_, f2_, diffusivity_node_k_, diffusivity_node_w_;

    void initialize_fields(const TurbulenceContext& ctx);
    void compute_sources(const TurbulenceContext& ctx);
    void compute_eddy_viscosity(const TurbulenceContext& ctx);
    void compute_diffusivity(const TurbulenceContext& ctx);
    GradientRecovery recovery_method() const { return config_.gradient_recovery; }
};

} // namespace hydrama
