#pragma once

// Langtry-Menter gamma-Re_theta (2009) 4-equation transition model coupled with
// Menter k-omega SST.
//
// Equations:
//   1. k: turbulent kinetic energy
//      Dk/Dt = P~_k^* - D_k^* + div((nu + sigma_k nu_t) grad k)
//      P~_k^* = gamma_eff P~_k
//      D_k^*  = min(max(gamma_eff, 0.1), 1.0) beta* k omega
//
//   2. w~ = ln(omega): log-transformed specific dissipation rate
//      Dw~/Dt = gamma S^2 e^-w~ - beta e^w~ + div(Gamma_w grad w~) + Gamma_w |grad w~|^2
//               + 2 (1 - F1) sigma_w2 grad k . grad w~ e^-w~
//      with modified blending function: F1_trans = max(F1, F3),
//      F3 = exp(-(R_y / 120)^8), R_y = d sqrt(k) / nu.
//
//   3. gamma: intermittency (transported scalar in [0, 2])
//      Dgamma/Dt = P_gamma - E_gamma + div((nu + nu_t / sigma_gamma) grad gamma)
//      P_gamma = F_length c_a1 S sqrt(gamma F_onset) (1 - c_e1 gamma)
//      E_gamma = c_a2 Omega gamma F_turb (c_e2 gamma - 1)
//      gamma_eff = max(gamma, gamma_sep)
//
//   4. Re_thetat_tilde: transition momentum thickness Reynolds number
//      D(Re_thetat_tilde)/Dt = P_thetat + div(sigma_thetat (nu + nu_t) grad Re_thetat_tilde)
//      P_thetat = c_thetat (1 / t_scale) (Re_thetat - Re_thetat_tilde) (1 - F_thetat)
//
// References:
//   R. B. Langtry, F. R. Menter, "Correlation-Based Transition Modeling for
//   Unstructured Parallel Codes", AIAA Journal 47(12) (2009) 2894-2906.
//   NASA Langley Turbulence Modeling Resource: Langtry-Menter Transition Model.

#include "turbulence/KOmegaSST.hpp"
#include "turbulence/TransportModelBase.hpp"
#include "turbulence/TurbulenceBC.hpp"
#include "turbulence/TurbulenceConfig.hpp"

#include <array>
#include <vector>

namespace hydrama {

namespace transition {

struct Constants {
    // SST constants
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

    // Intermittency gamma constants
    static constexpr double ca1 = 0.5;
    static constexpr double ce1 = 1.0;
    static constexpr double ca2 = 0.03;
    static constexpr double ce2 = 1.0;
    static constexpr double sigma_gamma = 1.0;

    // Transition Re_thetat constants
    static constexpr double c_thetat = 0.03;
    static constexpr double sigma_thetat = 2.0;
};

// Empirical transition correlations (Langtry-Menter 2009)
double re_theta_t(double tu) noexcept;
double re_theta_c(double re_thetat_tilde) noexcept;
double f_length(double re_thetat_tilde) noexcept;

// Blending and transition functions
double f_onset1(double rev, double re_thetac) noexcept;
double f_onset2(double f_onset1_val) noexcept;
double f_onset3(double rt) noexcept;
double f_onset(double rev, double re_thetac, double rt) noexcept;
double f_turb(double rt) noexcept;
double f_reattach(double rt) noexcept;
double gamma_sep(double rev, double re_thetac, double rt) noexcept;
double f3(double d, double k, double nu) noexcept;

struct PointState {
    double k = 0.0;
    double omega = 0.0;
    double gamma = 0.0;
    double re_thetat = 0.0;
    double nu = 0.0;
    double nut = 0.0;
    double strain = 0.0;
    double vorticity = 0.0;
    double d = 0.0;
    double u_mag = 0.0;
    double f1 = 0.0;
    double grad_k_dot_grad_log_omega = 0.0;
    double grad_log_omega_sq = 0.0;
    double diffusivity_omega = 0.0;
};

struct SourceTerms {
    double p_k_eff = 0.0;
    double d_k_eff = 0.0;
    double p_gamma = 0.0;
    double e_gamma = 0.0;
    double p_thetat = 0.0;
    double gamma_eff = 0.0;
    double f_thetat = 0.0;
    double log_omega_source = 0.0;

    SourceSplit k;
    SourceSplit log_omega;
    SourceSplit gamma;
    SourceSplit re_thetat;
};

SourceTerms source(const PointState& s, double production_limiter) noexcept;

} // namespace transition

class GammaReThetaSST final : public TransportModelBase<GammaReThetaSST, 4> {
public:
    // Transported: k, w~ = ln(omega), gamma, and Re_thetat_tilde
    static constexpr std::array<const char*, 4> kNames = {"k", "log_omega", "gamma", "re_thetat"};
    static constexpr std::array<bool, 4> kPositive = {true, false, true, true};

    explicit GammaReThetaSST(TurbulenceConfig config);

    std::string_view name() const override { return "gamma-re-theta-sst"; }

    // Output fields: k, omega, nut, gamma, re_thetat, gamma_eff, sst_f1
    std::vector<FieldRef> fields() override;

    const std::vector<double>& k() const { return phi_[0]; }
    const std::vector<double>& log_omega() const { return phi_[1]; }
    const std::vector<double>& gamma() const { return phi_[2]; }
    const std::vector<double>& re_thetat() const { return phi_[3]; }
    const std::vector<double>& omega() const { return omega_; }
    const std::vector<double>& gamma_eff() const { return gamma_eff_; }
    const std::vector<double>& f1() const { return f1_; }
    const std::vector<double>& f2() const { return f2_; }

    TwoEquationInlet inlet_values(int flag, double nu) const;
    TwoEquationInlet freestream_values(double nu) const;
    double inlet_re_thetat(double tu) const;

private:
    friend class TransportModelBase<GammaReThetaSST, 4>;

    TurbulenceConfig config_;
    std::vector<std::array<double, 2>> grad_k_, grad_lw_;
    std::vector<double> omega_, gamma_eff_, f1_, f2_;
    std::vector<double> diffusivity_node_k_, diffusivity_node_w_;

    void initialize_fields(const TurbulenceContext& ctx);
    void compute_sources(const TurbulenceContext& ctx);
    void compute_eddy_viscosity(const TurbulenceContext& ctx);
    void compute_diffusivity(const TurbulenceContext& ctx);
    GradientRecovery recovery_method() const { return config_.gradient_recovery; }
};

} // namespace hydrama
