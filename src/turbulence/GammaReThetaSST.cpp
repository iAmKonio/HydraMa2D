#include "turbulence/GammaReThetaSST.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace hydrama {

namespace transition {

using C = Constants;
inline constexpr double kTiny = 1.0e-30;

double re_theta_t(double tu) noexcept {
    // Local turbulence intensity Tu is in percent (e.g. 1.0 = 1%)
    const double tu_val = std::max(tu, 0.013);
    if (tu_val <= 1.3) {
        const double tu2 = tu_val * tu_val;
        return std::max(1173.51 - 22.63 * tu_val + 0.1471 / tu2, 20.0);
    }
    const double diff = std::max(tu_val - 0.5658, 1.0e-6);
    return std::max(331.5 * std::pow(diff, -0.671), 20.0);
}

double re_theta_c(double re_thetat_tilde) noexcept {
    const double rt = std::max(re_thetat_tilde, 1.0);
    if (rt <= 1870.0) {
        const double p = 396.035e-2 - 120.656e-4 * rt + 868.230e-6 * rt * rt
                         - 696.506e-9 * rt * rt * rt + 174.105e-12 * rt * rt * rt * rt;
        return std::max(rt - p, 1.0);
    }
    return std::max(rt - (593.11 + 0.482 * (rt - 1870.0)), 1.0);
}

double f_length(double re_thetat_tilde) noexcept {
    const double rt = std::max(re_thetat_tilde, 1.0);
    double fl = 0.3188;
    if (rt < 400.0) {
        fl = 398.189e-1 - 119.270e-4 * rt - 132.567e-6 * rt * rt;
    } else if (rt < 596.5) {
        fl = 263.404 - 123.939e-2 * rt + 194.548e-5 * rt * rt - 101.695e-8 * rt * rt * rt;
    } else if (rt < 1200.0) {
        fl = 0.5 - 3.0e-4 * (rt - 596.5);
    } else {
        fl = 0.3188;
    }
    return std::max(fl, 0.1);
}

double f_onset1(double rev, double re_thetac) noexcept {
    return rev / (2.193 * std::max(re_thetac, 1.0));
}

double f_onset2(double f_onset1_val) noexcept {
    const double f1 = std::max(f_onset1_val, 0.0);
    const double f1_4 = f1 * f1 * f1 * f1;
    return std::min(std::max(f1, f1_4), 2.0);
}

double f_onset3(double rt) noexcept {
    const double r = std::max(rt, 0.0);
    const double ratio = r / 2.5;
    return std::max(1.0 - ratio * ratio * ratio, 0.0);
}

double f_onset(double rev, double re_thetac, double rt) noexcept {
    const double fo1 = f_onset1(rev, re_thetac);
    const double fo2 = f_onset2(fo1);
    const double fo3 = f_onset3(rt);
    return std::max(fo2 - fo3, 0.0);
}

double f_turb(double rt) noexcept {
    const double r = std::max(rt, 0.0);
    const double r4 = (r / 4.0) * (r / 4.0) * (r / 4.0) * (r / 4.0);
    return std::exp(-std::min(r4, 50.0));
}

double f_reattach(double rt) noexcept {
    const double r = std::max(rt, 0.0);
    const double r4 = (r / 20.0) * (r / 20.0) * (r / 20.0) * (r / 20.0);
    return std::exp(-std::min(r4, 50.0));
}

double gamma_sep(double rev, double re_thetac, double rt) noexcept {
    const double ratio = rev / (3.235 * std::max(re_thetac, 1.0));
    const double diff = std::max(ratio - 1.0, 0.0);
    const double f_reatt = f_reattach(rt);
    return std::min(2.0 * diff * f_reatt, 2.0);
}

double f3(double d, double k, double nu) noexcept {
    if (!(d > 0.0) || !(k > 0.0) || !(nu > 0.0)) return 0.0;
    const double ry = d * std::sqrt(k) / nu;
    const double arg = ry / 120.0;
    const double arg2 = arg * arg;
    const double arg4 = arg2 * arg2;
    const double arg8 = arg4 * arg4;
    return std::exp(-std::min(arg8, 50.0));
}

SourceTerms source(const PointState& s, double production_limiter) noexcept {
    SourceTerms t;
    const double k = std::max(s.k, 0.0);
    const double w = std::max(s.omega, kTiny);
    const double gamma = std::max(s.gamma, 0.0);
    const double re_tt = std::max(s.re_thetat, 1.0);
    const double nu = std::max(s.nu, kTiny);
    const double d = std::max(s.d, 0.0);
    const double s_mag = std::max(s.strain, 0.0);
    const double omega_mag = std::max(s.vorticity, 0.0);
    const double u_mag = std::max(s.u_mag, 1.0e-6);

    // 1. Transition onset correlations
    const double tu_local = 100.0 * std::sqrt(2.0 * k / 3.0) / u_mag;
    const double re_tt_target = re_theta_t(tu_local);
    const double re_tc = re_theta_c(re_tt);
    const double f_len = f_length(re_tt);

    const double rev = d * d * s_mag / nu;
    const double rt = k / (nu * w);

    const double fonset = f_onset(rev, re_tc, rt);
    const double fturb = f_turb(rt);
    const double g_sep = gamma_sep(rev, re_tc, rt);
    const double g_eff = std::max(gamma, g_sep);
    t.gamma_eff = g_eff;

    // 2. SST F1 modification with F3
    const double f3_val = f3(d, k, nu);
    const double f1_trans = std::max(s.f1, f3_val);

    const double beta = sst::blend(f1_trans, C::beta1, C::beta2);
    const double gamma_sst = sst::blend(f1_trans, C::gamma1, C::gamma2);
    const double s2 = s_mag * s_mag;

    // 3. SST k and omega sources with transition coupling
    double p_k = s.nut * s2;
    if (production_limiter > 0.0 && p_k > production_limiter * C::beta_star * k * w) {
        p_k = production_limiter * C::beta_star * k * w;
    }
    const double p_k_eff = g_eff * p_k;
    const double d_k_eff = std::min(std::max(g_eff, 0.1), 1.0) * C::beta_star * k * w;
    t.p_k_eff = p_k_eff;
    t.d_k_eff = d_k_eff;

    t.k.add_value(p_k_eff, k);
    t.k.add_linear(-std::min(std::max(g_eff, 0.1), 1.0) * C::beta_star * w, k);

    const double inv_w = 1.0 / w;
    const double cd_coef = 2.0 * (1.0 - f1_trans) * C::sigma_w2 * s.grad_k_dot_grad_log_omega;
    const double omega_prod = gamma_sst * s2;
    const double gradient_term = s.diffusivity_omega * s.grad_log_omega_sq;
    t.log_omega_source = omega_prod * inv_w - beta * w + cd_coef + gradient_term;

    const double lw = std::log(w);
    t.log_omega.add_newton(omega_prod * inv_w, -omega_prod * inv_w, lw);
    t.log_omega.add_newton(-beta * w, -beta * w, lw);
    t.log_omega.add_newton(cd_coef, -cd_coef, lw);
    t.log_omega.add_value(gradient_term, lw);

    // 4. Intermittency gamma source terms
    // P_gamma = F_length * c_a1 * S * sqrt(gamma * F_onset) * (1 - c_e1 * gamma)
    const double sqrt_g_fonset = std::sqrt(std::max(gamma * fonset, 0.0));
    const double p_gamma_coef = f_len * C::ca1 * s_mag * sqrt_g_fonset;
    const double p_gamma = p_gamma_coef * (1.0 - C::ce1 * gamma);
    t.p_gamma = p_gamma;

    t.gamma.add_value(p_gamma_coef, gamma);
    t.gamma.add_linear(-p_gamma_coef * C::ce1, gamma);

    // E_gamma = c_a2 * Omega * gamma * F_turb * (c_e2 * gamma - 1)
    const double e_gamma_coef = C::ca2 * omega_mag * fturb;
    const double e_gamma = e_gamma_coef * gamma * (C::ce2 * gamma - 1.0);
    t.e_gamma = e_gamma;

    if (C::ce2 * gamma >= 1.0) {
        t.gamma.add_linear(-e_gamma_coef * (C::ce2 * gamma - 1.0), gamma);
    } else {
        t.gamma.add_value(-e_gamma_coef * gamma * (C::ce2 * gamma - 1.0), gamma);
    }

    // 5. Re_thetat_tilde source terms
    const double t_scale = 500.0 * nu / (u_mag * u_mag);
    const double inv_t_scale = 1.0 / std::max(t_scale, 1.0e-12);

    const double theta_bl = re_tt * nu / u_mag;
    const double delta_bl = 7.5 * theta_bl;
    const double delta = 50.0 * (omega_mag * d / u_mag) * delta_bl;
    const double re_omega = w * d * d / nu;
    const double f_wake = std::exp(-std::min(std::pow(re_omega / 1.0e5, 2.0), 50.0));
    const double y_norm = d / std::max(delta, 1.0e-8);
    const double term1 = f_wake * std::exp(-std::min(std::pow(y_norm, 4.0), 50.0));
    const double g_diff = gamma - 1.0;
    const double term2 = 1.0 - g_diff * g_diff;
    const double f_thetat = std::clamp(std::max(term1, term2), 0.0, 1.0);
    t.f_thetat = f_thetat;

    const double p_thetat = C::c_thetat * inv_t_scale * (re_tt_target - re_tt) * (1.0 - f_thetat);
    t.p_thetat = p_thetat;

    const double p_tt_coef = C::c_thetat * inv_t_scale * (1.0 - f_thetat);
    t.re_thetat.add_value(p_tt_coef * re_tt_target, re_tt);
    t.re_thetat.add_linear(-p_tt_coef, re_tt);

    return t;
}

} // namespace transition

GammaReThetaSST::GammaReThetaSST(TurbulenceConfig config)
    : config_(std::move(config)) {}

std::vector<FieldRef> GammaReThetaSST::fields() {
    std::vector<FieldRef> r;
    r.push_back({"k", &phi_[0]});
    r.push_back({"omega", &omega_});
    r.push_back({"nut", &nut_});
    r.push_back({"gamma", &phi_[2]});
    r.push_back({"re_thetat", &phi_[3]});
    r.push_back({"gamma_eff", &gamma_eff_});
    r.push_back({"sst_f1", &f1_});
    return r;
}

TwoEquationInlet GammaReThetaSST::inlet_values(int flag, double nu) const {
    return two_equation_inlet(config_.inlet_for(flag), nu, config_.reference_velocity);
}

TwoEquationInlet GammaReThetaSST::freestream_values(double nu) const {
    return two_equation_inlet(config_.inlet, nu, config_.reference_velocity);
}

double GammaReThetaSST::inlet_re_thetat(double tu) const {
    return transition::re_theta_t(tu);
}

void GammaReThetaSST::initialize_fields(const TurbulenceContext& ctx) {
    const Mesh& mesh = ctx.mesh;
    const double nu = ctx.nu;
    const auto fs = freestream_values(nu);
    const double tu_fs = 100.0 * std::sqrt(2.0 * fs.k / 3.0) / std::max(config_.reference_velocity, 1.0e-6);
    const double re_tt_fs = inlet_re_thetat(tu_fs);

    phi_[0].assign(mesh.npoin, fs.k);
    phi_[1].assign(mesh.npoin, std::log(std::max(fs.omega, sst::kTiny)));
    phi_[2].assign(mesh.npoin, 1.0); // free-stream intermittency = 1.0
    phi_[3].assign(mesh.npoin, re_tt_fs);

    omega_.assign(mesh.npoin, fs.omega);
    gamma_eff_.assign(mesh.npoin, 1.0);
    f1_.assign(mesh.npoin, 1.0);
    f2_.assign(mesh.npoin, 1.0);

    // First-cell height at every node of a (turbulence) wall side:
    std::vector<double> height(mesh.npoin, 0.0), length(mesh.npoin, 0.0);
    for (const BoundarySide& side : mesh.iside) {
        if (config_.boundary_kind(side.flag) != BoundaryKind::Wall) continue;
        const double L = std::hypot(mesh.x[side.node2] - mesh.x[side.node1], mesh.y[side.node2] - mesh.y[side.node1]);
        const double h = mesh.is_quad(side.elem) ? mesh.height_above_edge(side.elem, side.node1, side.node2)
                                                 : std::abs(mesh.detJ[side.elem]) / L;
        for (int ip : {side.node1, side.node2}) {
            height[ip] += h * L;
            length[ip] += L;
        }
    }

    auto k_in = [&](int flag) { return inlet_values(flag, nu).k; };
    auto w_in = [&](int flag) { return std::log(std::max(inlet_values(flag, nu).omega, sst::kTiny)); };
    auto g_in = [&](int) { return 1.0; };
    auto rt_in = [&](int flag) {
        const auto in = inlet_values(flag, nu);
        const double tu = 100.0 * std::sqrt(2.0 * in.k / 3.0) / std::max(config_.reference_velocity, 1.0e-6);
        return inlet_re_thetat(tu);
    };

    bcs_[0] = build_turbulence_dirichlet(mesh, config_, k_in, std::optional<double>(0.0));
    bcs_[1] = build_turbulence_dirichlet(mesh, config_, w_in, std::function<double(int)>([&](int node) {
                                             return std::log(sst::wall_omega(nu, height[node] / length[node]));
                                         }));
    bcs_[2] = build_turbulence_dirichlet(mesh, config_, g_in, std::nullopt);
    bcs_[3] = build_turbulence_dirichlet(mesh, config_, rt_in, std::nullopt);
}

void GammaReThetaSST::compute_eddy_viscosity(const TurbulenceContext& ctx) {
    const std::vector<double>& k = phi_[0];
    const std::vector<double>& lw = phi_[1];
    const std::vector<double>& d = ctx.walls.wall_distance();
    const std::vector<double>& strain = kinematics_.strain_node;

    recovery_.gradient(k, grad_k_);
    recovery_.gradient(lw, grad_lw_);

    const double cap = config_.nut_ratio_max * ctx.nu;
    const int npoin = ctx.mesh.npoin;
    omega_.resize(npoin);
    f1_.resize(npoin);
    f2_.resize(npoin);
    gamma_eff_.resize(npoin);
    nut_.resize(npoin);

    for (int ip = 0; ip < npoin; ++ip) {
        const double w = std::exp(lw[ip]);
        omega_[ip] = w;
        const double dot = grad_k_[ip][0] * grad_lw_[ip][0] + grad_k_[ip][1] * grad_lw_[ip][1];
        f1_[ip] = sst::f1(k[ip], w, d[ip], ctx.nu, sst::cross_diffusion(dot));
        f2_[ip] = sst::f2(k[ip], w, d[ip], ctx.nu);
        nut_[ip] = std::min(sst::eddy_viscosity(k[ip], w, strain[ip], f2_[ip]), cap);

        const double re_tc = transition::re_theta_c(std::max(phi_[3][ip], 1.0));
        const double rev = d[ip] * d[ip] * strain[ip] / ctx.nu;
        const double rt = k[ip] / (ctx.nu * w);
        const double g_sep = transition::gamma_sep(rev, re_tc, rt);
        gamma_eff_[ip] = std::max(phi_[2][ip], g_sep);
    }
}

void GammaReThetaSST::compute_diffusivity(const TurbulenceContext& ctx) {
    const int npoin = ctx.mesh.npoin;
    const double nu = ctx.nu;

    diffusivity_node_k_.resize(npoin);
    diffusivity_node_w_.resize(npoin);

    std::vector<double> diff_node_g(npoin), diff_node_rt(npoin);

    for (int ip = 0; ip < npoin; ++ip) {
        const double f = f1_[ip];
        diffusivity_node_k_[ip] = nu + sst::blend(f, sst::Constants::sigma_k1, sst::Constants::sigma_k2) * nut_[ip];
        diffusivity_node_w_[ip] = nu + sst::blend(f, sst::Constants::sigma_w1, sst::Constants::sigma_w2) * nut_[ip];
        diff_node_g[ip] = nu + nut_[ip] / transition::Constants::sigma_gamma;
        diff_node_rt[ip] = transition::Constants::sigma_thetat * (nu + nut_[ip]);
    }

    element_mean(ctx.mesh, diffusivity_node_k_, diffusivity_elem_[0]);
    element_mean(ctx.mesh, diffusivity_node_w_, diffusivity_elem_[1]);
    element_mean(ctx.mesh, diff_node_g, diffusivity_elem_[2]);
    element_mean(ctx.mesh, diff_node_rt, diffusivity_elem_[3]);
}

void GammaReThetaSST::compute_sources(const TurbulenceContext& ctx) {
    const std::vector<double>& k = phi_[0];
    const std::vector<double>& strain = kinematics_.strain_node;
    const std::vector<double>& vorticity = kinematics_.vorticity_node;
    const std::vector<double>& speed = kinematics_.speed_node;
    const auto& d = ctx.walls.wall_distance();

    for (int ip = 0; ip < ctx.mesh.npoin; ++ip) {
        const auto& gk = grad_k_[ip];
        const auto& gw = grad_lw_[ip];
        transition::PointState s;
        s.k = k[ip];
        s.omega = omega_[ip];
        s.gamma = phi_[2][ip];
        s.re_thetat = phi_[3][ip];
        s.nu = ctx.nu;
        s.nut = nut_[ip];
        s.strain = strain[ip];
        s.vorticity = vorticity[ip];
        s.d = d[ip];
        s.u_mag = speed.empty() ? 1.0 : std::max(speed[ip], 1.0e-6);
        s.f1 = f1_[ip];
        s.grad_k_dot_grad_log_omega = gk[0] * gw[0] + gk[1] * gw[1];
        s.grad_log_omega_sq = gw[0] * gw[0] + gw[1] * gw[1];
        s.diffusivity_omega = diffusivity_node_w_[ip];

        const auto src = transition::source(s, config_.production_limiter);

        production_[0][ip] = src.k.P;
        destruction_[0][ip] = src.k.D;
        quadratic_[0][ip] = src.k.B;

        production_[1][ip] = src.log_omega.P;
        destruction_[1][ip] = src.log_omega.D;
        quadratic_[1][ip] = src.log_omega.B;

        production_[2][ip] = src.gamma.P;
        destruction_[2][ip] = src.gamma.D;
        quadratic_[2][ip] = src.gamma.B;

        production_[3][ip] = src.re_thetat.P;
        destruction_[3][ip] = src.re_thetat.D;
        quadratic_[3][ip] = src.re_thetat.B;
    }
}

} // namespace hydrama
