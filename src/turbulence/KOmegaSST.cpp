#include "turbulence/KOmegaSST.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace hydrama {

namespace sst {

using C = Constants;

double cross_diffusion(double grad_k_dot_grad_log_omega) noexcept {
    return std::max(2.0 * C::sigma_w2 * grad_k_dot_grad_log_omega, C::cd_min);
}

double f1(double k, double omega, double d, double nu, double cd_kw) noexcept {
    if (!(d > 0.0)) return 1.0;
    const double kk = std::max(k, 0.0);
    const double w = std::max(omega, kTiny);
    const double d2 = d * d; // inf for d = inf: all arguments vanish
    const double a = std::sqrt(kk) / (C::beta_star * w * d);
    const double b = 500.0 * nu / (d2 * w);
    const double c = 4.0 * C::sigma_w2 * kk / (std::max(cd_kw, C::cd_min) * d2);
    const double arg = std::min(std::max(a, b), c);
    const double arg2 = arg * arg;
    return std::tanh(arg2 * arg2);
}

double f2(double k, double omega, double d, double nu) noexcept {
    if (!(d > 0.0)) return 1.0;
    const double w = std::max(omega, kTiny);
    const double arg = std::max(2.0 * std::sqrt(std::max(k, 0.0)) / (C::beta_star * w * d), 500.0 * nu / (d * d * w));
    return std::tanh(arg * arg);
}

double eddy_viscosity(double k, double omega, double strain, double f2_value) noexcept {
    if (!(k > 0.0)) return 0.0;
    return C::a1 * k / std::max({C::a1 * omega, strain * f2_value, kTiny});
}

double wall_omega(double nu, double y1) noexcept { return 60.0 * nu / (C::beta1 * y1 * y1); }

SourceTerms source(const PointState& s, double production_limiter) noexcept {
    SourceTerms t;
    const double k = std::max(s.k, 0.0);
    const double w = std::max(s.omega, 0.0);
    const double beta = blend(s.f1, C::beta1, C::beta2);
    const double gamma = blend(s.f1, C::gamma1, C::gamma2);
    const double s2 = s.strain * s.strain;

    double p = s.nut * s2;
    if (production_limiter > 0.0 && p > production_limiter * C::beta_star * k * w) {
        p = production_limiter * C::beta_star * k * w;
        t.limited = true;
    }
    const double inv_w = 1.0 / std::max(w, kTiny);
    const double cd_coef = 2.0 * (1.0 - s.f1) * C::sigma_w2 * s.grad_k_dot_grad_log_omega; // * e^-w~
    t.production = p;
    t.k_destruction = C::beta_star * k * w;
    t.omega_production = gamma * s2;
    t.omega_destruction = beta * w * w;
    t.cross_diffusion = cd_coef * w; // (1/omega) grad k . grad omega = grad k . grad w~
    const double gradient_term = s.diffusivity_omega * s.grad_log_omega_sq;
    t.log_omega_source = t.omega_production * inv_w - beta * w + cd_coef + gradient_term;

    t.k.add_value(p, k);
    t.k.add_linear(-C::beta_star * w, k);

    // w~ equation, with e^-w~ = 1/omega and e^w~ = omega:
    const double lw = std::log(std::max(w, kTiny));
    t.log_omega.add_newton(t.omega_production * inv_w, -t.omega_production * inv_w, lw); // gamma S^2 e^-w~
    t.log_omega.add_newton(-beta * w, -beta * w, lw);                                     // -beta e^w~
    t.log_omega.add_newton(cd_coef, -cd_coef, lw); // cross diffusion c e^-w~ (c = cd_coef * omega)
    t.log_omega.add_value(gradient_term, lw);      // Gamma_w |grad w~|^2 >= 0
    return t;
}

} // namespace sst

KOmegaSST::KOmegaSST(TurbulenceConfig config) : config_(std::move(config)) {}

std::vector<FieldRef> KOmegaSST::fields() {
    return {{"k", &phi_[0]}, {"omega", &omega_}, {"nut", &nut_}, {"sst_f1", &f1_}};
}

TwoEquationInlet KOmegaSST::inlet_values(int flag, double nu) const {
    return two_equation_inlet(config_.inlet_for(flag), nu, config_.reference_velocity);
}

TwoEquationInlet KOmegaSST::freestream_values(double nu) const {
    return two_equation_inlet(config_.inlet, nu, config_.reference_velocity);
}

void KOmegaSST::initialize_fields(const TurbulenceContext& ctx) {
    const Mesh& mesh = ctx.mesh;
    const TwoEquationInlet fs = freestream_values(ctx.nu);
    phi_[0].assign(mesh.npoin, fs.k);
    phi_[1].assign(mesh.npoin, std::log(fs.omega));

    // First-cell height at every node of a (turbulence) wall side:
    // length-weighted mean of 2 A / L over its wall sides.
    std::vector<double> height(mesh.npoin, 0.0), length(mesh.npoin, 0.0);
    for (const BoundarySide& side : mesh.iside) {
        if (config_.boundary_kind(side.flag) != BoundaryKind::Wall) continue;
        const double L = std::hypot(mesh.x[side.node2] - mesh.x[side.node1], mesh.y[side.node2] - mesh.y[side.node1]);
        const double h = std::abs(mesh.detJ[side.elem]) / L;
        for (int ip : {side.node1, side.node2}) {
            height[ip] += h * L;
            length[ip] += L;
        }
    }
    auto k_in = [&](int flag) { return inlet_values(flag, ctx.nu).k; };
    auto w_in = [&](int flag) { return std::log(inlet_values(flag, ctx.nu).omega); };
    bcs_[0] = build_turbulence_dirichlet(mesh, config_, k_in, std::optional<double>(0.0));
    bcs_[1] = build_turbulence_dirichlet(mesh, config_, w_in, std::function<double(int)>([&](int node) {
                                             return std::log(sst::wall_omega(ctx.nu, height[node] / length[node]));
                                         }));
    omega_.assign(mesh.npoin, fs.omega);
    f1_.assign(mesh.npoin, 1.0);
    f2_.assign(mesh.npoin, 1.0);
}

void KOmegaSST::compute_eddy_viscosity(const TurbulenceContext& ctx) {
    // Blending functions and nu_t of the current state (also used by the
    // sources and diffusivities of the next step).
    const std::vector<double>& k = phi_[0];
    const std::vector<double>& lw = phi_[1];
    const std::vector<double>& d = ctx.walls.wall_distance();
    const std::vector<double>& strain = kinematics_.strain_node;
    recovery_.gradient(k, grad_k_);
    recovery_.gradient(lw, grad_lw_);
    const double cap = config_.nut_ratio_max * ctx.nu;
    omega_.resize(ctx.mesh.npoin);
    for (int ip = 0; ip < ctx.mesh.npoin; ++ip) {
        const double w = std::exp(lw[ip]);
        omega_[ip] = w;
        const double dot = grad_k_[ip][0] * grad_lw_[ip][0] + grad_k_[ip][1] * grad_lw_[ip][1];
        f1_[ip] = sst::f1(k[ip], w, d[ip], ctx.nu, sst::cross_diffusion(dot));
        f2_[ip] = sst::f2(k[ip], w, d[ip], ctx.nu);
        nut_[ip] = std::min(sst::eddy_viscosity(k[ip], w, strain[ip], f2_[ip]), cap);
    }
}

void KOmegaSST::compute_sources(const TurbulenceContext& ctx) {
    const std::vector<double>& k = phi_[0];
    const std::vector<double>& strain = kinematics_.strain_node;
    for (int ip = 0; ip < ctx.mesh.npoin; ++ip) {
        const auto& gk = grad_k_[ip];
        const auto& gw = grad_lw_[ip];
        const sst::PointState s{k[ip],
                                omega_[ip],
                                ctx.nu,
                                nut_[ip],
                                strain[ip],
                                f1_[ip],
                                gk[0] * gw[0] + gk[1] * gw[1],
                                gw[0] * gw[0] + gw[1] * gw[1],
                                diffusivity_node_w_[ip]};
        const sst::SourceTerms t = sst::source(s, config_.production_limiter);
        production_[0][ip] = t.k.P;
        destruction_[0][ip] = t.k.D;
        production_[1][ip] = t.log_omega.P;
        destruction_[1][ip] = t.log_omega.D;
    }
}

void KOmegaSST::compute_diffusivity(const TurbulenceContext& ctx) {
    const int npoin = ctx.mesh.npoin;
    diffusivity_node_k_.resize(npoin);
    diffusivity_node_w_.resize(npoin);
    for (int ip = 0; ip < npoin; ++ip) {
        const double f = f1_[ip];
        diffusivity_node_k_[ip] = ctx.nu + sst::blend(f, sst::Constants::sigma_k1, sst::Constants::sigma_k2) * nut_[ip];
        diffusivity_node_w_[ip] = ctx.nu + sst::blend(f, sst::Constants::sigma_w1, sst::Constants::sigma_w2) * nut_[ip];
    }
    element_mean(ctx.mesh, diffusivity_node_k_, diffusivity_elem_[0]);
    element_mean(ctx.mesh, diffusivity_node_w_, diffusivity_elem_[1]);
}

} // namespace hydrama
