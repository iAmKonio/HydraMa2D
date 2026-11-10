#include "turbulence/SpalartAllmaras.hpp"

#include "turbulence/TurbulenceBC.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace hydrama {

namespace sa {

using C = Constants;

double fv1(double chi) noexcept {
    const double chi3 = chi * chi * chi;
    return chi3 / (chi3 + C::cv1 * C::cv1 * C::cv1);
}

double fv2(double chi, double fv1_value) noexcept { return 1.0 - chi / (1.0 + chi * fv1_value); }

double fw(double r) noexcept {
    const double g = r + C::cw2 * (std::pow(r, 6) - r);
    const double cw3_6 = std::pow(C::cw3, 6);
    return g * std::pow((1.0 + cw3_6) / (std::pow(g, 6) + cw3_6), 1.0 / 6.0);
}

double ft2(double chi) noexcept { return C::ct3 * std::exp(-C::ct4 * chi * chi); }

double fn(double chi) noexcept {
    const double chi3 = chi * chi * chi;
    return (C::cn1 + chi3) / (C::cn1 - chi3);
}

double s_tilde(double omega, double s_bar) noexcept {
    if (s_bar >= -C::cv2 * omega) return omega + s_bar;
    return omega + omega * (C::cv2 * C::cv2 * omega + C::cv3 * s_bar) / ((C::cv3 - 2.0 * C::cv2) * omega - s_bar);
}

double eddy_viscosity(double nu_tilde, double nu) noexcept {
    return nu_tilde > 0.0 ? nu_tilde * fv1(nu_tilde / nu) : 0.0;
}

double diffusivity(double nu_tilde, double nu) noexcept {
    const double f = nu_tilde >= 0.0 ? 1.0 : fn(nu_tilde / nu);
    return (nu + f * nu_tilde) / C::sigma;
}

double nu_tilde_ratio_from_viscosity_ratio(double viscosity_ratio) {
    if (!(viscosity_ratio >= 0.0)) throw std::invalid_argument("SA: viscosity ratio must be >= 0");
    // chi fv1(chi) is increasing and r <= chi <= r + cv1 (chi fv1 <= chi and
    // chi fv1 >= chi - cv1^3/chi^2); bisection to machine precision.
    double lo = viscosity_ratio, hi = viscosity_ratio + C::cv1;
    for (int it = 0; it < 200 && hi - lo > 0.0; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (mid == lo || mid == hi) break;
        if (mid * fv1(mid) < viscosity_ratio) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

SourceTerms source(const PointState& s, bool use_ft2) noexcept {
    SourceTerms t;
    if (!(s.d > 0.0)) return t; // wall node: nu~ is prescribed

    const double nt = s.nu_tilde;
    const double chi = nt / s.nu;
    const double inv_d2 = 1.0 / (s.d * s.d); // 0 for d = inf
    const double k2 = C::kappa * C::kappa;

    t.cross_diffusion = C::cb2 / C::sigma * s.grad_sq;

    if (nt >= 0.0) {
        const double f_v1 = fv1(chi);
        const double s_bar = nt * fv2(chi, f_v1) * inv_d2 / k2;
        const double st = s_tilde(s.omega, s_bar);
        const double denom = st * k2 * s.d * s.d;
        const double r = (denom > 0.0) ? std::min(nt / denom, C::r_max) : C::r_max;
        const double f_t2 = use_ft2 ? ft2(chi) : 0.0;

        const double prod_coef = C::cb1 * (1.0 - f_t2) * st;                    // * nu~
        const double dest_coef = (C::cw1 * fw(r) - C::cb1 / k2 * f_t2) * nt * inv_d2; // * nu~
        t.production = prod_coef * nt;
        t.destruction = dest_coef * nt;
        t.split.add_linear(prod_coef, nt);
        t.split.add_linear(-dest_coef, nt);
    } else {
        const double prod_coef = C::cb1 * (1.0 - C::ct3) * s.omega; // <= 0: decay towards 0
        const double dest_coef = -C::cw1 * nt * inv_d2;              // destruction = dest_coef * nu~ <= 0
        t.production = prod_coef * nt;
        t.destruction = dest_coef * nt;
        t.split.add_linear(prod_coef, nt);
        t.split.add_linear(-dest_coef, nt);
    }
    t.split.add_value(t.cross_diffusion, nt);
    return t;
}

} // namespace sa

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

SpalartAllmaras::SpalartAllmaras(TurbulenceConfig config) : config_(std::move(config)) {}

namespace {

// nu~/nu from an inlet specification: explicit ratio, else converted from
// nu_t/nu, else 3 (TMR free-stream recommendation).
double nu_tilde_ratio(const InletTurbulence& in) {
    if (in.nu_tilde_ratio) return *in.nu_tilde_ratio;
    if (in.viscosity_ratio) return sa::nu_tilde_ratio_from_viscosity_ratio(*in.viscosity_ratio);
    return 3.0;
}

} // namespace

double SpalartAllmaras::inlet_nu_tilde_ratio(int flag) const { return nu_tilde_ratio(config_.inlet_for(flag)); }

double SpalartAllmaras::freestream_nu_tilde(double nu) const { return nu_tilde_ratio(config_.inlet) * nu; }

void SpalartAllmaras::initialize_fields(const TurbulenceContext& ctx) {
    phi_[0].assign(ctx.mesh.npoin, freestream_nu_tilde(ctx.nu));
    bcs_[0] = build_turbulence_dirichlet(
        ctx.mesh, config_, [&](int flag) { return inlet_nu_tilde_ratio(flag) * ctx.nu; }, 0.0);
}

void SpalartAllmaras::compute_sources(const TurbulenceContext& ctx) {
    const Mesh& mesh = ctx.mesh;
    const std::vector<double>& nt = phi_[0];

    // Recovered nodal gradient of nu~ (cb2 |grad nu~|^2 term).
    recovery_.gradient(nt, grad_node_);

    const std::vector<double>& d = ctx.walls.wall_distance();
    const std::vector<double>& omega = kinematics_.vorticity_node;
    std::vector<double>& P = production_[0];
    std::vector<double>& D = destruction_[0];
    for (int ip = 0; ip < mesh.npoin; ++ip) {
        const double grad_sq = grad_node_[ip][0] * grad_node_[ip][0] + grad_node_[ip][1] * grad_node_[ip][1];
        const sa::SourceTerms t = sa::source({nt[ip], ctx.nu, d[ip], omega[ip], grad_sq}, config_.sa_ft2);
        P[ip] = t.split.P;
        D[ip] = t.split.D;
    }
}

void SpalartAllmaras::compute_eddy_viscosity(const TurbulenceContext& ctx) {
    const double cap = config_.nut_ratio_max * ctx.nu;
    const std::vector<double>& nt = phi_[0];
    for (int ip = 0; ip < ctx.mesh.npoin; ++ip) nut_[ip] = std::min(sa::eddy_viscosity(nt[ip], ctx.nu), cap);
}

void SpalartAllmaras::compute_diffusivity(const TurbulenceContext& ctx) {
    const std::vector<double>& nt = phi_[0];
    diffusivity_node_.resize(ctx.mesh.npoin);
    for (int ip = 0; ip < ctx.mesh.npoin; ++ip) diffusivity_node_[ip] = sa::diffusivity(nt[ip], ctx.nu);
    element_mean(ctx.mesh, diffusivity_node_, diffusivity_elem_[0]);
}

} // namespace hydrama
