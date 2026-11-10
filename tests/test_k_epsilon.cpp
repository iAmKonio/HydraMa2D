// k-epsilon family: standard k-epsilon with log-law wall functions and the
// Launder-Sharma low-Re model.
//
// Closure: constants (log-layer consistency kappa^2 = (C2 - C1) sigma_e sqrt(C_mu)),
// Launder-Sharma damping functions f_mu, f2 (values, limits, monotonicity),
// eddy viscosity, source terms and the production limiter at reference points
// (independent Python evaluation), linearisations, the exact backward-Euler
// (Riccati) update of the dissipation, the low-Re terms D = 2 nu |grad sqrt k|^2
// and E = 2 nu nu_t |grad grad u|^2 on fields where they are known exactly.
//
// Decay of homogeneous isotropic turbulence: analytical solution of the
// standard model (and Launder-Sharma at high R_t), RK4 reference for the
// low-R_t Launder-Sharma decay (f2 -> 0.7), discrete recurrence and
// first-order convergence in time.
//
// Exact log-layer solution (second order), wall functions (y+_lam, nu_t,w,
// eps_w, wall-node production) and a log-law equilibrium test.

#include "TurbulenceTestUtils.hpp"
#include "turbulence/KEpsilon.hpp"
#include "turbulence/WallFunctions.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using KS = ke::Standard;

namespace {

const double kKappaEps = std::sqrt((KS::c2 - KS::c1) * KS::sigma_eps * std::sqrt(KS::c_mu));

template <class Model>
TurbulenceModelType model_type() {
    if constexpr (std::is_same_v<Model, StandardKEpsilon>) {
        return TurbulenceModelType::KEpsilon;
    } else {
        return TurbulenceModelType::LaunderSharmaKEpsilon;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Closure functions
// ---------------------------------------------------------------------------

TEST_CASE("k-epsilon constants", "[kepsilon]") {
    CHECK(KS::c_mu == 0.09);
    CHECK(KS::c1 == 1.44);
    CHECK(KS::c2 == 1.92);
    CHECK(KS::sigma_k == 1.0);
    CHECK(KS::sigma_eps == 1.3);
    CHECK_THAT(kKappaEps, WithinRel(0.43266615305567874, 1e-12));
    CHECK(std::string(StandardKEpsilon::kNames[1]) == "epsilon");
    CHECK(std::string(LaunderSharmaKEpsilon::kNames[1]) == "epsilon_tilde");
}

TEST_CASE("Launder-Sharma damping functions", "[kepsilon][damping]") {
    using LS = ke::LaunderSharma;
    CHECK_THAT(LS::f_mu(0.0), WithinRel(0.03337326996032608, 1e-14)); // exp(-3.4)
    CHECK_THAT(LS::f_mu(50.0), WithinRel(0.4274149319487267, 1e-14));
    CHECK_THAT(LS::f_mu(2.0), WithinRel(0.04313195573373551, 1e-14));
    CHECK_THAT(LS::f_mu(1e8), WithinAbs(1.0, 1e-9));
    CHECK_THAT(LS::f2(0.0), WithinRel(0.7, 1e-15));
    CHECK_THAT(LS::f2(1.0), WithinRel(0.8896361676485673, 1e-14));
    CHECK_THAT(LS::f2(0.5), WithinRel(0.7663597650785785, 1e-14));
    CHECK(LS::f2(10.0) == 1.0);
    double pm = 0.0, p2 = 0.0;
    for (double rt = 0.0; rt < 500.0; rt += 0.25) {
        CHECK(LS::f_mu(rt) > pm);
        CHECK(LS::f2(rt) >= p2);
        pm = LS::f_mu(rt);
        p2 = LS::f2(rt);
    }
    // High-Re model: no damping.
    CHECK(ke::NoDamping::f_mu(0.0) == 1.0);
    CHECK(ke::NoDamping::f2(0.0) == 1.0);
}

TEST_CASE("k-epsilon eddy viscosity and turbulence Reynolds number", "[kepsilon]") {
    const double nu = 1e-3;
    CHECK_THAT(ke::turbulence_reynolds(1e-3, 2e-3, nu), WithinRel(0.5, 1e-14));
    CHECK(ke::turbulence_reynolds(0.0, 1e-3, nu) == 0.0);
    CHECK(ke::turbulence_reynolds(1e-3, 0.0, nu) > 1e20);
    CHECK_THAT((ke::eddy_viscosity<KS, ke::NoDamping>(2e-2, 3e-3, nu)), WithinRel(0.09 * 4e-4 / 3e-3, 1e-14));
    CHECK_THAT((ke::eddy_viscosity<KS, ke::LaunderSharma>(1e-3, 2e-3, nu)), WithinRel(1.6058543524618828e-06, 1e-12));
    CHECK((ke::eddy_viscosity<KS, ke::LaunderSharma>(0.0, 0.0, nu)) == 0.0);
    CHECK(std::isfinite(ke::eddy_viscosity<KS, ke::NoDamping>(1e-3, 0.0, nu)));
    // Near a wall (R_t -> 0) Launder-Sharma damps nu_t strongly: f_mu -> exp(-3.4).
    const double k = 1e-8, e = 1e-2; // R_t = 1e-11
    CHECK_THAT((ke::eddy_viscosity<KS, ke::LaunderSharma>(k, e, nu)) /
                   (ke::eddy_viscosity<KS, ke::NoDamping>(k, e, nu)),
               WithinRel(std::exp(-3.4), 1e-9));
}

TEST_CASE("k-epsilon source terms at reference points", "[kepsilon][source]") {
    SECTION("standard model") {
        const ke::SourceTerms t = ke::source<KS, ke::NoDamping>({2e-2, 3e-3, 1e-5, 1e-2, 0.7, 0.0, 0.0}, 10.0);
        CHECK_FALSE(t.limited);
        CHECK_THAT(t.production, WithinRel(0.0049, 1e-14));
        CHECK_THAT(t.dissipation, WithinRel(3e-3, 1e-15));
        CHECK_THAT(t.eps_production, WithinRel(0.0010584, 1e-13));
        CHECK_THAT(t.eps_destruction, WithinRel(0.000864, 1e-13));
    }
    SECTION("production limiter P_k <= 10 eps") {
        const ke::SourceTerms t = ke::source<KS, ke::NoDamping>({2e-2, 3e-3, 1e-5, 1e-2, 3.0, 0.0, 0.0}, 10.0);
        CHECK(t.limited);
        CHECK_THAT(t.production, WithinRel(0.03, 1e-14));
        CHECK_THAT(t.eps_production, WithinRel(0.00648, 1e-13));
        CHECK_FALSE((ke::source<KS, ke::NoDamping>({2e-2, 3e-3, 1e-5, 1e-2, 3.0, 0.0, 0.0}, 0.0).limited));
    }
    SECTION("Launder-Sharma with D and E terms") {
        const ke::SourceTerms t = ke::source<KS, ke::LaunderSharma>({1e-3, 2e-3, 1e-3, 2e-5, 5.0, 4e-4, 1e-4}, 10.0);
        CHECK_THAT(t.production, WithinRel(0.0005, 1e-13));
        CHECK_THAT(t.dissipation, WithinRel(0.0024, 1e-14)); // eps = eps~ + D
        CHECK_THAT(t.eps_production, WithinRel(0.00154, 1e-13));
        CHECK_THAT(t.eps_destruction, WithinRel(0.005885642995803483, 1e-13)); // f2(R_t = 0.5)
    }
    SECTION("wall-function production override") {
        const ke::SourceTerms t = ke::source<KS, ke::NoDamping>({2e-2, 3e-3, 1e-5, 1e-2, 0.7, 0.0, 0.0}, 10.0, 2e-3);
        CHECK(t.production == 2e-3);
    }
}

TEST_CASE("k-epsilon linearisations reproduce the sources", "[kepsilon][source]") {
    for (double k : {1e-8, 1e-3, 0.1}) {
        for (double e : {1e-9, 1e-3, 1.0}) {
            for (double s : {0.0, 2.0, 50.0}) {
                const double nut = ke::eddy_viscosity<KS, ke::LaunderSharma>(k, e, 1e-4);
                const ke::PointState p{k, e, 1e-4, nut, s, 1e-5, 2e-6};
                const ke::SourceTerms t = ke::source<KS, ke::LaunderSharma>(p, 10.0);
                CAPTURE(k, e, s);
                CHECK(t.k.D >= 0.0);
                CHECK(t.e.D >= 0.0);
                CHECK(t.e.B >= 0.0);
                CHECK_THAT(t.k.value(k), WithinAbs(t.production - t.dissipation, 1e-12 * (1 + t.production)));
                CHECK_THAT(t.e.value(e), WithinAbs(t.eps_production - t.eps_destruction,
                                                   1e-12 * (1 + t.eps_production + t.eps_destruction)));
            }
        }
    }
}

TEST_CASE("Exact backward-Euler update of a Riccati sink", "[kepsilon][point_implicit]") {
    // d phi/dt = A - B phi^2: backward Euler phi1 = phi0 + dt (A - B phi1^2).
    const double A = 2.0, B = 3.0, phi0 = 0.1;
    for (double dt : {1e-3, 0.1, 10.0, 1e6}) {
        std::vector<double> phi = {phi0};
        point_implicit_update({1.0}, {dt}, {0.0}, {A}, {0.0}, {B}, true, phi);
        CAPTURE(dt);
        CHECK_THAT(phi[0], WithinAbs(phi0 + dt * (A - B * phi[0] * phi[0]), 1e-12 * (1 + dt)));
        CHECK(phi[0] > 0.0);
        CHECK(phi[0] <= std::max(phi0, std::sqrt(A / B)) + 1e-15); // no overshoot of the equilibrium
    }
    // Implicit growth (D < 0) with a quadratic sink.
    std::vector<double> phi = {1e-3};
    point_implicit_update({1.0}, {100.0}, {0.0}, {0.0}, {-5.0}, {2.0}, true, phi);
    CHECK_THAT(phi[0], WithinRel(1e-3 + 100.0 * (5.0 * phi[0] - 2.0 * phi[0] * phi[0]), 1e-10));
    // B = 0 is the linear update, bit for bit.
    std::vector<double> a = {0.3, 0.2}, b = {0.3, 0.2};
    point_implicit_update({1.0, 2.0}, {0.1, 0.05}, {1.0, -4.0}, {0.5, 0.0}, {2.0, 1.0}, true, a);
    point_implicit_update({1.0, 2.0}, {0.1, 0.05}, {1.0, -4.0}, {0.5, 0.0}, {2.0, 1.0}, {0.0, 0.0}, true, b);
    CHECK(a == b);
}

TEST_CASE("Launder-Sharma low-Re terms D and E", "[kepsilon][damping]") {
    // sqrt(k) linear in y: D = 2 nu a^2 everywhere (PPR is exact).
    // u = c y^2: d2u/dy2 = 2c, E = 2 nu nu_t (2c)^2.
    const double a = 0.3, c = 0.8, nu = 1e-3;
    TurbSetup s(turb_grid(4, 6, 1.0, 1.0, 504), nu, 1e-6);
    // k = (a y + 0.01)^2 keeps sqrt(k) linear and k > 0.
    std::vector<double> u(s.mesh.npoin), v(s.mesh.npoin, 0.0), k(s.mesh.npoin), e(s.mesh.npoin, 1e-3);
    for (int ip = 0; ip < s.mesh.npoin; ++ip) {
        u[ip] = c * s.mesh.y[ip] * s.mesh.y[ip];
        k[ip] = std::pow(a * s.mesh.y[ip] + 0.01, 2);
    }
    LaunderSharmaKEpsilon model(model_config(TurbulenceModelType::LaunderSharmaKEpsilon));
    model.initialize(s.ctx(), {u, v});
    model.set_field(s.ctx(), 0, k);
    model.set_field(s.ctx(), 1, e);
    const std::vector<double> nut = model.nut();
    model.advance(s.ctx(), {u, v});
    for (int ip = 0; ip < s.mesh.npoin; ++ip) {
        CAPTURE(ip);
        CHECK_THAT(model.extra_dissipation()[ip], WithinRel(2.0 * nu * a * a, 1e-9));
        CHECK_THAT(model.extra_production()[ip], WithinRel(2.0 * nu * nut[ip] * 4.0 * c * c, 1e-8));
    }
}

// ---------------------------------------------------------------------------
// Homogeneous decay
// ---------------------------------------------------------------------------

namespace {

struct DecayResult {
    double k = 0.0, e = 0.0;         // model
    double k_rec = 0.0, e_rec = 0.0; // discrete recurrence
};

template <class Model>
DecayResult run_decay(double nu, double k0, double e0, double t_end, int steps) {
    const double dt = t_end / steps;
    // Large cells: the explicit diffusion of round-off non-uniformities stays
    // stable even for nu = 10 (low-R_t case).
    TurbSetup s(turb_grid(3, 2, 1000.0, 1000.0, 504), nu, dt);
    const std::vector<double> zero(s.mesh.npoin, 0.0);
    TurbulenceConfig c = model_config(model_type<Model>());
    c.inlet.k = k0;
    c.inlet.epsilon = e0;
    c.nut_ratio_max = 1e30;
    Model model(c);
    model.initialize(s.ctx(), {zero, zero});
    REQUIRE(model.boundary_conditions(0).empty());
    DecayResult r{0.0, 0.0, k0, e0};
    for (int n = 0; n < steps; ++n) {
        model.advance(s.ctx(), {zero, zero});
        // Recurrence: k_{n+1} = k / (1 + dt e/k); e solves dt B e^2 + e - e_n = 0.
        const double rt = ke::turbulence_reynolds(r.k_rec, r.e_rec, nu);
        const double f2 = std::is_same_v<Model, StandardKEpsilon> ? 1.0 : ke::LaunderSharma::f2(rt);
        const double b = dt * KS::c2 * f2 / r.k_rec;
        const double k_next = r.k_rec / (1.0 + dt * r.e_rec / r.k_rec);
        r.e_rec = 2.0 * r.e_rec / (1.0 + std::sqrt(1.0 + 4.0 * b * r.e_rec));
        r.k_rec = k_next;
    }
    r.k = model.k()[0];
    r.e = model.epsilon()[0];
    for (int ip = 0; ip < s.mesh.npoin; ++ip) {
        CHECK_THAT(model.k()[ip], WithinRel(r.k, 1e-12));
        CHECK_THAT(model.epsilon()[ip], WithinRel(r.e, 1e-12));
    }
    return r;
}

// Analytical decay of the standard model: k = k0 (1 + (C2-1) e0 t / k0)^(-1/(C2-1)).
double k_analytic(double k0, double e0, double t) {
    return k0 * std::pow(1.0 + (KS::c2 - 1.0) * e0 * t / k0, -1.0 / (KS::c2 - 1.0));
}
double e_analytic(double k0, double e0, double t) {
    return e0 * std::pow(1.0 + (KS::c2 - 1.0) * e0 * t / k0, -KS::c2 / (KS::c2 - 1.0));
}

// RK4 reference of dk/dt = -e, de/dt = -C2 f2(R_t) e^2/k (Launder-Sharma, no gradients).
void rk4_launder_sharma(double nu, double& k, double& e, double t_end, int steps) {
    auto rhs = [&](double kk, double ee, double& dk, double& de) {
        dk = -ee;
        de = -KS::c2 * ke::LaunderSharma::f2(kk * kk / (nu * ee)) * ee * ee / kk;
    };
    const double h = t_end / steps;
    for (int n = 0; n < steps; ++n) {
        double k1, e1, k2, e2, k3, e3, k4, e4;
        rhs(k, e, k1, e1);
        rhs(k + 0.5 * h * k1, e + 0.5 * h * e1, k2, e2);
        rhs(k + 0.5 * h * k2, e + 0.5 * h * e2, k3, e3);
        rhs(k + h * k3, e + h * e3, k4, e4);
        k += h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
        e += h / 6.0 * (e1 + 2 * e2 + 2 * e3 + e4);
    }
}

} // namespace

TEST_CASE("Standard k-epsilon: analytical decay of homogeneous isotropic turbulence", "[kepsilon][decay]") {
    const double nu = 1e-5, k0 = 1e-2, e0 = 5e-3, t_end = 4.0;
    const DecayResult a = run_decay<StandardKEpsilon>(nu, k0, e0, t_end, 400);
    const DecayResult b = run_decay<StandardKEpsilon>(nu, k0, e0, t_end, 800);
    CHECK_THAT(a.k, WithinRel(a.k_rec, 1e-10));
    CHECK_THAT(a.e, WithinRel(a.e_rec, 1e-10));
    const double ke = k_analytic(k0, e0, t_end), ee = e_analytic(k0, e0, t_end);
    CHECK_THAT(b.k, WithinRel(ke, 5e-3));
    CHECK_THAT(b.e, WithinRel(ee, 1e-2));
    CHECK_THAT(std::abs(a.k - ke) / std::abs(b.k - ke), WithinAbs(2.0, 0.15));
    CHECK_THAT(std::abs(a.e - ee) / std::abs(b.e - ee), WithinAbs(2.0, 0.15));
}

TEST_CASE("Launder-Sharma: homogeneous decay at high and low turbulence Reynolds number", "[kepsilon][decay][damping]") {
    SECTION("high R_t: f2 = 1, the standard analytical decay") {
        const double nu = 1e-9, k0 = 1e-2, e0 = 5e-3, t_end = 4.0; // R_t > 1e6 throughout
        const DecayResult r = run_decay<LaunderSharmaKEpsilon>(nu, k0, e0, t_end, 800);
        CHECK_THAT(r.k, WithinRel(r.k_rec, 1e-10));
        CHECK_THAT(r.k, WithinRel(k_analytic(k0, e0, t_end), 5e-3));
        CHECK_THAT(r.e, WithinRel(e_analytic(k0, e0, t_end), 1e-2));
    }
    SECTION("low R_t: f2 -> 0.7 slows the decay of eps~ (RK4 reference)") {
        const double nu = 10.0, k0 = 1e-2, e0 = 5e-3, t_end = 4.0; // R_t = 2e-3 -> f2 ~ 0.7
        double k_ref = k0, e_ref = e0;
        rk4_launder_sharma(nu, k_ref, e_ref, t_end, 100000);
        const DecayResult a = run_decay<LaunderSharmaKEpsilon>(nu, k0, e0, t_end, 400);
        const DecayResult b = run_decay<LaunderSharmaKEpsilon>(nu, k0, e0, t_end, 800);
        CHECK_THAT(b.k, WithinRel(b.k_rec, 1e-10));
        CHECK_THAT(b.k, WithinRel(k_ref, 5e-3));
        CHECK_THAT(b.e, WithinRel(e_ref, 1e-2));
        CHECK_THAT(std::abs(a.k - k_ref) / std::abs(b.k - k_ref), WithinAbs(2.0, 0.15));
        // f2 < 1 weakens the destruction of eps~: eps~ decays more slowly, so
        // k (dk/dt = -eps~) decays faster than in the undamped model.
        CHECK(e_ref > e_analytic(k0, e0, t_end) * 1.05);
        CHECK(k_ref < k_analytic(k0, e0, t_end));
        CHECK(b.e > e_analytic(k0, e0, t_end) * 1.05);
    }
}

// ---------------------------------------------------------------------------
// Exact log-layer solution
// ---------------------------------------------------------------------------

TEST_CASE("Standard k-epsilon: exact log-layer solution with second-order convergence", "[kepsilon][model][verification]") {
    // Frozen log-law velocity u = (u_tau/kappa) ln(d/y0), d = y + y0, with
    // kappa = kappa_eps: k = u_tau^2/sqrt(C_mu), eps = u_tau^3/(kappa d) solve
    // the k-epsilon equations exactly. No walls (d is not used by k-epsilon);
    // Dirichlet values at bottom (503) / top (507), symmetry at the sides.
    const double u_tau = 1.0, y0 = 0.5, nu = 1e-8, H = 1.0;
    const double kappa = kKappaEps;
    const double k_exact = u_tau * u_tau / std::sqrt(KS::c_mu);
    auto e_exact = [&](double y) { return u_tau * u_tau * u_tau / (kappa * (y + y0)); };

    struct Errors {
        double k = 0.0, eps = 0.0;
    };
    auto solve = [&](int ny) {
        Mesh m = turb_grid(2, ny, 0.25, H);
        set_flags_by_side_id(m, {{1, 503}, {2, 506}, {3, 507}, {4, 506}});
        const double h = H / ny;
        const double gamma_max = kappa * u_tau * (H + y0) / KS::sigma_k;
        TurbSetup s(std::move(m), nu, 0.2 * h * h / gamma_max);
        TurbulenceConfig c = model_config(TurbulenceModelType::KEpsilon);
        TurbulenceBoundaryOverride bottom = boundary_override(503, BoundaryKind::Inlet);
        bottom.k = k_exact;
        bottom.epsilon = e_exact(0.0);
        TurbulenceBoundaryOverride top = boundary_override(507, BoundaryKind::Inlet);
        top.k = k_exact;
        top.epsilon = e_exact(H);
        c.boundaries.push_back(bottom);
        c.boundaries.push_back(top);
        c.inlet.k = 0.5 * k_exact;
        c.nut_ratio_max = 1e12; // nu = 1e-8: the default cap (1e5 nu) would clip nu_t ~ 0.2
        c.inlet.epsilon = e_exact(0.5 * H);
        std::vector<double> u(s.mesh.npoin), v(s.mesh.npoin, 0.0);
        for (int ip = 0; ip < s.mesh.npoin; ++ip) u[ip] = u_tau / kappa * std::log((s.mesh.y[ip] + y0) / y0);
        StandardKEpsilon model(c);
        model.initialize(s.ctx(), {u, v});
        for (int n = 0; n < 400000; ++n) {
            model.advance(s.ctx(), {u, v});
            const auto r = model.residuals();
            if (std::max(r[0].relative_change, r[1].relative_change) < 1e-14) break;
        }
        Errors e;
        for (int ip = 0; ip < s.mesh.npoin; ++ip) {
            e.k = std::max(e.k, std::abs(model.k()[ip] - k_exact) / k_exact);
            e.eps = std::max(e.eps, std::abs(model.epsilon()[ip] - e_exact(s.mesh.y[ip])) / e_exact(s.mesh.y[ip]));
        }
        return e;
    };
    // As for SST: k second order; max |eps error| ~ order 1.7, limited by the
    // natural inflow side boundary (measured 1.76, 1.72, 1.59 on 10/20/40/80 cells).
    const Errors e20 = solve(20), e40 = solve(40);
    const double order_k = std::log2(e20.k / e40.k), order_e = std::log2(e20.eps / e40.eps);
    INFO("k error " << e20.k << " -> " << e40.k << " (order " << order_k << "), eps error " << e20.eps << " -> "
                    << e40.eps << " (order " << order_e << ")");
    CHECK(e40.k < 5e-4);
    CHECK(e40.eps < 1e-3);
    CHECK(order_k > 1.9);
    CHECK(order_e > 1.6);
}

// ---------------------------------------------------------------------------
// Wall functions
// ---------------------------------------------------------------------------

TEST_CASE("Log-law wall function formulas", "[kepsilon][wall_function]") {
    const double lam = wallfn::y_plus_laminar();
    CHECK_THAT(lam, WithinRel(11.530107402304532, 1e-12));
    // Both layers agree at y+_lam; nu_t,w vanishes there and below.
    CHECK_THAT(std::log(9.8 * lam) / 0.41, WithinRel(lam, 1e-12));
    CHECK(wallfn::u_plus(5.0) == 5.0);
    CHECK_THAT(wallfn::u_plus(100.0), WithinRel(std::log(980.0) / 0.41, 1e-15));
    CHECK(wallfn::nut_wall(5.0, 1e-3, lam) == 0.0);
    CHECK_THAT(wallfn::nut_wall(lam * (1 + 1e-9), 1e-3, lam), WithinAbs(0.0, 1e-11));
    CHECK_THAT(wallfn::nut_wall(100.0, 1e-3, lam), WithinRel(1e-3 * 4.952767630213662, 1e-12));
    CHECK_THAT(wallfn::friction_velocity(0.04, 0.09), WithinRel(std::pow(0.09, 0.25) * 0.2, 1e-15));
    CHECK_THAT(wallfn::epsilon_wall(0.04, 0.01, 0.09), WithinRel(std::pow(0.09, 0.75) * 0.008 / (0.41 * 0.01), 1e-14));
    // Consistency: (nu + nu_t,w) U_p / y_p = u_tau^2 for U_p from the log law.
    const double nu = 1e-5, u_tau = 0.05, y_p = 0.05, y_plus = u_tau * y_p / nu;
    const double u_p = u_tau * wallfn::u_plus(y_plus);
    CHECK_THAT((nu + wallfn::nut_wall(y_plus, nu, lam)) * u_p / y_p, WithinRel(u_tau * u_tau, 1e-12));
}

TEST_CASE("Standard k-epsilon: wall-function boundary treatment", "[kepsilon][wall_function]") {
    // Couette flow u = gamma y between walls (y = 0 and y = ly).
    const double nu = 1e-4, gamma = 2.0, lx = 1.0, ly = 0.5;
    const int nx = 4, ny = 5;
    TurbSetup s(turb_channel(nx, ny, lx, ly), nu, 1e-3);
    std::vector<double> u(s.mesh.npoin), v(s.mesh.npoin, 0.0);
    for (int ip = 0; ip < s.mesh.npoin; ++ip) u[ip] = gamma * s.mesh.y[ip];
    TurbulenceConfig c = model_config(TurbulenceModelType::KEpsilon);
    c.inlet.k = 0.01;
    c.inlet.epsilon = 1e-3;
    StandardKEpsilon model(c);
    model.initialize(s.ctx(), {u, v});
    const double y_p = ly / ny;

    // k is natural at walls (no Dirichlet entries at pure wall nodes), eps is
    // set from the local k.
    for (int ip : model.boundary_conditions(0).nodes) CHECK((!s.walls.is_wall_node(ip) || s.mesh.x[ip] == 0.0));
    model.advance(s.ctx(), {u, v});
    const auto& k = model.k();
    const auto& e = model.epsilon();
    for (int iw = 0; iw < s.walls.num_wall_nodes(); ++iw) {
        const int ip = s.walls.wall_nodes()[iw];
        CHECK_THAT(e[ip], WithinRel(wallfn::epsilon_wall(k[ip], s.walls.wall_node_yp(iw), 0.09), 1e-12));
    }
    // Wall-adjacent elements carry the log-law nu_t; the others the element mean.
    std::vector<int> wall_elem(s.mesh.nelem, 0);
    for (const WallFace& f : s.walls.faces()) {
        ++wall_elem[f.elem];
        const double kf = 0.5 * (k[f.node1] + k[f.node2]);
        const double y_plus = wallfn::friction_velocity(kf, 0.09) * f.yp / nu;
        CHECK_THAT(model.nut_elem()[f.elem], WithinAbs(wallfn::nut_wall(y_plus, nu, model.y_plus_laminar()), 1e-15));
        CHECK(y_plus > model.y_plus_laminar()); // log layer in this set-up
    }
    for (int ie = 0; ie < s.mesh.nelem; ++ie) {
        if (wall_elem[ie]) continue;
        const auto& n = s.mesh.intma[ie];
        CHECK_THAT(model.nut_elem()[ie], WithinRel((model.nut()[n[0]] + model.nut()[n[1]] + model.nut()[n[2]]) / 3.0, 1e-14));
    }
    // Production at the wall nodes: P_w = tau_w u_tau / (kappa y_p) with
    // tau_w = (nu + nu_t,w) gamma from the wall elements (Couette: du/dy = gamma).
    std::vector<double> k_before = model.k(), nut_elem_before = model.nut_elem();
    model.advance(s.ctx(), {u, v});
    const auto& prod = model.production(0);
    for (int iw = 0; iw < s.walls.num_wall_nodes(); ++iw) {
        const int ip = s.walls.wall_nodes()[iw];
        double tau = 0.0, len = 0.0;
        for (int f : s.walls.wall_node_faces(iw)) {
            const WallFace& face = s.walls.face(f);
            tau += (nu + nut_elem_before[face.elem]) * gamma * face.length;
            len += face.length;
        }
        const double expected = tau / len * wallfn::friction_velocity(k_before[ip], 0.09) / (0.41 * y_p);
        CAPTURE(ip);
        CHECK_THAT(prod[ip], WithinRel(expected, 1e-10));
    }
}

TEST_CASE("Standard k-epsilon: wall functions are consistent with the log-law equilibrium",
          "[kepsilon][wall_function][verification]") {
    // Log-layer equilibrium above a wall at y = 0: u = u_tau u+(u_tau y/nu) at
    // the nodes y > 0 (first row at y+ = 250), u = 0 at the wall,
    // k = u_tau^2 / sqrt(C_mu), eps = u_tau^3 / (kappa y). The wall-function
    // treatment must reproduce this state exactly at the wall-function layer:
    //  * wall shear seen by Step 1: (nu + nu_t,w) U_p / y_p = u_tau^2,
    //  * production = dissipation at the wall nodes and the first node row,
    //  * eps at the wall nodes = u_tau^3 / (kappa y_p).
    // (A converged fully developed state is not used: the frozen flow enters
    // through a natural side boundary, where it is not x-invariant.)
    const double u_tau = 0.05, nu = 1e-5, H = 1.0, kappa = 0.41;
    const int ny = 20;
    Mesh m = turb_grid(4, ny, 0.5, H);
    set_flags_by_side_id(m, {{1, 500}, {2, 506}, {3, 507}, {4, 506}});
    const double h = H / ny;
    const double k_eq = u_tau * u_tau / std::sqrt(KS::c_mu);
    TurbSetup s(std::move(m), nu, 1e-6); // tiny step: evaluate the sources of the initial state
    TurbulenceConfig c = model_config(TurbulenceModelType::KEpsilon);
    c.inlet.k = k_eq;
    c.inlet.epsilon = 1.0; // replaced below
    std::vector<double> u(s.mesh.npoin, 0.0), v(s.mesh.npoin, 0.0);
    for (int ip = 0; ip < s.mesh.npoin; ++ip)
        if (s.mesh.y[ip] > 0.0) u[ip] = u_tau * wallfn::u_plus(u_tau * s.mesh.y[ip] / nu);

    StandardKEpsilon model(c);
    model.initialize(s.ctx(), {u, v});
    std::vector<double> eps(s.mesh.npoin);
    for (int ip = 0; ip < s.mesh.npoin; ++ip)
        eps[ip] = u_tau * u_tau * u_tau / (kappa * std::max(s.mesh.y[ip], h));
    model.set_field(s.ctx(), 1, eps);
    model.advance(s.ctx(), {u, v}); // sources evaluated on the equilibrium state

    for (const WallFace& f : s.walls.faces()) {
        const double tau = (nu + model.nut_elem()[f.elem]) * u[f.opposite_node] / f.yp;
        CHECK_THAT(tau, WithinRel(u_tau * u_tau, 1e-10));
    }
    int layer = 0;
    for (int ip = 0; ip < s.mesh.npoin; ++ip) {
        if (s.mesh.y[ip] > h + 1e-12) continue;
        ++layer;
        const double dissipation = u_tau * u_tau * u_tau / (kappa * h);
        CAPTURE(ip, s.mesh.x[ip], s.mesh.y[ip]);
        CHECK_THAT(model.production(0)[ip], WithinRel(dissipation, 1e-9));
        CHECK_THAT(model.destruction(0)[ip] * k_eq, WithinRel(dissipation, 1e-9));
        if (s.mesh.y[ip] == 0.0) CHECK_THAT(model.epsilon()[ip], WithinRel(dissipation, 1e-6));
    }
    CHECK(layer == 2 * 5);
}

// ---------------------------------------------------------------------------
// Model set-up
// ---------------------------------------------------------------------------

TEST_CASE("k-epsilon models: boundary conditions and outputs", "[kepsilon][model]") {
    const double nu = 1e-3;
    TurbSetup s(turb_channel(6, 4, 1.5, 0.5), nu, 1e-3);
    const std::vector<double> u(s.mesh.npoin, 1.0), v(s.mesh.npoin, 0.0);
    TurbulenceConfig c = model_config(TurbulenceModelType::LaunderSharmaKEpsilon);
    c.inlet.intensity = 0.02;
    c.inlet.viscosity_ratio = 5.0;
    LaunderSharmaKEpsilon ls(c);
    ls.initialize(s.ctx(), {u, v});
    const double k_in = 1.5 * 0.02 * 0.02;
    const double e_in = 0.09 * k_in * k_in / (5.0 * nu);
    CHECK_THAT(ls.freestream_values(nu).k, WithinRel(k_in, 1e-14));
    CHECK_THAT(ls.freestream_values(nu).epsilon, WithinRel(e_in, 1e-14));
    for (int ip = 0; ip < s.mesh.npoin; ++ip) {
        if (s.walls.is_wall_node(ip)) {
            CHECK(ls.k()[ip] == 0.0);
            CHECK(ls.epsilon()[ip] == 0.0);
            CHECK(ls.nut()[ip] == 0.0);
        } else if (s.mesh.x[ip] == 0.0) {
            CHECK_THAT(ls.k()[ip], WithinRel(k_in, 1e-14));
            CHECK_THAT(ls.epsilon()[ip], WithinRel(e_in, 1e-14));
            // nu_t/nu = 5 only up to the damping f_mu(R_t).
            const double rt = k_in * k_in / (nu * e_in);
            CHECK_THAT(ls.nut()[ip], WithinRel(5.0 * nu * ke::LaunderSharma::f_mu(rt), 1e-12));
        }
    }
    CHECK(ls.name() == "launder-sharma-k-epsilon");
    const auto f = ls.fields();
    REQUIRE(f.size() == 3);
    CHECK(f[0].name == "k");
    CHECK(f[1].name == "epsilon_tilde");
    CHECK(f[2].name == "nut");

    StandardKEpsilon st(model_config(TurbulenceModelType::KEpsilon));
    st.initialize(s.ctx(), {u, v});
    CHECK(st.name() == "k-epsilon");
    CHECK(st.fields()[1].name == "epsilon");
}

TEST_CASE("k-epsilon models keep k and epsilon non-negative for large time steps", "[kepsilon][positivity]") {
    const double nu = 1e-4;
    TurbSetup s(turb_channel(8, 6, 2.0, 1.0), nu, 5.0); // far beyond any stability limit
    std::vector<double> u(s.mesh.npoin), v(s.mesh.npoin);
    for (int ip = 0; ip < s.mesh.npoin; ++ip) {
        u[ip] = std::sin(3.0 * s.mesh.y[ip]) * 4.0;
        v[ip] = std::cos(2.0 * s.mesh.x[ip]);
    }
    const bool low_re = GENERATE(false, true);
    CAPTURE(low_re);
    auto check = [&](const auto& model) {
        for (int ip = 0; ip < s.mesh.npoin; ++ip) {
            CHECK(model.k()[ip] >= 0.0);
            CHECK(model.epsilon()[ip] >= 0.0);
            CHECK(std::isfinite(model.k()[ip]));
        }
    };
    if (low_re) {
        LaunderSharmaKEpsilon model(model_config(TurbulenceModelType::LaunderSharmaKEpsilon));
        model.initialize(s.ctx(), {u, v});
        for (int n = 0; n < 5; ++n) model.advance(s.ctx(), {u, v});
        check(model);
    } else {
        StandardKEpsilon model(model_config(TurbulenceModelType::KEpsilon));
        model.initialize(s.ctx(), {u, v});
        for (int n = 0; n < 5; ++n) model.advance(s.ctx(), {u, v});
        check(model);
    }
}
