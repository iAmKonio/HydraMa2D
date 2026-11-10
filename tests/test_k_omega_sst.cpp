// Menter k-omega SST (2003) with logarithmic omega.
//
// Closure: constants and the log-layer consistency kappa^2 = (beta1/beta* -
// gamma1) sqrt(beta*)/sigma_w1; blending functions F1 and F2 (limits,
// reference values from an independent Python evaluation, monotonicity);
// cross-diffusion floor; shear stress limiter on nu_t; source terms, the
// production limiter and the Newton linearisation of the ln(omega) equation.
//
// Model: wall / inlet boundary conditions, output fields, analytical decay of
// homogeneous isotropic turbulence (F1 = 0: beta2, F1 = 1: beta1), the exact
// log-layer solution with second-order convergence, and a regression test
// for the near-wall omega collapse of the untransformed omega equation.

#include "BoundaryConditions.hpp"
#include "TurbulenceTestUtils.hpp"
#include "WallTestUtils.hpp"
#include "turbulence/KOmegaSST.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using C = sst::Constants;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// k-omega log-layer constant implied by the set-1 coefficients.
const double kKappaOmega = std::sqrt((C::beta1 / C::beta_star - C::gamma1) * std::sqrt(C::beta_star) / C::sigma_w1);

} // namespace

// ---------------------------------------------------------------------------
// Closure functions
// ---------------------------------------------------------------------------

TEST_CASE("SST constants", "[sst]") {
    CHECK(C::beta_star == 0.09);
    CHECK(C::a1 == 0.31);
    CHECK(C::sigma_k1 == 0.85);
    CHECK(C::sigma_w1 == 0.5);
    CHECK(C::beta1 == 0.075);
    CHECK(C::gamma1 == 5.0 / 9.0);
    CHECK(C::sigma_k2 == 1.0);
    CHECK(C::sigma_w2 == 0.856);
    CHECK(C::beta2 == 0.0828);
    CHECK(C::gamma2 == 0.44);
    // The inner (k-omega) set reproduces the log law with kappa ~ 0.41.
    CHECK_THAT(kKappaOmega, WithinRel(0.408248290463863, 1e-12));
    CHECK(sst::blend(1.0, 2.0, 5.0) == 2.0);
    CHECK(sst::blend(0.0, 2.0, 5.0) == 5.0);
    CHECK(sst::blend(0.25, 2.0, 6.0) == 5.0);
    CHECK_THAT(sst::wall_omega(1e-3, 0.01), WithinRel(60.0 * 1e-3 / (0.075 * 1e-4), 1e-15));
}

TEST_CASE("SST blending function F1", "[sst][blending]") {
    SECTION("reference values") {
        CHECK_THAT(sst::f1(1e-2, 50.0, 0.02, 1e-5, 1e-3), WithinRel(0.90941951863331, 1e-12));
        CHECK_THAT(sst::f1(4e-3, 2.0, 0.3, 1e-5, 5e-2), WithinRel(0.9546409838547052, 1e-12));
        CHECK(sst::f1(1e-2, 10.0, 0.05, 1e-4, 2e-1) == 1.0);
    }
    SECTION("limits: 1 at and near walls, 0 far away and without walls") {
        CHECK(sst::f1(1e-2, 10.0, 0.0, 1e-4, 1e-3) == 1.0);
        CHECK(sst::f1(1e-2, 10.0, -1.0, 1e-4, 1e-3) == 1.0);
        CHECK(sst::f1(1e-2, 10.0, 1e-6, 1e-4, 1e-3) == 1.0);
        CHECK(sst::f1(1e-2, 10.0, kInf, 1e-4, 1e-3) == 0.0);
        CHECK(sst::f1(1e-2, 10.0, 50.0, 1e-6, 1e-3) < 1e-6);
        CHECK(sst::f1(0.0, 10.0, 0.5, 1e-4, 1e-3) >= 0.0);
    }
    SECTION("monotone decreasing with the wall distance") {
        double previous = 1.0;
        for (double d = 1e-4; d < 10.0; d *= 1.3) {
            const double f = sst::f1(1e-2, 5.0, d, 1e-5, 1e-3);
            CHECK(f <= previous + 1e-15);
            previous = f;
        }
    }
    SECTION("the cross-diffusion argument limits F1 and CD_kw is floored") {
        // With a large CD_kw the third argument becomes active and reduces F1.
        CHECK(sst::f1(1e-2, 5.0, 0.1, 1e-5, 1e3) < sst::f1(1e-2, 5.0, 0.1, 1e-5, 1e-10));
        CHECK(sst::cross_diffusion(-5.0) == C::cd_min);
        CHECK(sst::cross_diffusion(0.0) == C::cd_min);
        CHECK_THAT(sst::cross_diffusion(2.0), WithinRel(2.0 * C::sigma_w2 * 2.0, 1e-15));
    }
}

TEST_CASE("SST blending function F2", "[sst][blending]") {
    CHECK_THAT(sst::f2(1e-2, 50.0, 0.02, 1e-5), WithinRel(0.9998972742141851, 1e-12));
    CHECK_THAT(sst::f2(4e-3, 2.0, 0.3, 1e-5), WithinRel(0.9999657151455275, 1e-12));
    CHECK(sst::f2(1e-2, 10.0, 0.0, 1e-4) == 1.0);
    CHECK(sst::f2(1e-2, 10.0, kInf, 1e-4) == 0.0);
    double previous = 1.0;
    for (double d = 1e-4; d < 50.0; d *= 1.3) {
        const double f = sst::f2(1e-2, 5.0, d, 1e-5);
        CHECK(f <= previous + 1e-15);
        CHECK(f >= sst::f1(1e-2, 5.0, d, 1e-5, 1e-10) - 1e-12); // F2 extends further than F1
        previous = f;
    }
}

TEST_CASE("SST shear stress limiter on the eddy viscosity", "[sst][limiter]") {
    const double k = 1e-2, omega = 5.0;
    // a1 omega > S F2: nu_t = k / omega.
    CHECK_THAT(sst::eddy_viscosity(k, omega, 1.0, 0.9), WithinRel(k / omega, 1e-15));
    CHECK_THAT(sst::eddy_viscosity(k, omega, 1.0, 0.9), WithinRel(0.002, 1e-15));
    // a1 omega < S F2: nu_t = a1 k / (S F2) (Bradshaw: -u'v' = nu_t S <= a1 k).
    CHECK_THAT(sst::eddy_viscosity(k, omega, 10.0, 0.9), WithinRel(0.0003444444444444444, 1e-14));
    for (double s = 0.0; s < 100.0; s += 0.5) {
        const double nut = sst::eddy_viscosity(k, omega, s, 1.0);
        CHECK(nut <= k / omega * (1.0 + 1e-15));
        CHECK(nut * s <= C::a1 * k * (1.0 + 1e-12));
    }
    // Continuous at the switch a1 omega = S F2.
    const double s_switch = C::a1 * omega / 0.8;
    CHECK_THAT(sst::eddy_viscosity(k, omega, s_switch * (1 + 1e-12), 0.8), WithinRel(k / omega, 1e-9));
    CHECK(sst::eddy_viscosity(0.0, omega, 1.0, 1.0) == 0.0);
    CHECK(std::isfinite(sst::eddy_viscosity(k, 0.0, 0.0, 0.0)));
}

TEST_CASE("SST source terms at reference points", "[sst][source]") {
    SECTION("blended coefficients, cross diffusion") {
        const sst::PointState s{1e-2, 20.0, 1e-5, 4e-4, 15.0, 0.3, 0.5, 4.0, 2e-4};
        const sst::SourceTerms t = sst::source(s, 10.0);
        CHECK_FALSE(t.limited);
        CHECK_THAT(t.production, WithinRel(0.09, 1e-14));
        CHECK_THAT(t.k_destruction, WithinRel(0.018, 1e-14));
        CHECK_THAT(t.omega_production, WithinRel(106.8, 1e-14));
        CHECK_THAT(t.omega_destruction, WithinRel(32.184, 1e-14));
        CHECK_THAT(t.cross_diffusion, WithinRel(11.984, 1e-13));
        CHECK_THAT(t.log_omega_source, WithinRel(4.3308, 1e-13));
    }
    SECTION("production limiter P_k <= 10 beta* k omega") {
        const sst::PointState s{1e-2, 2.0, 1e-5, 4e-2, 40.0, 0.8, -0.2, 1.0, 1e-2};
        const sst::SourceTerms t = sst::source(s, 10.0);
        CHECK(t.limited);
        CHECK_THAT(t.production, WithinRel(10.0 * C::beta_star * 1e-2 * 2.0, 1e-14));
        CHECK_THAT(t.cross_diffusion, WithinRel(-0.13696, 1e-13));
        CHECK_THAT(t.log_omega_source, WithinRel(425.7439555555556, 1e-13));
        const sst::SourceTerms off = sst::source(s, 0.0); // limiter switched off
        CHECK_FALSE(off.limited);
        CHECK_THAT(off.production, WithinRel(4e-2 * 1600.0, 1e-14));
    }
    SECTION("no cross diffusion near walls (F1 = 1)") {
        const sst::SourceTerms t = sst::source({1e-2, 20.0, 1e-5, 4e-4, 15.0, 1.0, 3.0, 0.0, 0.0}, 10.0);
        CHECK(t.cross_diffusion == 0.0);
    }
}

TEST_CASE("SST linearisations reproduce the sources", "[sst][source]") {
    int cases = 0;
    for (double k : {0.0, 1e-6, 1e-2, 1.0}) {
        for (double w : {1e-3, 1.0, 1e4}) {
            for (double f1 : {0.0, 0.6, 1.0}) {
                for (double dot : {-3.0, 0.0, 2.0}) {
                    const sst::PointState s{k, w, 1e-5, k / w, 7.0, f1, dot, 1.5, 1e-3};
                    const sst::SourceTerms t = sst::source(s, 10.0);
                    CAPTURE(k, w, f1, dot);
                    CHECK(t.k.D >= 0.0);
                    CHECK(t.log_omega.D >= 0.0);
                    CHECK_THAT(t.k.value(k), WithinAbs(t.production - t.k_destruction, 1e-12 * (1 + t.production)));
                    CHECK_THAT(t.log_omega.value(std::log(w)),
                               WithinAbs(t.log_omega_source, 1e-11 * (1 + std::abs(t.log_omega_source))));
                    // Consistency of the log-form source with the omega equation:
                    // (omega sources)/omega + Gamma_w |grad ln omega|^2.
                    CHECK_THAT(t.log_omega_source,
                               WithinAbs((t.omega_production - t.omega_destruction + t.cross_diffusion) / w + 1e-3 * 1.5,
                                         1e-11 * (1 + std::abs(t.log_omega_source))));
                    ++cases;
                }
            }
        }
    }
    CHECK(cases == 108);
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

TEST_CASE("SST model: wall and inlet boundary conditions", "[sst][model]") {
    const double nu = 1e-3, lx = 1.5, ly = 0.5;
    const int nx = 6, ny = 5;
    TurbSetup s(turb_channel(nx, ny, lx, ly), nu, 1e-3);
    const std::vector<double> u(s.mesh.npoin, 1.0), v(s.mesh.npoin, 0.0);
    TurbulenceConfig c = model_config(TurbulenceModelType::KOmegaSST);
    c.inlet.intensity = 0.04;
    c.inlet.length_scale = 0.05;
    KOmegaSST model(c);
    model.initialize(s.ctx(), {u, v});

    const double k_in = 1.5 * 0.04 * 0.04;
    const double eps_in = std::pow(0.09, 0.75) * std::pow(k_in, 1.5) / 0.05;
    const double w_in = eps_in / (0.09 * k_in);
    CHECK_THAT(model.freestream_values(nu).k, WithinRel(k_in, 1e-14));
    CHECK_THAT(model.freestream_values(nu).omega, WithinRel(w_in, 1e-14));
    const double w_wall = 60.0 * nu / (C::beta1 * (ly / ny) * (ly / ny));

    int walls = 0, inlets = 0;
    for (int ip = 0; ip < s.mesh.npoin; ++ip) {
        if (s.walls.is_wall_node(ip)) {
            ++walls;
            CHECK(model.k()[ip] == 0.0);
            CHECK_THAT(model.omega()[ip], WithinRel(w_wall, 1e-12));
            CHECK_THAT(model.log_omega()[ip], WithinRel(std::log(w_wall), 1e-14));
        } else if (s.mesh.x[ip] == 0.0) {
            ++inlets;
            CHECK_THAT(model.k()[ip], WithinRel(k_in, 1e-14));
            CHECK_THAT(model.omega()[ip], WithinRel(w_in, 1e-12));
        }
    }
    CHECK(walls == 2 * (nx + 1));
    CHECK(inlets == ny - 1);

    SECTION("fields, state and names") {
        CHECK(model.name() == "k-omega-sst");
        const auto f = model.fields();
        REQUIRE(f.size() == 4);
        CHECK(f[0].name == "k");
        CHECK(f[1].name == "omega");
        CHECK(f[2].name == "nut");
        CHECK(f[3].name == "sst_f1");
        const auto st = model.state_fields();
        REQUIRE(st.size() == 2);
        CHECK(st[0].name == "k");
        CHECK(st[1].name == "log_omega");
        for (int ip = 0; ip < s.mesh.npoin; ++ip)
            CHECK_THAT(model.omega()[ip], WithinRel(std::exp(model.log_omega()[ip]), 1e-15));
        // F1 = 1 at the walls.
        for (int ip : s.walls.wall_nodes()) CHECK(model.f1()[ip] == 1.0);
    }

    SECTION("direct inlet values and the viscosity ratio") {
        TurbulenceConfig d = c;
        TurbulenceBoundaryOverride b = boundary_override(503, BoundaryKind::Inlet);
        b.k = 2e-3;
        b.omega = 7.0;
        d.boundaries.push_back(b);
        d.inlet.viscosity_ratio = 20.0;
        KOmegaSST m2(d);
        m2.initialize(s.ctx(), {u, v});
        CHECK(m2.inlet_values(503, nu).k == 2e-3);
        CHECK(m2.inlet_values(503, nu).omega == 7.0);
        // Free stream from nu_t / nu = 20: omega = k / (20 nu).
        CHECK_THAT(m2.freestream_values(nu).omega, WithinRel(k_in / (20.0 * nu), 1e-12));
        const int inlet_node = find_node(s.mesh, 0.0, 0.2);
        REQUIRE(inlet_node >= 0);
        CHECK(m2.k()[inlet_node] == 2e-3);
        CHECK_THAT(m2.omega()[inlet_node], WithinRel(7.0, 1e-14));
    }
}

TEST_CASE("SST model: analytical decay of homogeneous isotropic turbulence", "[sst][decay]") {
    // No velocity, uniform k0, omega0:  dk/dt = -beta* k omega,  dw/dt = -beta w^2
    //   omega(t) = omega0 / (1 + beta omega0 t),  k(t) = k0 (1 + beta omega0 t)^(-beta*/beta),
    // beta = beta2 without walls (F1 = 0), beta1 close to a wall (F1 = 1).
    const bool near_wall = GENERATE(false, true);
    CAPTURE(near_wall);
    const double nu = 1e-5, k0 = 1e-2, w0 = 10.0, t_end = 1.0;
    const double beta = near_wall ? C::beta1 : C::beta2;

    auto run = [&](int steps, double& k_end, double& w_end, double& k_rec, double& w_rec) {
        const double dt = t_end / steps;
        const ConstantDistance near(1e-3);
        TurbSetup s = near_wall ? TurbSetup(turb_grid(3, 2, 1.0, 1.0, 504), nu, dt, near)
                                : TurbSetup(turb_grid(3, 2, 1.0, 1.0, 504), nu, dt);
        const std::vector<double> zero(s.mesh.npoin, 0.0);
        TurbulenceConfig c = model_config(TurbulenceModelType::KOmegaSST);
        c.inlet.k = k0;
        c.inlet.omega = w0;
        KOmegaSST model(c);
        model.initialize(s.ctx(), {zero, zero});
        REQUIRE(model.boundary_conditions(0).empty());
        CHECK(model.f1()[0] == (near_wall ? 1.0 : 0.0));
        // Discrete recurrence: implicit k decay, Newton step for ln(omega).
        k_rec = k0;
        double lw = std::log(w0);
        for (int n = 0; n < steps; ++n) {
            model.advance(s.ctx(), {zero, zero});
            const double w = std::exp(lw);
            k_rec = k_rec / (1.0 + dt * C::beta_star * w);
            lw = lw - dt * beta * w / (1.0 + dt * beta * w);
        }
        w_rec = std::exp(lw);
        k_end = model.k()[0];
        w_end = model.omega()[0];
        for (int ip = 0; ip < s.mesh.npoin; ++ip) { // homogeneous
            CHECK_THAT(model.k()[ip], WithinRel(k_end, 1e-12));
            CHECK_THAT(model.omega()[ip], WithinRel(w_end, 1e-12));
        }
    };

    const double w_exact = w0 / (1.0 + beta * w0 * t_end);
    const double k_exact = k0 * std::pow(1.0 + beta * w0 * t_end, -C::beta_star / beta);
    double k1, w1, k1r, w1r, k2, w2, k2r, w2r;
    run(200, k1, w1, k1r, w1r);
    run(400, k2, w2, k2r, w2r);
    // The model follows the discrete recurrence ...
    CHECK_THAT(k1, WithinRel(k1r, 1e-10));
    CHECK_THAT(w1, WithinRel(w1r, 1e-10));
    // ... which converges to the analytical solution at first order in dt.
    CHECK_THAT(k2, WithinRel(k_exact, 5e-3));
    CHECK_THAT(w2, WithinRel(w_exact, 5e-3));
    CHECK_THAT(std::abs(k1 - k_exact) / std::abs(k2 - k_exact), WithinAbs(2.0, 0.1));
    CHECK_THAT(std::abs(w1 - w_exact) / std::abs(w2 - w_exact), WithinAbs(2.0, 0.1));
}

TEST_CASE("SST model: exact log-layer solution with second-order convergence", "[sst][model][verification]") {
    // Virtual wall at y = -y0 (d = y + y0), frozen log-law velocity
    // u = (u_tau/kappa) ln(d/y0) with kappa = kappa_omega. Then F1 = 1, the
    // limiters are inactive and k = u_tau^2/sqrt(beta*), omega = u_tau/(sqrt(beta*) kappa d)
    // solve the SST equations exactly (production = destruction in both).
    const double u_tau = 1.0, y0 = 0.5, nu = 1e-8, H = 1.0;
    const double kappa = kKappaOmega;
    const double k_exact = u_tau * u_tau / std::sqrt(C::beta_star);
    auto w_exact = [&](double y) { return u_tau / (std::sqrt(C::beta_star) * kappa * (y + y0)); };

    struct Errors {
        double k = 0.0, omega = 0.0;
    };
    auto solve = [&](int ny) {
        Mesh m = turb_grid(2, ny, 0.25, H);
        set_flags_by_side_id(m, {{1, 503}, {2, 506}, {3, 507}, {4, 506}});
        const VirtualWallDistance distance(y0);
        const double h = H / ny;
        const double gamma_max = C::sigma_k1 * kappa * u_tau * (H + y0);
        TurbSetup s(std::move(m), nu, 0.2 * h * h / gamma_max, distance);

        TurbulenceConfig c = model_config(TurbulenceModelType::KOmegaSST);
        TurbulenceBoundaryOverride bottom = boundary_override(503, BoundaryKind::Inlet);
        bottom.k = k_exact;
        bottom.omega = w_exact(0.0);
        TurbulenceBoundaryOverride top = boundary_override(507, BoundaryKind::Inlet);
        top.k = k_exact;
        top.omega = w_exact(H);
        c.boundaries.push_back(bottom);
        c.boundaries.push_back(top);
        c.inlet.k = 0.5 * k_exact; // initial field far from the solution
        c.nut_ratio_max = 1e12;    // nu = 1e-8: the default cap (1e5 nu) would clip nu_t ~ 0.2
        c.inlet.omega = w_exact(0.5 * H);
        std::vector<double> u(s.mesh.npoin), v(s.mesh.npoin, 0.0);
        for (int ip = 0; ip < s.mesh.npoin; ++ip) u[ip] = u_tau / kappa * std::log((s.mesh.y[ip] + y0) / y0);

        KOmegaSST model(c);
        model.initialize(s.ctx(), {u, v});
        for (int n = 0; n < 400000; ++n) {
            model.advance(s.ctx(), {u, v});
            const auto r = model.residuals();
            if (std::max(r[0].relative_change, r[1].relative_change) < 1e-14) break;
        }
        Errors e;
        for (int ip = 0; ip < s.mesh.npoin; ++ip) {
            e.k = std::max(e.k, std::abs(model.k()[ip] - k_exact) / k_exact);
            e.omega = std::max(e.omega, std::abs(model.omega()[ip] - w_exact(s.mesh.y[ip])) / w_exact(s.mesh.y[ip]));
            CHECK(model.f1()[ip] > 0.999999);
        }
        return e;
    };

    // k converges at second order. omega ~ 1/d is second order in the
    // interior; at the side boundaries the frozen flow enters through a
    // natural (symmetry) boundary, the discrete solution is not exactly
    // x-invariant there and the observed order of max |omega error| is ~1.7
    // (measured 1.71, 1.70, 1.58 on 10/20/40/80 cells).
    const Errors e20 = solve(20), e40 = solve(40);
    const double order_k = std::log2(e20.k / e40.k), order_w = std::log2(e20.omega / e40.omega);
    INFO("k error " << e20.k << " -> " << e40.k << " (order " << order_k << "), omega error " << e20.omega << " -> "
                    << e40.omega << " (order " << order_w << ")");
    CHECK(e40.k < 5e-4);
    CHECK(e40.omega < 1e-3);
    CHECK(order_k > 1.9);
    CHECK(order_w > 1.6);
}

TEST_CASE("SST model: omega stays positive and bounded near walls on the benchmark mesh", "[sst][model][hinterkante]") {
    // Regression: with omega (instead of ln omega) as the transported variable,
    // convection next to the steep near-wall omega profile drove omega to zero
    // at interior nodes within ~10 steps, and nu_t ~ 1/omega then blew up.
    Mesh m = benchmark_mesh();
    const double nu = 1.0 / 200.0;
    TurbSetup s(std::move(m), nu, 1.0);
    std::vector<double> u(s.mesh.npoin, 1.0), v(s.mesh.npoin, 0.0);
    apply_velocity_bcs(s.mesh, u, v);
    std::vector<double> speed(s.mesh.npoin);
    for (int ip = 0; ip < s.mesh.npoin; ++ip) speed[ip] = std::sqrt(u[ip] * u[ip] + v[ip] * v[ip] + 1e-16);

    KOmegaSST model(model_config(TurbulenceModelType::KOmegaSST));
    model.initialize(s.ctx(), {u, v});
    const double w_inf = model.freestream_values(nu).omega;
    for (int n = 0; n < 100; ++n) {
        const auto& g = model.max_diffusivity_elem();
        std::vector<double> nu_eff(s.mesh.nelem);
        for (int ie = 0; ie < s.mesh.nelem; ++ie) nu_eff[ie] = std::max(nu + model.nut_elem()[ie], g[ie]);
        compute_timestep(1, -1, 1, 1.0, nu_eff, 0.6, s.mesh, speed, s.delte);
        update_lhs(s.mesh, s.delte, s.elcoe2, s.pdiag, s.gstif);
        model.advance(s.ctx(), {u, v});
    }
    double min_w = kInf, max_nut = 0.0;
    for (int ip = 0; ip < s.mesh.npoin; ++ip) {
        REQUIRE(std::isfinite(model.k()[ip]));
        REQUIRE(std::isfinite(model.omega()[ip]));
        CHECK(model.k()[ip] >= 0.0);
        min_w = std::min(min_w, model.omega()[ip]);
        max_nut = std::max(max_nut, model.nut()[ip]);
    }
    INFO("min omega / omega_inf = " << min_w / w_inf << ", max nu_t / nu = " << max_nut / nu);
    CHECK(min_w > 0.1 * w_inf);
    CHECK(max_nut < 10.0 * nu);
}
