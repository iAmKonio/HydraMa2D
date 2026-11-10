// Spalart-Allmaras model with negative-SA extension.
//
// Pointwise: closure constants, fv1, fv2, fw, ft2, fn, the S~ limiter, the
// eddy viscosity and diffusivity, source terms at hand-computed points
// (reference values from an independent Python evaluation of the formulas in
// Allmaras, Johnson & Spalart 2012), the linearisation used by the
// point-implicit update, the log-layer equilibrium and the negative branch.
//
// Model: initial and boundary conditions (incl. config overrides), exact
// production ODE in homogeneous shear (with and without ft2), free-stream
// preservation, relaxation of negative nu~, outputs, and a smoke run on the
// benchmark mesh.

#include "BoundaryConditions.hpp"
#include "LinearSolver.hpp"
#include "TurbulenceTestUtils.hpp"
#include "WallTestUtils.hpp"
#include "turbulence/SpalartAllmaras.hpp"

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
using C = sa::Constants;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Mesh + wall geometry + time step data for driving the model directly.
struct Setup {
    Mesh mesh;
    WallGeometry walls;
    double nu = 1.0e-3;
    std::vector<double> delte, elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;

    Setup(Mesh m, double nu_, double dt) : mesh(std::move(m)), nu(nu_) {
        compute_all_geometry(mesh);
        walls = WallGeometry(mesh);
        set_dt(dt);
    }
    void set_dt(double dt) {
        delte.assign(mesh.nelem, dt);
        update_lhs(mesh, delte, elcoe2, pdiag, gstif); // elcoe2 exactly as in the solver
    }
    TurbulenceContext ctx() const { return {mesh, walls, nu, delte, elcoe2}; }
};

Mesh grid(int nx, int ny, double lx, double ly, int flag = 500) {
    GridSpec g;
    g.lx = lx;
    g.ly = ly;
    g.nx = nx;
    g.ny = ny;
    return make_grid_mesh(g, flag);
}

// Channel: bottom/top walls, left inlet, right outlet.
Mesh channel(int nx, int ny, double lx, double ly) {
    Mesh m = grid(nx, ny, lx, ly);
    set_flags_by_side_id(m, {{1, 500}, {2, 504}, {3, 500}, {4, 503}});
    return m;
}

TurbulenceConfig sa_config() {
    TurbulenceConfig c;
    c.present = true;
    c.model = TurbulenceModelType::SpalartAllmaras;
    return c;
}

} // namespace

// ---------------------------------------------------------------------------
// Closure functions
// ---------------------------------------------------------------------------

TEST_CASE("SA closure constants", "[sa]") {
    CHECK(C::cb1 == 0.1355);
    CHECK(C::sigma == 2.0 / 3.0);
    CHECK(C::cb2 == 0.622);
    CHECK(C::kappa == 0.41);
    CHECK_THAT(C::cw1, WithinRel(3.2390678167757287, 1e-15)); // cb1/kappa^2 + (1 + cb2)/sigma
    CHECK(C::cw2 == 0.3);
    CHECK(C::cw3 == 2.0);
    CHECK(C::cv1 == 7.1);
    CHECK(C::cv2 == 0.7);
    CHECK(C::cv3 == 0.9);
    CHECK(C::ct3 == 1.2);
    CHECK(C::ct4 == 0.5);
    CHECK(C::cn1 == 16.0);
}

TEST_CASE("SA damping functions fv1, fv2, ft2, fn", "[sa]") {
    CHECK(sa::fv1(0.0) == 0.0);
    CHECK_THAT(sa::fv1(C::cv1), WithinRel(0.5, 1e-15));
    CHECK_THAT(sa::fv1(1.0), WithinRel(0.0027862060510823023, 1e-14));
    CHECK_THAT(sa::fv1(1.0e4), WithinAbs(1.0, 1e-9));
    for (double chi = 0.0; chi < 50.0; chi += 0.5) CHECK(sa::fv1(chi + 0.5) > sa::fv1(chi));

    CHECK(sa::fv2(0.0, sa::fv1(0.0)) == 1.0);
    CHECK_THAT(sa::fv2(10.0, sa::fv1(10.0)), WithinRel(-0.19556404342312583, 1e-14));
    CHECK_THAT(sa::fv2(1.0e6, sa::fv1(1.0e6)), WithinAbs(0.0, 1e-5)); // ~ 1/chi

    CHECK(sa::ft2(0.0) == C::ct3);
    CHECK_THAT(sa::ft2(2.0), WithinRel(0.16240233988393524, 1e-14));
    CHECK(sa::ft2(20.0) < 1e-80);

    CHECK(sa::fn(0.0) == 1.0);
    CHECK_THAT(sa::fn(-2.0), WithinRel(1.0 / 3.0, 1e-15));
    CHECK_THAT(sa::fn(-1.0e3), WithinAbs(-1.0, 1e-7));
}

TEST_CASE("SA wall function fw", "[sa]") {
    CHECK(sa::fw(0.0) == 0.0);
    CHECK_THAT(sa::fw(1.0), WithinRel(1.0, 1e-15)); // log-layer equilibrium
    CHECK_THAT(sa::fw(0.47429873864969324), WithinRel(0.3362910709879365, 1e-13));
    CHECK_THAT(sa::fw(1.568625391927122), WithinRel(2.004457339097171, 1e-13));
    // Saturates at (1 + cw3^6)^(1/6) = 65^(1/6) at r = r_max.
    CHECK_THAT(sa::fw(C::r_max), WithinRel(std::pow(65.0, 1.0 / 6.0), 1e-12));
    // Strictly increasing for small r, saturating (to round-off) for large r.
    for (double r = 0.0; r < 1.5; r += 0.05) CHECK(sa::fw(r + 0.05) > sa::fw(r));
    for (double r = 1.5; r < C::r_max; r += 0.05) CHECK(sa::fw(r + 0.05) >= sa::fw(r) * (1.0 - 1e-14));
}

TEST_CASE("SA modified vorticity S~ with the Allmaras-Johnson-Spalart limiter", "[sa]") {
    const double omega = GENERATE(0.0, 0.5, 20.0);
    CAPTURE(omega);
    // Unmodified branch.
    CHECK(sa::s_tilde(omega, 0.3) == omega + 0.3);
    CHECK(sa::s_tilde(omega, -C::cv2 * omega) == omega - C::cv2 * omega);
    if (omega == 0.0) {
        CHECK(sa::s_tilde(0.0, -1.0) == 0.0);
        return;
    }
    // Continuous at S_bar = -cv2 Omega, monotone, bounded below by (1 - cv3) Omega.
    const double edge = -C::cv2 * omega;
    CHECK_THAT(sa::s_tilde(omega, edge * (1.0 + 1e-9)), WithinRel(sa::s_tilde(omega, edge), 1e-7));
    double previous = sa::s_tilde(omega, -1.0e6 * omega);
    CHECK_THAT(previous, WithinRel((1.0 - C::cv3) * omega, 1e-5));
    for (double s_bar = -1.0e3 * omega; s_bar < 2.0 * omega; s_bar += 0.01 * omega) {
        const double st = sa::s_tilde(omega, s_bar);
        CHECK(st > (1.0 - C::cv3) * omega * (1.0 - 1e-12));
        CHECK(st >= previous * (1.0 - 1e-12));
        previous = st;
    }
}

TEST_CASE("SA eddy viscosity, diffusivity and inlet ratio conversion", "[sa]") {
    const double nu = 2.0e-3;
    CHECK(sa::eddy_viscosity(0.0, nu) == 0.0);
    CHECK(sa::eddy_viscosity(-5.0 * nu, nu) == 0.0); // negative SA: nu_t = 0
    CHECK_THAT(sa::eddy_viscosity(7.1 * nu, nu), WithinRel(0.5 * 7.1 * nu, 1e-15));
    CHECK_THAT(sa::eddy_viscosity(1.0e5 * nu, nu), WithinRel(1.0e5 * nu, 1e-9));

    CHECK_THAT(sa::diffusivity(3.0 * nu, nu), WithinRel(1.5 * 4.0 * nu, 1e-15));
    CHECK_THAT(sa::diffusivity(-2.0 * nu, nu), WithinRel(1.5 * (1.0 - 2.0 / 3.0) * nu, 1e-14));
    for (double chi = -100.0; chi <= 100.0; chi += 0.25) CHECK(sa::diffusivity(chi * nu, nu) > 0.0);
    CHECK_THAT(sa::diffusivity(-1e-12 * nu, nu), WithinRel(sa::diffusivity(0.0, nu), 1e-10)); // continuous at 0

    // TMR free stream nu~ = 3 nu corresponds to nu_t / nu = 3 fv1(3).
    CHECK_THAT(sa::nu_tilde_ratio_from_viscosity_ratio(0.21043825715555026), WithinRel(3.0, 1e-12));
    for (double r : {0.0, 1e-3, 1.0, 10.0, 1000.0}) {
        const double chi = sa::nu_tilde_ratio_from_viscosity_ratio(r);
        CHECK_THAT(chi * sa::fv1(chi), WithinAbs(r, 1e-12 * std::max(1.0, r)));
    }
    CHECK_THROWS_AS(sa::nu_tilde_ratio_from_viscosity_ratio(-1.0), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Source terms
// ---------------------------------------------------------------------------

TEST_CASE("SA source terms at hand-computed points", "[sa][source]") {
    struct Case {
        sa::PointState s;
        bool ft2;
        double production, destruction, cross, total;
    };
    // Reference values: independent Python evaluation of the [AJS12] formulas.
    const Case c = GENERATE(
        // chi = 2: S_bar < -cv2 Omega (limiter branch), r clipped at 10.
        Case{{2.0, 1.0, 1.0, 1.0, 0.0}, false, 0.028142229184193273, 25.979587936112836, 0.0, -25.95144570692864},
        // limiter branch with ft2 and a cross-diffusion term.
        Case{{5e-3, 1e-3, 0.1, 2.0, 0.04}, true, 0.0001787215719973895, 0.01623723344826496, 0.037320000000000006,
             0.021261488123732436},
        // regular branch, r = 1.57.
        Case{{0.02, 1e-3, 0.05, 30.0, 0.0}, false, 0.08221902523563672, 1.0388117211471295, 0.0, -0.9565926959114928},
        // regular branch, r = 0.47, with ft2.
        Case{{0.02, 1e-3, 0.05, 100.0, 0.0}, true, 0.2719190252356367, 0.17428313361697068, 0.0, 0.09763589161866604},
        // negative SA.
        Case{{-3e-3, 1e-3, 0.2, 5.0, 0.01}, true, 0.00040649999999999996, -0.0007287902587745389, 0.009330000000000001,
             0.01046529025877454});
    CAPTURE(c.s.nu_tilde, c.s.nu, c.s.d, c.s.omega, c.s.grad_sq, c.ft2);

    const sa::SourceTerms t = sa::source(c.s, c.ft2);
    CHECK_THAT(t.production, WithinRel(c.production, 1e-12));
    CHECK_THAT(t.destruction, WithinRel(c.destruction, 1e-12));
    CHECK_THAT(t.cross_diffusion, WithinAbs(c.cross, 1e-15));
    CHECK_THAT(t.total(), WithinRel(c.total, 1e-12));
    CHECK_THAT(t.split.value(c.s.nu_tilde), WithinAbs(c.total, 1e-12 * std::abs(c.total)));
}

TEST_CASE("SA source linearisation reproduces the source with D >= 0", "[sa][source]") {
    const double nu = 1.0e-3;
    int negative_production_coefficients = 0;
    for (double chi : {-50.0, -3.0, -0.1, 0.0, 0.05, 0.5, 1.0, 3.0, 30.0, 3000.0}) {
        for (double d : {1e-4, 1e-2, 0.3, kInf}) {
            for (double omega : {0.0, 0.7, 400.0}) {
                for (double grad_sq : {0.0, 0.02}) {
                    for (bool ft2 : {false, true}) {
                        const sa::PointState s{chi * nu, nu, d, omega, grad_sq};
                        const sa::SourceTerms t = sa::source(s, ft2);
                        CAPTURE(chi, d, omega, grad_sq, ft2);
                        REQUIRE(std::isfinite(t.total()));
                        CHECK(t.split.D >= 0.0);
                        CHECK_THAT(t.split.value(s.nu_tilde), WithinAbs(t.total(), 1e-12 * (1.0 + std::abs(t.total()))));
                        if (chi < 0.0) {
                            // Negative branch: every term pushes nu~ towards zero.
                            CHECK(t.production >= 0.0);
                            CHECK(t.destruction <= 0.0);
                            CHECK(t.split.P == t.cross_diffusion);
                        }
                        if (ft2 && chi > 0.0 && chi < 0.6 && omega > 0.0) {
                            // ft2 > 1: production coefficient < 0, treated implicitly.
                            CHECK(t.production < 0.0);
                            ++negative_production_coefficients;
                        }
                    }
                }
            }
        }
    }
    CHECK(negative_production_coefficients > 0);
}

TEST_CASE("SA sources vanish at walls and without walls / shear", "[sa][source]") {
    const double nu = 1e-3;
    // Wall node: nu~ is prescribed, no source.
    const sa::SourceTerms wall = sa::source({5 * nu, nu, 0.0, 10.0, 1.0}, true);
    CHECK(wall.total() == 0.0);
    CHECK(wall.split.P == 0.0);
    CHECK(wall.split.D == 0.0);
    // Far from walls (d = inf) without vorticity: production = destruction = 0.
    const sa::SourceTerms free = sa::source({3 * nu, nu, kInf, 0.0, 0.0}, true);
    CHECK(free.production == 0.0);
    CHECK(free.destruction == 0.0);
    // d = inf with shear: pure production cb1 (1 - ft2) Omega nu~.
    const sa::SourceTerms shear = sa::source({30 * nu, nu, kInf, 2.0, 0.0}, false);
    CHECK_THAT(shear.production, WithinRel(C::cb1 * 2.0 * 30 * nu, 1e-15));
    CHECK(shear.destruction == 0.0);
}

TEST_CASE("SA log-layer equilibrium: production, destruction and diffusion balance", "[sa][source]") {
    // In the log layer nu~ = kappa u_tau d and Omega = u_tau / (kappa d). The
    // diffusion operator gives (1/sigma)(1 + cb2) |grad nu~|^2 = (1 + cb2) kappa^2 u_tau^2 / sigma
    // (nu~ linear, molecular viscosity negligible). With fv2 -> 0 (chi >> 1)
    // r -> 1, fw -> 1 and the balance holds by the definition of cw1:
    //   cb1 u_tau^2 - cw1 kappa^2 u_tau^2 + (1 + cb2) kappa^2 u_tau^2 / sigma = 0.
    const double u_tau = 0.05, nu = 1.0e-9;
    for (double d : {0.01, 0.1, 0.4}) {
        const double nt = C::kappa * u_tau * d;
        const double grad_sq = C::kappa * C::kappa * u_tau * u_tau;
        const sa::SourceTerms t = sa::source({nt, nu, d, u_tau / (C::kappa * d), grad_sq}, false);
        const double diffusion_without_cb2 = grad_sq / C::sigma; // (1/sigma) nu~ lap(nu~) + |grad nu~|^2 / sigma
        const double residual = t.total() + diffusion_without_cb2;
        CAPTURE(d, t.production, t.destruction);
        CHECK(std::abs(residual) < 1e-4 * C::cb1 * u_tau * u_tau);
        CHECK_THAT(t.production, WithinRel(C::cb1 * u_tau * u_tau, 1e-4));
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

TEST_CASE("SA model: initial field and boundary conditions", "[sa][model]") {
    const StepSpec spec;
    Setup s(make_step_mesh(spec), 1.0e-3, 1.0e-3);
    const std::vector<double> u(s.mesh.npoin, 1.0), v(s.mesh.npoin, 0.0);

    SECTION("defaults: free stream and inlet nu~ = 3 nu, walls nu~ = 0") {
        SpalartAllmaras model(sa_config());
        model.initialize(s.ctx(), {u, v});
        CHECK(model.freestream_nu_tilde(s.nu) == 3.0 * s.nu);
        CHECK(model.inlet_nu_tilde_ratio(503) == 3.0);
        for (int ip = 0; ip < s.mesh.npoin; ++ip) {
            const double expected = s.walls.is_wall_node(ip) ? 0.0 : 3.0 * s.nu;
            CHECK(model.nu_tilde()[ip] == expected);
        }
        const DirichletSet& bc = model.boundary_conditions(0);
        int wall = 0, inlet = 0;
        for (int k = 0; k < bc.size(); ++k) {
            const int ip = bc.nodes[k];
            if (s.walls.is_wall_node(ip)) {
                ++wall;
                CHECK(bc.values[k] == 0.0);
            } else {
                ++inlet;
                CHECK(s.mesh.x[ip] == 0.0);
                CHECK(bc.values[k] == 3.0 * s.nu);
            }
        }
        CHECK(wall == s.walls.num_wall_nodes());
        CHECK(inlet == spec.ny - static_cast<int>(spec.hs / (spec.ly / spec.ny)) - 1); // inlet minus its 2 corners
        // Outlet nodes (not on walls) are natural: no Dirichlet entry.
        for (int ip : bc.nodes) CHECK((s.mesh.x[ip] < spec.lx || s.walls.is_wall_node(ip)));
    }

    SECTION("config: nu_t/nu ratio, per-flag override") {
        TurbulenceConfig c = sa_config();
        c.inlet.viscosity_ratio = 0.21043825715555026; // <=> nu~/nu = 3
        TurbulenceBoundaryOverride b = boundary_override(503, BoundaryKind::Inlet);
        b.nu_tilde_ratio = 5.0;
        c.boundaries.push_back(b);
        SpalartAllmaras model(c);
        model.initialize(s.ctx(), {u, v});
        CHECK_THAT(model.freestream_nu_tilde(s.nu), WithinRel(3.0 * s.nu, 1e-12));
        CHECK(model.inlet_nu_tilde_ratio(503) == 5.0);
        CHECK_THAT(model.inlet_nu_tilde_ratio(507), WithinRel(3.0, 1e-12));
        const int inlet_node = find_node(s.mesh, 0.0, 1.0);
        REQUIRE(inlet_node >= 0);
        CHECK(model.nu_tilde()[inlet_node] == 5.0 * s.nu);
    }

    SECTION("config: treating the outlet as a wall for the turbulence equations") {
        TurbulenceConfig c = sa_config();
        c.boundaries.push_back(boundary_override(504, BoundaryKind::Wall));
        SpalartAllmaras model(c);
        model.initialize(s.ctx(), {u, v});
        const int outlet_node = find_node(s.mesh, spec.lx, 0.75);
        REQUIRE(outlet_node >= 0);
        CHECK(model.nu_tilde()[outlet_node] == 0.0);
    }
}

TEST_CASE("SA model: homogeneous shear follows the exact production ODE", "[sa][model]") {
    // No walls (d = inf), uniform nu~, u = gamma (y - 1/2): transport and
    // destruction vanish; Omega = gamma at every node.
    //  ft2 off: nu~_{n+1} = nu~_n (1 + dt cb1 gamma)         (explicit growth)
    //  ft2 on:  coefficient c = cb1 (1 - ft2(chi_n)) gamma; growth explicit,
    //           decay implicit: nu~_{n+1} = nu~_n / (1 - dt c).
    const double nu = 1.0e-3, gamma = 2.0, dt = 0.05;
    Setup s(grid(6, 4, 1.5, 1.0, 504), nu, dt);
    REQUIRE_FALSE(s.walls.has_walls());
    std::vector<double> u(s.mesh.npoin), v(s.mesh.npoin, 0.0);
    for (int ip = 0; ip < s.mesh.npoin; ++ip) u[ip] = gamma * (s.mesh.y[ip] - 0.5);

    const bool ft2 = GENERATE(false, true);
    const double chi0 = GENERATE(0.5, 3.0);
    CAPTURE(ft2, chi0);
    TurbulenceConfig c = sa_config();
    c.sa_ft2 = ft2;
    SpalartAllmaras model(c);
    model.initialize(s.ctx(), {u, v});
    REQUIRE(model.boundary_conditions(0).empty());
    model.set_field(s.ctx(), 0, std::vector<double>(s.mesh.npoin, chi0 * nu));

    double ref = chi0 * nu;
    for (int n = 0; n < 40; ++n) {
        model.advance(s.ctx(), {u, v});
        const double coef = C::cb1 * (1.0 - (ft2 ? sa::ft2(ref / nu) : 0.0)) * gamma;
        ref = coef >= 0.0 ? ref * (1.0 + dt * coef) : ref / (1.0 - dt * coef);
    }
    if (ft2 && chi0 == 0.5) {
        CHECK(ref < chi0 * nu); // ft2 makes small nu~ decay
    } else {
        CHECK(ref > chi0 * nu);
    }
    for (int ip = 0; ip < s.mesh.npoin; ++ip) {
        CAPTURE(ip);
        CHECK_THAT(model.nu_tilde()[ip], WithinRel(ref, 1e-11));
        CHECK_THAT(model.nut()[ip], WithinRel(ref * sa::fv1(ref / nu), 1e-11));
    }
}

TEST_CASE("SA model: a uniform free stream is preserved", "[sa][model]") {
    // Channel without walls (inlet 503 left, outlets elsewhere), uniform flow.
    const double nu = 1.0e-3;
    Mesh m = grid(8, 4, 2.0, 1.0, 504);
    set_flags_by_side_id(m, {{4, 503}});
    Setup s(std::move(m), nu, 0.02);
    const std::vector<double> u(s.mesh.npoin, 1.0), v(s.mesh.npoin, 0.25);
    SpalartAllmaras model(sa_config());
    model.initialize(s.ctx(), {u, v});
    for (int n = 0; n < 50; ++n) model.advance(s.ctx(), {u, v});
    for (int ip = 0; ip < s.mesh.npoin; ++ip) CHECK_THAT(model.nu_tilde()[ip], WithinRel(3.0 * nu, 1e-12));
    CHECK(model.residuals()[0].relative_change < 1e-12);
}

TEST_CASE("SA model: negative nu~ relaxes towards zero with nu_t = 0", "[sa][model][negative]") {
    // Quiescent channel between walls: all negative-branch sources decay
    // towards zero (implicit), diffusion with the positive coefficient
    // (nu + fn nu~)/sigma.
    // Channel height 0.1: destruction time scale d^2 / (cw1 |nu~|) ~ 0.2 at
    // the centre line, so 200 steps of 0.01 cover several time scales.
    const double nu = 1.0e-3;
    Mesh m = grid(12, 8, 0.3, 0.1);
    set_flags_by_side_id(m, {{2, 504}, {4, 504}}); // walls bottom/top, no inlet
    Setup s(std::move(m), nu, 0.01);
    const std::vector<double> zero(s.mesh.npoin, 0.0);
    SpalartAllmaras model(sa_config());
    model.initialize(s.ctx(), {zero, zero});

    std::vector<double> initial(s.mesh.npoin);
    for (int ip = 0; ip < s.mesh.npoin; ++ip) initial[ip] = s.walls.is_wall_node(ip) ? 0.0 : -2.0 * nu;
    model.set_field(s.ctx(), 0, initial);

    double previous_min = -2.0 * nu;
    for (int n = 0; n < 200; ++n) {
        model.advance(s.ctx(), {zero, zero});
        const auto& nt = model.nu_tilde();
        const double mn = *std::min_element(nt.begin(), nt.end());
        const double mx = *std::max_element(nt.begin(), nt.end());
        CHECK(mn >= previous_min);        // monotone relaxation
        CHECK(mx <= 1e-3 * 2.0 * nu);     // no significant overshoot above zero
        previous_min = mn;
        for (int ip = 0; ip < s.mesh.npoin; ++ip) {
            if (nt[ip] <= 0.0) CHECK(model.nut()[ip] == 0.0);
        }
    }
    CHECK(previous_min > -0.5 * nu); // substantially relaxed
}

TEST_CASE("SA model: outputs, caps and life cycle", "[sa][model]") {
    const double nu = 1.0e-3;
    Setup s(channel(6, 4, 1.5, 1.0), nu, 0.01);
    const std::vector<double> u(s.mesh.npoin, 1.0), v(s.mesh.npoin, 0.0);

    SECTION("advance before initialize") {
        SpalartAllmaras model(sa_config());
        CHECK_THROWS_AS(model.advance(s.ctx(), {u, v}), std::logic_error);
    }

    SpalartAllmaras model(sa_config());
    model.initialize(s.ctx(), {u, v});

    SECTION("nut nodal = nu~ fv1, element values = nodal means, max diffusivity") {
        std::vector<double> field(s.mesh.npoin);
        for (int ip = 0; ip < s.mesh.npoin; ++ip) field[ip] = nu * (0.5 + 40.0 * s.mesh.y[ip]) * (ip % 3 == 0 ? -1 : 1);
        model.set_field(s.ctx(), 0, field);
        for (int ip = 0; ip < s.mesh.npoin; ++ip) CHECK(model.nut()[ip] == sa::eddy_viscosity(field[ip], nu));
        for (int ie = 0; ie < s.mesh.nelem; ++ie) {
            const auto& n = s.mesh.intma[ie];
            CHECK_THAT(model.nut_elem()[ie], WithinRel((model.nut()[n[0]] + model.nut()[n[1]] + model.nut()[n[2]]) / 3.0, 1e-15));
            const double gamma = (sa::diffusivity(field[n[0]], nu) + sa::diffusivity(field[n[1]], nu) +
                                  sa::diffusivity(field[n[2]], nu)) / 3.0;
            CHECK_THAT(model.max_diffusivity_elem()[ie], WithinRel(gamma, 1e-15));
            CHECK(model.max_diffusivity_elem()[ie] > 0.0);
        }
    }

    SECTION("nu_t / nu is capped by nut_ratio_max") {
        TurbulenceConfig c = sa_config();
        c.nut_ratio_max = 50.0;
        SpalartAllmaras capped(c);
        capped.initialize(s.ctx(), {u, v});
        capped.set_field(s.ctx(), 0, std::vector<double>(s.mesh.npoin, 1000.0 * nu));
        for (double x : capped.nut()) CHECK(x == 50.0 * nu);
    }

    SECTION("fields and residuals") {
        const auto f = model.fields();
        REQUIRE(f.size() == 2);
        CHECK(f[0].name == "nu_tilde");
        CHECK(f[0].values == &model.field(0));
        CHECK(f[1].name == "nut");
        CHECK(model.residuals().size() == 1);
        CHECK(model.residuals()[0].name == "nu_tilde");
        CHECK(model.name() == "spalart-allmaras");
        CHECK_THROWS_AS(model.set_field(s.ctx(), 0, {1.0}), std::invalid_argument);
    }
}

TEST_CASE("SA model: smoke run on the benchmark mesh with a frozen flow field", "[sa][model][hinterkante]") {
    // Velocity frozen at the inflow value with no-slip walls (shear only in
    // the wall elements); the time step includes the SA diffusivity, as the
    // solver coupling will do.
    const double re = GENERATE(200.0, 1.0e5);
    const double nu = 1.0 / re;
    CAPTURE(re);
    Mesh m = benchmark_mesh();
    Setup s(std::move(m), nu, 1.0);
    std::vector<double> u(s.mesh.npoin, 1.0), v(s.mesh.npoin, 0.0);
    apply_velocity_bcs(s.mesh, u, v);
    std::vector<double> speed(s.mesh.npoin);
    for (int ip = 0; ip < s.mesh.npoin; ++ip) speed[ip] = std::sqrt(u[ip] * u[ip] + v[ip] * v[ip] + 1e-16);

    SpalartAllmaras model(sa_config());
    model.initialize(s.ctx(), {u, v});
    for (int n = 0; n < 150; ++n) {
        const auto& g = model.max_diffusivity_elem();
        const double diffusivity = std::max(nu, *std::max_element(g.begin(), g.end()));
        compute_timestep(1, -1, 1, 1.0, diffusivity, 0.6, s.mesh, speed, s.delte);
        update_lhs(s.mesh, s.delte, s.elcoe2, s.pdiag, s.gstif);
        model.advance(s.ctx(), {u, v});
    }
    const auto& nt = model.nu_tilde();
    double near_wall_sum = 0.0;
    int near_wall = 0;
    for (int ip = 0; ip < s.mesh.npoin; ++ip) {
        REQUIRE(std::isfinite(nt[ip]));
        CHECK(model.nut()[ip] >= 0.0);
        CHECK(model.nut()[ip] <= 1.0e5 * nu);
        if (s.walls.is_wall_node(ip)) CHECK(nt[ip] == 0.0);
        const double d = s.walls.wall_distance()[ip];
        if (d > 0.0 && d < 0.05) {
            near_wall_sum += nt[ip];
            ++near_wall;
        }
    }
    const double mx = *std::max_element(nt.begin(), nt.end());
    const double mn = *std::min_element(nt.begin(), nt.end());
    const double near_wall_mean = near_wall_sum / near_wall;
    const double inflow = 3.0 * nu;
    INFO("nu~/nu in [" << mn / nu << ", " << mx / nu << "], near-wall mean " << near_wall_mean / nu);
    REQUIRE(near_wall > 100);
    if (re < 1000.0) {
        // Shear only in the wall elements, where destruction dominates.
        CHECK(mx <= inflow * (1.0 + 1e-12));
        CHECK(near_wall_mean < 0.5 * inflow);
        CHECK(mn > -0.01 * inflow);
    } else {
        // Small nu: production dominates in the sheared wall layers.
        CHECK(mx > 5.0 * inflow);
        CHECK(near_wall_mean > inflow);
        CHECK(mn > -0.1 * mx); // bounded negative excursions, handled by SA-neg
    }
}


TEST_CASE("SA model: exact log-layer solution nu~ = kappa u_tau (y + y0)", "[sa][model][verification]") {
    // Frozen log-law velocity u = (u_tau / kappa) ln((y + y0) / y0), i.e.
    // Omega = u_tau / (kappa d) with the wall distance d = y + y0 of a virtual
    // wall at y = -y0. For nu -> 0 the linear field nu~ = kappa u_tau d is an
    // exact steady solution of the full SA equation (production cb1 u_tau^2,
    // destruction cw1 kappa^2 u_tau^2, diffusion (1 + cb2) kappa^2 u_tau^2 / sigma).
    // Domain [0, 0.25] x [0, 1]: Dirichlet nu~ at the bottom (flag 503) and top
    // (flag 507) from the exact solution, symmetry (natural) at the sides.
    // Starting from a uniform field, pseudo-time marching must converge to it
    // with second-order accuracy when the nodal gradients are recovered with
    // PPR; the lumped projection is first order at the symmetry-side nodes.
    // (y0 = 0.5 keeps the log profile resolved on these grids; for y0 << h
    // the errors are pre-asymptotic.)
    const GradientRecovery method = GENERATE(GradientRecovery::PolynomialPreserving, GradientRecovery::LumpedProjection);
    CAPTURE(to_string(method));
    const double u_tau = 1.0, y0 = 0.5, nu = 1.0e-8, H = 1.0;
    const double kappa = C::kappa;
    auto exact = [&](double y) { return kappa * u_tau * (y + y0); };

    auto solve = [&](int ny) {
        Mesh m = grid(2, ny, 0.25, H);
        set_flags_by_side_id(m, {{1, 503}, {2, 506}, {3, 507}, {4, 506}});
        compute_all_geometry(m);
        const VirtualWallDistance distance(y0);
        const WallGeometry walls(m, distance);

        TurbulenceConfig c = sa_config();
        c.gradient_recovery = method;
        TurbulenceBoundaryOverride bottom = boundary_override(503, BoundaryKind::Inlet);
        bottom.nu_tilde_ratio = exact(0.0) / nu;
        TurbulenceBoundaryOverride top = boundary_override(507, BoundaryKind::Inlet);
        top.nu_tilde_ratio = exact(H) / nu;
        c.boundaries.push_back(bottom);
        c.boundaries.push_back(top);
        c.inlet.nu_tilde_ratio = exact(0.5 * H) / nu; // uniform initial field

        std::vector<double> u(m.npoin), v(m.npoin, 0.0);
        for (int ip = 0; ip < m.npoin; ++ip) u[ip] = u_tau / kappa * std::log((m.y[ip] + y0) / y0);

        const double h = H / ny;
        const double dt = 0.2 * h * h / (exact(H) / C::sigma);
        std::vector<double> delte(m.nelem, dt), elcoe2, pdiag;
        std::vector<std::array<double, 3>> gstif;
        update_lhs(m, delte, elcoe2, pdiag, gstif);
        const TurbulenceContext ctx{m, walls, nu, delte, elcoe2};

        SpalartAllmaras model(c);
        model.initialize(ctx, {u, v});
        for (int n = 0; n < 200000; ++n) {
            model.advance(ctx, {u, v});
            if (model.residuals()[0].relative_change < 1e-13) break;
        }
        double err = 0.0;
        for (int ip = 0; ip < m.npoin; ++ip) {
            err = std::max(err, std::abs(model.nu_tilde()[ip] - exact(m.y[ip])) / exact(m.y[ip]));
        }
        return err;
    };

    const double e20 = solve(20), e40 = solve(40);
    const double order = std::log2(e20 / e40);
    INFO("max relative error ny=20: " << e20 << ", ny=40: " << e40 << ", observed order " << order);
    if (method == GradientRecovery::PolynomialPreserving) {
        CHECK(e40 < 5e-4);
        CHECK(order > 1.8);
    } else {
        CHECK(order < 1.5);
    }
}
