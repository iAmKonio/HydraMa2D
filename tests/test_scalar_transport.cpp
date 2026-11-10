// Generic CBS scalar transport kernel (assemble_scalar_transport), the
// point-implicit source update (point_implicit_update, SourceSplit) and the
// compiled Dirichlet sets.
//
// Verified:
//  * the kernel reproduces the momentum Step 1 for phi = u, v and Gamma = 1/Re,
//  * fused multi-scalar assembly == separate assemblies (bit-for-bit),
//  * constant preservation, conservation of the diffusion and streamline terms,
//  * variable diffusivity: dense-matrix reference and the exact identity
//    rhs_i / M_i = grad(Gamma) . grad(phi) for linear Gamma and phi,
//  * manufactured solution of a steady diffusion-reaction problem with
//    variable diffusivity: second-order convergence,
//  * point-implicit update: exact implicit-Euler decay, explicit production,
//    local time steps, positivity (Patankar) for any time step.

#include "LinearSolverTestUtils.hpp"
#include "Step1.hpp"
#include "geometry/FieldOps.hpp"
#include "numerics/DirichletSet.hpp"
#include "numerics/PointImplicit.hpp"
#include "numerics/ScalarTransport.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kPi = 3.14159265358979323846;

Mesh grid(int nx, int ny, double lx, double ly, double x0 = 0.0, double y0 = 0.0) {
    GridSpec g;
    g.x0 = x0;
    g.y0 = y0;
    g.lx = lx;
    g.ly = ly;
    g.nx = nx;
    g.ny = ny;
    Mesh m = make_grid_mesh(g);
    compute_all_geometry(m);
    return m;
}

Mesh test_grid() { return grid(5, 4, 1.2, 0.9, 0.2, -0.3); }

double tg_u(double x, double y) { return 0.4 + std::sin(2.0 * x) * std::cos(1.5 * y); }
double tg_v(double x, double y) { return -0.2 - std::cos(2.0 * x) * std::sin(1.5 * y); }

std::vector<double> transport(const Mesh& m, const std::vector<double>& u, const std::vector<double>& v,
                              const std::vector<double>& delte, const std::vector<double>& phi,
                              const std::vector<double>& gamma) {
    std::vector<double> rhs(3, 42.0); // stale, wrongly sized output
    assemble_scalar_transport(m, u, v, delte, phi, gamma, rhs);
    return rhs;
}

bool is_boundary_node(const Mesh& m, int ip) {
    for (const auto& s : m.iside)
        if (s.node1 == ip || s.node2 == ip) return true;
    return false;
}

} // namespace

// ---------------------------------------------------------------------------
// Transport kernel
// ---------------------------------------------------------------------------

TEST_CASE("Scalar transport reproduces the CBS momentum Step 1 for phi = u, v", "[transport]") {
    const Mesh m = test_grid();
    const double nu = GENERATE(0.0, 1.0 / 200.0, 0.1);
    CAPTURE(nu);
    const auto delte = varying_dt(m.nelem);
    const auto u = sample(m, tg_u), v = sample(m, tg_v);

    std::vector<double> ru, rv;
    compute_step1(m, nu, delte, u, v, ru, rv);
    const std::vector<double> gamma(m.nelem, nu);
    const auto tu = transport(m, u, v, delte, u, gamma);
    const auto tv = transport(m, u, v, delte, v, gamma);
    REQUIRE(static_cast<int>(tu.size()) == m.npoin);
    const double su = scale_of(ru), sv = scale_of(rv);
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i);
        CHECK_THAT(tu[i], WithinAbs(ru[i], 1e-14 * su));
        CHECK_THAT(tv[i], WithinAbs(rv[i], 1e-14 * sv));
    }
}

TEST_CASE("Fused multi-scalar assembly equals separate assemblies bit-for-bit", "[transport]") {
    const Mesh m = test_grid();
    const auto delte = varying_dt(m.nelem);
    const auto u = sample(m, tg_u), v = sample(m, tg_v);
    const auto a = sample(m, [](double x, double y) { return std::exp(x) * y; });
    const auto b = sample(m, [](double x, double y) { return std::cos(3.0 * x + y); });
    const auto c = sample(m, [](double x, double) { return x * x; });
    std::vector<double> ga(m.nelem), gb(m.nelem), gc(m.nelem, 0.02);
    for (int ie = 0; ie < m.nelem; ++ie) {
        ga[ie] = 0.01 * (1 + ie % 3);
        gb[ie] = 0.05 / (1 + ie % 4);
    }

    std::vector<double> fa, fb, fc;
    assemble_scalar_transport<2>(m, u, v, delte, {TransportedScalar{&a, &ga, &fa}, TransportedScalar{&b, &gb, &fb}});
    CHECK(fa == transport(m, u, v, delte, a, ga));
    CHECK(fb == transport(m, u, v, delte, b, gb));

    std::vector<double> ha, hb, hc;
    assemble_scalar_transport<3>(m, u, v, delte,
                                 {TransportedScalar{&a, &ga, &ha}, TransportedScalar{&b, &gb, &hb},
                                  TransportedScalar{&c, &gc, &hc}});
    CHECK(ha == fa);
    CHECK(hb == fb);
    CHECK(hc == transport(m, u, v, delte, c, gc));

    std::vector<double> out;
    CHECK_THROWS_AS(assemble_scalar_transport<1>(m, u, v, delte, {TransportedScalar{&a, nullptr, &out}}),
                    std::invalid_argument);
}

TEST_CASE("Scalar transport preserves constants", "[transport]") {
    const Mesh m = test_grid();
    const double c = GENERATE(0.0, 1.0, -3.5e-4);
    const auto u = sample(m, tg_u), v = sample(m, tg_v);
    std::vector<double> gamma(m.nelem);
    for (int ie = 0; ie < m.nelem; ++ie) gamma[ie] = 0.1 + 0.01 * ie;
    const auto rhs = transport(m, u, v, varying_dt(m.nelem), std::vector<double>(m.npoin, c), gamma);
    for (double r : rhs) CHECK_THAT(r, WithinAbs(0.0, 1e-14));
}

TEST_CASE("Diffusion and streamline terms are conservative", "[transport]") {
    // sum_i grad N_i = 0 per element, so only the Galerkin convection term
    // contributes to sum_i rhs_i, for any diffusivity and time step.
    const Mesh m = test_grid();
    const auto u = sample(m, tg_u), v = sample(m, tg_v);
    const auto phi = sample(m, [](double x, double y) { return std::sin(3.0 * x) + y * y; });
    std::vector<double> gamma(m.nelem);
    for (int ie = 0; ie < m.nelem; ++ie) gamma[ie] = 0.5 + std::sin(1.0 * ie);

    const auto full = transport(m, u, v, varying_dt(m.nelem), phi, gamma);
    const auto conv = transport(m, u, v, std::vector<double>(m.nelem, 0.0), phi, std::vector<double>(m.nelem, 0.0));
    const double sum_full = std::accumulate(full.begin(), full.end(), 0.0);
    const double sum_conv = std::accumulate(conv.begin(), conv.end(), 0.0);
    CHECK_THAT(sum_full, WithinAbs(sum_conv, 1e-13));

    const std::vector<double> zero(m.npoin, 0.0);
    const auto diff = transport(m, zero, zero, varying_dt(m.nelem), phi, gamma);
    CHECK_THAT(std::accumulate(diff.begin(), diff.end(), 0.0), WithinAbs(0.0, 1e-13));
}

TEST_CASE("Variable diffusivity matches the dense element-weighted Laplacian", "[transport]") {
    const Mesh m = test_grid();
    std::vector<double> gamma(m.nelem);
    for (int ie = 0; ie < m.nelem; ++ie) gamma[ie] = 0.01 * (1 + (ie * 5) % 7);
    const auto phi = sample(m, [](double x, double y) { return std::exp(x - y) + x * y; });
    const std::vector<double> zero(m.npoin, 0.0);

    const auto rhs = transport(m, zero, zero, varying_dt(m.nelem), phi, gamma);
    // dense_reference_laplacian(m, w) assembles sum_e w_e K_e.
    auto ref = multiply(dense_reference_laplacian(m, gamma), phi);
    for (double& r : ref) r = -r;
    for (int i = 0; i < m.npoin; ++i) CHECK_THAT(rhs[i], WithinAbs(ref[i], 1e-14 * scale_of(ref)));
}

TEST_CASE("Variable diffusivity: rhs_i / M_i = grad(Gamma) . grad(phi) at interior nodes", "[transport]") {
    // For linear Gamma and phi, int Gamma grad N_i . grad phi = -M_i grad Gamma . grad phi
    // at interior nodes (integration by parts), so rhs_i = +M_i grad Gamma . grad phi.
    const Mesh m = grid(6, 5, 1.0, 1.0);
    const auto mass = lumped_nodal_mass(m);
    const auto gamma_node = sample(m, [](double x, double y) { return 0.3 + 0.2 * x - 0.1 * y; });
    std::vector<double> gamma;
    element_mean(m, gamma_node, gamma);
    const auto phi = sample(m, [](double x, double y) { return 2.0 * x + 5.0 * y; });
    const std::vector<double> zero(m.npoin, 0.0);
    const auto rhs = transport(m, zero, zero, std::vector<double>(m.nelem, 0.1), phi, gamma);

    const double expected = 0.2 * 2.0 - 0.1 * 5.0;
    int interior = 0;
    for (int ip = 0; ip < m.npoin; ++ip) {
        if (is_boundary_node(m, ip)) continue;
        ++interior;
        CHECK_THAT(rhs[ip] / mass[ip], WithinAbs(expected, 1e-12));
    }
    CHECK(interior == 20);
}

TEST_CASE("Scalar transport: manufactured steady diffusion-reaction, second-order convergence",
          "[transport][mms]") {
    // -div(Gamma grad phi) + lambda phi = f on the unit square, phi = 0 on the
    // boundary, Gamma = 1 + x/2, phi_exact = sin(pi x) sin(pi y):
    //   f = Gamma 2 pi^2 phi - (pi/2) cos(pi x) sin(pi y) + lambda phi.
    // Pseudo-time marching with the transport kernel (u = 0) and the
    // point-implicit update (P = f, D = lambda) to the steady state.
    const double lambda = 2.0;
    auto exact = [](double x, double y) { return std::sin(kPi * x) * std::sin(kPi * y); };
    auto source = [&](double x, double y) {
        return (1.0 + 0.5 * x) * 2.0 * kPi * kPi * exact(x, y) - 0.5 * kPi * std::cos(kPi * x) * std::sin(kPi * y) +
               lambda * exact(x, y);
    };

    auto solve = [&](int n) {
        const Mesh m = grid(n, n, 1.0, 1.0);
        const auto mass = lumped_nodal_mass(m);
        const auto gamma_node = sample(m, [](double x, double) { return 1.0 + 0.5 * x; });
        std::vector<double> gamma;
        element_mean(m, gamma_node, gamma);
        const double h = 1.0 / n, dt = 0.2 * h * h / 1.5;
        const std::vector<double> delte(m.nelem, dt), zero(m.npoin, 0.0);
        std::vector<double> elcoe2(m.npoin);
        for (int ip = 0; ip < m.npoin; ++ip) elcoe2[ip] = dt / mass[ip];
        const auto P = sample(m, source);
        const std::vector<double> D(m.npoin, lambda);

        DirichletBuilder b(m.npoin);
        for (const auto& s : m.iside) {
            b.set(s.node1, 0.0);
            b.set(s.node2, 0.0);
        }
        const DirichletSet bc = b.build();

        std::vector<double> phi(m.npoin, 0.0), rhs, old;
        for (int step = 0; step < 200000; ++step) {
            old = phi;
            assemble_scalar_transport(m, zero, zero, delte, phi, gamma, rhs);
            point_implicit_update(mass, elcoe2, rhs, P, D, false, phi);
            bc.apply(phi);
            double change = 0.0;
            for (int ip = 0; ip < m.npoin; ++ip) change = std::max(change, std::abs(phi[ip] - old[ip]));
            if (change < 1e-13) break;
        }
        double err = 0.0;
        for (int ip = 0; ip < m.npoin; ++ip) err = std::max(err, std::abs(phi[ip] - exact(m.x[ip], m.y[ip])));
        return err;
    };

    const double e16 = solve(16), e32 = solve(32);
    const double order = std::log2(e16 / e32);
    INFO("max error h=1/16: " << e16 << ", h=1/32: " << e32 << ", observed order " << order);
    CHECK(e32 < 5e-3);
    CHECK(order > 1.8);
    CHECK(order < 2.3);
}

// ---------------------------------------------------------------------------
// Point-implicit update
// ---------------------------------------------------------------------------

TEST_CASE("SourceSplit: decay implicit, growth explicit, sinks Patankar-weighted", "[transport][point_implicit]") {
    SourceSplit s;
    s.add_linear(-3.0, 2.0); // decay: D += 3
    CHECK(s.D == 3.0);
    CHECK(s.P == 0.0);
    s.add_linear(0.5, 2.0); // growth: P += 1
    CHECK(s.P == 1.0);
    s.add_linear(-1.0, -4.0); // decay of a negative value is still implicit
    CHECK(s.D == 4.0);
    s.add_value(0.25, 2.0); // explicit source
    CHECK(s.P == 1.25);
    s.add_value(-1.0, 2.0); // sink, phi > 0: D += 1/2
    CHECK(s.D == 4.5);
    s.add_value(-1.0, 0.0); // sink, phi <= 0: explicit
    CHECK(s.P == 0.25);
    CHECK(s.value(2.0) == 0.25 - 4.5 * 2.0); // Q = P - D phi

    // A single term is reproduced exactly at its linearisation point.
    for (double c : {-2.0, 0.0, 3.0}) {
        for (double phi : {-1.5, 0.0, 2.5}) {
            SourceSplit lin;
            lin.add_linear(c, phi);
            CHECK_THAT(lin.value(phi), WithinAbs(c * phi, 1e-15));
            SourceSplit val;
            val.add_value(c, phi);
            CHECK_THAT(val.value(phi), WithinAbs(c, 1e-15));
            CHECK(lin.D >= 0.0);
            CHECK(val.D >= 0.0);
        }
    }
}

TEST_CASE("Point-implicit update: exact formulas", "[transport][point_implicit]") {
    SECTION("pure decay is implicit Euler: phi_n = phi_0 / (1 + dt D)^n, never negative") {
        const double dt = GENERATE(0.01, 1.0, 1.0e6);
        const double D = 50.0;
        std::vector<double> phi = {1.0, 0.0, -2.0};
        const std::vector<double> mass(3, 2.0), elcoe2(3, dt / 2.0), zero(3, 0.0), Dv(3, D);
        for (int n = 1; n <= 5; ++n) {
            point_implicit_update(mass, elcoe2, zero, zero, Dv, false, phi);
            CHECK_THAT(phi[0], WithinRel(1.0 / std::pow(1.0 + dt * D, n), 1e-13));
            CHECK(phi[0] > 0.0);
            CHECK(phi[1] == 0.0);
            CHECK(phi[2] < 0.0); // decay never changes the sign
        }
    }

    SECTION("implicit Euler converges to exp(-D t) at first order") {
        auto error = [](int steps) {
            const double t = 1.0, D = 3.0, dt = t / steps;
            std::vector<double> phi = {1.0};
            for (int n = 0; n < steps; ++n)
                point_implicit_update({1.0}, {dt}, {0.0}, {0.0}, {D}, false, phi);
            return std::abs(phi[0] - std::exp(-D * t));
        };
        CHECK_THAT(error(200) / error(400), WithinAbs(2.0, 0.02));
    }

    SECTION("explicit part: phi + elcoe2 rhs + dt P, divided by 1 + dt D") {
        const std::vector<double> mass = {0.5, 2.0}, elcoe2 = {0.2, 0.05}; // nodal dt = 0.1 each
        std::vector<double> phi = {1.0, 3.0};
        point_implicit_update(mass, elcoe2, {4.0, -2.0}, {1.5, 0.0}, {0.0, 10.0}, false, phi);
        CHECK_THAT(phi[0], WithinRel(1.0 + 0.2 * 4.0 + 0.1 * 1.5, 1e-15));
        CHECK_THAT(phi[1], WithinRel((3.0 - 0.05 * 2.0) / (1.0 + 0.1 * 10.0), 1e-15));
    }

    SECTION("local time stepping: the nodal time step is M_i * elcoe2_i") {
        const std::vector<double> mass = {1.0, 4.0}, elcoe2 = {0.5, 0.5}; // dt = 0.5 and 2
        std::vector<double> phi = {1.0, 1.0};
        point_implicit_update(mass, elcoe2, {0.0, 0.0}, {0.0, 0.0}, {1.0, 1.0}, false, phi);
        CHECK_THAT(phi[0], WithinRel(1.0 / 1.5, 1e-15));
        CHECK_THAT(phi[1], WithinRel(1.0 / 3.0, 1e-15));
    }

    SECTION("size mismatch") {
        std::vector<double> phi = {1.0, 2.0};
        CHECK_THROWS_AS(point_implicit_update({1.0}, {1.0, 1.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, false, phi),
                        std::invalid_argument);
    }
}

TEST_CASE("Point-implicit update with positivity enforcement", "[transport][point_implicit][positivity]") {
    // Deterministic pseudo-random non-negative states and strongly negative
    // explicit increments.
    const int n = 500;
    std::vector<double> mass(n), elcoe2(n), rhs(n), P(n), D(n), phi0(n);
    for (int i = 0; i < n; ++i) {
        mass[i] = 0.1 + 0.01 * (i % 13);
        elcoe2[i] = (1.0 + i % 7) / mass[i];         // dt between 1 and 7
        rhs[i] = -50.0 * std::abs(std::sin(1.3 * i)) * (i % 3 == 0 ? -0.1 : 1.0);
        P[i] = (i % 5 == 0) ? -2.0 : 0.5 * (i % 4); // includes negative explicit sources
        D[i] = 0.3 * (i % 6);
        phi0[i] = (i % 11 == 0) ? 0.0 : 0.01 * (1 + i % 17);
    }

    std::vector<double> plain = phi0, positive = phi0;
    point_implicit_update(mass, elcoe2, rhs, P, D, false, plain);
    point_implicit_update(mass, elcoe2, rhs, P, D, true, positive);

    const auto negatives = std::count_if(plain.begin(), plain.end(), [](double x) { return x < 0.0; });
    CHECK(negatives > 100); // the plain update overshoots
    for (int i = 0; i < n; ++i) {
        INFO("node " << i);
        CHECK(positive[i] >= 0.0);
        const double explicit_increment = elcoe2[i] * rhs[i] + mass[i] * elcoe2[i] * P[i];
        const double dt = mass[i] * elcoe2[i];
        if (explicit_increment >= 0.0) {
            CHECK(positive[i] == plain[i]); // identical when not limiting
        } else if (phi0[i] > 0.0) {
            // Patankar weighting: phi^{n+1} = phi^n / (1 + dt D - E / phi^n).
            CHECK_THAT(positive[i], WithinRel(phi0[i] / (1.0 + dt * D[i] - explicit_increment / phi0[i]), 1e-13));
        } else {
            CHECK(positive[i] == 0.0); // negative increment on a zero value is dropped
        }
    }
}

TEST_CASE("Explicit advection with positivity enforcement stays non-negative", "[transport][positivity]") {
    // A narrow non-negative pulse advected diagonally at CFL ~ 1: the
    // unlimited explicit scheme produces negative undershoots, the Patankar
    // update does not.
    const Mesh m = grid(20, 20, 1.0, 1.0);
    const auto mass = lumped_nodal_mass(m);
    const double dt = 0.5 * (1.0 / 20.0); // CFL ~ 0.8 based on the minimum element height
    const std::vector<double> delte(m.nelem, dt);
    std::vector<double> elcoe2(m.npoin);
    for (int ip = 0; ip < m.npoin; ++ip) elcoe2[ip] = dt / mass[ip];
    const std::vector<double> u(m.npoin, 1.0), v(m.npoin, 0.5), gamma(m.nelem, 0.0), zero(m.npoin, 0.0);
    auto pulse = sample(m, [](double x, double y) { return std::exp(-400.0 * ((x - 0.3) * (x - 0.3) + (y - 0.3) * (y - 0.3))); });

    auto run = [&](bool positivity) {
        std::vector<double> phi = pulse, rhs;
        double min_value = 0.0;
        for (int step = 0; step < 8; ++step) {
            assemble_scalar_transport(m, u, v, delte, phi, gamma, rhs);
            point_implicit_update(mass, elcoe2, rhs, zero, zero, positivity, phi);
            min_value = std::min(min_value, *std::min_element(phi.begin(), phi.end()));
        }
        return min_value;
    };
    CHECK(run(false) < -1e-3);
    CHECK(run(true) >= 0.0);
}

// ---------------------------------------------------------------------------
// Dirichlet sets
// ---------------------------------------------------------------------------

TEST_CASE("Dirichlet builder: later values override, nodes sorted and unique", "[transport][dirichlet]") {
    DirichletBuilder b(6);
    b.set(4, 1.0);
    b.set(1, 2.0);
    b.set(4, 3.0); // override
    CHECK(b.is_set(4));
    CHECK_FALSE(b.is_set(0));
    const DirichletSet s = b.build();
    REQUIRE(s.size() == 2);
    CHECK(s.nodes == std::vector<int>{1, 4});
    CHECK(s.values == std::vector<double>{2.0, 3.0});

    std::vector<double> phi(6, -1.0);
    s.apply(phi);
    CHECK(phi == std::vector<double>{-1.0, 2.0, -1.0, -1.0, 3.0, -1.0});
    CHECK(DirichletBuilder(3).build().empty());
}
