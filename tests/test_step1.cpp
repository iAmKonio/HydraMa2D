// CBS Step 1 (intermediate momentum): compute_step1() assembles, for each
// velocity component phi in {u, v} and node i,
//
//   rhs_i = - int N_i (u . grad) phi dOmega                       (convection)
//           - dt_e / 2 * int (u_bar . grad N_i)(u_bar . grad phi)  (CBS stabilisation)
//           - nu * int grad N_i . grad phi dOmega                  (viscous diffusion)
//
// with u_bar the element-mean velocity, dt_e the element time step and
// nu = ani = 1/Re. The solver then updates phi_i += dt * rhs_i / M_i.

#include "LinearSolverTestUtils.hpp"
#include "IO.hpp"
#include "Step1.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <numeric>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

struct Rhs {
    std::vector<double> u, v;
};

Rhs step1(const Mesh& m, double ani, const std::vector<double>& delte,
          const std::vector<double>& u, const std::vector<double>& v) {
    Rhs r{std::vector<double>(3, 42.0), {}}; // stale, wrongly sized output
    compute_step1(m, ani, delte, u, v, r.u, r.v);
    return r;
}

// --- Independent reference, term by term ----------------------------------

// Convection, integrated with the 3-point edge-midpoint rule, which is exact
// for the quadratic integrand N_i * (u_h . grad phi_h) on each element.
std::vector<double> convection_ref(const Mesh& m, const std::vector<double>& u,
                                   const std::vector<double>& v, const std::vector<double>& phi) {
    std::vector<double> r(m.npoin, 0.0);
    for (int ie = 0; ie < m.nelem; ++ie) {
        const auto& n = m.intma[ie];
        const auto& d = m.dNkdx[ie];
        const double area = 0.5 * m.detJ[ie];
        double gx = 0.0, gy = 0.0;
        for (int k = 0; k < 3; ++k) {
            gx += d.dNdx[k] * phi[n[k]];
            gy += d.dNdy[k] * phi[n[k]];
        }
        for (int k = 0; k < 3; ++k) { // midpoint of edge (a, b)
            const int a = k, b = (k + 1) % 3;
            const double um = 0.5 * (u[n[a]] + u[n[b]]);
            const double vm = 0.5 * (v[n[a]] + v[n[b]]);
            const double integrand = um * gx + vm * gy;
            // N_a = N_b = 1/2 at the midpoint, the third shape function is 0.
            r[n[a]] -= area / 3.0 * 0.5 * integrand;
            r[n[b]] -= area / 3.0 * 0.5 * integrand;
        }
    }
    return r;
}

// Viscous term: -nu * K phi with the assembled Laplacian K.
std::vector<double> viscous_ref(const Mesh& m, double nu, const std::vector<double>& phi) {
    const DenseMatrix k = dense_reference_laplacian(m, std::vector<double>(m.nelem, 1.0));
    std::vector<double> r = multiply(k, phi);
    for (double& x : r) x *= -nu;
    return r;
}

// Stabilisation: -dt_e/2 * A_e * (u_bar . grad N_i)(u_bar . grad phi).
std::vector<double> stabilisation_ref(const Mesh& m, const std::vector<double>& delte,
                                      const std::vector<double>& u, const std::vector<double>& v,
                                      const std::vector<double>& phi) {
    std::vector<double> r(m.npoin, 0.0);
    for (int ie = 0; ie < m.nelem; ++ie) {
        const auto& n = m.intma[ie];
        const auto& d = m.dNkdx[ie];
        const double area = 0.5 * m.detJ[ie];
        const double ub = (u[n[0]] + u[n[1]] + u[n[2]]) / 3.0;
        const double vb = (v[n[0]] + v[n[1]] + v[n[2]]) / 3.0;
        double adv_phi = 0.0;
        for (int k = 0; k < 3; ++k) adv_phi += (ub * d.dNdx[k] + vb * d.dNdy[k]) * phi[n[k]];
        for (int k = 0; k < 3; ++k)
            r[n[k]] -= 0.5 * delte[ie] * area * (ub * d.dNdx[k] + vb * d.dNdy[k]) * adv_phi;
    }
    return r;
}

Rhs step1_ref(const Mesh& m, double nu, const std::vector<double>& delte,
              const std::vector<double>& u, const std::vector<double>& v) {
    Rhs r{convection_ref(m, u, v, u), convection_ref(m, u, v, v)};
    const auto vu = viscous_ref(m, nu, u), vv = viscous_ref(m, nu, v);
    const auto su = stabilisation_ref(m, delte, u, v, u), sv = stabilisation_ref(m, delte, u, v, v);
    for (int i = 0; i < m.npoin; ++i) {
        r.u[i] += vu[i] + su[i];
        r.v[i] += vv[i] + sv[i];
    }
    return r;
}

// --- Test setup -------------------------------------------------------------

Mesh step_grid() {
    GridSpec g;
    g.x0 = 0.2;
    g.y0 = -0.3;
    g.lx = 1.2;
    g.ly = 0.9;
    g.nx = 5;
    g.ny = 4;
    Mesh m = make_grid_mesh(g);
    compute_all_geometry(m);
    return m;
}

// Smooth, non-linear test velocity (Taylor-Green-like vortex plus drift).
double tg_u(double x, double y) { return 0.4 + std::sin(2.0 * x) * std::cos(1.5 * y); }
double tg_v(double x, double y) { return -0.2 - std::cos(2.0 * x) * std::sin(1.5 * y); }

void check_vectors_close(const std::vector<double>& a, const std::vector<double>& b, double tol) {
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        CAPTURE(i);
        CHECK_THAT(a[i], WithinAbs(b[i], tol));
    }
}

} // namespace

TEST_CASE("Step 1 on a single element, computed by hand", "[cbs][step1]") {
    // Unit right triangle (A = 1/2), u = x, v = 0:
    //   grad u = (1, 0), u_bar = (1/3, 0), dN/dx = (-1, 1, 0)
    //   convection   -int N_i x           = -(1, 2, 1) / 24
    //   viscous      -nu A dN_i/dx        =  nu/2 * (1, -1, 0)
    //   stabilisation -dt/2 A (1/3 dN_i/dx)(1/3) = dt/36 * (1, -1, 0)
    Mesh m = make_triangle_mesh(reference_triangles()[0]);
    compute_all_geometry(m);
    const double nu = GENERATE(0.0, 0.01, 0.5);
    const double dt = GENERATE(0.0, 0.1, 2.0);
    CAPTURE(nu, dt);

    const Rhs r = step1(m, nu, {dt}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0});
    REQUIRE(r.u.size() == 3);
    REQUIRE(r.v.size() == 3);
    CHECK_THAT(r.u[0], WithinAbs(-1.0 / 24.0 + nu / 2.0 + dt / 36.0, 1e-15));
    CHECK_THAT(r.u[1], WithinAbs(-2.0 / 24.0 - nu / 2.0 - dt / 36.0, 1e-15));
    CHECK_THAT(r.u[2], WithinAbs(-1.0 / 24.0, 1e-15));
    for (double x : r.v) CHECK(x == 0.0);
}

TEST_CASE("Step 1 vanishes for uniform flow (free-stream preservation)", "[cbs][step1]") {
    const Mesh m = step_grid();
    const double u0 = GENERATE(0.0, 1.0, -2.5);
    const double v0 = GENERATE(0.0, 0.75);
    CAPTURE(u0, v0);

    const Rhs r = step1(m, 0.01, varying_dt(m.nelem), std::vector<double>(m.npoin, u0),
                        std::vector<double>(m.npoin, v0));
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i);
        CHECK_THAT(r.u[i], WithinAbs(0.0, 1e-14));
        CHECK_THAT(r.v[i], WithinAbs(0.0, 1e-14));
    }
}

TEST_CASE("Step 1 matches the term-by-term reference for a non-linear flow", "[cbs][step1]") {
    const Mesh m = step_grid();
    const double nu = GENERATE(0.0, 1.0 / 200.0, 0.1);
    CAPTURE(nu);
    const auto delte = varying_dt(m.nelem);
    const auto u = sample(m, tg_u), v = sample(m, tg_v);

    const Rhs r = step1(m, nu, delte, u, v);
    const Rhs ref = step1_ref(m, nu, delte, u, v);
    check_vectors_close(r.u, ref.u, 1e-13 * scale_of(ref.u));
    check_vectors_close(r.v, ref.v, 1e-13 * scale_of(ref.v));
}

TEST_CASE("Step 1 convection term for a linear velocity field", "[cbs][step1]") {
    // With dt = 0 and nu = 0 only the Galerkin convection term remains.
    const Mesh m = step_grid();
    auto fu = [](double x, double y) { return 0.5 + 1.2 * x - 0.7 * y; };
    auto fv = [](double x, double y) { return -0.3 + 0.4 * x + 0.9 * y; };
    const auto u = sample(m, fu), v = sample(m, fv);

    const Rhs r = step1(m, 0.0, std::vector<double>(m.nelem, 0.0), u, v);
    check_vectors_close(r.u, convection_ref(m, u, v, u), 1e-14);
    check_vectors_close(r.v, convection_ref(m, u, v, v), 1e-14);

    SECTION("nodal sum equals -integral of (u . grad) u over the domain") {
        // (u.grad)u = 1.2 u - 0.7 v and (u.grad)v = 0.4 u + 0.9 v are linear,
        // so their integrals are area * value at the centroid (0.8, 0.15).
        const double area = 1.2 * 0.9, xc = 0.8, yc = 0.15;
        const double conv_u = 1.2 * fu(xc, yc) - 0.7 * fv(xc, yc);
        const double conv_v = 0.4 * fu(xc, yc) + 0.9 * fv(xc, yc);
        CHECK_THAT(std::accumulate(r.u.begin(), r.u.end(), 0.0), WithinRel(-area * conv_u, 1e-12));
        CHECK_THAT(std::accumulate(r.v.begin(), r.v.end(), 0.0), WithinRel(-area * conv_v, 1e-12));
    }
}

TEST_CASE("Step 1 viscous term is -nu K u", "[cbs][step1]") {
    // The difference between two viscosities isolates the viscous term.
    const Mesh m = step_grid();
    const auto delte = varying_dt(m.nelem);
    const auto u = sample(m, tg_u), v = sample(m, tg_v);
    const double nu1 = 0.3, nu2 = 0.05;

    const Rhs r1 = step1(m, nu1, delte, u, v);
    const Rhs r2 = step1(m, nu2, delte, u, v);
    const auto ku = viscous_ref(m, nu1 - nu2, u), kv = viscous_ref(m, nu1 - nu2, v);
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i);
        CHECK_THAT(r1.u[i] - r2.u[i], WithinAbs(ku[i], 1e-13));
        CHECK_THAT(r1.v[i] - r2.v[i], WithinAbs(kv[i], 1e-13));
    }
}

TEST_CASE("Step 1 stabilisation is linear in the element time step and element-local", "[cbs][step1]") {
    const Mesh m = step_grid();
    const auto u = sample(m, tg_u), v = sample(m, tg_v);
    const double nu = 0.01;
    const std::vector<double> zero(m.nelem, 0.0);
    const auto delte = varying_dt(m.nelem);

    SECTION("rhs(dt) - rhs(0) is the stabilisation term") {
        const Rhs r0 = step1(m, nu, zero, u, v);
        const Rhs r = step1(m, nu, delte, u, v);
        const auto su = stabilisation_ref(m, delte, u, v, u);
        for (int i = 0; i < m.npoin; ++i) {
            CAPTURE(i);
            CHECK_THAT(r.u[i] - r0.u[i], WithinAbs(su[i], 1e-14));
        }
    }
    SECTION("doubling every dt doubles the stabilisation term") {
        std::vector<double> twice = delte;
        for (double& d : twice) d *= 2.0;
        const Rhs r0 = step1(m, nu, zero, u, v);
        const Rhs r1 = step1(m, nu, delte, u, v);
        const Rhs r2 = step1(m, nu, twice, u, v);
        for (int i = 0; i < m.npoin; ++i) {
            CAPTURE(i);
            CHECK_THAT(r2.u[i] - r0.u[i], WithinAbs(2.0 * (r1.u[i] - r0.u[i]), 1e-14));
            CHECK_THAT(r2.v[i] - r0.v[i], WithinAbs(2.0 * (r1.v[i] - r0.v[i]), 1e-14));
        }
    }
    SECTION("changing one element's dt only affects that element's nodes") {
        const int ie = 13;
        std::vector<double> changed = delte;
        changed[ie] *= 5.0;
        const Rhs a = step1(m, nu, delte, u, v);
        const Rhs b = step1(m, nu, changed, u, v);
        const std::set<int> nodes(m.intma[ie].begin(), m.intma[ie].end());
        for (int i = 0; i < m.npoin; ++i) {
            CAPTURE(i);
            if (nodes.count(i)) {
                CHECK(a.u[i] != b.u[i]);
            } else {
                CHECK(a.u[i] == b.u[i]);
                CHECK(a.v[i] == b.v[i]);
            }
        }
    }
}

TEST_CASE("Step 1 viscous and stabilisation terms conserve momentum", "[cbs][step1]") {
    // Both terms contain grad N_i, and sum_i grad N_i = 0 on every element, so
    // the nodal sum of rhs is the convection integral alone: independent of
    // nu and dt.
    const Mesh m = step_grid();
    const auto u = sample(m, tg_u), v = sample(m, tg_v);
    const Rhs base = step1(m, 0.0, std::vector<double>(m.nelem, 0.0), u, v);
    const double su0 = std::accumulate(base.u.begin(), base.u.end(), 0.0);
    const double sv0 = std::accumulate(base.v.begin(), base.v.end(), 0.0);

    const double nu = GENERATE(0.01, 1.0);
    CAPTURE(nu);
    const Rhs r = step1(m, nu, varying_dt(m.nelem), u, v);
    CHECK_THAT(std::accumulate(r.u.begin(), r.u.end(), 0.0), WithinAbs(su0, 1e-13));
    CHECK_THAT(std::accumulate(r.v.begin(), r.v.end(), 0.0), WithinAbs(sv0, 1e-13));
}

TEST_CASE("Step 1 treats u and v symmetrically", "[cbs][step1]") {
    // Swapping the velocity components on a mesh mirrored about y = x (with
    // connectivity reordered to stay counter-clockwise) swaps the results.
    GridSpec g;
    g.lx = 1.0;
    g.ly = 1.0;
    g.nx = 4;
    g.ny = 4;
    Mesh m = make_grid_mesh(g);
    Mesh mirrored = m;
    std::swap(mirrored.x, mirrored.y);
    for (auto& e : mirrored.intma) std::swap(e[1], e[2]);
    compute_all_geometry(m);
    compute_all_geometry(mirrored);

    const auto u = sample(m, tg_u), v = sample(m, tg_v);
    const auto delte = varying_dt(m.nelem);
    const Rhs r = step1(m, 0.02, delte, u, v);
    const Rhs rm = step1(mirrored, 0.02, delte, v, u);
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i);
        CHECK_THAT(rm.u[i], WithinAbs(r.v[i], 1e-14));
        CHECK_THAT(rm.v[i], WithinAbs(r.u[i], 1e-14));
    }
}

TEST_CASE("Step 1 on the Hinterkante mesh", "[cbs][step1][hinterkante]") {
    const std::string dir = HYDRAMA2D_CASE_DIR;
    Mesh m;
    REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", m));
    REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", m));
    compute_all_geometry(m);
    const double nu = 1.0 / 200.0;
    const auto delte = varying_dt(m.nelem);

    SECTION("uniform inflow produces no update") {
        const Rhs r = step1(m, nu, delte, std::vector<double>(m.npoin, 1.0), std::vector<double>(m.npoin, 0.0));
        CHECK(scale_of(r.u) < 1e-14);
        CHECK(scale_of(r.v) < 1e-14);
    }
    SECTION("non-linear flow matches the convection + stabilisation reference") {
        // The dense viscous reference is too large here; use nu = 0.
        auto fu = [](double x, double y) { return 1.0 + 0.1 * std::sin(0.5 * x) * std::cos(y); };
        auto fv = [](double x, double y) { return 0.05 * std::cos(0.3 * x) * std::sin(0.7 * y); };
        const auto u = sample(m, fu), v = sample(m, fv);
        const Rhs r = step1(m, 0.0, delte, u, v);
        auto ru = convection_ref(m, u, v, u), rv = convection_ref(m, u, v, v);
        const auto su = stabilisation_ref(m, delte, u, v, u), sv = stabilisation_ref(m, delte, u, v, v);
        for (int i = 0; i < m.npoin; ++i) {
            ru[i] += su[i];
            rv[i] += sv[i];
        }
        double worst = 0.0;
        for (int i = 0; i < m.npoin; ++i)
            worst = std::max({worst, std::abs(r.u[i] - ru[i]), std::abs(r.v[i] - rv[i])});
        CAPTURE(worst, scale_of(ru));
        CHECK(worst < 1e-13 * scale_of(ru));
    }
}
