// CBS Step 2 (pressure Poisson right-hand side): compute_step2() assembles
//
//   rhs1_i = - int N_i div(u^n) dOmega + theta1 * int grad N_i . (u* - u^n) dOmega
//
// which the solver passes to solve_pcg together with the dt-scaled Laplacian.
// Integrating the second term by parts gives the identity used below:
//
//   rhs1_i = - int N_i div(u^n + theta1 (u* - u^n)) dOmega
//            + theta1 * closed int N_i (u* - u^n) . n ds

#include "IO.hpp"
#include "MeshTestUtils.hpp"
#include "Step2.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <numeric>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

struct Fields {
    std::vector<double> us, vs; // u*, v*
    std::vector<double> un, vn; // u^n, v^n
};

std::vector<double> step2(const Mesh& m, const Fields& f, double theta) {
    std::vector<double> rhs(5, -1.0); // stale, wrongly sized output
    compute_step2(m, f.us, f.vs, f.un, f.vn, theta, rhs);
    return rhs;
}

// Reference from the integrated-by-parts form (see file header):
//   domain part:   -div_e(w) * A_e / 3 per element node, w = u^n + theta * du
//   boundary part: theta * L / 6 * (2 du_i + du_j) . n on every boundary side
std::vector<double> step2_ref(const Mesh& m, const Fields& f, double theta) {
    std::vector<double> r(m.npoin, 0.0);
    for (int ie = 0; ie < m.nelem; ++ie) {
        const auto& n = m.intma[ie];
        const auto& d = m.dNkdx[ie];
        double div = 0.0;
        for (int k = 0; k < 3; ++k) {
            const int p = n[k];
            div += d.dNdx[k] * (f.un[p] + theta * (f.us[p] - f.un[p]));
            div += d.dNdy[k] * (f.vn[p] + theta * (f.vs[p] - f.vn[p]));
        }
        for (int k = 0; k < 3; ++k) r[n[k]] -= div * 0.5 * m.detJ[ie] / 3.0;
    }
    for (int ib = 0; ib < m.nboun; ++ib) {
        const int a = m.iside[ib].node1, b = m.iside[ib].node2;
        const auto& fn = m.face_norm[ib];
        auto dun = [&](int p) { return (f.us[p] - f.un[p]) * fn.nx + (f.vs[p] - f.vn[p]) * fn.ny; };
        r[a] += theta * fn.length / 6.0 * (2.0 * dun(a) + dun(b));
        r[b] += theta * fn.length / 6.0 * (2.0 * dun(b) + dun(a));
    }
    return r;
}

std::vector<double> lumped_mass(const Mesh& m) {
    std::vector<double> mass(m.npoin, 0.0);
    for (int ie = 0; ie < m.nelem; ++ie)
        for (int k = 0; k < 3; ++k) mass[m.intma[ie][k]] += m.elcoe_e[3 * ie + k];
    return mass;
}

GridSpec step_grid_spec() {
    GridSpec g;
    g.x0 = 0.2;
    g.y0 = -0.3;
    g.lx = 1.2;
    g.ly = 0.9;
    g.nx = 5;
    g.ny = 4;
    return g;
}

Mesh step_grid() {
    Mesh m = make_grid_mesh(step_grid_spec());
    compute_all_geometry(m);
    return m;
}

// Generic non-linear fields for u^n and u*.
Fields generic_fields(const Mesh& m) {
    Fields f;
    f.un = sample(m, [](double x, double y) { return 0.4 + std::sin(2.0 * x) * std::cos(1.5 * y); });
    f.vn = sample(m, [](double x, double y) { return -0.2 - std::cos(2.0 * x) * std::sin(1.5 * y); });
    f.us = sample(m, [](double x, double y) { return 0.5 + std::sin(2.1 * x) * std::cos(1.4 * y) + 0.1 * x * y; });
    f.vs = sample(m, [](double x, double y) { return -0.1 - std::cos(1.9 * x) * std::sin(1.6 * y) + 0.05 * y * y; });
    return f;
}

} // namespace

TEST_CASE("Step 2 on a single element, computed by hand", "[cbs][step2]") {
    // Unit right triangle (A = 1/2), dN/dx = (-1, 1, 0), dN/dy = (-1, 0, 1).
    //   u^n = (x, 0): div = 1  ->  -int N_i div = -A/3 = -1/6 for every node
    //   u* - u^n = (0, 3) uniform -> theta * A * dN_i/dy * 3 = theta * 1.5 * (-1, 0, 1)
    Mesh m = make_triangle_mesh(reference_triangles()[0]);
    compute_all_geometry(m);
    const double theta = GENERATE(0.5, 1.0);
    CAPTURE(theta);

    Fields f;
    f.un = {0.0, 1.0, 0.0};
    f.vn = {0.0, 0.0, 0.0};
    f.us = f.un;
    f.vs = {3.0, 3.0, 3.0};
    const auto r = step2(m, f, theta);
    REQUIRE(r.size() == 3);
    CHECK_THAT(r[0], WithinAbs(-1.0 / 6.0 - 1.5 * theta, 1e-15));
    CHECK_THAT(r[1], WithinAbs(-1.0 / 6.0, 1e-15));
    CHECK_THAT(r[2], WithinAbs(-1.0 / 6.0 + 1.5 * theta, 1e-15));
}

TEST_CASE("Step 2 vanishes for a divergence-free flow with u* = u^n", "[cbs][step2]") {
    // Uniform flow and the linear strain field (x, -y) are exactly
    // divergence-free on every element.
    const Mesh m = step_grid();
    const double theta = GENERATE(0.5, 1.0);
    CAPTURE(theta);

    Fields f;
    SECTION("uniform flow") {
        f.un = std::vector<double>(m.npoin, 1.3);
        f.vn = std::vector<double>(m.npoin, -0.4);
    }
    SECTION("linear strain") {
        f.un = sample(m, [](double x, double) { return 2.0 * x; });
        f.vn = sample(m, [](double, double y) { return -2.0 * y; });
    }
    f.us = f.un;
    f.vs = f.vn;
    for (double x : step2(m, f, theta)) CHECK_THAT(x, WithinAbs(0.0, 1e-14));
}

TEST_CASE("Step 2 with constant divergence is -div times the lumped mass", "[cbs][step2]") {
    // u^n = (a x, b y) has div = a + b everywhere and int N_i = M_i.
    const Mesh m = step_grid();
    const double a = 1.5, b = -0.25;
    Fields f;
    f.un = sample(m, [&](double x, double) { return a * x; });
    f.vn = sample(m, [&](double, double y) { return b * y; });
    f.us = f.un;
    f.vs = f.vn;

    const auto r = step2(m, f, 1.0);
    const auto mass = lumped_mass(m);
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i);
        CHECK_THAT(r[i], WithinRel(-(a + b) * mass[i], 1e-12));
    }
}

TEST_CASE("Step 2 is affine in theta1", "[cbs][step2]") {
    // rhs(theta) = rhs(0) + theta * int grad N_i . (u* - u^n)
    const Mesh m = step_grid();
    const Fields f = generic_fields(m);
    const auto r0 = step2(m, f, 0.0);
    const auto r1 = step2(m, f, 1.0);
    const double theta = GENERATE(0.5, 0.75);
    CAPTURE(theta);
    const auto rt = step2(m, f, theta);
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i);
        CHECK_THAT(rt[i], WithinAbs(r0[i] + theta * (r1[i] - r0[i]), 1e-14));
    }

    SECTION("theta = 0 depends on u^n only") {
        Fields g = f;
        g.us = sample(m, [](double x, double y) { return 10.0 * x - y; });
        g.vs = sample(m, [](double x, double) { return std::exp(x); });
        CHECK(step2(m, g, 0.0) == r0);
    }
}

TEST_CASE("Step 2 satisfies the integration-by-parts identity at every node", "[cbs][step2]") {
    // Interior nodes: rhs1_i = -int N_i div(u^n + theta du);
    // boundary nodes additionally receive theta * int N_i du . n ds.
    const Mesh m = step_grid();
    const Fields f = generic_fields(m);
    const double theta = GENERATE(0.5, 1.0);
    CAPTURE(theta);

    const auto r = step2(m, f, theta);
    const auto ref = step2_ref(m, f, theta);
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i, m.x[i], m.y[i]);
        CHECK_THAT(r[i], WithinAbs(ref[i], 1e-13 * scale_of(ref)));
    }
}

TEST_CASE("Step 2 nodal sum is minus the outflow of u^n", "[cbs][step2]") {
    // sum_i rhs1_i = -int div(u^n) = -closed int u^n . n ds (divergence
    // theorem); the theta term sums to zero because sum_i grad N_i = 0.
    const Mesh m = step_grid();
    const Fields f = generic_fields(m);
    double outflow = 0.0;
    for (int ib = 0; ib < m.nboun; ++ib) {
        const int a = m.iside[ib].node1, b = m.iside[ib].node2;
        const auto& n = m.face_norm[ib];
        outflow += n.length * 0.5 * ((f.un[a] + f.un[b]) * n.nx + (f.vn[a] + f.vn[b]) * n.ny);
    }
    const double theta = GENERATE(0.0, 0.5, 1.0);
    CAPTURE(theta);
    const auto r = step2(m, f, theta);
    CHECK_THAT(std::accumulate(r.begin(), r.end(), 0.0), WithinAbs(-outflow, 1e-13));
}

TEST_CASE("Step 2 on the Hinterkante mesh", "[cbs][step2][hinterkante]") {
    const std::string dir = HYDRAMA2D_CASE_DIR;
    Mesh m;
    REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", m));
    REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", m));
    compute_all_geometry(m);

    SECTION("uniform flow produces a zero right-hand side") {
        Fields f;
        f.un = f.us = std::vector<double>(m.npoin, 1.0);
        f.vn = f.vs = std::vector<double>(m.npoin, 0.0);
        CHECK(scale_of(step2(m, f, 1.0)) < 1e-14);
    }
    SECTION("integration-by-parts identity holds at every node") {
        Fields f;
        f.un = sample(m, [](double x, double y) { return 1.0 + 0.1 * std::sin(0.5 * x) * std::cos(y); });
        f.vn = sample(m, [](double x, double y) { return 0.05 * std::cos(0.3 * x) * std::sin(0.7 * y); });
        f.us = sample(m, [](double x, double y) { return 1.0 + 0.12 * std::sin(0.45 * x) * std::cos(1.1 * y); });
        f.vs = sample(m, [](double x, double y) { return 0.04 * std::cos(0.35 * x) * std::sin(0.6 * y); });
        const auto r = step2(m, f, 1.0);
        const auto ref = step2_ref(m, f, 1.0);
        double worst = 0.0;
        for (int i = 0; i < m.npoin; ++i) worst = std::max(worst, std::abs(r[i] - ref[i]));
        CAPTURE(worst, scale_of(ref));
        CHECK(worst < 1e-12 * scale_of(ref));
    }
}
