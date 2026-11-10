// Field operations: element gradients, velocity gradient tensor, strain-rate
// magnitude S = sqrt(2 S_ij S_ij), vorticity magnitude Omega = sqrt(2 W_ij W_ij),
// lumped nodal mass, lumped L2 projection and FlowKinematics.

#include "MeshTestUtils.hpp"
#include "geometry/FieldOps.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numeric>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

Mesh unit_grid(int nx, int ny, double lx = 1.0, double ly = 1.0, double x0 = 0.0, double y0 = 0.0) {
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

// S and Omega from the tensor definitions (independent of FieldOps).
double strain_from_tensor(const VelocityGradient& g) {
    const double L[2][2] = {{g.dudx, g.dudy}, {g.dvdx, g.dvdy}}; // L_ij = du_i/dx_j
    double sum = 0.0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            const double s = 0.5 * (L[i][j] + L[j][i]);
            sum += s * s;
        }
    return std::sqrt(2.0 * sum);
}

double vorticity_from_tensor(const VelocityGradient& g) {
    const double L[2][2] = {{g.dudx, g.dudy}, {g.dvdx, g.dvdy}};
    double sum = 0.0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            const double w = 0.5 * (L[i][j] - L[j][i]);
            sum += w * w;
        }
    return std::sqrt(2.0 * sum);
}

} // namespace

TEST_CASE("Strain rate and vorticity of elementary flows", "[field_ops]") {
    const double a = GENERATE(0.5, -2.0);
    CAPTURE(a);
    // Simple shear u = a y: S = Omega = |a|.
    CHECK_THAT(strain_rate_magnitude({0.0, a, 0.0, 0.0}), WithinRel(std::abs(a), 1e-15));
    CHECK_THAT(vorticity_magnitude({0.0, a, 0.0, 0.0}), WithinRel(std::abs(a), 1e-15));
    // Solid-body rotation u = -a y, v = a x: S = 0, Omega = 2|a|.
    CHECK(strain_rate_magnitude({0.0, -a, a, 0.0}) == 0.0);
    CHECK_THAT(vorticity_magnitude({0.0, -a, a, 0.0}), WithinRel(2.0 * std::abs(a), 1e-15));
    // Pure strain u = a x, v = -a y: S = 2|a|, Omega = 0.
    CHECK_THAT(strain_rate_magnitude({a, 0.0, 0.0, -a}), WithinRel(2.0 * std::abs(a), 1e-15));
    CHECK(vorticity_magnitude({a, 0.0, 0.0, -a}) == 0.0);
}

TEST_CASE("Strain rate and vorticity match the tensor definitions and are frame invariant", "[field_ops]") {
    const VelocityGradient g = GENERATE(VelocityGradient{0.3, -1.2, 0.7, -0.3}, VelocityGradient{1.0, 2.0, 3.0, 4.0},
                                        VelocityGradient{-0.5, 0.0, 0.25, 0.9});
    CHECK_THAT(strain_rate_magnitude(g), WithinRel(strain_from_tensor(g), 1e-14));
    CHECK_THAT(vorticity_magnitude(g), WithinAbs(vorticity_from_tensor(g), 1e-14));

    // Rotated frame: L' = R L R^T.
    for (double th : {0.3, 1.1, 2.7}) {
        const double c = std::cos(th), s = std::sin(th);
        const double R[2][2] = {{c, -s}, {s, c}};
        const double L[2][2] = {{g.dudx, g.dudy}, {g.dvdx, g.dvdy}};
        double Lr[2][2] = {{0, 0}, {0, 0}};
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k)
                    for (int l = 0; l < 2; ++l) Lr[i][j] += R[i][k] * L[k][l] * R[j][l];
        const VelocityGradient gr{Lr[0][0], Lr[0][1], Lr[1][0], Lr[1][1]};
        CHECK_THAT(strain_rate_magnitude(gr), WithinRel(strain_rate_magnitude(g), 1e-13));
        CHECK_THAT(vorticity_magnitude(gr), WithinAbs(vorticity_magnitude(g), 1e-13));
    }
}

TEST_CASE("Element gradients are exact for linear fields", "[field_ops]") {
    const Mesh m = unit_grid(5, 4, 1.2, 0.9, 0.2, -0.3);
    auto fu = [](double x, double y) { return 0.5 + 1.2 * x - 0.7 * y; };
    auto fv = [](double x, double y) { return -0.3 + 0.4 * x + 0.9 * y; };
    const auto u = sample(m, fu), v = sample(m, fv);

    std::vector<VelocityGradient> g;
    compute_velocity_gradients(m, u, v, g);
    std::vector<std::array<double, 2>> gs;
    compute_scalar_gradients(m, u, gs);
    REQUIRE(static_cast<int>(g.size()) == m.nelem);
    for (int ie = 0; ie < m.nelem; ++ie) {
        CHECK_THAT(g[ie].dudx, WithinAbs(1.2, 1e-13));
        CHECK_THAT(g[ie].dudy, WithinAbs(-0.7, 1e-13));
        CHECK_THAT(g[ie].dvdx, WithinAbs(0.4, 1e-13));
        CHECK_THAT(g[ie].dvdy, WithinAbs(0.9, 1e-13));
        CHECK_THAT(gs[ie][0], WithinAbs(1.2, 1e-13));
        CHECK_THAT(gs[ie][1], WithinAbs(-0.7, 1e-13));
    }

    SECTION("on arbitrary single triangles") {
        for (const Triangle& t : reference_triangles()) {
            Mesh tm = make_triangle_mesh(t);
            tm.compute_shape_function_derivatives();
            std::vector<VelocityGradient> tg;
            compute_velocity_gradients(tm, sample(tm, fu), sample(tm, fv), tg);
            INFO(t);
            // Absolute error scales with the coordinate magnitude / element size.
            const double tol = 1e-9;
            CHECK_THAT(tg[0].dudx, WithinAbs(1.2, tol));
            CHECK_THAT(tg[0].dudy, WithinAbs(-0.7, tol));
            CHECK_THAT(tg[0].dvdx, WithinAbs(0.4, tol));
            CHECK_THAT(tg[0].dvdy, WithinAbs(0.9, tol));
        }
    }
}

TEST_CASE("Lumped nodal mass and lumped L2 projection", "[field_ops]") {
    const Mesh m = unit_grid(6, 5, 1.2, 1.0);
    const std::vector<double> mass = lumped_nodal_mass(m);
    CHECK_THAT(std::accumulate(mass.begin(), mass.end(), 0.0), WithinRel(1.2, 1e-14));
    // Same as the legacy element mass contributions.
    std::vector<double> legacy(m.npoin, 0.0);
    for (int ie = 0; ie < m.nelem; ++ie)
        for (int k = 0; k < 3; ++k) legacy[m.intma[ie][k]] += m.elcoe_e[ie * 3 + k];
    for (int ip = 0; ip < m.npoin; ++ip) CHECK_THAT(mass[ip], WithinRel(legacy[ip], 1e-15));

    SECTION("constants are reproduced") {
        std::vector<double> nodal;
        project_to_nodes(m, mass, std::vector<double>(m.nelem, 3.25), nodal);
        for (double x : nodal) CHECK_THAT(x, WithinRel(3.25, 1e-14));
    }

    SECTION("the integral is preserved: sum_i M_i f_i = sum_e A_e f_e") {
        std::vector<double> fe(m.nelem);
        for (int ie = 0; ie < m.nelem; ++ie) fe[ie] = std::sin(0.37 * ie) + 2.0;
        std::vector<double> nodal;
        project_to_nodes(m, mass, fe, nodal);
        double lhs = 0.0, rhs = 0.0;
        for (int ip = 0; ip < m.npoin; ++ip) lhs += mass[ip] * nodal[ip];
        for (int ie = 0; ie < m.nelem; ++ie) rhs += 0.5 * m.detJ[ie] * fe[ie];
        CHECK_THAT(lhs, WithinRel(rhs, 1e-14));
    }

    SECTION("a linear field sampled at centroids is exact at interior nodes of a uniform grid") {
        auto f = [](double x, double y) { return 1.0 + 2.0 * x - 3.0 * y; };
        std::vector<double> fe(m.nelem);
        for (int ie = 0; ie < m.nelem; ++ie) {
            const auto& n = m.intma[ie];
            fe[ie] = f((m.x[n[0]] + m.x[n[1]] + m.x[n[2]]) / 3.0, (m.y[n[0]] + m.y[n[1]] + m.y[n[2]]) / 3.0);
        }
        std::vector<double> nodal;
        project_to_nodes(m, mass, fe, nodal);
        for (int ip = 0; ip < m.npoin; ++ip) {
            const bool interior = m.x[ip] > 1e-9 && m.x[ip] < 1.2 - 1e-9 && m.y[ip] > 1e-9 && m.y[ip] < 1.0 - 1e-9;
            if (interior) CHECK_THAT(nodal[ip], WithinAbs(f(m.x[ip], m.y[ip]), 1e-13));
        }
    }

    SECTION("element mean of a linear nodal field is its centroid value") {
        auto f = [](double x, double y) { return 0.3 - x + 0.5 * y; };
        std::vector<double> em;
        element_mean(m, sample(m, f), em);
        for (int ie = 0; ie < m.nelem; ++ie) {
            const auto& n = m.intma[ie];
            const double xc = (m.x[n[0]] + m.x[n[1]] + m.x[n[2]]) / 3.0;
            const double yc = (m.y[n[0]] + m.y[n[1]] + m.y[n[2]]) / 3.0;
            CHECK_THAT(em[ie], WithinAbs(f(xc, yc), 1e-14));
        }
    }
}

TEST_CASE("FlowKinematics: element and nodal S, Omega", "[field_ops]") {
    const GradientRecovery method = GENERATE(GradientRecovery::PolynomialPreserving, GradientRecovery::LumpedProjection);
    CAPTURE(to_string(method));
    SECTION("linear velocity: every element and node carries the exact values") {
        const Mesh m = unit_grid(4, 3, 1.0, 0.75);
        const auto mass = lumped_nodal_mass(m);
        const auto u = sample(m, [](double x, double y) { return 0.1 + 0.8 * x + 1.5 * y; });
        const auto v = sample(m, [](double x, double y) { return 0.2 - 0.3 * x - 0.8 * y; });
        const VelocityGradient exact{0.8, 1.5, -0.3, -0.8};
        FlowKinematics k;
        k.compute(m, NodalGradient(m, method), u, v);
        for (int ie = 0; ie < m.nelem; ++ie) {
            CHECK_THAT(k.strain_elem[ie], WithinRel(strain_rate_magnitude(exact), 1e-12));
            CHECK_THAT(k.vorticity_elem[ie], WithinRel(vorticity_magnitude(exact), 1e-12));
        }
        for (int ip = 0; ip < m.npoin; ++ip) {
            CHECK_THAT(k.grad_node[ip].dudy, WithinAbs(1.5, 1e-12));
            CHECK_THAT(k.strain_node[ip], WithinRel(strain_rate_magnitude(exact), 1e-12));
            CHECK_THAT(k.vorticity_node[ip], WithinRel(vorticity_magnitude(exact), 1e-12));
        }
    }

    SECTION("Poiseuille flow: nodal vorticity vanishes on the centre line") {
        // u = 1 - (2y - 1)^2 on [0, 2] x [0, 1]: du/dy = -4 (2y - 1) changes
        // sign at y = 1/2. Recovering the gradient (not |Omega|) gives 0 there.
        const int ny = 8;
        const Mesh m = unit_grid(6, ny, 2.0, 1.0);
        const auto mass = lumped_nodal_mass(m);
        const auto u = sample(m, [](double, double y) { return 1.0 - (2.0 * y - 1.0) * (2.0 * y - 1.0); });
        const std::vector<double> v(m.npoin, 0.0);
        FlowKinematics k;
        k.compute(m, NodalGradient(m, method), u, v);

        std::vector<double> omega_mag_projected;
        project_to_nodes(m, mass, k.vorticity_elem, omega_mag_projected);
        int centre = 0;
        for (int ip = 0; ip < m.npoin; ++ip) {
            if (std::abs(m.y[ip] - 0.5) > 1e-12 || m.x[ip] < 1e-9 || m.x[ip] > 2.0 - 1e-9) continue;
            ++centre;
            CHECK_THAT(k.vorticity_node[ip], WithinAbs(0.0, 1e-12));
            CHECK(omega_mag_projected[ip] > 0.4); // the magnitude projection would not vanish (|du/dy| = 4h = 0.5)
        }
        CHECK(centre == 5);

        // Interior nodes: the recovered gradient is exact for this quadratic
        // profile on a uniform grid (symmetric patches, both methods).
        for (int ip = 0; ip < m.npoin; ++ip) {
            const bool interior = m.x[ip] > 1e-9 && m.x[ip] < 2.0 - 1e-9 && m.y[ip] > 1e-9 && m.y[ip] < 1.0 - 1e-9;
            if (interior) CHECK_THAT(k.grad_node[ip].dudy, WithinAbs(-4.0 * (2.0 * m.y[ip] - 1.0), 1e-12));
        }
    }
}

// ---------------------------------------------------------------------------
// Recovered nodal gradients (polynomial preserving recovery vs lumped projection)
// ---------------------------------------------------------------------------

namespace {

// Uniform grid with interior nodes displaced by up to 0.25 h (deterministic):
// irregular patches, boundary nodes stay on the boundary.
Mesh perturbed_grid(int n) {
    GridSpec g;
    g.nx = n;
    g.ny = n;
    Mesh m = make_grid_mesh(g);
    const double h = 1.0 / n;
    for (int j = 1; j < n; ++j) {
        for (int i = 1; i < n; ++i) {
            const int ip = g.node(i, j);
            m.x[ip] += 0.25 * h * std::sin(12.9898 * i + 78.233 * j);
            m.y[ip] += 0.25 * h * std::cos(39.3468 * i + 11.135 * j);
        }
    }
    compute_all_geometry(m);
    return m;
}

double quad_f(double x, double y) { return 0.7 - 1.3 * x + 2.1 * y + 0.9 * x * x - 1.7 * x * y + 0.4 * y * y; }
double quad_fx(double x, double y) { return -1.3 + 1.8 * x - 1.7 * y; }
double quad_fy(double x, double y) { return 2.1 - 1.7 * x + 0.8 * y; }
double smooth_u(double x, double y) { return std::sin(2.0 * x + 0.5) * std::cos(1.5 * y); }
double smooth_dudy(double x, double y) { return -1.5 * std::sin(2.0 * x + 0.5) * std::sin(1.5 * y); }

bool on_boundary(const Mesh& m, int ip) {
    for (const auto& s : m.iside)
        if (s.node1 == ip || s.node2 == ip) return true;
    return false;
}

// Max error of the recovered du/dy over boundary or interior nodes.
double recovery_error(const Mesh& m, GradientRecovery method, bool boundary) {
    const NodalGradient g(m, method);
    std::vector<VelocityGradient> gn;
    g.velocity_gradient(sample(m, smooth_u), std::vector<double>(m.npoin, 0.0), gn);
    double err = 0.0;
    for (int ip = 0; ip < m.npoin; ++ip) {
        if (on_boundary(m, ip) != boundary) continue;
        err = std::max(err, std::abs(gn[ip].dudy - smooth_dudy(m.x[ip], m.y[ip])));
    }
    return err;
}

} // namespace

TEST_CASE("Nodal gradient recovery: method names", "[field_ops][recovery]") {
    CHECK(parse_gradient_recovery("ppr") == GradientRecovery::PolynomialPreserving);
    CHECK(parse_gradient_recovery("polynomial-preserving") == GradientRecovery::PolynomialPreserving);
    CHECK(parse_gradient_recovery("lumped") == GradientRecovery::LumpedProjection);
    CHECK(parse_gradient_recovery("lumped-projection") == GradientRecovery::LumpedProjection);
    CHECK_FALSE(parse_gradient_recovery("spr").has_value());
    CHECK(to_string(GradientRecovery::PolynomialPreserving) == "ppr");
    CHECK(to_string(GradientRecovery::LumpedProjection) == "lumped-projection");
}

TEST_CASE("Nodal gradient recovery: polynomial reproduction", "[field_ops][recovery]") {
    const bool perturbed = GENERATE(false, true);
    const Mesh m = perturbed ? perturbed_grid(9) : unit_grid(5, 4, 1.0, 0.8);
    CAPTURE(perturbed);

    for (GradientRecovery method : {GradientRecovery::PolynomialPreserving, GradientRecovery::LumpedProjection}) {
        CAPTURE(to_string(method));
        const NodalGradient g(m, method);
        REQUIRE(g.npoin() == m.npoin);
        CHECK(g.method() == method);
        CHECK(g.fallback_nodes() == 0);

        std::vector<std::array<double, 2>> grad;
        // Constants: zero gradient; linear fields: exact (both methods).
        g.gradient(std::vector<double>(m.npoin, 3.0), grad);
        for (const auto& gr : grad) {
            CHECK_THAT(gr[0], WithinAbs(0.0, 1e-11));
            CHECK_THAT(gr[1], WithinAbs(0.0, 1e-11));
        }
        g.gradient(sample(m, [](double x, double y) { return 0.5 - 2.0 * x + 3.0 * y; }), grad);
        for (const auto& gr : grad) {
            CHECK_THAT(gr[0], WithinAbs(-2.0, 1e-11));
            CHECK_THAT(gr[1], WithinAbs(3.0, 1e-11));
        }
        // Quadratic fields: PPR exact at every node incl. boundaries and corners.
        g.gradient(sample(m, quad_f), grad);
        int exact = 0;
        for (int ip = 0; ip < m.npoin; ++ip) {
            if (std::abs(grad[ip][0] - quad_fx(m.x[ip], m.y[ip])) < 1e-10 &&
                std::abs(grad[ip][1] - quad_fy(m.x[ip], m.y[ip])) < 1e-10)
                ++exact;
        }
        if (method == GradientRecovery::PolynomialPreserving) {
            CHECK(exact == m.npoin);
        } else {
            CHECK(exact < m.npoin);
        }
    }
}

TEST_CASE("Nodal gradient recovery: lumped stencil equals the projection of element gradients",
          "[field_ops][recovery]") {
    const Mesh m = perturbed_grid(6);
    const auto mass = lumped_nodal_mass(m);
    const auto phi = sample(m, smooth_u);
    const NodalGradient g(m, GradientRecovery::LumpedProjection);
    std::vector<std::array<double, 2>> grad, ge;
    g.gradient(phi, grad);
    compute_scalar_gradients(m, phi, ge);
    std::vector<double> gx(m.nelem), gy(m.nelem), px, py;
    for (int ie = 0; ie < m.nelem; ++ie) {
        gx[ie] = ge[ie][0];
        gy[ie] = ge[ie][1];
    }
    project_to_nodes(m, mass, gx, px);
    project_to_nodes(m, mass, gy, py);
    for (int ip = 0; ip < m.npoin; ++ip) {
        CHECK_THAT(grad[ip][0], WithinAbs(px[ip], 1e-12));
        CHECK_THAT(grad[ip][1], WithinAbs(py[ip], 1e-12));
    }
}

TEST_CASE("Nodal gradient recovery: PPR converges at second order everywhere", "[field_ops][recovery]") {
    SECTION("uniform grid, boundary nodes: PPR second order, lumped projection first order") {
        auto err = [](int n, GradientRecovery m) { return recovery_error(unit_grid(n, n), m, true); };
        const double ppr = std::log2(err(16, GradientRecovery::PolynomialPreserving) /
                                     err(32, GradientRecovery::PolynomialPreserving));
        const double lumped =
            std::log2(err(16, GradientRecovery::LumpedProjection) / err(32, GradientRecovery::LumpedProjection));
        INFO("boundary nodes: PPR order " << ppr << ", lumped order " << lumped);
        CHECK(ppr > 1.8);
        CHECK(lumped < 1.2);
    }

    SECTION("irregular mesh, interior nodes: PPR second order, lumped projection first order") {
        auto err = [](int n, GradientRecovery m) { return recovery_error(perturbed_grid(n), m, false); };
        const double ppr = std::log2(err(16, GradientRecovery::PolynomialPreserving) /
                                     err(32, GradientRecovery::PolynomialPreserving));
        const double lumped =
            std::log2(err(16, GradientRecovery::LumpedProjection) / err(32, GradientRecovery::LumpedProjection));
        INFO("interior nodes: PPR order " << ppr << ", lumped order " << lumped);
        CHECK(ppr > 1.8);
        CHECK(lumped < 1.3);
    }
}

TEST_CASE("Nodal gradient recovery: falls back to the lumped projection where no fit is possible",
          "[field_ops][recovery]") {
    Mesh m = make_triangle_mesh(reference_triangles()[1]);
    m.compute_shape_function_derivatives();
    const NodalGradient g(m, GradientRecovery::PolynomialPreserving);
    CHECK(g.fallback_nodes() == 3);
    std::vector<std::array<double, 2>> grad;
    g.gradient(sample(m, [](double x, double y) { return 2.0 * x - y; }), grad);
    for (const auto& gr : grad) {
        CHECK_THAT(gr[0], WithinAbs(2.0, 1e-12));
        CHECK_THAT(gr[1], WithinAbs(-1.0, 1e-12));
    }
}
