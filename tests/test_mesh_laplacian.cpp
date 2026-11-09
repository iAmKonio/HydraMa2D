// Element Laplacian: Mesh::compute_laplacian_stiffness() stores the P1
// stiffness matrix K_ij = A * grad N_i . grad N_j of every element as
//   pdiagE[ie][i]   diagonal K_ii, i = 0, 1, 2
//   gstifE[ie][0,1,2] off-diagonals K_01, K_02, K_12
// It is the pressure Poisson operator assembled by update_lhs() and applied
// matrix-free by matrix_vector_mult() inside the PCG solver.

#include "LinearSolver.hpp"
#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <set>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

using Mat3 = std::array<std::array<double, 3>, 3>;

// Full symmetric 3x3 element matrix from the packed storage.
Mat3 element_matrix(const Mesh& m, int ie) {
    Mat3 k{};
    for (int i = 0; i < 3; ++i) k[i][i] = m.pdiagE[ie][i];
    k[0][1] = k[1][0] = m.gstifE[ie][0];
    k[0][2] = k[2][0] = m.gstifE[ie][1];
    k[1][2] = k[2][1] = m.gstifE[ie][2];
    return k;
}

Mat3 stiffness_of(const Triangle& t) {
    Mesh m = make_triangle_mesh(t);
    m.compute_shape_function_derivatives();
    m.compute_laplacian_stiffness();
    REQUIRE(m.pdiagE.size() == 1);
    REQUIRE(m.gstifE.size() == 1);
    return element_matrix(m, 0);
}

// Independent reference: cotangent formula of the P1 Laplacian,
//   K_ij = -1/2 cot(theta_k)  (i != j, theta_k = angle at the third vertex k)
//   K_ii = -sum_{j != i} K_ij
Mat3 cotangent_stiffness(const Triangle& t) {
    auto cot_at = [&](int k) {
        const Point o = t.p[k], a = t.p[(k + 1) % 3], b = t.p[(k + 2) % 3];
        const double ux = a.x - o.x, uy = a.y - o.y, vx = b.x - o.x, vy = b.y - o.y;
        return (ux * vx + uy * vy) / std::abs(ux * vy - uy * vx);
    };
    Mat3 k{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i != j) k[i][j] = -0.5 * cot_at(3 - i - j);
        }
    }
    for (int i = 0; i < 3; ++i) k[i][i] = -(k[i][(i + 1) % 3] + k[i][(i + 2) % 3]);
    return k;
}

double max_abs(const Mat3& k) {
    double r = 0.0;
    for (const auto& row : k)
        for (double v : row) r = std::max(r, std::abs(v));
    return r;
}

double quadratic_form(const Mat3& k, const std::array<double, 3>& f) {
    double e = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) e += f[i] * k[i][j] * f[j];
    return e;
}

// Assembled global operator K (time step 1) applied to a nodal vector via the
// production code path: update_lhs() + matrix_vector_mult().
struct AssembledLaplacian {
    const Mesh& mesh;
    std::vector<double> pdiag;
    std::vector<std::array<double, 3>> gstif;

    explicit AssembledLaplacian(const Mesh& m, double dt = 1.0) : mesh(m) {
        std::vector<double> delte(m.nelem, dt), elcoe2;
        update_lhs(m, delte, elcoe2, pdiag, gstif);
    }
    std::vector<double> apply(const std::vector<double>& x) const {
        std::vector<double> b;
        matrix_vector_mult(mesh, gstif, pdiag, x, b);
        return b;
    }
};

Mesh laplacian_grid_mesh(const GridSpec& g) {
    Mesh m = make_grid_mesh(g);
    m.compute_shape_function_derivatives();
    m.compute_mass_matrix();
    m.compute_laplacian_stiffness();
    return m;
}

} // namespace

TEST_CASE("Element Laplacian of the unit right triangle", "[mesh][laplacian]") {
    // grad N = (-1,-1), (1,0), (0,1); A = 1/2
    // K = 1/2 * [[2, -1, -1], [-1, 1, 0], [-1, 0, 1]]
    Mesh m = make_triangle_mesh(reference_triangles()[0]);
    m.compute_shape_function_derivatives();
    m.compute_laplacian_stiffness();

    CHECK_THAT(m.pdiagE[0][0], WithinAbs(1.0, 1e-15));
    CHECK_THAT(m.pdiagE[0][1], WithinAbs(0.5, 1e-15));
    CHECK_THAT(m.pdiagE[0][2], WithinAbs(0.5, 1e-15));
    CHECK_THAT(m.gstifE[0][0], WithinAbs(-0.5, 1e-15)); // (0,1)
    CHECK_THAT(m.gstifE[0][1], WithinAbs(-0.5, 1e-15)); // (0,2)
    CHECK_THAT(m.gstifE[0][2], WithinAbs(0.0, 1e-15));  // (1,2)
}

TEST_CASE("Element Laplacian of the equilateral triangle", "[mesh][laplacian]") {
    // All angles 60 deg: K_ii = cot(60) = 1/sqrt(3), K_ij = -1/(2 sqrt(3)),
    // independent of the side length.
    const Mat3 k = stiffness_of(reference_triangles()[2]);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            CAPTURE(i, j);
            const double expected = (i == j) ? 1.0 / std::sqrt(3.0) : -0.5 / std::sqrt(3.0);
            CHECK_THAT(k[i][j], WithinRel(expected, 1e-14));
        }
    }
}

TEST_CASE("Element Laplacian matches the cotangent formula", "[mesh][laplacian]") {
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    const Mat3 k = stiffness_of(tri);
    const Mat3 ref = cotangent_stiffness(tri);
    const double tol = 1e-9 * max_abs(ref);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            CAPTURE(i, j);
            CHECK_THAT(k[i][j], WithinAbs(ref[i][j], tol));
        }
    }
}

TEST_CASE("Element Laplacian annihilates constants", "[mesh][laplacian]") {
    // grad of a constant field is zero  =>  every row of K sums to zero.
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    const Mat3 k = stiffness_of(tri);
    const double tol = 1e-12 * max_abs(k);

    for (int i = 0; i < 3; ++i) {
        CAPTURE(i);
        CHECK(k[i][i] > 0.0);
        CHECK_THAT(k[i][0] + k[i][1] + k[i][2], WithinAbs(0.0, tol));
    }
}

TEST_CASE("Element Laplacian energy of a linear field is A |grad f|^2", "[mesh][laplacian]") {
    // f^T K f = integral over the element of |grad f|^2, exact for linear f.
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    const Mat3 k = stiffness_of(tri);
    const double b = 0.7, c = -1.9;

    std::array<double, 3> f{};
    for (int i = 0; i < 3; ++i) f[i] = 4.0 + b * tri.p[i].x + c * tri.p[i].y;

    CHECK_THAT(quadratic_form(k, f), WithinRel(heron_area(tri) * (b * b + c * c), 1e-8));
}

TEST_CASE("Element Laplacian is positive semi-definite", "[mesh][laplacian]") {
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    const Mat3 k = stiffness_of(tri);
    const double tol = 1e-12 * max_abs(k);

    const std::array<double, 3> vectors[] = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, -1.0, 0.0},
        {1.0, 1.0, -2.0}, {0.3, -2.0, 5.0}, {-1.0, 4.0, 2.5}, {1.0, 1.0, 1.0},
    };
    for (const auto& v : vectors) {
        CAPTURE(v[0], v[1], v[2]);
        CHECK(quadratic_form(k, v) >= -tol);
    }
}

TEST_CASE("Element Laplacian is invariant under translation and uniform scaling", "[mesh][laplacian]") {
    // In 2D, A scales as L^2 and |grad N|^2 as 1/L^2: K is scale invariant.
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    const Mat3 k = stiffness_of(tri);
    const double tol = 1e-9 * max_abs(k);

    SECTION("translation") {
        const Mat3 kt = stiffness_of(translated(tri, 6.0, -2.5));
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) CHECK_THAT(kt[i][j], WithinAbs(k[i][j], tol));
    }
    SECTION("scaling") {
        const double s = GENERATE(1.0e-3, 7.0, 1.0e3);
        CAPTURE(s);
        const Mat3 ks = stiffness_of(scaled(tri, s));
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) CHECK_THAT(ks[i][j], WithinAbs(k[i][j], tol));
    }
}

TEST_CASE("Clockwise elements yield a negated element Laplacian", "[mesh][laplacian][characterization]") {
    // Documents current behaviour: K is scaled by the signed area (detJ / 2),
    // so a clockwise element contributes -K (negative definite) to the
    // pressure Poisson matrix. Meshes must be CCW.
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    const Mat3 k = stiffness_of(tri);
    const Mat3 kcw = stiffness_of(reversed(tri)); // vertices 0 and 1 swapped
    const int perm[3] = {1, 0, 2};
    const double tol = 1e-9 * max_abs(k);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) CHECK_THAT(kcw[i][j], WithinAbs(-k[perm[i]][perm[j]], tol));
}

TEST_CASE("Assembled Laplacian on a rectangular grid is the 5-point stencil", "[mesh][laplacian][assembly]") {
    // For this triangulation the diagonal edges carry zero coupling and the
    // P1 operator equals the classic finite-difference 5-point stencil:
    //   centre 2 (hy/hx + hx/hy), x-neighbours -hy/hx, y-neighbours -hx/hy.
    GridSpec g;
    g.lx = 2.0;
    g.ly = 1.2;
    g.nx = 4;
    g.ny = 6;
    const Mesh m = laplacian_grid_mesh(g);
    const AssembledLaplacian lap(m);
    const double rx = g.hy() / g.hx(), ry = g.hx() / g.hy();

    for (int j = 1; j < g.ny; ++j) {
        for (int i = 1; i < g.nx; ++i) {
            std::vector<double> e(m.npoin, 0.0);
            e[g.node(i, j)] = 1.0;
            const std::vector<double> col = lap.apply(e);

            std::vector<double> expected(m.npoin, 0.0);
            expected[g.node(i, j)] = 2.0 * (rx + ry);
            expected[g.node(i - 1, j)] = expected[g.node(i + 1, j)] = -rx;
            expected[g.node(i, j - 1)] = expected[g.node(i, j + 1)] = -ry;

            for (int ip = 0; ip < m.npoin; ++ip) {
                CAPTURE(i, j, ip);
                CHECK_THAT(col[ip], WithinAbs(expected[ip], 1e-12));
            }
        }
    }
}

TEST_CASE("Assembled Laplacian of constant and linear fields", "[mesh][laplacian][assembly]") {
    GridSpec g;
    g.x0 = -1.0;
    g.y0 = 0.5;
    g.lx = 3.0;
    g.ly = 2.0;
    g.nx = 6;
    g.ny = 5;
    const Mesh m = laplacian_grid_mesh(g);
    const AssembledLaplacian lap(m);

    std::set<int> boundary_nodes;
    for (const auto& s : m.iside) {
        boundary_nodes.insert(s.node1);
        boundary_nodes.insert(s.node2);
    }

    SECTION("constant field is in the null space") {
        const std::vector<double> r = lap.apply(std::vector<double>(m.npoin, 3.5));
        for (int ip = 0; ip < m.npoin; ++ip) {
            CAPTURE(ip);
            CHECK_THAT(r[ip], WithinAbs(0.0, 1e-12));
        }
    }
    SECTION("linear field is discretely harmonic at interior nodes") {
        const double b = 1.25, c = -0.5;
        std::vector<double> f(m.npoin);
        for (int ip = 0; ip < m.npoin; ++ip) f[ip] = 2.0 + b * m.x[ip] + c * m.y[ip];
        const std::vector<double> r = lap.apply(f);

        double energy = 0.0;
        for (int ip = 0; ip < m.npoin; ++ip) {
            energy += f[ip] * r[ip];
            if (boundary_nodes.count(ip)) continue;
            CAPTURE(ip);
            CHECK_THAT(r[ip], WithinAbs(0.0, 1e-12));
        }
        // Global energy f^T K f = integral of |grad f|^2 over the domain.
        CHECK_THAT(energy, WithinRel(g.lx * g.ly * (b * b + c * c), 1e-12));
    }
}

TEST_CASE("Assembled Laplacian scales linearly with the time step", "[mesh][laplacian][assembly]") {
    GridSpec g;
    g.nx = 3;
    g.ny = 3;
    const Mesh m = laplacian_grid_mesh(g);
    const double dt = GENERATE(0.5, 1.0e-3);
    CAPTURE(dt);

    const AssembledLaplacian unit(m, 1.0), scaled_lap(m, dt);
    for (int ip = 0; ip < m.npoin; ++ip) {
        CAPTURE(ip);
        CHECK_THAT(scaled_lap.pdiag[ip], WithinRel(dt * unit.pdiag[ip], 1e-14));
    }
    for (int ie = 0; ie < m.nelem; ++ie) {
        for (int k = 0; k < 3; ++k) {
            CAPTURE(ie, k);
            CHECK_THAT(scaled_lap.gstif[ie][k], WithinAbs(dt * unit.gstif[ie][k], 1e-15));
        }
    }
}
