// Matrix-free pressure operator:
//   update_lhs()         scales the element data by the element time step:
//                          elcoe2[ip]  = 1 / sum_e (elcoe_e / dt_e)   (inverse lumped mass / dt)
//                          pdiag[ip]   = sum_e dt_e * K_e,ii          (assembled diagonal)
//                          gstif[ie]   = dt_e * (K_e,01, K_e,02, K_e,12)
//   matrix_vector_mult() applies the assembled symmetric operator
//                          b = diag(pdiag) x + sum_e (element off-diagonal couplings) x

#include "LinearSolverTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

void check_matrices_equal(const DenseMatrix& a, const DenseMatrix& b, double tol) {
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a.size(); ++j) {
            CAPTURE(i, j);
            CHECK_THAT(a[i][j], WithinAbs(b[i][j], tol));
        }
    }
}

void check_symmetric(const DenseMatrix& a, double tol) {
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = i + 1; j < a.size(); ++j) {
            CAPTURE(i, j);
            CHECK_THAT(a[i][j], WithinAbs(a[j][i], tol));
        }
    }
}

GridSpec small_grid() {
    GridSpec g;
    g.x0 = 0.5;
    g.y0 = -1.0;
    g.lx = 1.5;
    g.ly = 1.0;
    g.nx = 3;
    g.ny = 2;
    return g;
}

} // namespace

// ---------------------------------------------------------------------------
// matrix_vector_mult
// ---------------------------------------------------------------------------

TEST_CASE("matrix_vector_mult maps element pairs to global nodes", "[linear][matvec]") {
    // One element with permuted connectivity: local nodes (0,1,2) = global (2,0,1).
    // gstif pairs: (0,1) -> global (2,0), (0,2) -> (2,1), (1,2) -> (0,1).
    Mesh m = make_mesh({{0, 0}, {1, 0}, {0, 1}}, {{2, 0, 1}});
    const std::vector<std::array<double, 3>> gstif = {{3.0, 5.0, 7.0}};
    const std::vector<double> pdiag = {11.0, 13.0, 17.0};

    const DenseMatrix a = dense_operator(m, gstif, pdiag);
    const DenseMatrix expected = {{11.0, 7.0, 3.0},
                                  {7.0, 13.0, 5.0},
                                  {3.0, 5.0, 17.0}};
    check_matrices_equal(a, expected, 0.0);
}

TEST_CASE("matrix_vector_mult accumulates couplings of shared edges", "[linear][matvec]") {
    // Two elements sharing edge (1,2): elements {0,1,2} and {1,3,2}.
    Mesh m = make_mesh({{0, 0}, {1, 0}, {0, 1}, {1, 1}}, {{0, 1, 2}, {1, 3, 2}});
    const std::vector<std::array<double, 3>> gstif = {{1.0, 2.0, 3.0},   // (0,1) (0,2) (1,2)
                                                      {4.0, 5.0, 6.0}};  // (1,3) (1,2) (3,2)
    const std::vector<double> pdiag = {10.0, 20.0, 30.0, 40.0};

    const DenseMatrix a = dense_operator(m, gstif, pdiag);
    const DenseMatrix expected = {{10.0, 1.0, 2.0, 0.0},
                                  {1.0, 20.0, 3.0 + 5.0, 4.0},
                                  {2.0, 3.0 + 5.0, 30.0, 6.0},
                                  {0.0, 4.0, 6.0, 40.0}};
    check_matrices_equal(a, expected, 0.0);
}

TEST_CASE("matrix_vector_mult is linear and overwrites its output", "[linear][matvec]") {
    const PoissonSystem s = make_poisson_system(make_grid_mesh(small_grid()));
    const int n = s.mesh.npoin;
    std::vector<double> x(n), y(n), comb(n);
    for (int i = 0; i < n; ++i) {
        x[i] = std::sin(1.0 + i);
        y[i] = std::cos(0.5 * i) - 0.3;
        comb[i] = 2.5 * x[i] - 4.0 * y[i];
    }

    std::vector<double> ax, ay, acomb(3 * n, 99.0); // stale, wrongly sized output
    matrix_vector_mult(s.mesh, s.gstif, s.pdiag, x, ax);
    matrix_vector_mult(s.mesh, s.gstif, s.pdiag, y, ay);
    matrix_vector_mult(s.mesh, s.gstif, s.pdiag, comb, acomb);

    REQUIRE(static_cast<int>(acomb.size()) == n);
    for (int i = 0; i < n; ++i) {
        CAPTURE(i);
        CHECK_THAT(acomb[i], WithinAbs(2.5 * ax[i] - 4.0 * ay[i], 1e-12));
    }
}

TEST_CASE("matrix_vector_mult is self-adjoint: x.(A y) == y.(A x)", "[linear][matvec]") {
    const PoissonSystem s = make_poisson_system(make_grid_mesh(small_grid()), varying_dt(12));
    const int n = s.mesh.npoin;
    std::vector<double> x(n), y(n), ax, ay;
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0 + 0.1 * i * i;
        y[i] = (i % 3) - 0.7 * i;
    }
    matrix_vector_mult(s.mesh, s.gstif, s.pdiag, x, ax);
    matrix_vector_mult(s.mesh, s.gstif, s.pdiag, y, ay);
    CHECK_THAT(dot(x, ay), WithinRel(dot(y, ax), 1e-12));
}

// ---------------------------------------------------------------------------
// update_lhs: assembled operator on a small grid
// ---------------------------------------------------------------------------

TEST_CASE("Assembled pressure operator matches the dense reference", "[linear][lhs]") {
    const GridSpec g = small_grid();
    Mesh mesh = make_grid_mesh(g);

    SECTION("uniform time step") {
        const double dt = GENERATE(1.0, 0.02);
        CAPTURE(dt);
        const PoissonSystem s = make_poisson_system(mesh, std::vector<double>(g.nelem(), dt));
        const DenseMatrix a = dense_operator(s.mesh, s.gstif, s.pdiag);
        check_matrices_equal(a, dense_reference_laplacian(s.mesh, s.delte), 1e-14);
    }
    SECTION("element-wise (local) time steps") {
        const PoissonSystem s = make_poisson_system(mesh, varying_dt(g.nelem()));
        const DenseMatrix a = dense_operator(s.mesh, s.gstif, s.pdiag);
        check_matrices_equal(a, dense_reference_laplacian(s.mesh, s.delte), 1e-14);
    }
}

TEST_CASE("Assembled pressure operator is symmetric", "[linear][lhs]") {
    const GridSpec g = GENERATE(small_grid(), GridSpec{-1.0, 2.0, 1.0, 3.0, 2, 4});
    const bool local_dt = GENERATE(false, true);
    CAPTURE(g.nx, g.ny, local_dt);

    const PoissonSystem s = make_poisson_system(
        make_grid_mesh(g), local_dt ? varying_dt(g.nelem()) : std::vector<double>{});
    const DenseMatrix a = dense_operator(s.mesh, s.gstif, s.pdiag);
    check_symmetric(a, 1e-15 * max_abs(a));
}

TEST_CASE("Assembled pressure operator structure on a uniform grid", "[linear][lhs]") {
    const GridSpec g = small_grid();
    const PoissonSystem s = make_poisson_system(make_grid_mesh(g));
    const DenseMatrix a = dense_operator(s.mesh, s.gstif, s.pdiag);
    const int n = s.mesh.npoin;

    SECTION("constants are in the null space (zero row sums)") {
        for (int i = 0; i < n; ++i) {
            double row = 0.0;
            for (int j = 0; j < n; ++j) row += a[i][j];
            CAPTURE(i);
            CHECK_THAT(row, WithinAbs(0.0, 1e-14));
        }
    }
    SECTION("positive diagonal, non-positive couplings (M-matrix pattern)") {
        for (int i = 0; i < n; ++i) {
            CAPTURE(i);
            CHECK(a[i][i] > 0.0);
            for (int j = 0; j < n; ++j) {
                if (i != j) CHECK(a[i][j] <= 1e-15);
            }
        }
    }
    SECTION("only nodes sharing an element are coupled") {
        std::set<std::pair<int, int>> connected;
        for (const auto& e : s.mesh.intma)
            for (int p : e)
                for (int q : e) connected.insert({p, q});
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!connected.count({i, j})) {
                    CAPTURE(i, j);
                    CHECK(a[i][j] == 0.0);
                }
            }
        }
    }
}

TEST_CASE("Pressure operator is SPD once a reference node is fixed", "[linear][lhs]") {
    // The full operator is singular (constants), but removing any Dirichlet
    // node - as the solver does with its reference pressure node - leaves a
    // symmetric positive definite matrix, which PCG requires.
    const GridSpec g = small_grid();
    const bool local_dt = GENERATE(false, true);
    CAPTURE(local_dt);
    const PoissonSystem s = make_poisson_system(
        make_grid_mesh(g), local_dt ? varying_dt(g.nelem()) : std::vector<double>{});
    const DenseMatrix a = dense_operator(s.mesh, s.gstif, s.pdiag);

    // Singular: the constant vector is a null vector (a Cholesky test is not
    // used here because the last pivot of a singular matrix is only ~0 in
    // floating point).
    CHECK(norm2(multiply(a, std::vector<double>(a.size(), 1.0))) < 1e-14);
    const int fixed = GENERATE(0, 5);
    CAPTURE(fixed);
    CHECK(cholesky_succeeds(reduced(a, {fixed})));
}

TEST_CASE("update_lhs scales element stiffness by the element time step", "[linear][lhs]") {
    const GridSpec g = small_grid();
    const PoissonSystem s = make_poisson_system(make_grid_mesh(g), varying_dt(g.nelem()));
    const Mesh& m = s.mesh;

    REQUIRE(static_cast<int>(s.gstif.size()) == m.nelem);
    for (int ie = 0; ie < m.nelem; ++ie) {
        for (int k = 0; k < 3; ++k) {
            CAPTURE(ie, k);
            CHECK_THAT(s.gstif[ie][k], WithinRel(s.delte[ie] * m.gstifE[ie][k], 1e-15));
        }
    }

    std::vector<double> diag(m.npoin, 0.0);
    for (int ie = 0; ie < m.nelem; ++ie)
        for (int in = 0; in < 3; ++in) diag[m.intma[ie][in]] += s.delte[ie] * m.pdiagE[ie][in];
    REQUIRE(static_cast<int>(s.pdiag.size()) == m.npoin);
    for (int ip = 0; ip < m.npoin; ++ip) {
        CAPTURE(ip);
        CHECK_THAT(s.pdiag[ip], WithinRel(diag[ip], 1e-14));
    }
}

TEST_CASE("update_lhs inverse lumped mass with element-wise time steps", "[linear][lhs]") {
    // elcoe2[ip] = 1 / sum_e (A_e / 3 / dt_e)
    const GridSpec g = small_grid();
    const PoissonSystem s = make_poisson_system(make_grid_mesh(g), varying_dt(g.nelem()));
    const Mesh& m = s.mesh;

    std::vector<double> sum(m.npoin, 0.0);
    for (int ie = 0; ie < m.nelem; ++ie)
        for (int in = 0; in < 3; ++in) sum[m.intma[ie][in]] += 0.5 * m.detJ[ie] / 3.0 / s.delte[ie];

    REQUIRE(static_cast<int>(s.elcoe2.size()) == m.npoin);
    for (int ip = 0; ip < m.npoin; ++ip) {
        CAPTURE(ip);
        CHECK_THAT(s.elcoe2[ip], WithinRel(1.0 / sum[ip], 1e-14));
    }

    SECTION("hand check: corner node (nx, 0) lies in a single element") {
        const int ip = g.node(g.nx, 0);
        const int ie = 0 + 2 * (g.nx - 1); // first triangle of the bottom-right cell
        const double area = 0.5 * g.hx() * g.hy();
        CHECK_THAT(s.elcoe2[ip], WithinRel(3.0 * s.delte[ie] / area, 1e-14));
    }
}

TEST_CASE("update_lhs overwrites stale outputs", "[linear][lhs]") {
    Mesh m = make_grid_mesh(small_grid());
    compute_all_geometry(m);
    const std::vector<double> delte(m.nelem, 0.5);

    std::vector<double> elcoe2_a, pdiag_a;
    std::vector<std::array<double, 3>> gstif_a;
    update_lhs(m, delte, elcoe2_a, pdiag_a, gstif_a);

    std::vector<double> elcoe2_b(2, 7.0), pdiag_b(50, -1.0);
    std::vector<std::array<double, 3>> gstif_b(3, {9.0, 9.0, 9.0});
    update_lhs(m, delte, elcoe2_b, pdiag_b, gstif_b); // first call with junk
    update_lhs(m, delte, elcoe2_b, pdiag_b, gstif_b); // second call must not accumulate

    CHECK(elcoe2_b == elcoe2_a);
    CHECK(pdiag_b == pdiag_a);
    CHECK(gstif_b == gstif_a);
}
