// Core linalg interfaces and their basic implementations:
//   LinearOperator        default diagonal, EbeOperator (matrix-free pressure
//                         operator), ConstrainedOperator (Dirichlet decorator)
//   DirichletConstraints  construction, validation, impose/zero/mask
//   Preconditioner        IdentityPreconditioner, JacobiPreconditioner,
//                         default update() -> setup()

#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "LinalgTestUtils.hpp"
#include "reference/LegacyPcg.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <set>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

GridSpec interface_grid() {
    GridSpec g;
    g.x0 = -0.25;
    g.y0 = 0.5;
    g.lx = 1.5;
    g.ly = 1.0;
    g.nx = 5;
    g.ny = 4;
    return g;
}

PoissonSystem grid_system() {
    const GridSpec g = interface_grid();
    return make_poisson_system(make_grid_mesh(g), varying_dt(g.nelem()));
}

const PoissonSystem& hinterkante_system() {
    static const PoissonSystem s = [] {
        Mesh mesh;
        CaptureStdout quiet;
        const std::string dir = HYDRAMA2D_CASE_DIR;
        REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", mesh));
        REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", mesh));
        return make_poisson_system(std::move(mesh), std::vector<double>(17377, 2.5e-3));
    }();
    return s;
}

} // namespace

// ---------------------------------------------------------------------------
// LinearOperator / EbeOperator
// ---------------------------------------------------------------------------

TEST_CASE("EbeOperator reproduces the original matrix-vector product bit for bit",
          "[linalg][operator]") {
    const bool real_mesh = GENERATE(false, true);
    CAPTURE(real_mesh);
    const PoissonSystem grid = real_mesh ? PoissonSystem{} : grid_system();
    const PoissonSystem& s = real_mesh ? hinterkante_system() : grid;
    const int n = s.mesh.npoin;

    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    REQUIRE(op.size() == static_cast<std::size_t>(n));

    const std::vector<double> x = test_vector(n);
    std::vector<double> y(n), y_legacy, y_wrapper;
    op.apply(x, y);
    legacy::matrix_vector_mult(s.mesh, s.gstif, s.pdiag, x, y_legacy);
    matrix_vector_mult(s.mesh, s.gstif, s.pdiag, x, y_wrapper);
    CHECK(y == y_legacy);
    CHECK(y_wrapper == y_legacy);
}

TEST_CASE("EbeOperator equals the assembled dense Laplacian", "[linalg][operator]") {
    const PoissonSystem s = grid_system();
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    const DenseMatrix ref = dense_reference_laplacian(s.mesh, s.delte);
    const std::vector<double> x = test_vector(s.mesh.npoin, 0.3);
    std::vector<double> y(s.mesh.npoin);
    op.apply(x, y);
    const std::vector<double> y_ref = multiply(ref, x);
    CHECK(max_abs_diff(y, y_ref) < 1e-13 * (1.0 + max_abs(ref)));
}

TEST_CASE("EbeOperator exposes diagonal, traits and name", "[linalg][operator]") {
    const PoissonSystem s = grid_system();
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    std::vector<double> d(s.mesh.npoin);
    op.diagonal(d);
    CHECK(d == s.pdiag);
    CHECK(op.traits().symmetric);
    CHECK(op.name() == "ebe");

    // Symmetry of the actual operator: <A x, y> == <x, A y>.
    const std::vector<double> x = test_vector(s.mesh.npoin, 0.1), y = test_vector(s.mesh.npoin, 2.0);
    std::vector<double> ax(x.size()), ay(y.size());
    op.apply(x, ax);
    op.apply(y, ay);
    CHECK_THAT(la::dot(ax, y), WithinRel(la::dot(x, ay), 1e-12));
}

TEST_CASE("EbeOperator is a view of the current coefficients", "[linalg][operator]") {
    PoissonSystem s = grid_system();
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    const std::vector<double> x = test_vector(s.mesh.npoin);
    std::vector<double> before(x.size()), after(x.size());
    op.apply(x, before);

    // A new time step scales the operator: the view sees the new values.
    update_lhs(s.mesh, std::vector<double>(s.mesh.nelem, 3.0), s.elcoe2, s.pdiag, s.gstif);
    op.apply(x, after);
    std::vector<double> expected;
    legacy::matrix_vector_mult(s.mesh, s.gstif, s.pdiag, x, expected);
    CHECK(after == expected);
    CHECK(after != before);
}

TEST_CASE("EbeOperator validates its input", "[linalg][operator]") {
    const PoissonSystem s = grid_system();
    SECTION("coefficient arrays must match the mesh") {
        std::vector<std::array<double, 3>> short_gstif(s.gstif.begin(), s.gstif.end() - 1);
        std::vector<double> short_pdiag(s.pdiag.begin(), s.pdiag.end() - 1);
        CHECK_THROWS_AS(la::EbeOperator(s.mesh, short_gstif, s.pdiag), std::invalid_argument);
        CHECK_THROWS_AS(la::EbeOperator(s.mesh, s.gstif, short_pdiag), std::invalid_argument);
    }
    SECTION("vectors must match the operator") {
        const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
        std::vector<double> x(s.mesh.npoin, 1.0), y(s.mesh.npoin + 1);
        CHECK_THROWS_AS(op.apply(x, y), std::invalid_argument);
        CHECK_THROWS_AS(op.apply(y, x), std::invalid_argument);
    }
}

TEST_CASE("LinearOperator without a diagonal reports it", "[linalg][operator]") {
    const NoDiagonalOperator op(4);
    std::vector<double> d(4);
    CHECK_THROWS_AS(op.diagonal(d), std::logic_error);
    CHECK_FALSE(op.traits().symmetric); // default traits claim nothing

    la::JacobiPreconditioner jacobi;
    CHECK_THROWS_AS(jacobi.setup(op), std::logic_error);
}

// ---------------------------------------------------------------------------
// DirichletConstraints
// ---------------------------------------------------------------------------

TEST_CASE("DirichletConstraints impose, zero and mask dofs", "[linalg][constraints]") {
    const la::DirichletConstraints c({1, 4}, {2.5, -1.0});
    REQUIRE(c.size() == 2);
    CHECK_FALSE(c.empty());
    CHECK(c.dofs() == std::vector<int>{1, 4});
    CHECK(c.values() == std::vector<double>{2.5, -1.0});

    std::vector<double> x(5, 9.0);
    c.impose(x);
    CHECK(x == std::vector<double>{9.0, 2.5, 9.0, 9.0, -1.0});
    c.zero(x);
    CHECK(x == std::vector<double>{9.0, 0.0, 9.0, 9.0, 0.0});
    std::vector<double> mask(5);
    c.free_mask(mask);
    CHECK(mask == std::vector<double>{1.0, 0.0, 1.0, 1.0, 0.0});
}

TEST_CASE("DirichletConstraints: duplicates keep the last value, empty set is a no-op",
          "[linalg][constraints]") {
    const la::DirichletConstraints dup({2, 2}, {1.0, 3.0});
    std::vector<double> x(3, 0.0);
    dup.impose(x);
    CHECK(x[2] == 3.0);

    const la::DirichletConstraints none;
    CHECK(none.empty());
    std::vector<double> y = {1.0, 2.0};
    none.impose(y);
    none.zero(y);
    CHECK(y == std::vector<double>{1.0, 2.0});
    CHECK_NOTHROW(none.validate(0));
}

TEST_CASE("DirichletConstraints reject invalid input", "[linalg][constraints]") {
    CHECK_THROWS_AS(la::DirichletConstraints({0, 1}, {0.0}), std::invalid_argument);
    CHECK_THROWS_AS(la::DirichletConstraints({-1}, {0.0}), std::invalid_argument);
    const la::DirichletConstraints c({0, 5}, {0.0, 0.0});
    CHECK_NOTHROW(c.validate(6));
    CHECK_THROWS_AS(c.validate(5), std::out_of_range);
}

// ---------------------------------------------------------------------------
// ConstrainedOperator
// ---------------------------------------------------------------------------

TEST_CASE("ConstrainedOperator zeroes constrained rows and unit-diagonalises them",
          "[linalg][operator][constraints]") {
    const PoissonSystem s = grid_system();
    const int n = s.mesh.npoin;
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    const std::vector<int> fixed = {0, 7, 13};
    const la::DirichletConstraints c(fixed, {1.0, 2.0, 3.0});
    const la::ConstrainedOperator cop(op, c);

    CHECK(cop.size() == op.size());
    CHECK(cop.traits().symmetric == op.traits().symmetric);
    CHECK(&cop.base() == &op);
    CHECK(&cop.constraints() == &c);
    CHECK(cop.name() == "constrained");

    const std::vector<double> x = test_vector(n);
    std::vector<double> y(n), ax(n);
    cop.apply(x, y);
    op.apply(x, ax);
    for (int i = 0; i < n; ++i) {
        CAPTURE(i);
        if (i == 0 || i == 7 || i == 13) {
            CHECK(y[i] == 0.0);
        } else {
            CHECK(y[i] == ax[i]);
        }
    }

    std::vector<double> d(n);
    cop.diagonal(d);
    for (int i = 0; i < n; ++i) {
        CAPTURE(i);
        CHECK(d[i] == ((i == 0 || i == 7 || i == 13) ? 1.0 : s.pdiag[i]));
    }
}

TEST_CASE("ConstrainedOperator is symmetric on the constrained subspace", "[linalg][operator][constraints]") {
    // For x, y vanishing at the constrained dofs: <P A x, y> == <x, P A y>,
    // and it equals the reduced matrix A_ff.
    const PoissonSystem s = grid_system();
    const int n = s.mesh.npoin;
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    const std::vector<int> fixed = boundary_node_list(s.mesh);
    const la::DirichletConstraints c(fixed, std::vector<double>(fixed.size(), 0.0));
    const la::ConstrainedOperator cop(op, c);

    std::vector<double> x = test_vector(n, 0.2), y = test_vector(n, 1.7);
    c.zero(x);
    c.zero(y);
    std::vector<double> cx(n), cy(n);
    cop.apply(x, cx);
    cop.apply(y, cy);
    CHECK_THAT(la::dot(cx, y), WithinRel(la::dot(x, cy), 1e-12));

    // SPD on the subspace: the reduced dense matrix admits a Cholesky factor.
    const DenseMatrix a = dense_operator(s.mesh, s.gstif, s.pdiag);
    CHECK(cholesky_succeeds(reduced(a, std::set<int>(fixed.begin(), fixed.end()))));
}

TEST_CASE("ConstrainedOperator without constraints equals the base operator", "[linalg][operator][constraints]") {
    const PoissonSystem s = grid_system();
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    const la::DirichletConstraints none;
    const la::ConstrainedOperator cop(op, none);
    const std::vector<double> x = test_vector(s.mesh.npoin);
    std::vector<double> y(x.size()), ax(x.size());
    cop.apply(x, y);
    op.apply(x, ax);
    CHECK(y == ax);
}

TEST_CASE("ConstrainedOperator rejects out-of-range constraints", "[linalg][operator][constraints]") {
    const DenseOperator a(laplacian_1d(4));
    const la::DirichletConstraints c({4}, {0.0});
    CHECK_THROWS_AS(la::ConstrainedOperator(a, c), std::out_of_range);
}

// ---------------------------------------------------------------------------
// Preconditioners
// ---------------------------------------------------------------------------

TEST_CASE("IdentityPreconditioner copies the residual", "[linalg][preconditioner]") {
    const DenseOperator a(laplacian_1d(5));
    la::IdentityPreconditioner m;
    CHECK(m.size() == 0);
    m.setup(a);
    CHECK(m.size() == 5);
    CHECK(m.name() == "none");
    CHECK(m.pointwise_scaling().empty());
    CHECK(m.traits().symmetric);
    CHECK_FALSE(m.traits().variable);

    const std::vector<double> r = {1.0, -2.0, 3.0, 0.0, 4.5};
    std::vector<double> z(5);
    m.apply(r, z);
    CHECK(z == r);
}

TEST_CASE("JacobiPreconditioner inverts the diagonal", "[linalg][preconditioner]") {
    const DenseOperator a(diagonal_matrix({2.0, -4.0, 0.5, 1e-31, 0.0, 8.0}));
    la::JacobiPreconditioner m;
    m.setup(a);
    REQUIRE(m.size() == 6);
    CHECK(m.name() == "jacobi");
    CHECK(m.traits().symmetric);
    CHECK_FALSE(m.traits().variable);

    // |d| <= 1e-30 is not inverted (legacy guard): scaling 1.
    const std::vector<double> expected = {0.5, -0.25, 2.0, 1.0, 1.0, 0.125};
    CHECK(m.inverse_diagonal() == expected);

    const la::ConstVectorView s = m.pointwise_scaling();
    REQUIRE(s.size() == 6);
    CHECK(s.data() == m.inverse_diagonal().data());

    const std::vector<double> r = {1.0, 1.0, 1.0, 3.0, 3.0, 16.0};
    std::vector<double> z(6);
    m.apply(r, z);
    CHECK(z == std::vector<double>{0.5, -0.25, 2.0, 3.0, 3.0, 2.0});
}

TEST_CASE("JacobiPreconditioner on the pressure operator uses the assembled diagonal",
          "[linalg][preconditioner]") {
    const PoissonSystem s = grid_system();
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    const la::DirichletConstraints c({3}, {0.0});
    la::JacobiPreconditioner m;
    m.setup(la::ConstrainedOperator(op, c));
    for (int i = 0; i < s.mesh.npoin; ++i) {
        CAPTURE(i);
        CHECK(m.inverse_diagonal()[i] == (i == 3 ? 1.0 : 1.0 / s.pdiag[i]));
    }
}

TEST_CASE("Preconditioners must be set up before use", "[linalg][preconditioner]") {
    std::vector<double> r(3, 1.0), z(3);
    la::IdentityPreconditioner identity;
    la::JacobiPreconditioner jacobi;
    CHECK_THROWS_AS(identity.apply(r, z), std::invalid_argument);
    CHECK_THROWS_AS(jacobi.apply(r, z), std::invalid_argument);

    const DenseOperator a(laplacian_1d(4));
    jacobi.setup(a);
    CHECK_THROWS_AS(jacobi.apply(r, z), std::invalid_argument); // wrong size
}

TEST_CASE("Preconditioner::update defaults to a full setup", "[linalg][preconditioner]") {
    DenseOperator a(laplacian_1d(4));
    DensePreconditioner m(inverse(laplacian_1d(4)));
    m.setup(a);
    REQUIRE(m.setups == 1);
    m.update(a, la::OperatorChange::Values);
    m.update(a, la::OperatorChange::Structure);
    CHECK(m.setups == 3);

    SECTION("Jacobi picks up new operator values on update") {
        la::JacobiPreconditioner jacobi;
        jacobi.setup(DenseOperator(laplacian_1d(4, 1.0)));
        CHECK(jacobi.inverse_diagonal()[0] == 0.5);
        jacobi.update(DenseOperator(laplacian_1d(4, 4.0)), la::OperatorChange::Values);
        CHECK(jacobi.inverse_diagonal()[0] == 0.125);
    }
}
