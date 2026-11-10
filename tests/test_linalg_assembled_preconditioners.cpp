// Assembled preconditioners, each verified against its defining property:
//   SSOR   M^{-1} equals the inverse of the closed form
//          (omega (2 - omega))^{-1} (D + omega L) D^{-1} (D + omega U)
//   IC(0)  (U^T U)_ij = a_ij on the pattern; exact without fill-in;
//          breakdown on Kershaw's matrix cured by the diagonal shift
//   ILU(0) (L U)_ij = a_ij on the pattern (non-symmetric); exact without
//          fill-in; zero pivots cured by the shift; equals IC(0) for SPD A
//   ReorderedPreconditioner
//          M^{-1} = P^T M_inner(P A P^T)^{-1} P, ordering computed once
// plus symmetry/definiteness, structure reuse on update(), error handling,
// and CG with every preconditioner on grids and the benchmark mesh.

#include "BoundaryConditions.hpp"
#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "LinalgTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <map>
#include <set>
#include <string>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Strictly lower / diagonal / strictly upper parts of a dense matrix.
DenseMatrix part(const DenseMatrix& a, int which) {
    DenseMatrix p = zeros(static_cast<int>(a.size()));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a.size(); ++j)
            if ((which < 0 && j < i) || (which == 0 && j == i) || (which > 0 && j > i)) p[i][j] = a[i][j];
    return p;
}

DenseMatrix add(const DenseMatrix& a, const DenseMatrix& b, double s = 1.0) {
    DenseMatrix c = a;
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < a.size(); ++j) c[i][j] += s * b[i][j];
    return c;
}

DenseMatrix ssor_matrix(const DenseMatrix& a, double omega) {
    const DenseMatrix l = part(a, -1), d = part(a, 0), u = part(a, 1);
    DenseMatrix dinv = zeros(static_cast<int>(a.size()));
    for (size_t i = 0; i < a.size(); ++i) dinv[i][i] = 1.0 / d[i][i];
    DenseMatrix m = matmul(matmul(add(d, l, omega), dinv), add(d, u, omega));
    for (auto& row : m)
        for (double& v : row) v /= omega * (2.0 - omega);
    return m;
}

bool symmetric(const DenseMatrix& a, double tol) {
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (std::abs(a[i][j] - a[j][i]) > tol) return false;
    return true;
}

// SPD test matrix: constrained P1 Laplacian of a small grid (not tridiagonal).
la::CsrMatrix grid_spd(int nx = 5, int ny = 4, bool varying = true) {
    GridSpec g;
    g.lx = 1.0;
    g.ly = 0.8;
    g.nx = nx;
    g.ny = ny;
    PoissonSystem s = make_poisson_system(make_grid_mesh(g), varying ? varying_dt(g.nelem()) : std::vector<double>{});
    const la::P1CsrAssembler assembler(s.mesh);
    la::CsrMatrix a = assembler.create_matrix();
    assembler.assemble_ebe(s.mesh, s.gstif, s.pdiag, a);
    la::CsrMatrix ac;
    la::constrain(a, la::DirichletConstraints(boundary_node_list(s.mesh), std::vector<double>(
                                                  boundary_node_list(s.mesh).size(), 0.0)),
                  ac);
    return ac;
}

// Entries of M (= inverse of M^{-1}) on the pattern of A vs. A.
double max_pattern_mismatch(const la::CsrMatrix& a, const la::Preconditioner& m) {
    const DenseMatrix mm = inverse(inverse_preconditioner_matrix(m));
    const DenseMatrix da = to_dense(a);
    double worst = 0.0, scale = 0.0;
    const auto& ptr = a.pattern().row_ptr();
    const auto& col = a.pattern().col_idx();
    for (int i = 0; i < a.rows(); ++i) {
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
            worst = std::max(worst, std::abs(mm[i][col[k]] - da[i][col[k]]));
            scale = std::max(scale, std::abs(da[i][col[k]]));
        }
    }
    return worst / scale;
}

} // namespace

// ---------------------------------------------------------------------------
// SSOR
// ---------------------------------------------------------------------------

TEST_CASE("SSOR applies the inverse of its closed-form matrix", "[linalg][preconditioner][ssor]") {
    const double omega = GENERATE(1.0, 0.7, 1.4);
    const bool nonsymmetric = GENERATE(false, true);
    CAPTURE(omega, nonsymmetric);
    const DenseMatrix d = nonsymmetric ? convection_diffusion(4, 3) : to_dense(grid_spd());
    const la::CsrMatrix a = from_dense(d, !nonsymmetric);

    la::SsorPreconditioner m(omega);
    m.setup(a);
    CHECK(m.omega() == omega);
    CHECK(m.name() == "ssor");
    CHECK(m.size() == d.size());
    CHECK(m.traits().symmetric == !nonsymmetric);
    const DenseMatrix product = matmul(inverse_preconditioner_matrix(m), ssor_matrix(d, omega));
    CHECK(is_identity(product, 1e-12));
}

TEST_CASE("SSOR of an SPD matrix is symmetric positive definite", "[linalg][preconditioner][ssor]") {
    const la::CsrMatrix a = grid_spd();
    const double omega = GENERATE(0.5, 1.0, 1.8);
    la::SsorPreconditioner m(omega);
    m.setup(a);
    const DenseMatrix minv = inverse_preconditioner_matrix(m);
    CHECK(symmetric(minv, 1e-12));
    CHECK(cholesky_succeeds(minv));
}

TEST_CASE("SSOR validates omega and the matrix", "[linalg][preconditioner][ssor]") {
    for (double bad : {0.0, 2.0, -0.5, 2.5, std::numeric_limits<double>::quiet_NaN()}) {
        CAPTURE(bad);
        CHECK_THROWS_AS(la::SsorPreconditioner(bad), std::invalid_argument);
    }
    la::SsorPreconditioner m;
    const DenseOperator dense(laplacian_1d(3));
    CHECK_THROWS_AS(m.setup(dense), std::invalid_argument); // not assembled
    CHECK_THROWS_AS(m.setup(la::CsrMatrix::from_triplets(2, 2, {{0, 0, 1.0}, {1, 0, 1.0}})),
                    std::invalid_argument); // missing diagonal
    CHECK_THROWS_AS(m.setup(la::CsrMatrix::from_triplets(2, 2, {{0, 0, 1.0}, {1, 1, 0.0}})),
                    std::invalid_argument); // zero diagonal
    std::vector<double> r(3, 1.0), z(3);
    CHECK_THROWS_AS(m.apply(r, z), std::invalid_argument); // not set up
}

// ---------------------------------------------------------------------------
// IC(0)
// ---------------------------------------------------------------------------

TEST_CASE("IC(0) is exact when the Cholesky factor has no fill-in", "[linalg][preconditioner][ic0]") {
    SECTION("tridiagonal matrix") {
        const la::CsrMatrix a = from_dense(laplacian_1d(12, 3.0), true);
        la::IncompleteCholesky m;
        m.setup(a);
        CHECK(m.shift() == 0.0);
        CHECK(m.attempts() == 1);
        CHECK(is_identity(matmul(inverse_preconditioner_matrix(m), to_dense(a)), 1e-11));
    }
    SECTION("full pattern: IC(0) = complete Cholesky") {
        // B^T B + I, dense and SPD.
        DenseMatrix b = zeros(6);
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j) b[i][j] = std::sin(1.0 + 2.0 * i + 0.7 * j);
        DenseMatrix spd = zeros(6);
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j) {
                for (int k = 0; k < 6; ++k) spd[i][j] += b[k][i] * b[k][j];
                if (i == j) spd[i][j] += 1.0;
            }
        la::IncompleteCholesky m;
        m.setup(from_dense(spd, true));
        CHECK(is_identity(matmul(inverse_preconditioner_matrix(m), spd), 1e-10));
    }
}

TEST_CASE("IC(0) reproduces A on its sparsity pattern", "[linalg][preconditioner][ic0]") {
    // The defining property of a zero fill-in factorisation.
    const la::CsrMatrix a = grid_spd();
    la::IncompleteCholesky m;
    m.setup(a);
    CHECK(m.shift() == 0.0);
    CHECK(max_pattern_mismatch(a, m) < 1e-12);
    // ... but it is not exact: fill-in was dropped.
    CHECK_FALSE(is_identity(matmul(inverse_preconditioner_matrix(m), to_dense(a)), 1e-6));
    CHECK(m.factor_nnz() == (a.nnz() + a.rows()) / 2); // triu(A)
}

TEST_CASE("IC(0) of an SPD matrix is symmetric positive definite", "[linalg][preconditioner][ic0]") {
    la::IncompleteCholesky m;
    m.setup(grid_spd());
    const DenseMatrix minv = inverse_preconditioner_matrix(m);
    CHECK(symmetric(minv, 1e-12));
    CHECK(cholesky_succeeds(minv));
    CHECK(m.traits().symmetric);
    CHECK(m.name() == "ic0");
}

TEST_CASE("IC(0) reads only the upper triangle", "[linalg][preconditioner][ic0]") {
    const la::CsrMatrix a = grid_spd();
    la::CsrMatrix perturbed = a;
    for (int i = 0; i < a.rows(); ++i)
        for (int k = a.pattern().row_begin(i); k < a.pattern().row_end(i); ++k)
            if (a.pattern().col_idx()[k] < i) perturbed.values()[k] = 99.0;
    la::IncompleteCholesky m1, m2;
    m1.setup(a);
    m2.setup(perturbed);
    CHECK(inverse_preconditioner_matrix(m1) == inverse_preconditioner_matrix(m2));
}

TEST_CASE("IC(0) breakdown on Kershaw's matrix is cured by a diagonal shift",
          "[linalg][preconditioner][ic0]") {
    const DenseMatrix k = kershaw_matrix();
    REQUIRE(cholesky_succeeds(k)); // SPD
    const la::CsrMatrix a = from_dense(k, true);

    la::IncompleteCholesky m;
    m.setup(a);
    // alpha = 0 fails (pivot -5), then 1e-3 * 2^j; 0.256 is the first that works.
    CHECK(m.attempts() == 10);
    CHECK_THAT(m.shift(), WithinRel(0.256, 1e-12));
    const DenseMatrix minv = inverse_preconditioner_matrix(m);
    CHECK(symmetric(minv, 1e-12));
    CHECK(cholesky_succeeds(minv));

    SECTION("an initial shift that is large enough needs a single attempt") {
        la::DiagonalShift s;
        s.initial = 0.3;
        la::IncompleteCholesky shifted(s);
        shifted.setup(a);
        CHECK(shifted.attempts() == 1);
        CHECK(shifted.shift() == 0.3);
    }
    SECTION("giving up after max_attempts") {
        la::DiagonalShift s;
        s.max_attempts = 3;
        la::IncompleteCholesky limited(s);
        CHECK_THROWS_AS(limited.setup(a), std::runtime_error);
    }
}

TEST_CASE("IC(0) rejects unsuitable input", "[linalg][preconditioner][ic0]") {
    la::IncompleteCholesky m;
    CHECK_THROWS_AS(m.setup(DenseOperator(laplacian_1d(3))), std::invalid_argument);
    CHECK_THROWS_AS(m.setup(from_dense(diagonal_matrix({1.0, -2.0}))), std::invalid_argument);
    CHECK_THROWS_AS(m.setup(la::CsrMatrix::from_triplets(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}})),
                    std::invalid_argument); // missing diagonal in row 1
    la::DiagonalShift bad;
    bad.initial = -1.0;
    CHECK_THROWS_AS(la::IncompleteCholesky(bad), std::invalid_argument);
}

TEST_CASE("IC(0) update with new values reuses the structure", "[linalg][preconditioner][ic0]") {
    const la::CsrMatrix a = grid_spd();
    la::IncompleteCholesky m;
    m.setup(a);
    const DenseMatrix minv = inverse_preconditioner_matrix(m);

    // A scaled by 4 (same pattern object): M^{-1} scales by 1/4.
    la::CsrMatrix a4 = a;
    for (double& v : a4.values()) v *= 4.0;
    m.update(a4, la::OperatorChange::Values);
    const DenseMatrix minv4 = inverse_preconditioner_matrix(m);
    for (size_t i = 0; i < minv.size(); ++i)
        for (size_t j = 0; j < minv.size(); ++j) CHECK_THAT(minv4[i][j], WithinAbs(0.25 * minv[i][j], 1e-13));

    SECTION("a different pattern triggers a full setup") {
        const la::CsrMatrix other = from_dense(laplacian_1d(7), true);
        m.update(other, la::OperatorChange::Values);
        CHECK(m.size() == 7);
        CHECK(is_identity(matmul(inverse_preconditioner_matrix(m), to_dense(other)), 1e-11));
    }
}

// ---------------------------------------------------------------------------
// ILU(0)
// ---------------------------------------------------------------------------

TEST_CASE("ILU(0) is exact for tridiagonal non-symmetric matrices", "[linalg][preconditioner][ilu0]") {
    DenseMatrix t = zeros(10);
    for (int i = 0; i < 10; ++i) {
        t[i][i] = 4.0 + 0.1 * i;
        if (i > 0) t[i][i - 1] = -1.5;
        if (i < 9) t[i][i + 1] = -0.5;
    }
    la::IncompleteLU m;
    m.setup(from_dense(t));
    CHECK(m.shift() == 0.0);
    CHECK(is_identity(matmul(inverse_preconditioner_matrix(m), t), 1e-12));
}

TEST_CASE("ILU(0) reproduces A on its sparsity pattern", "[linalg][preconditioner][ilu0]") {
    const DenseMatrix d = convection_diffusion(5, 4);
    const la::CsrMatrix a = from_dense(d);
    la::IncompleteLU m;
    m.setup(a);
    CHECK(max_pattern_mismatch(a, m) < 1e-12);
    CHECK_FALSE(is_identity(matmul(inverse_preconditioner_matrix(m), d), 1e-6));
    CHECK_FALSE(m.traits().symmetric);
    CHECK(m.name() == "ilu0");
}

TEST_CASE("ILU(0) equals IC(0) for a symmetric positive definite matrix", "[linalg][preconditioner][ilu0]") {
    // U = D L^T in exact arithmetic, hence L U = (D^{1/2} L^T)^T (D^{1/2} L^T).
    const la::CsrMatrix a = grid_spd();
    la::IncompleteLU ilu;
    la::IncompleteCholesky ic;
    ilu.setup(a);
    ic.setup(a);
    CHECK(max_abs_diff(inverse_preconditioner_matrix(ilu), inverse_preconditioner_matrix(ic)) < 1e-12);
}

TEST_CASE("ILU(0) zero pivot is cured by a diagonal shift", "[linalg][preconditioner][ilu0]") {
    // u_11 = a_11 - l_10 u_01 = 1 - 1 = 0 without a shift.
    const la::CsrMatrix a = from_dense({{1, 1, 0}, {1, 1, 1}, {0, 1, 1}});
    la::IncompleteLU m;
    m.setup(a);
    CHECK(m.attempts() > 1);
    CHECK(m.shift() > 0.0);
    const DenseMatrix minv = inverse_preconditioner_matrix(m);
    for (const auto& row : minv)
        for (double v : row) CHECK(std::isfinite(v));
}

TEST_CASE("ILU(0) rejects unsuitable input", "[linalg][preconditioner][ilu0]") {
    la::IncompleteLU m;
    CHECK_THROWS_AS(m.setup(DenseOperator(laplacian_1d(3))), std::invalid_argument);
    CHECK_THROWS_AS(m.setup(la::CsrMatrix::from_triplets(2, 2, {{0, 0, 0.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}})),
                    std::invalid_argument); // zero diagonal
    CHECK_THROWS_AS(m.setup(la::CsrMatrix::from_triplets(2, 3, {{0, 0, 1.0}})), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// ReorderedPreconditioner
// ---------------------------------------------------------------------------

TEST_CASE("Reordered preconditioner applies P^T M(P A P^T)^{-1} P", "[linalg][preconditioner][reordering]") {
    const la::CsrMatrix a = grid_spd(6, 5);
    const int n = a.rows();
    const la::Permutation p(shuffled_indices(n, 17));
    const std::string inner_name = GENERATE(std::string("ic0"), std::string("ssor"), std::string("ilu0"));
    CAPTURE(inner_name);

    la::ReorderedPreconditioner m(la::make_preconditioner(inner_name), [&p](const la::SparsityPattern&) { return p; },
                                  "custom");
    m.setup(a);
    CHECK(m.name() == "custom+" + inner_name);
    CHECK(m.size() == static_cast<size_t>(n));
    CHECK(m.permutation().new_to_old() == p.new_to_old());

    // Reference: inner preconditioner on the permuted matrix, conjugated.
    auto inner = la::make_preconditioner(inner_name);
    const la::CsrMatrix b = la::SymmetricPermutation(a.pattern_ptr(), p).apply(a);
    inner->setup(b);
    const DenseMatrix inner_minv = inverse_preconditioner_matrix(*inner);
    const DenseMatrix minv = inverse_preconditioner_matrix(m);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            CHECK(minv[i][j] == inner_minv[p.old_to_new()[i]][p.old_to_new()[j]]);
}

TEST_CASE("Reordering with the identity changes nothing", "[linalg][preconditioner][reordering]") {
    const la::CsrMatrix a = grid_spd();
    la::ReorderedPreconditioner m(std::make_unique<la::IncompleteCholesky>(),
                                  [](const la::SparsityPattern& s) { return la::Permutation::identity(s.rows()); },
                                  "none");
    la::IncompleteCholesky plain;
    m.setup(a);
    plain.setup(a);
    CHECK(inverse_preconditioner_matrix(m) == inverse_preconditioner_matrix(plain));
    CHECK(m.traits().symmetric == plain.traits().symmetric);
}

TEST_CASE("Reordered preconditioner computes the ordering once per pattern", "[linalg][preconditioner][reordering]") {
    const la::CsrMatrix a = grid_spd();
    int orderings = 0;
    la::ReorderedPreconditioner m(
        std::make_unique<la::IncompleteCholesky>(),
        [&orderings](const la::SparsityPattern& s) {
            ++orderings;
            return la::reverse_cuthill_mckee(s);
        },
        "rcm");
    m.setup(a);
    CHECK(orderings == 1);

    la::CsrMatrix a2 = a; // same pattern object, new values
    for (double& v : a2.values()) v *= 3.0;
    m.update(a2, la::OperatorChange::Values);
    CHECK(orderings == 1);
    la::IncompleteCholesky fresh;
    fresh.setup(la::SymmetricPermutation(a2.pattern_ptr(), m.permutation()).apply(a2));
    const DenseMatrix expected = inverse_preconditioner_matrix(fresh);
    const DenseMatrix minv = inverse_preconditioner_matrix(m);
    const auto& o2n = m.permutation().old_to_new();
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.rows(); ++j) CHECK(minv[i][j] == expected[o2n[i]][o2n[j]]);

    m.update(a2, la::OperatorChange::Structure);
    CHECK(orderings == 2);
}

TEST_CASE("Reordered preconditioner input checks", "[linalg][preconditioner][reordering]") {
    CHECK_THROWS_AS(la::ReorderedPreconditioner(nullptr, la::reverse_cuthill_mckee, "rcm"), std::invalid_argument);
    CHECK_THROWS_AS(la::ReorderedPreconditioner(std::make_unique<la::IncompleteCholesky>(), nullptr, "x"),
                    std::invalid_argument);
    auto m = la::ReorderedPreconditioner::rcm(std::make_unique<la::IncompleteCholesky>());
    CHECK(m->name() == "rcm+ic0");
    CHECK(m->size() == 0);
    std::vector<double> r(3), z(3);
    CHECK_THROWS_AS(m->apply(r, z), std::invalid_argument);
    CHECK_THROWS_AS(m->permutation(), std::logic_error);
    CHECK_THROWS_AS(m->setup(DenseOperator(laplacian_1d(3))), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// CG with every preconditioner
// ---------------------------------------------------------------------------

TEST_CASE("CG with assembled preconditioners agrees with a dense solve", "[linalg][preconditioner][cg]") {
    // Pressure-type system: EBE operator for the Krylov iteration, the
    // preconditioner built from the constrained CSR matrix.
    GridSpec g;
    g.lx = 1.5;
    g.ly = 1.0;
    g.nx = 9;
    g.ny = 6;
    const PoissonSystem s = make_poisson_system(make_grid_mesh(g), varying_dt(g.nelem()));
    const int n = s.mesh.npoin;
    const int ref = 11;
    const la::DirichletConstraints bc({ref}, {0.25});
    const std::vector<double> rhs = test_vector(n, 0.9);

    const DenseMatrix dense = dense_operator(s.mesh, s.gstif, s.pdiag);
    std::vector<double> b;
    for (int i = 0; i < n; ++i)
        if (i != ref) b.push_back(rhs[i] - dense[i][ref] * 0.25);
    const std::vector<double> xf = solve_dense(reduced(dense, {ref}), b);

    const la::P1CsrAssembler assembler(s.mesh);
    la::CsrMatrix a = assembler.create_matrix(), ac;
    assembler.assemble_ebe(s.mesh, s.gstif, s.pdiag, a);
    la::constrain(a, bc, ac);

    const std::string name = GENERATE(std::string("ssor"), std::string("ic0"), std::string("ilu0"));
    const std::string ordering = GENERATE(std::string("none"), std::string("rcm"));
    const bool csr_operator = GENERATE(false, true);
    CAPTURE(name, ordering, csr_operator);
    la::PreconditionerOptions o;
    o.ordering = ordering;
    auto m = la::make_preconditioner(name, o);
    m->setup(ac);

    const la::EbeOperator ebe(s.mesh, s.gstif, s.pdiag);
    const la::LinearOperator& op = csr_operator ? static_cast<const la::LinearOperator&>(a) : ebe;
    la::ConvergenceCriteria c;
    c.rel_tol = 1e-12;
    la::ConjugateGradient cg(c);
    std::vector<double> x(n, 0.0);
    const la::SolveResult r = cg.solve(op, *m, rhs, x, bc);
    REQUIRE(r.converged);
    CHECK(x[ref] == 0.25);
    for (int i = 0, k = 0; i < n; ++i) {
        if (i == ref) continue;
        CHECK_THAT(x[i], WithinAbs(xf[k++], 1e-9));
    }
}

TEST_CASE("Preconditioner quality on the benchmark mesh", "[linalg][preconditioner][hinterkante]") {
    // Iterations of CG to a relative residual of 1e-8 on the pressure matrix.
    // Expected ordering of quality: jacobi > ssor > ic0 = ilu0, and RCM
    // improves the factorisations and SSOR.
    static const auto counts = [] {
        Mesh mesh;
        PressureBC pbc;
        {
            CaptureStdout quiet;
            const std::string dir = HYDRAMA2D_CASE_DIR;
            REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", mesh));
            REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", mesh));
            compute_all_geometry(mesh);
            pbc = setup_pressure_bcs(mesh, 0);
        }
        PoissonSystem s = make_poisson_system(std::move(mesh), std::vector<double>(17377, 4.9e-3));
        const la::P1CsrAssembler assembler(s.mesh);
        la::CsrMatrix a = assembler.create_matrix(), ac;
        assembler.assemble_ebe(s.mesh, s.gstif, s.pdiag, a);
        const la::DirichletConstraints bc(pbc.nodes, pbc.values);
        la::constrain(a, bc, ac);
        const la::EbeOperator ebe(s.mesh, s.gstif, s.pdiag);
        std::vector<double> rhs = test_vector(s.mesh.npoin, 0.7);

        std::map<std::string, int> it;
        for (std::string name : {"jacobi", "ssor", "ic0", "ilu0"}) {
            for (std::string ord : {"none", "rcm"}) {
                if (name == "jacobi" && ord == "rcm") continue;
                la::PreconditionerOptions o;
                o.ordering = ord;
                auto m = la::make_preconditioner(name, o);
                m->setup(ac);
                la::ConvergenceCriteria c;
                c.rel_tol = 1e-8;
                la::ConjugateGradient cg(c);
                std::vector<double> x(s.mesh.npoin, 0.0);
                const la::SolveResult r = cg.solve(ebe, *m, rhs, x, bc);
                REQUIRE(r.converged);
                it[std::string(m->name())] = r.iterations;
            }
        }
        return it;
    }();
    for (const auto& [name, n] : counts) {
        CAPTURE(name, n);
        CHECK(n > 0);
    }
    CHECK(counts.at("ssor") < counts.at("jacobi"));
    CHECK(counts.at("ic0") < counts.at("ssor"));
    CHECK(std::abs(counts.at("ilu0") - counts.at("ic0")) <= 1);
    CHECK(counts.at("rcm+ic0") < counts.at("ic0"));
    CHECK(counts.at("rcm+ssor") < counts.at("ssor"));
    CHECK(2 * counts.at("rcm+ic0") < counts.at("jacobi"));
}
