// Compressed sparse row storage: SparsityPattern (validation, queries,
// bandwidth/profile) and CsrMatrix (construction from triplets, matrix-vector
// product against dense references, diagonal, symmetry, constrained copy,
// capability query).

#include "LinalgTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;

namespace {

// 4 x 4 example:   [ 4 -1  .  2 ]
//                  [-1  5 -2  . ]
//                  [ .  .  6  . ]
//                  [ 2  . -3  7 ]
la::CsrMatrix example() {
    return la::CsrMatrix::from_triplets(4, 4, {{0, 0, 4}, {0, 1, -1}, {0, 3, 2}, {1, 0, -1}, {1, 1, 5}, {1, 2, -2},
                                               {2, 2, 6}, {3, 0, 2}, {3, 2, -3}, {3, 3, 7}});
}

} // namespace

// ---------------------------------------------------------------------------
// SparsityPattern
// ---------------------------------------------------------------------------

TEST_CASE("SparsityPattern stores and queries the structure", "[linalg][csr]") {
    const la::SparsityPattern p(3, 4, {0, 2, 2, 5}, {0, 3, 1, 2, 3});
    CHECK(p.rows() == 3);
    CHECK(p.cols() == 4);
    CHECK_FALSE(p.square());
    CHECK(p.nnz() == 5);
    CHECK(p.row_begin(2) == 2);
    CHECK(p.row_end(2) == 5);
    CHECK(p.find(0, 3) == 1);
    CHECK(p.find(2, 2) == 3);
    CHECK(p.find(1, 0) == -1); // empty row
    CHECK(p.find(0, 2) == -1);
    CHECK(p.find(5, 0) == -1); // row out of range
    CHECK(p.diagonal_positions() == std::vector<int>{0, -1, 3});
    CHECK_FALSE(p.has_full_diagonal());
    CHECK_FALSE(p.structurally_symmetric());
}

TEST_CASE("SparsityPattern rejects malformed structures", "[linalg][csr]") {
    CHECK_THROWS_AS(la::SparsityPattern(2, 2, {0, 1}, {0}), std::invalid_argument);          // short row_ptr
    CHECK_THROWS_AS(la::SparsityPattern(2, 2, {1, 1, 2}, {0, 1}), std::invalid_argument);    // row_ptr[0] != 0
    CHECK_THROWS_AS(la::SparsityPattern(2, 2, {0, 1, 3}, {0, 1}), std::invalid_argument);    // end != nnz
    CHECK_THROWS_AS(la::SparsityPattern(2, 2, {0, 2, 1}, {0, 1}), std::invalid_argument);    // decreasing
    CHECK_THROWS_AS(la::SparsityPattern(2, 2, {0, 1, 2}, {0, 2}), std::invalid_argument);    // column range
    CHECK_THROWS_AS(la::SparsityPattern(1, 3, {0, 2}, {1, 0}), std::invalid_argument);       // unsorted
    CHECK_THROWS_AS(la::SparsityPattern(1, 3, {0, 2}, {1, 1}), std::invalid_argument);       // duplicate
    CHECK_THROWS_AS(la::SparsityPattern(-1, 2, {0}, {}), std::invalid_argument);
    CHECK_NOTHROW(la::SparsityPattern(0, 0, {0}, {}));
}

TEST_CASE("SparsityPattern symmetry, bandwidth, profile and equality", "[linalg][csr]") {
    const la::CsrMatrix a = example();
    const la::SparsityPattern& p = a.pattern();
    CHECK(p.has_full_diagonal());
    CHECK_FALSE(p.structurally_symmetric()); // (1,2) without (2,1)
    CHECK(p.bandwidth() == 3);               // (0,3), (3,0)
    // Lower envelope: row 1 from col 0 (1), row 2 diagonal only (0), row 3 from col 0 (3).
    CHECK(p.profile() == 4);

    const la::CsrMatrix t = from_dense(laplacian_1d(6), true);
    CHECK(t.pattern().structurally_symmetric());
    CHECK(t.pattern().bandwidth() == 1);
    CHECK(t.pattern().profile() == 5);

    const la::SparsityPattern same(4, 4, p.row_ptr(), p.col_idx());
    CHECK(same == p);
    CHECK(t.pattern() != p);
}

// ---------------------------------------------------------------------------
// CsrMatrix
// ---------------------------------------------------------------------------

TEST_CASE("CsrMatrix from triplets sums duplicates and keeps explicit zeros", "[linalg][csr]") {
    const la::CsrMatrix a = la::CsrMatrix::from_triplets(
        2, 3, {{1, 2, 1.0}, {0, 1, 2.0}, {1, 2, 0.5}, {0, 0, 0.0}, {1, 0, -1.0}, {0, 1, 3.0}});
    CHECK(a.rows() == 2);
    CHECK(a.cols() == 3);
    CHECK(a.nnz() == 4); // (0,0) (0,1) (1,0) (1,2)
    CHECK(a.pattern().find(0, 0) >= 0);
    CHECK(a.coeff(0, 0) == 0.0);
    CHECK(a.coeff(0, 1) == 5.0);
    CHECK(a.coeff(1, 2) == 1.5);
    CHECK(a.coeff(1, 0) == -1.0);
    CHECK(a.coeff(1, 1) == 0.0); // not stored

    CHECK_THROWS_AS(la::CsrMatrix::from_triplets(2, 2, {{2, 0, 1.0}}), std::invalid_argument);
    CHECK_THROWS_AS(la::CsrMatrix::from_triplets(2, 2, {{0, -1, 1.0}}), std::invalid_argument);
}

TEST_CASE("CsrMatrix product equals the dense product", "[linalg][csr]") {
    const int which = GENERATE(0, 1, 2);
    CAPTURE(which);
    const DenseMatrix d = which == 0 ? to_dense(example())
                        : which == 1 ? convection_diffusion(5, 4)
                                     : laplacian_1d(17, 2.5);
    const la::CsrMatrix a = from_dense(d);
    REQUIRE(to_dense(a) == d);

    const int n = static_cast<int>(d.size());
    const std::vector<double> x = test_vector(n, 0.4);
    std::vector<double> y(n, 123.0); // overwritten, not accumulated
    a.apply(x, y);
    const std::vector<double> ref = multiply(d, x);
    for (int i = 0; i < n; ++i) CHECK_THAT(y[i], WithinAbs(ref[i], 1e-13));
}

TEST_CASE("CsrMatrix product of rectangular matrices", "[linalg][csr]") {
    const la::CsrMatrix a = la::CsrMatrix::from_triplets(2, 3, {{0, 0, 1.0}, {0, 2, 2.0}, {1, 1, -3.0}});
    std::vector<double> x = {1.0, 2.0, 3.0}, y(2);
    a.apply(x, y);
    CHECK(y == std::vector<double>{7.0, -6.0});
    CHECK(a.size() == 2);
    std::vector<double> wrong(2), y3(3);
    CHECK_THROWS_AS(a.apply(wrong, y), std::invalid_argument);
    CHECK_THROWS_AS(a.apply(x, y3), std::invalid_argument);
}

TEST_CASE("CsrMatrix diagonal, symmetry, values and capability query", "[linalg][csr]") {
    la::CsrMatrix a = example();
    std::vector<double> d(4);
    a.diagonal(d);
    CHECK(d == std::vector<double>{4, 5, 6, 7});
    CHECK(a.name() == "csr");
    CHECK(a.assembled() == &a);
    CHECK(&la::require_assembled(a, "test") == &a);

    CHECK_FALSE(a.is_symmetric());
    CHECK_FALSE(a.traits().symmetric); // nothing declared
    la::CsrMatrix s = from_dense(laplacian_1d(5));
    CHECK(s.is_symmetric());
    s.declare_symmetric(true);
    CHECK(s.traits().symmetric);
    s.values()[1] += 1e-9; // (0,1) no longer equals (1,0)
    CHECK_FALSE(s.is_symmetric());
    CHECK(s.is_symmetric(1e-8));

    la::CsrMatrix z = a;
    z.set_zero();
    CHECK(z.nnz() == a.nnz());
    for (double v : z.values()) CHECK(v == 0.0);
    CHECK(z.pattern_ptr() == a.pattern_ptr()); // copies share the pattern

    SECTION("a missing diagonal entry reads as zero") {
        const la::CsrMatrix off = la::CsrMatrix::from_triplets(2, 2, {{0, 1, 1.0}, {1, 1, 2.0}});
        std::vector<double> dd(2);
        off.diagonal(dd);
        CHECK(dd == std::vector<double>{0.0, 2.0});
    }
}

TEST_CASE("CsrMatrix construction checks", "[linalg][csr]") {
    auto p = std::make_shared<const la::SparsityPattern>(2, 2, std::vector<int>{0, 1, 2}, std::vector<int>{0, 1});
    CHECK_THROWS_AS(la::CsrMatrix(p, {1.0}), std::invalid_argument);
    CHECK_THROWS_AS(la::CsrMatrix(nullptr), std::invalid_argument);
    const la::CsrMatrix empty;
    CHECK(empty.rows() == 0);
    CHECK(empty.nnz() == 0);
    CHECK(empty.coeff(0, 0) == 0.0);
}

TEST_CASE("require_assembled rejects matrix-free operators", "[linalg][csr]") {
    const NoDiagonalOperator free_op(3);
    CHECK(free_op.assembled() == nullptr);
    CHECK_THROWS_AS(la::require_assembled(free_op, "ic0"), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Constrained matrix
// ---------------------------------------------------------------------------

TEST_CASE("constrain() builds P A P + (I - P) on the same pattern", "[linalg][csr][constraints]") {
    const la::CsrMatrix a = from_dense(laplacian_1d(6, 2.0), true);
    const la::DirichletConstraints c({0, 3}, {5.0, -1.0});
    la::CsrMatrix ac;
    la::constrain(a, c, ac);
    CHECK(ac.pattern_ptr() == a.pattern_ptr());
    CHECK(ac.traits().symmetric);

    const DenseMatrix d = to_dense(a), dc = to_dense(ac);
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            CAPTURE(i, j);
            const bool fixed = (i == 0 || i == 3 || j == 0 || j == 3);
            CHECK(dc[i][j] == (fixed ? (i == j ? 1.0 : 0.0) : d[i][j]));
        }
    }

    SECTION("agrees with ConstrainedOperator on vectors vanishing at the constrained dofs") {
        std::vector<double> x = test_vector(6);
        c.zero(x);
        std::vector<double> y1(6), y2(6);
        ac.apply(x, y1);
        la::ConstrainedOperator(a, c).apply(x, y2);
        CHECK(y1 == y2);
    }
    SECTION("storage is reused on repeated calls") {
        const double* storage = ac.values().data();
        la::constrain(a, c, ac);
        CHECK(ac.values().data() == storage);
    }
    SECTION("constrained dofs need a stored diagonal") {
        const la::CsrMatrix off = la::CsrMatrix::from_triplets(2, 2, {{0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 2.0}});
        la::CsrMatrix out;
        CHECK_THROWS_AS(la::constrain(off, la::DirichletConstraints({0}, {0.0}), out), std::invalid_argument);
        CHECK_THROWS_AS(la::constrain(a, la::DirichletConstraints({6}, {0.0}), out), std::out_of_range);
    }
}
