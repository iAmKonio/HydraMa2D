// Permutations, symmetric permutation of CSR matrices (B = P A P^T), and the
// (reverse) Cuthill-McKee ordering: validity, bandwidth/profile reduction on
// graphs with known optimal orderings (paths, grids) after random
// relabelling, disconnected graphs, non-symmetric patterns, the Liu-Sherman
// property (RCM envelope <= CM envelope) and the benchmark mesh.

#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "LinalgTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <set>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;

namespace {

// Symmetric matrix of a graph given by edges (diagonal = degree + 1).
la::CsrMatrix graph_matrix(int n, const std::vector<std::pair<int, int>>& edges) {
    std::vector<la::Triplet> t;
    std::vector<int> degree(n, 0);
    for (auto [a, b] : edges) {
        t.push_back({a, b, -1.0});
        t.push_back({b, a, -1.0});
        ++degree[a];
        ++degree[b];
    }
    for (int i = 0; i < n; ++i) t.push_back({i, i, degree[i] + 1.0});
    la::CsrMatrix m = la::CsrMatrix::from_triplets(n, n, t);
    m.declare_symmetric(true);
    return m;
}

// Path 0-1-2-...-(n-1) with node labels relabelled by `label` (label[v] = new id).
la::CsrMatrix relabelled_path(int n, const std::vector<int>& label) {
    std::vector<std::pair<int, int>> edges;
    for (int v = 0; v + 1 < n; ++v) edges.push_back({label[v], label[v + 1]});
    return graph_matrix(n, edges);
}

la::CsrMatrix permuted(const la::CsrMatrix& a, const la::Permutation& p) {
    return la::SymmetricPermutation(a.pattern_ptr(), p).apply(a);
}

la::CsrMatrix grid_laplacian_csr(const Mesh& m) {
    PoissonSystem s = make_poisson_system(m);
    const la::P1CsrAssembler assembler(s.mesh);
    la::CsrMatrix a = assembler.create_matrix();
    assembler.assemble_ebe(s.mesh, s.gstif, s.pdiag, a);
    return a;
}

} // namespace

// ---------------------------------------------------------------------------
// Permutation
// ---------------------------------------------------------------------------

TEST_CASE("Permutation maps between old and new numbering", "[linalg][reordering]") {
    const la::Permutation p({2, 0, 3, 1}); // new 0 <- old 2, ...
    CHECK(p.size() == 4);
    CHECK(p.new_to_old() == std::vector<int>{2, 0, 3, 1});
    CHECK(p.old_to_new() == std::vector<int>{1, 3, 0, 2});
    CHECK_FALSE(p.is_identity());
    CHECK(la::Permutation::identity(5).is_identity());
    CHECK(p.inverse().new_to_old() == p.old_to_new());
    CHECK(p.inverse().inverse().new_to_old() == p.new_to_old());

    const std::vector<double> x_old = {10, 11, 12, 13};
    std::vector<double> x_new(4), back(4);
    p.gather(x_old, x_new);
    CHECK(x_new == std::vector<double>{12, 10, 13, 11});
    p.scatter(x_new, back);
    CHECK(back == x_old);

    std::vector<double> wrong(3);
    CHECK_THROWS_AS(p.gather(x_old, wrong), std::invalid_argument);
    CHECK_THROWS_AS(p.scatter(wrong, back), std::invalid_argument);
}

TEST_CASE("Permutation rejects non-permutations", "[linalg][reordering]") {
    CHECK_THROWS_AS(la::Permutation({0, 0, 1}), std::invalid_argument);
    CHECK_THROWS_AS(la::Permutation({0, 3, 1}), std::invalid_argument);
    CHECK_THROWS_AS(la::Permutation({-1, 0}), std::invalid_argument);
    CHECK(la::Permutation().size() == 0);
}

// ---------------------------------------------------------------------------
// SymmetricPermutation
// ---------------------------------------------------------------------------

TEST_CASE("SymmetricPermutation computes P A P^T", "[linalg][reordering]") {
    const DenseMatrix d = convection_diffusion(4, 3); // non-symmetric values, symmetric pattern
    const la::CsrMatrix a = from_dense(d);
    const int n = a.rows();
    const la::Permutation p(shuffled_indices(n, 7));
    const la::SymmetricPermutation plan(a.pattern_ptr(), p);
    const la::CsrMatrix b = plan.apply(a);

    CHECK(b.nnz() == a.nnz());
    CHECK(b.pattern_ptr() == plan.permuted_pattern());
    CHECK(plan.source_pattern() == a.pattern_ptr());
    const DenseMatrix db = to_dense(b);
    for (int k = 0; k < n; ++k)
        for (int l = 0; l < n; ++l) CHECK(db[k][l] == d[p.new_to_old()[k]][p.new_to_old()[l]]);

    SECTION("products are consistent: B (P x) = P (A x)") {
        const std::vector<double> x = test_vector(n);
        std::vector<double> ax(n), pax(n), px(n), bpx(n);
        a.apply(x, ax);
        p.gather(ax, pax);
        p.gather(x, px);
        b.apply(px, bpx);
        CHECK(max_abs_diff(bpx, pax) < 1e-13);
    }
    SECTION("the plan re-permutes new values in place") {
        la::CsrMatrix a2 = a;
        for (double& v : a2.values()) v *= -2.0;
        la::CsrMatrix b2 = b;
        const double* storage = b2.values().data();
        plan.apply(a2, b2);
        CHECK(b2.values().data() == storage);
        for (size_t k = 0; k < b.nnz(); ++k) CHECK(b2.values()[k] == -2.0 * b.values()[k]);
    }
    SECTION("symmetry is carried over") {
        la::CsrMatrix s = from_dense(laplacian_1d(n), true);
        CHECK(permuted(s, p).traits().symmetric);
        CHECK(permuted(s, p).is_symmetric());
    }
}

TEST_CASE("SymmetricPermutation checks its input", "[linalg][reordering]") {
    const la::CsrMatrix a = from_dense(laplacian_1d(5));
    CHECK_THROWS_AS(la::SymmetricPermutation(a.pattern_ptr(), la::Permutation::identity(4)), std::invalid_argument);
    const la::CsrMatrix rect = la::CsrMatrix::from_triplets(2, 3, {{0, 0, 1.0}});
    CHECK_THROWS_AS(la::SymmetricPermutation(rect.pattern_ptr(), la::Permutation::identity(2)), std::invalid_argument);

    const la::SymmetricPermutation plan(a.pattern_ptr(), la::Permutation::identity(5));
    // Same size, different structure (tridiagonal plus a (0, 4) coupling).
    DenseMatrix d = laplacian_1d(5);
    d[0][4] = d[4][0] = -0.5;
    const la::CsrMatrix other = from_dense(d);
    CHECK(plan.matches(a.pattern_ptr()));
    CHECK_FALSE(plan.matches(other.pattern_ptr()));
    la::CsrMatrix out;
    CHECK_THROWS_AS(plan.apply(other, out), std::invalid_argument);
    // Identity permutation reproduces the matrix.
    CHECK(to_dense(plan.apply(a)) == to_dense(a));
}

// ---------------------------------------------------------------------------
// Cuthill-McKee / reverse Cuthill-McKee
// ---------------------------------------------------------------------------

TEST_CASE("RCM recovers the optimal ordering of a scrambled path", "[linalg][reordering][rcm]") {
    const int n = 40;
    const la::CsrMatrix a = relabelled_path(n, shuffled_indices(n, 3));
    REQUIRE(a.pattern().bandwidth() > 10);
    const la::Permutation p = la::reverse_cuthill_mckee(a.pattern());
    REQUIRE(p.size() == n);
    const la::CsrMatrix b = permuted(a, p);
    CHECK(b.pattern().bandwidth() == 1);
    CHECK(b.pattern().profile() == n - 1);
}

TEST_CASE("Pseudo-peripheral node of a path is an end point", "[linalg][reordering][rcm]") {
    const int n = 25;
    const std::vector<int> label = shuffled_indices(n, 11);
    const la::CsrMatrix a = relabelled_path(n, label);
    const int start = label[n / 2]; // middle of the path
    const int root = la::pseudo_peripheral_node(a.pattern(), start);
    CHECK((root == label[0] || root == label[n - 1]));
    CHECK_THROWS_AS(la::pseudo_peripheral_node(a.pattern(), n), std::out_of_range);
}

TEST_CASE("RCM restores a narrow band for a scrambled grid", "[linalg][reordering][rcm]") {
    GridSpec g;
    g.nx = 12;
    g.ny = 6;
    const Mesh structured = make_grid_mesh(g);
    const la::CsrMatrix natural = grid_laplacian_csr(structured);
    // Row-major numbering with diagonal couplings: bandwidth nx + 2.
    REQUIRE(natural.pattern().bandwidth() == g.nx + 2);

    const Mesh scrambled = renumbered(structured, shuffled_indices(structured.npoin, 5));
    const la::CsrMatrix a = grid_laplacian_csr(scrambled);
    REQUIRE(a.pattern().bandwidth() > 40);

    const la::CsrMatrix b = permuted(a, la::reverse_cuthill_mckee(a.pattern()));
    CAPTURE(b.pattern().bandwidth(), b.pattern().profile(), natural.pattern().profile());
    // RCM orders along the short side: the band is at most that of the
    // natural ordering along the short direction (ny + 2).
    CHECK(b.pattern().bandwidth() <= g.ny + 2);
    CHECK(b.pattern().profile() <= natural.pattern().profile());
}

TEST_CASE("RCM envelope is never larger than the Cuthill-McKee envelope", "[linalg][reordering][rcm]") {
    // Liu & Sherman (1976); reversal also preserves the bandwidth.
    const std::uint32_t seed = GENERATE(1u, 2u, 3u);
    GridSpec g;
    g.nx = 9;
    g.ny = 7;
    const Mesh m = renumbered(make_grid_mesh(g), shuffled_indices(g.npoin(), seed));
    const la::CsrMatrix a = grid_laplacian_csr(m);
    const la::CsrMatrix cm = permuted(a, la::cuthill_mckee(a.pattern()));
    const la::CsrMatrix rcm = permuted(a, la::reverse_cuthill_mckee(a.pattern()));
    CAPTURE(seed, cm.pattern().profile(), rcm.pattern().profile());
    CHECK(rcm.pattern().profile() <= cm.pattern().profile());
    CHECK(rcm.pattern().bandwidth() == cm.pattern().bandwidth());

    // RCM is CM reversed.
    std::vector<int> reversed = la::cuthill_mckee(a.pattern()).new_to_old();
    std::reverse(reversed.begin(), reversed.end());
    CHECK(la::reverse_cuthill_mckee(a.pattern()).new_to_old() == reversed);
}

TEST_CASE("RCM handles disconnected graphs and isolated nodes", "[linalg][reordering][rcm]") {
    // Two scrambled paths (sizes 6 and 5), interleaved labels, plus isolated
    // nodes 11 and 12.
    const std::vector<int> label = shuffled_indices(11, 9);
    std::vector<std::pair<int, int>> edges;
    for (int v = 0; v + 1 < 6; ++v) edges.push_back({label[v], label[v + 1]});
    for (int v = 6; v + 1 < 11; ++v) edges.push_back({label[v], label[v + 1]});
    const la::CsrMatrix a = graph_matrix(13, edges);

    const la::Permutation p = la::reverse_cuthill_mckee(a.pattern());
    REQUIRE(p.size() == 13);
    const la::CsrMatrix b = permuted(a, p);
    CHECK(b.pattern().bandwidth() == 1);

    // Each component occupies a contiguous block of the new numbering.
    std::set<int> first(label.begin(), label.begin() + 6), second(label.begin() + 6, label.end());
    auto block_is_contiguous = [&](const std::set<int>& comp) {
        int lo = 13, hi = -1;
        for (int v : comp) {
            lo = std::min(lo, p.old_to_new()[v]);
            hi = std::max(hi, p.old_to_new()[v]);
        }
        return hi - lo + 1 == static_cast<int>(comp.size());
    };
    CHECK(block_is_contiguous(first));
    CHECK(block_is_contiguous(second));
}

TEST_CASE("RCM of a structurally non-symmetric pattern uses A + A^T", "[linalg][reordering][rcm]") {
    // Directed path: only (v, v+1) stored off the diagonal, scrambled labels.
    const int n = 20;
    const std::vector<int> label = shuffled_indices(n, 21);
    std::vector<la::Triplet> t;
    for (int i = 0; i < n; ++i) t.push_back({i, i, 2.0});
    for (int v = 0; v + 1 < n; ++v) t.push_back({label[v], label[v + 1], -1.0});
    const la::CsrMatrix a = la::CsrMatrix::from_triplets(n, n, t);
    REQUIRE_FALSE(a.pattern().structurally_symmetric());
    const la::CsrMatrix b = permuted(a, la::reverse_cuthill_mckee(a.pattern()));
    CHECK(b.pattern().bandwidth() == 1);
}

TEST_CASE("RCM is deterministic and trivial cases work", "[linalg][reordering][rcm]") {
    const la::CsrMatrix a = graph_matrix(30, {{0, 17}, {17, 4}, {4, 29}, {29, 8}, {8, 0}, {3, 5}});
    CHECK(la::reverse_cuthill_mckee(a.pattern()).new_to_old() ==
          la::reverse_cuthill_mckee(a.pattern()).new_to_old());
    const la::CsrMatrix diag = from_dense(diagonal_matrix({1, 2, 3}));
    CHECK(la::reverse_cuthill_mckee(diag.pattern()).size() == 3);
    const la::CsrMatrix empty = la::CsrMatrix::from_triplets(0, 0, {});
    CHECK(la::reverse_cuthill_mckee(empty.pattern()).size() == 0);
    const la::CsrMatrix rect = la::CsrMatrix::from_triplets(2, 3, {});
    CHECK_THROWS_AS(la::reverse_cuthill_mckee(rect.pattern()), std::invalid_argument);
}

TEST_CASE("Named orderings", "[linalg][reordering]") {
    const la::CsrMatrix a = relabelled_path(10, shuffled_indices(10, 4));
    CHECK(la::ordering_names() == std::vector<std::string_view>{"none", "rcm"});
    CHECK(la::is_ordering("rcm"));
    CHECK_FALSE(la::is_ordering("amd"));
    CHECK(la::compute_ordering("none", a.pattern()).is_identity());
    CHECK(la::compute_ordering("rcm", a.pattern()).new_to_old() ==
          la::reverse_cuthill_mckee(a.pattern()).new_to_old());
    CHECK_THROWS_AS(la::compute_ordering("amd", a.pattern()), std::invalid_argument);
}

TEST_CASE("RCM on the benchmark mesh reduces bandwidth and envelope", "[linalg][reordering][rcm][hinterkante]") {
    Mesh mesh;
    {
        CaptureStdout quiet;
        const std::string dir = HYDRAMA2D_CASE_DIR;
        REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", mesh));
    }
    const la::P1CsrAssembler assembler(mesh);
    const la::SparsityPattern& p = *assembler.pattern();
    const la::Permutation rcm = la::reverse_cuthill_mckee(p);
    const la::SymmetricPermutation plan(assembler.pattern(), rcm);
    const la::SparsityPattern& q = *plan.permuted_pattern();
    CAPTURE(p.bandwidth(), q.bandwidth(), p.profile(), q.profile());
    CHECK(p.bandwidth() == 8376);
    CHECK(q.bandwidth() < 150);            // 126 with this implementation
    CHECK(q.profile() * 20 < p.profile()); // 20.4e6 -> 0.63e6
    CHECK(q.nnz() == p.nnz());
    CHECK(q.structurally_symmetric());
}
