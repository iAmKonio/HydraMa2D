// Element-to-CSR assembly (P1CsrAssembler): sparsity pattern of the P1 node
// graph, element scatter map, and the assembled pressure operator against
//   * the matrix-free EBE operator (dense matrix: bit-identical entries;
//     products: equal to round-off),
//   * the independent dense reference sum_e dt_e K_e,
//   * generic element-matrix assembly,
// on analytic grids and on the Hinterkante benchmark mesh.

#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "LinalgTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <set>
#include <utility>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;

namespace {

GridSpec assembly_grid() {
    GridSpec g;
    g.x0 = 0.25;
    g.y0 = -0.5;
    g.lx = 2.0;
    g.ly = 1.25;
    g.nx = 7;
    g.ny = 5;
    return g;
}

PoissonSystem grid_system() {
    const GridSpec g = assembly_grid();
    return make_poisson_system(make_grid_mesh(g), varying_dt(g.nelem()));
}

const PoissonSystem& hinterkante_system() {
    static const PoissonSystem s = [] {
        Mesh mesh;
        CaptureStdout quiet;
        const std::string dir = HYDRAMA2D_CASE_DIR;
        REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", mesh));
        REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", mesh));
        return make_poisson_system(std::move(mesh), std::vector<double>(17377, 4.9e-3));
    }();
    return s;
}

la::CsrMatrix assemble(const PoissonSystem& s, const la::P1CsrAssembler& assembler) {
    la::CsrMatrix a = assembler.create_matrix();
    assembler.assemble_ebe(s.mesh, s.gstif, s.pdiag, a);
    return a;
}

// ||A_csr x - A_ebe x||_inf relative to ||A_ebe x||_inf over a few vectors.
double worst_product_mismatch(const PoissonSystem& s, const la::CsrMatrix& a) {
    const la::EbeOperator ebe(s.mesh, s.gstif, s.pdiag);
    const int n = s.mesh.npoin;
    double worst = 0.0;
    for (double phase : {0.1, 1.3, 2.9}) {
        const std::vector<double> x = test_vector(n, phase);
        std::vector<double> y_csr(n), y_ebe(n);
        a.apply(x, y_csr);
        ebe.apply(x, y_ebe);
        double scale = 0.0;
        for (double v : y_ebe) scale = std::max(scale, std::abs(v));
        worst = std::max(worst, max_abs_diff(y_csr, y_ebe) / scale);
    }
    return worst;
}

} // namespace

// ---------------------------------------------------------------------------
// Pattern and scatter map
// ---------------------------------------------------------------------------

TEST_CASE("P1 pattern is the node graph of the mesh plus the diagonal", "[linalg][assembly]") {
    const GridSpec g = assembly_grid();
    Mesh m = make_grid_mesh(g);
    const la::P1CsrAssembler assembler(m);
    const la::SparsityPattern& p = *assembler.pattern();

    // Edges of the split grid: horizontal + vertical + one diagonal per cell.
    const int edges = g.nx * (g.ny + 1) + g.ny * (g.nx + 1) + g.nx * g.ny;
    CHECK(p.rows() == m.npoin);
    CHECK(p.square());
    CHECK(p.nnz() == static_cast<size_t>(m.npoin + 2 * edges));
    CHECK(p.has_full_diagonal());
    CHECK(p.structurally_symmetric());

    // Brute force: (i, j) stored iff i == j or i and j share an element.
    std::set<std::pair<int, int>> coupled;
    for (const auto& e : m.intma)
        for (int a : e)
            for (int b : e) coupled.insert({a, b});
    for (int i = 0; i < m.npoin; ++i) {
        for (int j = 0; j < m.npoin; ++j) {
            const bool expected = (i == j) || coupled.count({i, j}) > 0;
            if ((p.find(i, j) >= 0) != expected) {
                CAPTURE(i, j);
                FAIL_CHECK("pattern entry mismatch");
            }
        }
    }
}

TEST_CASE("Element scatter map points at the coupled nodes", "[linalg][assembly]") {
    const Mesh m = make_grid_mesh(assembly_grid());
    const la::P1CsrAssembler assembler(m);
    const la::SparsityPattern& p = *assembler.pattern();
    CHECK(assembler.nelem() == m.nelem);
    CHECK(assembler.npoin() == m.npoin);
    for (int e = 0; e < m.nelem; ++e) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                CAPTURE(e, a, b);
                const int pos = assembler.position(e, a, b);
                REQUIRE(pos >= 0);
                CHECK(pos == p.find(m.intma[e][a], m.intma[e][b]));
            }
        }
    }
}

TEST_CASE("Nodes without elements still get a diagonal entry", "[linalg][assembly]") {
    Mesh m = make_mesh({{0, 0}, {1, 0}, {0, 1}, {5, 5}}, {{0, 1, 2}}); // node 3 is isolated
    const la::P1CsrAssembler assembler(m);
    const la::SparsityPattern& p = *assembler.pattern();
    CHECK(p.has_full_diagonal());
    CHECK(p.row_end(3) - p.row_begin(3) == 1);
    CHECK(p.nnz() == 10);
}

TEST_CASE("P1CsrAssembler rejects invalid meshes", "[linalg][assembly]") {
    Mesh m = make_mesh({{0, 0}, {1, 0}, {0, 1}}, {{0, 1, 3}});
    CHECK_THROWS_AS(la::P1CsrAssembler(m), std::invalid_argument);
    Mesh inconsistent = make_mesh({{0, 0}, {1, 0}, {0, 1}}, {{0, 1, 2}});
    inconsistent.nelem = 2;
    CHECK_THROWS_AS(la::P1CsrAssembler(inconsistent), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Assembled pressure operator
// ---------------------------------------------------------------------------

TEST_CASE("Assembled CSR matrix has exactly the entries of the EBE operator", "[linalg][assembly]") {
    // Both sum the same element contributions in element order, so the dense
    // matrices agree bit for bit.
    const PoissonSystem s = grid_system();
    const la::P1CsrAssembler assembler(s.mesh);
    const la::CsrMatrix a = assemble(s, assembler);
    CHECK(a.traits().symmetric);
    CHECK(a.is_symmetric());
    CHECK(to_dense(a) == dense_operator(s.mesh, s.gstif, s.pdiag));
}

TEST_CASE("Assembled CSR matrix matches the independent dense Laplacian reference", "[linalg][assembly]") {
    const PoissonSystem s = grid_system();
    const la::P1CsrAssembler assembler(s.mesh);
    const DenseMatrix ref = dense_reference_laplacian(s.mesh, s.delte);
    CHECK(max_abs_diff(to_dense(assemble(s, assembler)), ref) < 1e-14 * max_abs(ref));
}

TEST_CASE("CSR and EBE products agree to round-off", "[linalg][assembly]") {
    const bool real_mesh = GENERATE(false, true);
    CAPTURE(real_mesh);
    const PoissonSystem grid = real_mesh ? PoissonSystem{} : grid_system();
    const PoissonSystem& s = real_mesh ? hinterkante_system() : grid;
    const la::P1CsrAssembler assembler(s.mesh);
    const la::CsrMatrix a = assemble(s, assembler);
    CHECK(worst_product_mismatch(s, a) < 1e-13);

    std::vector<double> d(s.mesh.npoin);
    a.diagonal(d);
    CHECK(d == s.pdiag);
}

TEST_CASE("Assembled Laplacian annihilates constants", "[linalg][assembly]") {
    // Row sums of the (unconstrained) P1 Laplacian vanish.
    const PoissonSystem s = grid_system();
    const la::P1CsrAssembler assembler(s.mesh);
    const la::CsrMatrix a = assemble(s, assembler);
    const std::vector<double> ones(s.mesh.npoin, 1.0);
    std::vector<double> y(s.mesh.npoin);
    a.apply(ones, y);
    for (int i = 0; i < s.mesh.npoin; ++i) CHECK_THAT(y[i], WithinAbs(0.0, 1e-14 * s.pdiag[i]));
}

TEST_CASE("Generic element-matrix assembly reproduces the EBE assembly", "[linalg][assembly]") {
    // K_e dt_e added as full 3x3 element matrices accumulates the same numbers
    // in the same element order as update_lhs + assemble_ebe.
    const PoissonSystem s = grid_system();
    const la::P1CsrAssembler assembler(s.mesh);
    la::CsrMatrix generic = assembler.create_matrix();
    const int pair[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int e = 0; e < s.mesh.nelem; ++e) {
        la::P1CsrAssembler::ElementMatrix ke{};
        for (int a = 0; a < 3; ++a) ke[a][a] = s.mesh.pdiagE[e][a] * s.delte[e];
        for (int k = 0; k < 3; ++k) {
            ke[pair[k][0]][pair[k][1]] = s.mesh.gstifE[e][k] * s.delte[e];
            ke[pair[k][1]][pair[k][0]] = s.mesh.gstifE[e][k] * s.delte[e];
        }
        assembler.add_element_matrix(generic, e, ke);
    }
    CHECK(generic.values() == assemble(s, assembler).values());
    CHECK_THROWS_AS(assembler.add_element_matrix(generic, s.mesh.nelem, {}), std::out_of_range);
}

TEST_CASE("Re-assembly with new time steps reuses pattern and storage", "[linalg][assembly]") {
    PoissonSystem s = grid_system();
    const la::P1CsrAssembler assembler(s.mesh);
    la::CsrMatrix a = assemble(s, assembler);
    const double* storage = a.values().data();
    const std::vector<double> before = a.values();

    // Uniform time step 4x larger: every entry scales by 4.
    std::vector<double> dt(s.mesh.nelem, 0.04);
    update_lhs(s.mesh, dt, s.elcoe2, s.pdiag, s.gstif);
    std::vector<double> dt1(s.mesh.nelem, 0.01);
    PoissonSystem s1 = s;
    update_lhs(s1.mesh, dt1, s1.elcoe2, s1.pdiag, s1.gstif);
    la::CsrMatrix a1 = assemble(s1, assembler);

    assembler.assemble_ebe(s.mesh, s.gstif, s.pdiag, a);
    CHECK(a.values().data() == storage);
    CHECK(a.pattern_ptr() == assembler.pattern());
    CHECK(a.values() != before);
    for (size_t k = 0; k < a.nnz(); ++k) CHECK_THAT(a.values()[k], WithinAbs(4.0 * a1.values()[k], 1e-15));
}

TEST_CASE("Assembly validates its arguments", "[linalg][assembly]") {
    const PoissonSystem s = grid_system();
    const la::P1CsrAssembler assembler(s.mesh);
    SECTION("matrix with a foreign pattern") {
        la::CsrMatrix foreign = from_dense(laplacian_1d(s.mesh.npoin));
        CHECK_THROWS_AS(assembler.assemble_ebe(s.mesh, s.gstif, s.pdiag, foreign), std::invalid_argument);
        CHECK_THROWS_AS(assembler.add_element_matrix(foreign, 0, {}), std::invalid_argument);
    }
    SECTION("coefficient arrays of the wrong size") {
        la::CsrMatrix a = assembler.create_matrix();
        std::vector<double> short_pdiag(s.pdiag.begin(), s.pdiag.end() - 1);
        std::vector<std::array<double, 3>> short_gstif(s.gstif.begin(), s.gstif.end() - 1);
        CHECK_THROWS_AS(assembler.assemble_ebe(s.mesh, s.gstif, short_pdiag, a), std::invalid_argument);
        CHECK_THROWS_AS(assembler.assemble_ebe(s.mesh, short_gstif, s.pdiag, a), std::invalid_argument);
    }
}

TEST_CASE("Benchmark mesh: assembled pressure matrix", "[linalg][assembly][hinterkante]") {
    const PoissonSystem& s = hinterkante_system();
    const la::P1CsrAssembler assembler(s.mesh);
    const la::CsrMatrix a = assemble(s, assembler);
    CHECK(a.rows() == 8956);
    CHECK(a.nnz() == 61620); // 8956 nodes + 2 x 26332 edges
    CHECK(a.pattern().structurally_symmetric());
    CHECK(a.is_symmetric());
    CHECK(a.pattern().bandwidth() == 8376); // original node numbering
}
