// Mesh geometry checks on the bundled benchmark case (Hinterkante.dev.Re100):
// 17377 elements, 8956 nodes, 533 boundary sides. The domain is the box
// [-4, 12] x [-5, 5] minus a blunt plate [-4, -0.5] x [-0.5, 0.5] with a
// semicircular trailing edge of radius 0.5 (polygonally approximated).

#include "IO.hpp"
#include "LinearSolver.hpp"
#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <map>
#include <numeric>
#include <set>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

const Mesh& hinterkante_mesh() {
    static const Mesh mesh = [] {
        const std::string dir = HYDRAMA2D_CASE_DIR;
        Mesh m;
        REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", m));
        REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", m));
        compute_all_geometry(m);
        return m;
    }();
    return mesh;
}

double boundary_enclosed_area(const Mesh& m) {
    // Divergence theorem: area = closed integral of x n_x dS.
    double area = 0.0;
    for (int ib = 0; ib < m.nboun; ++ib) {
        const double xm = 0.5 * (m.x[m.iside[ib].node1] + m.x[m.iside[ib].node2]);
        area += xm * m.face_norm[ib].nx * m.face_norm[ib].length;
    }
    return area;
}

constexpr double kPi = 3.14159265358979323846;

} // namespace

TEST_CASE("Hinterkante mesh loads with the expected sizes", "[mesh][hinterkante]") {
    const Mesh& m = hinterkante_mesh();
    CHECK(m.nelem == 17377);
    CHECK(m.npoin == 8956);
    CHECK(m.nboun == 533);
    CHECK(static_cast<int>(m.detJ.size()) == m.nelem);
    CHECK(static_cast<int>(m.dNkdx.size()) == m.nelem);
    CHECK(static_cast<int>(m.elcoe_e.size()) == Mesh::nep * m.nelem);
    CHECK(static_cast<int>(m.face_norm.size()) == m.nboun);
    CHECK(static_cast<int>(m.alen_e.size()) == m.nelem);
    CHECK(static_cast<int>(m.pdiagE.size()) == m.nelem);
    CHECK(static_cast<int>(m.gstifE.size()) == m.nelem);
    CHECK(m.wall_node_norm.size() == m.wall_node_list.size());
}

TEST_CASE("Hinterkante elements are all counter-clockwise", "[mesh][hinterkante][area]") {
    const Mesh& m = hinterkante_mesh();
    int non_positive = 0;
    for (double d : m.detJ) {
        if (!(d > 0.0)) ++non_positive;
    }
    CHECK(non_positive == 0);
    CHECK(m.min_vol > 0.0);
    CHECK(m.max_vol >= m.min_vol);
}

TEST_CASE("Hinterkante element areas sum to the enclosed domain area", "[mesh][hinterkante][area]") {
    const Mesh& m = hinterkante_mesh();
    double total = 0.0;
    for (double d : m.detJ) total += 0.5 * d;

    // Exact consistency with the area enclosed by the boundary polygon.
    CHECK_THAT(total, WithinRel(boundary_enclosed_area(m), 1e-12));
    // Sanity check against the analytic geometry (polygonal semicircle).
    const double analytic = 16.0 * 10.0 - (3.5 * 1.0 + 0.5 * kPi * 0.25);
    CHECK_THAT(total, WithinRel(analytic, 1e-5));
}

TEST_CASE("Hinterkante lumped mass sums to the domain area", "[mesh][hinterkante][mass]") {
    const Mesh& m = hinterkante_mesh();
    const double total_mass = std::accumulate(m.elcoe_e.begin(), m.elcoe_e.end(), 0.0);
    double total_area = 0.0;
    for (double d : m.detJ) total_area += 0.5 * d;
    CHECK_THAT(total_mass, WithinRel(total_area, 1e-12));
}

TEST_CASE("Hinterkante shape functions satisfy the Kronecker delta property", "[mesh][hinterkante][shape]") {
    const Mesh& m = hinterkante_mesh();
    int failures = 0;
    double worst = 0.0;
    for (int ie = 0; ie < m.nelem; ++ie) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (i == j) continue;
                const int ni = m.intma[ie][i], nj = m.intma[ie][j];
                const double v = m.dNkdx[ie].dNdx[i] * (m.x[nj] - m.x[ni]) +
                                 m.dNkdx[ie].dNdy[i] * (m.y[nj] - m.y[ni]);
                const double err = std::abs(v + 1.0);
                worst = std::max(worst, err);
                if (err > 1e-9) ++failures;
            }
        }
    }
    CAPTURE(worst);
    CHECK(failures == 0);
}

TEST_CASE("Hinterkante boundary normals form a closed, outward-oriented boundary", "[mesh][hinterkante][normals]") {
    const Mesh& m = hinterkante_mesh();

    double snx = 0.0, sny = 0.0;
    int not_unit = 0, inward = 0;
    for (int ib = 0; ib < m.nboun; ++ib) {
        const FaceNormal& n = m.face_norm[ib];
        const BoundarySide& s = m.iside[ib];
        snx += n.nx * n.length;
        sny += n.ny * n.length;
        if (std::abs(n.nx * n.nx + n.ny * n.ny - 1.0) > 1e-12) ++not_unit;

        int opposite = -1;
        for (int node : m.intma[s.elem]) {
            if (node != s.node1 && node != s.node2) opposite = node;
        }
        REQUIRE(opposite >= 0); // the side really belongs to its element
        const double mx = 0.5 * (m.x[s.node1] + m.x[s.node2]);
        const double my = 0.5 * (m.y[s.node1] + m.y[s.node2]);
        if (n.nx * (mx - m.x[opposite]) + n.ny * (my - m.y[opposite]) <= 0.0) ++inward;
    }

    CHECK(not_unit == 0);
    CHECK(inward == 0);
    CHECK_THAT(snx, WithinAbs(0.0, 1e-10));
    CHECK_THAT(sny, WithinAbs(0.0, 1e-10));
}

TEST_CASE("Hinterkante outer box normals are axis-aligned", "[mesh][hinterkante][normals]") {
    // Side ids (see hinterkante.bofl): 1 bottom wall, 2 outlet, 3 top wall,
    // 4 and 8 inlet. Their outward normals are known exactly.
    const Mesh& m = hinterkante_mesh();
    struct Expect { int side_id; double nx, ny; };
    const Expect expected[] = {{1, 0.0, -1.0}, {2, 1.0, 0.0}, {3, 0.0, 1.0},
                               {4, -1.0, 0.0}, {8, -1.0, 0.0}};

    for (const auto& e : expected) {
        int count = 0;
        for (int ib = 0; ib < m.nboun; ++ib) {
            if (m.iside[ib].side_id != e.side_id) continue;
            ++count;
            CAPTURE(e.side_id, ib);
            CHECK_THAT(m.face_norm[ib].nx, WithinAbs(e.nx, 1e-12));
            CHECK_THAT(m.face_norm[ib].ny, WithinAbs(e.ny, 1e-12));
        }
        CAPTURE(e.side_id);
        CHECK(count > 0);
    }
}

// ---------------------------------------------------------------------------
// Element sizes
// ---------------------------------------------------------------------------

TEST_CASE("Hinterkante element sizes are the altitudes onto the longest sides", "[mesh][hinterkante][size]") {
    // area = 1/2 * longest side * smallest altitude, for every element.
    const Mesh& m = hinterkante_mesh();
    int non_positive = 0, mismatches = 0;
    double worst = 0.0;
    for (int ie = 0; ie < m.nelem; ++ie) {
        const auto& n = m.intma[ie];
        double lmax = 0.0;
        for (int k = 0; k < 3; ++k) {
            const int a = n[k], b = n[(k + 1) % 3];
            lmax = std::max(lmax, std::hypot(m.x[b] - m.x[a], m.y[b] - m.y[a]));
        }
        if (!(m.alen_e[ie] > 0.0)) ++non_positive;
        const double rel = std::abs(0.5 * lmax * m.alen_e[ie] - 0.5 * m.detJ[ie]) / (0.5 * m.detJ[ie]);
        worst = std::max(worst, rel);
        if (rel > 1e-12) ++mismatches;
    }
    CAPTURE(worst);
    CHECK(non_positive == 0);
    CHECK(mismatches == 0);
}

// ---------------------------------------------------------------------------
// Laplacian stiffness (assembled through the production solver path)
// ---------------------------------------------------------------------------

TEST_CASE("Hinterkante assembled Laplacian annihilates constant and linear fields", "[mesh][hinterkante][laplacian]") {
    const Mesh& m = hinterkante_mesh();
    std::vector<double> delte(m.nelem, 1.0), elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;
    update_lhs(m, delte, elcoe2, pdiag, gstif);

    std::set<int> boundary_nodes;
    for (const auto& s : m.iside) {
        boundary_nodes.insert(s.node1);
        boundary_nodes.insert(s.node2);
    }
    int non_positive_diag = 0;
    for (double d : pdiag) {
        if (!(d > 0.0)) ++non_positive_diag;
    }
    CHECK(non_positive_diag == 0);

    SECTION("constant field: K 1 == 0 at every node") {
        std::vector<double> r;
        matrix_vector_mult(m, gstif, pdiag, std::vector<double>(m.npoin, 1.0), r);
        double worst = 0.0;
        for (int ip = 0; ip < m.npoin; ++ip) worst = std::max(worst, std::abs(r[ip]) / pdiag[ip]);
        CHECK(worst < 1e-12);
    }
    SECTION("linear field: K f == 0 at interior nodes") {
        std::vector<double> f(m.npoin), r;
        for (int ip = 0; ip < m.npoin; ++ip) f[ip] = 1.0 + 0.3 * m.x[ip] - 0.8 * m.y[ip];
        matrix_vector_mult(m, gstif, pdiag, f, r);
        double worst = 0.0;
        for (int ip = 0; ip < m.npoin; ++ip) {
            if (!boundary_nodes.count(ip)) worst = std::max(worst, std::abs(r[ip]) / pdiag[ip]);
        }
        CHECK(worst < 1e-10);
    }
}

// ---------------------------------------------------------------------------
// Wall nodes
// ---------------------------------------------------------------------------

TEST_CASE("Hinterkante wall nodes are exactly the nodes of the wall sides", "[mesh][hinterkante][wall]") {
    // Walls (flag 500): bottom (1), top (3) and the plate (5, 6, 7) - three
    // open wall chains: 52 + 52 + 279 = 383 nodes.
    const Mesh& m = hinterkante_mesh();
    std::set<int> expected;
    for (const auto& s : m.iside) {
        if (s.flag == 500 || s.flag == 501 || s.flag == 502) {
            expected.insert(s.node1);
            expected.insert(s.node2);
        }
    }
    CHECK(m.wall_node_list.size() == 383);
    CHECK(expected.size() == 383);
    CHECK(std::set<int>(m.wall_node_list.begin(), m.wall_node_list.end()) == expected);
}

TEST_CASE("Hinterkante wall normals point out of the fluid", "[mesh][hinterkante][wall]") {
    const Mesh& m = hinterkante_mesh();
    std::map<int, std::set<int>> side_ids_of_node;
    for (const auto& s : m.iside) {
        side_ids_of_node[s.node1].insert(s.side_id);
        side_ids_of_node[s.node2].insert(s.side_id);
    }

    int checked_flat = 0, checked_round = 0;
    for (size_t k = 0; k < m.wall_node_list.size(); ++k) {
        const int ip = m.wall_node_list[k];
        const auto& ids = side_ids_of_node[ip];
        const auto& n = m.wall_node_norm[k];
        if (n[0] == 0.0 && n[1] == 0.0) continue; // see characterization test below
        CAPTURE(ip, m.x[ip], m.y[ip]);
        CHECK_THAT(std::hypot(n[0], n[1]), WithinAbs(1.0, 1e-12));

        if (ids.size() != 1) continue; // junction nodes between two side ids
        const int id = *ids.begin();
        if (id == 1 || id == 5) { // bottom wall, top of plate: fluid above
            CHECK_THAT(n[0], WithinAbs(0.0, 1e-12));
            CHECK_THAT(n[1], WithinAbs(-1.0, 1e-12));
            ++checked_flat;
        } else if (id == 3 || id == 7) { // top wall, bottom of plate: fluid below
            CHECK_THAT(n[0], WithinAbs(0.0, 1e-12));
            CHECK_THAT(n[1], WithinAbs(1.0, 1e-12));
            ++checked_flat;
        } else if (id == 6) { // semicircular trailing edge, centre (-0.5, 0)
            const double rx = -0.5 - m.x[ip], ry = -m.y[ip];
            const double r = std::hypot(rx, ry);
            CHECK((n[0] * rx + n[1] * ry) / r > 0.999);
            ++checked_round;
        }
    }
    // Nodes on a single side id (junction nodes excluded):
    // flat: sides 1, 3 (52 nodes - 2 each) + sides 5, 7 (102 nodes - 2 each); round: 77 - 2.
    CHECK(checked_flat == 2 * 50 + 2 * 100);
    CHECK(checked_round == 75);
}

TEST_CASE("Hinterkante trailing edge classification", "[mesh][hinterkante][wall][characterization]") {
    // Documents current behaviour (suspected defect, pending a decision): the
    // plate has a rounded trailing edge, yet the single-face endpoints of the
    // open wall chains whose face normal has nx + ny < -0.2 are classified as
    // trailing edge (see test_mesh_wall_nodes.cpp). Their wall normals are
    // zeroed and the last one, the domain corner (-4, -5), is reported.
    const Mesh& m = hinterkante_mesh();

    const int corner = find_node(m, -4.0, -5.0);
    REQUIRE(corner >= 0);
    CHECK(m.trailing_edge_node == corner);

    std::set<std::pair<double, double>> zeroed;
    for (size_t k = 0; k < m.wall_node_list.size(); ++k) {
        if (m.wall_node_norm[k][0] == 0.0 && m.wall_node_norm[k][1] == 0.0) {
            const int ip = m.wall_node_list[k];
            zeroed.insert({m.x[ip], m.y[ip]});
        }
    }
    const std::set<std::pair<double, double>> expected = {{-4.0, -5.0}, {12.0, -5.0}, {-4.0, 0.5}};
    CHECK(zeroed == expected);
}
