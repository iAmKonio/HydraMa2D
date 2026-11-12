// Meshes with quadrilaterals: element numbering and helpers of Mesh, the
// geometry computed for quadrilaterals (areas, lumped masses, length scales,
// Laplacians), update_lhs / EBE storage, wall geometry, VTK output, and the
// importer (orientation, validity, duplicates, boundary sides, all Gmsh
// format variants).

#include "GmshTestWriter.hpp"
#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "LinearSolver.hpp"
#include "QuadTestUtils.hpp"
#include "geometry/FieldOps.hpp"
#include "geometry/WallGeometry.hpp"
#include "linalg/EbeOperator.hpp"
#include "mesh/GmshReader.hpp"
#include "mesh/MeshBuilder.hpp"
#include "mesh/MeshCheck.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <stdexcept>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using hydrama::mesh::check_mesh;
using Catch::Matchers::WithinRel;

namespace {

void prepare(Mesh& m) {
    CaptureStdout quiet;
    compute_all_geometry(m);
}

// Two rectangles 0.5 x 0.2 (quadrilaterals) and a triangle on the right:
//   3 --- 4 --- 5
//   |  Q0 |  Q1 |  T0      T0 = (2, 6, 5)
//   0 --- 1 --- 2 --- 6
HybridSpec small_mixed() {
    HybridSpec s;
    s.nodes = {{0, 0}, {0.5, 0}, {1.0, 0}, {0, 0.2}, {0.5, 0.2}, {1.0, 0.2}, {1.4, 0}};
    s.quads = {{0, 1, 4, 3}, {1, 2, 5, 4}};
    s.triangles = {{2, 6, 5}};
    return s;
}

mesh::MeshData data_of(const HybridSpec& s, bool with_lines = true) {
    mesh::MeshData d;
    d.source = "test.msh";
    for (std::size_t i = 0; i < s.nodes.size(); ++i) {
        d.node_tags.push_back(static_cast<std::int64_t>(i + 1));
        d.x.push_back(s.nodes[i].x);
        d.y.push_back(s.nodes[i].y);
        d.z.push_back(0.0);
    }
    std::int64_t tag = 1;
    for (const auto& q : s.quads)
        d.elements.push_back({mesh::ElementType::Quad4, {q[0] + 1, q[1] + 1, q[2] + 1, q[3] + 1}, 0, 0, tag++});
    for (const auto& t : s.triangles) d.elements.push_back({mesh::ElementType::Tri3, {t[0] + 1, t[1] + 1, t[2] + 1, 0}, 0, 0, tag++});
    if (with_lines) {
        const Mesh m = make_hybrid_mesh(s);
        for (const auto& side : m.iside)
            d.elements.push_back({mesh::ElementType::Line2, {side.node1 + 1, side.node2 + 1, 0, 0}, 1, 0, tag++});
        d.physical_names[{1, 1}] = "wall";
    }
    return d;
}

std::string build_error(const mesh::MeshData& d) {
    try {
        mesh::build_mesh(d);
    } catch (const mesh::MeshImportError& e) {
        return e.what();
    }
    FAIL("expected MeshImportError");
    return {};
}

} // namespace

// =============================================================================
// Mesh
// =============================================================================

TEST_CASE("Mesh: element numbering with triangles and quadrilaterals", "[quad][mesh]") {
    Mesh m = make_hybrid_mesh(small_mixed());
    CHECK(m.nelem == 3);
    CHECK(m.num_triangles() == 1);
    CHECK(m.num_quads() == 2);
    CHECK(m.has_quads());
    CHECK_FALSE(m.is_quad(0));
    CHECK(m.is_quad(1));
    CHECK(m.quad_of(2) == 1);
    CHECK(m.element_of_quad(0) == 1);
    CHECK(m.nodes_per_element(0) == 3);
    CHECK(m.nodes_per_element(2) == 4);
    CHECK(m.num_offdiag_blocks() == 1 + 2 * 2);
    const ElementNodes t = m.element_nodes(0), q = m.element_nodes(2);
    CHECK(std::vector<int>(t.begin(), t.end()) == std::vector<int>{2, 6, 5});
    CHECK(std::vector<int>(q.begin(), q.end()) == std::vector<int>{1, 2, 5, 4});
    CHECK(q.size() == 4);
    CHECK(q[3] == 4);
    REQUIRE(check_mesh(m).empty());
    CHECK(m.nboun == 7);
}

TEST_CASE("Mesh: geometry of quadrilaterals", "[quad][mesh]") {
    Mesh m = make_hybrid_mesh(small_mixed());
    prepare(m);
    REQUIRE(m.quad_geom.size() == 2);
    REQUIRE(m.quad_lap.size() == 2);
    REQUIRE(m.detJ.size() == 3);
    REQUIRE(m.alen_e.size() == 3);
    CHECK(m.elcoe_e.size() == 3); // triangles only
    for (int q = 0; q < 2; ++q) {
        const int e = m.element_of_quad(q);
        CHECK_THAT(m.detJ[static_cast<std::size_t>(e)], WithinRel(2.0 * 0.1, 1e-15)); // 2 x area
        CHECK_THAT(m.alen_e[static_cast<std::size_t>(e)], WithinRel(0.2, 1e-15));    // min(hx, hy)
        for (double mass : m.quad_geom[static_cast<std::size_t>(q)].mass) CHECK_THAT(mass, WithinRel(0.025, 1e-14));
        // Mean gradients of a rectangle: +-1/(2 hx), +-1/(2 hy).
        CHECK_THAT(m.dNkdx[static_cast<std::size_t>(e)].dNdx[0], WithinRel(-1.0, 1e-14));
        CHECK_THAT(m.dNkdx[static_cast<std::size_t>(e)].dNdy[0], WithinRel(-2.5, 1e-14));
    }
    CHECK_THAT(m.max_vol, WithinRel(0.1, 1e-14));
    CHECK_THAT(m.min_vol, WithinRel(0.04, 1e-14)); // triangle 0.4 x 0.2 / 2
    // Lumped nodal masses: quarter areas from the quadrilaterals, thirds from the triangle.
    const std::vector<double> mass = lumped_nodal_mass(m);
    CHECK_THAT(mass[2], WithinRel(0.025 + 0.04 / 3, 1e-14));
    CHECK_THAT(mass[1], WithinRel(0.05, 1e-14));
    CHECK_THAT(mass[6], WithinRel(0.04 / 3, 1e-14));
    // Heights above edges.
    CHECK_THAT(m.height_above_edge(1, 0, 1), WithinRel(0.2, 1e-14));
    CHECK_THAT(m.height_above_edge(1, 1, 4), WithinRel(0.5, 1e-14));
    CHECK_THAT(m.height_above_edge(0, 2, 6), WithinRel(0.2, 1e-14)); // triangle: 2 A / L
    // A trapezoid: mean distance of the two other nodes.
    Mesh trap = make_hybrid_mesh(HybridSpec{{{0, 0}, {1, 0}, {1, 0.6}, {0, 0.2}}, {}, {{0, 1, 2, 3}}});
    prepare(trap);
    CHECK_THAT(trap.height_above_edge(0, 0, 1), WithinRel(0.4, 1e-14));
    CHECK_THAT(trap.alen_e[0], WithinRel(trap.quad_geom[0].area / std::hypot(1.0, 0.4), 1e-14));
}

TEST_CASE("Mesh: clockwise quadrilaterals built without the importer are rejected", "[quad][mesh]") {
    Mesh m = make_hybrid_mesh(HybridSpec{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}, {}, {{0, 3, 2, 1}}});
    CaptureStdout quiet;
    CHECK_THROWS_WITH(m.compute_shape_function_derivatives(),
                      ContainsSubstring("quadrilateral element 0 has a non-positive Jacobian determinant"));
}

TEST_CASE("update_lhs: quadrilateral blocks of the element-by-element storage", "[quad][mesh][lhs]") {
    Mesh m = make_hybrid_mesh(small_mixed());
    prepare(m);
    const std::vector<double> dt = {0.5, 0.25, 0.125};
    std::vector<double> elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;
    update_lhs(m, dt, elcoe2, pdiag, gstif);
    REQUIRE(gstif.size() == 5);
    for (int q = 0; q < 2; ++q) {
        const QuadLaplacian& k = m.quad_lap[static_cast<std::size_t>(q)];
        const double d = dt[static_cast<std::size_t>(1 + q)];
        const auto& b0 = gstif[static_cast<std::size_t>(1 + 2 * q)];
        const auto& b1 = gstif[static_cast<std::size_t>(2 + 2 * q)];
        CHECK(b0 == std::array<double, 3>{k.off[0] * d, k.off[1] * d, k.off[2] * d});
        CHECK(b1 == std::array<double, 3>{k.off[3] * d, k.off[4] * d, k.off[5] * d});
    }
    // 1 / sum_e (M_e,i / dt_e) at node 1 (two quadrilaterals).
    CHECK_THAT(elcoe2[1], WithinRel(1.0 / (0.025 / 0.25 + 0.025 / 0.125), 1e-14));
    // EbeOperator checks the block count.
    std::vector<std::array<double, 3>> wrong(3);
    CHECK_THROWS_WITH(linalg::EbeOperator(m, wrong, pdiag), ContainsSubstring("(5 off-diagonal blocks)"));
}

TEST_CASE("compute_timestep: quadrilateral length scales and node velocities", "[quad][mesh][timestep]") {
    Mesh m = make_hybrid_mesh(small_mixed());
    prepare(m);
    std::vector<double> vel(7, 0.0), dt;
    vel[5] = 2.0; // node of Q1 and T0
    compute_timestep(1, 0, 0, 0.0, 0.01, 0.5, m, vel, dt); // local time steps
    // Q0: diffusion limit 0.5 * 0.2^2 * 0.5 / 0.01 = 1.0, convection 0.5 * 0.2 / 1e-15.
    CHECK_THAT(dt[1], WithinRel(1.0, 1e-14));
    // Q1: convection limit 0.5 * 0.2 / 2 = 0.05.
    CHECK_THAT(dt[2], WithinRel(0.05, 1e-14));
}

TEST_CASE("WallGeometry: wall faces on quadrilaterals", "[quad][mesh][wallgeom]") {
    // Channel with quadrilateral rows at the walls (inflation layers), walls at y = 0 and y = 1.
    GridOptions g;
    g.cells = GridCells::InflationLayers;
    g.nx = 6;
    g.ny = 5;
    g.layers = 1;
    Mesh m = make_hybrid_mesh(grid(g), [](double, double y) { return (y == 0.0 || y == 1.0) ? 500 : 504; });
    prepare(m);
    const WallGeometry w(m);
    REQUIRE(w.num_faces() == 12);
    for (int f = 0; f < w.num_faces(); ++f) {
        const WallFace& face = w.face(f);
        CHECK(m.is_quad(face.elem));
        CHECK(face.opposite_node == -1);
        CHECK_THAT(face.yp, WithinRel(0.2, 1e-14)); // first cell height
        const double ym = 0.5 * (m.y[face.node1] + m.y[face.node2]);
        CHECK(face.ny == (ym == 0.0 ? 1.0 : -1.0)); // into the fluid
        CHECK(face.nx == 0.0);
    }
    for (int i = 0; i < m.npoin; ++i) // exact wall distance in a channel
        CHECK_THAT(w.wall_distance()[static_cast<std::size_t>(i)], WithinAbs(std::min(m.y[i], 1.0 - m.y[i]), 1e-14));
}

TEST_CASE("write_vtk: triangles (type 5) and quadrilaterals (type 9)", "[quad][mesh][vtk]") {
    Mesh m = make_hybrid_mesh(small_mixed());
    TempDir dir("vtk-quads");
    const std::vector<double> zero(7, 0.0);
    REQUIRE(IO::write_vtk((dir / "m.vtk").string(), m, zero, zero, zero));
    const std::string text = read_text(dir / "m.vtk");
    CHECK_THAT(text, ContainsSubstring("POINTS 7 double\n"));
    CHECK_THAT(text, ContainsSubstring("\nCELLS 3 14\n3 2 6 5\n4 0 1 4 3\n4 1 2 5 4\nCELL_TYPES 3\n5\n9\n9\nPOINT_DATA 7\n"));
}

// =============================================================================
// Importer
// =============================================================================

TEST_CASE("build_mesh: quadrilaterals are numbered after the triangles, in file order", "[quad][mesh_import]") {
    mesh::ImportReport r;
    const Mesh m = mesh::build_mesh(data_of(small_mixed()), {}, &r);
    REQUIRE(check_mesh(m).empty());
    CHECK(m.intma == std::vector<std::array<int, 3>>{{2, 6, 5}});
    CHECK(m.quads == std::vector<std::array<int, 4>>{{0, 1, 4, 3}, {1, 2, 5, 4}});
    CHECK(r.triangles == 1);
    CHECK(r.quadrilaterals == 2);
    CHECK_THAT(mesh::format_report(r), ContainsSubstring("Mesh import: 1 triangles, 2 quadrilaterals, 7 nodes, 7 boundary sides"));
    // Boundary sides of quadrilaterals run counter-clockwise in their element.
    for (const BoundarySide& s : m.iside) {
        const ElementNodes n = m.element_nodes(s.elem);
        int k1 = -1, k2 = -1;
        for (int k = 0; k < n.size(); ++k) {
            if (n[k] == s.node1) k1 = k;
            if (n[k] == s.node2) k2 = k;
        }
        CHECK((k1 + 1) % n.size() == k2);
    }
}

TEST_CASE("build_mesh: clockwise quadrilaterals are reversed, keeping the first node", "[quad][mesh_import]") {
    HybridSpec s = small_mixed();
    for (auto& q : s.quads) q = {q[0], q[3], q[2], q[1]};
    mesh::ImportReport r;
    const Mesh m = mesh::build_mesh(data_of(s), {}, &r);
    CHECK(r.reoriented_quadrilaterals == 2);
    CHECK(m.quads == std::vector<std::array<int, 4>>{{0, 1, 4, 3}, {1, 2, 5, 4}});
    CHECK_THAT(mesh::format_report(r), ContainsSubstring("2 clockwise quadrilateral(s) reoriented counter-clockwise"));
}

TEST_CASE("build_mesh: invalid quadrilaterals", "[quad][mesh_import][errors]") {
    auto one_quad = [](std::array<Point, 4> p) {
        HybridSpec s;
        s.nodes.assign(p.begin(), p.end());
        s.quads = {{0, 1, 2, 3}};
        mesh::MeshData d = data_of(s, false);
        return build_error(d);
    };
    CHECK_THAT(one_quad({Point{0, 0}, {2, 0}, {0.6, 0.6}, {0, 2}}),
               ContainsSubstring("quadrilateral 1 is not convex or self-intersecting (its Jacobian determinant changes sign)"));
    CHECK_THAT(one_quad({Point{0, 0}, {2, 0}, {0.9, 0.9}, {0, 2}}), ContainsSubstring("is not convex")); // det J > 0 at the Gauss points
    CHECK_THAT(one_quad({Point{0, 0}, {1, 0}, {0, 1}, {1, 1}}), ContainsSubstring("is not convex or self-intersecting"));
    CHECK_THAT(one_quad({Point{0, 0}, {1, 0}, {2, 0}, {0, 1}}),
               ContainsSubstring("quadrilateral 1 is degenerate (a corner angle of 0 or 180 degrees: zero Jacobian determinant): "
                                 "node 1 (0, 0), node 2 (1, 0), node 3 (2, 0), node 4 (0, 1)"));
    CHECK_THAT(one_quad({Point{0, 0}, {1, 0}, {1, 1}, {1 + 1e-14, 1}}), ContainsSubstring("is degenerate"));

    mesh::MeshData undefined = data_of(small_mixed());
    undefined.elements[0].nodes[2] = 99;
    CHECK_THAT(build_error(undefined), ContainsSubstring("quadrilateral 1 references undefined node 99"));
}

TEST_CASE("build_mesh: hanging nodes are not conforming", "[quad][mesh_import][errors]") {
    // One quadrilateral on the left, two on the right: node 5 (1, 1) hangs on
    // the edge (1, 0) - (1, 2) of the left one. Only the outer boundary has
    // line elements, so the three edges around the hanging node are open.
    HybridSpec s;
    s.nodes = {{0, 0}, {1, 0}, {2, 0}, {0, 2}, {1, 2}, {1, 1}, {2, 1}, {2, 2}};
    s.quads = {{0, 1, 4, 3}, {1, 2, 6, 5}, {5, 6, 7, 4}};
    mesh::MeshData d = data_of(s, false);
    std::int64_t tag = 100;
    for (const auto& [a, b] : std::vector<std::pair<int, int>>{{0, 1}, {1, 2}, {2, 6}, {6, 7}, {7, 4}, {4, 3}, {3, 0}})
        d.elements.push_back({mesh::ElementType::Line2, {a + 1, b + 1, 0, 0}, 1, 0, tag++});
    d.physical_names[{1, 1}] = "wall";
    CHECK_THAT(build_error(d), ContainsSubstring("3 boundary edges have no line element"));
}

TEST_CASE("build_mesh: quadrilaterals in several physical groups are merged", "[quad][mesh_import]") {
    mesh::MeshData d = data_of(small_mixed());
    mesh::RawElement dup = d.elements[0];
    dup.physical = 7;
    dup.nodes = {dup.nodes[2], dup.nodes[3], dup.nodes[0], dup.nodes[1]}; // rotated
    d.elements.push_back(dup);
    mesh::ImportReport r;
    const Mesh m = mesh::build_mesh(d, {}, &r);
    CHECK(m.num_quads() == 2);
    CHECK(r.duplicate_elements_merged == 1);
}

TEST_CASE("Gmsh files with quadrilaterals: every format variant gives the same mesh", "[quad][mesh_import][gmsh]") {
    GridOptions g;
    g.cells = GridCells::InflationLayers;
    g.nx = 5;
    g.ny = 4;
    g.layers = 1;
    g.perturbation = 0.15;
    const Mesh ref = make_hybrid_mesh(grid(g), [](double, double y) { return y == 0.0 ? 1 : 2; });
    GmshModel model;
    for (int i = 0; i < ref.npoin; ++i) model.add_node(1000 + 3 * i, ref.x[static_cast<std::size_t>(i)], ref.y[static_cast<std::size_t>(i)]);
    auto tag = [](int i) { return static_cast<std::int64_t>(1000 + 3 * i); };
    for (const auto& t : ref.intma) model.add_element(2, {tag(t[0]), tag(t[1]), tag(t[2])}, {10});
    for (const auto& q : ref.quads) model.add_element(3, {tag(q[0]), tag(q[1]), tag(q[2]), tag(q[3])}, {10});
    for (const auto& s : ref.iside) model.add_element(1, {tag(s.node1), tag(s.node2)}, {s.side_id});
    model.names[{1, 1}] = "bottom_wall";
    model.names[{1, 2}] = "other walls";
    for (const std::string version : {"2.2", "4.1"}) {
        for (const bool binary : {false, true}) {
            CAPTURE(version, binary);
            GmshWriteOptions o;
            o.version = version;
            o.binary = binary;
            const Mesh m = mesh::build_mesh(mesh::parse_gmsh(write_gmsh(model, o), "q.msh"));
            REQUIRE(check_mesh(m).empty());
            CHECK(m.intma == ref.intma);
            CHECK(m.quads == ref.quads);
            CHECK(m.x == ref.x);
            CHECK(m.y == ref.y);
            CHECK(m.nboun == ref.nboun);
        }
    }
}
