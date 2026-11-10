// WallGeometry: wall faces (adjacent element, opposite node, length, inward
// normal, first-cell height y_p), wall nodes (normals, mean y_p, node -> face
// map) and input validation.

#include "WallTestUtils.hpp"
#include "BoundaryKind.hpp"
#include "geometry/WallGeometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <set>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kPi = 3.14159265358979323846;
const double kInvSqrt2 = 1.0 / std::sqrt(2.0);

Mesh channel(int nx, int ny, double lx, double ly) {
    GridSpec g;
    g.lx = lx;
    g.ly = ly;
    g.nx = nx;
    g.ny = ny;
    Mesh m = make_grid_mesh(g);
    set_flags_by_side_id(m, {{1, 500}, {2, 504}, {3, 501}, {4, 503}}); // bottom 500, top 501
    return m;
}

void check_unit(const std::array<double, 2>& n) { CHECK_THAT(std::hypot(n[0], n[1]), WithinAbs(1.0, 1e-14)); }

} // namespace

TEST_CASE("WallGeometry on a channel: faces, normals, first-cell heights", "[wallgeom]") {
    const int nx = 8, ny = 5;
    const double lx = 4.0, ly = 1.25;
    const Mesh m = channel(nx, ny, lx, ly);
    const WallGeometry walls(m);
    const double hx = lx / nx, hy = ly / ny;

    REQUIRE(walls.num_faces() == 2 * nx);
    CHECK_THAT(walls.total_length(), WithinAbs(2.0 * lx, 1e-13));
    for (const WallFace& f : walls.faces()) {
        const bool bottom = m.y[f.node1] == 0.0;
        INFO("face on side " << f.side << (bottom ? " (bottom)" : " (top)"));
        CHECK(is_wall_flag(m.iside[f.side].flag));
        CHECK(f.node1 == m.iside[f.side].node1);
        CHECK(f.node2 == m.iside[f.side].node2);
        CHECK(f.elem == m.iside[f.side].elem);
        CHECK_THAT(f.length, WithinAbs(hx, 1e-14));
        CHECK_THAT(f.yp, WithinAbs(hy, 1e-14));
        CHECK(f.nx == 0.0);
        CHECK(f.ny == (bottom ? 1.0 : -1.0)); // into the fluid
        // The opposite node is one grid line away from the wall.
        CHECK_THAT(std::abs(m.y[f.opposite_node] - m.y[f.node1]), WithinAbs(hy, 1e-14));
        const auto& e = m.intma[f.elem];
        CHECK((e[0] == f.opposite_node || e[1] == f.opposite_node || e[2] == f.opposite_node));
    }

    REQUIRE(walls.num_wall_nodes() == 2 * (nx + 1));
    std::set<int> expected;
    for (int i = 0; i <= nx; ++i) {
        expected.insert(i);                 // bottom row
        expected.insert(ny * (nx + 1) + i); // top row
    }
    CHECK(std::set<int>(walls.wall_nodes().begin(), walls.wall_nodes().end()) == expected);
    CHECK(std::is_sorted(walls.wall_nodes().begin(), walls.wall_nodes().end()));

    for (int iw = 0; iw < walls.num_wall_nodes(); ++iw) {
        const int ip = walls.wall_nodes()[iw];
        CHECK(walls.wall_index(ip) == iw);
        CHECK(walls.is_wall_node(ip));
        const bool bottom = m.y[ip] == 0.0;
        const auto& n = walls.wall_node_normal(iw);
        CHECK(n[0] == 0.0);
        CHECK(n[1] == (bottom ? 1.0 : -1.0));
        CHECK_THAT(walls.wall_node_yp(iw), WithinAbs(hy, 1e-14));
        // End nodes (at the inlet / outlet) touch one wall face, the rest two.
        const bool end = m.x[ip] == 0.0 || m.x[ip] == lx;
        CHECK(walls.wall_node_faces(iw).size() == (end ? 1 : 2));
        for (int f : walls.wall_node_faces(iw)) {
            CHECK((walls.face(f).node1 == ip || walls.face(f).node2 == ip));
        }
    }
    for (int ip = 0; ip < m.npoin; ++ip) {
        if (!expected.count(ip)) CHECK(walls.wall_index(ip) == -1);
    }
}

TEST_CASE("WallGeometry: inward normals do not depend on the side orientation", "[wallgeom]") {
    const StepSpec spec;
    const Mesh m = make_step_mesh(spec);
    Mesh flipped = m;
    for (auto& s : flipped.iside) std::swap(s.node1, s.node2);

    const WallGeometry a(m), b(flipped);
    REQUIRE(a.num_faces() == b.num_faces());
    for (int f = 0; f < a.num_faces(); ++f) {
        CHECK(a.face(f).nx == b.face(f).nx + 0.0); // +0.0: -0.0 == 0.0 anyway
        CHECK(a.face(f).ny == b.face(f).ny + 0.0);
        CHECK(a.face(f).yp == b.face(f).yp);
        CHECK(a.face(f).opposite_node == b.face(f).opposite_node);
    }
    for (int iw = 0; iw < a.num_wall_nodes(); ++iw) {
        CHECK_THAT(a.wall_node_normal(iw)[0], WithinAbs(b.wall_node_normal(iw)[0], 1e-15));
        CHECK_THAT(a.wall_node_normal(iw)[1], WithinAbs(b.wall_node_normal(iw)[1], 1e-15));
    }
}

TEST_CASE("WallGeometry on a backward-facing step: corner normals are bisectors", "[wallgeom]") {
    const StepSpec spec; // uniform square cells: all wall faces have equal length
    const Mesh m = make_step_mesh(spec);
    const WallGeometry walls(m);

    struct Expect {
        double x, y, nx, ny;
        int faces;
    };
    const std::vector<Expect> expected = {
        {spec.xs, 0.0, kInvSqrt2, kInvSqrt2, 2},     // concave corner: step face + bottom
        {spec.xs, spec.hs, kInvSqrt2, kInvSqrt2, 2}, // convex corner: step top + step face
        {0.0, spec.hs, 0.0, 1.0, 1},                 // step top meets the inlet
        {0.0, spec.ly, 0.0, -1.0, 1},                // top wall meets the inlet
        {spec.lx, spec.ly, 0.0, -1.0, 1},            // top wall meets the outlet
        {spec.lx, 0.0, 0.0, 1.0, 1},                 // bottom wall meets the outlet
        {spec.xs, 0.25, 1.0, 0.0, 2},                // on the step face
        {0.5, spec.hs, 0.0, 1.0, 2},                 // on the step top
    };
    for (const Expect& e : expected) {
        const int ip = find_node(m, e.x, e.y);
        REQUIRE(ip >= 0);
        const int iw = walls.wall_index(ip);
        INFO("node at (" << e.x << ", " << e.y << ")");
        REQUIRE(iw >= 0);
        CHECK_THAT(walls.wall_node_normal(iw)[0], WithinAbs(e.nx, 1e-14));
        CHECK_THAT(walls.wall_node_normal(iw)[1], WithinAbs(e.ny, 1e-14));
        CHECK(walls.wall_node_faces(iw).size() == e.faces);
    }

    // Step face: inward normal +x, the adjacent element extends one cell in x.
    for (const WallFace& f : walls.faces()) {
        if (m.x[f.node1] == spec.xs && m.x[f.node2] == spec.xs) {
            CHECK(f.nx == 1.0);
            CHECK_THAT(f.yp, WithinAbs(spec.lx / spec.nx, 1e-14));
        }
    }
}

TEST_CASE("WallGeometry: y_p = 2 * area / length and inward normals on single triangles", "[wallgeom]") {
    for (const Triangle& t : reference_triangles()) {
        Mesh m = make_triangle_mesh(t, 500);
        m.compute_shape_function_derivatives();
        const WallGeometry walls(m);
        INFO(t);
        REQUIRE(walls.num_faces() == 3);
        REQUIRE(walls.num_wall_nodes() == 3);
        const double cx = (t.p[0].x + t.p[1].x + t.p[2].x) / 3.0;
        const double cy = (t.p[0].y + t.p[1].y + t.p[2].y) / 3.0;
        for (const WallFace& f : walls.faces()) {
            CHECK_THAT(f.yp, WithinRel(m.detJ[0] / f.length, 1e-10));
            CHECK_THAT(std::hypot(f.nx, f.ny), WithinAbs(1.0, 1e-14));
            const double mx = 0.5 * (m.x[f.node1] + m.x[f.node2]);
            const double my = 0.5 * (m.y[f.node1] + m.y[f.node2]);
            CHECK((cx - mx) * f.nx + (cy - my) * f.ny > 0.0); // points into the element
            // Perpendicular to the face.
            CHECK_THAT((m.x[f.node2] - m.x[f.node1]) * f.nx + (m.y[f.node2] - m.y[f.node1]) * f.ny,
                       WithinAbs(0.0, 1e-12 * f.length));
        }
        for (int iw = 0; iw < 3; ++iw) check_unit(walls.wall_node_normal(iw));
    }
}

TEST_CASE("WallGeometry: only wall flags produce wall faces", "[wallgeom]") {
    const int flag = GENERATE(0, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508);
    GridSpec g;
    g.nx = 3;
    g.ny = 2;
    const Mesh m = make_grid_mesh(g, flag);
    const WallGeometry walls(m);
    INFO("flag " << flag);
    CHECK(walls.has_walls() == is_wall_flag(flag));
    CHECK(walls.num_faces() == (is_wall_flag(flag) ? m.nboun : 0));
}

TEST_CASE("WallGeometry on the benchmark mesh", "[wallgeom][hinterkante]") {
    Mesh m = benchmark_mesh();
    const WallGeometry walls(m);

    // Side ids 1, 3, 5, 6, 7 are walls: 51 + 51 + 101 + 76 + 101 faces.
    CHECK(walls.num_faces() == 380);
    // Channel walls 2 x 16, plate flanks 2 x 3.5, semicircle (inscribed polygon).
    const double straight = 2.0 * 16.0 + 2.0 * 3.5;
    CHECK(walls.total_length() < straight + 0.5 * kPi + 1e-7);
    CHECK(walls.total_length() > straight + 0.5 * kPi - 3e-4);

    for (const WallFace& f : walls.faces()) {
        const auto& e = m.intma[f.elem];
        const double cx = (m.x[e[0]] + m.x[e[1]] + m.x[e[2]]) / 3.0;
        const double cy = (m.y[e[0]] + m.y[e[1]] + m.y[e[2]]) / 3.0;
        const double mx = 0.5 * (m.x[f.node1] + m.x[f.node2]);
        const double my = 0.5 * (m.y[f.node1] + m.y[f.node2]);
        CHECK((cx - mx) * f.nx + (cy - my) * f.ny > 0.0);
        CHECK(f.yp > 0.0);
    }

    // Same wall node set as the legacy routine; where the legacy (outward)
    // normal is defined, ours is its negative. Ours is never zero on this mesh,
    // including the nodes the legacy routine flags as "trailing edge".
    compute_all_geometry(m);
    REQUIRE(static_cast<int>(m.wall_node_list.size()) == walls.num_wall_nodes());
    int legacy_zero = 0;
    for (std::size_t k = 0; k < m.wall_node_list.size(); ++k) {
        const int ip = m.wall_node_list[k];
        const int iw = walls.wall_index(ip);
        REQUIRE(iw >= 0);
        check_unit(walls.wall_node_normal(iw));
        const auto& legacy = m.wall_node_norm[k];
        if (legacy[0] == 0.0 && legacy[1] == 0.0) {
            ++legacy_zero;
            continue;
        }
        CHECK_THAT(walls.wall_node_normal(iw)[0], WithinAbs(-legacy[0], 1e-12));
        CHECK_THAT(walls.wall_node_normal(iw)[1], WithinAbs(-legacy[1], 1e-12));
    }
    CHECK(legacy_zero >= 1); // documented legacy behaviour (see test_mesh_wall_nodes)
}

TEST_CASE("WallGeometry rejects inconsistent meshes", "[wallgeom]") {
    GridSpec g;
    g.nx = 2;
    g.ny = 2;
    const Mesh good = make_grid_mesh(g, 500);
    REQUIRE_NOTHROW(WallGeometry(good));

    SECTION("side is not an edge of its element") {
        Mesh m = good;
        m.iside[0].elem = m.nelem - 1; // far corner element
        CHECK_THROWS_AS(WallGeometry(m), std::invalid_argument);
    }
    SECTION("element id out of range") {
        Mesh m = good;
        m.iside[1].elem = m.nelem;
        CHECK_THROWS_AS(WallGeometry(m), std::invalid_argument);
    }
    SECTION("node id out of range") {
        Mesh m = good;
        m.iside[2].node2 = -1;
        CHECK_THROWS_AS(WallGeometry(m), std::invalid_argument);
    }
    SECTION("zero-length side") {
        Mesh m = good;
        m.iside[0].node2 = m.iside[0].node1;
        CHECK_THROWS_AS(WallGeometry(m), std::invalid_argument);
    }
    SECTION("degenerate adjacent element") {
        Mesh m = good;
        const int opposite = m.intma[m.iside[0].elem][2]; // element {p00, p10, p11}: p11
        m.y[opposite] = m.y[m.iside[0].node1];             // collapse onto the wall line
        CHECK_THROWS_AS(WallGeometry(m), std::invalid_argument);
    }
    SECTION("array sizes inconsistent with the counters") {
        Mesh m = good;
        m.x.pop_back();
        CHECK_THROWS_AS(WallGeometry(m), std::invalid_argument);
    }
    SECTION("non-wall sides are not validated") {
        Mesh m = good;
        m.iside[0].flag = 504;
        m.iside[0].elem = m.nelem - 1;
        CHECK_NOTHROW(WallGeometry(m));
    }
}
