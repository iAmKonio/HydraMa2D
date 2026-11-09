// Wall nodes: Mesh::determine_wall_nodes() collects the nodes of all wall
// sides (flags 500, 501, 502) and computes for each of them
//   * wall_node_list : unique node ids, in order of first appearance in iside,
//   * wall_node_norm : length-weighted average of the adjacent face normals,
//                      normalised to unit length,
//   * trailing_edge_node : a node where the adjacent wall faces meet at a sharp
//                      angle; its wall normal is set to zero.
// The trailing edge test uses rtemp = component-wise product of the adjacent
// face normals; for a node with two wall faces rtemp.x + rtemp.y = n1 . n2, and
// the node is flagged when n1 . n2 < -0.2 (angle between normals > ~101.5 deg).
// Requires face_norm, i.e. compute_face_normals().
//
// wall_node_list drives the no-slip condition in apply_velocity_bcs();
// wall_node_norm and trailing_edge_node are currently not used by the solver.

#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <map>
#include <set>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;

namespace {

constexpr double kPi = 3.14159265358979323846;

bool is_wall_flag(int flag) { return flag == 500 || flag == 501 || flag == 502; }

// Expected wall normal at every wall node, computed directly from the node
// coordinates: normalise( sum over adjacent wall sides of (dy, -dx) ).
std::map<int, std::array<double, 2>> expected_wall_normals(const Mesh& m) {
    std::map<int, std::array<double, 2>> sum;
    for (const auto& s : m.iside) {
        if (!is_wall_flag(s.flag)) continue;
        const double dx = m.x[s.node2] - m.x[s.node1];
        const double dy = m.y[s.node2] - m.y[s.node1];
        for (int ip : {s.node1, s.node2}) {
            sum[ip][0] += dy;
            sum[ip][1] -= dx;
        }
    }
    for (auto& [ip, n] : sum) {
        const double len = std::hypot(n[0], n[1]);
        n = {n[0] / len, n[1] / len};
    }
    return sum;
}

// Wall nodes in order of first appearance in iside.
std::vector<int> expected_wall_list(const Mesh& m) {
    std::vector<int> list;
    std::set<int> seen;
    for (const auto& s : m.iside) {
        if (!is_wall_flag(s.flag)) continue;
        for (int ip : {s.node1, s.node2}) {
            if (seen.insert(ip).second) list.push_back(ip);
        }
    }
    return list;
}

std::array<double, 2> wall_normal_of(const Mesh& m, int node) {
    for (size_t k = 0; k < m.wall_node_list.size(); ++k) {
        if (m.wall_node_list[k] == node) return m.wall_node_norm[k];
    }
    FAIL("node " << node << " is not a wall node");
    return {0.0, 0.0};
}

Mesh walled_grid(const GridSpec& g, int flag = 500) {
    Mesh m = make_grid_mesh(g, flag);
    m.compute_face_normals();
    return m;
}

// Closed wall loop around a body with a blunt front and a wedge-shaped tail:
//
//        B ------------- C
//        |                 \      tip angle at T
//        |                   T
//        |                 /
//        A ------------- D
//
// A = (0,-h), B = (0,h), C = (lb,h), T = (lb + h/tan(phi), 0), D = (lb,-h),
// with phi = tip_angle / 2. The fluid is outside, so the loop A-B-C-T-D is
// traversed clockwise and the face normals point into the body. Only the
// boundary is needed by determine_wall_nodes(), so the mesh has no elements.
struct WedgeBody {
    Mesh mesh;
    int A = 0, B = 1, C = 2, T = 3, D = 4;
};

WedgeBody wedge_body(double tip_angle_deg, double h = 0.1, double lb = 0.5) {
    const double phi = 0.5 * tip_angle_deg * kPi / 180.0;
    WedgeBody w;
    w.mesh = make_mesh({{0.0, -h}, {0.0, h}, {lb, h}, {lb + h / std::tan(phi), 0.0}, {lb, -h}}, {},
                       {{0, 1, 0, 1, 500}, {1, 2, 0, 1, 500}, {2, 3, 0, 1, 500},
                        {3, 4, 0, 1, 500}, {4, 0, 0, 1, 500}});
    w.mesh.compute_face_normals();
    return w;
}

} // namespace

TEST_CASE("Wall node list contains every wall node once, in order of first appearance", "[mesh][wall]") {
    GridSpec g;
    g.nx = 4;
    g.ny = 3;
    Mesh m = walled_grid(g);
    m.determine_wall_nodes();

    CHECK(static_cast<int>(m.wall_node_list.size()) == g.nboun()); // closed loop: nodes == sides
    CHECK(m.wall_node_list == expected_wall_list(m));
    CHECK(m.wall_node_norm.size() == m.wall_node_list.size());
    CHECK(std::set<int>(m.wall_node_list.begin(), m.wall_node_list.end()).size() ==
          m.wall_node_list.size());
}

TEST_CASE("Wall flags 500, 501 and 502 are walls; other boundary flags are not", "[mesh][wall]") {
    GridSpec g;
    g.nx = 3;
    g.ny = 2;

    SECTION("wall flags") {
        const int flag = GENERATE(500, 501, 502);
        CAPTURE(flag);
        Mesh m = walled_grid(g, flag);
        m.determine_wall_nodes();
        CHECK(static_cast<int>(m.wall_node_list.size()) == g.nboun());
    }
    SECTION("non-wall flags") {
        const int flag = GENERATE(503, 504, 506, 507);
        CAPTURE(flag);
        Mesh m = walled_grid(g, flag);
        m.determine_wall_nodes();
        CHECK(m.wall_node_list.empty());
        CHECK(m.wall_node_norm.empty());
        CHECK(m.trailing_edge_node == -1);
    }
}

TEST_CASE("Wall nodes shared with non-wall boundaries are included", "[mesh][wall]") {
    // Channel: bottom (1) and top (3) walls, outlet (2) and inlet (4).
    // The corner nodes also belong to the inlet/outlet but must be wall nodes
    // (apply_velocity_bcs applies the no-slip condition last at corners).
    GridSpec g;
    g.nx = 5;
    g.ny = 3;
    Mesh m = walled_grid(g);
    set_flags_by_side_id(m, {{1, 500}, {2, 504}, {3, 501}, {4, 503}});
    m.determine_wall_nodes();

    std::set<int> expected;
    for (int i = 0; i <= g.nx; ++i) {
        expected.insert(g.node(i, 0));
        expected.insert(g.node(i, g.ny));
    }
    CHECK(std::set<int>(m.wall_node_list.begin(), m.wall_node_list.end()) == expected);
    CHECK(m.wall_node_list.size() == expected.size());
    CHECK(m.wall_node_list == expected_wall_list(m));
}

TEST_CASE("Wall normals on a closed rectangular wall", "[mesh][wall]") {
    // hx != hy so that the corner normals reveal the length weighting.
    GridSpec g;
    g.x0 = 1.0;
    g.y0 = -1.0;
    g.lx = 2.0;
    g.ly = 0.75;
    g.nx = 4;
    g.ny = 3;
    Mesh m = walled_grid(g);
    m.determine_wall_nodes();

    SECTION("no trailing edge on a box: corner normals are perpendicular") {
        CHECK(m.trailing_edge_node == -1);
    }
    SECTION("wall normals are the normalised length-weighted face normals") {
        const auto expected = expected_wall_normals(m);
        for (size_t k = 0; k < m.wall_node_list.size(); ++k) {
            const int ip = m.wall_node_list[k];
            CAPTURE(ip, m.x[ip], m.y[ip]);
            CHECK_THAT(m.wall_node_norm[k][0], WithinAbs(expected.at(ip)[0], 1e-14));
            CHECK_THAT(m.wall_node_norm[k][1], WithinAbs(expected.at(ip)[1], 1e-14));
        }
    }
    SECTION("straight-wall nodes take the face normal") {
        auto n = wall_normal_of(m, g.node(2, 0)); // bottom
        CHECK_THAT(n[0], WithinAbs(0.0, 1e-15));
        CHECK_THAT(n[1], WithinAbs(-1.0, 1e-15));
        n = wall_normal_of(m, g.node(g.nx, 1)); // right
        CHECK_THAT(n[0], WithinAbs(1.0, 1e-15));
        CHECK_THAT(n[1], WithinAbs(0.0, 1e-15));
        n = wall_normal_of(m, g.node(1, g.ny)); // top
        CHECK_THAT(n[0], WithinAbs(0.0, 1e-15));
        CHECK_THAT(n[1], WithinAbs(1.0, 1e-15));
        n = wall_normal_of(m, g.node(0, 2)); // left
        CHECK_THAT(n[0], WithinAbs(-1.0, 1e-15));
        CHECK_THAT(n[1], WithinAbs(0.0, 1e-15));
    }
    SECTION("corner normals are weighted by the adjacent face lengths") {
        // Bottom-right corner: hx * (0,-1) + hy * (1,0), normalised.
        const double hx = g.hx(), hy = g.hy(), len = std::hypot(hx, hy);
        const auto n = wall_normal_of(m, g.node(g.nx, 0));
        CHECK_THAT(n[0], WithinAbs(hy / len, 1e-15));
        CHECK_THAT(n[1], WithinAbs(-hx / len, 1e-15));
    }
}

TEST_CASE("Sharp trailing edge of a closed wall loop is detected", "[mesh][wall]") {
    // At the tip, n1 . n2 = -cos(tip angle): flagged for tip angles < ~78.5 deg.
    const double tip_angle = GENERATE(10.0, 30.0, 60.0, 75.0);
    CAPTURE(tip_angle);
    WedgeBody w = wedge_body(tip_angle);
    w.mesh.determine_wall_nodes();
    const Mesh& m = w.mesh;

    REQUIRE(m.wall_node_list.size() == 5);
    CHECK(m.trailing_edge_node == w.T);

    const auto tip = wall_normal_of(m, w.T);
    CHECK(tip[0] == 0.0);
    CHECK(tip[1] == 0.0);

    // All other wall normals are unit vectors equal to the weighted average.
    const auto expected = expected_wall_normals(m);
    for (int ip : {w.A, w.B, w.C, w.D}) {
        CAPTURE(ip);
        const auto n = wall_normal_of(m, ip);
        CHECK_THAT(n[0], WithinAbs(expected.at(ip)[0], 1e-14));
        CHECK_THAT(n[1], WithinAbs(expected.at(ip)[1], 1e-14));
    }
}

TEST_CASE("Blunt tail is not detected as a trailing edge", "[mesh][wall]") {
    const double tip_angle = GENERATE(85.0, 100.0, 130.0, 160.0);
    CAPTURE(tip_angle);
    WedgeBody w = wedge_body(tip_angle);
    w.mesh.determine_wall_nodes();

    CHECK(w.mesh.trailing_edge_node == -1);
    const auto n = wall_normal_of(w.mesh, w.T); // symmetric tip: normal points along -x
    CHECK_THAT(n[0], WithinAbs(-1.0, 1e-14));
    CHECK_THAT(n[1], WithinAbs(0.0, 1e-14));
}

TEST_CASE("Wall normals of a body point into the body (out of the fluid)", "[mesh][wall]") {
    WedgeBody w = wedge_body(30.0, 0.1, 0.5);
    w.mesh.determine_wall_nodes();

    // Front corner B: faces A->B (length 2h, normal (1,0)) and B->C
    // (length lb, normal (0,-1)): normalise(2h, -lb) = (0.2, -0.5) / |.|
    const auto nb = wall_normal_of(w.mesh, w.B);
    const double len = std::hypot(0.2, 0.5);
    CHECK_THAT(nb[0], WithinAbs(0.2 / len, 1e-14));
    CHECK_THAT(nb[1], WithinAbs(-0.5 / len, 1e-14));
}

TEST_CASE("Recomputing wall nodes does not accumulate results", "[mesh][wall]") {
    GridSpec g;
    g.nx = 3;
    g.ny = 3;
    Mesh m = walled_grid(g);
    m.determine_wall_nodes();
    const auto list = m.wall_node_list;
    const auto norms = m.wall_node_norm;
    const int te = m.trailing_edge_node;

    m.determine_wall_nodes();
    CHECK(m.wall_node_list == list);
    CHECK(m.wall_node_norm == norms);
    CHECK(m.trailing_edge_node == te);

    SECTION("and reflects changed boundary flags") {
        set_flags_by_side_id(m, {{1, 503}, {2, 504}, {3, 503}, {4, 503}});
        m.determine_wall_nodes();
        CHECK(m.wall_node_list.empty());
        CHECK(m.trailing_edge_node == -1);
    }
}

TEST_CASE("Trailing edge is cleared when it no longer exists", "[mesh][wall]") {
    WedgeBody w = wedge_body(30.0);
    Mesh& m = w.mesh;
    m.determine_wall_nodes();
    REQUIRE(m.trailing_edge_node == w.T);

    SECTION("tip blunted") {
        m.x[w.T] = 0.5 + 0.1 / std::tan(0.5 * 120.0 * kPi / 180.0); // 120 deg tip
        m.compute_face_normals();
        m.determine_wall_nodes();
        CHECK(m.trailing_edge_node == -1);
        const auto n = wall_normal_of(m, w.T);
        CHECK_THAT(n[0], WithinAbs(-1.0, 1e-14));
        CHECK_THAT(n[1], WithinAbs(0.0, 1e-14));
    }
    SECTION("body no longer a wall") {
        for (auto& s : m.iside) s.flag = 503;
        m.determine_wall_nodes();
        CHECK(m.trailing_edge_node == -1);
        CHECK(m.wall_node_list.empty());
    }
}

TEST_CASE("Endpoints of open wall segments are classified by face direction",
          "[mesh][wall][characterization]") {
    // Documents current behaviour (suspected defect, pending a decision):
    // an endpoint of an open wall segment has only ONE wall face, so rtemp is
    // that face normal itself and the test becomes nx + ny < -0.2. Whether
    // such an endpoint is flagged as "trailing edge" (and gets a zero wall
    // normal) therefore depends only on which way the wall faces.
    // On the benchmark mesh this flags domain corners (see
    // test_mesh_hinterkante.cpp). wall_node_norm / trailing_edge_node are
    // currently unused by the solver, so results are not affected today.
    struct Case { int wall_side; bool flagged; };
    // bottom n=(0,-1), right n=(1,0), top n=(0,1), left n=(-1,0)
    const Case c = GENERATE(Case{1, true}, Case{2, false}, Case{3, false}, Case{4, true});
    CAPTURE(c.wall_side, c.flagged);

    GridSpec g;
    g.nx = 3;
    g.ny = 2;
    Mesh m = walled_grid(g, 503);
    set_flags_by_side_id(m, {{c.wall_side, 500}});
    m.determine_wall_nodes();

    // The two segment endpoints are the first and last wall nodes.
    REQUIRE(m.wall_node_list.size() >= 2);
    const int first = m.wall_node_list.front();
    const int last = m.wall_node_list.back();
    const auto n_first = wall_normal_of(m, first);
    const auto n_last = wall_normal_of(m, last);

    if (c.flagged) {
        CHECK(m.trailing_edge_node == last); // the last flagged node wins
        CHECK(n_first == std::array<double, 2>{0.0, 0.0});
        CHECK(n_last == std::array<double, 2>{0.0, 0.0});
    } else {
        CHECK(m.trailing_edge_node == -1);
        CHECK_THAT(std::hypot(n_first[0], n_first[1]), WithinAbs(1.0, 1e-14));
        CHECK_THAT(std::hypot(n_last[0], n_last[1]), WithinAbs(1.0, 1e-14));
    }
}
