// Velocity boundary conditions (applied after Steps 1 and 3 in Solver::run(),
// symmetry first):
//
// apply_symmetry_bc():  on every slip/symmetry side (flag 506) remove the
//                       normal velocity component at both nodes, u -= (u.n) n,
//                       using that side's face normal.
// apply_velocity_bcs(): in this order
//   1. flag 507: backward-facing-step inflow profile
//                u = 0.6624 y^6 - 7.5547 y^5 + 33.9 y^4 - 75.283 y^3
//                    + 83.368 y^2 - 37.793 y + 2.6959,  v = 0
//   2. flag 503: uniform inflow u = 1, v = 0
//   3. wall nodes (mesh.wall_node_list, flags 500/501/502): u = v = 0
// so later conditions win at shared corner nodes.

#include "BoundaryConditions.hpp"
#include "IO.hpp"
#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <map>
#include <set>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Velocity {
    std::vector<double> u, v;
};

// Distinct, non-trivial values at every node.
Velocity generic_velocity(const Mesh& m) {
    Velocity vel{sample(m, [](double x, double y) { return 0.7 + std::sin(2.0 * x + y); }),
                 sample(m, [](double x, double y) { return -0.4 + std::cos(x - 1.5 * y); })};
    return vel;
}

Mesh grid_with_flags(const GridSpec& g, const std::vector<SideFlag>& flags) {
    Mesh m = make_grid_mesh(g, 504);
    set_flags_by_side_id(m, flags);
    compute_all_geometry(m);
    return m;
}

// Rotate the whole mesh about the origin (orientation and normals rotate).
Mesh rotated(Mesh m, double alpha) {
    const double c = std::cos(alpha), s = std::sin(alpha);
    for (int i = 0; i < m.npoin; ++i) {
        const double x = m.x[i], y = m.y[i];
        m.x[i] = c * x - s * y;
        m.y[i] = s * x + c * y;
    }
    return m;
}

std::set<int> nodes_with_flag(const Mesh& m, int flag) {
    std::set<int> s;
    for (const auto& side : m.iside) {
        if (side.flag == flag) {
            s.insert(side.node1);
            s.insert(side.node2);
        }
    }
    return s;
}

void check_untouched(const Velocity& before, const Velocity& after, const std::set<int>& changed) {
    for (size_t i = 0; i < before.u.size(); ++i) {
        if (changed.count(static_cast<int>(i))) continue;
        CAPTURE(i);
        CHECK(after.u[i] == before.u[i]);
        CHECK(after.v[i] == before.v[i]);
    }
}

// Backward-facing-step profile, evaluated in Horner form (independent of the
// power sum used in the solver).
double bfs_profile(double y) {
    return ((((((0.6624 * y - 7.5547) * y + 33.9) * y - 75.283) * y + 83.368) * y - 37.793) * y + 2.6959);
}

// Inlet channel of the profile: x in [0, 2], y in [1, 3], hy = 0.5.
GridSpec bfs_grid() {
    GridSpec g;
    g.x0 = 0.0;
    g.y0 = 1.0;
    g.lx = 2.0;
    g.ly = 2.0;
    g.nx = 4;
    g.ny = 4;
    return g;
}

} // namespace

// =============================================================================
// apply_symmetry_bc
// =============================================================================

TEST_CASE("Symmetry condition on a single diagonal side, computed by hand", "[bc][symmetry]") {
    // Side (0,0) -> (1,1): n = (1, -1)/sqrt(2).
    //   (2, 0): u.n = sqrt(2)   -> (2,0) - sqrt(2) n = (1, 1)
    //   (0, 3): u.n = -3/sqrt(2) -> (0,3) + 3/sqrt(2) n = (1.5, 1.5)
    Mesh m = make_mesh({{0, 0}, {1, 1}, {1, 0}}, {{0, 2, 1}}, {{0, 1, 0, 1, 506}});
    m.compute_face_normals();
    std::vector<double> u = {2.0, 0.0, 5.0}, v = {0.0, 3.0, 6.0};
    apply_symmetry_bc(m, u, v);

    CHECK_THAT(u[0], WithinAbs(1.0, 1e-15));
    CHECK_THAT(v[0], WithinAbs(1.0, 1e-15));
    CHECK_THAT(u[1], WithinAbs(1.5, 1e-15));
    CHECK_THAT(v[1], WithinAbs(1.5, 1e-15));
    CHECK(u[2] == 5.0); // node not on the side
    CHECK(v[2] == 6.0);
}

TEST_CASE("Horizontal symmetry line removes v and keeps u", "[bc][symmetry]") {
    GridSpec g;
    g.nx = 4;
    g.ny = 3;
    const Mesh m = grid_with_flags(g, {{1, 506}});
    const Velocity before = generic_velocity(m);
    Velocity vel = before;
    apply_symmetry_bc(m, vel.u, vel.v);

    const auto sym = nodes_with_flag(m, 506);
    REQUIRE(sym.size() == static_cast<size_t>(g.nx + 1));
    for (int i : sym) {
        CAPTURE(i);
        CHECK(vel.u[i] == before.u[i]);
        CHECK_THAT(vel.v[i], WithinAbs(0.0, 1e-16));
    }
    check_untouched(before, vel, sym);
}

TEST_CASE("Inclined symmetry line: zero normal, unchanged tangential velocity", "[bc][symmetry]") {
    const double alpha = GENERATE(kPi / 6.0, 0.9, -2.0);
    CAPTURE(alpha);
    GridSpec g;
    g.nx = 3;
    g.ny = 3;
    Mesh m = rotated(make_grid_mesh(g, 504), alpha);
    set_flags_by_side_id(m, {{1, 506}});
    compute_all_geometry(m);

    const Velocity before = generic_velocity(m);
    Velocity vel = before;
    apply_symmetry_bc(m, vel.u, vel.v);

    // Rotated bottom side: n = R(0,-1), t = R(1,0).
    const double nx = std::sin(alpha), ny = -std::cos(alpha);
    const double tx = std::cos(alpha), ty = std::sin(alpha);
    const auto sym = nodes_with_flag(m, 506);
    for (int i : sym) {
        CAPTURE(i);
        CHECK_THAT(vel.u[i] * nx + vel.v[i] * ny, WithinAbs(0.0, 1e-15));
        CHECK_THAT(vel.u[i] * tx + vel.v[i] * ty,
                   WithinAbs(before.u[i] * tx + before.v[i] * ty, 1e-15));
    }
    check_untouched(before, vel, sym);
}

TEST_CASE("Symmetry condition leaves tangential flow unchanged and is idempotent", "[bc][symmetry]") {
    GridSpec g;
    g.nx = 3;
    g.ny = 2;
    const Mesh m = grid_with_flags(g, {{3, 506}}); // top: n = (0, 1)

    SECTION("tangential flow") {
        Velocity vel{std::vector<double>(m.npoin, 2.5), std::vector<double>(m.npoin, 0.0)};
        const Velocity before = vel;
        apply_symmetry_bc(m, vel.u, vel.v);
        CHECK(vel.u == before.u);
        CHECK(vel.v == before.v);
    }
    SECTION("applying twice equals applying once") {
        Velocity once = generic_velocity(m);
        apply_symmetry_bc(m, once.u, once.v);
        Velocity twice = once;
        apply_symmetry_bc(m, twice.u, twice.v);
        for (int i = 0; i < m.npoin; ++i) {
            CAPTURE(i);
            CHECK_THAT(twice.u[i], WithinAbs(once.u[i], 1e-16));
            CHECK_THAT(twice.v[i], WithinAbs(once.v[i], 1e-16));
        }
    }
}

TEST_CASE("Only flag 506 sides are treated as symmetry boundaries", "[bc][symmetry]") {
    const int flag = GENERATE(500, 501, 502, 503, 504, 507);
    CAPTURE(flag);
    GridSpec g;
    g.nx = 3;
    g.ny = 3;
    Mesh m = make_grid_mesh(g, flag);
    compute_all_geometry(m);
    const Velocity before = generic_velocity(m);
    Velocity vel = before;
    apply_symmetry_bc(m, vel.u, vel.v);
    CHECK(vel.u == before.u);
    CHECK(vel.v == before.v);
}

TEST_CASE("Corner between perpendicular symmetry lines is a stagnation point", "[bc][symmetry]") {
    // Bottom (n = (0,-1)) and right (n = (1,0)) symmetry sides meet at (nx, 0).
    GridSpec g;
    g.nx = 3;
    g.ny = 3;
    const Mesh m = grid_with_flags(g, {{1, 506}, {2, 506}});
    Velocity vel = generic_velocity(m);
    apply_symmetry_bc(m, vel.u, vel.v);

    const int corner = g.node(g.nx, 0);
    CHECK_THAT(vel.u[corner], WithinAbs(0.0, 1e-16));
    CHECK_THAT(vel.v[corner], WithinAbs(0.0, 1e-16));
    CHECK_THAT(vel.v[g.node(1, 0)], WithinAbs(0.0, 1e-16)); // bottom: v = 0
    CHECK_THAT(vel.u[g.node(g.nx, 1)], WithinAbs(0.0, 1e-16)); // right: u = 0
    CHECK(vel.u[g.node(1, 0)] != 0.0);
    CHECK(vel.v[g.node(g.nx, 1)] != 0.0);
}

TEST_CASE("Kinked symmetry boundary: projections are applied side by side",
          "[bc][symmetry][characterization]") {
    // Documents current behaviour: at a node shared by two symmetry sides with
    // different normals, the side normals are projected out one after the
    // other, so the result is tangent only to the LAST processed side and
    // depends on the order of iside. (Perpendicular corners and straight
    // symmetry lines are unaffected.)
    //
    // A(0,0) -> B(1,0): n1 = (0,-1);  B -> C(2,0.5): n2 = (1,-2)/sqrt(5); u_B = (1,1)
    //   n1 first: (1,1) -> (1,0) -> (0.8, 0.4)    (u.n1 = -0.4 != 0)
    //   n2 first: (1,1) -> (1.2,0.6) -> (1.2, 0)  (u.n2 = 0.537 != 0)
    const std::vector<Point> nodes = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.5}};
    const BoundarySide ab{0, 1, 0, 1, 506}, bc{1, 2, 0, 2, 506};

    SECTION("side A-B processed first") {
        Mesh m = make_mesh(nodes, {}, {ab, bc});
        m.compute_face_normals();
        std::vector<double> u = {0.0, 1.0, 0.0}, v = {0.0, 1.0, 0.0};
        apply_symmetry_bc(m, u, v);
        CHECK_THAT(u[1], WithinAbs(0.8, 1e-15));
        CHECK_THAT(v[1], WithinAbs(0.4, 1e-15));
    }
    SECTION("side B-C processed first") {
        Mesh m = make_mesh(nodes, {}, {bc, ab});
        m.compute_face_normals();
        std::vector<double> u = {0.0, 1.0, 0.0}, v = {0.0, 1.0, 0.0};
        apply_symmetry_bc(m, u, v);
        CHECK_THAT(u[1], WithinAbs(1.2, 1e-15));
        CHECK_THAT(v[1], WithinAbs(0.0, 1e-15));
    }
}

// =============================================================================
// apply_velocity_bcs
// =============================================================================

TEST_CASE("Uniform inflow (503) prescribes u = 1, v = 0", "[bc][velocity_bc]") {
    GridSpec g;
    g.nx = 4;
    g.ny = 3;
    const Mesh m = grid_with_flags(g, {{4, 503}}); // left inlet, rest outlet
    REQUIRE(m.wall_node_list.empty());
    const Velocity before = generic_velocity(m);
    Velocity vel = before;
    apply_velocity_bcs(m, vel.u, vel.v);

    const auto inlet = nodes_with_flag(m, 503);
    REQUIRE(inlet.size() == static_cast<size_t>(g.ny + 1));
    for (int i : inlet) {
        CAPTURE(i);
        CHECK(vel.u[i] == 1.0);
        CHECK(vel.v[i] == 0.0);
    }
    check_untouched(before, vel, inlet);
}

TEST_CASE("Backward-facing-step inflow profile (507)", "[bc][velocity_bc]") {
    // Exact values of the polynomial (decimal coefficients, rational arithmetic):
    //   u(1.5) = 83179/64000, u(2) = 13611/10000, u(2.5) = 364913/320000
    const GridSpec g = bfs_grid();
    const Mesh m = grid_with_flags(g, {{4, 507}});
    const Velocity before = generic_velocity(m);
    Velocity vel = before;
    apply_velocity_bcs(m, vel.u, vel.v);

    CHECK_THAT(vel.u[g.node(0, 1)], WithinRel(83179.0 / 64000.0, 1e-12));   // y = 1.5
    CHECK_THAT(vel.u[g.node(0, 2)], WithinRel(13611.0 / 10000.0, 1e-12));   // y = 2
    CHECK_THAT(vel.u[g.node(0, 3)], WithinRel(364913.0 / 320000.0, 1e-12)); // y = 2.5

    const auto inlet = nodes_with_flag(m, 507);
    for (int i : inlet) {
        CAPTURE(i, m.y[i]);
        CHECK_THAT(vel.u[i], WithinAbs(bfs_profile(m.y[i]), 1e-11));
        CHECK(vel.v[i] == 0.0);
    }
    check_untouched(before, vel, inlet);
}

TEST_CASE("Backward-facing-step profile depends only on y", "[bc][velocity_bc]") {
    // The same profile is imposed on an inlet placed at any x.
    const GridSpec g = bfs_grid();
    const Mesh m = grid_with_flags(g, {{2, 507}}); // right side x = 2 as inlet
    Velocity vel = generic_velocity(m);
    apply_velocity_bcs(m, vel.u, vel.v);
    for (int j = 0; j <= g.ny; ++j) {
        const int i = g.node(g.nx, j);
        CAPTURE(j, m.y[i]);
        CHECK_THAT(vel.u[i], WithinAbs(bfs_profile(m.y[i]), 1e-11));
    }
}

TEST_CASE("Backward-facing-step profile is slightly negative at the channel walls",
          "[bc][velocity_bc][characterization]") {
    // Documents current behaviour: the fitted polynomial is meant for an inlet
    // channel y in [1, 3] but is not zero at its walls: u(1) = -0.0044 and
    // u(3) = -0.0146 (weak backflow; roots at y = 1.0009 and y = 2.9950).
    // With wall sides at y = 1 and y = 3 the wall condition overrides these
    // corner values; without them the negative values are imposed.
    const GridSpec g = bfs_grid();

    SECTION("inlet without adjacent walls") {
        const Mesh m = grid_with_flags(g, {{4, 507}});
        Velocity vel = generic_velocity(m);
        apply_velocity_bcs(m, vel.u, vel.v);
        CHECK_THAT(vel.u[g.node(0, 0)], WithinAbs(-0.0044, 1e-11));    // y = 1
        CHECK_THAT(vel.u[g.node(0, g.ny)], WithinAbs(-0.0146, 1e-11)); // y = 3
    }
    SECTION("walls at y = 1 and y = 3 override the corner values") {
        const Mesh m = grid_with_flags(g, {{1, 500}, {3, 500}, {4, 507}});
        Velocity vel = generic_velocity(m);
        apply_velocity_bcs(m, vel.u, vel.v);
        CHECK(vel.u[g.node(0, 0)] == 0.0);
        CHECK(vel.u[g.node(0, g.ny)] == 0.0);
    }
}

TEST_CASE("Wall nodes get a no-slip condition", "[bc][velocity_bc]") {
    const int wall_flag = GENERATE(500, 501, 502);
    CAPTURE(wall_flag);
    GridSpec g;
    g.nx = 4;
    g.ny = 3;
    const Mesh m = grid_with_flags(g, {{1, wall_flag}, {3, wall_flag}});
    const Velocity before = generic_velocity(m);
    Velocity vel = before;
    apply_velocity_bcs(m, vel.u, vel.v);

    const std::set<int> walls(m.wall_node_list.begin(), m.wall_node_list.end());
    REQUIRE(walls.size() == static_cast<size_t>(2 * (g.nx + 1)));
    for (int i : walls) {
        CAPTURE(i);
        CHECK(vel.u[i] == 0.0);
        CHECK(vel.v[i] == 0.0);
    }
    check_untouched(before, vel, walls);
}

TEST_CASE("No-slip nodes come from wall_node_list, not from the side flags", "[bc][velocity_bc]") {
    // apply_velocity_bcs() relies on Mesh::determine_wall_nodes() having been
    // called: it zeroes exactly the nodes in wall_node_list.
    GridSpec g;
    g.nx = 3;
    g.ny = 3;
    Mesh m = make_grid_mesh(g, 500); // wall flags everywhere ...
    m.compute_face_normals();        // ... but determine_wall_nodes() not called

    SECTION("empty wall list: nothing is changed") {
        const Velocity before = generic_velocity(m);
        Velocity vel = before;
        apply_velocity_bcs(m, vel.u, vel.v);
        CHECK(vel.u == before.u);
        CHECK(vel.v == before.v);
    }
    SECTION("explicit wall list: exactly those nodes are zeroed") {
        m.wall_node_list = {g.node(1, 1), g.node(2, 2)};
        const Velocity before = generic_velocity(m);
        Velocity vel = before;
        apply_velocity_bcs(m, vel.u, vel.v);
        for (int i : m.wall_node_list) {
            CHECK(vel.u[i] == 0.0);
            CHECK(vel.v[i] == 0.0);
        }
        check_untouched(before, vel, {g.node(1, 1), g.node(2, 2)});
    }
}

TEST_CASE("Precedence at shared corner nodes: 507 < 503 < wall", "[bc][velocity_bc]") {
    // left 507, top 503, bottom wall, right outlet:
    //   (0, ny):  507 + 503  -> inflow (1, 0)
    //   (0, 0):   507 + wall -> (0, 0)
    //   (nx, ny): 503 + 504  -> inflow (1, 0)
    //   (nx, 0):  wall + 504 -> (0, 0)
    const GridSpec g = bfs_grid();
    const Mesh m = grid_with_flags(g, {{1, 500}, {2, 504}, {3, 503}, {4, 507}});
    Velocity vel = generic_velocity(m);
    apply_velocity_bcs(m, vel.u, vel.v);

    CHECK(vel.u[g.node(0, g.ny)] == 1.0);
    CHECK(vel.v[g.node(0, g.ny)] == 0.0);
    CHECK(vel.u[g.node(0, 0)] == 0.0);
    CHECK(vel.v[g.node(0, 0)] == 0.0);
    CHECK(vel.u[g.node(g.nx, g.ny)] == 1.0);
    CHECK(vel.v[g.node(g.nx, g.ny)] == 0.0);
    CHECK(vel.u[g.node(g.nx, 0)] == 0.0);
    CHECK(vel.v[g.node(g.nx, 0)] == 0.0);
    // Interior of the 507 side keeps the profile.
    CHECK_THAT(vel.u[g.node(0, 2)], WithinAbs(bfs_profile(2.0), 1e-11));
}

TEST_CASE("Outlet and symmetry sides are not touched by apply_velocity_bcs", "[bc][velocity_bc]") {
    const int flag = GENERATE(504, 506);
    CAPTURE(flag);
    GridSpec g;
    g.nx = 3;
    g.ny = 3;
    Mesh m = make_grid_mesh(g, flag);
    compute_all_geometry(m);
    const Velocity before = generic_velocity(m);
    Velocity vel = before;
    apply_velocity_bcs(m, vel.u, vel.v);
    CHECK(vel.u == before.u);
    CHECK(vel.v == before.v);
}

TEST_CASE("apply_velocity_bcs is idempotent", "[bc][velocity_bc]") {
    const GridSpec g = bfs_grid();
    const Mesh m = grid_with_flags(g, {{1, 500}, {2, 504}, {3, 503}, {4, 507}});
    Velocity once = generic_velocity(m);
    apply_velocity_bcs(m, once.u, once.v);
    Velocity twice = once;
    apply_velocity_bcs(m, twice.u, twice.v);
    CHECK(twice.u == once.u);
    CHECK(twice.v == once.v);
}

TEST_CASE("Boundary conditions in solver order: symmetry, then velocity", "[bc][symmetry][velocity_bc]") {
    // bottom symmetry, left inflow, top wall, right outlet.
    GridSpec g;
    g.nx = 4;
    g.ny = 3;
    const Mesh m = grid_with_flags(g, {{1, 506}, {2, 504}, {3, 500}, {4, 503}});
    const Velocity before = generic_velocity(m);
    Velocity vel = before;
    apply_symmetry_bc(m, vel.u, vel.v);
    apply_velocity_bcs(m, vel.u, vel.v);

    for (int i = 1; i < g.nx; ++i) { // symmetry interior: v = 0, u kept
        const int p = g.node(i, 0);
        CAPTURE(i);
        CHECK(vel.u[p] == before.u[p]);
        CHECK(vel.v[p] == 0.0);
    }
    CHECK(vel.u[g.node(0, 0)] == 1.0); // symmetry + inflow -> inflow
    CHECK(vel.v[g.node(0, 0)] == 0.0);
    CHECK(vel.u[g.node(g.nx, 0)] == before.u[g.node(g.nx, 0)]); // symmetry + outlet
    CHECK(vel.v[g.node(g.nx, 0)] == 0.0);
    CHECK(vel.u[g.node(0, g.ny)] == 0.0); // inflow + wall -> wall
    CHECK(vel.v[g.node(0, g.ny)] == 0.0);
    for (int j = 1; j < g.ny; ++j) { // outlet interior untouched
        const int p = g.node(g.nx, j);
        CHECK(vel.u[p] == before.u[p]);
        CHECK(vel.v[p] == before.v[p]);
    }
}

TEST_CASE("Velocity boundary conditions on the Hinterkante mesh", "[bc][velocity_bc][hinterkante]") {
    // Inlet: side ids 4 and 8 (flag 503) at x = -4, 52 nodes each; both ends of
    // each inlet segment are wall nodes (domain corners, plate corners), so
    // 2 * 50 nodes get u = 1. Walls: 383 nodes. No symmetry sides.
    const std::string dir = HYDRAMA2D_CASE_DIR;
    Mesh m;
    REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", m));
    REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", m));
    compute_all_geometry(m);

    const Velocity before = generic_velocity(m);
    Velocity vel = before;
    apply_symmetry_bc(m, vel.u, vel.v);
    CHECK(vel.u == before.u); // no flag 506 in this case
    CHECK(vel.v == before.v);
    apply_velocity_bcs(m, vel.u, vel.v);

    const std::set<int> walls(m.wall_node_list.begin(), m.wall_node_list.end());
    const auto inlet = nodes_with_flag(m, 503);
    REQUIRE(inlet.size() == 104);
    int inflow_nodes = 0;
    for (int i : inlet) {
        CAPTURE(i, m.x[i], m.y[i]);
        CHECK(m.x[i] == -4.0);
        if (walls.count(i)) continue;
        ++inflow_nodes;
        CHECK(vel.u[i] == 1.0);
        CHECK(vel.v[i] == 0.0);
    }
    CHECK(inflow_nodes == 100);
    for (double y : {-5.0, -0.5, 0.5, 5.0}) { // inlet ends are no-slip
        const int c = find_node(m, -4.0, y);
        CAPTURE(y);
        REQUIRE(c >= 0);
        CHECK(vel.u[c] == 0.0);
        CHECK(vel.v[c] == 0.0);
    }
    int nonzero_walls = 0;
    for (int i : walls) {
        if (vel.u[i] != 0.0 || vel.v[i] != 0.0) ++nonzero_walls;
    }
    CHECK(walls.size() == 383);
    CHECK(nonzero_walls == 0);

    std::set<int> changed = walls;
    changed.insert(inlet.begin(), inlet.end());
    check_untouched(before, vel, changed);
}
