// Pressure boundary conditions: setup_pressure_bcs() returns the Dirichlet
// nodes and values used by the pressure solve (solve_pcg):
//   * every node of a pressure-outlet side (flag 504), once, in order of first
//     appearance in iside, with p = 0;
//   * if there is no outlet, a single reference node (default_pnode, or node 0
//     if default_pnode is out of range) with p = 0, so that the otherwise
//     singular pressure operator becomes definite.

#include "BoundaryConditions.hpp"
#include "IO.hpp"
#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <set>

using namespace hydrama;
using namespace hydrama::test;

namespace {

// Grid with side ids 1=bottom, 2=right, 3=top, 4=left.
GridSpec bc_grid() {
    GridSpec g;
    g.nx = 4;
    g.ny = 3;
    return g;
}

Mesh flagged_grid(const std::vector<SideFlag>& flags) {
    Mesh m = make_grid_mesh(bc_grid(), 500);
    set_flags_by_side_id(m, flags);
    return m;
}

void check_all_zero(const PressureBC& bc) {
    REQUIRE(bc.values.size() == bc.nodes.size());
    for (double v : bc.values) CHECK(v == 0.0);
}

} // namespace

TEST_CASE("Outlet nodes become zero-pressure Dirichlet nodes", "[bc][pressure_bc]") {
    const GridSpec g = bc_grid();
    const Mesh m = flagged_grid({{1, 500}, {2, 504}, {3, 500}, {4, 503}});
    const PressureBC bc = setup_pressure_bcs(m, 0);

    // Right side, traversed bottom -> top.
    std::vector<int> expected;
    for (int j = 0; j <= g.ny; ++j) expected.push_back(g.node(g.nx, j));
    CHECK(bc.nodes == expected);
    check_all_zero(bc);
}

TEST_CASE("Nodes shared by several outlet sides are listed once", "[bc][pressure_bc]") {
    // Outlets on the right (bottom -> top) and top (right -> left): the corner
    // (nx, ny) belongs to both.
    const GridSpec g = bc_grid();
    const Mesh m = flagged_grid({{1, 500}, {2, 504}, {3, 504}, {4, 503}});
    const PressureBC bc = setup_pressure_bcs(m, 0);

    std::vector<int> expected;
    for (int j = 0; j <= g.ny; ++j) expected.push_back(g.node(g.nx, j));
    for (int i = g.nx - 1; i >= 0; --i) expected.push_back(g.node(i, g.ny));
    CHECK(bc.nodes == expected);
    CHECK(std::set<int>(bc.nodes.begin(), bc.nodes.end()).size() == bc.nodes.size());
    check_all_zero(bc);
}

TEST_CASE("Only flag 504 creates pressure Dirichlet nodes", "[bc][pressure_bc]") {
    const int other = GENERATE(500, 501, 502, 503, 506, 507);
    CAPTURE(other);
    const GridSpec g = bc_grid();
    const Mesh m = flagged_grid({{1, other}, {2, 504}, {3, other}, {4, other}});
    const PressureBC bc = setup_pressure_bcs(m, 7);

    CHECK(bc.nodes.size() == static_cast<size_t>(g.ny + 1));
    for (int ip : bc.nodes) {
        CAPTURE(ip);
        CHECK(m.x[ip] == 1.0); // right side of the unit square
    }
}

TEST_CASE("Reference pressure node is ignored when an outlet exists", "[bc][pressure_bc]") {
    const GridSpec g = bc_grid();
    const Mesh m = flagged_grid({{2, 504}});
    const int pnode = g.node(1, 1); // interior node
    const PressureBC bc = setup_pressure_bcs(m, pnode);
    CHECK(std::find(bc.nodes.begin(), bc.nodes.end(), pnode) == bc.nodes.end());
}

TEST_CASE("Without an outlet the reference node is used", "[bc][pressure_bc]") {
    const Mesh m = flagged_grid({{1, 500}, {2, 506}, {3, 500}, {4, 503}});
    const int pnode = GENERATE(0, 6, 19); // npoin = 20
    CAPTURE(pnode);
    const PressureBC bc = setup_pressure_bcs(m, pnode);

    CHECK(bc.nodes == std::vector<int>{pnode});
    CHECK(bc.values == std::vector<double>{0.0});
}

TEST_CASE("Out-of-range reference node falls back to node 0", "[bc][pressure_bc]") {
    // E.g. pnode = 0 in the 1-based *.physical file becomes -1 after
    // IO::read_physical; the fallback is silent (no warning is printed).
    const Mesh m = flagged_grid({});
    REQUIRE(m.npoin == 20);
    const int pnode = GENERATE(-1, -100, 20, 1000);
    CAPTURE(pnode);
    const PressureBC bc = setup_pressure_bcs(m, pnode);

    CHECK(bc.nodes == std::vector<int>{0});
    CHECK(bc.values == std::vector<double>{0.0});
}

TEST_CASE("Mesh without boundary sides gets a single reference node", "[bc][pressure_bc]") {
    const Mesh m = make_mesh({{0, 0}, {1, 0}, {0, 1}}, {{0, 1, 2}});
    const PressureBC bc = setup_pressure_bcs(m, 2);
    CHECK(bc.nodes == std::vector<int>{2});
}

TEST_CASE("Hinterkante pressure boundary conditions", "[bc][pressure_bc][hinterkante]") {
    // The outlet (side id 2, flag 504) is the edge x = 12 with 51 sides.
    const std::string dir = HYDRAMA2D_CASE_DIR;
    Mesh m;
    REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", m));
    REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", m));
    PhysicalConfig phys;
    REQUIRE(IO::read_physical(dir + "/system/hinterkante.physical", phys));

    const PressureBC bc = setup_pressure_bcs(m, phys.pnode);
    CHECK(bc.nodes.size() == 52);
    CHECK(std::set<int>(bc.nodes.begin(), bc.nodes.end()).size() == 52);
    check_all_zero(bc);
    for (int ip : bc.nodes) {
        CAPTURE(ip, m.y[ip]);
        CHECK(m.x[ip] == 12.0);
    }
}
