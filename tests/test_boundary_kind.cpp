// BoundaryKind: semantic classification of boundary flag codes, and its
// agreement with the flag codes hard-wired in the legacy routines
// (Mesh::determine_wall_nodes, setup_pressure_bcs, apply_velocity_bcs,
// apply_symmetry_bc).

#include "BoundaryConditions.hpp"
#include "BoundaryKind.hpp"
#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>

using namespace hydrama;
using namespace hydrama::test;

// Compile-time usable.
static_assert(classify_boundary_flag(500) == BoundaryKind::Wall);
static_assert(is_wall_flag(flags::kWallCold));
static_assert(!is_wall_flag(flags::kSymmetry));
static_assert(to_string(BoundaryKind::Outlet) == "outlet");

TEST_CASE("BoundaryKind: flag code table", "[boundary_kind]") {
    CHECK(classify_boundary_flag(500) == BoundaryKind::Wall);
    CHECK(classify_boundary_flag(501) == BoundaryKind::Wall);
    CHECK(classify_boundary_flag(502) == BoundaryKind::Wall);
    CHECK(classify_boundary_flag(503) == BoundaryKind::Inlet);
    CHECK(classify_boundary_flag(504) == BoundaryKind::Outlet);
    CHECK(classify_boundary_flag(505) == BoundaryKind::Periodic);
    CHECK(classify_boundary_flag(506) == BoundaryKind::Symmetry);
    CHECK(classify_boundary_flag(507) == BoundaryKind::Inlet);
    CHECK(classify_boundary_flag(508) == BoundaryKind::Inlet);
    CHECK(classify_boundary_flag(509) == BoundaryKind::Wall);
    CHECK(classify_boundary_flag(510) == BoundaryKind::Wall);
    for (int unknown : {0, -1, 499, 511, 600}) {
        INFO("flag " << unknown);
        CHECK(classify_boundary_flag(unknown) == BoundaryKind::Unknown);
        CHECK_FALSE(is_wall_flag(unknown));
    }
    CHECK(flags::kWall == 500);
    CHECK(flags::kWallHot == 501);
    CHECK(flags::kWallCold == 502);
    CHECK(flags::kInlet == 503);
    CHECK(flags::kOutlet == 504);
    CHECK(flags::kPeriodic == 505);
    CHECK(flags::kSymmetry == 506);
    CHECK(flags::kInletProfile == 507);
    CHECK(flags::kInletParabolic == 508);
    CHECK(flags::kMovingWall == 509);
    CHECK(flags::kRotatingWall == 510);
}

TEST_CASE("BoundaryKind: names", "[boundary_kind]") {
    CHECK(to_string(BoundaryKind::Wall) == "wall");
    CHECK(to_string(BoundaryKind::Inlet) == "inlet");
    CHECK(to_string(BoundaryKind::Outlet) == "outlet");
    CHECK(to_string(BoundaryKind::Periodic) == "periodic");
    CHECK(to_string(BoundaryKind::Symmetry) == "symmetry");
    CHECK(to_string(BoundaryKind::Unknown) == "unknown");
}

TEST_CASE("BoundaryKind agrees with the legacy boundary routines", "[boundary_kind]") {
    const int flag = GENERATE(0, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510);
    const BoundaryKind kind = classify_boundary_flag(flag);
    INFO("flag " << flag << " (" << to_string(kind) << ")");

    // A triangle whose three edges all carry `flag`; no edge is axis-aligned,
    // so the symmetry projection changes any generic velocity.
    Mesh m = make_triangle_mesh({"scalene", {{{1.0, 1.0}, {4.0, 2.0}, {2.0, 5.0}}}}, flag);
    // A parabolic inlet (508) must be an open curve: only the first edge.
    if (flag == flags::kInletParabolic) m.iside[1].flag = m.iside[2].flag = 0;
    compute_geometry(m);

    SECTION("walls: Mesh::determine_wall_nodes") {
        m.determine_wall_nodes();
        CHECK(m.wall_node_list.empty() == (kind != BoundaryKind::Wall));
    }

    SECTION("outlets: setup_pressure_bcs") {
        // Without outlet sides the legacy routine falls back to one reference node.
        const PressureBC bc = setup_pressure_bcs(m, 0);
        CHECK((bc.nodes.size() == 3) == (kind == BoundaryKind::Outlet));
    }

    SECTION("inlets and walls: apply_velocity_bcs") {
        m.determine_wall_nodes(); // the no-slip condition uses Mesh::wall_node_list
        std::vector<double> u(3, 42.0), v(3, 42.0);
        apply_velocity_bcs(m, u, v);
        const bool changed = std::any_of(v.begin(), v.end(), [](double x) { return x != 42.0; });
        CHECK(changed == (kind == BoundaryKind::Inlet || kind == BoundaryKind::Wall));
    }

    SECTION("symmetry: apply_symmetry_bc") {
        std::vector<double> u(3, 0.3), v(3, -0.7);
        apply_symmetry_bc(m, u, v);
        const bool changed = std::any_of(u.begin(), u.end(), [](double x) { return x != 0.3; });
        CHECK(changed == (kind == BoundaryKind::Symmetry));
    }
}
