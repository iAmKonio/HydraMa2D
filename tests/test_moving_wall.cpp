// Moving and rotating wall boundary conditions (flags 509 and 510):
// MovingWallSpec configuration, moving_wall_velocities(), apply_velocity_bcs(),
// corner discontinuity handling (corners_zero), *.walls configuration parser,
// and lid-driven cavity simulation.

#include "BoundaryConditions.hpp"
#include "BoundaryKind.hpp"
#include "IO.hpp"
#include "MeshCaseTestUtils.hpp"
#include "QuadTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>

using namespace hydrama;
using namespace hydrama::test;
namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Unit square [0, 1] x [0, 1] mesh with specified boundary flags for the 4 sides.
Mesh square_mesh(GridCells cells, int nx, int ny, int bottom_flag, int right_flag, int top_flag, int left_flag) {
    GridOptions g;
    g.lx = 1.0;
    g.ly = 1.0;
    g.nx = nx;
    g.ny = ny;
    g.cells = cells;
    return make_hybrid_mesh(grid(g), [=](double x, double y) {
        if (y == 0.0) return bottom_flag;
        if (x == 1.0) return right_flag;
        if (y == 1.0) return top_flag;
        if (x == 0.0) return left_flag;
        return 500;
    });
}

} // namespace

TEST_CASE("MovingWallSpec: defaults and enum", "[bc][moving_wall]") {
    MovingWallSpec spec;
    CHECK(spec.kind == MovingWallSpec::Kind::Translating);
    CHECK(spec.u == 1.0);
    CHECK(spec.v == 0.0);
    CHECK(spec.omega == 1.0);
    CHECK(spec.center_x == 0.0);
    CHECK(spec.center_y == 0.0);
    CHECK(spec.auto_center == true);
    CHECK(spec.corners_zero == false);
}

TEST_CASE("Translating wall: default and custom velocities", "[bc][moving_wall]") {
    const GridCells cells = GENERATE(GridCells::Triangles, GridCells::Quads);
    Mesh m = square_mesh(cells, 4, 4, 500, 500, 509, 500);

    SECTION("default specification (empty wall_specs map)") {
        const auto vels = moving_wall_velocities(m, {});
        REQUIRE_FALSE(vels.empty());
        for (const auto& nv : vels) {
            CHECK_THAT(m.y[nv.node], WithinAbs(1.0, 1e-12));
            CHECK(nv.u == 1.0);
            CHECK(nv.v == 0.0);
        }
    }

    SECTION("custom translating velocity") {
        std::map<int, MovingWallSpec> specs;
        MovingWallSpec spec;
        spec.kind = MovingWallSpec::Kind::Translating;
        spec.u = -2.5;
        spec.v = 0.75;
        specs[509] = spec;

        const auto vels = moving_wall_velocities(m, specs);
        REQUIRE_FALSE(vels.empty());
        for (const auto& nv : vels) {
            CHECK_THAT(m.y[nv.node], WithinAbs(1.0, 1e-12));
            CHECK(nv.u == -2.5);
            CHECK(nv.v == 0.75);
        }
    }
}

TEST_CASE("Rotating wall: auto-centering and explicit center", "[bc][rotating_wall]") {
    // Construct an 8-sided regular polygon ring centered at (2.0, 3.0) with radius 1.0
    const double cx0 = 2.0;
    const double cy0 = 3.0;
    const double R = 1.0;
    const int N = 8;
    const double pi = 3.14159265358979323846;

    Mesh m;
    m.npoin = N;
    m.nelem = 0;
    m.nboun = N;
    for (int i = 0; i < N; ++i) {
        double theta = 2.0 * pi * i / N;
        m.x.push_back(cx0 + R * std::cos(theta));
        m.y.push_back(cy0 + R * std::sin(theta));
    }
    for (int i = 0; i < N; ++i) {
        BoundarySide side;
        side.node1 = i;
        side.node2 = (i + 1) % N;
        side.elem = 0;
        side.flag = flags::kRotatingWall; // 510
        side.side_id = 1;
        m.iside.push_back(side);
    }

    SECTION("auto-centering calculates centroid and assigns omega x r") {
        std::map<int, MovingWallSpec> specs;
        MovingWallSpec spec;
        spec.kind = MovingWallSpec::Kind::Rotating;
        spec.omega = 2.5;
        spec.auto_center = true;
        specs[flags::kRotatingWall] = spec;

        const auto vels = moving_wall_velocities(m, specs);
        REQUIRE(vels.size() == static_cast<std::size_t>(N));
        for (const auto& nv : vels) {
            double dx = m.x[nv.node] - cx0;
            double dy = m.y[nv.node] - cy0;
            double expected_u = -2.5 * dy;
            double expected_v = 2.5 * dx;
            CHECK_THAT(nv.u, WithinAbs(expected_u, 1e-12));
            CHECK_THAT(nv.v, WithinAbs(expected_v, 1e-12));
        }
    }

    SECTION("explicit center specifies rotation axis") {
        std::map<int, MovingWallSpec> specs;
        MovingWallSpec spec;
        spec.kind = MovingWallSpec::Kind::Rotating;
        spec.omega = -1.5;
        spec.center_x = cx0;
        spec.center_y = cy0;
        spec.auto_center = false;
        specs[flags::kRotatingWall] = spec;

        const auto vels = moving_wall_velocities(m, specs);
        REQUIRE(vels.size() == static_cast<std::size_t>(N));
        for (const auto& nv : vels) {
            double dx = m.x[nv.node] - cx0;
            double dy = m.y[nv.node] - cy0;
            double expected_u = -(-1.5) * dy;
            double expected_v = (-1.5) * dx;
            CHECK_THAT(nv.u, WithinAbs(expected_u, 1e-12));
            CHECK_THAT(nv.v, WithinAbs(expected_v, 1e-12));
        }
    }
}

TEST_CASE("Lid-driven cavity corner treatment: corners_zero", "[bc][moving_wall]") {
    const GridCells cells = GENERATE(GridCells::Triangles, GridCells::Quads);
    Mesh m = square_mesh(cells, 4, 4, 500, 500, 509, 500);
    compute_all_geometry(m);
    m.determine_wall_nodes();

    std::vector<double> u(m.npoin, 99.0);
    std::vector<double> v(m.npoin, 99.0);

    SECTION("corners_zero = false (default): moving lid velocity wins at top corners") {
        std::map<int, MovingWallSpec> specs;
        MovingWallSpec spec;
        spec.kind = MovingWallSpec::Kind::Translating;
        spec.u = 1.0;
        spec.v = 0.0;
        spec.corners_zero = false;
        specs[509] = spec;

        apply_velocity_bcs(m, u, v, specs);

        for (int i = 0; i < m.npoin; ++i) {
            if (m.y[i] == 1.0) {
                // All top nodes including corners (0,1) and (1,1) have lid velocity
                CHECK_THAT(u[i], WithinAbs(1.0, 1e-14));
                CHECK_THAT(v[i], WithinAbs(0.0, 1e-14));
            } else if (m.x[i] == 0.0 || m.x[i] == 1.0 || m.y[i] == 0.0) {
                // Stationary wall nodes are zero
                CHECK_THAT(u[i], WithinAbs(0.0, 1e-14));
                CHECK_THAT(v[i], WithinAbs(0.0, 1e-14));
            }
        }
    }

    SECTION("corners_zero = true: nodes shared with stationary walls stay zero") {
        std::map<int, MovingWallSpec> specs;
        MovingWallSpec spec;
        spec.kind = MovingWallSpec::Kind::Translating;
        spec.u = 1.0;
        spec.v = 0.0;
        spec.corners_zero = true;
        specs[509] = spec;

        apply_velocity_bcs(m, u, v, specs);

        for (int i = 0; i < m.npoin; ++i) {
            if (m.y[i] == 1.0) {
                if (m.x[i] == 0.0 || m.x[i] == 1.0) {
                    // Top corners shared with stationary side walls stay 0
                    CHECK_THAT(u[i], WithinAbs(0.0, 1e-14));
                    CHECK_THAT(v[i], WithinAbs(0.0, 1e-14));
                } else {
                    // Interior lid nodes get u = 1.0
                    CHECK_THAT(u[i], WithinAbs(1.0, 1e-14));
                    CHECK_THAT(v[i], WithinAbs(0.0, 1e-14));
                }
            } else if (m.x[i] == 0.0 || m.x[i] == 1.0 || m.y[i] == 0.0) {
                CHECK_THAT(u[i], WithinAbs(0.0, 1e-14));
                CHECK_THAT(v[i], WithinAbs(0.0, 1e-14));
            }
        }
    }
}

TEST_CASE("IO: read_walls_config parser", "[io][moving_wall]") {
    TempDir dir("test_walls_parser");
    const std::string valid_file = (dir / "test.walls").string();

    SECTION("parses valid translating and rotating wall entries") {
        write_text(valid_file,
            "# Moving and rotating wall configuration\n"
            "// Comment line with slashes\n"
            "wall 509 translating u 1.5 v -0.2 corners_zero\n"
            "wall 510 rotating omega 3.14 center 1.0 2.0\n"
            "wall lid u 0.8 v 0.1\n"
            "wall rotor omega 10.0 auto_center\n"
        );

        std::map<int, MovingWallSpec> specs;
        std::string error;
        REQUIRE(IO::read_walls_config(valid_file, specs, error));
        CHECK(error.empty());

        REQUIRE(specs.count(509) == 1);
        CHECK(specs[509].kind == MovingWallSpec::Kind::Translating);
        CHECK_THAT(specs[509].u, WithinAbs(0.8, 1e-14)); // overwritten by 'wall lid'
        CHECK_THAT(specs[509].v, WithinAbs(0.1, 1e-14));

        REQUIRE(specs.count(510) == 1);
        CHECK(specs[510].kind == MovingWallSpec::Kind::Rotating);
        CHECK_THAT(specs[510].omega, WithinAbs(10.0, 1e-14)); // overwritten by 'wall rotor'
        CHECK(specs[510].auto_center == true);
    }

    SECTION("missing file returns true and empty specs") {
        std::map<int, MovingWallSpec> specs;
        std::string error;
        REQUIRE(IO::read_walls_config((dir / "nonexistent.walls").string(), specs, error));
        CHECK(specs.empty());
        CHECK(error.empty());
    }

    SECTION("reports invalid value for u") {
        write_text(valid_file, "wall 509 translating u not_a_number\n");
        std::map<int, MovingWallSpec> specs;
        std::string error;
        CHECK_FALSE(IO::read_walls_config(valid_file, specs, error));
        CHECK_THAT(error, ContainsSubstring("invalid value for 'u'"));
    }

    SECTION("reports missing coordinates for center") {
        write_text(valid_file, "wall 510 rotating center 1.0\n");
        std::map<int, MovingWallSpec> specs;
        std::string error;
        CHECK_FALSE(IO::read_walls_config(valid_file, specs, error));
        CHECK_THAT(error, ContainsSubstring("requires two coordinates"));
    }

    SECTION("reports unknown keyword") {
        write_text(valid_file, "wall 509 bogus_keyword\n");
        std::map<int, MovingWallSpec> specs;
        std::string error;
        CHECK_FALSE(IO::read_walls_config(valid_file, specs, error));
        CHECK_THAT(error, ContainsSubstring("unknown keyword 'bogus_keyword'"));
    }
}

TEST_CASE("Solver: Lid-driven cavity simulation with moving lid", "[solver][moving_wall]") {
    // 8 x 8 cavity mesh in Gmsh MSH format
    const int N = 8;
    GridOptions g;
    g.lx = 1.0;
    g.ly = 1.0;
    g.nx = N;
    g.ny = N;
    g.cells = GridCells::Triangles;

    const Mesh m = make_hybrid_mesh(grid(g), [](double x, double y) {
        if (y == 1.0) return 509; // Moving lid
        (void)x;
        return 500; // Stationary wall
    });

    GmshModel gmsh = gmsh_model_from_mesh(m, {{500, "wall"}, {509, "moving_wall"}}, 2);
    for (const auto& q : m.quads) gmsh.add_element(3, {q[0] + 1, q[1] + 1, q[2] + 1, q[3] + 1}, {2});

    GmshWriteOptions gmsh_opts;
    gmsh_opts.version = "2.2";
    gmsh_opts.binary = false;
    const std::string msh_content = write_gmsh(gmsh, gmsh_opts);

    CaseOptions opts;
    opts.steps = 50;

    MeshCaseOptions extra;
    extra.system_files["hinterkante.walls"] = "wall moving_wall translating u 1.0 v 0.0\n";

    const MeshCaseResult r = run_mesh_case("cavity_re100", msh_content, opts, extra);
    REQUIRE(r.run.initialized);
    CHECK(r.run.log.find("Simulation Completed") != std::string::npos);

    // Verify velocity field:
    // Top boundary nodes must have u = 1.0, v = 0.0
    // Recirculation: u near the bottom (y < 0.5) must develop negative flow (u < 0)
    bool has_negative_u = false;
    for (int i = 0; i < r.mesh.npoin; ++i) {
        if (r.mesh.y[i] == 1.0) {
            CHECK_THAT(r.run.u[i], WithinAbs(1.0, 1e-14));
            CHECK_THAT(r.run.v[i], WithinAbs(0.0, 1e-14));
        } else if (r.mesh.y[i] < 0.5 && r.mesh.x[i] > 0.2 && r.mesh.x[i] < 0.8) {
            if (r.run.u[i] < -1e-4) {
                has_negative_u = true;
            }
        }
    }
    CHECK(has_negative_u);
}
