#include "BoundaryConditions.hpp"
#include "IO.hpp"
#include "MeshCaseTestUtils.hpp"
#include "MeshTestUtils.hpp"
#include "Solver.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kPi = 3.14159265358979323846;

Mesh make_inlet_channel() {
    // 2x1 rectangle:
    // Left (x = 0): inlet (503)
    // Right (x = 2): outlet (504)
    // Bottom (y = 0) and Top (y = 1): wall (500)
    GridSpec g;
    g.nx = 4;
    g.ny = 2;
    g.lx = 2.0;
    g.ly = 1.0;
    Mesh m = make_grid_mesh(g, 504);
    // Boundary sides: bottom = 1, right = 2, top = 3, left = 4
    set_flags_by_side_id(m, {{1, 500}, {2, 504}, {3, 500}, {4, 503}});
    compute_all_geometry(m);
    return m;
}

} // namespace

TEST_CASE("InflowConfig defaults and accessors", "[inflow][bc]") {
    InflowConfig cfg;
    CHECK(cfg.empty());
    CHECK_FALSE(cfg.has_default);
    CHECK_FALSE(cfg.has_boundary(503));

    // When empty, defaults to u = 1, v = 0
    InflowSpec s = cfg.get_spec(503);
    CHECK_THAT(s.u, WithinAbs(1.0, 1e-12));
    CHECK_THAT(s.v, WithinAbs(0.0, 1e-12));

    // Set default
    cfg.has_default = true;
    cfg.default_u = 2.0;
    cfg.default_v = -0.5;
    CHECK_FALSE(cfg.empty());
    s = cfg.get_spec(503);
    CHECK_THAT(s.u, WithinAbs(2.0, 1e-12));
    CHECK_THAT(s.v, WithinAbs(-0.5, 1e-12));

    // Override specific boundary
    cfg.boundary_specs[503] = InflowSpec{0.8, 0.6};
    CHECK(cfg.has_boundary(503));
    s = cfg.get_spec(503);
    CHECK_THAT(s.u, WithinAbs(0.8, 1e-12));
    CHECK_THAT(s.v, WithinAbs(0.6, 1e-12));

    // Other boundary gets default
    s = cfg.get_spec(508);
    CHECK_THAT(s.u, WithinAbs(2.0, 1e-12));
    CHECK_THAT(s.v, WithinAbs(-0.5, 1e-12));
}

TEST_CASE("apply_velocity_bcs with custom angle of attack and speed", "[inflow][bc]") {
    Mesh mesh = make_inlet_channel();
    std::vector<double> u(mesh.npoin, 0.0);
    std::vector<double> v(mesh.npoin, 0.0);

    SECTION("Default uniform inflow gives u = 1, v = 0") {
        apply_velocity_bcs(mesh, u, v);
        // Find an interior inlet node (x = 0, y = 0.5)
        for (int ip = 0; ip < mesh.npoin; ++ip) {
            if (std::abs(mesh.x[ip]) < 1e-6 && std::abs(mesh.y[ip] - 0.5) < 1e-6) {
                CHECK_THAT(u[ip], WithinAbs(1.0, 1e-12));
                CHECK_THAT(v[ip], WithinAbs(0.0, 1e-12));
            }
            // Wall nodes must remain 0
            if (std::abs(mesh.y[ip]) < 1e-6 || std::abs(mesh.y[ip] - 1.0) < 1e-6) {
                CHECK_THAT(u[ip], WithinAbs(0.0, 1e-12));
                CHECK_THAT(v[ip], WithinAbs(0.0, 1e-12));
            }
        }
    }

    SECTION("Angle of attack 30 deg, speed 2.0") {
        const double alpha_deg = 30.0;
        const double speed = 2.0;
        const double rad = alpha_deg * (kPi / 180.0);
        const double expected_u = speed * std::cos(rad);
        const double expected_v = speed * std::sin(rad);

        InflowConfig cfg;
        cfg.has_default = true;
        cfg.default_u = expected_u;
        cfg.default_v = expected_v;

        apply_velocity_bcs(mesh, u, v, {}, cfg);

        for (int ip = 0; ip < mesh.npoin; ++ip) {
            if (std::abs(mesh.x[ip]) < 1e-6 && std::abs(mesh.y[ip] - 0.5) < 1e-6) {
                CHECK_THAT(u[ip], WithinAbs(expected_u, 1e-12));
                CHECK_THAT(v[ip], WithinAbs(expected_v, 1e-12));
            }
            // Corner nodes shared with wall are clamped to wall (no-slip precedence)
            if (std::abs(mesh.y[ip]) < 1e-6 || std::abs(mesh.y[ip] - 1.0) < 1e-6) {
                CHECK_THAT(u[ip], WithinAbs(0.0, 1e-12));
                CHECK_THAT(v[ip], WithinAbs(0.0, 1e-12));
            }
        }
    }
}

TEST_CASE("IO::read_inflow_config parser", "[inflow][io]") {
    namespace fs = std::filesystem;
    const fs::path test_dir = fs::temp_directory_path() / "hydrama_test_inflow";
    fs::create_directories(test_dir);

    SECTION("Missing file leaves config empty and returns true") {
        InflowConfig cfg;
        std::string err;
        CHECK(IO::read_inflow_config((test_dir / "nonexistent.inflow").string(), cfg, err));
        CHECK(cfg.empty());
        CHECK(err.empty());
    }

    SECTION("Speed and angle format") {
        const fs::path p = test_dir / "speed_angle.inflow";
        {
            std::ofstream out(p);
            out << "# Aerodynamic angle of attack configuration\n";
            out << "speed 1.5\n";
            out << "angle 10.0 // degrees\n";
        }
        InflowConfig cfg;
        std::string err;
        REQUIRE(IO::read_inflow_config(p.string(), cfg, err));
        CHECK(cfg.has_default);
        const double rad = 10.0 * (kPi / 180.0);
        CHECK_THAT(cfg.default_u, WithinAbs(1.5 * std::cos(rad), 1e-10));
        CHECK_THAT(cfg.default_v, WithinAbs(1.5 * std::sin(rad), 1e-10));
    }

    SECTION("Direct u and v format") {
        const fs::path p = test_dir / "uv.inflow";
        {
            std::ofstream out(p);
            out << "u 0.9961947\n";
            out << "v 0.0871557\n";
        }
        InflowConfig cfg;
        std::string err;
        REQUIRE(IO::read_inflow_config(p.string(), cfg, err));
        CHECK(cfg.has_default);
        CHECK_THAT(cfg.default_u, WithinAbs(0.9961947, 1e-7));
        CHECK_THAT(cfg.default_v, WithinAbs(0.0871557, 1e-7));
    }

    SECTION("Per-boundary override with alpha keyword") {
        const fs::path p = test_dir / "boundary.inflow";
        {
            std::ofstream out(p);
            out << "boundary 503 alpha 5.0 speed 2.0\n";
        }
        InflowConfig cfg;
        std::string err;
        REQUIRE(IO::read_inflow_config(p.string(), cfg, err));
        CHECK(cfg.has_boundary(503));
        const double rad = 5.0 * (kPi / 180.0);
        CHECK_THAT(cfg.boundary_specs[503].u, WithinAbs(2.0 * std::cos(rad), 1e-10));
        CHECK_THAT(cfg.boundary_specs[503].v, WithinAbs(2.0 * std::sin(rad), 1e-10));
    }

    SECTION("Invalid syntax returns error") {
        const fs::path p = test_dir / "invalid.inflow";
        {
            std::ofstream out(p);
            out << "speed not_a_number\n";
        }
        InflowConfig cfg;
        std::string err;
        CHECK_FALSE(IO::read_inflow_config(p.string(), cfg, err));
        CHECK_THAT(err, Catch::Matchers::ContainsSubstring("invalid value for 'speed'"));
    }

    fs::remove_all(test_dir);
}

TEST_CASE("Solver integration with angle of attack", "[inflow][solver]") {
    TempDir dir("test_inflow_solver");
    prepare_case(dir.path(), {});

    SimulationConfig cfg;
    cfg.casename = "hinterkante";
    cfg.case_dir = dir.path().string();

    set_nonempty_lines(dir / "system/hinterkante.control", {{2, "1"}}); // ntime = 1
    Solver solver(cfg);
    solver.set_angle_of_attack(5.0, 1.2);
    REQUIRE(solver.initialize());
    solver.run();

    const double rad = 5.0 * (kPi / 180.0);
    const double expected_u = 1.2 * std::cos(rad);
    const double expected_v = 1.2 * std::sin(rad);

    // Verify velocity on interior inlet nodes (excluding corner nodes shared with walls)
    const auto& mesh = solver.mesh();
    const auto& u = solver.u();
    const auto& v = solver.v();
    std::set<int> wall_nodes(mesh.wall_node_list.begin(), mesh.wall_node_list.end());
    int checked_nodes = 0;
    for (int ib = 0; ib < mesh.nboun; ++ib) {
        if (mesh.iside[ib].flag == flags::kInlet) {
            for (int ip : {mesh.iside[ib].node1, mesh.iside[ib].node2}) {
                if (wall_nodes.count(ip) == 0) {
                    CHECK_THAT(u[ip], WithinAbs(expected_u, 1e-10));
                    CHECK_THAT(v[ip], WithinAbs(expected_v, 1e-10));
                    ++checked_nodes;
                }
            }
        }
    }
    CHECK(checked_nodes > 0);
}
