// Unit and benchmark tests for periodic boundary conditions (Phase 2.2):
// 1. Parsing of Gmsh $Periodic blocks in MSH 2.2 and MSH 4.1.
// 2. PeriodicMapping and PeriodicReducedOperator: preserving SPD symmetry and CG convergence.
// 3. apply_periodic_field: lumped mass weighted averaging across periodic node groups.
// 4. match_periodic_boundary_nodes: automatic translation detection and node matching.
// 5. Poiseuille channel flow benchmark: periodic inlet/outlet with body force driving,
//    comparing steady-state profile against analytical parabolic solution u(y) = 6 y (1 - y).

#include "BoundaryKind.hpp"
#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "LinearSolver.hpp"
#include "Mesh.hpp"
#include "MeshTestUtils.hpp"
#include "PressureSolver.hpp"
#include "Solver.hpp"
#include "SolverTestUtils.hpp"
#include "linalg/ConjugateGradient.hpp"
#include "linalg/EbeOperator.hpp"
#include "mesh/GmshReader.hpp"
#include "mesh/MeshBuilder.hpp"
#include "numerics/Periodic.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

using namespace hydrama;
using namespace hydrama::mesh;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("GmshReader parses $Periodic sections in MSH 2.2 and 4.1", "[periodic][gmsh]") {
    SECTION("MSH 2.2 ASCII with $Periodic") {
        const std::string msh2 =
            "$MeshFormat\n"
            "2.2 0 8\n"
            "$EndMeshFormat\n"
            "$PhysicalNames\n"
            "4\n"
            "1 1 \"bottom_wall\"\n"
            "1 2 \"periodic_slave\"\n"
            "1 3 \"top_wall\"\n"
            "1 4 \"periodic_master\"\n"
            "$EndPhysicalNames\n"
            "$Nodes\n"
            "4\n"
            "1 0.0 0.0 0.0\n"
            "2 1.0 0.0 0.0\n"
            "3 1.0 1.0 0.0\n"
            "4 0.0 1.0 0.0\n"
            "$EndNodes\n"
            "$Elements\n"
            "6\n"
            "1 1 2 1 1 1 2\n" // line bottom: tag 1
            "2 1 2 2 2 2 3\n" // line right (slave): tag 2
            "3 1 2 3 3 3 4\n" // line top: tag 3
            "4 1 2 4 4 4 1\n" // line left (master): tag 4
            "5 2 2 10 10 1 2 3\n"
            "6 2 2 10 10 1 3 4\n"
            "$EndElements\n"
            "$Periodic\n"
            "1\n"
            "1 2 4\n" // dim=1, slave_tag=2, master_tag=4
            "0\n"     // 0 affine parameters
            "2\n"     // 2 node pairs (slave, master)
            "2 1\n"
            "3 4\n"
            "$EndPeriodic\n";

        MeshData data = parse_gmsh(msh2, "test.msh");
        REQUIRE(data.periodic_links.size() == 1);
        CHECK(data.periodic_links[0].dim == 1);
        CHECK(data.periodic_links[0].entity_tag == 2);
        CHECK(data.periodic_links[0].master_tag == 4);
        REQUIRE(data.periodic_links[0].node_tags.size() == 2);
        CHECK(data.periodic_links[0].node_tags[0].first == 2);
        CHECK(data.periodic_links[0].node_tags[0].second == 1);
        CHECK(data.periodic_links[0].node_tags[1].first == 3);
        CHECK(data.periodic_links[0].node_tags[1].second == 4);

        Mesh mesh = build_mesh(data);
        CHECK(mesh.has_periodic());
        CHECK(mesh.periodic_pairs.size() == 2);
        CHECK(mesh.periodic_slave_nodes.size() == 2);

        // Check canonical master mapping: node 2 -> node 1, node 3 -> node 4
        // In internal 0-based indices:
        int n1 = -1, n2 = -1, n3 = -1, n4 = -1;
        for (int i = 0; i < mesh.npoin; ++i) {
            if (std::abs(mesh.x[i] - 0.0) < 1e-6 && std::abs(mesh.y[i] - 0.0) < 1e-6) n1 = i;
            if (std::abs(mesh.x[i] - 1.0) < 1e-6 && std::abs(mesh.y[i] - 0.0) < 1e-6) n2 = i;
            if (std::abs(mesh.x[i] - 1.0) < 1e-6 && std::abs(mesh.y[i] - 1.0) < 1e-6) n3 = i;
            if (std::abs(mesh.x[i] - 0.0) < 1e-6 && std::abs(mesh.y[i] - 1.0) < 1e-6) n4 = i;
        }
        REQUIRE(n1 >= 0);
        REQUIRE(n2 >= 0);
        REQUIRE(n3 >= 0);
        REQUIRE(n4 >= 0);
        CHECK(mesh.canonical_master(n2) == n1);
        CHECK(mesh.canonical_master(n3) == n4);
    }

    SECTION("MSH 4.1 ASCII with $Periodic") {
        const std::string msh4 =
            "$MeshFormat\n"
            "4.1 0 8\n"
            "$EndMeshFormat\n"
            "$Entities\n"
            "0 4 1 0\n" // 0 points, 4 curves, 1 surface, 0 volumes
            "1 0 0 0 1 0 0 1 1 0\n" // curve 1, tag 1 (bottom_wall)
            "2 1 0 0 1 1 0 1 2 0\n" // curve 2, tag 2 (periodic_slave)
            "3 1 1 0 0 1 0 1 3 0\n" // curve 3, tag 3 (top_wall)
            "4 0 1 0 0 0 0 1 4 0\n" // curve 4, tag 4 (periodic_master)
            "1 0 0 0 1 1 0 1 10 4 1 2 3 4\n" // surface 1, physical tag 10
            "$EndEntities\n"
            "$Nodes\n"
            "1 4 1 4\n"
            "2 1 0 4\n"
            "1\n2\n3\n4\n"
            "0.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "1.0 1.0 0.0\n"
            "0.0 1.0 0.0\n"
            "$EndNodes\n"
            "$Elements\n"
            "2 6 1 6\n"
            "1 1 1 4\n" // 4 lines
            "1 1 2\n"
            "2 2 3\n"
            "3 3 4\n"
            "4 4 1\n"
            "2 1 2 2\n" // 2 triangles
            "5 1 2 3\n"
            "6 1 3 4\n"
            "$EndElements\n"
            "$Periodic\n"
            "1\n"
            "1 2 4 0\n" // dim=1, slave_tag=2, master_tag=4, num_affine=0
            "2\n"
            "2 1\n"
            "3 4\n"
            "$EndPeriodic\n";

        MeshData data = parse_gmsh(msh4, "test.msh");
        REQUIRE(data.periodic_links.size() == 1);
        CHECK(data.periodic_links[0].entity_tag == 2);
        CHECK(data.periodic_links[0].master_tag == 4);
        REQUIRE(data.periodic_links[0].node_tags.size() == 2);
    }
}

TEST_CASE("PeriodicMapping and PeriodicReducedOperator preserve SPD symmetry", "[periodic][linalg]") {
    GridSpec g;
    g.lx = 1.0;
    g.ly = 1.0;
    g.nx = 3;
    g.ny = 3;
    Mesh mesh = make_grid_mesh(g, flags::kPeriodic);
    compute_all_geometry(mesh);

    // Identify left (x=0) and right (x=1) nodes and match them
    std::vector<int> left_nodes, right_nodes;
    for (int i = 0; i < mesh.npoin; ++i) {
        if (std::abs(mesh.x[i] - 0.0) < 1e-6) left_nodes.push_back(i);
        if (std::abs(mesh.x[i] - 1.0) < 1e-6) right_nodes.push_back(i);
    }
    std::sort(left_nodes.begin(), left_nodes.end());
    std::sort(right_nodes.begin(), right_nodes.end());

    int matched = match_periodic_boundary_nodes(mesh, right_nodes, left_nodes);
    CHECK(matched == 4);
    REQUIRE(mesh.has_periodic());

    // Assemble LHS
    std::vector<double> delte(mesh.nelem, 0.01);
    std::vector<double> elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;
    update_lhs(mesh, delte, elcoe2, pdiag, gstif);

    linalg::EbeOperator full_ebe(mesh, gstif, pdiag);
    PeriodicMapping map = PeriodicMapping::build(mesh);
    CHECK(map.has_periodic);
    CHECK(map.num_nodes == 16);
    CHECK(map.num_independent == 12); // 16 - 4 slave nodes
    CHECK(map.periodic_slaves.size() == 4);

    linalg::PeriodicReducedOperator red_op(full_ebe, map);
    CHECK(red_op.size() == 12);
    CHECK(red_op.traits().symmetric);

    // Test operator symmetry: x_hat^T (A_hat y_hat) == y_hat^T (A_hat x_hat)
    std::vector<double> x_hat(12), y_hat(12), Ax_hat(12), Ay_hat(12);
    for (int i = 0; i < 12; ++i) {
        x_hat[i] = std::sin(1.7 * (i + 1));
        y_hat[i] = std::cos(2.3 * (i + 1));
    }
    red_op.apply(x_hat, Ax_hat);
    red_op.apply(y_hat, Ay_hat);

    double x_dot_Ay = 0.0, y_dot_Ax = 0.0;
    for (int i = 0; i < 12; ++i) {
        x_dot_Ay += x_hat[i] * Ay_hat[i];
        y_dot_Ax += y_hat[i] * Ax_hat[i];
    }
    CHECK_THAT(x_dot_Ay, WithinRel(y_dot_Ax, 1e-12));

    // Test solving with Dirichlet constraint at node 0
    PressureBC pbc;
    pbc.nodes = {0};
    pbc.values = {0.0};

    PressureSolver solver(SolverConfig{}, pbc);
    solver.update(mesh, gstif, pdiag);

    std::vector<double> rhs(mesh.npoin, 1.0);
    std::vector<double> p(mesh.npoin, 0.0);
    linalg::SolveResult res = solver.solve(rhs, p);
    CHECK(res.converged);

    // Verify exact equality across all periodic pairs
    for (const auto& pair : mesh.periodic_pairs) {
        CHECK_THAT(p[pair.slave], WithinRel(p[pair.master], 1e-12));
    }
}

TEST_CASE("apply_periodic_field enforces mass-weighted periodic consistency", "[periodic][fields]") {
    Mesh mesh;
    mesh.npoin = 4;
    mesh.periodic_pairs = {{1, 0}, {3, 2}};
    mesh.periodic_master_map = {0, 0, 2, 2};
    mesh.periodic_slave_nodes = {1, 3};
    mesh.periodic_groups = {{0, 1}, {2, 3}};

    // M_0 = 3.0, M_1 = 1.0 => elcoe2 = 1 / mass = {1/3, 1}
    // M_2 = 2.0, M_3 = 2.0 => elcoe2 = {0.5, 0.5}
    std::vector<double> elcoe2 = {1.0 / 3.0, 1.0, 0.5, 0.5};
    std::vector<double> field = {10.0, 2.0, 5.0, 9.0};

    apply_periodic_field(mesh, elcoe2, field);

    // Group 0: (10.0 * 3.0 + 2.0 * 1.0) / (3.0 + 1.0) = 32 / 4 = 8.0
    CHECK_THAT(field[0], WithinRel(8.0, 1e-14));
    CHECK_THAT(field[1], WithinRel(8.0, 1e-14));

    // Group 1: (5.0 * 2.0 + 9.0 * 2.0) / 4.0 = 28 / 4 = 7.0
    CHECK_THAT(field[2], WithinRel(7.0, 1e-14));
    CHECK_THAT(field[3], WithinRel(7.0, 1e-14));
}

TEST_CASE("Poiseuille channel flow with periodic boundary conditions", "[periodic][poiseuille][benchmark]") {
    // 2D channel flow [0, 1] x [0, 1] driven by body force fx.
    // Periodic in x: inlet (left) and outlet (right) are periodic.
    // Top and bottom walls are no-slip (u = v = 0).
    // Re = 100, fx = 12 / Re = 0.12.
    // Analytical solution: u(y) = 6 * y * (1 - y), peak u_max = 1.5 at y=0.5.
    TempDir dir("test_poiseuille_case");
    CaseOptions opts;
    opts.steps = 150;
    opts.transient = 0; // steady state
    prepare_case(dir.path(), opts);

    // Set fixed time step for stable CBS convergence (dt = 0.08, 600 steps)
    set_nonempty_lines(dir.path() / "system/hinterkante.control",
                       {{kDtfixed, "1"}, {26, "0.08"}, {kNtime, "600"}, {kTransient, "0"}});
    // Set Re = 100.0
    set_nonempty_lines(dir.path() / "system/hinterkante.physical",
                       {{2, "100.0"}});

    // Generate mesh using GridSpec: 4 x 10 cells
    GridSpec g;
    g.lx = 1.0;
    g.ly = 1.0;
    g.nx = 4;
    g.ny = 10;
    Mesh grid_m = make_grid_mesh(g);

    // Tag sides:
    // 1: bottom (y=0) -> 501 (wall)
    // 2: right (x=1) -> 505 (periodic slave)
    // 3: top (y=1) -> 501 (wall)
    // 4: left (x=0) -> 505 (periodic master)
    set_flags_by_side_id(grid_m, {
        {1, flags::kWall},
        {2, flags::kPeriodic},
        {3, flags::kWall},
        {4, flags::kPeriodic}
    });

    // Write constant/hinterkante.mesh and hinterkante.bofl
    {
        std::ofstream fm(dir.path() / "constant/hinterkante.mesh");
        fm << grid_m.nelem << " " << grid_m.npoin << " " << grid_m.nboun << "\n";
        for (int i = 0; i < grid_m.nelem; ++i) {
            fm << (i + 1) << " " << (grid_m.intma[i][0] + 1) << " " << (grid_m.intma[i][1] + 1) << " " << (grid_m.intma[i][2] + 1) << "\n";
        }
        for (int i = 0; i < grid_m.npoin; ++i) {
            fm << (i + 1) << " " << grid_m.x[i] << " " << grid_m.y[i] << "\n";
        }
        for (int i = 0; i < grid_m.nboun; ++i) {
            fm << (grid_m.iside[i].node1 + 1) << " " << (grid_m.iside[i].node2 + 1) << " "
               << (grid_m.iside[i].elem + 1) << " " << grid_m.iside[i].side_id << "\n";
        }

        std::map<int, int> side_to_flag;
        for (int i = 0; i < grid_m.nboun; ++i) {
            side_to_flag[grid_m.iside[i].side_id] = grid_m.iside[i].flag;
        }
        std::ofstream fb(dir.path() / "constant/hinterkante.bofl");
        fb << side_to_flag.size() << "\n";
        for (const auto& [sid, flg] : side_to_flag) {
            fb << sid << " " << flg << "\n";
        }
    }

    // Write system/hinterkante.periodic with driving body force fx = 0.12
    write_text(dir.path() / "system/hinterkante.periodic",
               "# Periodic channel configuration\n"
               "periodic on\n"
               "driving_force 0.12 0.0\n");

    SimulationConfig cfg;
    cfg.casename = "hinterkante";
    cfg.case_dir = dir.path().string();

    Solver solver(cfg);
    REQUIRE(solver.initialize());
    CHECK(solver.mesh().has_periodic());
    CHECK(solver.periodic_config().has_driving_force());

    solver.run();

    const Mesh& final_mesh = solver.mesh();
    const auto& u = solver.u();
    const auto& v = solver.v();

    // 1. Verify exact machine-precision periodic consistency between inlet and outlet
    for (const auto& pair : final_mesh.periodic_pairs) {
        CHECK_THAT(u[pair.slave], WithinRel(u[pair.master], 1e-10));
        CHECK_THAT(v[pair.slave], WithinAbs(v[pair.master], 1e-10));
    }

    // 2. Check velocity at channel centerline (y = 0.5)
    // Analytical: u_max = 1.5
    double u_center_max = 0.0;
    for (int i = 0; i < final_mesh.npoin; ++i) {
        if (std::abs(final_mesh.y[i] - 0.5) < 1e-4) {
            u_center_max = std::max(u_center_max, u[i]);
        }
    }
    CHECK(u_center_max > 1.0);
    CHECK(u_center_max <= 1.6);
}

TEST_CASE("Periodic cascade boundary matching with pitch translation", "[periodic][cascade]") {
    // Turbomachinery cascade with pitch spacing = 0.5 along y
    Mesh mesh;
    mesh.npoin = 10;
    mesh.x.resize(10);
    mesh.y.resize(10);

    // Master boundary (lower periodic curve at y = 0.0)
    for (int i = 0; i < 5; ++i) {
        mesh.x[i] = 0.25 * i;
        mesh.y[i] = 0.0;
    }

    // Slave boundary (upper periodic curve shifted by pitch dy = 0.5)
    for (int i = 0; i < 5; ++i) {
        mesh.x[5 + i] = 0.25 * i;
        mesh.y[5 + i] = 0.5;
    }

    std::vector<int> master_nodes = {0, 1, 2, 3, 4};
    std::vector<int> slave_nodes = {5, 6, 7, 8, 9};

    SECTION("Auto-deduced pitch translation") {
        Mesh m = mesh;
        int matched = match_periodic_boundary_nodes(m, slave_nodes, master_nodes);
        CHECK(matched == 5);
        REQUIRE(m.has_periodic());
        REQUIRE(m.periodic_boundaries.size() == 1);
        CHECK_THAT(m.periodic_boundaries[0].translation[0], WithinAbs(0.0, 1e-12));
        CHECK_THAT(m.periodic_boundaries[0].translation[1], WithinAbs(0.5, 1e-12));

        for (int i = 0; i < 5; ++i) {
            CHECK(m.canonical_master(5 + i) == i);
        }
    }

    SECTION("Explicit pitch translation") {
        Mesh m = mesh;
        std::array<double, 2> translation = {0.0, 0.5};
        int matched = match_periodic_boundary_nodes(m, slave_nodes, master_nodes, translation);
        CHECK(matched == 5);
        REQUIRE(m.has_periodic());
        for (int i = 0; i < 5; ++i) {
            CHECK(m.canonical_master(5 + i) == i);
        }
    }
}
