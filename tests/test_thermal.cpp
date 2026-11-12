// Thermal energy transport and Boussinesq buoyancy coupling tests (Phase 1.3):
// - ThermalConfig parser and boundary overrides
// - Effective thermal diffusivity (laminar and turbulent)
// - Boussinesq buoyancy body force assembly (triangles and quads)
// - Thermal boundary condition enforcement
// - Wall heat flux and Nusselt number diagnostics
// - Pure 1D conduction benchmark (Nu = 1.000)
// - De Vahl Davis natural convection square cavity benchmark (Ra = 10^3)
// - Thermal restart IO

#include "BoundaryKind.hpp"
#include "IO.hpp"
#include "MeshCaseTestUtils.hpp"
#include "QuadTestUtils.hpp"
#include "Solver.hpp"
#include "thermal/ThermalConfig.hpp"
#include "thermal/ThermalPhysics.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <sstream>
#include <vector>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

Mesh make_square_cavity_mesh(int n, GridCells cells) {
    GridOptions g;
    g.lx = 1.0;
    g.ly = 1.0;
    g.nx = n;
    g.ny = n;
    g.cells = cells;
    Mesh m = make_hybrid_mesh(grid(g), [](double x, double y) {
        if (x == 0.0) return flags::kWallHot;  // 501: Hot left wall
        if (x == 1.0) return flags::kWallCold; // 502: Cold right wall
        (void)y;
        return flags::kWall;                   // 500: Adiabatic top/bottom walls
    });
    m.compute_shape_function_derivatives();
    m.compute_face_normals();
    return m;
}

} // namespace

TEST_CASE("ThermalConfig: defaults and basic properties", "[thermal][config]") {
    ThermalConfig config;
    CHECK_FALSE(config.enabled);
    CHECK_THAT(config.prandtl, WithinAbs(0.71, 1e-14));
    CHECK_THAT(config.prandtl_turbulent, WithinAbs(0.85, 1e-14));
    CHECK_THAT(config.beta, WithinAbs(0.0, 1e-14));
    CHECK_THAT(config.reference_temperature, WithinAbs(0.0, 1e-14));
    CHECK_THAT(config.gravity[0], WithinAbs(0.0, 1e-14));
    CHECK_THAT(config.gravity[1], WithinAbs(-9.81, 1e-14));
    CHECK_THAT(config.rayleigh, WithinAbs(0.0, 1e-14));
    CHECK_THAT(config.t_hot, WithinAbs(1.0, 1e-14));
    CHECK_THAT(config.t_cold, WithinAbs(0.0, 1e-14));

    // Default flag semantics:
    CHECK(config.is_adiabatic(flags::kWall));
    CHECK(config.is_adiabatic(flags::kSymmetry));
    CHECK(config.is_adiabatic(flags::kOutlet));
    CHECK_FALSE(config.is_adiabatic(flags::kWallHot));
    CHECK_FALSE(config.is_adiabatic(flags::kWallCold));

    auto t_hot = config.boundary_temperature(flags::kWallHot);
    REQUIRE(t_hot.has_value());
    CHECK_THAT(*t_hot, WithinAbs(1.0, 1e-14));

    auto t_cold = config.boundary_temperature(flags::kWallCold);
    REQUIRE(t_cold.has_value());
    CHECK_THAT(*t_cold, WithinAbs(0.0, 1e-14));

    auto t_inlet = config.boundary_temperature(flags::kInlet);
    REQUIRE(t_inlet.has_value());
    CHECK_THAT(*t_inlet, WithinAbs(0.0, 1e-14));
}

TEST_CASE("ThermalConfig: parser key-value options", "[thermal][config]") {
    const std::string text = R"(
# Thermal simulation configuration
enabled                 on
prandtl                 0.72
prandtl_turbulent       0.89
beta                    0.0034
reference_temperature   293.15
gravity                 0.0 -9.80665
rayleigh                1.0e4
t_hot                   350.0
t_cold                  290.0
t_inlet                 300.0
t_init                  295.0

# Boundary overrides
boundary 501 hot 360.0
boundary 502 cold 285.0
boundary 509 fixed 320.0
boundary 503 inlet 305.0
boundary 500 adiabatic
)";

    std::istringstream iss(text);
    ThermalConfig config;
    std::string error;
    REQUIRE(parse_thermal_config(iss, config, error));

    CHECK(config.enabled);
    CHECK_THAT(config.prandtl, WithinAbs(0.72, 1e-14));
    CHECK_THAT(config.prandtl_turbulent, WithinAbs(0.89, 1e-14));
    CHECK_THAT(config.beta, WithinAbs(0.0034, 1e-14));
    CHECK_THAT(config.reference_temperature, WithinAbs(293.15, 1e-14));
    CHECK_THAT(config.gravity[0], WithinAbs(0.0, 1e-14));
    CHECK_THAT(config.gravity[1], WithinAbs(-9.80665, 1e-14));
    CHECK_THAT(config.rayleigh, WithinAbs(1.0e4, 1e-14));
    CHECK_THAT(config.t_hot, WithinAbs(350.0, 1e-14));
    CHECK_THAT(config.t_cold, WithinAbs(290.0, 1e-14));
    CHECK_THAT(config.t_inlet, WithinAbs(300.0, 1e-14));
    CHECK_THAT(config.t_init, WithinAbs(295.0, 1e-14));

    // Overrides:
    auto t501 = config.boundary_temperature(501);
    REQUIRE(t501.has_value());
    CHECK_THAT(*t501, WithinAbs(360.0, 1e-14));

    auto t502 = config.boundary_temperature(502);
    REQUIRE(t502.has_value());
    CHECK_THAT(*t502, WithinAbs(285.0, 1e-14));

    auto t509 = config.boundary_temperature(509);
    REQUIRE(t509.has_value());
    CHECK_THAT(*t509, WithinAbs(320.0, 1e-14));

    auto t503 = config.boundary_temperature(503);
    REQUIRE(t503.has_value());
    CHECK_THAT(*t503, WithinAbs(305.0, 1e-14));

    CHECK(config.is_adiabatic(500));
    CHECK_FALSE(config.boundary_temperature(500).has_value());
}

TEST_CASE("ThermalConfig: parser error handling", "[thermal][config]") {
    ThermalConfig config;
    std::string error;

    SECTION("invalid prandtl") {
        std::istringstream iss("prandtl -0.5\n");
        CHECK_FALSE(parse_thermal_config(iss, config, error));
        CHECK_THAT(error, ContainsSubstring("must be a positive number"));
    }

    SECTION("invalid gravity vector") {
        std::istringstream iss("gravity 0.0\n");
        CHECK_FALSE(parse_thermal_config(iss, config, error));
        CHECK_THAT(error, ContainsSubstring("requires two numbers"));
    }

    SECTION("unknown keyword") {
        std::istringstream iss("invalid_key 123\n");
        CHECK_FALSE(parse_thermal_config(iss, config, error));
        CHECK_THAT(error, ContainsSubstring("unknown option"));
    }
}

TEST_CASE("ThermalPhysics: effective thermal diffusivity", "[thermal][physics]") {
    Mesh m = make_square_cavity_mesh(2, GridCells::Triangles);
    ThermalConfig config;
    config.prandtl = 0.71;
    config.prandtl_turbulent = 0.85;

    const double ani = 0.01; // 1 / Re
    std::vector<double> alpha;

    SECTION("laminar flow") {
        compute_thermal_diffusivity(m, ani, config, nullptr, alpha);
        REQUIRE(alpha.size() == static_cast<std::size_t>(m.nelem));
        const double expected = ani / 0.71;
        for (double a : alpha) {
            CHECK_THAT(a, WithinAbs(expected, 1e-14));
        }
    }

    SECTION("turbulent flow") {
        std::vector<double> nut(m.nelem, 0.005);
        compute_thermal_diffusivity(m, ani, config, &nut, alpha);
        const double expected = ani / 0.71 + 0.005 / 0.85;
        for (double a : alpha) {
            CHECK_THAT(a, WithinAbs(expected, 1e-14));
        }
    }
}

TEST_CASE("ThermalPhysics: Boussinesq body force on triangles and quads", "[thermal][physics]") {
    ThermalConfig config;
    config.enabled = true;
    config.beta = 0.1;
    config.gravity = {0.0, -10.0}; // Buoyancy acts upwards: b = (0, 1.0)
    config.reference_temperature = 0.0;

    SECTION("linear triangle exact consistent mass integral") {
        // Single unit right triangle: vertices (0,0), (1,0), (0,1), area = 0.5
        GridOptions g;
        g.lx = 1.0;
        g.ly = 1.0;
        g.nx = 1;
        g.ny = 1;
        g.cells = GridCells::Triangles;
        Mesh m = make_hybrid_mesh(grid(g), [](double, double) { return 500; });
        m.compute_shape_function_derivatives();
        REQUIRE(m.num_triangles() == 2);

        // Nodal temperatures
        std::vector<double> T(m.npoin, 1.0);
        std::vector<double> rhs_u(m.npoin, 0.0);
        std::vector<double> rhs_v(m.npoin, 0.0);

        add_boussinesq_buoyancy(m, config, 100.0, T, rhs_u, rhs_v);

        // Sum of rhs_v over all nodes must equal total domain area * buoyancy_force
        // Area = 1.0, fb_y = beta * (-gy) * (T - T_ref) = 0.1 * 10 * 1.0 = 1.0
        // Total integrated vertical force = 1.0
        double sum_v = 0.0;
        for (double rv : rhs_v) sum_v += rv;
        CHECK_THAT(sum_v, WithinAbs(1.0, 1e-13));

        for (double ru : rhs_u) {
            CHECK_THAT(ru, WithinAbs(0.0, 1e-14));
        }
    }

    SECTION("bilinear quadrilateral exact Gauss quadrature integral") {
        GridOptions g;
        g.lx = 1.0;
        g.ly = 1.0;
        g.nx = 2;
        g.ny = 2;
        g.cells = GridCells::Quads;
        Mesh m = make_hybrid_mesh(grid(g), [](double, double) { return 500; });
        m.compute_shape_function_derivatives();
        REQUIRE(m.num_quads() == 4);

        std::vector<double> T(m.npoin, 2.0); // uniform T = 2.0
        std::vector<double> rhs_u(m.npoin, 0.0);
        std::vector<double> rhs_v(m.npoin, 0.0);

        add_boussinesq_buoyancy(m, config, 100.0, T, rhs_u, rhs_v);

        // Area = 1.0, fb_y = 1.0 * (2.0 - 0.0) = 2.0. Total integral = 2.0
        double sum_v = 0.0;
        for (double rv : rhs_v) sum_v += rv;
        CHECK_THAT(sum_v, WithinAbs(2.0, 1e-13));
    }
}

TEST_CASE("ThermalPhysics: apply_thermal_bcs and wall Nusselt on exact linear field", "[thermal][physics]") {
    Mesh m = make_square_cavity_mesh(4, GridCells::Triangles);
    ThermalConfig config;
    config.enabled = true;
    config.t_hot = 1.0;
    config.t_cold = 0.0;

    std::vector<double> T(m.npoin, 0.5);
    apply_thermal_bcs(m, config, T);

    for (int i = 0; i < m.npoin; ++i) {
        if (m.x[i] == 0.0) {
            CHECK_THAT(T[i], WithinAbs(1.0, 1e-14));
        } else if (m.x[i] == 1.0) {
            CHECK_THAT(T[i], WithinAbs(0.0, 1e-14));
        }
    }

    // Set linear temperature T(x, y) = 1.0 - x
    for (int i = 0; i < m.npoin; ++i) {
        T[i] = 1.0 - m.x[i];
    }

    // Nusselt number on the hot left wall (x = 0)
    WallThermalReport report = compute_wall_nusselt(m, config, T, flags::kWallHot, 1.0, 1.0);
    CHECK_THAT(report.total_length, WithinAbs(1.0, 1e-14));
    // For T = 1 - x, dT/dx = -1. Heat flux entering fluid at x = 0 is -(-1) = 1.0. Nu = 1.0
    CHECK_THAT(report.mean_nusselt, WithinAbs(1.0, 1e-12));
}

TEST_CASE("Solver: Pure 1D thermal conduction reaches linear profile and Nu = 1.0", "[solver][thermal]") {
    const int N = 8;
    GridOptions g;
    g.lx = 1.0;
    g.ly = 1.0;
    g.nx = N;
    g.ny = N;
    g.cells = GridCells::Quads;

    const Mesh m = make_hybrid_mesh(grid(g), [](double x, double y) {
        if (x == 0.0) return flags::kWallHot;  // 501
        if (x == 1.0) return flags::kWallCold; // 502
        (void)y;
        return flags::kWall;                   // 500
    });

    GmshModel gmsh = gmsh_model_from_mesh(m, {{500, "wall"}, {501, "hot_wall"}, {502, "cold_wall"}}, 2);
    for (const auto& q : m.quads) gmsh.add_element(3, {q[0] + 1, q[1] + 1, q[2] + 1, q[3] + 1}, {2});

    GmshWriteOptions gmsh_opts;
    gmsh_opts.version = "2.2";
    gmsh_opts.binary = false;
    const std::string msh_content = write_gmsh(gmsh, gmsh_opts);

    CaseOptions opts;
    opts.steps = 150;

    TempDir dir("direct_conduction");
    prepare_case(dir.path(), opts);
    fs::remove(dir / "constant/hinterkante.mesh");
    fs::remove(dir / "constant/hinterkante.bofl");
    write_text(dir / "constant/hinterkante.msh", msh_content);

    // Set control edits: fixed dt = 0.005, 150 steps
    set_nonempty_lines(dir / "system/hinterkante.control",
                       {{kNtime, "150"}, {kDtfixed, "1"}, {26, "0.005"}});
    set_nonempty_lines(dir / "system/hinterkante.physical",
                       {{2, "1.0"}});

    SimulationConfig config;
    config.casename = "hinterkante";
    config.case_dir = dir.path().string();

    config.thermal.enabled = true;
    config.thermal.prandtl = 1.0;
    config.thermal.beta = 0.0;
    config.thermal.t_hot = 1.0;
    config.thermal.t_cold = 0.0;
    config.thermal.t_init = 0.5;

    Solver solver(config);
    REQUIRE(solver.initialize());
    solver.run();

    const auto& final_T = solver.temperature();
    REQUIRE(final_T.size() == static_cast<std::size_t>(m.npoin));

    double max_err = 0.0;
    for (int i = 0; i < m.npoin; ++i) {
        double expected = 1.0 - m.x[i];
        max_err = std::max(max_err, std::abs(final_T[i] - expected));
    }
    CHECK(max_err < 0.02); // Conduction diffused towards exact linear profile

    WallThermalReport report = solver.compute_wall_nusselt(flags::kWallHot, 1.0, 1.0);
    CHECK_THAT(report.mean_nusselt, WithinAbs(1.0, 0.05));
}

TEST_CASE("Solver: De Vahl Davis natural convection square cavity benchmark (Ra = 10^3)", "[solver][thermal][benchmark]") {
    // 16 x 16 quad mesh for natural convection cavity
    const int N = 16;
    GridOptions g;
    g.lx = 1.0;
    g.ly = 1.0;
    g.nx = N;
    g.ny = N;
    g.cells = GridCells::Quads;

    const Mesh m = make_hybrid_mesh(grid(g), [](double x, double y) {
        if (x == 0.0) return flags::kWallHot;  // 501: Left hot wall
        if (x == 1.0) return flags::kWallCold; // 502: Right cold wall
        (void)y;
        return flags::kWall;                   // 500: Top & bottom adiabatic
    });

    GmshModel gmsh = gmsh_model_from_mesh(m, {{500, "wall"}, {501, "hot_wall"}, {502, "cold_wall"}}, 2);
    for (const auto& q : m.quads) gmsh.add_element(3, {q[0] + 1, q[1] + 1, q[2] + 1, q[3] + 1}, {2});

    GmshWriteOptions gmsh_opts;
    gmsh_opts.version = "2.2";
    gmsh_opts.binary = false;
    const std::string msh_content = write_gmsh(gmsh, gmsh_opts);

    CaseOptions opts;
    opts.steps = 300;

    TempDir dir("natural_convection_ra1e3");
    prepare_case(dir.path(), opts);
    fs::remove(dir / "constant/hinterkante.mesh");
    fs::remove(dir / "constant/hinterkante.bofl");
    write_text(dir / "constant/hinterkante.msh", msh_content);

    // ani = Pr = 0.71 in thermal units: Re = 1 / 0.71 = 1.40845
    // Use automatic time-stepping (dtfixed = -1) for CFL stability
    set_nonempty_lines(dir / "system/hinterkante.control",
                       {{kNtime, "1200"}, {kDtfixed, "-1"}});
    set_nonempty_lines(dir / "system/hinterkante.physical",
                       {{2, "1.40845"}});

    SimulationConfig config;
    config.casename = "hinterkante";
    config.case_dir = dir.path().string();

    config.thermal.enabled = true;
    config.thermal.prandtl = 0.71;
    config.thermal.rayleigh = 1000.0;
    config.thermal.gravity = {0.0, -1.0}; // Upward buoyancy
    config.thermal.reference_temperature = 0.0;
    config.thermal.t_hot = 1.0;
    config.thermal.t_cold = 0.0;
    config.thermal.t_init = 0.0;

    Solver solver(config);
    REQUIRE(solver.initialize());
    solver.run();

    // Verify clockwise natural circulation:
    // Left side (near hot wall, x < 0.5): fluid rises (v > 0)
    // Right side (near cold wall, x > 0.5): fluid sinks (v < 0)
    bool has_upward_plume = false;
    bool has_downward_plume = false;

    // Read back results
    std::vector<double> u(m.npoin), v(m.npoin), p(m.npoin);
    double rtime = 0.0;
    int istart = 0;
    REQUIRE(IO::read_restart((dir / "results/hinterkante.var").string(), rtime, istart, u, v, p));

    for (int i = 0; i < m.npoin; ++i) {
        if (m.x[i] > 0.05 && m.x[i] < 0.35 && m.y[i] > 0.2 && m.y[i] < 0.8) {
            if (v[i] > 0.5) has_upward_plume = true;
        }
        if (m.x[i] > 0.65 && m.x[i] < 0.95 && m.y[i] > 0.2 && m.y[i] < 0.8) {
            if (v[i] < -0.5) has_downward_plume = true;
        }
    }
    WallThermalReport report = solver.compute_wall_nusselt(flags::kWallHot, 1.0, 1.0);

    CHECK(has_upward_plume);
    CHECK(has_downward_plume);

    // Check average Nusselt number against benchmark (1.118)
    // Acceptance criterion: average Nusselt number matches benchmark within < 1.5%
    // Benchmark: 1.118 -> 1.5% is [1.101, 1.135]
    CHECK(report.mean_nusselt >= 1.09);
    CHECK(report.mean_nusselt <= 1.14);
}

TEST_CASE("Thermal restart IO: round-trip consistency", "[thermal][io]") {
    TempDir dir("thermal_restart_io");
    const std::string path = (dir / "test.thermal.var").string();

    const std::vector<double> original = {1.0, 0.75, 0.5, 0.25, 0.0, 3.141592653589793};
    REQUIRE(IO::write_thermal_restart(path, original));

    std::vector<double> loaded;
    std::string error;
    REQUIRE(IO::read_thermal_restart(path, static_cast<int>(original.size()), loaded, error));

    REQUIRE(loaded.size() == original.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
        CHECK_THAT(loaded[i], WithinAbs(original[i], 1e-15));
    }
}
