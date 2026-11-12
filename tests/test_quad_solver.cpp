// The solver on meshes with quadrilaterals:
//   * fully developed laminar channel flow (Poiseuille) on quadrilateral,
//     triangle and hybrid (inflation-layer) meshes of the same channel;
//   * the hybrid inflation-layer mesh written by Gmsh, with its boundary
//     conditions, laminar and with every turbulence model;
//   * determinism.

#include "MeshCaseTestUtils.hpp"
#include "QuadTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <map>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace {

Mesh channel(GridCells cells, int nx, int ny, double length) {
    GridOptions g;
    g.lx = length;
    g.ly = 1.0;
    g.nx = nx;
    g.ny = ny;
    g.cells = cells;
    g.layers = 2;
    return make_hybrid_mesh(grid(g), [length](double x, double y) {
        if (x == 0.0) return 503;
        if (std::abs(x - length) < 1e-12) return 504;
        (void)y;
        return 500;
    });
}

} // namespace

TEST_CASE("Solver: fully developed channel flow on quadrilateral, triangle and hybrid meshes", "[quad][solver]") {
    const double length = 6.0;
    std::map<int, double> centre; // cells -> outlet centreline velocity
    for (const GridCells cells : {GridCells::Quads, GridCells::InflationLayers, GridCells::Triangles}) {
        CAPTURE(static_cast<int>(cells));
        const Mesh m = channel(cells, 48, 8, length);
        CaseOptions opts;
        opts.steps = 6000;
        opts.transient = 0; // steady: local time steps, stops when converged
        opts.dtfixed = 0;
        const MeshCaseResult r = run_mesh_case("quad-channel", gmsh_file(m), opts, {{{2, "20.0"}}, {}, {}, false}); // Re = 20
        INFO(r.run.errors);
        REQUIRE(r.run.initialized);
        REQUIRE(r.run.state_read);
        REQUIRE_FALSE(r.run.steps.empty());
        CHECK(r.run.steps.back().step < 6000); // converged before the step limit
        CHECK(count_non_finite(r.run.u) + count_non_finite(r.run.v) + count_non_finite(r.run.p) == 0);

        // Outlet: fully developed (Poiseuille) flow. At the inlet corners the
        // wall wins (u = 0), so the discrete inflow is the piecewise-linear
        // profile 0, 1, ..., 1, 0 with flux Q = 1 - h (h = 1/8). A parabolic
        // nodal profile carrying that flux (trapezoidal rule = exact flux of
        // the piecewise-linear field) is u_i = 6 Q / (1 - h^2) y (1 - y)
        // = 6 y (1 - y) / (1 + h).
        std::vector<std::pair<double, double>> profile; // (y, u)
        for (int i = 0; i < r.mesh.npoin; ++i)
            if (std::abs(r.mesh.x[static_cast<std::size_t>(i)] - length) < 1e-12)
                profile.push_back({r.mesh.y[static_cast<std::size_t>(i)], r.run.u[static_cast<std::size_t>(i)]});
        std::sort(profile.begin(), profile.end());
        REQUIRE(profile.size() == 9);
        const double h = 1.0 / 8.0;
        double flux = 0.0, worst = 0.0;
        for (std::size_t k = 0; k < profile.size(); ++k) {
            const auto [y, u] = profile[k];
            worst = std::max(worst, std::abs(u - 6.0 * y * (1.0 - y) / (1.0 + h)));
            if (k > 0) flux += 0.5 * (u + profile[k - 1].second) * (y - profile[k - 1].first);
        }
        CAPTURE(worst, flux);
        CHECK(worst < 0.02);
        CHECK_THAT(flux, WithinAbs(1.0 - h, 0.01)); // mass conservation
        centre[static_cast<int>(cells)] = profile[4].second;
        CHECK(max_abs(r.run.v) < 0.2);
        CHECK_THAT(r.run.log, ContainsSubstring("Mesh import: "));
    }
    CHECK_THAT(centre[static_cast<int>(GridCells::Quads)], WithinAbs(centre[static_cast<int>(GridCells::Triangles)], 0.03));
    CHECK_THAT(centre[static_cast<int>(GridCells::InflationLayers)], WithinAbs(centre[static_cast<int>(GridCells::Triangles)], 0.03));
}

TEST_CASE("Solver: the Gmsh inflation-layer mesh runs with its boundary conditions", "[quad][solver][fixtures]") {
    CaseOptions opts;
    opts.steps = 60;
    const MeshCaseResult r = run_mesh_case("quad-inflation", data_file("cylinder_inflation_4.1_binary.msh"), opts);
    INFO(r.run.errors);
    REQUIRE(r.run.initialized);
    REQUIRE(r.run.state_read);
    CHECK_THAT(r.run.log, ContainsSubstring("318 triangles, 48 quadrilaterals"));
    CHECK(count_non_finite(r.run.u) + count_non_finite(r.run.v) + count_non_finite(r.run.p) == 0);
    const Mesh& m = r.mesh;
    for (const BoundarySide& s : m.iside) {
        for (int ip : {s.node1, s.node2}) {
            CAPTURE(s.flag, ip);
            if (s.flag == 500) {
                CHECK(r.run.u[static_cast<std::size_t>(ip)] == 0.0);
                CHECK(r.run.v[static_cast<std::size_t>(ip)] == 0.0);
            } else if (s.flag == 503) {
                CHECK(r.run.u[static_cast<std::size_t>(ip)] == 1.0);
            } else if (s.flag == 506) {
                CHECK(std::abs(r.run.v[static_cast<std::size_t>(ip)]) < 1e-15);
            } else if (s.flag == 504) {
                CHECK(r.run.p[static_cast<std::size_t>(ip)] == 0.0);
            }
        }
    }
    CHECK(max_abs(r.run.u) > 1.0); // accelerated around the cylinder
    CHECK(max_abs(r.run.u) < 3.0);
    // Deterministic: the same run gives the same fields.
    const MeshCaseResult again = run_mesh_case("quad-inflation-2", data_file("cylinder_inflation_4.1_binary.msh"), opts);
    CHECK(again.run.u == r.run.u);
    CHECK(again.run.p == r.run.p);
}

TEST_CASE("Solver: turbulence models on the hybrid inflation-layer mesh", "[quad][solver][turbulence]") {
    for (const char* model : {"spalart-allmaras", "k-omega-sst", "k-epsilon", "launder-sharma-k-epsilon"}) {
        CAPTURE(model);
        CaseOptions opts;
        opts.steps = 20;
        opts.turbulence = std::string("model ") + model + "\n";
        const MeshCaseResult r = run_mesh_case("quad-turbulence", data_file("cylinder_inflation_4.1_binary.msh"), opts);
        INFO(r.run.errors);
        REQUIRE(r.run.initialized);
        REQUIRE(r.run.state_read);
        CHECK_THAT(r.run.log, ContainsSubstring("Turbulence model: " + std::string(model)));
        CHECK_THAT(r.run.log, ContainsSubstring("Wall faces: 16"));
        CHECK(count_non_finite(r.run.u) + count_non_finite(r.run.v) + count_non_finite(r.run.p) == 0);
        REQUIRE(r.run.turb_read);
        for (const std::vector<double>& field : r.run.turb.values) CHECK(count_non_finite(field) == 0);
        CHECK_THAT(r.run.log, ContainsSubstring("y+ (first cell) min"));
    }
}
