// Fully developed (parabolic) inflow, flag 508 / boundary type
// inlet_parabolic: the profile of parabolic_inlet_velocities(), the order of
// apply_velocity_bcs(), invalid inlets, and a solver run that keeps a
// Poiseuille flow fully developed.

#include "BoundaryConditions.hpp"
#include "MeshCaseTestUtils.hpp"
#include "QuadTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace {

// Channel [0, L] x [0, H]; the flag of a side from its midpoint.
Mesh channel(GridCells cells, int nx, int ny, double length, double height,
             const std::function<int(double, double)>& flag_of) {
    GridOptions g;
    g.lx = length;
    g.ly = height;
    g.nx = nx;
    g.ny = ny;
    g.cells = cells;
    return make_hybrid_mesh(grid(g), flag_of);
}

std::map<int, std::pair<double, double>> by_node(const std::vector<NodeVelocity>& list) {
    std::map<int, std::pair<double, double>> out;
    for (const NodeVelocity& nv : list) out[nv.node] = {nv.u, nv.v};
    return out;
}

double profile(double s, double length) { return 6.0 * s * (length - s) / (length * length); }

} // namespace

TEST_CASE("Parabolic inlet: Poiseuille profile with mean velocity 1 along the inward normal", "[bc][inlet_parabolic]") {
    const GridCells cells = GENERATE(GridCells::Triangles, GridCells::Quads);
    const double H = 2.5;
    SECTION("left boundary: flow in +x") {
        const Mesh m = channel(cells, 4, 10, 3.0, H, [](double x, double) { return x == 0.0 ? 508 : 500; });
        const auto v = by_node(parabolic_inlet_velocities(m));
        REQUIRE(v.size() == 11);
        double peak = 0.0;
        for (const auto& [node, uv] : v) {
            CHECK(m.x[node] == 0.0);
            CHECK_THAT(uv.first, WithinAbs(profile(m.y[node], H), 1e-14));
            CHECK(uv.second == 0.0);
            peak = std::max(peak, uv.first);
        }
        CHECK_THAT(peak, WithinAbs(1.5, 1e-14)); // node at the centre
        // Flux of the piecewise-linear nodal profile (trapezoidal rule):
        // H (1 - 1 / n^2) for a parabola of mean 1 sampled at n + 1 points.
        double flux = 0.0;
        for (int j = 0; j < 10; ++j) flux += 0.5 * (profile(j * H / 10, H) + profile((j + 1) * H / 10, H)) * H / 10;
        CHECK_THAT(flux, WithinAbs(H * (1.0 - 1.0 / 100.0), 1e-13));
    }
    SECTION("right boundary: flow in -x") {
        const Mesh m = channel(cells, 4, 10, 3.0, H, [](double x, double) { return x > 2.9 ? 508 : 500; });
        for (const auto& [node, uv] : by_node(parabolic_inlet_velocities(m))) {
            CHECK_THAT(uv.first, WithinAbs(-profile(m.y[node], H), 1e-14));
            CHECK(uv.second == 0.0);
        }
    }
    SECTION("rotated mesh: along the rotated normal") {
        Mesh m = channel(cells, 4, 10, 3.0, H, [](double x, double) { return x == 0.0 ? 508 : 500; });
        const Mesh ref = m;
        const double angle = 0.6, c = std::cos(angle), s = std::sin(angle);
        for (int i = 0; i < m.npoin; ++i) {
            m.x[i] = c * ref.x[i] - s * ref.y[i] + 1.0;
            m.y[i] = s * ref.x[i] + c * ref.y[i] - 2.0;
        }
        for (const auto& [node, uv] : by_node(parabolic_inlet_velocities(m))) {
            const double speed = profile(ref.y[node], H);
            CHECK_THAT(uv.first, WithinAbs(speed * c, 1e-13));
            CHECK_THAT(uv.second, WithinAbs(speed * s, 1e-13));
        }
    }
    SECTION("independent of the node order of the sides") {
        Mesh m = channel(cells, 4, 10, 3.0, H, [](double x, double) { return x == 0.0 ? 508 : 500; });
        const auto a = by_node(parabolic_inlet_velocities(m));
        for (BoundarySide& side : m.iside) std::swap(side.node1, side.node2);
        const auto b = by_node(parabolic_inlet_velocities(m));
        REQUIRE(a.size() == b.size());
        for (const auto& [node, uv] : a) {
            CHECK_THAT(b.at(node).first, WithinAbs(uv.first, 1e-14));
            CHECK(b.at(node).second == uv.second);
        }
    }
}

TEST_CASE("Parabolic inlet: separate inlet segments get their own parabola", "[bc][inlet_parabolic]") {
    // Left boundary y in [0, 1]: inlet on [0, 0.5] and [0.75, 1], wall between.
    const Mesh m = channel(GridCells::Quads, 2, 8, 1.0, 1.0, [](double x, double y) {
        return (x == 0.0 && (y < 0.5 || y > 0.75)) ? 508 : 500;
    });
    const auto v = by_node(parabolic_inlet_velocities(m));
    REQUIRE(v.size() == 5 + 3);
    for (const auto& [node, uv] : v) {
        const double y = m.y[node];
        const double expected = y <= 0.5 ? profile(y, 0.5) : profile(y - 0.75, 0.25);
        CHECK_THAT(uv.first, WithinAbs(expected, 1e-14));
    }
    CHECK_THAT(v.at(2 * 3).first, WithinAbs(1.5, 1e-14)); // node (0, 0.25): centre of the lower segment
}

TEST_CASE("Parabolic inlet: invalid inlets are reported", "[bc][inlet_parabolic][errors]") {
    SECTION("closed curve") {
        const Mesh m = channel(GridCells::Quads, 2, 2, 1.0, 1.0, [](double, double) { return 508; });
        CHECK_THROWS_WITH(parabolic_inlet_velocities(m), ContainsSubstring("is a closed curve; a parabolic inlet needs two ends"));
    }
    SECTION("branching") {
        // Two triangles touching at one node: four inlet sides meet there.
        HybridSpec s;
        s.nodes = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}};
        s.triangles = {{0, 1, 2}, {0, 3, 4}};
        const Mesh m = make_hybrid_mesh(s, [](double, double) { return 508; });
        CHECK_THROWS_WITH(parabolic_inlet_velocities(m), ContainsSubstring("branches at node 0 (0.000000, 0.000000): it is in 4 inlet sides"));
    }
    SECTION("zero-length side") {
        Mesh m = channel(GridCells::Quads, 2, 2, 1.0, 1.0, [](double x, double) { return x == 0.0 ? 508 : 500; });
        for (const BoundarySide& side : m.iside)
            if (side.flag == 508) { // two nodes at one position
                m.x[side.node2] = m.x[side.node1];
                m.y[side.node2] = m.y[side.node1];
                break;
            }
        CHECK_THROWS_WITH(parabolic_inlet_velocities(m), ContainsSubstring("has zero length"));
    }
    SECTION("no inlet: empty") {
        const Mesh m = channel(GridCells::Quads, 2, 2, 1.0, 1.0, [](double, double) { return 500; });
        CHECK(parabolic_inlet_velocities(m).empty());
    }
}

TEST_CASE("apply_velocity_bcs: order 507, 503, 508, walls", "[bc][inlet_parabolic]") {
    // Left: 508 inlet; bottom: 503 inlet; top: wall; right: outlet.
    Mesh m = channel(GridCells::Triangles, 3, 4, 3.0, 1.0, [](double x, double y) {
        if (x == 0.0) return 508;
        if (y == 0.0) return 503;
        if (y == 1.0) return 500;
        return 504;
    });
    compute_all_geometry(m);
    std::vector<double> u(static_cast<std::size_t>(m.npoin), 42.0), v(u.size(), 42.0);
    apply_velocity_bcs(m, u, v);
    auto node = [&](double x, double y) {
        for (int i = 0; i < m.npoin; ++i)
            if (m.x[i] == x && m.y[i] == y) return static_cast<std::size_t>(i);
        FAIL("no node");
        return std::size_t{0};
    };
    CHECK(u[node(0.0, 0.0)] == 0.0);  // 503 then 508: the end of the parabola wins
    CHECK(u[node(1.0, 0.0)] == 1.0);  // 503
    CHECK(u[node(0.0, 1.0)] == 0.0);  // 508 end and wall
    CHECK(u[node(0.0, 0.5)] == 1.5);  // 508 centre
    CHECK(v[node(0.0, 0.5)] == 0.0);
    CHECK(u[node(2.0, 1.0)] == 0.0);  // wall
    CHECK(u[node(3.0, 0.5)] == 42.0); // outlet: untouched
}

TEST_CASE("Solver: a parabolic inlet keeps a channel flow fully developed", "[bc][inlet_parabolic][solver]") {
    const GridCells cells = GENERATE(GridCells::Triangles, GridCells::InflationLayers);
    CAPTURE(static_cast<int>(cells));
    const double length = 4.0;
    GridOptions g;
    g.lx = length;
    g.nx = 24;
    g.ny = 8;
    g.layers = 2;
    g.cells = cells;
    const Mesh m = make_hybrid_mesh(grid(g), [length](double x, double) {
        if (x == 0.0) return 508;
        if (std::abs(x - length) < 1e-12) return 504;
        return 500;
    });
    CaseOptions opts;
    opts.steps = 20000;
    opts.transient = 0;
    opts.dtfixed = 0;
    MeshCaseOptions extra;
    extra.physical_edits = {{2, "20.0"}}; // Re = 20
    extra.control_edits = {{kL2Norm, "1.0d-12"}, {kIwrite, "50"}};
    const MeshCaseResult r = run_mesh_case("parabolic-channel", gmsh_file(m), opts, extra);
    INFO(r.run.errors);
    REQUIRE(r.run.initialized);
    REQUIRE(r.run.state_read);
    CHECK_THAT(r.run.log, ContainsSubstring("Boundary 'parabolic_inlet' (tag 508): inlet_parabolic, flag 508, 8 sides [name]"));
    CHECK_THAT(r.run.log, ContainsSubstring("Parabolic inlet (508): 9 nodes, peak velocity 1.5"));
    CHECK(r.run.steps.back().step < 20000);
    // The inlet nodes carry the exact profile. Downstream the flow stays
    // fully developed within the discretisation error of the grid (the
    // alternating diagonals make the discrete Laplacian inexact for
    // quadratics at every other node: errors alternate in y, < 1.7 % of the
    // peak), with the inflow mass flux and mirror symmetry about y = 1/2.
    double worst_downstream = 0.0;
    std::map<double, std::map<double, std::pair<double, double>>> columns; // x -> y -> (u, v)
    for (int i = 0; i < r.mesh.npoin; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        const double x = r.mesh.x[k], y = r.mesh.y[k];
        columns[x][y] = {r.run.u[k], r.run.v[k]};
        if (x == 0.0) CHECK_THAT(r.run.u[k], WithinAbs(6.0 * y * (1.0 - y), 1e-14));
        if (x >= 1.0) worst_downstream = std::max(worst_downstream, std::abs(r.run.u[k] - 6.0 * y * (1.0 - y)));
    }
    CAPTURE(worst_downstream);
    CHECK(worst_downstream < 0.025);
    auto flux = [](const std::map<double, std::pair<double, double>>& column) {
        double q = 0.0;
        for (auto it = std::next(column.begin()); it != column.end(); ++it)
            q += 0.5 * (it->second.first + std::prev(it)->second.first) * (it->first - std::prev(it)->first);
        return q;
    };
    const double inflow = flux(columns.begin()->second);
    CHECK_THAT(inflow, WithinAbs(1.0 - 1.0 / 64.0, 1e-14)); // trapezoidal flux of the nodal parabola, 8 cells
    CHECK_THAT(flux(columns.rbegin()->second), WithinAbs(inflow, 0.01 * inflow));
    for (const auto& [x, column] : columns)
        for (const auto& [y, uv] : column) {
            const auto mirror = column.find(1.0 - y);
            if (mirror == column.end()) continue; // (no node at 1 - y in rounding)
            CHECK_THAT(uv.first, WithinAbs(mirror->second.first, 1e-9));
            CHECK_THAT(uv.second, WithinAbs(-mirror->second.second, 1e-9));
        }
    CHECK(max_abs(r.run.v) < 0.015);
}
