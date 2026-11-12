// Point probes (post/Probes.hpp): location and interpolation on triangle,
// quadrilateral and hybrid meshes.

#include "QuadTestUtils.hpp"
#include "post/Probes.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;

namespace {

Mesh square(GridCells cells, double perturbation) {
    GridOptions g;
    g.x0 = -1.0;
    g.y0 = 0.5;
    g.lx = 2.0;
    g.ly = 1.5;
    g.nx = 5;
    g.ny = 6;
    g.layers = 2;
    g.cells = cells;
    g.perturbation = perturbation;
    return make_hybrid_mesh(grid(g));
}

template <typename F>
std::vector<double> nodal(const Mesh& m, F f) {
    std::vector<double> v(static_cast<std::size_t>(m.npoin));
    for (int i = 0; i < m.npoin; ++i) v[static_cast<std::size_t>(i)] = f(m.x[i], m.y[i]);
    return v;
}

} // namespace

TEST_CASE("Probes: linear fields are interpolated exactly everywhere in the mesh", "[post][probes]") {
    const GridCells cells = GENERATE(GridCells::Triangles, GridCells::Quads, GridCells::InflationLayers);
    const double perturbation = GENERATE(0.0, 0.2);
    CAPTURE(static_cast<int>(cells), perturbation);
    const Mesh m = square(cells, perturbation);
    auto f = [](double x, double y) { return 0.7 - 1.3 * x + 2.1 * y; };
    const std::vector<double> field = nodal(m, f);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> ux(-1.0, 1.0), uy(0.5, 2.0);
    for (int k = 0; k < 200; ++k) {
        const double x = ux(rng), y = uy(rng);
        const auto loc = post::locate_point(m, x, y);
        REQUIRE(loc.has_value());
        CHECK(loc->count == m.nodes_per_element(loc->element));
        double sum = 0.0;
        for (int a = 0; a < loc->count; ++a) {
            sum += loc->weights[a];
            CHECK(loc->weights[a] > -1e-9);
        }
        CHECK_THAT(sum, WithinAbs(1.0, 1e-12));
        CHECK_THAT(post::interpolate(*loc, field), WithinAbs(f(x, y), 1e-12));
    }
}

TEST_CASE("Probes: bilinear fields are exact on quadrilaterals, in reference coordinates", "[post][probes]") {
    // On a parallelogram grid the map is affine, so x*y is in the element space.
    HybridSpec s = grid(GridOptions{});
    for (Point& p : s.nodes) p.x += 0.3 * p.y; // shear: parallelograms
    const Mesh m = make_hybrid_mesh(s);
    auto f = [](double x, double y) { return 1.0 + x - 2.0 * y + 3.0 * (x - 0.3 * y) * y; }; // bilinear in (x - 0.3 y, y)
    const std::vector<double> field = nodal(m, f);
    for (const auto& [x, y] : std::vector<std::pair<double, double>>{{0.41, 0.37}, {0.9, 0.95}, {0.5, 0.11}}) {
        const auto loc = post::locate_point(m, x, y);
        REQUIRE(loc.has_value());
        CHECK_THAT(post::interpolate(*loc, field), WithinAbs(f(x, y), 1e-12));
    }
}

TEST_CASE("Probes: nodes, edges, the boundary and points outside", "[post][probes]") {
    const Mesh m = square(GridCells::InflationLayers, 0.15);
    // Every node is found, with weight 1 on itself.
    for (int i = 0; i < m.npoin; ++i) {
        const auto loc = post::locate_point(m, m.x[i], m.y[i]);
        REQUIRE(loc.has_value());
        double w = 0.0;
        for (int a = 0; a < loc->count; ++a)
            if (loc->nodes[a] == i) w = loc->weights[a];
        CHECK_THAT(w, WithinAbs(1.0, 1e-9));
    }
    // Midpoints of boundary sides.
    for (const BoundarySide& s : m.iside) {
        const double x = 0.5 * (m.x[s.node1] + m.x[s.node2]), y = 0.5 * (m.y[s.node1] + m.y[s.node2]);
        CHECK(post::locate_point(m, x, y).has_value());
    }
    // Outside.
    CHECK_FALSE(post::locate_point(m, -1.01, 1.0).has_value());
    CHECK_FALSE(post::locate_point(m, 0.0, 2.0 + 1e-6).has_value());
    CHECK_FALSE(post::locate_point(m, 5.0, 5.0).has_value());
    // Inside a hole.
    GridOptions g;
    g.nx = g.ny = 3;
    const Mesh holed = make_hybrid_mesh(without_cells(grid(g), [](double x, double y) {
        return x > 1.0 / 3.0 && x < 2.0 / 3.0 && y > 1.0 / 3.0 && y < 2.0 / 3.0;
    }));
    CHECK_FALSE(post::locate_point(holed, 0.5, 0.5).has_value());
    CHECK(post::locate_point(holed, 0.2, 0.5).has_value());
}
