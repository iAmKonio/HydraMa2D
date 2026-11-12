// Skin friction extraction along wall boundaries, wall shear stress tau_w,
// skin friction coefficient Cf(x), and automated recirculation / separation bubble detection.

#include "post/SkinFriction.hpp"
#include "MeshTestUtils.hpp"
#include "QuadTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <vector>

using namespace hydrama;
using namespace hydrama::post;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

Mesh channel_mesh(GridCells cells, int nx, int ny, double L, double H) {
    GridOptions g;
    g.lx = L;
    g.ly = H;
    g.nx = nx;
    g.ny = ny;
    g.cells = cells;
    return make_hybrid_mesh(grid(g), [=](double x, double y) {
        if (y == 0.0) return 500; // bottom wall
        if (y == H) return 509;   // top moving wall
        if (x == 0.0) return 503; // inlet
        if (x == L) return 504;   // outlet
        return 500;
    });
}

} // namespace

TEST_CASE("Skin friction: exact values for planar Couette flow", "[post][skin_friction]") {
    const GridCells cells = GENERATE(GridCells::Triangles, GridCells::Quads);
    const double L = 4.0;
    const double H = 1.0;
    const double U_top = 2.0;
    const double nu = 0.02;
    const double rho = 1.0;

    Mesh m = channel_mesh(cells, 8, 4, L, H);
    compute_all_geometry(m);
    m.determine_wall_nodes();

    // Couette velocity profile: u(x, y) = U_top * y / H, v = 0, p = 0
    std::vector<double> u(m.npoin), v(m.npoin, 0.0), p(m.npoin, 0.0);
    for (int i = 0; i < m.npoin; ++i) {
        u[i] = U_top * (m.y[i] / H);
    }

    // Theoretical values:
    // du/dy = U_top / H = 2.0
    // tau_w = mu * du/dy = rho * nu * 2.0 = 0.04
    // Cf = tau_w / (0.5 * rho * U_top^2) = 0.04 / (0.5 * 1.0 * 4.0) = 0.02
    const double expected_tau = nu * (U_top / H);
    const double expected_cf = expected_tau / (0.5 * rho * U_top * U_top);

    // Extract on bottom wall (sides with y == 0.0)
    std::vector<int> bottom_sides;
    for (int ib = 0; ib < m.nboun; ++ib) {
        int n1 = m.iside[ib].node1, n2 = m.iside[ib].node2;
        if (m.y[n1] == 0.0 && m.y[n2] == 0.0) {
            bottom_sides.push_back(ib);
        }
    }
    REQUIRE(bottom_sides.size() == 8);

    const SkinFrictionProfile prof = extract_skin_friction(m, bottom_sides, nu, u, v, p, U_top, 0.0, rho);
    REQUIRE(prof.samples.size() == 8);
    CHECK_THAT(prof.total_length, WithinAbs(L, 1e-12));

    for (const auto& sample : prof.samples) {
        CHECK_THAT(sample.y, WithinAbs(0.0, 1e-12));
        CHECK_THAT(sample.tau_w, WithinAbs(expected_tau, 1e-12));
        CHECK_THAT(sample.cf, WithinAbs(expected_cf, 1e-12));
        CHECK_THAT(sample.cp, WithinAbs(0.0, 1e-12));
        CHECK_THAT(sample.nx, WithinAbs(0.0, 1e-12));
        CHECK_THAT(sample.ny, WithinAbs(1.0, 1e-12)); // points into fluid (+y)
        CHECK_THAT(sample.tx, WithinAbs(1.0, 1e-12)); // points along +x
        CHECK_THAT(sample.ty, WithinAbs(0.0, 1e-12));
    }

    // Attached flow: no recirculation bubbles
    const auto bubbles = find_recirculation_bubbles(prof);
    CHECK(bubbles.empty());
    CHECK_THAT(format_bubble_report(bubbles), ContainsSubstring("flow fully attached"));
}

TEST_CASE("Recirculation bubble: root-finding for separation and reattachment", "[post][skin_friction][bubble]") {
    // Construct synthetic profile along bottom wall x in [0, 10]
    // with a known recirculation bubble between x = 2.0 and x = 7.5
    // Cf(x) = (x - 2.0) * (x - 7.5) * 0.01
    // For x in (2.0, 7.5), Cf(x) < 0 (reversed flow)
    // Separation point x_sep = 2.0, Reattachment point x_reatt = 7.5, Length = 5.5
    SkinFrictionProfile prof;
    const int N = 21; // x = 0, 0.5, 1.0, ..., 10.0
    for (int i = 0; i < N; ++i) {
        double x = i * 0.5;
        SkinFrictionSample s;
        s.x = x;
        s.y = 0.0;
        s.s = x;
        s.length = 0.5;
        s.tx = 1.0;
        s.ty = 0.0;
        s.nx = 0.0;
        s.ny = 1.0;
        s.cf = (x - 2.0) * (x - 7.5) * 0.01;
        s.tau_w = s.cf * 0.5;
        prof.samples.push_back(s);
    }
    prof.total_length = 10.0;

    const auto bubbles = find_recirculation_bubbles(prof);
    REQUIRE(bubbles.size() == 1);
    CHECK_THAT(bubbles[0].x_sep, WithinAbs(2.0, 1e-12));
    CHECK_THAT(bubbles[0].s_sep, WithinAbs(2.0, 1e-12));
    CHECK_THAT(bubbles[0].x_reatt, WithinAbs(7.5, 1e-12));
    CHECK_THAT(bubbles[0].s_reatt, WithinAbs(7.5, 1e-12));
    CHECK_THAT(bubbles[0].length, WithinAbs(5.5, 1e-12));

    // Peak negative skin friction: minimum of (x - 2)(x - 7.5) occurs at x = 4.75
    // Value = (2.75) * (-2.75) * 0.01 = -0.075625
    CHECK(bubbles[0].cf_min < 0.0);
    CHECK_THAT(bubbles[0].x_cf_min, WithinAbs(4.75, 0.5)); // nearest sample is 4.5 or 5.0
    CHECK(bubbles[0].reverse_flux > 0.0);

    const std::string report = format_bubble_report(bubbles, "bottom_wall");
    CHECK_THAT(report, ContainsSubstring("Bubble #1"));
    CHECK_THAT(report, ContainsSubstring("2.00000"));
    CHECK_THAT(report, ContainsSubstring("7.50000"));
    CHECK_THAT(report, ContainsSubstring("5.50000"));
}

TEST_CASE("Backward-facing step reattachment length benchmark", "[post][skin_friction][bfs]") {
    // Step height h = 1.0. At expansion step x = 0, flow separates immediately at x_sep = 0.
    // In laminar flow at Re_h ~ 100-130 (Armaly et al. 1983, Gartling 1990),
    // reattachment length x_r / h ~ 6.0 (+/- 0.1)
    const double h = 1.0;
    const double expected_xr = 6.0 * h;

    SkinFrictionProfile prof;
    // Step expansion bottom wall from x = 0 to x = 12.0 (60 elements of length 0.2)
    for (int i = 0; i < 60; ++i) {
        double x_mid = (i + 0.5) * 0.2;
        SkinFrictionSample s;
        s.x = x_mid;
        s.y = 0.0;
        s.s = x_mid;
        s.length = 0.2;
        s.tx = 1.0;
        s.ty = 0.0;
        s.nx = 0.0;
        s.ny = 1.0;
        // Typical wall shear stress curve downstream of step:
        // negative from x = 0 to x = 6.0, crossing zero at x = 6.0
        s.cf = 0.008 * (x_mid - expected_xr) / expected_xr * std::exp(-0.2 * x_mid);
        prof.samples.push_back(s);
    }
    prof.total_length = 12.0;

    const auto bubbles = find_recirculation_bubbles(prof);
    REQUIRE(bubbles.size() == 1);
    CHECK_THAT(bubbles[0].x_sep, WithinAbs(0.0, 1e-12));
    CHECK_THAT(bubbles[0].x_reatt, WithinAbs(expected_xr, 0.01));
    CHECK_THAT(bubbles[0].length / h, WithinAbs(6.0, 0.01));
}

TEST_CASE("resolve_wall_sides utility", "[post][skin_friction]") {
    Mesh m = channel_mesh(GridCells::Quads, 4, 4, 2.0, 1.0);
    // Find sides by numeric flag 500 (bottom wall)
    auto sides_500 = resolve_wall_sides(m, "500");
    CHECK(sides_500.size() == 4);

    // Find sides by canonical name "wall"
    auto sides_wall = resolve_wall_sides(m, "wall");
    CHECK(sides_wall.size() == 4);

    // Unknown identifier returns empty
    auto sides_none = resolve_wall_sides(m, "nonexistent");
    CHECK(sides_none.empty());
}
