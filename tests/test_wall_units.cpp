// Wall shear stress, friction velocity and y+ (turbulence/WallUnits).
// Couette flow u = gamma y between walls at y = 0 and y = H: tau_w = nu gamma
// on both walls, u_tau = sqrt(nu gamma), first-cell y+ = h_y u_tau / nu and
// the nodal y+ = d u_tau / nu.

#include "WallTestUtils.hpp"
#include "turbulence/WallUnits.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

Mesh couette_mesh(int nx, int ny, double lx, double ly) {
    GridSpec g;
    g.lx = lx;
    g.ly = ly;
    g.nx = nx;
    g.ny = ny;
    Mesh m = make_grid_mesh(g);
    set_flags_by_side_id(m, {{1, 500}, {2, 504}, {3, 500}, {4, 503}});
    compute_all_geometry(m);
    return m;
}

} // namespace

TEST_CASE("Wall units in Couette flow", "[wall_units]") {
    const double gamma = 3.0, nu = 1.0e-3, lx = 2.0, ly = 0.5;
    const int nx = 8, ny = 5;
    const Mesh m = couette_mesh(nx, ny, lx, ly);
    const WallGeometry walls(m);
    const auto u = sample(m, [&](double, double y) { return gamma * y; });
    const std::vector<double> v(m.npoin, 0.0);

    const std::vector<double> u_tau = wall_friction_velocity(m, walls, u, v, nu);
    REQUIRE(static_cast<int>(u_tau.size()) == walls.num_faces());
    const double expected = std::sqrt(nu * gamma);
    for (double ut : u_tau) CHECK_THAT(ut, WithinRel(expected, 1e-12));

    const WallUnitStats s = first_cell_yplus(walls, u_tau, nu);
    CHECK(s.faces == 2 * nx);
    const double yp1 = (ly / ny) * expected / nu;
    CHECK_THAT(s.min, WithinRel(yp1, 1e-12));
    CHECK_THAT(s.mean, WithinRel(yp1, 1e-12));
    CHECK_THAT(s.max, WithinRel(yp1, 1e-12));

    std::vector<double> yplus;
    yplus_field(walls, u_tau, nu, yplus);
    for (int ip = 0; ip < m.npoin; ++ip) {
        const double d = std::min(m.y[ip], ly - m.y[ip]);
        CHECK_THAT(yplus[ip], WithinAbs(d * expected / nu, 1e-9));
    }
}

TEST_CASE("Wall units are frame invariant and independent of the side orientation", "[wall_units]") {
    const double gamma = 2.0, nu = 5.0e-3;
    Mesh m = couette_mesh(6, 4, 1.5, 1.0);
    const double angle = GENERATE(0.0, 0.6, 2.0);
    const bool flip = GENERATE(false, true);
    CAPTURE(angle, flip);
    const double c = std::cos(angle), s = std::sin(angle);
    // Rotate the mesh and the Couette velocity field.
    std::vector<double> u(m.npoin), v(m.npoin);
    for (int ip = 0; ip < m.npoin; ++ip) {
        const double x = m.x[ip], y = m.y[ip];
        const double speed = gamma * y; // along the (rotated) x axis
        m.x[ip] = c * x - s * y;
        m.y[ip] = s * x + c * y;
        u[ip] = c * speed;
        v[ip] = s * speed;
    }
    if (flip)
        for (auto& side : m.iside) std::swap(side.node1, side.node2);
    compute_all_geometry(m);
    const WallGeometry walls(m);
    for (double ut : wall_friction_velocity(m, walls, u, v, nu)) CHECK_THAT(ut, WithinRel(std::sqrt(nu * gamma), 1e-12));
}

TEST_CASE("Wall units without walls", "[wall_units]") {
    GridSpec g;
    g.nx = 3;
    g.ny = 2;
    Mesh m = make_grid_mesh(g, 504);
    compute_all_geometry(m);
    const WallGeometry walls(m);
    const std::vector<double> u(m.npoin, 1.0), v(m.npoin, 0.0);
    const auto u_tau = wall_friction_velocity(m, walls, u, v, 1e-3);
    CHECK(u_tau.empty());
    CHECK(first_cell_yplus(walls, u_tau, 1e-3).faces == 0);
    std::vector<double> yplus;
    yplus_field(walls, u_tau, 1e-3, yplus);
    CHECK(yplus == std::vector<double>(m.npoin, 0.0));
}
