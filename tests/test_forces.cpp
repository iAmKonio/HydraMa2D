// Forces on a body (post/Forces.hpp): the surface traction and the volume
// (reaction) form on triangle, quadrilateral and hybrid meshes, against
// closed-form values:
//   * Couette flow u = (y, 0), p = p0 on a channel, body = bottom wall:
//     F = (nu L, -p0 L), exact for both methods on distorted meshes;
//   * identities of the volume form on a closed body (a hole) for linear
//     pressures, constant time derivatives and a convective field, checked
//     against lumped masses and an independent 3 x 3 Gauss rule;
//   * the first-order error of the surface form for Poiseuille flow;
//   * rotation invariance, independence of the side orientation, validation.

#include "QuadTestUtils.hpp"
#include "numerics/Quadrature.hpp"
#include "post/Forces.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

const char* cell_name(GridCells c) {
    return c == GridCells::Quads ? "quadrilaterals" : c == GridCells::Triangles ? "triangles" : "hybrid";
}

// Channel [0, L] x [0, 1]; bottom wall flag 500, every other side 503.
Mesh channel(GridCells cells, double perturbation, int nx = 6, int ny = 4, double length = 3.0) {
    GridOptions g;
    g.lx = length;
    g.ly = 1.0;
    g.nx = nx;
    g.ny = ny;
    g.cells = cells;
    g.perturbation = perturbation;
    Mesh m = make_hybrid_mesh(grid(g), [](double, double y) { return y == 0.0 ? 500 : 503; });
    compute_all_geometry(m);
    return m;
}

std::vector<int> sides_with_flag(const Mesh& m, int flag) {
    std::vector<int> out;
    for (int ib = 0; ib < m.nboun; ++ib)
        if (m.iside[static_cast<std::size_t>(ib)].flag == flag) out.push_back(ib);
    return out;
}

// Square [0, 4]^2 with the hole [1.5, 2.5]^2 (walls 500); outer sides 503.
Mesh holed_square(GridCells cells, double perturbation) {
    GridOptions g;
    g.lx = g.ly = 4.0;
    g.nx = g.ny = 8;
    g.cells = cells;
    g.layers = 2;
    g.perturbation = perturbation;
    // Keep the nodes around the hole on its straight sides: no perturbation
    // there would need a special grid, so the hole is cut from an unperturbed
    // copy of the node positions next to it.
    HybridSpec s = grid(g);
    GridOptions flat = g;
    flat.perturbation = 0.0;
    const HybridSpec ref = grid(flat);
    for (std::size_t i = 0; i < s.nodes.size(); ++i) {
        const Point& p = ref.nodes[i];
        if (p.x >= 1.0 && p.x <= 3.0 && p.y >= 1.0 && p.y <= 3.0) s.nodes[i] = p;
    }
    const HybridSpec holed = without_cells(s, [](double x, double y) { return x > 1.5 && x < 2.5 && y > 1.5 && y < 2.5; });
    Mesh m = make_hybrid_mesh(holed, [](double x, double y) {
        return (x > 1.4 && x < 2.6 && y > 1.4 && y < 2.6) ? 500 : 503;
    });
    compute_all_geometry(m);
    return m;
}

// Lumped mass int N_i dOmega summed over the nodes of the body sides.
double body_mass(const Mesh& m, const std::vector<int>& body) {
    std::vector<char> is_body(static_cast<std::size_t>(m.npoin), 0);
    for (int ib : body) is_body[m.iside[ib].node1] = is_body[m.iside[ib].node2] = 1;
    double mass = 0.0;
    for (int e = 0; e < m.num_triangles(); ++e)
        for (int a = 0; a < 3; ++a)
            if (is_body[m.intma[e][a]]) mass += m.detJ[e] / 6.0;
    for (int q = 0; q < m.num_quads(); ++q)
        for (int a = 0; a < 4; ++a)
            if (is_body[m.quads[q][a]]) mass += m.quad_geom[q].mass[a];
    return mass;
}

template <typename F>
std::vector<double> nodal(const Mesh& m, F f) {
    std::vector<double> v(static_cast<std::size_t>(m.npoin));
    for (int i = 0; i < m.npoin; ++i) v[static_cast<std::size_t>(i)] = f(m.x[i], m.y[i]);
    return v;
}

} // namespace

TEST_CASE("Forces: Couette flow gives the exact wall force with both methods", "[post][forces]") {
    const GridCells cells = GENERATE(GridCells::Triangles, GridCells::Quads, GridCells::InflationLayers);
    const double perturbation = GENERATE(0.0, 0.2);
    CAPTURE(cell_name(cells), perturbation);
    const Mesh m = channel(cells, perturbation);
    const std::vector<int> body = sides_with_flag(m, 500);
    REQUIRE(body.size() == 6);
    const double nu = 0.037, p0 = 1.7, L = 3.0;
    const auto u = nodal(m, [](double, double y) { return y; });
    const std::vector<double> v(static_cast<std::size_t>(m.npoin), 0.0), p(static_cast<std::size_t>(m.npoin), p0);

    const post::BodyForce s = post::surface_force(m, body, nu, u, v, p);
    CHECK_THAT(s.fx, WithinRel(nu * L, 1e-12));
    CHECK_THAT(s.fy, WithinRel(-p0 * L, 1e-12));
    CHECK_THAT(s.pressure_fx, WithinAbs(0.0, 1e-13));
    CHECK_THAT(s.pressure_fy, WithinRel(-p0 * L, 1e-12));
    CHECK_THAT(s.viscous_fx(), WithinRel(nu * L, 1e-12));

    const post::BodyForce w = post::volume_force(m, body, nu, u, v, p);
    CHECK_THAT(w.fx, WithinRel(nu * L, 1e-12));
    CHECK_THAT(w.fy, WithinRel(-p0 * L, 1e-12));
}

TEST_CASE("Forces: identities of the volume form on a closed body", "[post][forces]") {
    const GridCells cells = GENERATE(GridCells::Triangles, GridCells::Quads, GridCells::InflationLayers);
    const double perturbation = GENERATE(0.0, 0.2);
    CAPTURE(cell_name(cells), perturbation);
    const Mesh m = holed_square(cells, perturbation);
    const std::vector<int> body = sides_with_flag(m, 500);
    REQUIRE(body.size() == 8); // the hole: 4 sides of 2 cells
    const double mass = body_mass(m, body);
    const std::vector<double> zero(static_cast<std::size_t>(m.npoin), 0.0);

    SECTION("linear pressure: F_volume = F_surface - grad p * sum of the body masses") {
        const double a = 0.3, b = -1.2, c = 0.7;
        const auto p = nodal(m, [&](double x, double y) { return a + b * x + c * y; });
        const post::BodyForce s = post::surface_force(m, body, 0.01, zero, zero, p);
        const post::BodyForce w = post::volume_force(m, body, 0.01, zero, zero, p);
        // Surface: the pressure force on the closed square [1.5, 2.5]^2 is -int p n_b = +int p n_outward(hole)... exactly:
        // F = -(grad p) * area(hole) (Archimedes).
        CHECK_THAT(s.fx, WithinAbs(-b * 1.0, 1e-12));
        CHECK_THAT(s.fy, WithinAbs(-c * 1.0, 1e-12));
        CHECK_THAT(w.fx, WithinAbs(s.fx - b * mass, 1e-12));
        CHECK_THAT(w.fy, WithinAbs(s.fy - c * mass, 1e-12));
    }
    SECTION("constant time derivative: F_volume = -du/dt * sum of the body masses") {
        const std::vector<double> dudt(static_cast<std::size_t>(m.npoin), 0.4), dvdt(static_cast<std::size_t>(m.npoin), -2.5);
        const post::BodyForce w = post::volume_force(m, body, 0.01, zero, zero, zero, &dudt, &dvdt);
        CHECK_THAT(w.fx, WithinRel(-0.4 * mass, 1e-12));
        CHECK_THAT(w.fy, WithinRel(2.5 * mass, 1e-12));
    }
    SECTION("convection: F_volume = -int (u . grad u) phi, independent 3 x 3 Gauss rule") {
        // u = (x, 0): (u . grad) u = (x, 0).
        const auto u = nodal(m, [](double x, double) { return x; });
        const post::BodyForce w = post::volume_force(m, body, 0.0, u, zero, zero);
        std::vector<char> is_body(static_cast<std::size_t>(m.npoin), 0);
        for (int ib : body) is_body[m.iside[ib].node1] = is_body[m.iside[ib].node2] = 1;
        const auto rule = quadrature::gauss_legendre(3);
        double expected = 0.0;
        for (int e = 0; e < m.nelem; ++e) {
            const ElementNodes n = m.element_nodes(e);
            if (!m.is_quad(e)) {
                // int_T x N_i = A / 12 (2 x_i + x_j + x_k)
                const double area = 0.5 * m.detJ[e];
                for (int a = 0; a < 3; ++a)
                    if (is_body[n[a]]) expected += area / 12.0 * (2.0 * m.x[n[a]] + m.x[n[(a + 1) % 3]] + m.x[n[(a + 2) % 3]]);
                continue;
            }
            double qx[4], qy[4];
            for (int a = 0; a < 4; ++a) {
                qx[a] = m.x[n[a]];
                qy[a] = m.y[n[a]];
            }
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j) {
                    double N[4], px = 0.0, py = 0.0;
                    q1::shape_functions(rule.points[i], rule.points[j], N);
                    q1::map(qx, qy, rule.points[i], rule.points[j], px, py);
                    const double det = q1::jacobian(qx, qy, rule.points[i], rule.points[j]).det;
                    for (int a = 0; a < 4; ++a)
                        if (is_body[n[a]]) expected += rule.weights[i] * rule.weights[j] * det * px * N[a];
                }
        }
        CHECK_THAT(w.fx, WithinRel(-expected, 1e-12));
        CHECK_THAT(w.fy, WithinAbs(0.0, 1e-13));
    }
    SECTION("a constant state exerts no force on a closed body") {
        const std::vector<double> u(static_cast<std::size_t>(m.npoin), 0.8), v(static_cast<std::size_t>(m.npoin), -0.3),
            p(static_cast<std::size_t>(m.npoin), 2.0);
        const post::BodyForce s = post::surface_force(m, body, 0.05, u, v, p);
        const post::BodyForce w = post::volume_force(m, body, 0.05, u, v, p);
        CHECK_THAT(s.fx, WithinAbs(0.0, 1e-13));
        CHECK_THAT(s.fy, WithinAbs(0.0, 1e-13));
        CHECK_THAT(w.fx, WithinAbs(0.0, 1e-13));
        CHECK_THAT(w.fy, WithinAbs(0.0, 1e-13));
    }
}

TEST_CASE("Forces: the surface form is first order for Poiseuille flow", "[post][forces]") {
    // u = 6 y (1 - y): wall shear 6 nu. The gradient of the interpolant in the
    // first cell row is the secant slope 6 (1 - h), for triangles and
    // quadrilaterals (at the side midpoint): F_x = 6 nu L (1 - h).
    const GridCells cells = GENERATE(GridCells::Triangles, GridCells::Quads);
    const int ny = GENERATE(4, 8, 16);
    CAPTURE(cell_name(cells), ny);
    const Mesh m = channel(cells, 0.0, 6, ny);
    const std::vector<int> body = sides_with_flag(m, 500);
    const double nu = 0.1, L = 3.0, h = 1.0 / ny;
    const auto u = nodal(m, [](double, double y) { return 6.0 * y * (1.0 - y); });
    const auto p = nodal(m, [&](double x, double) { return -12.0 * nu * x; });
    const std::vector<double> v(static_cast<std::size_t>(m.npoin), 0.0);
    const post::BodyForce s = post::surface_force(m, body, nu, u, v, p);
    CHECK_THAT(s.fx, WithinRel(6.0 * nu * L * (1.0 - h), 1e-12));
    // Pressure: -int p n_b dx with n_b = (0, 1), p linear: exact.
    CHECK_THAT(s.fy, WithinRel(12.0 * nu * L * L / 2.0, 1e-12));
}

TEST_CASE("Forces: rotating the configuration rotates the force", "[post][forces]") {
    const GridCells cells = GENERATE(GridCells::Triangles, GridCells::InflationLayers);
    CAPTURE(cell_name(cells));
    const Mesh base = holed_square(cells, 0.2);
    const std::vector<int> body = sides_with_flag(base, 500);
    // A generic (non-solution) state.
    auto uf = [](double x, double y) { return 0.3 + 0.2 * x - 0.1 * y + 0.05 * x * y; };
    auto vf = [](double x, double y) { return -0.2 + 0.1 * x + 0.3 * y - 0.02 * x * x; };
    auto pf = [](double x, double y) { return 1.0 + 0.4 * x * y - 0.3 * y; };
    const double nu = 0.02;
    auto forces = [&](const Mesh& m, double angle) {
        // Fields defined in the original frame, velocity rotated with the frame.
        const double c = std::cos(angle), s = std::sin(angle);
        std::vector<double> u(static_cast<std::size_t>(m.npoin)), v(u.size()), p(u.size());
        for (int i = 0; i < m.npoin; ++i) {
            const double x0 = c * m.x[i] + s * m.y[i], y0 = -s * m.x[i] + c * m.y[i]; // back-rotated position
            const double ur = uf(x0, y0), vr = vf(x0, y0);
            u[static_cast<std::size_t>(i)] = c * ur - s * vr;
            v[static_cast<std::size_t>(i)] = s * ur + c * vr;
            p[static_cast<std::size_t>(i)] = pf(x0, y0);
        }
        return std::pair{post::surface_force(m, body, nu, u, v, p), post::volume_force(m, body, nu, u, v, p)};
    };
    const auto [s0, w0] = forces(base, 0.0);
    const double angle = 0.7;
    Mesh rotated = base;
    for (int i = 0; i < rotated.npoin; ++i) {
        rotated.x[i] = std::cos(angle) * base.x[i] - std::sin(angle) * base.y[i];
        rotated.y[i] = std::sin(angle) * base.x[i] + std::cos(angle) * base.y[i];
    }
    compute_all_geometry(rotated);
    const auto [s1, w1] = forces(rotated, angle);
    CHECK_THAT(s1.fx, WithinAbs(std::cos(angle) * s0.fx - std::sin(angle) * s0.fy, 1e-12));
    CHECK_THAT(s1.fy, WithinAbs(std::sin(angle) * s0.fx + std::cos(angle) * s0.fy, 1e-12));
    CHECK_THAT(w1.fx, WithinAbs(std::cos(angle) * w0.fx - std::sin(angle) * w0.fy, 1e-12));
    CHECK_THAT(w1.fy, WithinAbs(std::sin(angle) * w0.fx + std::cos(angle) * w0.fy, 1e-12));
    CHECK(std::abs(w0.fx) > 1e-3); // not a trivial zero

    // The node order of the body sides does not matter.
    Mesh reversed = base;
    for (BoundarySide& side : reversed.iside) std::swap(side.node1, side.node2);
    const auto [s2, w2] = forces(reversed, 0.0);
    CHECK(s2.fx == s0.fx);
    CHECK(s2.fy == s0.fy);
    CHECK(w2.fx == w0.fx);
    CHECK(w2.fy == w0.fy);
}

TEST_CASE("Forces: coefficients and input validation", "[post][forces]") {
    CHECK(post::force_coefficient(2.5, 1.0, 1.0) == 5.0);
    CHECK_THAT(post::force_coefficient(0.5, 2.0, 0.1), WithinRel(0.5 / (0.5 * 4.0 * 0.1), 1e-15));
    CHECK_THROWS_AS(post::force_coefficient(1.0, 0.0, 1.0), std::invalid_argument);
    CHECK_THROWS_AS(post::force_coefficient(1.0, 1.0, -1.0), std::invalid_argument);

    const Mesh m = channel(GridCells::InflationLayers, 0.1);
    const std::vector<double> f(static_cast<std::size_t>(m.npoin), 0.0), shorter(3, 0.0);
    CHECK_THROWS_AS(post::surface_force(m, {m.nboun}, 0.1, f, f, f), std::invalid_argument);
    CHECK_THROWS_AS(post::volume_force(m, {-1}, 0.1, f, f, f), std::invalid_argument);
    CHECK_THROWS_AS(post::surface_force(m, {0}, 0.1, f, shorter, f), std::invalid_argument);
    CHECK_THROWS_AS(post::volume_force(m, {0}, 0.1, f, f, f, &f, nullptr), std::invalid_argument);
    CHECK_THROWS_AS(post::volume_force(m, {0}, 0.1, f, f, f, &shorter, &shorter), std::invalid_argument);
    Mesh raw = m;
    raw.dNkdx.clear();
    CHECK_THROWS_AS(post::volume_force(raw, {0}, 0.1, f, f, f), std::invalid_argument);
    CHECK(post::volume_force(m, {}, 0.1, f, f, f).fx == 0.0); // empty body
}
