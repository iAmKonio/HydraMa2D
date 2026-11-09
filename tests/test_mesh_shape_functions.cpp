// Shape function derivatives: Mesh::compute_shape_function_derivatives()
// stores the (constant) gradients of the three linear P1 basis functions
// N_0, N_1, N_2 of every element in dNkdx.

#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;

namespace {

ElementDeriv derivatives_of(const Triangle& t) {
    Mesh m = make_triangle_mesh(t);
    m.compute_shape_function_derivatives();
    REQUIRE(m.dNkdx.size() == 1);
    return m.dNkdx[0];
}

// Absolute tolerance for quantities of order `scale`.
constexpr double kTol = 1e-9;

} // namespace

TEST_CASE("Shape function derivatives of the unit right triangle", "[mesh][shape]") {
    // N0 = 1 - x - y, N1 = x, N2 = y
    const ElementDeriv d = derivatives_of(reference_triangles()[0]);

    CHECK_THAT(d.dNdx[0], WithinAbs(-1.0, 1e-15));
    CHECK_THAT(d.dNdx[1], WithinAbs(1.0, 1e-15));
    CHECK_THAT(d.dNdx[2], WithinAbs(0.0, 1e-15));
    CHECK_THAT(d.dNdy[0], WithinAbs(-1.0, 1e-15));
    CHECK_THAT(d.dNdy[1], WithinAbs(0.0, 1e-15));
    CHECK_THAT(d.dNdy[2], WithinAbs(1.0, 1e-15));
}

TEST_CASE("Shape function derivatives of a scalene triangle", "[mesh][shape]") {
    // Nodes (1,1), (4,2), (2,5); 2A = 11.
    // N_i = (a_i + b_i x + c_i y) / 2A with b = (-3, 4, -1), c = (-2, -1, 3).
    const ElementDeriv d = derivatives_of(reference_triangles()[1]);

    CHECK_THAT(d.dNdx[0], WithinAbs(-3.0 / 11.0, 1e-15));
    CHECK_THAT(d.dNdx[1], WithinAbs(4.0 / 11.0, 1e-15));
    CHECK_THAT(d.dNdx[2], WithinAbs(-1.0 / 11.0, 1e-15));
    CHECK_THAT(d.dNdy[0], WithinAbs(-2.0 / 11.0, 1e-15));
    CHECK_THAT(d.dNdy[1], WithinAbs(-1.0 / 11.0, 1e-15));
    CHECK_THAT(d.dNdy[2], WithinAbs(3.0 / 11.0, 1e-15));
}

TEST_CASE("Shape functions satisfy the Kronecker delta property", "[mesh][shape]") {
    // N_i is linear with N_i(x_j) = delta_ij, hence for j != i:
    //   grad N_i . (x_j - x_i) = N_i(x_j) - N_i(x_i) = -1
    // Together these uniquely determine grad N_i.
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    const ElementDeriv d = derivatives_of(tri);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == j) continue;
            CAPTURE(i, j);
            const double dx = tri.p[j].x - tri.p[i].x;
            const double dy = tri.p[j].y - tri.p[i].y;
            CHECK_THAT(d.dNdx[i] * dx + d.dNdy[i] * dy, WithinAbs(-1.0, kTol));
        }
    }
}

TEST_CASE("Shape function derivatives sum to zero (partition of unity)", "[mesh][shape]") {
    // sum_i N_i == 1  =>  sum_i grad N_i == 0
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    const ElementDeriv d = derivatives_of(tri);

    // Scale tolerance with the gradient magnitude (large for slender elements).
    double gmax = 0.0;
    for (int i = 0; i < 3; ++i) gmax = std::max({gmax, std::abs(d.dNdx[i]), std::abs(d.dNdy[i])});

    CHECK_THAT(d.dNdx[0] + d.dNdx[1] + d.dNdx[2], WithinAbs(0.0, 1e-12 * gmax));
    CHECK_THAT(d.dNdy[0] + d.dNdy[1] + d.dNdy[2], WithinAbs(0.0, 1e-12 * gmax));
}

TEST_CASE("Gradient of a linear field is reproduced exactly", "[mesh][shape]") {
    // f(x, y) = a + b x + c y  =>  sum_i f(x_i) grad N_i == (b, c)
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    const double a = 3.0, b = -2.0, c = 5.0;

    auto grad_of_linear_field = [&](const Triangle& t) {
        const ElementDeriv d = derivatives_of(t);
        double gx = 0.0, gy = 0.0;
        for (int i = 0; i < 3; ++i) {
            const double f = a + b * t.p[i].x + c * t.p[i].y;
            gx += d.dNdx[i] * f;
            gy += d.dNdy[i] * f;
        }
        return std::array<double, 2>{gx, gy};
    };

    SECTION("counter-clockwise ordering") {
        const auto g = grad_of_linear_field(tri);
        CHECK_THAT(g[0], WithinAbs(b, kTol));
        CHECK_THAT(g[1], WithinAbs(c, kTol));
    }
    SECTION("clockwise ordering gives the same gradient") {
        // The sign of the area cancels in dN/dx = b_i / 2A.
        const auto g = grad_of_linear_field(reversed(tri));
        CHECK_THAT(g[0], WithinAbs(b, kTol));
        CHECK_THAT(g[1], WithinAbs(c, kTol));
    }
}

TEST_CASE("Shape function derivatives are translation invariant and scale as 1/L", "[mesh][shape]") {
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    const ElementDeriv d = derivatives_of(tri);

    SECTION("translation") {
        const ElementDeriv dt = derivatives_of(translated(tri, -12.5, 4.0));
        for (int i = 0; i < 3; ++i) {
            CAPTURE(i);
            CHECK_THAT(dt.dNdx[i], WithinAbs(d.dNdx[i], 1e-9 * (1.0 + std::abs(d.dNdx[i]))));
            CHECK_THAT(dt.dNdy[i], WithinAbs(d.dNdy[i], 1e-9 * (1.0 + std::abs(d.dNdy[i]))));
        }
    }
    SECTION("uniform scaling") {
        const double s = 4.0;
        const ElementDeriv ds = derivatives_of(scaled(tri, s));
        for (int i = 0; i < 3; ++i) {
            CAPTURE(i);
            CHECK_THAT(ds.dNdx[i] * s, WithinAbs(d.dNdx[i], 1e-12 * (1.0 + std::abs(d.dNdx[i]))));
            CHECK_THAT(ds.dNdy[i] * s, WithinAbs(d.dNdy[i], 1e-12 * (1.0 + std::abs(d.dNdy[i]))));
        }
    }
}

TEST_CASE("Per-element derivatives on a grid use each element's own connectivity", "[mesh][shape]") {
    // A linear field sampled at the nodes must have the same gradient on every
    // element; any mix-up of element/node indices would break this.
    GridSpec g;
    g.x0 = 0.5;
    g.y0 = -1.0;
    g.lx = 2.0;
    g.ly = 3.0;
    g.nx = 4;
    g.ny = 3;
    Mesh m = make_grid_mesh(g);
    m.compute_shape_function_derivatives();
    REQUIRE(static_cast<int>(m.dNkdx.size()) == g.nelem());

    const double b = 1.5, c = -0.75;
    std::vector<double> f(m.npoin);
    for (int ip = 0; ip < m.npoin; ++ip) f[ip] = 2.0 + b * m.x[ip] + c * m.y[ip];

    for (int ie = 0; ie < m.nelem; ++ie) {
        CAPTURE(ie);
        double gx = 0.0, gy = 0.0;
        for (int in = 0; in < 3; ++in) {
            gx += m.dNkdx[ie].dNdx[in] * f[m.intma[ie][in]];
            gy += m.dNkdx[ie].dNdy[in] * f[m.intma[ie][in]];
        }
        CHECK_THAT(gx, WithinAbs(b, 1e-12));
        CHECK_THAT(gy, WithinAbs(c, 1e-12));
    }
}
