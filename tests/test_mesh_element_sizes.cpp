// Element sizes: Mesh::compute_element_sizes() stores in alen_e the minimum
// height (altitude) of every element, min_k(2A / s_k) = 2A / longest side.
// It is the length scale of the convective and diffusive time step limits in
// compute_timestep(). Requires detJ, i.e. compute_shape_function_derivatives().

#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinRel;

namespace {

double element_size(const Triangle& t) {
    Mesh m = make_triangle_mesh(t);
    m.compute_shape_function_derivatives();
    m.compute_element_sizes();
    REQUIRE(m.alen_e.size() == 1);
    return m.alen_e[0];
}

// Distance from point c to the line through a and b, computed by orthogonal
// projection (independent of the area formula used in Mesh).
double distance_to_line(Point c, Point a, Point b) {
    const double ex = b.x - a.x, ey = b.y - a.y;
    const double t = ((c.x - a.x) * ex + (c.y - a.y) * ey) / (ex * ex + ey * ey);
    return std::hypot(c.x - (a.x + t * ex), c.y - (a.y + t * ey));
}

double min_altitude(const Triangle& t) {
    const auto& p = t.p;
    return std::min({distance_to_line(p[0], p[1], p[2]),
                     distance_to_line(p[1], p[2], p[0]),
                     distance_to_line(p[2], p[0], p[1])});
}

} // namespace

TEST_CASE("Element size of hand-computed triangles", "[mesh][size]") {
    const auto tris = reference_triangles();

    SECTION("unit right triangle: altitude onto the hypotenuse, 1/sqrt(2)") {
        CHECK_THAT(element_size(tris[0]), WithinRel(1.0 / std::sqrt(2.0), 1e-14));
    }
    SECTION("scalene (1,1),(4,2),(2,5): 2A / longest side = 11 / sqrt(17)") {
        CHECK_THAT(element_size(tris[1]), WithinRel(11.0 / std::sqrt(17.0), 1e-14));
    }
    SECTION("equilateral with side 2: height sqrt(3)") {
        CHECK_THAT(element_size(tris[2]), WithinRel(std::sqrt(3.0), 1e-14));
    }
    SECTION("high aspect ratio: height 0.01 above a base of length 100") {
        CHECK_THAT(element_size(tris[6]), WithinRel(0.01, 1e-9));
    }
}

TEST_CASE("Element size equals the smallest altitude", "[mesh][size]") {
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);

    const double h = element_size(tri);
    CHECK(h > 0.0);
    CHECK_THAT(h, WithinRel(min_altitude(tri), 1e-9));

    // The smallest altitude is never longer than any side.
    for (int k = 0; k < 3; ++k) {
        const Point a = tri.p[k], b = tri.p[(k + 1) % 3];
        CHECK(h <= std::hypot(b.x - a.x, b.y - a.y) * (1.0 + 1e-12));
    }
}

TEST_CASE("Element size is invariant under translation and vertex rotation", "[mesh][size]") {
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    const double h = element_size(tri);

    SECTION("translation") {
        CHECK_THAT(element_size(translated(tri, -3.0, 11.5)), WithinRel(h, 1e-9));
    }
    SECTION("cyclic vertex permutation") {
        Triangle rot = tri;
        rot.p = {tri.p[2], tri.p[0], tri.p[1]};
        CHECK_THAT(element_size(rot), WithinRel(h, 1e-12));
    }
}

TEST_CASE("Element size scales linearly with the length scale", "[mesh][size]") {
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    const double s = GENERATE(1.0e-3, 0.5, 3.0, 1.0e3);
    CAPTURE(tri, s);

    CHECK_THAT(element_size(scaled(tri, s)), WithinRel(s * element_size(tri), 1e-12));
}

TEST_CASE("Element sizes on a structured grid", "[mesh][size]") {
    // Every element is a right triangle with legs hx, hy; its smallest
    // altitude is the one onto the hypotenuse: hx * hy / sqrt(hx^2 + hy^2).
    GridSpec g;
    g.lx = 2.0;
    g.ly = 1.5;
    g.nx = 5;
    g.ny = 4;
    Mesh m = make_grid_mesh(g);
    m.compute_shape_function_derivatives();
    m.compute_element_sizes();

    REQUIRE(static_cast<int>(m.alen_e.size()) == g.nelem());
    const double expected = g.hx() * g.hy() / std::hypot(g.hx(), g.hy());
    for (int ie = 0; ie < m.nelem; ++ie) {
        CAPTURE(ie);
        CHECK_THAT(m.alen_e[ie], WithinRel(expected, 1e-12));
    }
}

TEST_CASE("Element sizes follow the current element geometry when recomputed", "[mesh][size]") {
    Mesh m = make_triangle_mesh(reference_triangles()[0]);
    m.compute_shape_function_derivatives();
    m.compute_element_sizes();
    CHECK_THAT(m.alen_e[0], WithinRel(1.0 / std::sqrt(2.0), 1e-14));

    for (auto& x : m.x) x *= 2.0;
    for (auto& y : m.y) y *= 2.0;
    m.compute_shape_function_derivatives();
    m.compute_element_sizes();
    REQUIRE(m.alen_e.size() == 1);
    CHECK_THAT(m.alen_e[0], WithinRel(std::sqrt(2.0), 1e-14));
}

TEST_CASE("Clockwise elements yield a negative element size", "[mesh][size][characterization]") {
    // Documents current behaviour: the size is computed from the signed detJ,
    // so a clockwise element gets a negative size (and compute_timestep would
    // then produce a negative convective time step). Meshes must be CCW.
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    CHECK(element_size(reversed(tri)) < 0.0);
}
