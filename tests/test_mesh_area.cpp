// Element areas: Mesh::compute_shape_function_derivatives() stores
// detJ = 2 * (signed element area) and the max/min area statistics.

#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinRel;

namespace {
double element_area(const Triangle& t) {
    Mesh m = make_triangle_mesh(t);
    m.compute_shape_function_derivatives();
    return 0.5 * m.detJ[0];
}
} // namespace

TEST_CASE("Element area of hand-computed triangles", "[mesh][area]") {
    SECTION("unit right triangle has area 1/2, detJ 1") {
        Mesh m = make_triangle_mesh(reference_triangles()[0]);
        m.compute_shape_function_derivatives();
        REQUIRE(m.detJ.size() == 1);
        CHECK_THAT(m.detJ[0], WithinRel(1.0, 1e-14));
    }
    SECTION("scalene triangle (1,1),(4,2),(2,5) has area 11/2") {
        Mesh m = make_triangle_mesh(reference_triangles()[1]);
        m.compute_shape_function_derivatives();
        CHECK_THAT(m.detJ[0], WithinRel(11.0, 1e-14));
    }
    SECTION("equilateral triangle with side 2 has area sqrt(3)") {
        CHECK_THAT(element_area(reference_triangles()[2]), WithinRel(std::sqrt(3.0), 1e-14));
    }
}

TEST_CASE("Element area matches Heron's formula for counter-clockwise triangles", "[mesh][area]") {
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);

    const double area = element_area(tri);
    CHECK(area > 0.0);
    CHECK_THAT(area, WithinRel(heron_area(tri), 1e-9));
}

TEST_CASE("Element area is invariant under translation and vertex rotation", "[mesh][area]") {
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    const double area = element_area(tri);

    SECTION("translation") {
        CHECK_THAT(element_area(translated(tri, 7.25, -3.5)), WithinRel(area, 1e-9));
    }
    SECTION("cyclic vertex permutation keeps orientation and area") {
        Triangle rot = tri;
        rot.p = {tri.p[1], tri.p[2], tri.p[0]};
        CHECK_THAT(element_area(rot), WithinRel(area, 1e-12));
    }
}

TEST_CASE("Element area scales with the square of the length scale", "[mesh][area]") {
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    const double s = GENERATE(1.0e-3, 0.5, 3.0, 1.0e3);
    CAPTURE(tri, s);

    CHECK_THAT(element_area(scaled(tri, s)), WithinRel(s * s * element_area(tri), 1e-12));
}

TEST_CASE("Clockwise element ordering yields a negative signed area", "[mesh][area]") {
    // Documents current behaviour: detJ is a signed quantity and is NOT made
    // positive, so meshes must use counter-clockwise connectivity.
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);

    CHECK_THAT(element_area(reversed(tri)), WithinRel(-element_area(tri), 1e-12));
}

TEST_CASE("Element areas of a structured grid tile the rectangle", "[mesh][area]") {
    GridSpec g;
    g.x0 = -1.0;
    g.y0 = 2.0;
    g.lx = 3.0;
    g.ly = 2.0;
    g.nx = 6;
    g.ny = 4;
    Mesh m = make_grid_mesh(g);
    m.compute_shape_function_derivatives();

    REQUIRE(static_cast<int>(m.detJ.size()) == g.nelem());

    const double expected = 0.5 * g.hx() * g.hy();
    double total = 0.0;
    for (int ie = 0; ie < m.nelem; ++ie) {
        CAPTURE(ie);
        CHECK_THAT(0.5 * m.detJ[ie], WithinRel(expected, 1e-12));
        total += 0.5 * m.detJ[ie];
    }
    CHECK_THAT(total, WithinRel(g.lx * g.ly, 1e-12));
    CHECK_THAT(m.max_vol, WithinRel(expected, 1e-12));
    CHECK_THAT(m.min_vol, WithinRel(expected, 1e-12));
}

TEST_CASE("Max/min element volume statistics", "[mesh][area]") {
    // Two disjoint triangles with areas 1/2 and 2.
    Mesh m = make_mesh({{0, 0}, {1, 0}, {0, 1}, {5, 5}, {7, 5}, {5, 7}},
                       {{0, 1, 2}, {3, 4, 5}});
    m.compute_shape_function_derivatives();

    CHECK_THAT(m.min_vol, WithinRel(0.5, 1e-14));
    CHECK_THAT(m.max_vol, WithinRel(2.0, 1e-14));

    SECTION("statistics are reset when recomputed") {
        // New areas 1 and 3/2: stale values (1/2, 2) would fail both checks.
        m.x = {0, 2, 0, 5, 8, 5};
        m.y = {0, 0, 1, 5, 5, 6};
        m.compute_shape_function_derivatives();
        CHECK_THAT(m.min_vol, WithinRel(1.0, 1e-14));
        CHECK_THAT(m.max_vol, WithinRel(1.5, 1e-14));
    }
}
