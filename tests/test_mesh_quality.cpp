// Element quality metrics and the mesh quality report (mesh/MeshQuality.hpp),
// against values computed by hand.

#include "QuadTestUtils.hpp"
#include "mesh/MeshQuality.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>

using namespace hydrama;
using namespace hydrama::mesh;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

namespace {

const double kSqrt3 = std::sqrt(3.0);

ElementQuality tri(std::array<double, 3> x, std::array<double, 3> y) { return triangle_quality(x.data(), y.data()); }
ElementQuality quad(std::array<double, 4> x, std::array<double, 4> y) { return quad_quality(x.data(), y.data()); }

} // namespace

TEST_CASE("Mesh quality: triangles", "[mesh_quality]") {
    SECTION("equilateral: ideal") {
        const ElementQuality q = tri({0.0, 1.0, 0.5}, {0.0, 0.0, 0.5 * kSqrt3});
        CHECK_THAT(q.min_angle, WithinAbs(60.0, 1e-12));
        CHECK_THAT(q.max_angle, WithinAbs(60.0, 1e-12));
        CHECK_THAT(q.aspect_ratio, WithinAbs(1.0, 1e-15));
        CHECK_THAT(q.skewness, WithinAbs(0.0, 1e-12));
        CHECK_THAT(q.scaled_jacobian, WithinAbs(1.0, 1e-15));
    }
    SECTION("right isosceles") {
        const ElementQuality q = tri({0.0, 1.0, 0.0}, {0.0, 0.0, 1.0});
        CHECK_THAT(q.min_angle, WithinAbs(45.0, 1e-12));
        CHECK_THAT(q.max_angle, WithinAbs(90.0, 1e-12));
        CHECK_THAT(q.aspect_ratio, WithinAbs(std::sqrt(2.0), 1e-15));
        CHECK_THAT(q.skewness, WithinAbs(0.25, 1e-12)); // max(30/120, 15/60)
        CHECK_THAT(q.scaled_jacobian, WithinAbs(std::sin(std::acos(-1.0) / 4) * 2.0 / kSqrt3, 1e-15));
    }
    SECTION("obtuse sliver") {
        // Angles 20, 20, 140.
        const double h = std::tan(20.0 * std::acos(-1.0) / 180.0);
        const ElementQuality q = tri({0.0, 2.0, 1.0}, {0.0, 0.0, h});
        CHECK_THAT(q.min_angle, WithinAbs(20.0, 1e-12));
        CHECK_THAT(q.max_angle, WithinAbs(140.0, 1e-12));
        CHECK_THAT(q.skewness, WithinAbs(std::max(80.0 / 120.0, 40.0 / 60.0), 1e-12));
        CHECK_THAT(q.aspect_ratio, WithinAbs(2.0 / std::hypot(1.0, h), 1e-14));
        CHECK_THAT(q.scaled_jacobian, WithinAbs(std::sin(20.0 * std::acos(-1.0) / 180.0) * 2.0 / kSqrt3, 1e-14));
    }
    SECTION("degenerate: skewness 1, scaled Jacobian 0") {
        const ElementQuality q = tri({0.0, 1.0, 2.0}, {0.0, 0.0, 0.0});
        CHECK_THAT(q.skewness, WithinAbs(1.0, 1e-12));
        CHECK_THAT(q.scaled_jacobian, WithinAbs(0.0, 1e-15));
    }
    SECTION("clockwise: negative scaled Jacobian") {
        CHECK(tri({0.0, 0.5, 1.0}, {0.0, 0.5 * kSqrt3, 0.0}).scaled_jacobian < 0.0);
    }
}

TEST_CASE("Mesh quality: quadrilaterals", "[mesh_quality]") {
    SECTION("square and rectangles: angles ideal, aspect ratio = edge ratio") {
        for (double r : {1.0, 4.0, 50.0}) {
            CAPTURE(r);
            const ElementQuality q = quad({0.0, r, r, 0.0}, {0.0, 0.0, 1.0, 1.0});
            CHECK_THAT(q.min_angle, WithinAbs(90.0, 1e-12));
            CHECK_THAT(q.max_angle, WithinAbs(90.0, 1e-12));
            CHECK_THAT(q.aspect_ratio, WithinAbs(r, 1e-12));
            CHECK_THAT(q.skewness, WithinAbs(0.0, 1e-12));
            CHECK_THAT(q.scaled_jacobian, WithinAbs(1.0, 1e-15));
        }
    }
    SECTION("rhombus with 60 / 120 degree corners") {
        const ElementQuality q = quad({0.0, 1.0, 1.5, 0.5}, {0.0, 0.0, 0.5 * kSqrt3, 0.5 * kSqrt3});
        CHECK_THAT(q.min_angle, WithinAbs(60.0, 1e-12));
        CHECK_THAT(q.max_angle, WithinAbs(120.0, 1e-12));
        CHECK_THAT(q.aspect_ratio, WithinAbs(1.0, 1e-15));
        CHECK_THAT(q.skewness, WithinAbs(1.0 / 3.0, 1e-12));
        CHECK_THAT(q.scaled_jacobian, WithinAbs(0.5 * kSqrt3, 1e-15));
    }
    SECTION("trapezoid") {
        // Base 2, top 1, height 1: corners atan(2) = 63.4 and 116.6 degrees.
        const ElementQuality q = quad({0.0, 2.0, 1.5, 0.5}, {0.0, 0.0, 1.0, 1.0});
        CHECK_THAT(q.min_angle, WithinAbs(63.43494882292201, 1e-10)); // atan(2)
        CHECK_THAT(q.max_angle, WithinAbs(180.0 - 63.43494882292201, 1e-10));
        CHECK_THAT(q.aspect_ratio, WithinAbs(2.0, 1e-15));
    }
    SECTION("non-convex: a reflex corner, negative scaled Jacobian") {
        const ElementQuality q = quad({0.0, 2.0, 0.6, 0.0}, {0.0, 0.0, 0.6, 2.0});
        CHECK(q.max_angle > 180.0);
        CHECK(q.scaled_jacobian < 0.0);
        CHECK(q.skewness == 1.0);
    }
}

TEST_CASE("Mesh quality: invariant under translation, rotation and scaling", "[mesh_quality]") {
    const std::array<double, 4> x{0.1, 1.3, 1.1, -0.2}, y{0.0, 0.2, 1.4, 0.9};
    const ElementQuality ref = quad(x, y);
    std::array<double, 4> xt{}, yt{};
    const double c = std::cos(1.1), s = std::sin(1.1);
    for (std::size_t k = 0; k < 4; ++k) {
        xt[k] = 3.0 * (c * x[k] - s * y[k]) + 7.0;
        yt[k] = 3.0 * (s * x[k] + c * y[k]) - 2.0;
    }
    const ElementQuality q = quad(xt, yt);
    CHECK_THAT(q.min_angle, WithinAbs(ref.min_angle, 1e-11));
    CHECK_THAT(q.max_angle, WithinAbs(ref.max_angle, 1e-11));
    CHECK_THAT(q.aspect_ratio, WithinAbs(ref.aspect_ratio, 1e-13));
    CHECK_THAT(q.skewness, WithinAbs(ref.skewness, 1e-13));
    CHECK_THAT(q.scaled_jacobian, WithinAbs(ref.scaled_jacobian, 1e-13));
    // Starting the node list at another corner changes nothing either.
    const ElementQuality r = quad({x[2], x[3], x[0], x[1]}, {y[2], y[3], y[0], y[1]});
    CHECK_THAT(r.skewness, WithinAbs(ref.skewness, 1e-15));
    CHECK_THAT(r.scaled_jacobian, WithinAbs(ref.scaled_jacobian, 1e-15));
}

TEST_CASE("Mesh quality: report of a hybrid mesh", "[mesh_quality]") {
    GridOptions g;
    g.cells = GridCells::InflationLayers;
    g.nx = 4;
    g.ny = 6;
    g.layers = 1;
    g.ly = 3.0;
    const Mesh m = make_hybrid_mesh(grid(g));
    const MeshQualityReport r = assess_mesh_quality(m);
    CHECK(r.triangles.count == 32);
    CHECK(r.quads.count == 8);
    // Quadrilaterals 0.25 x 0.5: aspect ratio 2, rectangles.
    CHECK_THAT(r.quads.aspect_ratio.max, WithinAbs(2.0, 1e-14));
    CHECK_THAT(r.quads.skewness.max, WithinAbs(0.0, 1e-12));
    CHECK_THAT(r.quads.scaled_jacobian.min, WithinAbs(1.0, 1e-14));
    // Triangles: halves of 0.25 x 0.5 rectangles, angles atan(2) and atan(1/2), 90.
    CHECK_THAT(r.triangles.min_angle.min, WithinAbs(std::atan(0.5) * 180.0 / std::acos(-1.0), 1e-10));
    CHECK_THAT(r.triangles.max_angle.max, WithinAbs(90.0, 1e-10));
    CHECK(m.is_quad(r.quads.aspect_ratio.worst_element));
    CHECK_FALSE(m.is_quad(r.triangles.skewness.worst_element));
    CHECK_FALSE(m.is_quad(r.worst_element));
    int total = 0;
    for (int c : r.skewness_histogram) total += c;
    CHECK(total == m.nelem);
    CHECK(r.skewness_histogram[static_cast<std::size_t>(SkewnessClass::Excellent)] == 8);

    const std::string text = format_quality_report(m, r);
    CHECK_THAT(text, ContainsSubstring("Mesh quality: 32 triangles: angles 26.6 - 90.0 deg"));
    CHECK_THAT(text, ContainsSubstring("Mesh quality: 8 quadrilaterals: angles 90.0 - 90.0 deg, aspect ratio max 2.00"));
    CHECK_THAT(text, ContainsSubstring("Mesh quality: equiangle skewness 8 excellent, 0 good, 32 fair, 0 poor, 0 bad"));
    CHECK_THAT(text, ContainsSubstring("most skewed element: triangle"));
    CHECK_THAT(text, !ContainsSubstring("WARNING"));
}

TEST_CASE("Mesh quality: skewness classes and the warning for bad elements", "[mesh_quality]") {
    CHECK(skewness_class(0.0) == SkewnessClass::Excellent);
    CHECK(skewness_class(0.2499) == SkewnessClass::Excellent);
    CHECK(skewness_class(0.25) == SkewnessClass::Good);
    CHECK(skewness_class(0.5) == SkewnessClass::Fair);
    CHECK(skewness_class(0.75) == SkewnessClass::Poor);
    CHECK(skewness_class(0.9) == SkewnessClass::Bad);
    CHECK(std::string(to_string(SkewnessClass::Poor)) == "poor");

    // A mesh with one very flat triangle (angles ~ 2.3, 2.3, 175.4 degrees).
    HybridSpec s;
    s.nodes = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.5, -0.02}};
    s.quads = {{0, 1, 2, 3}};
    s.triangles = {{0, 4, 1}};
    const Mesh m = make_hybrid_mesh(s);
    const MeshQualityReport r = assess_mesh_quality(m);
    CHECK(r.worst_element == 0);
    CHECK(r.worst_skewness > 0.9);
    CHECK(r.skewness_histogram[static_cast<std::size_t>(SkewnessClass::Bad)] == 1);
    const std::string text = format_quality_report(m, r);
    CHECK_THAT(text, ContainsSubstring("WARNING: 1 element(s) with equiangle skewness >= 0.9"));
    CHECK_THAT(text, ContainsSubstring("most skewed element: triangle (element index 0, 0-based), skewness 0.9"));
    CHECK_THAT(text, ContainsSubstring("at (0.5, -0.00666667)"));
}
