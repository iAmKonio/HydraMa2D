// Wall distance d(x): exact Euclidean distance to the nearest point of the
// discrete (piecewise-linear) wall, computed by BruteForceWallDistance and
// AabbTreeWallDistance (SegmentAabbTree, branch-and-bound search).
//
// Verified against analytic distances for
//   * a plane channel (uniform and wall-clustered grids): d = min(y, H - y),
//   * a backward-facing step, including the circular-arc region around the
//     convex step corner and the concave corner,
//   * the bundled Hinterkante benchmark (plate with semicircular trailing edge
//     in a channel), where the polygonal wall approximates a circle and the
//     chord error is bounded by the sagitta,
// plus bit-for-bit agreement between the AABB tree and brute force, pruning
// efficiency, degenerate inputs and metric properties (1-Lipschitz,
// invariance under rigid motions).

#include "WallTestUtils.hpp"
#include "geometry/SegmentTree.hpp"
#include "geometry/WallDistance.hpp"
#include "geometry/WallGeometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Portable deterministic random numbers (std distributions are not
// reproducible across standard libraries).
class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed) {}
    double uniform() { // [0, 1)
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state_ >> 11) * (1.0 / 9007199254740992.0);
    }
    double uniform(double a, double b) { return a + (b - a) * uniform(); }

private:
    std::uint64_t state_;
};

Mesh channel_mesh(int nx, int ny, double lx, double ly, bool clustered) {
    GridSpec g;
    g.lx = lx;
    g.ly = ly;
    g.nx = nx;
    g.ny = ny;
    Mesh m = make_grid_mesh(g);
    // side ids: 1 = bottom, 2 = right, 3 = top, 4 = left
    set_flags_by_side_id(m, {{1, 500}, {2, 504}, {3, 500}, {4, 503}});
    if (clustered) {
        // tanh clustering towards both walls in y, geometric stretching in x;
        // the walls stay at y = 0 and y = ly exactly.
        const double beta = 2.5;
        map_nodes(m, [&](double x, double y) {
            const double eta = y / ly, xi = x / lx;
            const double yy = (eta == 0.0 || eta == 1.0)
                                  ? y
                                  : 0.5 * ly * (1.0 - std::tanh(beta * (1.0 - 2.0 * eta)) / std::tanh(beta));
            const double xx = lx * (std::pow(8.0, xi) - 1.0) / 7.0;
            return std::pair<double, double>{xx, yy};
        });
    }
    return m;
}

// Distance from (x, y) to the horizontal segment y = c, x in [x0, x1].
double horizontal_segment_distance(double x, double y, double x0, double x1, double c) {
    return std::hypot(std::max({x0 - x, 0.0, x - x1}), y - c);
}

std::unique_ptr<WallDistanceCalculator> calculator(const std::string& name) {
    auto method = parse_wall_distance_method(name);
    REQUIRE(method.has_value());
    return make_wall_distance_calculator(*method);
}

// Largest chord sagitta R - sqrt(R^2 - (c/2)^2) of the benchmark trailing
// edge (side id 6), i.e. the maximum gap between the polygon and the circle.
double benchmark_max_sagitta(const Mesh& m) {
    const double r = 0.5;
    double sag = 0.0;
    for (const auto& s : m.iside) {
        if (s.side_id != 6) continue;
        const double c = std::hypot(m.x[s.node2] - m.x[s.node1], m.y[s.node2] - m.y[s.node1]);
        sag = std::max(sag, r - std::sqrt(r * r - 0.25 * c * c));
    }
    return sag;
}

} // namespace

// ---------------------------------------------------------------------------
// Point-segment kernel
// ---------------------------------------------------------------------------

TEST_CASE("Point-segment distance: interior, end point and degenerate cases", "[walldist]") {
    const Segment2D horizontal{0.0, 0.0, 2.0, 0.0};
    CHECK(point_segment_distance_sq(1.0, 1.0, horizontal) == 1.0);   // interior foot point
    CHECK(point_segment_distance_sq(1.0, -3.0, horizontal) == 9.0);  // either side
    CHECK(point_segment_distance_sq(-1.0, 0.0, horizontal) == 1.0);  // beyond a (collinear)
    CHECK(point_segment_distance_sq(3.0, 1.0, horizontal) == 2.0);   // beyond b
    CHECK(point_segment_distance_sq(0.5, 0.0, horizontal) == 0.0);   // on the segment
    CHECK(point_segment_distance_sq(0.0, 0.0, horizontal) == 0.0);   // end points exactly zero
    CHECK(point_segment_distance_sq(2.0, 0.0, horizontal) == 0.0);

    // Oblique 3-4-5 segment: perpendicular offsets of length 5 at a, the
    // midpoint and b (exact in binary floating point).
    const Segment2D oblique{0.0, 0.0, 3.0, 4.0};
    CHECK(point_segment_distance_sq(4.0, -3.0, oblique) == 25.0);
    CHECK(point_segment_distance_sq(5.5, -1.0, oblique) == 25.0);
    CHECK(point_segment_distance_sq(7.0, 1.0, oblique) == 25.0);
    CHECK(point_segment_distance_sq(3.0, 4.0, oblique) == 0.0);

    // Zero-length segment degrades to a point distance.
    const Segment2D point{1.0, 1.0, 1.0, 1.0};
    CHECK(point_segment_distance_sq(4.0, 5.0, point) == 25.0);
    CHECK(point_segment_distance_sq(1.0, 1.0, point) == 0.0);

    // Independent of the segment direction (up to round-off).
    Rng rng(7);
    const Segment2D s{0.3, -1.7, 2.9, 0.4};
    const Segment2D r{s.x2, s.y2, s.x1, s.y1};
    for (int k = 0; k < 200; ++k) {
        const double px = rng.uniform(-3.0, 5.0), py = rng.uniform(-4.0, 3.0);
        CHECK_THAT(point_segment_distance_sq(px, py, s),
                   WithinRel(point_segment_distance_sq(px, py, r), 1e-12));
    }
}

// ---------------------------------------------------------------------------
// Analytic geometries
// ---------------------------------------------------------------------------

TEST_CASE("Channel: nodal wall distance equals min(y, H - y)", "[walldist][analytic]") {
    const std::string method = GENERATE("brute-force", "exact");
    const bool clustered = GENERATE(false, true);
    const double lx = 6.0, ly = 2.0;
    const Mesh m = channel_mesh(30, 16, lx, ly, clustered);
    const WallGeometry walls(m, *calculator(method));
    INFO("method " << method << (clustered ? ", clustered grid" : ", uniform grid"));

    REQUIRE(walls.has_walls());
    REQUIRE(static_cast<int>(walls.wall_distance().size()) == m.npoin);
    for (int ip = 0; ip < m.npoin; ++ip) {
        const double y = m.y[ip];
        const double exact = std::min(y, ly - y);
        INFO("node " << ip << " at (" << m.x[ip] << ", " << y << ")");
        CHECK_THAT(walls.wall_distance()[ip], WithinAbs(exact, 1e-12));

        // The nearest face lies on the nearer wall (skip the centre line tie).
        const int f = walls.nearest_wall_face()[ip];
        REQUIRE(f >= 0);
        const double wall_y = m.y[walls.face(f).node1];
        if (y < 0.5 * ly - 1e-9) CHECK(wall_y == 0.0);
        if (y > 0.5 * ly + 1e-9) CHECK(wall_y == ly);
    }
}

TEST_CASE("Channel: distance at arbitrary points, inside and outside the wall extent",
          "[walldist][analytic]") {
    const double lx = 6.0, ly = 2.0;
    const Mesh m = channel_mesh(24, 8, lx, ly, true);
    const WallGeometry walls(m, BruteForceWallDistance());
    const SegmentAabbTree tree(wall_segments(m, walls.faces()));

    Rng rng(11);
    for (int k = 0; k < 2000; ++k) {
        // Points beyond the channel ends see the wall end points (corners).
        const double px = rng.uniform(-2.0, lx + 2.0), py = rng.uniform(-1.0, ly + 1.0);
        const double exact = std::min(horizontal_segment_distance(px, py, 0.0, lx, 0.0),
                                      horizontal_segment_distance(px, py, 0.0, lx, ly));
        const auto nearest = tree.nearest(px, py);
        INFO("point (" << px << ", " << py << ")");
        CHECK_THAT(std::sqrt(nearest.distance_sq), WithinAbs(exact, 1e-12));
    }
}

TEST_CASE("Backward-facing step: nodal wall distance matches the analytic distance",
          "[walldist][analytic]") {
    const std::string method = GENERATE("brute-force", "exact");
    // Dyadic grid (exact coordinates) and a non-dyadic one.
    const StepSpec spec = GENERATE(StepSpec{4.0, 1.5, 1.0, 0.5, 32, 12}, StepSpec{3.3, 1.1, 0.9, 0.4, 33, 11});
    const Mesh m = make_step_mesh(spec);
    const WallGeometry walls(m, *calculator(method));
    INFO("method " << method << ", step lx=" << spec.lx << " ly=" << spec.ly);

    REQUIRE(m.npoin == (spec.nx + 1) * (spec.ny + 1) -
                           static_cast<int>(std::lround(spec.xs * spec.nx / spec.lx)) *
                               static_cast<int>(std::lround(spec.hs * spec.ny / spec.ly)));
    for (int ip = 0; ip < m.npoin; ++ip) {
        INFO("node " << ip << " at (" << m.x[ip] << ", " << m.y[ip] << ")");
        CHECK_THAT(walls.wall_distance()[ip], WithinAbs(step_wall_distance(spec, m.x[ip], m.y[ip]), 1e-12));
    }
}

TEST_CASE("Backward-facing step: arbitrary points incl. the convex-corner arc region",
          "[walldist][analytic]") {
    const StepSpec spec;
    const Mesh m = make_step_mesh(spec);
    const WallGeometry walls(m);
    const SegmentAabbTree tree(wall_segments(m, walls.faces()));
    const int corner = find_node(m, spec.xs, spec.hs);
    REQUIRE(corner >= 0);

    SECTION("random points in the fluid") {
        Rng rng(23);
        int tested = 0;
        while (tested < 3000) {
            const double px = rng.uniform(0.0, spec.lx), py = rng.uniform(0.0, spec.ly);
            if (px < spec.xs && py < spec.hs) continue; // inside the step block
            ++tested;
            INFO("point (" << px << ", " << py << ")");
            CHECK_THAT(std::sqrt(tree.nearest(px, py).distance_sq),
                       WithinAbs(step_wall_distance(spec, px, py), 1e-12));
        }
    }

    SECTION("circular arc around the convex corner: d = r, nearest face touches the corner") {
        const double r = 0.2;
        for (int k = 1; k < 20; ++k) {
            const double theta = 0.5 * 3.14159265358979323846 * k / 20.0;
            const double px = spec.xs + r * std::cos(theta), py = spec.hs + r * std::sin(theta);
            const auto nearest = tree.nearest(px, py);
            INFO("theta " << theta);
            CHECK_THAT(std::sqrt(nearest.distance_sq), WithinAbs(r, 1e-12));
            const WallFace& f = walls.face(nearest.index);
            CHECK((f.node1 == corner || f.node2 == corner));
        }
    }

    SECTION("concave corner: bisector points are equidistant from bottom and step face") {
        for (double t : {0.05, 0.1, 0.2, 0.3}) {
            CHECK_THAT(std::sqrt(tree.nearest(spec.xs + t, t).distance_sq), WithinAbs(t, 1e-12));
        }
    }
}

TEST_CASE("Benchmark mesh: wall distance matches the plate-in-channel geometry",
          "[walldist][analytic][hinterkante]") {
    const Mesh& m = benchmark_mesh();
    const WallGeometry walls(m);
    const double sagitta = benchmark_max_sagitta(m);
    REQUIRE(sagitta > 0.0);
    REQUIRE(sagitta < 2e-4); // 76 chords on a semicircle of radius 0.5
    // Trailing-edge nodes lie on the circle to ~2e-8 (rounded coordinates).
    const double node_tol = 1e-7;

    int flat_dominated = 0, circle_region = 0;
    for (int ip = 0; ip < m.npoin; ++ip) {
        const double d = walls.wall_distance()[ip];
        const BenchmarkDistance exact = benchmark_wall_distance(m.x[ip], m.y[ip]);
        INFO("node " << ip << " at (" << m.x[ip] << ", " << m.y[ip] << "), d = " << d);
        if (exact.flat < exact.circle - 1e-6) {
            // Straight walls are represented exactly by the mesh.
            ++flat_dominated;
            CHECK_THAT(d, WithinAbs(exact.flat, 1e-12));
        } else {
            // The polygon is inscribed in the circle: its distance is larger
            // by at most the sagitta.
            ++circle_region;
            CHECK(d >= exact.exact() - node_tol);
            CHECK(d <= std::min(exact.flat, exact.circle + sagitta) + node_tol);
        }
    }
    // Both branches are exercised substantially.
    INFO(flat_dominated << " nodes nearest to straight walls, " << circle_region << " near the trailing edge");
    CHECK(flat_dominated > m.npoin / 2);
    CHECK(circle_region > 1000);
}

// ---------------------------------------------------------------------------
// AABB tree: exactness, efficiency, robustness
// ---------------------------------------------------------------------------

TEST_CASE("AABB tree and brute force agree bit-for-bit on the benchmark mesh", "[walldist][aabb]") {
    const Mesh& m = benchmark_mesh();
    const WallGeometry reference(m, BruteForceWallDistance());
    const int leaf_size = GENERATE(1, 2, 4, 16);
    const WallGeometry accelerated(m, AabbTreeWallDistance(leaf_size));
    INFO("leaf size " << leaf_size);

    int mismatches = 0;
    for (int ip = 0; ip < m.npoin; ++ip) {
        if (accelerated.wall_distance()[ip] != reference.wall_distance()[ip] ||
            accelerated.nearest_wall_face()[ip] != reference.nearest_wall_face()[ip]) {
            ++mismatches;
        }
    }
    CHECK(mismatches == 0);
}

TEST_CASE("AABB tree and brute force agree bit-for-bit at arbitrary points", "[walldist][aabb]") {
    const Mesh& m = benchmark_mesh();
    const WallGeometry walls(m, BruteForceWallDistance());
    const std::vector<Segment2D> segments = wall_segments(m, walls.faces());
    const int leaf_size = GENERATE(1, 4, 8);
    const SegmentAabbTree tree(segments, leaf_size);

    Rng rng(99);
    int mismatches = 0;
    for (int k = 0; k < 20000; ++k) {
        // Mostly around the domain, some far away (up to 1e4 lengths).
        const double scale = (k % 10 == 0) ? 1.0e4 : 15.0;
        const double px = rng.uniform(-scale, scale), py = rng.uniform(-scale, scale);
        const int hint = (k % 3 == 0) ? -1 : static_cast<int>(rng.uniform() * segments.size());
        const auto a = tree.nearest(px, py, hint);
        const auto b = SegmentAabbTree::nearest_brute_force(segments, px, py);
        if (a.distance_sq != b.distance_sq || a.index != b.index) ++mismatches;
    }
    CHECK(mismatches == 0);
}

TEST_CASE("AABB tree ties: duplicate and touching segments resolve to the lowest index",
          "[walldist][aabb]") {
    // Segments 0..3 share the corner (1, 1); 4 and 5 are duplicates of 0.
    const std::vector<Segment2D> segments = {
        {0.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 2.0}, {1.0, 1.0, 2.0, 1.0},
        {1.0, 0.0, 1.0, 1.0}, {0.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 0.0, 1.0},
    };
    const SegmentAabbTree tree(segments, 1);
    for (int hint = -1; hint < static_cast<int>(segments.size()); ++hint) {
        const auto at_corner = tree.nearest(1.0, 1.0, hint);
        CHECK(at_corner.distance_sq == 0.0);
        CHECK(at_corner.index == 0);
        const auto above_left = tree.nearest(0.5, 1.5, hint); // equidistant to 0 (and its duplicates) and 1
        CHECK(above_left.distance_sq == 0.25);
        CHECK(above_left.index == 0);
        const auto right = tree.nearest(1.5, 0.5, hint); // equidistant to 2 and 3
        CHECK(right.index == 2);
    }
}

TEST_CASE("AABB tree prunes: far fewer segment tests than a brute-force scan", "[walldist][aabb]") {
    SECTION("benchmark mesh nodes with warm start") {
        const Mesh& m = benchmark_mesh();
        const WallGeometry walls(m, BruteForceWallDistance());
        const SegmentAabbTree tree(wall_segments(m, walls.faces()));
        SegmentAabbTree::QueryStats stats;
        int hint = -1;
        for (int ip = 0; ip < m.npoin; ++ip) hint = tree.nearest(m.x[ip], m.y[ip], hint, &stats).index;
        const double per_query = static_cast<double>(stats.segments_tested) / m.npoin;
        INFO(walls.num_faces() << " wall faces, " << per_query << " segment tests per node");
        CHECK(per_query < 0.1 * walls.num_faces());
    }

    SECTION("polygon with 16384 edges, random points") {
        const int n = 16384;
        std::vector<Segment2D> circle;
        for (int k = 0; k < n; ++k) {
            const double a0 = 2.0 * 3.14159265358979323846 * k / n;
            const double a1 = 2.0 * 3.14159265358979323846 * (k + 1) / n;
            circle.push_back({std::cos(a0), std::sin(a0), std::cos(a1), std::sin(a1)});
        }
        const SegmentAabbTree tree(circle);
        CHECK(tree.depth() <= 14); // ceil(log2(16384 / 4)) + 1 = 13 levels
        SegmentAabbTree::QueryStats stats;
        Rng rng(5);
        const int queries = 2000;
        for (int k = 0; k < queries; ++k) {
            const double r = rng.uniform(0.1, 3.0), a = rng.uniform(0.0, 6.283185307179586);
            const double px = r * std::cos(a), py = r * std::sin(a);
            const auto nearest = tree.nearest(px, py, -1, &stats);
            CHECK_THAT(std::sqrt(nearest.distance_sq), WithinAbs(std::abs(r - 1.0), 2e-8)); // chord error
        }
        const double per_query = static_cast<double>(stats.segments_tested) / queries;
        INFO(per_query << " segment tests per query");
        CHECK(per_query < 0.01 * n);
    }
}

TEST_CASE("Wall distance: degenerate inputs", "[walldist]") {
    SECTION("empty tree") {
        const SegmentAabbTree tree(std::vector<Segment2D>{});
        CHECK(tree.empty());
        CHECK(tree.depth() == 0);
        const auto nearest = tree.nearest(1.0, 2.0, 0);
        CHECK(nearest.index == -1);
        CHECK(nearest.distance_sq == kInf);
    }

    SECTION("single and zero-length segments") {
        const SegmentAabbTree single({{0.0, 0.0, 1.0, 0.0}});
        CHECK(single.nearest(0.5, 2.0).distance_sq == 4.0);
        const SegmentAabbTree points({{0.0, 0.0, 0.0, 0.0}, {3.0, 0.0, 3.0, 0.0}});
        const auto nearest = points.nearest(2.0, 0.0);
        CHECK(nearest.index == 1);
        CHECK(nearest.distance_sq == 1.0);
    }

    SECTION("invalid leaf size") {
        CHECK_THROWS_AS(SegmentAabbTree({{0.0, 0.0, 1.0, 0.0}}, 0), std::invalid_argument);
    }

    SECTION("mesh without walls: infinite distance, no nearest face") {
        const std::string method = GENERATE("brute-force", "exact");
        GridSpec g;
        g.nx = 3;
        g.ny = 2;
        const Mesh m = make_grid_mesh(g, 504);
        const WallGeometry walls(m, *calculator(method));
        CHECK_FALSE(walls.has_walls());
        CHECK(walls.num_wall_nodes() == 0);
        for (int ip = 0; ip < m.npoin; ++ip) {
            CHECK(walls.wall_distance()[ip] == kInf);
            CHECK(walls.nearest_wall_face()[ip] == -1);
        }
    }
}

// ---------------------------------------------------------------------------
// Metric properties
// ---------------------------------------------------------------------------

TEST_CASE("Wall nodes have zero distance, all other nodes a positive one", "[walldist]") {
    const StepSpec spec;
    const Mesh step = make_step_mesh(spec);
    const Mesh& bench = benchmark_mesh();
    for (const Mesh* m : {&step, &bench}) {
        const WallGeometry walls(*m);
        int zero = 0;
        for (int ip = 0; ip < m->npoin; ++ip) {
            const double d = walls.wall_distance()[ip];
            if (walls.is_wall_node(ip)) {
                ++zero;
                CHECK(d == 0.0);
                const WallFace& f = walls.face(walls.nearest_wall_face()[ip]);
                CHECK((f.node1 == ip || f.node2 == ip));
            } else {
                CHECK(d > 0.0);
            }
        }
        CHECK(zero == walls.num_wall_nodes());
    }
}

TEST_CASE("Wall distance is 1-Lipschitz", "[walldist]") {
    const StepSpec spec;
    const Mesh m = make_step_mesh(spec);
    const WallGeometry walls(m);
    const SegmentAabbTree tree(wall_segments(m, walls.faces()));
    Rng rng(3);
    for (int k = 0; k < 2000; ++k) {
        const double px = rng.uniform(-1.0, 5.0), py = rng.uniform(-1.0, 2.5);
        const double qx = px + rng.uniform(-0.3, 0.3), qy = py + rng.uniform(-0.3, 0.3);
        const double dp = std::sqrt(tree.nearest(px, py).distance_sq);
        const double dq = std::sqrt(tree.nearest(qx, qy).distance_sq);
        CHECK(std::abs(dp - dq) <= std::hypot(px - qx, py - qy) * (1.0 + 1e-12) + 1e-14);
    }
}

TEST_CASE("Wall distance is invariant under rigid motions of the mesh", "[walldist]") {
    const StepSpec spec;
    const Mesh m = make_step_mesh(spec);
    const WallGeometry reference(m);

    const double angle = GENERATE(0.5 * 3.14159265358979323846, 0.7, -2.1);
    const double cx = 123.25, cy = -45.5; // translation
    Mesh moved = m;
    map_nodes(moved, [&](double x, double y) {
        return std::pair<double, double>{cx + std::cos(angle) * x - std::sin(angle) * y,
                                          cy + std::sin(angle) * x + std::cos(angle) * y};
    });
    const WallGeometry walls(moved);
    INFO("rotation angle " << angle);
    for (int ip = 0; ip < m.npoin; ++ip) {
        CHECK_THAT(walls.wall_distance()[ip], WithinAbs(reference.wall_distance()[ip], 1e-12));
    }
}

TEST_CASE("Wall distance method names and factory", "[walldist]") {
    CHECK(parse_wall_distance_method("exact") == WallDistanceMethod::AabbTree);
    CHECK(parse_wall_distance_method("aabb-tree") == WallDistanceMethod::AabbTree);
    CHECK(parse_wall_distance_method("brute-force") == WallDistanceMethod::BruteForce);
    CHECK_FALSE(parse_wall_distance_method("poisson").has_value()); // not implemented yet
    CHECK_FALSE(parse_wall_distance_method("").has_value());

    CHECK(make_wall_distance_calculator(WallDistanceMethod::AabbTree)->name() == "aabb-tree");
    CHECK(make_wall_distance_calculator(WallDistanceMethod::BruteForce)->name() == "brute-force");
}
