#pragma once

// Exact nearest-segment queries in 2D accelerated by an axis-aligned bounding
// box (AABB) tree.
//
// Algorithms (see docs/TURBULENCE_DESIGN.md, section 11):
//  * point-segment distance: C. Ericson, Real-Time Collision Detection (2005),
//    section 5.1.2 (closest point on a segment, parameter clamped to [0, 1]);
//  * point-box distance ("MINDIST"): Ericson (2005), section 5.1.3;
//    Roussopoulos, Kelley & Vincent, ACM SIGMOD (1995);
//  * top-down bounding volume hierarchy, median split along the longest axis
//    of the centroid bounds: Ericson (2005), chapter 6;
//  * depth-first branch-and-bound nearest-neighbour search (nearer child
//    first, prune when MINDIST exceeds the best distance): Friedman, Bentley
//    & Finkel, ACM TOMS 3(3) (1977); Roussopoulos et al. (1995).
//
// Exactness contract: nearest() returns bit-for-bit the same (distance,
// index) pair as nearest_brute_force(). Pruning keeps a round-off slack of
// 1e-12 relative to the scene size, and ties are resolved in favour of the
// lowest segment index in both routines.

#include <limits>
#include <vector>

namespace hydrama {

struct Segment2D {
    double x1 = 0.0, y1 = 0.0; // end point a
    double x2 = 0.0, y2 = 0.0; // end point b
};

// Squared Euclidean distance from (px, py) to the closed segment [a, b].
// Closest points at the end points are handled by exact branches, so the
// distance of an end point to its own segment is exactly zero. Zero-length
// segments degrade gracefully to point distances.
inline double point_segment_distance_sq(double px, double py, const Segment2D& s) noexcept {
    const double abx = s.x2 - s.x1;
    const double aby = s.y2 - s.y1;
    const double apx = px - s.x1;
    const double apy = py - s.y1;
    const double proj = apx * abx + apy * aby; // (p - a) . (b - a)
    if (proj <= 0.0) {                        // closest point: a
        return apx * apx + apy * apy;
    }
    const double len2 = abx * abx + aby * aby;
    if (proj >= len2) { // closest point: b
        const double bpx = px - s.x2;
        const double bpy = py - s.y2;
        return bpx * bpx + bpy * bpy;
    }
    const double t = proj / len2; // 0 < t < 1
    const double dx = apx - t * abx;
    const double dy = apy - t * aby;
    return dx * dx + dy * dy;
}

class SegmentAabbTree {
public:
    // Result of a nearest-segment query. index is the position of the segment
    // in the vector passed to the constructor, or -1 if the tree is empty.
    struct Nearest {
        double distance_sq = std::numeric_limits<double>::infinity();
        int index = -1;
    };

    // Optional work counters (for tests and profiling).
    struct QueryStats {
        long long nodes_visited = 0;
        long long segments_tested = 0;
    };

    static constexpr int kDefaultLeafSize = 4;

    SegmentAabbTree() = default;
    explicit SegmentAabbTree(const std::vector<Segment2D>& segments,
                             int leaf_size = kDefaultLeafSize);

    bool empty() const noexcept { return segments_.empty(); }
    int size() const noexcept { return static_cast<int>(segments_.size()); }
    int node_count() const noexcept { return static_cast<int>(nodes_.size()); }
    int depth() const noexcept { return depth_; } // number of levels (0 if empty)

    // Nearest segment to (px, py). hint (optional) is a segment index that is
    // likely to be close - e.g. the answer for a neighbouring point - and is
    // used to initialise the pruning bound. It never changes the result.
    Nearest nearest(double px, double py, int hint = -1,
                    QueryStats* stats = nullptr) const;

    // Reference O(N) scan with the same distance function and tie-breaking.
    static Nearest nearest_brute_force(const std::vector<Segment2D>& segments,
                                       double px, double py) noexcept;

private:
    struct Node {
        double xmin, ymin, xmax, ymax; // bounding box of all segments below
        int first;                     // leaf: first segment (tree order)
        int count;                     // leaf: number of segments, 0 for inner nodes
        int left, right;               // inner node: children
    };

    static constexpr int kMaxStack = 128;

    std::vector<Node> nodes_;
    std::vector<Segment2D> segments_; // segments in tree (leaf) order
    std::vector<int> original_;       // tree position -> original index
    std::vector<int> position_;       // original index -> tree position
    double slack_ = 0.0;              // absolute round-off slack for pruning
    int depth_ = 0;

    int build(const std::vector<Segment2D>& segments, std::vector<int>& order,
              int begin, int end, int leaf_size, int level,
              const std::vector<double>& cx, const std::vector<double>& cy);

    static double box_distance_sq(const Node& n, double px, double py) noexcept;
};

} // namespace hydrama
