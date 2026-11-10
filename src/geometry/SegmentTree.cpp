#include "geometry/SegmentTree.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hydrama {

namespace {

// Pruning slack relative to the scene size / current best distance. Round-off
// in the point-box and point-segment distances is a few ulps of the operand
// differences, i.e. ~1e-16 relative; 1e-12 is a wide safety margin that costs
// no measurable pruning efficiency.
constexpr double kRelativeSlack = 1.0e-12;

} // namespace

SegmentAabbTree::SegmentAabbTree(const std::vector<Segment2D>& segments, int leaf_size) {
    if (leaf_size < 1) {
        throw std::invalid_argument("SegmentAabbTree: leaf_size must be >= 1");
    }
    const int n = static_cast<int>(segments.size());
    if (n == 0) return;

    // Centroids drive the split; the scene diagonal scales the pruning slack.
    std::vector<double> cx(n), cy(n);
    double xmin = std::numeric_limits<double>::max(), ymin = xmin;
    double xmax = std::numeric_limits<double>::lowest(), ymax = xmax;
    for (int i = 0; i < n; ++i) {
        const Segment2D& s = segments[i];
        cx[i] = 0.5 * (s.x1 + s.x2);
        cy[i] = 0.5 * (s.y1 + s.y2);
        xmin = std::min({xmin, s.x1, s.x2});
        xmax = std::max({xmax, s.x1, s.x2});
        ymin = std::min({ymin, s.y1, s.y2});
        ymax = std::max({ymax, s.y1, s.y2});
    }
    slack_ = kRelativeSlack * std::hypot(xmax - xmin, ymax - ymin);

    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;

    // A median split gives depth <= ceil(log2(n)) + 1, so 2n nodes suffice.
    nodes_.reserve(2 * static_cast<std::size_t>(n));
    build(segments, order, 0, n, leaf_size, 1, cx, cy);

    if (depth_ + 1 > kMaxStack) { // unreachable for a median split and int-sized n
        throw std::runtime_error("SegmentAabbTree: tree too deep");
    }

    // Store segments contiguously in leaf order (cache-friendly leaf scans).
    segments_.resize(n);
    original_ = order;
    position_.resize(n);
    for (int k = 0; k < n; ++k) {
        segments_[k] = segments[order[k]];
        position_[order[k]] = k;
    }
}

int SegmentAabbTree::build(const std::vector<Segment2D>& segments, std::vector<int>& order,
                           int begin, int end, int leaf_size, int level,
                           const std::vector<double>& cx, const std::vector<double>& cy) {
    depth_ = std::max(depth_, level);

    // Bounding box of all segments in order[begin, end).
    Node node{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
              std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(),
              begin, 0, -1, -1};
    for (int k = begin; k < end; ++k) {
        const Segment2D& s = segments[order[k]];
        node.xmin = std::min({node.xmin, s.x1, s.x2});
        node.xmax = std::max({node.xmax, s.x1, s.x2});
        node.ymin = std::min({node.ymin, s.y1, s.y2});
        node.ymax = std::max({node.ymax, s.y1, s.y2});
    }
    const int id = static_cast<int>(nodes_.size());
    nodes_.push_back(node);

    const int count = end - begin;
    if (count <= leaf_size) {
        nodes_[id].count = count;
        return id;
    }

    // Split at the median centroid along the longest axis of the centroid bounds.
    double cxmin = std::numeric_limits<double>::max(), cymin = cxmin;
    double cxmax = std::numeric_limits<double>::lowest(), cymax = cxmax;
    for (int k = begin; k < end; ++k) {
        const int i = order[k];
        cxmin = std::min(cxmin, cx[i]);
        cxmax = std::max(cxmax, cx[i]);
        cymin = std::min(cymin, cy[i]);
        cymax = std::max(cymax, cy[i]);
    }
    const bool split_x = (cxmax - cxmin) >= (cymax - cymin);
    const std::vector<double>& key = split_x ? cx : cy;

    const int mid = begin + count / 2;
    // Ties on the key are broken by index so the tree is fully deterministic.
    std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
                     [&key](int a, int b) { return key[a] < key[b] || (key[a] == key[b] && a < b); });

    const int left = build(segments, order, begin, mid, leaf_size, level + 1, cx, cy);
    const int right = build(segments, order, mid, end, leaf_size, level + 1, cx, cy);
    nodes_[id].left = left;
    nodes_[id].right = right;
    return id;
}

double SegmentAabbTree::box_distance_sq(const Node& n, double px, double py) noexcept {
    const double dx = std::max({n.xmin - px, 0.0, px - n.xmax});
    const double dy = std::max({n.ymin - py, 0.0, py - n.ymax});
    return dx * dx + dy * dy;
}

SegmentAabbTree::Nearest SegmentAabbTree::nearest(double px, double py, int hint,
                                                   QueryStats* stats) const {
    Nearest best;
    if (nodes_.empty()) return best;

    // Subtrees whose box distance exceeds `bound` cannot contain a segment
    // that beats (or ties with) the current best; see kRelativeSlack.
    double bound = std::numeric_limits<double>::infinity();
    auto update_bound = [&]() {
        const double d = std::sqrt(best.distance_sq);
        const double b = d + kRelativeSlack * d + slack_;
        bound = b * b;
    };

    if (hint >= 0 && hint < size()) {
        best.distance_sq = point_segment_distance_sq(px, py, segments_[position_[hint]]);
        best.index = hint;
        update_bound();
        if (stats) ++stats->segments_tested;
    }

    struct Item {
        int node;
        double d2;
    };
    Item stack[kMaxStack];
    int top = 0;

    const double d2_root = box_distance_sq(nodes_[0], px, py);
    if (d2_root <= bound) stack[top++] = {0, d2_root};

    while (top > 0) {
        const Item item = stack[--top];
        if (item.d2 > bound) continue; // bound tightened since the push
        const Node& node = nodes_[item.node];
        if (stats) ++stats->nodes_visited;

        if (node.count > 0) {
            for (int k = node.first; k < node.first + node.count; ++k) {
                const double d2 = point_segment_distance_sq(px, py, segments_[k]);
                const int idx = original_[k];
                if (stats) ++stats->segments_tested;
                if (d2 < best.distance_sq || (d2 == best.distance_sq && idx < best.index)) {
                    best.distance_sq = d2;
                    best.index = idx;
                    update_bound();
                }
            }
            continue;
        }

        const double dl = box_distance_sq(nodes_[node.left], px, py);
        const double dr = box_distance_sq(nodes_[node.right], px, py);
        // Push the farther child first so the nearer one is processed next.
        if (dl <= dr) {
            if (dr <= bound) stack[top++] = {node.right, dr};
            if (dl <= bound) stack[top++] = {node.left, dl};
        } else {
            if (dl <= bound) stack[top++] = {node.left, dl};
            if (dr <= bound) stack[top++] = {node.right, dr};
        }
    }
    return best;
}

SegmentAabbTree::Nearest SegmentAabbTree::nearest_brute_force(const std::vector<Segment2D>& segments,
                                                              double px, double py) noexcept {
    Nearest best;
    const int n = static_cast<int>(segments.size());
    for (int i = 0; i < n; ++i) {
        const double d2 = point_segment_distance_sq(px, py, segments[i]);
        if (d2 < best.distance_sq) { // strict: the lowest index wins ties
            best.distance_sq = d2;
            best.index = i;
        }
    }
    return best;
}

} // namespace hydrama
