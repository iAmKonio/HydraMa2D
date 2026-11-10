#include "geometry/WallDistance.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace hydrama {

std::optional<WallDistanceMethod> parse_wall_distance_method(std::string_view name) {
    if (name == "exact" || name == "aabb-tree") return WallDistanceMethod::AabbTree;
    if (name == "brute-force") return WallDistanceMethod::BruteForce;
    return std::nullopt;
}

std::vector<Segment2D> wall_segments(const Mesh& mesh, const std::vector<WallFace>& faces) {
    std::vector<Segment2D> segments;
    segments.reserve(faces.size());
    for (const WallFace& f : faces) {
        segments.push_back({mesh.x[f.node1], mesh.y[f.node1], mesh.x[f.node2], mesh.y[f.node2]});
    }
    return segments;
}

namespace {

WallDistanceField empty_field(int npoin) {
    WallDistanceField field;
    field.distance.assign(npoin, std::numeric_limits<double>::infinity());
    field.nearest_face.assign(npoin, -1);
    return field;
}

} // namespace

WallDistanceField BruteForceWallDistance::compute(const Mesh& mesh,
                                                  const std::vector<WallFace>& faces) const {
    WallDistanceField field = empty_field(mesh.npoin);
    const std::vector<Segment2D> segments = wall_segments(mesh, faces);
    for (int ip = 0; ip < mesh.npoin; ++ip) {
        const auto best = SegmentAabbTree::nearest_brute_force(segments, mesh.x[ip], mesh.y[ip]);
        field.distance[ip] = std::sqrt(best.distance_sq);
        field.nearest_face[ip] = best.index;
    }
    return field;
}

WallDistanceField AabbTreeWallDistance::compute(const Mesh& mesh,
                                                const std::vector<WallFace>& faces) const {
    WallDistanceField field = empty_field(mesh.npoin);
    if (faces.empty()) return field;

    const SegmentAabbTree tree(wall_segments(mesh, faces), leaf_size_);

    // Warm start: mesh numbering is spatially coherent, so the nearest face of
    // the previous node is a good initial bound (it never changes the result).
    int hint = -1;
    for (int ip = 0; ip < mesh.npoin; ++ip) {
        const auto best = tree.nearest(mesh.x[ip], mesh.y[ip], hint);
        field.distance[ip] = std::sqrt(best.distance_sq);
        field.nearest_face[ip] = best.index;
        hint = best.index;
    }
    return field;
}

std::unique_ptr<WallDistanceCalculator> make_wall_distance_calculator(WallDistanceMethod method) {
    switch (method) {
    case WallDistanceMethod::AabbTree: return std::make_unique<AabbTreeWallDistance>();
    case WallDistanceMethod::BruteForce: return std::make_unique<BruteForceWallDistance>();
    }
    throw std::invalid_argument("make_wall_distance_calculator: unknown method");
}

} // namespace hydrama
