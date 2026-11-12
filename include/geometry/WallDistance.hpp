#pragma once

// Wall distance d(x) at the mesh nodes: Strategy interface and
// implementations.
//
// d is the Euclidean distance to the nearest point of the discrete
// (piecewise-linear) wall, i.e. the minimum over all wall faces of the
// point-segment distance. It is required by the Spalart-Allmaras destruction
// term, the SST blending functions F1/F2, low-Re damping functions and wall
// functions (y+). It is computed once per run (static mesh).
//
// Implementations:
//   BruteForceWallDistance  O(N * Nw) reference scan (tests, tiny meshes)
//   AabbTreeWallDistance    O(N log Nw) exact search with an AABB tree
//                           (default); bit-for-bit identical to brute force
// Planned: Poisson-equation based distance (Spalding 1994, Tucker 2003).

#include "Mesh.hpp"
#include "geometry/SegmentTree.hpp"

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace hydrama {

// One wall face (boundary side whose flag classifies as BoundaryKind::Wall).
struct WallFace {
    int side = -1;          // index into Mesh::iside
    int node1 = -1;         // 0-based end nodes, in the order of Mesh::iside
    int node2 = -1;
    int elem = -1;          // element adjacent to the face
    int opposite_node = -1; // vertex of a triangle `elem` that is not on the face (-1 for quadrilaterals)
    double nx = 0.0;        // unit normal pointing INTO the fluid
    double ny = 0.0;
    double length = 0.0;
    double yp = 0.0;        // height of `elem` above the face: triangles 2 * area / length,
                            // quadrilaterals the mean distance of the two off-face nodes
};

struct WallDistanceField {
    std::vector<double> distance; // per mesh node; +inf if there are no walls
    std::vector<int> nearest_face; // per mesh node; index into the face list, -1 if none
};

enum class WallDistanceMethod {
    AabbTree,   // exact, accelerated (default)
    BruteForce, // exact, O(N * Nw) reference
};

// "exact" | "aabb-tree" -> AabbTree, "brute-force" -> BruteForce.
std::optional<WallDistanceMethod> parse_wall_distance_method(std::string_view name);

class WallDistanceCalculator {
public:
    virtual ~WallDistanceCalculator() = default;
    virtual std::string_view name() const = 0;
    virtual WallDistanceField compute(const Mesh& mesh, const std::vector<WallFace>& faces) const = 0;
};

class BruteForceWallDistance final : public WallDistanceCalculator {
public:
    std::string_view name() const override { return "brute-force"; }
    WallDistanceField compute(const Mesh& mesh, const std::vector<WallFace>& faces) const override;
};

class AabbTreeWallDistance final : public WallDistanceCalculator {
public:
    explicit AabbTreeWallDistance(int leaf_size = SegmentAabbTree::kDefaultLeafSize)
        : leaf_size_(leaf_size) {}
    std::string_view name() const override { return "aabb-tree"; }
    WallDistanceField compute(const Mesh& mesh, const std::vector<WallFace>& faces) const override;

private:
    int leaf_size_;
};

std::unique_ptr<WallDistanceCalculator> make_wall_distance_calculator(WallDistanceMethod method);

// Geometric segments of the wall faces (same order as `faces`).
std::vector<Segment2D> wall_segments(const Mesh& mesh, const std::vector<WallFace>& faces);

} // namespace hydrama
