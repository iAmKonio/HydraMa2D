#pragma once

// Wall metrics for turbulence modelling and wall functions.
//
// Built once from the raw mesh data (coordinates, connectivity, boundary
// sides with flags); it does not depend on the Mesh::compute_* routines and
// does not keep a reference to the mesh.
//
//  * wall faces: boundary sides whose flag classifies as BoundaryKind::Wall,
//    with adjacent element, opposite node, length, INWARD unit normal and the
//    first-cell height y_p = 2 * area / length (triangles). On a
//    quadrilateral, y_p is the mean distance of its two off-face nodes from
//    the face, and the opposite node is -1;
//  * wall nodes (ascending node id) with a length-weighted inward normal
//    (bisector at corners; unlike Mesh::wall_node_norm it is never zeroed
//    except at cusps where the face normals cancel exactly), a
//    length-weighted mean y_p and a CSR node -> face map;
//  * wall distance d and nearest wall face for every mesh node.
//
// Inward normals are oriented towards the opposite node of the adjacent
// element (quadrilaterals: the mean of the off-face nodes), so they do not
// depend on the node order of the boundary side.
//
// Throws std::invalid_argument for inconsistent input (node / element ids out
// of range, a wall side that is not an edge of its element, zero-length
// faces, degenerate adjacent elements).

#include "Mesh.hpp"
#include "geometry/WallDistance.hpp"

#include <array>
#include <vector>

namespace hydrama {

class WallGeometry {
public:
    // Contiguous range of face indices (for range-based for loops).
    struct IndexRange {
        const int* first = nullptr;
        const int* last = nullptr;
        const int* begin() const noexcept { return first; }
        const int* end() const noexcept { return last; }
        int size() const noexcept { return static_cast<int>(last - first); }
    };

    WallGeometry() = default;
    explicit WallGeometry(const Mesh& mesh); // exact distance, AABB tree
    WallGeometry(const Mesh& mesh, const WallDistanceCalculator& distance);

    bool has_walls() const noexcept { return !faces_.empty(); }
    int npoin() const noexcept { return static_cast<int>(node_to_wall_.size()); }

    // --- Wall faces -------------------------------------------------------
    int num_faces() const noexcept { return static_cast<int>(faces_.size()); }
    const std::vector<WallFace>& faces() const noexcept { return faces_; }
    const WallFace& face(int f) const { return faces_[f]; }
    double total_length() const noexcept { return total_length_; }

    // --- Wall nodes -------------------------------------------------------
    int num_wall_nodes() const noexcept { return static_cast<int>(wall_nodes_.size()); }
    const std::vector<int>& wall_nodes() const noexcept { return wall_nodes_; }
    int wall_index(int ip) const noexcept { return node_to_wall_[ip]; } // -1: not a wall node
    bool is_wall_node(int ip) const noexcept { return node_to_wall_[ip] >= 0; }
    const std::array<double, 2>& wall_node_normal(int iw) const { return wall_node_normal_[iw]; }
    double wall_node_yp(int iw) const { return wall_node_yp_[iw]; }
    IndexRange wall_node_faces(int iw) const {
        return {node_faces_.data() + node_face_offset_[iw], node_faces_.data() + node_face_offset_[iw + 1]};
    }

    // --- Wall distance (per mesh node) ------------------------------------
    const std::vector<double>& wall_distance() const noexcept { return distance_.distance; }
    const std::vector<int>& nearest_wall_face() const noexcept { return distance_.nearest_face; }

private:
    std::vector<WallFace> faces_;
    double total_length_ = 0.0;

    std::vector<int> wall_nodes_;
    std::vector<int> node_to_wall_;
    std::vector<std::array<double, 2>> wall_node_normal_;
    std::vector<double> wall_node_yp_;
    std::vector<int> node_face_offset_; // size num_wall_nodes + 1
    std::vector<int> node_faces_;

    WallDistanceField distance_;

    void build_faces(const Mesh& mesh);
    void build_wall_nodes(int npoin);
};

} // namespace hydrama
