#include "geometry/WallGeometry.hpp"

#include "BoundaryKind.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace hydrama {

namespace {

[[noreturn]] void fail(const std::string& what) {
    throw std::invalid_argument("WallGeometry: " + what);
}

std::string side_label(const Mesh& mesh, int ib) {
    return "boundary side " + std::to_string(ib) + " (side id " + std::to_string(mesh.iside[ib].side_id) +
           ", flag " + std::to_string(mesh.iside[ib].flag) + ")";
}

} // namespace

WallGeometry::WallGeometry(const Mesh& mesh) : WallGeometry(mesh, AabbTreeWallDistance()) {}

WallGeometry::WallGeometry(const Mesh& mesh, const WallDistanceCalculator& distance) {
    if (mesh.npoin < 0 || static_cast<int>(mesh.x.size()) != mesh.npoin ||
        static_cast<int>(mesh.y.size()) != mesh.npoin ||
        static_cast<int>(mesh.intma.size()) != mesh.nelem ||
        static_cast<int>(mesh.iside.size()) != mesh.nboun) {
        fail("mesh arrays are inconsistent with npoin / nelem / nboun");
    }
    build_faces(mesh);
    build_wall_nodes(mesh.npoin);
    distance_ = distance.compute(mesh, faces_);
}

void WallGeometry::build_faces(const Mesh& mesh) {
    faces_.clear();
    total_length_ = 0.0;
    auto node_ok = [&](int ip) { return ip >= 0 && ip < mesh.npoin; };

    for (int ib = 0; ib < mesh.nboun; ++ib) {
        const BoundarySide& side = mesh.iside[ib];
        if (!is_wall_flag(side.flag)) continue;

        if (!node_ok(side.node1) || !node_ok(side.node2)) fail(side_label(mesh, ib) + " references a node out of range");
        if (side.elem < 0 || side.elem >= mesh.nelem) fail(side_label(mesh, ib) + " references an element out of range");

        // The face must be an edge of its element; the remaining vertex is the
        // opposite node.
        const auto& e = mesh.intma[side.elem];
        int on_face = 0, opposite = -1;
        for (int k = 0; k < Mesh::nep; ++k) {
            if (e[k] == side.node1 || e[k] == side.node2) {
                ++on_face;
            } else {
                opposite = e[k];
            }
        }
        if (side.node1 == side.node2 || on_face != 2 || opposite < 0 || !node_ok(opposite)) {
            fail(side_label(mesh, ib) + " is not an edge of element " + std::to_string(side.elem));
        }

        const double x1 = mesh.x[side.node1], y1 = mesh.y[side.node1];
        const double dx = mesh.x[side.node2] - x1;
        const double dy = mesh.y[side.node2] - y1;
        const double length = std::hypot(dx, dy);
        if (!(length > 0.0)) fail(side_label(mesh, ib) + " has zero length");

        // Unit normal to the face; its signed projection onto (opposite - node1)
        // is the element height above the face. Orient it into the element.
        double nx = dy / length, ny = -dx / length;
        const double h = nx * (mesh.x[opposite] - x1) + ny * (mesh.y[opposite] - y1);
        if (!(std::abs(h) > 0.0)) fail("element " + std::to_string(side.elem) + " adjacent to " + side_label(mesh, ib) + " is degenerate");
        if (h < 0.0) {
            nx = -nx;
            ny = -ny;
        }

        WallFace f;
        f.side = ib;
        f.node1 = side.node1;
        f.node2 = side.node2;
        f.elem = side.elem;
        f.opposite_node = opposite;
        f.nx = nx;
        f.ny = ny;
        f.length = length;
        f.yp = std::abs(h);
        faces_.push_back(f);
        total_length_ += length;
    }
}

void WallGeometry::build_wall_nodes(int npoin) {
    // Wall nodes in ascending node order and the number of faces per node.
    std::vector<int> face_count(npoin, 0);
    for (const WallFace& f : faces_) {
        ++face_count[f.node1];
        ++face_count[f.node2];
    }

    node_to_wall_.assign(npoin, -1);
    wall_nodes_.clear();
    for (int ip = 0; ip < npoin; ++ip) {
        if (face_count[ip] > 0) {
            node_to_wall_[ip] = static_cast<int>(wall_nodes_.size());
            wall_nodes_.push_back(ip);
        }
    }
    const int nw = num_wall_nodes();

    // CSR node -> faces (faces listed in ascending face index).
    node_face_offset_.assign(nw + 1, 0);
    for (int iw = 0; iw < nw; ++iw) node_face_offset_[iw + 1] = node_face_offset_[iw] + face_count[wall_nodes_[iw]];
    node_faces_.assign(node_face_offset_[nw], -1);
    std::vector<int> fill(node_face_offset_.begin(), node_face_offset_.end() - 1);
    for (int f = 0; f < num_faces(); ++f) {
        node_faces_[fill[node_to_wall_[faces_[f].node1]]++] = f;
        node_faces_[fill[node_to_wall_[faces_[f].node2]]++] = f;
    }

    // Length-weighted normals and first-cell heights.
    wall_node_normal_.assign(nw, {0.0, 0.0});
    wall_node_yp_.assign(nw, 0.0);
    for (int iw = 0; iw < nw; ++iw) {
        double sx = 0.0, sy = 0.0, sl = 0.0, syp = 0.0;
        for (int f : wall_node_faces(iw)) {
            const WallFace& face = faces_[f];
            sx += face.length * face.nx;
            sy += face.length * face.ny;
            sl += face.length;
            syp += face.length * face.yp;
        }
        const double norm = std::hypot(sx, sy);
        // Cusp (e.g. both sides of a zero-thickness plate): normals cancel.
        if (norm > 1.0e-12 * sl) wall_node_normal_[iw] = {sx / norm, sy / norm};
        wall_node_yp_[iw] = syp / sl;
    }
}

} // namespace hydrama
