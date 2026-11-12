#pragma once

// Independent check of the invariants the solver relies on. Every mesh
// source (legacy *.mesh reader, build_mesh for Gmsh files, future SU2 /
// CGNS readers) must produce meshes that pass it:
//
//   1. array sizes match npoin / nelem / nboun, all indices in range;
//   2. every triangle has three distinct nodes and a strictly positive
//      signed area (counter-clockwise); every quadrilateral has four
//      distinct nodes and strictly positive Jacobian determinants at its four
//      corners (counter-clockwise and convex, hence det J > 0 everywhere);
//   3. the elements are conforming: an edge belongs to one (boundary) or two
//      (interior) elements, traversed in opposite directions by the two;
//   4. every boundary side is an edge of its element `elem`, traversed in the
//      element's counter-clockwise direction (node1 -> node2), so the face
//      normal (dy, -dx) / L of Mesh::compute_face_normals points outwards;
//   5. every boundary edge is covered by exactly one boundary side, and no
//      boundary side lies on an interior edge.
//
// It reads only npoin/nelem/nboun, x, y, intma, quads and iside (no derived data).

#include "Mesh.hpp"

#include <string>
#include <vector>

namespace hydrama::mesh {

// Returns human-readable problems (at most max_problems; empty if valid).
std::vector<std::string> check_mesh(const Mesh& mesh, std::size_t max_problems = 20);

} // namespace hydrama::mesh
