#pragma once

// Point probes: the finite element interpolation of nodal fields at a point,
// on meshes of triangles and quadrilaterals.
//
// locate_point finds an element that contains the point (within a relative
// tolerance, so points on element edges and on the boundary are found) and
// the shape function values there: barycentric coordinates in triangles, the
// bilinear shape functions at the preimage (xi, eta) in quadrilaterals
// (Newton's method on the isoparametric map; unique for convex elements).
// Interpolation is then exact for fields that are linear (triangles) or in
// the element space (quadrilaterals: bilinear in xi, eta).

#include "Mesh.hpp"

#include <array>
#include <optional>
#include <vector>

namespace hydrama::post {

struct PointLocation {
    int element = -1;
    int count = 0;                 // 3 (triangle) or 4 (quadrilateral)
    std::array<int, 4> nodes{};
    std::array<double, 4> weights{}; // shape functions at the point (sum 1)
};

// nullopt if the point lies outside the mesh. The tolerance is relative to
// the element size.
std::optional<PointLocation> locate_point(const Mesh& mesh, double x, double y, double tolerance = 1e-9);

double interpolate(const PointLocation& location, const std::vector<double>& field);

} // namespace hydrama::post
