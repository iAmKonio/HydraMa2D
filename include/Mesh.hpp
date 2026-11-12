#pragma once

// Unstructured 2D mesh of linear triangles (P1) and bilinear quadrilaterals
// (Q1), see docs/QUADRILATERAL_ELEMENTS.md.
//
// Elements are numbered triangles first, then quadrilaterals:
//     element e in [0, num_triangles())           triangle  intma[e]
//     element e in [num_triangles(), nelem)       quadrilateral quads[e - num_triangles()]
// nelem = num_triangles() + num_quads(). Per-element arrays (detJ, dNkdx,
// alen_e, time steps, eddy viscosity, ...) have nelem entries. A mesh of
// triangles only is exactly the original triangle mesh: its triangle
// computations are unchanged.

#include "numerics/QuadElement.hpp"

#include <array>
#include <string>
#include <vector>

namespace hydrama {

struct BoundarySide {
    int node1 = 0;   // 0-based
    int node2 = 0;   // 0-based
    int elem = 0;    // 0-based
    int side_id = 0; // original boundary side identifier
    int flag = 0;    // boundary flag code (e.g. 500, 503, 504, 506, 507, 508)
};

// A boundary group of an imported mesh (Gmsh physical curve group, SU2
// marker, CGNS BC): its tag, its name ("" if unnamed) and the boundary
// sides in it (indices into Mesh::iside, ascending). A side can be in several
// groups; its side_id is the smallest of their tags.
struct BoundaryGroup {
    int tag = 0;
    std::string name;
    std::vector<int> sides;
};

struct FaceNormal {
    double nx = 0.0;
    double ny = 0.0;
    double length = 0.0;
};

// Shape function derivatives of an element: the constant gradients of a
// triangle (entries 0..2), or the area-averaged gradients
// (1/A) int grad N_a dOmega of a quadrilateral (entries 0..3).
struct ElementDeriv {
    double dNdx[4] = {0.0, 0.0, 0.0, 0.0};
    double dNdy[4] = {0.0, 0.0, 0.0, 0.0};
};

// The nodes of one element (3 or 4), usable in range-for loops.
struct ElementNodes {
    std::array<int, 4> node{};
    int count = 0;
    const int* begin() const { return node.data(); }
    const int* end() const { return node.data() + count; }
    int size() const { return count; }
    int operator[](int k) const { return node[static_cast<std::size_t>(k)]; }
};

class Mesh {
public:
    int nelem = 0; // triangles + quadrilaterals
    int npoin = 0;
    int nboun = 0;
    static constexpr int nep = 3;   // Nodes per triangle
    static constexpr int nep_quad = 4; // Nodes per quadrilateral
    static constexpr int nsid = 3;  // Sides per triangle
    static constexpr int nsidp = 2; // Nodes per side
    static constexpr int gsdim = 3; // Off-diagonal components per triangle (per block)

    // Element connectivity, 0-based node indices, counter-clockwise.
    std::vector<std::array<int, nep>> intma;      // triangles: elements 0 .. num_triangles() - 1
    std::vector<std::array<int, nep_quad>> quads; // quadrilaterals: elements num_triangles() .. nelem - 1

    // Nodal coordinates
    std::vector<double> x;
    std::vector<double> y;

    // Boundary edge information
    std::vector<BoundarySide> iside;

    // Boundary groups of an imported mesh, ascending tags (empty for legacy
    // meshes, whose sides are identified by side_id only). Informational: the
    // solver uses iside[].flag.
    std::vector<BoundaryGroup> boundary_groups;

    // Boundary normals and lengths
    std::vector<FaceNormal> face_norm;

    // Wall information
    std::vector<int> wall_node_list;
    std::vector<std::array<double, 2>> wall_node_norm;
    int trailing_edge_node = -1;

    // Geometric element properties (nelem entries unless noted)
    std::vector<double> detJ;               // 2.0 * element area (triangles: the Jacobian of the P1 map)
    std::vector<ElementDeriv> dNkdx;        // Shape function derivatives (see ElementDeriv)
    std::vector<double> elcoe_e;            // Lumped triangle mass: detJ / 6.0 (nep * num_triangles())
    std::vector<double> alen_e;             // Element length scale (minimum height) for the time step
    std::vector<std::array<double, 3>> pdiagE; // Triangle Laplacian diagonal components (num_triangles())
    std::vector<std::array<double, 3>> gstifE; // Triangle Laplacian off-diagonal components (num_triangles())
    std::vector<QuadGeometry> quad_geom;    // Quadrilateral quadrature data, incl. lumped mass (num_quads())
    std::vector<QuadLaplacian> quad_lap;    // Quadrilateral Laplacian (num_quads())

    // Volume statistics
    double max_vol = 0.0;
    double min_vol = 0.0;

    int num_triangles() const { return static_cast<int>(intma.size()); }
    int num_quads() const { return static_cast<int>(quads.size()); }
    bool has_quads() const { return !quads.empty(); }
    bool is_quad(int e) const { return e >= num_triangles(); }
    int quad_of(int e) const { return e - num_triangles(); }           // element -> quadrilateral index
    int element_of_quad(int q) const { return num_triangles() + q; }  // quadrilateral index -> element
    int nodes_per_element(int e) const { return is_quad(e) ? nep_quad : nep; }
    ElementNodes element_nodes(int e) const;

    // Element-by-element storage of the off-diagonal couplings of symmetric
    // element matrices (update_lhs, EbeOperator): blocks of 3 values, one per
    // triangle (pairs (0,1), (0,2), (1,2)) followed by two per quadrilateral
    // (pairs (0,1), (0,2), (0,3), then (1,2), (1,3), (2,3)).
    int num_offdiag_blocks() const { return num_triangles() + 2 * num_quads(); }

    // Height of the element above its edge (n1, n2): triangles 2 A / L,
    // quadrilaterals the mean distance of the two other nodes from the edge.
    double height_above_edge(int e, int n1, int n2) const;

    void compute_shape_function_derivatives();
    void compute_mass_matrix();
    void compute_face_normals();
    void determine_wall_nodes();
    void compute_element_sizes();
    void compute_laplacian_stiffness();
};

} // namespace hydrama
