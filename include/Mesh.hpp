#pragma once

#include <vector>
#include <array>
#include <string>

namespace hydrama {

struct BoundarySide {
    int node1 = 0;   // 0-based
    int node2 = 0;   // 0-based
    int elem = 0;    // 0-based
    int side_id = 0; // original boundary side identifier
    int flag = 0;    // boundary flag code (e.g. 500, 503, 504, 506, 507)
};

struct FaceNormal {
    double nx = 0.0;
    double ny = 0.0;
    double length = 0.0;
};

struct ElementDeriv {
    double dNdx[3] = {0.0, 0.0, 0.0};
    double dNdy[3] = {0.0, 0.0, 0.0};
};

class Mesh {
public:
    int nelem = 0;
    int npoin = 0;
    int nboun = 0;
    static constexpr int nep = 3;   // Nodes per element (triangle)
    static constexpr int nsid = 3;  // Sides per element
    static constexpr int nsidp = 2; // Nodes per side
    static constexpr int gsdim = 3; // Off-diagonal components per element

    // Element connectivity: 0-based node indices
    std::vector<std::array<int, nep>> intma;

    // Nodal coordinates
    std::vector<double> x;
    std::vector<double> y;

    // Boundary edge information
    std::vector<BoundarySide> iside;

    // Boundary normals and lengths
    std::vector<FaceNormal> face_norm;

    // Wall information
    std::vector<int> wall_node_list;
    std::vector<std::array<double, 2>> wall_node_norm;
    int trailing_edge_node = -1;

    // Geometric element properties
    std::vector<double> detJ;               // 2.0 * element area
    std::vector<ElementDeriv> dNkdx;        // Shape function derivatives
    std::vector<double> elcoe_e;            // Lumped element mass: detJ / 6.0 (nep * nelem)
    std::vector<double> alen_e;             // Minimum element height
    std::vector<std::array<double, 3>> pdiagE; // Element Laplacian diagonal components
    std::vector<std::array<double, 3>> gstifE; // Element Laplacian off-diagonal components

    // Volume statistics
    double max_vol = 0.0;
    double min_vol = 0.0;

    void compute_shape_function_derivatives();
    void compute_mass_matrix();
    void compute_face_normals();
    void determine_wall_nodes();
    void compute_element_sizes();
    void compute_laplacian_stiffness();
};

} // namespace hydrama
