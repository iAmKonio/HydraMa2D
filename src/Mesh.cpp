#include "Mesh.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>

namespace hydrama {

void Mesh::compute_shape_function_derivatives() {
    dNkdx.resize(nelem);
    detJ.resize(nelem);

    max_vol = 0.0;
    min_vol = std::numeric_limits<double>::max();

    for (int ie = 0; ie < nelem; ++ie) {
        int n0 = intma[ie][0];
        int n1 = intma[ie][1];
        int n2 = intma[ie][2];

        double x0 = x[n0], y0 = y[n0];
        double x1 = x[n1], y1 = y[n1];
        double x2 = x[n2], y2 = y[n2];

        double b0 = y1 - y2;
        double b1 = y2 - y0;
        double b2 = y0 - y1;

        double c0 = x2 - x1;
        double c1 = x0 - x2;
        double c2 = x1 - x0;

        double area = 0.5 * (c2 * b1 - b2 * c1);
        double twoa = 1.0 / (2.0 * area);

        dNkdx[ie].dNdx[0] = twoa * b0;
        dNkdx[ie].dNdx[1] = twoa * b1;
        dNkdx[ie].dNdx[2] = twoa * b2;

        dNkdx[ie].dNdy[0] = twoa * c0;
        dNkdx[ie].dNdy[1] = twoa * c1;
        dNkdx[ie].dNdy[2] = twoa * c2;

        detJ[ie] = 2.0 * area;

        max_vol = std::max(max_vol, area);
        min_vol = std::min(min_vol, area);
    }

    std::cout << "Maximum Volume " << max_vol << "\n";
    std::cout << "Minimum Volume " << min_vol << "\n";
    if (min_vol > 0.0) {
        std::cout << "Ratio of Max to Min " << (max_vol / min_vol) << "\n";
    }
}

void Mesh::compute_mass_matrix() {
    // Lumped mass matrix component per node in each element: detJ / 6.0 = area / 3.0
    elcoe_e.resize(nep * nelem);
    for (int ie = 0; ie < nelem; ++ie) {
        double ldetJ = detJ[ie] / 6.0;
        for (int in = 0; in < nep; ++in) {
            elcoe_e[ie * nep + in] = ldetJ;
        }
    }
}

void Mesh::compute_face_normals() {
    face_norm.resize(nboun);
    for (int ib = 0; ib < nboun; ++ib) {
        int n1 = iside[ib].node1;
        int n2 = iside[ib].node2;

        double dx = x[n2] - x[n1];
        double dy = y[n2] - y[n1];
        double length = std::sqrt(dx * dx + dy * dy);

        face_norm[ib].nx = dy / length;
        face_norm[ib].ny = -dx / length;
        face_norm[ib].length = length;
    }
}

void Mesh::determine_wall_nodes() {
    std::vector<int> itemp(npoin, 0);
    std::vector<std::array<double, 2>> rtemp;

    wall_node_list.clear();
    wall_node_norm.clear();

    for (int ib = 0; ib < nboun; ++ib) {
        int flag = iside[ib].flag;
        if (flag == 500 || flag == 501 || flag == 502) {
            int nodes[2] = {iside[ib].node1, iside[ib].node2};
            for (int in = 0; in < 2; ++in) {
                int ip = nodes[in];
                int ip2 = 0;
                if (itemp[ip] == 0) {
                    wall_node_list.push_back(ip);
                    ip2 = static_cast<int>(wall_node_list.size()) - 1; // 0-based
                    itemp[ip] = ip2 + 1; // 1-based marker
                    wall_node_norm.push_back({0.0, 0.0});
                    rtemp.push_back({face_norm[ib].nx, face_norm[ib].ny});
                } else {
                    ip2 = itemp[ip] - 1;
                    rtemp[ip2][0] *= face_norm[ib].nx;
                    rtemp[ip2][1] *= face_norm[ib].ny;
                }
                wall_node_norm[ip2][0] += face_norm[ib].nx * face_norm[ib].length;
                wall_node_norm[ip2][1] += face_norm[ib].ny * face_norm[ib].length;
            }
        }
    }

    int npoin_wall = static_cast<int>(wall_node_list.size());
    std::cout << "Wall Nodes Detected: " << npoin_wall << "\n";

    trailing_edge_node = -1;
    for (int ip2 = 0; ip2 < npoin_wall; ++ip2) {
        if (rtemp[ip2][0] + rtemp[ip2][1] < -0.2) {
            // Trailing edge detected
            trailing_edge_node = wall_node_list[ip2];
            wall_node_norm[ip2][0] = 0.0;
            wall_node_norm[ip2][1] = 0.0;
            continue;
        }
        double len = std::sqrt(wall_node_norm[ip2][0] * wall_node_norm[ip2][0] +
                               wall_node_norm[ip2][1] * wall_node_norm[ip2][1]);
        if (len > 0.0) {
            wall_node_norm[ip2][0] /= len;
            wall_node_norm[ip2][1] /= len;
        }
    }
}

void Mesh::compute_element_sizes() {
    alen_e.assign(nelem, std::numeric_limits<double>::max());

    for (int ie = 0; ie < nelem; ++ie) {
        int n0 = intma[ie][0];
        int n1 = intma[ie][1];
        int n2 = intma[ie][2];

        double x0 = x[n0], y0 = y[n0];
        double x1 = x[n1], y1 = y[n1];
        double x2 = x[n2], y2 = y[n2];

        // Three side lengths
        double s0 = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)); // between 1 and 2
        double s1 = std::sqrt((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)); // between 2 and 0
        double s2 = std::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)); // between 0 and 1

        double volum = detJ[ie]; // 2 * Area
        double h0 = volum / s0;
        double h1 = volum / s1;
        double h2 = volum / s2;

        alen_e[ie] = std::min({h0, h1, h2});
    }
}

void Mesh::compute_laplacian_stiffness() {
    pdiagE.resize(nelem);
    gstifE.resize(nelem);

    for (int ie = 0; ie < nelem; ++ie) {
        double ldetJ = detJ[ie] * 0.5; // Area

        // Diagonal components for the 3 nodes
        for (int in = 0; in < nep; ++in) {
            double dNx = dNkdx[ie].dNdx[in];
            double dNy = dNkdx[ie].dNdy[in];
            pdiagE[ie][in] = (dNx * dNx + dNy * dNy) * ldetJ;
        }

        // Off-diagonal components:
        // Pair 0: (0, 1)
        gstifE[ie][0] = (dNkdx[ie].dNdx[0] * dNkdx[ie].dNdx[1] +
                         dNkdx[ie].dNdy[0] * dNkdx[ie].dNdy[1]) * ldetJ;
        // Pair 1: (0, 2)
        gstifE[ie][1] = (dNkdx[ie].dNdx[0] * dNkdx[ie].dNdx[2] +
                         dNkdx[ie].dNdy[0] * dNkdx[ie].dNdy[2]) * ldetJ;
        // Pair 2: (1, 2)
        gstifE[ie][2] = (dNkdx[ie].dNdx[1] * dNkdx[ie].dNdx[2] +
                         dNkdx[ie].dNdy[1] * dNkdx[ie].dNdy[2]) * ldetJ;
    }
}

} // namespace hydrama
