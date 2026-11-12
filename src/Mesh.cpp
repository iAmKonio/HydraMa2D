#include "Mesh.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace hydrama {

ElementNodes Mesh::element_nodes(int e) const {
    ElementNodes n;
    if (is_quad(e)) {
        const auto& q = quads[static_cast<std::size_t>(quad_of(e))];
        n.node = {q[0], q[1], q[2], q[3]};
        n.count = 4;
    } else {
        const auto& t = intma[static_cast<std::size_t>(e)];
        n.node = {t[0], t[1], t[2], -1};
        n.count = 3;
    }
    return n;
}

double Mesh::height_above_edge(int e, int n1, int n2) const {
    const double dx = x[n2] - x[n1], dy = y[n2] - y[n1];
    const double L = std::sqrt(dx * dx + dy * dy);
    if (!is_quad(e)) return std::abs(detJ[e]) / L; // 2 A / L
    // Quadrilateral: mean distance of the two nodes that are not on the edge.
    double sum = 0.0;
    for (int a : element_nodes(e)) {
        if (a == n1 || a == n2) continue;
        sum += std::abs(dy * (x[a] - x[n1]) - dx * (y[a] - y[n1])) / L;
    }
    return 0.5 * sum;
}

void Mesh::compute_shape_function_derivatives() {
    dNkdx.resize(nelem);
    detJ.resize(nelem);

    max_vol = 0.0;
    min_vol = std::numeric_limits<double>::max();

    const int ntri = num_triangles();
    for (int ie = 0; ie < ntri; ++ie) {
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

    // Quadrilaterals: 2 x 2 Gauss-Legendre data (numerics/QuadElement.hpp).
    quad_geom.resize(quads.size());
    for (int q = 0; q < num_quads(); ++q) {
        const auto& n = quads[static_cast<std::size_t>(q)];
        const double qx[4] = {x[n[0]], x[n[1]], x[n[2]], x[n[3]]};
        const double qy[4] = {y[n[0]], y[n[1]], y[n[2]], y[n[3]]};
        const int ie = element_of_quad(q);
        try {
            quad_geom[static_cast<std::size_t>(q)] = q1::compute_geometry(qx, qy);
        } catch (const std::domain_error&) {
            throw std::domain_error("quadrilateral element " + std::to_string(ie) +
                                    " has a non-positive Jacobian determinant (clockwise, non-convex or degenerate)");
        }
        const QuadGeometry& g = quad_geom[static_cast<std::size_t>(q)];
        for (int a = 0; a < 4; ++a) {
            dNkdx[ie].dNdx[a] = g.mean_dNdx[a];
            dNkdx[ie].dNdy[a] = g.mean_dNdy[a];
        }
        detJ[ie] = 2.0 * g.area;
        max_vol = std::max(max_vol, g.area);
        min_vol = std::min(min_vol, g.area);
    }

    std::cout << "Maximum Volume " << max_vol << "\n";
    std::cout << "Minimum Volume " << min_vol << "\n";
    if (min_vol > 0.0) {
        std::cout << "Ratio of Max to Min " << (max_vol / min_vol) << "\n";
    }
}

void Mesh::compute_mass_matrix() {
    // Lumped mass matrix component per node in each triangle: detJ / 6.0 = area / 3.0.
    // Quadrilaterals: row-sum lumped mass int N_a dOmega in quad_geom (computed
    // with the shape function derivatives).
    elcoe_e.resize(nep * intma.size());
    for (int ie = 0; ie < num_triangles(); ++ie) {
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

    for (int ie = 0; ie < num_triangles(); ++ie) {
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

    // Quadrilaterals: minimum of area / edge length over the four edges (for a
    // rectangle hx x hy: min(hx, hy); for a parallelogram: its smaller height).
    for (int q = 0; q < num_quads(); ++q) {
        const auto& n = quads[static_cast<std::size_t>(q)];
        double longest = 0.0;
        for (int k = 0; k < 4; ++k) {
            const int a = n[static_cast<std::size_t>(k)], b = n[static_cast<std::size_t>((k + 1) % 4)];
            longest = std::max(longest, std::sqrt((x[b] - x[a]) * (x[b] - x[a]) + (y[b] - y[a]) * (y[b] - y[a])));
        }
        alen_e[element_of_quad(q)] = quad_geom[static_cast<std::size_t>(q)].area / longest;
    }
}

void Mesh::compute_laplacian_stiffness() {
    pdiagE.resize(intma.size());
    gstifE.resize(intma.size());

    for (int ie = 0; ie < num_triangles(); ++ie) {
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
    // Quadrilaterals: 2 x 2 Gauss-Legendre (numerics/QuadElement.hpp).
    quad_lap.resize(quads.size());
    for (int q = 0; q < num_quads(); ++q) quad_lap[static_cast<std::size_t>(q)] = q1::laplacian(quad_geom[static_cast<std::size_t>(q)]);
}

} // namespace hydrama
