#include "BoundaryConditions.hpp"
#include <cmath>
#include <iostream>

namespace hydrama {

PressureBC setup_pressure_bcs(const Mesh& mesh, int default_pnode) {
    PressureBC bc;
    std::vector<int> itemp(mesh.npoin, 0);

    for (int ib = 0; ib < mesh.nboun; ++ib) {
        if (mesh.iside[ib].flag == 504) {
            int nodes[2] = {mesh.iside[ib].node1, mesh.iside[ib].node2};
            for (int i = 0; i < 2; ++i) {
                int ip = nodes[i];
                if (itemp[ip] == 0) {
                    itemp[ip] = 1;
                    bc.nodes.push_back(ip);
                    bc.values.push_back(0.0);
                }
            }
        }
    }

    // If no pressure boundaries exist, set reference pressure node
    if (bc.nodes.empty()) {
        int ref_node = (default_pnode >= 0 && default_pnode < mesh.npoin) ? default_pnode : 0;
        bc.nodes.push_back(ref_node);
        bc.values.push_back(0.0);
    }

    std::cout << "Number of Pressure BC nodes: " << bc.nodes.size() << "\n";
    return bc;
}

void apply_symmetry_bc(const Mesh& mesh,
                       std::vector<double>& u,
                       std::vector<double>& v) {
    for (int ib = 0; ib < mesh.nboun; ++ib) {
        if (mesh.iside[ib].flag != 506) continue;

        int ip1 = mesh.iside[ib].node1;
        int ip2 = mesh.iside[ib].node2;
        double nx = mesh.face_norm[ib].nx;
        double ny = mesh.face_norm[ib].ny;

        double vnorm1 = u[ip1] * nx + v[ip1] * ny;
        u[ip1] -= vnorm1 * nx;
        v[ip1] -= vnorm1 * ny;

        double vnorm2 = u[ip2] * nx + v[ip2] * ny;
        u[ip2] -= vnorm2 * nx;
        v[ip2] -= vnorm2 * ny;
    }
}

void apply_velocity_bcs(const Mesh& mesh,
                        std::vector<double>& u,
                        std::vector<double>& v) {
    // 1. Backward Facing Step (Flag 507) - Parabolic / polynomial inflow profile
    for (int ib = 0; ib < mesh.nboun; ++ib) {
        if (mesh.iside[ib].flag == 507) {
            int nodes[2] = {mesh.iside[ib].node1, mesh.iside[ib].node2};
            for (int i = 0; i < 2; ++i) {
                int ip = nodes[i];
                double y_coord = mesh.y[ip];
                double y2 = y_coord * y_coord;
                double y3 = y2 * y_coord;
                double y4 = y3 * y_coord;
                double y5 = y4 * y_coord;
                double y6 = y5 * y_coord;

                u[ip] = 0.6624 * y6 - 7.5547 * y5 + 33.9 * y4
                        - 75.283 * y3 + 83.368 * y2 - 37.793 * y_coord + 2.6959;
                v[ip] = 0.0;
            }
        }
    }

    // 2. Prescribed inflow velocity (Flag 503)
    for (int ib = 0; ib < mesh.nboun; ++ib) {
        if (mesh.iside[ib].flag == 503) {
            int nodes[2] = {mesh.iside[ib].node1, mesh.iside[ib].node2};
            for (int i = 0; i < 2; ++i) {
                int ip = nodes[i];
                u[ip] = 1.0;
                v[ip] = 0.0;
            }
        }
    }

    // 3. Wall nodes (No-Slip) - applied last to ensure corners don't leak
    for (int ip : mesh.wall_node_list) {
        u[ip] = 0.0;
        v[ip] = 0.0;
    }
}

} // namespace hydrama
