#include "BoundaryConditions.hpp"
#include "BoundaryKind.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

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
                        std::vector<double>& v,
                        const std::map<int, MovingWallSpec>& wall_specs,
                        const InflowConfig& inflow) {
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

    // 2. Prescribed inflow velocity (Flag 503 or configured in inflow)
    for (int ib = 0; ib < mesh.nboun; ++ib) {
        int flag = mesh.iside[ib].flag;
        if (flag == flags::kInlet || inflow.has_boundary(flag)) {
            const InflowSpec spec = inflow.get_spec(flag);
            int nodes[2] = {mesh.iside[ib].node1, mesh.iside[ib].node2};
            for (int i = 0; i < 2; ++i) {
                int ip = nodes[i];
                u[ip] = spec.u;
                v[ip] = spec.v;
            }
        }
    }

    // 3. Fully developed (parabolic) inflow (Flag 508)
    bool has_parabolic = false;
    for (const BoundarySide& s : mesh.iside) has_parabolic = has_parabolic || s.flag == flags::kInletParabolic;
    if (has_parabolic) {
        for (const NodeVelocity& nv : parabolic_inlet_velocities(mesh)) {
            u[nv.node] = nv.u;
            v[nv.node] = nv.v;
        }
    }

    // 4. Wall nodes (No-Slip) - applied to all wall nodes
    for (int ip : mesh.wall_node_list) {
        u[ip] = 0.0;
        v[ip] = 0.0;
    }

    // 5. Moving and rotating wall nodes (Flags 509, 510 or configured in wall_specs)
    for (const NodeVelocity& nv : moving_wall_velocities(mesh, wall_specs)) {
        u[nv.node] = nv.u;
        v[nv.node] = nv.v;
    }
}

std::vector<NodeVelocity> moving_wall_velocities(
    const Mesh& mesh,
    const std::map<int, MovingWallSpec>& wall_specs) {
    // 1. Group boundary sides by flag for any flag that is a moving/rotating wall or in wall_specs
    std::map<int, std::vector<int>> sides_by_flag;
    for (int ib = 0; ib < mesh.nboun; ++ib) {
        int flag = mesh.iside[ib].flag;
        if (flag == flags::kMovingWall || flag == flags::kRotatingWall || wall_specs.count(flag) > 0) {
            sides_by_flag[flag].push_back(ib);
        }
    }
    if (sides_by_flag.empty()) return {};

    // 2. Identify stationary wall nodes if any spec has corners_zero == true
    bool any_corners_zero = false;
    for (const auto& [flag, sides] : sides_by_flag) {
        auto it = wall_specs.find(flag);
        if (it != wall_specs.end() && it->second.corners_zero) {
            any_corners_zero = true;
            break;
        }
    }

    std::set<int> stationary_wall_nodes;
    if (any_corners_zero) {
        for (int ib = 0; ib < mesh.nboun; ++ib) {
            int flag = mesh.iside[ib].flag;
            if (is_wall_flag(flag) && sides_by_flag.find(flag) == sides_by_flag.end()) {
                stationary_wall_nodes.insert(mesh.iside[ib].node1);
                stationary_wall_nodes.insert(mesh.iside[ib].node2);
            }
        }
    }

    // 3. Compute velocities per node
    std::map<int, NodeVelocity> node_vel_map;

    for (const auto& [flag, sides] : sides_by_flag) {
        MovingWallSpec spec;
        auto it = wall_specs.find(flag);
        if (it != wall_specs.end()) {
            spec = it->second;
        } else if (flag == flags::kMovingWall) {
            spec.kind = MovingWallSpec::Kind::Translating;
            spec.u = 1.0;
            spec.v = 0.0;
        } else if (flag == flags::kRotatingWall) {
            spec.kind = MovingWallSpec::Kind::Rotating;
            spec.omega = 1.0;
            spec.auto_center = true;
        }

        double cx = spec.center_x;
        double cy = spec.center_y;

        if (spec.kind == MovingWallSpec::Kind::Rotating && spec.auto_center) {
            double total_len = 0.0;
            double sum_x = 0.0, sum_y = 0.0;
            for (int ib : sides) {
                int n1 = mesh.iside[ib].node1;
                int n2 = mesh.iside[ib].node2;
                double x1 = mesh.x[n1], y1 = mesh.y[n1];
                double x2 = mesh.x[n2], y2 = mesh.y[n2];
                double len = std::hypot(x2 - x1, y2 - y1);
                total_len += len;
                sum_x += 0.5 * (x1 + x2) * len;
                sum_y += 0.5 * (y1 + y2) * len;
            }
            if (total_len > 0.0) {
                cx = sum_x / total_len;
                cy = sum_y / total_len;
            } else {
                std::set<int> unique_nodes;
                for (int ib : sides) {
                    unique_nodes.insert(mesh.iside[ib].node1);
                    unique_nodes.insert(mesh.iside[ib].node2);
                }
                if (!unique_nodes.empty()) {
                    cx = 0.0;
                    cy = 0.0;
                    for (int ip : unique_nodes) {
                        cx += mesh.x[ip];
                        cy += mesh.y[ip];
                    }
                    cx /= static_cast<double>(unique_nodes.size());
                    cy /= static_cast<double>(unique_nodes.size());
                }
            }
        }

        for (int ib : sides) {
            int nodes[2] = {mesh.iside[ib].node1, mesh.iside[ib].node2};
            for (int in = 0; in < 2; ++in) {
                int ip = nodes[in];
                if (spec.corners_zero && stationary_wall_nodes.count(ip) > 0) {
                    node_vel_map[ip] = {ip, 0.0, 0.0};
                } else if (spec.kind == MovingWallSpec::Kind::Translating) {
                    node_vel_map[ip] = {ip, spec.u, spec.v};
                } else { // Rotating
                    double dx = mesh.x[ip] - cx;
                    double dy = mesh.y[ip] - cy;
                    // u = -omega * (y - cy), v = omega * (x - cx)
                    node_vel_map[ip] = {ip, -spec.omega * dy, spec.omega * dx};
                }
            }
        }
    }

    std::vector<NodeVelocity> out;
    out.reserve(node_vel_map.size());
    for (const auto& [node, nv] : node_vel_map) {
        out.push_back(nv);
    }
    return out;
}

std::vector<NodeVelocity> parabolic_inlet_velocities(const Mesh& mesh) {
    // Sides of the inlet and, per node, the (at most two) sides it belongs to.
    std::vector<int> sides;
    std::map<int, std::vector<int>> sides_of_node;
    for (int ib = 0; ib < mesh.nboun; ++ib) {
        const BoundarySide& s = mesh.iside[static_cast<std::size_t>(ib)];
        if (s.flag != flags::kInletParabolic) continue;
        sides.push_back(ib);
        sides_of_node[s.node1].push_back(ib);
        sides_of_node[s.node2].push_back(ib);
    }
    if (sides.empty()) return {};

    auto node_label = [&](int ip) {
        return "node " + std::to_string(ip) + " (" + std::to_string(mesh.x[static_cast<std::size_t>(ip)]) + ", " +
               std::to_string(mesh.y[static_cast<std::size_t>(ip)]) + ")";
    };
    for (const auto& [ip, list] : sides_of_node) {
        if (list.size() > 2)
            throw std::invalid_argument("inlet_parabolic (508) boundary branches at " + node_label(ip) + ": it is in " +
                                        std::to_string(list.size()) + " inlet sides; each inlet must be a simple curve");
    }

    // Unit normal of side ib pointing into its element (towards the centroid).
    auto inward_normal = [&](int ib, double& nx, double& ny) {
        const BoundarySide& s = mesh.iside[static_cast<std::size_t>(ib)];
        const double x1 = mesh.x[static_cast<std::size_t>(s.node1)], y1 = mesh.y[static_cast<std::size_t>(s.node1)];
        const double dx = mesh.x[static_cast<std::size_t>(s.node2)] - x1;
        const double dy = mesh.y[static_cast<std::size_t>(s.node2)] - y1;
        const double len = std::hypot(dx, dy);
        if (!(len > 0.0)) throw std::invalid_argument("inlet_parabolic (508) side " + std::to_string(ib) + " has zero length");
        nx = -dy / len;
        ny = dx / len;
        double cx = 0.0, cy = 0.0;
        const ElementNodes e = mesh.element_nodes(s.elem);
        for (int k = 0; k < e.size(); ++k) {
            cx += mesh.x[static_cast<std::size_t>(e[k])];
            cy += mesh.y[static_cast<std::size_t>(e[k])];
        }
        cx /= e.size();
        cy /= e.size();
        if (nx * (cx - x1) + ny * (cy - y1) < 0.0) {
            nx = -nx;
            ny = -ny;
        }
        return len;
    };

    std::map<int, std::array<double, 2>> normal_sum; // per node: sum of the inward normals of its sides
    std::vector<NodeVelocity> out;
    std::map<int, bool> side_done;
    // Walk every chain from one of its ends (a node in one inlet side).
    for (const auto& [start, start_sides] : sides_of_node) {
        if (start_sides.size() != 1 || side_done[start_sides[0]]) continue;
        std::vector<int> chain_nodes{start};
        std::vector<double> arc{0.0};
        int node = start;
        int side = start_sides[0];
        while (side >= 0) {
            side_done[side] = true;
            const BoundarySide& s = mesh.iside[static_cast<std::size_t>(side)];
            double nx = 0.0, ny = 0.0;
            const double len = inward_normal(side, nx, ny);
            const int next = s.node1 == node ? s.node2 : s.node1;
            for (int ip : {node, next}) {
                normal_sum[ip][0] += nx;
                normal_sum[ip][1] += ny;
            }
            chain_nodes.push_back(next);
            arc.push_back(arc.back() + len);
            node = next;
            side = -1;
            for (int other : sides_of_node[node])
                if (!side_done[other]) side = other;
        }
        const double total = arc.back();
        for (std::size_t k = 0; k < chain_nodes.size(); ++k) {
            const int ip = chain_nodes[k];
            const double s = arc[k];
            const double speed = 6.0 * s * (total - s) / (total * total);
            const std::array<double, 2> n = normal_sum[ip];
            const double nn = std::hypot(n[0], n[1]);
            if (!(nn > 0.0))
                throw std::invalid_argument("inlet_parabolic (508) boundary folds back on itself at " + node_label(ip));
            out.push_back({ip, speed * n[0] / nn, speed * n[1] / nn});
        }
    }
    for (int ib : sides) {
        if (!side_done[ib]) {
            const BoundarySide& s = mesh.iside[static_cast<std::size_t>(ib)];
            throw std::invalid_argument("inlet_parabolic (508) boundary through " + node_label(s.node1) +
                                        " is a closed curve; a parabolic inlet needs two ends");
        }
    }
    std::sort(out.begin(), out.end(), [](const NodeVelocity& a, const NodeVelocity& b) { return a.node < b.node; });
    return out;
}

} // namespace hydrama
