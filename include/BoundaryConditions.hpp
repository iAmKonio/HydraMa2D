#pragma once

#include "Mesh.hpp"
#include <vector>

namespace hydrama {

struct PressureBC {
    std::vector<int> nodes;
    std::vector<double> values;
};

PressureBC setup_pressure_bcs(const Mesh& mesh, int default_pnode);

void apply_symmetry_bc(const Mesh& mesh,
                       std::vector<double>& u,
                       std::vector<double>& v);

// Velocity Dirichlet conditions, in this order (the last one wins at
// shared nodes): 507 step profile, 503 uniform inlet, 508 parabolic inlet,
// walls (Mesh::wall_node_list). Throws std::invalid_argument if the 508
// sides do not form open chains (see parabolic_inlet_velocities).
void apply_velocity_bcs(const Mesh& mesh,
                        std::vector<double>& u,
                        std::vector<double>& v);

// Prescribed velocity of one node.
struct NodeVelocity {
    int node = 0;
    double u = 0.0;
    double v = 0.0;
};

// Fully developed inflow of the inlet_parabolic sides (flag 508).
//
// The 508 sides form connected chains (polylines). Along each chain, with
// arc length s in [0, L] from one end, the speed is the Poiseuille profile
//     |u|(s) = 6 s (L - s) / L^2      (mean 1 over the chain, 1.5 at s = L/2,
//                                      0 at both ends),
// directed along the inward normal of the boundary: the normal of a side
// that points into its element, averaged over the (one or two) chain sides
// of a node. It does not depend on the node order of the sides.
//
// Nodes in ascending order. Empty if the mesh has no 508 sides. Throws
// std::invalid_argument if a chain is closed (no ends) or branches (a node
// in more than two 508 sides), or if a side has zero length.
std::vector<NodeVelocity> parabolic_inlet_velocities(const Mesh& mesh);

} // namespace hydrama
