#pragma once

#include "Mesh.hpp"
#include <map>
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
// stationary walls, moving/rotating walls. Throws std::invalid_argument if the 508
// sides do not form open chains (see parabolic_inlet_velocities).
struct MovingWallSpec {
    enum class Kind : unsigned char {
        Translating, // constant velocity (u, v), e.g. a cavity lid
        Rotating,    // rotating about an axis (center_x, center_y) with angular velocity omega
    };
    Kind kind = Kind::Translating;
    double u = 1.0;
    double v = 0.0;
    double omega = 1.0;
    double center_x = 0.0;
    double center_y = 0.0;
    bool auto_center = true; // if true, center is computed from centroid of patch nodes
    bool corners_zero = false; // if true, nodes shared with stationary walls stay zero
};

void apply_velocity_bcs(const Mesh& mesh,
                        std::vector<double>& u,
                        std::vector<double>& v,
                        const std::map<int, MovingWallSpec>& wall_specs = {});

// Prescribed velocity of one node.
struct NodeVelocity {
    int node = 0;
    double u = 0.0;
    double v = 0.0;
};

// Prescribed velocities of all moving (509) and rotating (510) wall nodes.
// If specs does not define a boundary, default values (u=1, v=0 for 509;
// omega=1 about centroid for 510) are used.
std::vector<NodeVelocity> moving_wall_velocities(
    const Mesh& mesh,
    const std::map<int, MovingWallSpec>& wall_specs = {});

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
