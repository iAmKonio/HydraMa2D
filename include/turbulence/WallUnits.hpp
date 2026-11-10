#pragma once

// Wall shear stress, friction velocity and y+ diagnostics.
//
// For every wall face the velocity gradient G_ij = du_i/dx_j of the adjacent
// element (constant for P1) gives the molecular wall shear stress
//     tau_w = nu | t . (G n) |
// with the inward unit normal n and the unit tangent t of the face, and the
// friction velocity u_tau = sqrt(tau_w) (kinematic, non-dimensional). With a
// resolved (low-Re) wall treatment nu_t = 0 at the wall, so this is the full
// wall shear stress.
//
//   first-cell y+ of a face:  y+_1 = y_p u_tau / nu   (mesh resolution check)
//   nodal y+ field:           y+_i = d_i u_tau(nearest face) / nu

#include "Mesh.hpp"
#include "geometry/WallGeometry.hpp"

#include <vector>

namespace hydrama {

// u_tau per wall face (size walls.num_faces()).
std::vector<double> wall_friction_velocity(const Mesh& mesh, const WallGeometry& walls,
                                           const std::vector<double>& u, const std::vector<double>& v,
                                           double nu);

// Nodal y+ from the wall distance and the friction velocity of the nearest
// face; 0 where there is no wall (no faces).
void yplus_field(const WallGeometry& walls, const std::vector<double>& u_tau_face, double nu,
                 std::vector<double>& yplus);

struct WallUnitStats {
    int faces = 0;
    double min = 0.0;
    double mean = 0.0; // length-weighted
    double max = 0.0;
};

// Statistics of the first-cell y+ = y_p u_tau / nu over all wall faces.
WallUnitStats first_cell_yplus(const WallGeometry& walls, const std::vector<double>& u_tau_face, double nu);

} // namespace hydrama
