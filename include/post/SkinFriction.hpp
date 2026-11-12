#pragma once

// Wall skin friction extraction and automated recirculation / separation bubble detection.
//
// Evaluates wall shear stress tau_w and skin friction coefficient
//     Cf(s) = tau_w(s) / (0.5 * rho * U_ref^2)
// along boundary walls of triangles and quadrilaterals, where
//     tau_w = (sigma . n_b) . t
// is the tangential viscous traction along the wall tangent t pointing in the
// direction of increasing arc length (or +x).
//
// Automated root-finding on Cf(s) = 0 locates separation points (where Cf changes
// from positive to negative) and reattachment points (where Cf changes from
// negative to positive), returning the recirculation bubble length L_bubble = s_reatt - s_sep.

#include "Mesh.hpp"

#include <string>
#include <vector>

namespace hydrama::post {

struct SkinFrictionSample {
    int side_index = -1;      // index in mesh.iside
    int node1 = -1;
    int node2 = -1;
    double x = 0.0;           // midpoint coordinates
    double y = 0.0;
    double s = 0.0;           // arc length along the boundary patch
    double length = 0.0;      // side length
    double nx = 0.0;          // unit normal into fluid
    double ny = 0.0;
    double tx = 0.0;          // unit tangent along increasing s
    double ty = 0.0;
    double tau_w = 0.0;       // wall shear stress (signed along tangent)
    double cf = 0.0;          // skin friction coefficient
    double p = 0.0;           // surface pressure
    double cp = 0.0;          // pressure coefficient (p - p_ref) / (0.5 * rho * U_ref^2)
};

struct SkinFrictionProfile {
    std::vector<SkinFrictionSample> samples;
    double u_ref = 1.0;
    double p_ref = 0.0;
    double rho = 1.0;
    double total_length = 0.0;
};

struct RecirculationBubble {
    double x_sep = 0.0;       // separation point coordinates
    double y_sep = 0.0;
    double s_sep = 0.0;       // separation arc length
    double x_reatt = 0.0;     // reattachment point coordinates
    double y_reatt = 0.0;
    double s_reatt = 0.0;     // reattachment arc length
    double length = 0.0;      // s_reatt - s_sep
    double cf_min = 0.0;      // peak reverse skin friction
    double x_cf_min = 0.0;    // coordinate of peak reverse skin friction
    double reverse_flux = 0.0;// integral of -Cf ds over the recirculation zone
};

// Extracts skin friction along the designated boundary sides of the mesh.
// Sides are sorted into a continuous geometric polyline ordered along increasing arc length s.
SkinFrictionProfile extract_skin_friction(
    const Mesh& mesh,
    const std::vector<int>& boundary_sides,
    double nu,
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& p,
    double u_ref = 1.0,
    double p_ref = 0.0,
    double rho = 1.0);

// Resolves a boundary name, physical group tag, or flag code to a list of boundary sides.
std::vector<int> resolve_wall_sides(const Mesh& mesh, const std::string& patch_identifier);

// Identifies separation points, reattachment points, and recirculation bubbles
// by computing zero-crossings of Cf(s) along the skin friction profile.
std::vector<RecirculationBubble> find_recirculation_bubbles(const SkinFrictionProfile& profile);

// Formats a clean ASCII summary table of the detected recirculation bubbles.
std::string format_bubble_report(
    const std::vector<RecirculationBubble>& bubbles,
    const std::string& patch_name = "wall");

} // namespace hydrama::post
