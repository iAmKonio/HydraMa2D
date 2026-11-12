#pragma once

// Forces exerted by the fluid on a body (a set of boundary sides), per unit
// span and for density 1 (the solver's non-dimensional units), on meshes of
// triangles and quadrilaterals. See docs/CYLINDER_BENCHMARK.md, section 2.
//
// Two evaluations of the same quantity
//
//     F = integral over the body of  sigma n_b ds,
//     sigma = -p I + nu (grad u + grad u^T),
//
// with n_b the unit normal pointing out of the body into the fluid:
//
//  * surface_force: the traction integrated along the body sides. The
//    pressure is linear along a side (exact), the velocity gradient is that of
//    the adjacent element (triangles: constant; quadrilaterals: at the side
//    midpoint, exact for its linear variation along the side). First order in
//    the mesh size for the viscous part.
//
//  * volume_force: the weak ("reaction", "consistent") form. With a test
//    function w = e_k on the body, 0 on every other boundary,
//
//        F_k = - integral over the domain of
//                  [ (du/dt + (u . grad) u) . w + nu grad u : grad w - p div w ],
//
//    which follows from the momentum equation by integration by parts. w is
//    the finite element function that is 1 at the body nodes and 0 elsewhere,
//    so only the first layer of elements around the body contributes. This
//    form converges at the rate of the energy norm squared for smooth
//    solutions (Babuska & Miller 1984; Giles, Larson, Levenstam & Suli 1997)
//    and is the one used for the reference values of Schafer & Turek (1996)
//    and John (2004). nu grad u : grad w is the viscous form of the solver
//    (Step 1); on a no-slip wall it gives the same force as the symmetric
//    form. If the body shares nodes with other boundaries (e.g. a body that
//    touches a channel wall), the traction on those boundaries within one
//    element of the shared nodes is included: an O(h) error.
//
// Integrals: triangles with the 3-point edge-midpoint rule, quadrilaterals
// with the 2 x 2 Gauss rule of the solver; exact for the polynomial
// integrands of triangles and parallelograms.

#include "Mesh.hpp"

#include <vector>

namespace hydrama::post {

struct BodyForce {
    double fx = 0.0; // total force
    double fy = 0.0;
    double mz = 0.0; // total pitching moment about reference point (z-axis, counter-clockwise / pitch-up)
    double pressure_fx = 0.0; // pressure part (surface_force only; volume_force: 0)
    double pressure_fy = 0.0;
    double pressure_mz = 0.0;
    double viscous_fx() const { return fx - pressure_fx; }
    double viscous_fy() const { return fy - pressure_fy; }
    double viscous_mz() const { return mz - pressure_mz; }
};

// Boundary sides of the body: indices into mesh.iside.
// Both functions need the geometry of Mesh::compute_shape_function_derivatives
// (and quad_geom for quadrilaterals); they throw std::invalid_argument for
// side indices out of range or field sizes different from npoin.
BodyForce surface_force(const Mesh& mesh, const std::vector<int>& body_sides, double nu,
                        const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& p,
                        double x_ref = 0.0, double y_ref = 0.0);

// dudt / dvdt: the time derivative of the velocity (nullptr for steady
// states, where it vanishes).
BodyForce volume_force(const Mesh& mesh, const std::vector<int>& body_sides, double nu,
                       const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& p,
                       const std::vector<double>* dudt = nullptr, const std::vector<double>* dvdt = nullptr);

// Drag / lift coefficient of a force component: F / (0.5 U_ref^2 L_ref).
double force_coefficient(double force, double reference_velocity, double reference_length);

// Pitching moment coefficient of a moment component: M / (0.5 U_ref^2 L_ref^2).
double moment_coefficient(double moment, double reference_velocity, double reference_length);

} // namespace hydrama::post
