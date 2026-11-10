#pragma once

// Turbulence boundary conditions resolved from the mesh boundary flags and
// the .turbulence configuration into compiled Dirichlet sets.
//
//   inlet sides    -> inlet_value(flag)   (per-flag value, e.g. nu_tilde = 3 nu)
//   wall sides     -> wall_value          (if given; e.g. nu_tilde = 0, k = 0)
//   outlet, symmetry, unknown -> natural (zero diffusive flux): no entry
//
// Walls are applied after inlets, so nodes shared by an inlet and a wall
// (corners) take the wall value - the same precedence as the velocity BCs.
// The boundary kind of each flag is TurbulenceConfig::boundary_kind(flag),
// i.e. the BoundaryKind classification unless overridden in the config.

#include "Mesh.hpp"
#include "numerics/DirichletSet.hpp"
#include "turbulence/TurbulenceConfig.hpp"

#include <functional>
#include <optional>

namespace hydrama {

DirichletSet build_turbulence_dirichlet(const Mesh& mesh,
                                        const TurbulenceConfig& config,
                                        const std::function<double(int flag)>& inlet_value,
                                        std::optional<double> wall_value);

// Same with a wall value per wall node (e.g. omega_w = 60 nu / (beta_1 y_1^2)).
DirichletSet build_turbulence_dirichlet(const Mesh& mesh,
                                        const TurbulenceConfig& config,
                                        const std::function<double(int flag)>& inlet_value,
                                        const std::function<double(int node)>& wall_value);

// Inlet / free-stream values of the two-equation variables from an inlet
// specification, with C_mu = beta* = 0.09:
//   k     = given k, else 1.5 (U_ref I)^2
//   eps   = given epsilon; else from omega: C_mu k omega; else from nu_t/nu = r:
//           C_mu k^2 / (r nu); else from the length scale: C_mu^(3/4) k^(3/2) / l
//   omega = given omega, else eps / (C_mu k)    (consistent with eps)
// (eps, omega are floored at tiny positive values for k = 0.)
struct TwoEquationInlet {
    double k = 0.0;
    double epsilon = 0.0;
    double omega = 0.0;
};
TwoEquationInlet two_equation_inlet(const InletTurbulence& inlet, double nu, double reference_velocity);

} // namespace hydrama
