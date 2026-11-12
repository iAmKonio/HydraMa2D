#pragma once

#include "Mesh.hpp"
#include "thermal/ThermalConfig.hpp"

#include <vector>

namespace hydrama {

struct WallThermalReport {
    double total_heat_rate = 0.0; // Integral of -grad(T) . n_fluid ds
    double total_length = 0.0;    // Total boundary edge length
    double mean_heat_flux = 0.0;  // total_heat_rate / total_length
    double mean_nusselt = 0.0;    // Dimensionless mean Nusselt number
};

// Computes effective thermal diffusivity alpha_eff = nu / Pr + nu_t / Pr_t per element
void compute_thermal_diffusivity(const Mesh& mesh,
                                 double ani,
                                 const ThermalConfig& config,
                                 const std::vector<double>* nut_elem,
                                 std::vector<double>& alpha_elem);

// Assembles Boussinesq buoyancy body force into Step 1 momentum RHS:
//   rhs_u += int N_i f_{b,x} dOmega
//   rhs_v += int N_i f_{b,y} dOmega
// where f_b = b * (T - T_ref)
void add_boussinesq_buoyancy(const Mesh& mesh,
                             const ThermalConfig& config,
                             double re,
                             const std::vector<double>& T,
                             std::vector<double>& rhs_u,
                             std::vector<double>& rhs_v);

// Applies thermal Dirichlet boundary conditions (hot wall, cold wall, inlet, overrides)
void apply_thermal_bcs(const Mesh& mesh,
                       const ThermalConfig& config,
                       std::vector<double>& T);

// Computes wall heat transfer and average Nusselt number across sides matching `boundary_flag`
WallThermalReport compute_wall_nusselt(const Mesh& mesh,
                                       const ThermalConfig& config,
                                       const std::vector<double>& T,
                                       int boundary_flag,
                                       double delta_T = 1.0,
                                       double ref_length = 1.0);

} // namespace hydrama
