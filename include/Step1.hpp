#pragma once

#include "Mesh.hpp"
#include <vector>

namespace hydrama {

// CBS Step 1 (intermediate momentum), laminar: convection, CBS streamline
// stabilisation and the molecular viscous term ani * Laplacian(u).
void compute_step1(const Mesh& mesh,
                   double ani,
                   const std::vector<double>& delte,
                   const std::vector<double>& u,
                   const std::vector<double>& v,
                   std::vector<double>& rhs_u,
                   std::vector<double>& rhs_v);

// CBS Step 1 with an eddy viscosity nu_t (element values, e.g. element means
// of a nodal P1 field). The molecular part is unchanged (Laplacian form); the
// turbulent stress is added in symmetric form
//
//   rhs_i -= int grad N_i . nu_t (grad u + grad u^T) dOmega
//
// i.e. per element with A_e = detJ/2:
//   rhs_u_i -= A_e nu_t,e [dN_i/dx 2 du/dx + dN_i/dy (du/dy + dv/dx)]
//   rhs_v_i -= A_e nu_t,e [dN_i/dx (du/dy + dv/dx) + dN_i/dy 2 dv/dy]
// (exact for a linear nu_t represented by its element mean). This form is
// objective (a rigid rotation produces no stress) and correct for spatially
// varying nu_t; the isotropic part -2/3 k I is absorbed into the pressure.
//
// Elements with nu_t,e == 0 skip the turbulent term, so an empty nut_elem or
// one that is identically zero reproduces the laminar overload bit for bit.
void compute_step1(const Mesh& mesh,
                   double ani,
                   const std::vector<double>& nut_elem,
                   const std::vector<double>& delte,
                   const std::vector<double>& u,
                   const std::vector<double>& v,
                   std::vector<double>& rhs_u,
                   std::vector<double>& rhs_v);

} // namespace hydrama
