#pragma once

// CBS (characteristic-Galerkin) explicit transport kernel for nodal scalars
//
//   d(phi)/dt + u . grad(phi) = div(Gamma grad(phi)) + Q
//
// For every transported scalar and node i the kernel assembles the
// integrated explicit right-hand side (without the source Q, which is handled
// nodally by point_implicit_update()):
//
//   rhs_i = - int N_i (u_h . grad phi) dOmega                        convection (exact for P1)
//           - dt_e/2 * A_e (u_bar . grad N_i)(u_bar . grad phi)        CBS streamline term
//           - Gamma_e A_e grad N_i . grad phi                          diffusion
//
// u_bar is the element-mean velocity, dt_e the element time step and
// Gamma_e the element diffusivity (for a linear nodal diffusivity its element
// mean makes the diffusion integral exact). The explicit update is
// phi_i += rhs_i * elcoe2_i, exactly as for the momentum in Step 1: with
// phi = u and Gamma = 1/Re this kernel reproduces compute_step1().
// The source stabilisation term dt/2 u . grad(Q) of the full characteristic-
// Galerkin scheme is omitted (standard practice for turbulence equations).
//
// Several scalars (k and epsilon/omega) are assembled in ONE element pass
// sharing the velocity data (template parameter N).
//
// Reference: O. C. Zienkiewicz, R. L. Taylor, P. Nithiarasu, The Finite
// Element Method for Fluid Dynamics, 7th ed. (2014), CBS scheme / scalar
// transport with the characteristic-Galerkin procedure.

#include "Mesh.hpp"

#include <array>
#include <vector>

namespace hydrama {

struct TransportedScalar {
    const std::vector<double>* phi = nullptr;         // nodal values
    const std::vector<double>* diffusivity = nullptr; // element diffusivity Gamma_e >= 0
    std::vector<double>* rhs = nullptr;               // output (resized, overwritten)
};

// Instantiated for N = 1, 2, 3, 4.
template <int N>
void assemble_scalar_transport(const Mesh& mesh,
                               const std::vector<double>& u,
                               const std::vector<double>& v,
                               const std::vector<double>& delte,
                               const std::array<TransportedScalar, N>& scalars);

// Single-scalar convenience wrapper.
void assemble_scalar_transport(const Mesh& mesh,
                               const std::vector<double>& u,
                               const std::vector<double>& v,
                               const std::vector<double>& delte,
                               const std::vector<double>& phi,
                               const std::vector<double>& diffusivity,
                               std::vector<double>& rhs);

} // namespace hydrama
