#pragma once

#include "Mesh.hpp"
#include <vector>
#include <array>

namespace hydrama {

// Result of the legacy solve_pcg() interface. initial_residual is the
// residual after the FIRST iteration (legacy semantics, see
// linalg::ResidualReference::FirstIteration).
struct PCGResult {
    int iterations = 0;
    double initial_residual = 0.0;
    double final_residual = 0.0;
};

// b = A x for the element-by-element pressure operator (gstif, pdiag).
// Thin wrapper over linalg::ebe_apply (see linalg/EbeOperator.hpp).
void matrix_vector_mult(const Mesh& mesh,
                        const std::vector<std::array<double, 3>>& gstif,
                        const std::vector<double>& pdiag,
                        const std::vector<double>& x,
                        std::vector<double>& b);

void update_lhs(const Mesh& mesh,
                const std::vector<double>& delte,
                std::vector<double>& elcoe2,
                std::vector<double>& pdiag,
                std::vector<std::array<double, 3>>& gstif);

// Legacy PCG interface, kept for compatibility. Adapter over the linalg
// framework: EbeOperator + JacobiPreconditioner + ConjugateGradient with
// ConvergenceCriteria::legacy_pcg(relToler, absToler); reproduces the
// original implementation bit for bit. New code should use PressureSolver
// or the linalg classes directly.
PCGResult solve_pcg(const Mesh& mesh,
                    const std::vector<std::array<double, 3>>& gstif,
                    const std::vector<double>& pdiag,
                    const std::vector<double>& rhs1,
                    const std::vector<int>& bc_list,
                    const std::vector<double>& bc_values,
                    double relToler,
                    double absToler,
                    std::vector<double>& press);

void compute_timestep(int solver_opt, int dtfixed, int transient_on,
                      double dtfix, double ani, double csafm,
                      const Mesh& mesh,
                      const std::vector<double>& velocity,
                      std::vector<double>& delte);

// Same, with an effective diffusivity per element in the diffusive limit,
// dt_diff,e = csafm * h_e^2 / (2 nu_eff,e). With turbulence, nu_eff,e is
// max(nu + nu_t,e, largest diffusivity of the turbulence equations in e).
// For nu_eff,e == ani in every element the result is bit-identical to the
// overload above.
void compute_timestep(int solver_opt, int dtfixed, int transient_on,
                      double dtfix, const std::vector<double>& nu_eff_elem, double csafm,
                      const Mesh& mesh,
                      const std::vector<double>& velocity,
                      std::vector<double>& delte);

} // namespace hydrama
