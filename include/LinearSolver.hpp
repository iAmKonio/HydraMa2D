#pragma once

#include "Mesh.hpp"
#include <vector>
#include <array>

namespace hydrama {

struct PCGResult {
    int iterations = 0;
    double initial_residual = 0.0;
    double final_residual = 0.0;
};

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

} // namespace hydrama
