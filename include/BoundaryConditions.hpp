#pragma once

#include "Mesh.hpp"
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

void apply_velocity_bcs(const Mesh& mesh,
                        std::vector<double>& u,
                        std::vector<double>& v);

} // namespace hydrama
