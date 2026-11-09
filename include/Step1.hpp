#pragma once

#include "Mesh.hpp"
#include <vector>

namespace hydrama {

void compute_step1(const Mesh& mesh,
                   double ani,
                   const std::vector<double>& delte,
                   const std::vector<double>& u,
                   const std::vector<double>& v,
                   std::vector<double>& rhs_u,
                   std::vector<double>& rhs_v);

} // namespace hydrama
