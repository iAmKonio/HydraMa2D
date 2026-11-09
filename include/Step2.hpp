#pragma once

#include "Mesh.hpp"
#include <vector>

namespace hydrama {

void compute_step2(const Mesh& mesh,
                   const std::vector<double>& u_star,
                   const std::vector<double>& v_star,
                   const std::vector<double>& u_n,
                   const std::vector<double>& v_n,
                   double theta1,
                   std::vector<double>& rhs1);

} // namespace hydrama
