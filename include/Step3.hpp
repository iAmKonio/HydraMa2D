#pragma once

#include "Mesh.hpp"
#include <vector>

namespace hydrama {

void compute_step3(const Mesh& mesh,
                   const std::vector<double>& u_n,
                   const std::vector<double>& v_n,
                   const std::vector<double>& p_new,
                   const std::vector<double>& p_old,
                   const std::vector<double>& delte,
                   std::vector<double>& rhs_u,
                   std::vector<double>& rhs_v);

} // namespace hydrama
