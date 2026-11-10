#include "linalg/DirichletConstraints.hpp"

#include "linalg/VectorOps.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace hydrama::linalg {

DirichletConstraints::DirichletConstraints(std::vector<int> dofs, std::vector<double> values)
    : dofs_(std::move(dofs)), values_(std::move(values)) {
    if (dofs_.size() != values_.size()) {
        throw std::invalid_argument("DirichletConstraints: " + std::to_string(dofs_.size()) + " dofs but " +
                                    std::to_string(values_.size()) + " values");
    }
    for (int d : dofs_) {
        if (d < 0) throw std::invalid_argument("DirichletConstraints: negative dof " + std::to_string(d));
    }
}

void DirichletConstraints::validate(std::size_t n) const {
    for (int d : dofs_) {
        if (static_cast<std::size_t>(d) >= n) {
            throw std::out_of_range("DirichletConstraints: dof " + std::to_string(d) +
                                    " out of range for a system of size " + std::to_string(n));
        }
    }
}

void DirichletConstraints::impose(VectorView x) const {
    for (std::size_t k = 0; k < dofs_.size(); ++k) x[dofs_[k]] = values_[k];
}

void DirichletConstraints::zero(VectorView v) const {
    for (int d : dofs_) v[d] = 0.0;
}

void DirichletConstraints::free_mask(VectorView mask) const {
    fill(mask, 1.0);
    zero(mask);
}

} // namespace hydrama::linalg
