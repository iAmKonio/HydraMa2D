#include "linalg/ConstrainedOperator.hpp"

namespace hydrama::linalg {

ConstrainedOperator::ConstrainedOperator(const LinearOperator& a, const DirichletConstraints& constraints)
    : a_(a), constraints_(constraints) {
    constraints_.validate(a_.size());
}

void ConstrainedOperator::apply(ConstVectorView x, VectorView y) const {
    a_.apply(x, y);
    constraints_.zero(y);
}

void ConstrainedOperator::diagonal(VectorView d) const {
    a_.diagonal(d);
    for (int dof : constraints_.dofs()) d[dof] = 1.0;
}

} // namespace hydrama::linalg
