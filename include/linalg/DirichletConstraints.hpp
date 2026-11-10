#pragma once

// Dirichlet (essential) constraints x[dof] = value for a linear system.
//
// The solvers use the "homogenised projection" formulation of the original
// PCG: the prescribed values are imposed on the iterate, the residual is
// zeroed at the constrained dofs, and the Krylov iteration runs on the
// constrained operator P A (see ConstrainedOperator), so that every search
// direction vanishes at the constrained dofs and the prescribed values are
// kept exactly.

#include "linalg/VectorView.hpp"

#include <cstddef>
#include <vector>

namespace hydrama::linalg {

class DirichletConstraints {
public:
    DirichletConstraints() = default;

    // dofs[k] is fixed to values[k]. Throws std::invalid_argument if the
    // sizes differ or a dof is negative. Duplicate dofs are allowed; the last
    // value wins when imposing (as in the original implementation).
    DirichletConstraints(std::vector<int> dofs, std::vector<double> values);

    std::size_t size() const { return dofs_.size(); }
    bool empty() const { return dofs_.empty(); }
    const std::vector<int>& dofs() const { return dofs_; }
    const std::vector<double>& values() const { return values_; }

    // Throws std::out_of_range if a dof is not a valid index of an
    // n-vector.
    void validate(std::size_t n) const;

    // x[dof] = value for every constraint.
    void impose(VectorView x) const;

    // v[dof] = 0 for every constraint.
    void zero(VectorView v) const;

    // mask[i] = 0 at constrained dofs, 1 elsewhere (mask must have n entries).
    void free_mask(VectorView mask) const;

private:
    std::vector<int> dofs_;
    std::vector<double> values_;
};

} // namespace hydrama::linalg
