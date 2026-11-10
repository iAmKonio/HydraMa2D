#pragma once

// Decorator that restricts an operator to the Dirichlet-free subspace:
//
//     apply:     y = P A x,   i.e. A x with the constrained rows zeroed
//     diagonal:  diag(A) with 1 at the constrained dofs
//
// where P zeroes the constrained dofs. On vectors that vanish at the
// constrained dofs (all Krylov search directions do) this is the operator
// P A P + (I - P): the constrained rows/columns are removed and replaced by
// the identity, which keeps a symmetric positive (semi-)definite A symmetric
// and positive definite. This is exactly the operator the original PCG
// iterated with (A p followed by zeroing at the Dirichlet nodes).
//
// The decorator is non-owning: the wrapped operator and constraints must
// outlive it.

#include "linalg/DirichletConstraints.hpp"
#include "linalg/LinearOperator.hpp"

namespace hydrama::linalg {

class ConstrainedOperator final : public LinearOperator {
public:
    // Throws std::out_of_range if a constrained dof is out of range for A.
    ConstrainedOperator(const LinearOperator& a, const DirichletConstraints& constraints);

    std::size_t size() const override { return a_.size(); }
    void apply(ConstVectorView x, VectorView y) const override;
    void diagonal(VectorView d) const override;
    OperatorTraits traits() const override { return a_.traits(); }
    std::string_view name() const override { return "constrained"; }

    const LinearOperator& base() const { return a_; }
    const DirichletConstraints& constraints() const { return constraints_; }

private:
    const LinearOperator& a_;
    const DirichletConstraints& constraints_;
};

} // namespace hydrama::linalg
