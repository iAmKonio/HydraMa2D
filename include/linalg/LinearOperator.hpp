#pragma once

// Abstract linear operator y = A x (Strategy pattern).
//
// The Krylov solvers and preconditioners only see this interface, so the
// storage of A is interchangeable: the matrix-free element-by-element
// operator of the pressure equation (EbeOperator), assembled sparse matrices
// (CsrMatrix), decorators such as the Dirichlet-constrained operator
// (ConstrainedOperator), and in future block operators, ...
//
// Granularity: one virtual call per O(n) operation, so dynamic dispatch cost
// is negligible compared to the work done. Concrete operators should be
// declared `final`.

#include "linalg/VectorView.hpp"

#include <cstddef>
#include <string_view>

namespace hydrama::linalg {

class CsrMatrix;

// Structural properties advertised by an operator. They are hints used to
// validate solver/preconditioner combinations (e.g. CG requires symmetry);
// they are not verified numerically.
struct OperatorTraits {
    bool symmetric = false;
};

class LinearOperator {
public:
    virtual ~LinearOperator() = default;

    // Number of rows (= columns; all operators are square).
    virtual std::size_t size() const = 0;

    // y = A x. x and y must have size() entries and must not alias.
    virtual void apply(ConstVectorView x, VectorView y) const = 0;

    // d = diag(A). Used by diagonal-based preconditioners (Jacobi, and in
    // future Chebyshev/AMG smoothers). Operators that cannot provide their
    // diagonal cheaply keep the default, which throws std::logic_error.
    virtual void diagonal(VectorView d) const;

    virtual OperatorTraits traits() const { return {}; }

    // Capability query: the operator as an assembled CSR matrix, or nullptr
    // for matrix-free operators. Preconditioners that need matrix entries
    // (SSOR, incomplete factorisations) use this instead of dynamic_cast.
    virtual const CsrMatrix* assembled() const { return nullptr; }

    virtual std::string_view name() const = 0;
};

} // namespace hydrama::linalg
