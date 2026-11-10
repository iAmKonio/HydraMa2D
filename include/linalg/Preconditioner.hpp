#pragma once

// Abstract preconditioner z = M^{-1} r (Strategy pattern).
//
// Life cycle:
//   setup(A)            build from the operator (symbolic + numeric work);
//                       must be called before the first apply()
//   update(A, change)   the operator changed; lets expensive preconditioners
//                       reuse work (e.g. keep a sparsity pattern or an AMG
//                       hierarchy when only values changed). The default
//                       simply calls setup(A).
//   apply(r, z)         z = M^{-1} r; r and z must not alias.
//
// Pointwise preconditioners (Jacobi, identity-like scalings) expose their
// scaling vector through pointwise_scaling() so that Krylov solvers can fuse
// z = s .* r into their vector update loops instead of making an extra pass.

#include "linalg/LinearOperator.hpp"
#include "linalg/VectorView.hpp"

#include <cstddef>
#include <string_view>

namespace hydrama::linalg {

struct PreconditionerTraits {
    // M^{-1} is symmetric (required by CG).
    bool symmetric = true;
    // M^{-1} changes between applications (e.g. inner iterations); requires
    // a flexible Krylov method (FCG / FGMRES).
    bool variable = false;
    // Homogeneous of degree one: M(c A) = c M(A) for c > 0. Then a
    // preconditioner set up for A can serve c A as (1/c) M(A)^{-1}
    // (ScaledPreconditioner) without a new setup.
    bool homogeneous = false;
};

enum class OperatorChange {
    Values,    // same sparsity/structure, new coefficients
    Structure  // anything may have changed
};

class Preconditioner {
public:
    virtual ~Preconditioner() = default;

    virtual void setup(const LinearOperator& a) = 0;
    virtual void update(const LinearOperator& a, OperatorChange change);
    virtual void apply(ConstVectorView r, VectorView z) const = 0;

    // Size of the operator the preconditioner was set up for (0 before
    // setup()).
    virtual std::size_t size() const = 0;

    virtual PreconditionerTraits traits() const { return {}; }
    virtual std::string_view name() const = 0;

    // For pointwise preconditioners z[i] = s[i] * r[i]: the scaling s.
    // Empty view otherwise.
    virtual ConstVectorView pointwise_scaling() const { return {}; }
};

} // namespace hydrama::linalg
