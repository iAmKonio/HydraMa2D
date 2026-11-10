#pragma once

// Preconditioned conjugate gradient method for symmetric positive definite
// systems (symmetric positive definite preconditioner).
//
// Per iteration: one operator application, one preconditioner application,
// two inner products and one residual norm. With a pointwise preconditioner
// (Jacobi) the updates of x, r and z, the residual norm and r.z are fused
// into a single pass over the vectors.
//
// With ConvergenceCriteria::legacy_pcg() the iteration reproduces the
// original HydraMa PCG (solve_pcg) bit for bit: same operations, same
// summation order, same breakdown guards and stopping rules.

#include "linalg/KrylovSolver.hpp"

namespace hydrama::linalg {

class ConjugateGradient final : public KrylovSolver {
public:
    // Denominators |d| <= kBreakdownGuard give a zero step coefficient
    // under BreakdownPolicy::Continue (legacy guard).
    static constexpr double kBreakdownGuard = 1.0e-30;

    using KrylovSolver::KrylovSolver;

    std::string_view name() const override { return "CG"; }

protected:
    SolveResult iterate(const LinearOperator& a,
                        const Preconditioner& m,
                        VectorView x,
                        VectorView r,
                        ConvergenceControl& control) override;

private:
    std::vector<double> z_; // preconditioned residual
    std::vector<double> p_; // search direction
    std::vector<double> q_; // A p
};

} // namespace hydrama::linalg
