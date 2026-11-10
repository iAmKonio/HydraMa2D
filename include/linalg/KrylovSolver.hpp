#pragma once

// Base class of the Krylov solvers (Template Method / non-virtual interface).
//
// solve() is non-virtual and performs everything that is common to all
// methods:
//   1. validates sizes (operator, preconditioner, vectors, constraints),
//   2. imposes the Dirichlet values on x,
//   3. computes the projected initial residual r_0 = P (b - A x),
//   4. starts the convergence control (criteria + monitor),
// and then hands over to the method-specific iterate(), which iterates on
// the constrained operator P A (ConstrainedOperator) and reports every
// iteration to the convergence control.
//
// Solver objects own their work vectors and keep them between solves, so
// repeated solves of the same size (one per time step) do not allocate.
// Solvers are therefore stateful and not safe for concurrent solve() calls.

#include "linalg/Convergence.hpp"
#include "linalg/DirichletConstraints.hpp"
#include "linalg/LinearOperator.hpp"
#include "linalg/Preconditioner.hpp"

#include <string_view>
#include <vector>

namespace hydrama::linalg {

class IterationMonitor;

class KrylovSolver {
public:
    explicit KrylovSolver(ConvergenceCriteria criteria = {});
    virtual ~KrylovSolver() = default;

    KrylovSolver(const KrylovSolver&) = delete;
    KrylovSolver& operator=(const KrylovSolver&) = delete;

    // Solve A x = b subject to x[dof] = value for the given constraints.
    // x holds the initial guess on entry and the solution on exit. The
    // preconditioner must be set up for an operator of the same size. b and
    // x must not alias. Throws std::invalid_argument / std::out_of_range on
    // inconsistent input.
    SolveResult solve(const LinearOperator& a,
                      const Preconditioner& m,
                      ConstVectorView b,
                      VectorView x,
                      const DirichletConstraints& constraints = DirichletConstraints{});

    const ConvergenceCriteria& criteria() const { return criteria_; }
    void set_criteria(const ConvergenceCriteria& criteria) { criteria_ = criteria; }

    // Non-owning; nullptr detaches.
    void set_monitor(IterationMonitor* monitor) { monitor_ = monitor; }
    IterationMonitor* monitor() const { return monitor_; }

    virtual std::string_view name() const = 0;

protected:
    // Method-specific iteration. On entry x satisfies the constraints and
    // r = P (b - A x); `a` is the constrained operator. Must report each
    // iteration through control.check() and return control.finish(reason).
    virtual SolveResult iterate(const LinearOperator& a,
                                const Preconditioner& m,
                                VectorView x,
                                VectorView r,
                                ConvergenceControl& control) = 0;

    // Resizes a work vector only when the size changes.
    static void ensure_size(std::vector<double>& v, std::size_t n);

    // Helpers for iterate(), valid only during solve():
    // r = P (b - A x) with the unconstrained operator; returns ||r||.
    double true_residual(ConstVectorView x, VectorView r) const;
    // v[dof] = 0 at the constrained dofs. Applied to preconditioned
    // directions so that the iterate keeps its prescribed values even if
    // the preconditioner does not decouple the constrained dofs.
    void project(VectorView v) const;

private:
    ConvergenceCriteria criteria_;
    IterationMonitor* monitor_ = nullptr;
    std::vector<double> r_;   // residual
    std::vector<double> tmp_; // projected rhs (RightHandSide reference only)

    // Problem of the current solve() (non-owning).
    const LinearOperator* a_ = nullptr;
    ConstVectorView b_;
    const DirichletConstraints* constraints_ = nullptr;
};

} // namespace hydrama::linalg
