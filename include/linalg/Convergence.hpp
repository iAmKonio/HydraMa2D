#pragma once

// Convergence criteria, stop reasons and solve results shared by all Krylov
// solvers, plus ConvergenceControl: the single implementation of the
// stopping logic every solver uses.
//
// Residual norms are unpreconditioned 2-norms over the free
// (non-Dirichlet) dofs.
//
// Stopping rules, checked after every iteration k >= 1 in this order:
//   1. residual not finite                      -> NonFinite    (not converged)
//   2. reference > 0 and r_k / reference < rel  -> RelativeTolerance
//   3. r_k < abs                                -> AbsoluteTolerance
//   4. r_k == 0 and on_breakdown == Stop        -> ExactSolution ("happy breakdown")
//   5. k >= max_iterations                      -> MaxIterations (not converged)
// A tolerance of 0 switches that criterion off (the comparisons are strict).
//
// Before the first iteration (k = 0) the same rules are applied to r_0,
// except in the FirstIteration (legacy) mode, which always iterates at least
// once and only stops early on a non-finite r_0 or max_iterations <= 0.

#include <optional>
#include <string_view>

namespace hydrama::linalg {

class IterationMonitor;

// Denominator of the relative tolerance.
enum class ResidualReference {
    InitialResidual, // ||r_0|| = ||b - A x_0||  (default)
    RightHandSide,   // ||b||   (over the free dofs)
    FirstIteration   // ||r_1||, the residual AFTER the first iteration.
                     // Legacy PCG semantics of HydraMa; only used to
                     // reproduce the original pressure solver.
};

// What a solver does when a division by a (near) zero quantity occurs.
enum class BreakdownPolicy {
    // Stop: r == 0 exactly ends the solve as converged (ExactSolution); a
    // serious breakdown (e.g. p.Ap <= 0 in CG) ends it as Breakdown.
    Stop,
    // Legacy PCG behaviour: a step with a denominator |d| <= 1e-30 is taken
    // with a zero coefficient and the iteration continues.
    Continue
};

enum class StopReason {
    None,              // solve not finished
    RelativeTolerance,
    AbsoluteTolerance,
    ExactSolution,
    MaxIterations,
    Breakdown,
    NonFinite,
    Stagnation // the true residual stalls above the tolerance (attainable accuracy)
};

std::string_view to_string(StopReason reason);
std::string_view to_string(ResidualReference reference);

// True for the reasons that mean "the requested accuracy was reached".
constexpr bool is_converged(StopReason reason) {
    return reason == StopReason::RelativeTolerance || reason == StopReason::AbsoluteTolerance ||
           reason == StopReason::ExactSolution;
}

struct ConvergenceCriteria {
    double rel_tol = 0.0;          // 0 = off
    double abs_tol = 0.0;          // 0 = off
    int max_iterations = 100000;
    ResidualReference reference = ResidualReference::InitialResidual;
    BreakdownPolicy on_breakdown = BreakdownPolicy::Stop;

    // The criteria of the original HydraMa PCG: tolerances relative to the
    // residual after the first iteration, 100000 iterations, zero-step
    // breakdown guards. Used only for the legacy PCG configuration.
    static ConvergenceCriteria legacy_pcg(double rel_tol, double abs_tol);
};

struct SolveResult {
    int iterations = 0;
    double initial_residual = 0.0;   // ||r_0||
    double reference_residual = 0.0; // denominator of the relative test
                                     // (||r_1|| in FirstIteration mode)
    double final_residual = 0.0;     // residual of the returned iterate
    bool converged = false;
    StopReason reason = StopReason::None;
};

// Guard against endless "false convergence": solvers that verify a
// recursively updated (or estimated) residual against the true residual
// restart when only the former meets the tolerance. Near the attainable
// accuracy of the problem (rounding floor of ||b - A x||) this would repeat
// until the iteration cap. The guard reports stagnation when the true
// residual fails to halve over kMaxStalls consecutive false convergences.
class StagnationGuard {
public:
    static constexpr int kMaxStalls = 2;
    static constexpr double kRequiredReduction = 0.5;

    // Call after a false convergence with the true residual norm; true means
    // "stop: stagnation".
    bool stalled(double true_residual);
    void reset();

private:
    double last_ = -1.0; // true residual at the previous false convergence (< 0: none)
    int stalls_ = 0;
};

// Stateful stopping logic for one solve. Krylov solvers call begin() once
// with ||r_0||, check() after every iteration, and finish() with the stop
// reason. The monitor (optional, non-owning) is notified of every step.
class ConvergenceControl {
public:
    ConvergenceControl(const ConvergenceCriteria& criteria,
                       IterationMonitor* monitor,
                       std::string_view solver_name);

    // Returns a stop reason if no iteration is needed/allowed.
    // rhs_norm is only used with ResidualReference::RightHandSide.
    std::optional<StopReason> begin(double initial_residual, double rhs_norm = 0.0);

    // Record the residual of iteration k (k = 1, 2, ...); returns a stop
    // reason when the solve must end.
    std::optional<StopReason> check(int iteration, double residual);

    // Builds the result and notifies the monitor.
    SolveResult finish(StopReason reason);

    // Would `residual` meet the tolerances (rules 2-4 above)? No side
    // effects: lets a solver verify a recursively updated residual against
    // the true residual before reporting convergence.
    bool satisfied(double residual) const { return test(residual).has_value(); }

    const ConvergenceCriteria& criteria() const { return criteria_; }
    int iteration() const { return result_.iterations; }

private:
    std::optional<StopReason> test(double residual) const;

    ConvergenceCriteria criteria_;
    IterationMonitor* monitor_;
    std::string_view solver_name_;
    SolveResult result_;
};

} // namespace hydrama::linalg
