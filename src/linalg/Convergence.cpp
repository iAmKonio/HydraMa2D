#include "linalg/Convergence.hpp"

#include "linalg/IterationMonitor.hpp"

#include <cmath>

namespace hydrama::linalg {

std::string_view to_string(StopReason reason) {
    switch (reason) {
        case StopReason::None: return "none";
        case StopReason::RelativeTolerance: return "relative tolerance";
        case StopReason::AbsoluteTolerance: return "absolute tolerance";
        case StopReason::ExactSolution: return "exact solution";
        case StopReason::MaxIterations: return "maximum iterations";
        case StopReason::Breakdown: return "breakdown";
        case StopReason::NonFinite: return "non-finite residual";
        case StopReason::Stagnation: return "stagnation (attainable accuracy reached)";
    }
    return "unknown";
}

std::string_view to_string(ResidualReference reference) {
    switch (reference) {
        case ResidualReference::InitialResidual: return "initial";
        case ResidualReference::RightHandSide: return "rhs";
        case ResidualReference::FirstIteration: return "first_iteration";
    }
    return "unknown";
}

bool StagnationGuard::stalled(double true_residual) {
    if (last_ >= 0.0 && !(true_residual < kRequiredReduction * last_)) {
        ++stalls_;
    } else {
        stalls_ = 0;
    }
    last_ = true_residual;
    return stalls_ >= kMaxStalls;
}

void StagnationGuard::reset() {
    last_ = -1.0;
    stalls_ = 0;
}

ConvergenceCriteria ConvergenceCriteria::legacy_pcg(double rel_tol, double abs_tol) {
    ConvergenceCriteria c;
    c.rel_tol = rel_tol;
    c.abs_tol = abs_tol;
    c.max_iterations = 100000;
    c.reference = ResidualReference::FirstIteration;
    c.on_breakdown = BreakdownPolicy::Continue;
    return c;
}

ConvergenceControl::ConvergenceControl(const ConvergenceCriteria& criteria,
                                       IterationMonitor* monitor,
                                       std::string_view solver_name)
    : criteria_(criteria), monitor_(monitor), solver_name_(solver_name) {}

std::optional<StopReason> ConvergenceControl::test(double residual) const {
    const double reference = result_.reference_residual;
    if (reference > 0.0 && (residual / reference) < criteria_.rel_tol) return StopReason::RelativeTolerance;
    if (residual < criteria_.abs_tol) return StopReason::AbsoluteTolerance;
    if (residual == 0.0 && criteria_.on_breakdown == BreakdownPolicy::Stop) return StopReason::ExactSolution;
    return std::nullopt;
}

std::optional<StopReason> ConvergenceControl::begin(double initial_residual, double rhs_norm) {
    result_ = SolveResult{};
    result_.initial_residual = initial_residual;
    result_.final_residual = initial_residual;
    if (monitor_) monitor_->on_start(solver_name_, initial_residual);

    if (!std::isfinite(initial_residual)) return StopReason::NonFinite;

    switch (criteria_.reference) {
        case ResidualReference::InitialResidual: result_.reference_residual = initial_residual; break;
        case ResidualReference::RightHandSide: result_.reference_residual = rhs_norm; break;
        case ResidualReference::FirstIteration: result_.reference_residual = 0.0; break; // set at k = 1
    }

    if (criteria_.reference != ResidualReference::FirstIteration) {
        if (auto stop = test(initial_residual)) return stop;
    }
    if (criteria_.max_iterations <= 0) return StopReason::MaxIterations;
    return std::nullopt;
}

std::optional<StopReason> ConvergenceControl::check(int iteration, double residual) {
    result_.iterations = iteration;
    result_.final_residual = residual;
    if (monitor_) monitor_->on_iteration(iteration, residual);

    if (!std::isfinite(residual)) return StopReason::NonFinite;
    if (criteria_.reference == ResidualReference::FirstIteration && iteration == 1) {
        result_.reference_residual = residual;
    }
    if (auto stop = test(residual)) return stop;
    if (iteration >= criteria_.max_iterations) return StopReason::MaxIterations;
    return std::nullopt;
}

SolveResult ConvergenceControl::finish(StopReason reason) {
    result_.reason = reason;
    result_.converged = is_converged(reason);
    if (monitor_) monitor_->on_finish(result_);
    return result_;
}

} // namespace hydrama::linalg
