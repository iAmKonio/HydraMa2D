#pragma once

// Observers of iterative solves (Observer pattern).
//
// A Krylov solver notifies its monitor (if any) through ConvergenceControl:
//   on_start(solver, r0)      once, before the first iteration
//   on_iteration(k, r_k)      after every iteration k = 1, 2, ...
//   on_finish(result)         once, when the solve ends (for any reason)
// Monitors are non-owning attachments: they must outlive the solves they
// observe, and they never influence the iteration. Without a monitor the
// cost is one null-pointer check per iteration.

#include "linalg/Convergence.hpp"

#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace hydrama::linalg {

class IterationMonitor {
public:
    virtual ~IterationMonitor() = default;
    virtual void on_start(std::string_view /*solver*/, double /*initial_residual*/) {}
    virtual void on_iteration(int iteration, double residual) = 0;
    virtual void on_finish(const SolveResult& /*result*/) {}
};

// Records the residual history of the most recent solve:
// residuals()[0] = ||r_0||, residuals()[k] = ||r_k||.
class ResidualHistory final : public IterationMonitor {
public:
    void on_start(std::string_view solver, double initial_residual) override;
    void on_iteration(int iteration, double residual) override;
    void on_finish(const SolveResult& result) override;

    const std::vector<double>& residuals() const { return residuals_; }
    const std::string& solver() const { return solver_; }
    const SolveResult& result() const { return result_; }
    bool finished() const { return finished_; }
    int solves() const { return solves_; } // number of solves observed

    // Worst ratio r_k / r_{k-1} over the history (1 if fewer than 2 entries
    // or a zero residual); a measure of monotonicity.
    double max_growth_factor() const;

private:
    std::vector<double> residuals_;
    std::string solver_;
    SolveResult result_;
    bool finished_ = false;
    int solves_ = 0;
};

// Writes one line per `every` iterations (and for the first/last) to a
// stream:  "<prefix><solver> iter <k> residual <r_k>".
class StreamMonitor final : public IterationMonitor {
public:
    explicit StreamMonitor(std::ostream& out, int every = 1, std::string prefix = "  ");

    void on_start(std::string_view solver, double initial_residual) override;
    void on_iteration(int iteration, double residual) override;
    void on_finish(const SolveResult& result) override;

private:
    std::ostream& out_;
    int every_;
    std::string prefix_;
    std::string solver_;
    int last_written_ = -1;
};

// Forwards every notification to a list of monitors (Composite).
class MonitorGroup final : public IterationMonitor {
public:
    // Adds a non-owning monitor; nullptr is ignored.
    void add(IterationMonitor* monitor);
    std::size_t size() const { return monitors_.size(); }

    void on_start(std::string_view solver, double initial_residual) override;
    void on_iteration(int iteration, double residual) override;
    void on_finish(const SolveResult& result) override;

private:
    std::vector<IterationMonitor*> monitors_;
};

} // namespace hydrama::linalg
