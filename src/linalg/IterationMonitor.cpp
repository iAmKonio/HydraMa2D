#include "linalg/IterationMonitor.hpp"

#include <algorithm>
#include <ios>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace hydrama::linalg {

// ---------------------------------------------------------------------------
// ResidualHistory
// ---------------------------------------------------------------------------

void ResidualHistory::on_start(std::string_view solver, double initial_residual) {
    residuals_.clear();
    residuals_.push_back(initial_residual);
    solver_ = std::string(solver);
    result_ = SolveResult{};
    finished_ = false;
    ++solves_;
}

void ResidualHistory::on_iteration(int /*iteration*/, double residual) {
    residuals_.push_back(residual);
}

void ResidualHistory::on_finish(const SolveResult& result) {
    result_ = result;
    finished_ = true;
}

double ResidualHistory::max_growth_factor() const {
    double worst = 1.0;
    bool any = false;
    for (std::size_t k = 1; k < residuals_.size(); ++k) {
        if (residuals_[k - 1] > 0.0) {
            const double g = residuals_[k] / residuals_[k - 1];
            worst = any ? std::max(worst, g) : g;
            any = true;
        }
    }
    return any ? worst : 1.0;
}

// ---------------------------------------------------------------------------
// StreamMonitor
// ---------------------------------------------------------------------------

StreamMonitor::StreamMonitor(std::ostream& out, int every, std::string prefix)
    : out_(out), every_(every), prefix_(std::move(prefix)) {
    if (every_ < 1) throw std::invalid_argument("StreamMonitor: 'every' must be >= 1");
}

void StreamMonitor::on_start(std::string_view solver, double initial_residual) {
    solver_ = std::string(solver);
    last_written_ = 0;
    out_ << prefix_ << solver_ << " iter " << 0 << " residual " << std::scientific << initial_residual
         << std::defaultfloat << '\n';
}

void StreamMonitor::on_iteration(int iteration, double residual) {
    if (iteration % every_ == 0 || iteration == 1) {
        out_ << prefix_ << solver_ << " iter " << iteration << " residual " << std::scientific << residual
             << std::defaultfloat << '\n';
        last_written_ = iteration;
    }
}

void StreamMonitor::on_finish(const SolveResult& result) {
    if (result.iterations != last_written_) {
        out_ << prefix_ << solver_ << " iter " << result.iterations << " residual " << std::scientific
             << result.final_residual << std::defaultfloat << '\n';
    }
    out_ << prefix_ << solver_ << (result.converged ? " converged" : " NOT converged") << " ("
         << to_string(result.reason) << ") after " << result.iterations << " iterations\n";
}

// ---------------------------------------------------------------------------
// MonitorGroup
// ---------------------------------------------------------------------------

void MonitorGroup::add(IterationMonitor* monitor) {
    if (monitor) monitors_.push_back(monitor);
}

void MonitorGroup::on_start(std::string_view solver, double initial_residual) {
    for (auto* m : monitors_) m->on_start(solver, initial_residual);
}

void MonitorGroup::on_iteration(int iteration, double residual) {
    for (auto* m : monitors_) m->on_iteration(iteration, residual);
}

void MonitorGroup::on_finish(const SolveResult& result) {
    for (auto* m : monitors_) m->on_finish(result);
}

} // namespace hydrama::linalg
