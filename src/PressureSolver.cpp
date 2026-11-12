#include "PressureSolver.hpp"

#include "linalg/ConstrainedOperator.hpp"
#include "linalg/Factory.hpp"
#include "linalg/ScaledPreconditioner.hpp"

#include <cmath>

#include <stdexcept>
#include <string>

namespace hydrama {

linalg::ConvergenceCriteria pressure_criteria(const SolverConfig& config) {
    linalg::ConvergenceCriteria c;
    if (config.is_legacy_pcg()) {
        c = linalg::ConvergenceCriteria::legacy_pcg(config.relToler, config.absToler);
    } else {
        c.rel_tol = config.relToler;
        c.abs_tol = config.absToler;
        c.reference = linalg::ResidualReference::InitialResidual;
        c.on_breakdown = linalg::BreakdownPolicy::Stop;
    }
    c.max_iterations = config.max_iterations;
    if (config.residual_reference == "first_iteration") {
        c.reference = linalg::ResidualReference::FirstIteration;
    } else if (config.residual_reference == "initial") {
        c.reference = linalg::ResidualReference::InitialResidual;
    } else if (config.residual_reference == "rhs") {
        c.reference = linalg::ResidualReference::RightHandSide;
    } else if (!config.residual_reference.empty()) {
        throw std::invalid_argument("PressureSolver: unknown residual_reference '" + config.residual_reference + "'");
    }
    return c;
}

std::optional<double> uniform_scaling(const std::vector<std::array<double, 3>>& gstif,
                                      const std::vector<double>& pdiag,
                                      const std::vector<std::array<double, 3>>& gstif_ref,
                                      const std::vector<double>& pdiag_ref,
                                      double tol) {
    if (gstif.size() != gstif_ref.size() || pdiag.size() != pdiag_ref.size()) return std::nullopt;
    double sum = 0.0, sum_ref = 0.0;
    for (std::size_t i = 0; i < pdiag.size(); ++i) {
        sum += pdiag[i];
        sum_ref += pdiag_ref[i];
    }
    if (!(sum > 0.0) || !(sum_ref > 0.0) || !std::isfinite(sum) || !std::isfinite(sum_ref)) return std::nullopt;
    const double c = sum / sum_ref;
    auto close = [c, tol](double v, double ref) {
        const double expected = c * ref;
        return std::abs(v - expected) <= tol * std::abs(expected);
    };
    for (std::size_t i = 0; i < pdiag.size(); ++i)
        if (!close(pdiag[i], pdiag_ref[i])) return std::nullopt;
    for (std::size_t e = 0; e < gstif.size(); ++e)
        for (int k = 0; k < 3; ++k)
            if (!close(gstif[e][k], gstif_ref[e][k])) return std::nullopt;
    return c;
}

PressureSolver::PressureSolver(const SolverConfig& config, const PressureBC& bc)
    : config_(config), constraints_(bc.nodes, bc.values) {
    // solver_opt keeps its original meaning: it is only read by
    // compute_timestep() (solver_opt > 1 forces a global time step).
    if (config_.operator_format != "ebe" && config_.operator_format != "csr") {
        throw std::invalid_argument("PressureSolver: unknown operator_format '" + config_.operator_format + "'");
    }
    const linalg::PreconditionerInfo* info = linalg::find_preconditioner(config_.preconditioner);
    if (!info) {
        throw std::invalid_argument("PressureSolver: unknown preconditioner '" + config_.preconditioner + "'");
    }
    if (config_.preconditioner_reuse != "rescale" && config_.preconditioner_reuse != "rebuild") {
        throw std::invalid_argument("PressureSolver: unknown preconditioner_reuse '" + config_.preconditioner_reuse +
                                    "' (rescale | rebuild)");
    }
    legacy_ = config_.is_legacy_pcg();
    csr_operator_ = config_.operator_format == "csr";
    assembled_precond_ = info->requires_assembled;

    linalg::PreconditionerOptions options;
    options.ssor_omega = config_.ssor_omega;
    options.diagonal_shift = config_.diagonal_shift;
    options.ordering = config_.reordering;
    precond_ = linalg::make_preconditioner(config_.preconditioner, options);
    // Reuse pays off (and is used) only for assembled preconditioners; it is
    // exact only for homogeneous ones.
    reuse_ = config_.preconditioner_reuse == "rescale" && assembled_precond_ && precond_->traits().homogeneous;
    linalg::KrylovOptions krylov_options;
    krylov_options.gmres_restart = config_.gmres_restart;
    krylov_options.gmres_orthogonalization = config_.gmres_orthogonalization;
    krylov_ = linalg::make_krylov_solver(config_.method, pressure_criteria(config_), krylov_options);
}

void PressureSolver::update(const Mesh& mesh,
                            const std::vector<std::array<double, 3>>& gstif,
                            const std::vector<double>& pdiag) {
    ebe_.emplace(mesh, gstif, pdiag);
    op_ = &*ebe_;

    if (csr_operator_ || assembled_precond_) {
        // Pattern, scatter map (and, inside the preconditioner, orderings)
        // are built once per mesh.
        if (!assembler_ || assembled_mesh_ != &mesh || assembler_->nelem() != mesh.nelem ||
            assembler_->npoin() != mesh.npoin) {
            assembler_ = std::make_unique<linalg::P1CsrAssembler>(mesh);
            assembled_mesh_ = &mesh;
            csr_ = assembler_->create_matrix();
            precond_ready_ = false;
            reference_valid_ = false;
            ++stats_.structure_builds;
        }
        if (csr_operator_) {
            assembler_->assemble_ebe(mesh, gstif, pdiag, csr_);
            op_ = &csr_;
        }
    }

    const linalg::OperatorChange change = linalg::OperatorChange::Values;
    periodic_map_ = PeriodicMapping::build(mesh);

    if (periodic_map_.has_periodic) {
        std::vector<int> red_dofs;
        std::vector<double> red_vals;
        for (std::size_t k = 0; k < constraints_.size(); ++k) {
            int orig_dof = constraints_.dofs()[k];
            red_dofs.push_back(periodic_map_.to_indep[orig_dof]);
            red_vals.push_back(constraints_.values()[k]);
        }
        reduced_constraints_ = linalg::DirichletConstraints(red_dofs, red_vals);

        reduced_op_ = std::make_unique<linalg::PeriodicReducedOperator>(*op_, periodic_map_);
        op_ = reduced_op_.get();

        const linalg::ConstrainedOperator constrained(*op_, reduced_constraints_);
        if (precond_ready_) {
            precond_->update(constrained, change);
        } else {
            precond_->setup(constrained);
        }
    } else if (assembled_precond_) {
        // A = c A_ref: keep the setup, rescale its inverse.
        if (reuse_ && precond_ready_ && reference_valid_) {
            if (const auto c = uniform_scaling(gstif, pdiag, gstif_ref_, pdiag_ref_)) {
                scale_ = 1.0 / *c;
                ++stats_.rescaled_reuses;
                return;
            }
        }
        if (!csr_operator_) assembler_->assemble_ebe(mesh, gstif, pdiag, csr_);
        linalg::constrain(csr_, constraints_, csr_constrained_);
        if (precond_ready_) {
            precond_->update(csr_constrained_, change);
        } else {
            precond_->setup(csr_constrained_);
        }
        if (reuse_) {
            gstif_ref_ = gstif;
            pdiag_ref_ = pdiag;
            reference_valid_ = true;
        }
    } else {
        const linalg::ConstrainedOperator constrained(*op_, constraints_);
        if (precond_ready_) {
            precond_->update(constrained, change);
        } else {
            precond_->setup(constrained);
        }
    }
    scale_ = 1.0;
    precond_ready_ = true;
    ++stats_.numeric_setups;
}

const linalg::LinearOperator& PressureSolver::op() const {
    if (!op_) throw std::logic_error("PressureSolver: update() has not been called");
    return *op_;
}

linalg::SolveResult PressureSolver::solve(const std::vector<double>& rhs, std::vector<double>& p) {
    if (!op_) throw std::logic_error("PressureSolver::solve called before update()");
    if (periodic_map_.has_periodic) {
        const int n_indep = periodic_map_.num_independent;
        std::vector<double> rhs_hat(static_cast<std::size_t>(n_indep), 0.0);
        for (int i = 0; i < periodic_map_.num_nodes; ++i) {
            rhs_hat[static_cast<std::size_t>(periodic_map_.to_indep[i])] += rhs[static_cast<std::size_t>(i)];
        }
        std::vector<double> p_hat(static_cast<std::size_t>(n_indep), 0.0);
        for (int j = 0; j < n_indep; ++j) {
            p_hat[static_cast<std::size_t>(j)] = p[static_cast<std::size_t>(periodic_map_.from_indep[static_cast<std::size_t>(j)])];
        }

        linalg::SolveResult res;
        if (assembled_precond_) {
            const linalg::ScaledPreconditioner m(*precond_, scale_);
            res = krylov_->solve(*op_, m, rhs_hat, p_hat, reduced_constraints_);
        } else {
            res = krylov_->solve(*op_, *precond_, rhs_hat, p_hat, reduced_constraints_);
        }

        for (int i = 0; i < periodic_map_.num_nodes; ++i) {
            p[static_cast<std::size_t>(i)] = p_hat[static_cast<std::size_t>(periodic_map_.to_indep[i])];
        }
        return res;
    }

    if (assembled_precond_) {
        // After a numeric setup scale_ == 1 and the view is transparent.
        const linalg::ScaledPreconditioner m(*precond_, scale_);
        return krylov_->solve(*op_, m, rhs, p, constraints_);
    }
    return krylov_->solve(*op_, *precond_, rhs, p, constraints_);
}

void PressureSolver::set_monitor(linalg::IterationMonitor* monitor) {
    krylov_->set_monitor(monitor);
}

std::string PressureSolver::label() const {
    return config_.method == "cg" ? std::string("PCG") : std::string(krylov_->name());
}

std::string PressureSolver::description() const {
    const std::string family = config_.method == "cg" ? "PCG" : "Krylov";
    return family + " (" + std::string(krylov_->name()) + " + " + std::string(precond_->name()) + ", " +
           config_.operator_format + ")";
}

} // namespace hydrama
