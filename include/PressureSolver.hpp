#pragma once

// Facade of the linear algebra layer for the CBS pressure Poisson equation
// (Step 2). The time loop only calls
//
//     update(mesh, gstif, pdiag)   after update_lhs() changed the operator
//     solve(rhs, p)                p: initial guess in, solution out
//
// Composition (from the *.solver settings, see SolverConfig):
//
//   operator        operator_format = ebe : matrix-free EbeOperator (default)
//                                     csr : CsrMatrix assembled from the
//                                           element data every update
//   preconditioner  linalg::make_preconditioner(preconditioner, options);
//                   assembled preconditioners (ssor, ic0, ilu0) are built
//                   from the Dirichlet-constrained CSR matrix, which is
//                   assembled for them even when the Krylov operator is EBE;
//                   reordering = rcm wraps them in a ReorderedPreconditioner
//   Krylov method   linalg::make_krylov_solver(method, criteria, options):
//                   cg (SPD), bicgstab, gmres (restart, orthogonalisation)
//   criteria        legacy PCG (default settings): legacy_pcg(); otherwise
//                   the reference of residual_reference ("initial" if not
//                   given) and stop on breakdown
//
// All objects, matrices and work vectors persist between time steps; the
// sparsity pattern, element scatter map and orderings are computed once
// per mesh.
//
// Setup reuse (preconditioner_reuse = rescale, assembled preconditioners):
// with global/uniform time stepping the operator is A = dt K with constant
// K. update() compares the element data with the data of the last numeric
// setup and, if A = c A_ref (every entry within 1e-12 relative, c from the
// assembled diagonal), skips assembly, constraint, permutation and
// refactorisation: the Krylov solver then uses (1/c) M(A_ref)^{-1}
// (ScaledPreconditioner; exact for homogeneous preconditioners). Any other
// change (local time steps, geometry, mesh) triggers a new numeric setup,
// and a new mesh also a new pattern/ordering. On the constrained dofs
// M(A_ref) keeps unit rows while A's are 1 as well; the difference is never
// seen because residuals and corrections vanish there. Jacobi (O(n) setup)
// and the legacy path are always set up from the current values.

#include "BoundaryConditions.hpp"
#include "Config.hpp"
#include "Mesh.hpp"
#include "linalg/Convergence.hpp"
#include "linalg/CsrAssembly.hpp"
#include "linalg/CsrMatrix.hpp"
#include "linalg/DirichletConstraints.hpp"
#include "linalg/EbeOperator.hpp"
#include "linalg/KrylovSolver.hpp"
#include "linalg/Preconditioner.hpp"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hydrama {

namespace linalg {
class IterationMonitor;
}

class PressureSolver {
public:
    // Throws std::invalid_argument for unknown/inconsistent settings or
    // invalid pressure constraints. (solver_opt is not used here; it keeps
    // its original meaning and only affects compute_timestep().)
    PressureSolver(const SolverConfig& config, const PressureBC& bc);

    PressureSolver(const PressureSolver&) = delete;
    PressureSolver& operator=(const PressureSolver&) = delete;

    // (Re)binds the operator to the current coefficient arrays, re-assembles
    // the CSR matrices if needed and updates the preconditioner. Must be
    // called before solve() and whenever the coefficients change. The arrays
    // and the mesh must outlive subsequent solve() calls.
    void update(const Mesh& mesh,
                const std::vector<std::array<double, 3>>& gstif,
                const std::vector<double>& pdiag);

    // Solves A p = rhs with the pressure constraints. Throws std::logic_error
    // if update() has not been called.
    linalg::SolveResult solve(const std::vector<double>& rhs, std::vector<double>& p);

    // Non-owning; nullptr detaches.
    void set_monitor(linalg::IterationMonitor* monitor);

    // E.g. "PCG (CG + jacobi, ebe)", "PCG (CG + rcm+ic0, csr)" or
    // "Krylov (GMRES(30) + ilu0, csr)".
    std::string description() const;
    // Short name for per-step logs: "PCG" for CG, else the method name.
    std::string label() const;

    bool legacy() const { return legacy_; }
    const SolverConfig& config() const { return config_; }
    const linalg::KrylovSolver& krylov() const { return *krylov_; }
    const linalg::Preconditioner& preconditioner() const { return *precond_; }
    const linalg::DirichletConstraints& constraints() const { return constraints_; }

    // The operator used by the Krylov iteration (after update()).
    const linalg::LinearOperator& op() const;
    // Setup work done so far (diagnostics and tests).
    struct SetupStatistics {
        int structure_builds = 0; // sparsity pattern + scatter map (once per mesh)
        int numeric_setups = 0;   // preconditioner setups/refactorisations
        int rescaled_reuses = 0;  // updates served by rescaling the last setup
    };
    const SetupStatistics& statistics() const { return stats_; }
    // Factor applied to the last setup's M^{-1} (1 after a numeric setup).
    double preconditioner_scale() const { return scale_; }
    bool setup_reuse_enabled() const { return reuse_; }

    // Assembled matrices, if built: A and the constrained P A P + (I - P).
    const linalg::CsrMatrix* assembled_matrix() const { return assembler_ ? &csr_ : nullptr; }
    const linalg::CsrMatrix* constrained_matrix() const { return assembled_precond_ && assembler_ ? &csr_constrained_ : nullptr; }

private:
    SolverConfig config_;
    bool legacy_ = true;
    bool csr_operator_ = false;
    bool assembled_precond_ = false;

    linalg::DirichletConstraints constraints_;
    std::unique_ptr<linalg::Preconditioner> precond_;
    std::unique_ptr<linalg::KrylovSolver> krylov_;

    std::optional<linalg::EbeOperator> ebe_;
    std::unique_ptr<linalg::P1CsrAssembler> assembler_;
    const Mesh* assembled_mesh_ = nullptr;
    linalg::CsrMatrix csr_;
    linalg::CsrMatrix csr_constrained_;

    const linalg::LinearOperator* op_ = nullptr;
    bool precond_ready_ = false;

    // Setup reuse: element data of the last numeric setup.
    bool reuse_ = false;
    bool reference_valid_ = false;
    std::vector<std::array<double, 3>> gstif_ref_;
    std::vector<double> pdiag_ref_;
    double scale_ = 1.0;
    SetupStatistics stats_;
};

// c with gstif = c gstif_ref and pdiag = c pdiag_ref (every entry within
// relative tolerance `tol`), or nothing. c is taken from the sums of the
// (positive) assembled diagonals; identical data gives exactly c = 1.
std::optional<double> uniform_scaling(const std::vector<std::array<double, 3>>& gstif,
                                      const std::vector<double>& pdiag,
                                      const std::vector<std::array<double, 3>>& gstif_ref,
                                      const std::vector<double>& pdiag_ref,
                                      double tol = 1.0e-12);

// The convergence criteria PressureSolver uses for a configuration.
linalg::ConvergenceCriteria pressure_criteria(const SolverConfig& config);

} // namespace hydrama
