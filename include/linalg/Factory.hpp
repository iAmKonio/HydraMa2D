#pragma once

// Name-based creation of Krylov solvers and preconditioners (Factory with an
// explicit registry).
//
// The catalogues below are the single source of truth for the names that
// may appear in configuration files: IO::read_solver validates against them
// and PressureSolver creates through them. Adding a method = implementing
// the class + one catalogue entry in Factory.cpp. (The registry is an
// explicit table rather than self-registering static objects, which a
// static library would silently drop at link time.)

#include "linalg/Convergence.hpp"
#include "linalg/KrylovSolver.hpp"
#include "linalg/Preconditioner.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace hydrama::linalg {

struct PreconditionerOptions {
    double ssor_omega = 1.0;     // SSOR relaxation, 0 < omega < 2
    double diagonal_shift = 0.0; // initial shift alpha_0 of ic0 / ilu0
    std::string ordering = "none"; // "none" | "rcm" (assembled preconditioners only)
};

struct PreconditionerInfo {
    std::string_view name;
    std::string_view description;
    bool requires_assembled; // needs a CsrMatrix (setup() throws otherwise)
    bool symmetric;          // M^{-1} symmetric for symmetric A (CG-compatible)
};

struct KrylovOptions {
    int gmres_restart = 30;                       // GMRES(m): m >= 1
    std::string gmres_orthogonalization = "mgs";  // "mgs" | "cgs2"
};

struct KrylovInfo {
    std::string_view name;
    std::string_view description;
    bool requires_symmetric; // operator and preconditioner must be symmetric
};

const std::vector<PreconditionerInfo>& preconditioner_catalog();
const std::vector<KrylovInfo>& krylov_catalog();

// nullptr for unknown names.
const PreconditionerInfo* find_preconditioner(std::string_view name);
const KrylovInfo* find_krylov_solver(std::string_view name);

// Throws std::invalid_argument for unknown names, invalid options, or an
// ordering requested for a preconditioner that does not use an assembled
// matrix.
std::unique_ptr<Preconditioner> make_preconditioner(std::string_view name,
                                                    const PreconditionerOptions& options = {});
std::unique_ptr<KrylovSolver> make_krylov_solver(std::string_view name,
                                                 const ConvergenceCriteria& criteria = {},
                                                 const KrylovOptions& options = {});

// True for the GMRES orthogonalisation names "mgs" and "cgs2".
bool is_orthogonalization(std::string_view name);

// "a | b | c" lists for messages.
std::string preconditioner_names();
std::string krylov_solver_names();

} // namespace hydrama::linalg
