#include "linalg/Factory.hpp"

#include "linalg/BiCGStab.hpp"
#include "linalg/ConjugateGradient.hpp"
#include "linalg/Gmres.hpp"
#include "linalg/IdentityPreconditioner.hpp"
#include "linalg/IncompleteCholesky.hpp"
#include "linalg/IncompleteLU.hpp"
#include "linalg/JacobiPreconditioner.hpp"
#include "linalg/ReorderedPreconditioner.hpp"
#include "linalg/Reordering.hpp"
#include "linalg/SsorPreconditioner.hpp"

#include <algorithm>
#include <stdexcept>

namespace hydrama::linalg {

namespace {

using PreconditionerCreator = std::unique_ptr<Preconditioner> (*)(const PreconditionerOptions&);
using KrylovCreator = std::unique_ptr<KrylovSolver> (*)(const ConvergenceCriteria&, const KrylovOptions&);

struct PreconditionerEntry {
    PreconditionerInfo info;
    PreconditionerCreator create;
};

struct KrylovEntry {
    KrylovInfo info;
    KrylovCreator create;
};

DiagonalShift shift_from(const PreconditionerOptions& o) {
    if (!(o.diagonal_shift >= 0.0)) throw std::invalid_argument("diagonal_shift must be >= 0");
    DiagonalShift s;
    s.initial = o.diagonal_shift;
    return s;
}

// ---- The registries: one entry per method ----------------------------------

const std::vector<PreconditionerEntry>& preconditioner_entries() {
    static const std::vector<PreconditionerEntry> entries = {
        {{"none", "no preconditioning (M = I)", false, true},
         [](const PreconditionerOptions&) -> std::unique_ptr<Preconditioner> {
             return std::make_unique<IdentityPreconditioner>();
         }},
        {{"jacobi", "diagonal scaling M = diag(A)", false, true},
         [](const PreconditionerOptions&) -> std::unique_ptr<Preconditioner> {
             return std::make_unique<JacobiPreconditioner>();
         }},
        {{"ssor", "symmetric successive over-relaxation (omega = ssor_omega)", true, true},
         [](const PreconditionerOptions& o) -> std::unique_ptr<Preconditioner> {
             return std::make_unique<SsorPreconditioner>(o.ssor_omega);
         }},
        {{"ic0", "incomplete Cholesky, zero fill-in", true, true},
         [](const PreconditionerOptions& o) -> std::unique_ptr<Preconditioner> {
             return std::make_unique<IncompleteCholesky>(shift_from(o));
         }},
        {{"ilu0", "incomplete LU, zero fill-in", true, false},
         [](const PreconditionerOptions& o) -> std::unique_ptr<Preconditioner> {
             return std::make_unique<IncompleteLU>(shift_from(o));
         }},
    };
    return entries;
}

const std::vector<KrylovEntry>& krylov_entries() {
    static const std::vector<KrylovEntry> entries = {
        {{"cg", "preconditioned conjugate gradients (SPD systems)", true},
         [](const ConvergenceCriteria& c, const KrylovOptions&) -> std::unique_ptr<KrylovSolver> {
             return std::make_unique<ConjugateGradient>(c);
         }},
        {{"bicgstab", "BiCGSTAB, right-preconditioned (general systems)", false},
         [](const ConvergenceCriteria& c, const KrylovOptions&) -> std::unique_ptr<KrylovSolver> {
             return std::make_unique<BiCGStab>(c);
         }},
        {{"gmres", "restarted GMRES(m), right-preconditioned (general systems)", false},
         [](const ConvergenceCriteria& c, const KrylovOptions& o) -> std::unique_ptr<KrylovSolver> {
             GmresOptions g;
             g.restart = o.gmres_restart;
             if (o.gmres_orthogonalization == "mgs") {
                 g.orthogonalization = Orthogonalization::ModifiedGramSchmidt;
             } else if (o.gmres_orthogonalization == "cgs2") {
                 g.orthogonalization = Orthogonalization::ClassicalGramSchmidtTwice;
             } else {
                 throw std::invalid_argument("unknown GMRES orthogonalization '" + o.gmres_orthogonalization +
                                             "' (mgs | cgs2)");
             }
             return std::make_unique<Gmres>(c, g);
         }},
    };
    return entries;
}

template <class Entries>
std::string join_names(const Entries& entries) {
    std::string s;
    for (const auto& e : entries) {
        if (!s.empty()) s += " | ";
        s += e.info.name;
    }
    return s;
}

} // namespace

const std::vector<PreconditionerInfo>& preconditioner_catalog() {
    static const std::vector<PreconditionerInfo> infos = [] {
        std::vector<PreconditionerInfo> v;
        for (const auto& e : preconditioner_entries()) v.push_back(e.info);
        return v;
    }();
    return infos;
}

const std::vector<KrylovInfo>& krylov_catalog() {
    static const std::vector<KrylovInfo> infos = [] {
        std::vector<KrylovInfo> v;
        for (const auto& e : krylov_entries()) v.push_back(e.info);
        return v;
    }();
    return infos;
}

const PreconditionerInfo* find_preconditioner(std::string_view name) {
    for (const auto& e : preconditioner_entries())
        if (e.info.name == name) return &e.info;
    return nullptr;
}

const KrylovInfo* find_krylov_solver(std::string_view name) {
    for (const auto& e : krylov_entries())
        if (e.info.name == name) return &e.info;
    return nullptr;
}

std::unique_ptr<Preconditioner> make_preconditioner(std::string_view name, const PreconditionerOptions& options) {
    const auto& entries = preconditioner_entries();
    const auto it = std::find_if(entries.begin(), entries.end(), [&](const auto& e) { return e.info.name == name; });
    if (it == entries.end()) {
        throw std::invalid_argument("unknown preconditioner '" + std::string(name) + "' (available: " +
                                    preconditioner_names() + ")");
    }
    if (!is_ordering(options.ordering)) {
        throw std::invalid_argument("unknown ordering '" + options.ordering + "'");
    }
    std::unique_ptr<Preconditioner> m = it->create(options);
    if (options.ordering != "none") {
        if (!it->info.requires_assembled) {
            throw std::invalid_argument("ordering '" + options.ordering + "' requires an assembled preconditioner, '" +
                                        std::string(name) + "' is not");
        }
        const std::string ordering = options.ordering;
        m = std::make_unique<ReorderedPreconditioner>(
            std::move(m), [ordering](const SparsityPattern& p) { return compute_ordering(ordering, p); }, ordering);
    }
    return m;
}

std::unique_ptr<KrylovSolver> make_krylov_solver(std::string_view name, const ConvergenceCriteria& criteria,
                                                 const KrylovOptions& options) {
    const auto& entries = krylov_entries();
    const auto it = std::find_if(entries.begin(), entries.end(), [&](const auto& e) { return e.info.name == name; });
    if (it == entries.end()) {
        throw std::invalid_argument("unknown Krylov solver '" + std::string(name) + "' (available: " +
                                    krylov_solver_names() + ")");
    }
    return it->create(criteria, options);
}

bool is_orthogonalization(std::string_view name) {
    return name == "mgs" || name == "cgs2";
}

std::string preconditioner_names() {
    return join_names(preconditioner_entries());
}

std::string krylov_solver_names() {
    return join_names(krylov_entries());
}

} // namespace hydrama::linalg
