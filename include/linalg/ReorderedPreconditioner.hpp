#pragma once

// Decorator that builds an assembled preconditioner in a different dof
// numbering (e.g. reverse Cuthill-McKee):
//
//     M^{-1} = P^T  M_inner(P A P^T)^{-1}  P
//
// The Krylov iteration stays in the original numbering; only the
// preconditioner sees the reordered matrix. For incomplete factorisations
// and Gauss-Seidel-type sweeps the ordering changes the preconditioner
// itself; a bandwidth/profile reducing ordering typically improves its
// quality and the memory locality of its sweeps (Duff & Meurant 1989;
// Benzi, Szyld & van Duin 1999; Saad 2003, Sec. 3.3 and 10.3).
//
// The ordering is computed once per sparsity pattern; update() with
// OperatorChange::Values only permutes the new values (O(nnz)) and updates
// the inner preconditioner numerically.

#include "linalg/CsrMatrix.hpp"
#include "linalg/Permutation.hpp"
#include "linalg/Preconditioner.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hydrama::linalg {

using OrderingFunction = std::function<Permutation(const SparsityPattern&)>;

class ReorderedPreconditioner final : public Preconditioner {
public:
    // Throws std::invalid_argument for a null inner preconditioner or
    // ordering function.
    ReorderedPreconditioner(std::unique_ptr<Preconditioner> inner,
                            OrderingFunction ordering,
                            std::string ordering_name);

    // Reverse Cuthill-McKee ordering.
    static std::unique_ptr<ReorderedPreconditioner> rcm(std::unique_ptr<Preconditioner> inner);

    void setup(const LinearOperator& a) override;
    void update(const LinearOperator& a, OperatorChange change) override;
    void apply(ConstVectorView r, VectorView z) const override;
    std::size_t size() const override { return plan_ ? static_cast<std::size_t>(plan_->permutation().size()) : 0; }
    PreconditionerTraits traits() const override { return inner_->traits(); }
    std::string_view name() const override { return name_; } // e.g. "rcm+ic0"

    const Preconditioner& inner() const { return *inner_; }
    // Number of times the ordering was computed (once per sparsity pattern).
    int orderings_computed() const { return orderings_; }
    const Permutation& permutation() const;
    const CsrMatrix& permuted_matrix() const { return permuted_; }

private:
    std::unique_ptr<Preconditioner> inner_;
    OrderingFunction ordering_;
    std::string name_;
    std::optional<SymmetricPermutation> plan_;
    CsrMatrix permuted_;
    mutable std::vector<double> r_perm_; // work vectors of apply()
    mutable std::vector<double> z_perm_;
    int orderings_ = 0;
};

} // namespace hydrama::linalg
