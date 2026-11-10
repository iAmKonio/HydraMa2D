#include "linalg/ReorderedPreconditioner.hpp"

#include "linalg/Reordering.hpp"

#include <stdexcept>
#include <utility>

namespace hydrama::linalg {

ReorderedPreconditioner::ReorderedPreconditioner(std::unique_ptr<Preconditioner> inner,
                                                 OrderingFunction ordering,
                                                 std::string ordering_name)
    : inner_(std::move(inner)), ordering_(std::move(ordering)) {
    if (!inner_) throw std::invalid_argument("ReorderedPreconditioner: null inner preconditioner");
    if (!ordering_) throw std::invalid_argument("ReorderedPreconditioner: null ordering function");
    name_ = ordering_name + "+" + std::string(inner_->name());
}

std::unique_ptr<ReorderedPreconditioner> ReorderedPreconditioner::rcm(std::unique_ptr<Preconditioner> inner) {
    return std::make_unique<ReorderedPreconditioner>(
        std::move(inner), [](const SparsityPattern& p) { return reverse_cuthill_mckee(p); }, "rcm");
}

void ReorderedPreconditioner::setup(const LinearOperator& a) {
    const CsrMatrix& m = require_assembled(a, name_ + " preconditioner");
    if (!m.pattern_ptr() || !m.pattern().square()) throw std::invalid_argument(name_ + ": square matrix required");
    plan_.emplace(m.pattern_ptr(), ordering_(m.pattern()));
    ++orderings_;
    plan_->apply(m, permuted_);
    inner_->setup(permuted_);
    r_perm_.assign(m.rows(), 0.0);
    z_perm_.assign(m.rows(), 0.0);
}

void ReorderedPreconditioner::update(const LinearOperator& a, OperatorChange change) {
    const CsrMatrix& m = require_assembled(a, name_ + " preconditioner");
    if (change == OperatorChange::Values && plan_ && plan_->matches(m.pattern_ptr())) {
        plan_->apply(m, permuted_);
        inner_->update(permuted_, OperatorChange::Values);
    } else {
        setup(a);
    }
}

void ReorderedPreconditioner::apply(ConstVectorView r, VectorView z) const {
    if (!plan_) throw std::invalid_argument(name_ + "::apply: setup() has not been called");
    const Permutation& p = plan_->permutation();
    p.gather(r, r_perm_);
    inner_->apply(r_perm_, z_perm_);
    p.scatter(z_perm_, z);
}

const Permutation& ReorderedPreconditioner::permutation() const {
    if (!plan_) throw std::logic_error("ReorderedPreconditioner: setup() has not been called");
    return plan_->permutation();
}

} // namespace hydrama::linalg
