#include "linalg/KrylovSolver.hpp"

#include "linalg/ConstrainedOperator.hpp"
#include "linalg/VectorOps.hpp"

#include <stdexcept>
#include <string>

namespace hydrama::linalg {

KrylovSolver::KrylovSolver(ConvergenceCriteria criteria) : criteria_(criteria) {}

void KrylovSolver::ensure_size(std::vector<double>& v, std::size_t n) {
    if (v.size() != n) v.resize(n);
}

SolveResult KrylovSolver::solve(const LinearOperator& a,
                                const Preconditioner& m,
                                ConstVectorView b,
                                VectorView x,
                                const DirichletConstraints& constraints) {
    const std::size_t n = a.size();
    // Error messages are only built on the error path (no allocation otherwise).
    auto fail = [&](const std::string& what) {
        throw std::invalid_argument(std::string(name()) + "::solve: " + what);
    };
    if (b.size() != n) {
        fail("rhs has " + std::to_string(b.size()) + " entries, operator " + std::to_string(n));
    }
    if (x.size() != n) {
        fail("solution has " + std::to_string(x.size()) + " entries, operator " + std::to_string(n));
    }
    if (m.size() != n) {
        fail("preconditioner '" + std::string(m.name()) + "' is set up for size " + std::to_string(m.size()) +
             ", operator has " + std::to_string(n) + " (was setup() called?)");
    }
    if (n > 0 && b.data() == x.data()) fail("rhs and solution must not alias");
    constraints.validate(n);

    // x_0 with the prescribed values, r_0 = P (b - A x_0).
    constraints.impose(x);
    ensure_size(r_, n);
    a.apply(x, r_);
    for (std::size_t i = 0; i < n; ++i) r_[i] = b[i] - r_[i];
    constraints.zero(r_);

    double rhs_norm = 0.0;
    if (criteria_.reference == ResidualReference::RightHandSide) {
        ensure_size(tmp_, n);
        copy(b, tmp_);
        constraints.zero(tmp_);
        rhs_norm = norm2(tmp_);
    }

    ConvergenceControl control(criteria_, monitor_, name());
    if (auto stop = control.begin(norm2(r_), rhs_norm)) return control.finish(*stop);

    a_ = &a;
    b_ = b;
    constraints_ = &constraints;
    const ConstrainedOperator constrained(a, constraints);
    const SolveResult result = iterate(constrained, m, x, r_, control);
    a_ = nullptr;
    b_ = {};
    constraints_ = nullptr;
    return result;
}

double KrylovSolver::true_residual(ConstVectorView x, VectorView r) const {
    a_->apply(x, r);
    const std::size_t n = r.size();
    for (std::size_t i = 0; i < n; ++i) r[i] = b_[i] - r[i];
    constraints_->zero(r);
    return norm2(r);
}

void KrylovSolver::project(VectorView v) const {
    constraints_->zero(v);
}

} // namespace hydrama::linalg
