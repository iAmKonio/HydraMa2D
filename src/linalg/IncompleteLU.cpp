#include "linalg/IncompleteLU.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace hydrama::linalg {

IncompleteLU::IncompleteLU(DiagonalShift shift) : policy_(shift) {
    if (policy_.initial < 0.0 || policy_.minimum <= 0.0 || policy_.max_attempts < 1) {
        throw std::invalid_argument("IncompleteLU: invalid diagonal shift policy");
    }
}

void IncompleteLU::symbolic(const CsrMatrix& a) {
    if (!a.pattern_ptr() || !a.pattern().square()) throw std::invalid_argument("ilu0: square matrix required");
    if (!a.pattern().has_full_diagonal()) throw std::invalid_argument("ilu0: all diagonal entries must be stored");
    pattern_ = a.pattern_ptr();
    n_ = pattern_->rows();
    a_.resize(pattern_->nnz());
    lu_.resize(pattern_->nnz());
    inv_pivot_.resize(n_);
    where_.assign(n_, -1);
}

void IncompleteLU::numeric(const CsrMatrix& a) {
    std::copy(a.values().begin(), a.values().end(), a_.begin());
    const std::vector<int>& dpos = pattern_->diagonal_positions();
    for (int i = 0; i < n_; ++i) {
        if (a_[dpos[i]] == 0.0) throw std::invalid_argument("ilu0: zero diagonal entry in row " + std::to_string(i));
    }
    double alpha = policy_.initial;
    for (attempts_ = 1;; ++attempts_) {
        if (factor(alpha)) {
            shift_ = alpha;
            return;
        }
        if (attempts_ >= policy_.max_attempts) {
            throw std::runtime_error("ilu0: factorisation broke down for every diagonal shift up to " +
                                     std::to_string(alpha));
        }
        alpha = std::max(2.0 * alpha, policy_.minimum);
    }
}

bool IncompleteLU::factor(double alpha) {
    const int* const ptr = pattern_->row_ptr().data();
    const int* const col = pattern_->col_idx().data();
    const int* const dpos = pattern_->diagonal_positions().data();
    std::copy(a_.begin(), a_.end(), lu_.begin());
    if (alpha != 0.0) {
        for (int i = 0; i < n_; ++i) lu_[dpos[i]] *= 1.0 + alpha;
    }

    // IKJ variant (Saad 2003, Algorithm 10.4).
    for (int i = 0; i < n_; ++i) {
        for (int q = ptr[i]; q < ptr[i + 1]; ++q) where_[col[q]] = q;
        for (int p = ptr[i]; p < dpos[i]; ++p) { // k < i, increasing
            const int k = col[p];
            const double lik = lu_[p] / lu_[dpos[k]];
            lu_[p] = lik;
            for (int q = dpos[k] + 1; q < ptr[k + 1]; ++q) { // u_kj, j > k
                const int pos = where_[col[q]];
                if (pos >= 0) lu_[pos] -= lik * lu_[q];
            }
        }
        for (int q = ptr[i]; q < ptr[i + 1]; ++q) where_[col[q]] = -1;

        const double pivot = lu_[dpos[i]];
        if (!std::isfinite(pivot) || !(std::abs(pivot) > policy_.pivot_tolerance * std::abs(a_[dpos[i]]))) {
            return false;
        }
        inv_pivot_[i] = 1.0 / pivot;
    }
    return true;
}

void IncompleteLU::setup(const LinearOperator& a) {
    const CsrMatrix& m = require_assembled(a, "ilu0 preconditioner");
    symbolic(m);
    numeric(m);
}

void IncompleteLU::update(const LinearOperator& a, OperatorChange change) {
    const CsrMatrix& m = require_assembled(a, "ilu0 preconditioner");
    if (change == OperatorChange::Values && m.pattern_ptr() == pattern_) {
        numeric(m);
    } else {
        setup(a);
    }
}

void IncompleteLU::apply(ConstVectorView r, VectorView z) const {
    if (r.size() != size() || z.size() != size()) {
        throw std::invalid_argument("IncompleteLU::apply: vector size does not match the set-up operator "
                                    "(was setup() called?)");
    }
    const int* const ptr = pattern_->row_ptr().data();
    const int* const col = pattern_->col_idx().data();
    const int* const dpos = pattern_->diagonal_positions().data();
    const double* const lu = lu_.data();

    // L y = r (unit lower)
    for (int i = 0; i < n_; ++i) {
        double s = r[i];
        for (int p = ptr[i]; p < dpos[i]; ++p) s -= lu[p] * z[col[p]];
        z[i] = s;
    }
    // U z = y
    for (int i = n_ - 1; i >= 0; --i) {
        double s = z[i];
        for (int p = dpos[i] + 1; p < ptr[i + 1]; ++p) s -= lu[p] * z[col[p]];
        z[i] = s * inv_pivot_[i];
    }
}

} // namespace hydrama::linalg
