#include "linalg/IncompleteCholesky.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace hydrama::linalg {

IncompleteCholesky::IncompleteCholesky(DiagonalShift shift) : policy_(shift) {
    if (policy_.initial < 0.0 || policy_.minimum <= 0.0 || policy_.max_attempts < 1) {
        throw std::invalid_argument("IncompleteCholesky: invalid diagonal shift policy");
    }
}

void IncompleteCholesky::symbolic(const CsrMatrix& a) {
    if (!a.pattern_ptr() || !a.pattern().square()) throw std::invalid_argument("ic0: square matrix required");
    const SparsityPattern& p = a.pattern();
    n_ = p.rows();
    const auto& ptr = p.row_ptr();
    const auto& col = p.col_idx();

    u_ptr_.assign(n_ + 1, 0);
    u_col_.clear();
    src_pos_.clear();
    for (int i = 0; i < n_; ++i) {
        const int d = p.diagonal_positions()[i];
        if (d < 0) throw std::invalid_argument("ic0: diagonal entry of row " + std::to_string(i) + " is not stored");
        for (int k = d; k < ptr[i + 1]; ++k) { // columns >= i, diagonal first
            u_col_.push_back(col[k]);
            src_pos_.push_back(k);
        }
        u_ptr_[i + 1] = static_cast<int>(u_col_.size());
    }
    a_upper_.resize(u_col_.size());
    u_.resize(u_col_.size());
    inv_diag_.resize(n_);
    where_.assign(n_, -1);
    source_ = a.pattern_ptr();
}

void IncompleteCholesky::numeric(const CsrMatrix& a) {
    const std::vector<double>& v = a.values();
    for (std::size_t q = 0; q < src_pos_.size(); ++q) a_upper_[q] = v[src_pos_[q]];
    for (int i = 0; i < n_; ++i) {
        const double d = a_upper_[u_ptr_[i]];
        if (!(d > 0.0)) {
            throw std::invalid_argument("ic0: non-positive diagonal entry " + std::to_string(d) + " in row " +
                                        std::to_string(i) + " (matrix not positive definite)");
        }
    }
    double alpha = policy_.initial;
    for (attempts_ = 1;; ++attempts_) {
        if (factor(alpha)) {
            shift_ = alpha;
            return;
        }
        if (attempts_ >= policy_.max_attempts) {
            throw std::runtime_error("ic0: factorisation broke down for every diagonal shift up to " +
                                     std::to_string(alpha));
        }
        alpha = std::max(2.0 * alpha, policy_.minimum);
    }
}

bool IncompleteCholesky::factor(double alpha) {
    std::copy(a_upper_.begin(), a_upper_.end(), u_.begin());
    if (alpha != 0.0) {
        for (int i = 0; i < n_; ++i) u_[u_ptr_[i]] *= 1.0 + alpha;
    }
    // Right-looking elimination, dropping every update outside triu(A).
    for (int k = 0; k < n_; ++k) {
        const int dk = u_ptr_[k];
        const int end_k = u_ptr_[k + 1];
        const double pivot = u_[dk];
        if (!std::isfinite(pivot) || !(pivot > policy_.pivot_tolerance * a_upper_[dk])) return false;
        const double ukk = std::sqrt(pivot);
        u_[dk] = ukk;
        inv_diag_[k] = 1.0 / ukk;
        for (int p = dk + 1; p < end_k; ++p) u_[p] /= ukk;

        // Row j of the trailing matrix, for every u_kj != 0 (j > k):
        //     u_ji -= u_kj u_ki   for i >= j, if (j, i) is in the pattern.
        for (int p = dk + 1; p < end_k; ++p) {
            const int j = u_col_[p];
            const double ukj = u_[p];
            for (int q = u_ptr_[j]; q < u_ptr_[j + 1]; ++q) where_[u_col_[q]] = q;
            for (int p2 = p; p2 < end_k; ++p2) {
                const int q = where_[u_col_[p2]];
                if (q >= 0) u_[q] -= ukj * u_[p2];
            }
            for (int q = u_ptr_[j]; q < u_ptr_[j + 1]; ++q) where_[u_col_[q]] = -1;
        }
    }
    return true;
}

void IncompleteCholesky::setup(const LinearOperator& a) {
    const CsrMatrix& m = require_assembled(a, "ic0 preconditioner");
    symbolic(m);
    numeric(m);
}

void IncompleteCholesky::update(const LinearOperator& a, OperatorChange change) {
    const CsrMatrix& m = require_assembled(a, "ic0 preconditioner");
    if (change == OperatorChange::Values && m.pattern_ptr() == source_) {
        numeric(m);
    } else {
        setup(a);
    }
}

void IncompleteCholesky::apply(ConstVectorView r, VectorView z) const {
    if (r.size() != size() || z.size() != size()) {
        throw std::invalid_argument("IncompleteCholesky::apply: vector size does not match the set-up operator "
                                    "(was setup() called?)");
    }
    const int* const ptr = u_ptr_.data();
    const int* const col = u_col_.data();
    const double* const u = u_.data();
    for (int i = 0; i < n_; ++i) z[i] = r[i];

    // U^T y = r (column-oriented: U stored by rows = U^T by columns)
    for (int k = 0; k < n_; ++k) {
        const double yk = z[k] * inv_diag_[k];
        z[k] = yk;
        for (int p = ptr[k] + 1; p < ptr[k + 1]; ++p) z[col[p]] -= u[p] * yk;
    }
    // U z = y
    for (int k = n_ - 1; k >= 0; --k) {
        double s = z[k];
        for (int p = ptr[k] + 1; p < ptr[k + 1]; ++p) s -= u[p] * z[col[p]];
        z[k] = s * inv_diag_[k];
    }
}

} // namespace hydrama::linalg
