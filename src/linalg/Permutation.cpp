#include "linalg/Permutation.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace hydrama::linalg {

// ---------------------------------------------------------------------------
// Permutation
// ---------------------------------------------------------------------------

Permutation::Permutation(std::vector<int> new_to_old) : new_to_old_(std::move(new_to_old)) {
    const int n = static_cast<int>(new_to_old_.size());
    old_to_new_.assign(n, -1);
    for (int k = 0; k < n; ++k) {
        const int i = new_to_old_[k];
        if (i < 0 || i >= n) {
            throw std::invalid_argument("Permutation: index " + std::to_string(i) + " out of range 0.." +
                                        std::to_string(n - 1));
        }
        if (old_to_new_[i] != -1) {
            throw std::invalid_argument("Permutation: index " + std::to_string(i) + " appears twice");
        }
        old_to_new_[i] = k;
    }
}

Permutation Permutation::identity(int n) {
    std::vector<int> p(n);
    std::iota(p.begin(), p.end(), 0);
    return Permutation(std::move(p));
}

bool Permutation::is_identity() const {
    for (int k = 0; k < size(); ++k)
        if (new_to_old_[k] != k) return false;
    return true;
}

Permutation Permutation::inverse() const {
    return Permutation(old_to_new_);
}

void Permutation::gather(ConstVectorView x_old, VectorView x_new) const {
    if (x_old.size() != new_to_old_.size() || x_new.size() != new_to_old_.size()) {
        throw std::invalid_argument("Permutation::gather: size mismatch");
    }
    for (std::size_t k = 0; k < new_to_old_.size(); ++k) x_new[k] = x_old[new_to_old_[k]];
}

void Permutation::scatter(ConstVectorView x_new, VectorView x_old) const {
    if (x_old.size() != new_to_old_.size() || x_new.size() != new_to_old_.size()) {
        throw std::invalid_argument("Permutation::scatter: size mismatch");
    }
    for (std::size_t k = 0; k < new_to_old_.size(); ++k) x_old[new_to_old_[k]] = x_new[k];
}

// ---------------------------------------------------------------------------
// SymmetricPermutation
// ---------------------------------------------------------------------------

SymmetricPermutation::SymmetricPermutation(std::shared_ptr<const SparsityPattern> pattern, Permutation permutation)
    : source_(std::move(pattern)), permutation_(std::move(permutation)) {
    if (!source_ || !source_->square()) throw std::invalid_argument("SymmetricPermutation: square pattern required");
    const int n = source_->rows();
    if (permutation_.size() != n) {
        throw std::invalid_argument("SymmetricPermutation: permutation of size " +
                                    std::to_string(permutation_.size()) + " for a pattern of size " +
                                    std::to_string(n));
    }
    const auto& ptr = source_->row_ptr();
    const auto& col = source_->col_idx();
    const auto& n2o = permutation_.new_to_old();
    const auto& o2n = permutation_.old_to_new();

    std::vector<int> row_ptr(n + 1, 0), col_idx(source_->nnz());
    value_map_.resize(source_->nnz());
    std::vector<std::pair<int, int>> row; // (new column, source position)
    for (int k = 0; k < n; ++k) {
        const int i = n2o[k];
        row.clear();
        for (int q = ptr[i]; q < ptr[i + 1]; ++q) row.emplace_back(o2n[col[q]], q);
        std::sort(row.begin(), row.end());
        int out = row_ptr[k];
        for (const auto& [c, q] : row) {
            col_idx[out] = c;
            value_map_[out] = q;
            ++out;
        }
        row_ptr[k + 1] = out;
    }
    permuted_ = std::make_shared<const SparsityPattern>(n, n, std::move(row_ptr), std::move(col_idx));
}

bool SymmetricPermutation::matches(const std::shared_ptr<const SparsityPattern>& pattern) const {
    return pattern == source_ || (pattern && *pattern == *source_);
}

void SymmetricPermutation::apply(const CsrMatrix& a, CsrMatrix& b) const {
    if (!matches(a.pattern_ptr())) {
        throw std::invalid_argument("SymmetricPermutation::apply: matrix pattern does not match the plan");
    }
    if (b.pattern_ptr() != permuted_) b = CsrMatrix(permuted_);
    const std::vector<double>& av = a.values();
    std::vector<double>& bv = b.values();
    for (std::size_t k = 0; k < value_map_.size(); ++k) bv[k] = av[value_map_[k]];
    b.declare_symmetric(a.traits().symmetric);
}

CsrMatrix SymmetricPermutation::apply(const CsrMatrix& a) const {
    CsrMatrix b;
    apply(a, b);
    return b;
}

} // namespace hydrama::linalg
