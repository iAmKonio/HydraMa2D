#include "linalg/CsrMatrix.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace hydrama::linalg {

CsrMatrix::CsrMatrix(std::shared_ptr<const SparsityPattern> pattern)
    : pattern_(std::move(pattern)) {
    if (!pattern_) throw std::invalid_argument("CsrMatrix: null pattern");
    values_.assign(pattern_->nnz(), 0.0);
}

CsrMatrix::CsrMatrix(std::shared_ptr<const SparsityPattern> pattern, std::vector<double> values)
    : pattern_(std::move(pattern)), values_(std::move(values)) {
    if (!pattern_) throw std::invalid_argument("CsrMatrix: null pattern");
    if (values_.size() != pattern_->nnz()) {
        throw std::invalid_argument("CsrMatrix: " + std::to_string(values_.size()) + " values for a pattern with " +
                                    std::to_string(pattern_->nnz()) + " entries");
    }
}

CsrMatrix CsrMatrix::from_triplets(int rows, int cols, const std::vector<Triplet>& triplets) {
    if (rows < 0 || cols < 0) throw std::invalid_argument("CsrMatrix::from_triplets: negative dimensions");
    for (const Triplet& t : triplets) {
        if (t.row < 0 || t.row >= rows || t.col < 0 || t.col >= cols) {
            throw std::invalid_argument("CsrMatrix::from_triplets: entry (" + std::to_string(t.row) + ", " +
                                        std::to_string(t.col) + ") out of range");
        }
    }
    // Stable sort by (row, col) keeps the input order of duplicates, so
    // their sum is accumulated in input order (deterministic).
    std::vector<Triplet> sorted(triplets);
    std::stable_sort(sorted.begin(), sorted.end(), [](const Triplet& a, const Triplet& b) {
        return a.row != b.row ? a.row < b.row : a.col < b.col;
    });

    std::vector<int> row_ptr(rows + 1, 0), col_idx;
    std::vector<double> values;
    col_idx.reserve(sorted.size());
    values.reserve(sorted.size());
    for (std::size_t k = 0; k < sorted.size(); ++k) {
        const Triplet& t = sorted[k];
        if (k > 0 && t.row == sorted[k - 1].row && t.col == sorted[k - 1].col) {
            values.back() += t.value;
        } else {
            col_idx.push_back(t.col);
            values.push_back(t.value);
            ++row_ptr[t.row + 1];
        }
    }
    for (int i = 0; i < rows; ++i) row_ptr[i + 1] += row_ptr[i];

    auto pattern = std::make_shared<const SparsityPattern>(rows, cols, std::move(row_ptr), std::move(col_idx));
    return CsrMatrix(std::move(pattern), std::move(values));
}

void CsrMatrix::apply(ConstVectorView x, VectorView y) const {
    const int n = rows();
    if (x.size() != static_cast<std::size_t>(cols()) || y.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("CsrMatrix::apply: vector sizes do not match the matrix");
    }
    if (n == 0) return;
    const int* const ptr = pattern_->row_ptr().data();
    const int* const col = pattern_->col_idx().data();
    const double* const val = values_.data();
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) s += val[k] * x[col[k]];
        y[i] = s;
    }
}

void CsrMatrix::diagonal(VectorView d) const {
    if (d.size() != size()) throw std::invalid_argument("CsrMatrix::diagonal: size mismatch");
    if (!pattern_) return;
    const std::vector<int>& pos = pattern_->diagonal_positions();
    for (std::size_t i = 0; i < d.size(); ++i) {
        d[i] = (i < pos.size() && pos[i] >= 0) ? values_[pos[i]] : 0.0;
    }
}

double CsrMatrix::coeff(int i, int j) const {
    if (!pattern_) return 0.0;
    const int k = pattern_->find(i, j);
    return k >= 0 ? values_[k] : 0.0;
}

void CsrMatrix::set_zero() {
    std::fill(values_.begin(), values_.end(), 0.0);
}

bool CsrMatrix::is_symmetric(double tol) const {
    if (!pattern_) return true;
    if (!pattern_->square()) return false;
    const auto& ptr = pattern_->row_ptr();
    const auto& col = pattern_->col_idx();
    for (int i = 0; i < rows(); ++i) {
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
            const int j = col[k];
            if (j <= i) continue;
            const int kt = pattern_->find(j, i);
            const double aji = kt >= 0 ? values_[kt] : 0.0;
            if (std::abs(values_[k] - aji) > tol) return false;
        }
    }
    return true;
}

void constrain(const CsrMatrix& a, const DirichletConstraints& constraints, CsrMatrix& out) {
    if (!a.pattern_ptr() || !a.pattern().square()) {
        throw std::invalid_argument("constrain: the matrix must be square");
    }
    const SparsityPattern& p = a.pattern();
    constraints.validate(static_cast<std::size_t>(p.rows()));
    for (int d : constraints.dofs()) {
        if (p.diagonal_positions()[d] < 0) {
            throw std::invalid_argument("constrain: diagonal entry of constrained dof " + std::to_string(d) +
                                        " is not stored");
        }
    }
    if (out.pattern_ptr() != a.pattern_ptr()) out = CsrMatrix(a.pattern_ptr());

    std::vector<char> fixed(p.rows(), 0);
    for (int d : constraints.dofs()) fixed[d] = 1;

    const auto& ptr = p.row_ptr();
    const auto& col = p.col_idx();
    const std::vector<double>& av = a.values();
    std::vector<double>& ov = out.values();
    for (int i = 0; i < p.rows(); ++i) {
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) {
            const int j = col[k];
            if (fixed[i] || fixed[j]) {
                ov[k] = (i == j) ? 1.0 : 0.0;
            } else {
                ov[k] = av[k];
            }
        }
    }
    out.declare_symmetric(a.traits().symmetric);
}

const CsrMatrix& require_assembled(const LinearOperator& a, std::string_view who) {
    const CsrMatrix* m = a.assembled();
    if (!m) {
        throw std::invalid_argument(std::string(who) + " requires an assembled (CSR) operator, got '" +
                                    std::string(a.name()) + "'");
    }
    return *m;
}

} // namespace hydrama::linalg
