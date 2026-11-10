#include "linalg/SparsityPattern.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace hydrama::linalg {

SparsityPattern::SparsityPattern(int rows, int cols, std::vector<int> row_ptr, std::vector<int> col_idx)
    : rows_(rows), cols_(cols), row_ptr_(std::move(row_ptr)), col_idx_(std::move(col_idx)) {
    auto fail = [](const std::string& what) { throw std::invalid_argument("SparsityPattern: " + what); };
    if (rows_ < 0 || cols_ < 0) fail("negative dimensions");
    if (row_ptr_.size() != static_cast<std::size_t>(rows_) + 1) {
        fail("row_ptr must have rows + 1 = " + std::to_string(rows_ + 1) + " entries, has " +
             std::to_string(row_ptr_.size()));
    }
    if (row_ptr_.front() != 0) fail("row_ptr[0] must be 0");
    if (static_cast<std::size_t>(row_ptr_.back()) != col_idx_.size()) {
        fail("row_ptr[rows] = " + std::to_string(row_ptr_.back()) + " does not match " +
             std::to_string(col_idx_.size()) + " column indices");
    }
    for (int i = 0; i < rows_; ++i) {
        if (row_ptr_[i + 1] < row_ptr_[i]) fail("row_ptr decreases at row " + std::to_string(i));
        for (int k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            const int j = col_idx_[k];
            if (j < 0 || j >= cols_) {
                fail("column " + std::to_string(j) + " out of range in row " + std::to_string(i));
            }
            if (k > row_ptr_[i] && j <= col_idx_[k - 1]) {
                fail("columns of row " + std::to_string(i) + " are not strictly increasing");
            }
        }
    }

    const int ndiag = std::min(rows_, cols_);
    diag_pos_.assign(ndiag, -1);
    full_diagonal_ = true;
    for (int i = 0; i < ndiag; ++i) {
        diag_pos_[i] = find(i, i);
        if (diag_pos_[i] < 0) full_diagonal_ = false;
    }

    symmetric_ = square();
    for (int i = 0; i < rows_ && symmetric_; ++i) {
        for (int k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            if (find(col_idx_[k], i) < 0) {
                symmetric_ = false;
                break;
            }
        }
    }
}

int SparsityPattern::find(int i, int j) const {
    if (i < 0 || i >= rows_) return -1;
    const auto first = col_idx_.begin() + row_ptr_[i];
    const auto last = col_idx_.begin() + row_ptr_[i + 1];
    const auto it = std::lower_bound(first, last, j);
    return (it != last && *it == j) ? static_cast<int>(it - col_idx_.begin()) : -1;
}

int SparsityPattern::bandwidth() const {
    int b = 0;
    for (int i = 0; i < rows_; ++i) {
        if (row_ptr_[i + 1] > row_ptr_[i]) {
            // Columns are sorted: the extremes are the first and last entry.
            b = std::max(b, std::abs(i - col_idx_[row_ptr_[i]]));
            b = std::max(b, std::abs(col_idx_[row_ptr_[i + 1] - 1] - i));
        }
    }
    return b;
}

long long SparsityPattern::profile() const {
    long long p = 0;
    for (int i = 0; i < rows_; ++i) {
        if (row_ptr_[i + 1] > row_ptr_[i] && col_idx_[row_ptr_[i]] < i) p += i - col_idx_[row_ptr_[i]];
    }
    return p;
}

bool SparsityPattern::operator==(const SparsityPattern& other) const {
    return rows_ == other.rows_ && cols_ == other.cols_ && row_ptr_ == other.row_ptr_ && col_idx_ == other.col_idx_;
}

} // namespace hydrama::linalg
