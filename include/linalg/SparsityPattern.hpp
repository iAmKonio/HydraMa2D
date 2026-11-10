#pragma once

// Immutable compressed sparse row (CSR) structure: row pointers and sorted,
// unique column indices (Saad 2003, Sec. 3.4; Barrett et al. 1994,
// Sec. 4.3.1).
//
// The pattern is held through std::shared_ptr<const SparsityPattern> so that
// several matrices on the same graph (the assembled operator, its
// constrained copy, future mass/viscous matrices) share one structure, and
// so that preconditioners can detect "same structure, new values" by pointer
// identity and skip their symbolic setup.
//
// Indices are 0-based ints, consistent with Mesh.

#include <cstddef>
#include <vector>

namespace hydrama::linalg {

class SparsityPattern {
public:
    // Validates the structure and throws std::invalid_argument unless
    // row_ptr has rows + 1 non-decreasing entries starting at 0 and ending at
    // col_idx.size(), and the columns of every row are strictly increasing
    // and within [0, cols).
    SparsityPattern(int rows, int cols, std::vector<int> row_ptr, std::vector<int> col_idx);

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    bool square() const { return rows_ == cols_; }
    std::size_t nnz() const { return col_idx_.size(); }

    const std::vector<int>& row_ptr() const { return row_ptr_; }
    const std::vector<int>& col_idx() const { return col_idx_; }
    int row_begin(int i) const { return row_ptr_[static_cast<std::size_t>(i)]; }
    int row_end(int i) const { return row_ptr_[static_cast<std::size_t>(i) + 1]; }

    // Position of entry (i, j) in the value array, or -1 if not stored
    // (binary search within the row).
    int find(int i, int j) const;

    // diagonal_positions()[i] = position of (i, i), or -1 (first
    // min(rows, cols) rows).
    const std::vector<int>& diagonal_positions() const { return diag_pos_; }
    bool has_full_diagonal() const { return full_diagonal_; }

    // True if (i, j) stored <=> (j, i) stored.
    bool structurally_symmetric() const { return symmetric_; }

    // max |i - j| over the stored entries (0 for an empty/diagonal pattern).
    int bandwidth() const;

    // Envelope (profile) size of the lower triangle:
    // sum_i (i - min{j : (i, j) stored, j <= i}) (George & Liu 1981, Ch. 4).
    long long profile() const;

    bool operator==(const SparsityPattern& other) const;
    bool operator!=(const SparsityPattern& other) const { return !(*this == other); }

private:
    int rows_ = 0;
    int cols_ = 0;
    std::vector<int> row_ptr_;
    std::vector<int> col_idx_;
    std::vector<int> diag_pos_;
    bool full_diagonal_ = false;
    bool symmetric_ = false;
};

} // namespace hydrama::linalg
