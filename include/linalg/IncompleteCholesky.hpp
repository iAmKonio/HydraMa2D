#pragma once

// Incomplete Cholesky factorisation with zero fill-in, IC(0)
// (Meijerink & van der Vorst 1977; Saad 2003, Sec. 10.3; Golub & Van Loan
// 2013, Ch. 11).
//
//     A ~ M = U^T U,   U upper triangular with the sparsity of triu(A).
//
// The factor is computed row by row with a right-looking (KIJ) elimination
// in which every update outside the pattern of triu(A) is dropped; the
// defining property is (U^T U)_ij = a_ij for every (i, j) in the pattern.
// Only the upper triangle of A is read (A is assumed symmetric).
//
// For symmetric positive definite M-matrices IC(0) exists (Meijerink & van
// der Vorst 1977). For general SPD matrices it can break down with a
// non-positive pivot (Kershaw 1978); the factorisation is then repeated for
// A + alpha diag(A) with increasing alpha (DiagonalShift; Manteuffel 1980,
// Lin & More 1999). shift() reports the alpha actually used.
//
// Applying M^{-1}: forward substitution with U^T, backward with U.
// Setup on a matrix with the same pattern object as before (update() with
// OperatorChange::Values) reuses the symbolic structure.

#include "linalg/CsrMatrix.hpp"
#include "linalg/DiagonalShift.hpp"
#include "linalg/Preconditioner.hpp"

#include <memory>
#include <vector>

namespace hydrama::linalg {

class IncompleteCholesky final : public Preconditioner {
public:
    explicit IncompleteCholesky(DiagonalShift shift = {});

    // Throws std::invalid_argument for a non-assembled or non-square
    // operator, a missing or non-positive diagonal entry, and
    // std::runtime_error if no shift up to max_attempts succeeds.
    void setup(const LinearOperator& a) override;
    void update(const LinearOperator& a, OperatorChange change) override;
    void apply(ConstVectorView r, VectorView z) const override;
    std::size_t size() const override { return static_cast<std::size_t>(n_); }
    PreconditionerTraits traits() const override { return {true, false, true}; }
    std::string_view name() const override { return "ic0"; }

    double shift() const { return shift_; } // alpha of the last factorisation
    int attempts() const { return attempts_; }
    std::size_t factor_nnz() const { return u_.size(); }

private:
    void symbolic(const CsrMatrix& a);
    void numeric(const CsrMatrix& a);
    bool factor(double alpha);

    DiagonalShift policy_;
    int n_ = 0;
    std::shared_ptr<const SparsityPattern> source_;
    std::vector<int> u_ptr_;       // rows of U (diagonal first in each row)
    std::vector<int> u_col_;
    std::vector<int> src_pos_;     // position of U entry q in A's value array
    std::vector<double> a_upper_;  // triu(A) values
    std::vector<double> u_;        // factor values
    std::vector<double> inv_diag_; // 1 / u_kk
    std::vector<int> where_;       // work: column -> position in a row of U
    double shift_ = 0.0;
    int attempts_ = 0;
};

} // namespace hydrama::linalg
