#pragma once

// Assembled sparse matrix in compressed sparse row (CSR) format
// (Saad 2003, Sec. 3.4; Barrett et al. 1994, Sec. 4.3.1).
//
// A CsrMatrix is a LinearOperator (y = A x) with a shared, immutable
// SparsityPattern and its own value array. The product is a plain row loop
// with index-ordered accumulation, so it is deterministic and naturally
// parallel over rows (no write conflicts, unlike the element-by-element
// product).
//
// Also provided:
//   from_triplets()       build from (i, j, a_ij) lists (duplicates summed)
//   constrain()           the Dirichlet-constrained matrix P A P + (I - P)
//   require_assembled()   capability check used by assembled preconditioners

#include "linalg/DirichletConstraints.hpp"
#include "linalg/LinearOperator.hpp"
#include "linalg/SparsityPattern.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace hydrama::linalg {

struct Triplet {
    int row = 0;
    int col = 0;
    double value = 0.0;
};

class CsrMatrix final : public LinearOperator {
public:
    CsrMatrix() = default; // empty 0 x 0 matrix

    // All-zero matrix on the given pattern.
    explicit CsrMatrix(std::shared_ptr<const SparsityPattern> pattern);

    // Throws std::invalid_argument if values.size() != pattern->nnz().
    CsrMatrix(std::shared_ptr<const SparsityPattern> pattern, std::vector<double> values);

    // rows x cols matrix from triplets; duplicate (i, j) entries are summed,
    // explicitly given zeros are stored. Throws std::invalid_argument for
    // indices out of range.
    static CsrMatrix from_triplets(int rows, int cols, const std::vector<Triplet>& triplets);

    // LinearOperator interface. size() is the number of rows; apply()
    // requires x.size() == cols() and y.size() == rows().
    std::size_t size() const override { return static_cast<std::size_t>(rows()); }
    void apply(ConstVectorView x, VectorView y) const override;
    void diagonal(VectorView d) const override; // zero where (i, i) is not stored
    OperatorTraits traits() const override { return {symmetric_}; }
    std::string_view name() const override { return "csr"; }
    const CsrMatrix* assembled() const override { return this; }

    int rows() const { return pattern_ ? pattern_->rows() : 0; }
    int cols() const { return pattern_ ? pattern_->cols() : 0; }
    std::size_t nnz() const { return values_.size(); }

    const SparsityPattern& pattern() const { return *pattern_; }
    const std::shared_ptr<const SparsityPattern>& pattern_ptr() const { return pattern_; }

    std::vector<double>& values() { return values_; }
    const std::vector<double>& values() const { return values_; }

    // a_ij, 0 if (i, j) is not stored.
    double coeff(int i, int j) const;

    void set_zero();

    // Symmetry is declared by whoever builds the matrix (e.g. the assembler
    // of a symmetric bilinear form) and reported through traits(); it is
    // not re-checked when values change. is_symmetric() checks numerically.
    void declare_symmetric(bool symmetric) { symmetric_ = symmetric; }
    bool is_symmetric(double tol = 0.0) const;

private:
    std::shared_ptr<const SparsityPattern> pattern_;
    std::vector<double> values_;
    bool symmetric_ = false;
};

// out = P A P + (I - P): rows and columns of the constrained dofs are zeroed
// and their diagonal set to 1 (same pattern as A; out may be a matrix from a
// previous call, its storage is reused). This is the matrix the Krylov
// iteration effectively works with (see ConstrainedOperator) and the one
// assembled preconditioners must be built from. Requires a square matrix
// with all constrained diagonal entries stored; throws otherwise.
void constrain(const CsrMatrix& a, const DirichletConstraints& constraints, CsrMatrix& out);

// The operator as a CSR matrix; throws std::invalid_argument naming `who`
// if it is matrix-free.
const CsrMatrix& require_assembled(const LinearOperator& a, std::string_view who);

} // namespace hydrama::linalg
