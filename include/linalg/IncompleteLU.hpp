#pragma once

// Incomplete LU factorisation with zero fill-in, ILU(0)
// (Meijerink & van der Vorst 1977; Saad 2003, Sec. 10.3.2, Algorithm 10.4
// "ILU(0)", IKJ variant).
//
//     A ~ M = L U,  L unit lower and U upper triangular, both restricted to
//                   the sparsity pattern of A (stored in place of A).
//
// Defining property: (L U)_ij = a_ij for every (i, j) in the pattern of A.
// Works for non-symmetric matrices (the intended partner of BiCGStab/GMRES);
// for a symmetric matrix it coincides in exact arithmetic with IC(0)
// (U = D L^T), so CG may also be used, but IC(0) needs half the storage and
// work.
//
// A zero (or non-finite) pivot triggers a refactorisation of
// A + alpha diag(A) (DiagonalShift, Manteuffel 1980). Requires an assembled
// square matrix with a stored, non-zero diagonal.

#include "linalg/CsrMatrix.hpp"
#include "linalg/DiagonalShift.hpp"
#include "linalg/Preconditioner.hpp"

#include <memory>
#include <vector>

namespace hydrama::linalg {

class IncompleteLU final : public Preconditioner {
public:
    explicit IncompleteLU(DiagonalShift shift = {});

    void setup(const LinearOperator& a) override;
    void update(const LinearOperator& a, OperatorChange change) override;
    void apply(ConstVectorView r, VectorView z) const override;
    std::size_t size() const override { return static_cast<std::size_t>(n_); }
    PreconditionerTraits traits() const override { return {false, false, true}; }
    std::string_view name() const override { return "ilu0"; }

    double shift() const { return shift_; }
    int attempts() const { return attempts_; }

private:
    void symbolic(const CsrMatrix& a);
    void numeric(const CsrMatrix& a);
    bool factor(double alpha);

    DiagonalShift policy_;
    int n_ = 0;
    std::shared_ptr<const SparsityPattern> pattern_;
    std::vector<double> a_;         // values of A
    std::vector<double> lu_;        // L (strictly lower) and U (upper) in A's pattern
    std::vector<double> inv_pivot_; // 1 / u_ii
    std::vector<int> where_;        // work: column -> position in the current row
    double shift_ = 0.0;
    int attempts_ = 0;
};

} // namespace hydrama::linalg
