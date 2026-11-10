#pragma once

// Symmetric successive over-relaxation (SSOR) preconditioner
// (Evans 1968; Axelsson 1994, Ch. 7; Saad 2003, Sec. 4.1.2 and 10.2).
//
// With A = L + D + U (strictly lower, diagonal, strictly upper part):
//
//     M = 1 / (omega (2 - omega)) (D + omega L) D^{-1} (D + omega U),
//
// 0 < omega < 2. omega = 1 is symmetric Gauss-Seidel. For symmetric
// positive definite A, M is symmetric positive definite, so it can be used
// with CG. Applying M^{-1} is one forward and one backward sweep over the
// matrix (no extra storage beyond a copy of A's values; no fill-in):
//
//     (D + omega L) y = omega (2 - omega) r
//     (D + omega U) z = D y
//
// Requires an assembled (CSR) square matrix with a non-zero diagonal.

#include "linalg/CsrMatrix.hpp"
#include "linalg/Preconditioner.hpp"

#include <memory>
#include <vector>

namespace hydrama::linalg {

class SsorPreconditioner final : public Preconditioner {
public:
    // Throws std::invalid_argument unless 0 < omega < 2.
    explicit SsorPreconditioner(double omega = 1.0);

    void setup(const LinearOperator& a) override;
    void update(const LinearOperator& a, OperatorChange change) override;
    void apply(ConstVectorView r, VectorView z) const override;
    std::size_t size() const override { return static_cast<std::size_t>(n_); }
    PreconditionerTraits traits() const override { return {symmetric_, false, true}; }
    std::string_view name() const override { return "ssor"; }

    double omega() const { return omega_; }

private:
    void load_values(const CsrMatrix& a);

    double omega_;
    int n_ = 0;
    bool symmetric_ = true;
    std::shared_ptr<const SparsityPattern> pattern_;
    std::vector<double> off_;      // omega * a_ij off the diagonal (diagonal slots unused)
    std::vector<double> diag_;     // a_ii
    std::vector<double> inv_diag_; // 1 / a_ii
};

} // namespace hydrama::linalg
