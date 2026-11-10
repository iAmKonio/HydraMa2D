#pragma once

// Jacobi (diagonal) preconditioner: z[i] = r[i] / A[i][i].
//
// Entries with |A[i][i]| <= 1e-30 are not inverted; they use a scaling of
// 1 instead (the guard of the original PCG implementation).
//
// Pointwise: exposes 1/diag(A) via pointwise_scaling() so Krylov solvers can
// fuse the preconditioner into their vector updates.

#include "linalg/Preconditioner.hpp"

#include <vector>

namespace hydrama::linalg {

class JacobiPreconditioner final : public Preconditioner {
public:
    // |A[i][i]| at or below this threshold is treated as zero.
    static constexpr double kZeroDiagonal = 1.0e-30;

    void setup(const LinearOperator& a) override;
    void apply(ConstVectorView r, VectorView z) const override;
    std::size_t size() const override { return inv_diag_.size(); }
    PreconditionerTraits traits() const override { return {true, false, true}; }
    std::string_view name() const override { return "jacobi"; }
    ConstVectorView pointwise_scaling() const override { return inv_diag_; }

    const std::vector<double>& inverse_diagonal() const { return inv_diag_; }

private:
    std::vector<double> inv_diag_;
};

} // namespace hydrama::linalg
