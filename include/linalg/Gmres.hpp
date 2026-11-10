#pragma once

// Restarted GMRES(m) (Saad & Schultz 1986; Saad 2003, Algorithms 6.9 and
// 9.5), right-preconditioned:
//
//   cycle:  r = b - A x,  beta = ||r||,  v_1 = r / beta,  g = beta e_1
//     for j = 1 .. m:
//       w = A M^{-1} v_j                                   (one A, one M^{-1})
//       Arnoldi: h_ij = (w, v_i), w -= h_ij v_i  (i <= j)  (orthogonalisation)
//       h_{j+1,j} = ||w||,  v_{j+1} = w / h_{j+1,j}
//       apply the previous Givens rotations G_1..G_{j-1} to column j of H,
//       compute G_j annihilating h_{j+1,j}, apply it to g:
//       |g_{j+1}| = ||b - A x_j||  (residual norm at no extra cost)
//     y = R^{-1} g (back substitution),  x += M^{-1} (V y)
//
// The Hessenberg matrix is kept in QR form by the progressive Givens
// rotations, so no least-squares problem is ever formed or stored beyond the
// (m+1) x m Hessenberg array and m+1 basis vectors; the residual norm of the
// current iterate is available after every inner step. With right
// preconditioning this is the norm of the unpreconditioned residual, so the
// convergence criteria have the same meaning as for CG/BiCGSTAB.
//
// Orthogonalisation:
//   ModifiedGramSchmidt        MGS (default): backward stable for GMRES
//                              (Paige, Rozloznik & Strakos 2006)
//   ClassicalGramSchmidtTwice  CGS2, "twice is enough" (Giraud et al. 2005):
//                              orthogonal to working precision, two
//                              block-like passes (better for future
//                              parallel/BLAS-2 implementations)
//
// Termination within a cycle: convergence, m steps, the iteration cap, or a
// "happy breakdown" (h_{j+1,j} <= kHappyBreakdownTolerance ||A M^{-1} v_j||:
// the Krylov space is invariant and the iterate is exact in it). At the end
// of every cycle the true residual b - A x is computed (one extra operator
// application) and reported, so the residual of the returned iterate is
// always a true residual. If the estimate repeatedly claims convergence
// while the true residual stalls above the tolerance (attainable accuracy),
// the solve ends with StopReason::Stagnation (StagnationGuard). A singular
// projected matrix (R_jj = 0, only possible for singular A M^{-1}) ends the
// solve with StopReason::Breakdown.

#include "linalg/KrylovSolver.hpp"

#include <string>
#include <vector>

namespace hydrama::linalg {

enum class Orthogonalization { ModifiedGramSchmidt, ClassicalGramSchmidtTwice };

std::string_view to_string(Orthogonalization o);

struct GmresOptions {
    int restart = 30; // m: Krylov dimension per cycle (>= 1)
    Orthogonalization orthogonalization = Orthogonalization::ModifiedGramSchmidt;
};

class Gmres final : public KrylovSolver {
public:
    static constexpr double kHappyBreakdownTolerance = 1.0e-14;

    // Throws std::invalid_argument for restart < 1.
    explicit Gmres(ConvergenceCriteria criteria = {}, GmresOptions options = {});

    std::string_view name() const override { return name_; } // "GMRES(m)"
    const GmresOptions& options() const { return options_; }

    // Restart cycles of the last solve.
    int cycles() const { return cycles_; }

protected:
    SolveResult iterate(const LinearOperator& a,
                        const Preconditioner& m,
                        VectorView x,
                        VectorView r,
                        ConvergenceControl& control) override;

private:
    double& h(std::size_t i, std::size_t j) { return hessenberg_[j * (m_ + 1) + i]; }
    double* basis(std::size_t i) { return basis_.data() + i * n_; }
    void orthogonalize(std::size_t j);
    void update_solution(std::size_t columns, const Preconditioner& m, VectorView x);

    GmresOptions options_;
    std::string name_;
    std::size_t m_ = 0;  // Krylov dimension used (min(restart, n))
    std::size_t n_ = 0;  // system size
    std::vector<double> basis_;      // (m+1) x n, v_i contiguous
    std::vector<double> hessenberg_; // (m+1) x m, column-major; R after rotations
    std::vector<double> cs_, sn_;    // Givens rotations
    std::vector<double> g_;          // rotated right-hand side beta e_1
    std::vector<double> y_;
    std::vector<double> w_;          // new Arnoldi vector
    std::vector<double> z_;          // M^{-1} v_j, later M^{-1} V y
    int cycles_ = 0;
};

} // namespace hydrama::linalg
