#pragma once

// BiCGSTAB: biconjugate gradient stabilised method for general
// (non-symmetric) systems (van der Vorst 1992; Saad 2003, Algorithm 7.7;
// Barrett et al. 1994, Sec. 2.3.8), right-preconditioned as in the
// preconditioned version of Barrett et al.:
//
//     rho_k  = (r~, r_k)                       r~ = r_0 (shadow residual)
//     p      = r_k + beta (p - omega v),        beta = (rho_k / rho_{k-1}) (alpha / omega)
//     p^ = M^{-1} p,   v = A p^,   alpha = rho_k / (r~, v)
//     s      = r_k - alpha v                   (half step)
//     s^ = M^{-1} s,   t = A s^,   omega = (t, s) / (t, t)
//     x     += alpha p^ + omega s^,   r_{k+1} = s - omega t
//
// The tracked residual is the unpreconditioned residual b - A x, so the
// convergence criteria have the same meaning as for CG. Work per iteration:
// two operator and two preconditioner applications, four inner products;
// the vector updates are fused (4 passes).
//
// Convergence is verified with the true residual b - A x (one extra
// operator application): if the recursively updated residual meets the
// tolerance but the true residual does not, the recursion is restarted from
// the true residual with a new shadow residual (residual replacement,
// cf. van der Vorst & Ye 2000). Reported residuals at convergence are
// therefore true residuals. If the true residual stalls above the tolerance
// (the tolerance is below the attainable accuracy), the solve ends with
// StopReason::Stagnation instead of cycling until the iteration cap
// (StagnationGuard).
//
// Breakdowns (BreakdownPolicy::Stop and ::Continue alike; the zero-step
// guards of the legacy CG do not apply to BiCGSTAB):
//   |(r~, r_k)| <= tol ||r~|| ||r_k||   Lanczos (rho) breakdown      -> Breakdown
//   |(r~, v)|   <= tol ||r~|| ||v||     pivot (alpha) breakdown      -> Breakdown
//   |(t, s)|    <= tol ||t|| ||s||      stagnation (omega = 0): the
//                                       half step x + alpha p^ is
//                                       kept, then                   -> Breakdown
// with tol = kBreakdownTolerance; r_k == 0 exactly is an exact solution.

#include "linalg/KrylovSolver.hpp"

namespace hydrama::linalg {

class BiCGStab final : public KrylovSolver {
public:
    static constexpr double kBreakdownTolerance = 1.0e-14;

    using KrylovSolver::KrylovSolver;

    std::string_view name() const override { return "BiCGStab"; }

    // Number of restarts from the true residual in the last solve.
    int residual_replacements() const { return replacements_; }

protected:
    SolveResult iterate(const LinearOperator& a,
                        const Preconditioner& m,
                        VectorView x,
                        VectorView r,
                        ConvergenceControl& control) override;

private:
    std::vector<double> r_shadow_; // r~
    std::vector<double> p_;
    std::vector<double> v_;
    std::vector<double> p_hat_;
    std::vector<double> s_;
    std::vector<double> s_hat_;
    std::vector<double> t_;
    int replacements_ = 0;
};

} // namespace hydrama::linalg
