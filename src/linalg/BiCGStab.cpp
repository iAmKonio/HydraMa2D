#include "linalg/BiCGStab.hpp"

#include "linalg/VectorOps.hpp"

#include <cmath>

namespace hydrama::linalg {

SolveResult BiCGStab::iterate(const LinearOperator& a,
                              const Preconditioner& m,
                              VectorView x,
                              VectorView r,
                              ConvergenceControl& control) {
    const std::size_t n = a.size();
    for (auto* v : {&r_shadow_, &p_, &v_, &p_hat_, &s_, &s_hat_, &t_}) ensure_size(*v, n);
    double* const rs = r_shadow_.data();
    double* const p = p_.data();
    double* const v = v_.data();
    double* const ph = p_hat_.data();
    double* const s = s_.data();
    double* const sh = s_hat_.data();
    double* const t = t_.data();
    replacements_ = 0;

    // (Re)start the recursion from the current residual: r~ = r.
    double r_norm = 0.0, rs_norm = 0.0, rho = 0.0;
    bool restart = true;
    auto start_recursion = [&] {
        copy(r, r_shadow_);
        r_norm = norm2(r);
        rs_norm = r_norm;
        rho = r_norm * r_norm; // (r~, r) with r~ = r
        restart = true;
    };
    start_recursion();

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;

    // Verifies a residual that meets the tolerance against the true residual.
    // Returns a stop reason, or none (after restarting from the true residual).
    StagnationGuard guard;
    auto verify = [&](int k) -> std::optional<StopReason> {
        const double true_norm = true_residual(x, r);
        if (control.satisfied(true_norm)) return control.check(k, true_norm);
        // False convergence: replace the residual and restart the recursion,
        // unless the true residual has stopped improving.
        ++replacements_;
        if (auto stop = control.check(k, true_norm)) return stop; // cap, non-finite
        if (guard.stalled(true_norm)) return StopReason::Stagnation;
        start_recursion();
        return std::nullopt;
    };

    for (int k = 1;; ++k) {
        // Lanczos breakdown: the shadow residual is orthogonal to r.
        if (!std::isfinite(rho)) return control.finish(StopReason::NonFinite);
        if (std::abs(rho) <= kBreakdownTolerance * rs_norm * r_norm) {
            return control.finish(r_norm == 0.0 ? StopReason::ExactSolution : StopReason::Breakdown);
        }

        // p = r + beta (p - omega v)
        if (restart) {
            copy(r, p_);
            restart = false;
        } else {
            const double beta = (rho / rho_old) * (alpha / omega);
            for (std::size_t i = 0; i < n; ++i) p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // v = A M^{-1} p, alpha = rho / (r~, v)
        m.apply(p_, p_hat_);
        project(p_hat_);
        a.apply(p_hat_, v_);
        double rv = 0.0, vv = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            rv += rs[i] * v[i];
            vv += v[i] * v[i];
        }
        if (!std::isfinite(rv)) return control.finish(StopReason::NonFinite);
        if (std::abs(rv) <= kBreakdownTolerance * rs_norm * std::sqrt(vv)) {
            return control.finish(StopReason::Breakdown);
        }
        alpha = rho / rv;

        // Half step: s = r - alpha v
        double ss = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            s[i] = r[i] - alpha * v[i];
            ss += s[i] * s[i];
        }
        const double s_norm = std::sqrt(ss);
        if (control.satisfied(s_norm)) {
            axpy(alpha, p_hat_, x);
            if (auto stop = verify(k)) return control.finish(*stop);
            continue;
        }

        // t = A M^{-1} s, omega = (t, s) / (t, t)
        m.apply(s_, s_hat_);
        project(s_hat_);
        a.apply(s_hat_, t_);
        double ts = 0.0, tt = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            ts += t[i] * s[i];
            tt += t[i] * t[i];
        }
        if (!std::isfinite(ts) || !std::isfinite(tt)) return control.finish(StopReason::NonFinite);
        if (std::abs(ts) <= kBreakdownTolerance * std::sqrt(tt) * s_norm) {
            // Stagnation: keep the half step (its residual is s), then stop.
            axpy(alpha, p_hat_, x);
            copy(s_, r);
            if (auto stop = control.check(k, s_norm)) return control.finish(*stop);
            return control.finish(StopReason::Breakdown);
        }
        omega = ts / tt;

        // x += alpha p^ + omega s^,  r = s - omega t,  next rho = (r~, r)
        double rr = 0.0, rho_next = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += alpha * ph[i] + omega * sh[i];
            r[i] = s[i] - omega * t[i];
            rr += r[i] * r[i];
            rho_next += rs[i] * r[i];
        }
        r_norm = std::sqrt(rr);
        rho_old = rho;
        rho = rho_next;

        if (control.satisfied(r_norm)) {
            if (auto stop = verify(k)) return control.finish(*stop);
            continue;
        }
        if (auto stop = control.check(k, r_norm)) return control.finish(*stop);
    }
}

} // namespace hydrama::linalg
