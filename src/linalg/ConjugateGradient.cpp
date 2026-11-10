#include "linalg/ConjugateGradient.hpp"

#include "linalg/VectorOps.hpp"

#include <cmath>

namespace hydrama::linalg {

SolveResult ConjugateGradient::iterate(const LinearOperator& a,
                                       const Preconditioner& m,
                                       VectorView x,
                                       VectorView r,
                                       ConvergenceControl& control) {
    const std::size_t n = a.size();
    ensure_size(z_, n);
    ensure_size(p_, n);
    ensure_size(q_, n);
    double* const z = z_.data();
    double* const p = p_.data();
    double* const q = q_.data();

    const bool legacy_guards = control.criteria().on_breakdown == BreakdownPolicy::Continue;

    // Pointwise preconditioner (Jacobi): fuse z = s .* r into the update loop.
    const ConstVectorView scaling = m.pointwise_scaling();
    const bool fused = scaling.size() == n;
    const double* const s = scaling.data();

    // z_0 = M^{-1} r_0, p_0 = z_0, rz = r_0 . z_0
    if (fused) {
        for (std::size_t i = 0; i < n; ++i) z[i] = r[i] * s[i];
    } else {
        m.apply(r, z_);
    }
    copy(z_, p_);
    double rz = dot(r, z_);

    if (!legacy_guards && !(rz > 0.0)) {
        // r_0 != 0 here (begin() stops on r_0 == 0 except in legacy mode), so
        // r.z <= 0 means M is not positive definite.
        return control.finish(norm2(r) == 0.0 ? StopReason::ExactSolution : StopReason::Breakdown);
    }

    for (int k = 1;; ++k) {
        // q = A p
        a.apply(p_, q_);

        // alpha = (r . z) / (p . A p)
        const double pq = dot(p_, q_);
        double alpha = 0.0;
        if (legacy_guards) {
            alpha = (std::abs(pq) > kBreakdownGuard) ? (rz / pq) : 0.0;
        } else {
            if (!(pq > 0.0) || !std::isfinite(pq)) return control.finish(StopReason::Breakdown);
            alpha = rz / pq;
        }

        // x += alpha p, r -= alpha q, ||r||^2 (and z = s .* r, r . z when fused)
        double rr = 0.0;
        double rz_new = 0.0;
        if (fused) {
            for (std::size_t i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
                r[i] -= alpha * q[i];
                z[i] = r[i] * s[i];
                rr += r[i] * r[i];
                rz_new += z[i] * r[i];
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
                r[i] -= alpha * q[i];
                rr += r[i] * r[i];
            }
        }

        if (auto stop = control.check(k, std::sqrt(rr))) return control.finish(*stop);

        if (!fused) {
            m.apply(r, z_);
            rz_new = dot(z_, r);
        }

        // beta = (r_{k+1} . z_{k+1}) / (r_k . z_k)
        double beta = 0.0;
        if (legacy_guards) {
            beta = (std::abs(rz) > kBreakdownGuard) ? (rz_new / rz) : 0.0;
        } else {
            if (!(rz_new > 0.0) || !std::isfinite(rz_new)) return control.finish(StopReason::Breakdown);
            beta = rz_new / rz;
        }
        rz = rz_new;

        // p = z + beta p
        for (std::size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    }
}

} // namespace hydrama::linalg
