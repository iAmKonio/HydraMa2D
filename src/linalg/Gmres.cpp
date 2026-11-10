#include "linalg/Gmres.hpp"

#include "linalg/VectorOps.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace hydrama::linalg {

std::string_view to_string(Orthogonalization o) {
    switch (o) {
        case Orthogonalization::ModifiedGramSchmidt: return "mgs";
        case Orthogonalization::ClassicalGramSchmidtTwice: return "cgs2";
    }
    return "unknown";
}

Gmres::Gmres(ConvergenceCriteria criteria, GmresOptions options)
    : KrylovSolver(criteria), options_(options) {
    if (options_.restart < 1) {
        throw std::invalid_argument("Gmres: restart must be >= 1, got " + std::to_string(options_.restart));
    }
    name_ = "GMRES(" + std::to_string(options_.restart) + ")";
}

// w_ := w_ - sum_{i <= j} h_ij v_i with h_ij stored in column j of H.
void Gmres::orthogonalize(std::size_t j) {
    double* const w = w_.data();
    const std::size_t n = n_;
    if (options_.orthogonalization == Orthogonalization::ModifiedGramSchmidt) {
        for (std::size_t i = 0; i <= j; ++i) {
            const double* const vi = basis(i);
            double hij = 0.0;
            for (std::size_t l = 0; l < n; ++l) hij += w[l] * vi[l];
            for (std::size_t l = 0; l < n; ++l) w[l] -= hij * vi[l];
            h(i, j) = hij;
        }
        return;
    }
    // Classical Gram-Schmidt, two passes: all coefficients of a pass use the
    // same w, the second pass removes what rounding left behind.
    for (std::size_t i = 0; i <= j; ++i) h(i, j) = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t i = 0; i <= j; ++i) {
            const double* const vi = basis(i);
            double c = 0.0;
            for (std::size_t l = 0; l < n; ++l) c += w[l] * vi[l];
            y_[i] = c;
        }
        for (std::size_t i = 0; i <= j; ++i) {
            const double* const vi = basis(i);
            const double c = y_[i];
            for (std::size_t l = 0; l < n; ++l) w[l] -= c * vi[l];
            h(i, j) += c;
        }
    }
}

// y = R^{-1} g (upper triangular, first `columns` columns), x += M^{-1} V y.
// Uses w_ and z_ as work space; the preconditioner is applied once per cycle.
void Gmres::update_solution(std::size_t columns, const Preconditioner& m, VectorView x) {
    for (std::size_t i = columns; i-- > 0;) {
        double s = g_[i];
        for (std::size_t l = i + 1; l < columns; ++l) s -= h(i, l) * y_[l];
        y_[i] = s / h(i, i);
    }
    fill(w_, 0.0);
    for (std::size_t i = 0; i < columns; ++i) axpy(y_[i], ConstVectorView(basis(i), n_), w_);
    m.apply(w_, z_);
    project(z_);
    axpy(1.0, z_, x);
}

SolveResult Gmres::iterate(const LinearOperator& a,
                           const Preconditioner& m,
                           VectorView x,
                           VectorView r,
                           ConvergenceControl& control) {
    n_ = a.size();
    m_ = std::min(static_cast<std::size_t>(options_.restart), std::max<std::size_t>(n_, 1));
    ensure_size(basis_, (m_ + 1) * n_);
    ensure_size(hessenberg_, (m_ + 1) * m_);
    ensure_size(cs_, m_);
    ensure_size(sn_, m_);
    ensure_size(g_, m_ + 1);
    ensure_size(y_, m_ + 1);
    ensure_size(w_, n_);
    ensure_size(z_, n_);
    cycles_ = 0;

    const int max_iterations = control.criteria().max_iterations;
    StagnationGuard guard;
    int k = 0; // total inner iterations
    for (;;) {
        ++cycles_;
        const double beta = norm2(r);
        if (beta == 0.0) return control.finish(StopReason::ExactSolution);
        if (!std::isfinite(beta)) return control.finish(StopReason::NonFinite);

        double* const v0 = basis(0);
        for (std::size_t l = 0; l < n_; ++l) v0[l] = r[l] / beta;
        std::fill(g_.begin(), g_.end(), 0.0);
        g_[0] = beta;

        std::size_t columns = 0;
        bool singular = false;
        bool claimed = false; // the residual estimate met the tolerance
        for (std::size_t j = 0; j < m_; ++j) {
            // w = A M^{-1} v_j
            m.apply(ConstVectorView(basis(j), n_), z_);
            project(z_);
            a.apply(z_, w_);
            const double w_norm = norm2(w_);
            orthogonalize(j);
            const double h_next = norm2(w_);
            if (!std::isfinite(w_norm) || !std::isfinite(h_next)) {
                if (columns > 0) update_solution(columns, m, x);
                return control.finish(StopReason::NonFinite);
            }
            const bool happy = !(h_next > kHappyBreakdownTolerance * w_norm);
            if (!happy) {
                double* const vnext = basis(j + 1);
                for (std::size_t l = 0; l < n_; ++l) vnext[l] = w_[l] / h_next;
            }

            // Previous rotations on column j, then the rotation G_j that
            // annihilates h_{j+1,j}.
            for (std::size_t i = 0; i < j; ++i) {
                const double hij = h(i, j), hi1j = h(i + 1, j);
                h(i, j) = cs_[i] * hij + sn_[i] * hi1j;
                h(i + 1, j) = -sn_[i] * hij + cs_[i] * hi1j;
            }
            const double diag = h(j, j);
            const double sub = happy ? 0.0 : h_next;
            const double rho = std::hypot(diag, sub);
            if (rho == 0.0) { // R_jj = 0: A M^{-1} is singular on the Krylov space
                singular = true;
                break;
            }
            cs_[j] = diag / rho;
            sn_[j] = sub / rho;
            h(j, j) = rho;
            h(j + 1, j) = 0.0;
            g_[j + 1] = -sn_[j] * g_[j];
            g_[j] = cs_[j] * g_[j];

            ++k;
            columns = j + 1;
            const double estimate = std::abs(g_[j + 1]);
            claimed = control.satisfied(estimate);
            const bool last = happy || claimed || columns == m_ || k >= max_iterations || !std::isfinite(estimate);
            if (last) break;
            if (auto stop = control.check(k, estimate)) { // not expected: `last` covers all stops
                update_solution(columns, m, x);
                return control.finish(*stop);
            }
        }

        if (columns > 0) update_solution(columns, m, x);
        if (singular && columns == 0) return control.finish(StopReason::Breakdown);
        if (singular) {
            // The iterate minimises the residual over the columns before the
            // singular one; its residual norm was the last one reported.
            true_residual(x, r);
            return control.finish(StopReason::Breakdown);
        }

        // End of cycle: report the true residual of the updated iterate.
        const double true_norm = true_residual(x, r);
        if (auto stop = control.check(k, true_norm)) return control.finish(*stop);
        // Not converged: restart. If the estimate claimed convergence but the
        // true residual keeps missing the tolerance, the attainable accuracy
        // is reached.
        if (claimed && guard.stalled(true_norm)) return control.finish(StopReason::Stagnation);
    }
}

} // namespace hydrama::linalg
