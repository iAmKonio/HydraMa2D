#include "linalg/SsorPreconditioner.hpp"

#include <stdexcept>
#include <string>

namespace hydrama::linalg {

SsorPreconditioner::SsorPreconditioner(double omega) : omega_(omega) {
    if (!(omega > 0.0 && omega < 2.0)) {
        throw std::invalid_argument("SsorPreconditioner: omega must satisfy 0 < omega < 2, got " +
                                    std::to_string(omega));
    }
}

void SsorPreconditioner::setup(const LinearOperator& a) {
    const CsrMatrix& m = require_assembled(a, "ssor preconditioner");
    if (!m.pattern_ptr() || !m.pattern().square()) throw std::invalid_argument("ssor: square matrix required");
    if (!m.pattern().has_full_diagonal()) throw std::invalid_argument("ssor: all diagonal entries must be stored");
    pattern_ = m.pattern_ptr();
    n_ = m.rows();
    load_values(m);
}

void SsorPreconditioner::update(const LinearOperator& a, OperatorChange change) {
    const CsrMatrix& m = require_assembled(a, "ssor preconditioner");
    if (change == OperatorChange::Values && m.pattern_ptr() == pattern_) {
        load_values(m); // same structure: no symbolic work
    } else {
        setup(a);
    }
}

void SsorPreconditioner::load_values(const CsrMatrix& m) {
    const std::vector<int>& dpos = pattern_->diagonal_positions();
    const std::vector<double>& v = m.values();
    off_.resize(v.size());
    for (std::size_t k = 0; k < v.size(); ++k) off_[k] = omega_ * v[k];
    diag_.resize(n_);
    inv_diag_.resize(n_);
    for (int i = 0; i < n_; ++i) {
        const double d = v[dpos[i]];
        if (d == 0.0) throw std::invalid_argument("ssor: zero diagonal entry in row " + std::to_string(i));
        diag_[i] = d;
        inv_diag_[i] = 1.0 / d;
    }
    symmetric_ = m.traits().symmetric;
}

void SsorPreconditioner::apply(ConstVectorView r, VectorView z) const {
    if (r.size() != size() || z.size() != size()) {
        throw std::invalid_argument("SsorPreconditioner::apply: vector size does not match the set-up operator "
                                    "(was setup() called?)");
    }
    const int* const ptr = pattern_->row_ptr().data();
    const int* const col = pattern_->col_idx().data();
    const int* const dpos = pattern_->diagonal_positions().data();
    const double* const off = off_.data();
    const double scale = omega_ * (2.0 - omega_);

    // Forward sweep: (D + omega L) y = scale * r   (y stored in z)
    for (int i = 0; i < n_; ++i) {
        double s = scale * r[i];
        for (int k = ptr[i]; k < dpos[i]; ++k) s -= off[k] * z[col[k]];
        z[i] = s * inv_diag_[i];
    }
    // Backward sweep: (D + omega U) z = D y
    for (int i = n_ - 1; i >= 0; --i) {
        double s = diag_[i] * z[i];
        for (int k = dpos[i] + 1; k < ptr[i + 1]; ++k) s -= off[k] * z[col[k]];
        z[i] = s * inv_diag_[i];
    }
}

} // namespace hydrama::linalg
