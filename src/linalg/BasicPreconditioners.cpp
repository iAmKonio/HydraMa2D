// Identity and Jacobi preconditioners.

#include "linalg/IdentityPreconditioner.hpp"
#include "linalg/JacobiPreconditioner.hpp"
#include "linalg/VectorOps.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace hydrama::linalg {

namespace {

void require_setup(std::size_t n, ConstVectorView r, VectorView z, const char* who) {
    if (r.size() != n || z.size() != n) {
        throw std::invalid_argument(std::string(who) +
                                    "::apply: vector size does not match the set-up operator "
                                    "(was setup() called?)");
    }
}

} // namespace

void IdentityPreconditioner::apply(ConstVectorView r, VectorView z) const {
    require_setup(n_, r, z, "IdentityPreconditioner");
    copy(r, z);
}

void JacobiPreconditioner::setup(const LinearOperator& a) {
    const std::size_t n = a.size();
    inv_diag_.resize(n);
    a.diagonal(inv_diag_);
    for (double& d : inv_diag_) {
        d = (std::abs(d) > kZeroDiagonal) ? (1.0 / d) : 1.0;
    }
}

void JacobiPreconditioner::apply(ConstVectorView r, VectorView z) const {
    require_setup(inv_diag_.size(), r, z, "JacobiPreconditioner");
    pointwise_multiply(r, inv_diag_, z);
}

} // namespace hydrama::linalg
