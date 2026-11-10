#include "linalg/ScaledPreconditioner.hpp"

#include "linalg/VectorOps.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace hydrama::linalg {

ScaledPreconditioner::ScaledPreconditioner(const Preconditioner& inner, double factor)
    : inner_(inner), factor_(factor) {
    if (!(factor > 0.0) || !std::isfinite(factor)) {
        throw std::invalid_argument("ScaledPreconditioner: factor must be finite and > 0, got " +
                                    std::to_string(factor));
    }
}

void ScaledPreconditioner::setup(const LinearOperator& /*a*/) {
    throw std::logic_error("ScaledPreconditioner is a view: set up the wrapped preconditioner");
}

void ScaledPreconditioner::update(const LinearOperator& /*a*/, OperatorChange /*change*/) {
    throw std::logic_error("ScaledPreconditioner is a view: update the wrapped preconditioner");
}

void ScaledPreconditioner::apply(ConstVectorView r, VectorView z) const {
    inner_.apply(r, z);
    if (factor_ != 1.0) scale(z, factor_);
}

} // namespace hydrama::linalg
