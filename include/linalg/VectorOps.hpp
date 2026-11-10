#pragma once

// Vector kernels of the linear algebra layer.
//
// All O(n) vector work of the iterative solvers goes through this file (or
// through solver-specific fused loops), so that this is the single place to
// add threading/SIMD later.
//
// Determinism: reductions (dot, norm2) accumulate sequentially in index
// order. Results are therefore reproducible bit for bit and identical to the
// plain loops of the original PCG implementation. Any future parallel
// version must keep a fixed, thread-count independent reduction order.
//
// Every kernel checks that its arguments have matching sizes and throws
// std::invalid_argument otherwise (one check per call, not per element).

#include "linalg/VectorView.hpp"

namespace hydrama::linalg {

// sum_i x[i] * y[i]
double dot(ConstVectorView x, ConstVectorView y);

// sqrt(sum_i x[i]^2)
double norm2(ConstVectorView x);

// y = x
void copy(ConstVectorView x, VectorView y);

// x = value
void fill(VectorView x, double value);

// x *= alpha
void scale(VectorView x, double alpha);

// y += alpha * x
void axpy(double alpha, ConstVectorView x, VectorView y);

// y = x + beta * y
void xpby(ConstVectorView x, double beta, VectorView y);

// z = x - y   (z may alias x or y)
void subtract(ConstVectorView x, ConstVectorView y, VectorView z);

// z[i] = x[i] * y[i]   (z may alias x or y)
void pointwise_multiply(ConstVectorView x, ConstVectorView y, VectorView z);

} // namespace hydrama::linalg
