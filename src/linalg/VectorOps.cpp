#include "linalg/VectorOps.hpp"
#include "numerics/OpenMP.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace hydrama::linalg {

namespace {

void require_same_size(std::size_t a, std::size_t b, const char* kernel) {
    if (a != b) {
        throw std::invalid_argument(std::string("linalg::") + kernel + ": size mismatch (" +
                                    std::to_string(a) + " vs " + std::to_string(b) + ")");
    }
}

} // namespace

double dot(ConstVectorView x, ConstVectorView y) {
    require_same_size(x.size(), y.size(), "dot");
    const std::size_t n = x.size();
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

double norm2(ConstVectorView x) {
    const std::size_t n = x.size();
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += x[i] * x[i];
    return std::sqrt(s);
}

void copy(ConstVectorView x, VectorView y) {
    require_same_size(x.size(), y.size(), "copy");
    const std::size_t n = x.size();
    #pragma omp parallel for schedule(static) if(n > 2048)
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i];
}

void fill(VectorView x, double value) {
    const std::size_t n = x.size();
    #pragma omp parallel for schedule(static) if(n > 2048)
    for (std::size_t i = 0; i < n; ++i) x[i] = value;
}

void scale(VectorView x, double alpha) {
    const std::size_t n = x.size();
    #pragma omp parallel for schedule(static) if(n > 2048)
    for (std::size_t i = 0; i < n; ++i) x[i] *= alpha;
}

void axpy(double alpha, ConstVectorView x, VectorView y) {
    require_same_size(x.size(), y.size(), "axpy");
    const std::size_t n = x.size();
    #pragma omp parallel for schedule(static) if(n > 2048)
    for (std::size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

void xpby(ConstVectorView x, double beta, VectorView y) {
    require_same_size(x.size(), y.size(), "xpby");
    const std::size_t n = x.size();
    #pragma omp parallel for schedule(static) if(n > 2048)
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i] + beta * y[i];
}

void subtract(ConstVectorView x, ConstVectorView y, VectorView z) {
    require_same_size(x.size(), y.size(), "subtract");
    require_same_size(x.size(), z.size(), "subtract");
    const std::size_t n = x.size();
    #pragma omp parallel for schedule(static) if(n > 2048)
    for (std::size_t i = 0; i < n; ++i) z[i] = x[i] - y[i];
}

void pointwise_multiply(ConstVectorView x, ConstVectorView y, VectorView z) {
    require_same_size(x.size(), y.size(), "pointwise_multiply");
    require_same_size(x.size(), z.size(), "pointwise_multiply");
    const std::size_t n = x.size();
    #pragma omp parallel for schedule(static) if(n > 2048)
    for (std::size_t i = 0; i < n; ++i) z[i] = x[i] * y[i];
}

} // namespace hydrama::linalg
