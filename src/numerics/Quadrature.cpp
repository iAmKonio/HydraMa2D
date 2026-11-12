#include "numerics/Quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace hydrama::quadrature {

namespace {

// P_n(x) and P_{n-1}(x) by the three-term recurrence (A&S 22.7.10).
void legendre(int n, double x, double& pn, double& pn1) {
    double p0 = 1.0; // P_0
    double p1 = x;   // P_1
    if (n == 0) {
        pn = p0;
        pn1 = 0.0;
        return;
    }
    for (int k = 1; k < n; ++k) {
        const double p2 = ((2.0 * k + 1.0) * x * p1 - k * p0) / (k + 1.0);
        p0 = p1;
        p1 = p2;
    }
    pn = p1;
    pn1 = p0;
}

} // namespace

Rule1D gauss_legendre(int n) {
    if (n < 1 || n > 64) throw std::invalid_argument("gauss_legendre: n must be in 1..64, got " + std::to_string(n));
    const double pi = std::acos(-1.0);
    Rule1D rule;
    rule.points.assign(static_cast<std::size_t>(n), 0.0);
    rule.weights.assign(static_cast<std::size_t>(n), 0.0);
    const int half = (n + 1) / 2; // roots are symmetric: compute the positive half
    for (int i = 1; i <= half; ++i) {
        double x = std::cos(pi * (i - 0.25) / (n + 0.5)); // Numerical Recipes 4.6.
        double pn = 0.0, pn1 = 0.0, dp = 0.0;
        for (int iter = 0; iter < 100; ++iter) {
            legendre(n, x, pn, pn1);
            dp = n * (x * pn - pn1) / (x * x - 1.0); // A&S 22.8.5
            const double dx = pn / dp;
            x -= dx;
            if (std::abs(dx) <= 1e-16 * std::max(1.0, std::abs(x))) break;
        }
        legendre(n, x, pn, pn1);
        dp = n * (x * pn - pn1) / (x * x - 1.0);
        const double w = 2.0 / ((1.0 - x * x) * dp * dp); // A&S 25.4.29
        // x decreases with i: place -x at the front and +x at the back.
        rule.points[static_cast<std::size_t>(i - 1)] = -x;
        rule.points[static_cast<std::size_t>(n - i)] = x;
        rule.weights[static_cast<std::size_t>(i - 1)] = w;
        rule.weights[static_cast<std::size_t>(n - i)] = w;
    }
    if (n % 2 == 1) rule.points[static_cast<std::size_t>(n / 2)] = 0.0; // exact middle node
    return rule;
}

Rule2D gauss_legendre_square(int n) {
    const Rule1D r = gauss_legendre(n);
    Rule2D rule;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            rule.points.push_back({r.points[static_cast<std::size_t>(i)], r.points[static_cast<std::size_t>(j)]});
            rule.weights.push_back(r.weights[static_cast<std::size_t>(i)] * r.weights[static_cast<std::size_t>(j)]);
        }
    }
    return rule;
}

} // namespace hydrama::quadrature
