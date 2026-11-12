#pragma once

// Gauss-Legendre quadrature (docs/QUADRILATERAL_ELEMENTS.md, section 2).
//
// The n-point Gauss-Legendre rule on [-1, 1],
//
//     int_{-1}^{1} f(x) dx  ~=  sum_{i=1}^{n} w_i f(x_i),
//
// has as nodes x_i the n roots of the Legendre polynomial P_n and the weights
//
//     w_i = 2 / ((1 - x_i^2) P_n'(x_i)^2)
//
// (Abramowitz & Stegun 1964, eq. 25.4.29). It integrates every polynomial of
// degree <= 2n - 1 exactly (and no rule with n points does better).
//
// gauss_legendre(n) computes the rule for any n >= 1 by Newton's method on
// P_n, evaluated with the three-term recurrence
//     (k + 1) P_{k+1}(x) = (2k + 1) x P_k(x) - k P_{k-1}(x)   (A&S 22.7.10)
// and P_n'(x) = n (x P_n(x) - P_{n-1}(x)) / (x^2 - 1)         (A&S 22.8.5),
// starting from x_i ~ cos(pi (i - 1/4) / (n + 1/2)) (Numerical Recipes, 3rd
// ed., section 4.6). The result is accurate to a few units in the last place;
// the tests compare it with the closed forms for n <= 5 and with the
// tabulated values of A&S Table 25.4.
//
// gauss_legendre_square(n) is the tensor-product rule on [-1, 1]^2 with n^2
// points: exact for every polynomial x^a y^b with a, b <= 2n - 1.

#include <array>
#include <vector>

namespace hydrama::quadrature {

struct Rule1D {
    std::vector<double> points;  // increasing, symmetric about 0
    std::vector<double> weights; // positive, sum 2
};

struct Rule2D {
    std::vector<std::array<double, 2>> points; // (xi, eta)
    std::vector<double> weights;               // positive, sum 4
};

// n-point Gauss-Legendre rule on [-1, 1]. Throws std::invalid_argument for
// n < 1 or n > 64.
Rule1D gauss_legendre(int n);

// n x n tensor-product Gauss-Legendre rule on [-1, 1]^2. Points are ordered
// eta-major: point (i, j) = (x_i, x_j) has index j * n + i. For n = 2 this is
// (-,-), (+,-), (-,+), (+,+); the bilinear quadrilateral uses its own
// counter-clockwise ordering (QuadElement.hpp).
Rule2D gauss_legendre_square(int n);

// The 2-point rule in closed form: nodes -+1/sqrt(3), weights 1.
inline constexpr double kGauss2Point = 0.57735026918962576451; // 1 / sqrt(3)

} // namespace hydrama::quadrature
