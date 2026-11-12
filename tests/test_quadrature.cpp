// Gauss-Legendre quadrature (numerics/Quadrature.hpp): nodes and weights
// against closed forms and the tables of Abramowitz & Stegun (1964, Table
// 25.4), symmetry and positivity, exactness for polynomials of degree
// <= 2n - 1 (and not beyond), the tensor-product rule on the square.

#include "numerics/Quadrature.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numeric>
#include <stdexcept>

using namespace hydrama::quadrature;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// int_{-1}^{1} x^k dx
double monomial_integral(int k) { return k % 2 == 1 ? 0.0 : 2.0 / (k + 1); }

double apply(const Rule1D& r, int k) {
    double s = 0.0;
    for (std::size_t i = 0; i < r.points.size(); ++i) s += r.weights[i] * std::pow(r.points[i], k);
    return s;
}

} // namespace

TEST_CASE("Gauss-Legendre: closed forms for n = 1..5", "[quadrature]") {
    const double tol = 4e-16;
    const Rule1D r1 = gauss_legendre(1);
    CHECK(r1.points == std::vector<double>{0.0});
    CHECK(r1.weights == std::vector<double>{2.0});

    const Rule1D r2 = gauss_legendre(2);
    CHECK_THAT(r2.points[1], WithinAbs(1.0 / std::sqrt(3.0), tol));
    CHECK_THAT(r2.points[1], WithinAbs(kGauss2Point, tol));
    CHECK_THAT(r2.weights[0], WithinAbs(1.0, tol));

    const Rule1D r3 = gauss_legendre(3);
    CHECK(r3.points[1] == 0.0);
    CHECK_THAT(r3.points[2], WithinAbs(std::sqrt(3.0 / 5.0), tol));
    CHECK_THAT(r3.weights[1], WithinAbs(8.0 / 9.0, tol));
    CHECK_THAT(r3.weights[2], WithinAbs(5.0 / 9.0, tol));

    const Rule1D r4 = gauss_legendre(4); // +-sqrt(3/7 -+ 2/7 sqrt(6/5)), (18 +- sqrt 30) / 36
    CHECK_THAT(r4.points[2], WithinAbs(std::sqrt(3.0 / 7.0 - 2.0 / 7.0 * std::sqrt(6.0 / 5.0)), tol));
    CHECK_THAT(r4.points[3], WithinAbs(std::sqrt(3.0 / 7.0 + 2.0 / 7.0 * std::sqrt(6.0 / 5.0)), tol));
    CHECK_THAT(r4.weights[2], WithinAbs((18.0 + std::sqrt(30.0)) / 36.0, tol));
    CHECK_THAT(r4.weights[3], WithinAbs((18.0 - std::sqrt(30.0)) / 36.0, tol));

    const Rule1D r5 = gauss_legendre(5); // 0, (1/3) sqrt(5 -+ 2 sqrt(10/7))
    CHECK(r5.points[2] == 0.0);
    CHECK_THAT(r5.weights[2], WithinAbs(128.0 / 225.0, tol));
    CHECK_THAT(r5.points[3], WithinAbs(std::sqrt(5.0 - 2.0 * std::sqrt(10.0 / 7.0)) / 3.0, tol));
    CHECK_THAT(r5.points[4], WithinAbs(std::sqrt(5.0 + 2.0 * std::sqrt(10.0 / 7.0)) / 3.0, tol));
    CHECK_THAT(r5.weights[3], WithinAbs((322.0 + 13.0 * std::sqrt(70.0)) / 900.0, tol));
    CHECK_THAT(r5.weights[4], WithinAbs((322.0 - 13.0 * std::sqrt(70.0)) / 900.0, tol));
}

TEST_CASE("Gauss-Legendre: Abramowitz & Stegun Table 25.4", "[quadrature]") {
    // Positive abscissas and weights, 15 significant digits.
    struct Row {
        int n;
        double x, w;
    };
    const Row table[] = {
        {2, 0.577350269189626, 1.000000000000000},
        {3, 0.000000000000000, 0.888888888888889}, {3, 0.774596669241483, 0.555555555555556},
        {4, 0.339981043584856, 0.652145154862546}, {4, 0.861136311594053, 0.347854845137454},
        {5, 0.000000000000000, 0.568888888888889}, {5, 0.538469310105683, 0.478628670499366},
        {5, 0.906179845938664, 0.236926885056189},
        {6, 0.238619186083197, 0.467913934572691}, {6, 0.661209386466265, 0.360761573048139},
        {6, 0.932469514203152, 0.171324492379170},
        {8, 0.183434642495650, 0.362683783378362}, {8, 0.525532409916329, 0.313706645877887},
        {8, 0.796666477413627, 0.222381034453374}, {8, 0.960289856497536, 0.101228536290376},
    };
    for (const Row& row : table) {
        CAPTURE(row.n, row.x);
        const Rule1D r = gauss_legendre(row.n);
        bool found = false;
        for (std::size_t i = 0; i < r.points.size(); ++i) {
            if (std::abs(r.points[i] - row.x) < 1e-14) {
                found = true;
                CHECK_THAT(r.weights[i], WithinAbs(row.w, 1e-15));
            }
        }
        CHECK(found);
    }
}

TEST_CASE("Gauss-Legendre: symmetric, increasing, positive weights summing to 2", "[quadrature]") {
    for (int n = 1; n <= 64; ++n) {
        CAPTURE(n);
        const Rule1D r = gauss_legendre(n);
        REQUIRE(r.points.size() == static_cast<std::size_t>(n));
        REQUIRE(r.weights.size() == static_cast<std::size_t>(n));
        CHECK(r.points.front() > -1.0);
        CHECK(r.points.back() < 1.0);
        for (int i = 0; i < n; ++i) {
            const std::size_t k = static_cast<std::size_t>(i), m = static_cast<std::size_t>(n - 1 - i);
            if (i > 0) CHECK(r.points[k] > r.points[k - 1]);
            CHECK(r.points[k] == -r.points[m]);
            CHECK(r.weights[k] == r.weights[m]);
            CHECK(r.weights[k] > 0.0);
        }
        CHECK_THAT(std::accumulate(r.weights.begin(), r.weights.end(), 0.0), WithinAbs(2.0, 1e-13));
    }
}

TEST_CASE("Gauss-Legendre: exact up to degree 2n - 1, not for degree 2n", "[quadrature]") {
    for (int n = 1; n <= 20; ++n) {
        CAPTURE(n);
        const Rule1D r = gauss_legendre(n);
        for (int k = 0; k <= 2 * n - 1; ++k) {
            CAPTURE(k);
            CHECK_THAT(apply(r, k), WithinAbs(monomial_integral(k), 2e-14));
        }
    }
    // x^{2n} is not integrated exactly: the Gauss error term
    //   int f - Q(f) = 2^{2n+1} (n!)^4 / ((2n+1) ((2n)!)^3) f^{(2n)}(xi)
    // with f^{(2n)} = (2n)! gives 2^{2n+1} (n!)^4 / ((2n+1) ((2n)!)^2) > 0.
    for (int n = 1; n <= 8; ++n) {
        CAPTURE(n);
        double fact_n = 1.0, fact_2n = 1.0;
        for (int i = 2; i <= n; ++i) fact_n *= i;
        for (int i = 2; i <= 2 * n; ++i) fact_2n *= i;
        const double expected = std::pow(2.0, 2 * n + 1) * std::pow(fact_n, 4) / ((2 * n + 1) * fact_2n * fact_2n);
        const double error = monomial_integral(2 * n) - apply(gauss_legendre(n), 2 * n);
        CHECK(error > 0.0);
        CHECK_THAT(error, WithinRel(expected, 1e-9));
    }
}

TEST_CASE("Gauss-Legendre: integrates smooth functions with spectral accuracy", "[quadrature]") {
    // int_{-1}^{1} exp(x) dx = e - 1/e.
    const double exact = std::exp(1.0) - std::exp(-1.0);
    double previous = 1.0;
    for (int n = 1; n <= 8; ++n) {
        const Rule1D r = gauss_legendre(n);
        double s = 0.0;
        for (std::size_t i = 0; i < r.points.size(); ++i) s += r.weights[i] * std::exp(r.points[i]);
        const double err = std::abs(s - exact);
        CAPTURE(n, err);
        if (n > 1 && previous > 1e-14) CHECK(err < 0.1 * previous);
        previous = err;
    }
    CHECK(previous < 1e-14);
}

TEST_CASE("Gauss-Legendre: tensor rule on the square", "[quadrature]") {
    for (int n = 1; n <= 5; ++n) {
        CAPTURE(n);
        const Rule2D r = gauss_legendre_square(n);
        REQUIRE(r.points.size() == static_cast<std::size_t>(n * n));
        CHECK_THAT(std::accumulate(r.weights.begin(), r.weights.end(), 0.0), WithinAbs(4.0, 1e-13));
        for (int a = 0; a <= 2 * n - 1; ++a) {
            for (int b = 0; b <= 2 * n - 1; ++b) {
                double s = 0.0;
                for (std::size_t i = 0; i < r.points.size(); ++i)
                    s += r.weights[i] * std::pow(r.points[i][0], a) * std::pow(r.points[i][1], b);
                CAPTURE(a, b);
                CHECK_THAT(s, WithinAbs(monomial_integral(a) * monomial_integral(b), 1e-13));
            }
        }
    }
    const Rule2D r2 = gauss_legendre_square(2); // eta-major order: (-,-), (+,-), (-,+), (+,+)
    const double sx[4] = {-1, 1, -1, 1}, sy[4] = {-1, -1, 1, 1};
    for (int i = 0; i < 4; ++i) {
        CHECK_THAT(r2.points[static_cast<std::size_t>(i)][0], WithinAbs(sx[i] * kGauss2Point, 1e-16));
        CHECK_THAT(r2.points[static_cast<std::size_t>(i)][1], WithinAbs(sy[i] * kGauss2Point, 1e-16));
        CHECK_THAT(r2.weights[static_cast<std::size_t>(i)], WithinAbs(1.0, 1e-15));
    }
}

TEST_CASE("Gauss-Legendre: invalid point counts", "[quadrature]") {
    CHECK_THROWS_AS(gauss_legendre(0), std::invalid_argument);
    CHECK_THROWS_AS(gauss_legendre(-3), std::invalid_argument);
    CHECK_THROWS_AS(gauss_legendre(65), std::invalid_argument);
}
