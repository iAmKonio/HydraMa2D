// Bilinear quadrilateral (numerics/QuadElement.hpp): shape functions and
// their derivatives, the isoparametric map and its metric derivatives
// (Jacobian, determinant, chain rule), the corner criterion for valid
// elements, and the element integrals (area, lumped mass, mean gradients,
// Laplacian) with 2 x 2 Gauss-Legendre.

#include "numerics/QuadElement.hpp"
#include "numerics/Quadrature.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

using namespace hydrama;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

struct Quad {
    double x[4], y[4];
};

// A random convex quadrilateral: the unit square with corners moved by up to
// 0.2 (which keeps it convex), then a random rotation, stretch, shear and shift.
Quad random_quad(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const double sx[4] = {0, 1, 1, 0}, sy[4] = {0, 0, 1, 1};
    const double th = 3.0 * u(rng), a = 1.0 + 0.5 * u(rng), b = 0.2 + 0.15 * (1.0 + u(rng)), sh = 0.5 * u(rng);
    const double tx = 10.0 * u(rng), ty = 10.0 * u(rng);
    Quad q;
    for (int i = 0; i < 4; ++i) {
        const double px = sx[i] + 0.2 * u(rng), py = sy[i] + 0.2 * u(rng);
        const double qx = a * px + sh * py, qy = b * py;                // stretch + shear (det a b > 0)
        q.x[i] = tx + std::cos(th) * qx - std::sin(th) * qy;             // rotation
        q.y[i] = ty + std::sin(th) * qx + std::cos(th) * qy;
    }
    return q;
}

double shoelace(const Quad& q) {
    double a = 0.0;
    for (int i = 0; i < 4; ++i) a += 0.5 * (q.x[i] * q.y[(i + 1) % 4] - q.x[(i + 1) % 4] * q.y[i]);
    return a;
}

// int f(x, y) N_a dOmega with an n x n Gauss rule (reference for the 2 x 2 data).
template <typename F>
double integrate(const Quad& q, int n, F f) {
    const quadrature::Rule2D r = quadrature::gauss_legendre_square(n);
    double s = 0.0;
    for (std::size_t g = 0; g < r.points.size(); ++g) {
        const q1::Jacobian J = q1::jacobian(q.x, q.y, r.points[g][0], r.points[g][1]);
        s += r.weights[g] * J.det * f(r.points[g][0], r.points[g][1], J);
    }
    return s;
}

const double kNodes[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};

} // namespace

// =============================================================================
// Shape functions
// =============================================================================

TEST_CASE("Q1 shape functions: Kronecker property at the nodes", "[quad][shape]") {
    for (int b = 0; b < 4; ++b) {
        double N[4];
        q1::shape_functions(kNodes[b][0], kNodes[b][1], N);
        for (int a = 0; a < 4; ++a) CHECK(N[a] == (a == b ? 1.0 : 0.0));
        CHECK(q1::kNodeXi[b] == kNodes[b][0]);
        CHECK(q1::kNodeEta[b] == kNodes[b][1]);
    }
    double N[4];
    q1::shape_functions(0.0, 0.0, N);
    for (double v : N) CHECK(v == 0.25);
}

TEST_CASE("Q1 shape functions: partition of unity and reference completeness", "[quad][shape]") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int t = 0; t < 200; ++t) {
        const double xi = u(rng), eta = u(rng);
        double N[4], dxi[4], deta[4];
        q1::shape_functions(xi, eta, N);
        q1::shape_derivatives(xi, eta, dxi, deta);
        double s = 0.0, sxi = 0.0, seta = 0.0, sxe = 0.0, d1 = 0.0, d2 = 0.0;
        for (int a = 0; a < 4; ++a) {
            s += N[a];
            sxi += N[a] * q1::kNodeXi[a];
            seta += N[a] * q1::kNodeEta[a];
            sxe += N[a] * q1::kNodeXi[a] * q1::kNodeEta[a];
            d1 += dxi[a];
            d2 += deta[a];
            CHECK(N[a] >= 0.0); // inside the element
        }
        CHECK_THAT(s, WithinAbs(1.0, 1e-15));
        CHECK_THAT(sxi, WithinAbs(xi, 1e-15));         // 1, xi, eta, xi*eta are reproduced
        CHECK_THAT(seta, WithinAbs(eta, 1e-15));
        CHECK_THAT(sxe, WithinAbs(xi * eta, 1e-15));
        CHECK_THAT(d1, WithinAbs(0.0, 1e-15));
        CHECK_THAT(d2, WithinAbs(0.0, 1e-15));
    }
}

TEST_CASE("Q1 shape functions: derivatives agree with finite differences", "[quad][shape]") {
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(-0.99, 0.99);
    const double h = 1e-6;
    for (int t = 0; t < 100; ++t) {
        const double xi = u(rng), eta = u(rng);
        double dxi[4], deta[4], Np[4], Nm[4];
        q1::shape_derivatives(xi, eta, dxi, deta);
        q1::shape_functions(xi + h, eta, Np);
        q1::shape_functions(xi - h, eta, Nm);
        for (int a = 0; a < 4; ++a) CHECK_THAT(dxi[a], WithinAbs((Np[a] - Nm[a]) / (2 * h), 1e-9));
        q1::shape_functions(xi, eta + h, Np);
        q1::shape_functions(xi, eta - h, Nm);
        for (int a = 0; a < 4; ++a) CHECK_THAT(deta[a], WithinAbs((Np[a] - Nm[a]) / (2 * h), 1e-9));
        // Bilinear: d/dxi is linear in eta only, d/deta linear in xi only.
        double dxi2[4], deta2[4];
        q1::shape_derivatives(-xi, eta, dxi2, deta2);
        for (int a = 0; a < 4; ++a) CHECK(dxi2[a] == dxi[a]);
    }
}

// =============================================================================
// Metric derivatives
// =============================================================================

TEST_CASE("Q1 metrics: rectangles and affine maps have constant Jacobians", "[quad][metric]") {
    SECTION("rectangle") {
        const double a = 2.5, b = 0.4;
        const Quad q{{1.0, 1.0 + a, 1.0 + a, 1.0}, {-2.0, -2.0, -2.0 + b, -2.0 + b}};
        for (double xi : {-1.0, -0.3, 0.0, 0.7, 1.0}) {
            for (double eta : {-1.0, 0.5, 1.0}) {
                const q1::Jacobian J = q1::jacobian(q.x, q.y, xi, eta);
                CHECK_THAT(J.x_xi, WithinAbs(a / 2, 1e-15));
                CHECK_THAT(J.y_eta, WithinAbs(b / 2, 1e-15));
                CHECK_THAT(J.x_eta, WithinAbs(0.0, 1e-15));
                CHECK_THAT(J.y_xi, WithinAbs(0.0, 1e-15));
                CHECK_THAT(J.det, WithinRel(a * b / 4, 1e-15));
            }
        }
    }
    SECTION("parallelogram: x = c + A (xi, eta)") {
        const double A[2][2] = {{1.3, 0.4}, {-0.2, 0.9}}, c[2] = {5.0, -1.0};
        Quad q;
        for (int k = 0; k < 4; ++k) {
            q.x[k] = c[0] + A[0][0] * kNodes[k][0] + A[0][1] * kNodes[k][1];
            q.y[k] = c[1] + A[1][0] * kNodes[k][0] + A[1][1] * kNodes[k][1];
        }
        std::mt19937 rng(3);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (int t = 0; t < 50; ++t) {
            const double xi = u(rng), eta = u(rng);
            const q1::Jacobian J = q1::jacobian(q.x, q.y, xi, eta);
            CHECK_THAT(J.x_xi, WithinAbs(A[0][0], 1e-15));
            CHECK_THAT(J.x_eta, WithinAbs(A[0][1], 1e-15));
            CHECK_THAT(J.y_xi, WithinAbs(A[1][0], 1e-15));
            CHECK_THAT(J.y_eta, WithinAbs(A[1][1], 1e-15));
            CHECK_THAT(J.det, WithinAbs(A[0][0] * A[1][1] - A[0][1] * A[1][0], 1e-14));
            double px, py;
            q1::map(q.x, q.y, xi, eta, px, py);
            CHECK_THAT(px, WithinAbs(c[0] + A[0][0] * xi + A[0][1] * eta, 1e-14));
            CHECK_THAT(py, WithinAbs(c[1] + A[1][0] * xi + A[1][1] * eta, 1e-14));
        }
    }
}

TEST_CASE("Q1 metrics: the Jacobian is the derivative of the map", "[quad][metric]") {
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> u(-0.9, 0.9);
    const double h = 1e-6;
    for (int t = 0; t < 50; ++t) {
        const Quad q = random_quad(rng);
        const double xi = u(rng), eta = u(rng);
        const q1::Jacobian J = q1::jacobian(q.x, q.y, xi, eta);
        double xp, yp, xm, ym;
        q1::map(q.x, q.y, xi + h, eta, xp, yp);
        q1::map(q.x, q.y, xi - h, eta, xm, ym);
        CHECK_THAT(J.x_xi, WithinAbs((xp - xm) / (2 * h), 1e-7));
        CHECK_THAT(J.y_xi, WithinAbs((yp - ym) / (2 * h), 1e-7));
        q1::map(q.x, q.y, xi, eta + h, xp, yp);
        q1::map(q.x, q.y, xi, eta - h, xm, ym);
        CHECK_THAT(J.x_eta, WithinAbs((xp - xm) / (2 * h), 1e-7));
        CHECK_THAT(J.y_eta, WithinAbs((yp - ym) / (2 * h), 1e-7));
        CHECK_THAT(J.det, WithinAbs(J.x_xi * J.y_eta - J.x_eta * J.y_xi, 1e-14));
        // The nodes are mapped to themselves.
        for (int k = 0; k < 4; ++k) {
            q1::map(q.x, q.y, kNodes[k][0], kNodes[k][1], xp, yp);
            CHECK_THAT(xp, WithinAbs(q.x[k], 1e-13));
            CHECK_THAT(yp, WithinAbs(q.y[k], 1e-13));
        }
    }
}

TEST_CASE("Q1 metrics: det J is affine in (xi, eta) and fixed by its corner values", "[quad][metric]") {
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int t = 0; t < 100; ++t) {
        const Quad q = random_quad(rng);
        const std::array<double, 4> corner = q1::corner_jacobians(q.x, q.y);
        for (int k = 0; k < 4; ++k) {
            // det J at a corner = cross(x_{k+1} - x_k, x_{k-1} - x_k) / 4.
            CHECK_THAT(q1::jacobian(q.x, q.y, kNodes[k][0], kNodes[k][1]).det, WithinRel(corner[static_cast<std::size_t>(k)], 1e-12));
            CHECK(corner[static_cast<std::size_t>(k)] > 0.0);
        }
        // Affine: det J = bilinear interpolation of the corner values, which
        // has no xi*eta term (corner[0] - corner[1] + corner[2] - corner[3] = 0).
        CHECK_THAT(corner[0] - corner[1] + corner[2] - corner[3], WithinAbs(0.0, 1e-12 * std::abs(corner[0])));
        for (int s = 0; s < 10; ++s) {
            const double xi = u(rng), eta = u(rng);
            double N[4];
            q1::shape_functions(xi, eta, N);
            double interp = 0.0;
            for (int k = 0; k < 4; ++k) interp += N[k] * corner[static_cast<std::size_t>(k)];
            CHECK_THAT(q1::jacobian(q.x, q.y, xi, eta).det, WithinRel(interp, 1e-12));
        }
        // Area = int det J = 4 det J(0, 0) (affine) = the polygon area.
        CHECK_THAT(4.0 * q1::jacobian(q.x, q.y, 0.0, 0.0).det, WithinRel(shoelace(q), 1e-13));
    }
}

TEST_CASE("Q1 metrics: physical derivatives by the chain rule", "[quad][metric]") {
    std::mt19937 rng(13);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int t = 0; t < 100; ++t) {
        const Quad q = random_quad(rng);
        const double xi = u(rng), eta = u(rng);
        const q1::Jacobian J = q1::jacobian(q.x, q.y, xi, eta);
        double dxi[4], deta[4], dx[4], dy[4];
        q1::shape_derivatives(xi, eta, dxi, deta);
        q1::physical_derivatives(J, dxi, deta, dx, dy);
        const double scale = std::sqrt(std::abs(J.det));
        for (int a = 0; a < 4; ++a) {
            // dN/dxi = x_xi dN/dx + y_xi dN/dy, dN/deta = x_eta dN/dx + y_eta dN/dy.
            CHECK_THAT(J.x_xi * dx[a] + J.y_xi * dy[a], WithinAbs(dxi[a], 1e-13));
            CHECK_THAT(J.x_eta * dx[a] + J.y_eta * dy[a], WithinAbs(deta[a], 1e-13));
        }
        // Linear completeness in physical coordinates on distorted elements:
        // sum_a grad N_a f(x_a) = grad f for f = c + p x + r y.
        const double c = 0.7, p = -1.9, r = 3.1;
        double gx = 0.0, gy = 0.0, s1 = 0.0, s2 = 0.0;
        for (int a = 0; a < 4; ++a) {
            const double f = c + p * q.x[a] + r * q.y[a];
            gx += dx[a] * f;
            gy += dy[a] * f;
            s1 += dx[a];
            s2 += dy[a];
        }
        CHECK_THAT(gx, WithinAbs(p, 1e-12 * (1.0 + 10.0 / scale)));
        CHECK_THAT(gy, WithinAbs(r, 1e-12 * (1.0 + 10.0 / scale)));
        CHECK_THAT(s1, WithinAbs(0.0, 1e-12 / scale));
        CHECK_THAT(s2, WithinAbs(0.0, 1e-12 / scale));
    }
}

TEST_CASE("Q1 metrics: the corner criterion for valid elements", "[quad][metric]") {
    auto corners = [](Quad q) { return q1::corner_jacobians(q.x, q.y); };
    const Quad square{{0, 1, 1, 0}, {0, 0, 1, 1}};
    for (double c : corners(square)) CHECK(c == 0.25);

    const Quad clockwise{{0, 0, 1, 1}, {0, 1, 1, 0}};
    for (double c : corners(clockwise)) CHECK(c < 0.0);
    CHECK_THROWS_AS(q1::compute_geometry(clockwise.x, clockwise.y), std::domain_error);

    // Reflex corner at node 2 (corner values 1, 0.45, -0.1, 0.45); det J is
    // positive at all four Gauss points, so only the corner test detects it.
    const Quad arrowhead{{0, 2, 0.9, 0}, {0, 0, 0.9, 2}};
    const auto ca = corners(arrowhead);
    CHECK(ca[2] < 0.0);
    CHECK(ca[0] > 0.0);
    CHECK(ca[1] > 0.0);
    CHECK(ca[3] > 0.0);
    CHECK_NOTHROW(q1::compute_geometry(arrowhead.x, arrowhead.y));

    const Quad bowtie{{0, 1, 0, 1}, {0, 0, 1, 1}}; // self-intersecting
    int positive = 0;
    for (double c : corners(bowtie)) positive += c > 0.0 ? 1 : 0;
    CHECK(positive == 2); // (+, +, -, -)

    const Quad degenerate{{0, 1, 2, 0}, {0, 0, 0, 1}}; // nodes 0, 1, 2 collinear: 180 degree corner at node 1
    CHECK(corners(degenerate)[1] == 0.0);

    const Quad collapsed{{0, 1, 1, 1}, {0, 0, 1, 1}}; // nodes 2 and 3 coincide
    CHECK(corners(collapsed)[2] == 0.0);
    CHECK(corners(collapsed)[3] == 0.0);
}

// =============================================================================
// Element integrals with 2 x 2 Gauss-Legendre
// =============================================================================

TEST_CASE("Q1 element: Gauss data, area and lumped mass", "[quad][element]") {
    // The 2 x 2 points are ordered like the nodes (counter-clockwise).
    const double s = quadrature::kGauss2Point;
    const auto& gp = q1::gauss_points();
    for (int g = 0; g < 4; ++g) {
        CHECK(gp[static_cast<std::size_t>(g)][0] == s * kNodes[g][0]);
        CHECK(gp[static_cast<std::size_t>(g)][1] == s * kNodes[g][1]);
        CHECK(q1::gauss_weight(g) == 1.0);
        // The shape function of node a is largest at the Gauss point next to it.
        for (int a = 0; a < 4; ++a)
            if (a != g) CHECK(q1::gauss_shape()[static_cast<std::size_t>(g)][static_cast<std::size_t>(g)] > q1::gauss_shape()[static_cast<std::size_t>(g)][static_cast<std::size_t>(a)]);
    }

    std::mt19937 rng(17);
    for (int t = 0; t < 100; ++t) {
        const Quad q = random_quad(rng);
        const QuadGeometry geo = q1::compute_geometry(q.x, q.y);
        CHECK_THAT(geo.area, WithinRel(shoelace(q), 1e-13));
        double msum = 0.0;
        for (int a = 0; a < 4; ++a) {
            CHECK(geo.mass[static_cast<std::size_t>(a)] > 0.0);
            msum += geo.mass[static_cast<std::size_t>(a)];
            // Exact: N_a det J is a polynomial of degree <= 2 in each variable.
            const double exact = integrate(q, 5, [a](double xi, double eta, const q1::Jacobian&) {
                double N[4];
                q1::shape_functions(xi, eta, N);
                return N[a];
            });
            CHECK_THAT(geo.mass[static_cast<std::size_t>(a)], WithinRel(exact, 1e-13));
        }
        CHECK_THAT(msum, WithinRel(geo.area, 1e-14));
    }
    // Parallelogram: every node gets a quarter of the area.
    const Quad para{{0, 2, 2.5, 0.5}, {0, 0, 1, 1}};
    const QuadGeometry pg = q1::compute_geometry(para.x, para.y);
    for (double m : pg.mass) CHECK_THAT(m, WithinRel(0.5, 1e-15));
}

TEST_CASE("Q1 element: mean gradients are exact boundary integrals", "[quad][element]") {
    std::mt19937 rng(19);
    for (int t = 0; t < 100; ++t) {
        const Quad q = random_quad(rng);
        const QuadGeometry geo = q1::compute_geometry(q.x, q.y);
        for (int a = 0; a < 4; ++a) {
            // int grad N_a = oint N_a n ds: over the two edges at node a, each (L/2) n = (dy, -dx) / 2.
            const int next = (a + 1) % 4, prev = (a + 3) % 4;
            const double bx = 0.5 * ((q.y[next] - q.y[a]) + (q.y[a] - q.y[prev]));
            const double by = -0.5 * ((q.x[next] - q.x[a]) + (q.x[a] - q.x[prev]));
            CHECK_THAT(geo.mean_dNdx[static_cast<std::size_t>(a)] * geo.area, WithinAbs(bx, 1e-13));
            CHECK_THAT(geo.mean_dNdy[static_cast<std::size_t>(a)] * geo.area, WithinAbs(by, 1e-13));
        }
        // Linear completeness of the mean gradients.
        double gxx = 0.0, gxy = 0.0, gyx = 0.0, gyy = 0.0;
        for (int a = 0; a < 4; ++a) {
            gxx += geo.mean_dNdx[static_cast<std::size_t>(a)] * q.x[a];
            gxy += geo.mean_dNdx[static_cast<std::size_t>(a)] * q.y[a];
            gyx += geo.mean_dNdy[static_cast<std::size_t>(a)] * q.x[a];
            gyy += geo.mean_dNdy[static_cast<std::size_t>(a)] * q.y[a];
        }
        CHECK_THAT(gxx, WithinAbs(1.0, 1e-12));
        CHECK_THAT(gxy, WithinAbs(0.0, 1e-12));
        CHECK_THAT(gyx, WithinAbs(0.0, 1e-12));
        CHECK_THAT(gyy, WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("Q1 element: Laplacian of rectangles in closed form", "[quad][element]") {
    const double hx = 0.3, hy = 1.7;
    const Quad q{{0, hx, hx, 0}, {0, 0, hy, hy}};
    const QuadLaplacian k = q1::laplacian(q1::compute_geometry(q.x, q.y));
    // K = hy / (6 hx) Kx + hx / (6 hy) Ky (e.g. Hughes 2000, section 3.2).
    const double Kx[4][4] = {{2, -2, -1, 1}, {-2, 2, 1, -1}, {-1, 1, 2, -2}, {1, -1, -2, 2}};
    const double Ky[4][4] = {{2, 1, -1, -2}, {1, 2, -2, -1}, {-1, -2, 2, 1}, {-2, -1, 1, 2}};
    auto exact = [&](int a, int b) { return hy / (6 * hx) * Kx[a][b] + hx / (6 * hy) * Ky[a][b]; };
    for (int a = 0; a < 4; ++a) CHECK_THAT(k.diag[static_cast<std::size_t>(a)], WithinRel(exact(a, a), 1e-14));
    for (int p = 0; p < 6; ++p) {
        const int a = q1::kPairs[p][0], b = q1::kPairs[p][1];
        CHECK_THAT(k.off[static_cast<std::size_t>(p)], WithinAbs(exact(a, b), 1e-14 * exact(0, 0)));
    }
    // Unit square: (1/6) [4 -1 -2 -1; ...].
    const Quad sq{{0, 1, 1, 0}, {0, 0, 1, 1}};
    const QuadLaplacian s = q1::laplacian(q1::compute_geometry(sq.x, sq.y));
    CHECK_THAT(s.diag[0], WithinAbs(4.0 / 6.0, 1e-15));
    CHECK_THAT(s.off[0], WithinAbs(-1.0 / 6.0, 1e-15));
    CHECK_THAT(s.off[1], WithinAbs(-2.0 / 6.0, 1e-15));
    CHECK_THAT(s.off[2], WithinAbs(-1.0 / 6.0, 1e-15));
}

TEST_CASE("Q1 element: Laplacian properties on distorted elements", "[quad][element]") {
    std::mt19937 rng(23);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (int t = 0; t < 100; ++t) {
        const Quad q = random_quad(rng);
        const QuadGeometry geo = q1::compute_geometry(q.x, q.y);
        const QuadLaplacian k = q1::laplacian(geo);
        double K[4][4];
        for (int a = 0; a < 4; ++a) K[a][a] = k.diag[static_cast<std::size_t>(a)];
        for (int p = 0; p < 6; ++p) {
            K[q1::kPairs[p][0]][q1::kPairs[p][1]] = k.off[static_cast<std::size_t>(p)];
            K[q1::kPairs[p][1]][q1::kPairs[p][0]] = k.off[static_cast<std::size_t>(p)];
        }
        // Constants in the null space; K f = int grad N_a . grad f for linear f.
        const double c = 2.0, px = 0.6, py = -1.1;
        for (int a = 0; a < 4; ++a) {
            double row = 0.0, lin = 0.0;
            for (int b = 0; b < 4; ++b) {
                row += K[a][b];
                lin += K[a][b] * (c + px * q.x[b] + py * q.y[b]);
            }
            CHECK_THAT(row, WithinAbs(0.0, 1e-13 * K[0][0]));
            const double flux = geo.area * (geo.mean_dNdx[static_cast<std::size_t>(a)] * px + geo.mean_dNdy[static_cast<std::size_t>(a)] * py);
            CHECK_THAT(lin, WithinAbs(flux, 1e-12 * K[0][0]));
        }
        // Positive semi-definite, positive on non-constant vectors.
        for (int s = 0; s < 10; ++s) {
            double v[4], quad = 0.0, mean = 0.0;
            for (double& vi : v) {
                vi = u(rng);
                mean += 0.25 * vi;
            }
            for (double& vi : v) vi -= mean;
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) quad += v[a] * K[a][b] * v[b];
            CHECK(quad > 0.0);
        }
        // 2 x 2 Gauss vs. a 6 x 6 reference: exact for parallelograms, close otherwise.
        const double ref00 = integrate(q, 6, [](double xi, double eta, const q1::Jacobian& J) {
            double dxi[4], deta[4], dx[4], dy[4];
            q1::shape_derivatives(xi, eta, dxi, deta);
            q1::physical_derivatives(J, dxi, deta, dx, dy);
            return dx[0] * dx[0] + dy[0] * dy[0];
        });
        CHECK_THAT(k.diag[0], WithinRel(ref00, 0.05));
    }
    const double A[2][2] = {{1.3, 0.4}, {-0.2, 0.9}};
    Quad para;
    for (int k = 0; k < 4; ++k) {
        para.x[k] = A[0][0] * kNodes[k][0] + A[0][1] * kNodes[k][1];
        para.y[k] = A[1][0] * kNodes[k][0] + A[1][1] * kNodes[k][1];
    }
    const QuadLaplacian kp = q1::laplacian(q1::compute_geometry(para.x, para.y));
    for (int p = 0; p < 6; ++p) {
        const int a = q1::kPairs[p][0], b = q1::kPairs[p][1];
        const double ref = integrate(para, 6, [a, b](double xi, double eta, const q1::Jacobian& J) {
            double dxi[4], deta[4], dx[4], dy[4];
            q1::shape_derivatives(xi, eta, dxi, deta);
            q1::physical_derivatives(J, dxi, deta, dx, dy);
            return dx[a] * dx[b] + dy[a] * dy[b];
        });
        CHECK_THAT(kp.off[static_cast<std::size_t>(p)], WithinAbs(ref, 1e-14));
    }
}
