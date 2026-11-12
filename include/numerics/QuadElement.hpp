#pragma once

// Bilinear four-node quadrilateral element (Q1), see
// docs/QUADRILATERAL_ELEMENTS.md, sections 3-5.
//
// Reference square [-1, 1]^2 with the nodes numbered counter-clockwise
//
//     3 (-1, 1) ------ 2 (1, 1)
//        |                |            N_a(xi, eta) = (1 + xi_a xi)(1 + eta_a eta) / 4
//        |                |
//     0 (-1,-1) ------ 1 (1,-1)
//
// Isoparametric map x(xi, eta) = sum_a N_a x_a, y = sum_a N_a y_a, Jacobian
//
//     J = [ dx/dxi  dx/deta ]      det J = x_xi y_eta - x_eta y_xi,
//         [ dy/dxi  dy/deta ]
//
// and physical derivatives by the chain rule (J^{-T}):
//
//     dN/dx = ( y_eta dN/dxi - y_xi dN/deta) / det J
//     dN/dy = (-x_eta dN/dxi + x_xi dN/deta) / det J.
//
// det J is an affine function of (xi, eta) (the xi*eta terms cancel), so it
// is positive on the whole element iff it is positive at the four corners,
// where it equals a quarter of the cross product of the two edges at the
// corner: det J(corner a) = cross(x_{a+1} - x_a, x_{a-1} - x_a) / 4.
//
// Element integrals use the 2 x 2 Gauss-Legendre rule (Quadrature.hpp), with
// the Gauss points ordered like the nodes (counter-clockwise):
//     g0 = (-s, -s), g1 = (s, -s), g2 = (s, s), g3 = (-s, s),  s = 1/sqrt(3).

#include <array>

namespace hydrama {

// Quadrature data of one quadrilateral (computed once per mesh).
struct QuadGeometry {
    std::array<double, 4> weight{};                   // w_g det J(g): physical integration weights
    std::array<std::array<double, 4>, 4> dNdx{};       // dN_a/dx at Gauss point g: dNdx[g][a]
    std::array<std::array<double, 4>, 4> dNdy{};       // dN_a/dy at Gauss point g
    std::array<double, 4> mass{};                     // int N_a dOmega (row-sum lumped mass)
    std::array<double, 4> mean_dNdx{};                // (1 / area) int dN_a/dx dOmega
    std::array<double, 4> mean_dNdy{};
    double area = 0.0;                                // sum_g weight[g] (exact)
};

// Element Laplacian K_ab = int grad N_a . grad N_b dOmega in element-by-element
// storage: 4 diagonal entries and the 6 off-diagonal pairs of q1::kPairs.
struct QuadLaplacian {
    std::array<double, 4> diag{};
    std::array<double, 6> off{};
};

namespace q1 {

inline constexpr int kNodes = 4;
inline constexpr int kGaussPoints = 4;
inline constexpr double kNodeXi[4] = {-1.0, 1.0, 1.0, -1.0};
inline constexpr double kNodeEta[4] = {-1.0, -1.0, 1.0, 1.0};

// Off-diagonal node pairs of the element-by-element storage (a < b).
inline constexpr int kPairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

// Edges of the element, counter-clockwise: edge k joins node k and k + 1.
inline constexpr int kEdges[4][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

void shape_functions(double xi, double eta, double N[4]);
void shape_derivatives(double xi, double eta, double dN_dxi[4], double dN_deta[4]);

struct Jacobian {
    double x_xi = 0.0, x_eta = 0.0, y_xi = 0.0, y_eta = 0.0;
    double det = 0.0;
};

// Jacobian of the map of the element with node coordinates x[4], y[4].
Jacobian jacobian(const double x[4], const double y[4], double xi, double eta);

// Physical derivatives from the reference derivatives (chain rule).
void physical_derivatives(const Jacobian& J, const double dN_dxi[4], const double dN_deta[4],
                          double dNdx[4], double dNdy[4]);

// Image of (xi, eta).
void map(const double x[4], const double y[4], double xi, double eta, double& px, double& py);

// det J at the four corners (equals cross(x_{a+1} - x_a, x_{a-1} - x_a) / 4).
std::array<double, 4> corner_jacobians(const double x[4], const double y[4]);

// The 2 x 2 Gauss points (counter-clockwise) and the shape functions there.
const std::array<std::array<double, 2>, 4>& gauss_points();
double gauss_weight(int g); // 1 for every point of the 2 x 2 rule
const std::array<std::array<double, 4>, 4>& gauss_shape(); // N_a at g: [g][a]

// Quadrature data of an element. Throws std::domain_error if det J <= 0 at a
// Gauss point (elements must be valid: see corner_jacobians).
QuadGeometry compute_geometry(const double x[4], const double y[4]);

// Element Laplacian from the quadrature data (2 x 2 Gauss).
QuadLaplacian laplacian(const QuadGeometry& g);

} // namespace q1
} // namespace hydrama
