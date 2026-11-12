#include "numerics/QuadElement.hpp"

#include "numerics/Quadrature.hpp"

#include <stdexcept>

namespace hydrama::q1 {

void shape_functions(double xi, double eta, double N[4]) {
    for (int a = 0; a < 4; ++a) N[a] = 0.25 * (1.0 + kNodeXi[a] * xi) * (1.0 + kNodeEta[a] * eta);
}

void shape_derivatives(double xi, double eta, double dN_dxi[4], double dN_deta[4]) {
    for (int a = 0; a < 4; ++a) {
        dN_dxi[a] = 0.25 * kNodeXi[a] * (1.0 + kNodeEta[a] * eta);
        dN_deta[a] = 0.25 * kNodeEta[a] * (1.0 + kNodeXi[a] * xi);
    }
}

Jacobian jacobian(const double x[4], const double y[4], double xi, double eta) {
    double dN_dxi[4], dN_deta[4];
    shape_derivatives(xi, eta, dN_dxi, dN_deta);
    Jacobian J;
    for (int a = 0; a < 4; ++a) {
        J.x_xi += dN_dxi[a] * x[a];
        J.x_eta += dN_deta[a] * x[a];
        J.y_xi += dN_dxi[a] * y[a];
        J.y_eta += dN_deta[a] * y[a];
    }
    J.det = J.x_xi * J.y_eta - J.x_eta * J.y_xi;
    return J;
}

void physical_derivatives(const Jacobian& J, const double dN_dxi[4], const double dN_deta[4],
                          double dNdx[4], double dNdy[4]) {
    const double inv = 1.0 / J.det;
    for (int a = 0; a < 4; ++a) {
        dNdx[a] = (J.y_eta * dN_dxi[a] - J.y_xi * dN_deta[a]) * inv;
        dNdy[a] = (-J.x_eta * dN_dxi[a] + J.x_xi * dN_deta[a]) * inv;
    }
}

void map(const double x[4], const double y[4], double xi, double eta, double& px, double& py) {
    double N[4];
    shape_functions(xi, eta, N);
    px = py = 0.0;
    for (int a = 0; a < 4; ++a) {
        px += N[a] * x[a];
        py += N[a] * y[a];
    }
}

std::array<double, 4> corner_jacobians(const double x[4], const double y[4]) {
    std::array<double, 4> det{};
    for (int a = 0; a < 4; ++a) {
        const int next = (a + 1) % 4, prev = (a + 3) % 4;
        const double ex = x[next] - x[a], ey = y[next] - y[a]; // edge to the next node
        const double fx = x[prev] - x[a], fy = y[prev] - y[a]; // edge to the previous node
        det[static_cast<std::size_t>(a)] = 0.25 * (ex * fy - ey * fx);
    }
    return det;
}

const std::array<std::array<double, 2>, 4>& gauss_points() {
    static const std::array<std::array<double, 2>, 4> points = [] {
        constexpr double s = quadrature::kGauss2Point;
        return std::array<std::array<double, 2>, 4>{{{-s, -s}, {s, -s}, {s, s}, {-s, s}}};
    }();
    return points;
}

double gauss_weight(int) { return 1.0; }

const std::array<std::array<double, 4>, 4>& gauss_shape() {
    static const std::array<std::array<double, 4>, 4> shape = [] {
        std::array<std::array<double, 4>, 4> s{};
        for (int g = 0; g < 4; ++g) shape_functions(gauss_points()[g][0], gauss_points()[g][1], s[g].data());
        return s;
    }();
    return shape;
}

QuadGeometry compute_geometry(const double x[4], const double y[4]) {
    QuadGeometry geo;
    const auto& shape = gauss_shape();
    for (int g = 0; g < 4; ++g) {
        const double xi = gauss_points()[g][0], eta = gauss_points()[g][1];
        double dN_dxi[4], dN_deta[4];
        shape_derivatives(xi, eta, dN_dxi, dN_deta);
        const Jacobian J = jacobian(x, y, xi, eta);
        if (!(J.det > 0.0)) throw std::domain_error("quadrilateral with non-positive Jacobian determinant");
        physical_derivatives(J, dN_dxi, dN_deta, geo.dNdx[g].data(), geo.dNdy[g].data());
        const double w = gauss_weight(g) * J.det;
        geo.weight[g] = w;
        geo.area += w;
        for (int a = 0; a < 4; ++a) {
            geo.mass[a] += w * shape[g][a];
            geo.mean_dNdx[a] += w * geo.dNdx[g][a];
            geo.mean_dNdy[a] += w * geo.dNdy[g][a];
        }
    }
    for (int a = 0; a < 4; ++a) {
        geo.mean_dNdx[a] /= geo.area;
        geo.mean_dNdy[a] /= geo.area;
    }
    return geo;
}

QuadLaplacian laplacian(const QuadGeometry& geo) {
    QuadLaplacian k;
    for (int g = 0; g < 4; ++g) {
        const double w = geo.weight[g];
        const auto& nx = geo.dNdx[g];
        const auto& ny = geo.dNdy[g];
        for (int a = 0; a < 4; ++a) k.diag[a] += w * (nx[a] * nx[a] + ny[a] * ny[a]);
        for (int p = 0; p < 6; ++p) {
            const int a = kPairs[p][0], b = kPairs[p][1];
            k.off[p] += w * (nx[a] * nx[b] + ny[a] * ny[b]);
        }
    }
    return k;
}

} // namespace hydrama::q1
