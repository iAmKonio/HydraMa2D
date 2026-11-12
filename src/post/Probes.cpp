#include "post/Probes.hpp"

#include <algorithm>
#include <cmath>

namespace hydrama::post {

namespace {

bool locate_in_triangle(const Mesh& mesh, int e, double px, double py, double tolerance, PointLocation& out) {
    const std::array<int, 3>& t = mesh.intma[static_cast<std::size_t>(e)];
    const double x0 = mesh.x[static_cast<std::size_t>(t[0])], y0 = mesh.y[static_cast<std::size_t>(t[0])];
    const double x1 = mesh.x[static_cast<std::size_t>(t[1])], y1 = mesh.y[static_cast<std::size_t>(t[1])];
    const double x2 = mesh.x[static_cast<std::size_t>(t[2])], y2 = mesh.y[static_cast<std::size_t>(t[2])];
    const double det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if (det == 0.0) return false;
    const double l1 = ((px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)) / det;
    const double l2 = ((x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)) / det;
    const double l0 = 1.0 - l1 - l2;
    if (l0 < -tolerance || l1 < -tolerance || l2 < -tolerance) return false;
    out.element = e;
    out.count = 3;
    out.nodes = {t[0], t[1], t[2], 0};
    out.weights = {l0, l1, l2, 0.0};
    return true;
}

bool locate_in_quad(const Mesh& mesh, int e, double px, double py, double tolerance, PointLocation& out) {
    const std::array<int, 4>& q = mesh.quads[static_cast<std::size_t>(mesh.quad_of(e))];
    double x[4], y[4];
    for (int a = 0; a < 4; ++a) {
        x[a] = mesh.x[static_cast<std::size_t>(q[static_cast<std::size_t>(a)])];
        y[a] = mesh.y[static_cast<std::size_t>(q[static_cast<std::size_t>(a)])];
    }
    // Newton's method for x(xi, eta) = p, from the centre.
    double xi = 0.0, eta = 0.0;
    bool converged = false;
    for (int it = 0; it < 50; ++it) {
        double mx = 0.0, my = 0.0;
        q1::map(x, y, xi, eta, mx, my);
        const double rx = px - mx, ry = py - my;
        const q1::Jacobian J = q1::jacobian(x, y, xi, eta);
        if (!(std::abs(J.det) > 0.0)) return false;
        const double dxi = (J.y_eta * rx - J.x_eta * ry) / J.det;
        const double deta = (-J.y_xi * rx + J.x_xi * ry) / J.det;
        xi += dxi;
        eta += deta;
        if (!std::isfinite(xi) || !std::isfinite(eta) || std::abs(xi) > 10.0 || std::abs(eta) > 10.0) return false;
        if (std::abs(dxi) + std::abs(deta) < 1e-14) {
            converged = true;
            break;
        }
    }
    if (!converged) return false;
    if (std::abs(xi) > 1.0 + 2.0 * tolerance || std::abs(eta) > 1.0 + 2.0 * tolerance) return false;
    double N[4];
    q1::shape_functions(xi, eta, N);
    out.element = e;
    out.count = 4;
    out.nodes = q;
    out.weights = {N[0], N[1], N[2], N[3]};
    return true;
}

} // namespace

std::optional<PointLocation> locate_point(const Mesh& mesh, double px, double py, double tolerance) {
    PointLocation loc;
    for (int e = 0; e < mesh.nelem; ++e) {
        // Bounding box test first (cheap), with a margin relative to the element size.
        const ElementNodes nodes = mesh.element_nodes(e);
        double xmin = mesh.x[static_cast<std::size_t>(nodes[0])], xmax = xmin;
        double ymin = mesh.y[static_cast<std::size_t>(nodes[0])], ymax = ymin;
        for (int k = 1; k < nodes.size(); ++k) {
            xmin = std::min(xmin, mesh.x[static_cast<std::size_t>(nodes[k])]);
            xmax = std::max(xmax, mesh.x[static_cast<std::size_t>(nodes[k])]);
            ymin = std::min(ymin, mesh.y[static_cast<std::size_t>(nodes[k])]);
            ymax = std::max(ymax, mesh.y[static_cast<std::size_t>(nodes[k])]);
        }
        const double margin = tolerance * std::max(xmax - xmin, ymax - ymin);
        if (px < xmin - margin || px > xmax + margin || py < ymin - margin || py > ymax + margin) continue;
        const bool found = mesh.is_quad(e) ? locate_in_quad(mesh, e, px, py, tolerance, loc)
                                           : locate_in_triangle(mesh, e, px, py, tolerance, loc);
        if (found) return loc;
    }
    return std::nullopt;
}

double interpolate(const PointLocation& location, const std::vector<double>& field) {
    double value = 0.0;
    for (int a = 0; a < location.count; ++a)
        value += location.weights[static_cast<std::size_t>(a)] * field[static_cast<std::size_t>(location.nodes[static_cast<std::size_t>(a)])];
    return value;
}

} // namespace hydrama::post
