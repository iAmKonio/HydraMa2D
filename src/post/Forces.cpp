#include "post/Forces.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace hydrama::post {

namespace {

void check_inputs(const Mesh& mesh, const std::vector<int>& body_sides, const std::vector<double>& u,
                  const std::vector<double>& v, const std::vector<double>& p, const char* what) {
    const auto n = static_cast<std::size_t>(mesh.npoin);
    if (u.size() != n || v.size() != n || p.size() != n)
        throw std::invalid_argument(std::string(what) + ": the fields must have one value per node");
    if (static_cast<int>(mesh.dNkdx.size()) != mesh.nelem ||
        static_cast<int>(mesh.quad_geom.size()) != mesh.num_quads())
        throw std::invalid_argument(std::string(what) + ": the element geometry has not been computed");
    for (int ib : body_sides)
        if (ib < 0 || ib >= mesh.nboun)
            throw std::invalid_argument(std::string(what) + ": boundary side " + std::to_string(ib) + " out of range");
}

// Velocity gradient of element e at the midpoint of its side (n1, n2).
struct Gradient {
    double ux = 0.0, uy = 0.0, vx = 0.0, vy = 0.0;
};

Gradient side_gradient(const Mesh& mesh, int e, int n1, int n2, const std::vector<double>& u,
                       const std::vector<double>& v) {
    Gradient g;
    const ElementNodes nodes = mesh.element_nodes(e);
    if (!mesh.is_quad(e)) {
        const ElementDeriv& d = mesh.dNkdx[static_cast<std::size_t>(e)];
        for (int a = 0; a < 3; ++a) {
            const auto ip = static_cast<std::size_t>(nodes[a]);
            g.ux += u[ip] * d.dNdx[a];
            g.uy += u[ip] * d.dNdy[a];
            g.vx += v[ip] * d.dNdx[a];
            g.vy += v[ip] * d.dNdy[a];
        }
        return g;
    }
    double x[4], y[4];
    double xi = 0.0, eta = 0.0;
    int on_side = 0;
    for (int a = 0; a < 4; ++a) {
        x[a] = mesh.x[static_cast<std::size_t>(nodes[a])];
        y[a] = mesh.y[static_cast<std::size_t>(nodes[a])];
        if (nodes[a] == n1 || nodes[a] == n2) {
            xi += 0.5 * q1::kNodeXi[a];
            eta += 0.5 * q1::kNodeEta[a];
            ++on_side;
        }
    }
    if (on_side != 2) throw std::invalid_argument("surface_force: a body side is not an edge of its element");
    double dxi[4], deta[4], dx[4], dy[4];
    q1::shape_derivatives(xi, eta, dxi, deta);
    q1::physical_derivatives(q1::jacobian(x, y, xi, eta), dxi, deta, dx, dy);
    for (int a = 0; a < 4; ++a) {
        const auto ip = static_cast<std::size_t>(nodes[a]);
        g.ux += u[ip] * dx[a];
        g.uy += u[ip] * dy[a];
        g.vx += v[ip] * dx[a];
        g.vy += v[ip] * dy[a];
    }
    return g;
}

} // namespace

BodyForce surface_force(const Mesh& mesh, const std::vector<int>& body_sides, double nu,
                        const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& p) {
    check_inputs(mesh, body_sides, u, v, p, "surface_force");
    BodyForce f;
    for (int ib : body_sides) {
        const BoundarySide& s = mesh.iside[static_cast<std::size_t>(ib)];
        const auto i1 = static_cast<std::size_t>(s.node1), i2 = static_cast<std::size_t>(s.node2);
        const double dx = mesh.x[i2] - mesh.x[i1], dy = mesh.y[i2] - mesh.y[i1];
        const double length = std::hypot(dx, dy);
        // n_b: unit normal out of the body = into the fluid = towards the
        // element (independent of the node order of the side).
        double nx = -dy / length, ny = dx / length;
        const ElementNodes e = mesh.element_nodes(s.elem);
        double cx = 0.0, cy = 0.0;
        for (int k = 0; k < e.size(); ++k) {
            cx += mesh.x[static_cast<std::size_t>(e[k])] / e.size();
            cy += mesh.y[static_cast<std::size_t>(e[k])] / e.size();
        }
        if (nx * (cx - mesh.x[i1]) + ny * (cy - mesh.y[i1]) < 0.0) {
            nx = -nx;
            ny = -ny;
        }
        const double pm = 0.5 * (p[i1] + p[i2]); // exact mean of the linear pressure
        const Gradient g = side_gradient(mesh, s.elem, s.node1, s.node2, u, v);
        const double sxy = g.uy + g.vx;
        const double px = -pm * nx * length, py = -pm * ny * length;
        f.pressure_fx += px;
        f.pressure_fy += py;
        f.fx += px + nu * length * (2.0 * g.ux * nx + sxy * ny);
        f.fy += py + nu * length * (sxy * nx + 2.0 * g.vy * ny);
    }
    return f;
}

BodyForce volume_force(const Mesh& mesh, const std::vector<int>& body_sides, double nu,
                       const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& p,
                       const std::vector<double>* dudt, const std::vector<double>* dvdt) {
    check_inputs(mesh, body_sides, u, v, p, "volume_force");
    if ((dudt == nullptr) != (dvdt == nullptr))
        throw std::invalid_argument("volume_force: give both components of the time derivative, or none");
    if (dudt && (dudt->size() != u.size() || dvdt->size() != u.size()))
        throw std::invalid_argument("volume_force: the time derivative must have one value per node");

    std::vector<char> body(static_cast<std::size_t>(mesh.npoin), 0);
    for (int ib : body_sides) {
        body[static_cast<std::size_t>(mesh.iside[static_cast<std::size_t>(ib)].node1)] = 1;
        body[static_cast<std::size_t>(mesh.iside[static_cast<std::size_t>(ib)].node2)] = 1;
    }

    // Integrand at one quadrature point, given N_a and grad N_a there.
    double sx = 0.0, sy = 0.0;
    auto accumulate = [&](const ElementNodes& nodes, const double* N, const double* Nx, const double* Ny,
                          double weight) {
        double uq = 0.0, vq = 0.0, pq = 0.0, w = 0.0, ut = 0.0, vt = 0.0;
        double ux = 0.0, uy = 0.0, vx = 0.0, vy = 0.0, wx = 0.0, wy = 0.0;
        for (int a = 0; a < nodes.size(); ++a) {
            const auto ip = static_cast<std::size_t>(nodes[a]);
            uq += N[a] * u[ip];
            vq += N[a] * v[ip];
            pq += N[a] * p[ip];
            ux += Nx[a] * u[ip];
            uy += Ny[a] * u[ip];
            vx += Nx[a] * v[ip];
            vy += Ny[a] * v[ip];
            if (body[ip]) {
                w += N[a];
                wx += Nx[a];
                wy += Ny[a];
            }
            if (dudt) {
                ut += N[a] * (*dudt)[ip];
                vt += N[a] * (*dvdt)[ip];
            }
        }
        // w = phi e_x: div w = phi_x; w = phi e_y: div w = phi_y.
        sx += weight * ((ut + uq * ux + vq * uy) * w + nu * (ux * wx + uy * wy) - pq * wx);
        sy += weight * ((vt + uq * vx + vq * vy) * w + nu * (vx * wx + vy * wy) - pq * wy);
    };

    for (int e = 0; e < mesh.nelem; ++e) {
        const ElementNodes nodes = mesh.element_nodes(e);
        bool touches = false;
        for (int a = 0; a < nodes.size(); ++a) touches = touches || body[static_cast<std::size_t>(nodes[a])];
        if (!touches) continue;
        if (!mesh.is_quad(e)) {
            // Edge-midpoint rule: exact for quadratics on triangles.
            const ElementDeriv& d = mesh.dNkdx[static_cast<std::size_t>(e)];
            const double weight = mesh.detJ[static_cast<std::size_t>(e)] / 6.0; // area / 3
            static constexpr double kMid[3][3] = {{0.5, 0.5, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}};
            for (const auto& N : kMid) accumulate(nodes, N, d.dNdx, d.dNdy, weight);
        } else {
            const QuadGeometry& g = mesh.quad_geom[static_cast<std::size_t>(mesh.quad_of(e))];
            const auto& shape = q1::gauss_shape();
            for (int q = 0; q < q1::kGaussPoints; ++q) {
                const auto gq = static_cast<std::size_t>(q);
                accumulate(nodes, shape[gq].data(), g.dNdx[gq].data(), g.dNdy[gq].data(), g.weight[gq]);
            }
        }
    }
    BodyForce f;
    f.fx = -sx;
    f.fy = -sy;
    return f;
}

double force_coefficient(double force, double reference_velocity, double reference_length) {
    if (!(reference_velocity > 0.0) || !(reference_length > 0.0))
        throw std::invalid_argument("force_coefficient: the reference velocity and length must be > 0");
    return force / (0.5 * reference_velocity * reference_velocity * reference_length);
}

} // namespace hydrama::post
