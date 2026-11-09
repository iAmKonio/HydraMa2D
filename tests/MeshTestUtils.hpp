#pragma once

// Helpers for building small, analytically tractable meshes for unit tests.
//
// Conventions (matching the production mesh files, see test_mesh_hinterkante.cpp):
//   * element connectivity is counter-clockwise (positive signed area),
//   * boundary sides are traversed with the domain on their left, so the
//     face normal (dy, -dx) / L computed by Mesh points out of the domain.

#include "Mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <ostream>
#include <string>
#include <vector>

namespace hydrama::test {

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct Triangle {
    std::string name;
    std::array<Point, 3> p;
};

inline std::ostream& operator<<(std::ostream& os, const Triangle& t) {
    os << t.name << " [(" << t.p[0].x << ", " << t.p[0].y << "), ("
       << t.p[1].x << ", " << t.p[1].y << "), ("
       << t.p[2].x << ", " << t.p[2].y << ")]";
    return os;
}

// Build a mesh from raw arrays and keep the element/node/side counters in sync.
inline Mesh make_mesh(const std::vector<Point>& nodes,
                      const std::vector<std::array<int, 3>>& elements,
                      const std::vector<BoundarySide>& sides = {}) {
    Mesh m;
    m.npoin = static_cast<int>(nodes.size());
    m.nelem = static_cast<int>(elements.size());
    m.nboun = static_cast<int>(sides.size());
    for (const auto& n : nodes) {
        m.x.push_back(n.x);
        m.y.push_back(n.y);
    }
    m.intma = elements;
    m.iside = sides;
    return m;
}

// One-element mesh whose boundary is the element's own three edges
// (closed, domain on the left when the triangle is counter-clockwise).
inline Mesh make_triangle_mesh(const Triangle& t, int flag = 500) {
    std::vector<BoundarySide> sides = {
        {0, 1, 0, 1, flag},
        {1, 2, 0, 2, flag},
        {2, 0, 0, 3, flag},
    };
    return make_mesh({t.p[0], t.p[1], t.p[2]}, {{0, 1, 2}}, sides);
}

// Structured rectangle [x0, x0+lx] x [y0, y0+ly] with nx * ny cells, each cell
// split along its (i,j)->(i+1,j+1) diagonal into two counter-clockwise
// triangles. Node (i, j) has index j*(nx+1)+i. Boundary sides are emitted in
// counter-clockwise order with side ids 1=bottom, 2=right, 3=top, 4=left.
struct GridSpec {
    double x0 = 0.0, y0 = 0.0;
    double lx = 1.0, ly = 1.0;
    int nx = 1, ny = 1;

    double hx() const { return lx / nx; }
    double hy() const { return ly / ny; }
    int node(int i, int j) const { return j * (nx + 1) + i; }
    int npoin() const { return (nx + 1) * (ny + 1); }
    int nelem() const { return 2 * nx * ny; }
    int nboun() const { return 2 * (nx + ny); }
};

inline Mesh make_grid_mesh(const GridSpec& g, int flag = 500) {
    std::vector<Point> nodes;
    for (int j = 0; j <= g.ny; ++j) {
        for (int i = 0; i <= g.nx; ++i) {
            nodes.push_back({g.x0 + i * g.hx(), g.y0 + j * g.hy()});
        }
    }

    std::vector<std::array<int, 3>> elems;
    for (int j = 0; j < g.ny; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            int p00 = g.node(i, j), p10 = g.node(i + 1, j);
            int p01 = g.node(i, j + 1), p11 = g.node(i + 1, j + 1);
            elems.push_back({p00, p10, p11}); // element 2*(j*nx+i)
            elems.push_back({p00, p11, p01}); // element 2*(j*nx+i)+1
        }
    }
    auto cell_elem = [&](int i, int j, int t) { return 2 * (j * g.nx + i) + t; };

    std::vector<BoundarySide> sides;
    for (int i = 0; i < g.nx; ++i) // bottom, left -> right
        sides.push_back({g.node(i, 0), g.node(i + 1, 0), cell_elem(i, 0, 0), 1, flag});
    for (int j = 0; j < g.ny; ++j) // right, bottom -> top
        sides.push_back({g.node(g.nx, j), g.node(g.nx, j + 1), cell_elem(g.nx - 1, j, 0), 2, flag});
    for (int i = g.nx - 1; i >= 0; --i) // top, right -> left
        sides.push_back({g.node(i + 1, g.ny), g.node(i, g.ny), cell_elem(i, g.ny - 1, 1), 3, flag});
    for (int j = g.ny - 1; j >= 0; --j) // left, top -> bottom
        sides.push_back({g.node(0, j + 1), g.node(0, j), cell_elem(0, j, 1), 4, flag});

    return make_mesh(nodes, elems, sides);
}

// Basic geometric pre-processing: areas/derivatives, lumped mass, face normals.
inline void compute_geometry(Mesh& m) {
    m.compute_shape_function_derivatives();
    m.compute_mass_matrix();
    m.compute_face_normals();
}

// Full geometric pre-processing in the same order as Solver::initialize().
inline void compute_all_geometry(Mesh& m) {
    m.compute_shape_function_derivatives();
    m.compute_mass_matrix();
    m.compute_face_normals();
    m.determine_wall_nodes();
    m.compute_element_sizes();
    m.compute_laplacian_stiffness();
}

// Override boundary flags by side id, e.g. {{1, 500}, {2, 504}} for a grid
// (side ids 1=bottom, 2=right, 3=top, 4=left).
struct SideFlag {
    int side_id;
    int flag;
};

inline void set_flags_by_side_id(Mesh& m, const std::vector<SideFlag>& flags) {
    for (auto& side : m.iside) {
        for (const auto& sf : flags) {
            if (side.side_id == sf.side_id) side.flag = sf.flag;
        }
    }
}

// Nodal values f(x_i, y_i) of a field.
template <typename F>
std::vector<double> sample(const Mesh& m, F f) {
    std::vector<double> v(m.npoin);
    for (int ip = 0; ip < m.npoin; ++ip) v[ip] = f(m.x[ip], m.y[ip]);
    return v;
}

// Deterministic element time steps that differ from element to element
// (local time stepping): 0.01 * {1, 3, 5, 2, 4, ...}.
inline std::vector<double> varying_dt(int nelem) {
    std::vector<double> dt(nelem);
    for (int ie = 0; ie < nelem; ++ie) dt[ie] = 0.01 * (1 + (ie * 7) % 5);
    return dt;
}

// Largest absolute entry (scale for absolute tolerances).
inline double scale_of(const std::vector<double>& a) {
    double s = 0.0;
    for (double x : a) s = std::max(s, std::abs(x));
    return s;
}

// Index of the node at (x, y), or -1 if there is none.
inline int find_node(const Mesh& m, double x, double y, double tol = 1e-9) {
    for (int ip = 0; ip < m.npoin; ++ip) {
        if (std::abs(m.x[ip] - x) < tol && std::abs(m.y[ip] - y) < tol) return ip;
    }
    return -1;
}

// Unsigned triangle area from side lengths only (Kahan's numerically stable
// Heron formula), i.e. independent of the coordinate formula used in Mesh.
inline double heron_area(const Triangle& t) {
    auto dist = [](Point a, Point b) { return std::hypot(b.x - a.x, b.y - a.y); };
    std::array<double, 3> s = {dist(t.p[0], t.p[1]), dist(t.p[1], t.p[2]), dist(t.p[2], t.p[0])};
    std::sort(s.begin(), s.end(), [](double l, double r) { return l > r; });
    const double a = s[0], b = s[1], c = s[2];
    return 0.25 * std::sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c)));
}

// A deterministic set of valid, counter-clockwise triangles covering
// different shapes, sizes, aspect ratios and positions.
inline std::vector<Triangle> reference_triangles() {
    return {
        {"unit right", {{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}}},
        {"scalene", {{{1.0, 1.0}, {4.0, 2.0}, {2.0, 5.0}}}},
        {"equilateral", {{{0.0, 0.0}, {2.0, 0.0}, {1.0, std::sqrt(3.0)}}}},
        {"obtuse", {{{0.0, 0.0}, {10.0, 0.0}, {9.0, 0.5}}}},
        {"negative quadrant", {{{-3.0, -2.0}, {-1.0, -2.5}, {-1.5, -0.5}}}},
        {"small (mesh scale)", {{{0.1, 0.2}, {0.115, 0.2}, {0.105, 0.2125}}}},
        {"high aspect ratio", {{{0.0, 0.0}, {100.0, 0.0}, {50.0, 0.01}}}},
        {"far from origin", {{{1000.0, -500.0}, {1002.0, -500.0}, {1001.0, -498.0}}}},
    };
}

inline Triangle translated(const Triangle& t, double dx, double dy) {
    Triangle r = t;
    for (auto& p : r.p) {
        p.x += dx;
        p.y += dy;
    }
    return r;
}

inline Triangle scaled(const Triangle& t, double s) {
    Triangle r = t;
    for (auto& p : r.p) {
        p.x *= s;
        p.y *= s;
    }
    return r;
}

// Same triangle with the first two vertices swapped (clockwise ordering).
inline Triangle reversed(const Triangle& t) {
    Triangle r = t;
    std::swap(r.p[0], r.p[1]);
    r.name += " (clockwise)";
    return r;
}

} // namespace hydrama::test
