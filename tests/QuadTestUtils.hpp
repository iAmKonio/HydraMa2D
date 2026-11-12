#pragma once

// Meshes with bilinear quadrilaterals for tests (built directly, without the
// importer), including the patches of the patch tests:
//
//   * macneal_harder_patch():  the 5-element patch of MacNeal & Harder (1985),
//     rectangle 0.24 x 0.12 with four interior nodes of valence 3;
//   * pentagon_patch():        5 kites around a centre node of valence 5;
//   * triangle_split_patch():  a triangle split into 3 quadrilaterals (centre
//     node of valence 3);
//   * hybrid_pentagon_patch(): the pentagon with two kites split into two
//     triangles each (centre shared by quadrilaterals and triangles);
//   * grid():                  structured grids of quadrilaterals, triangles
//     or both (rows of quadrilaterals next to the bottom and top boundaries:
//     an "inflation layer"), with optional random perturbation of the
//     interior nodes (<= 0.2 h keeps every quadrilateral convex).
//
// "Singular" refers to irregular vertices (valence != 4), the singularities of
// a quadrilateral mesh. Boundary sides are generated from the elements
// (counter-clockwise, flag from the side's midpoint).

#include "Mesh.hpp"
#include "MeshTestUtils.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <map>
#include <random>
#include <vector>

namespace hydrama::test {

struct HybridSpec {
    std::vector<Point> nodes;
    std::vector<std::array<int, 3>> triangles;
    std::vector<std::array<int, 4>> quads;
};

// Mesh with boundary sides on every edge that belongs to one element,
// oriented counter-clockwise with respect to it. The flag and side id come
// from `flag_of(xm, ym)` at the midpoint (default: wall 500, id 1).
inline Mesh make_hybrid_mesh(const HybridSpec& s,
                             const std::function<int(double, double)>& flag_of = [](double, double) { return 500; }) {
    Mesh m;
    m.npoin = static_cast<int>(s.nodes.size());
    for (const Point& p : s.nodes) {
        m.x.push_back(p.x);
        m.y.push_back(p.y);
    }
    m.intma = s.triangles;
    m.quads = s.quads;
    m.nelem = m.num_triangles() + m.num_quads();

    std::map<std::pair<int, int>, std::vector<std::array<int, 3>>> uses; // edge -> (elem, from, to)
    for (int e = 0; e < m.nelem; ++e) {
        const ElementNodes n = m.element_nodes(e);
        for (int k = 0; k < n.size(); ++k) {
            const int a = n[k], b = n[(k + 1) % n.size()];
            uses[{std::min(a, b), std::max(a, b)}].push_back({e, a, b});
        }
    }
    for (const auto& [edge, list] : uses) {
        if (list.size() != 1) continue;
        const auto& [e, a, b] = list.front();
        const double xm = 0.5 * (m.x[a] + m.x[b]), ym = 0.5 * (m.y[a] + m.y[b]);
        const int flag = flag_of(xm, ym);
        m.iside.push_back({a, b, e, flag, flag});
    }
    m.nboun = static_cast<int>(m.iside.size());
    return m;
}

inline HybridSpec macneal_harder_patch() {
    HybridSpec s;
    s.nodes = {{0.0, 0.0}, {0.24, 0.0}, {0.24, 0.12}, {0.0, 0.12},        // corners 0..3
               {0.04, 0.02}, {0.18, 0.03}, {0.16, 0.08}, {0.08, 0.08}};  // interior 4..7
    s.quads = {{4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
    return s;
}

inline HybridSpec pentagon_patch(double radius = 1.0, double cx = 0.3, double cy = -0.2) {
    HybridSpec s;
    const double pi = std::acos(-1.0);
    s.nodes.push_back({cx, cy}); // centre 0
    for (int k = 0; k < 5; ++k) { // vertex V_k = 1 + 2k, midpoint M_k = 2 + 2k (between V_k and V_{k+1})
        const double a = pi / 2 + 2 * pi * k / 5, b = a + 2 * pi / 5;
        s.nodes.push_back({cx + radius * std::cos(a), cy + radius * std::sin(a)});
        s.nodes.push_back({cx + 0.5 * radius * (std::cos(a) + std::cos(b)), cy + 0.5 * radius * (std::sin(a) + std::sin(b))});
    }
    for (int k = 0; k < 5; ++k) {
        const int vk = 1 + 2 * k, mk = 2 + 2 * k, mprev = 2 + 2 * ((k + 4) % 5);
        s.quads.push_back({0, mprev, vk, mk});
    }
    return s;
}

inline HybridSpec triangle_split_patch() {
    HybridSpec s;
    const Point A{0.0, 0.0}, B{2.0, 0.3}, C{0.7, 1.6};
    auto mid = [](Point p, Point q) { return Point{0.5 * (p.x + q.x), 0.5 * (p.y + q.y)}; };
    const Point G{(A.x + B.x + C.x) / 3.0, (A.y + B.y + C.y) / 3.0};
    s.nodes = {A, B, C, mid(A, B), mid(B, C), mid(C, A), G}; // 0..6
    s.quads = {{0, 3, 6, 5}, {1, 4, 6, 3}, {2, 5, 6, 4}};
    return s;
}

// Pentagon with kites 1 and 3 split along (centre, vertex) into two triangles.
inline HybridSpec hybrid_pentagon_patch() {
    HybridSpec s = pentagon_patch();
    HybridSpec h;
    h.nodes = s.nodes;
    for (int k = 0; k < 5; ++k) {
        const auto& q = s.quads[static_cast<std::size_t>(k)];
        if (k == 1 || k == 3) {
            h.triangles.push_back({q[0], q[1], q[2]});
            h.triangles.push_back({q[0], q[2], q[3]});
        } else {
            h.quads.push_back(q);
        }
    }
    return h;
}

enum class GridCells { Quads, Triangles, InflationLayers };

struct GridOptions {
    double x0 = 0.0, y0 = 0.0, lx = 1.0, ly = 1.0;
    int nx = 4, ny = 4;
    GridCells cells = GridCells::Quads;
    int layers = 1;               // InflationLayers: rows of quadrilaterals at the bottom and at the top
    double perturbation = 0.0;    // interior nodes moved by up to this fraction of the cell size
    unsigned seed = 1;
    bool clockwise = false;       // write every element clockwise
};

inline HybridSpec grid(const GridOptions& o) {
    HybridSpec s;
    const double hx = o.lx / o.nx, hy = o.ly / o.ny;
    std::mt19937 rng(o.seed);
    std::uniform_real_distribution<double> r(-1.0, 1.0);
    for (int j = 0; j <= o.ny; ++j) {
        for (int i = 0; i <= o.nx; ++i) {
            double x = o.x0 + i * hx, y = o.y0 + j * hy;
            if (i > 0 && i < o.nx && j > 0 && j < o.ny && o.perturbation > 0.0) {
                x += o.perturbation * hx * r(rng);
                y += o.perturbation * hy * r(rng);
            }
            s.nodes.push_back({x, y});
        }
    }
    auto node = [&](int i, int j) { return j * (o.nx + 1) + i; };
    for (int j = 0; j < o.ny; ++j) {
        for (int i = 0; i < o.nx; ++i) {
            const int a = node(i, j), b = node(i + 1, j), c = node(i + 1, j + 1), d = node(i, j + 1);
            const bool quad = o.cells == GridCells::Quads ||
                              (o.cells == GridCells::InflationLayers && (j < o.layers || j >= o.ny - o.layers));
            if (quad) {
                s.quads.push_back(o.clockwise ? std::array<int, 4>{a, d, c, b} : std::array<int, 4>{a, b, c, d});
            } else if ((i + j) % 2 == 0) {
                s.triangles.push_back(o.clockwise ? std::array<int, 3>{a, c, b} : std::array<int, 3>{a, b, c});
                s.triangles.push_back(o.clockwise ? std::array<int, 3>{a, d, c} : std::array<int, 3>{a, c, d});
            } else {
                s.triangles.push_back(o.clockwise ? std::array<int, 3>{a, d, b} : std::array<int, 3>{a, b, d});
                s.triangles.push_back(o.clockwise ? std::array<int, 3>{b, d, c} : std::array<int, 3>{b, c, d});
            }
        }
    }
    return s;
}

// The mesh without the cells whose centre satisfies `remove(xc, yc)` (e.g. a
// hole), with the nodes that are no longer used dropped and renumbered.
inline HybridSpec without_cells(const HybridSpec& s, const std::function<bool(double, double)>& remove) {
    HybridSpec out;
    std::vector<int> new_index(s.nodes.size(), -1);
    auto map_node = [&](int n) {
        if (new_index[static_cast<std::size_t>(n)] < 0) {
            new_index[static_cast<std::size_t>(n)] = static_cast<int>(out.nodes.size());
            out.nodes.push_back(s.nodes[static_cast<std::size_t>(n)]);
        }
        return new_index[static_cast<std::size_t>(n)];
    };
    auto centre_removed = [&](const auto& cell) {
        double xc = 0.0, yc = 0.0;
        for (int n : cell) {
            xc += s.nodes[static_cast<std::size_t>(n)].x / static_cast<double>(cell.size());
            yc += s.nodes[static_cast<std::size_t>(n)].y / static_cast<double>(cell.size());
        }
        return remove(xc, yc);
    };
    for (const auto& t : s.triangles)
        if (!centre_removed(t)) out.triangles.push_back({map_node(t[0]), map_node(t[1]), map_node(t[2])});
    for (const auto& q : s.quads)
        if (!centre_removed(q)) out.quads.push_back({map_node(q[0]), map_node(q[1]), map_node(q[2]), map_node(q[3])});
    return out;
}

// Polygon area of the boundary loops (shoelace over the boundary sides).
inline double boundary_area(const Mesh& m) {
    double a = 0.0;
    for (const auto& s : m.iside) a += 0.5 * (m.x[s.node1] * m.y[s.node2] - m.x[s.node2] * m.y[s.node1]);
    return a;
}

inline std::vector<char> boundary_nodes(const Mesh& m) {
    std::vector<char> b(static_cast<std::size_t>(m.npoin), 0);
    for (const auto& s : m.iside) b[s.node1] = b[s.node2] = 1;
    return b;
}

} // namespace hydrama::test
