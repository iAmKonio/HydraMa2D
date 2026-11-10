#pragma once

// Mesh builders for wall-geometry and wall-distance tests:
//  * boundary extraction from the element list (edges used by one element),
//    oriented with the domain on the left for counter-clockwise elements,
//  * a backward-facing step (structured grid minus a rectangular block),
//  * coordinate mappings (stretching, rotation) that keep the topology,
//  * the bundled Hinterkante benchmark mesh and its analytic wall distance.

#include "IO.hpp"
#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace hydrama::test {

using FlagOf = std::function<int(double xm, double ym)>; // flag from the edge midpoint

// Boundary sides = element edges that belong to exactly one element. Each side
// keeps the node order of its element (counter-clockwise => domain on the
// left). side_id is set to the flag, which keeps .bofl-style lookups simple.
inline std::vector<BoundarySide> extract_boundary(const std::vector<Point>& nodes,
                                                  const std::vector<std::array<int, 3>>& elems,
                                                  const FlagOf& flag_of) {
    struct EdgeUse {
        int count = 0;
        int a = -1, b = -1, elem = -1;
    };
    std::map<std::pair<int, int>, EdgeUse> edges; // ordered => deterministic side order
    for (int ie = 0; ie < static_cast<int>(elems.size()); ++ie) {
        for (int k = 0; k < 3; ++k) {
            const int a = elems[ie][k], b = elems[ie][(k + 1) % 3];
            EdgeUse& use = edges[{std::min(a, b), std::max(a, b)}];
            ++use.count;
            use.a = a;
            use.b = b;
            use.elem = ie;
        }
    }
    std::vector<BoundarySide> sides;
    for (const auto& [key, use] : edges) {
        if (use.count != 1) continue;
        const double xm = 0.5 * (nodes[use.a].x + nodes[use.b].x);
        const double ym = 0.5 * (nodes[use.a].y + nodes[use.b].y);
        const int flag = flag_of(xm, ym);
        sides.push_back({use.a, use.b, use.elem, flag, flag});
    }
    return sides;
}

// Backward-facing step: the rectangle [0, lx] x [0, ly] without the block
// [0, xs] x [0, hs]. The grid has nx * ny square-split cells; xs and hs must
// lie on grid lines. Boundaries: x = 0 inlet (503), x = lx outlet (504), all
// others walls (500): step top, step face, bottom, top.
struct StepSpec {
    double lx = 4.0, ly = 1.5;
    double xs = 1.0, hs = 0.5;
    int nx = 32, ny = 12;
};

inline Mesh make_step_mesh(const StepSpec& s) {
    GridSpec g;
    g.lx = s.lx;
    g.ly = s.ly;
    g.nx = s.nx;
    g.ny = s.ny;
    const Mesh full = make_grid_mesh(g);

    // Keep the cells whose centre lies outside the block.
    std::vector<int> new_id(full.npoin, -1);
    std::vector<Point> nodes;
    std::vector<std::array<int, 3>> elems;
    for (int ie = 0; ie < full.nelem; ++ie) {
        const int cell = ie / 2;
        const int i = cell % g.nx, j = cell / g.nx;
        const double xc = (i + 0.5) * g.hx(), yc = (j + 0.5) * g.hy();
        if (xc < s.xs && yc < s.hs) continue;
        std::array<int, 3> e{};
        for (int k = 0; k < 3; ++k) {
            const int ip = full.intma[ie][k];
            if (new_id[ip] < 0) {
                new_id[ip] = static_cast<int>(nodes.size());
                nodes.push_back({full.x[ip], full.y[ip]});
            }
            e[k] = new_id[ip];
        }
        elems.push_back(e);
    }
    const double tol = 1e-9 * std::max(s.lx, s.ly);
    auto sides = extract_boundary(nodes, elems, [&](double xm, double /*ym*/) {
        if (std::abs(xm) < tol) return 503;
        if (std::abs(xm - s.lx) < tol) return 504;
        return 500;
    });
    return make_mesh(nodes, elems, sides);
}

// Analytic distance to the step walls (see make_step_mesh):
//   step top y = hs (0 <= x <= xs), step face x = xs (0 <= y <= hs),
//   bottom y = 0 (x >= xs), top y = ly.
inline double step_wall_distance(const StepSpec& s, double x, double y) {
    // Fluid points satisfy x >= xs or y >= hs.
    double lower;
    if (y >= s.hs && x <= s.xs) {
        lower = y - s.hs; // above the step: straight down to the step top
    } else if (y >= s.hs) {
        lower = std::min(y, std::hypot(x - s.xs, y - s.hs)); // bottom or convex corner
    } else {
        lower = std::min(y, x - s.xs); // below the step top (x >= xs): bottom or step face
    }
    return std::min(lower, s.ly - y);
}

// Apply a coordinate mapping to every node (topology unchanged).
template <typename F>
void map_nodes(Mesh& m, F f) {
    for (int ip = 0; ip < m.npoin; ++ip) {
        const auto [x, y] = f(m.x[ip], m.y[ip]);
        m.x[ip] = x;
        m.y[ip] = y;
    }
}

// --- Bundled benchmark mesh ------------------------------------------------

inline const Mesh& benchmark_mesh() {
    static const Mesh mesh = [] {
        const std::string dir = HYDRAMA2D_CASE_DIR;
        Mesh m;
        REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", m));
        REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", m));
        return m;
    }();
    return mesh;
}

// Exact distance to the continuous walls of the benchmark: channel walls
// y = +-5 and the plate [-4, -0.5] x [-0.5, 0.5] with a semicircular trailing
// edge of radius 0.5 centred at (-0.5, 0).
struct BenchmarkDistance {
    double flat = 0.0;     // distance to the straight walls
    double circle = 0.0;   // distance to the semicircle (inf where not applicable)
    double exact() const { return std::min(flat, circle); }
};

inline BenchmarkDistance benchmark_wall_distance(double x, double y) {
    BenchmarkDistance d;
    d.flat = std::min(5.0 - y, y + 5.0);
    d.circle = std::numeric_limits<double>::infinity();
    if (x <= -0.5) {
        d.flat = std::min(d.flat, std::abs(y) - 0.5);
    } else {
        d.circle = std::hypot(x + 0.5, y) - 0.5;
    }
    return d;
}

} // namespace hydrama::test
