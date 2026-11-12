#include "mesh/MeshCheck.hpp"

#include <algorithm>
#include <cstdint>
#include <map>

namespace hydrama::mesh {

std::vector<std::string> check_mesh(const Mesh& m, std::size_t max_problems) {
    std::vector<std::string> problems;
    bool truncated = false;
    auto add = [&](const std::string& msg) {
        if (problems.size() < max_problems) problems.push_back(msg);
        else truncated = true;
    };

    if (m.npoin < 0 || m.nelem < 0 || m.nboun < 0 || static_cast<int>(m.x.size()) != m.npoin ||
        static_cast<int>(m.y.size()) != m.npoin || m.num_triangles() + m.num_quads() != m.nelem ||
        static_cast<int>(m.iside.size()) != m.nboun) {
        add("array sizes do not match npoin / nelem / nboun");
        return problems;
    }
    auto node_ok = [&](int n) { return n >= 0 && n < m.npoin; };

    // 1.-2. Elements.
    for (int e = 0; e < m.num_triangles(); ++e) {
        const auto& n = m.intma[static_cast<std::size_t>(e)];
        if (!node_ok(n[0]) || !node_ok(n[1]) || !node_ok(n[2])) {
            add("element " + std::to_string(e) + " references a node out of range");
            continue;
        }
        if (n[0] == n[1] || n[0] == n[2] || n[1] == n[2]) {
            add("element " + std::to_string(e) + " has repeated nodes");
            continue;
        }
        const double twice_area = (m.x[n[1]] - m.x[n[0]]) * (m.y[n[2]] - m.y[n[0]]) -
                                  (m.x[n[2]] - m.x[n[0]]) * (m.y[n[1]] - m.y[n[0]]);
        if (!(twice_area > 0.0)) add("element " + std::to_string(e) + " is not counter-clockwise (signed area <= 0)");
    }
    for (int e = m.num_triangles(); e < m.nelem; ++e) {
        const ElementNodes n = m.element_nodes(e);
        bool ok = true;
        for (int a = 0; a < 4; ++a) ok = ok && node_ok(n[a]);
        if (!ok) {
            add("element " + std::to_string(e) + " references a node out of range");
            continue;
        }
        bool distinct = true;
        for (int a = 0; a < 4; ++a)
            for (int b = a + 1; b < 4; ++b) distinct = distinct && n[a] != n[b];
        if (!distinct) {
            add("element " + std::to_string(e) + " has repeated nodes");
            continue;
        }
        const double qx[4] = {m.x[n[0]], m.x[n[1]], m.x[n[2]], m.x[n[3]]};
        const double qy[4] = {m.y[n[0]], m.y[n[1]], m.y[n[2]], m.y[n[3]]};
        for (double c : q1::corner_jacobians(qx, qy)) {
            if (!(c > 0.0)) {
                add("element " + std::to_string(e) +
                    " (quadrilateral) has a non-positive corner Jacobian (clockwise, non-convex or degenerate)");
                break;
            }
        }
    }
    if (!problems.empty()) return problems;

    // 3. Edges: directed occurrences per undirected edge.
    struct Edge {
        int count = 0;
        int elem = -1, from = -1, to = -1;
    };
    std::map<std::pair<int, int>, Edge> edges;
    for (int e = 0; e < m.nelem; ++e) {
        const ElementNodes n = m.element_nodes(e);
        for (int k = 0; k < n.size(); ++k) {
            const int a = n[k], b = n[(k + 1) % n.size()];
            Edge& r = edges[{std::min(a, b), std::max(a, b)}];
            if (++r.count == 1) {
                r.elem = e;
                r.from = a;
                r.to = b;
            } else if (r.count == 2 && r.from == a) {
                add("elements " + std::to_string(r.elem) + " and " + std::to_string(e) +
                    " traverse their common edge in the same direction (overlap)");
            } else if (r.count > 2) {
                add("edge " + std::to_string(a) + "-" + std::to_string(b) + " is shared by more than two elements");
            }
        }
    }

    // 4.-5. Boundary sides.
    std::map<std::pair<int, int>, int> covered;
    for (int i = 0; i < m.nboun; ++i) {
        const BoundarySide& s = m.iside[static_cast<std::size_t>(i)];
        const std::string label = "boundary side " + std::to_string(i);
        if (!node_ok(s.node1) || !node_ok(s.node2) || s.elem < 0 || s.elem >= m.nelem) {
            add(label + " references a node or element out of range");
            continue;
        }
        const ElementNodes n = m.element_nodes(s.elem);
        int k1 = -1, k2 = -1;
        for (int k = 0; k < n.size(); ++k) {
            if (n[k] == s.node1) k1 = k;
            if (n[k] == s.node2) k2 = k;
        }
        const int nn = n.size();
        if (k1 < 0 || k2 < 0 || k1 == k2 || ((k1 + 1) % nn != k2 && (k2 + 1) % nn != k1)) {
            add(label + " is not an edge of its element " + std::to_string(s.elem));
            continue;
        }
        if ((k1 + 1) % nn != k2) add(label + " is not oriented counter-clockwise with respect to its element");
        const auto key = std::make_pair(std::min(s.node1, s.node2), std::max(s.node1, s.node2));
        const auto it = edges.find(key);
        if (it != edges.end() && it->second.count != 1) add(label + " lies on an interior edge");
        if (++covered[key] == 2) add(label + " duplicates another boundary side");
    }
    int uncovered = 0;
    for (const auto& [key, r] : edges)
        if (r.count == 1 && !covered.count(key)) ++uncovered;
    if (uncovered > 0) add(std::to_string(uncovered) + " boundary edge(s) are not covered by a boundary side");

    if (truncated) problems.push_back("... (more problems)");
    return problems;
}

} // namespace hydrama::mesh
