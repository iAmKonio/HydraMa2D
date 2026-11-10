#include "linalg/Reordering.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace hydrama::linalg {

namespace {

// Adjacency graph of A + A^T without self-loops, neighbour lists sorted by
// (degree, index): the visiting order of the Cuthill-McKee search.
struct Graph {
    std::vector<int> ptr;
    std::vector<int> adj;
    std::vector<int> degree;
    int size() const { return static_cast<int>(degree.size()); }
};

Graph build_graph(const SparsityPattern& p) {
    if (!p.square()) throw std::invalid_argument("reordering: square pattern required");
    const int n = p.rows();
    const auto& rp = p.row_ptr();
    const auto& ci = p.col_idx();

    std::vector<std::vector<int>> nbr(n);
    for (int i = 0; i < n; ++i) {
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
            const int j = ci[k];
            if (j == i) continue;
            nbr[i].push_back(j);
            if (!p.structurally_symmetric()) nbr[j].push_back(i);
        }
    }
    Graph g;
    g.ptr.assign(n + 1, 0);
    g.degree.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        auto& v = nbr[i];
        if (!p.structurally_symmetric()) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
        g.degree[i] = static_cast<int>(v.size());
        g.ptr[i + 1] = g.ptr[i] + g.degree[i];
    }
    g.adj.reserve(g.ptr[n]);
    for (int i = 0; i < n; ++i) {
        auto& v = nbr[i];
        std::sort(v.begin(), v.end(), [&g](int a, int b) {
            return g.degree[a] != g.degree[b] ? g.degree[a] < g.degree[b] : a < b;
        });
        g.adj.insert(g.adj.end(), v.begin(), v.end());
    }
    return g;
}

// Rooted level structure L(root) of root's connected component:
// level l = nodes[level_ptr[l] .. level_ptr[l+1]).
struct LevelStructure {
    std::vector<int> nodes;
    std::vector<int> level_ptr;
    int height() const { return static_cast<int>(level_ptr.size()) - 1; }
};

class LevelBuilder {
public:
    explicit LevelBuilder(const Graph& g) : g_(g), mark_(g.size(), 0) {}

    void build(int root, LevelStructure& ls) {
        ++stamp_;
        ls.nodes.clear();
        ls.level_ptr.assign(1, 0);
        ls.nodes.push_back(root);
        mark_[root] = stamp_;
        std::size_t begin = 0;
        while (begin < ls.nodes.size()) {
            const std::size_t end = ls.nodes.size();
            ls.level_ptr.push_back(static_cast<int>(end));
            for (std::size_t q = begin; q < end; ++q) {
                const int v = ls.nodes[q];
                for (int k = g_.ptr[v]; k < g_.ptr[v + 1]; ++k) {
                    const int w = g_.adj[k];
                    if (mark_[w] != stamp_) {
                        mark_[w] = stamp_;
                        ls.nodes.push_back(w);
                    }
                }
            }
            begin = end;
        }
    }

private:
    const Graph& g_;
    std::vector<int> mark_;
    int stamp_ = 0;
};

// George & Liu (1979): pseudo-peripheral node of start's component.
int find_pseudo_peripheral(const Graph& g, LevelBuilder& levels, int start) {
    LevelStructure ls;
    int root = start;
    levels.build(root, ls);
    for (;;) {
        // Minimum-degree node of the last level (ties: lowest index).
        const int h = ls.height();
        int candidate = -1;
        for (int q = ls.level_ptr[h - 1]; q < ls.level_ptr[h]; ++q) {
            const int v = ls.nodes[q];
            if (candidate < 0 || g.degree[v] < g.degree[candidate] ||
                (g.degree[v] == g.degree[candidate] && v < candidate)) {
                candidate = v;
            }
        }
        LevelStructure next;
        levels.build(candidate, next);
        if (next.height() > h) {
            root = candidate;
            ls = std::move(next);
        } else {
            return root;
        }
    }
}

std::vector<int> cuthill_mckee_order(const SparsityPattern& pattern) {
    const Graph g = build_graph(pattern);
    const int n = g.size();
    LevelBuilder levels(g);
    std::vector<char> numbered(n, 0);
    std::vector<int> order;
    order.reserve(n);
    for (int s = 0; s < n; ++s) {
        if (numbered[s]) continue;
        const int root = find_pseudo_peripheral(g, levels, s);
        std::size_t q = order.size();
        order.push_back(root);
        numbered[root] = 1;
        for (; q < order.size(); ++q) {
            const int v = order[q];
            for (int k = g.ptr[v]; k < g.ptr[v + 1]; ++k) {
                const int w = g.adj[k];
                if (!numbered[w]) {
                    numbered[w] = 1;
                    order.push_back(w);
                }
            }
        }
    }
    return order;
}

} // namespace

Permutation cuthill_mckee(const SparsityPattern& pattern) {
    return Permutation(cuthill_mckee_order(pattern));
}

Permutation reverse_cuthill_mckee(const SparsityPattern& pattern) {
    std::vector<int> order = cuthill_mckee_order(pattern);
    std::reverse(order.begin(), order.end());
    return Permutation(std::move(order));
}

int pseudo_peripheral_node(const SparsityPattern& pattern, int start) {
    const Graph g = build_graph(pattern);
    if (start < 0 || start >= g.size()) throw std::out_of_range("pseudo_peripheral_node: start out of range");
    LevelBuilder levels(g);
    return find_pseudo_peripheral(g, levels, start);
}

const std::vector<std::string_view>& ordering_names() {
    static const std::vector<std::string_view> names = {"none", "rcm"};
    return names;
}

bool is_ordering(std::string_view name) {
    const auto& names = ordering_names();
    return std::find(names.begin(), names.end(), name) != names.end();
}

Permutation compute_ordering(std::string_view name, const SparsityPattern& pattern) {
    if (name == "none") return Permutation::identity(pattern.rows());
    if (name == "rcm") return reverse_cuthill_mckee(pattern);
    throw std::invalid_argument("unknown ordering '" + std::string(name) + "'");
}

} // namespace hydrama::linalg
