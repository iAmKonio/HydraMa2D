#pragma once

// "Compiled" Dirichlet boundary condition for a nodal scalar: a flat list of
// (node, value) pairs resolved once at start-up, applied each step in a tight
// loop. Built with DirichletBuilder, where a later set() on the same node
// overrides an earlier one (e.g. walls after inlets, so that corners belong
// to the wall - the same precedence as apply_velocity_bcs()).

#include <cstddef>
#include <vector>

namespace hydrama {

struct DirichletSet {
    std::vector<int> nodes;     // ascending, unique
    std::vector<double> values;

    bool empty() const noexcept { return nodes.empty(); }
    int size() const noexcept { return static_cast<int>(nodes.size()); }

    void apply(std::vector<double>& phi) const {
        for (std::size_t k = 0; k < nodes.size(); ++k) phi[nodes[k]] = values[k];
    }
};

class DirichletBuilder {
public:
    explicit DirichletBuilder(int npoin) : value_(npoin, 0.0), set_(npoin, 0) {}

    void set(int node, double value) {
        value_[node] = value;
        set_[node] = 1;
    }
    bool is_set(int node) const { return set_[node] != 0; }

    DirichletSet build() const {
        DirichletSet s;
        for (int ip = 0; ip < static_cast<int>(set_.size()); ++ip) {
            if (set_[ip]) {
                s.nodes.push_back(ip);
                s.values.push_back(value_[ip]);
            }
        }
        return s;
    }

private:
    std::vector<double> value_;
    std::vector<char> set_;
};

} // namespace hydrama
