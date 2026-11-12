#include "numerics/Periodic.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace hydrama {

PeriodicMapping PeriodicMapping::build(const Mesh& mesh) {
    PeriodicMapping m;
    m.num_nodes = mesh.npoin;
    m.has_periodic = mesh.has_periodic();

    if (!m.has_periodic) {
        m.num_independent = mesh.npoin;
        m.to_indep.resize(mesh.npoin);
        m.from_indep.resize(mesh.npoin);
        m.slave_groups.resize(mesh.npoin);
        for (int i = 0; i < mesh.npoin; ++i) {
            m.to_indep[i] = i;
            m.from_indep[i] = i;
            m.slave_groups[i] = {i};
        }
        return m;
    }

    m.to_indep.assign(mesh.npoin, -1);
    int indep_count = 0;
    for (int i = 0; i < mesh.npoin; ++i) {
        if (mesh.periodic_master_map[i] == i) {
            m.to_indep[i] = indep_count++;
            m.from_indep.push_back(i);
        }
    }

    for (int i = 0; i < mesh.npoin; ++i) {
        if (mesh.periodic_master_map[i] != i) {
            int root = mesh.periodic_master_map[i];
            m.to_indep[i] = m.to_indep[root];
            m.periodic_slaves.push_back(i);
        }
    }
    m.num_independent = indep_count;

    m.slave_groups.resize(m.num_independent);
    for (int i = 0; i < mesh.npoin; ++i) {
        m.slave_groups[m.to_indep[i]].push_back(i);
    }

    return m;
}

namespace linalg {

PeriodicReducedOperator::PeriodicReducedOperator(const LinearOperator& full_op, const PeriodicMapping& map)
    : full_op_(full_op), map_(map), full_x_(map.num_nodes, 0.0), full_y_(map.num_nodes, 0.0) {
    if (full_op.size() != static_cast<std::size_t>(map.num_nodes)) {
        throw std::invalid_argument("PeriodicReducedOperator: base operator size (" +
                                    std::to_string(full_op.size()) + ") does not match mapping node count (" +
                                    std::to_string(map.num_nodes) + ")");
    }
}

void PeriodicReducedOperator::apply(ConstVectorView x_hat, VectorView y_hat) const {
    const double* const xh = x_hat.data();
    for (int i = 0; i < map_.num_nodes; ++i) {
        full_x_[i] = xh[map_.to_indep[i]];
    }

    full_op_.apply(full_x_, full_y_);

    std::fill(y_hat.begin(), y_hat.end(), 0.0);
    double* const yh = y_hat.data();
    for (int i = 0; i < map_.num_nodes; ++i) {
        yh[map_.to_indep[i]] += full_y_[i];
    }
}

void PeriodicReducedOperator::diagonal(VectorView d_hat) const {
    std::vector<double> full_d(map_.num_nodes, 0.0);
    full_op_.diagonal(full_d);

    std::fill(d_hat.begin(), d_hat.end(), 0.0);
    double* const dh = d_hat.data();
    for (int i = 0; i < map_.num_nodes; ++i) {
        dh[map_.to_indep[i]] += full_d[i];
    }
}

OperatorTraits PeriodicReducedOperator::traits() const {
    OperatorTraits t = full_op_.traits();
    t.symmetric = true; // Z^T A Z is symmetric whenever A is symmetric
    return t;
}

} // namespace linalg

void apply_periodic_field(const Mesh& mesh,
                          const std::vector<double>& elcoe2,
                          std::vector<double>& field) {
    if (!mesh.has_periodic()) return;
    if (static_cast<int>(field.size()) != mesh.npoin) return;

    const bool has_elcoe2 = static_cast<int>(elcoe2.size()) == mesh.npoin;

    if (!mesh.periodic_groups.empty()) {
        for (const auto& grp : mesh.periodic_groups) {
            double sum_w = 0.0;
            double sum_m = 0.0;
            for (int node : grp) {
                double mass = (has_elcoe2 && elcoe2[node] > 0.0) ? (1.0 / elcoe2[node]) : 1.0;
                sum_w += field[node] * mass;
                sum_m += mass;
            }
            if (sum_m > 0.0) {
                const double avg = sum_w / sum_m;
                for (int node : grp) {
                    field[node] = avg;
                }
            }
        }
        return;
    }

    // Fallback if periodic_groups is not built
    for (const auto& pair : mesh.periodic_pairs) {
        int m = mesh.canonical_master(pair.master);
        int s = pair.slave;
        double mass_m = (has_elcoe2 && elcoe2[m] > 0.0) ? (1.0 / elcoe2[m]) : 1.0;
        double mass_s = (has_elcoe2 && elcoe2[s] > 0.0) ? (1.0 / elcoe2[s]) : 1.0;
        double sum_m = mass_m + mass_s;
        if (sum_m > 0.0) {
            double avg = (field[m] * mass_m + field[s] * mass_s) / sum_m;
            field[m] = avg;
            field[s] = avg;
        }
    }
}

int match_periodic_boundary_nodes(Mesh& mesh,
                                  const std::vector<int>& slave_nodes,
                                  const std::vector<int>& master_nodes,
                                  std::optional<std::array<double, 2>> translation,
                                  double tolerance) {
    if (slave_nodes.empty() || master_nodes.empty()) return 0;

    double dx = 0.0, dy = 0.0;
    if (translation.has_value()) {
        dx = (*translation)[0];
        dy = (*translation)[1];
    } else {
        // Auto-deduce translation from centroids:
        // dx = slave_centroid - master_centroid
        double sx = 0.0, sy = 0.0;
        for (int s : slave_nodes) {
            sx += mesh.x[s];
            sy += mesh.y[s];
        }
        sx /= static_cast<double>(slave_nodes.size());
        sy /= static_cast<double>(slave_nodes.size());

        double mx = 0.0, my = 0.0;
        for (int m : master_nodes) {
            mx += mesh.x[m];
            my += mesh.y[m];
        }
        mx /= static_cast<double>(master_nodes.size());
        my /= static_cast<double>(master_nodes.size());

        dx = sx - mx;
        dy = sy - my;
    }

    PeriodicBoundary pb;
    pb.translation = {dx, dy};

    for (int s : slave_nodes) {
        const double target_x = mesh.x[s] - dx;
        const double target_y = mesh.y[s] - dy;

        int best_m = -1;
        double best_dist2 = std::numeric_limits<double>::infinity();

        for (int m : master_nodes) {
            const double d2 = (mesh.x[m] - target_x) * (mesh.x[m] - target_x) +
                              (mesh.y[m] - target_y) * (mesh.y[m] - target_y);
            if (d2 < best_dist2) {
                best_dist2 = d2;
                best_m = m;
            }
        }

        if (best_m >= 0 && std::sqrt(best_dist2) <= tolerance) {
            pb.pairs.push_back({s, best_m});
        }
    }

    if (pb.pairs.empty()) return 0;

    const int matched_count = static_cast<int>(pb.pairs.size());
    mesh.periodic_boundaries.push_back(std::move(pb));

    // Rebuild periodic_master_map
    if (mesh.periodic_master_map.size() != static_cast<std::size_t>(mesh.npoin)) {
        mesh.periodic_master_map.resize(mesh.npoin);
        for (int i = 0; i < mesh.npoin; ++i) mesh.periodic_master_map[i] = i;
    }

    std::function<int(int)> find_root = [&](int i) {
        if (mesh.periodic_master_map[i] == i) return i;
        return mesh.periodic_master_map[i] = find_root(mesh.periodic_master_map[i]);
    };

    for (const auto& boundary : mesh.periodic_boundaries) {
        for (const auto& p : boundary.pairs) {
            int rm = find_root(p.master);
            int rs = find_root(p.slave);
            if (rs != rm) {
                mesh.periodic_master_map[rs] = rm;
            }
        }
    }

    mesh.periodic_pairs.clear();
    mesh.periodic_slave_nodes.clear();
    mesh.periodic_groups.clear();
    std::map<int, std::vector<int>> master_to_slaves;
    for (int i = 0; i < mesh.npoin; ++i) {
        mesh.periodic_master_map[i] = find_root(i);
        if (mesh.periodic_master_map[i] != i) {
            int root = mesh.periodic_master_map[i];
            mesh.periodic_slave_nodes.push_back(i);
            mesh.periodic_pairs.push_back({i, root});
            master_to_slaves[root].push_back(i);
        }
    }
    for (auto& [master, slaves] : master_to_slaves) {
        std::vector<int> grp;
        grp.reserve(1 + slaves.size());
        grp.push_back(master);
        grp.insert(grp.end(), slaves.begin(), slaves.end());
        mesh.periodic_groups.push_back(std::move(grp));
    }

    return matched_count;
}

} // namespace hydrama
