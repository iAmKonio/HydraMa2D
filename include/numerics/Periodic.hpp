#pragma once

#include "Mesh.hpp"
#include "linalg/LinearOperator.hpp"
#include "linalg/VectorView.hpp"

#include <array>
#include <optional>
#include <vector>

namespace hydrama {

struct PeriodicMapping {
    int num_nodes = 0;
    int num_independent = 0;
    std::vector<int> to_indep;                  // size num_nodes: maps full node -> [0, num_independent - 1]
    std::vector<int> from_indep;                // size num_independent: maps indep -> canonical master node
    std::vector<std::vector<int>> slave_groups; // size num_independent: all full nodes in this DOF group
    std::vector<int> periodic_slaves;           // all unique slave nodes
    bool has_periodic = false;

    static PeriodicMapping build(const Mesh& mesh);
};

namespace linalg {

class PeriodicReducedOperator final : public LinearOperator {
public:
    PeriodicReducedOperator(const LinearOperator& full_op, const PeriodicMapping& map);

    std::size_t size() const override { return static_cast<std::size_t>(map_.num_independent); }
    void apply(ConstVectorView x_hat, VectorView y_hat) const override;
    void diagonal(VectorView d_hat) const override;
    OperatorTraits traits() const override;
    std::string_view name() const override { return "periodic_reduced"; }

    const LinearOperator& base() const { return full_op_; }
    const PeriodicMapping& mapping() const { return map_; }

private:
    const LinearOperator& full_op_;
    const PeriodicMapping& map_;
    mutable std::vector<double> full_x_;
    mutable std::vector<double> full_y_;
};

} // namespace linalg

// Enforces exact periodic consistency on field values across all periodic pairs
// using lumped mass weighting:
// field[master] = field[slaves] = (sum M_i * field[i]) / (sum M_i)
void apply_periodic_field(const Mesh& mesh,
                          const std::vector<double>& elcoe2,
                          std::vector<double>& field);

// Geometric matching between slave and master boundary nodes.
// If translation is not provided, it is deduced from the centroid difference.
// Returns the number of matched node pairs.
int match_periodic_boundary_nodes(Mesh& mesh,
                                  const std::vector<int>& slave_nodes,
                                  const std::vector<int>& master_nodes,
                                  std::optional<std::array<double, 2>> translation = std::nullopt,
                                  double tolerance = 1.0e-4);

} // namespace hydrama
