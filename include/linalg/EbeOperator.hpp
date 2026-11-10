#pragma once

// Matrix-free element-by-element (EBE) operator of a symmetric P1 element
// matrix with the assembled diagonal stored per node:
//
//     y = diag(pdiag) x + sum_e (off-diagonal couplings of element e) x
//
// gstif[e] holds the three off-diagonal entries of element e for the local
// node pairs (0,1), (0,2), (1,2). This is the storage produced by
// update_lhs() for the pressure Poisson operator, and the operator needs only
// O(nelem + npoin) memory with no global matrix assembly.
//
// The operator is a non-owning view: the mesh and the coefficient arrays
// must outlive it. Their contents may change between applications (e.g. a new
// time step size); the sizes must not.

#include "Mesh.hpp"
#include "linalg/LinearOperator.hpp"

#include <array>
#include <vector>

namespace hydrama::linalg {

class EbeOperator final : public LinearOperator {
public:
    // Throws std::invalid_argument if gstif/pdiag do not match the mesh.
    EbeOperator(const Mesh& mesh,
                const std::vector<std::array<double, 3>>& gstif,
                const std::vector<double>& pdiag);

    std::size_t size() const override { return static_cast<std::size_t>(mesh_->npoin); }
    void apply(ConstVectorView x, VectorView y) const override;
    void diagonal(VectorView d) const override;
    OperatorTraits traits() const override { return {true}; }
    std::string_view name() const override { return "ebe"; }

private:
    const Mesh* mesh_;
    const std::vector<std::array<double, 3>>* gstif_;
    const std::vector<double>* pdiag_;
};

// The EBE product kernel shared by EbeOperator and matrix_vector_mult():
// y = diag(pdiag) x + sum_e couplings(gstif[e]) x. x and y must hold npoin
// entries and must not alias; y is overwritten.
void ebe_apply(const Mesh& mesh,
               const std::array<double, 3>* gstif,
               const double* pdiag,
               const double* x,
               double* y);

} // namespace hydrama::linalg
