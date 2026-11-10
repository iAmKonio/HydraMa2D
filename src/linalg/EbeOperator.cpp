#include "linalg/EbeOperator.hpp"

#include "linalg/VectorOps.hpp"

#include <stdexcept>
#include <string>

namespace hydrama::linalg {

void ebe_apply(const Mesh& mesh,
               const std::array<double, 3>* gstif,
               const double* pdiag,
               const double* x,
               double* y) {
    const int npoin = mesh.npoin;
    for (int ip = 0; ip < npoin; ++ip) y[ip] = 0.0;

    for (int ie = 0; ie < mesh.nelem; ++ie) {
        const int n0 = mesh.intma[ie][0];
        const int n1 = mesh.intma[ie][1];
        const int n2 = mesh.intma[ie][2];

        const double k0 = gstif[ie][0]; // pair (0, 1)
        const double k1 = gstif[ie][1]; // pair (0, 2)
        const double k2 = gstif[ie][2]; // pair (1, 2)

        y[n0] += k0 * x[n1] + k1 * x[n2];
        y[n1] += k0 * x[n0] + k2 * x[n2];
        y[n2] += k1 * x[n0] + k2 * x[n1];
    }

    for (int ip = 0; ip < npoin; ++ip) {
        y[ip] += x[ip] * pdiag[ip];
    }
}

EbeOperator::EbeOperator(const Mesh& mesh,
                         const std::vector<std::array<double, 3>>& gstif,
                         const std::vector<double>& pdiag)
    : mesh_(&mesh), gstif_(&gstif), pdiag_(&pdiag) {
    if (static_cast<int>(gstif.size()) != mesh.nelem) {
        throw std::invalid_argument("EbeOperator: gstif has " + std::to_string(gstif.size()) +
                                    " entries, mesh has " + std::to_string(mesh.nelem) + " elements");
    }
    if (static_cast<int>(pdiag.size()) != mesh.npoin) {
        throw std::invalid_argument("EbeOperator: pdiag has " + std::to_string(pdiag.size()) +
                                    " entries, mesh has " + std::to_string(mesh.npoin) + " nodes");
    }
}

void EbeOperator::apply(ConstVectorView x, VectorView y) const {
    if (x.size() != size() || y.size() != size()) {
        throw std::invalid_argument("EbeOperator::apply: vector size does not match the operator");
    }
    ebe_apply(*mesh_, gstif_->data(), pdiag_->data(), x.data(), y.data());
}

void EbeOperator::diagonal(VectorView d) const {
    copy(*pdiag_, d);
}

} // namespace hydrama::linalg
