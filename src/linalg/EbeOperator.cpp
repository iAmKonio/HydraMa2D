#include "linalg/EbeOperator.hpp"

#include "linalg/VectorOps.hpp"
#include "numerics/OpenMP.hpp"

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

    const int ntri = mesh.num_triangles();
    for (int ie = 0; ie < ntri; ++ie) {
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

    // Quadrilaterals: two blocks per element, pairs (0,1), (0,2), (0,3) and
    // (1,2), (1,3), (2,3).
    for (int q = 0; q < mesh.num_quads(); ++q) {
        const auto& n = mesh.quads[static_cast<std::size_t>(q)];
        const auto& b0 = gstif[ntri + 2 * q];
        const auto& b1 = gstif[ntri + 2 * q + 1];
        const double k01 = b0[0], k02 = b0[1], k03 = b0[2], k12 = b1[0], k13 = b1[1], k23 = b1[2];
        const double x0 = x[n[0]], x1 = x[n[1]], x2 = x[n[2]], x3 = x[n[3]];
        y[n[0]] += k01 * x1 + k02 * x2 + k03 * x3;
        y[n[1]] += k01 * x0 + k12 * x2 + k13 * x3;
        y[n[2]] += k02 * x0 + k12 * x1 + k23 * x3;
        y[n[3]] += k03 * x0 + k13 * x1 + k23 * x2;
    }

    #pragma omp parallel for schedule(static)
    for (int ip = 0; ip < npoin; ++ip) {
        y[ip] += x[ip] * pdiag[ip];
    }
}

EbeOperator::EbeOperator(const Mesh& mesh,
                         const std::vector<std::array<double, 3>>& gstif,
                         const std::vector<double>& pdiag)
    : mesh_(&mesh), gstif_(&gstif), pdiag_(&pdiag) {
    if (static_cast<int>(gstif.size()) != mesh.num_offdiag_blocks()) {
        throw std::invalid_argument("EbeOperator: gstif has " + std::to_string(gstif.size()) + " entries, mesh has " +
                                    std::to_string(mesh.nelem) + " elements (" +
                                    std::to_string(mesh.num_offdiag_blocks()) + " off-diagonal blocks)");
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
