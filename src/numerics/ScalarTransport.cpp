#include "numerics/ScalarTransport.hpp"

#include <stdexcept>

namespace hydrama {

template <int N>
void assemble_scalar_transport(const Mesh& mesh,
                               const std::vector<double>& u,
                               const std::vector<double>& v,
                               const std::vector<double>& delte,
                               const std::array<TransportedScalar, N>& scalars) {
    static_assert(N >= 1, "at least one scalar");
    for (const TransportedScalar& s : scalars) {
        if (!s.phi || !s.diffusivity || !s.rhs) {
            throw std::invalid_argument("assemble_scalar_transport: null field");
        }
        s.rhs->assign(mesh.npoin, 0.0);
    }

    for (int ie = 0; ie < mesh.nelem; ++ie) {
        const int ipn[3] = {mesh.intma[ie][0], mesh.intma[ie][1], mesh.intma[ie][2]};
        const double lu[3] = {u[ipn[0]], u[ipn[1]], u[ipn[2]]};
        const double lv[3] = {v[ipn[0]], v[ipn[1]], v[ipn[2]]};

        const double* dNdx = mesh.dNkdx[ie].dNdx;
        const double* dNdy = mesh.dNkdx[ie].dNdy;
        const double ldetJ = mesh.detJ[ie];
        const double ldelte_half = 0.5 * delte[ie];

        // Velocity data shared by all scalars (same operation order as Step 1).
        const double lunksum_u = lu[0] + lu[1] + lu[2];
        const double lunksum_v = lv[0] + lv[1] + lv[2];
        const double umean_u = lunksum_u / 3.0;
        const double umean_v = lunksum_v / 3.0;
        const double lfact1 = ldetJ / 24.0;
        const double lfact2 = ldetJ * ldelte_half / 6.0;
        double u_adv[3];
        for (int in = 0; in < 3; ++in) u_adv[in] = dNdx[in] * umean_u + dNdy[in] * umean_v;

        for (const TransportedScalar& s : scalars) {
            const std::vector<double>& phi = *s.phi;
            const double lp[3] = {phi[ipn[0]], phi[ipn[1]], phi[ipn[2]]};
            const double gx = dNdx[0] * lp[0] + dNdx[1] * lp[1] + dNdx[2] * lp[2];
            const double gy = dNdy[0] * lp[0] + dNdy[1] * lp[1] + dNdy[2] * lp[2];

            const double gx1 = gx * lfact1;
            const double gy1 = gy * lfact1;
            const double stab = (lunksum_u * gx + lunksum_v * gy) * lfact2;
            const double lfact_diff = ldetJ * (*s.diffusivity)[ie] * 0.5;

            std::vector<double>& rhs = *s.rhs;
            for (int in = 0; in < 3; ++in) {
                double lrhs = 0.0;
                lrhs -= (lunksum_u + lu[in]) * gx1 + (lunksum_v + lv[in]) * gy1 + u_adv[in] * stab;
                lrhs -= (dNdx[in] * gx + dNdy[in] * gy) * lfact_diff;
                rhs[ipn[in]] += lrhs;
            }
        }
    }
}

template void assemble_scalar_transport<1>(const Mesh&, const std::vector<double>&, const std::vector<double>&,
                                           const std::vector<double>&, const std::array<TransportedScalar, 1>&);
template void assemble_scalar_transport<2>(const Mesh&, const std::vector<double>&, const std::vector<double>&,
                                           const std::vector<double>&, const std::array<TransportedScalar, 2>&);
template void assemble_scalar_transport<3>(const Mesh&, const std::vector<double>&, const std::vector<double>&,
                                           const std::vector<double>&, const std::array<TransportedScalar, 3>&);

void assemble_scalar_transport(const Mesh& mesh,
                               const std::vector<double>& u,
                               const std::vector<double>& v,
                               const std::vector<double>& delte,
                               const std::vector<double>& phi,
                               const std::vector<double>& diffusivity,
                               std::vector<double>& rhs) {
    assemble_scalar_transport<1>(mesh, u, v, delte, {TransportedScalar{&phi, &diffusivity, &rhs}});
}

} // namespace hydrama
