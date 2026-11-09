#include "Step1.hpp"

namespace hydrama {

void compute_step1(const Mesh& mesh,
                   double ani,
                   const std::vector<double>& delte,
                   const std::vector<double>& u,
                   const std::vector<double>& v,
                   std::vector<double>& rhs_u,
                   std::vector<double>& rhs_v) {
    rhs_u.assign(mesh.npoin, 0.0);
    rhs_v.assign(mesh.npoin, 0.0);

    for (int ie = 0; ie < mesh.nelem; ++ie) {
        int ipn[3] = {mesh.intma[ie][0], mesh.intma[ie][1], mesh.intma[ie][2]};
        double lu[3] = {u[ipn[0]], u[ipn[1]], u[ipn[2]]};
        double lv[3] = {v[ipn[0]], v[ipn[1]], v[ipn[2]]};

        const double* dNdx = mesh.dNkdx[ie].dNdx;
        const double* dNdy = mesh.dNkdx[ie].dNdy;
        double ldetJ = mesh.detJ[ie];
        double ldelte_half = 0.5 * delte[ie];

        // Velocity spatial gradients: dNudx(i, j) = du_i / dx_j
        // i=0 -> u, i=1 -> v
        // j=0 -> x, j=1 -> y
        double dNudx_ux = dNdx[0] * lu[0] + dNdx[1] * lu[1] + dNdx[2] * lu[2];
        double dNudx_vx = dNdx[0] * lv[0] + dNdx[1] * lv[1] + dNdx[2] * lv[2];
        double dNudx_uy = dNdy[0] * lu[0] + dNdy[1] * lu[1] + dNdy[2] * lu[2];
        double dNudx_vy = dNdy[0] * lv[0] + dNdy[1] * lv[1] + dNdy[2] * lv[2];

        // --- Convection & CBS Stabilisation ---
        double lunksum_u = lu[0] + lu[1] + lu[2];
        double lunksum_v = lv[0] + lv[1] + lv[2];
        double umean_u = lunksum_u / 3.0;
        double umean_v = lunksum_v / 3.0;

        double lfact1 = ldetJ / 24.0;
        double dNudx2_ux = dNudx_ux * lfact1;
        double dNudx2_uy = dNudx_uy * lfact1;
        double dNudx2_vx = dNudx_vx * lfact1;
        double dNudx2_vy = dNudx_vy * lfact1;

        double lunksum2_u = lunksum_u * dNudx_ux + lunksum_v * dNudx_uy;
        double lunksum2_v = lunksum_u * dNudx_vx + lunksum_v * dNudx_vy;

        double lfact2 = ldetJ * ldelte_half / 6.0;
        lunksum2_u *= lfact2;
        lunksum2_v *= lfact2;

        double lrhs_u[3] = {0.0, 0.0, 0.0};
        double lrhs_v[3] = {0.0, 0.0, 0.0};

        for (int in = 0; in < 3; ++in) {
            double u_adv = dNdx[in] * umean_u + dNdy[in] * umean_v;
            lrhs_u[in] -= (lunksum_u + lu[in]) * dNudx2_ux
                        + (lunksum_v + lv[in]) * dNudx2_uy
                        + u_adv * lunksum2_u;

            lrhs_v[in] -= (lunksum_u + lu[in]) * dNudx2_vx
                        + (lunksum_v + lv[in]) * dNudx2_vy
                        + u_adv * lunksum2_v;
        }

        // --- Viscous Diffusion ---
        double lfact_diff = ldetJ * ani * 0.5;
        for (int in = 0; in < 3; ++in) {
            lrhs_u[in] -= (dNdx[in] * dNudx_ux + dNdy[in] * dNudx_uy) * lfact_diff;
            lrhs_v[in] -= (dNdx[in] * dNudx_vx + dNdy[in] * dNudx_vy) * lfact_diff;
        }

        // Assemble into global RHS
        for (int in = 0; in < 3; ++in) {
            int ip = ipn[in];
            rhs_u[ip] += lrhs_u[in];
            rhs_v[ip] += lrhs_v[in];
        }
    }
}

} // namespace hydrama
