#include "Step2.hpp"

namespace hydrama {

void compute_step2(const Mesh& mesh,
                   const std::vector<double>& u_star,
                   const std::vector<double>& v_star,
                   const std::vector<double>& u_n,
                   const std::vector<double>& v_n,
                   double theta1,
                   std::vector<double>& rhs1) {
    rhs1.assign(mesh.npoin, 0.0);

    for (int ie = 0; ie < mesh.nelem; ++ie) {
        int ipn[3] = {mesh.intma[ie][0], mesh.intma[ie][1], mesh.intma[ie][2]};

        double lu_star[3] = {u_star[ipn[0]], u_star[ipn[1]], u_star[ipn[2]]};
        double lv_star[3] = {v_star[ipn[0]], v_star[ipn[1]], v_star[ipn[2]]};

        double lu_n[3] = {u_n[ipn[0]], u_n[ipn[1]], u_n[ipn[2]]};
        double lv_n[3] = {v_n[ipn[0]], v_n[ipn[1]], v_n[ipn[2]]};

        double ldiff_u[3] = {lu_star[0] - lu_n[0], lu_star[1] - lu_n[1], lu_star[2] - lu_n[2]};
        double ldiff_v[3] = {lv_star[0] - lv_n[0], lv_star[1] - lv_n[1], lv_star[2] - lv_n[2]};

        const double* dNdx = mesh.dNkdx[ie].dNdx;
        const double* dNdy = mesh.dNkdx[ie].dNdy;
        double ldetJ = mesh.detJ[ie];

        // Term 1: - \int N_k (\nabla \cdot u^n) d\Omega
        double div_u_n = (dNdx[0] * lu_n[0] + dNdx[1] * lu_n[1] + dNdx[2] * lu_n[2])
                       + (dNdy[0] * lv_n[0] + dNdy[1] * lv_n[1] + dNdy[2] * lv_n[2]);
        double div_fact = div_u_n * ldetJ / 6.0;

        double lrhs[3];
        for (int in = 0; in < 3; ++in) {
            lrhs[in] = -div_fact;
        }

        // Term 2: \theta_1 \int \nabla N_k \cdot (u^* - u^n) d\Omega
        double sum_diff_u = ldiff_u[0] + ldiff_u[1] + ldiff_u[2];
        double sum_diff_v = ldiff_v[0] + ldiff_v[1] + ldiff_v[2];
        double theta_fact = ldetJ * theta1 / 6.0;
        sum_diff_u *= theta_fact;
        sum_diff_v *= theta_fact;

        for (int in = 0; in < 3; ++in) {
            lrhs[in] += dNdx[in] * sum_diff_u + dNdy[in] * sum_diff_v;
        }

        // Assemble into global RHS
        for (int in = 0; in < 3; ++in) {
            rhs1[ipn[in]] += lrhs[in];
        }
    }
}

} // namespace hydrama
