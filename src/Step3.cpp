#include "Step3.hpp"

namespace hydrama {

void compute_step3(const Mesh& mesh,
                   const std::vector<double>& u_n,
                   const std::vector<double>& v_n,
                   const std::vector<double>& p_new,
                   const std::vector<double>& p_old,
                   const std::vector<double>& delte,
                   std::vector<double>& rhs_u,
                   std::vector<double>& rhs_v) {
    rhs_u.assign(mesh.npoin, 0.0);
    rhs_v.assign(mesh.npoin, 0.0);

    for (int ie = 0; ie < mesh.nelem; ++ie) {
        int ipn[3] = {mesh.intma[ie][0], mesh.intma[ie][1], mesh.intma[ie][2]};

        double lu_n[3] = {u_n[ipn[0]], u_n[ipn[1]], u_n[ipn[2]]};
        double lv_n[3] = {v_n[ipn[0]], v_n[ipn[1]], v_n[ipn[2]]};

        double lp_new[3] = {p_new[ipn[0]], p_new[ipn[1]], p_new[ipn[2]]};
        double lp_old[3] = {p_old[ipn[0]], p_old[ipn[1]], p_old[ipn[2]]};

        const double* dNdx = mesh.dNkdx[ie].dNdx;
        const double* dNdy = mesh.dNkdx[ie].dNdy;
        double ldetJ = mesh.detJ[ie];

        double umean_u = (lu_n[0] + lu_n[1] + lu_n[2]) / 3.0;
        double umean_v = (lv_n[0] + lv_n[1] + lv_n[2]) / 3.0;

        // Pressure gradient for p^{n+1}:
        // dNpdx / 6.0
        double dNpdx_x = (dNdx[0] * lp_new[0] + dNdx[1] * lp_new[1] + dNdx[2] * lp_new[2]) / 6.0;
        double dNpdx_y = (dNdy[0] * lp_new[0] + dNdy[1] * lp_new[1] + dNdy[2] * lp_new[2]) / 6.0;

        // Old pressure gradient for p^n multiplied by dt / 4:
        double dt_fact = delte[ie] * 0.25;
        double dNpdx_old_x = (dNdx[0] * lp_old[0] + dNdx[1] * lp_old[1] + dNdx[2] * lp_old[2]) * dt_fact;
        double dNpdx_old_y = (dNdy[0] * lp_old[0] + dNdy[1] * lp_old[1] + dNdy[2] * lp_old[2]) * dt_fact;

        for (int in = 0; in < 3; ++in) {
            double lval = dNdx[in] * umean_u + dNdy[in] * umean_v;
            double lrhs_u = -ldetJ * (dNpdx_x + lval * dNpdx_old_x);
            double lrhs_v = -ldetJ * (dNpdx_y + lval * dNpdx_old_y);

            int ip = ipn[in];
            rhs_u[ip] += lrhs_u;
            rhs_v[ip] += lrhs_v;
        }
    }
}

} // namespace hydrama
