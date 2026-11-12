#include "Step3.hpp"
#include "numerics/OpenMP.hpp"

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

    auto process_triangle = [&](int ie) {
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
    };

    if (!mesh.triangle_color_groups.empty()) {
        for (const auto& grp : mesh.triangle_color_groups) {
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < grp.size(); ++i) {
                process_triangle(grp[i]);
            }
        }
    } else {
        for (int ie = 0; ie < mesh.num_triangles(); ++ie) {
            process_triangle(ie);
        }
    }

    // Bilinear quadrilaterals, 2 x 2 Gauss-Legendre:
    //   - int N_a grad p^{n+1}  -  dt/2 int (u_c . grad N_a) grad p^n     u_c: element-centre velocity
    const auto& shape = q1::gauss_shape();
    auto process_quad = [&](int q) {
        const int ie = mesh.element_of_quad(q);
        const auto& n = mesh.quads[static_cast<std::size_t>(q)];
        const QuadGeometry& geo = mesh.quad_geom[static_cast<std::size_t>(q)];
        double pnew[4], pold[4], uc = 0.0, vc = 0.0;
        for (int a = 0; a < 4; ++a) {
            pnew[a] = p_new[n[a]];
            pold[a] = p_old[n[a]];
            uc += 0.25 * u_n[n[a]];
            vc += 0.25 * v_n[n[a]];
        }
        const double dt_half = 0.5 * delte[ie];
        double lrhs_u[4] = {0.0, 0.0, 0.0, 0.0};
        double lrhs_v[4] = {0.0, 0.0, 0.0, 0.0};
        for (int g = 0; g < q1::kGaussPoints; ++g) {
            const double W = geo.weight[g];
            const auto& N = shape[g];
            const auto& dNdx = geo.dNdx[g];
            const auto& dNdy = geo.dNdy[g];
            double px = 0.0, py = 0.0, pox = 0.0, poy = 0.0;
            for (int a = 0; a < 4; ++a) {
                px += dNdx[a] * pnew[a];
                py += dNdy[a] * pnew[a];
                pox += dNdx[a] * pold[a];
                poy += dNdy[a] * pold[a];
            }
            for (int a = 0; a < 4; ++a) {
                const double adv = dNdx[a] * uc + dNdy[a] * vc;
                lrhs_u[a] -= W * (N[a] * px + dt_half * adv * pox);
                lrhs_v[a] -= W * (N[a] * py + dt_half * adv * poy);
            }
        }
        for (int a = 0; a < 4; ++a) {
            rhs_u[n[a]] += lrhs_u[a];
            rhs_v[n[a]] += lrhs_v[a];
        }
    };

    if (!mesh.quad_color_groups.empty()) {
        for (const auto& grp : mesh.quad_color_groups) {
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < grp.size(); ++i) {
                process_quad(grp[i]);
            }
        }
    } else {
        for (int q = 0; q < mesh.num_quads(); ++q) {
            process_quad(q);
        }
    }
}

} // namespace hydrama
