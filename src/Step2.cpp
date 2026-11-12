#include "Step2.hpp"
#include "numerics/OpenMP.hpp"

namespace hydrama {

void compute_step2(const Mesh& mesh,
                   const std::vector<double>& u_star,
                   const std::vector<double>& v_star,
                   const std::vector<double>& u_n,
                   const std::vector<double>& v_n,
                   double theta1,
                   std::vector<double>& rhs1) {
    rhs1.assign(mesh.npoin, 0.0);

    auto process_triangle = [&](int ie) {
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
    //   - int N_a div u^n  +  theta1 int grad N_a . (u* - u^n)
    const auto& shape = q1::gauss_shape();
    auto process_quad = [&](int q) {
        const auto& n = mesh.quads[static_cast<std::size_t>(q)];
        const QuadGeometry& geo = mesh.quad_geom[static_cast<std::size_t>(q)];
        double un[4], vn[4], du[4], dv[4];
        for (int a = 0; a < 4; ++a) {
            un[a] = u_n[n[a]];
            vn[a] = v_n[n[a]];
            du[a] = u_star[n[a]] - un[a];
            dv[a] = v_star[n[a]] - vn[a];
        }
        double lrhs[4] = {0.0, 0.0, 0.0, 0.0};
        for (int g = 0; g < q1::kGaussPoints; ++g) {
            const double W = geo.weight[g];
            const auto& N = shape[g];
            const auto& dNdx = geo.dNdx[g];
            const auto& dNdy = geo.dNdy[g];
            double div = 0.0, dug = 0.0, dvg = 0.0;
            for (int a = 0; a < 4; ++a) {
                div += dNdx[a] * un[a] + dNdy[a] * vn[a];
                dug += N[a] * du[a];
                dvg += N[a] * dv[a];
            }
            for (int a = 0; a < 4; ++a) lrhs[a] += W * (-N[a] * div + theta1 * (dNdx[a] * dug + dNdy[a] * dvg));
        }
        for (int a = 0; a < 4; ++a) rhs1[n[a]] += lrhs[a];
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
