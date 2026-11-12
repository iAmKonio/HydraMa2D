#include "Step1.hpp"
#include "numerics/OpenMP.hpp"

#include <stdexcept>

namespace hydrama {

namespace {

// Shared element kernel. kTurbulent == false is the original laminar Step 1
// (same expressions, same order); kTurbulent == true adds the symmetric
// eddy-viscosity stress after the molecular term.
template <bool kTurbulent>
void step1_kernel(const Mesh& mesh,
                  double ani,
                  const std::vector<double>* nut_elem,
                  const std::vector<double>& delte,
                  const std::vector<double>& u,
                  const std::vector<double>& v,
                  std::vector<double>& rhs_u,
                  std::vector<double>& rhs_v) {
    rhs_u.assign(mesh.npoin, 0.0);
    rhs_v.assign(mesh.npoin, 0.0);

    auto process_triangle = [&](int ie) {
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

        // --- Turbulent stress (symmetric form) ---
        if constexpr (kTurbulent) {
            const double nut = (*nut_elem)[ie];
            if (nut != 0.0) {
                const double lfact_turb = ldetJ * nut * 0.5; // A_e nu_t,e
                const double two_ux = 2.0 * dNudx_ux;
                const double two_vy = 2.0 * dNudx_vy;
                const double shear = dNudx_uy + dNudx_vx;
                for (int in = 0; in < 3; ++in) {
                    lrhs_u[in] -= (dNdx[in] * two_ux + dNdy[in] * shear) * lfact_turb;
                    lrhs_v[in] -= (dNdx[in] * shear + dNdy[in] * two_vy) * lfact_turb;
                }
            }
        }

        // Assemble into global RHS
        for (int in = 0; in < 3; ++in) {
            int ip = ipn[in];
            rhs_u[ip] += lrhs_u[in];
            rhs_v[ip] += lrhs_v[in];
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

    // Bilinear quadrilaterals: the same weak form, 2 x 2 Gauss-Legendre
    // (docs/QUADRILATERAL_ELEMENTS.md, section 6):
    //   - int N_a (u . grad) u
    //   - dt/2 int (u_c . grad N_a) (u_c . grad) u          u_c: element-centre velocity
    //   - nu int grad N_a . grad u    [- nu_t int grad N_a . (grad u + grad u^T)]
    const auto& shape = q1::gauss_shape();
    auto process_quad = [&](int q) {
        const int ie = mesh.element_of_quad(q);
        const auto& n = mesh.quads[static_cast<std::size_t>(q)];
        const QuadGeometry& geo = mesh.quad_geom[static_cast<std::size_t>(q)];
        const double lu[4] = {u[n[0]], u[n[1]], u[n[2]], u[n[3]]};
        const double lv[4] = {v[n[0]], v[n[1]], v[n[2]], v[n[3]]};
        const double uc = 0.25 * (lu[0] + lu[1] + lu[2] + lu[3]); // value at (0, 0)
        const double vc = 0.25 * (lv[0] + lv[1] + lv[2] + lv[3]);
        const double dt_half = 0.5 * delte[ie];
        double nut = 0.0;
        if constexpr (kTurbulent) nut = (*nut_elem)[ie];

        double lrhs_u[4] = {0.0, 0.0, 0.0, 0.0};
        double lrhs_v[4] = {0.0, 0.0, 0.0, 0.0};
        for (int g = 0; g < q1::kGaussPoints; ++g) {
            const double W = geo.weight[g];
            const auto& N = shape[g];
            const auto& dNdx = geo.dNdx[g];
            const auto& dNdy = geo.dNdy[g];
            double ug = 0.0, vg = 0.0, ux = 0.0, uy = 0.0, vx = 0.0, vy = 0.0;
            for (int a = 0; a < 4; ++a) {
                ug += N[a] * lu[a];
                vg += N[a] * lv[a];
                ux += dNdx[a] * lu[a];
                uy += dNdy[a] * lu[a];
                vx += dNdx[a] * lv[a];
                vy += dNdy[a] * lv[a];
            }
            const double conv_u = ug * ux + vg * uy; // (u . grad) u at g
            const double conv_v = ug * vx + vg * vy;
            const double stab_u = uc * ux + vc * uy; // (u_c . grad) u at g
            const double stab_v = uc * vx + vc * vy;
            for (int a = 0; a < 4; ++a) {
                const double adv = dNdx[a] * uc + dNdy[a] * vc;
                double ru = N[a] * conv_u + dt_half * adv * stab_u + ani * (dNdx[a] * ux + dNdy[a] * uy);
                double rv = N[a] * conv_v + dt_half * adv * stab_v + ani * (dNdx[a] * vx + dNdy[a] * vy);
                if constexpr (kTurbulent) {
                    const double shear = uy + vx;
                    ru += nut * (dNdx[a] * 2.0 * ux + dNdy[a] * shear);
                    rv += nut * (dNdx[a] * shear + dNdy[a] * 2.0 * vy);
                }
                lrhs_u[a] -= W * ru;
                lrhs_v[a] -= W * rv;
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

} // namespace

void compute_step1(const Mesh& mesh,
                   double ani,
                   const std::vector<double>& delte,
                   const std::vector<double>& u,
                   const std::vector<double>& v,
                   std::vector<double>& rhs_u,
                   std::vector<double>& rhs_v) {
    step1_kernel<false>(mesh, ani, nullptr, delte, u, v, rhs_u, rhs_v);
}

void compute_step1(const Mesh& mesh,
                   double ani,
                   const std::vector<double>& nut_elem,
                   const std::vector<double>& delte,
                   const std::vector<double>& u,
                   const std::vector<double>& v,
                   std::vector<double>& rhs_u,
                   std::vector<double>& rhs_v) {
    if (nut_elem.empty()) {
        step1_kernel<false>(mesh, ani, nullptr, delte, u, v, rhs_u, rhs_v);
        return;
    }
    if (static_cast<int>(nut_elem.size()) != mesh.nelem) {
        throw std::invalid_argument("compute_step1: nut_elem must have one value per element");
    }
    step1_kernel<true>(mesh, ani, &nut_elem, delte, u, v, rhs_u, rhs_v);
}

} // namespace hydrama
