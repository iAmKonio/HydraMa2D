#include "thermal/ThermalPhysics.hpp"
#include "numerics/QuadElement.hpp"

#include <cmath>
#include <stdexcept>

namespace hydrama {

void compute_thermal_diffusivity(const Mesh& mesh,
                                 double ani,
                                 const ThermalConfig& config,
                                 const std::vector<double>* nut_elem,
                                 std::vector<double>& alpha_elem) {
    alpha_elem.resize(mesh.nelem);
    const double inv_pr = 1.0 / config.prandtl;
    const double inv_pr_t = 1.0 / config.prandtl_turbulent;
    const double alpha_mol = ani * inv_pr;

    if (nut_elem && !nut_elem->empty()) {
        for (int ie = 0; ie < mesh.nelem; ++ie) {
            alpha_elem[ie] = alpha_mol + (*nut_elem)[ie] * inv_pr_t;
        }
    } else {
        alpha_elem.assign(mesh.nelem, alpha_mol);
    }
}

void add_boussinesq_buoyancy(const Mesh& mesh,
                             const ThermalConfig& config,
                             double re,
                             const std::vector<double>& T,
                             std::vector<double>& rhs_u,
                             std::vector<double>& rhs_v) {
    if (!config.enabled) return;
    if (static_cast<int>(T.size()) != mesh.npoin ||
        static_cast<int>(rhs_u.size()) != mesh.npoin ||
        static_cast<int>(rhs_v.size()) != mesh.npoin) {
        return;
    }

    if (mesh.num_triangles() > 0 && mesh.detJ.size() != static_cast<std::size_t>(mesh.nelem)) {
        return;
    }
    if (mesh.has_quads() && mesh.quad_geom.size() != static_cast<std::size_t>(mesh.num_quads())) {
        return;
    }

    const auto [bx, by] = config.effective_buoyancy_vector(re);
    if (std::abs(bx) < 1.0e-15 && std::abs(by) < 1.0e-15) return;

    const double t_ref = config.reference_temperature;

    // Linear triangles
    for (int ie = 0; ie < mesh.num_triangles(); ++ie) {
        const int n0 = mesh.intma[ie][0];
        const int n1 = mesh.intma[ie][1];
        const int n2 = mesh.intma[ie][2];

        const double dT0 = T[n0] - t_ref;
        const double dT1 = T[n1] - t_ref;
        const double dT2 = T[n2] - t_ref;
        const double sum_dT = dT0 + dT1 + dT2;

        // Area A_e = 0.5 * detJ
        // Consistent mass integral: int_e N_i (T - T_ref) dOmega = (A_e / 12) * (sum_dT + dT_i)
        // With detJ = 2 * A_e, factor = detJ / 24.0
        const double lfact = mesh.detJ[ie] / 24.0;

        const double int_T0 = (sum_dT + dT0) * lfact;
        const double int_T1 = (sum_dT + dT1) * lfact;
        const double int_T2 = (sum_dT + dT2) * lfact;

        rhs_u[n0] += bx * int_T0;
        rhs_v[n0] += by * int_T0;

        rhs_u[n1] += bx * int_T1;
        rhs_v[n1] += by * int_T1;

        rhs_u[n2] += bx * int_T2;
        rhs_v[n2] += by * int_T2;
    }

    // Bilinear quadrilaterals: 2x2 Gauss quadrature
    if (mesh.has_quads()) {
        const auto& shape = q1::gauss_shape();
        for (int q = 0; q < mesh.num_quads(); ++q) {
            const auto& n = mesh.quads[static_cast<std::size_t>(q)];
            const QuadGeometry& geo = mesh.quad_geom[static_cast<std::size_t>(q)];

            const double lT[4] = {T[n[0]], T[n[1]], T[n[2]], T[n[3]]};

            double lrhs_u[4] = {0.0, 0.0, 0.0, 0.0};
            double lrhs_v[4] = {0.0, 0.0, 0.0, 0.0};

            for (int g = 0; g < q1::kGaussPoints; ++g) {
                const double W = geo.weight[g];
                const auto& N = shape[g];

                double Tg = 0.0;
                for (int a = 0; a < 4; ++a) {
                    Tg += N[a] * lT[a];
                }
                const double dTg = Tg - t_ref;
                const double fb_x = bx * dTg;
                const double fb_y = by * dTg;

                for (int a = 0; a < 4; ++a) {
                    lrhs_u[a] += W * N[a] * fb_x;
                    lrhs_v[a] += W * N[a] * fb_y;
                }
            }

            for (int a = 0; a < 4; ++a) {
                rhs_u[n[a]] += lrhs_u[a];
                rhs_v[n[a]] += lrhs_v[a];
            }
        }
    }
}

void apply_thermal_bcs(const Mesh& mesh,
                       const ThermalConfig& config,
                       std::vector<double>& T) {
    if (!config.enabled) return;

    for (int ib = 0; ib < mesh.nboun; ++ib) {
        const int flag = mesh.iside[ib].flag;
        const auto t_bc = config.boundary_temperature(flag);
        if (t_bc.has_value()) {
            T[mesh.iside[ib].node1] = *t_bc;
            T[mesh.iside[ib].node2] = *t_bc;
        }
    }
}

WallThermalReport compute_wall_nusselt(const Mesh& mesh,
                                       const ThermalConfig& config,
                                       const std::vector<double>& T,
                                       int boundary_flag,
                                       double delta_T,
                                       double ref_length) {
    (void)config;
    WallThermalReport report;
    if (delta_T <= 0.0) delta_T = 1.0;
    if (ref_length <= 0.0) ref_length = 1.0;

    if (static_cast<int>(mesh.face_norm.size()) != mesh.nboun ||
        static_cast<int>(mesh.dNkdx.size()) != mesh.nelem ||
        static_cast<int>(T.size()) != mesh.npoin) {
        return report;
    }

    for (int ib = 0; ib < mesh.nboun; ++ib) {
        if (mesh.iside[ib].flag != boundary_flag) continue;

        const int ie = mesh.iside[ib].elem;
        const double nx = mesh.face_norm[ib].nx; // outward normal
        const double ny = mesh.face_norm[ib].ny;
        const double ds = mesh.face_norm[ib].length;

        double dTdx = 0.0;
        double dTdy = 0.0;

        if (ie < mesh.num_triangles()) {
            const int n0 = mesh.intma[ie][0];
            const int n1 = mesh.intma[ie][1];
            const int n2 = mesh.intma[ie][2];
            const double* dNdx = mesh.dNkdx[ie].dNdx;
            const double* dNdy = mesh.dNkdx[ie].dNdy;
            dTdx = dNdx[0] * T[n0] + dNdx[1] * T[n1] + dNdx[2] * T[n2];
            dTdy = dNdy[0] * T[n0] + dNdy[1] * T[n1] + dNdy[2] * T[n2];
        } else {
            const int q = mesh.quad_of(ie);
            const auto& n = mesh.quads[static_cast<std::size_t>(q)];
            const double* dNdx = mesh.dNkdx[ie].dNdx;
            const double* dNdy = mesh.dNkdx[ie].dNdy;
            dTdx = dNdx[0] * T[n[0]] + dNdx[1] * T[n[1]] + dNdx[2] * T[n[2]] + dNdx[3] * T[n[3]];
            dTdy = dNdy[0] * T[n[0]] + dNdy[1] * T[n[1]] + dNdy[2] * T[n[2]] + dNdy[3] * T[n[3]];
        }

        // Heat flux into the fluid domain across the side:
        // n is outward from fluid domain, so flux into fluid = (grad T . n)
        const double flux_in = (dTdx * nx + dTdy * ny);
        report.total_heat_rate += flux_in * ds;
        report.total_length += ds;
    }

    if (report.total_length > 0.0) {
        report.mean_heat_flux = report.total_heat_rate / report.total_length;
        report.mean_nusselt = (report.mean_heat_flux * ref_length) / delta_T;
    }

    return report;
}

} // namespace hydrama
