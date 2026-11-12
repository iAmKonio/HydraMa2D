#include "LinearSolver.hpp"

#include "linalg/ConjugateGradient.hpp"
#include "linalg/ConstrainedOperator.hpp"
#include "linalg/EbeOperator.hpp"
#include "linalg/JacobiPreconditioner.hpp"
#include "numerics/OpenMP.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace hydrama {

void matrix_vector_mult(const Mesh& mesh,
                        const std::vector<std::array<double, 3>>& gstif,
                        const std::vector<double>& pdiag,
                        const std::vector<double>& x,
                        std::vector<double>& b) {
    b.resize(mesh.npoin);
    linalg::ebe_apply(mesh, gstif.data(), pdiag.data(), x.data(), b.data());
}

void update_lhs(const Mesh& mesh,
                const std::vector<double>& delte,
                std::vector<double>& elcoe2,
                std::vector<double>& pdiag,
                std::vector<std::array<double, 3>>& gstif) {
    // 1. Update LHS Diagonal: Lumped Mass Matrix / dt
    elcoe2.assign(mesh.npoin, 0.0);
    for (int ie = 0; ie < mesh.num_triangles(); ++ie) {
        double dt = delte[ie];
        for (int in = 0; in < Mesh::nep; ++in) {
            int ip = mesh.intma[ie][in];
            elcoe2[ip] += mesh.elcoe_e[ie * Mesh::nep + in] / dt;
        }
    }
    for (int q = 0; q < mesh.num_quads(); ++q) {
        const double dt = delte[mesh.element_of_quad(q)];
        const auto& n = mesh.quads[static_cast<std::size_t>(q)];
        for (int a = 0; a < Mesh::nep_quad; ++a) {
            elcoe2[n[a]] += mesh.quad_geom[static_cast<std::size_t>(q)].mass[a] / dt;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int ip = 0; ip < mesh.npoin; ++ip) {
        elcoe2[ip] = 1.0 / elcoe2[ip];
    }

    // 2. Update LHS Laplacian Matrix scaled by delte(ie)
    pdiag.assign(mesh.npoin, 0.0);
    gstif.resize(static_cast<std::size_t>(mesh.num_offdiag_blocks()));

    #pragma omp parallel for schedule(static)
    for (int ie = 0; ie < mesh.num_triangles(); ++ie) {
        double dt = delte[ie];
        gstif[ie][0] = mesh.gstifE[ie][0] * dt;
        gstif[ie][1] = mesh.gstifE[ie][1] * dt;
        gstif[ie][2] = mesh.gstifE[ie][2] * dt;
    }

    for (int ie = 0; ie < mesh.num_triangles(); ++ie) {
        double dt = delte[ie];
        for (int in = 0; in < Mesh::nep; ++in) {
            int ip = mesh.intma[ie][in];
            pdiag[ip] += mesh.pdiagE[ie][in] * dt;
        }
    }

    const std::size_t first_quad_block = mesh.intma.size();
    #pragma omp parallel for schedule(static)
    for (int q = 0; q < mesh.num_quads(); ++q) {
        const double dt = delte[mesh.element_of_quad(q)];
        const QuadLaplacian& k = mesh.quad_lap[static_cast<std::size_t>(q)];
        auto& b0 = gstif[first_quad_block + 2 * static_cast<std::size_t>(q)];
        auto& b1 = gstif[first_quad_block + 2 * static_cast<std::size_t>(q) + 1];
        b0 = {k.off[0] * dt, k.off[1] * dt, k.off[2] * dt};
        b1 = {k.off[3] * dt, k.off[4] * dt, k.off[5] * dt};
    }

    for (int q = 0; q < mesh.num_quads(); ++q) {
        const double dt = delte[mesh.element_of_quad(q)];
        const QuadLaplacian& k = mesh.quad_lap[static_cast<std::size_t>(q)];
        const auto& n = mesh.quads[static_cast<std::size_t>(q)];
        for (int a = 0; a < Mesh::nep_quad; ++a) pdiag[n[a]] += k.diag[a] * dt;
    }
}

PCGResult solve_pcg(const Mesh& mesh,
                    const std::vector<std::array<double, 3>>& gstif,
                    const std::vector<double>& pdiag,
                    const std::vector<double>& rhs1,
                    const std::vector<int>& bc_list,
                    const std::vector<double>& bc_values,
                    double relToler,
                    double absToler,
                    std::vector<double>& press) {
    // Adapter over the linalg framework: legacy PCG = EBE operator + Jacobi
    // preconditioner + CG with the legacy convergence criteria.
    const linalg::EbeOperator op(mesh, gstif, pdiag);
    const linalg::DirichletConstraints constraints(bc_list, bc_values);

    linalg::JacobiPreconditioner jacobi;
    jacobi.setup(linalg::ConstrainedOperator(op, constraints));

    linalg::ConjugateGradient cg(linalg::ConvergenceCriteria::legacy_pcg(relToler, absToler));
    const linalg::SolveResult res = cg.solve(op, jacobi, rhs1, press, constraints);

    PCGResult result;
    result.iterations = res.iterations;
    result.initial_residual = res.reference_residual;
    result.final_residual = res.final_residual;
    return result;
}

namespace {

// Shared implementation: re_half_of(ie) returns 0.5 / nu for element ie.
template <typename ReHalf>
void timestep_kernel(int solver_opt, int dtfixed, int transient_on,
                     double dtfix, double csafm,
                     const Mesh& mesh,
                     const std::vector<double>& velocity,
                     ReHalf re_half_of,
                     std::vector<double>& delte) {
    delte.resize(mesh.nelem);

    if (dtfixed > 0) {
        delte.assign(mesh.nelem, dtfix);
        return;
    }

    double delte_min = std::numeric_limits<double>::max();

    #pragma omp parallel for reduction(min:delte_min) schedule(static)
    for (int ie = 0; ie < mesh.nelem; ++ie) {
        double amin = mesh.alen_e[ie];
        const ElementNodes nodes = mesh.element_nodes(ie);
        double velc = velocity[nodes[0]];
        for (int in = 1; in < nodes.size(); ++in) {
            velc = std::max(velc, velocity[nodes[in]]);
        }
        if (velc < 1.0e-15) velc = 1.0e-15;

        double dt_diff = csafm * amin * amin * re_half_of(ie);
        double dt_conv = csafm * amin / velc;
        delte[ie] = std::min(dt_diff, dt_conv);
        delte_min = std::min(delte_min, delte[ie]);
    }

    if (transient_on > 0 || solver_opt > 1 || dtfixed == -1) {
        delte.assign(mesh.nelem, delte_min);
    }
}

} // namespace

void compute_timestep(int solver_opt, int dtfixed, int transient_on,
                      double dtfix, double ani, double csafm,
                      const Mesh& mesh,
                      const std::vector<double>& velocity,
                      std::vector<double>& delte) {
    const double re_half = 0.5 / ani; // 0.5 * Re
    timestep_kernel(solver_opt, dtfixed, transient_on, dtfix, csafm, mesh, velocity,
                    [re_half](int) { return re_half; }, delte);
}

void compute_timestep(int solver_opt, int dtfixed, int transient_on,
                      double dtfix, const std::vector<double>& nu_eff_elem, double csafm,
                      const Mesh& mesh,
                      const std::vector<double>& velocity,
                      std::vector<double>& delte) {
    if (static_cast<int>(nu_eff_elem.size()) != mesh.nelem) {
        throw std::invalid_argument("compute_timestep: nu_eff_elem must have one value per element");
    }
    timestep_kernel(solver_opt, dtfixed, transient_on, dtfix, csafm, mesh, velocity,
                    [&nu_eff_elem](int ie) { return 0.5 / nu_eff_elem[ie]; }, delte);
}

} // namespace hydrama
