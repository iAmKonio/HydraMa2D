#include "LinearSolver.hpp"

#include "linalg/ConjugateGradient.hpp"
#include "linalg/ConstrainedOperator.hpp"
#include "linalg/EbeOperator.hpp"
#include "linalg/JacobiPreconditioner.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

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
    for (int ie = 0; ie < mesh.nelem; ++ie) {
        double dt = delte[ie];
        for (int in = 0; in < Mesh::nep; ++in) {
            int ip = mesh.intma[ie][in];
            elcoe2[ip] += mesh.elcoe_e[ie * Mesh::nep + in] / dt;
        }
    }
    for (int ip = 0; ip < mesh.npoin; ++ip) {
        elcoe2[ip] = 1.0 / elcoe2[ip];
    }

    // 2. Update LHS Laplacian Matrix scaled by delte(ie)
    pdiag.assign(mesh.npoin, 0.0);
    gstif.resize(mesh.nelem);

    for (int ie = 0; ie < mesh.nelem; ++ie) {
        double dt = delte[ie];

        // Off-diagonals
        gstif[ie][0] = mesh.gstifE[ie][0] * dt;
        gstif[ie][1] = mesh.gstifE[ie][1] * dt;
        gstif[ie][2] = mesh.gstifE[ie][2] * dt;

        // Diagonal
        for (int in = 0; in < Mesh::nep; ++in) {
            int ip = mesh.intma[ie][in];
            pdiag[ip] += mesh.pdiagE[ie][in] * dt;
        }
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

void compute_timestep(int solver_opt, int dtfixed, int transient_on,
                      double dtfix, double ani, double csafm,
                      const Mesh& mesh,
                      const std::vector<double>& velocity,
                      std::vector<double>& delte) {
    delte.resize(mesh.nelem);

    if (dtfixed > 0) {
        delte.assign(mesh.nelem, dtfix);
        return;
    }

    double delte_min = std::numeric_limits<double>::max();
    double re_half = 0.5 / ani; // 0.5 * Re

    for (int ie = 0; ie < mesh.nelem; ++ie) {
        double amin = mesh.alen_e[ie];
        double velc = velocity[mesh.intma[ie][0]];
        for (int in = 1; in < Mesh::nep; ++in) {
            velc = std::max(velc, velocity[mesh.intma[ie][in]]);
        }
        if (velc < 1.0e-15) velc = 1.0e-15;

        double dt_diff = csafm * amin * amin * re_half;
        double dt_conv = csafm * amin / velc;
        delte[ie] = std::min(dt_diff, dt_conv);
        delte_min = std::min(delte_min, delte[ie]);
    }

    if (transient_on > 0 || solver_opt > 1 || dtfixed == -1) {
        delte.assign(mesh.nelem, delte_min);
    }
}

} // namespace hydrama
