#include "LinearSolver.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

namespace hydrama {

void matrix_vector_mult(const Mesh& mesh,
                        const std::vector<std::array<double, 3>>& gstif,
                        const std::vector<double>& pdiag,
                        const std::vector<double>& x,
                        std::vector<double>& b) {
    b.assign(mesh.npoin, 0.0);

    for (int ie = 0; ie < mesh.nelem; ++ie) {
        int n0 = mesh.intma[ie][0];
        int n1 = mesh.intma[ie][1];
        int n2 = mesh.intma[ie][2];

        double k0 = gstif[ie][0]; // pair (0, 1)
        double k1 = gstif[ie][1]; // pair (0, 2)
        double k2 = gstif[ie][2]; // pair (1, 2)

        b[n0] += k0 * x[n1] + k1 * x[n2];
        b[n1] += k0 * x[n0] + k2 * x[n2];
        b[n2] += k1 * x[n0] + k2 * x[n1];
    }

    for (int ip = 0; ip < mesh.npoin; ++ip) {
        b[ip] += x[ip] * pdiag[ip];
    }
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
    PCGResult result;
    const int maxIter = 100000;
    const int npoin = mesh.npoin;
    const int bc_fixed = static_cast<int>(bc_list.size());

    // Apply Dirichlet Boundary Values to initial pressure guess
    for (int i = 0; i < bc_fixed; ++i) {
        int ip = bc_list[i];
        press[ip] = bc_values[i];
    }

    // Calculate Apold = [A]{press}
    std::vector<double> apold(npoin, 0.0);
    matrix_vector_mult(mesh, gstif, pdiag, press, apold);

    // Initial residual: r = rhs - A * x
    std::vector<double> residual(npoin);
    for (int ip = 0; ip < npoin; ++ip) {
        residual[ip] = rhs1[ip] - apold[ip];
    }

    // Zero residual at Dirichlet boundary nodes
    for (int i = 0; i < bc_fixed; ++i) {
        int ip = bc_list[i];
        residual[ip] = 0.0;
    }

    // Jacobi preconditioner: M = 1.0 / diag(A)
    std::vector<double> Mmatrix(npoin);
    for (int ip = 0; ip < npoin; ++ip) {
        Mmatrix[ip] = (std::abs(pdiag[ip]) > 1.0e-30) ? (1.0 / pdiag[ip]) : 1.0;
    }

    // z = M * r
    std::vector<double> Zip(npoin);
    std::vector<double> pvect(npoin);
    double alph1 = 0.0;
    for (int ip = 0; ip < npoin; ++ip) {
        Zip[ip] = residual[ip] * Mmatrix[ip];
        pvect[ip] = Zip[ip];
        alph1 += residual[ip] * Zip[ip];
    }
    double beta1 = alph1;

    double bmax0 = 0.0;
    double bmax = 0.0;

    // Iterative Loop
    for (int iiter = 1; iiter <= maxIter; ++iiter) {
        result.iterations = iiter;

        // apold = [A]{p}
        matrix_vector_mult(mesh, gstif, pdiag, pvect, apold);

        // Zero Ap at Dirichlet boundary nodes
        for (int i = 0; i < bc_fixed; ++i) {
            int ip = bc_list[i];
            apold[ip] = 0.0;
        }

        // alpha = (r . z) / (p . A p)
        alph1 = beta1;
        double alph2 = 0.0;
        for (int ip = 0; ip < npoin; ++ip) {
            alph2 += pvect[ip] * apold[ip];
        }

        double alpha = (std::abs(alph2) > 1.0e-30) ? (alph1 / alph2) : 0.0;

        for (int ip = 0; ip < npoin; ++ip) {
            press[ip] += alpha * pvect[ip];
            residual[ip] -= alpha * apold[ip];
            Zip[ip] = residual[ip] * Mmatrix[ip];
        }

        // Compute L2 norm of residual: ||r||_2
        double res_sq = 0.0;
        for (int ip = 0; ip < npoin; ++ip) {
            res_sq += residual[ip] * residual[ip];
        }
        bmax = std::sqrt(res_sq);

        if (iiter == 1) {
            bmax0 = bmax;
            result.initial_residual = bmax0;
        }

        result.final_residual = bmax;

        // Convergence check: relative or absolute residual tolerance
        if (bmax0 > 0.0 && (bmax / bmax0) < relToler) {
            break;
        }
        if (bmax < absToler) {
            break;
        }

        // beta = (z_{k+1} . r_{k+1}) / (z_k . r_k)
        double beta2 = alph1;
        beta1 = 0.0;
        for (int ip = 0; ip < npoin; ++ip) {
            beta1 += Zip[ip] * residual[ip];
        }

        double beta = (std::abs(beta2) > 1.0e-30) ? (beta1 / beta2) : 0.0;

        for (int ip = 0; ip < npoin; ++ip) {
            pvect[ip] = Zip[ip] + beta * pvect[ip];
        }
    }

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
