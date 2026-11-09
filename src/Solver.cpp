#include "Solver.hpp"
#include "IO.hpp"
#include "Step1.hpp"
#include "Step2.hpp"
#include "Step3.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <filesystem>

namespace hydrama {

Solver::Solver(SimulationConfig config) : config_(std::move(config)) {}

bool Solver::initialize() {
    std::string prefix = config_.case_dir.empty() ? "" : (config_.case_dir + "/");

    // Ensure results directory exists
    std::filesystem::create_directories(prefix + "results");

    // Read control, physical, and solver configs
    if (!IO::read_control(prefix + "system/" + config_.casename + ".control", config_.control)) {
        return false;
    }
    if (!IO::read_physical(prefix + "system/" + config_.casename + ".physical", config_.physical)) {
        return false;
    }
    if (!IO::read_solver(prefix + "system/" + config_.casename + ".solver", config_.solver)) {
        return false;
    }

    // Read mesh and boundary definitions
    if (!IO::read_mesh(prefix + "constant/" + config_.casename + ".mesh", mesh_)) {
        return false;
    }
    if (!IO::read_boundary_flags(prefix + "constant/" + config_.casename + ".bofl", mesh_)) {
        return false;
    }

    ani_ = 1.0 / config_.physical.re;

    // Geometric Pre-processing
    mesh_.compute_shape_function_derivatives();
    mesh_.compute_mass_matrix();
    mesh_.compute_face_normals();
    mesh_.determine_wall_nodes();
    mesh_.compute_element_sizes();
    mesh_.compute_laplacian_stiffness();

    // Allocate field unknowns
    int npoin = mesh_.npoin;
    u_.assign(npoin, 0.0);
    v_.assign(npoin, 0.0);
    u_old_.assign(npoin, 0.0);
    v_old_.assign(npoin, 0.0);

    p_.assign(npoin, 0.0001);
    p_old_.assign(npoin, 0.0001);

    velocity_.resize(npoin);
    velocity_old_.resize(npoin);
    for (int ip = 0; ip < npoin; ++ip) {
        velocity_[ip] = std::sqrt(u_[ip] * u_[ip] + v_[ip] * v_[ip] + 1.0e-16);
    }

    // Set up pressure boundary conditions
    pressure_bc_ = setup_pressure_bcs(mesh_, config_.physical.pnode);

    // Check for restart file
    std::string restart_file = prefix + "results/" + config_.casename + ".var";
    if (config_.control.restart_opt > 0) {
        if (IO::read_restart(restart_file, rtime_, istart_, u_, v_, p_)) {
            std::cout << "Restarting using previously calculated values\n";
            istart_ += 1;
            for (int ip = 0; ip < npoin; ++ip) {
                velocity_[ip] = std::sqrt(u_[ip] * u_[ip] + v_[ip] * v_[ip] + 1.0e-16);
            }
        } else {
            std::cout << "WARNING: Restart file does not exist or invalid. Using initial conditions.\n";
            istart_ = 1;
            rtime_ = 0.0;
        }
    } else {
        istart_ = 1;
        rtime_ = 0.0;
    }

    // Allocate work vectors
    rhs_u_.resize(npoin);
    rhs_v_.resize(npoin);
    rhs1_.resize(npoin);
    delte_.resize(mesh_.nelem);
    elcoe2_.resize(npoin);
    pdiag_.resize(npoin);
    gstif_.resize(mesh_.nelem);

    return true;
}

void Solver::output_results(int iitime, double time_val) {
    std::string prefix = config_.case_dir.empty() ? "" : (config_.case_dir + "/");
    std::string time_str;

    if (config_.control.transient_on < 1) {
        time_str = std::to_string(iitime);
    } else {
        time_str = std::to_string(static_cast<int>(time_val));
    }

    std::string vtk_path = prefix + "results/" + config_.casename + "_" + time_str + ".vtk";
    IO::write_vtk(vtk_path, mesh_, u_, v_, p_);

    std::string restart_path = prefix + "results/" + config_.casename + ".var";
    IO::write_restart(restart_path, time_val, iitime, config_.control.transient_on, u_, v_, p_);
}

void Solver::run() {
    auto t_start = std::chrono::steady_clock::now();
    auto t1 = t_start;

    double rtime_output = 0.0;
    bool iitime_start = true;
    int completed_time = istart_;

    // Time stepping loop
    for (int iitime = istart_; iitime <= config_.control.ntime; ++iitime) {
        completed_time = iitime;

        // Store old timestep values
        u_old_ = u_;
        v_old_ = v_;
        p_old_ = p_;

        // Calculate time step size
        compute_timestep(config_.solver.solver_opt,
                         config_.control.dtfixed,
                         config_.control.transient_on,
                         config_.control.dtfix,
                         ani_,
                         config_.control.csafm,
                         mesh_,
                         velocity_,
                         delte_);

        if (config_.control.transient_on > 0) {
            rtime_ += delte_[0];
        }

        // Update LHS lumped mass inverse and time-scaled Laplacian
        update_lhs(mesh_, delte_, elcoe2_, pdiag_, gstif_);

        // ==========================================
        // Step 1: Intermediate Velocity Calculation
        // ==========================================
        compute_step1(mesh_, ani_, delte_, u_, v_, rhs_u_, rhs_v_);

        for (int ip = 0; ip < mesh_.npoin; ++ip) {
            u_[ip] += rhs_u_[ip] * elcoe2_[ip];
            v_[ip] += rhs_v_[ip] * elcoe2_[ip];
        }

        apply_symmetry_bc(mesh_, u_, v_);
        apply_velocity_bcs(mesh_, u_, v_);

        // ==========================================
        // Step 2: Pressure Calculation (Poisson Eq)
        // ==========================================
        compute_step2(mesh_, u_, v_, u_old_, v_old_, config_.control.theta[0], rhs1_);

        PCGResult pcg_res = solve_pcg(mesh_, gstif_, pdiag_, rhs1_,
                                      pressure_bc_.nodes, pressure_bc_.values,
                                      config_.solver.relToler, config_.solver.absToler,
                                      p_);

        iiter_total_ += pcg_res.iterations;

        // ==========================================
        // Step 3: Velocity Correction
        // ==========================================
        compute_step3(mesh_, u_old_, v_old_, p_, p_old_, delte_, rhs_u_, rhs_v_);

        for (int ip = 0; ip < mesh_.npoin; ++ip) {
            u_[ip] += rhs_u_[ip] * elcoe2_[ip];
            v_[ip] += rhs_v_[ip] * elcoe2_[ip];
        }

        apply_symmetry_bc(mesh_, u_, v_);
        apply_velocity_bcs(mesh_, u_, v_);

        // Update Velocity Magnitude
        velocity_old_ = velocity_;
        for (int ip = 0; ip < mesh_.npoin; ++ip) {
            velocity_[ip] = std::sqrt(u_[ip] * u_[ip] + v_[ip] * v_[ip] + 1.0e-16);
        }

        auto t2 = std::chrono::steady_clock::now();

        // Convergence / Progress Logging
        if (config_.control.transient_on < 1) {
            // Steady State simulation
            if (iitime % config_.control.iwrite == 0 || iitime == 1) {
                double adel_vel_sq = 0.0, vel_sq = 0.0, res_vel_sq = 0.0;
                for (int ip = 0; ip < mesh_.npoin; ++ip) {
                    double dv = velocity_[ip] - velocity_old_[ip];
                    adel_vel_sq += dv * dv;
                    vel_sq += velocity_[ip] * velocity_[ip];
                    double rv = dv / elcoe2_[ip];
                    res_vel_sq += rv * rv;
                }
                double hb1 = std::sqrt(adel_vel_sq / (vel_sq + 1.0e-30));
                double hb3 = std::sqrt(res_vel_sq);
                (void)hb3;

                std::cout << "\n Time = " << iitime << "\n\n";
                std::cout << " PCG: Solving for p, Initial residual = " << pcg_res.initial_residual
                          << " Final residual = " << pcg_res.final_residual
                          << " Iter: " << pcg_res.iterations << "\n";

                if (config_.control.vel_check > 0 && hb1 < config_.control.l2norm_vel_tolerance) {
                    break;
                }
            }
        } else {
            // Transient simulation
            if (iitime % config_.control.iwrite == 0 || iitime_start) {
                iitime_start = false;
                std::cout << std::setw(8) << iitime
                          << " pInitial: " << std::setw(15) << std::setprecision(8) << std::scientific << pcg_res.initial_residual
                          << " pFinal: " << std::setw(15) << std::setprecision(8) << std::scientific << pcg_res.final_residual
                          << " Iter: " << std::setw(6) << pcg_res.iterations
                          << " IterCumulative: " << std::setw(9) << iiter_total_ << "\n";
                std::cout << " Time: " << std::setprecision(16) << std::scientific << rtime_ << "\n";
            }
        }

        // Output intervals
        if (iitime % config_.control.write_output == 0) {
            std::cout << " Iterations Threshold Exceeded\n";
            std::cout << " Outputting Data at: " << iitime << "\n";
            output_results(iitime, rtime_);
            t1 = t2;
        } else {
            double elapsed = std::chrono::duration<double>(t2 - t1).count();
            if (elapsed > config_.control.write_time_output) {
                std::cout << " CPU Time Threshold Exceeded\n";
                std::cout << " Outputting Data at: " << iitime << "\n";
                output_results(iitime, rtime_);
                t1 = t2;
            } else if (config_.control.transient_on > 0) {
                if (rtime_ > rtime_output) {
                    std::cout << " Transient Output Interval Exceeded\n";
                    std::cout << " Outputting Data at: " << iitime << " " << rtime_output << "\n";
                    output_results(iitime, rtime_);
                    rtime_output += config_.control.time_output_interval;
                    t1 = t2;
                }
                if (rtime_ > config_.control.end_rtime) {
                    break;
                }
            }
        }
    }

    std::cout << " Simulation Completed\n";

    if (config_.control.transient_on < 1) {
        std::cout << " Outputting Data\n";
        output_results(completed_time, rtime_);
    }

    auto t_final = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(t_final - t_start).count();
    std::cout << " Time Taken (s): " << total_sec << "\n";
}

} // namespace hydrama
