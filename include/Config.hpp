#pragma once

#include <string>
#include <array>

namespace hydrama {

struct ControlConfig {
    int restart_opt = 0;
    int ntime = 1000;
    int transient_on = 1;
    double time_output_interval = 1.0;
    double end_rtime = 500.0;
    int write_output = 100000;
    double write_time_output = 3600.0;
    int dtfixed = -1; // -1: Global min, 0: Local timestepping, 1: Fixed dt
    double dtfix = 1.0e-5;
    int iwrite = 100;
    double csafm = 0.6;
    std::array<double, 2> theta = {1.0, 1.0}; // theta[0] (0.5 <= theta1 <= 1.0), theta[1] = 1.0
    int vel_check = 1;
    double l2norm_vel_tolerance = 1.0e-6;
    int runtime_mod = 0;
};

struct PhysicalConfig {
    double re = 100.0;
    int pnode = 0; // 0-based reference pressure node
};

struct SolverConfig {
    int solver_opt = 1; // 1 = Conjugate Gradient
    double relToler = 0.1;
    double absToler = 1.0e-6;
};

struct SimulationConfig {
    std::string casename;
    std::string case_dir = ".";
    ControlConfig control;
    PhysicalConfig physical;
    SolverConfig solver;
};

} // namespace hydrama
