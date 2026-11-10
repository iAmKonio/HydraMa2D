#pragma once

#include "Config.hpp"
#include "Mesh.hpp"
#include "BoundaryConditions.hpp"
#include "LinearSolver.hpp"
#include "PressureSolver.hpp"
#include <memory>
#include <vector>
#include <string>

namespace hydrama {

class Solver {
public:
    Solver(SimulationConfig config);

    bool initialize();
    void run();

private:
    SimulationConfig config_;
    Mesh mesh_;
    PressureBC pressure_bc_;

    // Physical parameter
    double ani_ = 0.0; // 1.0 / Re

    // State unknowns
    std::vector<double> u_;
    std::vector<double> v_;
    std::vector<double> u_old_;
    std::vector<double> v_old_;
    std::vector<double> p_;
    std::vector<double> p_old_;

    std::vector<double> velocity_;
    std::vector<double> velocity_old_;

    // Intermediate and RHS vectors
    std::vector<double> rhs_u_;
    std::vector<double> rhs_v_;
    std::vector<double> rhs1_;

    // Solvers and operators
    std::vector<double> delte_;
    std::vector<double> elcoe2_;
    std::vector<double> pdiag_;
    std::vector<std::array<double, 3>> gstif_;

    // Pressure Poisson solver (Step 2); re-bound to gstif_/pdiag_ every step.
    std::unique_ptr<PressureSolver> pressure_solver_;

    // Simulation metrics
    int istart_ = 1;
    double rtime_ = 0.0;
    int iiter_total_ = 0;

    void output_results(int iitime, double time_val);
};

} // namespace hydrama
