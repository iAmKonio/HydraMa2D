#pragma once

#include "Config.hpp"
#include "Mesh.hpp"
#include "BoundaryConditions.hpp"
#include "LinearSolver.hpp"
#include "PressureSolver.hpp"
#include "geometry/WallGeometry.hpp"
#include "post/Monitors.hpp"
#include "turbulence/TurbulenceConfig.hpp"
#include "turbulence/TurbulenceModel.hpp"
#include <fstream>
#include <memory>
#include <vector>
#include <string>

namespace hydrama {

class Solver {
public:
    Solver(SimulationConfig config);

    bool initialize();
    void run();

    // Dependency injection of a turbulence model (tests, user-defined models).
    // Must be called before initialize(); the injected model replaces the one
    // configured in system/<case>.turbulence.
    void set_turbulence_model(std::unique_ptr<TurbulenceModel> model);

    // nullptr for laminar runs.
    const TurbulenceModel* turbulence_model() const { return turbulence_.get(); }

    // Moving and rotating wall specifications (from system/<case>.walls or set_wall_specs)
    void set_wall_specs(std::map<int, MovingWallSpec> specs) { wall_specs_ = std::move(specs); }
    const std::map<int, MovingWallSpec>& wall_specs() const noexcept { return wall_specs_; }

private:
    SimulationConfig config_;
    Mesh mesh_;
    PressureBC pressure_bc_;

    // Turbulence (inactive when turbulence_ == nullptr: laminar code path)
    TurbulenceConfig turbulence_config_;
    std::unique_ptr<TurbulenceModel> turbulence_;
    WallGeometry walls_;
    std::vector<double> nu_eff_elem_; // diffusivity for the time step limit

    // Moving and rotating walls
    std::map<int, MovingWallSpec> wall_specs_;

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

    // Force / probe monitors (system/<case>.monitors; nullptr without the file).
    std::unique_ptr<post::Monitors> monitors_;
    std::ofstream monitors_file_;
    std::vector<double> dudt_, dvdt_;
    post::MonitorSample last_sample_;
    bool has_sample_ = false;
    int last_sampled_step_ = -1;

    // Simulation metrics
    int istart_ = 1;
    double rtime_ = 0.0;
    int iiter_total_ = 0;

    void output_results(int iitime, double time_val);
    bool initialize_monitors(const std::string& prefix, bool restarted);
    // Samples the monitors at steps that are multiples of their interval, and
    // at the last step of the run (force = true).
    void record_monitors(int iitime, bool force = false);

    // Turbulence helpers (only called when turbulence_ != nullptr).
    TurbulenceContext turbulence_context() const;
    bool initialize_turbulence(const std::string& prefix, bool restarted);
    void update_effective_viscosity();
    std::string turbulence_log_line() const;
    double max_turbulence_residual() const;
};

} // namespace hydrama
