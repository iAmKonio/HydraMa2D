#pragma once

#include "Config.hpp"
#include "Mesh.hpp"
#include "BoundaryConditions.hpp"
#include "LinearSolver.hpp"
#include "PressureSolver.hpp"
#include "geometry/WallGeometry.hpp"
#include "post/Monitors.hpp"
#include "post/SkinFriction.hpp"
#include "post/Spectral.hpp"
#include "thermal/ThermalConfig.hpp"
#include "thermal/ThermalPhysics.hpp"
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

    // Automated post-processing diagnostics (Phase 1.2)
    post::SheddingAnalysis analyze_shedding(double t_transient = -1.0) const;
    std::vector<post::RecirculationBubble> detect_recirculation(const std::string& patch_name, double u_ref = 1.0) const;

    // Thermal energy equation and Boussinesq buoyancy (Phase 1.3)
    void set_thermal_config(ThermalConfig config) { thermal_config_ = std::move(config); }
    const ThermalConfig& thermal_config() const noexcept { return thermal_config_; }
    const std::vector<double>& temperature() const noexcept { return T_; }
    std::vector<double>& temperature() noexcept { return T_; }

    // Inflow velocity specifications (Phase 1.4: system/<case>.inflow, CLI flags, or set_inflow_config)
    void set_inflow_config(InflowConfig config) { inflow_config_ = std::move(config); }
    const InflowConfig& inflow_config() const noexcept { return inflow_config_; }
    void set_inflow_velocity(double u, double v) {
        inflow_config_.has_default = true;
        inflow_config_.default_u = u;
        inflow_config_.default_v = v;
    }
    void set_angle_of_attack(double alpha_deg, double speed = 1.0) {
        constexpr double kPi = 3.14159265358979323846;
        double rad = alpha_deg * (kPi / 180.0);
        set_inflow_velocity(speed * std::cos(rad), speed * std::sin(rad));
    }

    // Periodic boundary conditions and driving force (Phase 2.2)
    void set_periodic_config(PeriodicConfig config) { periodic_config_ = std::move(config); }
    const PeriodicConfig& periodic_config() const noexcept { return periodic_config_; }
    void set_driving_force(double fx, double fy) {
        periodic_config_.driving_force = {fx, fy};
        periodic_config_.enabled = true;
    }

    // Dual time-stepping (BDF2) controls (Phase 3.1)
    void set_dual_time(bool enable, int max_subiter = 30, double tol = 1.0e-4, double dt_phys = 0.0) {
        config_.control.dual_time = enable;
        config_.control.dual_time_max_subiter = max_subiter;
        config_.control.dual_time_tolerance = tol;
        if (dt_phys > 0.0) {
            config_.control.physical_dt = dt_phys;
        }
    }
    bool is_dual_time() const noexcept { return config_.control.dual_time; }
    int last_subiterations() const noexcept { return last_subiter_; }
    int total_subiterations() const noexcept { return total_subiter_; }
    double last_subiteration_residual() const noexcept { return last_sub_res_; }
    double physical_dt() const noexcept { return dt_phys_; }
    double rtime() const noexcept { return rtime_; }
    double compute_physical_cfl() const;

    void set_state(std::vector<double> u, std::vector<double> v, std::vector<double> p) {
        u_ = u;
        v_ = v;
        p_ = p;
        u_n_ = u;
        v_n_ = v;
        p_n_ = p;
        u_old_ = u;
        v_old_ = v;
        p_old_ = p;
        velocity_.resize(u_.size());
        for (std::size_t i = 0; i < u_.size(); ++i) {
            velocity_[i] = std::sqrt(u_[i] * u_[i] + v_[i] * v_[i] + 1.0e-16);
        }
    }

    void set_previous_state(std::vector<double> u_prev, std::vector<double> v_prev, std::vector<double> p_prev) {
        u_prev_ = std::move(u_prev);
        v_prev_ = std::move(v_prev);
        p_prev_ = std::move(p_prev);
        has_prev_state_ = true;
    }

    void set_max_steps(int n) {
        config_.override_ntime = n;
        config_.control.ntime = n;
    }
    void set_reynolds_number(double re) {
        config_.override_re = re;
        config_.physical.re = re;
    }
    const Mesh& mesh() const noexcept { return mesh_; }
    const std::vector<double>& u() const noexcept { return u_; }
    const std::vector<double>& v() const noexcept { return v_; }
    const std::vector<double>& p() const noexcept { return p_; }

    WallThermalReport compute_wall_nusselt(int boundary_flag, double delta_T = 1.0, double ref_length = 1.0) const {
        return hydrama::compute_wall_nusselt(mesh_, thermal_config_, T_, boundary_flag, delta_T, ref_length);
    }

private:
    SimulationConfig config_;
    Mesh mesh_;
    PressureBC pressure_bc_;

    // Turbulence (inactive when turbulence_ == nullptr: laminar code path)
    TurbulenceConfig turbulence_config_;
    std::unique_ptr<TurbulenceModel> turbulence_;
    WallGeometry walls_;
    std::vector<double> nu_eff_elem_; // diffusivity for the time step limit

    // Thermal energy transport and Boussinesq buoyancy
    ThermalConfig thermal_config_;
    std::vector<double> T_;
    std::vector<double> T_old_;
    std::vector<double> rhs_T_;
    std::vector<double> alpha_elem_;

    // Moving and rotating walls
    std::map<int, MovingWallSpec> wall_specs_;

    // Inflow boundary velocity specifications (system/<case>.inflow)
    InflowConfig inflow_config_;

    // Physical parameter
    double ani_ = 0.0; // 1.0 / Re

    // State unknowns
    std::vector<double> u_;
    std::vector<double> v_;
    std::vector<double> u_old_;
    std::vector<double> v_old_;
    std::vector<double> p_;
    std::vector<double> p_old_;

    // Physical time history for multi-step schemes (e.g. BDF2)
    std::vector<double> u_prev_;
    std::vector<double> v_prev_;
    std::vector<double> p_prev_;
    std::vector<double> u_n_;
    std::vector<double> v_n_;
    std::vector<double> p_n_;

    // Sub-iteration state in pseudo-time
    std::vector<double> u_sub_old_;
    std::vector<double> v_sub_old_;
    std::vector<double> p_sub_old_;
    std::vector<double> delte_star_;

    // Dual time diagnostics
    int last_subiter_ = 0;
    int total_subiter_ = 0;
    double last_sub_res_ = 0.0;
    double dt_phys_ = 0.0;
    bool has_prev_state_ = false;

    void advance_dual_time_step(int iitime);

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
    std::vector<double> nodal_mass_;

    // Pressure Poisson solver (Step 2); re-bound to gstif_/pdiag_ every step.
    std::unique_ptr<PressureSolver> pressure_solver_;

    // Force / probe monitors (system/<case>.monitors; nullptr without the file).
    std::unique_ptr<post::Monitors> monitors_;
    std::ofstream monitors_file_;
    std::vector<double> dudt_, dvdt_;
    post::MonitorSample last_sample_;
    bool has_sample_ = false;
    int last_sampled_step_ = -1;
    std::vector<double> monitor_time_history_;
    std::vector<double> monitor_cl_history_;
    std::vector<double> monitor_cd_history_;

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

    // Thermal helpers (only called when thermal_config_.enabled).
    bool initialize_thermal(const std::string& prefix, bool restarted);
    void advance_temperature();

    // Periodic helpers
    PeriodicConfig periodic_config_;
    bool initialize_periodic(const std::string& prefix);
};

} // namespace hydrama
