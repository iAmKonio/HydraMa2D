#include "Solver.hpp"
#include "IO.hpp"
#include "geometry/FieldOps.hpp"
#include "mesh/MeshQuality.hpp"
#include "numerics/Periodic.hpp"
#include "numerics/ScalarTransport.hpp"
#include "Step1.hpp"
#include "Step2.hpp"
#include "Step3.hpp"
#include "turbulence/TurbulenceFactory.hpp"
#include "turbulence/WallUnits.hpp"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace hydrama {

Solver::Solver(SimulationConfig config) : config_(std::move(config)) {}

void Solver::set_turbulence_model(std::unique_ptr<TurbulenceModel> model) {
    turbulence_ = std::move(model);
}

bool Solver::initialize() {
    std::string prefix = config_.case_dir.empty() ? "" : (config_.case_dir + "/");

    // Ensure results directory exists
    std::filesystem::create_directories(prefix + "results");

    // Read control, physical, and solver configs
    if (!IO::read_control(prefix + "system/" + config_.casename + ".control", config_.control)) {
        return false;
    }
    if (config_.override_ntime.has_value()) {
        config_.control.ntime = *config_.override_ntime;
    }
    if (!IO::read_physical(prefix + "system/" + config_.casename + ".physical", config_.physical)) {
        return false;
    }
    if (config_.override_re.has_value()) {
        config_.physical.re = *config_.override_re;
    }
    if (!IO::read_solver(prefix + "system/" + config_.casename + ".solver", config_.solver)) {
        return false;
    }

    // Optional turbulence configuration (missing file: laminar)
    if (!IO::read_turbulence(prefix + "system/" + config_.casename + ".turbulence", turbulence_config_)) {
        return false;
    }

    // Optional moving / rotating walls configuration (missing file: empty specs)
    if (wall_specs_.empty()) {
        if (!IO::read_walls(prefix + "system/" + config_.casename + ".walls", wall_specs_)) {
            return false;
        }
    }

    // Optional inflow boundary velocity configuration (missing file: default uniform u=1, v=0)
    if (inflow_config_.empty()) {
        if (!config_.inflow.empty()) {
            inflow_config_ = config_.inflow;
        } else {
            if (!IO::read_inflow(prefix + "system/" + config_.casename + ".inflow", inflow_config_)) {
                return false;
            }
        }
    }

    // Optional periodic boundary configuration (missing file: empty)
    if (!periodic_config_.enabled) {
        if (config_.periodic.enabled) {
            periodic_config_ = config_.periodic;
        } else {
            if (!IO::read_periodic(prefix + "system/" + config_.casename + ".periodic", periodic_config_)) {
                return false;
            }
        }
    }
    if (!turbulence_) {
        try {
            turbulence_ = make_turbulence_model(turbulence_config_);
        } catch (const std::invalid_argument& e) {
            std::cerr << "ERROR: " << e.what() << "\n";
            return false;
        }
    }

    // Read mesh and boundary definitions: constant/<case>.mesh + <case>.bofl
    // (legacy format), constant/<case>.msh (Gmsh), constant/<case>.su2 (SU2)
    // or constant/<case>.cgns (CGNS); for the last three <case>.bofl is optional.
    if (!IO::read_case_mesh(prefix + "constant", config_.casename, mesh_)) {
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
    initialize_periodic(prefix);
    std::cout << mesh::format_quality_report(mesh_, mesh::assess_mesh_quality(mesh_));

    // Parabolic (fully developed) inlets: check that they are open curves.
    try {
        const std::vector<NodeVelocity> inflow = parabolic_inlet_velocities(mesh_);
        if (!inflow.empty()) {
            double peak = 0.0;
            for (const NodeVelocity& nv : inflow) peak = std::max(peak, std::hypot(nv.u, nv.v));
            std::cout << "Parabolic inlet (508): " << inflow.size() << " nodes, peak velocity " << peak << "\n";
        }
    } catch (const std::invalid_argument& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return false;
    }

    // Allocate field unknowns
    int npoin = mesh_.npoin;
    u_.assign(npoin, 0.0);
    v_.assign(npoin, 0.0);
    u_old_.assign(npoin, 0.0);
    v_old_.assign(npoin, 0.0);

    p_.assign(npoin, 0.0001);
    p_old_.assign(npoin, 0.0001);

    // Phase 3.1: Dual time-stepping history and sub-iteration work vectors
    u_prev_.assign(npoin, 0.0);
    v_prev_.assign(npoin, 0.0);
    p_prev_.assign(npoin, 0.0001);
    u_n_.assign(npoin, 0.0);
    v_n_.assign(npoin, 0.0);
    p_n_.assign(npoin, 0.0001);
    u_sub_old_.assign(npoin, 0.0);
    v_sub_old_.assign(npoin, 0.0);
    p_sub_old_.assign(npoin, 0.0001);
    delte_star_.assign(mesh_.nelem, 0.0);
    last_subiter_ = 0;
    total_subiter_ = 0;
    last_sub_res_ = 0.0;
    dt_phys_ = 0.0;
    if (config_.control.physical_dt > 0.0) {
        dt_phys_ = config_.control.physical_dt;
    } else if (config_.control.dtfixed == 1 && config_.control.dtfix > 0.0) {
        dt_phys_ = config_.control.dtfix;
    }

    velocity_.resize(npoin);
    velocity_old_.resize(npoin);
    for (int ip = 0; ip < npoin; ++ip) {
        velocity_[ip] = std::sqrt(u_[ip] * u_[ip] + v_[ip] * v_[ip] + 1.0e-16);
    }

    // Set up pressure boundary conditions
    pressure_bc_ = setup_pressure_bcs(mesh_, config_.physical.pnode);
    try {
        pressure_solver_ = std::make_unique<PressureSolver>(config_.solver, pressure_bc_);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: INVALID PRESSURE SOLVER SETTINGS: " << e.what() << "\n";
        return false;
    }
    std::cout << "Pressure solver: " << pressure_solver_->description() << "\n";

    // Check for restart file
    bool restarted = false;
    std::string restart_file = prefix + "results/" + config_.casename + ".var";
    if (config_.control.restart_opt > 0) {
        if (IO::read_restart(restart_file, rtime_, istart_, u_, v_, p_)) {
            restarted = true;
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

    u_n_ = u_;
    v_n_ = v_;
    p_n_ = p_;
    u_prev_ = u_;
    v_prev_ = v_;
    p_prev_ = p_;

    // Allocate work vectors
    rhs_u_.resize(npoin);
    rhs_v_.resize(npoin);
    rhs1_.resize(npoin);
    delte_.resize(mesh_.nelem);
    elcoe2_.resize(npoin);
    pdiag_.resize(npoin);
    gstif_.resize(static_cast<std::size_t>(mesh_.num_offdiag_blocks()));
    nodal_mass_ = lumped_nodal_mass(mesh_);

    if (mesh_.has_periodic()) {
        apply_periodic_field(mesh_, {}, u_);
        apply_periodic_field(mesh_, {}, v_);
        apply_periodic_field(mesh_, {}, p_);
    }

    // Read optional thermal configuration if not already set programmatically
    if (!thermal_config_.enabled) {
        if (config_.thermal.enabled) {
            thermal_config_ = config_.thermal;
        } else {
            const std::string thermal_file = prefix + "system/" + config_.casename + ".thermal";
            std::error_code ec;
            if (std::filesystem::exists(thermal_file, ec)) {
                if (!IO::read_thermal(thermal_file, thermal_config_)) {
                    return false;
                }
            }
        }
    }

    if (thermal_config_.enabled && !initialize_thermal(prefix, restarted)) {
        return false;
    }

    if (turbulence_ && !initialize_turbulence(prefix, restarted)) {
        return false;
    }

    if (!initialize_monitors(prefix, restarted)) {
        return false;
    }

    return true;
}

bool Solver::initialize_monitors(const std::string& prefix, bool restarted) {
    const std::string path = prefix + "system/" + config_.casename + ".monitors";
    post::MonitorConfig mc;
    bool found = false;
    std::string error;
    if (!post::read_monitor_config(path, mc, found, error)) {
        std::cerr << "ERROR: *.monitors FILE HAS INVALID FORMAT: " << error << "\n";
        return false;
    }
    if (!found) return true; // no monitors: nothing changes
    try {
        monitors_ = std::make_unique<post::Monitors>(mesh_, std::move(mc));
    } catch (const std::invalid_argument& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return false;
    }
    std::cout << monitors_->description();
    const std::string out = prefix + "results/" + config_.casename + ".monitors.dat";
    // A restarted run continues the history of the previous one.
    const bool append = restarted && std::filesystem::exists(out);
    monitors_file_.open(out, append ? std::ios::app : std::ios::trunc);
    if (!monitors_file_) {
        std::cerr << "ERROR: CANNOT WRITE MONITOR FILE: " << out << "\n";
        return false;
    }
    if (!append) monitors_file_ << monitors_->file_header(config_.casename);
    std::cout << "Monitors: writing " << out << (append ? " (appending)" : "") << "\n";
    return true;
}

void Solver::record_monitors(int iitime, bool force) {
    if (!monitors_ || iitime == last_sampled_step_) return;
    if (!force && iitime % monitors_->config().interval != 0) return;
    // Transient runs have one global time step: the force includes du/dt.
    const bool transient = (config_.control.transient_on > 0 && !delte_.empty() && delte_[0] > 0.0) ||
                           (config_.control.dual_time && dt_phys_ > 0.0);
    if (transient) {
        dudt_.resize(u_.size());
        dvdt_.resize(v_.size());
        if (config_.control.dual_time && dt_phys_ > 0.0) {
            const int n_phys = iitime - istart_ + 1;
            const double a0 = (n_phys >= 2) ? 1.5 : 1.0;
            const double a1 = (n_phys >= 2) ? -2.0 : -1.0;
            const double a2 = (n_phys >= 2) ? 0.5 : 0.0;
            const double inv_dt = 1.0 / dt_phys_;
            for (std::size_t i = 0; i < u_.size(); ++i) {
                dudt_[i] = (a0 * u_[i] + a1 * u_n_[i] + a2 * u_prev_[i]) * inv_dt;
                dvdt_[i] = (a0 * v_[i] + a1 * v_n_[i] + a2 * v_prev_[i]) * inv_dt;
            }
        } else {
            for (std::size_t i = 0; i < u_.size(); ++i) {
                dudt_[i] = (u_[i] - u_old_[i]) / delte_[0];
                dvdt_[i] = (v_[i] - v_old_[i]) / delte_[0];
            }
        }
    }
    last_sample_ = monitors_->evaluate(mesh_, ani_, u_, v_, p_, transient ? &dudt_ : nullptr,
                                       transient ? &dvdt_ : nullptr);
    has_sample_ = true;
    last_sampled_step_ = iitime;
    if (monitors_->has_body()) {
        monitor_time_history_.push_back(rtime_);
        monitor_cl_history_.push_back(last_sample_.cl);
        monitor_cd_history_.push_back(last_sample_.cd);
    }
    monitors_file_ << monitors_->format_row(iitime, rtime_, last_sample_);
    monitors_file_.flush();
}

TurbulenceContext Solver::turbulence_context() const {
    return TurbulenceContext{mesh_, walls_, ani_, delte_, elcoe2_};
}

bool Solver::initialize_turbulence(const std::string& prefix, bool restarted) {
    try {
        walls_ = WallGeometry(mesh_, *make_wall_distance_calculator(turbulence_config_.wall_distance));
    } catch (const std::invalid_argument& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return false;
    }
    const TurbulenceContext ctx = turbulence_context();
    turbulence_->initialize(ctx, {u_, v_});
    nu_eff_elem_.assign(mesh_.nelem, ani_);

    std::cout << "Turbulence model: " << turbulence_->name() << "\n";
    std::cout << "Wall faces: " << walls_.num_faces() << ", wall nodes: " << walls_.num_wall_nodes();
    if (walls_.has_walls()) {
        const auto& d = walls_.wall_distance();
        std::cout << ", max wall distance: " << *std::max_element(d.begin(), d.end());
    }
    std::cout << "\n";

    // Turbulence restart: continue from results/<case>.turb.var when the flow
    // was restarted; otherwise (or if it does not match) start from the
    // free-stream values, e.g. a turbulent run started from a laminar solution.
    if (restarted) {
        const std::string path = prefix + "results/" + config_.casename + ".turb.var";
        TurbulenceRestartData data;
        std::string error;
        std::vector<FieldRef> state = turbulence_->state_fields();
        bool ok = IO::read_turbulence_restart(path, mesh_.npoin, data, error);
        if (ok && data.model != turbulence_->name()) {
            ok = false;
            error = "file is for model '" + data.model + "'";
        }
        if (ok && data.names.size() != state.size()) {
            ok = false;
            error = "field count does not match the model";
        }
        for (std::size_t k = 0; ok && k < state.size(); ++k) {
            if (data.names[k] != state[k].name) {
                ok = false;
                error = "field '" + data.names[k] + "' does not match '" + state[k].name + "'";
            }
        }
        if (ok) {
            for (std::size_t k = 0; k < state.size(); ++k) *state[k].values = data.values[k];
            turbulence_->refresh(ctx);
            std::cout << "Restarting turbulence from " << path << "\n";
        } else {
            std::cout << "WARNING: Turbulence restart not used (" << error
                      << "). Turbulence starts from free-stream values.\n";
        }
    }
    return true;
}

bool Solver::initialize_thermal(const std::string& prefix, bool restarted) {
    T_.assign(mesh_.npoin, thermal_config_.t_init);
    T_old_ = T_;
    rhs_T_.assign(mesh_.npoin, 0.0);
    alpha_elem_.assign(mesh_.nelem, ani_ / thermal_config_.prandtl);

    if (restarted) {
        const std::string path = prefix + "results/" + config_.casename + ".thermal.var";
        std::string error;
        std::error_code ec;
        if (std::filesystem::exists(path, ec)) {
            if (!IO::read_thermal_restart(path, mesh_.npoin, T_, error)) {
                std::cout << "WARNING: Could not read thermal restart file: " << error
                          << ". Initializing temperature with default values.\n";
                T_.assign(mesh_.npoin, thermal_config_.t_init);
            }
        }
    }

    apply_thermal_bcs(mesh_, thermal_config_, T_);
    T_old_ = T_;

    std::cout << "Thermal energy: enabled (Pr = " << thermal_config_.prandtl
              << ", Pr_t = " << thermal_config_.prandtl_turbulent;
    if (thermal_config_.rayleigh > 0.0) {
        std::cout << ", Ra = " << thermal_config_.rayleigh;
    } else if (thermal_config_.beta != 0.0) {
        std::cout << ", beta = " << thermal_config_.beta
                  << ", g = (" << thermal_config_.gravity[0] << ", " << thermal_config_.gravity[1] << ")";
    }
    std::cout << ")\n";

    return true;
}

bool Solver::initialize_periodic(const std::string& prefix) {
    (void)prefix;
    for (const auto& pair_cfg : periodic_config_.pairs) {
        int slave_tag = -1, master_tag = -1;
        try {
            slave_tag = std::stoi(pair_cfg.slave);
            master_tag = std::stoi(pair_cfg.master);
        } catch (...) {
            for (const auto& bg : mesh_.boundary_groups) {
                if (bg.name == pair_cfg.slave) slave_tag = bg.tag;
                if (bg.name == pair_cfg.master) master_tag = bg.tag;
            }
        }
        if (slave_tag >= 0 && master_tag >= 0) {
            std::vector<int> slave_nodes, master_nodes;
            if (!mesh_.boundary_groups.empty()) {
                for (const auto& bg : mesh_.boundary_groups) {
                    if (bg.tag == slave_tag) {
                        for (int s : bg.sides) {
                            slave_nodes.push_back(mesh_.iside[static_cast<std::size_t>(s)].node1);
                            slave_nodes.push_back(mesh_.iside[static_cast<std::size_t>(s)].node2);
                        }
                    } else if (bg.tag == master_tag) {
                        for (int s : bg.sides) {
                            master_nodes.push_back(mesh_.iside[static_cast<std::size_t>(s)].node1);
                            master_nodes.push_back(mesh_.iside[static_cast<std::size_t>(s)].node2);
                        }
                    }
                }
            } else {
                // Legacy mesh where sides have side_id
                for (const auto& side : mesh_.iside) {
                    if (side.side_id == slave_tag) {
                        slave_nodes.push_back(side.node1);
                        slave_nodes.push_back(side.node2);
                    } else if (side.side_id == master_tag) {
                        master_nodes.push_back(side.node1);
                        master_nodes.push_back(side.node2);
                    }
                }
            }
            std::sort(slave_nodes.begin(), slave_nodes.end());
            slave_nodes.erase(std::unique(slave_nodes.begin(), slave_nodes.end()), slave_nodes.end());
            std::sort(master_nodes.begin(), master_nodes.end());
            master_nodes.erase(std::unique(master_nodes.begin(), master_nodes.end()), master_nodes.end());

            std::optional<std::array<double, 2>> trans;
            if (!pair_cfg.auto_translation) trans = pair_cfg.translation;
            match_periodic_boundary_nodes(mesh_, slave_nodes, master_nodes, trans);
        }
    }

    // Auto-detect and match periodic boundaries if not already matched
    if (!mesh_.has_periodic() && periodic_config_.pairs.empty()) {
        std::map<int, std::vector<int>> periodic_patches;
        for (const auto& side : mesh_.iside) {
            if (side.flag == flags::kPeriodic) {
                periodic_patches[side.side_id].push_back(side.node1);
                periodic_patches[side.side_id].push_back(side.node2);
            }
        }
        for (auto& [id, nodes] : periodic_patches) {
            std::sort(nodes.begin(), nodes.end());
            nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        }
        if (periodic_patches.size() == 2) {
            auto it = periodic_patches.begin();
            auto patch_a = it->second;
            ++it;
            auto patch_b = it->second;
            match_periodic_boundary_nodes(mesh_, patch_a, patch_b);
        }
    }

    if (mesh_.has_periodic()) {
        std::cout << "Periodic boundary: enabled (" << mesh_.periodic_pairs.size()
                  << " node pairs, " << mesh_.periodic_slave_nodes.size() << " slave nodes)\n";
        if (periodic_config_.has_driving_force()) {
            std::cout << " Periodic driving force: (" << periodic_config_.driving_force[0]
                      << ", " << periodic_config_.driving_force[1] << ")\n";
        }
    }
    return true;
}

void Solver::advance_temperature() {
    if (!thermal_config_.enabled) return;

    const std::vector<double>* nut = turbulence_ ? &turbulence_->nut_elem() : nullptr;
    compute_thermal_diffusivity(mesh_, ani_, thermal_config_, nut, alpha_elem_);

    T_old_ = T_;
    assemble_scalar_transport(mesh_, u_, v_, delte_, T_, alpha_elem_, rhs_T_);

    for (int ip = 0; ip < mesh_.npoin; ++ip) {
        T_[ip] += rhs_T_[ip] * elcoe2_[ip];
    }

    if (mesh_.has_periodic()) {
        apply_periodic_field(mesh_, elcoe2_, T_);
    }

    apply_thermal_bcs(mesh_, thermal_config_, T_);
}

void Solver::update_effective_viscosity() {
    nu_eff_elem_.resize(mesh_.nelem);
    if (turbulence_) {
        const std::vector<double>& nut = turbulence_->nut_elem();
        const std::vector<double>& gamma = turbulence_->max_diffusivity_elem();
        for (int ie = 0; ie < mesh_.nelem; ++ie) {
            double eff = std::max(ani_ + nut[ie], gamma[ie]);
            if (thermal_config_.enabled && ie < static_cast<int>(alpha_elem_.size())) {
                eff = std::max(eff, alpha_elem_[ie]);
            }
            nu_eff_elem_[ie] = eff;
        }
    } else if (thermal_config_.enabled) {
        compute_thermal_diffusivity(mesh_, ani_, thermal_config_, nullptr, alpha_elem_);
        for (int ie = 0; ie < mesh_.nelem; ++ie) {
            nu_eff_elem_[ie] = std::max(ani_, alpha_elem_[ie]);
        }
    }
}

double Solver::max_turbulence_residual() const {
    double r = 0.0;
    for (const auto& e : turbulence_->residuals()) r = std::max(r, e.relative_change);
    return r;
}

std::string Solver::turbulence_log_line() const {
    std::ostringstream os;
    os << std::scientific << std::setprecision(6) << " Turbulence:";
    for (const auto& e : turbulence_->residuals()) os << " " << e.name << " change " << e.relative_change;
    const std::vector<double>& nut = turbulence_->nut();
    os << "  nut/nu max " << *std::max_element(nut.begin(), nut.end()) / ani_;
    if (walls_.has_walls()) {
        const WallUnitStats y = first_cell_yplus(walls_, wall_friction_velocity(mesh_, walls_, u_, v_, ani_), ani_);
        os << "  y+ (first cell) min " << y.min << " mean " << y.mean << " max " << y.max;
    }
    return os.str();
}

void Solver::output_results(int iitime, double time_val) {
    std::string prefix = config_.case_dir.empty() ? "" : (config_.case_dir + "/");
    std::string time_str;

    if (config_.control.transient_on < 1) {
        time_str = std::to_string(iitime);
    } else {
        time_str = std::to_string(static_cast<int>(time_val));
    }

    std::string restart_path = prefix + "results/" + config_.casename + ".var";

    std::vector<NamedField> scalars;
    if (thermal_config_.enabled) {
        scalars.push_back({"T", &T_});
    }

    std::vector<double> yplus;
    if (turbulence_) {
        for (const FieldRef& f : turbulence_->fields()) scalars.push_back({f.name, f.values});
        if (walls_.has_walls()) {
            yplus_field(walls_, wall_friction_velocity(mesh_, walls_, u_, v_, ani_), ani_, yplus);
            scalars.push_back({"wall_distance", &walls_.wall_distance()});
            scalars.push_back({"yplus", &yplus});
        }
    }

    if (config_.control.vtk_format == VtkFormat::AsciiVtk) {
        std::string vtk_path = prefix + "results/" + config_.casename + "_" + time_str + ".vtk";
        if (scalars.empty()) {
            IO::write_vtk(vtk_path, mesh_, u_, v_, p_);
        } else {
            IO::write_vtk(vtk_path, mesh_, u_, v_, p_, scalars);
        }
    } else {
        std::string vtu_path = prefix + "results/" + config_.casename + "_" + time_str + ".vtu";
        const bool compress = (config_.control.vtk_format == VtkFormat::CompressedVtu);
        if (scalars.empty()) {
            IO::write_vtu(vtu_path, mesh_, u_, v_, p_, compress);
        } else {
            IO::write_vtu(vtu_path, mesh_, u_, v_, p_, scalars, compress);
        }
    }
    IO::write_restart(restart_path, time_val, iitime, config_.control.transient_on, u_, v_, p_);

    if (thermal_config_.enabled) {
        const std::string thermal_restart_path = prefix + "results/" + config_.casename + ".thermal.var";
        IO::write_thermal_restart(thermal_restart_path, T_);
    }

    if (turbulence_) {
        std::vector<NamedField> state;
        for (const FieldRef& f : turbulence_->state_fields()) state.push_back({f.name, f.values});
        const std::string turb_path = prefix + "results/" + config_.casename + ".turb.var";
        if (!IO::write_turbulence_restart(turb_path, std::string(turbulence_->name()), state)) {
            std::cerr << "ERROR: CANNOT WRITE TURBULENCE RESTART FILE: " << turb_path << "\n";
        }
    }
}

double Solver::compute_physical_cfl() const {
    double dt = dt_phys_;
    if (dt <= 0.0) {
        if (config_.control.physical_dt > 0.0) {
            dt = config_.control.physical_dt;
        } else if (config_.control.dtfixed == 1 && config_.control.dtfix > 0.0) {
            dt = config_.control.dtfix;
        } else if (!delte_.empty() && delte_[0] > 0.0) {
            dt = delte_[0];
        } else {
            dt = 1.0e-3;
        }
    }

    double max_cfl = 0.0;
    double ref_vel = 1.0;
    if (inflow_config_.has_default) {
        ref_vel = std::hypot(inflow_config_.default_u, inflow_config_.default_v);
        if (ref_vel <= 1.0e-12) ref_vel = 1.0;
    }

    for (int ie = 0; ie < mesh_.nelem; ++ie) {
        const double amin = mesh_.alen_e[ie];
        if (amin <= 1.0e-15) continue;
        const ElementNodes nodes = mesh_.element_nodes(ie);
        double velc = 0.0;
        for (int in = 0; in < nodes.size(); ++in) {
            const int ip = nodes[in];
            velc = std::max(velc, std::hypot(u_[ip], v_[ip]));
        }
        if (velc <= 1.0e-5) velc = ref_vel;
        const double cfl_conv = velc * dt / amin;
        max_cfl = std::max(max_cfl, cfl_conv);
    }
    return max_cfl;
}

void Solver::advance_dual_time_step(int iitime) {
    const int n_phys = iitime - istart_ + 1;

    // 1. Determine physical time step size
    if (config_.control.physical_dt > 0.0) {
        dt_phys_ = config_.control.physical_dt;
    } else if (config_.control.dtfixed == 1 && config_.control.dtfix > 0.0) {
        dt_phys_ = config_.control.dtfix;
    } else if (dt_phys_ <= 0.0) {
        dt_phys_ = (config_.control.dtfix > 0.0) ? config_.control.dtfix : 1.0e-3;
    }

    // 2. Multi-step BDF coefficients:
    // n_phys == 1: BDF1 (first-order startup: (u^{n+1} - u^n) / dt)
    // n_phys >= 2: BDF2 (second-order: (3*u^{n+1} - 4*u^n + u^{n-1}) / (2*dt))
    double alpha0 = 1.0;
    double alpha1 = -1.0;
    double alpha2 = 0.0;
    if (n_phys >= 2 || has_prev_state_) {
        alpha0 = 1.5;
        alpha1 = -2.0;
        alpha2 = 0.5;
    }

    // 3. Compute pseudo-time step delta_tau across elements for CFL stability
    const double pseudo_cfl = (config_.control.csafm_tau > 0.0) ? config_.control.csafm_tau : config_.control.csafm;
    if (turbulence_ || thermal_config_.enabled) {
        update_effective_viscosity();
        compute_timestep(config_.solver.solver_opt,
                         config_.control.dtfixed,
                         0, // pseudo-time marches to steady state
                         config_.control.dtfix,
                         nu_eff_elem_,
                         pseudo_cfl,
                         mesh_,
                         velocity_,
                         delte_);
    } else {
        compute_timestep(config_.solver.solver_opt,
                         config_.control.dtfixed,
                         0, // pseudo-time marches to steady state
                         config_.control.dtfix,
                         ani_,
                         pseudo_cfl,
                         mesh_,
                         velocity_,
                         delte_);
    }

    if (dt_phys_ <= 0.0 && !delte_.empty() && delte_[0] > 0.0) {
        dt_phys_ = 10.0 * delte_[0];
    }

    // 4. Compute effective pseudo-time step delta_tau* = delta_tau / (1 + alpha0 * delta_tau / dt_phys)
    delte_star_.resize(mesh_.nelem);
    for (int ie = 0; ie < mesh_.nelem; ++ie) {
        delte_star_[ie] = delte_[ie] / (1.0 + alpha0 * (delte_[ie] / dt_phys_));
    }

    // 5. Update LHS lumped mass inverse and Laplacian scaled by delta_tau*
    update_lhs(mesh_, delte_star_, elcoe2_, pdiag_, gstif_);
    pressure_solver_->update(mesh_, gstif_, pdiag_);

    // 6. Sub-iteration initialization:
    // Initial guess at m = 0: u^m = u^n, v^m = v^n, p^m = p^n
    u_ = u_n_;
    v_ = v_n_;
    p_ = p_n_;

    int subiter_used = 0;
    double last_sub_res = 1.0;
    const int max_subiters = (config_.control.dual_time_max_subiter > 0) ? config_.control.dual_time_max_subiter : 30;
    const double tol = (config_.control.dual_time_tolerance > 0.0) ? config_.control.dual_time_tolerance : 1.0e-4;

    for (int sub = 0; sub < max_subiters; ++sub) {
        subiter_used = sub + 1;

        u_sub_old_ = u_;
        v_sub_old_ = v_;
        p_sub_old_ = p_;

        // --- Step 1: Intermediate Velocity Calculation ---
        if (turbulence_) {
            compute_step1(mesh_, ani_, turbulence_->nut_elem(), delte_, u_, v_, rhs_u_, rhs_v_);
        } else {
            compute_step1(mesh_, ani_, delte_, u_, v_, rhs_u_, rhs_v_);
        }

        if (thermal_config_.enabled) {
            add_boussinesq_buoyancy(mesh_, thermal_config_, 1.0 / ani_, T_, rhs_u_, rhs_v_);
        }

        if (periodic_config_.has_driving_force()) {
            const double fx = periodic_config_.driving_force[0];
            const double fy = periodic_config_.driving_force[1];
            for (int ip = 0; ip < mesh_.npoin; ++ip) {
                rhs_u_[ip] += fx * nodal_mass_[ip];
                rhs_v_[ip] += fy * nodal_mass_[ip];
            }
        }

        // Add BDF source term: S = -(alpha0 * u^m + alpha1 * u^n + alpha2 * u^{n-1}) / dt_phys
        const double inv_dt = 1.0 / dt_phys_;
        for (int ip = 0; ip < mesh_.npoin; ++ip) {
            const double su = -(alpha0 * u_[ip] + alpha1 * u_n_[ip] + alpha2 * u_prev_[ip]) * inv_dt;
            const double sv = -(alpha0 * v_[ip] + alpha1 * v_n_[ip] + alpha2 * v_prev_[ip]) * inv_dt;
            rhs_u_[ip] += su * nodal_mass_[ip];
            rhs_v_[ip] += sv * nodal_mass_[ip];
        }

        for (int ip = 0; ip < mesh_.npoin; ++ip) {
            u_[ip] += rhs_u_[ip] * elcoe2_[ip];
            v_[ip] += rhs_v_[ip] * elcoe2_[ip];
        }

        if (mesh_.has_periodic()) {
            apply_periodic_field(mesh_, elcoe2_, u_);
            apply_periodic_field(mesh_, elcoe2_, v_);
        }

        apply_symmetry_bc(mesh_, u_, v_);
        apply_velocity_bcs(mesh_, u_, v_, wall_specs_, inflow_config_);

        // --- Step 2: Pressure Calculation (Poisson Eq) ---
        compute_step2(mesh_, u_, v_, u_sub_old_, v_sub_old_, config_.control.theta[0], rhs1_);

        const linalg::SolveResult pcg_res = pressure_solver_->solve(rhs1_, p_);
        iiter_total_ += pcg_res.iterations;

        // --- Step 3: Velocity Correction ---
        compute_step3(mesh_, u_sub_old_, v_sub_old_, p_, p_sub_old_, delte_, rhs_u_, rhs_v_);

        for (int ip = 0; ip < mesh_.npoin; ++ip) {
            u_[ip] += rhs_u_[ip] * elcoe2_[ip];
            v_[ip] += rhs_v_[ip] * elcoe2_[ip];
        }

        if (mesh_.has_periodic()) {
            apply_periodic_field(mesh_, elcoe2_, u_);
            apply_periodic_field(mesh_, elcoe2_, v_);
        }

        apply_symmetry_bc(mesh_, u_, v_);
        apply_velocity_bcs(mesh_, u_, v_, wall_specs_, inflow_config_);

        // Check pseudo-time convergence
        double du_sq = 0.0;
        double u_sq = 0.0;
        for (int ip = 0; ip < mesh_.npoin; ++ip) {
            const double du = u_[ip] - u_sub_old_[ip];
            const double dv = v_[ip] - v_sub_old_[ip];
            du_sq += du * du + dv * dv;
            u_sq += u_[ip] * u_[ip] + v_[ip] * v_[ip];
        }
        last_sub_res = std::sqrt(du_sq / (u_sq + 1.0e-30));

        if (last_sub_res < tol && sub >= 1) {
            break;
        }
    }

    last_subiter_ = subiter_used;
    total_subiter_ += subiter_used;
    last_sub_res_ = last_sub_res;

    // 7. Advance physical time
    rtime_ += dt_phys_;

    // 8. Turbulence & Thermal coupling
    if (turbulence_) {
        turbulence_->advance(turbulence_context(), {u_, v_});
    }
    if (thermal_config_.enabled) {
        advance_temperature();
    }

    record_monitors(iitime);

    // 9. Update physical time history
    u_prev_ = u_n_;
    v_prev_ = v_n_;
    p_prev_ = p_n_;

    u_n_ = u_;
    v_n_ = v_;
    p_n_ = p_;

    u_old_ = u_;
    v_old_ = v_;
    p_old_ = p_;

    // 10. Update velocity magnitude
    velocity_old_ = velocity_;
    for (int ip = 0; ip < mesh_.npoin; ++ip) {
        velocity_[ip] = std::sqrt(u_[ip] * u_[ip] + v_[ip] * v_[ip] + 1.0e-16);
    }
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

        if (config_.control.dual_time) {
            advance_dual_time_step(iitime);

            auto t2 = std::chrono::steady_clock::now();

            if (iitime % config_.control.iwrite == 0 || iitime_start) {
                iitime_start = false;
                std::cout << std::setw(8) << iitime
                          << " DualSub: " << std::setw(3) << last_subiter_
                          << " DualRes: " << std::setw(15) << std::setprecision(8) << std::scientific << last_sub_res_
                          << " IterCum: " << std::setw(9) << iiter_total_ << "\n";
                std::cout << " Time: " << std::setprecision(16) << std::scientific << rtime_ << "\n";
                if (turbulence_) {
                    std::cout << turbulence_log_line() << "\n";
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
                } else {
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
        } else {
            // Store old timestep values
            u_old_ = u_;
            v_old_ = v_;
            p_old_ = p_;

        // Calculate time step size
        if (turbulence_ || thermal_config_.enabled) {
            // Diffusive limit with nu_eff = max(nu + nu_t, turbulence diffusivities).
            update_effective_viscosity();
            compute_timestep(config_.solver.solver_opt,
                             config_.control.dtfixed,
                             config_.control.transient_on,
                             config_.control.dtfix,
                             nu_eff_elem_,
                             config_.control.csafm,
                             mesh_,
                             velocity_,
                             delte_);
        } else {
            compute_timestep(config_.solver.solver_opt,
                             config_.control.dtfixed,
                             config_.control.transient_on,
                             config_.control.dtfix,
                             ani_,
                             config_.control.csafm,
                             mesh_,
                             velocity_,
                             delte_);
        }

        if (config_.control.transient_on > 0) {
            rtime_ += delte_[0];
        }

        // Update LHS lumped mass inverse and time-scaled Laplacian
        update_lhs(mesh_, delte_, elcoe2_, pdiag_, gstif_);
        pressure_solver_->update(mesh_, gstif_, pdiag_);

        // ==========================================
        // Step 1: Intermediate Velocity Calculation
        // ==========================================
        if (turbulence_) {
            compute_step1(mesh_, ani_, turbulence_->nut_elem(), delte_, u_, v_, rhs_u_, rhs_v_);
        } else {
            compute_step1(mesh_, ani_, delte_, u_, v_, rhs_u_, rhs_v_);
        }

        if (thermal_config_.enabled) {
            add_boussinesq_buoyancy(mesh_, thermal_config_, 1.0 / ani_, T_, rhs_u_, rhs_v_);
        }

        if (periodic_config_.has_driving_force()) {
            const double fx = periodic_config_.driving_force[0];
            const double fy = periodic_config_.driving_force[1];
            for (int ip = 0; ip < mesh_.npoin; ++ip) {
                rhs_u_[ip] += fx * nodal_mass_[ip];
                rhs_v_[ip] += fy * nodal_mass_[ip];
            }
        }

        for (int ip = 0; ip < mesh_.npoin; ++ip) {
            u_[ip] += rhs_u_[ip] * elcoe2_[ip];
            v_[ip] += rhs_v_[ip] * elcoe2_[ip];
        }

        if (mesh_.has_periodic()) {
            apply_periodic_field(mesh_, elcoe2_, u_);
            apply_periodic_field(mesh_, elcoe2_, v_);
        }

        apply_symmetry_bc(mesh_, u_, v_);
        apply_velocity_bcs(mesh_, u_, v_, wall_specs_, inflow_config_);

        // ==========================================
        // Step 2: Pressure Calculation (Poisson Eq)
        // ==========================================
        compute_step2(mesh_, u_, v_, u_old_, v_old_, config_.control.theta[0], rhs1_);

        const linalg::SolveResult pcg_res = pressure_solver_->solve(rhs1_, p_);

        iiter_total_ += pcg_res.iterations;
        if (!pcg_res.converged) {
            std::cout << " WARNING: pressure solve at step " << iitime << " did not converge ("
                      << linalg::to_string(pcg_res.reason) << ", " << pcg_res.iterations << " iterations)\n";
        }

        // ==========================================
        // Step 3: Velocity Correction
        // ==========================================
        compute_step3(mesh_, u_old_, v_old_, p_, p_old_, delte_, rhs_u_, rhs_v_);

        for (int ip = 0; ip < mesh_.npoin; ++ip) {
            u_[ip] += rhs_u_[ip] * elcoe2_[ip];
            v_[ip] += rhs_v_[ip] * elcoe2_[ip];
        }

        if (mesh_.has_periodic()) {
            apply_periodic_field(mesh_, elcoe2_, u_);
            apply_periodic_field(mesh_, elcoe2_, v_);
        }

        apply_symmetry_bc(mesh_, u_, v_);
        apply_velocity_bcs(mesh_, u_, v_, wall_specs_, inflow_config_);

        // ==========================================
        // Turbulence: advance with u^{n+1}, update nu_t
        // ==========================================
        if (turbulence_) {
            turbulence_->advance(turbulence_context(), {u_, v_});
        }

        // ==========================================
        // Thermal: advance with u^{n+1}
        // ==========================================
        if (thermal_config_.enabled) {
            advance_temperature();
        }

        record_monitors(iitime);

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
                std::cout << " " << pressure_solver_->label() << ": Solving for p, Initial residual = "
                          << pcg_res.reference_residual
                          << " Final residual = " << pcg_res.final_residual
                          << " Iter: " << pcg_res.iterations << "\n";
                if (turbulence_) {
                    std::cout << turbulence_log_line() << "\n";
                }

                // Turbulent runs also require converged turbulence equations.
                const bool turbulence_converged =
                    !turbulence_ || max_turbulence_residual() < config_.control.l2norm_vel_tolerance;
                if (config_.control.vel_check > 0 && hb1 < config_.control.l2norm_vel_tolerance &&
                    turbulence_converged) {
                    break;
                }
            }
        } else {
            // Transient simulation
            if (iitime % config_.control.iwrite == 0 || iitime_start) {
                iitime_start = false;
                std::cout << std::setw(8) << iitime
                          << " pInitial: " << std::setw(15) << std::setprecision(8) << std::scientific << pcg_res.reference_residual
                          << " pFinal: " << std::setw(15) << std::setprecision(8) << std::scientific << pcg_res.final_residual
                          << " Iter: " << std::setw(6) << pcg_res.iterations
                          << " IterCumulative: " << std::setw(9) << iiter_total_ << "\n";
                std::cout << " Time: " << std::setprecision(16) << std::scientific << rtime_ << "\n";
                if (turbulence_) {
                    std::cout << turbulence_log_line() << "\n";
                }
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
        } // end of non-dual-time else branch
    }

    // The final state of the run is always sampled (e.g. a steady run that
    // converged between two monitor intervals). Not for an empty run (restart
    // beyond the last step).
    if (completed_time >= istart_ && completed_time <= config_.control.ntime) {
        record_monitors(completed_time, true);
    }

    std::cout << " Simulation Completed\n";
    if (!pressure_solver_->legacy()) {
        const PressureSolver::SetupStatistics& st = pressure_solver_->statistics();
        std::cout << " Pressure solver setup: " << st.numeric_setups << " numeric setups, " << st.rescaled_reuses
                  << " rescaled reuses, " << st.structure_builds << " structure builds\n";
    }

    if (config_.control.transient_on < 1 && !config_.control.dual_time) {
        std::cout << " Outputting Data\n";
        output_results(completed_time, rtime_);
    }

    if (monitors_ && has_sample_ && monitors_->has_body()) {
        std::ostringstream line; // own stream: the format of std::cout stays as it is
        line << std::setprecision(8) << " Monitors (last sample): cd " << last_sample_.cd << ", cl " << last_sample_.cl
             << " (surface: cd " << last_sample_.cd_surface << ", cl " << last_sample_.cl_surface << ")\n";
        std::cout << line.str();

        if (monitor_time_history_.size() >= 20 && (config_.control.transient_on > 0 || config_.control.dual_time)) {
            const auto shedding = analyze_shedding();
            if (shedding.is_periodic) {
                const std::string bname = monitors_->config().body.empty() ? "body" : monitors_->config().body.front();
                std::cout << post::format_spectral_report(shedding, bname);
            }
        }
    }

    auto t_final = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(t_final - t_start).count();
    std::cout << " Time Taken (s): " << total_sec << "\n";
}

post::SheddingAnalysis Solver::analyze_shedding(double t_transient) const {
    if (!monitors_ || monitor_time_history_.size() < 10) return {};
    return post::analyze_shedding(monitor_time_history_, monitor_cl_history_, monitor_cd_history_,
                                  monitors_->config().reference_length,
                                  monitors_->config().reference_velocity,
                                  t_transient);
}

std::vector<post::RecirculationBubble> Solver::detect_recirculation(const std::string& patch_name, double u_ref) const {
    const auto sides = post::resolve_wall_sides(mesh_, patch_name);
    if (sides.empty()) return {};
    const auto profile = post::extract_skin_friction(mesh_, sides, ani_, u_, v_, p_, u_ref);
    return post::find_recirculation_bubbles(profile);
}

} // namespace hydrama
