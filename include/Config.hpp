#pragma once

#include "BoundaryConditions.hpp"
#include "PeriodicConfig.hpp"
#include "thermal/ThermalConfig.hpp"
#include <string>
#include <array>
#include <optional>

namespace hydrama {

enum class VtkFormat {
    AsciiVtk,      // Legacy ASCII .vtk (default)
    BinaryVtu,     // Uncompressed raw binary appended XML .vtu
    CompressedVtu  // Zlib-compressed raw binary appended XML .vtu
};

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
    VtkFormat vtk_format = VtkFormat::AsciiVtk;
};

struct PhysicalConfig {
    double re = 100.0;
    int pnode = 0; // 0-based reference pressure node
};

struct SolverConfig {
    // Positional values of *.solver.
    int solver_opt = 1; // 1 = Conjugate Gradient (also read by compute_timestep)
    double relToler = 0.1;
    double absToler = 1.0e-6;

    // Optional "key = value" settings of *.solver (see IO::read_solver).
    // The defaults are the original (legacy) PCG.
    std::string method = "cg";              // "cg" | "bicgstab" | "gmres" (linalg::krylov_catalog)
    std::string preconditioner = "jacobi";  // linalg::preconditioner_catalog
    std::string operator_format = "ebe";    // "ebe" (matrix-free) | "csr" (assembled)
    std::string reordering = "none";        // "none" | "rcm" (assembled preconditioners)
    int max_iterations = 100000;
    std::string residual_reference;         // "" = automatic: "first_iteration" for the
                                            // legacy PCG, "initial" otherwise;
                                            // or "first_iteration" | "initial" | "rhs"
    double ssor_omega = 1.0;                // 0 < omega < 2
    double diagonal_shift = 0.0;            // initial diagonal shift of ic0 / ilu0
    std::string preconditioner_reuse = "rescale"; // "rescale": reuse the setup of assembled
                                            // preconditioners while A = c A_ref (uniform dt);
                                            // "rebuild": numeric setup every update
    int gmres_restart = 30;                 // GMRES(m) restart length m >= 1
    std::string gmres_orthogonalization = "mgs"; // "mgs" | "cgs2"

    // The original solver: CG + Jacobi on the element-by-element operator,
    // natural ordering. Only this configuration uses the legacy convergence
    // criteria (relative tolerance measured after the first iteration,
    // zero-step breakdown guards) unless residual_reference is set.
    bool is_legacy_pcg() const {
        return method == "cg" && preconditioner == "jacobi" && operator_format == "ebe" && reordering == "none";
    }
};

struct SimulationConfig {
    std::string casename;
    std::string case_dir = ".";
    ControlConfig control;
    PhysicalConfig physical;
    SolverConfig solver;
    ThermalConfig thermal;
    InflowConfig inflow;
    PeriodicConfig periodic;
    std::optional<int> override_ntime;
    std::optional<double> override_re;
};

} // namespace hydrama
