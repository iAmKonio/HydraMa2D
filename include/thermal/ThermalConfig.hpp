#pragma once

// Thermal configuration: system/<case>.thermal
//
// Optional key-value file. If it does not exist, thermal transport is disabled
// and the solver behaves as pure isothermal flow.
//
// Format (one entry per line, '#' or '//' starts a comment, case-insensitive keys):
//   enabled                 true              # on | off | true | false
//   prandtl                 0.71              # molecular Prandtl number Pr = nu / alpha (default: 0.71 for air)
//   prandtl_turbulent       0.85              # turbulent Prandtl number Pr_t = nu_t / alpha_t (default: 0.85)
//   beta                    0.0               # thermal expansion coefficient beta (1/K)
//   reference_temperature   0.0               # T_ref (default: 0.0)
//   gravity                 0.0 -9.81         # gravity vector (gx gy)
//   rayleigh                0.0               # Rayleigh number Ra (dimensionless mode, overrides beta if > 0)
//   t_hot                   1.0               # default temperature for hot walls (flag 501)
//   t_cold                  0.0               # default temperature for cold walls (flag 502)
//   t_inlet                 0.0               # default temperature for inflow boundaries (flags 503, 507, 508)
//   t_init                  0.0               # initial temperature for uninitialized nodes
//
// Boundary overrides:
//   boundary 501 hot 1.0                      # wall flag 501 isothermal with T = 1.0
//   boundary 502 cold 0.0                     # wall flag 502 isothermal with T = 0.0
//   boundary 500 adiabatic                    # wall flag 500 zero normal heat flux
//   boundary 503 inlet 0.5                    # inflow flag 503 with T = 0.5
//   boundary 509 fixed 2.0                    # custom flag fixed temperature T = 2.0

#include <array>
#include <istream>
#include <map>
#include <optional>
#include <set>
#include <string>

namespace hydrama {

struct ThermalBoundaryOverride {
    enum class Kind {
        Adiabatic,
        FixedTemperature
    };

    Kind kind = Kind::Adiabatic;
    double temperature = 0.0;
};

struct ThermalConfig {
    bool enabled = false;
    double prandtl = 0.71;
    double prandtl_turbulent = 0.85;
    double beta = 0.0;
    double reference_temperature = 0.0;
    std::array<double, 2> gravity = {0.0, -9.81};
    double rayleigh = 0.0;
    double t_hot = 1.0;
    double t_cold = 0.0;
    double t_inlet = 0.0;
    double t_init = 0.0;

    std::map<int, ThermalBoundaryOverride> boundaries;

    // Checks whether boundary flag is adiabatic (zero normal heat flux)
    bool is_adiabatic(int flag) const noexcept;

    // Returns prescribed temperature for a boundary flag if fixed, or std::nullopt if adiabatic
    std::optional<double> boundary_temperature(int flag) const noexcept;

    // Computes effective buoyancy acceleration vector b such that f_b = b * (T - T_ref)
    std::array<double, 2> effective_buoyancy_vector(double re) const noexcept;
};

// Parse a .thermal stream into `config`.
bool parse_thermal_config(std::istream& in, ThermalConfig& config, std::string& error);

// Read system/<case>.thermal. Missing file is not an error (leaves enabled = false).
bool read_thermal_config(const std::string& filepath, ThermalConfig& config, std::string& error);

} // namespace hydrama
