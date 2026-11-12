#include "thermal/ThermalConfig.hpp"
#include "BoundaryKind.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace hydrama {

namespace {

std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool parse_bool(const std::string& token, bool& out) {
    std::string s = to_lower(token);
    if (s == "1" || s == "true" || s == "on" || s == "yes") {
        out = true;
        return true;
    }
    if (s == "0" || s == "false" || s == "off" || s == "no") {
        out = false;
        return true;
    }
    return false;
}

bool parse_double(const std::string& token, double& out) {
    std::string s = token;
    for (char& c : s) {
        if (c == 'd' || c == 'D') c = 'e';
    }
    try {
        std::size_t pos = 0;
        double val = std::stod(s, &pos);
        if (pos == s.size()) {
            out = val;
            return true;
        }
    } catch (...) {}
    return false;
}

} // namespace

bool ThermalConfig::is_adiabatic(int flag) const noexcept {
    auto it = boundaries.find(flag);
    if (it != boundaries.end()) {
        return it->second.kind == ThermalBoundaryOverride::Kind::Adiabatic;
    }
    // Default behaviors:
    // 500: adiabatic wall
    // 506: symmetry
    // 504: outlet
    // 509, 510: moving / rotating wall default adiabatic unless overridden
    if (flag == flags::kWall || flag == flags::kSymmetry || flag == flags::kOutlet ||
        flag == flags::kMovingWall || flag == flags::kRotatingWall) {
        return true;
    }
    return false;
}

std::optional<double> ThermalConfig::boundary_temperature(int flag) const noexcept {
    auto it = boundaries.find(flag);
    if (it != boundaries.end()) {
        if (it->second.kind == ThermalBoundaryOverride::Kind::FixedTemperature) {
            return it->second.temperature;
        }
        return std::nullopt; // Explicitly adiabatic
    }

    // Default flag semantics:
    if (flag == flags::kWallHot) {
        return t_hot;
    }
    if (flag == flags::kWallCold) {
        return t_cold;
    }
    if (flag == flags::kInlet || flag == flags::kInletProfile || flag == flags::kInletParabolic) {
        return t_inlet;
    }

    return std::nullopt;
}

std::array<double, 2> ThermalConfig::effective_buoyancy_vector(double re) const noexcept {
    // If Rayleigh number is specified (dimensionless mode):
    // In dimensionless Boussinesq equations where momentum is non-dimensionalized by Re:
    // f_b = Ra / (Pr * Re^2) * (T - T_ref) * (-g / ||g||)
    if (rayleigh > 0.0) {
        double g_norm = std::sqrt(gravity[0] * gravity[0] + gravity[1] * gravity[1]);
        if (g_norm < 1.0e-14) {
            // Default gravity downwards (0, -1), so buoyancy acts upwards (0, 1)
            double coeff = rayleigh / (prandtl * re * re);
            return {0.0, coeff};
        }
        double unit_buoyancy_x = -gravity[0] / g_norm;
        double unit_buoyancy_y = -gravity[1] / g_norm;
        double coeff = rayleigh / (prandtl * re * re);
        return {coeff * unit_buoyancy_x, coeff * unit_buoyancy_y};
    }

    // Physical / dimensional mode: f_b = -beta * (T - T_ref) * g
    // With g = (gx, gy), b = (-beta * gx, -beta * gy)
    return {-beta * gravity[0], -beta * gravity[1]};
}

bool parse_thermal_config(std::istream& in, ThermalConfig& config, std::string& error) {
    config = ThermalConfig{};
    error.clear();

    std::string line;
    int line_num = 0;

    while (std::getline(in, line)) {
        ++line_num;
        auto hash_pos = line.find('#');
        if (hash_pos != std::string::npos) line.erase(hash_pos);
        auto slash_pos = line.find("//");
        if (slash_pos != std::string::npos) line.erase(slash_pos);

        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) tokens.push_back(token);
        if (tokens.empty()) continue;

        std::string key = to_lower(tokens[0]);

        if (key == "enabled" || key == "thermal") {
            if (tokens.size() < 2) {
                config.enabled = true; // bare "enabled" or "thermal" turns on
            } else {
                bool b = false;
                if (!parse_bool(tokens[1], b)) {
                    error = "line " + std::to_string(line_num) + ": invalid boolean for '" + key + "'";
                    return false;
                }
                config.enabled = b;
            }
        } else if (key == "prandtl" || key == "pr") {
            if (tokens.size() < 2 || !parse_double(tokens[1], config.prandtl) || config.prandtl <= 0.0) {
                error = "line " + std::to_string(line_num) + ": 'prandtl' must be a positive number";
                return false;
            }
        } else if (key == "prandtl_turbulent" || key == "pr_t" || key == "prt") {
            if (tokens.size() < 2 || !parse_double(tokens[1], config.prandtl_turbulent) || config.prandtl_turbulent <= 0.0) {
                error = "line " + std::to_string(line_num) + ": 'prandtl_turbulent' must be a positive number";
                return false;
            }
        } else if (key == "beta") {
            if (tokens.size() < 2 || !parse_double(tokens[1], config.beta)) {
                error = "line " + std::to_string(line_num) + ": invalid value for 'beta'";
                return false;
            }
        } else if (key == "reference_temperature" || key == "t_ref" || key == "tref") {
            if (tokens.size() < 2 || !parse_double(tokens[1], config.reference_temperature)) {
                error = "line " + std::to_string(line_num) + ": invalid value for 'reference_temperature'";
                return false;
            }
        } else if (key == "gravity") {
            if (tokens.size() < 3 || !parse_double(tokens[1], config.gravity[0]) || !parse_double(tokens[2], config.gravity[1])) {
                error = "line " + std::to_string(line_num) + ": 'gravity' requires two numbers (gx gy)";
                return false;
            }
        } else if (key == "gravity_x") {
            if (tokens.size() < 2 || !parse_double(tokens[1], config.gravity[0])) {
                error = "line " + std::to_string(line_num) + ": invalid value for 'gravity_x'";
                return false;
            }
        } else if (key == "gravity_y") {
            if (tokens.size() < 2 || !parse_double(tokens[1], config.gravity[1])) {
                error = "line " + std::to_string(line_num) + ": invalid value for 'gravity_y'";
                return false;
            }
        } else if (key == "rayleigh" || key == "ra") {
            if (tokens.size() < 2 || !parse_double(tokens[1], config.rayleigh) || config.rayleigh < 0.0) {
                error = "line " + std::to_string(line_num) + ": 'rayleigh' must be a non-negative number";
                return false;
            }
        } else if (key == "t_hot" || key == "thot") {
            if (tokens.size() < 2 || !parse_double(tokens[1], config.t_hot)) {
                error = "line " + std::to_string(line_num) + ": invalid value for 't_hot'";
                return false;
            }
        } else if (key == "t_cold" || key == "tcold") {
            if (tokens.size() < 2 || !parse_double(tokens[1], config.t_cold)) {
                error = "line " + std::to_string(line_num) + ": invalid value for 't_cold'";
                return false;
            }
        } else if (key == "t_inlet" || key == "tinlet") {
            if (tokens.size() < 2 || !parse_double(tokens[1], config.t_inlet)) {
                error = "line " + std::to_string(line_num) + ": invalid value for 't_inlet'";
                return false;
            }
        } else if (key == "t_init" || key == "tinit") {
            if (tokens.size() < 2 || !parse_double(tokens[1], config.t_init)) {
                error = "line " + std::to_string(line_num) + ": invalid value for 't_init'";
                return false;
            }
        } else if (key == "boundary") {
            // boundary <flag> <type> [value]
            // type: hot [val], cold [val], adiabatic, inlet [val], fixed <val>
            if (tokens.size() < 3) {
                error = "line " + std::to_string(line_num) + ": 'boundary' requires <flag> <type> [value]";
                return false;
            }
            int flag = 0;
            try {
                flag = std::stoi(tokens[1]);
            } catch (...) {
                error = "line " + std::to_string(line_num) + ": invalid boundary flag '" + tokens[1] + "'";
                return false;
            }

            std::string btype = to_lower(tokens[2]);
            ThermalBoundaryOverride b_ov;

            if (btype == "adiabatic" || btype == "isolated" || btype == "neumann") {
                b_ov.kind = ThermalBoundaryOverride::Kind::Adiabatic;
            } else if (btype == "hot") {
                b_ov.kind = ThermalBoundaryOverride::Kind::FixedTemperature;
                b_ov.temperature = config.t_hot;
                if (tokens.size() >= 4 && !parse_double(tokens[3], b_ov.temperature)) {
                    error = "line " + std::to_string(line_num) + ": invalid temperature for hot boundary";
                    return false;
                }
            } else if (btype == "cold") {
                b_ov.kind = ThermalBoundaryOverride::Kind::FixedTemperature;
                b_ov.temperature = config.t_cold;
                if (tokens.size() >= 4 && !parse_double(tokens[3], b_ov.temperature)) {
                    error = "line " + std::to_string(line_num) + ": invalid temperature for cold boundary";
                    return false;
                }
            } else if (btype == "inlet") {
                b_ov.kind = ThermalBoundaryOverride::Kind::FixedTemperature;
                b_ov.temperature = config.t_inlet;
                if (tokens.size() >= 4 && !parse_double(tokens[3], b_ov.temperature)) {
                    error = "line " + std::to_string(line_num) + ": invalid temperature for inlet boundary";
                    return false;
                }
            } else if (btype == "fixed" || btype == "dirichlet" || btype == "isothermal") {
                if (tokens.size() < 4 || !parse_double(tokens[3], b_ov.temperature)) {
                    error = "line " + std::to_string(line_num) + ": 'fixed' boundary requires a temperature value";
                    return false;
                }
                b_ov.kind = ThermalBoundaryOverride::Kind::FixedTemperature;
            } else {
                error = "line " + std::to_string(line_num) + ": unknown boundary type '" + btype + "' (expected adiabatic|hot|cold|inlet|fixed)";
                return false;
            }
            config.boundaries[flag] = b_ov;
        } else {
            error = "line " + std::to_string(line_num) + ": unknown option '" + tokens[0] + "'";
            return false;
        }
    }

    return true;
}

bool read_thermal_config(const std::string& filepath, ThermalConfig& config, std::string& error) {
    config = ThermalConfig{};
    error.clear();
    std::error_code ec;
    if (!std::filesystem::exists(filepath, ec)) {
        // Missing file -> disabled by default, not an error
        return true;
    }
    std::ifstream in(filepath);
    if (!in.is_open()) {
        error = "cannot open " + filepath;
        return false;
    }
    if (!parse_thermal_config(in, config, error)) {
        return false;
    }
    // If the file exists and parsed successfully without an explicit "enabled = false", enable by default
    config.enabled = true;
    return true;
}

} // namespace hydrama
