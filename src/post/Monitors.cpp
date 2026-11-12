#include "post/Monitors.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

namespace hydrama::post {

namespace {

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool parse_double(std::string token, double& value) {
    for (char& c : token)
        if (c == 'd' || c == 'D') c = 'e';
    errno = 0;
    char* end = nullptr;
    value = std::strtod(token.c_str(), &end);
    return !token.empty() && end == token.c_str() + token.size() && errno != ERANGE && std::isfinite(value);
}

bool parse_integer(const std::string& token, long& value) {
    errno = 0;
    char* end = nullptr;
    value = std::strtol(token.c_str(), &end, 10);
    return !token.empty() && end == token.c_str() + token.size() && errno != ERANGE;
}

bool valid_probe_name(const std::string& name) {
    if (name.empty()) return false;
    for (char c : name)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-' && c != '.') return false;
    return true;
}

std::string format_point(double x, double y) {
    std::ostringstream os;
    os << "(" << x << ", " << y << ")";
    return os.str();
}

} // namespace

bool parse_monitor_config(std::istream& in, MonitorConfig& config, std::string& error) {
    config = MonitorConfig{};
    std::set<std::string> seen;
    std::set<std::string> probe_names;
    std::string line;
    int number = 0;
    while (std::getline(in, line)) {
        ++number;
        std::istringstream ss(line.substr(0, line.find('#')));
        std::vector<std::string> t;
        for (std::string w; ss >> w;) t.push_back(w);
        if (t.empty()) continue;
        const std::string where = "line " + std::to_string(number) + ": ";
        const std::string key = lower(t[0]);
        auto once = [&]() {
            if (!seen.insert(key).second) {
                error = where + "'" + key + "' is given twice";
                return false;
            }
            return true;
        };
        auto one_value = [&]() {
            if (t.size() != 2) {
                error = where + "'" + key + "' needs exactly one value";
                return false;
            }
            return true;
        };
        if (key == "body") {
            if (!once()) return false;
            if (t.size() < 2) {
                error = where + "'body' needs at least one boundary group (name or tag)";
                return false;
            }
            config.body.assign(t.begin() + 1, t.end());
        } else if (key == "reference_velocity" || key == "reference_length") {
            if (!once() || !one_value()) return false;
            double value = 0.0;
            if (!parse_double(t[1], value) || !(value > 0.0)) {
                error = where + "'" + key + "' must be a number > 0, got '" + t[1] + "'";
                return false;
            }
            (key == "reference_velocity" ? config.reference_velocity : config.reference_length) = value;
        } else if (key == "interval") {
            if (!once() || !one_value()) return false;
            long value = 0;
            if (!parse_integer(t[1], value) || value < 1 || value > 1000000000L) {
                error = where + "'interval' must be an integer >= 1, got '" + t[1] + "'";
                return false;
            }
            config.interval = static_cast<int>(value);
        } else if (key == "probe") {
            if (t.size() != 4) {
                error = where + "'probe' needs a name and two coordinates: probe <name> <x> <y>";
                return false;
            }
            ProbeSpec p;
            p.name = t[1];
            if (!valid_probe_name(p.name)) {
                error = where + "invalid probe name '" + p.name + "' (letters, digits, '_', '-', '.')";
                return false;
            }
            if (!probe_names.insert(p.name).second) {
                error = where + "probe '" + p.name + "' is defined twice";
                return false;
            }
            if (!parse_double(t[2], p.x) || !parse_double(t[3], p.y)) {
                error = where + "invalid coordinates for probe '" + p.name + "'";
                return false;
            }
            config.probes.push_back(p);
        } else if (key == "moment_center" || key == "moment_origin" || key == "pitching_moment_center") {
            if (!once()) return false;
            if (t.size() != 3) {
                error = where + "'" + key + "' needs exactly two coordinates (x y)";
                return false;
            }
            if (!parse_double(t[1], config.moment_center_x) || !parse_double(t[2], config.moment_center_y)) {
                error = where + "'" + key + "' coordinates must be numbers";
                return false;
            }
            config.compute_moment = true;
        } else if (key == "pitching_moment" || key == "moment") {
            if (!once()) return false;
            config.compute_moment = true;
            if (t.size() == 3) {
                if (!parse_double(t[1], config.moment_center_x) || !parse_double(t[2], config.moment_center_y)) {
                    error = where + "'" + key + "' coordinates must be numbers";
                    return false;
                }
            }
        } else {
            error = where + "unknown key '" + t[0] +
                    "' (known: body, reference_velocity, reference_length, interval, probe, moment_center)";
            return false;
        }
    }
    if (config.empty()) {
        error = "the file defines neither a body nor a probe";
        return false;
    }
    return true;
}

bool read_monitor_config(const std::string& path, MonitorConfig& config, bool& found, std::string& error) {
    config = MonitorConfig{};
    found = std::filesystem::exists(path);
    if (!found) return true;
    std::ifstream in(path);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    return parse_monitor_config(in, config, error);
}

Monitors::Monitors(const Mesh& mesh, MonitorConfig config) : config_(std::move(config)) {
    if (!(config_.reference_velocity > 0.0) || !(config_.reference_length > 0.0))
        throw std::invalid_argument("monitors: the reference velocity and length must be > 0");
    if (config_.interval < 1) throw std::invalid_argument("monitors: the interval must be >= 1");

    std::set<int> sides;
    for (const std::string& entry : config_.body) {
        std::vector<int> matched;
        std::string label;
        for (const BoundaryGroup& g : mesh.boundary_groups) {
            if (!g.name.empty() && (g.name == entry || lower(g.name) == lower(entry))) {
                matched.insert(matched.end(), g.sides.begin(), g.sides.end());
                label += (label.empty() ? "" : ", ") + ("'" + g.name + "' (tag " + std::to_string(g.tag) + ")");
            }
        }
        long tag = 0;
        if (matched.empty() && parse_integer(entry, tag)) {
            if (!mesh.boundary_groups.empty()) {
                for (const BoundaryGroup& g : mesh.boundary_groups)
                    if (g.tag == tag) {
                        matched = g.sides;
                        label = (g.name.empty() ? "unnamed group" : "'" + g.name + "'") + std::string(" (tag ") +
                                std::to_string(g.tag) + ")";
                    }
            } else {
                for (int ib = 0; ib < mesh.nboun; ++ib)
                    if (mesh.iside[static_cast<std::size_t>(ib)].side_id == tag) matched.push_back(ib);
                label = "side id " + entry;
            }
        }
        if (matched.empty()) {
            std::string known;
            for (const BoundaryGroup& g : mesh.boundary_groups)
                known += (known.empty() ? "" : ", ") + (g.name.empty() ? std::to_string(g.tag) : g.name);
            throw std::invalid_argument("monitors: body '" + entry + "' matches no boundary group" +
                                        (known.empty() ? std::string(" or side id") : " (groups: " + known + ")"));
        }
        body_labels_.push_back(label);
        sides.insert(matched.begin(), matched.end());
    }
    body_sides_.assign(sides.begin(), sides.end());
    for (int ib : body_sides_) {
        const BoundarySide& s = mesh.iside[static_cast<std::size_t>(ib)];
        body_length_ += std::hypot(mesh.x[static_cast<std::size_t>(s.node2)] - mesh.x[static_cast<std::size_t>(s.node1)],
                                   mesh.y[static_cast<std::size_t>(s.node2)] - mesh.y[static_cast<std::size_t>(s.node1)]);
    }

    for (const ProbeSpec& p : config_.probes) {
        const std::optional<PointLocation> loc = locate_point(mesh, p.x, p.y);
        if (!loc)
            throw std::invalid_argument("monitors: probe '" + p.name + "' at " + format_point(p.x, p.y) +
                                        " lies outside the mesh");
        probes_.push_back(*loc);
    }
}

MonitorSample Monitors::evaluate(const Mesh& mesh, double nu, const std::vector<double>& u,
                                 const std::vector<double>& v, const std::vector<double>& p,
                                 const std::vector<double>* dudt, const std::vector<double>* dvdt) const {
    MonitorSample s;
    if (has_body()) {
        s.volume = volume_force(mesh, body_sides_, nu, u, v, p, dudt, dvdt);
        s.surface = surface_force(mesh, body_sides_, nu, u, v, p, config_.moment_center_x, config_.moment_center_y);
        const double U = config_.reference_velocity, L = config_.reference_length;
        s.cd = force_coefficient(s.volume.fx, U, L);
        s.cl = force_coefficient(s.volume.fy, U, L);
        s.cd_surface = force_coefficient(s.surface.fx, U, L);
        s.cl_surface = force_coefficient(s.surface.fy, U, L);
        s.cd_pressure = force_coefficient(s.surface.pressure_fx, U, L);
        s.cl_pressure = force_coefficient(s.surface.pressure_fy, U, L);
        s.cm = moment_coefficient(s.surface.mz, U, L);
        s.cm_pressure = moment_coefficient(s.surface.pressure_mz, U, L);
    }
    for (const PointLocation& loc : probes_) {
        s.probe_u.push_back(interpolate(loc, u));
        s.probe_v.push_back(interpolate(loc, v));
        s.probe_p.push_back(interpolate(loc, p));
    }
    return s;
}

std::string Monitors::description() const {
    std::ostringstream os;
    if (has_body()) {
        os << "Monitors: body";
        for (std::size_t k = 0; k < body_labels_.size(); ++k) os << (k ? ", " : " ") << body_labels_[k];
        os << ": " << body_sides_.size() << " sides, length " << body_length_ << "; U_ref "
           << config_.reference_velocity << ", L_ref " << config_.reference_length << "\n";
    }
    for (std::size_t k = 0; k < probes_.size(); ++k) {
        const ProbeSpec& p = config_.probes[k];
        os << "Monitors: probe '" << p.name << "' at " << format_point(p.x, p.y) << " in element "
           << probes_[k].element << (probes_[k].count == 4 ? " (quadrilateral)" : " (triangle)") << "\n";
    }
    return os.str();
}

std::string Monitors::file_header(const std::string& casename) const {
    std::ostringstream os;
    os << "# HydraMa2D monitors, case " << casename << "\n";
    if (has_body()) {
        os << "# body:";
        for (std::size_t k = 0; k < body_labels_.size(); ++k) os << (k ? ", " : " ") << body_labels_[k];
        os << "; " << body_sides_.size() << " sides, length " << std::setprecision(10) << body_length_ << "\n";
        os << "# coefficients C = F / (0.5 U_ref^2 L_ref), U_ref = " << config_.reference_velocity
           << ", L_ref = " << config_.reference_length << "\n";
        os << "# cd, cl: volume (reaction) form; *_surface: surface traction; *_pressure: its pressure part\n";
    }
    for (std::size_t k = 0; k < probes_.size(); ++k)
        os << "# probe " << config_.probes[k].name << " at " << format_point(config_.probes[k].x, config_.probes[k].y)
           << "\n";
    os << "step time";
    if (has_body()) {
        if (config_.compute_moment) {
            os << " cd cl cm cd_surface cl_surface cm_surface cd_pressure cl_pressure cm_pressure";
        } else {
            os << " cd cl cd_surface cl_surface cd_pressure cl_pressure";
        }
    }
    for (const ProbeSpec& p : config_.probes) os << " u_" << p.name << " v_" << p.name << " p_" << p.name;
    os << "\n";
    return os.str();
}

std::string Monitors::format_row(int step, double time, const MonitorSample& s) const {
    std::ostringstream os;
    os << step << " " << std::setprecision(12) << std::scientific << time;
    if (has_body()) {
        if (config_.compute_moment) {
            os << " " << s.cd << " " << s.cl << " " << s.cm
               << " " << s.cd_surface << " " << s.cl_surface << " " << s.cm
               << " " << s.cd_pressure << " " << s.cl_pressure << " " << s.cm_pressure;
        } else {
            os << " " << s.cd << " " << s.cl << " " << s.cd_surface << " " << s.cl_surface << " " << s.cd_pressure << " "
               << s.cl_pressure;
        }
    }
    for (std::size_t k = 0; k < s.probe_u.size(); ++k) os << " " << s.probe_u[k] << " " << s.probe_v[k] << " " << s.probe_p[k];
    os << "\n";
    return os.str();
}

} // namespace hydrama::post
