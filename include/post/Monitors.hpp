#pragma once

// Force and point monitors: system/<case>.monitors (optional).
//
// Without the file nothing is computed or written and the solver behaves
// exactly as before. Format (one entry per line, '#' starts a comment, numbers
// accept Fortran 'd' exponents):
//
//   body                cylinder      # boundary groups of the body: group names (as in the
//                                     # mesh file) or tags / side ids; several allowed
//   reference_velocity  1.0           # U_ref   (C = F / (0.5 U_ref^2 L_ref))
//   reference_length    1.0           # L_ref   (e.g. the cylinder diameter)
//   interval            1             # write every N time steps (default 1)
//   probe  front  1.45  2.0           # probe name, x, y (several allowed)
//
// Every run writes results/<case>.monitors.dat: comment lines describing the
// body and probes, a column header, and one row per interval and for the last
// step of the run:
//
//   step time cd cl cd_surface cl_surface cd_pressure cl_pressure [u_<probe> v_<probe> p_<probe>]...
//
// cd / cl: the volume (reaction) form of the force (post/Forces.hpp), with the
// time derivative in transient runs; *_surface: the surface traction;
// *_pressure: its pressure part. The x direction is the drag direction.
// A restarted run appends to the file.

#include "Mesh.hpp"
#include "post/Forces.hpp"
#include "post/Probes.hpp"

#include <istream>
#include <string>
#include <vector>

namespace hydrama::post {

struct ProbeSpec {
    std::string name;
    double x = 0.0;
    double y = 0.0;
};

struct MonitorConfig {
    std::vector<std::string> body; // group names or numeric tags / side ids
    double reference_velocity = 1.0;
    double reference_length = 1.0;
    int interval = 1;
    std::vector<ProbeSpec> probes;

    bool empty() const { return body.empty() && probes.empty(); }
};

// Parses the monitor file; false with a message (including the line number).
bool parse_monitor_config(std::istream& in, MonitorConfig& config, std::string& error);

// A missing file is not an error: config stays empty and `found` is false.
bool read_monitor_config(const std::string& path, MonitorConfig& config, bool& found, std::string& error);

// One evaluation of the monitors.
struct MonitorSample {
    BodyForce volume;   // reaction (weak) form
    BodyForce surface;  // surface traction
    double cd = 0.0, cl = 0.0;                 // from `volume`
    double cd_surface = 0.0, cl_surface = 0.0; // from `surface`
    double cd_pressure = 0.0, cl_pressure = 0.0;
    std::vector<double> probe_u, probe_v, probe_p;
};

class Monitors {
public:
    // Resolves the body sides and locates the probes. A body entry is a
    // boundary group name (exact, or ignoring case) or an integer: the tag
    // of a boundary group, or for legacy meshes without groups the side_id
    // of the sides. Throws std::invalid_argument if an entry matches no side,
    // a probe lies outside the mesh, or a reference value is not positive.
    Monitors(const Mesh& mesh, MonitorConfig config);

    const MonitorConfig& config() const { return config_; }
    bool has_body() const { return !body_sides_.empty(); }
    const std::vector<int>& body_sides() const { return body_sides_; }
    const std::vector<PointLocation>& probe_locations() const { return probes_; }

    // Needs the element geometry of the mesh (Mesh::compute_shape_function_derivatives).
    MonitorSample evaluate(const Mesh& mesh, double nu, const std::vector<double>& u, const std::vector<double>& v,
                           const std::vector<double>& p, const std::vector<double>* dudt = nullptr,
                           const std::vector<double>* dvdt = nullptr) const;

    // Run log lines: body (groups, sides, length) and probes (element).
    std::string description() const;
    // Comment lines and the column header of the data file.
    std::string file_header(const std::string& casename) const;
    std::string format_row(int step, double time, const MonitorSample& sample) const;

private:
    MonitorConfig config_;
    std::vector<int> body_sides_;
    std::vector<std::string> body_labels_; // per body entry: what it matched
    double body_length_ = 0.0;
    std::vector<PointLocation> probes_;
};

} // namespace hydrama::post
