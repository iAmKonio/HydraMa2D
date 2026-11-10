#pragma once

#include "Config.hpp"
#include "Mesh.hpp"
#include "turbulence/TurbulenceConfig.hpp"
#include <string>
#include <vector>

namespace hydrama {

// Named nodal field for output (VTK point scalars, turbulence restart).
struct NamedField {
    std::string name;
    const std::vector<double>* values = nullptr;
};

// Contents of a turbulence restart file.
struct TurbulenceRestartData {
    std::string model;                        // model name, e.g. "spalart-allmaras"
    std::vector<std::string> names;           // transported variables
    std::vector<std::vector<double>> values;  // values[k][node]
};

class IO {
public:
    static double parse_fortran_double(std::string str);

    static bool read_control(const std::string& filepath, ControlConfig& config);
    static bool read_physical(const std::string& filepath, PhysicalConfig& config);
    static bool read_solver(const std::string& filepath, SolverConfig& config);

    // Optional system/<case>.turbulence: a missing file yields a laminar
    // configuration (config.present == false) and returns true.
    static bool read_turbulence(const std::string& filepath, TurbulenceConfig& config);

    static bool read_mesh(const std::string& filepath, Mesh& mesh);
    static bool read_boundary_flags(const std::string& filepath, Mesh& mesh);

    static bool read_restart(const std::string& filepath,
                             double& rtime, int& istart,
                             std::vector<double>& u,
                             std::vector<double>& v,
                             std::vector<double>& p);

    static bool write_restart(const std::string& filepath,
                              double rtime, int iitime,
                              int transient_on,
                              const std::vector<double>& u,
                              const std::vector<double>& v,
                              const std::vector<double>& p);

    static bool write_vtk(const std::string& filepath,
                          const Mesh& mesh,
                          const std::vector<double>& u,
                          const std::vector<double>& v,
                          const std::vector<double>& p);

    // Same file with additional point scalars appended after P (one
    // "SCALARS <name> float 1" section each, in the given order). With no
    // extra fields the output is identical to the overload above.
    static bool write_vtk(const std::string& filepath,
                          const Mesh& mesh,
                          const std::vector<double>& u,
                          const std::vector<double>& v,
                          const std::vector<double>& p,
                          const std::vector<NamedField>& point_scalars);

    // Turbulence restart file (results/<case>.turb.var), written with 17
    // significant digits so that restarted runs continue bit for bit:
    //   HydraMa2D-turbulence-restart 1
    //   model <name>
    //   npoin <n>
    //   fields <m> <name_1> ... <name_m>
    //   <value_1> ... <value_m>          (one line per node)
    static bool write_turbulence_restart(const std::string& filepath,
                                         const std::string& model,
                                         const std::vector<NamedField>& fields);

    // Reads a turbulence restart file for a mesh with `npoin` nodes. Returns
    // false with a message in `error` for missing files, format errors, a
    // different node count, or non-finite values.
    static bool read_turbulence_restart(const std::string& filepath, int npoin,
                                        TurbulenceRestartData& data, std::string& error);
};

} // namespace hydrama
