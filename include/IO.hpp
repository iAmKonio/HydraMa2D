#pragma once

#include "Config.hpp"
#include "Mesh.hpp"
#include <string>
#include <vector>

namespace hydrama {

class IO {
public:
    static double parse_fortran_double(std::string str);

    static bool read_control(const std::string& filepath, ControlConfig& config);
    static bool read_physical(const std::string& filepath, PhysicalConfig& config);
    static bool read_solver(const std::string& filepath, SolverConfig& config);

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
};

} // namespace hydrama
