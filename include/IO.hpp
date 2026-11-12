#pragma once

#include "Config.hpp"
#include "Mesh.hpp"
#include "BoundaryConditions.hpp"
#include "turbulence/TurbulenceConfig.hpp"
#include <map>
#include <string>
#include <vector>

namespace hydrama {

// Mesh file of a case: constant/<case>.mesh (legacy HydraMa format, with
// constant/<case>.bofl), constant/<case>.msh (Gmsh MSH 2.2 / 4.1, ASCII or
// binary), constant/<case>.su2 (SU2 native, ASCII) or constant/<case>.cgns
// (CGNS, ADF or HDF5; needs a build with the CGNS library); the last three
// with an optional constant/<case>.bofl.
enum class MeshFileFormat { Legacy, Gmsh, Su2, Cgns };

struct MeshFileLocation {
    std::string path;
    MeshFileFormat format = MeshFileFormat::Legacy;
};

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

    // Optional system/<case>.thermal: missing file leaves thermal disabled (config.enabled == false) and returns true.
    static bool read_thermal(const std::string& filepath, ThermalConfig& config);

    // Optional system/<case>.walls: missing file leaves wall_specs empty and returns true.
    static bool read_walls(const std::string& filepath, std::map<int, MovingWallSpec>& wall_specs);
    static bool read_walls_config(const std::string& filepath, std::map<int, MovingWallSpec>& wall_specs, std::string& error);

    // Optional system/<case>.inflow: missing file leaves inflow empty (default uniform u=1, v=0) and returns true.
    static bool read_inflow(const std::string& filepath, InflowConfig& inflow);
    static bool read_inflow_config(const std::string& filepath, InflowConfig& inflow, std::string& error);

    static bool read_mesh(const std::string& filepath, Mesh& mesh);
    static bool read_boundary_flags(const std::string& filepath, Mesh& mesh);

    // Finds the mesh file of case `casename` in `constant_dir`. Exactly one of
    // <case>.mesh, <case>.msh, <case>.su2 and <case>.cgns must exist; prints
    // an error otherwise.
    static bool locate_mesh_file(const std::string& constant_dir, const std::string& casename,
                                 MeshFileLocation& location);

    // The mesh of a case, as a run reads it: locate_mesh_file, then the
    // legacy reader with <case>.bofl, or the Gmsh / SU2 / CGNS reader with
    // the optional <case>.bofl. Prints the import summary or the errors.
    static bool read_case_mesh(const std::string& constant_dir, const std::string& casename, Mesh& mesh);

    // Reads a Gmsh mesh (see docs/MESH_IMPORT.md): physical group names give
    // the boundary conditions (docs/BOUNDARY_NAMING.md); if `bofl_path` names
    // an existing file, its entries (physical tag -> flag code) take
    // precedence. Node tags are remapped, clockwise triangles reoriented,
    // boundary sides matched to their elements; every problem is reported.
    // Prints an import summary; returns false (with a message) on error.
    static bool read_gmsh_mesh(const std::string& msh_path, const std::string& bofl_path, Mesh& mesh);

    // Reads an SU2 mesh, otherwise as read_gmsh_mesh: marker names give the
    // boundary conditions; *.bofl entries are keyed by marker number (the
    // i-th marker of the file has physical tag i, 1-based).
    static bool read_su2_mesh(const std::string& su2_path, const std::string& bofl_path, Mesh& mesh);

    // Reads a CGNS mesh, otherwise as read_gmsh_mesh: BC names and BCTypes
    // give the boundary conditions; *.bofl entries are keyed by BC number
    // (the k-th BC_t of the zone has physical tag k, 1-based). Fails with a
    // clear message if HydraMa2D was built without the CGNS library.
    static bool read_cgns_mesh(const std::string& cgns_path, const std::string& bofl_path, Mesh& mesh);

    // Strict reader of a *.bofl table: a count n, then n pairs "id code"
    // (integers, ids unique). Returns false with a message on error.
    static bool read_boundary_flag_table(const std::string& filepath, std::map<int, int>& table);

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

    // Thermal restart file (results/<case>.thermal.var), written with 17 significant digits:
    static bool write_thermal_restart(const std::string& filepath,
                                      const std::vector<double>& T);

    // Reads a thermal restart file for a mesh with `npoin` nodes.
    static bool read_thermal_restart(const std::string& filepath, int npoin,
                                     std::vector<double>& T, std::string& error);
};

} // namespace hydrama
