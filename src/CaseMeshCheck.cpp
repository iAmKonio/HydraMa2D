#include "CaseMeshCheck.hpp"

#include "BoundaryConditions.hpp"
#include "BoundaryKind.hpp"
#include "IO.hpp"
#include "mesh/BoundaryNaming.hpp"
#include "mesh/MeshCheck.hpp"
#include "mesh/MeshQuality.hpp"

#include <iostream>
#include <map>
#include <stdexcept>

namespace hydrama {

bool check_case_mesh(const std::string& case_dir, const std::string& casename) {
    const std::string prefix = case_dir.empty() ? "" : case_dir + "/";
    Mesh mesh;
    if (!IO::read_case_mesh(prefix + "constant", casename, mesh)) return false;

    const std::vector<std::string> problems = mesh::check_mesh(mesh);
    if (!problems.empty()) {
        std::cerr << "ERROR: the mesh violates the solver's invariants:\n";
        for (const std::string& p : problems) std::cerr << "  - " << p << "\n";
        return false;
    }
    try {
        mesh.compute_shape_function_derivatives();
        mesh.compute_face_normals();
        mesh.determine_wall_nodes();
        mesh.compute_element_sizes();
        (void)parabolic_inlet_velocities(mesh);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return false;
    }

    std::cout << "Mesh check: " << mesh.num_triangles() << " triangles, " << mesh.num_quads() << " quadrilaterals, "
              << mesh.npoin << " nodes, " << mesh.nboun << " boundary sides: the solver's invariants hold\n";
    std::map<int, int> sides_per_flag;
    for (const BoundarySide& s : mesh.iside) ++sides_per_flag[s.flag];
    for (const auto& [flag, count] : sides_per_flag) {
        const mesh::BoundaryType type = mesh::boundary_type_from_flag(flag);
        std::cout << "Mesh check: flag " << flag << " ("
                  << (type == mesh::BoundaryType::Custom ? std::string("no known condition")
                                                         : std::string(mesh::to_string(type)))
                  << "): " << count << " boundary sides\n";
    }
    std::cout << mesh::format_quality_report(mesh, mesh::assess_mesh_quality(mesh));
    return true;
}

} // namespace hydrama
