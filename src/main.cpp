#include "CaseMeshCheck.hpp"
#include "Config.hpp"
#include "Solver.hpp"
#include "mesh/BoundaryNaming.hpp"
#include "mesh/CgnsReader.hpp"

#include <iostream>
#include <string>
#include <filesystem>

namespace {

// Mesh file extensions a case can use (see IO::locate_mesh_file).
constexpr const char* kMeshExtensions[] = {".mesh", ".msh", ".su2", ".cgns"};

bool is_mesh_file(const std::filesystem::path& p) {
    for (const char* ext : kMeshExtensions)
        if (p.extension() == ext) return true;
    return false;
}

bool case_mesh_exists(const std::filesystem::path& dir, const std::string& casename) {
    for (const char* ext : kMeshExtensions)
        if (std::filesystem::exists(dir / "constant" / (casename + ext))) return true;
    return false;
}

struct CaseLocation {
    std::string dir = ".";
    std::string name;
};

// Case directory and problem name from the command-line argument: a case
// directory (the mesh in its constant/ names the case), a path to
// <dir>/<case>, or a case name in the current directory or in one of its
// subdirectories.
CaseLocation locate_case(const std::string& input_arg) {
    CaseLocation c;
    std::filesystem::path p(input_arg);
    c.name = input_arg;
    if (std::filesystem::is_directory(p)) {
        c.dir = p.string();
        // Look inside case_dir/constant/ for a .mesh (legacy), .msh (Gmsh), .su2 (SU2) or .cgns (CGNS) file
        if (std::filesystem::is_directory(p / "constant")) {
            for (const auto& entry : std::filesystem::directory_iterator(p / "constant")) {
                if (is_mesh_file(entry.path())) {
                    c.name = entry.path().stem().string();
                    break;
                }
            }
        }
    } else if (p.has_parent_path()) {
        c.dir = p.parent_path().string();
        c.name = p.filename().string();
    } else if (!case_mesh_exists(".", c.name)) {
        // If not found in ., check if a subdirectory matching casename exists
        for (const auto& entry : std::filesystem::directory_iterator(".")) {
            if (entry.is_directory() && case_mesh_exists(entry.path(), c.name)) {
                c.dir = entry.path().string();
                break;
            }
        }
    }
    return c;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc > 1 && (std::string(argv[1]) == "--boundary-names" || std::string(argv[1]) == "--list-boundary-names")) {
        std::cout << hydrama::mesh::boundary_naming_summary();
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--check-mesh") {
        if (argc != 3) {
            std::cerr << "usage: HydraMa2D --check-mesh <case directory | case name>\n";
            return 2;
        }
        const CaseLocation c = locate_case(argv[2]);
        std::cout << "Checking the mesh of case " << c.name << (c.dir != "." ? " in " + c.dir : std::string()) << "\n";
        return hydrama::check_case_mesh(c.dir, c.name) ? 0 : 1;
    }

    std::cout << "\n";
    std::cout << "==============================================================\n";
    std::cout << "  HydraMa (2d): Program Start (C++ Translation)               \n";
    std::cout << "  Mesh formats: HydraMa, Gmsh, SU2"
              << (hydrama::mesh::cgns_support_available() ? ", CGNS (CGNS " + hydrama::mesh::cgns_library_version() + ")"
                                                           : std::string(" (built without CGNS)"))
              << "\n";
    std::cout << "--------------------------------------------------------------\n";
    std::cout << "==============================================================\n";

    std::string input_arg;
    int num_args = argc - 1;
    std::cout << " Number of Command-Line Input Arguments: " << num_args << "\n";

    if (num_args < 1) {
        std::cout << " Enter Problem Name?\n";
        if (!(std::cin >> input_arg)) {
            std::cerr << "Error reading problem name.\n";
            return 1;
        }
    } else {
        input_arg = argv[1];
    }

    hydrama::SimulationConfig config;

    // Detect case directory and problem name
    const CaseLocation location = locate_case(input_arg);
    const std::string& case_dir = location.dir;
    config.casename = location.name;
    config.case_dir = case_dir;

    std::cout << " Problem Name: " << config.casename << "\n";
    if (case_dir != ".") {
        std::cout << " Case Directory: " << config.case_dir << "\n";
    }

    hydrama::Solver solver(config);
    if (!solver.initialize()) {
        std::cerr << "ERROR: Failed to initialize simulation.\n";
        return 1;
    }

    solver.run();

    return 0;
}
