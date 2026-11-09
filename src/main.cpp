#include "Config.hpp"
#include "Solver.hpp"

#include <iostream>
#include <string>
#include <filesystem>

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "==============================================================\n";
    std::cout << "  HydraMa (2d): Program Start (C++ Translation)               \n";
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
    std::filesystem::path p(input_arg);
    std::string case_dir = ".";
    std::string casename = input_arg;

    if (std::filesystem::is_directory(p)) {
        case_dir = p.string();
        // Look inside case_dir/constant/ for a .mesh file
        for (const auto& entry : std::filesystem::directory_iterator(p / "constant")) {
            if (entry.path().extension() == ".mesh") {
                casename = entry.path().stem().string();
                break;
            }
        }
    } else if (p.has_parent_path()) {
        case_dir = p.parent_path().string();
        casename = p.filename().string();
    } else {
        casename = input_arg;
        case_dir = ".";
        // If not found in ., check if a subdirectory matching casename exists
        if (!std::filesystem::exists("constant/" + casename + ".mesh")) {
            for (const auto& entry : std::filesystem::directory_iterator(".")) {
                if (entry.is_directory()) {
                    if (std::filesystem::exists(entry.path() / "constant" / (casename + ".mesh"))) {
                        case_dir = entry.path().string();
                        break;
                    }
                }
            }
        }
    }

    config.casename = casename;
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
