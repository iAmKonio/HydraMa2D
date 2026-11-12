#pragma once

// HydraMa2D --check-mesh <case>: reads the mesh of a case exactly as a run
// does (IO::read_case_mesh), computes its geometry and reports
//   * the import summary (imported formats) or the legacy reader's output;
//   * the solver's mesh invariants (mesh::check_mesh);
//   * the parabolic inlets (open curves);
//   * the boundary sides per condition;
//   * the element quality (mesh/MeshQuality.hpp).
// Nothing else of the case is read and nothing is written.

#include <string>

namespace hydrama {

// True if the mesh can be run; the report goes to std::cout, errors to std::cerr.
bool check_case_mesh(const std::string& case_dir, const std::string& casename);

} // namespace hydrama
