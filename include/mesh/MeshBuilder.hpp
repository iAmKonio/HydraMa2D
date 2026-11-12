#pragma once

// MeshData (as read from a file) -> validated hydrama::Mesh.
//
// build_mesh() establishes the invariants the solver relies on (the same
// ones the legacy *.mesh files satisfy, see check_mesh in MeshCheck.hpp):
//
//   * nodes: 0-based and contiguous. Arbitrary / non-contiguous / sparse file
//     tags are remapped; file order is kept; nodes used by no triangle are
//     dropped (they would have no mass);
//   * elements: linear triangles, then bilinear quadrilaterals, each in file
//     order (Mesh numbers triangles first), with strictly positive Jacobian
//     determinants (counter-clockwise). Clockwise triangles swap their 2nd
//     and 3rd node; clockwise quadrilaterals are reversed (a, d, c, b), keeping
//     the first node. Quadrilaterals must be convex: det J is positive on the
//     element iff it is positive at its four corners. Degenerate, non-convex
//     and self-intersecting elements are errors. Duplicates (MSH 2.2 writes an
//     element once per physical group) are merged;
//   * boundary sides: every edge that belongs to exactly one triangle is
//     covered by exactly one side, oriented counter-clockwise with respect to
//     its element (domain on the left, so the face normal (dy, -dx) / L points
//     outwards, also for quadrilaterals), with its element index, side_id =
//     physical tag and the flag
//     code of its boundary condition.
//
// Boundary conditions come from the physical group names (BoundaryNaming.hpp),
// combined with the boundary type declared in the file if the reader provides
// one (MeshData::boundary_hints, e.g. CGNS BCType), unless
// options.flag_overrides (the *.bofl file) gives the physical tag a flag code. Every problem found is reported (up to a limit) in one
// MeshImportError.

#include "Mesh.hpp"
#include "mesh/BoundaryNaming.hpp"
#include "mesh/MeshData.hpp"

#include <map>
#include <string>
#include <vector>

namespace hydrama::mesh {

struct BuildOptions {
    // Physical line tag -> flag code (from a *.bofl file). Takes precedence
    // over the physical group name.
    std::map<int, int> flag_overrides;
    // Keep boundaries of recognised but unimplemented types (periodic, ...)
    // with flag 0 instead of rejecting the mesh. For tools and tests: the
    // solver would treat them as natural (do-nothing) boundaries.
    bool allow_unimplemented_boundaries = false;
    // Relative tolerances: |z - z0| <= planarity_tolerance * extent, and
    // |2 A| <= degenerate_tolerance * (longest edge)^2 is a degenerate triangle.
    double planarity_tolerance = 1.0e-9;
    double degenerate_tolerance = 1.0e-12;
    // Maximum number of individual problems listed in the error message.
    int max_reported_problems = 10;
};

enum class ConditionSource {
    Name,     // resolved from the physical group name (possibly refined by a declared type)
    FlagFile, // flag code from the *.bofl file
    FileType, // boundary type declared in the file (e.g. CGNS BCType); the name has no keyword
};

// One physical line group that defines a boundary condition.
struct BoundaryPatch {
    int tag = 0;                              // physical tag (= BoundarySide::side_id)
    std::string name;                         // physical name ("" if unnamed)
    BoundaryType type = BoundaryType::Custom;
    int flag = 0;                             // flag code given to its sides (0: none)
    ConditionSource source = ConditionSource::Name;
    std::string matched;                      // alias / code / declared type that decided the type
    int sides = 0;                            // boundary sides in this group. A side in several
                                              // groups (which must agree) counts in each; its
                                              // side_id is the smallest of their tags.
};

struct ImportReport {
    std::string format;                 // e.g. "Gmsh MSH 4.1 binary"
    int nodes_in_file = 0;
    int unused_nodes_dropped = 0;
    int triangles = 0;
    int quadrilaterals = 0;
    int reoriented_triangles = 0;       // clockwise in the file
    int reoriented_quadrilaterals = 0;  // clockwise in the file
    int duplicate_elements_merged = 0;  // triangles and lines
    int boundary_sides = 0;
    int interior_lines_ignored = 0;     // lines inside the domain without a boundary condition
    std::vector<BoundaryPatch> patches; // by physical tag
    std::vector<std::string> warnings;
};

// Throws MeshImportError.
Mesh build_mesh(const MeshData& data, const BuildOptions& options = {}, ImportReport* report = nullptr);

// Multi-line human-readable summary of a report (for the run log).
std::string format_report(const ImportReport& report);

} // namespace hydrama::mesh
