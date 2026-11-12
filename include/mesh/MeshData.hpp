#pragma once

// Format-neutral mesh as read from a file, before any checking: original
// node tags, coordinates, elements with original node tags and their
// physical group. Readers (Gmsh, SU2, CGNS) produce MeshData;
// build_mesh() (MeshBuilder.hpp) turns it into a validated hydrama::Mesh.
//
// Elements carry ONE physical tag each. An element in several physical
// groups appears once per group (this is how MSH 2.2 stores it, and the MSH
// 4.1 reader expands its entity-based groups the same way); build_mesh()
// merges such duplicates.

#include "mesh/BoundaryNaming.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hydrama::mesh {

// Thrown by the readers and by build_mesh() for invalid input. The message
// names the file and, where possible, the line / byte offset and the entity.
class MeshImportError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class ElementType : unsigned char {
    Line2, // 2-node line (boundary side)
    Tri3,  // 3-node triangle
    Quad4, // 4-node bilinear quadrilateral
};

constexpr int node_count(ElementType t) {
    return t == ElementType::Line2 ? 2 : (t == ElementType::Tri3 ? 3 : 4);
}

constexpr int dimension(ElementType t) { return t == ElementType::Line2 ? 1 : 2; }

struct RawElement {
    ElementType type = ElementType::Tri3;
    std::array<std::int64_t, 4> nodes{}; // original node tags; first node_count(type) are used
    int physical = 0;                    // physical group tag, 0 = in no physical group
    int entity = 0;                      // elementary entity tag (diagnostics only)
    std::int64_t tag = 0;                // original element tag (diagnostics only)
};

struct MeshData {
    std::string source; // file name, for messages
    std::string format; // e.g. "Gmsh MSH 4.1 binary"

    // Nodes in file order.
    std::vector<std::int64_t> node_tags;
    std::vector<double> x, y, z;

    // Line / surface elements in file order (points and unsupported
    // dimensions are not stored).
    std::vector<RawElement> elements;

    // (dimension, physical tag) -> name. Unnamed groups have no entry.
    std::map<std::pair<int, int>, std::string> physical_names;

    // Boundary types declared by the file for a physical group (dimension,
    // tag), e.g. the CGNS BCType of a BC. build_mesh combines them with the
    // group name (resolve_boundary_name with a hint).
    struct BoundaryHint {
        BoundaryType type = BoundaryType::Wall;
        std::string source; // for messages, e.g. "CGNS BCType BCWallViscous"
    };
    std::map<std::pair<int, int>, BoundaryHint> boundary_hints;

    // Non-fatal observations of the reader (copied into the import report).
    std::vector<std::string> warnings;

    // Name of a physical group, "" if unnamed.
    std::string physical_name(int dim, int tag) const {
        const auto it = physical_names.find({dim, tag});
        return it == physical_names.end() ? std::string() : it->second;
    }
};

} // namespace hydrama::mesh
