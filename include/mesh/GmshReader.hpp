#pragma once

// Reader for Gmsh MSH files: format 2.2 (also 2.0 / 2.1) and 4.1, each in
// ASCII and binary (either byte order).
//
// Read: $MeshFormat, $PhysicalNames, $Entities (4.1), $Nodes, $Elements.
// Skipped: every other section ($Periodic, $NodeData, $Comments, ...), as the
// format specification requires. Rejected with a clear message: MSH 1.x /
// 3.x / 4.0, partitioned 4.1 files ($PartitionedEntities), higher-order and
// 3D elements.
//
// Stored element types: 2-node lines, 3-node triangles and 4-node quadrangles.
// Point elements are ignored.
//
// The reader checks the syntax only; topology and geometry are checked by
// build_mesh() (MeshBuilder.hpp).

#include "mesh/MeshData.hpp"

#include <string>
#include <string_view>

namespace hydrama::mesh {

// Reads a file. Throws MeshImportError.
MeshData read_gmsh(const std::string& path);

// Parses file contents held in memory; `source` is used in messages.
// Throws MeshImportError.
MeshData parse_gmsh(std::string_view contents, const std::string& source);

// Human-readable name of a Gmsh element type number, e.g. 9 ->
// "6-node second-order triangle"; "" for unknown numbers.
std::string_view gmsh_element_name(int type);

} // namespace hydrama::mesh
