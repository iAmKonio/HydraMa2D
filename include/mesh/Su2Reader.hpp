#pragma once

// Reader for SU2 native mesh files (*.su2, ASCII; SU2 has no binary mesh
// format). Produces MeshData for build_mesh() (MeshBuilder.hpp).
//
// File structure (keywords in any order; NDIME= before the data sections):
//
//   % comment                        '%' starts a comment, also after data
//   NDIME= 2                         dimension: 2 (3 is rejected)
//   NELEM= n                         n element lines:  VTK-type node... [index]
//   5 0 1 4 0                          5 triangle, 9 quadrilateral
//   NPOIN= n [n_domain]              n point lines:    x y [index]
//   0.0 0.0 0
//   NMARK= m                         m markers:
//   MARKER_TAG= inlet                  name
//   MARKER_ELEMS= k                    k boundary elements: 3 node node [index]
//   3 0 1
//
// Node indices are 0-based positions in the NPOIN section (the optional
// trailing index is ignored, as SU2 does; a mismatch gives a warning).
// Also accepted: NZONE= 1 / IZONE= 1 headers, legacy NPERIODIC= blocks
// (skipped with a warning), and the FFD_* free-form-deformation definitions
// at the end of the file (ignored).
//
// Mapping to MeshData:
//   * node tag = SU2 point index (0-based), z = 0;
//   * triangles / quadrilaterals: physical 0, tag = element index (0-based);
//   * the i-th marker (1-based) is physical group i of dimension 1 named by
//     its MARKER_TAG; marker lines carry that physical tag. Their element tag
//     is their 0-based position among all marker elements of the file.
// Boundary conditions then follow the naming convention
// (docs/BOUNDARY_NAMING.md); *.bofl entries are keyed by marker number.

#include "mesh/MeshData.hpp"

#include <string>
#include <string_view>

namespace hydrama::mesh {

// Reads a file. Throws MeshImportError.
MeshData read_su2(const std::string& path);

// Parses file contents held in memory; `source` is used in messages.
// Throws MeshImportError.
MeshData parse_su2(std::string_view contents, const std::string& source);

// Name of a VTK cell type number as used by SU2, e.g. 5 -> "triangle";
// "" for unknown numbers.
std::string_view su2_element_name(int vtk_type);

} // namespace hydrama::mesh
