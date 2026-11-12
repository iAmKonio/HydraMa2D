#pragma once

// Reader for CGNS files (ADF or HDF5), an optional feature: the CGNS library
// (with HDF5) is used if CMake finds it (HYDRAMA2D_WITH_CGNS, see
// docs/MESH_IMPORT.md). Without it HydraMa2D builds and runs normally, and
// reading a CGNS file gives a clear error.
//
// Supported: one CGNSBase_t with CellDimension 2, one unstructured zone,
// Cartesian coordinates, element sections BAR_2, TRI_3, QUAD_4 and MIXED (of those),
// NODE sections (ignored), and boundary conditions (ZoneBC_t / BC_t) on edges
// (ElementRange / ElementList, or PointRange / PointList at a cell, face or
// edge location) or on vertices (PointRange / PointList at Vertex: the
// boundary edges with both end points in the BC).
// Everything else (3D and higher-order elements, NGON_n / NFACE_n, several
// bases or zones, structured zones, non-Cartesian coordinates, ...) is
// reported as a problem of the mesh.
//
// Mapping to MeshData:
//   * node tag = CGNS vertex index (1-based);
//   * element tag = CGNS element index; physical 0;
//   * the k-th BC_t of the zone (1-based) is physical group k of dimension 1.
//     Its name is the BC name, or its family name if only that contains a
//     boundary keyword; its BCType (or the FamilyBC BCType of its family for
//     FamilySpecified) is the declared boundary type (MeshData::boundary_hints)
//     that build_mesh combines with the name. *.bofl entries are keyed by k.
//   * BCs on 2D cells or NODE elements (e.g. Gmsh writes one for the surface)
//     are regions, not boundaries: ignored with a warning.

#include "mesh/CgnsModel.hpp"
#include "mesh/MeshData.hpp"

#include <optional>
#include <string>

namespace hydrama::mesh {

// True if HydraMa2D was built with the CGNS library.
bool cgns_support_available();

// Version of the CGNS library HydraMa2D was built with, e.g. "4.5.0";
// "" without CGNS support.
std::string cgns_library_version();

// Reads the CGNS structure of a file. Throws MeshImportError (also when
// HydraMa2D was built without CGNS support).
cgns::CgnsFile read_cgns_file(const std::string& path);

// Translates a CGNS structure into MeshData. Always available (no library
// needed). Throws MeshImportError listing every problem found.
MeshData cgns_to_mesh_data(const cgns::CgnsFile& file, const std::string& source);

// Declared boundary type of a CGNS BCType_t; nullopt for types that carry no
// usable information (BCTypeNull, BCTypeUserDefined, BCGeneral, BCDirichlet,
// BCNeumann, BCExtrapolate, FamilySpecified) and the unsupported
// BCDegenerateLine / BCDegeneratePoint.
std::optional<BoundaryType> boundary_type_of_cgns_bc(int bc_type);

// read_cgns_file + cgns_to_mesh_data.
MeshData read_cgns(const std::string& path);

} // namespace hydrama::mesh
