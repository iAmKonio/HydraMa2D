#pragma once

// In-memory model of the part of a CGNS file that HydraMa2D reads (CGNS
// Standard Interface Data Structures, SIDS), independent of the CGNS library:
//
//   CgnsFile
//     Base (CGNSBase_t: CellDimension, PhysicalDimension)
//       Family (Family_t, with the BCType of its FamilyBC_t)
//       Zone (Zone_t: type, vertex / cell counts)
//         coordinates (GridCoordinates_t: CoordinateX, CoordinateY, ...)
//         Section (Elements_t: element type, element range, connectivity)
//         BoundaryCondition (ZoneBC_t/BC_t: BCType, point set, grid location,
//                            points, FamilyName)
//
// read_cgns_file() (CgnsReader.hpp) fills it with the CGNS library;
// cgns_to_mesh_data() translates it into MeshData for build_mesh(). Keeping
// the translation independent of the library means it is compiled and tested
// in every build, with or without CGNS.
//
// The integer codes are those of the CGNS standard (cgnslib.h, identical in
// CGNS 3.x and 4.x). CgnsReader.cpp checks them against the library headers
// at compile time.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace hydrama::mesh::cgns {

// ElementType_t
enum ElementTypeCode : int {
    kElementTypeNull = 0,
    kElementTypeUserDefined = 1,
    kNode = 2,
    kBar2 = 3,
    kTri3 = 5,
    kTri6 = 6,
    kQuad4 = 7,
    kTetra4 = 10,
    kHexa8 = 17,
    kMixed = 20,
    kNGon = 22,
    kNFace = 23,
    kLastElementType = 56, // HEXA_125
};

// BCType_t
enum BCTypeCode : int {
    kBCTypeNull = 0,
    kBCTypeUserDefined = 1,
    kBCAxisymmetricWedge = 2,
    kBCDegenerateLine = 3,
    kBCDegeneratePoint = 4,
    kBCDirichlet = 5,
    kBCExtrapolate = 6,
    kBCFarfield = 7,
    kBCGeneral = 8,
    kBCInflow = 9,
    kBCInflowSubsonic = 10,
    kBCInflowSupersonic = 11,
    kBCNeumann = 12,
    kBCOutflow = 13,
    kBCOutflowSubsonic = 14,
    kBCOutflowSupersonic = 15,
    kBCSymmetryPlane = 16,
    kBCSymmetryPolar = 17,
    kBCTunnelInflow = 18,
    kBCTunnelOutflow = 19,
    kBCWall = 20,
    kBCWallInviscid = 21,
    kBCWallViscous = 22,
    kBCWallViscousHeatFlux = 23,
    kBCWallViscousIsothermal = 24,
    kFamilySpecified = 25,
};

// PointSetType_t
enum PointSetTypeCode : int {
    kPointSetTypeNull = 0,
    kPointSetTypeUserDefined = 1,
    kPointList = 2,
    kPointListDonor = 3,
    kPointRange = 4,
    kPointRangeDonor = 5,
    kElementRange = 6,
    kElementList = 7,
    kCellListDonor = 8,
};

// GridLocation_t
enum GridLocationCode : int {
    kGridLocationNull = 0,
    kGridLocationUserDefined = 1,
    kVertex = 2,
    kCellCenter = 3,
    kFaceCenter = 4,
    kIFaceCenter = 5,
    kJFaceCenter = 6,
    kKFaceCenter = 7,
    kEdgeCenter = 8,
};

// ZoneType_t
enum ZoneTypeCode : int {
    kZoneTypeNull = 0,
    kZoneTypeUserDefined = 1,
    kStructured = 2,
    kUnstructured = 3,
};

struct ElementTypeInfo {
    std::string_view name; // CGNS name, e.g. "TRI_3"
    int nodes;             // nodes per element; 0 for MIXED, NGON_n, NFACE_n, Null, UserDefined
    int dimension;         // 0 NODE, 1 BAR_*, 2 TRI_* / QUAD_*, 3 volume types, -1 other
    bool linear;           // BAR_2, TRI_3, QUAD_4, TETRA_4, PYRA_5, PENTA_6, HEXA_8, NODE
};

// Information on an ElementType_t code; name "" for unknown codes.
const ElementTypeInfo& element_type_info(int type);
std::string element_type_name(int type); // "TRI_3", or "ElementType 99"
std::string bc_type_name(int type);      // "BCWall", or "BCType 99"
std::string point_set_type_name(int type);
std::string grid_location_name(int type);

struct Section {
    std::string name;
    int type = kElementTypeNull;
    std::int64_t start = 0, end = 0;           // element range (1-based, inclusive)
    std::vector<std::int64_t> connectivity;    // 1-based vertex indices; MIXED: type code before each element
    std::vector<std::int64_t> offsets;         // MIXED / NGON_n / NFACE_n (CGNS 4): ElementStartOffset, may be empty
    bool connectivity_read = true;             // false: the reader skipped an unsupported type
};

struct BoundaryCondition {
    std::string name;
    int type = kBCTypeNull;                    // BCType_t
    int point_set = kPointSetTypeNull;         // PointSetType_t
    int location = kVertex;                    // GridLocation_t
    std::vector<std::int64_t> points;          // vertex or element indices (1-based); ranges: {first, last}
    std::string family;                        // FamilyName_t, "" if none
};

struct Family {
    std::string name;
    bool has_bc = false;                       // has a FamilyBC_t
    int bc_type = kBCTypeNull;                 // its BCType
};

struct Zone {
    std::string name;
    int type = kUnstructured;
    std::int64_t vertices = 0, cells = 0;
    int grids = 1;                             // GridCoordinates_t nodes (the first is read)
    std::vector<std::string> coordinate_names; // e.g. CoordinateX
    std::vector<std::vector<double>> coordinates;
    std::vector<Section> sections;
    std::vector<BoundaryCondition> bcs;
};

struct Base {
    std::string name;
    int cell_dimension = 2;
    int physical_dimension = 2;
    std::vector<Family> families;
    std::vector<Zone> zones;
};

struct CgnsFile {
    std::string format;                        // e.g. "CGNS 3.40 HDF5"
    std::vector<Base> bases;
};

} // namespace hydrama::mesh::cgns
