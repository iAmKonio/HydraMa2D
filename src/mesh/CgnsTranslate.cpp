// CGNS structure (CgnsModel.hpp) -> MeshData. Independent of the CGNS
// library: compiled and tested in every build.

#include "Problems.hpp"
#include "mesh/CgnsReader.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>

namespace hydrama::mesh {

namespace cgns {

namespace {

// ElementType_t codes 0..56 of the CGNS standard.
constexpr ElementTypeInfo kElementTypes[] = {
    {"ElementTypeNull", 0, -1, false}, {"ElementTypeUserDefined", 0, -1, false},
    {"NODE", 1, 0, true},      {"BAR_2", 2, 1, true},     {"BAR_3", 3, 1, false},    {"TRI_3", 3, 2, true},
    {"TRI_6", 6, 2, false},    {"QUAD_4", 4, 2, true},    {"QUAD_8", 8, 2, false},   {"QUAD_9", 9, 2, false},
    {"TETRA_4", 4, 3, true},   {"TETRA_10", 10, 3, false}, {"PYRA_5", 5, 3, true},   {"PYRA_14", 14, 3, false},
    {"PENTA_6", 6, 3, true},   {"PENTA_15", 15, 3, false}, {"PENTA_18", 18, 3, false}, {"HEXA_8", 8, 3, true},
    {"HEXA_20", 20, 3, false}, {"HEXA_27", 27, 3, false}, {"MIXED", 0, -1, false},   {"PYRA_13", 13, 3, false},
    {"NGON_n", 0, -1, false},  {"NFACE_n", 0, -1, false}, {"BAR_4", 4, 1, false},    {"TRI_9", 9, 2, false},
    {"TRI_10", 10, 2, false},  {"QUAD_12", 12, 2, false}, {"QUAD_16", 16, 2, false}, {"TETRA_16", 16, 3, false},
    {"TETRA_20", 20, 3, false}, {"PYRA_21", 21, 3, false}, {"PYRA_29", 29, 3, false}, {"PYRA_30", 30, 3, false},
    {"PENTA_24", 24, 3, false}, {"PENTA_38", 38, 3, false}, {"PENTA_40", 40, 3, false}, {"HEXA_32", 32, 3, false},
    {"HEXA_56", 56, 3, false}, {"HEXA_64", 64, 3, false}, {"BAR_5", 5, 1, false},    {"TRI_12", 12, 2, false},
    {"TRI_15", 15, 2, false},  {"QUAD_P4_16", 16, 2, false}, {"QUAD_25", 25, 2, false}, {"TETRA_22", 22, 3, false},
    {"TETRA_34", 34, 3, false}, {"TETRA_35", 35, 3, false}, {"PYRA_P4_29", 29, 3, false}, {"PYRA_50", 50, 3, false},
    {"PYRA_55", 55, 3, false}, {"PENTA_33", 33, 3, false}, {"PENTA_66", 66, 3, false}, {"PENTA_75", 75, 3, false},
    {"HEXA_44", 44, 3, false}, {"HEXA_98", 98, 3, false}, {"HEXA_125", 125, 3, false},
};
static_assert(sizeof(kElementTypes) / sizeof(kElementTypes[0]) == kLastElementType + 1);

constexpr std::string_view kBCTypeNames[] = {
    "BCTypeNull",      "BCTypeUserDefined",  "BCAxisymmetricWedge", "BCDegenerateLine", "BCDegeneratePoint",
    "BCDirichlet",     "BCExtrapolate",      "BCFarfield",          "BCGeneral",        "BCInflow",
    "BCInflowSubsonic", "BCInflowSupersonic", "BCNeumann",          "BCOutflow",        "BCOutflowSubsonic",
    "BCOutflowSupersonic", "BCSymmetryPlane", "BCSymmetryPolar",    "BCTunnelInflow",   "BCTunnelOutflow",
    "BCWall",          "BCWallInviscid",     "BCWallViscous",       "BCWallViscousHeatFlux",
    "BCWallViscousIsothermal", "FamilySpecified",
};
constexpr std::string_view kPointSetNames[] = {"PointSetTypeNull", "PointSetTypeUserDefined", "PointList",
                                               "PointListDonor",   "PointRange",              "PointRangeDonor",
                                               "ElementRange",     "ElementList",             "CellListDonor"};
constexpr std::string_view kGridLocationNames[] = {"GridLocationNull", "GridLocationUserDefined", "Vertex",
                                                   "CellCenter",       "FaceCenter",              "IFaceCenter",
                                                   "JFaceCenter",      "KFaceCenter",             "EdgeCenter"};

template <std::size_t N>
std::string name_of(const std::string_view (&names)[N], int code, const char* what) {
    if (code >= 0 && static_cast<std::size_t>(code) < N) return std::string(names[code]);
    return std::string(what) + " " + std::to_string(code);
}

} // namespace

const ElementTypeInfo& element_type_info(int type) {
    static const ElementTypeInfo unknown{"", 0, -1, false};
    if (type >= 0 && type <= kLastElementType) return kElementTypes[type];
    return unknown;
}

std::string element_type_name(int type) {
    const ElementTypeInfo& info = element_type_info(type);
    return info.name.empty() ? "ElementType " + std::to_string(type) : std::string(info.name);
}

std::string bc_type_name(int type) { return name_of(kBCTypeNames, type, "BCType"); }
std::string point_set_type_name(int type) { return name_of(kPointSetNames, type, "PointSetType"); }
std::string grid_location_name(int type) { return name_of(kGridLocationNames, type, "GridLocation"); }

} // namespace cgns

using namespace cgns;

std::optional<BoundaryType> boundary_type_of_cgns_bc(int bc_type) {
    switch (bc_type) {
    case kBCWall:
    case kBCWallViscous:
    case kBCWallViscousHeatFlux:
    case kBCWallViscousIsothermal: return BoundaryType::Wall;
    case kBCWallInviscid:
    case kBCSymmetryPlane: return BoundaryType::Symmetry;
    case kBCSymmetryPolar: return BoundaryType::Axis;
    case kBCInflow:
    case kBCInflowSubsonic:
    case kBCTunnelInflow: return BoundaryType::Inlet;
    case kBCInflowSupersonic: return BoundaryType::SupersonicInlet;
    case kBCOutflow:
    case kBCOutflowSubsonic:
    case kBCTunnelOutflow: return BoundaryType::Outlet;
    case kBCOutflowSupersonic: return BoundaryType::SupersonicOutlet;
    case kBCFarfield: return BoundaryType::Farfield;
    case kBCAxisymmetricWedge: return BoundaryType::Wedge;
    default: return std::nullopt; // no usable information, or unsupported (degenerate)
    }
}

namespace {

using detail::Problems;

enum class Kind : unsigned char { None, Node, Bar, Cell };

struct ElementRef {
    Kind kind = Kind::None;
    std::size_t index = 0; // into bars or into MeshData::elements (cells)
};

// Elements of a section. Single-type sections are described arithmetically
// (no per-element data: a NODE section may be huge); MIXED sections store one
// reference per element.
struct SectionRange {
    std::int64_t start, end;
    std::size_t section;
    Kind kind = Kind::None;       // single type: kind of every element
    std::size_t first_index = 0;  // single type: index of the first element
    std::vector<ElementRef> refs; // MIXED: one per element

    ElementRef at(std::int64_t id) const {
        if (!refs.empty()) return refs[static_cast<std::size_t>(id - start)];
        return {kind, first_index + static_cast<std::size_t>(id - start)};
    }
};

struct Bar {
    RawElement element;
    std::vector<int> bcs; // BCs (physical tags) that contain it
};

std::string section_label(const Section& s) {
    return "section '" + s.name + "' (" + element_type_name(s.type) + ", elements " + std::to_string(s.start) + "-" +
           std::to_string(s.end) + ")";
}

std::string list_names(const std::vector<std::string>& names) {
    std::string out;
    for (std::size_t i = 0; i < names.size() && i < 5; ++i) out += (i ? ", '" : "'") + names[i] + "'";
    if (names.size() > 5) out += ", ...";
    return out;
}

// Why an element type cannot be used; "" if it can (NODE, BAR_2, TRI_3, QUAD_4).
std::string unsupported_reason(int type) {
    const ElementTypeInfo& info = element_type_info(type);
    if (type == kNode || type == kBar2 || type == kTri3 || type == kQuad4) return "";
    if (type == kNGon || type == kNFace)
        return "polygonal / polyhedral elements (NGON_n, NFACE_n) are not supported";
    if (info.dimension == 3) return "3D elements; HydraMa2D needs a 2D mesh";
    if (info.dimension == 1 || info.dimension == 2)
        return "higher-order elements; HydraMa2D needs linear elements (BAR_2, TRI_3, QUAD_4)";
    return "element type " + element_type_name(type) + " is not supported";
}

} // namespace

MeshData cgns_to_mesh_data(const CgnsFile& file, const std::string& source) {
    MeshData d;
    d.source = source;
    d.format = file.format.empty() ? std::string("CGNS") : file.format;
    Problems problems(source, 10);

    // ------------------------------------------------------------------
    // 1. Base and zone
    // ------------------------------------------------------------------
    if (file.bases.empty()) problems.add("the file contains no CGNSBase_t");
    if (file.bases.size() > 1) {
        std::vector<std::string> names;
        for (const Base& b : file.bases) names.push_back(b.name);
        problems.add("the file contains " + std::to_string(file.bases.size()) + " bases (" + list_names(names) +
                     "); HydraMa2D reads files with a single CGNSBase_t");
    }
    problems.checkpoint();
    const Base& base = file.bases.front();
    if (base.cell_dimension == 3) {
        problems.add("base '" + base.name + "' is 3D (CellDimension 3); HydraMa2D needs a 2D mesh (CellDimension 2)");
    } else if (base.cell_dimension != 2) {
        problems.add("base '" + base.name + "' has CellDimension " + std::to_string(base.cell_dimension) +
                     "; HydraMa2D needs a 2D mesh (CellDimension 2)");
    }
    if (base.physical_dimension != 2 && base.physical_dimension != 3)
        problems.add("base '" + base.name + "' has PhysicalDimension " + std::to_string(base.physical_dimension) +
                     " (expected 2 or 3)");
    if (base.zones.empty()) problems.add("base '" + base.name + "' contains no zone");
    if (base.zones.size() > 1) {
        std::vector<std::string> names;
        for (const Zone& z : base.zones) names.push_back(z.name);
        problems.add("base '" + base.name + "' contains " + std::to_string(base.zones.size()) + " zones (" +
                     list_names(names) + "); multi-zone meshes are not supported yet: merge them into one zone");
    }
    problems.checkpoint();
    const Zone& zone = base.zones.front();
    if (zone.type == kStructured) {
        problems.add("zone '" + zone.name + "' is structured; HydraMa2D reads unstructured zones");
    } else if (zone.type != kUnstructured) {
        problems.add("zone '" + zone.name + "' has an unknown zone type (" + std::to_string(zone.type) + ")");
    }
    problems.checkpoint();

    // ------------------------------------------------------------------
    // 2. Coordinates
    // ------------------------------------------------------------------
    const std::int64_t nv = zone.vertices;
    if (nv <= 0 || nv > std::numeric_limits<int>::max() / 2) {
        problems.add("zone '" + zone.name + "' has an invalid number of vertices (" + std::to_string(nv) + ")");
        problems.checkpoint();
    }
    const std::vector<double>*xs = nullptr, *ys = nullptr, *zs = nullptr;
    for (std::size_t c = 0; c < zone.coordinate_names.size(); ++c) {
        const std::string& n = zone.coordinate_names[c];
        const std::vector<double>* values = c < zone.coordinates.size() ? &zone.coordinates[c] : nullptr;
        if (n == "CoordinateX") xs = values;
        else if (n == "CoordinateY") ys = values;
        else if (n == "CoordinateZ") zs = values;
        else problems.add("zone '" + zone.name + "' has coordinates '" + n + "'; only Cartesian coordinates "
                          "(CoordinateX, CoordinateY, CoordinateZ) are supported");
        if (values && static_cast<std::int64_t>(values->size()) != nv)
            problems.add("coordinate array '" + n + "' has " + std::to_string(values->size()) + " values, zone '" +
                         zone.name + "' has " + std::to_string(nv) + " vertices");
    }
    if (!xs || !ys) problems.add("zone '" + zone.name + "' has no CoordinateX / CoordinateY arrays");
    if (zone.grids > 1)
        d.warnings.push_back("zone '" + zone.name + "' has " + std::to_string(zone.grids) +
                             " GridCoordinates_t nodes; the first one is used");
    problems.checkpoint();
    for (std::int64_t i = 0; i < nv; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        d.node_tags.push_back(i + 1);
        d.x.push_back((*xs)[k]);
        d.y.push_back((*ys)[k]);
        d.z.push_back(zs && zs->size() == xs->size() ? (*zs)[k] : 0.0);
    }

    // ------------------------------------------------------------------
    // 3. Element sections
    // ------------------------------------------------------------------
    std::vector<SectionRange> ranges;
    std::vector<Bar> bars;
    auto vertex_ok = [nv](std::int64_t v) { return v >= 1 && v <= nv; };

    for (std::size_t s = 0; s < zone.sections.size(); ++s) {
        const Section& sec = zone.sections[s];
        const std::int64_t n = sec.end - sec.start + 1;
        if (sec.start < 1 || n < 1) {
            problems.add(section_label(sec) + ": invalid element range");
            continue;
        }
        SectionRange range{sec.start, sec.end, s, Kind::None, 0, {}};

        // Adds element `number` of type `type` with nodes conn[first..]; returns false on bad vertices.
        int bad_vertices = 0;
        auto add_element = [&](int type, const std::int64_t* nodes, std::int64_t number) {
            ElementRef ref;
            if (type == kNode) {
                ref.kind = Kind::Node;
            } else {
                const int nn = element_type_info(type).nodes;
                RawElement e;
                e.tag = number;
                for (int k = 0; k < nn; ++k) {
                    if (!vertex_ok(nodes[k]) && bad_vertices++ == 0) {
                        problems.add(section_label(sec) + ": element " + std::to_string(number) +
                                     " references vertex " + std::to_string(nodes[k]) + ", zone '" + zone.name +
                                     "' has " + std::to_string(nv) + " vertices");
                    }
                    e.nodes[static_cast<std::size_t>(k)] = nodes[k];
                }
                if (type == kBar2) {
                    e.type = ElementType::Line2;
                    ref = {Kind::Bar, bars.size()};
                    bars.push_back({e, {}});
                } else {
                    e.type = type == kTri3 ? ElementType::Tri3 : ElementType::Quad4;
                    ref = {Kind::Cell, d.elements.size()};
                    d.elements.push_back(e);
                }
            }
            range.refs.push_back(ref);
        };

        if (sec.type == kMixed) {
            if (!sec.connectivity_read) {
                problems.add(section_label(sec) + ": connectivity not read");
                continue;
            }
            const std::vector<std::int64_t>& c = sec.connectivity;
            if (!sec.offsets.empty() && static_cast<std::int64_t>(sec.offsets.size()) != n + 1)
                problems.add(section_label(sec) + ": " + std::to_string(sec.offsets.size()) +
                             " element start offsets for " + std::to_string(n) + " elements");
            std::map<int, std::int64_t> unsupported; // type -> count
            std::size_t pos = 0;
            bool broken = false;
            for (std::int64_t k = 0; k < n && !broken; ++k) {
                if (!sec.offsets.empty() && static_cast<std::int64_t>(sec.offsets.size()) == n + 1 &&
                    sec.offsets[static_cast<std::size_t>(k)] != static_cast<std::int64_t>(pos)) {
                    problems.add(section_label(sec) + ": element start offset of element " +
                                 std::to_string(sec.start + k) + " does not match the connectivity");
                    broken = true;
                    break;
                }
                if (pos >= c.size()) {
                    problems.add(section_label(sec) + ": connectivity ends after " + std::to_string(k) + " of " +
                                 std::to_string(n) + " elements");
                    broken = true;
                    break;
                }
                const std::int64_t code = c[pos++];
                const int type = code >= 0 && code <= kLastElementType ? static_cast<int>(code) : -1;
                const int nn = element_type_info(type).nodes;
                if (nn <= 0) {
                    problems.add(section_label(sec) + ": element " + std::to_string(sec.start + k) +
                                 " has the invalid type code " + std::to_string(code) + " in a MIXED section");
                    broken = true;
                    break;
                }
                if (pos + static_cast<std::size_t>(nn) > c.size()) {
                    problems.add(section_label(sec) + ": connectivity ends inside element " +
                                 std::to_string(sec.start + k));
                    broken = true;
                    break;
                }
                if (unsupported_reason(type).empty()) {
                    add_element(type, &c[pos], sec.start + k);
                } else {
                    ++unsupported[type];
                    range.refs.push_back({});
                }
                pos += static_cast<std::size_t>(nn);
            }
            if (!broken && pos != c.size())
                problems.add(section_label(sec) + ": " + std::to_string(c.size() - pos) +
                             " connectivity values after the last element");
            for (const auto& [type, count] : unsupported) {
                problems.add(section_label(sec) + ": " + std::to_string(count) + " " + element_type_name(type) +
                             " element" + (count == 1 ? "" : "s") + ": " + unsupported_reason(type));
            }
            if (broken) continue;
        } else if (const std::string why = unsupported_reason(sec.type); !why.empty()) {
            problems.add(section_label(sec) + ": " + why);
            continue;
        } else if (sec.type == kNode) {
            range.kind = Kind::Node;
        } else {
            const int nn = element_type_info(sec.type).nodes;
            if (!sec.connectivity_read ||
                sec.connectivity.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(nn)) {
                problems.add(section_label(sec) + ": connectivity has " + std::to_string(sec.connectivity.size()) +
                             " values, expected " + std::to_string(n * nn));
                continue;
            }
            range.kind = sec.type == kBar2 ? Kind::Bar : Kind::Cell;
            range.first_index = sec.type == kBar2 ? bars.size() : d.elements.size();
            for (std::int64_t k = 0; k < n; ++k)
                add_element(sec.type, &sec.connectivity[static_cast<std::size_t>(k * nn)], sec.start + k);
            range.refs.clear(); // described by kind / first_index
        }
        if (bad_vertices > 1)
            problems.add(section_label(sec) + ": " + std::to_string(bad_vertices - 1) + " more invalid vertex references");
        ranges.push_back(std::move(range));
    }
    std::sort(ranges.begin(), ranges.end(), [](const SectionRange& a, const SectionRange& b) { return a.start < b.start; });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].start <= ranges[i - 1].end)
            problems.add(section_label(zone.sections[ranges[i - 1].section]) + " and " +
                         section_label(zone.sections[ranges[i].section]) + " overlap");
    }
    problems.checkpoint();

    auto find_element = [&ranges](std::int64_t id) -> ElementRef {
        auto it = std::upper_bound(ranges.begin(), ranges.end(), id,
                                   [](std::int64_t v, const SectionRange& r) { return v < r.start; });
        if (it == ranges.begin()) return {};
        --it;
        if (id > it->end) return {};
        return it->at(id);
    };

    // Boundary edges (one cell) for BCs given on vertices; built on demand,
    // with the BAR_2 elements on them.
    std::vector<std::pair<std::int64_t, std::int64_t>> boundary_edges;
    std::map<std::pair<std::int64_t, std::int64_t>, std::size_t> bar_of_edge;
    bool have_boundary_edges = false;
    auto compute_boundary_edges = [&]() {
        for (std::size_t i = 0; i < bars.size(); ++i) {
            const std::int64_t a = bars[i].element.nodes[0], c = bars[i].element.nodes[1];
            bar_of_edge.emplace(std::make_pair(std::min(a, c), std::max(a, c)), i);
        }
        std::map<std::pair<std::int64_t, std::int64_t>, int> count;
        for (const RawElement& e : d.elements) {
            const int nn = node_count(e.type);
            for (int k = 0; k < nn; ++k) {
                std::int64_t a = e.nodes[static_cast<std::size_t>(k)], b = e.nodes[static_cast<std::size_t>((k + 1) % nn)];
                ++count[{std::min(a, b), std::max(a, b)}];
            }
        }
        for (const auto& [edge, c] : count)
            if (c == 1) boundary_edges.push_back(edge);
        have_boundary_edges = true;
    };

    // ------------------------------------------------------------------
    // 4. Boundary conditions
    // ------------------------------------------------------------------
    std::vector<RawElement> vertex_lines;
    std::int64_t total_elements = 0;
    for (const SectionRange& r : ranges) total_elements += r.end - r.start + 1;

    for (std::size_t b = 0; b < zone.bcs.size(); ++b) {
        const BoundaryCondition& bc = zone.bcs[b];
        const int tag = static_cast<int>(b + 1);
        const std::string label = "BC '" + bc.name + "' (BC " + std::to_string(tag) + ")";

        // Declared type: the BC's own BCType, or its family's for FamilySpecified
        // (and for BCs whose own type carries no information).
        const Family* family = nullptr;
        if (!bc.family.empty()) {
            for (const Family& f : base.families)
                if (f.name == bc.family) family = &f;
            if (!family)
                d.warnings.push_back(label + " refers to family '" + bc.family + "', which is not defined in base '" +
                                     base.name + "'");
        }
        int bc_type = bc.type;
        std::string hint_source = "CGNS BCType " + bc_type_name(bc.type);
        if ((bc_type == kFamilySpecified || !boundary_type_of_cgns_bc(bc_type)) && family && family->has_bc &&
            family->bc_type != kFamilySpecified) {
            bc_type = family->bc_type;
            hint_source = "CGNS FamilyBC " + bc_type_name(bc_type) + " of family '" + family->name + "'";
        }
        if (bc_type == kBCDegenerateLine || bc_type == kBCDegeneratePoint) {
            problems.add(label + ": " + bc_type_name(bc_type) + " is not supported");
            continue;
        }
        const std::optional<BoundaryType> hint = boundary_type_of_cgns_bc(bc_type);

        // Point set.
        bool on_elements = false;
        switch (bc.point_set) {
        case kElementRange:
        case kElementList: on_elements = true; break;
        case kPointRange:
        case kPointList:
            switch (bc.location) {
            case kGridLocationNull:
            case kVertex: on_elements = false; break;
            case kEdgeCenter:
            case kFaceCenter:
            case kCellCenter: on_elements = true; break;
            default:
                problems.add(label + ": grid location " + grid_location_name(bc.location) + " is not supported");
                continue;
            }
            break;
        default:
            problems.add(label + ": point set type " + point_set_type_name(bc.point_set) + " is not supported");
            continue;
        }
        const std::int64_t limit = on_elements ? total_elements : nv;
        const bool is_range = bc.point_set == kPointRange || bc.point_set == kElementRange;
        if (is_range) {
            if (bc.points.size() != 2 || bc.points[0] > bc.points[1]) {
                problems.add(label + ": invalid " + point_set_type_name(bc.point_set));
                continue;
            }
            if (bc.points[0] < 1 || bc.points[1] > (on_elements ? ranges.empty() ? 0 : ranges.back().end : nv)) {
                problems.add(label + ": " + point_set_type_name(bc.point_set) + " " + std::to_string(bc.points[0]) +
                             "-" + std::to_string(bc.points[1]) + " exceeds the " +
                             (on_elements ? "elements" : "vertices") + " of zone '" + zone.name + "'");
                continue;
            }
            if (bc.points[1] - bc.points[0] + 1 > limit) {
                problems.add(label + ": " + point_set_type_name(bc.point_set) + " larger than the zone");
                continue;
            }
        }
        // Visits the indices of the point set (ranges are not expanded).
        auto for_each_id = [&bc, is_range](auto&& visit) {
            if (is_range) {
                for (std::int64_t i = bc.points[0]; i <= bc.points[1]; ++i) visit(i);
            } else {
                for (std::int64_t i : bc.points) visit(i);
            }
        };

        if (on_elements) {
            std::int64_t nbar = 0, ncell = 0, nnode = 0, missing = 0, first_missing = 0;
            for_each_id([&](std::int64_t id) {
                const ElementRef ref = find_element(id);
                if (ref.kind == Kind::None) {
                    if (missing++ == 0) first_missing = id;
                } else if (ref.kind == Kind::Bar) {
                    ++nbar;
                } else if (ref.kind == Kind::Cell) {
                    ++ncell;
                } else {
                    ++nnode;
                }
            });
            if (missing > 0) {
                problems.add(label + ": refers to " + std::to_string(missing) + " element(s) that do not exist (first: " +
                             std::to_string(first_missing) + ")");
                continue;
            }
            if (nbar > 0 && ncell + nnode > 0) {
                problems.add(label + ": mixes " + std::to_string(nbar) + " boundary edges (BAR_2) with " +
                             std::to_string(ncell + nnode) + " other elements");
                continue;
            }
            if (nbar == 0) {
                d.warnings.push_back(label + (ncell > 0 ? " is defined on 2D cells (a region, not a boundary)"
                                                        : nnode > 0 ? " is defined on NODE elements" : " is empty") +
                                     ": ignored");
                continue;
            }
            for_each_id([&](std::int64_t id) {
                std::vector<int>& owners = bars[find_element(id).index].bcs;
                if (std::find(owners.begin(), owners.end(), tag) == owners.end()) owners.push_back(tag);
            });
        } else {
            std::vector<char> in_bc(static_cast<std::size_t>(nv) + 1, 0);
            std::int64_t bad = 0;
            for_each_id([&](std::int64_t v) {
                if (vertex_ok(v)) in_bc[static_cast<std::size_t>(v)] = 1;
                else ++bad;
            });
            if (bad > 0) {
                problems.add(label + ": " + std::to_string(bad) + " vertex index(es) outside 1-" + std::to_string(nv));
                continue;
            }
            if (!have_boundary_edges) compute_boundary_edges();
            std::int64_t edges = 0;
            for (const auto& [a, c] : boundary_edges) {
                if (!in_bc[static_cast<std::size_t>(a)] || !in_bc[static_cast<std::size_t>(c)]) continue;
                ++edges;
                // An existing BAR_2 on that edge joins the BC; otherwise a line is added.
                if (const auto bar = bar_of_edge.find({a, c}); bar != bar_of_edge.end()) {
                    std::vector<int>& owners = bars[bar->second].bcs;
                    if (std::find(owners.begin(), owners.end(), tag) == owners.end()) owners.push_back(tag);
                    continue;
                }
                RawElement e;
                e.type = ElementType::Line2;
                e.nodes = {a, c, 0, 0};
                e.physical = tag;
                vertex_lines.push_back(e);
            }
            d.warnings.push_back(label + " is given on vertices: " + std::to_string(edges) +
                                 " boundary edge(s) with both end points in the BC assigned to it");
        }

        // Name: the BC name, or the family name if only that one tells the type.
        std::string name = bc.name;
        const NameResolution by_bc = resolve_boundary_name(bc.name, hint);
        if ((!by_bc.resolved() || by_bc.from_hint) && !bc.family.empty()) {
            const NameResolution by_family = resolve_boundary_name(bc.family, hint);
            if (by_family.resolved() && !by_family.from_hint) name = bc.family;
        }
        d.physical_names[{1, tag}] = name;
        if (hint) d.boundary_hints[{1, tag}] = {*hint, hint_source};
    }
    problems.checkpoint();

    // Boundary lines: once per BC that contains them, else once without a group.
    for (const Bar& bar : bars) {
        RawElement e = bar.element;
        if (bar.bcs.empty()) {
            d.elements.push_back(e);
            continue;
        }
        for (int tag : bar.bcs) {
            e.physical = tag;
            d.elements.push_back(e);
        }
    }
    d.elements.insert(d.elements.end(), vertex_lines.begin(), vertex_lines.end());
    return d;
}

MeshData read_cgns(const std::string& path) { return cgns_to_mesh_data(read_cgns_file(path), path); }

} // namespace hydrama::mesh
