// CGNS structure -> MeshData -> Mesh (mesh/CgnsReader.hpp: cgns_to_mesh_data,
// boundary_type_of_cgns_bc; mesh/CgnsModel.hpp). No CGNS library needed: these
// tests run in every build. Files are tested in test_cgns_reader.cpp.
//
// Most cases use the channel of tests/CgnsTestModels.hpp, the same mesh as
// tests/data/su2/channel_regular.su2, so every CGNS layout is compared with
// the SU2 import exactly.

#include "CgnsTestModels.hpp"
#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "mesh/CgnsReader.hpp"
#include "mesh/MeshBuilder.hpp"
#include "mesh/MeshCheck.hpp"
#include "mesh/Su2Reader.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <set>
#include <tuple>

using namespace hydrama;
using namespace hydrama::mesh;
using namespace hydrama::test::cgnsmodels;
using Catch::Matchers::ContainsSubstring;

namespace {

const Mesh& su2_channel() {
    static const Mesh m = build_mesh(read_su2(std::string(HYDRAMA2D_TEST_DATA_DIR) + "/su2/channel_regular.su2"));
    return m;
}

Mesh build(const CgnsFile& f, ImportReport* r = nullptr, const BuildOptions& o = {}) {
    return build_mesh(cgns_to_mesh_data(f, "test.cgns"), o, r);
}

std::string error_of(const CgnsFile& f, const BuildOptions& o = {}) {
    try {
        build_mesh(cgns_to_mesh_data(f, "test.cgns"), o);
    } catch (const MeshImportError& e) {
        return e.what();
    }
    FAIL("expected MeshImportError");
    return {};
}

void require_valid(const Mesh& m) {
    const auto problems = check_mesh(m);
    for (const auto& p : problems) UNSCOPED_INFO(p);
    REQUIRE(problems.empty());
}

std::multiset<std::array<int, 4>> sides_of(const Mesh& m) {
    std::multiset<std::array<int, 4>> out;
    for (const auto& s : m.iside) out.insert({s.node1, s.node2, s.elem, s.flag});
    return out;
}

// Same mesh as the SU2 channel; `same_side_order`: also the same order of the sides.
void check_channel(const Mesh& m, bool same_side_order = true) {
    require_valid(m);
    const Mesh& ref = su2_channel();
    CHECK(m.intma == ref.intma);
    CHECK(m.x == ref.x);
    CHECK(m.y == ref.y);
    REQUIRE(m.nboun == ref.nboun);
    CHECK(sides_of(m) == sides_of(ref));
    if (same_side_order) {
        for (int i = 0; i < m.nboun; ++i) {
            CHECK(m.iside[static_cast<std::size_t>(i)].node1 == ref.iside[static_cast<std::size_t>(i)].node1);
            CHECK(m.iside[static_cast<std::size_t>(i)].side_id == ref.iside[static_cast<std::size_t>(i)].side_id);
        }
    }
}

Zone& zone_of(CgnsFile& f) { return f.bases.at(0).zones.at(0); }

} // namespace

// =============================================================================
// Tables
// =============================================================================

TEST_CASE("CGNS: element type table", "[cgns][translate]") {
    CHECK(element_type_info(kTri3).name == "TRI_3");
    CHECK(element_type_info(kTri3).nodes == 3);
    CHECK(element_type_info(kBar2).dimension == 1);
    CHECK(element_type_info(kQuad4).linear);
    CHECK_FALSE(element_type_info(kTri6).linear);
    CHECK(element_type_info(kHexa8).nodes == 8);
    CHECK(element_type_info(kMixed).nodes == 0);
    CHECK(element_type_info(kLastElementType).name == "HEXA_125");
    CHECK(element_type_info(kLastElementType).nodes == 125);
    CHECK(element_type_name(99) == "ElementType 99");
    CHECK(element_type_name(-1) == "ElementType -1");
    int volume = 0, higher_order = 0;
    for (int t = 0; t <= kLastElementType; ++t) {
        const ElementTypeInfo& i = element_type_info(t);
        CAPTURE(t, i.name);
        CHECK_FALSE(i.name.empty());
        if (i.dimension == 3) ++volume;
        if ((i.dimension == 1 || i.dimension == 2) && !i.linear) ++higher_order;
    }
    CHECK(volume == 34);       // TETRA_*, PYRA_*, PENTA_*, HEXA_*
    CHECK(higher_order == 14); // BAR_3..5, TRI_6..15, QUAD_8..25
    CHECK(bc_type_name(kBCWallViscous) == "BCWallViscous");
    CHECK(bc_type_name(26) == "BCType 26");
    CHECK(point_set_type_name(kElementRange) == "ElementRange");
    CHECK(grid_location_name(kEdgeCenter) == "EdgeCenter");
}

TEST_CASE("CGNS: BCType to boundary type", "[cgns][translate]") {
    using T = BoundaryType;
    const std::map<int, std::optional<T>> expected = {
        {kBCTypeNull, std::nullopt},       {kBCTypeUserDefined, std::nullopt},  {kBCAxisymmetricWedge, T::Wedge},
        {kBCDegenerateLine, std::nullopt}, {kBCDegeneratePoint, std::nullopt},  {kBCDirichlet, std::nullopt},
        {kBCExtrapolate, std::nullopt},    {kBCFarfield, T::Farfield},          {kBCGeneral, std::nullopt},
        {kBCInflow, T::Inlet},             {kBCInflowSubsonic, T::Inlet},       {kBCInflowSupersonic, T::SupersonicInlet},
        {kBCNeumann, std::nullopt},        {kBCOutflow, T::Outlet},             {kBCOutflowSubsonic, T::Outlet},
        {kBCOutflowSupersonic, T::SupersonicOutlet}, {kBCSymmetryPlane, T::Symmetry}, {kBCSymmetryPolar, T::Axis},
        {kBCTunnelInflow, T::Inlet},       {kBCTunnelOutflow, T::Outlet},       {kBCWall, T::Wall},
        {kBCWallInviscid, T::Symmetry},    {kBCWallViscous, T::Wall},           {kBCWallViscousHeatFlux, T::Wall},
        {kBCWallViscousIsothermal, T::Wall}, {kFamilySpecified, std::nullopt},
    };
    REQUIRE(expected.size() == 26);
    for (const auto& [code, type] : expected) {
        CAPTURE(bc_type_name(code));
        CHECK(boundary_type_of_cgns_bc(code) == type);
    }
    CHECK_FALSE(boundary_type_of_cgns_bc(99).has_value());
}

// =============================================================================
// Valid meshes and boundary conditions
// =============================================================================

TEST_CASE("CGNS: sections with ElementRange BCs give the SU2 channel", "[cgns][translate]") {
    ImportReport r;
    const Mesh m = build(channel_model(), &r);
    check_channel(m);
    CHECK(r.format == "CGNS 3.40 HDF5");
    REQUIRE(r.patches.size() == 4);
    const std::vector<std::tuple<std::string, int, std::string>> expected = {
        {"inlet", 503, "inlet + CGNS BCType BCInflowSubsonic"},
        {"outlet", 504, "outlet + CGNS BCType BCOutflow"},
        {"upper_wall", 500, "wall + CGNS BCType BCWallViscous"},
        {"lower_wall", 500, "wall + CGNS BCType BCWall"}};
    for (std::size_t i = 0; i < 4; ++i) {
        CAPTURE(i);
        CHECK(r.patches[i].tag == static_cast<int>(i + 1));
        CHECK(std::make_tuple(r.patches[i].name, r.patches[i].flag, r.patches[i].matched) == expected[i]);
        CHECK(r.patches[i].source == ConditionSource::Name);
    }
    CHECK(r.warnings.empty());
    CHECK(r.duplicate_elements_merged == 0);
    CHECK_THAT(format_report(r), ContainsSubstring("Boundary 'inlet' (tag 1): inlet, flag 503, 4 sides [name + declared type]"));
}

TEST_CASE("CGNS: every point set type and location for boundary edges", "[cgns][translate]") {
    struct Variant {
        int point_set, location;
    };
    for (const Variant v : {Variant{kElementRange, kVertex}, Variant{kElementList, kVertex},
                            Variant{kPointRange, kEdgeCenter}, Variant{kPointList, kEdgeCenter},
                            Variant{kPointRange, kFaceCenter}, Variant{kPointList, kCellCenter}}) {
        CAPTURE(point_set_type_name(v.point_set), grid_location_name(v.location));
        ChannelSpec s;
        s.point_set = v.point_set;
        s.location = v.location;
        ImportReport r;
        check_channel(build(channel_model(s), &r));
        CHECK(r.warnings.empty());
    }
}

TEST_CASE("CGNS: BCs on vertices give the boundary edges between their vertices", "[cgns][translate]") {
    for (const bool mixed : {false, true}) {
        CAPTURE(mixed);
        ChannelSpec s;
        s.mixed = mixed;
        s.point_set = kPointList;
        s.location = kVertex;
        ImportReport r;
        check_channel(build(channel_model(s), &r), false);
        REQUIRE(r.warnings.size() == 4);
        CHECK_THAT(r.warnings[2], ContainsSubstring("BC 'upper_wall' (BC 3) is given on vertices: 8 boundary edge(s)"));
        CHECK(r.duplicate_elements_merged == 0); // edges with a BAR_2 join it
    }
    // A vertex PointRange (here: the bottom row 1..9 = lower wall).
    ChannelSpec s;
    s.point_set = kPointList;
    s.location = kVertex;
    CgnsFile f = channel_model(s);
    BoundaryCondition& lower = zone_of(f).bcs[3];
    lower.point_set = kPointRange;
    lower.points = {1, 9};
    check_channel(build(f), false);
}

TEST_CASE("CGNS: MIXED sections with and without element start offsets", "[cgns][translate]") {
    ChannelSpec s;
    s.mixed = true;
    CgnsFile with_offsets = channel_model(s);
    check_channel(build(with_offsets));
    CgnsFile without = with_offsets; // CGNS 3.x style: type codes only
    zone_of(without).sections[0].offsets.clear();
    check_channel(build(without));
}

TEST_CASE("CGNS: clockwise triangles are repaired", "[cgns][translate]") {
    for (const bool mixed : {false, true}) {
        ChannelSpec s;
        s.clockwise = true;
        s.mixed = mixed;
        ImportReport r;
        check_channel(build(channel_model(s), &r));
        CHECK(r.reoriented_triangles == 64);
    }
}

TEST_CASE("CGNS: PhysicalDimension 3 with planar coordinates", "[cgns][translate]") {
    ChannelSpec s;
    s.with_z = true;
    CgnsFile f = channel_model(s);
    check_channel(build(f));
    zone_of(f).coordinates[2][7] = 0.5;
    CHECK_THAT(error_of(f), ContainsSubstring("not planar in the x-y plane"));
}

TEST_CASE("CGNS: families give names and boundary types", "[cgns][translate]") {
    ChannelSpec s;
    s.names = {"BC_1", "BC_2", "BC_3", "BC_4"};
    s.types = {kFamilySpecified, kFamilySpecified, kFamilySpecified, kBCTypeUserDefined};
    s.families = {"Inflow", "Outflow", "Walls", "Walls"};
    s.family_nodes = {{"Inflow", true, kBCInflow}, {"Outflow", true, kBCOutflow}, {"Walls", true, kBCWall}};
    ImportReport r;
    check_channel(build(channel_model(s), &r));
    REQUIRE(r.patches.size() == 4);
    CHECK(r.patches[0].name == "Inflow"); // the BC name has no keyword: the family name is used
    CHECK(r.patches[0].matched == "inflow + CGNS FamilyBC BCInflow of family 'Inflow'");
    CHECK(r.patches[3].name == "Walls"); // BCTypeUserDefined: the family's type applies
    CHECK(r.patches[3].flag == 500);

    SECTION("family without FamilyBC: names alone") {
        ChannelSpec t = s;
        t.family_nodes = {{"Inflow", false, kBCTypeNull}, {"Outflow", false, kBCTypeNull}, {"Walls", false, kBCTypeNull}};
        ImportReport q;
        check_channel(build(channel_model(t), &q));
        CHECK(q.patches[0].matched == "inflow");
    }
    SECTION("BC name with a keyword is preferred to the family name") {
        ChannelSpec t = s;
        t.names = {"inlet", "outlet", "top_wall", "bottom_wall"};
        ImportReport q;
        build(channel_model(t), &q);
        CHECK(q.patches[0].name == "inlet");
        CHECK(q.patches[2].name == "top_wall");
    }
    SECTION("undefined family") {
        ChannelSpec t = s;
        t.names = {"inlet", "outlet", "top_wall", "bottom_wall"};
        t.family_nodes.clear();
        ImportReport q;
        build(channel_model(t), &q);
        REQUIRE(q.warnings.size() == 4);
        CHECK_THAT(q.warnings[0], ContainsSubstring("BC 'inlet' (BC 1) refers to family 'Inflow', which is not defined in base 'Base'"));
    }
}

TEST_CASE("CGNS: BCType and name are combined by the naming rules", "[cgns][translate]") {
    auto flags_of = [](const std::array<std::string, 4>& names, const std::array<int, 4>& types) {
        ChannelSpec s;
        s.names = names;
        s.types = types;
        ImportReport r;
        build(channel_model(s), &r);
        std::vector<std::pair<int, ConditionSource>> out;
        for (const auto& p : r.patches) out.push_back({p.flag, p.source});
        return out;
    };
    using CS = ConditionSource;
    // Names without keywords: the declared type decides.
    CHECK(flags_of({"left", "right", "top", "bottom"}, {kBCInflow, kBCOutflowSubsonic, kBCWallViscousHeatFlux, kBCSymmetryPlane}) ==
          std::vector<std::pair<int, CS>>{{503, CS::FileType}, {504, CS::FileType}, {500, CS::FileType}, {506, CS::FileType}});
    // A specific name refines the declared family; an ambiguous name is settled by it.
    CHECK(flags_of({"inlet_profile", "exit", "hot_wall", "inlet_wall"}, {kBCInflow, kBCOutflow, kBCWallViscousIsothermal, kBCWall}) ==
          std::vector<std::pair<int, CS>>{{507, CS::Name}, {504, CS::Name}, {501, CS::Name}, {500, CS::Name}});
    // BCWallInviscid is a slip wall: symmetry.
    CHECK(flags_of({"inlet", "outlet", "upper", "lower"}, {kBCTypeNull, kBCExtrapolate, kBCWallInviscid, kBCWall}) ==
          std::vector<std::pair<int, CS>>{{503, CS::Name}, {504, CS::Name}, {506, CS::FileType}, {500, CS::FileType}});
    // Moving and rotating walls: specific wall names refine BCWall to 509 and 510.
    CHECK(flags_of({"rotor_wall", "moving_wall", "exit", "inlet"}, {kBCWall, kBCWall, kBCOutflow, kBCInflow}) ==
          std::vector<std::pair<int, CS>>{{510, CS::Name}, {509, CS::Name}, {504, CS::Name}, {503, CS::Name}});
}

TEST_CASE("CGNS: declared types that HydraMa2D rejects", "[cgns][translate][errors]") {
    auto with = [](int index, const std::string& name, int type) {
        ChannelSpec s;
        s.names[static_cast<std::size_t>(index)] = name;
        s.types[static_cast<std::size_t>(index)] = type;
        return error_of(channel_model(s));
    };
    CHECK_THAT(with(0, "inlet", kBCWall),
               ContainsSubstring("'inlet' (tag 1): the name means inlet, but CGNS BCType BCWall means wall; make them agree"));
    CHECK_THAT(with(1, "far", kBCFarfield), ContainsSubstring("'far' (tag 2) is a 'farfield' boundary"));
    CHECK_THAT(with(0, "inflow_1", kBCInflowSupersonic), ContainsSubstring("is a 'supersonic_inlet' boundary"));
    CHECK_THAT(with(0, "pressure_inlet", kBCInflow), ContainsSubstring("is a 'pressure_inlet' boundary"));
    CHECK_THAT(with(2, "periodic_1", kBCTypeNull), ContainsSubstring("is a 'periodic' boundary"));
    CHECK_THAT(with(3, "centreline", kBCSymmetryPolar), ContainsSubstring("is a 'axis' boundary"));
    CHECK_THAT(with(3, "front", kBCAxisymmetricWedge), ContainsSubstring("is a 'wedge' boundary"));
    CHECK_THAT(with(2, "upper", kBCDegenerateLine), ContainsSubstring("BC 'upper' (BC 3): BCDegenerateLine is not supported"));
    CHECK_THAT(with(2, "airfoil", kBCGeneral),
               ContainsSubstring("'airfoil' (tag 3): boundary name 'airfoil' (normalised 'airfoil') contains no boundary keyword"));
}

TEST_CASE("CGNS: *.bofl entries are keyed by BC number", "[cgns][translate][bofl]") {
    ChannelSpec s;
    s.names = {"L_1", "L_2", "L_3", "L_4"};
    s.types = {kFamilySpecified, kFamilySpecified, kFamilySpecified, kFamilySpecified}; // like Gmsh
    CHECK_THAT(error_of(channel_model(s)), ContainsSubstring("invalid mesh (4 problems)"));
    BuildOptions o;
    o.flag_overrides = {{1, 503}, {2, 504}, {3, 500}, {4, 500}};
    ImportReport r;
    const Mesh m = build(channel_model(s), &r, o);
    check_channel(m);
    CHECK(r.patches[3].flag == 500);
    CHECK(r.patches[3].source == ConditionSource::FlagFile);
}

TEST_CASE("CGNS: BCs that are not boundaries are ignored", "[cgns][translate]") {
    CgnsFile f = channel_model();
    Zone& z = zone_of(f);
    z.sections.push_back({"Points", kNode, 89, 90, {1, 45}, {}, true});
    BoundaryCondition surface{"S_1", kFamilySpecified, kPointRange, kCellCenter, {1, 64}, ""}; // Gmsh: the surface
    BoundaryCondition points{"corners", kBCTypeNull, kElementList, kVertex, {89, 90}, ""};
    BoundaryCondition empty{"nothing", kBCWall, kPointList, kEdgeCenter, {}, ""};
    z.bcs.insert(z.bcs.end(), {surface, points, empty});
    ImportReport r;
    check_channel(build(f, &r));
    REQUIRE(r.warnings.size() == 3);
    CHECK_THAT(r.warnings[0], ContainsSubstring("BC 'S_1' (BC 5) is defined on 2D cells (a region, not a boundary): ignored"));
    CHECK_THAT(r.warnings[1], ContainsSubstring("BC 'corners' (BC 6) is defined on NODE elements: ignored"));
    CHECK_THAT(r.warnings[2], ContainsSubstring("BC 'nothing' (BC 7) is empty: ignored"));
}

TEST_CASE("CGNS: the benchmark mesh as CGNS is the legacy mesh", "[cgns][translate][hinterkante]") {
    Mesh legacy;
    {
        test::CaptureStdout quiet;
        REQUIRE(IO::read_mesh(std::string(HYDRAMA2D_CASE_DIR) + "/constant/hinterkante.mesh", legacy));
        REQUIRE(IO::read_boundary_flags(std::string(HYDRAMA2D_CASE_DIR) + "/constant/hinterkante.bofl", legacy));
    }
    const std::map<int, std::string> names = {{1, "wall_1"}, {2, "outlet"}, {3, "wall_3"}, {4, "inflow"},
                                              {5, "top"},    {6, "wall_6"}, {7, "wall_7"}, {8, "inlet_8"}};
    const std::map<int, int> types = {{5, kBCWallViscous}, {2, kBCOutflow}};
    const Mesh m = build(cgns_model_from_mesh(legacy, names, types));
    CHECK(m.intma == legacy.intma);
    CHECK(m.x == legacy.x);
    CHECK(m.y == legacy.y);
    std::vector<BoundarySide> expected = legacy.iside; // sides grouped by BC (side id), stable
    std::stable_sort(expected.begin(), expected.end(), [](const BoundarySide& a, const BoundarySide& b) { return a.side_id < b.side_id; });
    REQUIRE(m.nboun == legacy.nboun);
    int differing = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const BoundarySide &a = m.iside[i], &b = expected[i];
        if (a.node1 != b.node1 || a.node2 != b.node2 || a.elem != b.elem || a.side_id != b.side_id || a.flag != b.flag)
            ++differing;
    }
    CHECK(differing == 0);
}

// =============================================================================
// Unsupported and invalid structures
// =============================================================================

TEST_CASE("CGNS: 3D, higher-order, polyhedral and unknown element types are reported", "[cgns][translate][errors]") {
    for (int type = 0; type <= kLastElementType + 1; ++type) {
        const ElementTypeInfo& info = element_type_info(type);
        if (type == kNode || type == kBar2 || type == kTri3 || type == kQuad4 || type == kMixed) continue;
        CAPTURE(type, info.name);
        CgnsFile f = channel_model();
        const std::int64_t n = std::max(info.nodes, 1);
        Section extra{"Extra", type, 89, 89, std::vector<std::int64_t>(static_cast<std::size_t>(n), 1), {}, false};
        zone_of(f).sections.push_back(extra);
        const std::string err = error_of(f);
        CHECK_THAT(err, ContainsSubstring("section 'Extra' (" + element_type_name(type) + ", elements 89-89)"));
        if (info.dimension == 3) CHECK_THAT(err, ContainsSubstring("3D elements; HydraMa2D needs a 2D mesh"));
        else if (type == kNGon || type == kNFace) CHECK_THAT(err, ContainsSubstring("(NGON_n, NFACE_n) are not supported"));
        else if (info.dimension >= 1) CHECK_THAT(err, ContainsSubstring("higher-order elements"));
        else CHECK_THAT(err, ContainsSubstring("is not supported"));
    }
}

TEST_CASE("CGNS: unsupported elements inside MIXED sections are counted per type", "[cgns][translate][errors]") {
    ChannelSpec s;
    s.mixed = true;
    CgnsFile f = channel_model(s);
    Section& all = zone_of(f).sections[0];
    all.offsets.clear();
    all.connectivity.insert(all.connectivity.end(), {kTetra4, 1, 2, 10, 11, kTetra4, 2, 3, 11, 12, kTri6, 1, 3, 19, 2, 11, 10});
    all.end += 3;
    const std::string err = error_of(f);
    CHECK_THAT(err, ContainsSubstring("2 problems"));
    CHECK_THAT(err, ContainsSubstring("section 'All' (MIXED, elements 1-91): 2 TETRA_4 elements: 3D elements"));
    CHECK_THAT(err, ContainsSubstring("section 'All' (MIXED, elements 1-91): 1 TRI_6 element: higher-order elements"));
}

TEST_CASE("CGNS: quadrilaterals and mixed meshes (QUAD_4 sections)", "[cgns][translate][quad]") {
    // The inflation-layer mesh of tests/data/su2/mixed_tri_quad.su2, exactly.
    const Mesh su2 = build_mesh(read_su2(std::string(HYDRAMA2D_TEST_DATA_DIR) + "/su2/mixed_tri_quad.su2"));
    ImportReport r;
    const Mesh m = build(channel_model_with_quads(1), &r);
    require_valid(m);
    CHECK(r.quadrilaterals == 8);
    CHECK(m.intma == su2.intma);
    CHECK(m.quads == su2.quads);
    CHECK(m.x == su2.x);
    CHECK(m.y == su2.y);
    CHECK(sides_of(m) == sides_of(su2));
    // All quadrilaterals, and quadrilaterals in MIXED sections.
    const Mesh all = build(channel_model_with_quads(kNy));
    require_valid(all);
    CHECK(all.num_quads() == 32);
    CHECK(all.num_triangles() == 0);
    CgnsFile mixed = channel_model_with_quads(2);
    Zone& z = zone_of(mixed);
    Section merged{"All", kMixed, 1, 0, {}, {}, true};
    for (const Section& s : {z.sections[0], z.sections[1]}) {
        const int nn = element_type_info(s.type).nodes;
        for (std::size_t k = 0; k < s.connectivity.size(); k += static_cast<std::size_t>(nn)) {
            merged.connectivity.push_back(s.type);
            merged.connectivity.insert(merged.connectivity.end(), s.connectivity.begin() + static_cast<std::ptrdiff_t>(k),
                                       s.connectivity.begin() + static_cast<std::ptrdiff_t>(k) + nn);
            ++merged.end;
        }
    }
    z.sections.erase(z.sections.begin(), z.sections.begin() + 2);
    z.sections.insert(z.sections.begin(), merged);
    const Mesh from_mixed = build(mixed);
    const Mesh from_sections = build(channel_model_with_quads(2));
    CHECK(from_mixed.quads == from_sections.quads);
    CHECK(from_mixed.intma == from_sections.intma);
    CHECK(from_mixed.num_quads() == 16);
    // A quadrilateral on top of the triangles is an overlap (build_mesh).
    CgnsFile f = channel_model();
    zone_of(f).sections.push_back({"Quad", kQuad4, 89, 89, {1, 2, 11, 10}, {}, true});
    CHECK_THAT(error_of(f), ContainsSubstring("quadrilateral 89 overlap"));
}

TEST_CASE("CGNS: invalid MIXED sections", "[cgns][translate][errors]") {
    ChannelSpec s;
    s.mixed = true;
    auto broken = [&](auto&& edit) {
        CgnsFile f = channel_model(s);
        edit(zone_of(f).sections[0]);
        return error_of(f);
    };
    CHECK_THAT(broken([](Section& a) { a.connectivity[0] = kMixed; }),
               ContainsSubstring("element 1 has the invalid type code 20 in a MIXED section"));
    CHECK_THAT(broken([](Section& a) { a.connectivity[0] = 77; }), ContainsSubstring("invalid type code 77"));
    CHECK_THAT(broken([](Section& a) { a.connectivity.resize(a.connectivity.size() - 2); a.offsets.clear(); }),
               ContainsSubstring("connectivity ends inside element 88"));
    CHECK_THAT(broken([](Section& a) { a.end += 1; a.offsets.clear(); }),
               ContainsSubstring("connectivity ends after 88 of 89 elements"));
    CHECK_THAT(broken([](Section& a) { a.connectivity.push_back(1); a.offsets.clear(); }),
               ContainsSubstring("1 connectivity values after the last element"));
    CHECK_THAT(broken([](Section& a) { a.offsets[5] += 1; }),
               ContainsSubstring("element start offset of element 6 does not match the connectivity"));
    CHECK_THAT(broken([](Section& a) { a.offsets.pop_back(); }), ContainsSubstring("88 element start offsets for 88 elements"));
    CHECK_THAT(broken([](Section& a) { a.connectivity_read = false; }), ContainsSubstring("connectivity not read"));
}

TEST_CASE("CGNS: invalid bases, zones and coordinates", "[cgns][translate][errors]") {
    CgnsFile none;
    CHECK_THAT(error_of(none), ContainsSubstring("the file contains no CGNSBase_t"));

    CgnsFile two = channel_model();
    two.bases.push_back(two.bases[0]);
    two.bases[1].name = "Other";
    CHECK_THAT(error_of(two), ContainsSubstring("the file contains 2 bases ('Base', 'Other'); HydraMa2D reads files with a single CGNSBase_t"));

    CgnsFile f = channel_model();
    f.bases[0].cell_dimension = 3;
    CHECK_THAT(error_of(f), ContainsSubstring("base 'Base' is 3D (CellDimension 3)"));
    f.bases[0].cell_dimension = 1;
    CHECK_THAT(error_of(f), ContainsSubstring("has CellDimension 1"));
    f = channel_model();
    f.bases[0].physical_dimension = 4;
    CHECK_THAT(error_of(f), ContainsSubstring("has PhysicalDimension 4 (expected 2 or 3)"));

    f = channel_model();
    f.bases[0].zones.clear();
    CHECK_THAT(error_of(f), ContainsSubstring("base 'Base' contains no zone"));
    f = channel_model();
    for (int k = 0; k < 6; ++k) f.bases[0].zones.push_back(f.bases[0].zones[0]);
    CHECK_THAT(error_of(f), ContainsSubstring("contains 7 zones ('Channel', 'Channel', 'Channel', 'Channel', 'Channel', ...)") &&
                                ContainsSubstring("multi-zone meshes are not supported yet"));
    f = channel_model();
    zone_of(f).type = kStructured;
    CHECK_THAT(error_of(f), ContainsSubstring("zone 'Channel' is structured"));
    zone_of(f).type = kZoneTypeUserDefined;
    CHECK_THAT(error_of(f), ContainsSubstring("unknown zone type"));

    f = channel_model();
    zone_of(f).coordinate_names[1] = "CoordinateTheta";
    const std::string polar = error_of(f);
    CHECK_THAT(polar, ContainsSubstring("coordinates 'CoordinateTheta'; only Cartesian coordinates"));
    CHECK_THAT(polar, ContainsSubstring("has no CoordinateX / CoordinateY arrays"));
    f = channel_model();
    zone_of(f).coordinates[0].pop_back();
    CHECK_THAT(error_of(f), ContainsSubstring("coordinate array 'CoordinateX' has 44 values, zone 'Channel' has 45 vertices"));
    f = channel_model();
    zone_of(f).vertices = 0;
    CHECK_THAT(error_of(f), ContainsSubstring("invalid number of vertices (0)"));

    f = channel_model();
    zone_of(f).grids = 2;
    ImportReport r;
    build(f, &r);
    REQUIRE(r.warnings.size() == 1);
    CHECK_THAT(r.warnings[0], ContainsSubstring("2 GridCoordinates_t nodes; the first one is used"));
}

TEST_CASE("CGNS: invalid sections", "[cgns][translate][errors]") {
    CgnsFile f = channel_model();
    zone_of(f).sections[1].start = 60; // bars 60..68 overlap the triangles 1..64
    zone_of(f).sections[1].end = 63;
    CHECK_THAT(error_of(f), ContainsSubstring("section 'Fluid' (TRI_3, elements 1-64) and section 'inlet_edges' (BAR_2, elements 60-63) overlap"));
    f = channel_model();
    zone_of(f).sections[1].end = zone_of(f).sections[1].start - 2;
    CHECK_THAT(error_of(f), ContainsSubstring("invalid element range"));
    f = channel_model();
    zone_of(f).sections[0].connectivity.pop_back();
    CHECK_THAT(error_of(f), ContainsSubstring("connectivity has 191 values, expected 192"));
    f = channel_model();
    zone_of(f).sections[0].connectivity[4] = 46;
    zone_of(f).sections[0].connectivity[7] = 0;
    const std::string err = error_of(f);
    CHECK_THAT(err, ContainsSubstring("section 'Fluid' (TRI_3, elements 1-64): element 2 references vertex 46, zone 'Channel' has 45 vertices"));
    CHECK_THAT(err, ContainsSubstring("1 more invalid vertex references"));
}

TEST_CASE("CGNS: invalid boundary conditions", "[cgns][translate][errors]") {
    auto with_bc = [](auto&& edit) {
        CgnsFile f = channel_model();
        edit(zone_of(f).bcs[0]);
        return error_of(f);
    };
    CHECK_THAT(with_bc([](BoundaryCondition& b) { b.points = {3, 2}; }), ContainsSubstring("BC 'inlet' (BC 1): invalid ElementRange"));
    CHECK_THAT(with_bc([](BoundaryCondition& b) { b.points = {80, 95}; }),
               ContainsSubstring("ElementRange 80-95 exceeds the elements of zone 'Channel'"));
    CHECK_THAT(with_bc([](BoundaryCondition& b) { b.point_set = kElementList; b.points = {65, 200, 300}; }),
               ContainsSubstring("refers to 2 element(s) that do not exist (first: 200)"));
    CHECK_THAT(with_bc([](BoundaryCondition& b) { b.point_set = kElementList; b.points = {65, 1}; }),
               ContainsSubstring("mixes 1 boundary edges (BAR_2) with 1 other elements"));
    CHECK_THAT(with_bc([](BoundaryCondition& b) { b.point_set = kPointListDonor; }),
               ContainsSubstring("point set type PointListDonor is not supported"));
    CHECK_THAT(with_bc([](BoundaryCondition& b) { b.point_set = kPointRange; b.location = kIFaceCenter; }),
               ContainsSubstring("grid location IFaceCenter is not supported"));
    CHECK_THAT(with_bc([](BoundaryCondition& b) { b.point_set = kPointList; b.location = kVertex; b.points = {1, 46, 47}; }),
               ContainsSubstring("2 vertex index(es) outside 1-45"));
    CHECK_THAT(with_bc([](BoundaryCondition& b) { b.point_set = kPointRange; b.location = kVertex; b.points = {0, 9}; }),
               ContainsSubstring("PointRange 0-9 exceeds the vertices"));
}

TEST_CASE("CGNS: element numbers appear in the messages of build_mesh", "[cgns][translate][errors]") {
    // A BAR_2 inside the domain with a wall BC: build_mesh names it by its CGNS element number.
    CgnsFile f = channel_model();
    Zone& z = zone_of(f);
    z.sections.push_back({"Baffle", kBar2, 89, 89, {11, 12}, {}, true});
    z.bcs.push_back({"baffle_wall", kBCWall, kElementRange, kVertex, {89, 89}, ""});
    CHECK_THAT(error_of(f), ContainsSubstring("line element 89 (node 11 (0.5, 0.25) - node 12 (1, 0.25))") &&
                                ContainsSubstring("lies inside the domain"));
    // Boundary bars without a BC: the boundary is not covered.
    CgnsFile g = channel_model();
    zone_of(g).bcs.pop_back();
    CHECK_THAT(error_of(g), ContainsSubstring("8 boundary line elements are in no physical group"));
}

TEST_CASE("CGNS: corrupted structures never crash the translation", "[cgns][translate][errors][robustness]") {
    std::uint64_t state = 0x853C49E6748FEA9Bull;
    auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    auto any = [&next](std::int64_t lo, std::int64_t hi) {
        return lo + static_cast<std::int64_t>(next() % static_cast<std::uint64_t>(hi - lo + 1));
    };
    int built = 0;
    for (int trial = 0; trial < 1500; ++trial) {
        ChannelSpec s;
        s.mixed = trial % 2 == 1;
        s.point_set = std::array<int, 4>{kElementRange, kElementList, kPointList, kPointRange}[trial % 4];
        s.location = trial % 3 == 0 ? kVertex : kEdgeCenter;
        if (s.point_set == kPointRange && s.location == kVertex) s.location = kEdgeCenter;
        CgnsFile f = channel_model(s);
        Zone& z = zone_of(f);
        for (int edit = 0; edit < 1 + static_cast<int>(next() % 3); ++edit) {
            Section& sec = z.sections[next() % z.sections.size()];
            BoundaryCondition& bc = z.bcs[next() % z.bcs.size()];
            switch (next() % 9) {
            case 0: if (!sec.connectivity.empty()) sec.connectivity[next() % sec.connectivity.size()] = any(-3, 120); break;
            case 1: sec.type = static_cast<int>(any(-2, 60)); break;
            case 2: sec.start += any(-100, 100); break;
            case 3: sec.end += any(-100, 100); break;
            case 4: if (!bc.points.empty()) bc.points[next() % bc.points.size()] = any(-5, 200); break;
            case 5: bc.point_set = static_cast<int>(any(0, 9)); break;
            case 6: bc.location = static_cast<int>(any(0, 9)); break;
            case 7: bc.type = static_cast<int>(any(0, 27)); break;
            default: if (!sec.offsets.empty()) sec.offsets[next() % sec.offsets.size()] = any(-3, 400); break;
            }
        }
        try {
            const Mesh m = build_mesh(cgns_to_mesh_data(f, "fuzz.cgns"));
            CHECK(check_mesh(m).empty()); // anything accepted is a valid mesh
            ++built;
        } catch (const MeshImportError&) {
        }
    }
    CHECK(built > 0); // some edits are harmless (e.g. a BC type that still agrees with its name)
}
