// CGNS files (mesh/CgnsReader.hpp: read_cgns_file, read_cgns; IO::read_cgns_mesh):
// the optional-feature behaviour of builds with and without the CGNS library,
// the fixtures of tests/data/cgns/ (synthetic ones written with CGNS 3.4 and
// files exported by Gmsh), round trips through HDF5 and ADF files, the
// benchmark case through CGNS, and CGNS cases in the solver.
//
// Without the CGNS library the file tests are skipped (SKIP) and the "not
// available" path is tested instead.

#include "CgnsTestModels.hpp"
#include "CgnsTestWriter.hpp"
#include "IO.hpp"
#include "SolverTestUtils.hpp"
#include "mesh/CgnsReader.hpp"
#include "mesh/GmshReader.hpp"
#include "mesh/MeshBuilder.hpp"
#include "mesh/MeshCheck.hpp"
#include "mesh/Su2Reader.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>

using namespace hydrama;
using namespace hydrama::mesh;
using namespace hydrama::test;
using namespace hydrama::test::cgnsmodels;
using Catch::Matchers::ContainsSubstring;

namespace {

const std::string kData = HYDRAMA2D_TEST_DATA_DIR;

std::string cgns_fixture(const std::string& name) { return kData + "/cgns/" + name; }

#define REQUIRE_CGNS_SUPPORT()                                                              \
    do {                                                                                    \
        if (!cgns_support_available()) SKIP("HydraMa2D was built without the CGNS library"); \
    } while (0)

const Mesh& su2_channel() {
    static const Mesh m = build_mesh(read_su2(kData + "/su2/channel_regular.su2"));
    return m;
}

void require_valid(const Mesh& m) {
    const auto problems = check_mesh(m);
    for (const auto& p : problems) UNSCOPED_INFO(p);
    REQUIRE(problems.empty());
}

std::multiset<std::array<int, 3>> sides_of(const Mesh& m) {
    std::multiset<std::array<int, 3>> out;
    for (const auto& s : m.iside) out.insert({s.node1, s.node2, s.flag});
    return out;
}

// The SU2 channel mesh; `with_flags`: also the same boundary conditions
// (inlet, outlet, two walls).
void check_channel(const Mesh& m, bool with_flags = true) {
    require_valid(m);
    const Mesh& ref = su2_channel();
    CHECK(m.intma == ref.intma);
    CHECK(m.x == ref.x);
    CHECK(m.y == ref.y);
    if (with_flags) {
        CHECK(sides_of(m) == sides_of(ref));
    } else {
        std::multiset<std::array<int, 2>> a, b;
        for (const auto& s : m.iside) a.insert({s.node1, s.node2});
        for (const auto& s : ref.iside) b.insert({s.node1, s.node2});
        CHECK(a == b);
    }
}

// Numbering-independent description of a mesh: triangles and boundary sides
// by their (rounded) coordinates, sides with their flag.
using Point = std::pair<long long, long long>;
Point point_of(const Mesh& m, int n) {
    return {std::llround(m.x[static_cast<std::size_t>(n)] * 1e9), std::llround(m.y[static_cast<std::size_t>(n)] * 1e9)};
}
std::multiset<std::array<Point, 3>> triangles_by_position(const Mesh& m) {
    std::multiset<std::array<Point, 3>> out;
    for (const auto& e : m.intma) {
        std::array<Point, 3> t = {point_of(m, e[0]), point_of(m, e[1]), point_of(m, e[2])};
        std::rotate(t.begin(), std::min_element(t.begin(), t.end()), t.end()); // counter-clockwise, canonical start
        out.insert(t);
    }
    return out;
}
std::multiset<std::tuple<Point, Point, int>> sides_by_position(const Mesh& m) {
    std::multiset<std::tuple<Point, Point, int>> out;
    for (const auto& s : m.iside) out.insert({point_of(m, s.node1), point_of(m, s.node2), s.flag});
    return out;
}

std::string import_error(const std::string& path, const BuildOptions& o = {}) {
    try {
        build_mesh(read_cgns(path), o);
    } catch (const MeshImportError& e) {
        return e.what();
    }
    FAIL("expected MeshImportError for " << path);
    return {};
}

Run run_case_dir(const fs::path& dir) {
    Run run;
    CaptureStdout capture;
    CaptureStderr errors;
    SimulationConfig config;
    config.casename = "hinterkante";
    config.case_dir = dir.string();
    Solver solver(config);
    run.initialized = solver.initialize();
    if (run.initialized) solver.run();
    run.log = capture.str();
    run.errors = errors.str();
    return run;
}

} // namespace

// =============================================================================
// The optional feature
// =============================================================================

TEST_CASE("CGNS support matches the build configuration", "[cgns][file]") {
#if defined(HYDRAMA2D_HAVE_CGNS) && HYDRAMA2D_HAVE_CGNS
    CHECK(cgns_support_available());
    const int v = CGNS_VERSION;
    CHECK(cgns_library_version() ==
          std::to_string(v / 1000) + "." + std::to_string((v / 100) % 10) + "." + std::to_string((v / 10) % 10));
#else
    CHECK_FALSE(cgns_support_available());
    CHECK(cgns_library_version().empty());
#endif
}

TEST_CASE("Without the CGNS library CGNS files give a clear error", "[cgns][file]") {
    if (cgns_support_available()) SKIP("HydraMa2D was built with the CGNS library");
    const std::string path = cgns_fixture("channel_sections.cgns");
    CHECK_THROWS_WITH(read_cgns_file(path), ContainsSubstring(path + ": HydraMa2D was built without CGNS support") &&
                                                ContainsSubstring("docs/MESH_IMPORT.md"));
    CHECK_THROWS_AS(read_cgns(path), MeshImportError);
    Mesh m;
    {
        CaptureStderr err;
        CHECK_FALSE(IO::read_cgns_mesh(path, "", m));
        CHECK_THAT(err.str(), ContainsSubstring("built without CGNS support"));
    }
    // In the solver: the case is found, the error is clear.
    TempDir dir("cgns-unavailable");
    CaseOptions opts;
    opts.steps = 1;
    prepare_case(dir.path(), opts);
    fs::remove(dir / "constant/hinterkante.mesh");
    fs::copy_file(path, dir / "constant/hinterkante.cgns");
    const Run run = run_case_dir(dir.path());
    CHECK_FALSE(run.initialized);
    CHECK_THAT(run.errors, ContainsSubstring("built without CGNS support"));
}

TEST_CASE("locate_mesh_file: CGNS meshes", "[cgns][file][case]") {
    TempDir dir("locate-cgns");
    const std::string c = dir.path().string();
    MeshFileLocation loc;
    write_text(dir / "x.cgns", "");
    REQUIRE(IO::locate_mesh_file(c, "x", loc));
    CHECK(loc.format == MeshFileFormat::Cgns);
    {
        CaptureStderr err;
        CHECK_FALSE(IO::locate_mesh_file(c, "y", loc));
        CHECK_THAT(err.str(), ContainsSubstring("y.su2 (SU2) OR ") && ContainsSubstring("y.cgns (CGNS)"));
    }
    write_text(dir / "x.su2", "");
    CaptureStderr err;
    CHECK_FALSE(IO::locate_mesh_file(c, "x", loc));
    CHECK_THAT(err.str(), ContainsSubstring("AMBIGUOUS MESH") && ContainsSubstring("x.cgns"));
}

// =============================================================================
// Committed fixtures (written with CGNS 3.4; Gmsh exports)
// =============================================================================

TEST_CASE("CGNS fixtures: valid meshes", "[cgns][file][fixtures]") {
    REQUIRE_CGNS_SUPPORT();
    struct Case {
        const char* file;
        const char* format;
        bool vertex_bcs;
    };
    for (const Case c : {Case{"channel_sections.cgns", "CGNS 3.40 HDF5", false},
                         Case{"channel_sections_adf.cgns", "CGNS 3.40 ADF", false},
                         Case{"channel_mixed_vertex_bcs.cgns", "CGNS 3.40 HDF5", true},
                         Case{"channel_families_clockwise.cgns", "CGNS 3.40 HDF5", false}}) {
        CAPTURE(c.file);
        ImportReport r;
        const Mesh m = build_mesh(read_cgns(cgns_fixture(c.file)), {}, &r);
        check_channel(m, !c.vertex_bcs); // the vertex-BC fixture has a symmetry line at the bottom
        CHECK(r.format == c.format);
        CHECK(r.warnings.size() == (c.vertex_bcs ? 4u : 0u));
    }
    // Boundary conditions: name + declared type, declared type only, family.
    ImportReport a, b, c;
    build_mesh(read_cgns(cgns_fixture("channel_sections.cgns")), {}, &a);
    build_mesh(read_cgns(cgns_fixture("channel_mixed_vertex_bcs.cgns")), {}, &b);
    const Mesh families = build_mesh(read_cgns(cgns_fixture("channel_families_clockwise.cgns")), {}, &c);
    CHECK(a.patches[0].matched == "inlet + CGNS BCType BCInflowSubsonic");
    CHECK(b.patches[3].name == "bottom");
    CHECK(b.patches[3].flag == 506);
    CHECK(b.patches[3].source == ConditionSource::FileType);
    CHECK(c.patches[2].name == "Walls");
    CHECK(c.patches[2].matched == "walls + CGNS FamilyBC BCWall of family 'Walls'");
    CHECK(c.reoriented_triangles == 64);
}

TEST_CASE("CGNS fixtures: a quadrilateral mesh", "[cgns][file][fixtures][quad]") {
    REQUIRE_CGNS_SUPPORT();
    ImportReport r;
    const Mesh m = build_mesh(read_cgns(cgns_fixture("channel_quads.cgns")), {}, &r);
    require_valid(m);
    CHECK(m.num_quads() == 32);
    CHECK(m.num_triangles() == 0);
    CHECK(m.npoin == 45);
    CHECK(m.nboun == 24);
    CHECK(r.format == "CGNS 3.40 ADF");
    double area = 0.0;
    for (const auto& s : m.iside) area += 0.5 * (m.x[s.node1] * m.y[s.node2] - m.x[s.node2] * m.y[s.node1]);
    CHECK(area == 4.0);
    std::map<int, int> flags;
    for (const auto& p : r.patches) flags[p.tag] = p.flag;
    CHECK(flags == std::map<int, int>{{1, 503}, {2, 504}, {3, 500}, {4, 500}});
}

TEST_CASE("CGNS fixtures: meshes HydraMa2D rejects", "[cgns][file][fixtures][errors]") {
    REQUIRE_CGNS_SUPPORT();
    const std::map<std::string, std::string> expected = {
        {"unsupported_3d.cgns", "base 'Base' is 3D (CellDimension 3); HydraMa2D needs a 2D mesh"},
        {"unsupported_tri6.cgns", "section 'Curved' (TRI_6, elements 89-89): higher-order elements"},
        {"unsupported_ngon.cgns", "section 'Polygons' (NGON_n, elements 89-90): polygonal / polyhedral elements"},
        {"unsupported_mixed_hexa.cgns", "section 'All' (MIXED, elements 1-89): 1 HEXA_8 element: 3D elements"},
        {"unsupported_multizone.cgns", "contains 2 zones ('Channel', 'Channel2'); multi-zone meshes are not supported yet"},
        {"unsupported_structured.cgns", "zone 'Block' is structured"},
        {"invalid_name_type_conflict.cgns", "'inlet' (tag 1): the name means inlet, but CGNS BCType BCWall means wall"},
    };
    for (const auto& [file, message] : expected) {
        CAPTURE(file);
        CHECK_THAT(import_error(cgns_fixture(file)), ContainsSubstring(cgns_fixture(file) + ": invalid mesh") &&
                                                         ContainsSubstring(message));
    }
}

TEST_CASE("CGNS fixtures: files written by Gmsh (BCs by number)", "[cgns][file][fixtures][gmsh][bofl]") {
    REQUIRE_CGNS_SUPPORT();
    // Gmsh names every BC after its curve (L_1, L_2, ...): no keywords.
    CHECK_THAT(import_error(cgns_fixture("channel_cylinder_gmsh.cgns")),
               ContainsSubstring("invalid mesh (8 problems)") && ContainsSubstring("'L_1' (tag 1)"));

    struct Case {
        const char* cgns;
        const char* msh;
        std::map<int, int> flags; // BC number (= curve) -> flag code of the .msh physical group
    };
    for (const Case& c : {Case{"square_gmsh.cgns", "square_4.1_ascii.msh", {{1, 500}, {2, 504}, {3, 500}, {4, 503}}},
                          Case{"channel_cylinder_gmsh.cgns", "channel_cylinder_4.1_binary.msh",
                               {{1, 506}, {2, 504}, {3, 506}, {4, 503}, {5, 500}, {6, 500}, {7, 500}, {8, 500}}}}) {
        CAPTURE(c.cgns);
        BuildOptions o;
        o.flag_overrides = c.flags;
        ImportReport r;
        const Mesh a = build_mesh(read_cgns(cgns_fixture(c.cgns)), o, &r);
        const Mesh b = build_mesh(read_gmsh(kData + "/gmsh/" + c.msh));
        require_valid(a);
        // Gmsh numbers the CGNS vertices differently: compare by position.
        CHECK(a.npoin == b.npoin);
        CHECK(a.nelem == b.nelem);
        CHECK(triangles_by_position(a) == triangles_by_position(b));
        CHECK(sides_by_position(a) == sides_by_position(b));
        // The BC that Gmsh writes for the surface is not a boundary.
        REQUIRE(r.warnings.size() == 1);
        CHECK_THAT(r.warnings[0], ContainsSubstring("BC 'S_1'") && ContainsSubstring("2D cells (a region, not a boundary): ignored"));
    }
}

TEST_CASE("CGNS files: read errors", "[cgns][file][errors]") {
    REQUIRE_CGNS_SUPPORT();
    TempDir dir("cgns-errors");
    CHECK_THROWS_WITH(read_cgns_file((dir / "missing.cgns").string()), ContainsSubstring("missing.cgns: cannot open mesh file"));
    CHECK_THROWS_WITH(read_cgns_file(dir.path().string()), ContainsSubstring("cannot open mesh file"));
    write_text(dir / "text.cgns", "this is not a CGNS file\n");
    CHECK_THROWS_WITH(read_cgns_file((dir / "text.cgns").string()), ContainsSubstring("text.cgns: cannot open CGNS file: "));
    // A Gmsh file with the wrong extension is not a CGNS file either.
    fs::copy_file(kData + "/gmsh/square_4.1_ascii.msh", dir / "gmsh.cgns");
    CHECK_THROWS_WITH(read_cgns_file((dir / "gmsh.cgns").string()), ContainsSubstring("cannot open CGNS file"));
}

#if defined(HYDRAMA2D_HAVE_CGNS) && HYDRAMA2D_HAVE_CGNS

// =============================================================================
// With the library: files written by the test writer
// =============================================================================

namespace {

const Mesh& legacy_mesh() {
    static const Mesh mesh = [] {
        Mesh m;
        CaptureStdout quiet;
        REQUIRE(IO::read_mesh(std::string(HYDRAMA2D_CASE_DIR) + "/constant/hinterkante.mesh", m));
        REQUIRE(IO::read_boundary_flags(std::string(HYDRAMA2D_CASE_DIR) + "/constant/hinterkante.bofl", m));
        return m;
    }();
    return mesh;
}

const std::map<int, std::string> kBenchmarkNames = {{1, "lower_wall"}, {2, "outlet"}, {3, "step_wall"}, {4, "inflow"},
                                                    {5, "upper"},      {6, "wall_6"}, {7, "wall_7"},    {8, "inlet"}};
const std::map<int, int> kBenchmarkTypes = {{5, kBCWall}, {2, kBCOutflow}, {4, kBCInflowSubsonic}};

} // namespace

TEST_CASE("CGNS library: nodes per element agree with the element table", "[cgns][file]") {
    for (int t = 0; t <= kLastElementType; ++t) {
        CAPTURE(t, element_type_info(t).name);
        int npe = -1;
        const int status = cg_npe(static_cast<CGNS_ENUMT(ElementType_t)>(t), &npe);
        if (element_type_info(t).nodes > 0) {
            REQUIRE(status == CG_OK);
            CHECK(npe == element_type_info(t).nodes);
        } else if (status == CG_OK) {
            CHECK(npe <= 0); // MIXED, NGON_n, NFACE_n: variable
        }
    }
}

TEST_CASE("CGNS files: ADF and HDF5 are read whatever file type was set before", "[cgns][file]") {
    // CGNS 4.x stops detecting the file type in cg_open once cg_set_file_type()
    // was called (e.g. by a writer in the same process); the reader detects it.
    for (const int type : {CG_FILE_NONE, CG_FILE_ADF, CG_FILE_HDF5}) {
        CAPTURE(type);
        REQUIRE(cg_set_file_type(type) == CG_OK);
        for (const char* file : {"channel_sections_adf.cgns", "channel_sections.cgns"}) {
            CAPTURE(file);
            check_channel(build_mesh(read_cgns(cgns_fixture(file))));
        }
    }
    cg_set_file_type(CG_FILE_NONE);
}

TEST_CASE("CGNS files: every layout round-trips through HDF5 and ADF", "[cgns][file]") {
    std::vector<std::pair<std::string, ChannelSpec>> specs;
    specs.push_back({"sections + ElementRange", {}});
    ChannelSpec list;
    list.point_set = kElementList;
    specs.push_back({"ElementList", list});
    ChannelSpec mixed;
    mixed.mixed = true;
    mixed.point_set = kPointList;
    mixed.location = kEdgeCenter;
    specs.push_back({"MIXED + PointList EdgeCenter", mixed});
    ChannelSpec vertex;
    vertex.point_set = kPointList;
    vertex.location = kVertex;
    vertex.clockwise = true;
    specs.push_back({"PointList Vertex, clockwise", vertex});
    ChannelSpec families;
    families.with_z = true;
    families.names = {"a", "b", "c", "d"};
    families.types = {kFamilySpecified, kFamilySpecified, kFamilySpecified, kFamilySpecified};
    families.families = {"In", "Out", "Top", "Bottom"};
    families.family_nodes = {{"In", true, kBCInflow}, {"Out", true, kBCOutflow}, {"Top", true, kBCWall}, {"Bottom", true, kBCSymmetryPlane}};
    specs.push_back({"families", families});

    TempDir dir("cgns-roundtrip");
    for (const auto& [label, spec] : specs) {
        for (const CgnsFileKind kind : {CgnsFileKind::Hdf5, CgnsFileKind::Adf}) {
            CAPTURE(label, kind == CgnsFileKind::Adf);
            const CgnsFile model = channel_model(spec);
            const std::string path = (dir / "m.cgns").string();
            fs::remove(path);
            write_cgns(model, path, kind);
            const CgnsFile read = read_cgns_file(path);

            // The structure comes back.
            CHECK_THAT(read.format, ContainsSubstring(kind == CgnsFileKind::Adf ? " ADF" : " HDF5"));
            REQUIRE(read.bases.size() == 1);
            const Base& b = read.bases[0];
            CHECK(b.physical_dimension == model.bases[0].physical_dimension);
            CHECK(b.families.size() == model.bases[0].families.size());
            for (std::size_t i = 0; i < b.families.size(); ++i) {
                CHECK(b.families[i].name == model.bases[0].families[i].name);
                CHECK(b.families[i].bc_type == model.bases[0].families[i].bc_type);
            }
            const Zone& z = b.zones.at(0);
            const Zone& mz = model.bases[0].zones[0];
            CHECK(z.vertices == mz.vertices);
            CHECK(z.coordinate_names == mz.coordinate_names);
            CHECK(z.coordinates == mz.coordinates);
            REQUIRE(z.sections.size() == mz.sections.size());
            for (std::size_t s = 0; s < z.sections.size(); ++s) {
                CHECK(z.sections[s].name == mz.sections[s].name);
                CHECK(z.sections[s].type == mz.sections[s].type);
                CHECK(z.sections[s].start == mz.sections[s].start);
                CHECK(z.sections[s].end == mz.sections[s].end);
                CHECK(z.sections[s].connectivity == mz.sections[s].connectivity);
                if (mz.sections[s].type == kMixed) CHECK(z.sections[s].offsets == mz.sections[s].offsets);
            }
            REQUIRE(z.bcs.size() == mz.bcs.size());
            for (std::size_t k = 0; k < z.bcs.size(); ++k) {
                CHECK(z.bcs[k].name == mz.bcs[k].name);
                CHECK(z.bcs[k].type == mz.bcs[k].type);
                CHECK(z.bcs[k].points == mz.bcs[k].points);
                CHECK(z.bcs[k].family == mz.bcs[k].family);
                // CGNS 3.x stores ElementRange / ElementList as PointRange /
                // PointList at an element location; both are read alike.
                if (mz.bcs[k].point_set == kElementRange || mz.bcs[k].point_set == kElementList) {
                    CHECK((z.bcs[k].point_set == mz.bcs[k].point_set ||
                           z.bcs[k].point_set == (mz.bcs[k].point_set == kElementRange ? kPointRange : kPointList)));
                } else {
                    CHECK(z.bcs[k].point_set == mz.bcs[k].point_set);
                    CHECK(z.bcs[k].location == mz.bcs[k].location);
                }
            }
            // ... and gives the same mesh as the in-memory model.
            const Mesh from_file = build_mesh(cgns_to_mesh_data(read, path));
            const Mesh from_model = build_mesh(cgns_to_mesh_data(model, path));
            CHECK(from_file.intma == from_model.intma);
            CHECK(from_file.x == from_model.x);
            CHECK(sides_of(from_file) == sides_of(from_model));
            check_channel(from_file, label != "families"); // the families case has a symmetry line
        }
    }
}

TEST_CASE("CGNS files: unsupported structures are reported from files", "[cgns][file][errors]") {
    // Element types the reader does not load are reported by type and range.
    TempDir dir("cgns-unsupported");
    for (const int type : {kTetra4, kHexa8, kTri6, kNGon}) {
        CAPTURE(element_type_name(type));
        CgnsFile f = channel_model();
        Zone& z = f.bases[0].zones[0];
        const int npe = element_type_info(type).nodes;
        Section extra{"Extra", type, 89, 90, {}, {}, true};
        if (type == kNGon) {
            extra.connectivity = {1, 2, 10, 2, 3, 11};
            extra.offsets = {0, 3, 6};
        } else {
            for (int k = 0; k < 2 * npe; ++k) extra.connectivity.push_back(1 + k % 45);
        }
        z.sections.push_back(extra);
        const std::string path = (dir / ("u" + std::to_string(type) + ".cgns")).string();
        write_cgns(f, path);
        const CgnsFile read = read_cgns_file(path);
        CHECK_FALSE(read.bases[0].zones[0].sections.back().connectivity_read);
        CHECK_THAT(import_error(path), ContainsSubstring("section 'Extra' (" + element_type_name(type) + ", elements 89-90)"));
    }
}

TEST_CASE("CGNS files: the benchmark mesh through a CGNS file", "[cgns][file][hinterkante]") {
    TempDir dir("cgns-benchmark");
    const std::string path = (dir / "hinterkante.cgns").string();
    write_cgns(cgns_model_from_mesh(legacy_mesh(), kBenchmarkNames, kBenchmarkTypes), path);
    ImportReport r;
    const Mesh m = build_mesh(read_cgns(path), {}, &r);
    CHECK(m.intma == legacy_mesh().intma);
    CHECK(m.x == legacy_mesh().x); // bit for bit (binary doubles)
    CHECK(m.y == legacy_mesh().y);
    std::vector<BoundarySide> expected = legacy_mesh().iside;
    std::stable_sort(expected.begin(), expected.end(),
                     [](const BoundarySide& a, const BoundarySide& b) { return a.side_id < b.side_id; });
    REQUIRE(m.nboun == static_cast<int>(expected.size()));
    int differing = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const BoundarySide &a = m.iside[i], &b = expected[i];
        if (a.node1 != b.node1 || a.node2 != b.node2 || a.elem != b.elem || a.side_id != b.side_id || a.flag != b.flag)
            ++differing;
    }
    CHECK(differing == 0);
    CHECK(r.patches.size() == 8);
    CHECK(r.patches[4].source == ConditionSource::FileType); // "upper" + BCWall
}

TEST_CASE("Solver: a CGNS case reproduces the legacy case bit for bit", "[cgns][file][hinterkante][solver]") {
    CaseOptions opts;
    opts.steps = 10;
    const Run legacy = run_case(opts, "cgns-legacy");
    REQUIRE(legacy.initialized);
    REQUIRE(legacy.state_read);

    TempDir dir("cgns-case");
    prepare_case(dir.path(), opts);
    fs::remove(dir / "constant/hinterkante.mesh");
    fs::remove(dir / "constant/hinterkante.bofl");
    write_cgns(cgns_model_from_mesh(legacy_mesh(), kBenchmarkNames, kBenchmarkTypes), (dir / "constant/hinterkante.cgns").string());
    const Run run = run_solver(dir.path());
    INFO(run.errors);
    REQUIRE(run.initialized);
    REQUIRE(run.state_read);
    CHECK(run.u == legacy.u);
    CHECK(run.v == legacy.v);
    CHECK(run.p == legacy.p);
    CHECK(run.rtime == legacy.rtime);
    REQUIRE(run.steps.size() == legacy.steps.size());
    for (std::size_t k = 0; k < run.steps.size(); ++k) CHECK(run.steps[k].iterations == legacy.steps[k].iterations);
    CHECK_THAT(run.log, ContainsSubstring("Mesh format: CGNS "));
    CHECK_THAT(run.log, ContainsSubstring("Boundary 'upper' (tag 5): wall, flag 500, 101 sides [declared type]"));
}

#endif // HYDRAMA2D_HAVE_CGNS

// =============================================================================
// CGNS cases in the solver (committed fixtures)
// =============================================================================

TEST_CASE("Solver: the CGNS channel fixture runs with its boundary conditions", "[cgns][file][solver][fixtures]") {
    REQUIRE_CGNS_SUPPORT();
    TempDir dir("cgns-channel");
    CaseOptions opts;
    opts.steps = 30;
    prepare_case(dir.path(), opts);
    fs::remove(dir / "constant/hinterkante.mesh");
    fs::remove(dir / "constant/hinterkante.bofl");
    fs::copy_file(cgns_fixture("channel_families_clockwise.cgns"), dir / "constant/hinterkante.cgns");
    const Run run = run_case_dir(dir.path());
    INFO(run.errors);
    REQUIRE(run.initialized);
    CHECK_THAT(run.log, ContainsSubstring("64 clockwise triangle(s) reoriented"));
    CHECK_THAT(run.log, ContainsSubstring("Boundary 'Inflow' (tag 1): inlet, flag 503, 4 sides [name + declared type]"));

    Mesh m;
    {
        CaptureStdout quiet;
        REQUIRE(IO::read_cgns_mesh((dir / "constant/hinterkante.cgns").string(), "", m));
    }
    std::vector<double> u(m.npoin), v(m.npoin), p(m.npoin);
    double rtime = 0.0;
    int istart = 0;
    REQUIRE(IO::read_restart((dir / "results/hinterkante.var").string(), rtime, istart, u, v, p));
    CHECK(istart == 30);
    CHECK(count_non_finite(u) + count_non_finite(v) + count_non_finite(p) == 0);
    for (const BoundarySide& s : m.iside) {
        for (int ip : {s.node1, s.node2}) {
            CAPTURE(s.side_id, ip, m.x[ip], m.y[ip]);
            if (s.flag == 500) {
                CHECK(u[ip] == 0.0);
                CHECK(v[ip] == 0.0);
            } else if (s.flag == 503 && m.y[ip] > 0.0 && m.y[ip] < 1.0) {
                CHECK(u[ip] == 1.0);
                CHECK(v[ip] == 0.0);
            } else if (s.flag == 504) {
                CHECK(p[ip] == 0.0);
            }
        }
    }
}

TEST_CASE("Solver: a Gmsh CGNS export runs with *.bofl entries by BC number", "[cgns][file][solver][bofl]") {
    REQUIRE_CGNS_SUPPORT();
    TempDir dir("cgns-gmsh-case");
    CaseOptions opts;
    opts.steps = 5;
    prepare_case(dir.path(), opts);
    fs::remove(dir / "constant/hinterkante.mesh");
    fs::copy_file(cgns_fixture("channel_cylinder_gmsh.cgns"), dir / "constant/hinterkante.cgns");
    write_text(dir / "constant/hinterkante.bofl", "8\n1 506\n2 504\n3 506\n4 503\n5 500\n6 500\n7 500\n8 500\n");
    const Run run = run_case_dir(dir.path());
    INFO(run.errors);
    REQUIRE(run.initialized);
    CHECK_THAT(run.log, ContainsSubstring("Boundary flag overrides:") && ContainsSubstring("(8 entries, by BC number)"));
    CHECK_THAT(run.log, ContainsSubstring("Boundary 'L_4' (tag 4): inlet, flag 503, 4 sides [*.bofl]"));
    CHECK_THAT(run.log, ContainsSubstring("WARNING: BC 'S_1' (BC 9) is defined on 2D cells"));
    CHECK_THAT(run.log, ContainsSubstring("Simulation Completed"));
}
