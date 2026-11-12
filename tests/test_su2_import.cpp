// SU2 meshes through the whole import path: the fixtures of tests/data/su2/
// (regular triangular meshes, boundary markers, orientation repair, every
// optional part of the format, malformed files, files exported by Gmsh), the
// benchmark mesh through SU2, and SU2 cases in the solver.

#include "IO.hpp"
#include "SolverTestUtils.hpp"
#include "Su2TestWriter.hpp"
#include "mesh/GmshReader.hpp"
#include "mesh/MeshBuilder.hpp"
#include "mesh/MeshCheck.hpp"
#include "mesh/Su2Reader.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <map>
#include <tuple>

using namespace hydrama;
using namespace hydrama::mesh;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

const std::string kCaseDir = HYDRAMA2D_CASE_DIR;

std::string fixture(const std::string& name) { return std::string(HYDRAMA2D_TEST_DATA_DIR) + "/su2/" + name; }

Mesh import(const std::string& name, const BuildOptions& o = {}, ImportReport* r = nullptr) {
    return build_mesh(read_su2(fixture(name)), o, r);
}

std::string import_error(const std::string& name, const BuildOptions& o = {}) {
    try {
        import(name, o);
    } catch (const MeshImportError& e) {
        return e.what();
    }
    FAIL("expected MeshImportError for " << name);
    return {};
}

void require_valid(const Mesh& m) {
    const auto problems = check_mesh(m);
    for (const auto& p : problems) UNSCOPED_INFO(p);
    REQUIRE(problems.empty());
}

double total_area(const Mesh& m) {
    double a = 0.0;
    for (const auto& n : m.intma)
        a += 0.5 * ((m.x[n[1]] - m.x[n[0]]) * (m.y[n[2]] - m.y[n[0]]) - (m.x[n[2]] - m.x[n[0]]) * (m.y[n[1]] - m.y[n[0]]));
    return a;
}

double boundary_area(const Mesh& m) {
    double a = 0.0;
    for (const auto& s : m.iside) a += 0.5 * (m.x[s.node1] * m.y[s.node2] - m.x[s.node2] * m.y[s.node1]);
    return a;
}

using SideKey = std::array<int, 5>; // node1, node2, elem, side_id, flag
SideKey key(const BoundarySide& s) { return {s.node1, s.node2, s.elem, s.side_id, s.flag}; }

void check_identical(const Mesh& a, const Mesh& b) {
    REQUIRE(a.npoin == b.npoin);
    REQUIRE(a.nelem == b.nelem);
    REQUIRE(a.nboun == b.nboun);
    CHECK(a.intma == b.intma);
    CHECK(a.x == b.x);
    CHECK(a.y == b.y);
    int differing = 0;
    for (int i = 0; i < a.nboun; ++i)
        if (key(a.iside[static_cast<std::size_t>(i)]) != key(b.iside[static_cast<std::size_t>(i)])) ++differing;
    CHECK(differing == 0);
}

// ---------------------------------------------------------------------------
// Benchmark mesh as SU2 (one marker per side id)
// ---------------------------------------------------------------------------
const Mesh& legacy_mesh() {
    static const Mesh mesh = [] {
        Mesh m;
        CaptureStdout quiet;
        REQUIRE(IO::read_mesh(kCaseDir + "/constant/hinterkante.mesh", m));
        REQUIRE(IO::read_boundary_flags(kCaseDir + "/constant/hinterkante.bofl", m));
        return m;
    }();
    return mesh;
}

const std::map<int, std::string> kBenchmarkMarkers = {{1, "lower_wall"}, {2, "outlet"},     {3, "step_wall"},
                                                      {4, "inlet_upper"}, {5, "upper_wall"}, {6, "wall_6"},
                                                      {7, "wall_7"},      {8, "inlet"}};

std::string export_benchmark(const Su2WriteOptions& o = {}, const std::map<int, std::string>& names = kBenchmarkMarkers) {
    return write_su2(su2_model_from_mesh(legacy_mesh(), names), o);
}

// The legacy boundary sides grouped by side id (stable): the order of an SU2
// file, which lists the sides marker by marker.
Mesh legacy_mesh_marker_order() {
    Mesh m = legacy_mesh();
    std::stable_sort(m.iside.begin(), m.iside.end(),
                     [](const BoundarySide& a, const BoundarySide& b) { return a.side_id < b.side_id; });
    return m;
}

} // namespace

// =============================================================================
// Synthetic fixtures
// =============================================================================

TEST_CASE("SU2 fixtures: regular triangular channel", "[su2][mesh_import][fixtures]") {
    ImportReport r;
    Mesh m = import("channel_regular.su2", {}, &r);
    require_valid(m);
    CHECK(m.npoin == 45);
    CHECK(m.nelem == 64);
    CHECK(m.nboun == 24);
    CHECK(r.format == "SU2 ASCII");
    CHECK(r.reoriented_triangles == 0);
    CHECK(r.warnings.empty());
    for (int e = 0; e < m.nelem; ++e) {
        const auto& n = m.intma[static_cast<std::size_t>(e)];
        const double area = 0.5 * ((m.x[n[1]] - m.x[n[0]]) * (m.y[n[2]] - m.y[n[0]]) -
                                   (m.x[n[2]] - m.x[n[0]]) * (m.y[n[1]] - m.y[n[0]]));
        CHECK(area == 0.5 * 0.5 * 0.25); // every triangle: half a 0.5 x 0.25 cell
    }
    CHECK(total_area(m) == 4.0);
    CHECK(boundary_area(m) == 4.0);

    // Boundary markers -> conditions, in file order (physical tag = marker number).
    REQUIRE(r.patches.size() == 4);
    const std::vector<std::tuple<std::string, int, int>> expected = {
        {"inlet", 503, 4}, {"outlet", 504, 4}, {"upper_wall", 500, 8}, {"lower_wall", 500, 8}};
    for (std::size_t i = 0; i < 4; ++i) {
        CAPTURE(i);
        CHECK(r.patches[i].tag == static_cast<int>(i + 1));
        CHECK(std::make_tuple(r.patches[i].name, r.patches[i].flag, r.patches[i].sides) == expected[i]);
    }
    // Sides keep the marker order; each side knows its marker.
    for (int i = 0; i < m.nboun; ++i) CHECK(m.iside[static_cast<std::size_t>(i)].side_id == (i < 4 ? 1 : i < 8 ? 2 : i < 16 ? 3 : 4));

    // Outward normals.
    m.compute_face_normals();
    for (int i = 0; i < m.nboun; ++i) {
        const FaceNormal& n = m.face_norm[static_cast<std::size_t>(i)];
        switch (m.iside[static_cast<std::size_t>(i)].side_id) {
        case 1: CHECK((n.nx == -1.0 && n.ny == 0.0)); break;
        case 2: CHECK((n.nx == 1.0 && n.ny == 0.0)); break;
        case 3: CHECK((n.nx == 0.0 && n.ny == 1.0)); break;
        case 4: CHECK((n.nx == 0.0 && n.ny == -1.0)); break;
        default: FAIL("unexpected side id");
        }
    }
}

TEST_CASE("SU2 fixtures: orientation repair of clockwise triangles and reversed marker lines", "[su2][mesh_import][fixtures]") {
    ImportReport r;
    const Mesh repaired = import("channel_regular_clockwise.su2", {}, &r);
    require_valid(repaired);
    CHECK(r.reoriented_triangles == 64);
    CHECK_THAT(format_report(r), ContainsSubstring("64 clockwise triangle(s) reoriented counter-clockwise"));
    // The clockwise file lists each triangle as (a, c, b): the repair swaps the
    // 2nd and 3rd node back, so the mesh is identical to the regular one.
    check_identical(repaired, import("channel_regular.su2"));
}

TEST_CASE("SU2 fixtures: every optional and legacy part of the format", "[su2][mesh_import][fixtures]") {
    // NZONE/IZONE, NPOIN before NELEM, "NPOIN= n n", blanks around '=', tabs,
    // inline comments, CRLF, marker element indices, NPERIODIC, FFD trailer.
    ImportReport r;
    const Mesh m = import("channel_regular_variants.su2", {}, &r);
    CHECK(r.warnings.empty());
    check_identical(m, import("channel_regular.su2"));
}

TEST_CASE("SU2 fixtures: classic airfoil / farfield marker names", "[su2][mesh_import][fixtures][bofl]") {
    const std::string err = import_error("annulus_airfoil_farfield.su2");
    CHECK_THAT(err, ContainsSubstring("invalid mesh (2 problems)"));
    CHECK_THAT(err, ContainsSubstring("'airfoil' (tag 1)") && ContainsSubstring("no boundary keyword") &&
                        ContainsSubstring("airfoil_wall"));
    CHECK_THAT(err, ContainsSubstring("'farfield' (tag 2) is a 'farfield' boundary (Far-field / free-stream boundary), "
                                      "which HydraMa2D does not implement yet"));

    // *.bofl entries keyed by marker number make it usable.
    BuildOptions o;
    o.flag_overrides = {{1, 500}, {2, 504}};
    ImportReport r;
    Mesh m = import("annulus_airfoil_farfield.su2", o, &r);
    require_valid(m);
    CHECK(m.nelem == 2 * 24 * 6);
    CHECK(m.nboun == 48);
    CHECK(r.patches[0].source == ConditionSource::FlagFile);
    CHECK(r.patches[1].flag == 504);
    CHECK_THAT(total_area(m), WithinRel(boundary_area(m), 1e-12));
    const double polygon = 0.5 * 24 * std::sin(2.0 * std::acos(-1.0) / 24); // regular 24-gon, r = 1
    CHECK_THAT(total_area(m), WithinRel(polygon * (4.0 - 0.25), 1e-12));
    // Normals point out of the fluid: away from the centre on the outer
    // circle, towards it on the inner one.
    m.compute_face_normals();
    for (int i = 0; i < m.nboun; ++i) {
        const BoundarySide& s = m.iside[static_cast<std::size_t>(i)];
        const double xm = 0.5 * (m.x[s.node1] + m.x[s.node2]), ym = 0.5 * (m.y[s.node1] + m.y[s.node2]);
        const double radial = m.face_norm[static_cast<std::size_t>(i)].nx * xm + m.face_norm[static_cast<std::size_t>(i)].ny * ym;
        CHECK((s.side_id == 1 ? radial < 0.0 : radial > 0.0));
    }
}

TEST_CASE("SU2 fixtures: a quadrilateral inflation layer with triangles above", "[su2][mesh_import][fixtures][quad]") {
    ImportReport r;
    Mesh m = import("mixed_tri_quad.su2", {}, &r);
    require_valid(m);
    CHECK(m.num_quads() == 8);      // the row of cells on the lower wall
    CHECK(m.num_triangles() == 48);
    CHECK(m.npoin == 45);
    CHECK(m.nboun == 24);
    CHECK(boundary_area(m) == 4.0);
    // Triangles are numbered first: the lower-wall sides belong to quadrilaterals.
    for (const BoundarySide& s : m.iside) {
        const bool lower = m.y[s.node1] == 0.0 && m.y[s.node2] == 0.0;
        CHECK(m.is_quad(s.elem) == (lower || ((m.x[s.node1] == 0.0 || m.x[s.node1] == 4.0) && std::min(m.y[s.node1], m.y[s.node2]) == 0.0)));
    }
    m.compute_face_normals();
    for (int i = 0; i < m.nboun; ++i) {
        const FaceNormal& n = m.face_norm[static_cast<std::size_t>(i)];
        const BoundarySide& s = m.iside[static_cast<std::size_t>(i)];
        if (s.side_id == 4) CHECK((n.nx == 0.0 && n.ny == -1.0)); // lower wall, on the quadrilaterals
    }
    CHECK_THAT(format_report(r), ContainsSubstring("48 triangles, 8 quadrilaterals, 45 nodes"));
}

TEST_CASE("SU2 fixtures: malformed files", "[su2][mesh_import][fixtures][errors]") {
    const std::map<std::string, std::string> expected = {
        {"truncated.su2", "line 66: unexpected end of file: NELEM= 70 declares 70 lines, found 64"},
        {"too_many_elements.su2", "line 66: unexpected data line '5 34 44 43 63' outside a NELEM=, NPOIN= or NMARK= section"},
        {"point_index_out_of_range.su2", "line 3: element references point 45, but NPOIN= 45"},
        {"ndime_3.su2", "line 1: 3D mesh (NDIME= 3)"},
        {"marker_without_elems.su2", "line 115: expected 'MARKER_ELEMS= count' after 'MARKER_TAG= inlet'"},
        {"bad_coordinate.su2", "line 69: expected finite coordinates 'x y [index]', found '0.5 abc 1'"},
        {"triangle_in_marker.su2", "line 116: marker 'inlet' contains a triangle (type 5)"},
        {"quad_with_three_nodes.su2", "line 3: a quadrilateral (type 9) needs 4 node indices"},
    };
    for (const auto& [name, message] : expected) {
        CAPTURE(name);
        const std::string path = fixture("malformed/" + name);
        try {
            read_su2(path);
            FAIL("expected an error");
        } catch (const MeshImportError& e) {
            CHECK_THAT(e.what(), ContainsSubstring(path + ": " + message));
        }
    }
}

// =============================================================================
// Files exported by Gmsh
// =============================================================================

TEST_CASE("SU2 fixtures: files written by Gmsh match the Gmsh source meshes", "[su2][mesh_import][fixtures][gmsh]") {
    const std::string gmsh_dir = std::string(HYDRAMA2D_TEST_DATA_DIR) + "/gmsh/";
    for (const auto& [su2, msh] : {std::pair<std::string, std::string>{"square_gmsh.su2", "square_4.1_ascii.msh"},
                                   {"channel_cylinder_gmsh.su2", "channel_cylinder_4.1_binary.msh"}}) {
        CAPTURE(su2);
        ImportReport r;
        const Mesh a = import(su2, {}, &r);
        const Mesh b = build_mesh(read_gmsh(gmsh_dir + msh));
        require_valid(a);
        CHECK(a.intma == b.intma);
        REQUIRE(a.npoin == b.npoin);
        for (int i = 0; i < a.npoin; ++i) {
            CHECK_THAT(a.x[static_cast<std::size_t>(i)], WithinAbs(b.x[static_cast<std::size_t>(i)], 1e-15));
            CHECK_THAT(a.y[static_cast<std::size_t>(i)], WithinAbs(b.y[static_cast<std::size_t>(i)], 1e-15));
        }
        REQUIRE(a.nboun == b.nboun);
        std::multiset<std::array<int, 3>> sa, sb;
        for (const auto& s : a.iside) sa.insert({s.node1, s.node2, s.flag});
        for (const auto& s : b.iside) sb.insert({s.node1, s.node2, s.flag});
        CHECK(sa == sb);
    }
    // Gmsh writes "Symmetry Plane" as "Symmetry_Plane"; both resolve to symmetry.
    ImportReport r;
    import("channel_cylinder_gmsh.su2", {}, &r);
    CHECK(r.patches[2].name == "Symmetry_Plane");
    CHECK(r.patches[2].flag == 506);
    // Recombined quadrilaterals exported by Gmsh: the same mesh as the .msh file.
    const Mesh qa = import("square_quads_gmsh.su2");
    const Mesh qb = build_mesh(read_gmsh(gmsh_dir + "square_quads_2.2_ascii.msh"));
    require_valid(qa);
    CHECK(qa.num_quads() == 21);
    CHECK(qa.quads == qb.quads);
    REQUIRE(qa.npoin == qb.npoin);
    for (int i = 0; i < qa.npoin; ++i) {
        CHECK_THAT(qa.x[static_cast<std::size_t>(i)], WithinAbs(qb.x[static_cast<std::size_t>(i)], 1e-15));
        CHECK_THAT(qa.y[static_cast<std::size_t>(i)], WithinAbs(qb.y[static_cast<std::size_t>(i)], 1e-15));
    }
}

// =============================================================================
// Benchmark mesh through SU2
// =============================================================================

TEST_CASE("SU2: the benchmark mesh exported to SU2 imports identically", "[su2][mesh_import][hinterkante]") {
    const Mesh expected = legacy_mesh_marker_order();
    Su2WriteOptions variants[4];
    variants[1].point_index = variants[1].element_index = variants[1].comments = false;
    variants[2].points_first = variants[2].npoin_domain = variants[2].crlf = true;
    variants[3].prefix = "NZONE= 1\nIZONE= 1\n";
    variants[3].trailer = "FFD_NBOX= 0\nFFD_NLEVEL= 0\n";
    for (const Su2WriteOptions& o : variants) {
        CAPTURE(o.point_index, o.points_first, o.crlf);
        ImportReport r;
        const Mesh m = build_mesh(parse_su2(export_benchmark(o), "hinterkante.su2"), {}, &r);
        check_identical(m, expected);
        CHECK(r.patches.size() == 8);
        CHECK(r.warnings.empty());
    }
}

TEST_CASE("SU2: *.bofl entries are keyed by marker number", "[su2][mesh_import][hinterkante][bofl]") {
    BuildOptions o;
    REQUIRE(IO::read_boundary_flag_table(kCaseDir + "/constant/hinterkante.bofl", o.flag_overrides));
    std::map<int, std::string> anonymous;
    for (const auto& [id, name] : kBenchmarkMarkers) anonymous[id] = "marker_" + std::to_string(id); // no keywords
    const Mesh m = build_mesh(parse_su2(export_benchmark({}, anonymous), "hinterkante.su2"), o);
    check_identical(m, legacy_mesh_marker_order());
}

TEST_CASE("Solver: an SU2 case reproduces the legacy case bit for bit", "[su2][mesh_import][hinterkante][solver]") {
    CaseOptions opts;
    opts.steps = 10;
    const Run legacy = run_case(opts, "su2-legacy");
    REQUIRE(legacy.initialized);
    REQUIRE(legacy.state_read);

    for (const bool keep_bofl : {false, true}) {
        CAPTURE(keep_bofl);
        TempDir dir("su2-case");
        prepare_case(dir.path(), opts);
        fs::remove(dir / "constant/hinterkante.mesh");
        if (!keep_bofl) fs::remove(dir / "constant/hinterkante.bofl");
        write_text(dir / "constant/hinterkante.su2", export_benchmark());
        const Run run = run_solver(dir.path());
        INFO(run.errors);
        REQUIRE(run.initialized);
        REQUIRE(run.state_read);
        // The boundary sides come in marker order, which changes no result.
        CHECK(run.u == legacy.u);
        CHECK(run.v == legacy.v);
        CHECK(run.p == legacy.p);
        CHECK(run.rtime == legacy.rtime);
        REQUIRE(run.steps.size() == legacy.steps.size());
        for (std::size_t k = 0; k < run.steps.size(); ++k) {
            CHECK(run.steps[k].iterations == legacy.steps[k].iterations);
            CHECK(run.steps[k].final == legacy.steps[k].final);
        }
        CHECK_THAT(run.log, ContainsSubstring("Mesh format: SU2 ASCII"));
        CHECK_THAT(run.log, ContainsSubstring("Boundary 'inlet' (tag 8): inlet, flag 503, 51 sides"));
        if (keep_bofl) CHECK_THAT(run.log, ContainsSubstring("entries, by marker number"));
    }
}

// =============================================================================
// Case files and solver
// =============================================================================

TEST_CASE("locate_mesh_file: SU2 meshes", "[su2][mesh_import][case]") {
    TempDir dir("locate-su2");
    const std::string c = dir.path().string();
    MeshFileLocation loc;
    write_text(dir / "x.su2", "");
    REQUIRE(IO::locate_mesh_file(c, "x", loc));
    CHECK(loc.format == MeshFileFormat::Su2);
    CHECK(loc.path == (dir / "x.su2").string());
    {
        CaptureStderr err;
        CHECK_FALSE(IO::locate_mesh_file(c, "y", loc));
        CHECK_THAT(err.str(), ContainsSubstring("y.mesh (legacy format), ") && ContainsSubstring("y.msh (Gmsh), ") &&
                                  ContainsSubstring("y.su2 (SU2)"));
    }
    for (const char* other : {"x.msh", "x.mesh"}) {
        CAPTURE(other);
        write_text(dir / other, "");
        CaptureStderr err;
        CHECK_FALSE(IO::locate_mesh_file(c, "x", loc));
        CHECK_THAT(err.str(), ContainsSubstring("AMBIGUOUS MESH") && ContainsSubstring("x.su2"));
        fs::remove(dir / other);
    }
}

TEST_CASE("Solver: errors of SU2 cases stop the initialisation", "[su2][mesh_import][case][solver]") {
    CaseOptions opts;
    opts.steps = 1;
    auto su2_case = [&](const std::string& tag, const std::string& contents) {
        TempDir dir(tag);
        prepare_case(dir.path(), opts);
        fs::remove(dir / "constant/hinterkante.mesh");
        fs::remove(dir / "constant/hinterkante.bofl");
        write_text(dir / "constant/hinterkante.su2", contents);
        return run_solver(dir.path());
    };
    SECTION("malformed file") {
        std::string text = export_benchmark();
        text.replace(text.find("NPOIN= 8956"), 11, "NPOIN= 8950");
        const Run run = su2_case("su2-bad", text);
        CHECK_FALSE(run.initialized);
        CHECK_THAT(run.errors, ContainsSubstring("hinterkante.su2: line") && ContainsSubstring("unexpected data line"));
    }
    SECTION("marker without boundary keyword") {
        auto names = kBenchmarkMarkers;
        names[3] = "step";
        const Run run = su2_case("su2-name", export_benchmark({}, names));
        CHECK_FALSE(run.initialized);
        CHECK_THAT(run.errors, ContainsSubstring("'step' (tag 3)") && ContainsSubstring("no boundary keyword"));
    }
    SECTION("unimplemented marker type") {
        auto names = kBenchmarkMarkers;
        names[2] = "farfield";
        const Run run = su2_case("su2-far", export_benchmark({}, names));
        CHECK_FALSE(run.initialized);
        CHECK_THAT(run.errors, ContainsSubstring("'farfield' (tag 2) is a 'farfield' boundary"));
    }
}

TEST_CASE("Solver: the regular SU2 channel runs with its boundary conditions", "[su2][mesh_import][solver][fixtures]") {
    TempDir dir("su2-channel");
    CaseOptions opts;
    opts.steps = 30;
    prepare_case(dir.path(), opts);
    fs::remove(dir / "constant/hinterkante.mesh");
    fs::remove(dir / "constant/hinterkante.bofl");
    fs::copy_file(fixture("channel_regular_clockwise.su2"), dir / "constant/hinterkante.su2");

    Run run;
    {
        CaptureStdout capture;
        CaptureStderr errors;
        SimulationConfig config;
        config.casename = "hinterkante";
        config.case_dir = dir.path().string();
        Solver solver(config);
        run.initialized = solver.initialize();
        if (run.initialized) solver.run();
        run.log = capture.str();
        run.errors = errors.str();
    }
    INFO(run.errors);
    REQUIRE(run.initialized);
    CHECK_THAT(run.log, ContainsSubstring("64 clockwise triangle(s) reoriented"));

    Mesh m;
    {
        CaptureStdout quiet;
        REQUIRE(IO::read_su2_mesh((dir / "constant/hinterkante.su2").string(), "", m));
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
            } else if (s.flag == 503 && m.y[ip] > 0.0 && m.y[ip] < 1.0) { // corners belong to the walls
                CHECK(u[ip] == 1.0);
                CHECK(v[ip] == 0.0);
            } else if (s.flag == 504) {
                CHECK(p[ip] == 0.0);
            }
        }
    }
    // Developing channel flow: faster than the inflow on the centre line.
    double u_centre = 0.0;
    for (int ip = 0; ip < m.npoin; ++ip)
        if (m.y[ip] == 0.5 && m.x[ip] > 2.0) u_centre = std::max(u_centre, u[ip]);
    CHECK(u_centre > 1.0);
    CHECK(u_centre < 2.0);
}
