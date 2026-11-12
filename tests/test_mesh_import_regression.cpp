// End-to-end: Gmsh cases in the solver.
//   * The legacy benchmark mesh exported to every Gmsh variant imports to the
//     identical Mesh, and the solver reproduces the legacy run bit for bit.
//   * Case discovery (constant/<case>.mesh or .msh), *.bofl handling, errors.
//   * A mesh written by Gmsh itself runs in the solver with the expected
//     boundary conditions.

#include "GmshTestWriter.hpp"
#include "IO.hpp"
#include "SolverTestUtils.hpp"
#include "mesh/GmshReader.hpp"
#include "mesh/MeshBuilder.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;

namespace {

const std::string kCaseDir = HYDRAMA2D_CASE_DIR;

// Legacy benchmark mesh with flags from its *.bofl file.
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

// Physical names of the benchmark side ids that reproduce its *.bofl flags
// (1,3,5,6,7: 500; 2: 504; 4,8: 503) through the naming convention, in
// assorted spellings.
const std::map<int, std::string> kBenchmarkNames = {{1, "bottom_wall"}, {2, "Outlet"},      {3, "step wall"},
                                                   {4, "inlet-upper"},  {5, "WallTop"},      {6, "noSlip"},
                                                   {7, "wall_7"},       {8, "velocityInlet"}};

std::string export_benchmark(const GmshWriteOptions& o, bool named) {
    return write_gmsh(gmsh_model_from_mesh(legacy_mesh(), named ? kBenchmarkNames : std::map<int, std::string>{}, 1), o);
}

void check_identical(const Mesh& a, const Mesh& b) {
    REQUIRE(a.npoin == b.npoin);
    REQUIRE(a.nelem == b.nelem);
    REQUIRE(a.nboun == b.nboun);
    CHECK(a.intma == b.intma);
    CHECK(a.x == b.x); // bit for bit
    CHECK(a.y == b.y);
    int differing = 0;
    for (int i = 0; i < a.nboun; ++i) {
        const BoundarySide &s = a.iside[static_cast<std::size_t>(i)], &t = b.iside[static_cast<std::size_t>(i)];
        if (s.node1 != t.node1 || s.node2 != t.node2 || s.elem != t.elem || s.side_id != t.side_id || s.flag != t.flag)
            ++differing;
    }
    CHECK(differing == 0);
}

std::vector<GmshWriteOptions> all_variants() {
    std::vector<GmshWriteOptions> v;
    for (const std::string version : {"2.2", "4.1"})
        for (const bool binary : {false, true}) v.push_back({version, binary, false, false, 8, ""});
    v.push_back({"4.1", true, true, false, 8, ""});  // foreign byte order
    v.push_back({"2.2", true, true, false, 8, ""});
    v.push_back({"4.1", true, false, false, 4, ""}); // 4-byte size_t
    v.push_back({"2.2", false, false, true, 8, ""}); // CRLF
    return v;
}

// Replaces the legacy mesh of a prepared case by a Gmsh file.
void use_gmsh_mesh(const fs::path& dir, const std::string& contents, bool keep_bofl) {
    fs::remove(dir / "constant/hinterkante.mesh");
    if (!keep_bofl) fs::remove(dir / "constant/hinterkante.bofl");
    write_text(dir / "constant/hinterkante.msh", contents);
}

} // namespace

// =============================================================================
// Legacy benchmark mesh through Gmsh files
// =============================================================================

TEST_CASE("The benchmark mesh exported to Gmsh imports identically (names)", "[mesh_import][hinterkante]") {
    for (const GmshWriteOptions& o : all_variants()) {
        CAPTURE(o.version, o.binary, o.swap_bytes, o.size_t_bytes, o.crlf);
        mesh::ImportReport r;
        const Mesh m = mesh::build_mesh(mesh::parse_gmsh(export_benchmark(o, true), "hinterkante.msh"), {}, &r);
        check_identical(m, legacy_mesh());
        CHECK(r.reoriented_triangles == 0);
        CHECK(r.warnings.empty());
        CHECK(r.patches.size() == 8);
    }
}

TEST_CASE("The benchmark mesh exported to Gmsh imports identically (unnamed + *.bofl)", "[mesh_import][hinterkante][bofl]") {
    mesh::BuildOptions o;
    REQUIRE(IO::read_boundary_flag_table(kCaseDir + "/constant/hinterkante.bofl", o.flag_overrides));
    CHECK(o.flag_overrides.size() == 8);
    for (const GmshWriteOptions& w : all_variants()) {
        CAPTURE(w.version, w.binary);
        const Mesh m = mesh::build_mesh(mesh::parse_gmsh(export_benchmark(w, false), "hinterkante.msh"), o);
        check_identical(m, legacy_mesh());
    }
}

TEST_CASE("Solver: a Gmsh case reproduces the legacy case bit for bit", "[mesh_import][hinterkante][solver]") {
    CaseOptions opts;
    opts.steps = 10;
    const Run legacy = run_case(opts, "gmsh-legacy");
    REQUIRE(legacy.initialized);
    REQUIRE(legacy.state_read);

    struct Variant {
        GmshWriteOptions write;
        bool named;
        bool keep_bofl;
    };
    const Variant variants[] = {
        {{"4.1", true, false, false, 8, ""}, true, false},  // names only
        {{"2.2", false, false, false, 8, ""}, false, true}, // unnamed groups + *.bofl
        {{"4.1", false, false, false, 8, ""}, true, true},  // names and *.bofl (agree)
    };
    for (const Variant& v : variants) {
        CAPTURE(v.write.version, v.write.binary, v.named, v.keep_bofl);
        TempDir dir("gmsh-case");
        prepare_case(dir.path(), opts);
        use_gmsh_mesh(dir.path(), export_benchmark(v.write, v.named), v.keep_bofl);
        const Run run = run_solver(dir.path());
        INFO(run.errors);
        REQUIRE(run.initialized);
        REQUIRE(run.state_read);
        CHECK(run.u == legacy.u);
        CHECK(run.v == legacy.v);
        CHECK(run.p == legacy.p);
        CHECK(run.rtime == legacy.rtime);
        REQUIRE(run.steps.size() == legacy.steps.size());
        for (std::size_t k = 0; k < run.steps.size(); ++k) {
            CHECK(run.steps[k].iterations == legacy.steps[k].iterations);
            CHECK(run.steps[k].final == legacy.steps[k].final);
        }
        CHECK_THAT(run.log, ContainsSubstring("Mesh format: Gmsh MSH " + v.write.version));
        CHECK_THAT(run.log, ContainsSubstring("Wall Nodes Detected: "));
        if (v.keep_bofl) CHECK_THAT(run.log, ContainsSubstring("Boundary flag overrides:"));
    }
}

// =============================================================================
// Case files
// =============================================================================

TEST_CASE("locate_mesh_file: legacy, Gmsh, ambiguous, missing", "[mesh_import][case]") {
    TempDir dir("locate");
    const std::string c = dir.path().string();
    MeshFileLocation loc;
    {
        CaptureStderr err;
        CHECK_FALSE(IO::locate_mesh_file(c, "x", loc));
        CHECK_THAT(err.str(), ContainsSubstring("MESH FILE DOES NOT EXIST") && ContainsSubstring("x.msh"));
    }
    write_text(dir / "x.mesh", "");
    REQUIRE(IO::locate_mesh_file(c, "x", loc));
    CHECK(loc.format == MeshFileFormat::Legacy);
    CHECK(loc.path == (dir / "x.mesh").string());

    write_text(dir / "x.msh", "");
    {
        CaptureStderr err;
        CHECK_FALSE(IO::locate_mesh_file(c, "x", loc));
        CHECK_THAT(err.str(), ContainsSubstring("AMBIGUOUS MESH"));
    }
    fs::remove(dir / "x.mesh");
    REQUIRE(IO::locate_mesh_file(c, "x", loc));
    CHECK(loc.format == MeshFileFormat::Gmsh);

    fs::create_directory(dir / "y.msh"); // a directory is not a mesh file
    CaptureStderr err;
    CHECK_FALSE(IO::locate_mesh_file(c, "y", loc));
}

TEST_CASE("Solver: errors of Gmsh cases stop the initialisation", "[mesh_import][case][solver]") {
    CaseOptions opts;
    opts.steps = 1;

    SECTION("both mesh files") {
        TempDir dir("gmsh-both");
        prepare_case(dir.path(), opts);
        write_text(dir / "constant/hinterkante.msh", export_benchmark({}, true));
        const Run run = run_solver(dir.path());
        CHECK_FALSE(run.initialized);
        CHECK_THAT(run.errors, ContainsSubstring("AMBIGUOUS MESH"));
    }
    SECTION("unnamed groups without *.bofl") {
        TempDir dir("gmsh-unnamed");
        prepare_case(dir.path(), opts);
        use_gmsh_mesh(dir.path(), export_benchmark({}, false), false);
        const Run run = run_solver(dir.path());
        CHECK_FALSE(run.initialized);
        CHECK_THAT(run.errors, ContainsSubstring("invalid mesh (8 problems)") &&
                                   ContainsSubstring("physical group 1 (unnamed)"));
    }
    SECTION("invalid *.bofl") {
        TempDir dir("gmsh-badbofl");
        prepare_case(dir.path(), opts);
        use_gmsh_mesh(dir.path(), export_benchmark({}, true), true);
        write_text(dir / "constant/hinterkante.bofl", "2\n1 500\n");
        const Run run = run_solver(dir.path());
        CHECK_FALSE(run.initialized);
        CHECK_THAT(run.errors, ContainsSubstring("ENTRY 2 OF 2"));
    }
    SECTION("unimplemented boundary type") {
        TempDir dir("gmsh-farfield");
        prepare_case(dir.path(), opts);
        auto names = kBenchmarkNames;
        names[5] = "farfield_top";
        use_gmsh_mesh(dir.path(), write_gmsh(gmsh_model_from_mesh(legacy_mesh(), names, 1)), false);
        const Run run = run_solver(dir.path());
        CHECK_FALSE(run.initialized);
        CHECK_THAT(run.errors, ContainsSubstring("'farfield_top' (tag 5) is a 'farfield' boundary"));
    }
    SECTION("corrupt file") {
        TempDir dir("gmsh-corrupt");
        prepare_case(dir.path(), opts);
        std::string text = export_benchmark({"4.1", true, false, false, 8, ""}, true);
        text.resize(text.size() / 2);
        use_gmsh_mesh(dir.path(), text, false);
        const Run run = run_solver(dir.path());
        CHECK_FALSE(run.initialized);
        CHECK_THAT(run.errors, ContainsSubstring("hinterkante.msh: byte"));
    }
}

TEST_CASE("read_boundary_flag_table is strict", "[mesh_import][bofl]") {
    TempDir dir("bofl");
    std::map<int, int> t;
    auto read = [&](const std::string& text) {
        write_text(dir / "f.bofl", text);
        CaptureStderr quiet;
        return IO::read_boundary_flag_table((dir / "f.bofl").string(), t);
    };
    REQUIRE(read("3\n1 500\n 7 503\n\n12 504 trailing text is ignored\n"));
    CHECK(t == std::map<int, int>{{1, 500}, {7, 503}, {12, 504}});
    REQUIRE(read("0\n"));
    CHECK(t.empty());
    CHECK_FALSE(read(""));
    CHECK_FALSE(read("-1\n"));
    CHECK_FALSE(read("two\n1 500\n2 504\n"));
    CHECK_FALSE(read("2\n1 500\n"));
    CHECK_FALSE(read("2\n1 500\n2 5x4\n"));
    CHECK_FALSE(read("2\n1 500\n1 504\n"));
    CaptureStderr quiet;
    CHECK_FALSE(IO::read_boundary_flag_table((dir / "missing.bofl").string(), t));
}

TEST_CASE("read_gmsh_mesh prints the import summary", "[mesh_import][io]") {
    TempDir dir("gmsh-io");
    write_text(dir / "m.msh", export_benchmark({"2.2", true, false, false, 8, ""}, true));
    Mesh m;
    CaptureStdout out;
    REQUIRE(IO::read_gmsh_mesh((dir / "m.msh").string(), (dir / "none.bofl").string(), m));
    CHECK(m.nelem == 17377);
    const std::string log = out.str();
    CHECK_THAT(log, ContainsSubstring("Mesh format: Gmsh MSH 2.2 binary"));
    CHECK_THAT(log, ContainsSubstring("Mesh import: 17377 triangles, 8956 nodes, 533 boundary sides"));
    CHECK_THAT(log, ContainsSubstring("Boundary 'velocityInlet' (tag 8): inlet, flag 503, 51 sides [name]"));
    CHECK_THAT(log, !ContainsSubstring("Boundary flag overrides"));
}

// =============================================================================
// A mesh written by Gmsh in the solver
// =============================================================================

TEST_CASE("Solver: the Gmsh channel-with-cylinder fixture runs with its boundary conditions", "[mesh_import][solver][fixtures]") {
    TempDir dir("gmsh-channel");
    CaseOptions opts;
    opts.steps = 20;
    prepare_case(dir.path(), opts);
    fs::remove(dir / "constant/hinterkante.mesh");
    fs::remove(dir / "constant/hinterkante.bofl");
    fs::copy_file(std::string(HYDRAMA2D_TEST_DATA_DIR) + "/gmsh/channel_cylinder_4.1_binary.msh",
                  dir / "constant/hinterkante.msh");

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
    CHECK_THAT(run.log, ContainsSubstring("Boundary 'Symmetry Plane' (tag 3): symmetry, flag 506, 32 sides [name]"));

    Mesh m;
    {
        CaptureStdout quiet;
        REQUIRE(IO::read_gmsh_mesh((dir / "constant/hinterkante.msh").string(), "", m));
    }
    std::vector<double> u(m.npoin), v(m.npoin), p(m.npoin);
    double rtime = 0.0;
    int istart = 0;
    REQUIRE(IO::read_restart((dir / "results/hinterkante.var").string(), rtime, istart, u, v, p));
    CHECK(istart == 20);
    CHECK(count_non_finite(u) + count_non_finite(v) + count_non_finite(p) == 0);

    m.compute_face_normals();
    m.determine_wall_nodes();
    for (int ib = 0; ib < m.nboun; ++ib) {
        const BoundarySide& s = m.iside[static_cast<std::size_t>(ib)];
        for (int ip : {s.node1, s.node2}) {
            CAPTURE(ib, s.flag, ip, m.x[ip], m.y[ip]);
            if (s.flag == 503) {
                CHECK(u[ip] == 1.0);
                CHECK(v[ip] == 0.0);
            } else if (s.flag == 500) {
                CHECK(u[ip] == 0.0);
                CHECK(v[ip] == 0.0);
            } else if (s.flag == 506) {
                CHECK(std::abs(v[ip]) < 1e-15); // u . n = 0 on y = 0 and y = 1
            } else if (s.flag == 504) {
                CHECK(p[ip] == 0.0);
            }
        }
    }
    // The flow is deflected by the cylinder: non-zero cross-flow inside.
    CHECK(max_abs(v) > 1e-3);
    CHECK(max_abs(u) < 3.0);
}
