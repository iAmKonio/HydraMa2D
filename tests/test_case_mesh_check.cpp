// HydraMa2D --check-mesh (CaseMeshCheck.hpp) and IO::read_case_mesh.

#include "CaseMeshCheck.hpp"
#include "MeshCaseTestUtils.hpp"
#include "QuadTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sstream>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;

namespace {

struct CheckResult {
    bool ok = false;
    std::string out, err;
};

CheckResult check(const fs::path& dir, const std::string& casename) {
    CheckResult r;
    CaptureStdout out;
    CaptureStderr err;
    r.ok = check_case_mesh(dir.string(), casename);
    r.out = out.str();
    r.err = err.str();
    return r;
}

} // namespace

TEST_CASE("check_case_mesh: the bundled legacy case", "[check_mesh]") {
    TempDir dir("check-legacy");
    copy_benchmark_case(dir.path());
    const CheckResult r = check(dir.path(), "hinterkante");
    INFO(r.err);
    REQUIRE(r.ok);
    CHECK_THAT(r.out, ContainsSubstring("Mesh check: 17377 triangles, 0 quadrilaterals, 8956 nodes, 533 boundary sides: "
                                        "the solver's invariants hold"));
    CHECK_THAT(r.out, ContainsSubstring("Mesh check: flag 500 (wall): 380 boundary sides"));
    CHECK_THAT(r.out, ContainsSubstring("Mesh check: flag 504 (outlet): 51 boundary sides"));
    CHECK_THAT(r.out, ContainsSubstring("Mesh quality: 17377 triangles: angles 15.7 - 128.1 deg"));
    CHECK_THAT(r.out, ContainsSubstring("Mesh quality: equiangle skewness 14376 excellent, 2975 good, 26 fair, 0 poor, 0 bad"));
    // Nothing is written.
    CHECK((!fs::exists(dir / "results") || fs::is_empty(dir / "results")));
}

TEST_CASE("check_case_mesh: Gmsh cases with quadrilaterals and a parabolic inlet", "[check_mesh]") {
    TempDir dir("check-gmsh");
    fs::create_directories(dir / "constant");
    write_text(dir / "constant/st.msh", data_file("schaefer_turek_bl_L0.msh"));
    const CheckResult r = check(dir.path(), "st");
    INFO(r.err);
    REQUIRE(r.ok);
    CHECK_THAT(r.out, ContainsSubstring("Mesh import: 951 triangles, 64 quadrilaterals, 585 nodes"));
    CHECK_THAT(r.out, ContainsSubstring("Mesh check: 951 triangles, 64 quadrilaterals, 585 nodes"));
    CHECK_THAT(r.out, ContainsSubstring("Mesh check: flag 508 (inlet_parabolic): "));
    CHECK_THAT(r.out, ContainsSubstring("Mesh quality: 64 quadrilaterals: angles "));
}

TEST_CASE("check_case_mesh: problems are reported", "[check_mesh][errors]") {
    SECTION("no mesh") {
        TempDir dir("check-none");
        fs::create_directories(dir / "constant");
        const CheckResult r = check(dir.path(), "nothing");
        CHECK_FALSE(r.ok);
        CHECK_THAT(r.err, ContainsSubstring("ERROR"));
    }
    SECTION("a closed parabolic inlet") {
        // Every boundary side of a square is a parabolic inlet.
        GridOptions g;
        g.nx = g.ny = 2;
        const Mesh m = make_hybrid_mesh(grid(g), [](double, double) { return 508; });
        TempDir dir("check-closed-inlet");
        fs::create_directories(dir / "constant");
        write_text(dir / "constant/sq.msh", gmsh_file(m));
        const CheckResult r = check(dir.path(), "sq");
        CHECK_FALSE(r.ok);
        CHECK_THAT(r.err, ContainsSubstring("is a closed curve; a parabolic inlet needs two ends"));
    }
    SECTION("a legacy mesh with a clockwise triangle") {
        TempDir dir("check-clockwise");
        copy_benchmark_case(dir.path());
        // Swap two nodes of the first triangle (line 2 of the mesh file).
        std::vector<std::string> lines = split_lines(read_text(dir / "constant/hinterkante.mesh"));
        std::istringstream in(lines[1]);
        int id = 0, a = 0, b = 0, c = 0;
        in >> id >> a >> b >> c;
        lines[1] = std::to_string(id) + " " + std::to_string(a) + " " + std::to_string(c) + " " + std::to_string(b);
        std::string text;
        for (const std::string& l : lines) text += l + "\n";
        write_text(dir / "constant/hinterkante.mesh", text);
        const CheckResult r = check(dir.path(), "hinterkante");
        CHECK_FALSE(r.ok);
        CHECK_THAT(r.err, ContainsSubstring("violates the solver's invariants"));
        CHECK_THAT(r.err, ContainsSubstring("element 0 is not counter-clockwise"));
    }
}
