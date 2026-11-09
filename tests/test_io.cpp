// File input/output (IO.cpp):
//   parse_fortran_double           Fortran-style reals ("1.0d-05")
//   read_control/physical/solver   positional config readers: values are taken
//                                  from fixed NON-EMPTY line indices
//   read_mesh, read_boundary_flags mesh topology/coordinates (1-based ids in
//                                  the file, 0-based in memory), flag table
//   write/read_restart             checkpoint files (*.var)
//   write_vtk                      legacy ASCII VTK output for ParaView

#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

const fs::path kCase = HYDRAMA2D_CASE_DIR;

// Bundled control file: value line index -> field
//   2 restart_opt, 5 ntime, 8 transient_on, 11 time_output_interval,
//   14 end_rtime, 17 write_output, 20 write_time_output, 23 dtfixed,
//   26 dtfix, 29 iwrite, 32 csafm, 35 theta[0], 38 vel_check,
//   41 l2norm_vel_tolerance, 44 runtime_mod
fs::path copy_config(const TempDir& dir, const std::string& name) {
    const fs::path dst = dir / name;
    fs::copy_file(kCase / "system" / name, dst);
    return dst;
}

// Two-element unit square; ids deliberately out of order in the file.
//   element 1 = nodes (1,2,3), element 2 = nodes (2,4,3)
//   node 1 (0,0), 2 (1,0), 3 (0,1), 4 (1,1)
//   sides: bottom 1-2, right 2-4, top 4-3, left 3-1
const char* kSquareMesh =
    "2 4 4\n"
    "2 2 4 3\n"
    "1 1 2 3\n"
    "3 0.0 1.0\n"
    "1 0.0 0.0\n"
    "4 1.0 1.0\n"
    "2 1.0 0.0\n"
    "1 2 1 1\n"
    "2 4 2 2\n"
    "4 3 2 3\n"
    "3 1 1 4\n";

const char* kSquareFlags = "4\n1 500\n2 504\n3 500\n4 503\n";

Mesh read_square(const TempDir& dir, const std::string& text = kSquareMesh) {
    write_text(dir / "sq.mesh", text);
    Mesh m;
    REQUIRE(IO::read_mesh((dir / "sq.mesh").string(), m));
    return m;
}

// max |a - b| / |a| bound for decimal output with `digits` significant digits
double decimal_rel_error(int digits) { return 0.5 * std::pow(10.0, 1 - digits); }

} // namespace

// =============================================================================
// parse_fortran_double
// =============================================================================

TEST_CASE("parse_fortran_double accepts Fortran and C exponents", "[io][parse]") {
    struct Case { const char* text; double value; };
    const Case c = GENERATE(Case{"1.0d-05", 1.0e-5}, Case{"1.0D-05", 1.0e-5}, Case{"0.6d00", 0.6},
                            Case{"200.0d00", 200.0}, Case{"1d3", 1000.0}, Case{"-2.5d-1", -0.25},
                            Case{".5d1", 5.0}, Case{"1.0e-3", 1.0e-3}, Case{"1.0E+2", 100.0},
                            Case{"3", 3.0}, Case{"-0.0", 0.0}, Case{"  7.5d0", 7.5});
    CAPTURE(c.text);
    // Bit-exact: identical to parsing the same decimal with a C exponent.
    CHECK(IO::parse_fortran_double(c.text) == c.value);
}

TEST_CASE("parse_fortran_double rejects non-numbers", "[io][parse]") {
    CHECK_THROWS_AS(IO::parse_fortran_double(""), std::invalid_argument);
    CHECK_THROWS_AS(IO::parse_fortran_double("abc"), std::invalid_argument);
    CHECK_THROWS_AS(IO::parse_fortran_double("d5"), std::invalid_argument);
    CHECK_THROWS_AS(IO::parse_fortran_double("1d999"), std::out_of_range);
}

TEST_CASE("parse_fortran_double silently ignores trailing text", "[io][parse][characterization]") {
    // Documents current behaviour: std::stod parses the longest valid prefix,
    // so typos after a number are not reported.
    CHECK(IO::parse_fortran_double("1.5d0xyz") == 1.5);
    CHECK(IO::parse_fortran_double("1.0dd5") == 1.0); // "1.0ee5" -> 1.0
    CHECK(IO::parse_fortran_double("0.6 ! safety factor") == 0.6);
}

// =============================================================================
// Configuration readers
// =============================================================================

TEST_CASE("read_control reads the bundled control file", "[io][config]") {
    ControlConfig c;
    REQUIRE(IO::read_control((kCase / "system/hinterkante.control").string(), c));
    CHECK(c.restart_opt == 0);
    CHECK(c.ntime == 20000);
    CHECK(c.transient_on == 1);
    CHECK(c.time_output_interval == 1.0);
    CHECK(c.end_rtime == 500.0);
    CHECK(c.write_output == 100000);
    CHECK(c.write_time_output == 3600.0);
    CHECK(c.dtfixed == -1);
    CHECK(c.dtfix == 1.0e-5);
    CHECK(c.iwrite == 100);
    CHECK(c.csafm == 0.6);
    CHECK(c.theta[0] == 1.0);
    CHECK(c.theta[1] == 1.0);
    CHECK(c.vel_check == 1);
    CHECK(c.l2norm_vel_tolerance == 1.0e-6);
    CHECK(c.runtime_mod == 0);
}

TEST_CASE("read_control maps every value line to its field", "[io][config]") {
    // Distinct values on every value line catch swapped line indices.
    TempDir dir("control");
    const fs::path f = copy_config(dir, "hinterkante.control");
    set_nonempty_lines(f, {{2, "1"}, {5, "123"}, {8, "0"}, {11, "2.5d0"}, {14, "42.0"},
                           {17, "7"}, {20, "99.5"}, {23, "0"}, {26, "3.0d-4"}, {29, "11"},
                           {32, "0.45"}, {35, "0.75d0"}, {38, "0"}, {41, "2.0d-8"}, {44, "1"}});
    ControlConfig c;
    c.theta[1] = -1.0;
    REQUIRE(IO::read_control(f.string(), c));
    CHECK(c.restart_opt == 1);
    CHECK(c.ntime == 123);
    CHECK(c.transient_on == 0);
    CHECK(c.time_output_interval == 2.5);
    CHECK(c.end_rtime == 42.0);
    CHECK(c.write_output == 7);
    CHECK(c.write_time_output == 99.5);
    CHECK(c.dtfixed == 0);
    CHECK(c.dtfix == 3.0e-4);
    CHECK(c.iwrite == 11);
    CHECK(c.csafm == 0.45);
    CHECK(c.theta[0] == 0.75);
    CHECK(c.theta[1] == 1.0); // always 1 (semi-implicit), not read from file
    CHECK(c.vel_check == 0);
    CHECK(c.l2norm_vel_tolerance == 2.0e-8);
    CHECK(c.runtime_mod == 1);
}

TEST_CASE("Config readers tolerate blank lines, indentation and line endings", "[io][config]") {
    // The bundled files use CRLF; LF files, extra blank lines and leading or
    // trailing whitespace must give identical results.
    TempDir dir("control-format");
    const std::string original = read_text(kCase / "system/hinterkante.control");
    std::string lf, messy;
    for (const auto& raw : split_lines(original)) {
        std::string line = raw;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lf += line + "\n";
        messy += "\n   \t\n  " + line + "  \t\r\n";
    }
    ControlConfig ref, a, b;
    REQUIRE(IO::read_control((kCase / "system/hinterkante.control").string(), ref));
    write_text(dir / "lf.control", lf);
    write_text(dir / "messy.control", messy);
    REQUIRE(IO::read_control((dir / "lf.control").string(), a));
    REQUIRE(IO::read_control((dir / "messy.control").string(), b));
    for (const ControlConfig* c : {&a, &b}) {
        CHECK(c->ntime == ref.ntime);
        CHECK(c->dtfix == ref.dtfix);
        CHECK(c->csafm == ref.csafm);
        CHECK(c->l2norm_vel_tolerance == ref.l2norm_vel_tolerance);
        CHECK(c->runtime_mod == ref.runtime_mod);
    }
}

TEST_CASE("read_physical and read_solver read the bundled files", "[io][config]") {
    PhysicalConfig p;
    REQUIRE(IO::read_physical((kCase / "system/hinterkante.physical").string(), p));
    CHECK(p.re == 200.0);
    CHECK(p.pnode == 0); // 1-based "1" in the file

    SolverConfig s;
    REQUIRE(IO::read_solver((kCase / "system/hinterkante.solver").string(), s));
    CHECK(s.solver_opt == 1);
    CHECK(s.relToler == 0.1);
    CHECK(s.absToler == 1.0e-6);
}

TEST_CASE("read_physical and read_solver map their value lines", "[io][config]") {
    TempDir dir("physical");
    SECTION("physical: Re and 1-based reference node") {
        const fs::path f = copy_config(dir, "hinterkante.physical");
        set_nonempty_lines(f, {{2, "1.5d3"}, {5, "17"}});
        PhysicalConfig p;
        REQUIRE(IO::read_physical(f.string(), p));
        CHECK(p.re == 1500.0);
        CHECK(p.pnode == 16);
    }
    SECTION("physical: pnode 0 becomes -1 (falls back to node 0 later)") {
        const fs::path f = copy_config(dir, "hinterkante.physical");
        set_nonempty_lines(f, {{5, "0"}});
        PhysicalConfig p;
        REQUIRE(IO::read_physical(f.string(), p));
        CHECK(p.pnode == -1);
    }
    SECTION("solver: option and tolerances") {
        const fs::path f = copy_config(dir, "hinterkante.solver");
        set_nonempty_lines(f, {{2, "2"}, {6, "0.0"}, {9, "1.0d-12"}});
        SolverConfig s;
        REQUIRE(IO::read_solver(f.string(), s));
        CHECK(s.solver_opt == 2);
        CHECK(s.relToler == 0.0);
        CHECK(s.absToler == 1.0e-12);
    }
}

TEST_CASE("Config readers reject missing and truncated files", "[io][config]") {
    CaptureStderr quiet; // expected error messages
    TempDir dir("truncated");
    auto truncated = [&](const std::string& name, int keep_nonempty) {
        std::string out;
        int k = 0;
        for (const auto& line : split_lines(read_text(kCase / "system" / name))) {
            if (!is_blank(line) && k++ >= keep_nonempty) break;
            out += line + "\n";
        }
        write_text(dir / name, out);
        return (dir / name).string();
    };
    ControlConfig c;
    PhysicalConfig p;
    SolverConfig s;
    CHECK_FALSE(IO::read_control((dir / "missing.control").string(), c));
    CHECK_FALSE(IO::read_physical((dir / "missing.physical").string(), p));
    CHECK_FALSE(IO::read_solver((dir / "missing.solver").string(), s));
    CHECK_FALSE(IO::read_control(truncated("hinterkante.control", 44), c)); // needs 45
    CHECK_FALSE(IO::read_physical(truncated("hinterkante.physical", 5), p)); // needs 6
    CHECK_FALSE(IO::read_solver(truncated("hinterkante.solver", 9), s));     // needs 10
    CHECK(IO::read_control(truncated("hinterkante.control", 45), c));
}

TEST_CASE("Integer config fields are parsed leniently", "[io][config][characterization]") {
    // Documents current behaviour: integer fields use std::stoi, which stops
    // at the first non-digit. A step count written as "1e4" runs ONE step,
    // "20000.0" is accepted as 20000; nothing is reported.
    TempDir dir("lenient");
    const fs::path f = copy_config(dir, "hinterkante.control");
    ControlConfig c;

    set_nonempty_lines(f, {{5, "1e4"}});
    REQUIRE(IO::read_control(f.string(), c));
    CHECK(c.ntime == 1);

    set_nonempty_lines(f, {{5, "20000.0"}});
    REQUIRE(IO::read_control(f.string(), c));
    CHECK(c.ntime == 20000);
}

TEST_CASE("Malformed config values throw instead of returning false", "[io][config][characterization]") {
    // Documents current behaviour: the readers only return false for missing
    // or short files. A non-numeric value escapes as an exception, which
    // Solver::initialize() does not catch (the program terminates).
    TempDir dir("malformed");
    const fs::path control = copy_config(dir, "hinterkante.control");
    set_nonempty_lines(control, {{5, "many"}});
    ControlConfig c;
    CHECK_THROWS_AS(IO::read_control(control.string(), c), std::invalid_argument);

    const fs::path solver = copy_config(dir, "hinterkante.solver");
    set_nonempty_lines(solver, {{6, "tight"}});
    SolverConfig s;
    CHECK_THROWS_AS(IO::read_solver(solver.string(), s), std::invalid_argument);
}

// =============================================================================
// Mesh and boundary flags
// =============================================================================

TEST_CASE("read_mesh reads the bundled benchmark mesh", "[io][mesh]") {
    Mesh m;
    REQUIRE(IO::read_mesh((kCase / "constant/hinterkante.mesh").string(), m));
    CHECK(m.nelem == 17377);
    CHECK(m.npoin == 8956);
    CHECK(m.nboun == 533);
    CHECK(m.intma.size() == 17377);
    CHECK(m.x.size() == 8956);
    CHECK(m.iside.size() == 533);

    // Spot checks against the file (1-based there, 0-based here).
    CHECK(m.intma[0] == std::array<int, 3>{50, 51, 52});   // "1 51 52 53"
    CHECK(m.intma[5] == std::array<int, 3>{532, 0, 1});    // "6 533 1 2"
    CHECK(m.x[0] == -4.0);                                  // "1 -4.0 -5.0"
    CHECK(m.y[0] == -5.0);
    CHECK(m.x[1] == -3.6862745299999999);                   // "2 -3.68627452999..."
    CHECK(m.iside[0].node1 == 50);                          // "51 52 1 1"
    CHECK(m.iside[0].node2 == 51);
    CHECK(m.iside[0].elem == 0);
    CHECK(m.iside[0].side_id == 1);
    CHECK(m.iside[532].elem == 2077);                       // "50 51 2078 1"
    for (const auto& s : m.iside) CHECK(s.flag == 0);       // set by read_boundary_flags

    int bad = 0;
    for (const auto& e : m.intma)
        for (int n : e)
            if (n < 0 || n >= m.npoin) ++bad;
    CHECK(bad == 0);
}

TEST_CASE("read_mesh places entries by id and converts to 0-based", "[io][mesh]") {
    TempDir dir("mesh");
    Mesh m = read_square(dir);
    REQUIRE(m.nelem == 2);
    REQUIRE(m.npoin == 4);
    REQUIRE(m.nboun == 4);
    CHECK(m.intma[0] == std::array<int, 3>{0, 1, 2});
    CHECK(m.intma[1] == std::array<int, 3>{1, 3, 2});
    CHECK(m.x == std::vector<double>{0.0, 1.0, 0.0, 1.0});
    CHECK(m.y == std::vector<double>{0.0, 0.0, 1.0, 1.0});
    CHECK(m.iside[1].node1 == 1);
    CHECK(m.iside[1].node2 == 3);
    CHECK(m.iside[1].elem == 1);
    CHECK(m.iside[1].side_id == 2);

    // The loaded mesh is geometrically valid: CCW unit square, outward normals.
    compute_geometry(m);
    CHECK(m.detJ[0] > 0.0);
    CHECK(m.detJ[1] > 0.0);
    CHECK_THAT(0.5 * (m.detJ[0] + m.detJ[1]), WithinRel(1.0, 1e-15));
    CHECK(m.face_norm[1].nx == 1.0); // right side
}

TEST_CASE("read_mesh does not depend on line layout", "[io][mesh]") {
    TempDir dir("mesh-layout");
    std::string one_line = kSquareMesh;
    for (char& ch : one_line)
        if (ch == '\n') ch = ' ';
    std::string crlf;
    for (const auto& line : split_lines(kSquareMesh)) crlf += "  " + line + "\r\n";

    const Mesh ref = read_square(dir);
    for (const std::string& text : {one_line, crlf}) {
        const Mesh m = read_square(dir, text);
        CHECK(m.intma == ref.intma);
        CHECK(m.x == ref.x);
        CHECK(m.y == ref.y);
        CHECK(m.iside.size() == ref.iside.size());
    }
}

TEST_CASE("read_mesh rejects missing and truncated files", "[io][mesh]") {
    CaptureStderr quiet; // expected error messages
    TempDir dir("mesh-bad");
    Mesh m;
    CHECK_FALSE(IO::read_mesh((dir / "missing.mesh").string(), m));

    const auto lines = split_lines(kSquareMesh);
    // Cut the file inside the header, elements, nodes and boundary sides; the
    // error message must name the section that is incomplete.
    struct Cut { int keep; const char* message; };
    const Cut c = GENERATE(Cut{0, "MESH DIMENSIONS"}, Cut{2, "ELEMENT CONNECTIVITY"},
                           Cut{4, "NODAL COORDINATE"}, Cut{9, "BOUNDARY SIDE"});
    CAPTURE(c.keep, c.message);
    std::string text;
    for (int i = 0; i < c.keep; ++i) text += lines[i] + "\n";
    if (c.keep == 0) text = "2 4\n";
    write_text(dir / "cut.mesh", text);
    Mesh cut;
    std::string err;
    {
        CaptureStderr capture;
        CHECK_FALSE(IO::read_mesh((dir / "cut.mesh").string(), cut));
        err = capture.str();
    }
    CHECK(err.find(c.message) != std::string::npos);
}

TEST_CASE("read_mesh rejects truncated nodes when there are no boundary sides", "[io][mesh]") {
    // Without a boundary section nothing after the nodes could catch the error.
    TempDir dir("mesh-noboun");
    write_text(dir / "nb.mesh", "1 3 0\n1 1 2 3\n1 0.0 0.0\n2 1.0 0.0\n");
    Mesh m;
    CaptureStderr quiet;
    CHECK_FALSE(IO::read_mesh((dir / "nb.mesh").string(), m));
}

TEST_CASE("read_mesh rejects Fortran-style exponents in coordinates", "[io][mesh]") {
    CaptureStderr quiet; // expected error messages
    // Unlike the config files, mesh coordinates must use C exponents.
    TempDir dir("mesh-fortran");
    std::string text = kSquareMesh;
    text.replace(text.find("3 0.0 1.0"), 9, "3 0.0d0 1.0");
    write_text(dir / "f.mesh", text);
    Mesh m;
    CHECK_FALSE(IO::read_mesh((dir / "f.mesh").string(), m));
}

TEST_CASE("read_mesh rejects invalid ids with a clear message", "[io][mesh]") {
    // Each case replaces one line of the 2-element square mesh. Before the
    // validation was added, most of these wrote out of bounds or left entries
    // unassigned; now they must fail with a message naming the problem.
    //   line 0: "2 4 4"      header (elements, nodes, sides)
    //   line 1: "2 2 4 3"    element 2        line 3: "3 0.0 1.0"  node 3
    //   line 2: "1 1 2 3"    element 1        line 4: "1 0.0 0.0"  node 1
    //   line 7: "1 2 1 1"    side 1           line 8: "2 4 2 2"    side 2
    //   line 9: "4 3 2 3"    side 3
    struct Case {
        const char* what;
        int line;
        const char* replacement;
        const char* message;
    };
    const Case c = GENERATE(
        Case{"no elements", 0, "0 4 4", "INVALID MESH DIMENSIONS"},
        Case{"negative node count", 0, "2 -4 4", "INVALID MESH DIMENSIONS"},
        Case{"negative side count", 0, "2 4 -1", "INVALID MESH DIMENSIONS"},
        Case{"element id 0", 1, "0 2 4 3", "ELEMENT ID 0 OUT OF RANGE 1..2"},
        Case{"element id above count", 1, "3 2 4 3", "ELEMENT ID 3 OUT OF RANGE 1..2"},
        Case{"negative element id", 1, "-1 2 4 3", "ELEMENT ID -1 OUT OF RANGE 1..2"},
        Case{"huge element id", 1, "2147483647 2 4 3", "ELEMENT ID 2147483647 OUT OF RANGE"},
        Case{"duplicate element id", 1, "1 2 4 3", "DUPLICATE ELEMENT ID 1"},
        Case{"element node above count", 1, "2 2 5 3", "ELEMENT 2 REFERENCES NODE 5 OUT OF RANGE 1..4"},
        Case{"element node 0", 2, "1 0 2 3", "ELEMENT 1 REFERENCES NODE 0 OUT OF RANGE 1..4"},
        Case{"repeated node in element", 1, "2 2 4 4", "ELEMENT 2 HAS REPEATED NODES 2 4 4"},
        Case{"node id 0", 3, "0 0.0 1.0", "NODE ID 0 OUT OF RANGE 1..4"},
        Case{"node id above count", 3, "5 0.0 1.0", "NODE ID 5 OUT OF RANGE 1..4"},
        Case{"duplicate node id", 3, "1 0.0 1.0", "DUPLICATE NODE ID 1"},
        Case{"side node above count", 8, "2 9 2 2", "BOUNDARY SIDE 2 REFERENCES NODE 9 OUT OF RANGE 1..4"},
        Case{"side node 0", 7, "0 2 1 1", "BOUNDARY SIDE 1 REFERENCES NODE 0 OUT OF RANGE 1..4"},
        Case{"side element above count", 9, "4 3 3 3", "BOUNDARY SIDE 3 REFERENCES ELEMENT 3 OUT OF RANGE 1..2"},
        Case{"side element 0", 9, "4 3 0 3", "BOUNDARY SIDE 3 REFERENCES ELEMENT 0 OUT OF RANGE 1..2"});
    CAPTURE(c.what, c.replacement);

    auto lines = split_lines(kSquareMesh);
    lines[c.line] = c.replacement;
    std::string text;
    for (const auto& l : lines) text += l + "\n";
    TempDir dir("mesh-invalid");
    write_text(dir / "bad.mesh", text);

    Mesh m;
    std::string err;
    {
        CaptureStderr capture;
        CHECK_FALSE(IO::read_mesh((dir / "bad.mesh").string(), m));
        err = capture.str();
    }
    CAPTURE(err);
    CHECK(err.find(c.message) != std::string::npos);
}

TEST_CASE("read_mesh accepts valid meshes in any id order", "[io][mesh]") {
    // The validation must not reject legitimate files: reversed id order and a
    // mesh without boundary sides.
    TempDir dir("mesh-valid");
    SECTION("ids in reverse order") {
        write_text(dir / "r.mesh", "2 4 1\n2 2 4 3\n1 1 2 3\n4 1.0 1.0\n3 0.0 1.0\n2 1.0 0.0\n1 0.0 0.0\n1 2 1 1\n");
        Mesh m;
        REQUIRE(IO::read_mesh((dir / "r.mesh").string(), m));
        CHECK(m.intma[1] == std::array<int, 3>{1, 3, 2});
        CHECK(m.x[3] == 1.0);
    }
    SECTION("no boundary sides") {
        write_text(dir / "n.mesh", "1 3 0\n1 1 2 3\n1 0.0 0.0\n2 1.0 0.0\n3 0.0 1.0\n");
        Mesh m;
        REQUIRE(IO::read_mesh((dir / "n.mesh").string(), m));
        CHECK(m.nboun == 0);
        CHECK(m.iside.empty());
    }
}

TEST_CASE("read_boundary_flags assigns flags by side id", "[io][flags]") {
    TempDir dir("flags");
    Mesh m = read_square(dir);

    SECTION("table in order") {
        write_text(dir / "sq.bofl", kSquareFlags);
    }
    SECTION("table order does not matter") {
        write_text(dir / "sq.bofl", "4\r\n3 500\r\n1 500\r\n4 503\r\n2 504\r\n");
    }
    REQUIRE(IO::read_boundary_flags((dir / "sq.bofl").string(), m));
    CHECK(m.iside[0].flag == 500); // side id 1
    CHECK(m.iside[1].flag == 504); // side id 2
    CHECK(m.iside[2].flag == 500); // side id 3
    CHECK(m.iside[3].flag == 503); // side id 4
}

TEST_CASE("read_boundary_flags on the bundled case", "[io][flags]") {
    Mesh m;
    REQUIRE(IO::read_mesh((kCase / "constant/hinterkante.mesh").string(), m));
    REQUIRE(IO::read_boundary_flags((kCase / "constant/hinterkante.bofl").string(), m));
    // 1,3,5,6,7 -> 500 (51+51+101+76+101 sides), 4,8 -> 503, 2 -> 504
    std::map<int, int> count;
    for (const auto& s : m.iside) count[s.flag]++;
    CHECK(count == std::map<int, int>{{500, 380}, {503, 102}, {504, 51}});
    const std::map<int, int> expected = {{1, 500}, {2, 504}, {3, 500}, {4, 503},
                                         {5, 500}, {6, 500}, {7, 500}, {8, 503}};
    for (const auto& s : m.iside) {
        CAPTURE(s.side_id);
        CHECK(s.flag == expected.at(s.side_id));
    }
}

TEST_CASE("read_boundary_flags error handling", "[io][flags]") {
    CaptureStderr quiet; // expected error messages
    TempDir dir("flags-bad");
    Mesh m = read_square(dir);

    SECTION("missing file") {
        CHECK_FALSE(IO::read_boundary_flags((dir / "missing.bofl").string(), m));
    }
    SECTION("empty file") {
        write_text(dir / "e.bofl", "");
        CHECK_FALSE(IO::read_boundary_flags((dir / "e.bofl").string(), m));
    }
    SECTION("side id missing from the table") {
        write_text(dir / "m.bofl", "3\n1 500\n2 504\n3 500\n");
        CHECK_FALSE(IO::read_boundary_flags((dir / "m.bofl").string(), m));
    }
    SECTION("truncated table") {
        write_text(dir / "t.bofl", "4\n1 500\n2 504\n");
        CHECK_FALSE(IO::read_boundary_flags((dir / "t.bofl").string(), m));
    }
    SECTION("unknown flag code is accepted with a warning") {
        write_text(dir / "u.bofl", "4\n1 510\n2 504\n3 500\n4 503\n");
        std::string log;
        {
            CaptureStdout capture;
            CHECK(IO::read_boundary_flags((dir / "u.bofl").string(), m));
            log = capture.str();
        }
        CHECK(log.find("WARNING") != std::string::npos);
        CHECK(m.iside[0].flag == 510);
    }
    SECTION("duplicate table entries: the first one wins") {
        write_text(dir / "d.bofl", "5\n1 500\n1 501\n2 504\n3 500\n4 503\n");
        REQUIRE(IO::read_boundary_flags((dir / "d.bofl").string(), m));
        CHECK(m.iside[0].flag == 500);
    }
}

// =============================================================================
// Restart files
// =============================================================================

namespace {
struct Fields {
    std::vector<double> u, v, p;
};
Fields sample_fields() {
    return {{1.0, -0.25, 1.0 / 3.0, 0.0, 12345.678901234, -1.0e-7},
            {0.0, 2.0 / 7.0, -9.87654321e-3, 1.0e-300, 4.0, 5.5},
            {0.1, -100.0, std::sqrt(2.0), 6.02214076e23, -0.0, 1.0}};
}

// Bit-level equality (distinguishes -0.0 from 0.0).
bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
} // namespace

TEST_CASE("Restart files round-trip time, step and fields bit-for-bit", "[io][restart]") {
    TempDir dir("restart");
    const Fields f = sample_fields();
    const std::string path = (dir / "case.var").string();
    const double rtime = 12.3456789012345;
    REQUIRE(IO::write_restart(path, rtime, 417, 1, f.u, f.v, f.p));

    // Header line + one line per node.
    CHECK(split_lines(read_text(path)).size() == f.u.size() + 1);

    Fields r{std::vector<double>(6), std::vector<double>(6), std::vector<double>(6)};
    double rt = -1.0;
    int istart = -1;
    REQUIRE(IO::read_restart(path, rt, istart, r.u, r.v, r.p));
    CHECK(istart == 417);
    CHECK(same_bits(rt, rtime));
    for (size_t i = 0; i < 6; ++i) {
        CAPTURE(i, f.u[i], f.v[i], f.p[i]);
        CHECK(same_bits(r.u[i], f.u[i]));
        CHECK(same_bits(r.v[i], f.v[i]));
        CHECK(same_bits(r.p[i], f.p[i]));
    }
}

TEST_CASE("Restart files round-trip arbitrary doubles bit-for-bit", "[io][restart]") {
    // 10000 random finite bit patterns per field: all exponents, signs,
    // subnormals; plus the extremes of the double range.
    std::mt19937_64 rng(20240531);
    auto random_double = [&] {
        for (;;) {
            const std::uint64_t bits = rng();
            double d;
            std::memcpy(&d, &bits, sizeof d);
            if (std::isfinite(d)) return d;
        }
    };
    const int n = 10000;
    Fields f{std::vector<double>(n), std::vector<double>(n), std::vector<double>(n)};
    for (int i = 0; i < n; ++i) {
        f.u[i] = random_double();
        f.v[i] = random_double();
        f.p[i] = random_double();
    }
    const double extremes[] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
                               std::numeric_limits<double>::min(), std::numeric_limits<double>::denorm_min(),
                               -std::numeric_limits<double>::denorm_min(), -0.0, 0.0, 1.0 / 3.0};
    for (size_t k = 0; k < std::size(extremes); ++k) f.u[k] = f.v[n - 1 - k] = f.p[k] = extremes[k];
    const double rtime = random_double();

    TempDir dir("restart-random");
    const std::string path = (dir / "r.var").string();
    REQUIRE(IO::write_restart(path, rtime, 7, 1, f.u, f.v, f.p));
    Fields r{std::vector<double>(n), std::vector<double>(n), std::vector<double>(n)};
    double rt;
    int is;
    REQUIRE(IO::read_restart(path, rt, is, r.u, r.v, r.p));
    CHECK(same_bits(rt, rtime));
    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (!same_bits(r.u[i], f.u[i]) || !same_bits(r.v[i], f.v[i]) || !same_bits(r.p[i], f.p[i])) ++mismatches;
    }
    CHECK(mismatches == 0);
}

TEST_CASE("Restart file format: 17 significant digits, scientific notation", "[io][restart]") {
    TempDir dir("restart-format");
    REQUIRE(IO::write_restart((dir / "t.var").string(), 1.0 / 7.0, 3, 1,
                              {1.0 / 3.0, -0.0}, {2.5, 1.0e-300}, {0.0, -123.456}));
    CHECK(read_text(dir / "t.var") ==
          "1.4285714285714285e-01 3\n"
          "3.3333333333333331e-01 2.5000000000000000e+00 0.0000000000000000e+00\n"
          "-0.0000000000000000e+00 1.0000000000000000e-300 -1.2345600000000000e+02\n");

    SECTION("steady-state files store time 0") {
        REQUIRE(IO::write_restart((dir / "s.var").string(), 99.0, 5, 0, {1.0}, {2.0}, {3.0}));
        CHECK(split_lines(read_text(dir / "s.var"))[0] == "0.0000000000000000e+00 5");
    }
}

TEST_CASE("Restart files written with the former 9-digit format remain readable", "[io][restart]") {
    TempDir dir("restart-legacy");
    write_text(dir / "old.var", "5.131891261e-02 10\n"
                                "1.00000000e+00 0.00000000e+00 2.25360267e+00\n");
    std::vector<double> u(1), v(1), p(1);
    double rt;
    int is;
    REQUIRE(IO::read_restart((dir / "old.var").string(), rt, is, u, v, p));
    CHECK(rt == 5.131891261e-02);
    CHECK(is == 10);
    CHECK(u[0] == 1.0);
    CHECK(p[0] == 2.25360267);
}

TEST_CASE("read_restart error handling", "[io][restart]") {
    TempDir dir("restart-bad");
    std::vector<double> u(3), v(3), p(3);
    double rt;
    int is;
    SECTION("missing file") {
        CHECK_FALSE(IO::read_restart((dir / "none.var").string(), rt, is, u, v, p));
    }
    SECTION("empty file") {
        write_text(dir / "e.var", "");
        CHECK_FALSE(IO::read_restart((dir / "e.var").string(), rt, is, u, v, p));
    }
    SECTION("fewer nodes than the mesh") {
        write_text(dir / "s.var", "1.0 3\n1 2 3\n4 5 6\n");
        CHECK_FALSE(IO::read_restart((dir / "s.var").string(), rt, is, u, v, p));
    }
    SECTION("non-numeric values") {
        write_text(dir / "n.var", "1.0 3\n1 2 3\nnan 5 6\n7 8 9\n");
        CHECK_FALSE(IO::read_restart((dir / "n.var").string(), rt, is, u, v, p));
    }
}

TEST_CASE("read_restart does not detect files from a larger mesh", "[io][restart][characterization]") {
    // Documents current behaviour: the node count is not stored in the file,
    // so a restart file with MORE nodes than the mesh is read without error
    // (the surplus lines are ignored).
    TempDir dir("restart-mismatch");
    write_text(dir / "big.var", "2.0 10\n1 2 3\n4 5 6\n7 8 9\n10 11 12\n");
    std::vector<double> u(3), v(3), p(3);
    double rt;
    int is;
    CHECK(IO::read_restart((dir / "big.var").string(), rt, is, u, v, p));
    CHECK(u == std::vector<double>{1.0, 4.0, 7.0});
}

TEST_CASE("write_restart and write_vtk report unwritable paths", "[io][restart][vtk]") {
    CaptureStderr quiet; // expected error messages
    TempDir dir("unwritable");
    const std::string bad = (dir / "no_such_dir" / "file").string();
    const Fields f = sample_fields();
    CHECK_FALSE(IO::write_restart(bad, 0.0, 1, 1, f.u, f.v, f.p));
    Mesh m = make_triangle_mesh(reference_triangles()[0]);
    CHECK_FALSE(IO::write_vtk(bad, m, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}));
}

// =============================================================================
// VTK output
// =============================================================================

TEST_CASE("VTK output of a single triangle, byte for byte", "[io][vtk]") {
    TempDir dir("vtk-exact");
    Mesh m = make_mesh({{0.0, 0.0}, {2.5, 0.0}, {0.0, -1.0e-3}}, {{0, 1, 2}});
    REQUIRE(IO::write_vtk((dir / "t.vtk").string(), m, {1.0, -0.5, 0.0}, {0.0, 0.25, 3.0}, {0.125, 0.0, -2.0}));
    const std::string expected =
        "# vtk DataFile Version 3.0\n"
        "Instananeous Variables Output\n"
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n"
        "POINTS 3 double\n"
        "0.0000000000e+00 0.0000000000e+00 0.0\n"
        "2.5000000000e+00 0.0000000000e+00 0.0\n"
        "0.0000000000e+00 -1.0000000000e-03 0.0\n"
        "CELLS 1 4\n"
        "3 0 1 2\n"
        "CELL_TYPES 1\n"
        "5\n"
        "POINT_DATA 3\n"
        "VECTORS velocity float\n"
        "1.0000000e+00 0.0000000e+00 0.0\n"
        "-5.0000000e-01 2.5000000e-01 0.0\n"
        "0.0000000e+00 3.0000000e+00 0.0\n"
        "SCALARS P float 1\n"
        "LOOKUP_TABLE default\n"
        "1.2500000e-01\n"
        "0.0000000e+00\n"
        "-2.0000000e+00\n";
    CHECK(read_text(dir / "t.vtk") == expected);
}

TEST_CASE("VTK output round-trips the benchmark mesh and fields", "[io][vtk]") {
    TempDir dir("vtk");
    Mesh m;
    REQUIRE(IO::read_mesh((kCase / "constant/hinterkante.mesh").string(), m));
    const auto u = sample(m, [](double x, double y) { return 1.0 + 0.1 * std::sin(x) * std::cos(y); });
    const auto v = sample(m, [](double x, double y) { return 0.05 * x * y; });
    const auto p = sample(m, [](double x, double y) { return std::exp(-0.1 * (x * x + y * y)) - 0.5; });
    REQUIRE(IO::write_vtk((dir / "h.vtk").string(), m, u, v, p));

    const VtkFile f = parse_vtk(dir / "h.vtk");
    CHECK(f.header[0] == "# vtk DataFile Version 3.0");
    CHECK(f.header[2] == "ASCII");
    CHECK(f.header[3] == "DATASET UNSTRUCTURED_GRID");
    REQUIRE(f.npoints == m.npoin);
    REQUIRE(f.ncells == m.nelem);
    CHECK(f.cells_size == 4 * m.nelem);
    CHECK(f.ntypes == m.nelem);
    CHECK(f.npointdata == m.npoin);
    CHECK(f.vectors_line == "VECTORS velocity float");
    CHECK(f.scalars_line == "SCALARS P float 1");
    CHECK(f.lookup_line == "LOOKUP_TABLE default");

    int bad_points = 0, bad_cells = 0, bad_types = 0, bad_data = 0;
    const double coord_tol = decimal_rel_error(11);       // 1 + 10 digits
    const double float_tol = 1.5 * std::numeric_limits<float>::epsilon();
    auto close = [](double a, double b, double rel) { return std::abs(a - b) <= rel * std::abs(b) + 1e-300; };
    for (int i = 0; i < m.npoin; ++i) {
        if (!close(f.points[i][0], m.x[i], coord_tol) || !close(f.points[i][1], m.y[i], coord_tol) ||
            f.points[i][2] != 0.0)
            ++bad_points;
        if (!close(f.velocity[i][0], u[i], float_tol) || !close(f.velocity[i][1], v[i], float_tol) ||
            f.velocity[i][2] != 0.0 || !close(f.pressure[i], p[i], float_tol))
            ++bad_data;
    }
    for (int e = 0; e < m.nelem; ++e) {
        const auto& c = f.cells[e];
        if (c[0] != 3 || c[1] != m.intma[e][0] || c[2] != m.intma[e][1] || c[3] != m.intma[e][2]) ++bad_cells;
        if (f.cell_types[e] != 5) ++bad_types; // VTK_TRIANGLE
    }
    CHECK(bad_points == 0);
    CHECK(bad_cells == 0);
    CHECK(bad_types == 0);
    CHECK(bad_data == 0);
}
