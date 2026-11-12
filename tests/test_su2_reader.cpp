// SU2 native mesh reader (mesh/Su2Reader.hpp): the format specification
// (sections, element and marker definitions, optional indices, legacy and
// multi-zone headers, comments, layout) and every kind of malformed input.
// Files are written inline or by tests/Su2TestWriter.hpp; the fixtures in
// tests/data/su2/ are tested in test_su2_import.cpp.

#include "IoTestUtils.hpp"
#include "Su2TestWriter.hpp"
#include "mesh/Su2Reader.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>

using namespace hydrama;
using namespace hydrama::mesh;
using hydrama::test::Su2Model;
using hydrama::test::Su2WriteOptions;
using hydrama::test::write_su2;
using Catch::Matchers::ContainsSubstring;

namespace {

// The example of the SU2 documentation, 2D: unit square, 2 triangles.
const std::string kSpecExample =
    "%\n"
    "% Problem dimension\n"
    "%\n"
    "NDIME= 2\n"
    "%\n"
    "% Inner element connectivity\n"
    "%\n"
    "NELEM= 2\n"
    "5 0 1 3 0\n"
    "5 1 2 3 1\n"
    "%\n"
    "% Node coordinates\n"
    "%\n"
    "NPOIN= 4\n"
    "0.0 0.0 0\n"
    "1.0 0.0 1\n"
    "1.0 1.0 2\n"
    "0.0 1.0 3\n"
    "%\n"
    "% Boundary elements\n"
    "%\n"
    "NMARK= 4\n"
    "MARKER_TAG= lower\n"
    "MARKER_ELEMS= 1\n"
    "3 0 1\n"
    "MARKER_TAG= right\n"
    "MARKER_ELEMS= 1\n"
    "3 1 2\n"
    "MARKER_TAG= upper\n"
    "MARKER_ELEMS= 1\n"
    "3 2 3\n"
    "MARKER_TAG= left\n"
    "MARKER_ELEMS= 1\n"
    "3 3 0\n";

MeshData parse(const std::string& text, const std::string& source = "test.su2") { return parse_su2(text, source); }

std::string parse_error(const std::string& text, const std::string& source = "test.su2") {
    try {
        parse_su2(text, source);
    } catch (const MeshImportError& e) {
        return e.what();
    }
    FAIL("expected MeshImportError");
    return {};
}

void check_spec_example(const MeshData& d) {
    CHECK(d.format == "SU2 ASCII");
    CHECK(d.node_tags == std::vector<std::int64_t>{0, 1, 2, 3});
    CHECK(d.x == std::vector<double>{0.0, 1.0, 1.0, 0.0});
    CHECK(d.y == std::vector<double>{0.0, 0.0, 1.0, 1.0});
    CHECK(d.z == std::vector<double>{0.0, 0.0, 0.0, 0.0});
    REQUIRE(d.elements.size() == 6);
    CHECK(d.elements[0].type == ElementType::Tri3);
    CHECK(d.elements[0].nodes == std::array<std::int64_t, 4>{0, 1, 3, 0});
    CHECK(d.elements[0].physical == 0);
    CHECK(d.elements[1].tag == 1);
    CHECK(d.elements[1].nodes[2] == 3);
    const std::vector<std::string> names = {"lower", "right", "upper", "left"};
    for (int m = 0; m < 4; ++m) {
        const RawElement& l = d.elements[static_cast<std::size_t>(2 + m)];
        CAPTURE(m);
        CHECK(l.type == ElementType::Line2);
        CHECK(l.physical == m + 1);
        CHECK(l.tag == m); // position among all marker elements
        CHECK(d.physical_name(1, m + 1) == names[static_cast<std::size_t>(m)]);
    }
    CHECK(d.elements[5].nodes[0] == 3);
    CHECK(d.elements[5].nodes[1] == 0);
    CHECK(d.warnings.empty());
}

// Replaces the first occurrence of `from` (which must exist).
std::string replaced(std::string text, const std::string& from, const std::string& to) {
    const auto pos = text.find(from);
    REQUIRE(pos != std::string::npos);
    return text.replace(pos, from.size(), to);
}

Su2Model two_triangles() {
    Su2Model m;
    m.x = {0.0, 1.0, 1.0, 0.0};
    m.y = {0.0, 0.0, 1.0, 1.0};
    m.elements = {{5, {0, 1, 3}}, {5, {1, 2, 3}}};
    m.markers = {{"lower", {{3, {0, 1}}}}, {"right", {{3, {1, 2}}}}, {"upper", {{3, {2, 3}}}}, {"left", {{3, {3, 0}}}}};
    return m;
}

} // namespace

// =============================================================================
// The format
// =============================================================================

TEST_CASE("SU2: the example of the format specification", "[su2][reader]") {
    check_spec_example(parse(kSpecExample));
}

TEST_CASE("SU2: optional indices, section order and layout do not change the data", "[su2][reader]") {
    for (const bool point_index : {true, false}) {
        for (const bool element_index : {true, false}) {
            for (const bool points_first : {false, true}) {
                for (const bool crlf : {false, true}) {
                    Su2WriteOptions o;
                    o.point_index = point_index;
                    o.element_index = element_index;
                    o.points_first = points_first;
                    o.crlf = crlf;
                    o.comments = crlf; // vary the comments as well
                    CAPTURE(point_index, element_index, points_first, crlf);
                    check_spec_example(parse(write_su2(two_triangles(), o)));
                }
            }
        }
    }
}

TEST_CASE("SU2: blanks, tabs, inline comments and keyword case", "[su2][reader]") {
    std::string text = kSpecExample;
    text = replaced(text, "NDIME= 2", "  ndime =\t2   % inline comment");
    text = replaced(text, "NELEM= 2", "NELEM=2");
    text = replaced(text, "5 0 1 3 0", "\t5   0  1\t3   0   % first triangle");
    text = replaced(text, "1.0 0.0 1", "  +1.0e0\t0.0   1  ");
    text = replaced(text, "MARKER_TAG= lower", "  Marker_Tag =   lower   ");
    text = replaced(text, "MARKER_ELEMS= 1\n3 0 1", "MARKER_ELEMS=  1 % one line\n\n\n3 0 1 0");
    check_spec_example(parse(text));
}

TEST_CASE("SU2: multi-zone header with one zone and NPOIN with domain count", "[su2][reader]") {
    Su2WriteOptions o;
    o.prefix = "NZONE= 1\n\nIZONE= 1\n";
    o.npoin_domain = true;
    check_spec_example(parse(write_su2(two_triangles(), o)));
    // Fewer domain points than points (partitioned file with halo points).
    check_spec_example(parse(replaced(kSpecExample, "NPOIN= 4", "NPOIN= 4 3")));
}

TEST_CASE("SU2: FFD definitions at the end are ignored", "[su2][reader]") {
    const std::string ffd = "FFD_NBOX= 1\nFFD_NLEVEL= 1\nFFD_TAG= box\nFFD_LEVEL= 0\nFFD_DEGREE_I= 2\n"
                            "FFD_CORNER_POINTS= 4\n0 0\n1 0\n1 1\n0 1\nFFD_CONTROL_POINTS= 0\nFFD_SURFACE_POINTS= 0\n";
    check_spec_example(parse(kSpecExample + ffd));
    check_spec_example(parse(kSpecExample + "FFD_NBOX= 0\nFFD_NLEVEL= 0\n"));
}

TEST_CASE("SU2: legacy NPERIODIC blocks", "[su2][reader]") {
    const std::string identity = "NPERIODIC= 1\nPERIODIC_INDEX= 0\n0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n";
    const MeshData d = parse(kSpecExample + identity);
    check_spec_example(d); // identity transformation: no warning

    const std::string real = "NPERIODIC= 2\nPERIODIC_INDEX= 0\n0 0 0\n0 0 0\n0 0 0\n"
                             "PERIODIC_INDEX= 1\n0 0\n0 0\n1 0\n";
    const MeshData p = parse(kSpecExample + real);
    REQUIRE(p.warnings.size() == 1);
    CHECK_THAT(p.warnings[0], ContainsSubstring("periodic transformations (NPERIODIC=) ignored"));
}

TEST_CASE("SU2: point indices that differ from the position give a warning", "[su2][reader]") {
    std::string text = replaced(kSpecExample, "1.0 0.0 1", "1.0 0.0 7");
    text = replaced(text, "1.0 1.0 2", "1.0 1.0 1");
    const MeshData d = parse(text);
    CHECK(d.node_tags == std::vector<std::int64_t>{0, 1, 2, 3}); // positions, as in SU2
    CHECK(d.x[1] == 1.0);
    REQUIRE(d.warnings.size() == 1);
    CHECK_THAT(d.warnings[0], ContainsSubstring("2 point index(es) differ") && ContainsSubstring("first: point 1"));
}

TEST_CASE("SU2: quadrilaterals, empty markers and names with blanks", "[su2][reader]") {
    Su2Model m = two_triangles();
    m.elements = {{9, {0, 1, 2, 3}}};
    m.markers.push_back({"unused marker", {}});
    m.markers[0].name = "lower wall";
    const MeshData d = parse(write_su2(m));
    REQUIRE(d.elements.size() == 5);
    CHECK(d.elements[0].type == ElementType::Quad4);
    CHECK(d.elements[0].nodes == std::array<std::int64_t, 4>{0, 1, 2, 3});
    CHECK(d.physical_name(1, 1) == "lower wall");
    CHECK(d.physical_name(1, 5) == "unused marker");

    // No markers at all is valid syntax (build_mesh reports the boundary).
    const MeshData none = parse(kSpecExample.substr(0, kSpecExample.find("NMARK=")) + "NMARK= 0\n");
    CHECK(none.physical_names.empty());
    CHECK(none.elements.size() == 2);
}

TEST_CASE("SU2: element type names", "[su2][reader]") {
    CHECK(su2_element_name(3) == "line");
    CHECK(su2_element_name(5) == "triangle");
    CHECK(su2_element_name(9) == "quadrilateral");
    CHECK(su2_element_name(10) == "tetrahedron");
    CHECK(su2_element_name(7).empty());
}

TEST_CASE("SU2: read_su2 reads files and reports missing ones", "[su2][reader]") {
    test::TempDir dir("su2-read");
    test::write_text(dir / "m.su2", kSpecExample);
    check_spec_example(read_su2((dir / "m.su2").string()));
    CHECK_THROWS_WITH(read_su2((dir / "missing.su2").string()), ContainsSubstring("cannot open"));
}

// =============================================================================
// Malformed files
// =============================================================================

TEST_CASE("SU2: malformed headers and sections", "[su2][reader][errors]") {
    const std::string& t = kSpecExample;
    CHECK_THAT(parse_error(""), ContainsSubstring("no NDIME= line: not an SU2 mesh file"));
    CHECK_THAT(parse_error("% only a comment\n"), ContainsSubstring("no NDIME="));
    CHECK_THAT(parse_error("$MeshFormat\n4.1 0 8\n"), ContainsSubstring("line 1: unexpected data line '$MeshFormat'"));
    CHECK_THAT(parse_error(replaced(t, "NDIME= 2", "NDIME= 3")), ContainsSubstring("line 4: 3D mesh (NDIME= 3)"));
    CHECK_THAT(parse_error(replaced(t, "NDIME= 2", "NDIME= 1")), ContainsSubstring("invalid dimension NDIME= 1"));
    CHECK_THAT(parse_error(replaced(t, "NDIME= 2", "NDIME= two")), ContainsSubstring("invalid NDIME= 'two'"));
    CHECK_THAT(parse_error(replaced(t, "NDIME= 2\n", "")), ContainsSubstring("NDIME= must come before NELEM="));
    CHECK_THAT(parse_error("NDIME= 2\n" + t), ContainsSubstring("duplicate NDIME="));
    CHECK_THAT(parse_error(t + "NELEM= 0\n"), ContainsSubstring("duplicate NELEM= section"));
    CHECK_THAT(parse_error(t + "NPOIN= 0\n"), ContainsSubstring("duplicate NPOIN= section"));
    CHECK_THAT(parse_error(t + "NMARK= 0\n"), ContainsSubstring("duplicate NMARK= section"));
    CHECK_THAT(parse_error("NZONE= 2\n" + t), ContainsSubstring("multi-zone meshes (NZONE= 2)"));
    CHECK_THAT(parse_error("NZONE= 1\nIZONE= 2\n" + t), ContainsSubstring("invalid IZONE= '2'"));
    CHECK_THAT(parse_error(t + "NVERTEX= 3\n"), ContainsSubstring("unknown keyword 'NVERTEX='"));
    CHECK_THAT(parse_error(t + "MARKER_TAG= extra\n"), ContainsSubstring("MARKER_TAG= outside its section"));
    CHECK_THAT(parse_error(t + "PERIODIC_INDEX= 1\n"), ContainsSubstring("PERIODIC_INDEX= outside its section"));
    CHECK_THAT(parse_error(t.substr(0, t.find("%\n% Node"))), ContainsSubstring("no NPOIN= section"));
    CHECK_THAT(parse_error(t.substr(0, t.find("%\n% Boundary"))), ContainsSubstring("no NMARK= section"));
    CHECK_THAT(parse_error("NDIME= 2\nNPOIN= 0\nNMARK= 0\n"), ContainsSubstring("no NELEM= section"));
}

TEST_CASE("SU2: malformed counts", "[su2][reader][errors]") {
    const std::string& t = kSpecExample;
    CHECK_THAT(parse_error(replaced(t, "NELEM= 2", "NELEM= -2")), ContainsSubstring("negative count after NELEM="));
    CHECK_THAT(parse_error(replaced(t, "NELEM= 2", "NELEM= two")), ContainsSubstring("expected a count after NELEM="));
    CHECK_THAT(parse_error(replaced(t, "NELEM= 2", "NELEM= 99999999999")),
               ContainsSubstring("NELEM= 99999999999 exceeds the size of the file"));
    CHECK_THAT(parse_error(replaced(t, "NPOIN= 4", "NPOIN= 4 5")), ContainsSubstring("more domain points than points"));
    CHECK_THAT(parse_error(replaced(t, "NPOIN= 4", "NPOIN= 4 2 1")), ContainsSubstring("expected 'NPOIN= count [domain count]'"));
    // Count larger than the section: the next keyword interrupts it ...
    CHECK_THAT(parse_error(replaced(t, "NELEM= 2", "NELEM= 3")),
               ContainsSubstring("line 14: NELEM= 3 declares 3 lines, found 2 before 'NPOIN='"));
    CHECK_THAT(parse_error(replaced(t, "MARKER_ELEMS= 1\n3 0 1", "MARKER_ELEMS= 2\n3 0 1")),
               ContainsSubstring("MARKER_ELEMS= (marker 'lower') 2 declares 2 lines, found 1 before 'MARKER_TAG='"));
    // ... or the end of the file.
    CHECK_THAT(parse_error(replaced(t, "NMARK= 4", "NMARK= 5")),
               ContainsSubstring("unexpected end of file: NMARK= 5 declares 5 markers, found 4"));
    CHECK_THAT(parse_error(t.substr(0, t.find("1.0 1.0 2"))),
               ContainsSubstring("unexpected end of file: NPOIN= 4 declares 4 lines, found 2"));
    // Count smaller than the section: the remaining lines are stray data.
    CHECK_THAT(parse_error(replaced(t, "NPOIN= 4", "NPOIN= 3")),
               ContainsSubstring("line 18: unexpected data line '0.0 1.0 3' outside a NELEM=, NPOIN= or NMARK= section"));
}

TEST_CASE("SU2: malformed elements", "[su2][reader][errors]") {
    const std::string& t = kSpecExample;
    auto with_first_element = [&](const std::string& line) { return parse_error(replaced(t, "5 0 1 3 0", line)); };
    CHECK_THAT(with_first_element("5 0 1"),
               ContainsSubstring("line 9: a triangle (type 5) needs 3 node indices (and an optional element index), found 2"));
    CHECK_THAT(with_first_element("5 0 1 3 0 7"), ContainsSubstring("found 5 values"));
    CHECK_THAT(with_first_element("7 0 1 3"), ContainsSubstring("unknown VTK element type 7 in NELEM="));
    CHECK_THAT(with_first_element("tri 0 1 3"), ContainsSubstring("expected a VTK element type"));
    CHECK_THAT(with_first_element("10 0 1 3 2"), ContainsSubstring("a tetrahedron is a 3D element"));
    CHECK_THAT(with_first_element("12 0 1 2 3 0 1 2 3"), ContainsSubstring("a hexahedron is a 3D element"));
    CHECK_THAT(with_first_element("3 0 1"),
               ContainsSubstring("a line (type 3) is not a 2D element") && ContainsSubstring("belong to the markers"));
    CHECK_THAT(with_first_element("1 0"), ContainsSubstring("a vertex (type 1) is not a 2D element"));
    CHECK_THAT(with_first_element("5 0 x 3"), ContainsSubstring("invalid node index 'x'"));
    CHECK_THAT(with_first_element("5 0 1.5 3"), ContainsSubstring("invalid node index '1.5'"));
    CHECK_THAT(with_first_element("5 0 -1 3"), ContainsSubstring("node index -1 out of range"));
    CHECK_THAT(with_first_element("5 0 1 3 first"), ContainsSubstring("invalid element index 'first'"));
    // Index range is checked against NPOIN=, which may come later in the file.
    CHECK_THAT(with_first_element("5 0 1 4 0"),
               ContainsSubstring("test.su2: line 9: element references point 4, but NPOIN= 4 (indices are 0-based)"));
    CHECK_THAT(parse_error(replaced(t, "3 2 3\n", "3 2 9\n")),
               ContainsSubstring("line 31: boundary line references point 9, but NPOIN= 4"));
}

TEST_CASE("SU2: malformed points", "[su2][reader][errors]") {
    const std::string& t = kSpecExample;
    auto with_point = [&](const std::string& line) { return parse_error(replaced(t, "1.0 0.0 1", line)); };
    CHECK_THAT(with_point("1.0"), ContainsSubstring("line 16: a point of a 2D mesh is 'x y [index]', found 1 values"));
    CHECK_THAT(with_point("1.0 0.0 0.0 1"), ContainsSubstring("found 4 values"));
    CHECK_THAT(with_point("1.0 zero 1"), ContainsSubstring("expected finite coordinates"));
    CHECK_THAT(with_point("nan 0.0 1"), ContainsSubstring("expected finite coordinates"));
    CHECK_THAT(with_point("inf 0.0 1"), ContainsSubstring("expected finite coordinates"));
    CHECK_THAT(with_point("1e999 0.0 1"), ContainsSubstring("expected finite coordinates"));
    CHECK_THAT(with_point("1.0d0 0.0 1"), ContainsSubstring("expected finite coordinates"));
    CHECK_THAT(with_point("1.0 0.0 0.5"), ContainsSubstring("must be its integer index") &&
                                              ContainsSubstring("3D coordinates in a NDIME= 2 file?"));
}

TEST_CASE("SU2: malformed markers", "[su2][reader][errors]") {
    const std::string& t = kSpecExample;
    CHECK_THAT(parse_error(replaced(t, "MARKER_TAG= lower\n", "")),
               ContainsSubstring("expected 'MARKER_TAG= name' of marker 1 of 4, found 'MARKER_ELEMS= 1'"));
    CHECK_THAT(parse_error(replaced(t, "MARKER_TAG= lower", "MARKER_TAG=")), ContainsSubstring("empty MARKER_TAG= name (marker 1)"));
    CHECK_THAT(parse_error(replaced(t, "MARKER_TAG= right", "MARKER_TAG= lower")), ContainsSubstring("duplicate marker 'lower'"));
    CHECK_THAT(parse_error(replaced(t, "MARKER_ELEMS= 1\n3 0 1\n", "")),
               ContainsSubstring("expected 'MARKER_ELEMS= count' after 'MARKER_TAG= lower'"));
    CHECK_THAT(parse_error(replaced(t, "3 0 1\n", "5 0 1 2\n")),
               ContainsSubstring("marker 'lower' contains a triangle (type 5); the markers of a 2D mesh contain lines (type 3)"));
    CHECK_THAT(parse_error(replaced(t, "3 0 1\n", "1 0\n")), ContainsSubstring("marker 'lower' contains a vertex"));
    CHECK_THAT(parse_error(replaced(t, "3 0 1\n", "3 0\n")), ContainsSubstring("a line (type 3) needs 2 node indices"));
    CHECK_THAT(parse_error(replaced(t, "MARKER_ELEMS= 1\n3 0 1", "MARKER_ELEMS= -1\n3 0 1")),
               ContainsSubstring("negative count after MARKER_ELEMS="));
    CHECK_THAT(parse_error(t + "NPERIODIC= 1\nPERIODIC_INDEX= 0\n0 0 0\n"),
               ContainsSubstring("unexpected end of file: PERIODIC_INDEX= 0 3 declares 3 lines, found 1"));
    CHECK_THAT(parse_error(t + "NPERIODIC= 1\n0 0 0\n"), ContainsSubstring("expected 'PERIODIC_INDEX= i'"));
    CHECK_THAT(parse_error(t + "NPERIODIC= 1\nPERIODIC_INDEX= 0\n0 0 x\n0 0 0\n0 0 0\n"),
               ContainsSubstring("invalid periodic transformation value 'x'"));
}

TEST_CASE("SU2: truncated files are rejected cleanly at every length", "[su2][reader][errors][robustness]") {
    for (const std::string& full : {kSpecExample, write_su2(two_triangles(), Su2WriteOptions{false, false, false, true, true, true, "", ""})}) {
        int parsed = 0;
        for (std::size_t n = 0; n < full.size(); ++n) {
            try {
                parse_su2(std::string_view(full).substr(0, n), "cut.su2");
                ++parsed;
            } catch (const MeshImportError&) {
            }
        }
        // Only the complete file without its final "\n" or "\r\n" can parse
        // ("3 3 0" cut to "3 3" fails).
        CHECK(parsed <= 2);
    }
}

TEST_CASE("SU2: corrupted bytes never crash the reader", "[su2][reader][errors][robustness]") {
    std::uint64_t state = 0x2545F4914F6CDD1Dull;
    auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    const char alphabet[] = "0123456789 \n\t%=-+.eE5MARKER_TAGNPOINELEMDIx\r";
    for (int trial = 0; trial < 3000; ++trial) {
        std::string text = kSpecExample;
        const int edits = 1 + static_cast<int>(next() % 4);
        for (int k = 0; k < edits; ++k) {
            const std::size_t at = next() % text.size();
            switch (next() % 3) {
            case 0: text[at] = static_cast<char>(next() & 0xFF); break;                     // any byte
            case 1: text[at] = alphabet[next() % (sizeof(alphabet) - 1)]; break;             // plausible byte
            default: text.erase(at, 1 + next() % 8); if (text.empty()) text = "x"; break;   // deletion
            }
        }
        try {
            const MeshData d = parse_su2(text, "fuzz.su2");
            CHECK(d.node_tags.size() == d.x.size());
            for (const RawElement& e : d.elements)
                for (int k = 0; k < node_count(e.type); ++k) CHECK(e.nodes[static_cast<std::size_t>(k)] < static_cast<std::int64_t>(d.x.size()));
        } catch (const MeshImportError&) {
        }
    }
}
