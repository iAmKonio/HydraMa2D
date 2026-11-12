// Gmsh MSH reader (mesh/GmshReader.hpp): the four format variants, byte
// order, line endings, sections, element types and every kind of invalid
// input. Files are produced by tests/GmshTestWriter.hpp or written inline;
// files written by Gmsh itself are tested in test_gmsh_fixtures.cpp.

#include "GmshTestWriter.hpp"
#include "IoTestUtils.hpp"
#include "mesh/GmshReader.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>

using namespace hydrama;
using namespace hydrama::mesh;
using hydrama::test::GmshModel;
using hydrama::test::GmshWriteOptions;
using hydrama::test::write_gmsh;
using Catch::Matchers::ContainsSubstring;

namespace {

// Two triangles on the unit square with sparse node tags (within the 32-bit
// range of MSH 2.2 binary files), a boundary line in
// two physical groups, an unnamed group, a point element and a quadrangle.
GmshModel sample_model() {
    GmshModel g;
    g.add_node(100, 0.0, 0.0);
    g.add_node(7, 1.0, 0.0);
    g.add_node(2000000000, 1.0, 1.0);
    g.add_node(3, 0.0, 1.0, 0.0);
    g.add_node(55, 0.1, 0.2, 0.0); // used by the point only
    g.add_element(15, {55}, {9});
    g.add_element(2, {100, 7, 2000000000}, {1});
    g.add_element(2, {100, 2000000000, 3}, {1});
    g.add_element(1, {100, 7}, {11, 12});
    g.add_element(1, {7, 2000000000}, {13});
    g.add_element(1, {2000000000, 3}, {});
    g.add_element(3, {100, 7, 2000000000, 3}, {2});
    g.names[{1, 11}] = "bottom wall";
    g.names[{1, 12}] = "noSlip";
    g.names[{2, 1}] = "fluid";
    g.names[{0, 9}] = "probe point";
    return g;
}

const GmshWriteOptions kVariants[] = {
    {"2.2", false, false, false, 8, ""},
    {"2.2", true, false, false, 8, ""},
    {"4.1", false, false, false, 8, ""},
    {"4.1", true, false, false, 8, ""},
};

std::string label(const GmshWriteOptions& o) {
    return o.version + (o.binary ? " binary" : " ASCII") + (o.swap_bytes ? " swapped" : "") +
           (o.size_t_bytes == 4 ? " size_t=4" : "") + (o.crlf ? " CRLF" : "");
}

// Expected MeshData content of sample_model() (independent of the format).
void check_sample(const MeshData& d) {
    REQUIRE(d.node_tags == std::vector<std::int64_t>{100, 7, 2000000000, 3, 55});
    CHECK(d.x == std::vector<double>{0.0, 1.0, 1.0, 0.0, 0.1});
    CHECK(d.y == std::vector<double>{0.0, 0.0, 1.0, 1.0, 0.2});
    CHECK(d.z == std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0});

    // Point ignored; the line in two groups appears twice; unnamed line physical 0.
    REQUIRE(d.elements.size() == 7);
    const auto& e = d.elements;
    CHECK(e[0].type == ElementType::Tri3);
    CHECK(e[0].physical == 1);
    CHECK((std::vector<std::int64_t>(e[0].nodes.begin(), e[0].nodes.begin() + 3) ==
           std::vector<std::int64_t>{100, 7, 2000000000}));
    CHECK(e[1].type == ElementType::Tri3);
    CHECK(e[2].type == ElementType::Line2);
    CHECK(e[2].physical == 11);
    CHECK(e[3].type == ElementType::Line2);
    CHECK(e[3].physical == 12);
    CHECK(e[2].nodes[0] == 100);
    CHECK(e[2].nodes[1] == 7);
    CHECK(e[4].physical == 13);
    CHECK(e[5].physical == 0);
    CHECK(e[6].type == ElementType::Quad4);
    CHECK(e[6].nodes == std::array<std::int64_t, 4>{100, 7, 2000000000, 3});
    CHECK(e[6].physical == 2);

    CHECK(d.physical_name(1, 11) == "bottom wall");
    CHECK(d.physical_name(1, 12) == "noSlip");
    CHECK(d.physical_name(2, 1) == "fluid");
    CHECK(d.physical_name(0, 9) == "probe point");
    CHECK(d.physical_name(1, 13).empty());
}

MeshData parse(const std::string& text, const std::string& source = "test.msh") { return parse_gmsh(text, source); }

std::string parse_error(const std::string& text, const std::string& source = "test.msh") {
    try {
        parse_gmsh(text, source);
    } catch (const MeshImportError& e) {
        return e.what();
    }
    FAIL("expected MeshImportError");
    return {};
}

const std::string kHeader22 = "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n";
const std::string kNodes22 = "$Nodes\n3\n1 0 0 0\n2 1 0 0\n3 0 1 0\n$EndNodes\n";
const std::string kHeader41 = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n";

} // namespace

// =============================================================================
// Format variants
// =============================================================================

TEST_CASE("Gmsh 2.2 and 4.1, ASCII and binary, give identical data", "[gmsh][reader]") {
    for (const GmshWriteOptions& o : kVariants) {
        CAPTURE(label(o));
        const MeshData d = parse(write_gmsh(sample_model(), o));
        check_sample(d);
        CHECK(d.format == "Gmsh MSH " + o.version + (o.binary ? " binary" : " ASCII"));
        CHECK(d.source == "test.msh");
    }
}

TEST_CASE("Binary files in the other byte order and with 4-byte size_t", "[gmsh][reader]") {
    for (const std::string version : {"2.2", "4.1"}) {
        GmshWriteOptions o{version, true, true, false, 8, ""};
        CAPTURE(label(o));
        check_sample(parse(write_gmsh(sample_model(), o)));
    }
    for (const bool swap : {false, true}) {
        GmshWriteOptions o{"4.1", true, swap, false, 4, ""};
        CAPTURE(label(o));
        check_sample(parse(write_gmsh(sample_model(), o)));
    }
}

TEST_CASE("ASCII files with CRLF line endings and extra blanks", "[gmsh][reader]") {
    for (const std::string version : {"2.2", "4.1"}) {
        GmshWriteOptions o{version, false, false, true, 8, ""};
        CAPTURE(label(o));
        check_sample(parse(write_gmsh(sample_model(), o)));

        // Trailing blanks, blank lines and indentation on every line.
        std::string spaced;
        for (char c : write_gmsh(sample_model(), GmshWriteOptions{version, false, false, false, 8, ""})) {
            if (c == '\n') spaced += " \t\r\n\n  \t";
            else spaced += c;
        }
        check_sample(parse(spaced));
    }
}

TEST_CASE("Node tags beyond 32 bits (MSH 4.1 size_t, MSH 2.2 ASCII)", "[gmsh][reader]") {
    GmshModel g;
    const std::int64_t big = 123456789012345LL;
    g.add_node(big, 0.0, 0.0);
    g.add_node(big + 2, 1.0, 0.0);
    g.add_node(1, 0.0, 1.0);
    g.add_element(2, {big, big + 2, 1}, {1});
    g.add_element(1, {big, big + 2}, {2}, 9876543210123LL);
    for (const GmshWriteOptions& o : {GmshWriteOptions{"2.2", false, false, false, 8, ""},
                                      GmshWriteOptions{"4.1", false, false, false, 8, ""},
                                      GmshWriteOptions{"4.1", true, false, false, 8, ""}}) {
        CAPTURE(label(o));
        const MeshData d = parse(write_gmsh(g, o));
        CHECK(d.node_tags == std::vector<std::int64_t>{big, big + 2, 1});
        REQUIRE(d.elements.size() == 2);
        CHECK(d.elements[0].nodes[1] == big + 2);
        if (o.version == "4.1") CHECK(d.elements[1].tag == 9876543210123LL);
    }
}

TEST_CASE("Binary data that looks like whitespace or newlines", "[gmsh][reader]") {
    // Tags 9, 10, 13 and 32 are '\t', '\n', '\r' and ' ' as bytes: a reader that
    // skips whitespace inside binary data would lose them.
    GmshModel g;
    for (int t : {9, 10, 13, 32}) g.add_node(t, t * 0.5, t * 0.25);
    g.add_element(2, {9, 10, 13}, {10});
    g.add_element(2, {9, 13, 32}, {32});
    g.add_element(1, {9, 10}, {9});
    for (const GmshWriteOptions& o : kVariants) {
        if (!o.binary) continue;
        CAPTURE(label(o));
        const MeshData d = parse(write_gmsh(g, o));
        CHECK(d.node_tags == std::vector<std::int64_t>{9, 10, 13, 32});
        CHECK(d.x[1] == 5.0);
        REQUIRE(d.elements.size() == 3);
        CHECK(d.elements[1].physical == 32);
        CHECK(d.elements[2].nodes[1] == 10);
    }
}

TEST_CASE("Unknown sections are skipped", "[gmsh][reader]") {
    const std::string extra =
        "$Comments\nany text, even $EndNodes or $Nodes\n$EndComments\n"
        "$InterpolationScheme\n\"x\"\n1\n$EndInterpolationScheme\n";
    for (GmshWriteOptions o : kVariants) {
        o.extra_sections = extra;
        CAPTURE(label(o));
        std::string text = write_gmsh(sample_model(), o);
        // ... also after the mesh, with binary-looking content.
        text += "$NodeData\n1\n\"p\"\n";
        text += std::string("\0\n\x01$End", 7);
        text += "\n$EndNodeData\n";
        check_sample(parse(text));
    }
}

TEST_CASE("MSH 2.0 and 2.1 headers are read as 2.2", "[gmsh][reader]") {
    for (const char* v : {"2", "2.0", "2.1"}) {
        CAPTURE(v);
        const std::string text = std::string("$MeshFormat\n") + v + " 0 8\n$EndMeshFormat\n" + kNodes22 +
                                 "$Elements\n1\n1 2 0 1 2 3\n$EndElements\n";
        const MeshData d = parse(text);
        REQUIRE(d.elements.size() == 1);
        CHECK(d.elements[0].physical == 0); // no tags at all
        CHECK(d.elements[0].entity == 0);
    }
}

TEST_CASE("MSH 2.2 element tags: physical, entity, partitions", "[gmsh][reader]") {
    const std::string text = kHeader22 + kNodes22 +
                             "$Elements\n3\n"
                             "1 2 1 5 1 2 3\n"          // physical only
                             "2 2 4 6 8 2 1 1 2 3\n"    // physical, entity, partitions
                             "3 1 2 0 4 1 2\n"          // no physical group
                             "$EndElements\n";
    const MeshData d = parse(text);
    REQUIRE(d.elements.size() == 3);
    CHECK(d.elements[0].physical == 5);
    CHECK(d.elements[0].entity == 0);
    CHECK(d.elements[1].physical == 6);
    CHECK(d.elements[1].entity == 8);
    CHECK(d.elements[1].tag == 2);
    CHECK(d.elements[2].physical == 0);
}

TEST_CASE("MSH 4.1 parametric nodes and entity physical groups", "[gmsh][reader]") {
    const std::string text = kHeader41 +
                             "$PhysicalNames\n2\n1 7 \"wall\"\n1 8 \"second group\"\n$EndPhysicalNames\n"
                             "$Entities\n1 1 1 0\n"
                             "1 0 0 0 0\n"
                             "3 0 0 0 1 0 0 2 7 8 2 1 -1\n"
                             "4 0 0 0 1 1 0 0 1 3\n"
                             "$EndEntities\n"
                             "$Nodes\n3 3 1 30\n"
                             "0 1 0 1\n1\n0 0 0\n"
                             "1 3 1 1\n20\n1 0 0 0.5\n"            // curve node: u
                             "2 4 1 1\n30\n0 1 0 0.25 0.75\n"      // surface node: u v
                             "$EndNodes\n"
                             "$Elements\n3 3 1 3\n"
                             "0 1 15 1\n1 1\n"
                             "1 3 1 1\n2 1 20\n"
                             "2 4 2 1\n3 1 20 30\n"
                             "$EndElements\n";
    const MeshData d = parse(text);
    CHECK(d.node_tags == std::vector<std::int64_t>{1, 20, 30});
    CHECK(d.x == std::vector<double>{0.0, 1.0, 0.0});
    CHECK(d.y == std::vector<double>{0.0, 0.0, 1.0});
    // The line's curve is in two groups: two records. The surface in none.
    REQUIRE(d.elements.size() == 3);
    CHECK(d.elements[0].type == ElementType::Line2);
    CHECK(d.elements[0].physical == 7);
    CHECK(d.elements[1].physical == 8);
    CHECK(d.elements[0].entity == 3);
    CHECK(d.elements[2].type == ElementType::Tri3);
    CHECK(d.elements[2].physical == 0);
    CHECK(d.physical_name(1, 8) == "second group");
}

TEST_CASE("MSH 4.1 without $Entities has no physical groups", "[gmsh][reader]") {
    const std::string text = kHeader41 +
                             "$Nodes\n1 3 1 3\n2 1 0 3\n1\n2\n3\n0 0 0\n1 0 0\n0 1 0\n$EndNodes\n"
                             "$Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\n$EndElements\n";
    const MeshData d = parse(text);
    REQUIRE(d.elements.size() == 1);
    CHECK(d.elements[0].physical == 0);
}

TEST_CASE("Physical names: spaces, quotes, no quotes", "[gmsh][reader]") {
    const std::string text = kHeader22 +
                             "$PhysicalNames\n3\n1 1 \"Symmetry Plane\"\n1 2 \"a \"quoted\" name\"\n2 3 fluid zone\n"
                             "$EndPhysicalNames\n" +
                             kNodes22 + "$Elements\n1\n1 2 2 3 1 1 2 3\n$EndElements\n";
    const MeshData d = parse(text);
    CHECK(d.physical_name(1, 1) == "Symmetry Plane");
    CHECK(d.physical_name(1, 2) == "a \"quoted\" name");
    CHECK(d.physical_name(2, 3) == "fluid zone");
}

TEST_CASE("read_gmsh reads files and reports missing ones", "[gmsh][reader]") {
    test::TempDir dir("gmsh-read");
    const auto path = (dir / "m.msh").string();
    test::write_text(path, write_gmsh(sample_model(), kVariants[3]));
    check_sample(read_gmsh(path));
    CHECK_THROWS_WITH(read_gmsh((dir / "missing.msh").string()), ContainsSubstring("cannot open"));
}

// =============================================================================
// Invalid input
// =============================================================================

TEST_CASE("Unsupported versions and non-Gmsh files", "[gmsh][reader][errors]") {
    CHECK_THAT(parse_error(""), ContainsSubstring("empty file"));
    CHECK_THAT(parse_error("hello\n"), ContainsSubstring("expected a section header"));
    CHECK_THAT(parse_error("$Nodes\n0\n$EndNodes\n"), ContainsSubstring("must be $MeshFormat"));
    CHECK_THAT(parse_error("$NOD\n1\n1 0 0 0\n$ENDNOD\n"), ContainsSubstring("MSH 1.0"));
    CHECK_THAT(parse_error("$MeshFormat\n4 0 8\n$EndMeshFormat\n"), ContainsSubstring("MSH 4.0 is not supported"));
    CHECK_THAT(parse_error("$MeshFormat\n4.0 0 8\n$EndMeshFormat\n"), ContainsSubstring("MSH 4.0 is not supported"));
    CHECK_THAT(parse_error("$MeshFormat\n3.0 0 8\n$EndMeshFormat\n"), ContainsSubstring("unsupported MSH version 3.0"));
    CHECK_THAT(parse_error("$MeshFormat\n4.1 2 8\n$EndMeshFormat\n"), ContainsSubstring("invalid file type 2"));
    CHECK_THAT(parse_error("$MeshFormat\n4.1\n$EndMeshFormat\n"), ContainsSubstring("invalid $MeshFormat"));
    CHECK_THAT(parse_error("$MeshFormat\n2.2 1 4\n"), ContainsSubstring("data size 4"));
    CHECK_THAT(parse_error("$MeshFormat\n4.1 1 2\n"), ContainsSubstring("data size 2"));
    CHECK_THAT(parse_error(std::string("$MeshFormat\n4.1 1 8\n\x02\0\0\x03\n$EndMeshFormat\n", 40)),
               ContainsSubstring("endianness"));
    CHECK_THAT(parse_error(kHeader22 + kHeader22), ContainsSubstring("duplicate $MeshFormat"));
    CHECK_THAT(parse_error(kHeader22), ContainsSubstring("no $Nodes"));
    CHECK_THAT(parse_error(kHeader22 + kNodes22), ContainsSubstring("no $Elements"));
    CHECK_THAT(parse_error(kHeader22 + "$Elements\n0\n$EndElements\n"), ContainsSubstring("$Elements before $Nodes"));
    CHECK_THAT(parse_error(kHeader22 + kNodes22 + kNodes22), ContainsSubstring("duplicate $Nodes"));
    CHECK_THAT(parse_error(kHeader41 + "$PartitionedEntities\n1\n$EndPartitionedEntities\n"),
               ContainsSubstring("partitioned"));
    CHECK_THAT(parse_error(kHeader22 + "$Comments\nno end\n"), ContainsSubstring("has no $EndComments"));
    // $EndX must be a whole line: "$EndCommentsX" is not the end of $Comments.
    CHECK_THAT(parse_error(kHeader22 + "$Comments\n$EndCommentsX\n"), ContainsSubstring("has no $EndComments"));
}

TEST_CASE("Element types that are not linear 2D elements are rejected", "[gmsh][reader][errors]") {
    auto with_element = [](const std::string& element) {
        return kHeader22 + "$Nodes\n6\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 .5 0 0\n5 .5 .5 0\n6 0 .5 0\n$EndNodes\n"
                           "$Elements\n1\n" + element + "\n$EndElements\n";
    };
    CHECK_THAT(parse_error(with_element("1 9 0 1 2 3 4 5 6")),
               ContainsSubstring("6-node second-order triangle") && ContainsSubstring("Mesh.ElementOrder = 1"));
    CHECK_THAT(parse_error(with_element("1 8 0 1 2 4")), ContainsSubstring("higher-order"));
    CHECK_THAT(parse_error(with_element("1 4 0 1 2 3 4")),
               ContainsSubstring("4-node tetrahedron") && ContainsSubstring("3D element"));
    CHECK_THAT(parse_error(with_element("1 99 0 1 2")), ContainsSubstring("unknown Gmsh element type 99"));
    CHECK(gmsh_element_name(2) == "3-node triangle");
    CHECK(gmsh_element_name(1000).empty());
}

TEST_CASE("Malformed numbers and counts", "[gmsh][reader][errors]") {
    auto nodes = [](const std::string& body) {
        return kHeader22 + "$Nodes\n" + body + "$EndNodes\n$Elements\n0\n$EndElements\n";
    };
    CHECK_THAT(parse_error(nodes("1\n1 0 zero 0\n")), ContainsSubstring("expected a finite real number (node y)"));
    CHECK_THAT(parse_error(nodes("1\n1 0 nan 0\n")), ContainsSubstring("finite real number"));
    CHECK_THAT(parse_error(nodes("1\n1 0 inf 0\n")), ContainsSubstring("finite real number"));
    CHECK_THAT(parse_error(nodes("1\n1 0 1e999 0\n")), ContainsSubstring("finite real number"));
    CHECK_THAT(parse_error(nodes("1\n1.5 0 0 0\n")), ContainsSubstring("expected an integer (node tag)"));
    CHECK_THAT(parse_error(nodes("-1\n")), ContainsSubstring("negative count"));
    CHECK_THAT(parse_error(nodes("99999999999\n1 0 0 0\n")), ContainsSubstring("exceeds the size of the file"));
    CHECK_THAT(parse_error(nodes("2\n1 0 0 0\n")), ContainsSubstring("expected an integer (node tag), found '$EndNodes'"));
    CHECK_THAT(parse_error(nodes("1\n1 0 0 0 0\n")), ContainsSubstring("expected $EndNodes"));
    // Positive signs and Fortran-free exponents are fine.
    const MeshData d = parse(nodes("1\n+1 +1.5e+00 -2E-1 0\n"));
    CHECK(d.x[0] == 1.5);
    CHECK(d.y[0] == -0.2);
}

TEST_CASE("MSH 4.1 header and block inconsistencies", "[gmsh][reader][errors]") {
    const std::string nodes = "$Nodes\n1 3 1 3\n2 1 0 3\n1\n2\n3\n0 0 0\n1 0 0\n0 1 0\n$EndNodes\n";
    CHECK_THAT(parse_error(kHeader41 + "$Nodes\n1 4 1 4\n2 1 0 3\n1\n2\n3\n0 0 0\n1 0 0\n0 1 0\n$EndNodes\n"),
               ContainsSubstring("the header declares 4"));
    CHECK_THAT(parse_error(kHeader41 + "$Nodes\n1 2 1 3\n2 1 0 3\n1\n2\n3\n0 0 0\n1 0 0\n0 1 0\n$EndNodes\n"),
               ContainsSubstring("more nodes than the declared total"));
    CHECK_THAT(parse_error(kHeader41 + "$Nodes\n1 3 1 3\n2 1 7 3\n$EndNodes\n"), ContainsSubstring("parametric flag 7"));
    CHECK_THAT(parse_error(kHeader41 + "$Nodes\n1 3 1 3\n5 1 0 3\n$EndNodes\n"), ContainsSubstring("entity dimension 5"));
    CHECK_THAT(parse_error(kHeader41 + nodes + "$Elements\n1 1 1 1\n1 1 2 1\n1 1 2 3\n$EndElements\n"),
               ContainsSubstring("entity dimension 1 contains 3-node triangle"));
    CHECK_THAT(parse_error(kHeader41 + nodes + "$Elements\n1 2 1 2\n2 1 2 1\n1 1 2 3\n$EndElements\n"),
               ContainsSubstring("the header declares 2"));
    CHECK_THAT(parse_error(kHeader41 + "$Entities\n1 0 0 0\n1 0 0 0 0\n1 0 0 0 0\n$EndEntities\n"),
               ContainsSubstring("expected $EndEntities"));
    CHECK_THAT(parse_error(kHeader41 + "$Entities\n2 0 0 0\n1 0 0 0 0\n1 0 0 0 0\n$EndEntities\n"),
               ContainsSubstring("duplicate entity"));
    CHECK_THAT(parse_error(kHeader22 + "$PhysicalNames\n1\n7 1 \"x\"\n$EndPhysicalNames\n"),
               ContainsSubstring("invalid physical name entry"));
    CHECK_THAT(parse_error(kHeader22 + "$PhysicalNames\n2\n1 1 \"x\"\n1 1 \"y\"\n$EndPhysicalNames\n"),
               ContainsSubstring("duplicate physical name"));
}

TEST_CASE("Errors name the file and the line or byte offset", "[gmsh][reader][errors]") {
    const std::string text = kHeader22 + "$Nodes\n2\n1 0 0 0\n2 0 x 0\n$EndNodes\n";
    CHECK_THAT(parse_error(text, "case/constant/c.msh"), ContainsSubstring("case/constant/c.msh: line 7:"));

    std::string binary = write_gmsh(sample_model(), kVariants[3]);
    binary.resize(binary.size() - 40);
    CHECK_THAT(parse_error(binary, "b.msh"), ContainsSubstring("b.msh: byte "));
}

TEST_CASE("Truncated files are rejected cleanly at every length", "[gmsh][reader][errors][robustness]") {
    for (const GmshWriteOptions& o : kVariants) {
        CAPTURE(label(o));
        const std::string full = write_gmsh(sample_model(), o);
        // Every prefix either parses (only if nothing essential was cut) or
        // throws MeshImportError; never anything else (run with ASan/UBSan).
        int parsed = 0;
        for (std::size_t n = 0; n < full.size(); ++n) {
            try {
                parse_gmsh(std::string_view(full).substr(0, n), "cut.msh");
                ++parsed;
            } catch (const MeshImportError&) {
            }
        }
        // Only cuts inside the final "$EndElements\n" line ... or after it can parse.
        CHECK(parsed <= 2);
    }
}

TEST_CASE("Corrupted bytes never crash the reader", "[gmsh][reader][errors][robustness]") {
    std::uint64_t state = 0x9E3779B97F4A7C15ull;
    auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    for (const GmshWriteOptions& o : kVariants) {
        CAPTURE(label(o));
        const std::string full = write_gmsh(sample_model(), o);
        for (int trial = 0; trial < 400; ++trial) {
            std::string text = full;
            const int flips = 1 + static_cast<int>(next() % 4);
            for (int k = 0; k < flips; ++k) text[next() % text.size()] = static_cast<char>(next() & 0xFF);
            try {
                const MeshData d = parse_gmsh(text, "fuzz.msh");
                CHECK(d.node_tags.size() == d.x.size());
            } catch (const MeshImportError&) {
            }
        }
    }
}
