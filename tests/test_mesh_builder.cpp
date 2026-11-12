// build_mesh (mesh/MeshBuilder.hpp) and check_mesh (mesh/MeshCheck.hpp):
// node tag remapping, orientation, boundary matching and orientation,
// boundary conditions from names and *.bofl overrides, and every invalid
// configuration.

#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "MeshTestUtils.hpp"
#include "mesh/MeshBuilder.hpp"
#include "mesh/MeshCheck.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <set>

using namespace hydrama;
using namespace hydrama::mesh;
using Catch::Matchers::ContainsSubstring;

namespace {

// ---------------------------------------------------------------------------
// MeshData builders
// ---------------------------------------------------------------------------
void add_node(MeshData& d, std::int64_t tag, double x, double y, double z = 0.0) {
    d.node_tags.push_back(tag);
    d.x.push_back(x);
    d.y.push_back(y);
    d.z.push_back(z);
}

void add_tri(MeshData& d, std::int64_t a, std::int64_t b, std::int64_t c, int physical = 0, std::int64_t tag = 0) {
    RawElement e;
    e.type = ElementType::Tri3;
    e.nodes = {a, b, c, 0};
    e.physical = physical;
    e.tag = tag ? tag : static_cast<std::int64_t>(d.elements.size() + 1);
    d.elements.push_back(e);
}

void add_line(MeshData& d, std::int64_t a, std::int64_t b, int physical, std::int64_t tag = 0) {
    RawElement e;
    e.type = ElementType::Line2;
    e.nodes = {a, b, 0, 0};
    e.physical = physical;
    e.tag = tag ? tag : static_cast<std::int64_t>(d.elements.size() + 1);
    d.elements.push_back(e);
}

// Unit square, 4 counter-clockwise triangles around the centre node 5.
//   4 ---- 3       bottom (1-2): "bottom_wall" (1)   right (2-3): "outlet" (2)
//   |  \/  |       top (3-4):    "top wall"    (3)   left (4-1):  "inlet"  (4)
//   |  /\  |
//   1 ---- 2
MeshData square() {
    MeshData d;
    d.source = "square.msh";
    d.format = "test";
    add_node(d, 1, 0.0, 0.0);
    add_node(d, 2, 1.0, 0.0);
    add_node(d, 3, 1.0, 1.0);
    add_node(d, 4, 0.0, 1.0);
    add_node(d, 5, 0.5, 0.5);
    add_tri(d, 1, 2, 5);
    add_tri(d, 2, 3, 5);
    add_tri(d, 3, 4, 5);
    add_tri(d, 4, 1, 5);
    add_line(d, 1, 2, 1);
    add_line(d, 2, 3, 2);
    add_line(d, 3, 4, 3);
    add_line(d, 4, 1, 4);
    d.physical_names[{1, 1}] = "bottom_wall";
    d.physical_names[{1, 2}] = "outlet";
    d.physical_names[{1, 3}] = "top wall";
    d.physical_names[{1, 4}] = "inlet";
    return d;
}

// nx x ny structured grid on [0,nx] x [0,ny] without the cells for which
// `hole(i, j)` is true. Triangles counter-clockwise, node tags 1-based.
// Boundary edges (one triangle) become lines in physical group
// `group(xm, ym)` of their midpoint.
MeshData grid(int nx, int ny, const std::function<bool(int, int)>& hole,
              const std::function<int(double, double)>& group) {
    MeshData d;
    d.source = "grid.msh";
    auto tag = [nx](int i, int j) { return static_cast<std::int64_t>(1 + j * (nx + 1) + i); };
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) add_node(d, tag(i, j), i, j);
    std::map<std::pair<std::int64_t, std::int64_t>, int> count;
    std::vector<std::array<std::int64_t, 3>> tris;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (hole(i, j)) continue;
            tris.push_back({tag(i, j), tag(i + 1, j), tag(i + 1, j + 1)});
            tris.push_back({tag(i, j), tag(i + 1, j + 1), tag(i, j + 1)});
        }
    }
    std::vector<std::pair<std::int64_t, std::int64_t>> directed;
    for (const auto& t : tris) {
        add_tri(d, t[0], t[1], t[2]);
        for (int k = 0; k < 3; ++k) {
            const std::int64_t a = t[static_cast<std::size_t>(k)], b = t[static_cast<std::size_t>((k + 1) % 3)];
            ++count[{std::min(a, b), std::max(a, b)}];
            directed.push_back({a, b});
        }
    }
    auto pos = [&](std::int64_t t) {
        const int k = static_cast<int>(t - 1);
        return std::make_pair(static_cast<double>(k % (nx + 1)), static_cast<double>(k / (nx + 1)));
    };
    for (const auto& [a, b] : directed) {
        if (count[{std::min(a, b), std::max(a, b)}] != 1) continue;
        const auto pa = pos(a), pb = pos(b);
        add_line(d, a, b, group(0.5 * (pa.first + pb.first), 0.5 * (pa.second + pb.second)));
    }
    return d;
}

Mesh build(const MeshData& d, const BuildOptions& o = {}, ImportReport* r = nullptr) {
    return build_mesh(d, o, r);
}

std::string build_error(const MeshData& d, const BuildOptions& o = {}) {
    try {
        build_mesh(d, o);
    } catch (const MeshImportError& e) {
        return e.what();
    }
    FAIL("expected MeshImportError");
    return {};
}

double signed_area(const Mesh& m, int e) {
    const auto& n = m.intma[static_cast<std::size_t>(e)];
    return 0.5 * ((m.x[n[1]] - m.x[n[0]]) * (m.y[n[2]] - m.y[n[0]]) - (m.x[n[2]] - m.x[n[0]]) * (m.y[n[1]] - m.y[n[0]]));
}

void require_valid(const Mesh& m) {
    const std::vector<std::string> problems = check_mesh(m);
    for (const auto& p : problems) UNSCOPED_INFO(p);
    REQUIRE(problems.empty());
}

// Mesh geometry as a set of coordinates per element / boundary side, independent of numbering.
using Coord = std::pair<double, double>;
std::multiset<std::vector<Coord>> element_shapes(const Mesh& m) {
    std::multiset<std::vector<Coord>> out;
    for (const auto& e : m.intma) {
        std::vector<Coord> c;
        for (int n : e) c.push_back({m.x[static_cast<std::size_t>(n)], m.y[static_cast<std::size_t>(n)]});
        std::rotate(c.begin(), std::min_element(c.begin(), c.end()), c.end()); // canonical cyclic rotation
        out.insert(c);
    }
    return out;
}

std::multiset<std::tuple<Coord, Coord, int, int>> side_shapes(const Mesh& m) {
    std::multiset<std::tuple<Coord, Coord, int, int>> out;
    for (const auto& s : m.iside)
        out.insert({{m.x[static_cast<std::size_t>(s.node1)], m.y[static_cast<std::size_t>(s.node1)]},
                    {m.x[static_cast<std::size_t>(s.node2)], m.y[static_cast<std::size_t>(s.node2)]},
                    s.side_id, s.flag});
    return out;
}

} // namespace

// =============================================================================
// Valid meshes
// =============================================================================

TEST_CASE("build_mesh: a valid mesh keeps file order and gets its flags from names", "[mesh_import]") {
    ImportReport r;
    const Mesh m = build(square(), {}, &r);
    require_valid(m);
    CHECK(m.npoin == 5);
    CHECK(m.nelem == 4);
    CHECK(m.nboun == 4);
    CHECK(m.intma[0] == std::array<int, 3>{0, 1, 4});
    CHECK(m.intma[3] == std::array<int, 3>{3, 0, 4});
    CHECK(m.x == std::vector<double>{0.0, 1.0, 1.0, 0.0, 0.5});

    const std::vector<std::array<int, 5>> expected = {
        // node1, node2, elem, side_id, flag
        {0, 1, 0, 1, 500}, {1, 2, 1, 2, 504}, {2, 3, 2, 3, 500}, {3, 0, 3, 4, 503}};
    for (int i = 0; i < 4; ++i) {
        const BoundarySide& s = m.iside[static_cast<std::size_t>(i)];
        CAPTURE(i);
        CHECK(std::array<int, 5>{s.node1, s.node2, s.elem, s.side_id, s.flag} == expected[static_cast<std::size_t>(i)]);
    }

    CHECK(r.triangles == 4);
    CHECK(r.boundary_sides == 4);
    CHECK(r.reoriented_triangles == 0);
    CHECK(r.unused_nodes_dropped == 0);
    REQUIRE(r.patches.size() == 4);
    CHECK(r.patches[2].name == "top wall");
    CHECK(r.patches[2].type == BoundaryType::Wall);
    CHECK(r.patches[2].matched == "wall");
    CHECK(r.patches[3].flag == 503);
    CHECK(r.patches[3].source == ConditionSource::Name);
    CHECK(r.patches[3].sides == 1);
    CHECK(r.warnings.empty());
    const std::string text = format_report(r);
    CHECK_THAT(text, ContainsSubstring("Boundary 'top wall' (tag 3): wall, flag 500, 1 sides [name]"));
}

TEST_CASE("build_mesh: the face normals of imported meshes point outwards", "[mesh_import]") {
    Mesh m = build(square());
    m.compute_face_normals();
    // bottom, right, top, left
    const std::vector<std::array<double, 2>> outward = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
    for (int i = 0; i < 4; ++i) {
        CAPTURE(i);
        CHECK(m.face_norm[static_cast<std::size_t>(i)].nx == outward[static_cast<std::size_t>(i)][0]);
        CHECK(m.face_norm[static_cast<std::size_t>(i)].ny == outward[static_cast<std::size_t>(i)][1]);
    }
}

TEST_CASE("build_mesh: arbitrary, sparse and huge node tags are remapped in file order", "[mesh_import]") {
    const MeshData ref = square();
    const std::map<std::int64_t, std::int64_t> retag = {
        {1, 9000000000000LL}, {2, -7}, {3, 0}, {4, 42}, {5, 3}};
    MeshData d = ref;
    for (auto& t : d.node_tags) t = retag.at(t);
    for (auto& e : d.elements)
        for (int k = 0; k < node_count(e.type); ++k) e.nodes[static_cast<std::size_t>(k)] = retag.at(e.nodes[static_cast<std::size_t>(k)]);
    const Mesh a = build(ref), b = build(d);
    CHECK(a.intma == b.intma);
    CHECK(a.x == b.x);
    CHECK(a.y == b.y);
    REQUIRE(a.nboun == b.nboun);
    for (int i = 0; i < a.nboun; ++i) {
        CHECK(a.iside[static_cast<std::size_t>(i)].node1 == b.iside[static_cast<std::size_t>(i)].node1);
        CHECK(a.iside[static_cast<std::size_t>(i)].node2 == b.iside[static_cast<std::size_t>(i)].node2);
        CHECK(a.iside[static_cast<std::size_t>(i)].elem == b.iside[static_cast<std::size_t>(i)].elem);
    }
}

TEST_CASE("build_mesh: nodes used by no triangle are dropped", "[mesh_import]") {
    MeshData d = square();
    // Unused nodes before, between and after the used ones.
    d.node_tags.insert(d.node_tags.begin(), 100);
    d.x.insert(d.x.begin(), 9.0);
    d.y.insert(d.y.begin(), 9.0);
    d.z.insert(d.z.begin(), 0.0);
    add_node(d, 101, 7.0, 7.0);
    ImportReport r;
    const Mesh m = build(d, {}, &r);
    require_valid(m);
    CHECK(m.npoin == 5);
    CHECK(r.nodes_in_file == 7);
    CHECK(r.unused_nodes_dropped == 2);
    CHECK(m.x == std::vector<double>{0.0, 1.0, 1.0, 0.0, 0.5});
    CHECK_THAT(format_report(r), ContainsSubstring("2 node(s) not used by any triangle dropped"));
}

TEST_CASE("build_mesh: clockwise triangles are reoriented, boundary sides follow", "[mesh_import]") {
    MeshData d = square();
    // Reverse triangles 0 and 2; reverse the direction of every line.
    std::swap(d.elements[0].nodes[1], d.elements[0].nodes[2]);
    std::swap(d.elements[2].nodes[0], d.elements[2].nodes[1]);
    for (auto& e : d.elements)
        if (e.type == ElementType::Line2) std::swap(e.nodes[0], e.nodes[1]);
    ImportReport r;
    const Mesh m = build(d, {}, &r);
    require_valid(m);
    CHECK(r.reoriented_triangles == 2);
    for (int e = 0; e < m.nelem; ++e) CHECK(signed_area(m, e) == 0.25);
    // Swap of the 2nd and 3rd node: the first node stays first.
    CHECK(m.intma[0] == std::array<int, 3>{0, 1, 4});
    CHECK(m.intma[2] == std::array<int, 3>{3, 4, 2});
    CHECK(side_shapes(m) == side_shapes(build(square())));
    CHECK_THAT(format_report(r), ContainsSubstring("2 clockwise triangle(s) reoriented"));
}

TEST_CASE("build_mesh: elements in several physical groups are merged", "[mesh_import]") {
    MeshData d = square();
    add_tri(d, 1, 2, 5, 77);      // same triangle, other surface group
    add_tri(d, 5, 1, 2, 78);      // same triangle, rotated
    add_line(d, 2, 1, 9);         // bottom line again, reversed, group "noSlip"
    add_line(d, 1, 2, 9);         // and once more
    d.physical_names[{1, 9}] = "noSlip";
    ImportReport r;
    const Mesh m = build(d, {}, &r);
    require_valid(m);
    CHECK(m.nelem == 4);
    CHECK(m.nboun == 4);
    CHECK(r.duplicate_elements_merged == 4);
    CHECK(m.iside[0].side_id == 1); // smallest tag of the agreeing groups 1 and 9
    CHECK(m.iside[0].flag == 500);
    REQUIRE(r.patches.size() == 5);
    CHECK(r.patches[4].tag == 9);
    CHECK(r.patches[4].sides == 1);
}

TEST_CASE("build_mesh: a mesh in the plane z = const is accepted", "[mesh_import]") {
    MeshData d = square();
    for (double& z : d.z) z = 5.0;
    d.z[2] = 5.0 + 1e-12;
    require_valid(build(d));
}

TEST_CASE("build_mesh: holes get boundary sides oriented around the hole", "[mesh_import]") {
    // 3 x 3 cells without the centre cell; outer boundary "wall", hole "cylinder_wall".
    const MeshData d = grid(3, 3, [](int i, int j) { return i == 1 && j == 1; },
                            [](double x, double y) { return (x > 0.0 && x < 3.0 && y > 0.0 && y < 3.0) ? 2 : 1; });
    MeshData named = d;
    named.physical_names[{1, 1}] = "outer_wall";
    named.physical_names[{1, 2}] = "cylinder_wall";
    Mesh m = build(named);
    require_valid(m);
    CHECK(m.nboun == 12 + 4);
    m.compute_face_normals();
    for (int i = 0; i < m.nboun; ++i) {
        const BoundarySide& s = m.iside[static_cast<std::size_t>(i)];
        const double xm = 0.5 * (m.x[s.node1] + m.x[s.node2]), ym = 0.5 * (m.y[s.node1] + m.y[s.node2]);
        const double dot = m.face_norm[static_cast<std::size_t>(i)].nx * (xm - 1.5) +
                           m.face_norm[static_cast<std::size_t>(i)].ny * (ym - 1.5);
        CAPTURE(i, xm, ym);
        // Outwards from the fluid: away from the centre on the outer boundary,
        // towards the centre (into the hole) on the hole boundary.
        CHECK((s.side_id == 1 ? dot > 0.0 : dot < 0.0));
    }
}

TEST_CASE("build_mesh: random numbering, orientation and line direction give the same mesh", "[mesh_import]") {
    const MeshData ref_data = grid(6, 4, [](int i, int j) { return i == 2 && j >= 1 && j <= 2; },
                                   [](double x, double) { return x == 0.0 ? 1 : (x == 6.0 ? 2 : 3); });
    MeshData named = ref_data;
    named.physical_names = {{{1, 1}, "inlet"}, {{1, 2}, "outlet"}, {{1, 3}, "wall"}};
    const Mesh ref = build(named);
    require_valid(ref);

    std::mt19937 rng(12345);
    for (int trial = 0; trial < 20; ++trial) {
        MeshData d = named;
        // Random tags (unique, sparse, possibly negative) and random node order.
        std::vector<std::size_t> perm(d.node_tags.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        std::map<std::int64_t, std::int64_t> retag;
        std::set<std::int64_t> used;
        std::uniform_int_distribution<std::int64_t> tags(-1000000, 1000000000000LL);
        for (std::int64_t t : d.node_tags) {
            std::int64_t nt;
            do nt = tags(rng);
            while (!used.insert(nt).second);
            retag[t] = nt;
        }
        MeshData s = d;
        for (std::size_t i = 0; i < perm.size(); ++i) {
            s.node_tags[i] = retag.at(d.node_tags[perm[i]]);
            s.x[i] = d.x[perm[i]];
            s.y[i] = d.y[perm[i]];
        }
        std::shuffle(s.elements.begin(), s.elements.end(), rng);
        for (auto& e : s.elements) {
            for (int k = 0; k < node_count(e.type); ++k) e.nodes[static_cast<std::size_t>(k)] = retag.at(e.nodes[static_cast<std::size_t>(k)]);
            if (rng() % 2) std::swap(e.nodes[0], e.nodes[1]); // flips triangles, reverses lines
        }
        ImportReport r;
        const Mesh m = build(s, {}, &r);
        CAPTURE(trial);
        require_valid(m);
        CHECK(m.npoin == ref.npoin);
        CHECK(element_shapes(m) == element_shapes(ref));
        CHECK(side_shapes(m) == side_shapes(ref));
        CHECK(r.reoriented_triangles > 0);
    }
}

TEST_CASE("build_mesh: lines inside the domain without a condition are ignored", "[mesh_import]") {
    MeshData d = square();
    add_line(d, 1, 5, 0);     // diagonal, no group
    add_line(d, 2, 5, 30);    // diagonal, unnamed group without *.bofl entry
    add_line(d, 3, 5, 31);    // diagonal, group without keyword
    d.physical_names[{1, 31}] = "refinement line";
    ImportReport r;
    const Mesh m = build(d, {}, &r);
    require_valid(m);
    CHECK(m.nboun == 4);
    CHECK(r.interior_lines_ignored == 3);
    CHECK_THAT(format_report(r), ContainsSubstring("3 line element(s) inside the domain without a boundary condition ignored"));
}

TEST_CASE("build_mesh: a group without keyword is ignored if other groups define its sides", "[mesh_import]") {
    MeshData d = square();
    for (auto [a, b] : {std::pair{1, 2}, {2, 3}, {3, 4}, {4, 1}}) add_line(d, a, b, 50);
    d.physical_names[{1, 50}] = "all boundaries";
    ImportReport r;
    const Mesh m = build(d, {}, &r);
    require_valid(m);
    CHECK(m.iside[1].flag == 504);
    REQUIRE(r.warnings.size() == 1);
    CHECK_THAT(r.warnings[0], ContainsSubstring("'all boundaries' (tag 50)") && ContainsSubstring("ignored"));
}

// =============================================================================
// Boundary conditions: *.bofl overrides, unimplemented types
// =============================================================================

TEST_CASE("build_mesh: *.bofl entries take precedence over names", "[mesh_import][bofl]") {
    MeshData d = square();
    d.physical_names.erase({1, 4});             // left side unnamed
    d.physical_names[{1, 2}] = "airfoil";        // no keyword
    BuildOptions o;
    o.flag_overrides = {{4, 507}, {2, 504}, {1, 506}, {99, 500}}; // 1 overrides the name "bottom_wall"
    ImportReport r;
    const Mesh m = build(d, o, &r);
    CHECK(m.iside[0].flag == 506);
    CHECK(m.iside[1].flag == 504);
    CHECK(m.iside[2].flag == 500);
    CHECK(m.iside[3].flag == 507);
    CHECK(r.patches[0].source == ConditionSource::FlagFile);
    CHECK(r.patches[0].type == BoundaryType::Symmetry);
    CHECK(r.patches[2].source == ConditionSource::Name);
    REQUIRE(r.warnings.size() == 1);
    CHECK_THAT(r.warnings[0], ContainsSubstring("physical tag 99 matches no boundary line group"));
}

TEST_CASE("build_mesh: unknown *.bofl codes are kept with a warning (legacy behaviour)", "[mesh_import][bofl]") {
    BuildOptions o;
    o.flag_overrides = {{3, 505}};
    ImportReport r;
    const Mesh m = build(square(), o, &r);
    CHECK(m.iside[2].flag == 505);
    CHECK(r.patches[2].type == BoundaryType::Custom);
    REQUIRE(r.warnings.size() == 1);
    CHECK_THAT(r.warnings[0], ContainsSubstring("flag code 505"));
}

TEST_CASE("build_mesh: recognised but unimplemented boundary types are rejected", "[mesh_import]") {
    MeshData d = square();
    d.physical_names[{1, 1}] = "periodic_bottom";
    d.physical_names[{1, 3}] = "periodic_top";
    d.physical_names[{1, 2}] = "farfield";
    const std::string err = build_error(d);
    CHECK_THAT(err, ContainsSubstring("3 problems"));
    CHECK_THAT(err, ContainsSubstring("'periodic_bottom' (tag 1) is a 'periodic' boundary"));
    CHECK_THAT(err, ContainsSubstring("'farfield' (tag 2) is a 'farfield' boundary"));
    CHECK_THAT(err, ContainsSubstring("does not implement yet"));

    BuildOptions o;
    o.allow_unimplemented_boundaries = true;
    ImportReport r;
    const Mesh m = build(d, o, &r);
    CHECK(m.iside[0].flag == 0);
    CHECK(m.iside[1].flag == 0);
    CHECK(m.iside[3].flag == 503);
    CHECK(r.patches[0].type == BoundaryType::Periodic);
    CHECK(r.warnings.size() == 3);
}

TEST_CASE("build_mesh: boundary name problems are reported once per group", "[mesh_import][errors]") {
    MeshData d = square();
    d.physical_names[{1, 1}] = "airfoil";
    d.physical_names[{1, 2}] = "inlet_wall";
    d.physical_names.erase({1, 3});
    d.physical_names[{1, 4}] = "505";
    add_line(d, 1, 2, 1); // bottom twice: still one message
    const std::string err = build_error(d);
    CHECK_THAT(err, ContainsSubstring("square.msh: invalid mesh (4 problems)"));
    CHECK_THAT(err, ContainsSubstring("'airfoil' (tag 1)") && ContainsSubstring("no boundary keyword"));
    CHECK_THAT(err, ContainsSubstring("'inlet_wall' (tag 2)") && ContainsSubstring("ambiguous"));
    CHECK_THAT(err, ContainsSubstring("physical group 3 (unnamed)") && ContainsSubstring("*.bofl"));
    CHECK_THAT(err, ContainsSubstring("'505' (tag 4)") && ContainsSubstring("not a known flag code"));
}

TEST_CASE("build_mesh: groups with different conditions on the same side", "[mesh_import][errors]") {
    MeshData d = square();
    add_line(d, 1, 2, 8);
    d.physical_names[{1, 8}] = "outlet_bottom";
    CHECK_THAT(build_error(d), ContainsSubstring("'bottom_wall' (tag 1) (wall) and in physical group 'outlet_bottom' "
                                                 "(tag 8) (outlet), which define different boundary conditions"));
}

// =============================================================================
// Invalid meshes
// =============================================================================

TEST_CASE("build_mesh: boundary edges without a line or group", "[mesh_import][errors]") {
    MeshData d = square();
    d.elements.pop_back(); // no line on the left side
    CHECK_THAT(build_error(d), ContainsSubstring("1 boundary edge has no line element, e.g. node 4 (0, 1) - node 1 (0, 0)"));

    MeshData u = square();
    u.elements.back().physical = 0; // left line in no group
    u.elements[u.elements.size() - 2].physical = 0;
    CHECK_THAT(build_error(u), ContainsSubstring("2 boundary line elements are in no physical group"));
}

TEST_CASE("build_mesh: lines that are not edges of the mesh", "[mesh_import][errors]") {
    MeshData d = square();
    add_line(d, 1, 3, 1, 500); // diagonal 1-3 is not an edge
    CHECK_THAT(build_error(d), ContainsSubstring("line element 500 (node 1 (0, 0) - node 3 (1, 1)) is not an edge of any triangle"));

    MeshData e = square();
    add_node(e, 6, 2.0, 2.0);
    add_line(e, 3, 6, 1, 501); // node 6 belongs to no triangle
    CHECK_THAT(build_error(e), ContainsSubstring("line element 501") && ContainsSubstring("not an edge"));

    MeshData f = square();
    add_line(f, 2, 2, 1, 502);
    CHECK_THAT(build_error(f), ContainsSubstring("line element 502 has two identical nodes"));

    MeshData g = square();
    add_line(g, 1, 5, 1, 503); // interior edge in a wall group
    CHECK_THAT(build_error(g), ContainsSubstring("line element 503") && ContainsSubstring("(wall) lies inside the domain") &&
                                   ContainsSubstring("internal boundaries are not supported"));
}

TEST_CASE("build_mesh: invalid nodes and elements", "[mesh_import][errors]") {
    MeshData dup = square();
    add_node(dup, 3, 4.0, 4.0);
    CHECK_THAT(build_error(dup), ContainsSubstring("duplicate node tag 3"));

    MeshData nan = square();
    nan.y[1] = std::nan("");
    CHECK_THAT(build_error(nan), ContainsSubstring("node 2 has a non-finite coordinate"));

    MeshData undef = square();
    add_tri(undef, 1, 2, 77, 0, 900);
    CHECK_THAT(build_error(undef), ContainsSubstring("triangle 900 references undefined node 77"));

    MeshData rep = square();
    add_tri(rep, 1, 2, 2, 0, 901);
    CHECK_THAT(build_error(rep), ContainsSubstring("triangle 901 has repeated nodes"));

    MeshData empty = square();
    empty.elements.erase(empty.elements.begin(), empty.elements.begin() + 4);
    CHECK_THAT(build_error(empty), ContainsSubstring("the mesh contains no triangles"));

    // A quadrilateral on top of the triangles: the elements overlap.
    MeshData quad = square();
    RawElement q;
    q.type = ElementType::Quad4;
    q.nodes = {1, 2, 3, 4};
    q.tag = 902;
    quad.elements.push_back(q);
    quad.elements.push_back(q); // the duplicate is merged first
    CHECK_THAT(build_error(quad), ContainsSubstring("triangle 1 and quadrilateral 902 overlap"));
    MeshData repeated = square();
    q.nodes = {1, 2, 2, 4};
    q.tag = 903;
    repeated.elements.push_back(q);
    CHECK_THAT(build_error(repeated), ContainsSubstring("quadrilateral 903 has repeated nodes"));

    MeshData bent = square();
    bent.z[4] = 0.1;
    CHECK_THAT(build_error(bent), ContainsSubstring("not planar in the x-y plane: z ranges from 0 to 0.1"));

    MeshData flat;
    flat.source = "flat.msh";
    add_node(flat, 1, 0.0, 0.0);
    add_node(flat, 2, 0.0, 0.0, 1.0);
    add_node(flat, 3, 0.0, 0.0, 2.0);
    add_tri(flat, 1, 2, 3);
    CHECK_THAT(build_error(flat), ContainsSubstring("no extent in the x-y plane"));
}

TEST_CASE("build_mesh: degenerate, overlapping and non-manifold triangles", "[mesh_import][errors]") {
    MeshData degenerate = square();
    add_node(degenerate, 6, 2.0, 0.0);
    add_node(degenerate, 7, 3.0, 0.0);
    add_tri(degenerate, 2, 6, 7, 0, 910); // collinear
    CHECK_THAT(build_error(degenerate), ContainsSubstring("triangle 910 is degenerate (zero area)"));

    // Sliver with aspect ratio 1e-9 is fine; 1e-14 is degenerate.
    for (const double h : {1e-9, 1e-14}) {
        MeshData s;
        add_node(s, 1, 0.0, 0.0);
        add_node(s, 2, 1.0, 0.0);
        add_node(s, 3, 0.5, h);
        add_tri(s, 1, 2, 3);
        add_line(s, 1, 2, 1);
        add_line(s, 2, 3, 1);
        add_line(s, 3, 1, 1);
        s.physical_names[{1, 1}] = "wall";
        CAPTURE(h);
        if (h > 1e-12) {
            require_valid(build(s));
        } else {
            CHECK_THAT(build_error(s), ContainsSubstring("degenerate"));
        }
    }

    MeshData overlap = square();
    add_node(overlap, 6, 0.5, 0.2);
    add_tri(overlap, 1, 2, 6, 0, 920); // on the same side of edge 1-2 as triangle 1
    CHECK_THAT(build_error(overlap), ContainsSubstring("triangles 1 and 920 overlap"));

    MeshData nonmanifold = square();
    add_node(nonmanifold, 6, 0.5, 0.2);
    add_node(nonmanifold, 7, 0.5, -0.5);
    add_tri(nonmanifold, 2, 1, 7, 0, 930);
    add_tri(nonmanifold, 1, 2, 6, 0, 931);
    CHECK_THAT(build_error(nonmanifold), ContainsSubstring("shared by more than two triangles"));
}

TEST_CASE("build_mesh: many problems are listed up to a limit", "[mesh_import][errors]") {
    MeshData d = square();
    for (int k = 0; k < 25; ++k) add_tri(d, 1, 2, 1000 + k, 0, 2000 + k);
    const std::string err = build_error(d);
    CHECK_THAT(err, ContainsSubstring("invalid mesh (25 problems)"));
    CHECK_THAT(err, ContainsSubstring("references undefined node 1000"));
    CHECK_THAT(err, ContainsSubstring("... and 15 more"));
    BuildOptions o;
    o.max_reported_problems = 3;
    CHECK_THAT(build_error(d, o), ContainsSubstring("... and 22 more"));
}

// =============================================================================
// check_mesh
// =============================================================================

TEST_CASE("check_mesh accepts the legacy meshes and test meshes", "[mesh_import][check]") {
    Mesh legacy;
    {
        test::CaptureStdout quiet;
        REQUIRE(IO::read_mesh(std::string(HYDRAMA2D_CASE_DIR) + "/constant/hinterkante.mesh", legacy));
    }
    require_valid(legacy);

    test::GridSpec g;
    g.nx = 5;
    g.ny = 3;
    require_valid(test::make_grid_mesh(g));
}

TEST_CASE("check_mesh detects every violated invariant", "[mesh_import][check]") {
    const Mesh good = build(square());
    require_valid(good);
    auto problems = [](const Mesh& m) {
        const auto p = check_mesh(m);
        std::string all;
        for (const auto& s : p) all += s + "\n";
        return all;
    };

    Mesh m = good;
    m.nboun = 3;
    CHECK_THAT(problems(m), ContainsSubstring("array sizes"));

    m = good;
    std::swap(m.intma[1][1], m.intma[1][2]);
    CHECK_THAT(problems(m), ContainsSubstring("element 1 is not counter-clockwise"));

    m = good;
    m.intma[0][2] = 99;
    CHECK_THAT(problems(m), ContainsSubstring("out of range"));

    m = good;
    std::swap(m.iside[2].node1, m.iside[2].node2);
    CHECK_THAT(problems(m), ContainsSubstring("boundary side 2 is not oriented counter-clockwise"));

    m = good;
    m.iside[1].elem = 0;
    CHECK_THAT(problems(m), ContainsSubstring("boundary side 1 is not an edge of its element 0"));

    m = good;
    m.iside.pop_back();
    m.nboun = 3;
    CHECK_THAT(problems(m), ContainsSubstring("1 boundary edge(s) are not covered"));

    m = good;
    m.iside.push_back(m.iside[0]);
    m.nboun = 5;
    CHECK_THAT(problems(m), ContainsSubstring("duplicates another boundary side"));

    m = good;
    m.iside.push_back({0, 4, 0, 9, 500}); // interior edge 0-4 of element 0
    m.nboun = 5;
    CHECK_THAT(problems(m), ContainsSubstring("lies on an interior edge"));
}
