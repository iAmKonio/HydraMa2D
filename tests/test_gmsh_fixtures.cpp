// Files written by Gmsh itself (tests/data/gmsh/, see generate_fixtures.py):
// all four format variants of the same mesh, clockwise triangles, parametric
// nodes, and the files HydraMa2D must reject.

#include "GmshTestWriter.hpp"
#include "mesh/GmshReader.hpp"
#include "mesh/MeshBuilder.hpp"
#include "mesh/MeshCheck.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <map>

using namespace hydrama;
using namespace hydrama::mesh;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

const double kPi = std::acos(-1.0);

std::string fixture(const std::string& name) { return std::string(HYDRAMA2D_TEST_DATA_DIR) + "/gmsh/" + name; }

Mesh import(const std::string& name, const BuildOptions& o = {}, ImportReport* r = nullptr) {
    return build_mesh(read_gmsh(fixture(name)), o, r);
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

// Area enclosed by the boundary sides (shoelace formula). Equals the mesh
// area only if the sides form closed loops oriented with the domain on the
// left (counter-clockwise outside, clockwise around holes).
double boundary_area(const Mesh& m) {
    double a = 0.0;
    for (const auto& s : m.iside) a += 0.5 * (m.x[s.node1] * m.y[s.node2] - m.x[s.node2] * m.y[s.node1]);
    return a;
}

const char* const kChannelVariants[] = {"channel_cylinder_2.2_ascii.msh", "channel_cylinder_2.2_binary.msh",
                                        "channel_cylinder_4.1_ascii.msh", "channel_cylinder_4.1_binary.msh"};

} // namespace

TEST_CASE("Gmsh fixtures: channel with cylinder, every format variant", "[gmsh][fixtures]") {
    for (const char* name : kChannelVariants) {
        CAPTURE(name);
        ImportReport r;
        Mesh m = import(name, {}, &r);
        require_valid(m);
        CHECK(m.nelem == 246);
        CHECK(m.npoin == 151);
        CHECK(m.nboun == 56);
        CHECK(r.reoriented_triangles == 0);
        CHECK(r.duplicate_elements_merged == 16); // the 16 cylinder lines are also in "noSlip"
        CHECK(r.warnings.empty());

        std::map<std::string, std::pair<int, int>> patches; // name -> (flag, sides)
        for (const BoundaryPatch& p : r.patches) patches[p.name] = {p.flag, p.sides};
        CHECK(patches == std::map<std::string, std::pair<int, int>>{{"inlet", {503, 4}},
                                                                    {"outlet", {504, 4}},
                                                                    {"Symmetry Plane", {506, 32}},
                                                                    {"cylinderWall", {500, 16}},
                                                                    {"noSlip", {500, 16}}});

        // Mesh::boundary_groups: every group with its sides; the cylinder
        // sides are in two groups (side_id = the smaller tag).
        std::map<std::string, std::vector<int>> groups;
        for (const BoundaryGroup& g : m.boundary_groups) {
            CHECK(std::is_sorted(g.sides.begin(), g.sides.end()));
            groups[g.name] = g.sides;
        }
        REQUIRE(groups.size() == 5);
        CHECK(groups["inlet"].size() == 4);
        CHECK(groups["Symmetry Plane"].size() == 32);
        CHECK(groups["cylinderWall"] == groups["noSlip"]);
        for (int ib : groups["noSlip"]) CHECK(m.iside[static_cast<std::size_t>(ib)].side_id == 4);

        // The mesh covers the channel minus the (polygonal) cylinder, and the
        // boundary sides enclose exactly that area.
        const double area = total_area(m);
        CHECK_THAT(area, WithinRel(boundary_area(m), 1e-12));
        CHECK(area > 4.0 - kPi * 0.04);
        CHECK(area < 4.0 - kPi * 0.04 + 0.01);

        // Outward normals: -x at the inlet, +x at the outlet, +-y on the
        // symmetry lines, towards the cylinder centre on the cylinder.
        m.compute_face_normals();
        for (int i = 0; i < m.nboun; ++i) {
            const BoundarySide& s = m.iside[static_cast<std::size_t>(i)];
            const FaceNormal& n = m.face_norm[static_cast<std::size_t>(i)];
            const double xm = 0.5 * (m.x[s.node1] + m.x[s.node2]), ym = 0.5 * (m.y[s.node1] + m.y[s.node2]);
            CAPTURE(i, s.flag, xm, ym);
            switch (s.flag) {
            case 503: CHECK((n.nx == -1.0 && n.ny == 0.0)); break;
            case 504: CHECK((n.nx == 1.0 && n.ny == 0.0)); break;
            case 506: CHECK(n.ny == (ym > 0.5 ? 1.0 : -1.0)); break;
            case 500: CHECK(n.nx * (1.0 - xm) + n.ny * (0.5 - ym) > 0.0); break;
            default: FAIL("unexpected flag");
            }
        }
    }
}

TEST_CASE("Gmsh fixtures: ASCII and binary files of the same mesh agree", "[gmsh][fixtures]") {
    // Gmsh writes ASCII coordinates with 16 significant digits, binary ones exactly.
    const Mesh ref = import("channel_cylinder_4.1_binary.msh");
    for (const char* name : kChannelVariants) {
        CAPTURE(name);
        const Mesh m = import(name);
        CHECK(m.intma == ref.intma);
        REQUIRE(m.npoin == ref.npoin);
        for (int i = 0; i < m.npoin; ++i) {
            CHECK_THAT(m.x[static_cast<std::size_t>(i)], WithinAbs(ref.x[static_cast<std::size_t>(i)], 1e-15));
            CHECK_THAT(m.y[static_cast<std::size_t>(i)], WithinAbs(ref.y[static_cast<std::size_t>(i)], 1e-15));
        }
        REQUIRE(m.nboun == ref.nboun);
        for (int i = 0; i < m.nboun; ++i) {
            const BoundarySide &a = m.iside[static_cast<std::size_t>(i)], &b = ref.iside[static_cast<std::size_t>(i)];
            CHECK(std::array<int, 5>{a.node1, a.node2, a.elem, a.side_id, a.flag} ==
                  std::array<int, 5>{b.node1, b.node2, b.elem, b.side_id, b.flag});
        }
    }
}

TEST_CASE("Gmsh fixtures: meshes round-trip through the test writer", "[gmsh][fixtures]") {
    const Mesh ref = import("channel_cylinder_4.1_binary.msh");
    const std::map<int, std::string> names = {{1, "inlet"}, {2, "outlet"}, {3, "Symmetry Plane"}, {4, "cylinderWall"}};
    for (const std::string version : {"2.2", "4.1"}) {
        for (const bool binary : {false, true}) {
            test::GmshWriteOptions o;
            o.version = version;
            o.binary = binary;
            CAPTURE(version, binary);
            const Mesh m = build_mesh(parse_gmsh(test::write_gmsh(test::gmsh_model_from_mesh(ref, names, 10), o), "rt.msh"));
            CHECK(m.intma == ref.intma);
            CHECK(m.x == ref.x);
            CHECK(m.y == ref.y);
            REQUIRE(m.nboun == ref.nboun);
            for (int i = 0; i < m.nboun; ++i) {
                CHECK(m.iside[static_cast<std::size_t>(i)].node1 == ref.iside[static_cast<std::size_t>(i)].node1);
                CHECK(m.iside[static_cast<std::size_t>(i)].flag == ref.iside[static_cast<std::size_t>(i)].flag);
            }
        }
    }
}

TEST_CASE("Gmsh fixtures: clockwise triangles from a reversed surface", "[gmsh][fixtures]") {
    const MeshData raw = read_gmsh(fixture("square_clockwise_2.2_ascii.msh"));
    ImportReport r;
    const Mesh m = build_mesh(raw, {}, &r);
    require_valid(m);
    CHECK(r.reoriented_triangles == m.nelem); // Gmsh wrote every triangle clockwise
    CHECK(m.nelem == 26);
    CHECK_THAT(total_area(m), WithinRel(1.0, 1e-12));
    CHECK_THAT(boundary_area(m), WithinRel(1.0, 1e-12));
}

TEST_CASE("Gmsh fixtures: parametric coordinates do not change the mesh", "[gmsh][fixtures]") {
    const Mesh plain = import("square_4.1_ascii.msh");
    const Mesh param = import("square_parametric_4.1_ascii.msh");
    require_valid(param);
    CHECK(param.intma == plain.intma);
    CHECK(param.nboun == plain.nboun);
    REQUIRE(param.npoin == plain.npoin);
    for (int i = 0; i < plain.npoin; ++i) {
        CHECK_THAT(param.x[static_cast<std::size_t>(i)], WithinAbs(plain.x[static_cast<std::size_t>(i)], 1e-12));
        CHECK_THAT(param.y[static_cast<std::size_t>(i)], WithinAbs(plain.y[static_cast<std::size_t>(i)], 1e-12));
    }
}

TEST_CASE("Gmsh fixtures: unnamed groups need *.bofl entries", "[gmsh][fixtures][bofl]") {
    CHECK_THAT(import_error("square_unnamed_4.1_binary.msh"),
               ContainsSubstring("4 problems") && ContainsSubstring("physical group 21 (unnamed)"));
    BuildOptions o;
    o.flag_overrides = {{21, 500}, {22, 504}, {23, 506}, {24, 507}};
    ImportReport r;
    const Mesh m = import("square_unnamed_4.1_binary.msh", o, &r);
    require_valid(m);
    std::map<int, int> flags;
    for (const auto& s : m.iside) flags[s.side_id] = s.flag;
    CHECK(flags == std::map<int, int>{{21, 500}, {22, 504}, {23, 506}, {24, 507}});
}

TEST_CASE("Gmsh fixtures: quadrilateral and hybrid meshes", "[gmsh][fixtures][quad]") {
    struct Case {
        const char* file;
        int triangles, quadrilaterals, nodes, sides;
        double area;
    };
    // Areas: unit square; channel 4 x 1; channel minus the 16-gon of radius 0.2.
    const double cylinder = 4.0 - 0.5 * 16 * 0.04 * std::sin(2.0 * kPi / 16);
    for (const Case& c : {Case{"square_quads_2.2_ascii.msh", 0, 21, 30, 16, 1.0},          // recombined (unstructured)
                          Case{"channel_quads_2.2_ascii.msh", 0, 80, 102, 42, 4.0},        // transfinite
                          Case{"cylinder_inflation_4.1_binary.msh", 318, 48, 240, 66, cylinder}}) { // BoundaryLayer
        CAPTURE(c.file);
        ImportReport r;
        Mesh m = import(c.file, {}, &r);
        require_valid(m);
        CHECK(m.num_triangles() == c.triangles);
        CHECK(m.num_quads() == c.quadrilaterals);
        CHECK(m.nelem == c.triangles + c.quadrilaterals);
        CHECK(m.npoin == c.nodes);
        CHECK(m.nboun == c.sides);
        CHECK(r.quadrilaterals == c.quadrilaterals);
        CHECK_THAT(boundary_area(m), WithinRel(c.area, 1e-9));
        double area = 0.0;
        for (const auto& q : m.quads) {
            double qx[4], qy[4];
            for (int a = 0; a < 4; ++a) {
                qx[a] = m.x[q[static_cast<std::size_t>(a)]];
                qy[a] = m.y[q[static_cast<std::size_t>(a)]];
            }
            area += q1::compute_geometry(qx, qy).area;
        }
        for (const auto& t : m.intma)
            area += 0.5 * ((m.x[t[1]] - m.x[t[0]]) * (m.y[t[2]] - m.y[t[0]]) - (m.x[t[2]] - m.x[t[0]]) * (m.y[t[1]] - m.y[t[0]]));
        CHECK_THAT(area, WithinRel(c.area, 1e-9));
    }
    // The inflation layer: 3 layers of 16 quadrilaterals around the cylinder,
    // the cylinder wall sides belong to quadrilaterals.
    ImportReport r;
    const Mesh m = import("cylinder_inflation_4.1_binary.msh", {}, &r);
    int wall_sides_on_quads = 0;
    for (const auto& s : m.iside)
        if (s.flag == 500 && m.is_quad(s.elem)) ++wall_sides_on_quads;
    CHECK(wall_sides_on_quads == 16);
    CHECK_THAT(format_report(r), ContainsSubstring("Mesh import: 318 triangles, 48 quadrilaterals, 240 nodes, 66 boundary sides"));
}

TEST_CASE("Gmsh fixtures: files HydraMa2D rejects", "[gmsh][fixtures][errors]") {
    CHECK_THAT(import_error("square_order2_4.1_ascii.msh"),
               ContainsSubstring("second-order") && ContainsSubstring("Mesh.ElementOrder = 1"));
    CHECK_THAT(import_error("cube_tet_4.1_ascii.msh"), ContainsSubstring("3D element"));
    CHECK_THAT(import_error("square_nogroups_4.1_ascii.msh"),
               ContainsSubstring("12 boundary line elements are in no physical group"));
}
