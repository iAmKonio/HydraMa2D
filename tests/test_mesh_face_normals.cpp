// Boundary face normals: Mesh::compute_face_normals() stores, for every
// boundary side node1 -> node2, the unit normal (dy, -dx) / L and the length L.
// For sides traversed with the domain on their left this is the outward normal.

#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

Mesh single_side_mesh(Point a, Point b) {
    // The element is irrelevant for the normal; only node coordinates matter.
    return make_mesh({a, b, {a.x - 1.0, a.y + 3.0}}, {{0, 1, 2}}, {{0, 1, 0, 1, 500}});
}

FaceNormal normal_of_side(Point a, Point b) {
    Mesh m = single_side_mesh(a, b);
    m.compute_face_normals();
    REQUIRE(m.face_norm.size() == 1);
    return m.face_norm[0];
}

} // namespace

TEST_CASE("Face normals of the unit square point outward", "[mesh][normals]") {
    GridSpec g; // unit square, one cell, sides: bottom, right, top, left
    Mesh m = make_grid_mesh(g);
    m.compute_face_normals();
    REQUIRE(m.face_norm.size() == 4);

    const double expected[4][2] = {{0.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}};
    for (int ib = 0; ib < 4; ++ib) {
        CAPTURE(ib, m.iside[ib].side_id);
        CHECK_THAT(m.face_norm[ib].nx, WithinAbs(expected[ib][0], 1e-15));
        CHECK_THAT(m.face_norm[ib].ny, WithinAbs(expected[ib][1], 1e-15));
        CHECK_THAT(m.face_norm[ib].length, WithinRel(1.0, 1e-15));
    }
}

TEST_CASE("Face normal of a diagonal side", "[mesh][normals]") {
    const FaceNormal n = normal_of_side({0.0, 0.0}, {1.0, 1.0});
    const double r = 1.0 / std::sqrt(2.0);
    CHECK_THAT(n.nx, WithinAbs(r, 1e-15));
    CHECK_THAT(n.ny, WithinAbs(-r, 1e-15));
    CHECK_THAT(n.length, WithinRel(std::sqrt(2.0), 1e-15));
}

TEST_CASE("Face normal is a unit vector perpendicular to the side", "[mesh][normals]") {
    struct Side { Point a, b; };
    const Side side = GENERATE(Side{{0.0, 0.0}, {3.0, 4.0}},
                               Side{{-1.0, 2.0}, {-1.0, -5.0}},
                               Side{{2.5, 2.5}, {-0.5, 2.5}},
                               Side{{1.0e-3, 2.0e-3}, {1.3e-3, 1.6e-3}},
                               Side{{100.0, -40.0}, {99.0, -41.5}});
    CAPTURE(side.a.x, side.a.y, side.b.x, side.b.y);

    const FaceNormal n = normal_of_side(side.a, side.b);
    const double dx = side.b.x - side.a.x;
    const double dy = side.b.y - side.a.y;
    const double len = std::hypot(dx, dy);

    CHECK_THAT(n.length, WithinRel(len, 1e-14));
    CHECK_THAT(n.nx * n.nx + n.ny * n.ny, WithinAbs(1.0, 1e-14));
    CHECK_THAT(n.nx * dx + n.ny * dy, WithinAbs(0.0, 1e-14 * len));
    // Right-hand normal: tangent x normal has negative z component.
    CHECK(dx * n.ny - dy * n.nx < 0.0);
}

TEST_CASE("Reversing a side's node order flips its normal", "[mesh][normals]") {
    const Point a{0.3, -1.2}, b{2.1, 0.7};
    const FaceNormal fwd = normal_of_side(a, b);
    const FaceNormal rev = normal_of_side(b, a);

    CHECK_THAT(rev.nx, WithinAbs(-fwd.nx, 1e-15));
    CHECK_THAT(rev.ny, WithinAbs(-fwd.ny, 1e-15));
    CHECK_THAT(rev.length, WithinRel(fwd.length, 1e-15));
}

TEST_CASE("Face normals point away from the owning element", "[mesh][normals]") {
    GridSpec g;
    g.lx = 2.0;
    g.ly = 1.0;
    g.nx = 5;
    g.ny = 3;
    Mesh m = make_grid_mesh(g);
    m.compute_face_normals();
    REQUIRE(static_cast<int>(m.face_norm.size()) == g.nboun());

    for (int ib = 0; ib < m.nboun; ++ib) {
        const BoundarySide& s = m.iside[ib];
        // Vertex of the owning element that is not on the side.
        int opposite = -1;
        for (int node : m.intma[s.elem]) {
            if (node != s.node1 && node != s.node2) opposite = node;
        }
        CAPTURE(ib, s.elem, opposite);
        REQUIRE(opposite >= 0);

        const double mx = 0.5 * (m.x[s.node1] + m.x[s.node2]);
        const double my = 0.5 * (m.y[s.node1] + m.y[s.node2]);
        const double outward = m.face_norm[ib].nx * (mx - m.x[opposite]) +
                               m.face_norm[ib].ny * (my - m.y[opposite]);
        CHECK(outward > 0.0);
    }
}

TEST_CASE("Face normals of a closed boundary satisfy the divergence theorem", "[mesh][normals]") {
    // For a closed boundary:  sum n L == 0  and  sum x_mid n_x L == area.
    SECTION("single triangles") {
        const Triangle tri = GENERATE(from_range(reference_triangles()));
        CAPTURE(tri);
        Mesh m = make_triangle_mesh(tri);
        compute_geometry(m);

        double snx = 0.0, sny = 0.0, flux = 0.0, perimeter = 0.0;
        for (int ib = 0; ib < m.nboun; ++ib) {
            const FaceNormal& n = m.face_norm[ib];
            const double xm = 0.5 * (m.x[m.iside[ib].node1] + m.x[m.iside[ib].node2]);
            snx += n.nx * n.length;
            sny += n.ny * n.length;
            flux += xm * n.nx * n.length;
            perimeter += n.length;
        }
        CHECK_THAT(snx, WithinAbs(0.0, 1e-12 * perimeter));
        CHECK_THAT(sny, WithinAbs(0.0, 1e-12 * perimeter));
        CHECK_THAT(flux, WithinRel(0.5 * m.detJ[0], 1e-9));
    }

    SECTION("structured grid") {
        GridSpec g;
        g.x0 = 1.0;
        g.y0 = -2.0;
        g.lx = 3.0;
        g.ly = 2.5;
        g.nx = 6;
        g.ny = 5;
        Mesh m = make_grid_mesh(g);
        m.compute_face_normals();

        double snx = 0.0, sny = 0.0, flux = 0.0, perimeter = 0.0;
        for (int ib = 0; ib < m.nboun; ++ib) {
            const FaceNormal& n = m.face_norm[ib];
            const double xm = 0.5 * (m.x[m.iside[ib].node1] + m.x[m.iside[ib].node2]);
            snx += n.nx * n.length;
            sny += n.ny * n.length;
            flux += xm * n.nx * n.length;
            perimeter += n.length;
        }
        CHECK_THAT(perimeter, WithinRel(2.0 * (g.lx + g.ly), 1e-12));
        CHECK_THAT(snx, WithinAbs(0.0, 1e-12));
        CHECK_THAT(sny, WithinAbs(0.0, 1e-12));
        CHECK_THAT(flux, WithinRel(g.lx * g.ly, 1e-12));
    }
}

TEST_CASE("Mesh without boundary sides has no face normals", "[mesh][normals]") {
    Mesh m = make_mesh({{0, 0}, {1, 0}, {0, 1}}, {{0, 1, 2}});
    m.compute_face_normals();
    CHECK(m.face_norm.empty());
}
