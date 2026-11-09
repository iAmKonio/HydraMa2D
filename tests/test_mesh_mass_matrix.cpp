// Lumped mass matrix: Mesh::compute_mass_matrix() stores the row-sum lumped
// P1 mass contribution of every element node, elcoe_e[ie*3 + in] = A_e / 3,
// which update_lhs() later assembles into the global diagonal mass matrix.

#include "LinearSolver.hpp"
#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <numeric>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinRel;

namespace {

// Assemble the element contributions into a global nodal (diagonal) mass vector.
std::vector<double> assemble_lumped_mass(const Mesh& m) {
    std::vector<double> mass(m.npoin, 0.0);
    for (int ie = 0; ie < m.nelem; ++ie) {
        for (int in = 0; in < Mesh::nep; ++in) {
            mass[m.intma[ie][in]] += m.elcoe_e[ie * Mesh::nep + in];
        }
    }
    return mass;
}

GridSpec test_grid() {
    GridSpec g;
    g.x0 = -2.0;
    g.y0 = 1.0;
    g.lx = 4.0;
    g.ly = 1.5;
    g.nx = 8;
    g.ny = 5;
    return g;
}

} // namespace

TEST_CASE("Element lumped mass is one third of the element area per node", "[mesh][mass]") {
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    Mesh m = make_triangle_mesh(tri);
    compute_geometry(m);

    REQUIRE(m.elcoe_e.size() == 3);
    const double area = heron_area(tri);
    for (int in = 0; in < 3; ++in) {
        CAPTURE(in);
        CHECK_THAT(m.elcoe_e[in], WithinRel(area / 3.0, 1e-9));
        CHECK_THAT(m.elcoe_e[in], WithinRel(m.detJ[0] / 6.0, 1e-15));
    }
}

TEST_CASE("Lumped mass equals the row sums of the consistent P1 mass matrix", "[mesh][mass]") {
    // Consistent P1 mass matrix: M_ij = A/12 * (1 + delta_ij).
    const Triangle tri = GENERATE(from_range(reference_triangles()));
    CAPTURE(tri);
    Mesh m = make_triangle_mesh(tri);
    compute_geometry(m);

    const double area = 0.5 * m.detJ[0];
    for (int i = 0; i < 3; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < 3; ++j) row_sum += area / 12.0 * (i == j ? 2.0 : 1.0);
        CAPTURE(i);
        CHECK_THAT(m.elcoe_e[i], WithinRel(row_sum, 1e-14));
    }
}

TEST_CASE("Element mass storage layout is ie * nep + in", "[mesh][mass]") {
    // Two elements with different areas (1/2 and 2) sharing no nodes.
    Mesh m = make_mesh({{0, 0}, {1, 0}, {0, 1}, {5, 5}, {7, 5}, {5, 7}},
                       {{0, 1, 2}, {3, 4, 5}});
    m.compute_shape_function_derivatives();
    m.compute_mass_matrix();

    REQUIRE(m.elcoe_e.size() == 6);
    for (int in = 0; in < 3; ++in) {
        CHECK_THAT(m.elcoe_e[0 * Mesh::nep + in], WithinRel(0.5 / 3.0, 1e-14));
        CHECK_THAT(m.elcoe_e[1 * Mesh::nep + in], WithinRel(2.0 / 3.0, 1e-14));
    }
}

TEST_CASE("Assembled lumped mass on a structured grid", "[mesh][mass]") {
    const GridSpec g = test_grid();
    Mesh m = make_grid_mesh(g);
    compute_geometry(m);
    REQUIRE(static_cast<int>(m.elcoe_e.size()) == 3 * g.nelem());

    const std::vector<double> mass = assemble_lumped_mass(m);
    const double cell = g.hx() * g.hy();
    const double tri_area = 0.5 * cell;

    SECTION("total mass equals the domain area") {
        const double total = std::accumulate(mass.begin(), mass.end(), 0.0);
        CHECK_THAT(total, WithinRel(g.lx * g.ly, 1e-12));
    }

    SECTION("nodal masses match the number of adjacent elements") {
        // With the (i,j)->(i+1,j+1) diagonal: interior nodes touch 6 elements,
        // edge nodes 3, corners (0,0) and (nx,ny) 2, corners (nx,0) and (0,ny) 1.
        for (int j = 0; j <= g.ny; ++j) {
            for (int i = 0; i <= g.nx; ++i) {
                const bool ix = (i == 0 || i == g.nx);
                const bool jy = (j == 0 || j == g.ny);
                int n_adjacent = 6;
                if (ix && jy) {
                    n_adjacent = ((i == 0) == (j == 0)) ? 2 : 1;
                } else if (ix || jy) {
                    n_adjacent = 3;
                }
                CAPTURE(i, j, n_adjacent);
                CHECK_THAT(mass[g.node(i, j)], WithinRel(n_adjacent * tri_area / 3.0, 1e-12));
            }
        }
    }

    SECTION("interior nodal mass equals the cell area h_x * h_y") {
        CHECK_THAT(mass[g.node(g.nx / 2, g.ny / 2)], WithinRel(cell, 1e-12));
    }
}

TEST_CASE("Lumped mass integrates piecewise-linear fields exactly", "[mesh][mass]") {
    // sum_i M_i f_i == integral of f over the domain for linear f.
    const GridSpec g = test_grid();
    Mesh m = make_grid_mesh(g);
    compute_geometry(m);
    const std::vector<double> mass = assemble_lumped_mass(m);

    const double a = 1.0, b = 2.0, c = -3.0; // f = a + b x + c y
    double lumped_integral = 0.0;
    for (int ip = 0; ip < m.npoin; ++ip) {
        lumped_integral += mass[ip] * (a + b * m.x[ip] + c * m.y[ip]);
    }

    const double x1 = g.x0 + g.lx, y1 = g.y0 + g.ly;
    const double exact = a * g.lx * g.ly
                       + b * 0.5 * (x1 * x1 - g.x0 * g.x0) * g.ly
                       + c * 0.5 * (y1 * y1 - g.y0 * g.y0) * g.lx;
    CHECK_THAT(lumped_integral, WithinRel(exact, 1e-12));
}

TEST_CASE("update_lhs assembles the inverse lumped mass scaled by the time step", "[mesh][mass][assembly]") {
    // Production assembly path (LinearSolver.cpp): elcoe2[ip] = 1 / sum_e (m_e / dt_e).
    const GridSpec g = test_grid();
    Mesh m = make_grid_mesh(g);
    compute_geometry(m);
    m.compute_laplacian_stiffness(); // update_lhs also scales the element Laplacian

    const std::vector<double> mass = assemble_lumped_mass(m);
    const double dt = GENERATE(1.0, 0.25, 1.0e-4);
    CAPTURE(dt);

    std::vector<double> delte(m.nelem, dt), elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;
    update_lhs(m, delte, elcoe2, pdiag, gstif);

    REQUIRE(static_cast<int>(elcoe2.size()) == m.npoin);
    for (int ip = 0; ip < m.npoin; ++ip) {
        CAPTURE(ip);
        CHECK_THAT(elcoe2[ip], WithinRel(dt / mass[ip], 1e-12));
    }
}
