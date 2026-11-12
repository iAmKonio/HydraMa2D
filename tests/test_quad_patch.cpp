// Patch tests (Irons & Razzaque 1972) for bilinear quadrilaterals and mixed
// triangle / quadrilateral meshes, on patches with singular vertices
// (valence != 4; MacNeal & Harder 1985 and others, tests/QuadTestUtils.hpp)
// and on distorted grids:
//
//   * every operator of the solver is consistent with linear fields: the
//     Laplacian annihilates them at interior nodes and a Dirichlet problem
//     reproduces them exactly; Steps 1-3 and the scalar transport give their
//     exact discrete values; gradients and recovery are exact;
//   * the assembled (CSR) operator equals the element-by-element one;
//   * the Poisson problem converges at second order on distorted quadrilateral
//     and hybrid meshes.

#include "IoTestUtils.hpp"
#include "LinearSolver.hpp"
#include "QuadTestUtils.hpp"
#include "Step1.hpp"
#include "Step2.hpp"
#include "Step3.hpp"
#include "geometry/FieldOps.hpp"
#include "linalg/CsrAssembly.hpp"
#include "linalg/EbeOperator.hpp"
#include "mesh/MeshBuilder.hpp"
#include "mesh/MeshCheck.hpp"
#include "numerics/ScalarTransport.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <string>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using hydrama::mesh::check_mesh;
using Catch::Matchers::WithinRel;

namespace {

struct Patch {
    std::string name;
    Mesh mesh;
};

void prepare(Mesh& m) {
    CaptureStdout quiet;
    compute_all_geometry(m);
}

std::vector<Patch> patches() {
    std::vector<Patch> out;
    auto add = [&](const std::string& name, const HybridSpec& s) {
        Patch p{name, make_hybrid_mesh(s)};
        prepare(p.mesh);
        out.push_back(std::move(p));
    };
    add("MacNeal-Harder (valence 3)", macneal_harder_patch());
    add("pentagon (valence 5)", pentagon_patch());
    add("triangle split into 3 quadrilaterals (valence 3)", triangle_split_patch());
    add("hybrid pentagon (quadrilaterals + triangles, valence 7)", hybrid_pentagon_patch());
    GridOptions g;
    g.nx = 5;
    g.ny = 4;
    g.lx = 2.0;
    g.perturbation = 0.2;
    add("perturbed quadrilateral grid", grid(g));
    g.cells = GridCells::InflationLayers;
    g.nx = g.ny = 6;
    g.layers = 2;
    g.perturbation = 0.15;
    add("perturbed inflation-layer grid", grid(g));
    return out;
}

const std::vector<Patch>& all_patches() {
    static const std::vector<Patch> p = patches();
    return p;
}

std::vector<double> linear(const Mesh& m, double c, double a, double b) {
    std::vector<double> f(static_cast<std::size_t>(m.npoin));
    for (int i = 0; i < m.npoin; ++i) f[static_cast<std::size_t>(i)] = c + a * m.x[static_cast<std::size_t>(i)] + b * m.y[static_cast<std::size_t>(i)];
    return f;
}

double max_abs(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s = std::max(s, std::abs(x));
    return s;
}

// Elements around each node (valence).
std::vector<int> valence(const Mesh& m) {
    std::vector<int> n(static_cast<std::size_t>(m.npoin), 0);
    for (int e = 0; e < m.nelem; ++e)
        for (int ip : m.element_nodes(e)) ++n[static_cast<std::size_t>(ip)];
    return n;
}

} // namespace

TEST_CASE("Patch test meshes: valid, singular vertices, consistent geometry", "[quad][patch]") {
    for (const Patch& p : all_patches()) {
        CAPTURE(p.name);
        const Mesh& m = p.mesh;
        REQUIRE(check_mesh(m).empty());
        REQUIRE(m.num_quads() > 0);
        const std::vector<int> val = valence(m);
        const std::vector<char> bnd = boundary_nodes(m);
        bool singular = false;
        for (int i = 0; i < m.npoin; ++i)
            if (!bnd[static_cast<std::size_t>(i)] && val[static_cast<std::size_t>(i)] != 4) singular = true;
        if (p.name.find("grid") == std::string::npos) CHECK(singular);

        const std::vector<double> mass = lumped_nodal_mass(m);
        double total = 0.0, area = 0.0;
        for (double v : mass) {
            CHECK(v > 0.0);
            total += v;
        }
        for (int e = 0; e < m.nelem; ++e) {
            area += 0.5 * m.detJ[static_cast<std::size_t>(e)];
            CHECK(m.alen_e[static_cast<std::size_t>(e)] > 0.0);
        }
        CHECK_THAT(total, WithinRel(boundary_area(m), 1e-13));
        CHECK_THAT(area, WithinRel(boundary_area(m), 1e-13));
    }
}

TEST_CASE("Patch test: the Laplacian annihilates linear fields at interior nodes", "[quad][patch]") {
    for (const Patch& p : all_patches()) {
        CAPTURE(p.name);
        const Mesh& m = p.mesh;
        std::vector<double> elcoe2, pdiag, y;
        std::vector<std::array<double, 3>> gstif;
        update_lhs(m, std::vector<double>(static_cast<std::size_t>(m.nelem), 1.0), elcoe2, pdiag, gstif);
        REQUIRE(static_cast<int>(gstif.size()) == m.num_offdiag_blocks());
        const std::vector<double> f = linear(m, 0.3, -1.7, 2.9);
        matrix_vector_mult(m, gstif, pdiag, f, y);
        const std::vector<char> bnd = boundary_nodes(m);
        const double scale = max_abs(pdiag) * max_abs(f);
        for (int i = 0; i < m.npoin; ++i) {
            CAPTURE(i);
            if (!bnd[static_cast<std::size_t>(i)]) CHECK_THAT(y[static_cast<std::size_t>(i)], WithinAbs(0.0, 1e-13 * scale));
        }
        // Constants are in the null space everywhere.
        matrix_vector_mult(m, gstif, pdiag, std::vector<double>(static_cast<std::size_t>(m.npoin), 1.0), y);
        CHECK(max_abs(y) < 1e-13 * max_abs(pdiag));
    }
}

TEST_CASE("Patch test: a Dirichlet problem reproduces linear fields", "[quad][patch]") {
    for (const Patch& p : all_patches()) {
        CAPTURE(p.name);
        const Mesh& m = p.mesh;
        std::vector<double> elcoe2, pdiag;
        std::vector<std::array<double, 3>> gstif;
        update_lhs(m, std::vector<double>(static_cast<std::size_t>(m.nelem), 0.01), elcoe2, pdiag, gstif);
        const std::vector<double> exact = linear(m, 1.0, 0.4, -0.8);
        std::vector<int> bc;
        std::vector<double> values;
        const std::vector<char> bnd = boundary_nodes(m);
        for (int i = 0; i < m.npoin; ++i) {
            if (bnd[static_cast<std::size_t>(i)]) {
                bc.push_back(i);
                values.push_back(exact[static_cast<std::size_t>(i)]);
            }
        }
        std::vector<double> phi(static_cast<std::size_t>(m.npoin), 0.0);
        solve_pcg(m, gstif, pdiag, std::vector<double>(static_cast<std::size_t>(m.npoin), 0.0), bc, values, 0.0, 1e-15, phi);
        for (int i = 0; i < m.npoin; ++i) {
            CAPTURE(i);
            CHECK_THAT(phi[static_cast<std::size_t>(i)], WithinAbs(exact[static_cast<std::size_t>(i)], 1e-11));
        }
    }
}

TEST_CASE("Patch test: Step 2 gives -div(u) M_i for linear velocities", "[quad][patch][step2]") {
    for (const Patch& p : all_patches()) {
        CAPTURE(p.name);
        const Mesh& m = p.mesh;
        const std::vector<double> u = linear(m, 0.5, 1.3, -0.4), v = linear(m, -0.2, 0.7, 0.9);
        const double div = 1.3 + 0.9;
        const std::vector<double> mass = lumped_nodal_mass(m);
        std::vector<double> rhs1;
        compute_step2(m, u, v, u, v, 1.0, rhs1); // u* = u^n: no theta1 term
        for (int i = 0; i < m.npoin; ++i)
            CHECK_THAT(rhs1[static_cast<std::size_t>(i)], WithinAbs(-div * mass[static_cast<std::size_t>(i)], 1e-14 * div * max_abs(mass)));
        // theta1 term with a constant correction: theta1 int grad N_i . c = 0 at interior nodes.
        std::vector<double> us = u, vs = v;
        for (double& x : us) x += 0.3;
        for (double& x : vs) x -= 0.1;
        std::vector<double> rhs2;
        compute_step2(m, us, vs, u, v, 0.7, rhs2);
        const std::vector<char> bnd = boundary_nodes(m);
        for (int i = 0; i < m.npoin; ++i)
            if (!bnd[static_cast<std::size_t>(i)])
                CHECK_THAT(rhs2[static_cast<std::size_t>(i)], WithinAbs(rhs1[static_cast<std::size_t>(i)], 1e-14 * div * max_abs(mass)));
    }
}

TEST_CASE("Patch test: Step 3 gives -grad(p) M_i for linear pressures", "[quad][patch][step3]") {
    for (const Patch& p : all_patches()) {
        CAPTURE(p.name);
        const Mesh& m = p.mesh;
        const std::vector<double> pn = linear(m, 0.1, -2.0, 0.6);
        const std::vector<double> zero(static_cast<std::size_t>(m.npoin), 0.0);
        const std::vector<double> uc(static_cast<std::size_t>(m.npoin), 0.8), vc(static_cast<std::size_t>(m.npoin), -0.3);
        const std::vector<double> dt(static_cast<std::size_t>(m.nelem), 0.05);
        const std::vector<double> mass = lumped_nodal_mass(m);
        std::vector<double> ru, rv;
        compute_step3(m, uc, vc, pn, zero, dt, ru, rv);
        for (int i = 0; i < m.npoin; ++i) {
            CHECK_THAT(ru[static_cast<std::size_t>(i)], WithinAbs(2.0 * mass[static_cast<std::size_t>(i)], 1e-14 * max_abs(mass)));
            CHECK_THAT(rv[static_cast<std::size_t>(i)], WithinAbs(-0.6 * mass[static_cast<std::size_t>(i)], 1e-14 * max_abs(mass)));
        }
        // Stabilisation with the old pressure: -dt/2 (u . int grad N_i) grad p = 0 at interior nodes.
        compute_step3(m, uc, vc, zero, pn, dt, ru, rv);
        const std::vector<char> bnd = boundary_nodes(m);
        for (int i = 0; i < m.npoin; ++i) {
            if (bnd[static_cast<std::size_t>(i)]) continue;
            CHECK_THAT(ru[static_cast<std::size_t>(i)], WithinAbs(0.0, 1e-14));
            CHECK_THAT(rv[static_cast<std::size_t>(i)], WithinAbs(0.0, 1e-14));
        }
    }
}

TEST_CASE("Patch test: Step 1 preserves free streams and is consistent for viscous terms", "[quad][patch][step1]") {
    for (const Patch& p : all_patches()) {
        CAPTURE(p.name);
        const Mesh& m = p.mesh;
        const std::size_t n = static_cast<std::size_t>(m.npoin);
        const std::vector<double> dt(static_cast<std::size_t>(m.nelem), 0.02), nut(static_cast<std::size_t>(m.nelem), 0.3);
        std::vector<double> ru, rv;
        // Uniform flow: every term vanishes at every node (laminar and turbulent).
        compute_step1(m, 0.05, dt, std::vector<double>(n, 1.4), std::vector<double>(n, -0.6), ru, rv);
        CHECK(max_abs(ru) < 1e-14);
        CHECK(max_abs(rv) < 1e-14);
        compute_step1(m, 0.05, nut, dt, std::vector<double>(n, 1.4), std::vector<double>(n, -0.6), ru, rv);
        CHECK(max_abs(ru) < 1e-14);
        CHECK(max_abs(rv) < 1e-14);
        // Linear velocity: the viscous (and eddy-viscous) parts vanish at interior nodes.
        const std::vector<double> u = linear(m, 0.2, 0.5, -1.0), v = linear(m, 1.0, 0.3, 0.4);
        std::vector<double> ru0, rv0, ru1, rv1, ru2, rv2;
        compute_step1(m, 0.0, dt, u, v, ru0, rv0);
        compute_step1(m, 0.8, dt, u, v, ru1, rv1);
        compute_step1(m, 0.8, nut, dt, u, v, ru2, rv2);
        const std::vector<char> bnd = boundary_nodes(m);
        for (std::size_t i = 0; i < n; ++i) {
            if (bnd[i]) continue;
            CHECK_THAT(ru1[i] - ru0[i], WithinAbs(0.0, 1e-13));
            CHECK_THAT(rv1[i] - rv0[i], WithinAbs(0.0, 1e-13));
            CHECK_THAT(ru2[i] - ru0[i], WithinAbs(0.0, 1e-13));
            CHECK_THAT(rv2[i] - rv0[i], WithinAbs(0.0, 1e-13));
        }
    }
}

TEST_CASE("Patch test: scalar transport equals Step 1 and preserves constants", "[quad][patch][transport]") {
    for (const Patch& p : all_patches()) {
        CAPTURE(p.name);
        const Mesh& m = p.mesh;
        const std::size_t n = static_cast<std::size_t>(m.npoin);
        const std::vector<double> dt(static_cast<std::size_t>(m.nelem), 0.03);
        const std::vector<double> u = linear(m, 0.4, 0.3, -0.2), v = linear(m, -0.1, 0.2, 0.5);
        const double nu = 0.07;
        std::vector<double> ru, rv, rhs;
        compute_step1(m, nu, dt, u, v, ru, rv);
        assemble_scalar_transport(m, u, v, dt, u, std::vector<double>(static_cast<std::size_t>(m.nelem), nu), rhs);
        for (std::size_t i = 0; i < n; ++i) CHECK_THAT(rhs[i], WithinAbs(ru[i], 1e-14));
        assemble_scalar_transport(m, u, v, dt, std::vector<double>(n, 3.0), std::vector<double>(static_cast<std::size_t>(m.nelem), nu), rhs);
        CHECK(max_abs(rhs) < 1e-14);
    }
}

TEST_CASE("Patch test: element and recovered gradients of linear fields are exact", "[quad][patch][field_ops]") {
    for (const Patch& p : all_patches()) {
        CAPTURE(p.name);
        const Mesh& m = p.mesh;
        const std::vector<double> u = linear(m, 1.0, 0.25, -0.5), v = linear(m, 0.0, 1.5, 0.75);
        std::vector<VelocityGradient> g;
        compute_velocity_gradients(m, u, v, g);
        for (const VelocityGradient& e : g) {
            CHECK_THAT(e.dudx, WithinAbs(0.25, 1e-12));
            CHECK_THAT(e.dudy, WithinAbs(-0.5, 1e-12));
            CHECK_THAT(e.dvdx, WithinAbs(1.5, 1e-12));
            CHECK_THAT(e.dvdy, WithinAbs(0.75, 1e-12));
        }
        for (const GradientRecovery method : {GradientRecovery::LumpedProjection, GradientRecovery::PolynomialPreserving}) {
            const NodalGradient rec(m, method);
            std::vector<std::array<double, 2>> gn;
            rec.gradient(u, gn);
            for (int i = 0; i < m.npoin; ++i) {
                CAPTURE(i, to_string(method));
                CHECK_THAT(gn[static_cast<std::size_t>(i)][0], WithinAbs(0.25, 1e-10));
                CHECK_THAT(gn[static_cast<std::size_t>(i)][1], WithinAbs(-0.5, 1e-10));
            }
        }
        // Projection of a constant element field is that constant.
        std::vector<double> nodal;
        project_to_nodes(m, lumped_nodal_mass(m), std::vector<double>(static_cast<std::size_t>(m.nelem), 2.5), nodal);
        for (double x : nodal) CHECK_THAT(x, WithinRel(2.5, 1e-14));
    }
}

TEST_CASE("Patch test: PPR recovers quadratic fields on a hybrid grid", "[quad][patch][field_ops]") {
    GridOptions g;
    g.cells = GridCells::InflationLayers;
    g.nx = g.ny = 8;
    g.layers = 2;
    g.perturbation = 0.1;
    Mesh m = make_hybrid_mesh(grid(g));
    prepare(m);
    std::vector<double> q(static_cast<std::size_t>(m.npoin));
    for (int i = 0; i < m.npoin; ++i) {
        const double x = m.x[static_cast<std::size_t>(i)], y = m.y[static_cast<std::size_t>(i)];
        q[static_cast<std::size_t>(i)] = 1.0 + x - 2.0 * y + 3.0 * x * x - x * y + 0.5 * y * y;
    }
    const NodalGradient rec(m, GradientRecovery::PolynomialPreserving);
    std::vector<std::array<double, 2>> gn;
    rec.gradient(q, gn);
    const std::vector<char> bnd = boundary_nodes(m);
    for (int i = 0; i < m.npoin; ++i) {
        if (bnd[static_cast<std::size_t>(i)]) continue;
        const double x = m.x[static_cast<std::size_t>(i)], y = m.y[static_cast<std::size_t>(i)];
        CAPTURE(i, x, y);
        CHECK_THAT(gn[static_cast<std::size_t>(i)][0], WithinAbs(1.0 + 6.0 * x - y, 1e-9));
        CHECK_THAT(gn[static_cast<std::size_t>(i)][1], WithinAbs(-2.0 - x + y, 1e-9));
    }
}

TEST_CASE("Patch test: the assembled operator equals the element-by-element operator", "[quad][patch][linalg]") {
    for (const Patch& p : all_patches()) {
        CAPTURE(p.name);
        const Mesh& m = p.mesh;
        std::vector<double> elcoe2, pdiag, y_ebe;
        std::vector<std::array<double, 3>> gstif;
        update_lhs(m, std::vector<double>(static_cast<std::size_t>(m.nelem), 0.3), elcoe2, pdiag, gstif);
        const linalg::P1CsrAssembler assembler(m);
        linalg::CsrMatrix a = assembler.create_matrix();
        assembler.assemble_ebe(m, gstif, pdiag, a);
        // Every pair of nodes of a quadrilateral is coupled (also its diagonals).
        for (const auto& q : m.quads)
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) CHECK(a.pattern().find(q[static_cast<std::size_t>(i)], q[static_cast<std::size_t>(j)]) >= 0);
        const std::vector<double> x = linear(m, 0.5, 3.0, -1.0);
        std::vector<double> y_csr(x.size());
        y_ebe.resize(x.size());
        linalg::ebe_apply(m, gstif.data(), pdiag.data(), x.data(), y_ebe.data());
        a.apply(x, y_csr);
        for (std::size_t i = 0; i < x.size(); ++i) CHECK_THAT(y_csr[i], WithinAbs(y_ebe[i], 1e-13 * max_abs(pdiag) * max_abs(x)));
        // Generic assembly of the quadrilateral Laplacians (dt = 0.3) reproduces the quadrilateral part.
        linalg::CsrMatrix b = assembler.create_matrix();
        for (int q = 0; q < m.num_quads(); ++q) {
            const QuadLaplacian& k = m.quad_lap[static_cast<std::size_t>(q)];
            linalg::P1CsrAssembler::QuadElementMatrix ke{};
            for (int i = 0; i < 4; ++i) ke[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 0.3 * k.diag[static_cast<std::size_t>(i)];
            for (int pr = 0; pr < 6; ++pr) {
                const std::size_t i = static_cast<std::size_t>(q1::kPairs[pr][0]), j = static_cast<std::size_t>(q1::kPairs[pr][1]);
                ke[i][j] = ke[j][i] = 0.3 * k.off[static_cast<std::size_t>(pr)];
            }
            assembler.add_quad_element_matrix(b, m.element_of_quad(q), ke);
        }
        for (int e = 0; e < m.num_triangles(); ++e) {
            linalg::P1CsrAssembler::ElementMatrix ke{};
            for (int i = 0; i < 3; ++i) ke[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 0.3 * m.pdiagE[static_cast<std::size_t>(e)][static_cast<std::size_t>(i)];
            ke[0][1] = ke[1][0] = 0.3 * m.gstifE[static_cast<std::size_t>(e)][0];
            ke[0][2] = ke[2][0] = 0.3 * m.gstifE[static_cast<std::size_t>(e)][1];
            ke[1][2] = ke[2][1] = 0.3 * m.gstifE[static_cast<std::size_t>(e)][2];
            assembler.add_element_matrix(b, e, ke);
        }
        std::vector<double> y_gen(x.size());
        b.apply(x, y_gen);
        for (std::size_t i = 0; i < x.size(); ++i) CHECK_THAT(y_gen[i], WithinAbs(y_ebe[i], 1e-13 * max_abs(pdiag) * max_abs(x)));
        if (m.num_triangles() > 0) CHECK_THROWS_AS(assembler.add_quad_element_matrix(b, 0, {}), std::invalid_argument);
        CHECK_THROWS_AS(assembler.add_element_matrix(b, m.nelem - 1, {}), std::invalid_argument);
    }
}

TEST_CASE("Patch test: patches through the importer (clockwise input) give the same operators", "[quad][patch][mesh_import]") {
    GridOptions g;
    g.cells = GridCells::InflationLayers;
    g.nx = g.ny = 5;
    g.layers = 1;
    g.perturbation = 0.15;
    const Mesh direct = [&] {
        Mesh m = make_hybrid_mesh(grid(g));
        prepare(m);
        return m;
    }();
    g.clockwise = true;
    const HybridSpec cw = grid(g);
    mesh::MeshData d;
    for (std::size_t i = 0; i < cw.nodes.size(); ++i) {
        d.node_tags.push_back(static_cast<std::int64_t>(100 + 7 * i)); // sparse tags
        d.x.push_back(cw.nodes[i].x);
        d.y.push_back(cw.nodes[i].y);
        d.z.push_back(0.0);
    }
    auto tag = [](int i) { return static_cast<std::int64_t>(100 + 7 * i); };
    for (const auto& q : cw.quads) d.elements.push_back({mesh::ElementType::Quad4, {tag(q[0]), tag(q[1]), tag(q[2]), tag(q[3])}, 0, 0, 0});
    for (const auto& t : cw.triangles) d.elements.push_back({mesh::ElementType::Tri3, {tag(t[0]), tag(t[1]), tag(t[2]), 0}, 0, 0, 0});
    for (const auto& s : direct.iside) d.elements.push_back({mesh::ElementType::Line2, {tag(s.node2), tag(s.node1), 0, 0}, 1, 0, 0});
    d.physical_names[{1, 1}] = "wall";
    mesh::ImportReport r;
    Mesh imported = mesh::build_mesh(d, {}, &r);
    CHECK(r.reoriented_quadrilaterals == direct.num_quads());
    CHECK(r.reoriented_triangles == direct.num_triangles());
    REQUIRE(check_mesh(imported).empty());
    prepare(imported);
    // The importer numbers triangles first; the direct mesh lists them first too.
    CHECK(imported.intma == direct.intma);
    CHECK(imported.quads == direct.quads);
    CHECK(lumped_nodal_mass(imported) == lumped_nodal_mass(direct));
}

TEST_CASE("Poisson problem: second-order convergence on distorted and hybrid meshes", "[quad][patch][convergence]") {
    const double pi = std::acos(-1.0);
    for (const GridCells cells : {GridCells::Quads, GridCells::InflationLayers, GridCells::Triangles}) {
        CAPTURE(static_cast<int>(cells));
        std::vector<double> errors;
        for (const int n : {8, 16, 32}) {
            GridOptions g;
            g.nx = g.ny = n;
            g.cells = cells;
            g.layers = n / 4;
            HybridSpec s = grid(g);
            for (Point& p : s.nodes) { // smooth distortion, fixed boundary
                const double dx = 0.06 * std::sin(2 * pi * p.x) * std::sin(2 * pi * p.y);
                p.x += dx;
                p.y += 0.5 * dx;
            }
            Mesh m = make_hybrid_mesh(s);
            prepare(m);
            REQUIRE(check_mesh(m).empty());
            // -lap u = 2 pi^2 sin(pi x) sin(pi y), u = 0 on the boundary.
            std::vector<double> elcoe2, pdiag;
            std::vector<std::array<double, 3>> gstif;
            update_lhs(m, std::vector<double>(static_cast<std::size_t>(m.nelem), 1.0), elcoe2, pdiag, gstif);
            const std::vector<double> mass = lumped_nodal_mass(m);
            std::vector<double> b(static_cast<std::size_t>(m.npoin)), uh(static_cast<std::size_t>(m.npoin), 0.0);
            for (int i = 0; i < m.npoin; ++i) {
                const double x = m.x[static_cast<std::size_t>(i)], y = m.y[static_cast<std::size_t>(i)];
                b[static_cast<std::size_t>(i)] = mass[static_cast<std::size_t>(i)] * 2 * pi * pi * std::sin(pi * x) * std::sin(pi * y);
            }
            std::vector<int> bc;
            const std::vector<char> bnd = boundary_nodes(m);
            for (int i = 0; i < m.npoin; ++i)
                if (bnd[static_cast<std::size_t>(i)]) bc.push_back(i);
            solve_pcg(m, gstif, pdiag, b, bc, std::vector<double>(bc.size(), 0.0), 0.0, 1e-14, uh);
            double err = 0.0;
            for (int i = 0; i < m.npoin; ++i) {
                const double x = m.x[static_cast<std::size_t>(i)], y = m.y[static_cast<std::size_t>(i)];
                err = std::max(err, std::abs(uh[static_cast<std::size_t>(i)] - std::sin(pi * x) * std::sin(pi * y)));
            }
            errors.push_back(err);
        }
        CAPTURE(errors[0], errors[1], errors[2]);
        const double order1 = std::log2(errors[0] / errors[1]), order2 = std::log2(errors[1] / errors[2]);
        CAPTURE(order1, order2);
        CHECK(order1 > 1.8);
        CHECK(order2 > 1.8);
        CHECK(errors[2] < 1e-2);
    }
}
