// CBS Step 3 (velocity correction): compute_step3() assembles, for each node i,
//
//   rhs_i = - int N_i grad p^{n+1} dOmega                          (pressure gradient)
//           - dt_e / 2 * int (u_bar . grad N_i) grad p^n dOmega     (CBS stabilisation)
//
// with u_bar the element mean of u^n. In the code the second term appears as
// detJ * (u_bar . grad N_i) * grad p^n * dt/4, with detJ = 2A, i.e. A dt / 2.
// The solver then corrects u_i += dt * rhs_i / M_i (elcoe2 from update_lhs).

#include "IO.hpp"
#include "LinearSolver.hpp"
#include "MeshTestUtils.hpp"
#include "Step3.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <numeric>
#include <set>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

struct Rhs {
    std::vector<double> u, v;
};

struct State {
    std::vector<double> un, vn;     // u^n
    std::vector<double> pnew, pold; // p^{n+1}, p^n
};

Rhs step3(const Mesh& m, const State& s, const std::vector<double>& delte) {
    Rhs r{std::vector<double>(2, 9.0), std::vector<double>(99, -9.0)}; // stale outputs
    compute_step3(m, s.un, s.vn, s.pnew, s.pold, delte, r.u, r.v);
    return r;
}

// --- Independent references --------------------------------------------------

// Pressure-gradient term via integration by parts:
//   -int N_i dp/dx = int dN_i/dx p - closed int N_i p n_x ds
// with int_e p = A_e * mean(p) and int_side N_i p = L/6 (2 p_i + p_j).
Rhs pressure_ref(const Mesh& m, const std::vector<double>& p) {
    Rhs r{std::vector<double>(m.npoin, 0.0), std::vector<double>(m.npoin, 0.0)};
    for (int ie = 0; ie < m.nelem; ++ie) {
        const auto& n = m.intma[ie];
        const double area = 0.5 * m.detJ[ie];
        const double pbar = (p[n[0]] + p[n[1]] + p[n[2]]) / 3.0;
        for (int k = 0; k < 3; ++k) {
            r.u[n[k]] += m.dNkdx[ie].dNdx[k] * area * pbar;
            r.v[n[k]] += m.dNkdx[ie].dNdy[k] * area * pbar;
        }
    }
    for (int ib = 0; ib < m.nboun; ++ib) {
        const int a = m.iside[ib].node1, b = m.iside[ib].node2;
        const auto& fn = m.face_norm[ib];
        const double wa = fn.length / 6.0 * (2.0 * p[a] + p[b]);
        const double wb = fn.length / 6.0 * (2.0 * p[b] + p[a]);
        r.u[a] -= wa * fn.nx;
        r.v[a] -= wa * fn.ny;
        r.u[b] -= wb * fn.nx;
        r.v[b] -= wb * fn.ny;
    }
    return r;
}

// Stabilisation term: -dt_e/2 * A_e * (u_bar . grad N_i) * grad p^n.
Rhs stabilisation_ref(const Mesh& m, const State& s, const std::vector<double>& delte) {
    Rhs r{std::vector<double>(m.npoin, 0.0), std::vector<double>(m.npoin, 0.0)};
    for (int ie = 0; ie < m.nelem; ++ie) {
        const auto& n = m.intma[ie];
        const auto& d = m.dNkdx[ie];
        const double area = 0.5 * m.detJ[ie];
        const double ub = (s.un[n[0]] + s.un[n[1]] + s.un[n[2]]) / 3.0;
        const double vb = (s.vn[n[0]] + s.vn[n[1]] + s.vn[n[2]]) / 3.0;
        double gx = 0.0, gy = 0.0;
        for (int k = 0; k < 3; ++k) {
            gx += d.dNdx[k] * s.pold[n[k]];
            gy += d.dNdy[k] * s.pold[n[k]];
        }
        for (int k = 0; k < 3; ++k) {
            const double adv = ub * d.dNdx[k] + vb * d.dNdy[k];
            r.u[n[k]] -= 0.5 * delte[ie] * area * adv * gx;
            r.v[n[k]] -= 0.5 * delte[ie] * area * adv * gy;
        }
    }
    return r;
}

Rhs step3_ref(const Mesh& m, const State& s, const std::vector<double>& delte) {
    Rhs r = pressure_ref(m, s.pnew);
    const Rhs st = stabilisation_ref(m, s, delte);
    for (int i = 0; i < m.npoin; ++i) {
        r.u[i] += st.u[i];
        r.v[i] += st.v[i];
    }
    return r;
}

Rhs difference(const Rhs& a, const Rhs& b) {
    Rhs d{a.u, a.v};
    for (size_t i = 0; i < a.u.size(); ++i) {
        d.u[i] -= b.u[i];
        d.v[i] -= b.v[i];
    }
    return d;
}

void check_close(const Rhs& a, const Rhs& b, double tol) {
    REQUIRE(a.u.size() == b.u.size());
    REQUIRE(a.v.size() == b.v.size());
    for (size_t i = 0; i < a.u.size(); ++i) {
        CAPTURE(i);
        CHECK_THAT(a.u[i], WithinAbs(b.u[i], tol));
        CHECK_THAT(a.v[i], WithinAbs(b.v[i], tol));
    }
}

// --- Test setup -----------------------------------------------------------------

GridSpec step_grid_spec() {
    GridSpec g;
    g.x0 = 0.2;
    g.y0 = -0.3;
    g.lx = 1.2;
    g.ly = 0.9;
    g.nx = 5;
    g.ny = 4;
    return g;
}

Mesh step_grid() {
    Mesh m = make_grid_mesh(step_grid_spec());
    compute_all_geometry(m);
    return m;
}

State generic_state(const Mesh& m) {
    State s;
    s.un = sample(m, [](double x, double y) { return 0.4 + std::sin(2.0 * x) * std::cos(1.5 * y); });
    s.vn = sample(m, [](double x, double y) { return -0.2 - std::cos(2.0 * x) * std::sin(1.5 * y); });
    s.pnew = sample(m, [](double x, double y) { return std::cos(1.3 * x) * std::exp(0.5 * y); });
    s.pold = sample(m, [](double x, double y) { return 0.8 * std::cos(1.2 * x + 0.1) * std::exp(0.6 * y); });
    return s;
}

std::set<int> boundary_nodes(const Mesh& m) {
    std::set<int> b;
    for (const auto& s : m.iside) {
        b.insert(s.node1);
        b.insert(s.node2);
    }
    return b;
}

// Velocity change of one corrector step, as applied in Solver::run():
// du_i = rhs_i * elcoe2_i with elcoe2 from update_lhs().
Rhs velocity_change(const Mesh& m, const State& s, const std::vector<double>& delte) {
    std::vector<double> elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;
    update_lhs(m, delte, elcoe2, pdiag, gstif);
    Rhs du = step3(m, s, delte);
    for (int i = 0; i < m.npoin; ++i) {
        du.u[i] *= elcoe2[i];
        du.v[i] *= elcoe2[i];
    }
    return du;
}

} // namespace

// =============================================================================
// Hand-calculated values
// =============================================================================

TEST_CASE("Step 3 on the unit right triangle, computed by hand", "[cbs][step3]") {
    // A = 1/2, dN/dx = (-1, 1, 0), dN/dy = (-1, 0, 1).
    // p^{n+1} = x   -> grad = (1, 0):  -int N_i dp/dx = -A/3 = -1/6 for every node
    // p^n     = y   -> grad = (0, 1)
    // u^n = (3,0,0), v^n = (0,0,3) -> u_bar = (1, 1), u_bar . grad N_i = (-2, 1, 1)
    // stabilisation (v only): -A dt/2 * (-2, 1, 1) * 1 = dt * (1/2, -1/4, -1/4)
    Mesh m = make_triangle_mesh(reference_triangles()[0]);
    compute_all_geometry(m);
    const double dt = GENERATE(0.0, 0.1, 1.0, 4.0);
    CAPTURE(dt);

    State s;
    s.un = {3.0, 0.0, 0.0};
    s.vn = {0.0, 0.0, 3.0};
    s.pnew = {0.0, 1.0, 0.0};
    s.pold = {0.0, 0.0, 1.0};
    const Rhs r = step3(m, s, {dt});
    REQUIRE(r.u.size() == 3);
    REQUIRE(r.v.size() == 3);
    for (int i = 0; i < 3; ++i) CHECK_THAT(r.u[i], WithinAbs(-1.0 / 6.0, 1e-15));
    CHECK_THAT(r.v[0], WithinAbs(0.5 * dt, 1e-15));
    CHECK_THAT(r.v[1], WithinAbs(-0.25 * dt, 1e-15));
    CHECK_THAT(r.v[2], WithinAbs(-0.25 * dt, 1e-15));
}

TEST_CASE("Step 3 on the scalene reference triangle, computed by hand", "[cbs][step3]") {
    // Nodes (1,1), (4,2), (2,5): A = 11/2, dN/dx = (-3, 4, -1)/11, dN/dy = (-2, -1, 3)/11.
    Mesh m = make_triangle_mesh(reference_triangles()[1]);
    compute_all_geometry(m);
    const Triangle t = reference_triangles()[1];
    auto at_nodes = [&](auto f) {
        return std::vector<double>{f(t.p[0].x, t.p[0].y), f(t.p[1].x, t.p[1].y), f(t.p[2].x, t.p[2].y)};
    };

    SECTION("pressure gradient only: p^{n+1} = 1 + 2x - y, fluid at rest") {
        // -A/3 * (2, -1) = (-11/3, 11/6) at every node.
        State s;
        s.un = s.vn = {0.0, 0.0, 0.0};
        s.pnew = at_nodes([](double x, double y) { return 1.0 + 2.0 * x - y; });
        s.pold = at_nodes([](double x, double y) { return 5.0 * x * y; }); // irrelevant at rest
        const Rhs r = step3(m, s, {0.7});
        for (int i = 0; i < 3; ++i) {
            CAPTURE(i);
            CHECK_THAT(r.u[i], WithinAbs(-11.0 / 3.0, 1e-14));
            CHECK_THAT(r.v[i], WithinAbs(11.0 / 6.0, 1e-14));
        }
    }
    SECTION("stabilisation only: p^n = x, u_bar = (1, 1)") {
        // u_bar . grad N = (-5, 3, 2)/11; -A dt/2 * (-5, 3, 2)/11 = dt * (5/4, -3/4, -1/2).
        const double dt = GENERATE(0.2, 1.0);
        CAPTURE(dt);
        State s;
        s.un = {0.0, 3.0, 0.0};
        s.vn = {0.0, 0.0, 3.0};
        s.pnew = {2.5, 2.5, 2.5}; // constant: no gradient
        s.pold = at_nodes([](double x, double) { return x; });
        const Rhs r = step3(m, s, {dt});
        CHECK_THAT(r.u[0], WithinAbs(1.25 * dt, 1e-14));
        CHECK_THAT(r.u[1], WithinAbs(-0.75 * dt, 1e-14));
        CHECK_THAT(r.u[2], WithinAbs(-0.5 * dt, 1e-14));
        for (int i = 0; i < 3; ++i) CHECK_THAT(r.v[i], WithinAbs(0.0, 1e-15));
    }
}

// =============================================================================
// Pressure-gradient term
// =============================================================================

TEST_CASE("Uniform pressure gradient produces the exact acceleration", "[cbs][step3]") {
    // Fluid at rest, p^{n+1} linear: every node, including boundary nodes, is
    // accelerated by exactly du = -dt * grad p (lumped mass cancels).
    const Mesh m = step_grid();
    const double gx = GENERATE(1.0, -0.3);
    const double gy = GENERATE(0.0, 2.0);
    const double dt = GENERATE(0.05, 0.5);
    CAPTURE(gx, gy, dt);

    State s;
    s.un = s.vn = std::vector<double>(m.npoin, 0.0);
    s.pnew = sample(m, [&](double x, double y) { return 4.0 + gx * x + gy * y; });
    s.pold = sample(m, [](double x, double y) { return std::sin(3.0 * x) * y; }); // no effect at rest

    const Rhs du = velocity_change(m, s, std::vector<double>(m.nelem, dt));
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i, m.x[i], m.y[i]);
        CHECK_THAT(du.u[i], WithinAbs(-dt * gx, 1e-14));
        CHECK_THAT(du.v[i], WithinAbs(-dt * gy, 1e-14));
    }
}

TEST_CASE("Uniform pressure gradient with local time steps", "[cbs][step3]") {
    // With element-wise dt each node is accelerated over the mass-weighted
    // harmonic mean of the adjacent element time steps:
    //   du_i = -grad p * (sum_e m_e) / (sum_e m_e / dt_e)
    const Mesh m = step_grid();
    const auto delte = varying_dt(m.nelem);
    State s;
    s.un = s.vn = std::vector<double>(m.npoin, 0.0);
    s.pnew = sample(m, [](double x, double y) { return 1.5 * x - 0.5 * y; });
    s.pold = s.pnew;

    std::vector<double> msum(m.npoin, 0.0), mdt(m.npoin, 0.0);
    for (int ie = 0; ie < m.nelem; ++ie) {
        for (int k = 0; k < 3; ++k) {
            const double me = m.elcoe_e[3 * ie + k];
            msum[m.intma[ie][k]] += me;
            mdt[m.intma[ie][k]] += me / delte[ie];
        }
    }
    const Rhs du = velocity_change(m, s, delte);
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i);
        const double dt_eff = msum[i] / mdt[i];
        CHECK_THAT(du.u[i], WithinRel(-1.5 * dt_eff, 1e-13));
        CHECK_THAT(du.v[i], WithinRel(0.5 * dt_eff, 1e-13));
    }
}

TEST_CASE("Constant pressure produces no correction", "[cbs][step3]") {
    const Mesh m = step_grid();
    State s = generic_state(m);
    s.pnew = std::vector<double>(m.npoin, 101.325);
    s.pold = std::vector<double>(m.npoin, -7.0);
    const Rhs r = step3(m, s, varying_dt(m.nelem));
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i);
        CHECK_THAT(r.u[i], WithinAbs(0.0, 1e-12));
        CHECK_THAT(r.v[i], WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("Pressure-gradient term satisfies the integration-by-parts identity", "[cbs][step3]") {
    // dt = 0 removes the stabilisation term; the remaining Galerkin term must
    // equal int grad N_i p - closed int N_i p n ds at every node.
    const Mesh m = step_grid();
    const State s = generic_state(m);
    const Rhs r = step3(m, s, std::vector<double>(m.nelem, 0.0));
    const Rhs ref = pressure_ref(m, s.pnew);
    check_close(r, ref, 1e-13 * std::max(scale_of(ref.u), scale_of(ref.v)));
}

TEST_CASE("Pressure-gradient term is independent of dt, u^n and p^n", "[cbs][step3]") {
    const Mesh m = step_grid();
    State s = generic_state(m);
    s.pold = std::vector<double>(m.npoin, 0.0);
    const Rhs ref = step3(m, s, std::vector<double>(m.nelem, 0.0));

    SECTION("time step") {
        check_close(step3(m, s, varying_dt(m.nelem)), ref, 0.0);
    }
    SECTION("velocity") {
        State s2 = s;
        for (double& x : s2.un) x *= -3.0;
        check_close(step3(m, s2, std::vector<double>(m.nelem, 0.0)), ref, 0.0);
    }
}

TEST_CASE("Net pressure force equals the boundary integral of -p n", "[cbs][step3]") {
    // sum_i rhs_i = -int grad p^{n+1} = -closed int p^{n+1} n ds; the
    // stabilisation term sums to zero because sum_i grad N_i = 0.
    const Mesh m = step_grid();
    const State s = generic_state(m);
    double fx = 0.0, fy = 0.0;
    for (int ib = 0; ib < m.nboun; ++ib) {
        const auto& fn = m.face_norm[ib];
        const double pm = 0.5 * (s.pnew[m.iside[ib].node1] + s.pnew[m.iside[ib].node2]);
        fx -= pm * fn.nx * fn.length;
        fy -= pm * fn.ny * fn.length;
    }
    const Rhs r = step3(m, s, varying_dt(m.nelem));
    CHECK_THAT(std::accumulate(r.u.begin(), r.u.end(), 0.0), WithinAbs(fx, 1e-13));
    CHECK_THAT(std::accumulate(r.v.begin(), r.v.end(), 0.0), WithinAbs(fy, 1e-13));
}

// =============================================================================
// Old-pressure (dt/4) stabilisation term
// =============================================================================

TEST_CASE("Stabilisation term matches -dt/2 int (u_bar . grad N_i) grad p^n", "[cbs][step3]") {
    const Mesh m = step_grid();
    const State s = generic_state(m);
    const auto delte = varying_dt(m.nelem);
    const Rhs stab = difference(step3(m, s, delte), step3(m, s, std::vector<double>(m.nelem, 0.0)));
    check_close(stab, stabilisation_ref(m, s, delte), 1e-15);
}

TEST_CASE("Stabilisation term scales linearly with dt, u^n and p^n", "[cbs][step3]") {
    const Mesh m = step_grid();
    const State s = generic_state(m);
    const auto delte = varying_dt(m.nelem);
    const std::vector<double> zero_dt(m.nelem, 0.0);
    auto stab_of = [&](const State& st, const std::vector<double>& dt) {
        return difference(step3(m, st, dt), step3(m, st, zero_dt));
    };
    const Rhs base = stab_of(s, delte);
    REQUIRE(scale_of(base.u) > 1e-4); // the term is not trivially zero
    auto scaled = [](const Rhs& r, double f) {
        Rhs o = r;
        for (double& x : o.u) x *= f;
        for (double& x : o.v) x *= f;
        return o;
    };
    const double f = GENERATE(2.0, 0.5, -1.0);
    CAPTURE(f);

    SECTION("time step") {
        std::vector<double> dt2 = delte;
        for (double& d : dt2) d *= std::abs(f);
        check_close(stab_of(s, dt2), scaled(base, std::abs(f)), 1e-15);
    }
    SECTION("advecting velocity u^n") {
        State s2 = s;
        for (double& x : s2.un) x *= f;
        for (double& x : s2.vn) x *= f;
        check_close(stab_of(s2, delte), scaled(base, f), 1e-15);
    }
    SECTION("old pressure p^n") {
        State s2 = s;
        for (double& x : s2.pold) x *= f;
        check_close(stab_of(s2, delte), scaled(base, f), 1e-15);
    }
    SECTION("independent of p^{n+1}") {
        State s2 = s;
        for (double& x : s2.pnew) x = 3.0 * x + 1.0;
        check_close(stab_of(s2, delte), base, 1e-15);
    }
}

TEST_CASE("Stabilisation term vanishes without flow, time step or old-pressure gradient", "[cbs][step3]") {
    const Mesh m = step_grid();
    State s = generic_state(m);
    const Rhs galerkin = pressure_ref(m, s.pnew);
    const double tol = 1e-13 * std::max(scale_of(galerkin.u), scale_of(galerkin.v));

    SECTION("u^n = 0") {
        s.un = s.vn = std::vector<double>(m.npoin, 0.0);
        check_close(step3(m, s, varying_dt(m.nelem)), galerkin, tol);
    }
    SECTION("dt = 0") {
        check_close(step3(m, s, std::vector<double>(m.nelem, 0.0)), galerkin, tol);
    }
    SECTION("constant p^n") {
        s.pold = std::vector<double>(m.npoin, 3.0);
        check_close(step3(m, s, varying_dt(m.nelem)), galerkin, tol);
    }
}

TEST_CASE("Stabilisation uses only the element-mean velocity of u^n", "[cbs][step3]") {
    // Permuting u^n among the nodes of a single element keeps u_bar.
    Mesh m = make_triangle_mesh(reference_triangles()[1]);
    compute_all_geometry(m);
    State s;
    s.un = {3.0, 0.0, 0.0};
    s.vn = {0.5, -1.0, 2.0};
    s.pnew = {0.3, -0.2, 0.9};
    s.pold = {1.0, 2.0, -0.5};
    const Rhs a = step3(m, s, {0.3});

    State b = s;
    b.un = {0.0, 1.0, 2.0};  // same mean 1
    b.vn = {2.0, 0.5, -1.0}; // same mean 0.5
    check_close(step3(m, b, {0.3}), a, 1e-15);
}

TEST_CASE("Stabilisation term conserves momentum", "[cbs][step3]") {
    const Mesh m = step_grid();
    const State s = generic_state(m);
    const Rhs stab = difference(step3(m, s, varying_dt(m.nelem)), step3(m, s, std::vector<double>(m.nelem, 0.0)));
    CHECK_THAT(std::accumulate(stab.u.begin(), stab.u.end(), 0.0), WithinAbs(0.0, 1e-15));
    CHECK_THAT(std::accumulate(stab.v.begin(), stab.v.end(), 0.0), WithinAbs(0.0, 1e-15));
}

TEST_CASE("Stabilisation uses each element's own time step", "[cbs][step3]") {
    const Mesh m = step_grid();
    const State s = generic_state(m);
    const auto delte = varying_dt(m.nelem);
    const int ie = 21;
    std::vector<double> changed = delte;
    changed[ie] *= 3.0;

    const Rhs a = step3(m, s, delte);
    const Rhs b = step3(m, s, changed);
    const std::set<int> nodes(m.intma[ie].begin(), m.intma[ie].end());
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i);
        if (nodes.count(i)) {
            CHECK((a.u[i] != b.u[i] || a.v[i] != b.v[i]));
        } else {
            CHECK(a.u[i] == b.u[i]);
            CHECK(a.v[i] == b.v[i]);
        }
    }
}

TEST_CASE("Uniform flow with a uniform old-pressure gradient", "[cbs][step3]") {
    // The continuous term u . grad(grad p^n) is zero. Discretely, the
    // stabilisation vanishes at interior nodes (int grad N_i = 0 there); at
    // boundary nodes the omitted boundary integral leaves
    //   -dt/2 * (u0 . closed int N_i n ds) * grad p^n,  closed int N_i n = sum L/2 n.
    const Mesh m = step_grid();
    const double dt = 0.2, u0 = 0.8, v0 = -0.6, gx = 1.5, gy = -2.0;
    State s;
    s.un = std::vector<double>(m.npoin, u0);
    s.vn = std::vector<double>(m.npoin, v0);
    s.pnew = std::vector<double>(m.npoin, 0.0);
    s.pold = sample(m, [&](double x, double y) { return gx * x + gy * y; });

    std::vector<double> bnx(m.npoin, 0.0), bny(m.npoin, 0.0);
    for (int ib = 0; ib < m.nboun; ++ib) {
        const auto& fn = m.face_norm[ib];
        for (int p : {m.iside[ib].node1, m.iside[ib].node2}) {
            bnx[p] += 0.5 * fn.length * fn.nx;
            bny[p] += 0.5 * fn.length * fn.ny;
        }
    }
    const Rhs r = step3(m, s, std::vector<double>(m.nelem, dt));
    const auto boundary = boundary_nodes(m);
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i, m.x[i], m.y[i]);
        const double flux = u0 * bnx[i] + v0 * bny[i];
        const double expected_u = boundary.count(i) ? -0.5 * dt * flux * gx : 0.0;
        const double expected_v = boundary.count(i) ? -0.5 * dt * flux * gy : 0.0;
        CHECK_THAT(r.u[i], WithinAbs(expected_u, 1e-14));
        CHECK_THAT(r.v[i], WithinAbs(expected_v, 1e-14));
    }
}

// =============================================================================
// Whole term, symmetry and interface
// =============================================================================

TEST_CASE("Step 3 matches the term-by-term reference for generic fields", "[cbs][step3]") {
    const Mesh m = step_grid();
    const State s = generic_state(m);
    const auto delte = varying_dt(m.nelem);
    const Rhs ref = step3_ref(m, s, delte);
    check_close(step3(m, s, delte), ref, 1e-13 * std::max(scale_of(ref.u), scale_of(ref.v)));
}

TEST_CASE("Step 3 treats x and y symmetrically", "[cbs][step3]") {
    // Mirror the mesh about y = x (reordering connectivity to stay CCW) and
    // swap the velocity components: the results swap.
    GridSpec g;
    g.nx = 4;
    g.ny = 4;
    Mesh m = make_grid_mesh(g);
    Mesh mirrored = m;
    std::swap(mirrored.x, mirrored.y);
    for (auto& e : mirrored.intma) std::swap(e[1], e[2]);
    compute_all_geometry(m);
    compute_all_geometry(mirrored);

    const State s = generic_state(m);
    State sm = s;
    std::swap(sm.un, sm.vn);
    const auto delte = varying_dt(m.nelem);
    const Rhs r = step3(m, s, delte);
    const Rhs rm = step3(mirrored, sm, delte);
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i);
        CHECK_THAT(rm.u[i], WithinAbs(r.v[i], 1e-14));
        CHECK_THAT(rm.v[i], WithinAbs(r.u[i], 1e-14));
    }
}

TEST_CASE("Step 3 on the Hinterkante mesh", "[cbs][step3][hinterkante]") {
    const std::string dir = HYDRAMA2D_CASE_DIR;
    Mesh m;
    REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", m));
    REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", m));
    compute_all_geometry(m);

    SECTION("uniform pressure gradient gives the exact acceleration") {
        const double dt = 6.6e-3;
        State s;
        s.un = s.vn = std::vector<double>(m.npoin, 0.0);
        s.pnew = sample(m, [](double x, double y) { return 0.5 - 0.02 * x + 0.01 * y; });
        s.pold = s.pnew;
        const Rhs du = velocity_change(m, s, std::vector<double>(m.nelem, dt));
        double worst = 0.0;
        for (int i = 0; i < m.npoin; ++i)
            worst = std::max({worst, std::abs(du.u[i] - 0.02 * dt), std::abs(du.v[i] + 0.01 * dt)});
        CAPTURE(worst);
        CHECK(worst < 1e-16);
    }
    SECTION("generic fields match the term-by-term reference") {
        State s;
        s.un = sample(m, [](double x, double y) { return 1.0 + 0.1 * std::sin(0.5 * x) * std::cos(y); });
        s.vn = sample(m, [](double x, double y) { return 0.05 * std::cos(0.3 * x) * std::sin(0.7 * y); });
        s.pnew = sample(m, [](double x, double y) { return 0.2 * std::cos(0.4 * x) * std::sin(0.5 * y); });
        s.pold = sample(m, [](double x, double y) { return 0.25 * std::cos(0.35 * x) * std::sin(0.45 * y); });
        const auto delte = varying_dt(m.nelem);
        const Rhs r = step3(m, s, delte);
        const Rhs ref = step3_ref(m, s, delte);
        double worst = 0.0;
        for (int i = 0; i < m.npoin; ++i)
            worst = std::max({worst, std::abs(r.u[i] - ref.u[i]), std::abs(r.v[i] - ref.v[i])});
        const double scale = std::max(scale_of(ref.u), scale_of(ref.v));
        CAPTURE(worst, scale);
        CHECK(worst < 1e-12 * scale);
    }
}
