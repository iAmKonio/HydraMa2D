// Coupling of the eddy viscosity into the CBS momentum step and the time
// step:
//  * compute_step1(mesh, ani, nut_elem, ...): laminar Laplacian term plus the
//    symmetric turbulent stress  -int grad N_i . nu_t (grad u + grad u^T);
//    bit-identical to the laminar overload for empty or zero nu_t; checked
//    against an independent tensor reference, for objectivity (rigid
//    rotation), the transpose term, momentum conservation and the exact
//    discrete dissipation identity  sum_i u_i . rhs_i = -sum_e A_e nu_t,e S_e^2.
//  * compute_timestep(..., nu_eff_elem, ...): diffusive limit with the
//    effective viscosity per element; bit-identical to the legacy overload
//    for nu_eff = 1/Re.

#include "LinearSolver.hpp"
#include "MeshTestUtils.hpp"
#include "Step1.hpp"
#include "geometry/FieldOps.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numeric>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

Mesh coupling_grid(int nx = 5, int ny = 4) {
    GridSpec g;
    g.x0 = 0.2;
    g.y0 = -0.3;
    g.lx = 1.2;
    g.ly = 0.9;
    g.nx = nx;
    g.ny = ny;
    Mesh m = make_grid_mesh(g);
    compute_all_geometry(m);
    return m;
}

double tg_u(double x, double y) { return 0.4 + std::sin(2.0 * x) * std::cos(1.5 * y); }
double tg_v(double x, double y) { return -0.2 - std::cos(2.0 * x) * std::sin(1.5 * y); }

std::vector<double> varying_nut(const Mesh& m) {
    std::vector<double> nut(m.nelem);
    for (int ie = 0; ie < m.nelem; ++ie) nut[ie] = 0.01 * (1.0 + 0.8 * std::sin(0.7 * ie));
    return nut;
}

struct Rhs {
    std::vector<double> u, v;
};

Rhs laminar(const Mesh& m, double nu, const std::vector<double>& dt, const std::vector<double>& u,
            const std::vector<double>& v) {
    Rhs r;
    compute_step1(m, nu, dt, u, v, r.u, r.v);
    return r;
}

Rhs turbulent(const Mesh& m, double nu, const std::vector<double>& nut, const std::vector<double>& dt,
              const std::vector<double>& u, const std::vector<double>& v) {
    Rhs r{std::vector<double>(2, 7.0), {}}; // stale, wrongly sized output
    compute_step1(m, nu, nut, dt, u, v, r.u, r.v);
    return r;
}

// Turbulent part only: turbulent - laminar Step 1.
Rhs turbulent_part(const Mesh& m, const std::vector<double>& nut, const std::vector<double>& u,
                   const std::vector<double>& v) {
    const std::vector<double> dt(m.nelem, 0.0);
    Rhs t = turbulent(m, 0.0, nut, dt, u, v);
    const Rhs l = laminar(m, 0.0, dt, u, v);
    for (int i = 0; i < m.npoin; ++i) {
        t.u[i] -= l.u[i];
        t.v[i] -= l.v[i];
    }
    return t;
}

// Independent reference: tau = nu_t (G + G^T), rhs_i -= A_e grad N_i . tau.
Rhs stress_reference(const Mesh& m, const std::vector<double>& nut, const std::vector<double>& u,
                     const std::vector<double>& v) {
    Rhs r{std::vector<double>(m.npoin, 0.0), std::vector<double>(m.npoin, 0.0)};
    std::vector<VelocityGradient> g;
    compute_velocity_gradients(m, u, v, g);
    for (int ie = 0; ie < m.nelem; ++ie) {
        const double G[2][2] = {{g[ie].dudx, g[ie].dudy}, {g[ie].dvdx, g[ie].dvdy}};
        double tau[2][2];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) tau[i][j] = nut[ie] * (G[i][j] + G[j][i]);
        const double area = 0.5 * m.detJ[ie];
        for (int k = 0; k < 3; ++k) {
            const double gN[2] = {m.dNkdx[ie].dNdx[k], m.dNkdx[ie].dNdy[k]};
            const int ip = m.intma[ie][k];
            r.u[ip] -= area * (gN[0] * tau[0][0] + gN[1] * tau[0][1]);
            r.v[ip] -= area * (gN[0] * tau[1][0] + gN[1] * tau[1][1]);
        }
    }
    return r;
}

bool is_boundary_node(const Mesh& m, int ip) {
    for (const auto& s : m.iside)
        if (s.node1 == ip || s.node2 == ip) return true;
    return false;
}

} // namespace

// ---------------------------------------------------------------------------
// Step 1
// ---------------------------------------------------------------------------

TEST_CASE("Turbulent Step 1 with zero or no eddy viscosity is the laminar Step 1 bit for bit",
          "[coupling][step1][laminar]") {
    const Mesh m = coupling_grid();
    const double nu = GENERATE(0.0, 1.0 / 200.0, 0.1);
    CAPTURE(nu);
    const auto dt = varying_dt(m.nelem);
    const auto u = sample(m, tg_u), v = sample(m, tg_v);
    const Rhs ref = laminar(m, nu, dt, u, v);

    const Rhs zero = turbulent(m, nu, std::vector<double>(m.nelem, 0.0), dt, u, v);
    CHECK(zero.u == ref.u);
    CHECK(zero.v == ref.v);
    const Rhs empty = turbulent(m, nu, {}, dt, u, v);
    CHECK(empty.u == ref.u);
    CHECK(empty.v == ref.v);

    // Elements with nu_t = 0 inside a turbulent field contribute exactly the
    // laminar terms: only nodes touching a turbulent element change.
    std::vector<double> nut(m.nelem, 0.0);
    nut[0] = 0.05;
    const Rhs one = turbulent(m, nu, nut, dt, u, v);
    for (int ip = 0; ip < m.npoin; ++ip) {
        const auto& e = m.intma[0];
        const bool touched = ip == e[0] || ip == e[1] || ip == e[2];
        if (!touched) {
            CHECK(one.u[ip] == ref.u[ip]);
            CHECK(one.v[ip] == ref.v[ip]);
        }
    }
    CHECK_THROWS_AS(turbulent(m, nu, std::vector<double>(3, 0.0), dt, u, v), std::invalid_argument);
}

TEST_CASE("Turbulent Step 1 adds the symmetric stress of an independent tensor reference",
          "[coupling][step1]") {
    const Mesh m = coupling_grid();
    const auto u = sample(m, tg_u), v = sample(m, tg_v);
    const auto nut = varying_nut(m);
    const Rhs t = turbulent_part(m, nut, u, v);
    const Rhs ref = stress_reference(m, nut, u, v);
    const double scale = std::max(scale_of(ref.u), scale_of(ref.v));
    for (int i = 0; i < m.npoin; ++i) {
        CAPTURE(i);
        CHECK_THAT(t.u[i], WithinAbs(ref.u[i], 1e-13 * scale));
        CHECK_THAT(t.v[i], WithinAbs(ref.v[i], 1e-13 * scale));
    }

    SECTION("with time step and molecular viscosity: full = laminar + turbulent part") {
        const double nu = 0.02;
        const auto dt = varying_dt(m.nelem);
        const Rhs full = turbulent(m, nu, nut, dt, u, v);
        const Rhs lam = laminar(m, nu, dt, u, v);
        for (int i = 0; i < m.npoin; ++i) {
            CHECK_THAT(full.u[i], WithinAbs(lam.u[i] + ref.u[i], 1e-13 * scale));
            CHECK_THAT(full.v[i], WithinAbs(lam.v[i] + ref.v[i], 1e-13 * scale));
        }
    }
}

TEST_CASE("Turbulent stress is objective: a rigid rotation produces no stress", "[coupling][step1]") {
    // u = U - Omega y, v = V + Omega x: G + G^T = 0 in every element, whatever
    // nu_t does. A Laplacian form nu_t Lap(u) would not vanish for varying nu_t.
    const Mesh m = coupling_grid();
    const double omega = 1.7;
    const auto u = sample(m, [&](double, double y) { return 0.3 - omega * y; });
    const auto v = sample(m, [&](double x, double) { return -0.1 + omega * x; });
    const auto nut = varying_nut(m);
    const Rhs t = turbulent_part(m, nut, u, v);
    for (int i = 0; i < m.npoin; ++i) {
        CHECK_THAT(t.u[i], WithinAbs(0.0, 1e-14));
        CHECK_THAT(t.v[i], WithinAbs(0.0, 1e-14));
    }

    // Laplacian form for comparison: -A nu_t grad N_i . grad u.
    std::vector<double> lap(m.npoin, 0.0);
    std::vector<VelocityGradient> g;
    compute_velocity_gradients(m, u, v, g);
    for (int ie = 0; ie < m.nelem; ++ie)
        for (int k = 0; k < 3; ++k)
            lap[m.intma[ie][k]] -= 0.5 * m.detJ[ie] * nut[ie] *
                                   (m.dNkdx[ie].dNdx[k] * g[ie].dudx + m.dNkdx[ie].dNdy[k] * g[ie].dudy);
    CHECK(scale_of(lap) > 1e-3);
}

TEST_CASE("Turbulent stress: the transpose term couples shear into the cross momentum", "[coupling][step1]") {
    // u = gamma y, v = 0, nu_t = a + b x (element means of a linear field).
    // d/dx (nu_t du/dy) = b gamma acts on the v-momentum; the u-momentum term
    // d/dy (nu_t du/dy) vanishes. Interior nodes: rhs_v / M_i = b gamma exactly.
    const Mesh m = coupling_grid(6, 5);
    const auto mass = lumped_nodal_mass(m);
    const double gamma = 2.5, a = 0.02, b = 0.03;
    const auto u = sample(m, [&](double, double y) { return gamma * y; });
    const std::vector<double> v(m.npoin, 0.0);
    std::vector<double> nut;
    element_mean(m, sample(m, [&](double x, double) { return a + b * x; }), nut);

    const Rhs t = turbulent_part(m, nut, u, v);
    int interior = 0;
    for (int ip = 0; ip < m.npoin; ++ip) {
        if (is_boundary_node(m, ip)) continue;
        ++interior;
        CHECK_THAT(t.v[ip] / mass[ip], WithinAbs(b * gamma, 1e-12));
        CHECK_THAT(t.u[ip] / mass[ip], WithinAbs(0.0, 1e-12));
    }
    CHECK(interior == 20);
}

TEST_CASE("Turbulent stress conserves momentum and dissipates kinetic energy", "[coupling][step1]") {
    const Mesh m = coupling_grid();
    const auto u = sample(m, tg_u), v = sample(m, tg_v);
    const auto nut = varying_nut(m);
    const Rhs t = turbulent_part(m, nut, u, v);

    // sum_i grad N_i = 0: internal stresses cancel.
    CHECK_THAT(std::accumulate(t.u.begin(), t.u.end(), 0.0), WithinAbs(0.0, 1e-14));
    CHECK_THAT(std::accumulate(t.v.begin(), t.v.end(), 0.0), WithinAbs(0.0, 1e-14));

    // sum_i (u_i rhs_u_i + v_i rhs_v_i) = -sum_e A_e nu_t,e S_e^2, S^2 = 2 S_ij S_ij.
    double power = 0.0;
    for (int i = 0; i < m.npoin; ++i) power += u[i] * t.u[i] + v[i] * t.v[i];
    std::vector<VelocityGradient> g;
    compute_velocity_gradients(m, u, v, g);
    double dissipation = 0.0;
    for (int ie = 0; ie < m.nelem; ++ie) {
        const double s = strain_rate_magnitude(g[ie]);
        dissipation += 0.5 * m.detJ[ie] * nut[ie] * s * s;
    }
    CHECK(dissipation > 0.0);
    CHECK_THAT(power, WithinRel(-dissipation, 1e-12));
}

// ---------------------------------------------------------------------------
// Time step
// ---------------------------------------------------------------------------

TEST_CASE("compute_timestep with nu_eff = 1/Re is the legacy time step bit for bit", "[coupling][timestep][laminar]") {
    const Mesh m = coupling_grid();
    std::vector<double> speed(m.npoin);
    for (int ip = 0; ip < m.npoin; ++ip) speed[ip] = 0.1 + std::abs(tg_u(m.x[ip], m.y[ip]));
    const double re = GENERATE(10.0, 200.0, 1.0e5);
    const int dtfixed = GENERATE(-1, 0, 1);
    const int transient = GENERATE(0, 1);
    const int solver_opt = GENERATE(1, 2);
    CAPTURE(re, dtfixed, transient, solver_opt);
    const double ani = 1.0 / re;

    std::vector<double> legacy, coupled(5, -1.0);
    compute_timestep(solver_opt, dtfixed, transient, 1.0e-3, ani, 0.6, m, speed, legacy);
    compute_timestep(solver_opt, dtfixed, transient, 1.0e-3, std::vector<double>(m.nelem, ani), 0.6, m, speed,
                     coupled);
    CHECK(coupled == legacy);
}

TEST_CASE("compute_timestep uses the effective viscosity of each element", "[coupling][timestep]") {
    const Mesh m = coupling_grid();
    const std::vector<double> speed(m.npoin, 1.0e-6); // diffusion-dominated
    std::vector<double> nu_eff(m.nelem);
    for (int ie = 0; ie < m.nelem; ++ie) nu_eff[ie] = 0.01 * (1 + ie % 4);
    const double csafm = 0.6;

    SECTION("local time stepping: dt_e = csafm * min(h^2 / (2 nu_eff,e), h / |u|)") {
        std::vector<double> dt;
        compute_timestep(1, 0, 0, 1.0, nu_eff, csafm, m, speed, dt);
        for (int ie = 0; ie < m.nelem; ++ie) {
            const double h = m.alen_e[ie];
            CHECK_THAT(dt[ie], WithinRel(csafm * h * h / (2.0 * nu_eff[ie]), 1e-14));
        }
    }

    SECTION("global time step: minimum over elements, limited by the largest nu_eff") {
        std::vector<double> dt, lam;
        compute_timestep(1, -1, 1, 1.0, nu_eff, csafm, m, speed, dt);
        compute_timestep(1, -1, 1, 1.0, 0.01, csafm, m, speed, lam);
        const double h = m.alen_e[0]; // uniform grid: all elements have the same h
        CHECK_THAT(dt[0], WithinRel(csafm * h * h / (2.0 * 0.04), 1e-14));
        CHECK_THAT(dt[0], WithinRel(lam[0] / 4.0, 1e-14));
    }

    SECTION("fixed time step ignores nu_eff") {
        std::vector<double> dt;
        compute_timestep(1, 1, 1, 2.5e-4, nu_eff, csafm, m, speed, dt);
        CHECK(dt == std::vector<double>(m.nelem, 2.5e-4));
    }

    SECTION("size mismatch") {
        std::vector<double> dt;
        CHECK_THROWS_AS(compute_timestep(1, 0, 0, 1.0, std::vector<double>(2, 0.1), csafm, m, speed, dt),
                        std::invalid_argument);
    }
}
