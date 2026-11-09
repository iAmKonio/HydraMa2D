// Time step selection: compute_timestep() fills delte (one value per element).
//
// Per-element stable step (h = alen_e, minimum element height; |u| = largest
// nodal velocity magnitude of the element, floored at 1e-15):
//     dt_e = csafm * min( h^2 * Re / 2  (diffusive),  h / |u|  (convective) )
// with Re = 1 / ani. Modes:
//     dtfixed  > 0 : fixed step dtfix everywhere
//     dtfixed == -1: global minimum of dt_e everywhere
//     dtfixed == 0 : local time stepping (dt_e per element), but the global
//                    minimum is forced for transient runs (transient_on > 0)
//                    and for solver_opt > 1.

#include "LinearSolver.hpp"
#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinRel;

namespace {

// Three disjoint right isosceles triangles with leg lengths 1, 2 and 0.5,
// i.e. element sizes h = s / sqrt(2): 0.7071, 1.4142, 0.3536.
// Element ie uses nodes 3*ie, 3*ie+1, 3*ie+2.
Mesh three_element_mesh() {
    const double legs[3] = {1.0, 2.0, 0.5};
    std::vector<Point> nodes;
    std::vector<std::array<int, 3>> elems;
    for (int ie = 0; ie < 3; ++ie) {
        const double s = legs[ie], x0 = 10.0 * ie;
        nodes.push_back({x0, 0.0});
        nodes.push_back({x0 + s, 0.0});
        nodes.push_back({x0, s});
        elems.push_back({3 * ie, 3 * ie + 1, 3 * ie + 2});
    }
    Mesh m = make_mesh(nodes, elems);
    m.compute_shape_function_derivatives();
    m.compute_element_sizes();
    return m;
}

const double kH[3] = {1.0 / std::sqrt(2.0), 2.0 / std::sqrt(2.0), 0.5 / std::sqrt(2.0)};

double stable_dt(double h, double umax, double csafm, double re) {
    return csafm * std::min(h * h * re / 2.0, h / std::max(umax, 1.0e-15));
}

// Nodal velocity magnitudes giving element maxima {umax0, umax1, umax2};
// the other two nodes of each element get smaller values.
std::vector<double> nodal_velocity(const std::array<double, 3>& umax) {
    std::vector<double> v;
    for (double u : umax) {
        v.push_back(0.25 * u);
        v.push_back(u);
        v.push_back(0.5 * u);
    }
    return v;
}

struct Params {
    int solver_opt = 1;
    int dtfixed = 0;
    int transient_on = 0;
    double dtfix = 1.0e-3;
    double re = 100.0;
    double csafm = 0.6;
};

std::vector<double> run(const Mesh& m, const Params& p, const std::vector<double>& vel) {
    std::vector<double> delte(7, -123.0); // wrong size and contents on purpose
    compute_timestep(p.solver_opt, p.dtfixed, p.transient_on, p.dtfix, 1.0 / p.re, p.csafm, m, vel, delte);
    return delte;
}

} // namespace

TEST_CASE("Test mesh element sizes", "[timestep]") {
    const Mesh m = three_element_mesh();
    for (int ie = 0; ie < 3; ++ie) CHECK_THAT(m.alen_e[ie], WithinRel(kH[ie], 1e-14));
}

TEST_CASE("Local time step: diffusive limit h^2 Re / 2 at rest", "[timestep]") {
    // Zero velocity is floored to 1e-15, so the convective limit is huge and
    // never produces a division by zero.
    const Mesh m = three_element_mesh();
    const double re = GENERATE(1.0, 100.0, 1000.0);
    Params p;
    p.re = re;
    CAPTURE(re);

    const auto dt = run(m, p, std::vector<double>(m.npoin, 0.0));
    REQUIRE(dt.size() == 3);
    for (int ie = 0; ie < 3; ++ie) {
        CAPTURE(ie);
        CHECK(std::isfinite(dt[ie]));
        CHECK_THAT(dt[ie], WithinRel(p.csafm * kH[ie] * kH[ie] * re / 2.0, 1e-14));
    }
}

TEST_CASE("Local time step: convective limit h / |u| at high velocity", "[timestep]") {
    const Mesh m = three_element_mesh();
    Params p;
    p.re = 1.0e6; // diffusive limit irrelevant
    const std::array<double, 3> umax = {2.0, 3.0, 0.5};

    const auto dt = run(m, p, nodal_velocity(umax));
    for (int ie = 0; ie < 3; ++ie) {
        CAPTURE(ie);
        CHECK_THAT(dt[ie], WithinRel(p.csafm * kH[ie] / umax[ie], 1e-14));
    }
}

TEST_CASE("Local time step switches limits at the crossover velocity", "[timestep]") {
    // h / u == h^2 Re / 2  <=>  u* = 2 / (h Re)
    const Mesh m = three_element_mesh();
    Params p;
    p.re = 50.0;
    std::array<double, 3> ustar{};
    for (int ie = 0; ie < 3; ++ie) ustar[ie] = 2.0 / (kH[ie] * p.re);

    SECTION("below u*: diffusive limit") {
        const auto dt = run(m, p, nodal_velocity({0.5 * ustar[0], 0.5 * ustar[1], 0.5 * ustar[2]}));
        for (int ie = 0; ie < 3; ++ie)
            CHECK_THAT(dt[ie], WithinRel(p.csafm * kH[ie] * kH[ie] * p.re / 2.0, 1e-14));
    }
    SECTION("above u*: convective limit") {
        const auto dt = run(m, p, nodal_velocity({2.0 * ustar[0], 2.0 * ustar[1], 2.0 * ustar[2]}));
        for (int ie = 0; ie < 3; ++ie)
            CHECK_THAT(dt[ie], WithinRel(p.csafm * kH[ie] / (2.0 * ustar[ie]), 1e-14));
    }
    SECTION("at u*: both limits coincide") {
        const auto dt = run(m, p, nodal_velocity(ustar));
        for (int ie = 0; ie < 3; ++ie)
            CHECK_THAT(dt[ie], WithinRel(p.csafm * kH[ie] * kH[ie] * p.re / 2.0, 1e-14));
    }
}

TEST_CASE("Element velocity is the largest nodal velocity of the element", "[timestep]") {
    const Mesh m = three_element_mesh();
    Params p;
    p.re = 1.0e6;
    // The maximum sits at a different local node in each element.
    const std::vector<double> vel = {4.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0};

    const auto dt = run(m, p, vel);
    for (int ie = 0; ie < 3; ++ie) {
        CAPTURE(ie);
        CHECK_THAT(dt[ie], WithinRel(p.csafm * kH[ie] / 4.0, 1e-14));
    }
}

TEST_CASE("Time step is proportional to the safety factor", "[timestep]") {
    const Mesh m = three_element_mesh();
    const auto vel = nodal_velocity({0.3, 1.0, 7.0});
    Params p1, p2;
    p1.csafm = 1.0;
    p2.csafm = 0.35;

    const auto dt1 = run(m, p1, vel);
    const auto dt2 = run(m, p2, vel);
    for (int ie = 0; ie < 3; ++ie) CHECK_THAT(dt2[ie], WithinRel(0.35 * dt1[ie], 1e-14));
}

TEST_CASE("Time step mode selection", "[timestep]") {
    enum class Mode { Fixed, Global, Local };
    struct Case {
        const char* name;
        int solver_opt, dtfixed, transient_on;
        Mode mode;
    };
    const Case c = GENERATE(
        Case{"fixed, steady", 1, 1, 0, Mode::Fixed},
        Case{"fixed, transient", 1, 1, 1, Mode::Fixed},
        Case{"fixed, solver_opt 2", 2, 1, 0, Mode::Fixed},
        Case{"fixed, any positive dtfixed", 1, 3, 0, Mode::Fixed},
        Case{"global minimum, steady", 1, -1, 0, Mode::Global},
        Case{"global minimum, transient", 1, -1, 1, Mode::Global},
        Case{"local, steady", 1, 0, 0, Mode::Local},
        Case{"local requested, transient -> global", 1, 0, 1, Mode::Global},
        Case{"local requested, solver_opt 2 -> global", 2, 0, 0, Mode::Global});
    CAPTURE(c.name, c.solver_opt, c.dtfixed, c.transient_on);

    const Mesh m = three_element_mesh();
    Params p;
    p.solver_opt = c.solver_opt;
    p.dtfixed = c.dtfixed;
    p.transient_on = c.transient_on;
    p.dtfix = 2.5e-4;
    p.re = 200.0;
    const std::array<double, 3> umax = {1.0, 0.2, 3.0};

    std::array<double, 3> local{};
    for (int ie = 0; ie < 3; ++ie) local[ie] = stable_dt(kH[ie], umax[ie], p.csafm, p.re);
    const double global = *std::min_element(local.begin(), local.end());
    // Sanity: the three local steps differ, so the modes are distinguishable.
    REQUIRE(local[0] != local[1]);
    REQUIRE(local[1] != local[2]);

    const auto dt = run(m, p, nodal_velocity(umax));
    REQUIRE(dt.size() == 3);
    for (int ie = 0; ie < 3; ++ie) {
        CAPTURE(ie);
        switch (c.mode) {
        case Mode::Fixed:  CHECK(dt[ie] == p.dtfix); break;
        case Mode::Global: CHECK_THAT(dt[ie], WithinRel(global, 1e-14)); break;
        case Mode::Local:  CHECK_THAT(dt[ie], WithinRel(local[ie], 1e-14)); break;
        }
    }
}

TEST_CASE("Transient runs always get a uniform time step", "[timestep]") {
    // Solver::run() advances physical time by delte[0], which is only
    // meaningful if every element uses the same step.
    const Mesh m = three_element_mesh();
    const int dtfixed = GENERATE(-1, 0, 1);
    const int solver_opt = GENERATE(1, 2);
    CAPTURE(dtfixed, solver_opt);

    Params p;
    p.transient_on = 1;
    p.dtfixed = dtfixed;
    p.solver_opt = solver_opt;
    const auto dt = run(m, p, nodal_velocity({0.1, 2.0, 0.7}));
    CHECK(dt[0] == dt[1]);
    CHECK(dt[1] == dt[2]);
}

TEST_CASE("Fixed time step ignores mesh, velocity and Reynolds number", "[timestep]") {
    const Mesh m = three_element_mesh();
    Params p;
    p.dtfixed = 1;
    p.dtfix = 0.125;
    p.re = 1.0e-3; // would give a tiny diffusive limit
    const auto dt = run(m, p, std::vector<double>(m.npoin, 1.0e6));
    for (double d : dt) CHECK(d == 0.125);
}

TEST_CASE("Time step on a uniform grid is uniform in every mode", "[timestep]") {
    GridSpec g;
    g.nx = 4;
    g.ny = 3;
    Mesh m = make_grid_mesh(g);
    m.compute_shape_function_derivatives();
    m.compute_element_sizes();
    const double h = g.hx() * g.hy() / std::hypot(g.hx(), g.hy());

    const int dtfixed = GENERATE(-1, 0);
    CAPTURE(dtfixed);
    std::vector<double> delte;
    compute_timestep(1, dtfixed, 0, 0.0, 1.0 / 100.0, 0.6, m, std::vector<double>(m.npoin, 1.5), delte);
    REQUIRE(static_cast<int>(delte.size()) == m.nelem);
    for (double d : delte) CHECK_THAT(d, WithinRel(stable_dt(h, 1.5, 0.6, 100.0), 1e-12));
}

TEST_CASE("Undocumented dtfixed values below -1 select local time stepping",
          "[timestep][characterization]") {
    // Documents current behaviour: only -1, 0 and 1 are documented in the
    // *.control file, but any dtfixed < -1 behaves like 0 (local stepping).
    const Mesh m = three_element_mesh();
    Params p;
    p.dtfixed = GENERATE(-2, -5);
    CAPTURE(p.dtfixed);
    const std::array<double, 3> umax = {1.0, 0.2, 3.0};

    const auto dt = run(m, p, nodal_velocity(umax));
    for (int ie = 0; ie < 3; ++ie)
        CHECK_THAT(dt[ie], WithinRel(stable_dt(kH[ie], umax[ie], p.csafm, p.re), 1e-14));
}
