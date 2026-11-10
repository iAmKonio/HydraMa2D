// Setup reuse for scaled operators (global / uniform time stepping, A = dt K):
//   * ScaledPreconditioner: z = factor M^{-1} r, transparent for factor 1
//   * homogeneity of every preconditioner: M(c A)^{-1} = (1/c) M(A)^{-1}
//   * uniform_scaling(): detection of A = c A_ref from the element data
//   * PressureSolver: one numeric setup for a whole uniform-dt sequence,
//     bit-identical results for constant dt, equal results for varying dt,
//     a new setup for local time steps and geometry changes, a new structure
//     for a new mesh, orderings computed once, the rebuild option
//   * configuration of preconditioner_reuse

#include "BoundaryConditions.hpp"
#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "LinalgTestUtils.hpp"
#include "PressureSolver.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinRel;

namespace {

Mesh grid_mesh(int nx = 12, int ny = 6) {
    GridSpec g;
    g.lx = 2.0;
    g.ly = 1.0;
    g.nx = nx;
    g.ny = ny;
    Mesh m = make_grid_mesh(g);
    set_flags_by_side_id(m, {{1, 500}, {2, 504}, {3, 500}, {4, 503}});
    CaptureStdout quiet;
    compute_all_geometry(m);
    return m;
}

// Non-affine deformation of the grid: changes the element stiffness.
void deform(Mesh& m) {
    for (int ip = 0; ip < m.npoin; ++ip) m.x[ip] += 0.05 * m.y[ip] * m.y[ip];
    CaptureStdout quiet;
    compute_all_geometry(m);
}

PressureBC outlet_bc(const Mesh& m) {
    CaptureStdout quiet;
    return setup_pressure_bcs(m, 0);
}

SolverConfig config(const std::string& preconditioner, const std::string& reuse, const std::string& method = "cg",
                    double rel = 1e-12) {
    SolverConfig c;
    c.method = method;
    c.preconditioner = preconditioner;
    c.reordering = preconditioner == "jacobi" || preconditioner == "none" ? "none" : "rcm";
    c.preconditioner_reuse = reuse;
    c.relToler = rel;
    c.absToler = 0.0;
    c.gmres_restart = 100;
    return c;
}

// One time step: operator for uniform dt (or a local dt field), then a solve.
struct Stepper {
    const Mesh& mesh;
    std::vector<double> elcoe2, pdiag;
    std::vector<std::array<double, 3>> gstif;

    void operator()(PressureSolver& s, const std::vector<double>& dt) {
        update_lhs(mesh, dt, elcoe2, pdiag, gstif);
        s.update(mesh, gstif, pdiag);
    }
};

std::vector<double> uniform(int nelem, double dt) { return std::vector<double>(nelem, dt); }

std::vector<double> local(int nelem, int k) {
    std::vector<double> dt(nelem);
    for (int e = 0; e < nelem; ++e) dt[e] = 1e-3 * (1.0 + 0.5 * std::sin(0.7 * e + k));
    return dt;
}

double max_abs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

} // namespace

// ---------------------------------------------------------------------------
// ScaledPreconditioner and homogeneity
// ---------------------------------------------------------------------------

TEST_CASE("ScaledPreconditioner scales the inner preconditioner", "[reuse][preconditioner]") {
    const la::CsrMatrix a = from_dense(laplacian_1d(9, 2.0), true);
    la::IncompleteCholesky ic;
    ic.setup(a);
    const std::vector<double> r = test_vector(9);
    std::vector<double> z_inner(9), z(9);
    ic.apply(r, z_inner);

    const la::ScaledPreconditioner scaled(ic, 0.25);
    scaled.apply(r, z);
    for (int i = 0; i < 9; ++i) CHECK(z[i] == 0.25 * z_inner[i]);
    CHECK(scaled.factor() == 0.25);
    CHECK(&scaled.inner() == &ic);
    CHECK(scaled.size() == ic.size());
    CHECK(scaled.name() == "ic0");
    CHECK(scaled.traits().symmetric == ic.traits().symmetric);
    CHECK(scaled.pointwise_scaling().empty());

    SECTION("factor 1 is bit-for-bit transparent") {
        const la::ScaledPreconditioner identity(ic, 1.0);
        std::vector<double> z1(9);
        identity.apply(r, z1);
        CHECK(z1 == z_inner);
    }
    SECTION("invalid factors and setup calls are rejected") {
        for (double bad : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
            CAPTURE(bad);
            CHECK_THROWS_AS(la::ScaledPreconditioner(ic, bad), std::invalid_argument);
        }
        la::ScaledPreconditioner view(ic, 2.0);
        CHECK_THROWS_AS(view.setup(a), std::logic_error);
        CHECK_THROWS_AS(view.update(a, la::OperatorChange::Values), std::logic_error);
    }
}

TEST_CASE("Preconditioners are homogeneous: M(c A)^{-1} = (1/c) M(A)^{-1}", "[reuse][preconditioner]") {
    const std::string name = GENERATE(std::string("jacobi"), std::string("ssor"), std::string("ic0"),
                                      std::string("ilu0"), std::string("rcm+ic0"), std::string("rcm+ilu0"));
    const double c = GENERATE(0.37, 4.0);
    CAPTURE(name, c);
    // Pressure-type SPD matrix (constrained grid Laplacian).
    PoissonSystem s = make_poisson_system(make_grid_mesh({0.0, 0.0, 1.0, 0.8, 6, 5}), varying_dt(60));
    const la::P1CsrAssembler assembler(s.mesh);
    la::CsrMatrix a = assembler.create_matrix(), ac;
    assembler.assemble_ebe(s.mesh, s.gstif, s.pdiag, a);
    la::constrain(a, la::DirichletConstraints({0}, {0.0}), ac);
    la::CsrMatrix cac = ac;
    for (double& v : cac.values()) v *= c;

    la::PreconditionerOptions o;
    o.ssor_omega = 1.3;
    std::string base = name;
    if (name.rfind("rcm+", 0) == 0) {
        o.ordering = "rcm";
        base = name.substr(4);
    }
    auto m = la::make_preconditioner(base, o);
    auto mc = la::make_preconditioner(base, o);
    CHECK(m->traits().homogeneous);
    m->setup(ac);
    mc->setup(cac);
    const DenseMatrix minv = inverse_preconditioner_matrix(*m);
    const DenseMatrix minv_c = inverse_preconditioner_matrix(*mc);
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < minv.size(); ++i)
        for (size_t j = 0; j < minv.size(); ++j) {
            worst = std::max(worst, std::abs(minv_c[i][j] - minv[i][j] / c));
            scale = std::max(scale, std::abs(minv[i][j] / c));
        }
    CHECK(worst <= 1e-12 * scale);
}

TEST_CASE("Homogeneity survives the IC(0) breakdown shift", "[reuse][preconditioner]") {
    // The shift decisions are scale invariant: same alpha for A and c A.
    const la::CsrMatrix a = from_dense(kershaw_matrix(), true);
    la::CsrMatrix a7 = a;
    for (double& v : a7.values()) v *= 7.0;
    la::IncompleteCholesky m, m7;
    m.setup(a);
    m7.setup(a7);
    CHECK(m.shift() == m7.shift());
    CHECK(m.attempts() == m7.attempts());
    CHECK(max_abs_diff(inverse_preconditioner_matrix(m7), [&] {
              DenseMatrix d = inverse_preconditioner_matrix(m);
              for (auto& row : d)
                  for (double& v : row) v /= 7.0;
              return d;
          }()) < 1e-12);
}

TEST_CASE("Only genuinely homogeneous preconditioners claim homogeneity", "[reuse][preconditioner]") {
    CHECK_FALSE(la::IdentityPreconditioner().traits().homogeneous); // M = I for every A
    CHECK_FALSE(DensePreconditioner(laplacian_1d(2)).traits().homogeneous);
    CHECK(la::JacobiPreconditioner().traits().homogeneous);
    CHECK(la::SsorPreconditioner().traits().homogeneous);
    CHECK(la::IncompleteCholesky().traits().homogeneous);
    CHECK(la::IncompleteLU().traits().homogeneous);
    CHECK(la::ReorderedPreconditioner::rcm(std::make_unique<la::IncompleteLU>())->traits().homogeneous);
}

// ---------------------------------------------------------------------------
// Detection of A = c A_ref
// ---------------------------------------------------------------------------

TEST_CASE("uniform_scaling detects uniformly scaled element data", "[reuse]") {
    const Mesh m = grid_mesh();
    std::vector<double> e2, p1, p2;
    std::vector<std::array<double, 3>> g1, g2;
    update_lhs(m, uniform(m.nelem, 1e-3), e2, p1, g1);

    SECTION("identical data: exactly 1") {
        const auto c = uniform_scaling(g1, p1, g1, p1);
        REQUIRE(c);
        CHECK(*c == 1.0);
    }
    SECTION("a new uniform time step: the ratio of the time steps") {
        update_lhs(m, uniform(m.nelem, 3.7e-3), e2, p2, g2);
        const auto c = uniform_scaling(g2, p2, g1, p1);
        REQUIRE(c);
        CHECK_THAT(*c, WithinRel(3.7, 1e-14));
    }
    SECTION("local time steps are not a uniform scaling") {
        update_lhs(m, local(m.nelem, 0), e2, p2, g2);
        CHECK_FALSE(uniform_scaling(g2, p2, g1, p1));
    }
    SECTION("a single changed element is detected") {
        g2 = g1;
        p2 = p1;
        // Perturb the element's largest coupling (on this right-triangle
        // grid the hypotenuse coupling is exactly 0 and would stay 0).
        auto& g = g2[m.nelem / 2];
        int k = 0;
        for (int j = 1; j < 3; ++j)
            if (std::abs(g[j]) > std::abs(g[k])) k = j;
        REQUIRE(g[k] != 0.0);
        g[k] *= 1.0 + 1e-9;
        CHECK_FALSE(uniform_scaling(g2, p2, g1, p1));
        g2 = g1;
        p2[m.npoin / 3] *= 1.0 + 1e-9;
        CHECK_FALSE(uniform_scaling(g2, p2, g1, p1));
    }
    SECTION("rounding-level differences are accepted") {
        g2 = g1;
        p2 = p1;
        for (auto& g : g2)
            for (double& v : g) v *= 1.0 + 1e-15;
        CHECK(uniform_scaling(g2, p2, g1, p1));
    }
    SECTION("inconsistent or degenerate data") {
        std::vector<double> short_p(p1.begin(), p1.end() - 1);
        CHECK_FALSE(uniform_scaling(g1, short_p, g1, p1));
        std::vector<double> zero(p1.size(), 0.0);
        CHECK_FALSE(uniform_scaling(g1, p1, g1, zero));
        CHECK_FALSE(uniform_scaling(g1, zero, g1, p1));
    }
}

// ---------------------------------------------------------------------------
// PressureSolver setup reuse
// ---------------------------------------------------------------------------

TEST_CASE("Uniform time steps: one numeric setup, then rescaling", "[reuse][pressure_solver]") {
    const std::string pre = GENERATE(std::string("ic0"), std::string("ilu0"), std::string("ssor"));
    CAPTURE(pre);
    const Mesh m = grid_mesh();
    const PressureBC bc = outlet_bc(m);
    PressureSolver reuse(config(pre, "rescale"), bc), rebuild(config(pre, "rebuild"), bc);
    CHECK(reuse.setup_reuse_enabled());
    CHECK_FALSE(rebuild.setup_reuse_enabled());
    Stepper step_a{m, {}, {}, {}}, step_b{m, {}, {}, {}};
    std::vector<double> pa(m.npoin, 0.0), pb(m.npoin, 0.0);

    const std::vector<double> dts = {2e-3, 2.5e-3, 1.7e-3, 1.7e-3, 3.1e-3};
    for (size_t k = 0; k < dts.size(); ++k) {
        CAPTURE(k, dts[k]);
        step_a(reuse, uniform(m.nelem, dts[k]));
        step_b(rebuild, uniform(m.nelem, dts[k]));
        CHECK_THAT(reuse.preconditioner_scale(), WithinRel(dts[0] / dts[k], 1e-13));
        const std::vector<double> rhs = test_vector(m.npoin, 0.4 * static_cast<double>(k));
        const la::SolveResult ra = reuse.solve(rhs, pa), rb = rebuild.solve(rhs, pb);
        REQUIRE(ra.converged);
        REQUIRE(rb.converged);
        CHECK(std::abs(ra.iterations - rb.iterations) <= 1);
        CHECK(max_abs_diff(pa, pb) < 1e-9 * max_abs(pb));
    }
    CHECK(reuse.statistics().structure_builds == 1);
    CHECK(reuse.statistics().numeric_setups == 1);
    CHECK(reuse.statistics().rescaled_reuses == 4);
    CHECK(rebuild.statistics().numeric_setups == 5);
    CHECK(rebuild.statistics().rescaled_reuses == 0);
}

TEST_CASE("Constant time step: reuse is bit-for-bit identical to rebuilding", "[reuse][pressure_solver]") {
    const std::string method = GENERATE(std::string("cg"), std::string("bicgstab"), std::string("gmres"));
    CAPTURE(method);
    const Mesh m = grid_mesh();
    const PressureBC bc = outlet_bc(m);
    PressureSolver reuse(config("ilu0", "rescale", method, 1e-3), bc), rebuild(config("ilu0", "rebuild", method, 1e-3), bc);
    Stepper step_a{m, {}, {}, {}}, step_b{m, {}, {}, {}};
    std::vector<double> pa(m.npoin, 1e-4), pb(m.npoin, 1e-4);
    for (int k = 0; k < 4; ++k) {
        step_a(reuse, uniform(m.nelem, 2e-3));
        step_b(rebuild, uniform(m.nelem, 2e-3));
        CHECK(reuse.preconditioner_scale() == 1.0);
        const std::vector<double> rhs = test_vector(m.npoin, 0.3 * k);
        const la::SolveResult ra = reuse.solve(rhs, pa), rb = rebuild.solve(rhs, pb);
        CHECK(ra.iterations == rb.iterations);
        CHECK(ra.final_residual == rb.final_residual);
        REQUIRE(pa == pb);
    }
    CHECK(reuse.statistics().numeric_setups == 1);
    CHECK(rebuild.statistics().numeric_setups == 4);
}

TEST_CASE("Local time steps and geometry changes trigger a new numeric setup", "[reuse][pressure_solver]") {
    Mesh m = grid_mesh();
    const PressureBC bc = outlet_bc(m);
    PressureSolver s(config("ic0", "rescale"), bc);
    Stepper step{m, {}, {}, {}};

    step(s, uniform(m.nelem, 1e-3)); // setup 1
    step(s, uniform(m.nelem, 2e-3)); // reuse
    step(s, local(m.nelem, 0));      // setup 2 (not a uniform scaling)
    step(s, local(m.nelem, 1));      // setup 3
    step(s, uniform(m.nelem, 1e-3)); // setup 4 (reference was local)
    step(s, uniform(m.nelem, 5e-4)); // reuse
    CHECK(s.statistics().numeric_setups == 4);
    CHECK(s.statistics().rescaled_reuses == 2);

    deform(m); // same mesh object and sizes, new element stiffness
    step(s, uniform(m.nelem, 5e-4)); // setup 5
    step(s, uniform(m.nelem, 1e-3)); // reuse against the deformed reference
    CHECK(s.statistics().numeric_setups == 5);
    CHECK(s.statistics().rescaled_reuses == 3);
    CHECK(s.statistics().structure_builds == 1); // same connectivity, same pattern

    // Every solve is still correct (against a fresh rebuild solver).
    PressureSolver fresh(config("ic0", "rebuild"), bc);
    Stepper step_f{m, {}, {}, {}};
    step_f(fresh, uniform(m.nelem, 1e-3));
    const std::vector<double> rhs = test_vector(m.npoin, 1.1);
    std::vector<double> p1(m.npoin, 0.0), p2(m.npoin, 0.0);
    REQUIRE(s.solve(rhs, p1).converged);
    REQUIRE(fresh.solve(rhs, p2).converged);
    CHECK(max_abs_diff(p1, p2) < 1e-9 * max_abs(p2));
}

TEST_CASE("A new mesh rebuilds the structure; orderings are computed once per structure",
          "[reuse][pressure_solver]") {
    const Mesh a = grid_mesh(12, 6), b = grid_mesh(9, 5);
    PressureBC bc; // node 0 exists in both meshes
    bc.nodes = {0};
    bc.values = {0.0};
    PressureSolver s(config("ilu0", "rebuild"), bc);
    Stepper step_a{a, {}, {}, {}}, step_b{b, {}, {}, {}};
    for (int k = 0; k < 5; ++k) step_a(s, local(a.nelem, k));
    const auto& reordered = dynamic_cast<const la::ReorderedPreconditioner&>(s.preconditioner());
    CHECK(reordered.orderings_computed() == 1);
    CHECK(s.statistics().structure_builds == 1);
    CHECK(s.statistics().numeric_setups == 5);

    step_b(s, uniform(b.nelem, 1e-3));
    CHECK(s.statistics().structure_builds == 2);
    CHECK(reordered.orderings_computed() == 2);
    std::vector<double> p(b.npoin, 0.0);
    CHECK(s.solve(test_vector(b.npoin), p).converged);
}

TEST_CASE("Matrix-free and legacy configurations do not reuse setups", "[reuse][pressure_solver]") {
    const Mesh m = grid_mesh();
    const PressureBC bc = outlet_bc(m);
    SolverConfig legacy;
    SolverConfig jacobi_csr = config("jacobi", "rescale");
    jacobi_csr.operator_format = "csr";
    PressureSolver s(legacy, bc), j(jacobi_csr, bc);
    CHECK_FALSE(s.setup_reuse_enabled()); // Jacobi: O(n) setup, legacy path untouched
    CHECK_FALSE(j.setup_reuse_enabled());
    Stepper step{m, {}, {}, {}};
    for (int k = 0; k < 3; ++k) step(s, uniform(m.nelem, 1e-3 * (k + 1)));
    CHECK(s.statistics().numeric_setups == 3);
    CHECK(s.statistics().rescaled_reuses == 0);
    CHECK(s.statistics().structure_builds == 0); // nothing assembled
}

TEST_CASE("preconditioner_reuse configuration", "[reuse][io][solver_options]") {
    TempDir dir("reuse-option");
    const fs::path file = dir / "case.solver";
    const std::string positional = "-\nsolver\n1\n-\ntol\nrel\n0.1\n-\nabs\n1.0d-6\n";
    SolverConfig c;
    CHECK(c.preconditioner_reuse == "rescale");
    {
        write_text(file, positional + "preconditioner = ic0\npreconditioner_reuse = REBUILD\n");
        CaptureStdout out;
        REQUIRE(IO::read_solver(file.string(), c));
        CHECK(c.preconditioner_reuse == "rebuild");
    }
    {
        write_text(file, positional + "preconditioner_reuse = sometimes\n");
        CaptureStderr err;
        SolverConfig d;
        CHECK_FALSE(IO::read_solver(file.string(), d));
        CHECK(err.str().find("rescale | rebuild") != std::string::npos);
    }
    PressureBC bc;
    bc.nodes = {0};
    bc.values = {0.0};
    SolverConfig bad = config("ic0", "sometimes");
    CHECK_THROWS_AS(PressureSolver(bad, bc), std::invalid_argument);
}
