// Refactored preconditioned conjugate gradient (linalg::ConjugateGradient)
// and the legacy solve_pcg() adapter built on it.
//
//   * Bit-for-bit equivalence with the frozen original implementation
//     (tests/reference/LegacyPcg.hpp) in legacy mode, on analytic grids and
//     the Hinterkante mesh, through both the adapter and the linalg API.
//   * Fused (pointwise preconditioner) and generic code paths agree exactly.
//   * Work per iteration: one operator and one preconditioner application.
//   * Default mode: correctness against direct solves, tolerances relative to
//     r_0 / ||b||, early exit, finite termination, breakdown and non-finite
//     detection, iteration cap, input validation, workspace reuse.
//
// The existing test_pcg.cpp exercises the same solver through solve_pcg()
// and is kept unchanged as the behavioural specification of the legacy PCG.

#include "BoundaryConditions.hpp"
#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "LinalgTestUtils.hpp"
#include "reference/LegacyPcg.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <set>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

GridSpec cg_grid() {
    GridSpec g;
    g.x0 = -0.5;
    g.y0 = 0.25;
    g.lx = 1.5;
    g.ly = 1.0;
    g.nx = 9;
    g.ny = 6;
    return g;
}

PoissonSystem grid_system() {
    const GridSpec g = cg_grid();
    return make_poisson_system(make_grid_mesh(g), varying_dt(g.nelem()));
}

// The benchmark mesh with the solver's pressure BCs (outlet nodes, flag 504).
struct BenchmarkSystem {
    PoissonSystem s;
    PressureBC bc;
};

const BenchmarkSystem& benchmark_system() {
    static const BenchmarkSystem b = [] {
        Mesh mesh;
        CaptureStdout quiet;
        const std::string dir = HYDRAMA2D_CASE_DIR;
        REQUIRE(IO::read_mesh(dir + "/constant/hinterkante.mesh", mesh));
        REQUIRE(IO::read_boundary_flags(dir + "/constant/hinterkante.bofl", mesh));
        BenchmarkSystem out{make_poisson_system(std::move(mesh), std::vector<double>(17377, 4.9e-3)), {}};
        out.bc = setup_pressure_bcs(out.s.mesh, 0);
        return out;
    }();
    return b;
}

double linear_field(double x, double y) { return 2.0 - 1.5 * x + 0.75 * y; }

struct Scenario {
    std::string name;
    std::vector<int> bc;
    std::vector<double> bc_values;
    double rel, abs;
    bool nonzero_guess;
};

std::vector<Scenario> grid_scenarios(const Mesh& m) {
    const std::vector<int> boundary = boundary_node_list(m);
    std::vector<double> boundary_values;
    for (int ip : boundary) boundary_values.push_back(linear_field(m.x[ip], m.y[ip]));
    return {
        {"reference node, bundled tolerances", {0}, {0.0}, 0.1, 1e-6, false},
        {"reference node, converged", {17}, {0.4}, 0.0, 1e-12, false},
        {"reference node, relative only", {5}, {0.0}, 1e-4, 0.0, true},
        {"relative tolerance above one", {0}, {0.0}, 1.5, 0.0, false},
        {"whole boundary, linear data", boundary, boundary_values, 0.0, 1e-11, true},
        {"whole boundary, loose", boundary, boundary_values, 0.3, 1e-3, true},
    };
}

std::vector<double> initial_guess(int n, bool nonzero) {
    std::vector<double> x(n, 0.0);
    if (nonzero)
        for (int i = 0; i < n; ++i) x[i] = 5.0 * std::cos(0.3 * i);
    return x;
}

struct LinalgSolve {
    la::SolveResult result;
    std::vector<double> x;
};

// Solve directly through the linalg API (what PressureSolver does).
LinalgSolve linalg_solve(const PoissonSystem& s, const std::vector<double>& rhs, const la::DirichletConstraints& bc,
                         const la::ConvergenceCriteria& crit, std::vector<double> x,
                         bool expose_scaling = true) {
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    la::JacobiPreconditioner jacobi;
    jacobi.setup(la::ConstrainedOperator(op, bc));
    ForwardingPreconditioner m(jacobi, expose_scaling);
    la::ConjugateGradient cg(crit);
    LinalgSolve out;
    out.x = std::move(x);
    out.result = cg.solve(op, m, rhs, out.x, bc);
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Bit-for-bit equivalence with the original implementation (legacy mode)
// ---------------------------------------------------------------------------

TEST_CASE("Refactored PCG reproduces the original solver bit for bit on analytic grids",
          "[linalg][cg][legacy]") {
    const PoissonSystem s = grid_system();
    const int n = s.mesh.npoin;
    const std::vector<double> rhs = test_vector(n);
    for (const Scenario& sc : grid_scenarios(s.mesh)) {
        CAPTURE(sc.name);
        std::vector<double> x_ref = initial_guess(n, sc.nonzero_guess);
        std::vector<double> x_new = x_ref;
        const PCGResult ref = legacy::solve_pcg(s.mesh, s.gstif, s.pdiag, rhs, sc.bc, sc.bc_values, sc.rel,
                                                sc.abs, x_ref);

        // 1. through the legacy interface (adapter)
        const PCGResult res = solve_pcg(s.mesh, s.gstif, s.pdiag, rhs, sc.bc, sc.bc_values, sc.rel, sc.abs, x_new);
        CHECK(res.iterations == ref.iterations);
        CHECK(res.initial_residual == ref.initial_residual);
        CHECK(res.final_residual == ref.final_residual);
        CHECK(x_new == x_ref);

        // 2. directly through the linalg API
        const LinalgSolve direct = linalg_solve(s, rhs, la::DirichletConstraints(sc.bc, sc.bc_values),
                                                la::ConvergenceCriteria::legacy_pcg(sc.rel, sc.abs),
                                                initial_guess(n, sc.nonzero_guess));
        CHECK(direct.result.iterations == ref.iterations);
        CHECK(direct.result.reference_residual == ref.initial_residual);
        CHECK(direct.result.final_residual == ref.final_residual);
        CHECK(direct.result.converged);
        CHECK(direct.x == x_ref);
    }
}

TEST_CASE("Refactored PCG reproduces the original solver bit for bit on the benchmark mesh",
          "[linalg][cg][legacy][hinterkante]") {
    const BenchmarkSystem& b = benchmark_system();
    const PoissonSystem& s = b.s;
    const int n = s.mesh.npoin;
    // Scale the rough data to the size of typical Step 2 right-hand sides.
    std::vector<double> rhs = test_vector(n, 0.7);
    la::scale(rhs, 1e-4);

    const auto tolerances = GENERATE(std::pair<double, double>{0.1, 1e-6}, std::pair<double, double>{0.0, 1e-10});
    CAPTURE(tolerances.first, tolerances.second);

    std::vector<double> x_ref(n, 1e-4), x_new(n, 1e-4);
    const PCGResult ref = legacy::solve_pcg(s.mesh, s.gstif, s.pdiag, rhs, b.bc.nodes, b.bc.values,
                                            tolerances.first, tolerances.second, x_ref);
    const PCGResult res = solve_pcg(s.mesh, s.gstif, s.pdiag, rhs, b.bc.nodes, b.bc.values, tolerances.first,
                                    tolerances.second, x_new);
    CAPTURE(ref.iterations);
    CHECK(res.iterations == ref.iterations);
    CHECK(res.initial_residual == ref.initial_residual);
    CHECK(res.final_residual == ref.final_residual);
    CHECK(x_new == x_ref);
}

TEST_CASE("Fused and generic preconditioner paths give identical results", "[linalg][cg]") {
    // Hiding Jacobi's pointwise scaling forces the generic path
    // (separate preconditioner application); the arithmetic must not change.
    const PoissonSystem s = grid_system();
    const int n = s.mesh.npoin;
    const std::vector<double> rhs = test_vector(n);
    const la::DirichletConstraints bc({0}, {0.0});
    const la::ConvergenceCriteria crit = GENERATE(la::ConvergenceCriteria::legacy_pcg(0.1, 1e-6),
                                                  la::ConvergenceCriteria::legacy_pcg(0.0, 1e-12));
    const LinalgSolve fused = linalg_solve(s, rhs, bc, crit, std::vector<double>(n, 0.0), true);
    const LinalgSolve generic = linalg_solve(s, rhs, bc, crit, std::vector<double>(n, 0.0), false);
    CHECK(fused.result.iterations == generic.result.iterations);
    CHECK(fused.result.final_residual == generic.result.final_residual);
    CHECK(fused.x == generic.x);
}

// ---------------------------------------------------------------------------
// Work per iteration
// ---------------------------------------------------------------------------

TEST_CASE("CG applies the operator and the preconditioner once per iteration", "[linalg][cg]") {
    const PoissonSystem s = grid_system();
    const int n = s.mesh.npoin;
    const la::EbeOperator ebe(s.mesh, s.gstif, s.pdiag);
    const CountingOperator op(ebe);
    const la::DirichletConstraints bc({0}, {0.0});
    la::JacobiPreconditioner jacobi;
    jacobi.setup(la::ConstrainedOperator(ebe, bc));
    const bool expose = GENERATE(true, false);
    CAPTURE(expose);
    ForwardingPreconditioner m(jacobi, expose);

    la::ConvergenceCriteria c;
    c.rel_tol = 1e-8;
    la::ConjugateGradient cg(c);
    std::vector<double> x(n, 0.0);
    const la::SolveResult r = cg.solve(op, m, test_vector(n), x, bc);
    REQUIRE(r.converged);
    // One product for r_0, then one per iteration.
    CHECK(op.applications == r.iterations + 1);
    // Generic path: z_0 plus one per non-final iteration; fused path: none.
    CHECK(m.applications == (expose ? 0 : r.iterations));
}

// ---------------------------------------------------------------------------
// Default mode: correctness and convergence control
// ---------------------------------------------------------------------------

TEST_CASE("CG agrees with a dense direct solve", "[linalg][cg]") {
    const PoissonSystem s = grid_system();
    const int n = s.mesh.npoin;
    const int ref = GENERATE(0, 23);
    const double ref_value = -0.3;
    CAPTURE(ref);
    const std::vector<double> rhs = test_vector(n, 2.0);

    const DenseMatrix a = dense_operator(s.mesh, s.gstif, s.pdiag);
    std::vector<double> b;
    for (int i = 0; i < n; ++i)
        if (i != ref) b.push_back(rhs[i] - a[i][ref] * ref_value);
    const std::vector<double> xf = solve_dense(reduced(a, {ref}), b);

    la::ConvergenceCriteria c;
    c.rel_tol = 1e-13;
    const LinalgSolve sol = linalg_solve(s, rhs, la::DirichletConstraints({ref}, {ref_value}), c,
                                         std::vector<double>(n, 0.0));
    REQUIRE(sol.result.converged);
    CHECK(sol.result.reason == la::StopReason::RelativeTolerance);
    CHECK(sol.x[ref] == ref_value);
    for (int i = 0, k = 0; i < n; ++i) {
        if (i == ref) continue;
        CAPTURE(i);
        CHECK_THAT(sol.x[i], WithinAbs(xf[k++], 1e-10));
    }
}

TEST_CASE("CG works with user-defined operators and preconditioners", "[linalg][cg]") {
    // Dense 1D Laplacian through the LinearOperator interface, with three
    // preconditioners: none, Jacobi, and a (non-pointwise) dense one.
    const int n = 30;
    const DenseOperator a(laplacian_1d(n, 3.0));
    const std::vector<double> b = test_vector(n);
    const std::vector<double> exact = solve_dense(a.matrix(), b);

    la::IdentityPreconditioner identity;
    la::JacobiPreconditioner jacobi;
    DensePreconditioner exact_inverse(inverse(a.matrix()));
    const int which = GENERATE(0, 1, 2);
    la::Preconditioner* const m = which == 0 ? static_cast<la::Preconditioner*>(&identity)
                                  : which == 1 ? static_cast<la::Preconditioner*>(&jacobi)
                                               : static_cast<la::Preconditioner*>(&exact_inverse);
    CAPTURE(m->name());
    m->setup(a);

    la::ConvergenceCriteria c;
    c.rel_tol = 1e-12;
    la::ConjugateGradient cg(c);
    std::vector<double> x(n, 0.0);
    const la::SolveResult r = cg.solve(a, *m, b, x);
    REQUIRE(r.converged);
    CHECK(max_abs_diff(x, exact) < 1e-9);
    CHECK_THAT(residual_norm(a.matrix(), b, x), WithinAbs(r.final_residual, 1e-12 * la::norm2(b)));
    if (m == &exact_inverse) CHECK(r.iterations == 1); // M = A^{-1}: one step
    if (m != &exact_inverse) CHECK(r.iterations <= n);  // finite termination
}

TEST_CASE("CG converges in as many iterations as distinct eigenvalues", "[linalg][cg]") {
    const DenseOperator a(diagonal_matrix({1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 7.0, 7.0}));
    la::IdentityPreconditioner m;
    m.setup(a);
    la::ConvergenceCriteria c;
    c.rel_tol = 1e-12;
    la::ConjugateGradient cg(c);
    std::vector<double> x(8, 0.0);
    const la::SolveResult r = cg.solve(a, m, test_vector(8), x);
    CHECK(r.converged);
    CHECK(r.iterations <= 4);
}

TEST_CASE("CG relative tolerance is measured against the chosen reference", "[linalg][cg]") {
    const PoissonSystem s = grid_system();
    const int n = s.mesh.npoin;
    const std::vector<double> rhs = test_vector(n);
    const la::DirichletConstraints bc({0}, {0.0});

    SECTION("initial residual (default)") {
        la::ConvergenceCriteria c;
        c.rel_tol = 1e-6;
        std::vector<double> guess = initial_guess(n, true);
        const LinalgSolve sol = linalg_solve(s, rhs, bc, c, guess);
        REQUIRE(sol.result.reason == la::StopReason::RelativeTolerance);
        CHECK(sol.result.reference_residual == sol.result.initial_residual);
        CHECK(sol.result.final_residual < 1e-6 * sol.result.initial_residual);

        // r_0 is the true initial residual ||P (b - A x_0)||.
        std::vector<double> x0 = guess;
        bc.impose(x0);
        std::vector<double> ax(n);
        la::EbeOperator(s.mesh, s.gstif, s.pdiag).apply(x0, ax);
        for (int i = 0; i < n; ++i) ax[i] = rhs[i] - ax[i];
        bc.zero(ax);
        CHECK_THAT(sol.result.initial_residual, WithinRel(la::norm2(ax), 1e-14));
    }
    SECTION("right-hand side") {
        la::ConvergenceCriteria c;
        c.rel_tol = 1e-6;
        c.reference = la::ResidualReference::RightHandSide;
        const LinalgSolve sol = linalg_solve(s, rhs, bc, c, initial_guess(n, true));
        std::vector<double> b_free = rhs;
        bc.zero(b_free);
        CHECK(sol.result.reference_residual == la::norm2(b_free));
        CHECK(sol.result.final_residual < 1e-6 * la::norm2(b_free));
    }
}

TEST_CASE("CG returns immediately when the initial guess already solves the system", "[linalg][cg]") {
    // Uniform dt: linear fields are discretely harmonic (A u = 0 at interior
    // nodes) only for a uniform coefficient.
    const PoissonSystem s = make_poisson_system(make_grid_mesh(cg_grid()));
    const int n = s.mesh.npoin;
    const std::vector<int> bc = boundary_node_list(s.mesh);
    std::vector<double> values, exact(n);
    for (int ip : bc) values.push_back(linear_field(s.mesh.x[ip], s.mesh.y[ip]));
    for (int ip = 0; ip < n; ++ip) exact[ip] = linear_field(s.mesh.x[ip], s.mesh.y[ip]);

    la::ConvergenceCriteria c;
    c.abs_tol = 1e-10;
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    const CountingOperator counting(op);
    la::JacobiPreconditioner m;
    m.setup(op);
    la::ConjugateGradient cg(c);
    std::vector<double> x = exact;
    const la::SolveResult r = cg.solve(counting, m, std::vector<double>(n, 0.0), x, la::DirichletConstraints(bc, values));
    CHECK(r.iterations == 0);
    CHECK(r.reason == la::StopReason::AbsoluteTolerance);
    CHECK(counting.applications == 1); // only r_0
    CHECK(x == exact);
}

TEST_CASE("CG with zero data stops with the exact solution", "[linalg][cg]") {
    const PoissonSystem s = grid_system();
    const int n = s.mesh.npoin;
    la::ConjugateGradient cg; // tolerances off
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    la::JacobiPreconditioner m;
    m.setup(op);
    std::vector<double> x(n, 0.0);
    const la::SolveResult r = cg.solve(op, m, std::vector<double>(n, 0.0), x, la::DirichletConstraints({0}, {0.0}));
    CHECK(r.iterations == 0);
    CHECK(r.reason == la::StopReason::ExactSolution);
    CHECK(r.converged);
    for (double v : x) CHECK(v == 0.0);
}

TEST_CASE("CG iteration cap is reported as not converged", "[linalg][cg]") {
    const PoissonSystem s = grid_system();
    const int n = s.mesh.npoin;
    la::ConvergenceCriteria c;
    c.rel_tol = 1e-14;
    c.max_iterations = 3;
    const LinalgSolve sol = linalg_solve(s, test_vector(n), la::DirichletConstraints({0}, {0.0}), c,
                                         std::vector<double>(n, 0.0));
    CHECK(sol.result.iterations == 3);
    CHECK(sol.result.reason == la::StopReason::MaxIterations);
    CHECK_FALSE(sol.result.converged);
}

TEST_CASE("CG detects breakdown on indefinite problems", "[linalg][cg]") {
    SECTION("indefinite operator: p.Ap = 0") {
        const DenseOperator a(diagonal_matrix({1.0, -1.0}));
        la::IdentityPreconditioner m;
        m.setup(a);
        std::vector<double> x(2, 0.0);

        la::ConjugateGradient cg; // BreakdownPolicy::Stop
        const la::SolveResult r = cg.solve(a, m, std::vector<double>{1.0, 1.0}, x);
        CHECK(r.reason == la::StopReason::Breakdown);
        CHECK_FALSE(r.converged);
        CHECK(r.iterations == 0);

        SECTION("the legacy policy takes zero steps until the cap instead") {
            la::ConvergenceCriteria c = la::ConvergenceCriteria::legacy_pcg(0.1, 1e-6);
            c.max_iterations = 50;
            la::ConjugateGradient legacy_cg(c);
            std::vector<double> y(2, 0.0);
            const la::SolveResult lr = legacy_cg.solve(a, m, std::vector<double>{1.0, 1.0}, y);
            CHECK(lr.reason == la::StopReason::MaxIterations);
            CHECK(lr.iterations == 50);
            CHECK(y == std::vector<double>{0.0, 0.0});
        }
    }
    SECTION("indefinite preconditioner: r.z < 0") {
        const DenseOperator a(laplacian_1d(4));
        DensePreconditioner m(diagonal_matrix({-1.0, -1.0, -1.0, -1.0}));
        m.setup(a);
        la::ConjugateGradient cg;
        std::vector<double> x(4, 0.0);
        const la::SolveResult r = cg.solve(a, m, test_vector(4), x);
        CHECK(r.reason == la::StopReason::Breakdown);
        CHECK(r.iterations == 0);
    }
}

TEST_CASE("CG stops on non-finite data", "[linalg][cg]") {
    const PoissonSystem s = grid_system();
    const int n = s.mesh.npoin;
    std::vector<double> rhs = test_vector(n);
    rhs[n / 2] = std::numeric_limits<double>::quiet_NaN();
    const la::ConvergenceCriteria crit = GENERATE(la::ConvergenceCriteria{}, la::ConvergenceCriteria::legacy_pcg(0.1, 1e-6));
    const LinalgSolve sol = linalg_solve(s, rhs, la::DirichletConstraints({0}, {0.0}), crit, std::vector<double>(n, 0.0));
    CHECK(sol.result.reason == la::StopReason::NonFinite);
    CHECK_FALSE(sol.result.converged);
    CHECK(sol.result.iterations == 0);
}

// ---------------------------------------------------------------------------
// Interface contract
// ---------------------------------------------------------------------------

TEST_CASE("CG validates its input", "[linalg][cg]") {
    const DenseOperator a(laplacian_1d(5));
    la::JacobiPreconditioner m;
    m.setup(a);
    la::ConjugateGradient cg;
    CHECK(cg.name() == "CG");
    std::vector<double> b(5, 1.0), x(5, 0.0), short_vec(4, 0.0);

    CHECK_THROWS_AS(cg.solve(a, m, short_vec, x), std::invalid_argument);
    CHECK_THROWS_AS(cg.solve(a, m, b, short_vec), std::invalid_argument);
    CHECK_THROWS_AS(cg.solve(a, m, b, b), std::invalid_argument); // aliasing
    CHECK_THROWS_AS(cg.solve(a, m, b, x, la::DirichletConstraints({5}, {0.0})), std::out_of_range);

    la::JacobiPreconditioner not_set_up;
    CHECK_THROWS_AS(cg.solve(a, not_set_up, b, x), std::invalid_argument);
    la::JacobiPreconditioner wrong_size;
    wrong_size.setup(DenseOperator(laplacian_1d(6)));
    CHECK_THROWS_AS(cg.solve(a, wrong_size, b, x), std::invalid_argument);
}

TEST_CASE("CG solver objects are reusable across solves and sizes", "[linalg][cg]") {
    la::ConvergenceCriteria c;
    c.rel_tol = 1e-10;
    la::ConjugateGradient cg(c);
    CHECK(cg.criteria().rel_tol == 1e-10);

    const PoissonSystem s = grid_system();
    const int n = s.mesh.npoin;
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    la::JacobiPreconditioner m;
    m.setup(op);
    const la::DirichletConstraints bc({0}, {0.0});
    const std::vector<double> rhs = test_vector(n);

    std::vector<double> x1(n, 0.0), x2(n, 0.0);
    const la::SolveResult r1 = cg.solve(op, m, rhs, x1, bc);

    // A different, smaller system in between (work vectors are resized).
    const DenseOperator small(laplacian_1d(7));
    la::JacobiPreconditioner ms;
    ms.setup(small);
    std::vector<double> y(7, 0.0);
    CHECK(cg.solve(small, ms, test_vector(7), y).converged);

    const la::SolveResult r2 = cg.solve(op, m, rhs, x2, bc);
    CHECK(r1.iterations == r2.iterations);
    CHECK(r1.final_residual == r2.final_residual);
    CHECK(x1 == x2);
}

TEST_CASE("CG on the benchmark mesh converges in default mode", "[linalg][cg][hinterkante]") {
    const BenchmarkSystem& b = benchmark_system();
    const PoissonSystem& s = b.s;
    const int n = s.mesh.npoin;
    std::vector<double> rhs = test_vector(n, 0.7);
    const la::DirichletConstraints bc(b.bc.nodes, b.bc.values);

    la::ConvergenceCriteria c;
    c.rel_tol = 1e-8;
    la::ResidualHistory history;
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    la::JacobiPreconditioner m;
    m.setup(la::ConstrainedOperator(op, bc));
    la::ConjugateGradient cg(c);
    cg.set_monitor(&history);
    std::vector<double> x(n, 0.0);
    const la::SolveResult r = cg.solve(op, m, rhs, x, bc);
    CAPTURE(r.iterations);
    REQUIRE(r.converged);
    CHECK(r.final_residual < 1e-8 * r.initial_residual);
    for (int d : b.bc.nodes) CHECK(x[d] == 0.0);

    // Reported residual equals the true residual of the returned solution.
    std::vector<double> ax(n);
    op.apply(x, ax);
    for (int i = 0; i < n; ++i) ax[i] = rhs[i] - ax[i];
    bc.zero(ax);
    CHECK_THAT(la::norm2(ax), WithinAbs(r.final_residual, 1e-10 * r.initial_residual));
    CHECK(history.residuals().size() == static_cast<size_t>(r.iterations) + 1);
}
