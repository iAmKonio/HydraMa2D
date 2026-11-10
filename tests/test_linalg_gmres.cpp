// Restarted GMRES(m) (linalg::Gmres): right-preconditioned Arnoldi with
// modified or classical (twice) Gram-Schmidt and progressive Givens QR of
// the Hessenberg matrix.
//   * agreement with dense direct solutions (schemes x preconditioners x
//     restart lengths x orthogonalisations); reported = true residual
//   * exact properties: minimal residual over the Krylov space (checked
//     against an independent dense least-squares solve), monotone residual
//     history, finite termination (restart >= n), termination within the
//     number of distinct eigenvalues, exact preconditioner, happy breakdown
//   * full GMRES never needs more iterations than restarted GMRES; GMRES(1)
//     converges for positive-real matrices (Elman)
//   * singular systems (breakdown), work per iteration, cap, NaN,
//     constraints, options

#include "LinalgTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

la::ConvergenceCriteria relative(double tol) {
    la::ConvergenceCriteria c;
    c.rel_tol = tol;
    return c;
}

la::GmresOptions options(int restart, la::Orthogonalization o = la::Orthogonalization::ModifiedGramSchmidt) {
    la::GmresOptions g;
    g.restart = restart;
    g.orthogonalization = o;
    return g;
}

struct Outcome {
    la::SolveResult result;
    std::vector<double> x;
    int cycles = 0;
};

Outcome solve(const la::LinearOperator& a, const la::Preconditioner& m, const std::vector<double>& b,
              const la::ConvergenceCriteria& c, const la::GmresOptions& o, la::IterationMonitor* monitor = nullptr) {
    la::Gmres solver(c, o);
    solver.set_monitor(monitor);
    Outcome out;
    out.x.assign(b.size(), 0.0);
    out.result = solver.solve(a, m, b, out.x);
    out.cycles = solver.cycles();
    return out;
}

double max_abs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

// min_{y in K_k(A, b)} ||b - A y|| by dense least squares: orthonormal basis
// Q of K_k (Gram-Schmidt, twice), then the normal equations of A Q.
double minimal_residual(const DenseMatrix& a, const std::vector<double>& b, int k) {
    const int n = static_cast<int>(b.size());
    std::vector<std::vector<double>> q;
    std::vector<double> w = b;
    for (int j = 0; j < k; ++j) {
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& qi : q) {
                const double c = la::dot(w, qi);
                for (int l = 0; l < n; ++l) w[l] -= c * qi[l];
            }
        const double nw = la::norm2(w);
        for (double& v : w) v /= nw;
        q.push_back(w);
        w = multiply(a, w);
    }
    std::vector<std::vector<double>> aq;
    for (const auto& qi : q) aq.push_back(multiply(a, qi));
    DenseMatrix normal = zeros(k);
    std::vector<double> rhs(k);
    for (int i = 0; i < k; ++i) {
        rhs[i] = la::dot(aq[i], b);
        for (int j = 0; j < k; ++j) normal[i][j] = la::dot(aq[i], aq[j]);
    }
    const std::vector<double> c = solve_dense(normal, rhs);
    std::vector<double> r = b;
    for (int i = 0; i < k; ++i)
        for (int l = 0; l < n; ++l) r[l] -= c[i] * aq[i][l];
    return la::norm2(r);
}

} // namespace

TEST_CASE("GMRES agrees with a direct solve on convection-diffusion operators", "[linalg][gmres]") {
    const Convection scheme = GENERATE(Convection::Upwind, Convection::Central);
    const std::string preconditioner = GENERATE(std::string("none"), std::string("jacobi"), std::string("ilu0"),
                                                std::string("rcm+ilu0"));
    const int restart = GENERATE(5, 30);
    const la::Orthogonalization orth =
        GENERATE(la::Orthogonalization::ModifiedGramSchmidt, la::Orthogonalization::ClassicalGramSchmidtTwice);
    CAPTURE(scheme == Convection::Upwind ? "upwind" : "central", preconditioner, restart, la::to_string(orth));

    const la::CsrMatrix a = convection_diffusion_csr(10, 8, 6.0, 3.0, scheme);
    const DenseMatrix d = to_dense(a);
    const std::vector<double> b = test_vector(a.rows(), 0.3);
    const std::vector<double> exact = solve_dense(d, b);

    la::PreconditionerOptions o;
    if (preconditioner == "rcm+ilu0") o.ordering = "rcm";
    auto m = la::make_preconditioner(preconditioner == "rcm+ilu0" ? "ilu0" : preconditioner, o);
    m->setup(a);

    const Outcome out = solve(a, *m, b, relative(1e-12), options(restart, orth));
    CAPTURE(out.result.iterations, out.cycles);
    REQUIRE(out.result.converged);
    CHECK(out.result.reason == la::StopReason::RelativeTolerance);
    CHECK(max_abs_diff(out.x, exact) < 1e-9 * max_abs(exact));
    CHECK_THAT(out.result.final_residual, WithinRel(residual_norm(d, b, out.x), 1e-10));
}

// ---------------------------------------------------------------------------
// Exact properties
// ---------------------------------------------------------------------------

TEST_CASE("GMRES residuals are the minimal residuals over the Krylov spaces", "[linalg][gmres]") {
    // Without preconditioning and before the first restart, the k-th residual
    // is min ||b - A y|| over y in K_k(A, b).
    const la::CsrMatrix a = convection_diffusion_csr(6, 5, 5.0, 2.0, Convection::Central);
    const DenseMatrix d = to_dense(a);
    const std::vector<double> b = test_vector(a.rows(), 0.8);
    la::IdentityPreconditioner m;
    m.setup(a);
    la::ResidualHistory history;
    la::ConvergenceCriteria c;
    c.max_iterations = 5;
    const Outcome out = solve(a, m, b, c, options(30), &history);
    REQUIRE(history.residuals().size() == 6);
    for (int k = 1; k <= 5; ++k) {
        CAPTURE(k);
        CHECK_THAT(history.residuals()[k], WithinRel(minimal_residual(d, b, k), 1e-8));
    }
}

TEST_CASE("GMRES residual history is monotone, also across restarts", "[linalg][gmres]") {
    const int restart = GENERATE(3, 10, 50);
    CAPTURE(restart);
    const la::CsrMatrix a = convection_diffusion_csr(12, 10, 8.0, 3.0, Convection::Central);
    la::JacobiPreconditioner m;
    m.setup(a);
    la::ResidualHistory history;
    const Outcome out = solve(a, m, test_vector(a.rows()), relative(1e-10), options(restart), &history);
    REQUIRE(out.result.converged);
    CHECK(history.max_growth_factor() <= 1.0 + 1e-10);
    CHECK(history.residuals().size() == static_cast<size_t>(out.result.iterations) + 1);
}

TEST_CASE("Full GMRES terminates within n iterations", "[linalg][gmres]") {
    const la::CsrMatrix a = convection_diffusion_csr(6, 5, 7.0, 3.0, Convection::Upwind);
    const int n = a.rows();
    la::IdentityPreconditioner m;
    m.setup(a);
    const int restart = GENERATE(30, 100); // >= n: no restart (100 is capped to n internally)
    CAPTURE(restart);
    const Outcome out = solve(a, m, test_vector(n), relative(1e-13), options(restart));
    CAPTURE(out.result.iterations, out.result.final_residual);
    CHECK(out.result.converged);
    CHECK(out.result.iterations <= n);
    CHECK(out.cycles == 1);
}

TEST_CASE("GMRES terminates within the number of distinct eigenvalues", "[linalg][gmres]") {
    const SpectralMatrix sm = matrix_with_spectrum({2, 5, 2, 9, 5, 2, 9, 5});
    const la::CsrMatrix a = from_dense(sm.a);
    la::IdentityPreconditioner m;
    m.setup(a);
    const Outcome out = solve(a, m, test_vector(8), relative(1e-10), options(30));
    CAPTURE(out.result.iterations, out.result.final_residual);
    CHECK(out.result.converged);
    CHECK(out.result.iterations <= 3);
}

TEST_CASE("GMRES happy breakdown: eigenvector right-hand side", "[linalg][gmres]") {
    // K_1(A, b) is invariant: h_21 = 0 and the first iterate is exact.
    const SpectralMatrix sm = matrix_with_spectrum({3, 7, 1, 4, 6});
    const la::CsrMatrix a = from_dense(sm.a);
    std::vector<double> b(5);
    for (int i = 0; i < 5; ++i) b[i] = sm.s[i][3]; // eigenvector of 4
    la::IdentityPreconditioner m;
    m.setup(a);
    const Outcome out = solve(a, m, b, relative(1e-12), options(30));
    CHECK(out.result.converged);
    CHECK(out.result.iterations == 1);
    for (int i = 0; i < 5; ++i) CHECK_THAT(out.x[i], WithinAbs(b[i] / 4.0, 1e-14));
}

TEST_CASE("GMRES with the exact inverse as preconditioner converges in one step", "[linalg][gmres]") {
    const la::CsrMatrix a = convection_diffusion_csr(5, 4, 4.0, 2.0);
    DensePreconditioner m(inverse(to_dense(a)));
    m.setup(a);
    const Outcome out = solve(a, m, test_vector(a.rows()), relative(1e-12), options(10));
    CHECK(out.result.converged);
    CHECK(out.result.iterations == 1);
}

TEST_CASE("Full GMRES never needs more iterations than restarted GMRES", "[linalg][gmres]") {
    // Full GMRES minimises over a superset of the restarted search spaces.
    // (Between two restart lengths no such ordering holds in general.)
    const la::CsrMatrix a = convection_diffusion_csr(14, 12, 12.0, 4.0, Convection::Central);
    la::IdentityPreconditioner m;
    m.setup(a);
    const std::vector<double> b = test_vector(a.rows(), 0.4);
    const Outcome full = solve(a, m, b, relative(1e-8), options(a.rows()));
    REQUIRE(full.result.converged);
    for (int restart : {2, 5, 20}) {
        CAPTURE(restart);
        const Outcome r = solve(a, m, b, relative(1e-8), options(restart));
        REQUIRE(r.result.converged);
        CHECK(full.result.iterations <= r.result.iterations);
        if (r.result.iterations > restart) CHECK(r.cycles > 1);
    }
}

TEST_CASE("GMRES(1) converges for positive-real matrices", "[linalg][gmres]") {
    // Central differences: symmetric part = SPD Laplacian (Elman 1982).
    const la::CsrMatrix a = convection_diffusion_csr(8, 8, 3.0, 2.0, Convection::Central);
    la::IdentityPreconditioner m;
    m.setup(a);
    la::ConvergenceCriteria c = relative(1e-8);
    c.max_iterations = 20000;
    const Outcome out = solve(a, m, test_vector(a.rows()), c, options(1));
    CAPTURE(out.result.iterations);
    CHECK(out.result.converged);
    CHECK(out.cycles == out.result.iterations); // one step per cycle
}

TEST_CASE("MGS and CGS2 orthogonalisation give the same iterates", "[linalg][gmres]") {
    const la::CsrMatrix a = convection_diffusion_csr(12, 10, 6.0, 6.0);
    la::IncompleteLU m;
    m.setup(a);
    const std::vector<double> b = test_vector(a.rows(), 2.0);
    const Outcome mgs = solve(a, m, b, relative(1e-10), options(8, la::Orthogonalization::ModifiedGramSchmidt));
    const Outcome cgs2 = solve(a, m, b, relative(1e-10), options(8, la::Orthogonalization::ClassicalGramSchmidtTwice));
    REQUIRE(mgs.result.converged);
    REQUIRE(cgs2.result.converged);
    CHECK(std::abs(mgs.result.iterations - cgs2.result.iterations) <= 1);
    CHECK(max_abs_diff(mgs.x, cgs2.x) < 1e-8 * max_abs(mgs.x));
}

// ---------------------------------------------------------------------------
// Breakdown, cost, control, contracts
// ---------------------------------------------------------------------------

TEST_CASE("GMRES detects a singular projected system", "[linalg][gmres][breakdown]") {
    // A v_1 = 0: R_11 = 0, the least-squares problem is singular.
    const la::CsrMatrix a = from_dense({{1, 0}, {0, 0}});
    la::IdentityPreconditioner m;
    m.setup(a);
    const Outcome out = solve(a, m, {0.0, 1.0}, relative(1e-10), options(10));
    CHECK(out.result.reason == la::StopReason::Breakdown);
    CHECK_FALSE(out.result.converged);
    CHECK(out.result.iterations == 0);
    CHECK(out.x == std::vector<double>{0.0, 0.0});

    SECTION("zero right-hand side instead: exact solution without iterating") {
        la::Gmres gmres; // tolerances off
        std::vector<double> x(2, 0.0);
        const la::SolveResult r = gmres.solve(a, m, std::vector<double>{0.0, 0.0}, x);
        CHECK(r.reason == la::StopReason::ExactSolution);
        CHECK(r.iterations == 0);
    }
}

TEST_CASE("GMRES work per iteration", "[linalg][gmres]") {
    // Per inner iteration one product and one preconditioner application; per
    // cycle one more of each (true residual, solution update); plus r_0.
    const la::CsrMatrix a = convection_diffusion_csr(12, 10, 5.0, 2.0);
    const CountingOperator op(a);
    la::JacobiPreconditioner jacobi;
    jacobi.setup(a);
    ForwardingPreconditioner m(jacobi, true);
    la::Gmres solver(relative(1e-10), options(6));
    std::vector<double> x(a.rows(), 0.0);
    const la::SolveResult r = solver.solve(op, m, test_vector(a.rows()), x);
    REQUIRE(r.converged);
    CAPTURE(r.iterations, solver.cycles());
    CHECK(solver.cycles() > 1);
    CHECK(op.applications == 1 + r.iterations + solver.cycles());
    CHECK(m.applications == r.iterations + solver.cycles());
}

TEST_CASE("GMRES iteration cap, non-finite data and reuse", "[linalg][gmres]") {
    const la::CsrMatrix a = convection_diffusion_csr(10, 10, 8.0, 4.0);
    la::JacobiPreconditioner m;
    m.setup(a);
    SECTION("the cap can end a cycle early") {
        la::ConvergenceCriteria c = relative(1e-14);
        c.max_iterations = 7;
        la::Gmres solver(c, options(5));
        std::vector<double> x(a.rows(), 0.0);
        const la::SolveResult r = solver.solve(a, m, test_vector(a.rows()), x);
        CHECK(r.reason == la::StopReason::MaxIterations);
        CHECK(r.iterations == 7);
        CHECK(solver.cycles() == 2);
        CHECK_THAT(r.final_residual, WithinRel(residual_norm(to_dense(a), test_vector(a.rows()), x), 1e-10));
    }
    SECTION("NaN in the right-hand side") {
        std::vector<double> b = test_vector(a.rows());
        b[3] = std::numeric_limits<double>::infinity();
        const Outcome out = solve(a, m, b, relative(1e-10), options(10));
        CHECK(out.result.reason == la::StopReason::NonFinite);
    }
    SECTION("repeated solves are identical") {
        la::Gmres solver(relative(1e-10), options(10));
        const std::vector<double> b = test_vector(a.rows());
        std::vector<double> x1(a.rows(), 0.0), x2(a.rows(), 0.0);
        solver.solve(a, m, b, x1);
        solver.solve(a, m, b, x2);
        CHECK(x1 == x2);
    }
}

TEST_CASE("GMRES honours Dirichlet constraints and matches CG on an SPD pressure system", "[linalg][gmres]") {
    GridSpec g;
    g.lx = 1.5;
    g.nx = 8;
    g.ny = 5;
    const PoissonSystem s = make_poisson_system(make_grid_mesh(g), varying_dt(g.nelem()));
    const int n = s.mesh.npoin;
    const la::EbeOperator op(s.mesh, s.gstif, s.pdiag);
    const la::DirichletConstraints bc({0, 9}, {0.5, -0.25});
    la::JacobiPreconditioner m;
    m.setup(la::ConstrainedOperator(op, bc));
    const std::vector<double> rhs = test_vector(n, 2.2);

    std::vector<double> x_cg(n, 0.0), x_gm(n, 3.0);
    la::ConjugateGradient cg(relative(1e-13));
    la::Gmres gmres(relative(1e-13), options(15));
    REQUIRE(cg.solve(op, m, rhs, x_cg, bc).converged);
    REQUIRE(gmres.solve(op, m, rhs, x_gm, bc).converged);
    CHECK(x_gm[0] == 0.5);
    CHECK(x_gm[9] == -0.25);
    CHECK(max_abs_diff(x_gm, x_cg) < 1e-9 * max_abs(x_cg));
}

TEST_CASE("GMRES options", "[linalg][gmres]") {
    CHECK_THROWS_AS(la::Gmres({}, options(0)), std::invalid_argument);
    const la::Gmres g({}, options(12, la::Orthogonalization::ClassicalGramSchmidtTwice));
    CHECK(g.name() == "GMRES(12)");
    CHECK(g.options().restart == 12);
    CHECK(g.options().orthogonalization == la::Orthogonalization::ClassicalGramSchmidtTwice);
    CHECK(la::to_string(la::Orthogonalization::ModifiedGramSchmidt) == "mgs");
    CHECK(la::to_string(la::Orthogonalization::ClassicalGramSchmidtTwice) == "cgs2");
    CHECK(la::Gmres().name() == "GMRES(30)");
}

TEST_CASE("GMRES: a tolerance below the attainable accuracy ends in stagnation, not at the cap",
          "[linalg][gmres]") {
    // A relative tolerance of 1e-19 cannot be met by the true residual
    // (rounding floor ~1e-16 relative): the recursive/estimated residual
    // claims convergence, the true residual does not improve any further.
    const la::CsrMatrix a = convection_diffusion_csr(10, 10, 6.0, 3.0);
    la::IncompleteLU m;
    m.setup(a);
    const std::vector<double> b = test_vector(a.rows(), 0.9);
    la::ConvergenceCriteria c;
    c.rel_tol = 1e-19;
    c.max_iterations = 100000;
    la::Gmres solver(c, options(20));
    std::vector<double> x(a.rows(), 0.0);
    const la::SolveResult r = solver.solve(a, m, b, x);
    CAPTURE(r.iterations, r.final_residual / r.initial_residual);
    CHECK(r.reason == la::StopReason::Stagnation);
    CHECK_FALSE(r.converged);
    CHECK(r.iterations < 1000);
    CHECK(r.final_residual < 1e-14 * r.initial_residual); // as accurate as attainable
    CHECK_THAT(r.final_residual, WithinRel(residual_norm(to_dense(a), b, x), 1e-8));
    const std::vector<double> exact = solve_dense(to_dense(a), b);
    CHECK(max_abs_diff(x, exact) < 1e-12 * max_abs(exact));
}
