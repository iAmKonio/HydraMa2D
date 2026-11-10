// BiCGSTAB (linalg::BiCGStab) on non-symmetric systems:
//   * agreement with dense direct solutions on upwind and central
//     convection-diffusion operators, with none / Jacobi / ILU(0) /
//     RCM+ILU(0) preconditioning; reported residual = true residual
//   * exact convergence properties: exact preconditioner (one step),
//     eigenvector right-hand side (one step), termination within the number
//     of distinct eigenvalues
//   * breakdown detection: pivot (alpha), stagnation (omega) and Lanczos
//     (rho) breakdowns on symbolically constructed systems, singular systems
//   * work per iteration, iteration cap, non-finite data, Dirichlet
//     constraints, reuse

#include "LinalgTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <map>

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

struct Outcome {
    la::SolveResult result;
    std::vector<double> x;
};

Outcome solve(const la::LinearOperator& a, la::Preconditioner& m, const std::vector<double>& b,
              const la::ConvergenceCriteria& c, la::IterationMonitor* monitor = nullptr) {
    la::BiCGStab solver(c);
    solver.set_monitor(monitor);
    Outcome o;
    o.x.assign(b.size(), 0.0);
    o.result = solver.solve(a, m, b, o.x);
    return o;
}

double max_abs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

} // namespace

TEST_CASE("BiCGStab agrees with a direct solve on convection-diffusion operators", "[linalg][bicgstab]") {
    const Convection scheme = GENERATE(Convection::Upwind, Convection::Central);
    const std::string preconditioner = GENERATE(std::string("none"), std::string("jacobi"), std::string("ilu0"),
                                                std::string("rcm+ilu0"));
    CAPTURE(scheme == Convection::Upwind ? "upwind" : "central", preconditioner);

    const la::CsrMatrix a = convection_diffusion_csr(10, 8, 6.0, 3.0, scheme);
    const DenseMatrix d = to_dense(a);
    REQUIRE_FALSE(a.is_symmetric());
    const std::vector<double> b = test_vector(a.rows(), 0.3);
    const std::vector<double> exact = solve_dense(d, b);

    la::PreconditionerOptions o;
    if (preconditioner == "rcm+ilu0") o.ordering = "rcm";
    auto m = la::make_preconditioner(preconditioner == "rcm+ilu0" ? "ilu0" : preconditioner, o);
    m->setup(a);

    const Outcome out = solve(a, *m, b, relative(1e-12));
    CAPTURE(out.result.iterations);
    REQUIRE(out.result.converged);
    CHECK(out.result.reason == la::StopReason::RelativeTolerance);
    CHECK(max_abs_diff(out.x, exact) < 1e-9 * max_abs(exact));
    // The reported residual is the true residual of the returned iterate.
    CHECK_THAT(out.result.final_residual, WithinRel(residual_norm(d, b, out.x), 1e-10));
    CHECK(out.result.final_residual < 1e-12 * out.result.initial_residual);
}

TEST_CASE("BiCGStab with ILU(0) needs far fewer iterations than without", "[linalg][bicgstab]") {
    const la::CsrMatrix a = convection_diffusion_csr(20, 20, 10.0, 5.0);
    const std::vector<double> b = test_vector(a.rows(), 1.1);
    std::map<std::string, int> iterations;
    for (std::string name : {"none", "jacobi", "ilu0"}) {
        auto m = la::make_preconditioner(name);
        m->setup(a);
        const Outcome out = solve(a, *m, b, relative(1e-10));
        REQUIRE(out.result.converged);
        iterations[name] = out.result.iterations;
    }
    CAPTURE(iterations["none"], iterations["jacobi"], iterations["ilu0"]);
    CHECK(3 * iterations["ilu0"] < iterations["jacobi"]);
    CHECK(3 * iterations["ilu0"] < iterations["none"]);
}

// ---------------------------------------------------------------------------
// Exact convergence properties
// ---------------------------------------------------------------------------

TEST_CASE("BiCGStab with the exact inverse as preconditioner converges in one step", "[linalg][bicgstab]") {
    const la::CsrMatrix a = convection_diffusion_csr(5, 4, 4.0, 2.0, Convection::Central);
    DensePreconditioner m(inverse(to_dense(a)));
    m.setup(a);
    const Outcome out = solve(a, m, test_vector(a.rows()), relative(1e-12));
    CHECK(out.result.converged);
    CHECK(out.result.iterations == 1);
}

TEST_CASE("BiCGStab terminates within the number of distinct eigenvalues", "[linalg][bicgstab]") {
    // BiCG converges in at most deg(minimal polynomial) steps, and the
    // BiCGSTAB residual polynomial contains the BiCG one as a factor.
    const SpectralMatrix sm = matrix_with_spectrum({2, 5, 2, 9, 5, 2, 9, 5});
    const la::CsrMatrix a = from_dense(sm.a);
    la::IdentityPreconditioner m;
    m.setup(a);
    const Outcome out = solve(a, m, test_vector(8), relative(1e-10));
    CAPTURE(out.result.iterations, out.result.final_residual);
    CHECK(out.result.converged);
    CHECK(out.result.iterations <= 3);
}

TEST_CASE("BiCGStab solves for an eigenvector right-hand side in one step", "[linalg][bicgstab]") {
    const SpectralMatrix sm = matrix_with_spectrum({3, 7, 1, 4, 6});
    const la::CsrMatrix a = from_dense(sm.a);
    std::vector<double> b(5);
    for (int i = 0; i < 5; ++i) b[i] = sm.s[i][1]; // eigenvector of 7
    la::IdentityPreconditioner m;
    m.setup(a);
    const Outcome out = solve(a, m, b, relative(1e-12));
    CHECK(out.result.converged);
    CHECK(out.result.iterations == 1);
    for (int i = 0; i < 5; ++i) CHECK_THAT(out.x[i], WithinAbs(b[i] / 7.0, 1e-14));
}

// ---------------------------------------------------------------------------
// Breakdown detection
// ---------------------------------------------------------------------------

TEST_CASE("BiCGStab detects a pivot breakdown", "[linalg][bicgstab][breakdown]") {
    // (r~, A r~) = 0 for the rotation: alpha = rho / (r~, v) is undefined.
    const BreakdownCase bc = rotation_breakdown();
    const la::CsrMatrix a = from_dense(bc.a);
    la::IdentityPreconditioner m;
    m.setup(a);
    const Outcome out = solve(a, m, bc.b, relative(1e-10));
    CHECK(out.result.reason == la::StopReason::Breakdown);
    CHECK_FALSE(out.result.converged);
    CHECK(out.result.iterations == 0);
    CHECK(out.x == std::vector<double>{0.0, 0.0}); // no step taken
}

TEST_CASE("BiCGStab detects stagnation (omega = 0) and keeps the half step", "[linalg][bicgstab][breakdown]") {
    const BreakdownCase bc = omega_breakdown();
    const la::CsrMatrix a = from_dense(bc.a);
    la::IdentityPreconditioner m;
    m.setup(a);
    la::ResidualHistory history;
    const Outcome out = solve(a, m, bc.b, relative(1e-10), &history);
    CHECK(out.result.reason == la::StopReason::Breakdown);
    CHECK(out.result.iterations == 1);
    // x = alpha r0 with alpha = ||r0||^2 / (r0, A r0); its residual is s.
    const double alpha = la::dot(bc.b, bc.b) / la::dot(bc.b, multiply(bc.a, bc.b));
    for (int i = 0; i < 3; ++i) CHECK_THAT(out.x[i], WithinRel(alpha * bc.b[i], 1e-14));
    CHECK_THAT(out.result.final_residual, WithinRel(1.1785113019775793, 1e-12));
    CHECK_THAT(out.result.final_residual, WithinRel(residual_norm(bc.a, bc.b, out.x), 1e-12));
    CHECK(history.residuals().size() == 2);
}

TEST_CASE("BiCGStab detects a Lanczos (rho) breakdown", "[linalg][bicgstab][breakdown]") {
    const BreakdownCase bc = rho_breakdown();
    const la::CsrMatrix a = from_dense(bc.a);
    la::IdentityPreconditioner m;
    m.setup(a);
    const Outcome out = solve(a, m, bc.b, relative(1e-10));
    CHECK(out.result.reason == la::StopReason::Breakdown);
    CHECK(out.result.iterations == 1); // one full step, then (r~, r_1) = 0
    CHECK(out.result.final_residual > 0.5);
    CHECK_THAT(out.result.final_residual, WithinRel(residual_norm(bc.a, bc.b, out.x), 1e-12));
}

TEST_CASE("GMRES solves the systems on which BiCGStab breaks down", "[linalg][bicgstab][gmres][breakdown]") {
    const int which = GENERATE(0, 1, 2);
    const BreakdownCase bc = which == 0 ? rotation_breakdown() : which == 1 ? omega_breakdown() : rho_breakdown();
    CAPTURE(which);
    const la::CsrMatrix a = from_dense(bc.a);
    la::IdentityPreconditioner m;
    m.setup(a);
    la::Gmres gmres(relative(1e-12));
    std::vector<double> x(bc.b.size(), 0.0);
    const la::SolveResult r = gmres.solve(a, m, bc.b, x);
    CHECK(r.converged);
    CHECK(r.iterations <= static_cast<int>(bc.b.size()));
    const std::vector<double> exact = solve_dense(bc.a, bc.b);
    CHECK(max_abs_diff(x, exact) < 1e-12);
}

TEST_CASE("BiCGStab on singular systems and zero data", "[linalg][bicgstab][breakdown]") {
    la::IdentityPreconditioner m;
    SECTION("A v = 0 for the first direction: breakdown") {
        const la::CsrMatrix a = from_dense({{1, 0}, {0, 0}});
        m.setup(a);
        const Outcome out = solve(a, m, {0.0, 1.0}, relative(1e-10));
        CHECK(out.result.reason == la::StopReason::Breakdown);
    }
    SECTION("zero right-hand side: exact solution without iterating") {
        const la::CsrMatrix a = convection_diffusion_csr(4, 3, 2.0, 1.0);
        m.setup(a);
        la::BiCGStab solver; // tolerances off
        std::vector<double> x(a.rows(), 0.0);
        const la::SolveResult r = solver.solve(a, m, std::vector<double>(a.rows(), 0.0), x);
        CHECK(r.reason == la::StopReason::ExactSolution);
        CHECK(r.iterations == 0);
    }
}

// ---------------------------------------------------------------------------
// Cost, control and contracts
// ---------------------------------------------------------------------------

TEST_CASE("BiCGStab work per iteration", "[linalg][bicgstab]") {
    // Two products and two preconditioner applications per iteration, plus
    // r_0 and the final true-residual check (one less of each if the solve
    // ends on a half step).
    const la::CsrMatrix a = convection_diffusion_csr(12, 10, 5.0, 2.0);
    const CountingOperator op(a);
    la::IncompleteLU ilu;
    ilu.setup(a);
    ForwardingPreconditioner m(ilu, false);
    la::BiCGStab solver(relative(1e-10));
    std::vector<double> x(a.rows(), 0.0);
    const la::SolveResult r = solver.solve(op, m, test_vector(a.rows()), x);
    REQUIRE(r.converged);
    REQUIRE(solver.residual_replacements() == 0);
    const int k = r.iterations;
    CAPTURE(k, op.applications, m.applications);
    CHECK((op.applications == 2 * k + 2 || op.applications == 2 * k + 1));
    CHECK((m.applications == 2 * k || m.applications == 2 * k - 1));
}

TEST_CASE("BiCGStab iteration cap and non-finite data", "[linalg][bicgstab]") {
    const la::CsrMatrix a = convection_diffusion_csr(10, 10, 8.0, 4.0);
    la::JacobiPreconditioner m;
    m.setup(a);
    SECTION("cap") {
        la::ConvergenceCriteria c = relative(1e-14);
        c.max_iterations = 4;
        const Outcome out = solve(a, m, test_vector(a.rows()), c);
        CHECK(out.result.reason == la::StopReason::MaxIterations);
        CHECK(out.result.iterations == 4);
        CHECK_FALSE(out.result.converged);
    }
    SECTION("NaN in the right-hand side") {
        std::vector<double> b = test_vector(a.rows());
        b[7] = std::numeric_limits<double>::quiet_NaN();
        const Outcome out = solve(a, m, b, relative(1e-10));
        CHECK(out.result.reason == la::StopReason::NonFinite);
        CHECK(out.result.iterations == 0);
    }
}

TEST_CASE("BiCGStab honours Dirichlet constraints and matches CG on an SPD pressure system",
          "[linalg][bicgstab]") {
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

    std::vector<double> x_cg(n, 0.0), x_bi(n, 3.0);
    la::ConjugateGradient cg(relative(1e-13));
    la::BiCGStab bi(relative(1e-13));
    REQUIRE(cg.solve(op, m, rhs, x_cg, bc).converged);
    const la::SolveResult r = bi.solve(op, m, rhs, x_bi, bc);
    REQUIRE(r.converged);
    CHECK(x_bi[0] == 0.5);
    CHECK(x_bi[9] == -0.25);
    CHECK(max_abs_diff(x_bi, x_cg) < 1e-9 * max_abs(x_cg));
}

TEST_CASE("BiCGStab solver objects are reusable", "[linalg][bicgstab]") {
    const la::CsrMatrix a = convection_diffusion_csr(9, 7, 3.0, 1.0);
    la::IncompleteLU m;
    m.setup(a);
    la::BiCGStab solver(relative(1e-10));
    CHECK(solver.name() == "BiCGStab");
    const std::vector<double> b = test_vector(a.rows());
    std::vector<double> x1(a.rows(), 0.0), x2(a.rows(), 0.0);
    const la::SolveResult r1 = solver.solve(a, m, b, x1);
    const la::SolveResult r2 = solver.solve(a, m, b, x2);
    CHECK(r1.iterations == r2.iterations);
    CHECK(x1 == x2);
    std::vector<double> wrong(3);
    CHECK_THROWS_AS(solver.solve(a, m, wrong, x1), std::invalid_argument);
}

TEST_CASE("BiCGStab: a tolerance below the attainable accuracy ends in stagnation, not at the cap",
          "[linalg][bicgstab]") {
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
    la::BiCGStab solver(c);
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
