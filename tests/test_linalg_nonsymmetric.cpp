// Convergence-rate comparison of the non-symmetric Krylov solvers
// (BiCGSTAB, GMRES(m)) with Jacobi and ILU(0) preconditioning on
// convection-diffusion operators, measured against dense direct solutions:
//   * error ||x_k - x*|| decreases as the tolerance is tightened and
//     reaches the direct solution
//   * average residual reduction per iteration (r_k / r_0)^(1/k):
//     ILU(0) < Jacobi < 1, for both methods and both discretisations
//   * ILU(0) is the right partner for strong convection: iteration counts
//   * all methods (CG, BiCGSTAB, GMRES) agree on a symmetric pressure system

#include "LinalgTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <map>

using namespace hydrama;
using namespace hydrama::test;

namespace {

struct Problem {
    la::CsrMatrix a;
    std::vector<double> b;
    std::vector<double> exact;
};

// 24 x 24 grid: 576 unknowns (dense direct solve for the reference).
const Problem& problem(Convection scheme) {
    static const Problem upwind = [] {
        Problem p{convection_diffusion_csr(24, 24, 20.0, 8.0, Convection::Upwind), test_vector(576, 0.6), {}};
        p.exact = solve_dense(to_dense(p.a), p.b);
        return p;
    }();
    static const Problem central = [] {
        Problem p{convection_diffusion_csr(24, 24, 3.0, 1.5, Convection::Central), test_vector(576, 0.6), {}};
        p.exact = solve_dense(to_dense(p.a), p.b);
        return p;
    }();
    return scheme == Convection::Upwind ? upwind : central;
}

struct Run {
    la::SolveResult result;
    std::vector<double> x;
    double rate = 1.0; // average residual reduction per iteration
};

Run run(const std::string& method, const std::string& preconditioner, const Problem& p, double tol) {
    auto m = la::make_preconditioner(preconditioner);
    m->setup(p.a);
    la::ConvergenceCriteria c;
    c.rel_tol = tol;
    c.max_iterations = 5000;
    auto solver = la::make_krylov_solver(method, c);
    la::ResidualHistory history;
    solver->set_monitor(&history);
    Run r;
    r.x.assign(p.b.size(), 0.0);
    r.result = solver->solve(p.a, *m, p.b, r.x);
    if (r.result.iterations > 0) {
        r.rate = std::pow(r.result.final_residual / r.result.initial_residual, 1.0 / r.result.iterations);
    }
    return r;
}

double relative_error(const std::vector<double>& x, const std::vector<double>& exact) {
    std::vector<double> e(x.size());
    for (size_t i = 0; i < x.size(); ++i) e[i] = x[i] - exact[i];
    return la::norm2(e) / la::norm2(exact);
}

} // namespace

TEST_CASE("Non-symmetric solvers converge to the direct solution as the tolerance tightens",
          "[linalg][nonsymmetric]") {
    const Convection scheme = GENERATE(Convection::Upwind, Convection::Central);
    const std::string method = GENERATE(std::string("bicgstab"), std::string("gmres"));
    const std::string preconditioner = GENERATE(std::string("jacobi"), std::string("ilu0"));
    CAPTURE(scheme == Convection::Upwind ? "upwind" : "central", method, preconditioner);
    const Problem& p = problem(scheme);

    double previous = 1e300;
    for (double tol : {1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12}) {
        const Run r = run(method, preconditioner, p, tol);
        const double err = relative_error(r.x, p.exact);
        CAPTURE(tol, r.result.iterations, err);
        REQUIRE(r.result.converged);
        CHECK(r.result.final_residual <= tol * r.result.initial_residual);
        // Error bound ||e|| / ||x|| <= cond(A) ||r|| / ||b||; cond(A) < 1e3 here.
        CHECK(err <= 1e3 * tol);
        CHECK(err < previous * 1.0001);
        previous = err;
    }
    CHECK(previous < 1e-10);
}

TEST_CASE("ILU(0) gives faster convergence rates than Jacobi", "[linalg][nonsymmetric]") {
    const Convection scheme = GENERATE(Convection::Upwind, Convection::Central);
    const std::string method = GENERATE(std::string("bicgstab"), std::string("gmres"));
    CAPTURE(scheme == Convection::Upwind ? "upwind" : "central", method);
    const Problem& p = problem(scheme);

    const Run jacobi = run(method, "jacobi", p, 1e-10);
    const Run ilu = run(method, "ilu0", p, 1e-10);
    CAPTURE(jacobi.result.iterations, jacobi.rate, ilu.result.iterations, ilu.rate);
    REQUIRE(jacobi.result.converged);
    REQUIRE(ilu.result.converged);
    CHECK(jacobi.rate < 1.0);
    CHECK(ilu.rate < jacobi.rate);
    CHECK(ilu.result.iterations < jacobi.result.iterations);
    CHECK(relative_error(ilu.x, p.exact) < 1e-8);
    CHECK(relative_error(jacobi.x, p.exact) < 1e-8);
}

TEST_CASE("ILU(0) is essential for strongly convection-dominated problems", "[linalg][nonsymmetric]") {
    // Upwind operator with cell Peclet numbers 20 / 8: ILU(0) captures the
    // dominant (lower triangular) transport, so few iterations suffice.
    const Problem& p = problem(Convection::Upwind);
    std::map<std::string, int> it;
    for (std::string method : {"bicgstab", "gmres"}) {
        for (std::string pre : {"none", "jacobi", "ilu0"}) {
            const Run r = run(method, pre, p, 1e-10);
            REQUIRE(r.result.converged);
            it[method + "+" + pre] = r.result.iterations;
        }
    }
    for (const auto& [name, n] : it) {
        CAPTURE(name, n);
        CHECK(n > 0);
    }
    CHECK(4 * it["bicgstab+ilu0"] < it["bicgstab+jacobi"]);
    CHECK(4 * it["gmres+ilu0"] < it["gmres+jacobi"]);
    CHECK(it["bicgstab+ilu0"] <= 15);
    CHECK(it["gmres+ilu0"] <= 20);
}

TEST_CASE("CG, BiCGSTAB and GMRES agree on a symmetric positive definite system", "[linalg][nonsymmetric]") {
    GridSpec g;
    g.lx = 2.0;
    g.nx = 12;
    g.ny = 6;
    const PoissonSystem s = make_poisson_system(make_grid_mesh(g), varying_dt(g.nelem()));
    const la::P1CsrAssembler assembler(s.mesh);
    la::CsrMatrix a = assembler.create_matrix(), ac;
    assembler.assemble_ebe(s.mesh, s.gstif, s.pdiag, a);
    const la::DirichletConstraints bc({0}, {0.0});
    la::constrain(a, bc, ac);
    la::IncompleteLU ilu; // = IC(0) for this SPD matrix, usable by all three methods
    ilu.setup(ac);
    const std::vector<double> rhs = test_vector(a.rows(), 1.5);

    la::ConvergenceCriteria c;
    c.rel_tol = 1e-12;
    std::map<std::string, std::vector<double>> x;
    std::map<std::string, int> iterations;
    for (std::string method : {"cg", "bicgstab", "gmres"}) {
        auto solver = la::make_krylov_solver(method, c);
        x[method].assign(a.rows(), 0.0);
        const la::SolveResult r = solver->solve(a, ilu, rhs, x[method], bc);
        REQUIRE(r.converged);
        iterations[method] = r.iterations;
    }
    CAPTURE(iterations["cg"], iterations["bicgstab"], iterations["gmres"]);
    CHECK(relative_error(x["bicgstab"], x["cg"]) < 1e-10);
    CHECK(relative_error(x["gmres"], x["cg"]) < 1e-10);
    // GMRES minimises the residual norm: never more iterations than CG,
    // whose k-th residual lies in the same (preconditioned) Krylov space.
    CHECK(iterations["gmres"] <= iterations["cg"]);
}
