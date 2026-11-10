// Iteration monitors (Observer pattern): ResidualHistory, StreamMonitor,
// MonitorGroup, the notification protocol of the Krylov solvers, and the
// guarantee that observing a solve never changes its result.

#include "IoTestUtils.hpp"
#include "LinalgTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sstream>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinRel;

namespace {

GridSpec monitor_grid() {
    GridSpec g;
    g.lx = 2.0;
    g.ly = 1.0;
    g.nx = 8;
    g.ny = 4;
    return g;
}

struct System {
    PoissonSystem s;
    std::vector<double> rhs;
    la::DirichletConstraints bc{{0}, {0.0}};
};

System make_system() {
    const GridSpec g = monitor_grid();
    System sys{make_poisson_system(make_grid_mesh(g), varying_dt(g.nelem())), {}};
    sys.rhs = test_vector(sys.s.mesh.npoin);
    return sys;
}

la::SolveResult solve(const System& sys, la::ConjugateGradient& cg, std::vector<double>& x) {
    const la::EbeOperator op(sys.s.mesh, sys.s.gstif, sys.s.pdiag);
    la::JacobiPreconditioner m;
    m.setup(la::ConstrainedOperator(op, sys.bc));
    return cg.solve(op, m, sys.rhs, x, sys.bc);
}

la::ConvergenceCriteria tight() {
    la::ConvergenceCriteria c;
    c.rel_tol = 1e-10;
    return c;
}

} // namespace

TEST_CASE("ResidualHistory records r_0 and every iteration of a solve", "[linalg][monitor]") {
    const System sys = make_system();
    la::ResidualHistory history;
    la::ConjugateGradient cg(tight());
    cg.set_monitor(&history);
    CHECK(cg.monitor() == &history);

    std::vector<double> x(sys.s.mesh.npoin, 0.0);
    const la::SolveResult r = solve(sys, cg, x);
    REQUIRE(r.converged);

    CHECK(history.finished());
    CHECK(history.solves() == 1);
    CHECK(history.solver() == "CG");
    REQUIRE(history.residuals().size() == static_cast<size_t>(r.iterations) + 1);
    CHECK(history.residuals().front() == r.initial_residual);
    CHECK(history.residuals().back() == r.final_residual);
    CHECK(history.result().iterations == r.iterations);
    CHECK(history.result().reason == r.reason);
    CHECK(history.residuals().back() / history.residuals().front() < 1e-10);
}

TEST_CASE("ResidualHistory restarts for every solve", "[linalg][monitor]") {
    const System sys = make_system();
    la::ResidualHistory history;
    la::ConjugateGradient cg(tight());
    cg.set_monitor(&history);

    std::vector<double> x(sys.s.mesh.npoin, 0.0);
    solve(sys, cg, x);
    const size_t first = history.residuals().size();

    // Second solve from the converged solution: (almost) nothing to do.
    la::ConvergenceCriteria c;
    c.abs_tol = 1e-6;
    cg.set_criteria(c);
    const la::SolveResult again = solve(sys, cg, x);
    CHECK(history.solves() == 2);
    CHECK(again.iterations == 0);
    CHECK(history.residuals().size() == 1);
    CHECK(history.residuals().size() < first);
}

TEST_CASE("In legacy mode the first recorded iteration is the reference residual", "[linalg][monitor]") {
    const System sys = make_system();
    la::ResidualHistory history;
    la::ConjugateGradient cg(la::ConvergenceCriteria::legacy_pcg(1e-3, 0.0));
    cg.set_monitor(&history);
    std::vector<double> x(sys.s.mesh.npoin, 0.0);
    const la::SolveResult r = solve(sys, cg, x);
    REQUIRE(history.residuals().size() >= 2);
    CHECK(history.residuals()[1] == r.reference_residual);
    CHECK(r.final_residual / r.reference_residual < 1e-3);
}

TEST_CASE("Solvers notify start, every iteration and finish in order", "[linalg][monitor]") {
    const System sys = make_system();
    EventLog log;
    la::ConvergenceCriteria c;
    c.max_iterations = 3;
    la::ConjugateGradient cg(c);
    cg.set_monitor(&log);
    std::vector<double> x(sys.s.mesh.npoin, 0.0);
    const la::SolveResult r = solve(sys, cg, x);
    CHECK(r.reason == la::StopReason::MaxIterations);
    CHECK(log.events == std::vector<std::string>{"start", "iter1", "iter2", "iter3", "finish"});

    SECTION("a solve that needs no iteration still reports start and finish") {
        log.events.clear();
        std::vector<double> zero_rhs(sys.s.mesh.npoin, 0.0), y(sys.s.mesh.npoin, 0.0);
        const la::EbeOperator op(sys.s.mesh, sys.s.gstif, sys.s.pdiag);
        la::JacobiPreconditioner m;
        m.setup(op);
        const la::SolveResult z = cg.solve(op, m, zero_rhs, y, sys.bc);
        CHECK(z.iterations == 0);
        CHECK(log.events == std::vector<std::string>{"start", "finish"});
    }
}

TEST_CASE("Monitoring never changes the result", "[linalg][monitor]") {
    const System sys = make_system();
    const la::ConvergenceCriteria crit = GENERATE(la::ConvergenceCriteria::legacy_pcg(0.1, 1e-6), tight());

    la::ConjugateGradient plain(crit), observed(crit);
    la::ResidualHistory history;
    std::ostringstream sink;
    la::StreamMonitor stream(sink, 5);
    la::MonitorGroup group;
    group.add(&history);
    group.add(&stream);
    observed.set_monitor(&group);

    std::vector<double> x1(sys.s.mesh.npoin, 0.0), x2(sys.s.mesh.npoin, 0.0);
    const la::SolveResult r1 = solve(sys, plain, x1);
    const la::SolveResult r2 = solve(sys, observed, x2);
    CHECK(x1 == x2);
    CHECK(r1.iterations == r2.iterations);
    CHECK(r1.final_residual == r2.final_residual);
    CHECK(r1.reference_residual == r2.reference_residual);
    CHECK_FALSE(sink.str().empty());

    SECTION("and a monitor can be detached") {
        observed.set_monitor(nullptr);
        CHECK(observed.monitor() == nullptr);
        const int solves = history.solves();
        std::vector<double> x3(sys.s.mesh.npoin, 0.0);
        solve(sys, observed, x3);
        CHECK(history.solves() == solves);
    }
}

TEST_CASE("ResidualHistory growth factor", "[linalg][monitor]") {
    la::ResidualHistory h;
    CHECK(h.max_growth_factor() == 1.0); // empty
    h.on_start("x", 1.0);
    CHECK(h.max_growth_factor() == 1.0); // single entry
    h.on_iteration(1, 0.5);
    h.on_iteration(2, 0.75);
    h.on_iteration(3, 0.25);
    CHECK_THAT(h.max_growth_factor(), WithinRel(1.5, 1e-15));
    CHECK_FALSE(h.finished());
    h.on_finish(la::SolveResult{});
    CHECK(h.finished());

    la::ResidualHistory monotone;
    monotone.on_start("x", 8.0);
    monotone.on_iteration(1, 4.0);
    monotone.on_iteration(2, 1.0);
    CHECK(monotone.max_growth_factor() == 0.5);
}

TEST_CASE("StreamMonitor writes the first, every n-th and the last iteration", "[linalg][monitor]") {
    std::ostringstream out;
    la::StreamMonitor m(out, 2, "> ");
    m.on_start("CG", 1.0);
    for (int k = 1; k <= 5; ++k) m.on_iteration(k, 1.0 / (k + 1));
    la::SolveResult r;
    r.iterations = 5;
    r.final_residual = 1.0 / 6.0;
    r.converged = true;
    r.reason = la::StopReason::RelativeTolerance;
    m.on_finish(r);

    const std::vector<std::string> lines = split_lines(out.str());
    REQUIRE(lines.size() == 6);
    CHECK(lines[0].rfind("> CG iter 0 residual ", 0) == 0);
    CHECK(lines[1].rfind("> CG iter 1 residual ", 0) == 0);
    CHECK(lines[2].rfind("> CG iter 2 residual ", 0) == 0);
    CHECK(lines[3].rfind("> CG iter 4 residual ", 0) == 0);
    CHECK(lines[4].rfind("> CG iter 5 residual ", 0) == 0); // last, although 5 % 2 != 0
    CHECK(lines[5] == "> CG converged (relative tolerance) after 5 iterations");

    SECTION("the last iteration is not repeated when it was already written") {
        std::ostringstream out2;
        la::StreamMonitor m2(out2, 1);
        m2.on_start("CG", 1.0);
        m2.on_iteration(1, 0.1);
        la::SolveResult r2;
        r2.iterations = 1;
        r2.reason = la::StopReason::MaxIterations;
        m2.on_finish(r2);
        const auto l2 = split_lines(out2.str());
        REQUIRE(l2.size() == 3);
        CHECK(l2[2] == "  CG NOT converged (maximum iterations) after 1 iterations");
    }
    SECTION("the interval must be positive") {
        CHECK_THROWS_AS(la::StreamMonitor(out, 0), std::invalid_argument);
    }
}

TEST_CASE("MonitorGroup forwards to all members", "[linalg][monitor]") {
    EventLog a, b;
    la::MonitorGroup group;
    group.add(&a);
    group.add(nullptr); // ignored
    group.add(&b);
    CHECK(group.size() == 2);

    group.on_start("CG", 1.0);
    group.on_iteration(1, 0.5);
    group.on_finish(la::SolveResult{});
    const std::vector<std::string> expected = {"start", "iter1", "finish"};
    CHECK(a.events == expected);
    CHECK(b.events == expected);
}
