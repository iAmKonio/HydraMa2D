// ConvergenceControl: the stopping logic shared by all Krylov solvers.
// Exercised directly with scripted residual sequences (no solver involved),
// so every rule of linalg/Convergence.hpp is pinned down independently.

#include "LinalgTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <limits>
#include <optional>

using namespace hydrama::test;
using la::BreakdownPolicy;
using la::ConvergenceControl;
using la::ConvergenceCriteria;
using la::ResidualReference;
using la::StopReason;

namespace {

ConvergenceCriteria criteria(double rel, double abs, ResidualReference ref = ResidualReference::InitialResidual,
                             int max_it = 100, BreakdownPolicy policy = BreakdownPolicy::Stop) {
    ConvergenceCriteria c;
    c.rel_tol = rel;
    c.abs_tol = abs;
    c.reference = ref;
    c.max_iterations = max_it;
    c.on_breakdown = policy;
    return c;
}

// Feeds residuals r_1, r_2, ... until a stop; returns the finished result.
la::SolveResult run(ConvergenceControl& c, double r0, const std::vector<double>& residuals, double rhs = 0.0) {
    if (auto stop = c.begin(r0, rhs)) return c.finish(*stop);
    for (size_t k = 0; k < residuals.size(); ++k) {
        if (auto stop = c.check(static_cast<int>(k) + 1, residuals[k])) return c.finish(*stop);
    }
    return c.finish(StopReason::None);
}

} // namespace

TEST_CASE("Default criteria: tolerances off, initial-residual reference, stop on breakdown",
          "[linalg][convergence]") {
    const ConvergenceCriteria c;
    CHECK(c.rel_tol == 0.0);
    CHECK(c.abs_tol == 0.0);
    CHECK(c.max_iterations == 100000);
    CHECK(c.reference == ResidualReference::InitialResidual);
    CHECK(c.on_breakdown == BreakdownPolicy::Stop);
}

TEST_CASE("Legacy PCG criteria", "[linalg][convergence]") {
    const ConvergenceCriteria c = ConvergenceCriteria::legacy_pcg(0.1, 1e-6);
    CHECK(c.rel_tol == 0.1);
    CHECK(c.abs_tol == 1e-6);
    CHECK(c.max_iterations == 100000);
    CHECK(c.reference == ResidualReference::FirstIteration);
    CHECK(c.on_breakdown == BreakdownPolicy::Continue);
}

TEST_CASE("Relative tolerance against the initial residual", "[linalg][convergence]") {
    ConvergenceControl c(criteria(0.1, 0.0), nullptr, "test");
    const la::SolveResult r = run(c, 10.0, {5.0, 2.0, 1.0, 0.99, 0.5});
    CHECK(r.reason == StopReason::RelativeTolerance);
    CHECK(r.converged);
    CHECK(r.iterations == 4); // 0.99 / 10 < 0.1 (1.0 / 10 is not: strict)
    CHECK(r.initial_residual == 10.0);
    CHECK(r.reference_residual == 10.0);
    CHECK(r.final_residual == 0.99);
}

TEST_CASE("Relative tolerance against the right-hand side", "[linalg][convergence]") {
    ConvergenceControl c(criteria(0.1, 0.0, ResidualReference::RightHandSide), nullptr, "test");
    const la::SolveResult r = run(c, 30.0, {5.0, 3.0, 1.5}, /*rhs=*/20.0);
    CHECK(r.reason == StopReason::RelativeTolerance);
    CHECK(r.iterations == 3); // 1.5 / 20 < 0.1
    CHECK(r.reference_residual == 20.0);
    CHECK(r.initial_residual == 30.0);
}

TEST_CASE("Relative tolerance against the first iteration (legacy)", "[linalg][convergence]") {
    ConvergenceControl c(criteria(0.1, 0.0, ResidualReference::FirstIteration), nullptr, "test");
    const la::SolveResult r = run(c, 1000.0, {4.0, 1.0, 0.5, 0.39});
    CHECK(r.reason == StopReason::RelativeTolerance);
    CHECK(r.iterations == 4); // 0.39 / 4 < 0.1; measured against r_1 = 4, not r_0 = 1000
    CHECK(r.reference_residual == 4.0);
    CHECK(r.initial_residual == 1000.0);
}

TEST_CASE("First-iteration mode always iterates at least once", "[linalg][convergence]") {
    // Even with r_0 = 0 and an absolute tolerance that r_0 meets.
    ConvergenceControl c(criteria(0.1, 1e-6, ResidualReference::FirstIteration), nullptr, "test");
    CHECK_FALSE(c.begin(0.0).has_value());
    const std::optional<StopReason> stop = c.check(1, 0.0);
    REQUIRE(stop.has_value());
    CHECK(*stop == StopReason::AbsoluteTolerance);
    CHECK(c.finish(*stop).iterations == 1);
}

TEST_CASE("A relative tolerance above 1 in first-iteration mode stops after one iteration",
          "[linalg][convergence]") {
    ConvergenceControl c(criteria(1.5, 0.0, ResidualReference::FirstIteration), nullptr, "test");
    const la::SolveResult r = run(c, 3.0, {2.0, 1.0});
    CHECK(r.iterations == 1);
    CHECK(r.reason == StopReason::RelativeTolerance);
}

TEST_CASE("Absolute tolerance", "[linalg][convergence]") {
    ConvergenceControl c(criteria(0.0, 1e-3), nullptr, "test");
    const la::SolveResult r = run(c, 1.0, {1e-1, 1e-2, 1e-3, 9e-4});
    CHECK(r.reason == StopReason::AbsoluteTolerance);
    CHECK(r.iterations == 4); // strict: 1e-3 is not below 1e-3
}

TEST_CASE("Relative tolerance is reported before absolute when both hold", "[linalg][convergence]") {
    ConvergenceControl c(criteria(0.5, 1.0), nullptr, "test");
    const la::SolveResult r = run(c, 1.0, {0.1});
    CHECK(r.reason == StopReason::RelativeTolerance);
}

TEST_CASE("Convergence before the first iteration", "[linalg][convergence]") {
    SECTION("initial residual below the absolute tolerance") {
        ConvergenceControl c(criteria(0.0, 1e-6), nullptr, "test");
        const la::SolveResult r = run(c, 1e-7, {});
        CHECK(r.reason == StopReason::AbsoluteTolerance);
        CHECK(r.iterations == 0);
        CHECK(r.converged);
    }
    SECTION("zero initial residual with tolerances off: exact solution") {
        ConvergenceControl c(criteria(0.0, 0.0), nullptr, "test");
        const la::SolveResult r = run(c, 0.0, {});
        CHECK(r.reason == StopReason::ExactSolution);
        CHECK(r.iterations == 0);
        CHECK(r.converged);
    }
    SECTION("relative tolerance above 1 against the initial residual") {
        ConvergenceControl c(criteria(2.0, 0.0), nullptr, "test");
        CHECK(run(c, 5.0, {}).reason == StopReason::RelativeTolerance);
    }
    SECTION("non-positive iteration budget") {
        ConvergenceControl c(criteria(0.0, 0.0, ResidualReference::InitialResidual, 0), nullptr, "test");
        const la::SolveResult r = run(c, 5.0, {});
        CHECK(r.reason == StopReason::MaxIterations);
        CHECK_FALSE(r.converged);
    }
}

TEST_CASE("Exact zero residual depends on the breakdown policy", "[linalg][convergence]") {
    SECTION("Stop: exact solution ends the solve") {
        ConvergenceControl c(criteria(0.0, 0.0), nullptr, "test");
        const la::SolveResult r = run(c, 1.0, {0.5, 0.0, 0.0});
        CHECK(r.reason == StopReason::ExactSolution);
        CHECK(r.iterations == 2);
        CHECK(r.converged);
    }
    SECTION("Continue (legacy): iterate until another criterion") {
        ConvergenceControl c(criteria(0.0, 0.0, ResidualReference::FirstIteration, 5, BreakdownPolicy::Continue),
                             nullptr, "test");
        const la::SolveResult r = run(c, 1.0, {0.5, 0.0, 0.0, 0.0, 0.0, 0.0});
        CHECK(r.reason == StopReason::MaxIterations);
        CHECK(r.iterations == 5);
        CHECK_FALSE(r.converged);
    }
}

TEST_CASE("Iteration cap", "[linalg][convergence]") {
    const int cap = GENERATE(1, 3, 10);
    ConvergenceControl c(criteria(1e-12, 0.0, ResidualReference::InitialResidual, cap), nullptr, "test");
    const la::SolveResult r = run(c, 1.0, std::vector<double>(20, 0.5));
    CHECK(r.reason == StopReason::MaxIterations);
    CHECK(r.iterations == cap);
    CHECK_FALSE(r.converged);
}

TEST_CASE("Non-finite residuals stop the solve", "[linalg][convergence]") {
    const double bad = GENERATE(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity());
    SECTION("during the iteration") {
        ConvergenceControl c(criteria(0.1, 1e-6), nullptr, "test");
        const la::SolveResult r = run(c, 1.0, {0.9, bad, 0.1});
        CHECK(r.reason == StopReason::NonFinite);
        CHECK(r.iterations == 2);
        CHECK_FALSE(r.converged);
    }
    SECTION("initially, in every reference mode") {
        const ResidualReference ref = GENERATE(ResidualReference::InitialResidual, ResidualReference::RightHandSide,
                                               ResidualReference::FirstIteration);
        ConvergenceControl c(criteria(0.1, 1e-6, ref), nullptr, "test");
        const la::SolveResult r = run(c, bad, {0.1});
        CHECK(r.reason == StopReason::NonFinite);
        CHECK(r.iterations == 0);
    }
}

TEST_CASE("Stop reasons: names and convergence classification", "[linalg][convergence]") {
    CHECK(la::is_converged(StopReason::RelativeTolerance));
    CHECK(la::is_converged(StopReason::AbsoluteTolerance));
    CHECK(la::is_converged(StopReason::ExactSolution));
    CHECK_FALSE(la::is_converged(StopReason::None));
    CHECK_FALSE(la::is_converged(StopReason::MaxIterations));
    CHECK_FALSE(la::is_converged(StopReason::Breakdown));
    CHECK_FALSE(la::is_converged(StopReason::NonFinite));
    CHECK_FALSE(la::is_converged(StopReason::Stagnation));
    CHECK(la::to_string(StopReason::Stagnation) == "stagnation (attainable accuracy reached)");

    CHECK(la::to_string(StopReason::RelativeTolerance) == "relative tolerance");
    CHECK(la::to_string(StopReason::MaxIterations) == "maximum iterations");
    CHECK(la::to_string(StopReason::Breakdown) == "breakdown");
    CHECK(la::to_string(ResidualReference::FirstIteration) == "first_iteration");
    CHECK(la::to_string(ResidualReference::InitialResidual) == "initial");
    CHECK(la::to_string(ResidualReference::RightHandSide) == "rhs");
}

TEST_CASE("ConvergenceControl reports every step to its monitor", "[linalg][convergence][monitor]") {
    EventLog log;
    ConvergenceControl c(criteria(0.1, 0.0), &log, "test");
    run(c, 1.0, {0.5, 0.05});
    CHECK(log.events == std::vector<std::string>{"start", "iter1", "iter2", "finish"});
    CHECK(c.iteration() == 2);
}

TEST_CASE("StagnationGuard stops after repeated false convergence without progress", "[linalg][convergence]") {
    la::StagnationGuard g;
    CHECK_FALSE(g.stalled(1e-10)); // first false convergence: nothing to compare
    CHECK_FALSE(g.stalled(0.4e-10)); // halved: progress, counter reset
    CHECK_FALSE(g.stalled(0.39e-10)); // stall 1
    CHECK(g.stalled(0.38e-10));       // stall 2: stop

    SECTION("progress in between resets the count") {
        la::StagnationGuard h;
        CHECK_FALSE(h.stalled(1.0));
        CHECK_FALSE(h.stalled(0.9)); // stall 1
        CHECK_FALSE(h.stalled(0.3)); // progress
        CHECK_FALSE(h.stalled(0.29)); // stall 1
        CHECK(h.stalled(0.28));
    }
    SECTION("reset") {
        g.reset();
        CHECK_FALSE(g.stalled(1.0));
        CHECK_FALSE(g.stalled(1.0));
        CHECK(g.stalled(1.0));
    }
    SECTION("a growing residual counts as a stall, NaN too") {
        la::StagnationGuard h;
        CHECK_FALSE(h.stalled(1.0));
        CHECK_FALSE(h.stalled(2.0));
        CHECK(h.stalled(std::numeric_limits<double>::quiet_NaN()));
    }
}
