// End-to-end regression tests of the full CBS time loop (Solver::initialize +
// Solver::run) on the bundled Hinterkante benchmark case.
//
// Each test copies the case into a private temporary directory, shortens the
// run to a few time steps and captures the solver log. Only three control
// values are changed for the "benchmark configuration" runs:
//   ntime = N, write_output = N (write the final state), iwrite = 1 (log every
//   step). Everything else (transient, global dt, Re = 200, CFL 0.6, PCG
//   tolerances 0.1 / 1e-6, ...) is the bundled configuration.
//
// Solver state is private, so results are checked black-box through the files
// the solver writes (results/*.var, results/*.vtk) and its console log.
//
// Reference values: with the bundled loose PCG tolerance (10 % residual
// reduction per step) pressure is only determined to within solver tolerance,
// and round-off differences between compilers/CPUs (e.g. fused multiply-add on
// ARM) change it by ~1 %. Reference values are therefore checked
//   * tightly (1e-7) in a variant with a converged PCG (abs. tol. 1e-10), where
//     platform differences were measured at ~1e-11, and
//   * loosely (1e-3, velocity and time only) for the bundled configuration.

#include "SolverTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <set>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

TEST_CASE("Benchmark case: 10 transient steps stay finite, bounded and converged", "[solver][regression]") {
    // Catch2 re-enters the test case once per SECTION: run the solver only once.
    static const Run r = run_case({}, "bench");
    check_healthy_run(r, 10, 0.1, 1.0e-6);

    SECTION("physical time advances monotonically and matches the restart file") {
        double previous = 0.0;
        for (const auto& s : r.steps) {
            CAPTURE(s.step, s.time);
            CHECK(s.time > previous);
            previous = s.time;
        }
        CHECK_THAT(r.rtime, WithinRel(r.steps.back().time, 1e-9));
    }
    SECTION("cumulative iteration count is consistent") {
        int total = 0;
        for (const auto& s : r.steps) {
            total += s.iterations;
            CHECK(s.cumulative == total);
        }
    }
    SECTION("output files are written") {
        CHECK(r.result_files == std::set<std::string>{"hinterkante.var", "hinterkante_0.vtk"});
    }
    SECTION("reference values (velocity and time, platform-robust tolerance)") {
        // Recorded with this configuration; see the file header for tolerances.
        CHECK_THAT(l2(r.u), WithinRel(88.78570299323, 1e-3));
        CHECK_THAT(l2(r.v), WithinRel(11.71591359276, 1e-3));
        CHECK_THAT(max_abs(r.u), WithinRel(1.36580567, 1e-3));
        CHECK_THAT(r.rtime, WithinRel(5.131891261e-02, 1e-3));
    }
}

TEST_CASE("Benchmark case: VTK output matches the restart state", "[solver][regression]") {
    TempDir dir("bench-vtk");
    prepare_case(dir.path(), {});
    const Run r = run_solver(dir.path());
    REQUIRE(r.state_read);
    const VtkFile f = parse_vtk(dir / "results/hinterkante_0.vtk");
    REQUIRE(f.npointdata == 8956);
    CHECK(f.ncells == 17377);
    int mismatches = 0;
    auto close = [](double a, double b) { return std::abs(a - b) <= 2e-7 * std::abs(b) + 1e-30; };
    for (int i = 0; i < f.npointdata; ++i) {
        if (!close(f.velocity[i][0], r.u[i]) || !close(f.velocity[i][1], r.v[i]) || !close(f.pressure[i], r.p[i]))
            ++mismatches;
    }
    CHECK(mismatches == 0);
}

TEST_CASE("Benchmark case: results are deterministic", "[solver][regression]") {
    // Two independent runs of the same binary must agree bit for bit.
    TempDir a("det-a"), b("det-b");
    prepare_case(a.path(), {});
    prepare_case(b.path(), {});
    const Run ra = run_solver(a.path());
    const Run rb = run_solver(b.path());
    REQUIRE(ra.state_read);
    REQUIRE(rb.state_read);

    CHECK(read_text(a / "results/hinterkante.var") == read_text(b / "results/hinterkante.var"));
    CHECK(read_text(a / "results/hinterkante_0.vtk") == read_text(b / "results/hinterkante_0.vtk"));

    // Logs agree except for the wall-clock time and the case path.
    auto normalise = [](const std::string& log, const std::string& dir) {
        std::string out;
        for (const auto& line : split_lines(log)) {
            if (line.find("Time Taken") != std::string::npos) continue;
            std::string l = line;
            for (auto pos = l.find(dir); pos != std::string::npos; pos = l.find(dir)) l.replace(pos, dir.size(), "<case>");
            out += l + "\n";
        }
        return out;
    };
    CHECK(normalise(ra.log, a.path().string()) == normalise(rb.log, b.path().string()));
}

TEST_CASE("Converged-pressure variant reproduces reference values", "[solver][regression]") {
    // Bundled case with the PCG converged to an absolute residual of 1e-10
    // (relative criterion off), so results reflect the discretisation only.
    CaseOptions o;
    o.rel_tol = "0.0";
    o.abs_tol = "1.0d-10";
    const Run r = run_case(o, "tight");
    check_healthy_run(r, 10, 0.0, 1.0e-10);

    CHECK_THAT(l2(r.u), WithinRel(88.81503913006, 1e-7));
    CHECK_THAT(l2(r.v), WithinRel(11.74480735349, 1e-7));
    CHECK_THAT(l2(r.p), WithinRel(15.5788580752, 1e-7));
    CHECK_THAT(max_abs(r.u), WithinRel(1.36615318, 1e-7));
    CHECK_THAT(max_abs(r.v), WithinRel(0.730081568, 1e-7));
    CHECK_THAT(max_abs(r.p), WithinRel(2.42867969, 1e-7));
    CHECK_THAT(r.rtime, WithinRel(4.976913663e-02, 1e-7));
    // Iteration counts differ by +-1 per step between platforms.
    CHECK_THAT(static_cast<double>(r.steps.back().cumulative), WithinRel(5244.0, 0.02));
}

TEST_CASE("Restarting reproduces an uninterrupted run bit-for-bit", "[solver][regression][restart]") {
    // Bundled configuration: 5 steps, then a restart from the written state to
    // step 10, must give exactly the same state as 10 uninterrupted steps.
    // Restart files store 17 significant digits, so the state (u, v, p and
    // time) is restored exactly and nothing else carries over between steps.
    TempDir straight("restart-a"), split("restart-b");
    prepare_case(straight.path(), {});
    const Run full = run_solver(straight.path());

    CaseOptions first;
    first.steps = 5;
    prepare_case(split.path(), first);
    const Run part1 = run_solver(split.path());
    REQUIRE(part1.state_read);
    REQUIRE(part1.istart == 5);
    set_nonempty_lines(split / "system/hinterkante.control",
                       {{kRestartOpt, "1"}, {kNtime, "10"}, {kWriteOutput, "10"}});
    const Run part2 = run_solver(split.path());

    REQUIRE(full.state_read);
    REQUIRE(part2.state_read);
    CHECK(part2.log.find("Restarting using previously calculated values") != std::string::npos);
    CHECK(part2.istart == 10);

    // Identical final state, byte for byte.
    CHECK(read_text(straight / "results/hinterkante.var") == read_text(split / "results/hinterkante.var"));
    CHECK(read_text(straight / "results/hinterkante_0.vtk") == read_text(split / "results/hinterkante_0.vtk"));

    // Steps 6..10 of both runs are identical: same residuals, iterations and
    // physical time. (IterCumulative restarts from 0 after a restart.)
    REQUIRE(full.steps.size() == 10);
    REQUIRE(part2.steps.size() == 5);
    for (size_t k = 0; k < 5; ++k) {
        const StepLog& a = full.steps[5 + k];
        const StepLog& b = part2.steps[k];
        CAPTURE(a.step, b.step);
        CHECK(b.step == a.step);
        CHECK(b.initial == a.initial);
        CHECK(b.final == a.final);
        CHECK(b.iterations == a.iterations);
        CHECK(b.time == a.time);
    }
}

TEST_CASE("Steady-state mode with local time stepping", "[solver][regression]") {
    // transient_on = 0, dtfixed = 0: the other branch of Solver::run(),
    // with per-element time steps and a final output named by step number.
    CaseOptions o;
    o.transient = 0;
    o.dtfixed = 0;
    const Run r = run_case(o, "steady");
    check_healthy_run(r, 10, 0.1, 1.0e-6);
    CHECK(r.rtime == 0.0); // steady restart files carry no physical time
    CHECK(r.result_files == std::set<std::string>{"hinterkante.var", "hinterkante_10.vtk"});
    CHECK(r.log.find("Outputting Data") != std::string::npos);
}

TEST_CASE("Solver::initialize fails cleanly on missing input", "[solver]") {
    SECTION("missing case directory") {
        TempDir dir("missing");
        const Run r = run_solver(dir / "no_such_case");
        CHECK_FALSE(r.initialized);
        CHECK(r.steps.empty());
    }
    SECTION("missing input file") {
        const std::string file = GENERATE(std::string("system/hinterkante.control"), std::string("system/hinterkante.physical"),
                                          std::string("system/hinterkante.solver"), std::string("constant/hinterkante.mesh"),
                                          std::string("constant/hinterkante.bofl"));
        CAPTURE(file);
        TempDir dir("missing-file");
        copy_benchmark_case(dir.path());
        fs::remove(dir / file);
        const Run r = run_solver(dir.path());
        CHECK_FALSE(r.initialized);
        CHECK_FALSE(r.state_read);
    }
}

TEST_CASE("Solver::initialize creates the results directory even when it fails",
          "[solver][characterization]") {
    // Documents current behaviour: results/ is created before any input is
    // read, so pointing the solver at a wrong path leaves an empty
    // <path>/results directory behind.
    TempDir dir("side-effect");
    const fs::path wrong = dir / "typo_case";
    const Run r = run_solver(wrong);
    CHECK_FALSE(r.initialized);
    CHECK(fs::is_directory(wrong / "results"));
}

TEST_CASE("Every linear solver configuration reproduces the converged reference values",
          "[solver][regression][configurations]") {
    // The converged-pressure variant with other operators/preconditioners/
    // methods: the flow must match the reference values of the legacy PCG
    // to the same tight tolerance.
    //
    // Tolerances: an absolute residual of 1e-10 on A = dt K only bounds the
    // pressure error by ||A^{-1}|| 1e-10 ~ 2e-4 (lambda_min(dt K) ~ 5e-7).
    // CG minimises the error in the energy norm and lands within 1e-8 of the
    // converged pressure; the residual-minimising BiCGSTAB/GMRES stop
    // elsewhere in that band (~1e-6 in ||p||). They are therefore run to
    // 1e-12 (all methods within 5e-9 of the converged pressure); 1e-13 is
    // below the attainable accuracy of some steps (true residual floor
    // ~5e-13). GMRES uses a long restart: on this ill-conditioned SPD
    // system short restarts (m = 20, 30) stall near 1e-12 and end in
    // StopReason::Stagnation (restarting discards the Krylov space).
    struct Configuration {
        std::string options;
        std::string abs_tol;
    };
    const Configuration c = GENERATE(
        Configuration{"operator_format = csr\n", "1.0d-10"},
        Configuration{"preconditioner = ssor\nreordering = rcm\n", "1.0d-10"},
        Configuration{"preconditioner = ic0\n", "1.0d-10"},
        Configuration{"preconditioner = ic0\nreordering = rcm\noperator_format = csr\n", "1.0d-10"},
        Configuration{"preconditioner = ilu0\nreordering = rcm\n", "1.0d-10"},
        Configuration{"method = bicgstab\npreconditioner = ilu0\nreordering = rcm\n", "1.0d-12"},
        Configuration{"method = gmres\npreconditioner = ic0\nreordering = rcm\ngmres_restart = 200\n", "1.0d-12"},
        Configuration{"method = gmres\ngmres_orthogonalization = cgs2\noperator_format = csr\n"
                      "preconditioner = ilu0\nreordering = rcm\ngmres_restart = 200\n",
                      "1.0d-12"});
    CAPTURE(c.options, c.abs_tol);
    CaseOptions o;
    o.rel_tol = "0.0";
    o.abs_tol = c.abs_tol;
    o.solver_options = c.options;
    const Run r = run_case(o, "configurations");
    check_healthy_run(r, 10, 0.0, c.abs_tol == "1.0d-10" ? 1.0e-10 : 1.0e-12);
    CHECK(r.log.find("Pressure solver: ") != std::string::npos);

    CHECK_THAT(l2(r.u), WithinRel(88.81503913006, 1e-7));
    CHECK_THAT(l2(r.v), WithinRel(11.74480735349, 1e-7));
    CHECK_THAT(l2(r.p), WithinRel(15.5788580752, 1e-7));
    CHECK_THAT(max_abs(r.u), WithinRel(1.36615318, 1e-7));
    CHECK_THAT(max_abs(r.v), WithinRel(0.730081568, 1e-7));
    CHECK_THAT(max_abs(r.p), WithinRel(2.42867969, 1e-7));
    CHECK_THAT(r.rtime, WithinRel(4.976913663e-02, 1e-7));
}

TEST_CASE("Solver log reports the pressure solver configuration", "[solver][configurations]") {
    CaseOptions o;
    o.steps = 2;
    o.solver_options = "preconditioner = ic0\nreordering = rcm\noperator_format = csr\n";
    const Run r = run_case(o, "config-log");
    REQUIRE(r.initialized);
    CHECK(r.log.find("Pressure solver: PCG (CG + rcm+ic0, csr)") != std::string::npos);

    CaseOptions bundled;
    bundled.steps = 2;
    const Run legacy = run_case(bundled, "config-log-legacy");
    CHECK(legacy.log.find("Pressure solver: PCG (CG + jacobi, ebe)") != std::string::npos);
}

TEST_CASE("Solver::initialize fails cleanly on invalid solver options", "[solver][configurations]") {
    CaseOptions o;
    o.solver_options = "preconditioner = amg\n";
    const Run r = run_case(o, "config-invalid");
    CHECK_FALSE(r.initialized);
    CHECK_FALSE(r.state_read);
}

TEST_CASE("Steady mode logs the Krylov method of the pressure solve", "[solver][configurations]") {
    CaseOptions o;
    o.transient = 0;
    o.dtfixed = 0;
    o.solver_options = "method = gmres\npreconditioner = ilu0\nreordering = rcm\n";
    const Run r = run_case(o, "steady-gmres");
    check_healthy_run(r, 10, 0.1, 1.0e-6);
    CHECK(r.log.find(" GMRES(30): Solving for p, Initial residual = ") != std::string::npos);
    CHECK(r.log.find("Pressure solver: Krylov (GMRES(30) + rcm+ilu0, ebe)") != std::string::npos);
}

TEST_CASE("Global time stepping factorises the preconditioner once per run", "[solver][configurations][reuse]") {
    // Transient, global dt (changes every step): one numeric setup, then
    // rescaling. Steady local time stepping: a new setup every step.
    CaseOptions transient;
    transient.solver_options = "preconditioner = ic0\nreordering = rcm\n";
    const Run t = run_case(transient, "reuse-transient");
    check_healthy_run(t, 10, 0.1, 1.0e-6);
    CHECK(t.log.find("Pressure solver setup: 1 numeric setups, 9 rescaled reuses, 1 structure builds") !=
          std::string::npos);

    CaseOptions rebuild = transient;
    rebuild.solver_options += "preconditioner_reuse = rebuild\n";
    const Run b = run_case(rebuild, "reuse-rebuild");
    CHECK(b.log.find("Pressure solver setup: 10 numeric setups, 0 rescaled reuses, 1 structure builds") !=
          std::string::npos);

    CaseOptions steady = transient;
    steady.transient = 0;
    steady.dtfixed = 0;
    const Run s = run_case(steady, "reuse-steady");
    check_healthy_run(s, 10, 0.1, 1.0e-6);
    CHECK(s.log.find("Pressure solver setup: 10 numeric setups, 0 rescaled reuses, 1 structure builds") !=
          std::string::npos);

    // The legacy configuration prints no setup line (log unchanged).
    const Run legacy = run_case({}, "reuse-legacy");
    CHECK(legacy.log.find("Pressure solver setup") == std::string::npos);
}
