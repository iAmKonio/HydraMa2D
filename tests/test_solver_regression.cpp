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

#include "BoundaryConditions.hpp"
#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "Solver.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <regex>
#include <set>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Non-empty line indices of the *.control / *.solver value lines.
constexpr int kRestartOpt = 2, kNtime = 5, kTransient = 8, kWriteOutput = 17, kDtfixed = 23, kIwrite = 29;
constexpr int kRelToler = 6, kAbsToler = 9;

constexpr double kInflowSpeed = 1.0;

struct StepLog {
    int step = 0;
    double initial = 0.0, final = 0.0;
    int iterations = 0;
    int cumulative = -1; // transient log only
    double time = -1.0;  // transient log only
};

struct Run {
    bool initialized = false;
    std::string log;
    std::vector<StepLog> steps;
    bool state_read = false;
    double rtime = -1.0;
    int istart = -1;
    std::vector<double> u, v, p;
    std::set<std::string> result_files;
};

struct CaseOptions {
    int steps = 10;
    int transient = 1;     // bundled: 1
    int dtfixed = -1;      // bundled: -1
    std::string rel_tol;   // empty: bundled 0.1
    std::string abs_tol;   // empty: bundled 1.0d-06
};

void prepare_case(const fs::path& dir, const CaseOptions& o) {
    copy_benchmark_case(dir);
    set_nonempty_lines(dir / "system/hinterkante.control",
                       {{kNtime, std::to_string(o.steps)}, {kWriteOutput, std::to_string(o.steps)},
                        {kIwrite, "1"}, {kTransient, std::to_string(o.transient)},
                        {kDtfixed, std::to_string(o.dtfixed)}});
    if (!o.rel_tol.empty())
        set_nonempty_lines(dir / "system/hinterkante.solver", {{kRelToler, o.rel_tol}, {kAbsToler, o.abs_tol}});
}

std::vector<StepLog> parse_log(const std::string& log) {
    static const std::regex transient(R"(^\s*(\d+) pInitial:\s*(\S+) pFinal:\s*(\S+) Iter:\s*(\d+) IterCumulative:\s*(\d+))");
    static const std::regex steady_step(R"(^ Time = (\d+))");
    static const std::regex steady_pcg(R"(Initial residual = (\S+) Final residual = (\S+) Iter: (\d+))");
    static const std::regex time_line(R"(^ Time: (\S+))");
    std::vector<StepLog> steps;
    int steady_current = -1;
    for (const auto& line : split_lines(log)) {
        std::smatch m;
        if (std::regex_search(line, m, transient)) {
            StepLog s;
            s.step = std::stoi(m[1]);
            s.initial = std::stod(m[2]);
            s.final = std::stod(m[3]);
            s.iterations = std::stoi(m[4]);
            s.cumulative = std::stoi(m[5]);
            steps.push_back(s);
        } else if (std::regex_search(line, m, time_line) && !steps.empty()) {
            steps.back().time = std::stod(m[1]);
        } else if (std::regex_search(line, m, steady_step)) {
            steady_current = std::stoi(m[1]);
        } else if (std::regex_search(line, m, steady_pcg)) {
            StepLog s;
            s.step = steady_current;
            s.initial = std::stod(m[1]);
            s.final = std::stod(m[2]);
            s.iterations = std::stoi(m[3]);
            steps.push_back(s);
        }
    }
    return steps;
}

Run run_solver(const fs::path& dir) {
    Run r;
    {
        CaptureStdout capture;
        CaptureStderr errors; // failing-input tests print expected errors
        SimulationConfig config;
        config.casename = "hinterkante";
        config.case_dir = dir.string();
        Solver solver(config);
        r.initialized = solver.initialize();
        if (r.initialized) solver.run();
        r.log = capture.str();
    }
    r.steps = parse_log(r.log);
    if (fs::exists(dir / "results")) {
        for (const auto& e : fs::directory_iterator(dir / "results")) r.result_files.insert(e.path().filename().string());
    }
    const int npoin = 8956;
    r.u.assign(npoin, 0.0);
    r.v.assign(npoin, 0.0);
    r.p.assign(npoin, 0.0);
    r.state_read = IO::read_restart((dir / "results/hinterkante.var").string(), r.rtime, r.istart, r.u, r.v, r.p);
    return r;
}

Run run_case(const CaseOptions& o, const std::string& tag) {
    TempDir dir(tag);
    prepare_case(dir.path(), o);
    return run_solver(dir.path());
}

double max_abs(const std::vector<double>& a) {
    double m = 0.0;
    for (double x : a) m = std::max(m, std::abs(x));
    return m;
}

double l2(const std::vector<double>& a) {
    double s = 0.0;
    for (double x : a) s += x * x;
    return std::sqrt(s);
}

int count_non_finite(const std::vector<double>& a) {
    int n = 0;
    for (double x : a)
        if (!std::isfinite(x)) ++n;
    return n;
}

// Mesh with flags and wall nodes, for boundary condition checks (loaded once).
const Mesh& benchmark_mesh() {
    static const Mesh mesh = [] {
        Mesh m;
        CaptureStdout quiet;
        REQUIRE(IO::read_mesh(std::string(HYDRAMA2D_CASE_DIR) + "/constant/hinterkante.mesh", m));
        REQUIRE(IO::read_boundary_flags(std::string(HYDRAMA2D_CASE_DIR) + "/constant/hinterkante.bofl", m));
        m.compute_shape_function_derivatives();
        m.compute_face_normals();
        m.determine_wall_nodes();
        return m;
    }();
    return mesh;
}

// Checks shared by every completed run.
void check_healthy_run(const Run& r, int steps, double rel_tol, double abs_tol) {
    REQUIRE(r.initialized);
    REQUIRE(r.state_read); // fails for missing files and for "nan"/"inf" values
    CHECK(r.istart == steps);

    // Free of NaN/Inf, bounded by the flow scales (inflow speed U = 1).
    CHECK(count_non_finite(r.u) == 0);
    CHECK(count_non_finite(r.v) == 0);
    CHECK(count_non_finite(r.p) == 0);
    CHECK(max_abs(r.u) <= 5.0 * kInflowSpeed);
    CHECK(max_abs(r.v) <= 5.0 * kInflowSpeed);
    CHECK(max_abs(r.p) <= 10.0 * kInflowSpeed * kInflowSpeed);
    static const std::regex non_finite_token(R"((^|[^A-Za-z])-?(nan|inf)([^A-Za-z]|$))", std::regex::icase);
    CHECK_FALSE(std::regex_search(r.log, non_finite_token));

    // One converged pressure solve per step.
    REQUIRE(static_cast<int>(r.steps.size()) == steps);
    for (size_t k = 0; k < r.steps.size(); ++k) {
        const StepLog& s = r.steps[k];
        CAPTURE(s.step, s.initial, s.final, s.iterations);
        CHECK(s.step == static_cast<int>(k) + 1);
        CHECK(std::isfinite(s.initial));
        CHECK(std::isfinite(s.final));
        CHECK(s.iterations >= 1);
        CHECK(s.iterations < 100000); // stopped by a tolerance, not the cap
        CHECK((s.final < rel_tol * s.initial || s.final < abs_tol));
    }

    // Boundary conditions hold in the written state.
    const Mesh& m = benchmark_mesh();
    const std::set<int> walls(m.wall_node_list.begin(), m.wall_node_list.end());
    int wall_violations = 0, inflow_violations = 0, outlet_violations = 0;
    for (int i : walls)
        if (r.u[i] != 0.0 || r.v[i] != 0.0) ++wall_violations;
    for (const auto& side : m.iside) {
        for (int i : {side.node1, side.node2}) {
            if (side.flag == 503 && !walls.count(i) && (r.u[i] != 1.0 || r.v[i] != 0.0)) ++inflow_violations;
            if (side.flag == 504 && r.p[i] != 0.0) ++outlet_violations;
        }
    }
    CHECK(wall_violations == 0);
    CHECK(inflow_violations == 0);
    CHECK(outlet_violations == 0);
}

} // namespace

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
