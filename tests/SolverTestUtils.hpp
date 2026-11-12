#pragma once

// Shared harness for end-to-end solver tests (test_solver_regression.cpp,
// test_solver_turbulence.cpp): each test copies the bundled Hinterkante case
// into a private temporary directory, edits a few control values, runs
// Solver::initialize + Solver::run with captured output and reads back the
// files the solver writes (results/*.var, results/*.turb.var, results/*.vtk)
// and its log.

#include "BoundaryConditions.hpp"
#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "Solver.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <functional>
#include <memory>
#include <cctype>
#include <sstream>
#include <set>
#include <string>
#include <vector>

namespace hydrama::test {

// Non-empty line indices of the *.control / *.solver value lines.
inline constexpr int kRestartOpt = 2, kNtime = 5, kTransient = 8, kWriteOutput = 17, kDtfixed = 23, kIwrite = 29;
inline constexpr int kVelCheck = 38, kL2Norm = 41; // non-empty line indices of *.control
inline constexpr int kRelToler = 6, kAbsToler = 9;

inline constexpr double kInflowSpeed = 1.0;

struct StepLog {
    int step = 0;
    double initial = 0.0, final = 0.0;
    int iterations = 0;
    int cumulative = -1; // transient log only
    double time = -1.0;  // transient log only
};

struct Run {
    bool initialized = false;
    std::string errors;              // captured stderr
    bool turb_read = false;          // results/hinterkante.turb.var read successfully
    TurbulenceRestartData turb;      // its contents
    std::string turb_error;
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
    std::string solver_options; // non-empty: replaces the optional "key = value" block of *.solver
    std::string turbulence; // non-empty: written to system/hinterkante.turbulence
};

// Keeps the ten positional (non-empty) lines of *.solver and replaces
// everything after them by `options`.
inline void replace_solver_options(const fs::path& file, const std::string& options) {
    std::string kept;
    int non_empty = 0;
    for (const auto& line : split_lines(read_text(file))) {
        if (non_empty == 10) break;
        kept += line + "\n";
        if (!is_blank(line)) ++non_empty;
    }
    write_text(file, kept + options);
}

inline void prepare_case(const fs::path& dir, const CaseOptions& o) {
    copy_benchmark_case(dir);
    set_nonempty_lines(dir / "system/hinterkante.control",
                       {{kNtime, std::to_string(o.steps)}, {kWriteOutput, std::to_string(o.steps)},
                        {kIwrite, "1"}, {kTransient, std::to_string(o.transient)},
                        {kDtfixed, std::to_string(o.dtfixed)}});
    if (!o.rel_tol.empty())
        set_nonempty_lines(dir / "system/hinterkante.solver", {{kRelToler, o.rel_tol}, {kAbsToler, o.abs_tol}});
    if (!o.solver_options.empty()) replace_solver_options(dir / "system/hinterkante.solver", o.solver_options);
    if (!o.turbulence.empty()) write_text(dir / "system/hinterkante.turbulence", o.turbulence);
}

// Optional turbulence model injected into the solver before initialize().
using ModelFactory = std::function<std::unique_ptr<TurbulenceModel>()>;

// Parses the per-step log lines of Solver::run (plain string parsing; std::regex
// triggers -Wmaybe-uninitialized false positives inside libstdc++ in GCC
// sanitizer builds):
//   transient: "<step> pInitial: <r> pFinal: <r> Iter: <n> IterCumulative: <n>"
//              followed by " Time: <t>"
//   steady:    " Time = <step>" followed by "... Initial residual = <r> Final residual = <r> Iter: <n>"
inline std::vector<StepLog> parse_log(const std::string& log) {
    auto starts_with = [](const std::string& s, const std::string& prefix) { return s.rfind(prefix, 0) == 0; };
    std::vector<StepLog> steps;
    int steady_current = -1;
    for (const auto& line : split_lines(log)) {
        std::istringstream in(line);
        std::string a, b, c, d, e, f, g, h, i;
        if (in >> a >> b >> c >> d >> e >> f >> g >> h >> i && b == "pInitial:" && d == "pFinal:" && f == "Iter:" &&
            h == "IterCumulative:") {
            StepLog s;
            s.step = std::stoi(a);
            s.initial = std::stod(c);
            s.final = std::stod(e);
            s.iterations = std::stoi(g);
            s.cumulative = std::stoi(i);
            steps.push_back(s);
        } else if (starts_with(line, " Time: ") && !steps.empty()) {
            steps.back().time = std::stod(line.substr(7));
        } else if (starts_with(line, " Time = ")) {
            steady_current = std::stoi(line.substr(8));
        } else if (const auto at = line.find("Initial residual = "); at != std::string::npos) {
            std::istringstream rest(line.substr(at));
            std::string w1, w2, eq1, initial, w3, w4, eq2, final, iter, n;
            rest >> w1 >> w2 >> eq1 >> initial >> w3 >> w4 >> eq2 >> final >> iter >> n;
            StepLog s;
            s.step = steady_current;
            s.initial = std::stod(initial);
            s.final = std::stod(final);
            s.iterations = std::stoi(n);
            steps.push_back(s);
        }
    }
    return steps;
}

// True if the text contains "nan" or "inf" (any case) as a word of its own,
// i.e. not inside a longer word such as "info".
inline bool contains_non_finite_token(const std::string& text) {
    std::string lower(text);
    for (char& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    auto letter = [&lower](std::size_t k) { return std::isalpha(static_cast<unsigned char>(lower[k])) != 0; };
    for (const std::string word : {"nan", "inf"}) {
        for (std::size_t at = lower.find(word); at != std::string::npos; at = lower.find(word, at + 1)) {
            const bool before = at > 0 && letter(at - 1);
            const bool after = at + word.size() < lower.size() && letter(at + word.size());
            if (!before && !after) return true;
        }
    }
    return false;
}

inline Run run_solver(const fs::path& dir, const ModelFactory& inject = {}) {
    Run r;
    {
        CaptureStdout capture;
        CaptureStderr errors; // failing-input tests print expected errors
        SimulationConfig config;
        config.casename = "hinterkante";
        config.case_dir = dir.string();
        Solver solver(config);
        if (inject) solver.set_turbulence_model(inject());
        r.initialized = solver.initialize();
        if (r.initialized) solver.run();
        r.log = capture.str();
        r.errors = errors.str();
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
    if (fs::exists(dir / "results/hinterkante.turb.var")) {
        r.turb_read = IO::read_turbulence_restart((dir / "results/hinterkante.turb.var").string(), npoin, r.turb,
                                                  r.turb_error);
    }
    return r;
}

inline Run run_case(const CaseOptions& o, const std::string& tag, const ModelFactory& inject = {}) {
    TempDir dir(tag);
    prepare_case(dir.path(), o);
    return run_solver(dir.path(), inject);
}

inline double max_abs(const std::vector<double>& a) {
    double m = 0.0;
    for (double x : a) m = std::max(m, std::abs(x));
    return m;
}

inline double l2(const std::vector<double>& a) {
    double s = 0.0;
    for (double x : a) s += x * x;
    return std::sqrt(s);
}

inline int count_non_finite(const std::vector<double>& a) {
    int n = 0;
    for (double x : a)
        if (!std::isfinite(x)) ++n;
    return n;
}

// Mesh with flags and wall nodes, for boundary condition checks (loaded once).
inline const Mesh& regression_mesh() {
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
inline void check_healthy_run(const Run& r, int steps, double rel_tol, double abs_tol) {
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
    CHECK_FALSE(contains_non_finite_token(r.log));

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
    const Mesh& m = regression_mesh();
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

} // namespace hydrama::test
