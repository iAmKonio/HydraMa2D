// End-to-end tests of the turbulence coupling in the full CBS time loop on
// the bundled Hinterkante benchmark (see SolverTestUtils.hpp):
//
// Laminar bit-identity
//   * a .turbulence file with "model none" gives byte-identical results to
//     no file at all;
//   * the complete turbulent code path (Step 1 with nu_t, time step with
//     nu_eff, model advance, turbulence restart, extra VTK fields) driven by a
//     zero-eddy-viscosity model reproduces the laminar u, v, p bit for bit,
//     transient and steady (local time stepping).
//
// Two-equation models (SST, k-epsilon with wall functions, Launder-Sharma):
//   health, wall / inlet values, VTK fields, bit-for-bit restart, steady
//   mode and reference values.
//
// Spalart-Allmaras regression
//   * healthy runs (finite, bounded, BCs, converged PCG, nu~ BCs),
//   * VTK turbulence fields consistent with the restart state,
//   * reference values, determinism, bit-for-bit restart (incl. the
//     turbulence restart file), restart from a laminar solution, mismatched
//     turbulence restart files, steady mode, invalid configurations.

#include "SolverTestUtils.hpp"
#include "WallTestUtils.hpp"
#include "turbulence/SpalartAllmaras.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <iterator>
#include <cmath>
#include <memory>
#include <set>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kNu = 1.0 / 200.0; // bundled Re = 200
const std::string kSaConfig = "model spalart-allmaras\n";

// Test double: a turbulence model with nu_t = 0 and no transported
// diffusivity. Drives the turbulent code path without changing the flow.
class ZeroModel final : public TurbulenceModel {
public:
    explicit ZeroModel(std::shared_ptr<int> advances) : advances_(std::move(advances)) {}
    std::string_view name() const override { return "zero-model"; }
    void initialize(const TurbulenceContext& ctx, const FlowView&) override {
        zero_.assign(ctx.mesh.npoin, 0.0);
        nut_.assign(ctx.mesh.npoin, 0.0);
        nut_elem_.assign(ctx.mesh.nelem, 0.0);
        gamma_.assign(ctx.mesh.nelem, 0.0);
    }
    void advance(const TurbulenceContext&, const FlowView&) override { ++*advances_; }
    const std::vector<double>& nut() const override { return nut_; }
    const std::vector<double>& nut_elem() const override { return nut_elem_; }
    const std::vector<double>& max_diffusivity_elem() const override { return gamma_; }
    std::vector<FieldRef> fields() override { return {{"zero", &zero_}, {"nut", &nut_}}; }
    std::vector<FieldRef> state_fields() override { return {{"zero", &zero_}}; }
    void refresh(const TurbulenceContext&) override {}
    std::vector<EquationResidual> residuals() const override { return {{"zero", 0.0}}; }

private:
    std::shared_ptr<int> advances_;
    std::vector<double> zero_, nut_, nut_elem_, gamma_;
};

std::vector<std::string> turbulence_lines(const std::string& log) {
    std::vector<std::string> out;
    for (const auto& line : split_lines(log))
        if (line.rfind(" Turbulence:", 0) == 0) out.push_back(line);
    return out;
}

std::string normalise_log(const std::string& log, const std::string& dir) {
    std::string out;
    for (const auto& line : split_lines(log)) {
        if (line.find("Time Taken") != std::string::npos) continue;
        std::string l = line;
        for (auto pos = l.find(dir); pos != std::string::npos; pos = l.find(dir)) l.replace(pos, dir.size(), "<case>");
        out += l + "\n";
    }
    return out;
}

// SA-specific checks of the written turbulence state. min_fraction bounds
// negative nu~ relative to the inflow value 3 nu (SA-neg allows transient
// negative excursions; see the steady start-up test).
void check_sa_state(const Run& r, double min_fraction = 0.05) {
    REQUIRE(r.turb_read);
    CHECK(r.turb.model == "spalart-allmaras");
    REQUIRE(r.turb.names == std::vector<std::string>{"nu_tilde"});
    const std::vector<double>& nt = r.turb.values[0];
    CHECK(count_non_finite(nt) == 0);

    const Mesh& m = regression_mesh();
    const std::set<int> walls(m.wall_node_list.begin(), m.wall_node_list.end());
    int wall_violations = 0, inlet_violations = 0;
    for (int i : walls)
        if (nt[i] != 0.0) ++wall_violations;
    for (const auto& side : m.iside)
        for (int i : {side.node1, side.node2})
            if (side.flag == 503 && !walls.count(i) && nt[i] != 3.0 * kNu) ++inlet_violations;
    CHECK(wall_violations == 0);
    CHECK(inlet_violations == 0);

    const double mx = *std::max_element(nt.begin(), nt.end());
    const double mn = *std::min_element(nt.begin(), nt.end());
    CHECK(mx <= 10.0 * kNu);     // bounded (laminar Re: no strong growth)
    CHECK(mn > -min_fraction * 3.0 * kNu); // small negative excursions only (SA-neg)
}

} // namespace

// =============================================================================
// Laminar bit-identity
// =============================================================================

TEST_CASE("A .turbulence file with model none is byte-identical to a laminar run", "[solver][turbulence][laminar]") {
    TempDir plain("lam-plain"), none("lam-none");
    prepare_case(plain.path(), {});
    CaseOptions o;
    o.turbulence = "# explicitly laminar\nmodel none\n";
    prepare_case(none.path(), o);
    const Run a = run_solver(plain.path());
    const Run b = run_solver(none.path());
    REQUIRE(a.state_read);
    REQUIRE(b.state_read);
    CHECK(read_text(plain / "results/hinterkante.var") == read_text(none / "results/hinterkante.var"));
    CHECK(read_text(plain / "results/hinterkante_0.vtk") == read_text(none / "results/hinterkante_0.vtk"));
    CHECK(normalise_log(a.log, plain.path().string()) == normalise_log(b.log, none.path().string()));
    CHECK(b.result_files == std::set<std::string>{"hinterkante.var", "hinterkante_0.vtk"}); // no .turb.var
}

TEST_CASE("The turbulent code path with zero eddy viscosity reproduces the laminar solution bit for bit",
          "[solver][turbulence][laminar]") {
    const bool steady = GENERATE(false, true);
    CAPTURE(steady);
    CaseOptions o;
    if (steady) {
        o.transient = 0;
        o.dtfixed = 0; // local time stepping: the nu_eff time step per element
    }
    const std::string vtk = steady ? "hinterkante_10.vtk" : "hinterkante_0.vtk";

    TempDir lam("zero-lam"), turb("zero-turb");
    prepare_case(lam.path(), o);
    prepare_case(turb.path(), o);
    const Run a = run_solver(lam.path());
    auto advances = std::make_shared<int>(0);
    const Run b = run_solver(turb.path(), [advances] { return std::make_unique<ZeroModel>(advances); });
    REQUIRE(a.state_read);
    REQUIRE(b.state_read);

    // The turbulent path was really taken ...
    CHECK(*advances == 10);
    CHECK_THAT(b.log, ContainsSubstring("Turbulence model: zero-model"));
    CHECK(turbulence_lines(b.log).size() == 10u); // iwrite = 1: logged every step
    REQUIRE(b.turb_read);
    CHECK(b.turb.model == "zero-model");

    // ... and the flow is bit-identical: restart file, PCG history, VTK u, v, p.
    CHECK(read_text(lam / "results/hinterkante.var") == read_text(turb / "results/hinterkante.var"));
    REQUIRE(a.steps.size() == b.steps.size());
    for (std::size_t k = 0; k < a.steps.size(); ++k) {
        CHECK(a.steps[k].initial == b.steps[k].initial);
        CHECK(a.steps[k].final == b.steps[k].final);
        CHECK(a.steps[k].iterations == b.steps[k].iterations);
    }
    const VtkFile fa = parse_vtk(lam / ("results/" + vtk));
    const VtkFile fb = parse_vtk(turb / ("results/" + vtk));
    CHECK(fa.velocity == fb.velocity);
    CHECK(fa.pressure == fb.pressure);
    CHECK(fa.extra_scalars.empty());
    REQUIRE(fb.extra_scalars.size() == 4);
    CHECK(fb.extra_scalars[0].first == "zero");
    CHECK(fb.extra_scalars[1].first == "nut");
    CHECK(fb.extra_scalars[2].first == "wall_distance");
    CHECK(fb.extra_scalars[3].first == "yplus");
}

// =============================================================================
// Spalart-Allmaras on the benchmark
// =============================================================================

TEST_CASE("SA benchmark: 10 transient steps stay finite, bounded and converged", "[solver][turbulence][regression]") {
    static const Run r = [] {
        CaseOptions o;
        o.turbulence = kSaConfig;
        return run_case(o, "sa-bench");
    }();
    check_healthy_run(r, 10, 0.1, 1.0e-6);
    check_sa_state(r);

    SECTION("log and files") {
        CHECK_THAT(r.log, ContainsSubstring("Turbulence model: spalart-allmaras"));
        CHECK_THAT(r.log, ContainsSubstring("Wall faces: 380, wall nodes: 383"));
        const auto lines = turbulence_lines(r.log);
        REQUIRE(lines.size() == 10);
        CHECK_THAT(lines.back(), ContainsSubstring("nu_tilde change"));
        CHECK_THAT(lines.back(), ContainsSubstring("y+ (first cell) min"));
        CHECK(r.result_files ==
              std::set<std::string>{"hinterkante.var", "hinterkante.turb.var", "hinterkante_0.vtk"});
    }

    SECTION("reference values (platform-robust tolerance)") {
        // Recorded with this configuration (Linux x86-64, GCC); see
        // test_solver_regression.cpp for the tolerance policy.
        CHECK_THAT(l2(r.u), WithinRel(89.1141033943, 1e-3));
        CHECK_THAT(l2(r.v), WithinRel(12.0128160519, 1e-3));
        CHECK_THAT(r.rtime, WithinRel(1.48608710663e-02, 1e-3));
        CHECK_THAT(l2(r.turb.values[0]), WithinRel(1.33734748663, 1e-3));
    }

    SECTION("the time step includes the SA diffusivity") {
        // Laminar: 10 steps reach t = 5.13e-2; SA diffusion (nu + nu~)/sigma = 6 nu
        // in the free stream limits the step.
        CHECK(r.rtime < 0.5 * 5.131891261e-02);
    }
}

TEST_CASE("SA benchmark: VTK turbulence fields match the restart state", "[solver][turbulence][regression]") {
    TempDir dir("sa-vtk");
    CaseOptions o;
    o.turbulence = kSaConfig;
    prepare_case(dir.path(), o);
    const Run r = run_solver(dir.path());
    REQUIRE(r.turb_read);
    const VtkFile f = parse_vtk(dir / "results/hinterkante_0.vtk");
    REQUIRE(f.npointdata == 8956);
    REQUIRE(f.extra_scalars.size() == 4);
    CHECK(f.extra_scalars[0].first == "nu_tilde");
    CHECK(f.extra_scalars[1].first == "nut");
    CHECK(f.extra_scalars[2].first == "wall_distance");
    CHECK(f.extra_scalars[3].first == "yplus");

    Mesh m = benchmark_mesh();
    const WallGeometry walls(m);
    const std::vector<double>& nt = r.turb.values[0];
    auto close = [](double a, double b) { return std::abs(a - b) <= 2e-7 * std::abs(b) + 1e-30; };
    int bad = 0;
    for (int i = 0; i < f.npointdata; ++i) {
        if (!close((*f.scalar("nu_tilde"))[i], nt[i])) ++bad;
        if (!close((*f.scalar("nut"))[i], sa::eddy_viscosity(nt[i], kNu))) ++bad;
        if (!close((*f.scalar("wall_distance"))[i], walls.wall_distance()[i])) ++bad;
        const double yp = (*f.scalar("yplus"))[i];
        if (!(yp >= 0.0) || (walls.is_wall_node(i) && yp != 0.0)) ++bad;
    }
    CHECK(bad == 0);
    CHECK(*std::max_element(f.scalar("yplus")->begin(), f.scalar("yplus")->end()) > 0.0);
}

TEST_CASE("SA benchmark: converged-pressure variant reproduces reference values", "[solver][turbulence][regression]") {
    CaseOptions o;
    o.turbulence = kSaConfig;
    o.rel_tol = "0.0";
    o.abs_tol = "1.0d-10";
    const Run r = run_case(o, "sa-tight");
    check_healthy_run(r, 10, 0.0, 1.0e-10);
    check_sa_state(r);
    const std::vector<double>& nt = r.turb.values[0];
    CHECK_THAT(l2(r.u), WithinRel(89.0998372396, 1e-7));
    CHECK_THAT(l2(r.v), WithinRel(12.0148775404, 1e-7));
    CHECK_THAT(l2(r.p), WithinRel(30.9388883667, 1e-7));
    CHECK_THAT(r.rtime, WithinRel(1.48598743585e-02, 1e-7));
    CHECK_THAT(l2(nt), WithinRel(1.33735115159, 1e-7));
    CHECK_THAT(*std::max_element(nt.begin(), nt.end()), WithinRel(1.50204823841e-02, 1e-7));
    // Iteration counts differ by +-1 per step between platforms.
    CHECK_THAT(static_cast<double>(r.steps.back().cumulative), WithinRel(5227.0, 0.02));
}

TEST_CASE("SA benchmark: results are deterministic", "[solver][turbulence][regression]") {
    TempDir a("sa-det-a"), b("sa-det-b");
    CaseOptions o;
    o.turbulence = kSaConfig;
    prepare_case(a.path(), o);
    prepare_case(b.path(), o);
    const Run ra = run_solver(a.path());
    const Run rb = run_solver(b.path());
    REQUIRE(ra.turb_read);
    REQUIRE(rb.turb_read);
    for (const char* file : {"results/hinterkante.var", "results/hinterkante.turb.var", "results/hinterkante_0.vtk"}) {
        CAPTURE(file);
        CHECK(read_text(a / file) == read_text(b / file));
    }
    CHECK(normalise_log(ra.log, a.path().string()) == normalise_log(rb.log, b.path().string()));
}

TEST_CASE("SA benchmark: restarting reproduces an uninterrupted run bit for bit",
          "[solver][turbulence][regression][restart]") {
    TempDir straight("sa-restart-a"), split("sa-restart-b");
    CaseOptions o;
    o.turbulence = kSaConfig;
    prepare_case(straight.path(), o);
    const Run full = run_solver(straight.path());

    CaseOptions first = o;
    first.steps = 5;
    prepare_case(split.path(), first);
    const Run part1 = run_solver(split.path());
    REQUIRE(part1.turb_read);
    set_nonempty_lines(split / "system/hinterkante.control", {{kRestartOpt, "1"}, {kNtime, "10"}, {kWriteOutput, "10"}});
    const Run part2 = run_solver(split.path());
    REQUIRE(part2.turb_read);
    CHECK_THAT(part2.log, ContainsSubstring("Restarting using previously calculated values"));
    CHECK_THAT(part2.log, ContainsSubstring("Restarting turbulence from"));
    CHECK(part2.istart == 10);

    for (const char* file : {"results/hinterkante.var", "results/hinterkante.turb.var", "results/hinterkante_0.vtk"}) {
        CAPTURE(file);
        CHECK(read_text(straight / file) == read_text(split / file));
    }
    // Steps 6..10: same pressure solves, time and turbulence residuals.
    REQUIRE(full.steps.size() == 10);
    REQUIRE(part2.steps.size() == 5);
    const auto full_turb = turbulence_lines(full.log), part_turb = turbulence_lines(part2.log);
    REQUIRE(full_turb.size() == 10);
    REQUIRE(part_turb.size() == 5);
    for (std::size_t k = 0; k < 5; ++k) {
        CHECK(full.steps[5 + k].initial == part2.steps[k].initial);
        CHECK(full.steps[5 + k].iterations == part2.steps[k].iterations);
        CHECK(full.steps[5 + k].time == part2.steps[k].time);
        CHECK(full_turb[5 + k] == part_turb[k]);
    }
}

TEST_CASE("SA benchmark: a turbulent run can restart from a laminar solution", "[solver][turbulence][restart]") {
    TempDir dir("sa-from-laminar");
    CaseOptions o;
    o.steps = 5;
    prepare_case(dir.path(), o); // laminar, no .turbulence file
    const Run lam = run_solver(dir.path());
    REQUIRE(lam.state_read);
    REQUIRE_FALSE(lam.turb_read);

    write_text(dir / "system/hinterkante.turbulence", kSaConfig);
    set_nonempty_lines(dir / "system/hinterkante.control", {{kRestartOpt, "1"}, {kNtime, "10"}, {kWriteOutput, "10"}});
    const Run r = run_solver(dir.path());
    CHECK_THAT(r.log, ContainsSubstring("WARNING: Turbulence restart not used"));
    CHECK_THAT(r.log, ContainsSubstring("Turbulence starts from free-stream values"));
    REQUIRE(r.state_read);
    CHECK(r.istart == 10);
    check_sa_state(r);
}

TEST_CASE("SA benchmark: mismatched turbulence restart files are not used", "[solver][turbulence][restart]") {
    TempDir dir("sa-mismatch");
    CaseOptions o;
    o.turbulence = kSaConfig;
    o.steps = 5;
    prepare_case(dir.path(), o);
    REQUIRE(run_solver(dir.path()).turb_read);

    struct Edit {
        std::string from, to, message;
    };
    const Edit edit = GENERATE(Edit{"model spalart-allmaras", "model k-omega-sst", "file is for model 'k-omega-sst'"},
                               Edit{"fields 1 nu_tilde", "fields 1 k", "field 'k' does not match 'nu_tilde'"},
                               Edit{"npoin 8956", "npoin 8957", "file has 8957 nodes, mesh has 8956"});
    const fs::path turb = dir / "results/hinterkante.turb.var";
    std::string text = read_text(turb);
    REQUIRE(text.find(edit.from) != std::string::npos);
    text.replace(text.find(edit.from), edit.from.size(), edit.to);
    write_text(turb, text);

    set_nonempty_lines(dir / "system/hinterkante.control", {{kRestartOpt, "1"}, {kNtime, "6"}, {kWriteOutput, "6"}});
    const Run r = run_solver(dir.path());
    CAPTURE(edit.message);
    CHECK_THAT(r.log, ContainsSubstring("WARNING: Turbulence restart not used (" + edit.message + ")"));
    REQUIRE(r.state_read);
    CHECK(r.istart == 6);
    check_sa_state(r); // continued from free-stream turbulence values
}

TEST_CASE("SA benchmark: steady mode with local time stepping", "[solver][turbulence][regression]") {
    CaseOptions o;
    o.turbulence = kSaConfig;
    o.transient = 0;
    o.dtfixed = 0;
    const Run r = run_case(o, "sa-steady");
    check_healthy_run(r, 10, 0.1, 1.0e-6);
    check_sa_state(r, 0.2); // start-up transient with large local time steps
    CHECK(r.result_files == std::set<std::string>{"hinterkante.var", "hinterkante.turb.var", "hinterkante_10.vtk"});
    CHECK(turbulence_lines(r.log).size() == 10); // iwrite = 1: logged every step
}

TEST_CASE("SA benchmark: steady start-up negative excursions of nu~ vanish", "[solver][turbulence][regression]") {
    // With local time stepping the first iterations produce small negative
    // nu~ near the channel walls (measured: -0.3 nu after 10 iterations);
    // the negative-SA branch drives them back to zero.
    CaseOptions o;
    o.turbulence = kSaConfig;
    o.transient = 0;
    o.dtfixed = 0;
    o.steps = 50;
    const Run r = run_case(o, "sa-steady-50");
    check_healthy_run(r, 50, 0.1, 1.0e-6);
    check_sa_state(r, 1e-3);
}

TEST_CASE("Turbulence configuration errors fail initialization cleanly", "[solver][turbulence]") {
    struct Bad {
        std::string config;
        std::string message;
    };
    const Bad bad = GENERATE(Bad{"model spalart-allmaras\nturbulence_intensity 0.1\n", "unknown key"},
                             Bad{"model rng-k-epsilon\n", "is not implemented yet"},
                             Bad{"model k-epsilon\nwall_treatment low-re\n", "use launder-sharma-k-epsilon"},
                             Bad{"model spalart-allmaras\nwall_treatment wall-function\n", "low-re only"});
    TempDir dir("turb-bad");
    CaseOptions o;
    o.turbulence = bad.config;
    prepare_case(dir.path(), o);
    const Run r = run_solver(dir.path());
    CAPTURE(bad.config);
    CHECK_FALSE(r.initialized);
    CHECK_FALSE(r.state_read);
    CHECK_THAT(r.errors, ContainsSubstring(bad.message));
}

// =============================================================================
// Two-equation models on the benchmark
// =============================================================================

namespace {

struct TwoEqCase {
    std::string model;
    std::string dissipation; // name of the second transported variable
};

const TwoEqCase kTwoEqCases[] = {
    {"k-omega-sst", "log_omega"},
    {"k-epsilon", "epsilon"},
    {"launder-sharma-k-epsilon", "epsilon_tilde"},
};

// Checks of the written two-equation state.
void check_two_equation_state(const Run& r, const TwoEqCase& c) {
    REQUIRE(r.turb_read);
    CHECK(r.turb.model == c.model);
    REQUIRE(r.turb.names == std::vector<std::string>{"k", c.dissipation});
    const std::vector<double>& k = r.turb.values[0];
    const std::vector<double>& d = r.turb.values[1];
    CHECK(count_non_finite(k) == 0);
    CHECK(count_non_finite(d) == 0);
    CHECK(*std::min_element(k.begin(), k.end()) >= 0.0);
    if (c.dissipation != "log_omega") CHECK(*std::min_element(d.begin(), d.end()) >= 0.0);

    const Mesh& m = regression_mesh();
    Mesh geometry = benchmark_mesh();
    const WallGeometry walls(geometry);
    const double k_in = 1.5 * 0.05 * 0.05; // default intensity 5 %, U_ref = 1
    int wall_violations = 0, inlet_violations = 0;
    for (int iw = 0; iw < walls.num_wall_nodes(); ++iw) {
        const int ip = walls.wall_nodes()[iw];
        if (c.model == "k-epsilon") {
            // Wall functions: natural k, eps from local equilibrium.
            const double eps_w = std::pow(0.09, 0.75) * std::pow(k[ip], 1.5) / (0.41 * walls.wall_node_yp(iw));
            if (std::abs(d[ip] - std::max(eps_w, 1e-30)) > 1e-12 * eps_w + 1e-300) ++wall_violations;
        } else {
            if (k[ip] != 0.0) ++wall_violations; // resolved walls: k = 0
            if (c.model == "launder-sharma-k-epsilon" && d[ip] != 0.0) ++wall_violations;
        }
    }
    for (const auto& side : m.iside)
        for (int i : {side.node1, side.node2})
            if (side.flag == 503 && !walls.is_wall_node(i) && k[i] != k_in) ++inlet_violations;
    CHECK(wall_violations == 0);
    CHECK(inlet_violations == 0);
}

} // namespace

TEST_CASE("Two-equation models on the benchmark: 10 transient steps stay finite, bounded and converged",
          "[solver][turbulence][regression]") {
    const TwoEqCase c = GENERATE(from_range(std::begin(kTwoEqCases), std::end(kTwoEqCases)));
    CAPTURE(c.model);
    CaseOptions o;
    o.turbulence = "model " + c.model + "\n";
    const Run r = run_case(o, "twoeq-" + c.model);
    check_healthy_run(r, 10, 0.1, 1.0e-6);
    check_two_equation_state(r, c);
    CHECK_THAT(r.log, ContainsSubstring("Turbulence model: " + c.model));
    const auto lines = turbulence_lines(r.log);
    REQUIRE(lines.size() == 10);
    CHECK_THAT(lines.back(), ContainsSubstring("k change"));
    CHECK_THAT(lines.back(), ContainsSubstring(c.dissipation + " change"));
    CHECK(r.result_files == std::set<std::string>{"hinterkante.var", "hinterkante.turb.var", "hinterkante_0.vtk"});
}

TEST_CASE("Two-equation models on the benchmark: VTK fields match the restart state", "[solver][turbulence][regression]") {
    const TwoEqCase c = GENERATE(from_range(std::begin(kTwoEqCases), std::end(kTwoEqCases)));
    CAPTURE(c.model);
    TempDir dir("twoeq-vtk");
    CaseOptions o;
    o.turbulence = "model " + c.model + "\n";
    prepare_case(dir.path(), o);
    const Run r = run_solver(dir.path());
    REQUIRE(r.turb_read);
    const VtkFile f = parse_vtk(dir / "results/hinterkante_0.vtk");
    std::vector<std::string> names;
    for (const auto& s : f.extra_scalars) names.push_back(s.first);
    const std::vector<std::string> expected =
        c.model == "k-omega-sst"
            ? std::vector<std::string>{"k", "omega", "nut", "sst_f1", "wall_distance", "yplus"}
            : std::vector<std::string>{"k", c.dissipation, "nut", "wall_distance", "yplus"};
    CHECK(names == expected);
    auto close = [](double a, double b) { return std::abs(a - b) <= 2e-7 * std::abs(b) + 1e-30; };
    int bad = 0;
    for (int i = 0; i < f.npointdata; ++i) {
        if (!close((*f.scalar("k"))[i], r.turb.values[0][i])) ++bad;
        if (c.model == "k-omega-sst") {
            if (!close((*f.scalar("omega"))[i], std::exp(r.turb.values[1][i]))) ++bad;
            const double f1 = (*f.scalar("sst_f1"))[i];
            if (!(f1 >= 0.0 && f1 <= 1.0)) ++bad;
        } else if (!close((*f.scalar(c.dissipation))[i], r.turb.values[1][i])) {
            ++bad;
        }
        if (!((*f.scalar("nut"))[i] >= 0.0)) ++bad;
    }
    CHECK(bad == 0);
}

TEST_CASE("Two-equation models on the benchmark: restarting reproduces an uninterrupted run bit for bit",
          "[solver][turbulence][regression][restart]") {
    const TwoEqCase c = GENERATE(from_range(std::begin(kTwoEqCases), std::end(kTwoEqCases)));
    CAPTURE(c.model);
    TempDir straight("twoeq-restart-a"), split("twoeq-restart-b");
    CaseOptions o;
    o.turbulence = "model " + c.model + "\n";
    prepare_case(straight.path(), o);
    const Run full = run_solver(straight.path());
    CaseOptions first = o;
    first.steps = 5;
    prepare_case(split.path(), first);
    REQUIRE(run_solver(split.path()).turb_read);
    set_nonempty_lines(split / "system/hinterkante.control", {{kRestartOpt, "1"}, {kNtime, "10"}, {kWriteOutput, "10"}});
    const Run part2 = run_solver(split.path());
    CHECK_THAT(part2.log, ContainsSubstring("Restarting turbulence from"));
    REQUIRE(full.turb_read);
    REQUIRE(part2.turb_read);
    for (const char* file : {"results/hinterkante.var", "results/hinterkante.turb.var", "results/hinterkante_0.vtk"}) {
        CAPTURE(file);
        CHECK(read_text(straight / file) == read_text(split / file));
    }
}

TEST_CASE("Two-equation models on the benchmark: steady mode with local time stepping", "[solver][turbulence][regression]") {
    const TwoEqCase c = GENERATE(from_range(std::begin(kTwoEqCases), std::end(kTwoEqCases)));
    CAPTURE(c.model);
    CaseOptions o;
    o.turbulence = "model " + c.model + "\n";
    o.transient = 0;
    o.dtfixed = 0;
    o.steps = 30;
    const Run r = run_case(o, "twoeq-steady");
    check_healthy_run(r, 30, 0.1, 1.0e-6);
    check_two_equation_state(r, c);
}

TEST_CASE("Two-equation models on the benchmark: converged-pressure reference values", "[solver][turbulence][regression]") {
    const TwoEqCase c = GENERATE(from_range(std::begin(kTwoEqCases), std::end(kTwoEqCases)));
    CAPTURE(c.model);
    CaseOptions o;
    o.turbulence = "model " + c.model + "\n";
    o.rel_tol = "0.0";
    o.abs_tol = "1.0d-10";
    const Run r = run_case(o, "twoeq-tight");
    check_healthy_run(r, 10, 0.0, 1.0e-10);
    check_two_equation_state(r, c);
    // Recorded on Linux x86-64 (GCC); converged PCG, see test_solver_regression.cpp.
    struct Reference {
        double l2u, l2v, l2p, rtime, l2k, l2d;
    };
    const Reference ref = c.model == "k-omega-sst"
                              ? Reference{88.8125786642, 11.7420324834, 15.8069691699, 0.0455372702522, 0.343560771434,
                                          183.377623282}
                          : c.model == "k-epsilon"
                              ? Reference{88.7627038932, 11.7042487845, 16.9519267469, 0.0453541283656, 0.361992390909,
                                          0.10887592279}
                              : Reference{88.8147431634, 11.7438399608, 15.6441835491, 0.0493124417979, 0.341885213895,
                                          0.180316786327};
    CHECK_THAT(l2(r.u), WithinRel(ref.l2u, 1e-7));
    CHECK_THAT(l2(r.v), WithinRel(ref.l2v, 1e-7));
    CHECK_THAT(l2(r.p), WithinRel(ref.l2p, 1e-7));
    CHECK_THAT(r.rtime, WithinRel(ref.rtime, 1e-7));
    CHECK_THAT(l2(r.turb.values[0]), WithinRel(ref.l2k, 1e-7));
    CHECK_THAT(l2(r.turb.values[1]), WithinRel(ref.l2d, 1e-7));
}
