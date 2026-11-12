// The Schafer-Turek cylinder benchmark (docs/CYLINDER_BENCHMARK.md) on the
// coarsest meshes of tools/cylinder_benchmark/generate_meshes.py (level 0,
// fixtures in tests/data/gmsh): a triangle mesh and the same mesh with a
// quadrilateral inflation layer on the cylinder.
//
//   * 2D-1, steady, Re = 20: converged drag, lift and pressure difference,
//     pinned (regression) and within 10 % of the reference values (level 0
//     is far too coarse for the benchmark accuracy; the tool runs the finer
//     levels);
//   * 2D-2, Re = 100: periodic vortex shedding on the triangle mesh.

#include "MeshCaseTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sstream>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinRel;

namespace {

const char* kMonitors = "body cylinder_wall\n"
                        "reference_velocity 1\n"
                        "reference_length 1\n"
                        "interval 10\n"
                        "probe front 1.5 2.0\n"
                        "probe back 2.5 2.0\n";

// The settings of tools/cylinder_benchmark/run_benchmark.py.
CaseOptions benchmark_options(int steps, bool steady) {
    CaseOptions opts;
    opts.steps = steps;
    opts.transient = steady ? 0 : 1;
    opts.dtfixed = steady ? 0 : -1;
    opts.rel_tol = "1.0d-08";
    opts.abs_tol = "1.0d-13";
    opts.solver_options = "method = cg\npreconditioner = ic0\noperator_format = csr\nreordering = rcm\n";
    return opts;
}

struct Column {
    std::vector<std::string> names;
    std::vector<std::vector<double>> rows;
    std::vector<double> operator()(const std::string& name) const {
        const auto it = std::find(names.begin(), names.end(), name);
        REQUIRE(it != names.end());
        const auto k = static_cast<std::size_t>(it - names.begin());
        std::vector<double> out;
        for (const auto& r : rows) out.push_back(r[k]);
        return out;
    }
};

Column parse_monitors(const std::string& text) {
    Column c;
    for (const std::string& line : split_lines(text)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream in(line);
        if (c.names.empty()) {
            for (std::string w; in >> w;) c.names.push_back(w);
            continue;
        }
        std::vector<double> row;
        for (double x; in >> x;) row.push_back(x);
        REQUIRE(row.size() == c.names.size());
        c.rows.push_back(row);
    }
    return c;
}

} // namespace

TEST_CASE("Cylinder benchmark 2D-1 (Re = 20): triangle and inflation-layer meshes", "[benchmark][cylinder][solver]") {
    struct Expected {
        const char* mesh;
        int triangles, quads;
        double cd, cl, dp; // recorded on Linux / x86-64 (GCC)
    };
    for (const Expected& e : {Expected{"schaefer_turek_tri_L0.msh", 977, 0, 6.087858631946, -0.02280632848, 2.834092189},
                              Expected{"schaefer_turek_bl_L0.msh", 951, 64, 5.894450468674, -0.01905052458, 2.963726471}}) {
        CAPTURE(e.mesh);
        MeshCaseOptions extra;
        extra.physical_edits = {{2, "20.0"}};                          // Re = 20
        extra.control_edits = {{kL2Norm, "1.0d-10"}, {kIwrite, "100"}}; // converge tightly
        extra.system_files["hinterkante.monitors"] = kMonitors;
        const MeshCaseResult r = run_mesh_case("cylinder-2d1", data_file(e.mesh), benchmark_options(5000, true), extra);
        INFO(r.run.errors);
        REQUIRE(r.run.initialized);
        const std::string counts = "Mesh import: " + std::to_string(e.triangles) + " triangles, " +
                                   (e.quads ? std::to_string(e.quads) + " quadrilaterals, " : std::string());
        CHECK_THAT(r.run.log, ContainsSubstring(counts));
        CHECK_THAT(r.run.log, ContainsSubstring("Boundary 'parabolic_inlet' (tag 1): inlet_parabolic, flag 508"));
        CHECK_THAT(r.run.log, ContainsSubstring("Mesh quality: "));
        CHECK_THAT(r.run.log, ContainsSubstring("Monitors: body 'cylinder_wall' (tag 4): 16 sides"));
        const Column c = parse_monitors(r.monitors);
        REQUIRE(c.rows.size() > 10);
        CHECK(c.rows.size() < 500); // converged well before the step limit

        const double cd = c("cd").back(), cl = c("cl").back();
        const double dp = c("p_front").back() - c("p_back").back();
        // Converged: the last two samples agree.
        CHECK_THAT(c("cd")[c.rows.size() - 2], WithinRel(cd, 1e-9));
        // Regression values.
        CHECK_THAT(cd, WithinRel(e.cd, 1e-6));
        CHECK_THAT(cl, WithinRel(e.cl, 1e-4));
        CHECK_THAT(dp, WithinRel(e.dp, 1e-6));
        // Schafer & Turek (1996): c_D in [5.57, 5.59], dp in [2.93, 2.94] (non-dimensional).
        CHECK_THAT(cd, WithinRel(5.58, 0.10));
        CHECK_THAT(dp, WithinRel(2.935, 0.05));
        // Surface and volume forms agree to the discretisation error.
        CHECK_THAT(c("cd_surface").back(), WithinRel(cd, 0.15));
    }
}

TEST_CASE("Cylinder benchmark 2D-2 (Re = 100): periodic vortex shedding", "[benchmark][cylinder][solver]") {
    MeshCaseOptions extra;
    extra.physical_edits = {{2, "100.0"}};
    extra.control_edits = {{kIwrite, "1000"}, {14, "60.0"}}; // end time 60
    extra.system_files["hinterkante.monitors"] = kMonitors;
    const MeshCaseResult r =
        run_mesh_case("cylinder-2d2", data_file("schaefer_turek_tri_L0.msh"), benchmark_options(100000, false), extra);
    INFO(r.run.errors);
    REQUIRE(r.run.initialized);
    const Column c = parse_monitors(r.monitors);
    const std::vector<double> t = c("time"), cl = c("cl"), cd = c("cd");
    REQUIRE(t.back() > 59.0);
    // Upward zero crossings of the lift after the start-up (t > 30).
    std::vector<double> up;
    double cl_max = 0.0, cd_max = 0.0;
    for (std::size_t i = 1; i < t.size(); ++i) {
        if (t[i] < 30.0) continue;
        cl_max = std::max(cl_max, cl[i]);
        cd_max = std::max(cd_max, cd[i]);
        if (cl[i - 1] < 0.0 && cl[i] >= 0.0) up.push_back(t[i - 1] - cl[i - 1] * (t[i] - t[i - 1]) / (cl[i] - cl[i - 1]));
    }
    REQUIRE(up.size() >= 5);
    const double period = (up.back() - up.front()) / static_cast<double>(up.size() - 1);
    const double strouhal = 1.0 / period;
    CAPTURE(strouhal, cl_max, cd_max);
    // Periodic: the periods agree.
    for (std::size_t k = 1; k < up.size(); ++k) CHECK_THAT(up[k] - up[k - 1], WithinRel(period, 0.02));
    // Level 0 is coarse (534 nodes): St 0.247, c_L max 1.43, c_D max 3.65
    // against 0.30, 1.00 and 3.23 on fine meshes; the values are pinned.
    CHECK_THAT(strouhal, WithinRel(0.247, 0.01));
    CHECK_THAT(cl_max, WithinRel(1.43, 0.02));
    CHECK_THAT(cd_max, WithinRel(3.646, 0.01));
}
