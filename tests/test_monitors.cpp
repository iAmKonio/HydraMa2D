// Force and probe monitors (post/Monitors.hpp): the system/<case>.monitors
// parser, body and probe resolution, the data file, and the solver
// integration (results unchanged, restart appends).

#include "MeshCaseTestUtils.hpp"
#include "QuadTestUtils.hpp"
#include "mesh/GmshReader.hpp"
#include "mesh/MeshBuilder.hpp"
#include "post/Monitors.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sstream>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::StartsWith;
using Catch::Matchers::WithinAbs;

namespace {

post::MonitorConfig parse_ok(const std::string& text) {
    std::istringstream in(text);
    post::MonitorConfig c;
    std::string error;
    INFO(error);
    REQUIRE(post::parse_monitor_config(in, c, error));
    return c;
}

std::string parse_error(const std::string& text) {
    std::istringstream in(text);
    post::MonitorConfig c;
    std::string error;
    CHECK_FALSE(post::parse_monitor_config(in, c, error));
    return error;
}

// Channel [0, 3] x [0, 1] with a square hole [1, 1.5] x [0.25, 0.75] (the
// body), imported from a Gmsh file: groups 'inlet' (1), 'outlet' (2),
// 'walls' (3), 'Square_Wall' (4), and 'noSlip' (5) on the same sides as 4.
Mesh holed_channel() {
    GridOptions g;
    g.lx = 3.0;
    g.nx = 12;
    g.ny = 4;
    g.cells = GridCells::InflationLayers;
    const Mesh m = make_hybrid_mesh(
        without_cells(grid(g), [](double x, double y) { return x > 1.0 && x < 1.5 && y > 0.25 && y < 0.75; }),
        [](double x, double y) {
            if (x == 0.0) return 1;
            if (x == 3.0) return 2;
            if (y == 0.0 || y == 1.0) return 3;
            return 4;
        });
    GmshModel model;
    for (int i = 0; i < m.npoin; ++i) model.add_node(10 + i, m.x[i], m.y[i]);
    for (const auto& t : m.intma) model.add_element(2, {10 + t[0], 10 + t[1], 10 + t[2]}, {7});
    for (const auto& q : m.quads) model.add_element(3, {10 + q[0], 10 + q[1], 10 + q[2], 10 + q[3]}, {7});
    for (const auto& s : m.iside) {
        model.add_element(1, {10 + s.node1, 10 + s.node2}, {s.flag});
        if (s.flag == 4) model.add_element(1, {10 + s.node1, 10 + s.node2}, {5});
    }
    model.names[{1, 1}] = "inlet";
    model.names[{1, 2}] = "outlet";
    model.names[{1, 3}] = "walls";
    model.names[{1, 4}] = "Square_Wall";
    model.names[{1, 5}] = "noSlip";
    CaptureStdout quiet;
    Mesh out = mesh::build_mesh(mesh::parse_gmsh(write_gmsh(model, {}), "holed.msh"));
    compute_all_geometry(out);
    return out;
}

std::vector<std::string> data_lines(const std::string& file) {
    std::vector<std::string> out;
    for (const std::string& line : split_lines(file))
        if (!line.empty() && line[0] != '#') out.push_back(line);
    return out;
}

} // namespace

TEST_CASE("Monitors: configuration file", "[post][monitors]") {
    const post::MonitorConfig c = parse_ok("# comment\n"
                                           "body  cylinder_wall  4   # groups\n"
                                           "reference_velocity 2.0d0\n"
                                           "REFERENCE_LENGTH   0.1\n"
                                           "interval 5\n"
                                           "probe front 1.5 2.0\n"
                                           "probe back_2 2.5e0 -2\n");
    CHECK(c.body == std::vector<std::string>{"cylinder_wall", "4"});
    CHECK(c.reference_velocity == 2.0);
    CHECK(c.reference_length == 0.1);
    CHECK(c.interval == 5);
    REQUIRE(c.probes.size() == 2);
    CHECK(c.probes[1].name == "back_2");
    CHECK(c.probes[1].x == 2.5);
    CHECK(c.probes[1].y == -2.0);
    CHECK(parse_ok("probe a 0 0\n").body.empty());
    CHECK(parse_ok("body 3\n").interval == 1);

    CHECK_THAT(parse_error("body\n"), ContainsSubstring("line 1: 'body' needs at least one boundary group"));
    CHECK_THAT(parse_error("body a\nbody b\n"), ContainsSubstring("line 2: 'body' is given twice"));
    CHECK_THAT(parse_error("body a\nreference_velocity 0\n"), ContainsSubstring("line 2: 'reference_velocity' must be a number > 0, got '0'"));
    CHECK_THAT(parse_error("body a\nreference_length x\n"), ContainsSubstring("'reference_length' must be a number > 0"));
    CHECK_THAT(parse_error("body a\nreference_length 1 2\n"), ContainsSubstring("needs exactly one value"));
    CHECK_THAT(parse_error("body a\ninterval 0\n"), ContainsSubstring("'interval' must be an integer >= 1, got '0'"));
    CHECK_THAT(parse_error("body a\ninterval 1.5\n"), ContainsSubstring("'interval' must be an integer >= 1"));
    CHECK_THAT(parse_error("probe a 1\n"), ContainsSubstring("'probe' needs a name and two coordinates"));
    CHECK_THAT(parse_error("probe a 1 nan\n"), ContainsSubstring("invalid coordinates for probe 'a'"));
    CHECK_THAT(parse_error("probe a/b 1 1\n"), ContainsSubstring("invalid probe name 'a/b'"));
    CHECK_THAT(parse_error("probe a 1 1\nprobe a 2 2\n"), ContainsSubstring("line 2: probe 'a' is defined twice"));
    CHECK_THAT(parse_error("body a\nforce x\n"), ContainsSubstring("line 2: unknown key 'force'"));
    CHECK_THAT(parse_error("# nothing\n\n"), ContainsSubstring("neither a body nor a probe"));

    TempDir dir("monitors-config");
    post::MonitorConfig missing;
    bool found = true;
    std::string error;
    CHECK(post::read_monitor_config((dir / "none.monitors").string(), missing, found, error));
    CHECK_FALSE(found);
    CHECK(missing.empty());
}

TEST_CASE("Monitors: the body by group name, tag or side id, and probes", "[post][monitors]") {
    const Mesh m = holed_channel();
    REQUIRE(m.boundary_groups.size() == 5);
    const std::vector<int> square = m.boundary_groups[3].sides;
    CHECK(m.boundary_groups[3].name == "Square_Wall");
    CHECK(square.size() == 8);
    CHECK(m.boundary_groups[4].sides == square); // noSlip: the same sides, side_id 4
    for (int ib : square) CHECK(m.iside[ib].side_id == 4);

    post::MonitorConfig c;
    for (const char* entry : {"Square_Wall", "square_wall", "4", "noSlip", "5"}) {
        CAPTURE(entry);
        c.body = {std::string(entry)};
        const post::Monitors mon(m, c);
        CHECK(mon.body_sides() == square);
    }
    c.body = {"inlet", "outlet"};
    CHECK(post::Monitors(m, c).body_sides().size() == 8);
    c.body = {"4", "noSlip"}; // the same sides twice: counted once
    CHECK(post::Monitors(m, c).body_sides() == square);

    c.body = {"cylinder"};
    CHECK_THROWS_WITH(post::Monitors(m, c), ContainsSubstring("body 'cylinder' matches no boundary group (groups: inlet, outlet, walls, Square_Wall, noSlip)"));
    c.body = {"9"};
    CHECK_THROWS_WITH(post::Monitors(m, c), ContainsSubstring("body '9' matches no boundary group"));

    c.body = {"Square_Wall"};
    c.probes = {{"front", 0.9, 0.5}, {"inside", 1.25, 0.5}};
    CHECK_THROWS_WITH(post::Monitors(m, c), ContainsSubstring("probe 'inside' at (1.25, 0.5) lies outside the mesh"));
    c.probes = {{"front", 0.9, 0.5}};
    const post::Monitors mon(m, c);
    REQUIRE(mon.probe_locations().size() == 1);

    // Legacy meshes (no groups): side ids.
    Mesh legacy = m;
    legacy.boundary_groups.clear();
    c.body = {"4"};
    CHECK(post::Monitors(legacy, c).body_sides() == square);
    c.body = {"Square_Wall"};
    CHECK_THROWS_WITH(post::Monitors(legacy, c), ContainsSubstring("matches no boundary group or side id"));
}

TEST_CASE("Monitors: evaluation and data file format", "[post][monitors]") {
    const Mesh m = holed_channel();
    post::MonitorConfig c;
    c.body = {"Square_Wall"};
    c.reference_velocity = 2.0;
    c.reference_length = 0.5;
    c.probes = {{"a", 0.5, 0.5}, {"b", 2.0, 0.1}};
    const post::Monitors mon(m, c);
    std::vector<double> u(static_cast<std::size_t>(m.npoin)), v(u.size()), p(u.size());
    for (int i = 0; i < m.npoin; ++i) {
        u[i] = 1.0 + 0.2 * m.x[i];
        v[i] = 0.3 * m.y[i];
        p[i] = 2.0 - m.x[i] + 0.5 * m.y[i];
    }
    const post::MonitorSample s = mon.evaluate(m, 0.01, u, v, p);
    const post::BodyForce vol = post::volume_force(m, mon.body_sides(), 0.01, u, v, p);
    const post::BodyForce sur = post::surface_force(m, mon.body_sides(), 0.01, u, v, p);
    const double q = 0.5 * 4.0 * 0.5;
    CHECK(s.cd == vol.fx / q);
    CHECK(s.cl == vol.fy / q);
    CHECK(s.cd_surface == sur.fx / q);
    CHECK(s.cl_surface == sur.fy / q);
    CHECK(s.cd_pressure == sur.pressure_fx / q);
    CHECK_THAT(s.probe_u[0], WithinAbs(1.1, 1e-12));
    CHECK_THAT(s.probe_v[1], WithinAbs(0.03, 1e-12));
    CHECK_THAT(s.probe_p[1], WithinAbs(0.05, 1e-12));

    const std::string header = mon.file_header("mycase");
    CHECK_THAT(header, StartsWith("# HydraMa2D monitors, case mycase\n# body: 'Square_Wall' (tag 4); 8 sides, length 2\n"));
    CHECK_THAT(header, ContainsSubstring("# coefficients C = F / (0.5 U_ref^2 L_ref), U_ref = 2, L_ref = 0.5\n"));
    CHECK_THAT(header, ContainsSubstring("# probe a at (0.5, 0.5)\n"));
    CHECK_THAT(header, ContainsSubstring("\nstep time cd cl cd_surface cl_surface cd_pressure cl_pressure u_a v_a p_a u_b v_b p_b\n"));
    const std::string row = mon.format_row(12, 0.25, s);
    std::istringstream in(row);
    std::vector<double> values;
    for (double x; in >> x;) values.push_back(x);
    REQUIRE(values.size() == 14);
    CHECK(values[0] == 12.0);
    CHECK(values[1] == 0.25);
    CHECK_THAT(values[2], WithinAbs(s.cd, 1e-11 * std::abs(s.cd)));
    CHECK_THAT(values[13], WithinAbs(0.05, 1e-12));
    CHECK_THAT(mon.description(), ContainsSubstring("Monitors: body 'Square_Wall' (tag 4): 8 sides, length 2; U_ref 2, L_ref 0.5\n"));
    CHECK_THAT(mon.description(), ContainsSubstring("Monitors: probe 'b' at (2, 0.1) in element"));

    c.body.clear(); // probes only
    const post::Monitors probes_only(m, c);
    CHECK_FALSE(probes_only.has_body());
    CHECK_THAT(probes_only.file_header("x"), ContainsSubstring("\nstep time u_a v_a p_a u_b v_b p_b\n"));
}

TEST_CASE("Solver: monitors do not change the results; a restart appends", "[post][monitors][solver]") {
    const std::string mesh = data_file("cylinder_inflation_4.1_binary.msh");
    CaseOptions opts;
    opts.steps = 30;
    const MeshCaseResult plain = run_mesh_case("monitors-off", mesh, opts);
    MeshCaseOptions extra;
    extra.system_files["hinterkante.monitors"] = "body cylinder_wall\ninterval 3\nprobe wake 1.5 0.5\n";
    const MeshCaseResult with = run_mesh_case("monitors-on", mesh, opts, extra);
    INFO(with.run.errors);
    REQUIRE(with.run.initialized);
    CHECK(plain.monitors.empty());
    CHECK_FALSE(plain.run.result_files.count("hinterkante.monitors.dat"));
    CHECK(with.run.u == plain.run.u); // bit for bit
    CHECK(with.run.v == plain.run.v);
    CHECK(with.run.p == plain.run.p);
    CHECK_THAT(with.run.log, ContainsSubstring("Monitors: body 'cylinder_wall' (tag 4): 16 sides"));
    CHECK_THAT(with.run.log, ContainsSubstring("Monitors (last sample): cd "));
    const std::vector<std::string> rows = data_lines(with.monitors);
    REQUIRE(rows.size() == 1 + 10); // header + steps 3, 6, ..., 30
    CHECK_THAT(rows[0], StartsWith("step time cd cl"));
    CHECK_THAT(rows.back(), StartsWith("30 "));

    // The last step is always sampled: steps 7, 14, 21, 28 and 30.
    extra.system_files["hinterkante.monitors"] = "body cylinder_wall\ninterval 7\n";
    const MeshCaseResult sevens = run_mesh_case("monitors-last", mesh, opts, extra);
    const std::vector<std::string> seven_rows = data_lines(sevens.monitors);
    REQUIRE(seven_rows.size() == 1 + 5);
    CHECK_THAT(seven_rows[4], StartsWith("28 "));
    CHECK_THAT(seven_rows[5], StartsWith("30 "));

    extra.system_files["hinterkante.monitors"] = "body cylinder_wall\ninterval 3\nprobe wake 1.5 0.5\n";
    extra.restart_twice = true;
    const MeshCaseResult twice = run_mesh_case("monitors-restart", mesh, opts, extra);
    REQUIRE(twice.run.initialized);
    const std::vector<std::string> all = data_lines(twice.monitors);
    // The restarted run (steps 31..30) adds nothing beyond the first run's rows: ntime is reached.
    CHECK(all.size() == rows.size());
    CHECK_THAT(twice.run.log, ContainsSubstring("(appending)"));

    extra.system_files["hinterkante.monitors"] = "body nowhere\n";
    extra.restart_twice = false;
    const MeshCaseResult bad = run_mesh_case("monitors-bad", mesh, opts, extra);
    CHECK_FALSE(bad.run.initialized);
    CHECK_THAT(bad.run.errors, ContainsSubstring("ERROR: monitors: body 'nowhere' matches no boundary group"));
    extra.system_files["hinterkante.monitors"] = "body cylinder_wall\nspeed 3\n";
    const MeshCaseResult syntax = run_mesh_case("monitors-syntax", mesh, opts, extra);
    CHECK_FALSE(syntax.run.initialized);
    CHECK_THAT(syntax.run.errors, ContainsSubstring("ERROR: *.monitors FILE HAS INVALID FORMAT: line 2: unknown key 'speed'"));
}

TEST_CASE("Monitors pitching moment evaluation and formatting", "[monitors][moment]") {
    const std::string text =
        "body Square_Wall\n"
        "moment_center 0.25 0.0\n"
        "reference_velocity 2.0\n"
        "reference_length 1.5\n";
    std::istringstream in(text);
    post::MonitorConfig c;
    std::string err;
    REQUIRE(post::parse_monitor_config(in, c, err));
    CHECK(c.compute_moment);
    CHECK(c.moment_center_x == 0.25);
    CHECK(c.moment_center_y == 0.0);
    CHECK(c.reference_velocity == 2.0);
    CHECK(c.reference_length == 1.5);

    Mesh m = holed_channel();
    post::Monitors mon(m, c);
    std::string header = mon.file_header("case_test");
    CHECK_THAT(header, ContainsSubstring("step time cd cl cm cd_surface cl_surface cm_surface cd_pressure cl_pressure cm_pressure\n"));

    post::MonitorSample s;
    s.cd = 1.0;
    s.cl = 2.0;
    s.cm = -0.05;
    s.cd_surface = 1.1;
    s.cl_surface = 2.1;
    s.cd_pressure = 0.9;
    s.cl_pressure = 1.9;
    s.cm_pressure = -0.04;
    std::string row = mon.format_row(10, 0.5, s);
    CHECK_THAT(row, ContainsSubstring("10 "));
    CHECK_THAT(row, ContainsSubstring("-5.000000000000e-02"));
}
