#pragma once

// Solver runs on a mesh file: a private copy of the benchmark case whose
// legacy mesh is replaced by a Gmsh file (constant/hinterkante.msh), with
// optional extra system files (e.g. hinterkante.monitors).

#include "GmshTestWriter.hpp"
#include "IO.hpp"
#include "SolverTestUtils.hpp"

#include <map>
#include <string>

namespace hydrama::test {

struct MeshCaseResult {
    Run run;
    Mesh mesh;
    std::string monitors; // results/hinterkante.monitors.dat ("" if not written)
};

struct MeshCaseOptions {
    std::map<int, std::string> physical_edits;           // non-empty line index -> value (*.physical)
    std::map<int, std::string> control_edits;            // non-empty line index -> value (*.control)
    std::map<std::string, std::string> system_files;     // file name in system/ -> content
    bool restart_twice = false;                          // run, then run again with restart_opt = 1
};

inline MeshCaseResult run_mesh_case(const std::string& tag, const std::string& mesh_file, const CaseOptions& opts,
                                    const MeshCaseOptions& extra = {}) {
    TempDir dir(tag);
    prepare_case(dir.path(), opts);
    fs::remove(dir / "constant/hinterkante.mesh");
    fs::remove(dir / "constant/hinterkante.bofl");
    write_text(dir / "constant/hinterkante.msh", mesh_file);
    if (!extra.physical_edits.empty()) set_nonempty_lines(dir / "system/hinterkante.physical", extra.physical_edits);
    if (!extra.control_edits.empty()) set_nonempty_lines(dir / "system/hinterkante.control", extra.control_edits);
    for (const auto& [name, content] : extra.system_files) write_text(dir / "system" / name, content);

    MeshCaseResult r;
    auto solve = [&]() {
        CaptureStdout capture;
        CaptureStderr errors;
        SimulationConfig config;
        config.casename = "hinterkante";
        config.case_dir = dir.path().string();
        Solver solver(config);
        r.run.initialized = solver.initialize();
        if (r.run.initialized) solver.run();
        r.run.log += capture.str();
        r.run.errors += errors.str();
    };
    solve();
    if (extra.restart_twice && r.run.initialized) {
        set_nonempty_lines(dir / "system/hinterkante.control", {{kRestartOpt, "1"}});
        solve();
    }
    r.run.steps = parse_log(r.run.log);
    for (const auto& entry : fs::directory_iterator(dir / "results")) r.run.result_files.insert(entry.path().filename().string());
    if (fs::exists(dir / "results/hinterkante.monitors.dat")) r.monitors = read_text(dir / "results/hinterkante.monitors.dat");
    if (!r.run.initialized) return r;
    {
        CaptureStdout quiet;
        REQUIRE(IO::read_gmsh_mesh((dir / "constant/hinterkante.msh").string(), "", r.mesh));
    }
    const std::size_t n = static_cast<std::size_t>(r.mesh.npoin);
    r.run.u.assign(n, 0.0);
    r.run.v.assign(n, 0.0);
    r.run.p.assign(n, 0.0);
    r.run.state_read = IO::read_restart((dir / "results/hinterkante.var").string(), r.run.rtime, r.run.istart, r.run.u,
                                        r.run.v, r.run.p);
    if (fs::exists(dir / "results/hinterkante.turb.var")) {
        r.run.turb_read = IO::read_turbulence_restart((dir / "results/hinterkante.turb.var").string(), r.mesh.npoin,
                                                      r.run.turb, r.run.turb_error);
    }
    return r;
}

// Gmsh file of a mesh built with QuadTestUtils: one physical group per
// boundary flag, named by `names`.
inline std::string gmsh_file(const Mesh& m,
                             const std::map<int, std::string>& names = {{500, "wall"}, {503, "inlet"}, {504, "outlet"},
                                                                        {506, "symmetry"}, {508, "parabolic_inlet"}}) {
    GmshModel g;
    for (int i = 0; i < m.npoin; ++i) g.add_node(i + 1, m.x[static_cast<std::size_t>(i)], m.y[static_cast<std::size_t>(i)]);
    for (const auto& t : m.intma) g.add_element(2, {t[0] + 1, t[1] + 1, t[2] + 1}, {1});
    for (const auto& q : m.quads) g.add_element(3, {q[0] + 1, q[1] + 1, q[2] + 1, q[3] + 1}, {1});
    for (const auto& s : m.iside) g.add_element(1, {s.node1 + 1, s.node2 + 1}, {s.flag});
    for (const auto& [flag, name] : names) g.names[{1, flag}] = name;
    return write_gmsh(g, GmshWriteOptions{"4.1", true, false, false, 8, ""});
}

inline std::string data_file(const std::string& name) {
    return read_text(std::string(HYDRAMA2D_TEST_DATA_DIR) + "/gmsh/" + name);
}

} // namespace hydrama::test
