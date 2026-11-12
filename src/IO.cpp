#include "IO.hpp"

#include "linalg/Factory.hpp"
#include "linalg/Reordering.hpp"
#include "mesh/GmshReader.hpp"
#include "mesh/MeshBuilder.hpp"
#include "mesh/CgnsReader.hpp"
#include "mesh/Su2Reader.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <vector>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <set>

namespace hydrama {

double IO::parse_fortran_double(std::string str) {
    for (char& c : str) {
        if (c == 'd' || c == 'D') {
            c = 'e';
        }
    }
    return std::stod(str);
}

static std::vector<std::string> read_non_empty_lines(const std::string& filepath) {
    std::vector<std::string> lines;
    std::ifstream in(filepath);
    if (!in.is_open()) return lines;
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        size_t start = line.find_first_not_of(" \t");
        if (start != std::string::npos) {
            lines.push_back(line.substr(start));
        }
    }
    return lines;
}

bool IO::read_control(const std::string& filepath, ControlConfig& config) {
    auto lines = read_non_empty_lines(filepath);
    if (lines.size() < 45) {
        std::cerr << "ERROR: *.control FILE DOES NOT EXIST OR HAS INVALID FORMAT: " << filepath << "\n";
        return false;
    }

    config.restart_opt          = std::stoi(lines[2]);
    config.ntime                = std::stoi(lines[5]);
    config.transient_on         = std::stoi(lines[8]);
    config.time_output_interval = parse_fortran_double(lines[11]);
    config.end_rtime            = parse_fortran_double(lines[14]);
    config.write_output         = std::stoi(lines[17]);
    config.write_time_output    = parse_fortran_double(lines[20]);
    config.dtfixed              = std::stoi(lines[23]);
    config.dtfix                = parse_fortran_double(lines[26]);
    config.iwrite               = std::stoi(lines[29]);
    config.csafm                = parse_fortran_double(lines[32]);
    config.theta[0]             = parse_fortran_double(lines[35]);
    config.theta[1]             = 1.0;
    config.vel_check            = std::stoi(lines[38]);
    config.l2norm_vel_tolerance = parse_fortran_double(lines[41]);
    config.runtime_mod          = std::stoi(lines[44]);

    return true;
}

bool IO::read_physical(const std::string& filepath, PhysicalConfig& config) {
    auto lines = read_non_empty_lines(filepath);
    if (lines.size() < 6) {
        std::cerr << "ERROR: *.physical FILE DOES NOT EXIST OR HAS INVALID FORMAT: " << filepath << "\n";
        return false;
    }

    config.re = parse_fortran_double(lines[2]);
    int pnode_1based = std::stoi(lines[5]);
    config.pnode = pnode_1based - 1;

    return true;
}

namespace {

std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool is_identifier(const std::string& s) {
    if (s.empty() || !(std::isalpha(static_cast<unsigned char>(s[0])) || s[0] == '_')) return false;
    for (char c : s)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
    return true;
}

// Whole-string conversions (no trailing garbage); Fortran 'd' exponents allowed.
bool parse_int(const std::string& s, int& out) {
    try {
        size_t pos = 0;
        const int v = std::stoi(s, &pos);
        if (pos != s.size()) return false;
        out = v;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_double(std::string s, double& out) {
    for (char& c : s)
        if (c == 'd' || c == 'D') c = 'e';
    try {
        size_t pos = 0;
        const double v = std::stod(s, &pos);
        if (pos != s.size() || !std::isfinite(v)) return false;
        out = v;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool option_error(const std::string& filepath, const std::string& message) {
    std::cerr << "ERROR: *.solver OPTION " << message << " (" << filepath << ")\n";
    return false;
}

// Optional "key = value" settings following the positional block of *.solver.
bool read_solver_options(const std::vector<std::string>& lines, size_t first, const std::string& filepath,
                         SolverConfig& config) {
    std::set<std::string> seen;
    for (size_t n = first; n < lines.size(); ++n) {
        std::string line = lines[n];
        const size_t comment = line.find_first_of("#!");
        if (comment != std::string::npos) line.erase(comment);
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;               // free text / separators
        const std::string key = to_lower(trim(line.substr(0, eq)));
        if (!is_identifier(key)) continue;                    // header text containing '='
        const std::string raw = trim(line.substr(eq + 1));
        const std::string value = to_lower(raw);
        const std::string where = "'" + key + " = " + raw + "'";

        if (raw.empty()) return option_error(filepath, where + ": missing value");
        if (!seen.insert(key).second) return option_error(filepath, "'" + key + "' is given more than once");

        if (key == "method") {
            if (!linalg::find_krylov_solver(value)) {
                return option_error(filepath, where + ": unknown method (available: " +
                                                  linalg::krylov_solver_names() + ")");
            }
            config.method = value;
        } else if (key == "preconditioner") {
            if (!linalg::find_preconditioner(value)) {
                return option_error(filepath, where + ": unknown preconditioner (available: " +
                                                  linalg::preconditioner_names() + ")");
            }
            config.preconditioner = value;
        } else if (key == "operator_format") {
            if (value != "ebe" && value != "csr") return option_error(filepath, where + ": expected ebe | csr");
            config.operator_format = value;
        } else if (key == "reordering") {
            if (!linalg::is_ordering(value)) return option_error(filepath, where + ": expected none | rcm");
            config.reordering = value;
        } else if (key == "max_iterations") {
            int v = 0;
            if (!parse_int(raw, v) || v < 1) return option_error(filepath, where + ": expected an integer >= 1");
            config.max_iterations = v;
        } else if (key == "residual_reference") {
            if (value != "first_iteration" && value != "initial" && value != "rhs") {
                return option_error(filepath, where + ": expected first_iteration | initial | rhs");
            }
            config.residual_reference = value;
        } else if (key == "ssor_omega") {
            double v = 0.0;
            if (!parse_double(raw, v) || !(v > 0.0 && v < 2.0)) {
                return option_error(filepath, where + ": expected a number with 0 < omega < 2");
            }
            config.ssor_omega = v;
        } else if (key == "diagonal_shift") {
            double v = 0.0;
            if (!parse_double(raw, v) || v < 0.0) return option_error(filepath, where + ": expected a number >= 0");
            config.diagonal_shift = v;
        } else if (key == "preconditioner_reuse") {
            if (value != "rescale" && value != "rebuild") {
                return option_error(filepath, where + ": expected rescale | rebuild");
            }
            config.preconditioner_reuse = value;
        } else if (key == "gmres_restart") {
            int v = 0;
            if (!parse_int(raw, v) || v < 1) return option_error(filepath, where + ": expected an integer >= 1");
            config.gmres_restart = v;
        } else if (key == "gmres_orthogonalization") {
            if (!linalg::is_orthogonalization(value)) return option_error(filepath, where + ": expected mgs | cgs2");
            config.gmres_orthogonalization = value;
        } else {
            return option_error(filepath, "'" + key + "': unknown option (available: method, preconditioner, "
                                          "operator_format, reordering, max_iterations, residual_reference, "
                                          "ssor_omega, diagonal_shift, preconditioner_reuse, "
                                          "gmres_restart, gmres_orthogonalization)");
        }
    }

    // Consistency checks that involve several options.
    const linalg::PreconditionerInfo* pre = linalg::find_preconditioner(config.preconditioner);
    if (config.reordering != "none" && pre && !pre->requires_assembled) {
        return option_error(filepath, "'reordering = " + config.reordering + "' requires an assembled preconditioner "
                                      "(ssor | ic0 | ilu0), got '" + config.preconditioner + "'");
    }
    const linalg::KrylovInfo* krylov = linalg::find_krylov_solver(config.method);
    if (krylov && krylov->requires_symmetric && pre && !pre->symmetric) {
        std::cout << "WARNING: preconditioner '" << config.preconditioner << "' is not symmetric in general; with '"
                  << config.method << "' it relies on the symmetry of the pressure matrix\n";
    }
    return true;
}

} // namespace

bool IO::read_solver(const std::string& filepath, SolverConfig& config) {
    auto lines = read_non_empty_lines(filepath);
    if (lines.size() < 10) {
        std::cerr << "ERROR: *.solver FILE DOES NOT EXIST OR HAS INVALID FORMAT: " << filepath << "\n";
        return false;
    }

    config.solver_opt = std::stoi(lines[2]);
    config.relToler   = parse_fortran_double(lines[6]);
    config.absToler   = parse_fortran_double(lines[9]);

    // Optional linear solver settings after the positional block.
    return read_solver_options(lines, 10, filepath, config);
}

bool IO::read_turbulence(const std::string& filepath, TurbulenceConfig& config) {
    std::string error;
    if (!read_turbulence_config(filepath, config, error)) {
        std::cerr << "ERROR: *.turbulence FILE HAS INVALID FORMAT: " << error << "\n";
        return false;
    }
    return true;
}

bool IO::read_case_mesh(const std::string& constant_dir, const std::string& casename, Mesh& mesh) {
    MeshFileLocation mesh_file;
    if (!locate_mesh_file(constant_dir, casename, mesh_file)) return false;
    const std::string bofl_path = constant_dir + "/" + casename + ".bofl";
    switch (mesh_file.format) {
    case MeshFileFormat::Legacy: return read_mesh(mesh_file.path, mesh) && read_boundary_flags(bofl_path, mesh);
    case MeshFileFormat::Gmsh: return read_gmsh_mesh(mesh_file.path, bofl_path, mesh);
    case MeshFileFormat::Su2: return read_su2_mesh(mesh_file.path, bofl_path, mesh);
    case MeshFileFormat::Cgns: return read_cgns_mesh(mesh_file.path, bofl_path, mesh);
    }
    return false;
}

bool IO::read_mesh(const std::string& filepath, Mesh& mesh) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        std::cerr << "ERROR: MESH FILE DOES NOT EXIST: " << filepath << "\n";
        return false;
    }

    if (!(in >> mesh.nelem >> mesh.npoin >> mesh.nboun)) {
        std::cerr << "ERROR: CANNOT READ MESH DIMENSIONS\n";
        return false;
    }
    if (mesh.nelem < 1 || mesh.npoin < 1 || mesh.nboun < 0) {
        std::cerr << "ERROR: INVALID MESH DIMENSIONS: " << mesh.nelem << " ELEMENTS, "
                  << mesh.npoin << " NODES, " << mesh.nboun << " BOUNDARY SIDES\n";
        return false;
    }

    mesh.intma.resize(mesh.nelem);
    mesh.x.resize(mesh.npoin);
    mesh.y.resize(mesh.npoin);
    mesh.iside.resize(mesh.nboun);

    // Ids in the file are 1-based. Every id must be in range and appear only
    // once; together with the entry counts this guarantees that every element
    // and node is assigned exactly once (no gaps, no out-of-bounds writes).
    auto in_range = [](int id, int n) { return id >= 1 && id <= n; };

    std::vector<char> elem_seen(mesh.nelem, 0);
    for (int i = 0; i < mesh.nelem; ++i) {
        int elem_id, n1, n2, n3;
        if (!(in >> elem_id >> n1 >> n2 >> n3)) {
            std::cerr << "ERROR: CANNOT READ ELEMENT CONNECTIVITY AT ELEM " << (i + 1) << "\n";
            return false;
        }
        if (!in_range(elem_id, mesh.nelem)) {
            std::cerr << "ERROR: ELEMENT ID " << elem_id << " OUT OF RANGE 1.." << mesh.nelem
                      << " (ENTRY " << (i + 1) << " OF ELEMENT SECTION)\n";
            return false;
        }
        if (elem_seen[elem_id - 1]) {
            std::cerr << "ERROR: DUPLICATE ELEMENT ID " << elem_id
                      << " (ENTRY " << (i + 1) << " OF ELEMENT SECTION)\n";
            return false;
        }
        for (int n : {n1, n2, n3}) {
            if (!in_range(n, mesh.npoin)) {
                std::cerr << "ERROR: ELEMENT " << elem_id << " REFERENCES NODE " << n
                          << " OUT OF RANGE 1.." << mesh.npoin << "\n";
                return false;
            }
        }
        if (n1 == n2 || n1 == n3 || n2 == n3) {
            std::cerr << "ERROR: ELEMENT " << elem_id << " HAS REPEATED NODES "
                      << n1 << " " << n2 << " " << n3 << "\n";
            return false;
        }
        elem_seen[elem_id - 1] = 1;
        mesh.intma[elem_id - 1] = {n1 - 1, n2 - 1, n3 - 1};
    }

    std::vector<char> node_seen(mesh.npoin, 0);
    for (int i = 0; i < mesh.npoin; ++i) {
        int node_id;
        double x_coord, y_coord;
        if (!(in >> node_id >> x_coord >> y_coord)) {
            std::cerr << "ERROR: CANNOT READ NODAL COORDINATE AT NODE " << (i + 1) << "\n";
            return false;
        }
        if (!in_range(node_id, mesh.npoin)) {
            std::cerr << "ERROR: NODE ID " << node_id << " OUT OF RANGE 1.." << mesh.npoin
                      << " (ENTRY " << (i + 1) << " OF NODE SECTION)\n";
            return false;
        }
        if (node_seen[node_id - 1]) {
            std::cerr << "ERROR: DUPLICATE NODE ID " << node_id
                      << " (ENTRY " << (i + 1) << " OF NODE SECTION)\n";
            return false;
        }
        node_seen[node_id - 1] = 1;
        mesh.x[node_id - 1] = x_coord;
        mesh.y[node_id - 1] = y_coord;
    }

    for (int i = 0; i < mesh.nboun; ++i) {
        int n1, n2, elem_id, side_id;
        if (!(in >> n1 >> n2 >> elem_id >> side_id)) {
            std::cerr << "ERROR: CANNOT READ BOUNDARY SIDE AT SIDE " << (i + 1) << "\n";
            return false;
        }
        if (!in_range(n1, mesh.npoin) || !in_range(n2, mesh.npoin)) {
            std::cerr << "ERROR: BOUNDARY SIDE " << (i + 1) << " REFERENCES NODE "
                      << (in_range(n1, mesh.npoin) ? n2 : n1) << " OUT OF RANGE 1.." << mesh.npoin << "\n";
            return false;
        }
        if (!in_range(elem_id, mesh.nelem)) {
            std::cerr << "ERROR: BOUNDARY SIDE " << (i + 1) << " REFERENCES ELEMENT " << elem_id
                      << " OUT OF RANGE 1.." << mesh.nelem << "\n";
            return false;
        }
        mesh.iside[i].node1 = n1 - 1;
        mesh.iside[i].node2 = n2 - 1;
        mesh.iside[i].elem = elem_id - 1;
        mesh.iside[i].side_id = side_id;
        mesh.iside[i].flag = 0;
    }

    return true;
}

bool IO::read_boundary_flags(const std::string& filepath, Mesh& mesh) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        std::cerr << "ERROR: BOUNDARY FLAG FILE DOES NOT EXIST: " << filepath << "\n";
        return false;
    }

    int nflag = 0;
    if (!(in >> nflag)) {
        std::cerr << "ERROR: CANNOT READ NFLAG FROM BOUNDARY FILE\n";
        return false;
    }

    std::vector<std::pair<int, int>> flag_list(nflag);
    for (int i = 0; i < nflag; ++i) {
        in >> flag_list[i].first >> flag_list[i].second;
        int code = flag_list[i].second;
        if (code < 500 || code > 508) {
            std::cout << "WARNING: BOUNDARY FLAG CODE IN FILE DOES NOT MATCH KNOWN TYPES: " << code << "\n";
        }
    }

    for (int ib = 0; ib < mesh.nboun; ++ib) {
        bool found = false;
        for (int in_f = 0; in_f < nflag; ++in_f) {
            if (flag_list[in_f].first == mesh.iside[ib].side_id) {
                mesh.iside[ib].flag = flag_list[in_f].second;
                found = true;
                break;
            }
        }
        if (!found) {
            std::cerr << mesh.iside[ib].side_id << " ERROR: BOUNDARY SIDE NUMBER NOT IN BOUNDARY FLAG CODE FILE\n";
            return false;
        }
    }

    return true;
}

bool IO::locate_mesh_file(const std::string& constant_dir, const std::string& casename,
                          MeshFileLocation& location) {
    namespace fs = std::filesystem;
    struct Candidate {
        fs::path path;
        MeshFileFormat format;
        const char* description;
    };
    const Candidate candidates[] = {
        {fs::path(constant_dir) / (casename + ".mesh"), MeshFileFormat::Legacy, "legacy format"},
        {fs::path(constant_dir) / (casename + ".msh"), MeshFileFormat::Gmsh, "Gmsh"},
        {fs::path(constant_dir) / (casename + ".su2"), MeshFileFormat::Su2, "SU2"},
        {fs::path(constant_dir) / (casename + ".cgns"), MeshFileFormat::Cgns, "CGNS"},
    };
    std::vector<const Candidate*> found;
    std::error_code ec;
    for (const Candidate& c : candidates)
        if (fs::is_regular_file(c.path, ec)) found.push_back(&c);
    if (found.size() > 1) {
        std::cerr << "ERROR: AMBIGUOUS MESH:";
        for (const Candidate* c : found) std::cerr << " " << c->path.string();
        std::cerr << " EXIST; KEEP ONLY ONE MESH FILE PER CASE\n";
        return false;
    }
    if (found.empty()) {
        std::cerr << "ERROR: MESH FILE DOES NOT EXIST:";
        for (std::size_t i = 0; i < std::size(candidates); ++i) {
            std::cerr << (i == 0 ? " " : (i + 1 == std::size(candidates) ? " OR " : ", "))
                      << candidates[i].path.string() << " (" << candidates[i].description << ")";
        }
        std::cerr << "\n";
        return false;
    }
    location.path = found.front()->path.string();
    location.format = found.front()->format;
    return true;
}

bool IO::read_boundary_flag_table(const std::string& filepath, std::map<int, int>& table) {
    table.clear();
    std::ifstream in(filepath);
    if (!in.is_open()) {
        std::cerr << "ERROR: BOUNDARY FLAG FILE DOES NOT EXIST: " << filepath << "\n";
        return false;
    }
    std::string token;
    int nflag = 0;
    if (!(in >> token) || !parse_int(token, nflag) || nflag < 0) {
        std::cerr << "ERROR: " << filepath << ": THE FIRST VALUE MUST BE THE NUMBER OF ENTRIES\n";
        return false;
    }
    for (int i = 0; i < nflag; ++i) {
        std::string id_token, code_token;
        int id = 0, code = 0;
        if (!(in >> id_token >> code_token) || !parse_int(id_token, id) || !parse_int(code_token, code)) {
            std::cerr << "ERROR: " << filepath << ": ENTRY " << (i + 1) << " OF " << nflag
                      << " IS MISSING OR NOT A PAIR OF INTEGERS \"id code\"\n";
            return false;
        }
        if (!table.emplace(id, code).second) {
            std::cerr << "ERROR: " << filepath << ": DUPLICATE ENTRY FOR ID " << id << "\n";
            return false;
        }
    }
    return true;
}

namespace {

// Shared path of the mesh readers that produce mesh::MeshData: optional
// *.bofl overrides, build_mesh, report, error handling.
template <typename Reader>
bool import_mesh(const std::string& path, const std::string& bofl_path, const char* key_name, Reader read,
                 Mesh& mesh) {
    mesh::BuildOptions options;
    std::error_code ec;
    if (!bofl_path.empty() && std::filesystem::is_regular_file(bofl_path, ec)) {
        if (!IO::read_boundary_flag_table(bofl_path, options.flag_overrides)) return false;
        std::cout << "Boundary flag overrides: " << bofl_path << " (" << options.flag_overrides.size()
                  << " entries, by " << key_name << ")\n";
    }
    try {
        const mesh::MeshData data = read(path);
        mesh::ImportReport report;
        mesh = mesh::build_mesh(data, options, &report);
        std::cout << "Mesh file: " << path << "\n" << mesh::format_report(report);
    } catch (const mesh::MeshImportError& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return false;
    } catch (const std::exception& e) {
        // E.g. std::bad_alloc for absurd sizes in a corrupt file.
        std::cerr << "ERROR: " << path << ": cannot read the mesh (" << e.what() << "); is the file corrupt?\n";
        return false;
    }
    return true;
}

} // namespace

bool IO::read_gmsh_mesh(const std::string& msh_path, const std::string& bofl_path, Mesh& mesh) {
    return import_mesh(msh_path, bofl_path, "physical tag", mesh::read_gmsh, mesh);
}

bool IO::read_su2_mesh(const std::string& su2_path, const std::string& bofl_path, Mesh& mesh) {
    return import_mesh(su2_path, bofl_path, "marker number", mesh::read_su2, mesh);
}

bool IO::read_cgns_mesh(const std::string& cgns_path, const std::string& bofl_path, Mesh& mesh) {
    return import_mesh(cgns_path, bofl_path, "BC number", mesh::read_cgns, mesh);
}

static bool parse_restart_double(std::string token, double& out) {
    if (token.empty()) return false;
    for (char& c : token) {
        if (c == 'd' || c == 'D') {
            c = 'e';
        }
    }
    char* end = nullptr;
    errno = 0;
    double val = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0') {
        return false;
    }
    if (std::isnan(val) || std::isinf(val)) {
        return false;
    }
    if (errno == ERANGE && (val == HUGE_VAL || val == -HUGE_VAL)) {
        return false;
    }
    out = val;
    return true;
}

bool IO::read_restart(const std::string& filepath,
                      double& rtime, int& istart,
                      std::vector<double>& u,
                      std::vector<double>& v,
                      std::vector<double>& p) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        return false;
    }

    std::string token_time;
    if (!(in >> token_time >> istart)) {
        return false;
    }
    if (!parse_restart_double(token_time, rtime)) {
        return false;
    }

    size_t npoin = u.size();
    std::string tu, tv, tp;
    for (size_t i = 0; i < npoin; ++i) {
        if (!(in >> tu >> tv >> tp)) {
            return false;
        }
        if (!parse_restart_double(tu, u[i]) ||
            !parse_restart_double(tv, v[i]) ||
            !parse_restart_double(tp, p[i])) {
            return false;
        }
    }

    return true;
}

bool IO::write_restart(const std::string& filepath,
                       double rtime, int iitime,
                       int transient_on,
                       const std::vector<double>& u,
                       const std::vector<double>& v,
                       const std::vector<double>& p) {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        return false;
    }

    // 17 significant digits (max_digits10) so that every double - time and
    // fields - reads back bit-for-bit and restarted runs continue exactly.
    // In scientific notation the precision counts the digits after the point.
    out << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (transient_on > 0) {
        out << rtime << " " << iitime << "\n";
    } else {
        out << 0.0 << " " << iitime << "\n";
    }

    size_t npoin = u.size();
    for (size_t i = 0; i < npoin; ++i) {
        out << u[i] << " " << v[i] << " " << p[i] << "\n";
    }

    return true;
}

bool IO::write_vtk(const std::string& filepath,
                   const Mesh& mesh,
                   const std::vector<double>& u,
                   const std::vector<double>& v,
                   const std::vector<double>& p) {
    return write_vtk(filepath, mesh, u, v, p, {});
}

bool IO::write_vtk(const std::string& filepath,
                   const Mesh& mesh,
                   const std::vector<double>& u,
                   const std::vector<double>& v,
                   const std::vector<double>& p,
                   const std::vector<NamedField>& point_scalars) {
    for (const NamedField& f : point_scalars) {
        if (!f.values || static_cast<int>(f.values->size()) != mesh.npoin || f.name.empty() ||
            f.name.find_first_of(" \t\r\n") != std::string::npos) {
            std::cerr << "ERROR: INVALID VTK POINT FIELD '" << f.name << "'\n";
            return false;
        }
    }
    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "ERROR: CANNOT OPEN VTK FILE FOR WRITING: " << filepath << "\n";
        return false;
    }

    out << "# vtk DataFile Version 3.0\n";
    out << "Instananeous Variables Output\n";
    out << "ASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n";

    // POINTS
    out << "POINTS " << mesh.npoin << " double\n";
    out << std::setprecision(10) << std::scientific;
    for (int i = 0; i < mesh.npoin; ++i) {
        out << mesh.x[i] << " " << mesh.y[i] << " 0.0\n";
    }

    // CELLS: each cell line: number of nodes (3 or 4) followed by node IDs
    const int ntri = mesh.num_triangles(), nquad = mesh.num_quads();
    out << "CELLS " << mesh.nelem << " " << (ntri * 4 + nquad * 5) << "\n";
    for (int i = 0; i < ntri; ++i) {
        out << "3 " << mesh.intma[i][0] << " " << mesh.intma[i][1] << " " << mesh.intma[i][2] << "\n";
    }
    for (const auto& q : mesh.quads) {
        out << "4 " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "\n";
    }

    // CELL_TYPES: 5 for VTK_TRIANGLE, 9 for VTK_QUAD
    out << "CELL_TYPES " << mesh.nelem << "\n";
    for (int i = 0; i < ntri; ++i) {
        out << "5\n";
    }
    for (int i = 0; i < nquad; ++i) {
        out << "9\n";
    }

    // POINT_DATA
    out << "POINT_DATA " << mesh.npoin << "\n";

    // VECTORS velocity
    out << "VECTORS velocity float\n";
    out << std::setprecision(7) << std::scientific;
    for (int i = 0; i < mesh.npoin; ++i) {
        out << static_cast<float>(u[i]) << " "
            << static_cast<float>(v[i]) << " 0.0\n";
    }

    // SCALARS P
    out << "SCALARS P float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int i = 0; i < mesh.npoin; ++i) {
        out << static_cast<float>(p[i]) << "\n";
    }

    for (const NamedField& f : point_scalars) {
        out << "SCALARS " << f.name << " float 1\n";
        out << "LOOKUP_TABLE default\n";
        for (int i = 0; i < mesh.npoin; ++i) {
            out << static_cast<float>((*f.values)[i]) << "\n";
        }
    }

    return true;
}

namespace {

constexpr const char* kTurbulenceRestartMagic = "HydraMa2D-turbulence-restart";
constexpr int kTurbulenceRestartVersion = 1;

} // namespace

bool IO::write_turbulence_restart(const std::string& filepath,
                                  const std::string& model,
                                  const std::vector<NamedField>& fields) {
    if (fields.empty() || !fields[0].values) return false;
    const std::size_t npoin = fields[0].values->size();
    for (const NamedField& f : fields) {
        if (!f.values || f.values->size() != npoin || f.name.empty() ||
            f.name.find_first_of(" \t\r\n") != std::string::npos) {
            return false;
        }
    }
    std::ofstream out(filepath);
    if (!out.is_open()) return false;

    out << kTurbulenceRestartMagic << " " << kTurbulenceRestartVersion << "\n";
    out << "model " << model << "\n";
    out << "npoin " << npoin << "\n";
    out << "fields " << fields.size();
    for (const NamedField& f : fields) out << " " << f.name;
    out << "\n";
    out << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    for (std::size_t i = 0; i < npoin; ++i) {
        for (std::size_t k = 0; k < fields.size(); ++k) {
            if (k > 0) out << " ";
            out << (*fields[k].values)[i];
        }
        out << "\n";
    }
    return static_cast<bool>(out);
}

bool IO::read_turbulence_restart(const std::string& filepath, int npoin,
                                 TurbulenceRestartData& data, std::string& error) {
    data = TurbulenceRestartData{};
    std::ifstream in(filepath);
    if (!in.is_open()) {
        error = "cannot open " + filepath;
        return false;
    }
    std::string magic, key;
    int version = 0;
    if (!(in >> magic >> version) || magic != kTurbulenceRestartMagic || version != kTurbulenceRestartVersion) {
        error = "not a turbulence restart file (version " + std::to_string(kTurbulenceRestartVersion) + ")";
        return false;
    }
    if (!(in >> key >> data.model) || key != "model") {
        error = "missing model line";
        return false;
    }
    long long file_npoin = -1;
    if (!(in >> key >> file_npoin) || key != "npoin") {
        error = "missing npoin line";
        return false;
    }
    if (file_npoin != npoin) {
        error = "file has " + std::to_string(file_npoin) + " nodes, mesh has " + std::to_string(npoin);
        return false;
    }
    int nfields = 0;
    if (!(in >> key >> nfields) || key != "fields" || nfields < 1 || nfields > 16) {
        error = "missing or invalid fields line";
        return false;
    }
    data.names.resize(nfields);
    for (auto& name : data.names) {
        if (!(in >> name)) {
            error = "missing field name";
            return false;
        }
    }
    data.values.assign(nfields, std::vector<double>(npoin));
    std::string token;
    for (int i = 0; i < npoin; ++i) {
        for (int k = 0; k < nfields; ++k) {
            if (!(in >> token) || !parse_restart_double(token, data.values[k][i])) {
                error = "invalid or missing value for node " + std::to_string(i + 1) + ", field " + data.names[k];
                return false;
            }
        }
    }
    if (in >> token) {
        error = "unexpected trailing data";
        return false;
    }
    error.clear();
    return true;
}

} // namespace hydrama
