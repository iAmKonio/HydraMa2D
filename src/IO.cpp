#include "IO.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

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

bool IO::read_solver(const std::string& filepath, SolverConfig& config) {
    auto lines = read_non_empty_lines(filepath);
    if (lines.size() < 10) {
        std::cerr << "ERROR: *.solver FILE DOES NOT EXIST OR HAS INVALID FORMAT: " << filepath << "\n";
        return false;
    }

    config.solver_opt = std::stoi(lines[2]);
    config.relToler   = parse_fortran_double(lines[6]);
    config.absToler   = parse_fortran_double(lines[9]);

    return true;
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

    mesh.intma.resize(mesh.nelem);
    mesh.x.resize(mesh.npoin);
    mesh.y.resize(mesh.npoin);
    mesh.iside.resize(mesh.nboun);

    for (int i = 0; i < mesh.nelem; ++i) {
        int elem_id, n1, n2, n3;
        if (!(in >> elem_id >> n1 >> n2 >> n3)) {
            std::cerr << "ERROR: CANNOT READ ELEMENT CONNECTIVITY AT ELEM " << (i + 1) << "\n";
            return false;
        }
        mesh.intma[elem_id - 1] = {n1 - 1, n2 - 1, n3 - 1};
    }

    for (int i = 0; i < mesh.npoin; ++i) {
        int node_id;
        double x_coord, y_coord;
        if (!(in >> node_id >> x_coord >> y_coord)) {
            std::cerr << "ERROR: CANNOT READ NODAL COORDINATE AT NODE " << (i + 1) << "\n";
            return false;
        }
        mesh.x[node_id - 1] = x_coord;
        mesh.y[node_id - 1] = y_coord;
    }

    for (int i = 0; i < mesh.nboun; ++i) {
        int n1, n2, elem_id, side_id;
        if (!(in >> n1 >> n2 >> elem_id >> side_id)) {
            std::cerr << "ERROR: CANNOT READ BOUNDARY SIDE AT SIDE " << (i + 1) << "\n";
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
        if (code < 500 || code > 507) {
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

bool IO::read_restart(const std::string& filepath,
                      double& rtime, int& istart,
                      std::vector<double>& u,
                      std::vector<double>& v,
                      std::vector<double>& p) {
    std::ifstream in(filepath);
    if (!in.is_open()) {
        return false;
    }

    if (!(in >> rtime >> istart)) {
        return false;
    }

    size_t npoin = u.size();
    for (size_t i = 0; i < npoin; ++i) {
        if (!(in >> u[i] >> v[i] >> p[i])) {
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

    out << std::setprecision(10);
    if (transient_on > 0) {
        out << rtime << " " << iitime << "\n";
    } else {
        out << 0.0 << " " << iitime << "\n";
    }

    size_t npoin = u.size();
    out << std::setprecision(8) << std::scientific;
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

    // CELLS: each cell line: nep (3) followed by node IDs
    out << "CELLS " << mesh.nelem << " " << (mesh.nelem * 4) << "\n";
    for (int i = 0; i < mesh.nelem; ++i) {
        out << "3 " << mesh.intma[i][0] << " " << mesh.intma[i][1] << " " << mesh.intma[i][2] << "\n";
    }

    // CELL_TYPES: 5 for VTK_TRIANGLE
    out << "CELL_TYPES " << mesh.nelem << "\n";
    for (int i = 0; i < mesh.nelem; ++i) {
        out << "5\n";
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

    return true;
}

} // namespace hydrama
