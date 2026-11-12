#pragma once

// Minimal SU2 native mesh writer for tests, with knobs for the optional parts
// of the format (point / element indices, section order, "NPOIN= n n",
// NZONE / IZONE headers, trailing FFD or periodic blocks, comments, CRLF).
// Its output for real meshes is checked against SU2 files written by Gmsh
// (tests/data/su2/*_gmsh.su2).

#include "Mesh.hpp"

#include <array>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace hydrama::test {

struct Su2Model {
    struct Element {
        int vtk_type = 5;                 // 5 triangle, 9 quadrilateral, 3 line, ...
        std::vector<std::int64_t> nodes;  // 0-based point indices
    };
    struct Marker {
        std::string name;
        std::vector<Element> elements;    // normally lines (type 3)
    };
    std::vector<double> x, y;
    std::vector<Element> elements;
    std::vector<Marker> markers;
};

struct Su2WriteOptions {
    bool point_index = true;     // "x y index"
    bool element_index = true;   // "type nodes... index" in NELEM
    bool comments = true;        // SU2-style '%' section comments
    bool points_first = false;   // NPOIN= before NELEM=
    bool npoin_domain = false;   // "NPOIN= n n"
    bool crlf = false;
    std::string prefix;          // inserted before NDIME= (e.g. "NZONE= 1\nIZONE= 1\n")
    std::string trailer;         // appended at the end (e.g. "FFD_NBOX= 0\n")
};

inline std::string su2_number(double v) {
    std::ostringstream os;
    os << std::setprecision(std::numeric_limits<double>::max_digits10) << v;
    return os.str();
}

inline std::string write_su2(const Su2Model& m, const Su2WriteOptions& o = {}) {
    std::ostringstream out;
    auto comment = [&](const char* text) {
        if (o.comments) out << "%\n% " << text << "\n%\n";
    };
    out << o.prefix;
    comment("Problem dimension");
    out << "NDIME= 2\n";

    auto elements = [&]() {
        comment("Inner element connectivity");
        out << "NELEM= " << m.elements.size() << "\n";
        for (std::size_t i = 0; i < m.elements.size(); ++i) {
            out << m.elements[i].vtk_type;
            for (std::int64_t n : m.elements[i].nodes) out << " " << n;
            if (o.element_index) out << " " << i;
            out << "\n";
        }
    };
    auto points = [&]() {
        comment("Node coordinates");
        out << "NPOIN= " << m.x.size();
        if (o.npoin_domain) out << " " << m.x.size();
        out << "\n";
        for (std::size_t i = 0; i < m.x.size(); ++i) {
            out << su2_number(m.x[i]) << " " << su2_number(m.y[i]);
            if (o.point_index) out << " " << i;
            out << "\n";
        }
    };
    if (o.points_first) {
        points();
        elements();
    } else {
        elements();
        points();
    }

    comment("Boundary elements");
    out << "NMARK= " << m.markers.size() << "\n";
    for (const auto& mk : m.markers) {
        out << "MARKER_TAG= " << mk.name << "\n";
        out << "MARKER_ELEMS= " << mk.elements.size() << "\n";
        for (const auto& e : mk.elements) {
            out << e.vtk_type;
            for (std::int64_t n : e.nodes) out << " " << n;
            out << "\n";
        }
    }
    out << o.trailer;

    std::string text = out.str();
    if (o.crlf) {
        std::string crlf;
        for (char c : text) {
            if (c == '\n') crlf += '\r';
            crlf += c;
        }
        return crlf;
    }
    return text;
}

// A Mesh as an SU2 model: one marker per boundary side id, in increasing id
// order, named by `names` (side_id -> name; default "marker_<id>"). A marker
// lists its sides in mesh order.
inline Su2Model su2_model_from_mesh(const Mesh& mesh, const std::map<int, std::string>& names) {
    Su2Model s;
    s.x = mesh.x;
    s.y = mesh.y;
    for (const auto& e : mesh.intma) s.elements.push_back({5, {e[0], e[1], e[2]}});
    std::map<int, std::size_t> marker_of_id;
    for (const auto& side : mesh.iside) marker_of_id.emplace(side.side_id, 0);
    for (auto& [id, index] : marker_of_id) {
        index = s.markers.size();
        const auto it = names.find(id);
        s.markers.push_back({it != names.end() ? it->second : "marker_" + std::to_string(id), {}});
    }
    for (const auto& side : mesh.iside)
        s.markers[marker_of_id.at(side.side_id)].elements.push_back({3, {side.node1, side.node2}});
    return s;
}

} // namespace hydrama::test
