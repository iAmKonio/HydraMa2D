#pragma once

// Minimal Gmsh MSH writer for tests: MSH 2.2 and 4.1, ASCII and binary, with
// knobs for the cases a robust reader must handle (arbitrary node tags,
// foreign byte order, CRLF line endings, 4-byte size_t, extra sections,
// elements in several physical groups).
//
// Its output is checked against Gmsh itself: the fixtures in
// tests/data/gmsh/ were written by the Gmsh SDK, and the test
// "Writer output matches real Gmsh files" compares the two on the same mesh.

#include "Mesh.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hydrama::test {

struct GmshModel {
    struct Element {
        int gmsh_type = 2;                 // 1 line, 2 triangle, 3 quadrangle, 15 point, ...
        std::vector<std::int64_t> nodes;   // node TAGS
        std::vector<int> physicals;        // physical tags (empty: none)
        std::int64_t tag = 0;              // 0: sequential
    };
    std::vector<std::int64_t> node_tags;   // node tags in file order
    std::vector<double> x, y, z;
    std::vector<Element> elements;         // in file order
    std::map<std::pair<int, int>, std::string> names; // (dim, tag) -> physical name

    void add_node(std::int64_t tag, double px, double py, double pz = 0.0) {
        node_tags.push_back(tag);
        x.push_back(px);
        y.push_back(py);
        z.push_back(pz);
    }
    void add_element(int type, std::vector<std::int64_t> nodes, std::vector<int> physicals = {}, std::int64_t tag = 0) {
        elements.push_back({type, std::move(nodes), std::move(physicals), tag});
    }
};

struct GmshWriteOptions {
    std::string version = "4.1";  // "2.2" or "4.1"
    bool binary = false;
    bool swap_bytes = false;      // binary: foreign byte order
    bool crlf = false;            // ASCII: Windows line endings
    int size_t_bytes = 8;         // 4.1 binary: width of size_t fields
    std::string extra_sections;   // raw text inserted after $MeshFormat
};

namespace detail {

inline int gmsh_nodes(int type) {
    switch (type) {
    case 1: return 2;
    case 2: return 3;
    case 3: return 4;
    case 15: return 1;
    case 9: return 6;
    case 4: return 4;
    default: return 0;
    }
}

inline int gmsh_dim(int type) {
    switch (type) {
    case 15: return 0;
    case 1: case 8: return 1;
    case 4: return 3;
    default: return 2;
    }
}

class Out {
public:
    Out(bool swap) : swap_(swap) {}
    template <typename T>
    void bin(T v) {
        unsigned char b[sizeof(T)];
        std::memcpy(b, &v, sizeof(T));
        if (swap_) std::reverse(b, b + sizeof(T));
        s.append(reinterpret_cast<const char*>(b), sizeof(T));
    }
    void text(const std::string& t) { s += t; }
    std::string s;

private:
    bool swap_;
};

inline std::string num(double v) {
    std::ostringstream os;
    os << std::setprecision(std::numeric_limits<double>::max_digits10) << v;
    return os.str();
}

} // namespace detail

inline std::string write_gmsh(const GmshModel& m, const GmshWriteOptions& o = {}) {
    using detail::Out;
    const bool v2 = o.version == "2.2";
    Out out(o.swap_bytes);
    auto size = [&](std::uint64_t v) {
        if (o.size_t_bytes == 4) out.bin(static_cast<std::uint32_t>(v));
        else out.bin(static_cast<std::uint64_t>(v));
    };
    auto sec_num = [&](std::uint64_t v) { // 4.1: size_t field
        if (o.binary) size(v);
        else out.text(std::to_string(v));
    };
    auto sep = [&](const char* s) {
        if (!o.binary) out.text(s);
    };

    out.text("$MeshFormat\n" + o.version + " " + (o.binary ? "1" : "0") + " " +
             std::to_string(v2 ? 8 : o.size_t_bytes) + "\n");
    if (o.binary) {
        out.bin(static_cast<std::int32_t>(1));
        out.text("\n");
    }
    out.text("$EndMeshFormat\n");
    out.text(o.extra_sections);

    if (!m.names.empty()) {
        out.text("$PhysicalNames\n" + std::to_string(m.names.size()) + "\n");
        for (const auto& [key, name] : m.names)
            out.text(std::to_string(key.first) + " " + std::to_string(key.second) + " \"" + name + "\"\n");
        out.text("$EndPhysicalNames\n");
    }

    const std::size_t nn = m.node_tags.size();
    std::vector<std::int64_t> etags(m.elements.size());
    for (std::size_t i = 0; i < m.elements.size(); ++i)
        etags[i] = m.elements[i].tag != 0 ? m.elements[i].tag : static_cast<std::int64_t>(i + 1);

    if (v2) {
        out.text("$Nodes\n" + std::to_string(nn) + "\n");
        for (std::size_t i = 0; i < nn; ++i) {
            if (o.binary) {
                out.bin(static_cast<std::int32_t>(m.node_tags[i]));
                out.bin(m.x[i]);
                out.bin(m.y[i]);
                out.bin(m.z[i]);
            } else {
                out.text(std::to_string(m.node_tags[i]) + " " + detail::num(m.x[i]) + " " + detail::num(m.y[i]) + " " +
                         detail::num(m.z[i]) + "\n");
            }
        }
        if (o.binary) out.text("\n");
        out.text("$EndNodes\n");

        // One record per (element, physical group), like Gmsh.
        std::size_t records = 0;
        for (const auto& e : m.elements) records += std::max<std::size_t>(1, e.physicals.size());
        out.text("$Elements\n" + std::to_string(records) + "\n");
        for (std::size_t i = 0; i < m.elements.size(); ++i) {
            const auto& e = m.elements[i];
            std::vector<int> phys = e.physicals;
            if (phys.empty()) phys.push_back(0);
            for (int p : phys) {
                if (o.binary) {
                    out.bin(static_cast<std::int32_t>(e.gmsh_type));
                    out.bin(static_cast<std::int32_t>(1));
                    out.bin(static_cast<std::int32_t>(2));
                    out.bin(static_cast<std::int32_t>(etags[i]));
                    out.bin(static_cast<std::int32_t>(p));
                    out.bin(static_cast<std::int32_t>(1));
                    for (std::int64_t n : e.nodes) out.bin(static_cast<std::int32_t>(n));
                } else {
                    std::string line = std::to_string(etags[i]) + " " + std::to_string(e.gmsh_type) + " 2 " +
                                       std::to_string(p) + " 1";
                    for (std::int64_t n : e.nodes) line += " " + std::to_string(n);
                    out.text(line + "\n");
                }
            }
        }
        if (o.binary) out.text("\n");
        out.text("$EndElements\n");
    } else {
        // Entities: one per run of consecutive elements of the same dimension
        // and physical groups (keeps the element order).
        struct Run {
            int dim, entity;
            int type;
            std::vector<int> phys;
            std::size_t first, last;
        };
        std::vector<Run> runs;
        int next_entity[4] = {1, 1, 1, 1};
        for (std::size_t i = 0; i < m.elements.size(); ++i) {
            const auto& e = m.elements[i];
            const int dim = detail::gmsh_dim(e.gmsh_type);
            if (!runs.empty() && runs.back().dim == dim && runs.back().phys == e.physicals &&
                runs.back().type == e.gmsh_type && runs.back().last == i) {
                runs.back().last = i + 1;
            } else {
                runs.push_back({dim, next_entity[dim]++, e.gmsh_type, e.physicals, i, i + 1});
            }
        }
        int counts[4] = {0, 0, 0, 0};
        for (const Run& r : runs) ++counts[r.dim];

        out.text("$Entities\n");
        for (int d = 0; d < 4; ++d) {
            sec_num(static_cast<std::uint64_t>(counts[d]));
            sep(d < 3 ? " " : "\n");
        }
        for (int d = 0; d < 4; ++d) {
            for (const Run& r : runs) {
                if (r.dim != d) continue;
                if (o.binary) out.bin(static_cast<std::int32_t>(r.entity));
                else out.text(std::to_string(r.entity));
                for (int k = 0; k < (d == 0 ? 3 : 6); ++k) {
                    if (o.binary) out.bin(0.0);
                    else out.text(" 0");
                }
                sep(" ");
                sec_num(r.phys.size());
                for (int p : r.phys) {
                    if (o.binary) out.bin(static_cast<std::int32_t>(p));
                    else out.text(" " + std::to_string(p));
                }
                if (d > 0) {
                    sep(" ");
                    sec_num(0);
                }
                sep("\n");
            }
        }
        if (o.binary) out.text("\n");
        out.text("$EndEntities\n");

        std::int64_t tmin = nn ? *std::min_element(m.node_tags.begin(), m.node_tags.end()) : 0;
        std::int64_t tmax = nn ? *std::max_element(m.node_tags.begin(), m.node_tags.end()) : 0;
        out.text("$Nodes\n");
        sec_num(1); sep(" "); sec_num(nn); sep(" "); sec_num(static_cast<std::uint64_t>(tmin)); sep(" ");
        sec_num(static_cast<std::uint64_t>(tmax)); sep("\n");
        if (o.binary) {
            out.bin(static_cast<std::int32_t>(2));
            out.bin(static_cast<std::int32_t>(1));
            out.bin(static_cast<std::int32_t>(0));
            size(nn);
            for (std::int64_t t : m.node_tags) size(static_cast<std::uint64_t>(t));
            for (std::size_t i = 0; i < nn; ++i) {
                out.bin(m.x[i]);
                out.bin(m.y[i]);
                out.bin(m.z[i]);
            }
            out.text("\n");
        } else {
            out.text("2 1 0 " + std::to_string(nn) + "\n");
            for (std::int64_t t : m.node_tags) out.text(std::to_string(t) + "\n");
            for (std::size_t i = 0; i < nn; ++i)
                out.text(detail::num(m.x[i]) + " " + detail::num(m.y[i]) + " " + detail::num(m.z[i]) + "\n");
        }
        out.text("$EndNodes\n");

        std::int64_t emin = etags.empty() ? 0 : *std::min_element(etags.begin(), etags.end());
        std::int64_t emax = etags.empty() ? 0 : *std::max_element(etags.begin(), etags.end());
        out.text("$Elements\n");
        sec_num(runs.size()); sep(" "); sec_num(m.elements.size()); sep(" ");
        sec_num(static_cast<std::uint64_t>(emin)); sep(" "); sec_num(static_cast<std::uint64_t>(emax)); sep("\n");
        for (const Run& r : runs) {
            if (o.binary) {
                out.bin(static_cast<std::int32_t>(r.dim));
                out.bin(static_cast<std::int32_t>(r.entity));
                out.bin(static_cast<std::int32_t>(r.type));
                size(r.last - r.first);
            } else {
                out.text(std::to_string(r.dim) + " " + std::to_string(r.entity) + " " + std::to_string(r.type) + " " +
                         std::to_string(r.last - r.first) + "\n");
            }
            for (std::size_t i = r.first; i < r.last; ++i) {
                if (o.binary) {
                    size(static_cast<std::uint64_t>(etags[i]));
                    for (std::int64_t n : m.elements[i].nodes) size(static_cast<std::uint64_t>(n));
                } else {
                    std::string line = std::to_string(etags[i]);
                    for (std::int64_t n : m.elements[i].nodes) line += " " + std::to_string(n);
                    out.text(line + "\n");
                }
            }
        }
        if (o.binary) out.text("\n");
        out.text("$EndElements\n");
    }

    if (o.crlf && !o.binary) {
        std::string crlf;
        for (char c : out.s) {
            if (c == '\n') crlf += '\r';
            crlf += c;
        }
        return crlf;
    }
    return out.s;
}

// A Mesh (legacy numbering, nodes tagged 1..npoin in order) as a Gmsh model:
// triangles in surface group `surface_physical` (0: none), boundary sides as
// lines in the physical group side_id, named by `side_names` (side_id -> name;
// ids without an entry stay unnamed).
inline GmshModel gmsh_model_from_mesh(const Mesh& mesh, const std::map<int, std::string>& side_names,
                                      int surface_physical = 0) {
    GmshModel g;
    for (int i = 0; i < mesh.npoin; ++i) g.add_node(i + 1, mesh.x[static_cast<std::size_t>(i)], mesh.y[static_cast<std::size_t>(i)]);
    std::vector<int> surf;
    if (surface_physical != 0) surf.push_back(surface_physical);
    for (const auto& e : mesh.intma) g.add_element(2, {e[0] + 1, e[1] + 1, e[2] + 1}, surf);
    for (const auto& s : mesh.iside) g.add_element(1, {s.node1 + 1, s.node2 + 1}, {s.side_id});
    for (const auto& [id, name] : side_names) g.names[{1, id}] = name;
    if (surface_physical != 0) g.names[{2, surface_physical}] = "fluid";
    return g;
}

} // namespace hydrama::test
