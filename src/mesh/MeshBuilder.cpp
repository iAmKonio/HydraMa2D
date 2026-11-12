#include "mesh/MeshBuilder.hpp"

#include "Problems.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>

namespace hydrama::mesh {

namespace {

using detail::Problems;


std::string fmt(double v) {
    std::ostringstream os;
    os << std::setprecision(10) << v;
    return os.str();
}

// Sorted node set of an element (duplicate detection).
template <int N>
struct NodeKey {
    std::array<int, N> n;
    bool operator==(const NodeKey& o) const { return n == o.n; }
};

struct NodeKeyHash {
    template <int N>
    std::size_t operator()(const NodeKey<N>& k) const {
        std::uint64_t h = 1469598103934665603ull;
        for (int v : k.n) {
            h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(v));
            h *= 1099511628211ull;
        }
        return static_cast<std::size_t>(h ^ (h >> 29));
    }
};

struct EdgeRecord {
    int elem = -1;  // first element containing the edge
    int from = -1;  // direction of the edge in that (counter-clockwise) element
    int to = -1;
    int count = 0;  // number of elements containing the edge
};

struct SideRecord {
    int from, to, elem;
    std::vector<int> tags; // physical tags (0 excluded), in order of appearance
};

struct InteriorLine {
    std::int64_t tag;
    int a, b;
    std::vector<int> tags;
};

struct LineRecord {
    int a, b; // file node indices
    int physical;
    std::int64_t tag;
};

struct GroupResolution {
    bool resolved = false;
    BoundaryPatch patch;
    std::string problem; // why it is not resolved
};

std::string group_label(int tag, const std::string& name) {
    if (name.empty()) return "physical group " + std::to_string(tag) + " (unnamed)";
    return "physical group '" + name + "' (tag " + std::to_string(tag) + ")";
}

} // namespace

Mesh build_mesh(const MeshData& data, const BuildOptions& options, ImportReport* report_out) {
    ImportReport report;
    report.format = data.format;
    report.nodes_in_file = static_cast<int>(data.node_tags.size());
    report.warnings = data.warnings;
    Problems problems(data.source.empty() ? std::string("mesh") : data.source, options.max_reported_problems);

    const std::size_t nfile = data.node_tags.size();
    if (data.x.size() != nfile || data.y.size() != nfile || data.z.size() != nfile) {
        throw MeshImportError(data.source + ": inconsistent node arrays");
    }
    if (nfile > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) {
        throw MeshImportError(data.source + ": too many nodes");
    }

    // ------------------------------------------------------------------
    // 1. Node tags -> file indices
    // ------------------------------------------------------------------
    std::unordered_map<std::int64_t, int> index_of_tag;
    index_of_tag.reserve(nfile);
    for (std::size_t i = 0; i < nfile; ++i) {
        if (!std::isfinite(data.x[i]) || !std::isfinite(data.y[i]) || !std::isfinite(data.z[i]))
            problems.add("node " + std::to_string(data.node_tags[i]) + " has a non-finite coordinate");
        if (!index_of_tag.emplace(data.node_tags[i], static_cast<int>(i)).second)
            problems.add("duplicate node tag " + std::to_string(data.node_tags[i]));
    }
    problems.checkpoint();

    auto node_label = [&](int file_index) {
        return "node " + std::to_string(data.node_tags[static_cast<std::size_t>(file_index)]) + " (" +
               fmt(data.x[static_cast<std::size_t>(file_index)]) + ", " +
               fmt(data.y[static_cast<std::size_t>(file_index)]) + ")";
    };

    // ------------------------------------------------------------------
    // 2. Elements: triangles and quadrilaterals (deduplicated) and lines
    // ------------------------------------------------------------------
    std::vector<std::array<int, 3>> cells; // triangles, file node indices
    std::vector<std::int64_t> cell_tags;
    std::vector<std::array<int, 4>> qcells; // quadrilaterals, file node indices
    std::vector<std::int64_t> qcell_tags;
    std::vector<LineRecord> lines;
    std::unordered_map<NodeKey<3>, int, NodeKeyHash> cell_of_key;
    std::unordered_map<NodeKey<4>, int, NodeKeyHash> qcell_of_key;

    auto kind_name = [](ElementType t) {
        return t == ElementType::Tri3 ? "triangle " : t == ElementType::Quad4 ? "quadrilateral " : "line element ";
    };
    for (const RawElement& e : data.elements) {
        const int nn = node_count(e.type);
        std::array<int, 4> n{};
        bool ok = true;
        for (int k = 0; k < nn; ++k) {
            const auto it = index_of_tag.find(e.nodes[static_cast<std::size_t>(k)]);
            if (it == index_of_tag.end()) {
                problems.add(kind_name(e.type) + std::to_string(e.tag) + " references undefined node " +
                             std::to_string(e.nodes[static_cast<std::size_t>(k)]));
                ok = false;
                break;
            }
            n[static_cast<std::size_t>(k)] = it->second;
        }
        if (!ok) continue;

        if (e.type == ElementType::Quad4) {
            NodeKey<4> key{n};
            std::sort(key.n.begin(), key.n.end());
            if (std::adjacent_find(key.n.begin(), key.n.end()) != key.n.end()) {
                problems.add("quadrilateral " + std::to_string(e.tag) + " has repeated nodes");
                continue;
            }
            if (!qcell_of_key.emplace(key, static_cast<int>(qcells.size())).second) {
                ++report.duplicate_elements_merged; // same quadrilateral in another physical group
                continue;
            }
            qcells.push_back(n);
            qcell_tags.push_back(e.tag);
            continue;
        }

        if (e.type == ElementType::Line2) {
            if (n[0] == n[1]) {
                problems.add("line element " + std::to_string(e.tag) + " has two identical nodes (" + node_label(n[0]) + ")");
                continue;
            }
            lines.push_back({n[0], n[1], e.physical, e.tag});
            continue;
        }

        if (n[0] == n[1] || n[0] == n[2] || n[1] == n[2]) {
            problems.add("triangle " + std::to_string(e.tag) + " has repeated nodes");
            continue;
        }
        NodeKey<3> key{{n[0], n[1], n[2]}};
        std::sort(key.n.begin(), key.n.end());
        if (!cell_of_key.emplace(key, static_cast<int>(cells.size())).second) {
            ++report.duplicate_elements_merged; // same triangle in another physical group
            continue;
        }
        cells.push_back({n[0], n[1], n[2]});
        cell_tags.push_back(e.tag);
    }
    if (cells.empty() && qcells.empty()) problems.add("the mesh contains no triangles or quadrilaterals");
    problems.checkpoint();
    const bool only_triangles = qcells.empty();
    const int ntri = static_cast<int>(cells.size());
    // Label of element e (triangles first, then quadrilaterals) in messages.
    auto element_label = [&](int e) {
        return e < ntri ? "triangle " + std::to_string(cell_tags[static_cast<std::size_t>(e)])
                        : "quadrilateral " + std::to_string(qcell_tags[static_cast<std::size_t>(e - ntri)]);
    };

    // ------------------------------------------------------------------
    // 3. Renumber the nodes used by elements, in file order
    // ------------------------------------------------------------------
    std::vector<int> new_index(nfile, -1);
    for (const auto& c : cells)
        for (int f : c) new_index[static_cast<std::size_t>(f)] = 0;
    for (const auto& c : qcells)
        for (int f : c) new_index[static_cast<std::size_t>(f)] = 0;
    std::vector<int> file_index_of; // new -> file
    for (std::size_t i = 0; i < nfile; ++i) {
        if (new_index[i] == 0) {
            new_index[i] = static_cast<int>(file_index_of.size());
            file_index_of.push_back(static_cast<int>(i));
        }
    }
    const int npoin = static_cast<int>(file_index_of.size());
    report.unused_nodes_dropped = static_cast<int>(nfile) - npoin;

    Mesh mesh;
    mesh.npoin = npoin;
    mesh.x.resize(static_cast<std::size_t>(npoin));
    mesh.y.resize(static_cast<std::size_t>(npoin));
    double xmin = HUGE_VAL, xmax = -HUGE_VAL, ymin = HUGE_VAL, ymax = -HUGE_VAL, zmin = HUGE_VAL, zmax = -HUGE_VAL;
    for (int i = 0; i < npoin; ++i) {
        const std::size_t f = static_cast<std::size_t>(file_index_of[static_cast<std::size_t>(i)]);
        mesh.x[static_cast<std::size_t>(i)] = data.x[f];
        mesh.y[static_cast<std::size_t>(i)] = data.y[f];
        xmin = std::min(xmin, data.x[f]);
        xmax = std::max(xmax, data.x[f]);
        ymin = std::min(ymin, data.y[f]);
        ymax = std::max(ymax, data.y[f]);
        zmin = std::min(zmin, data.z[f]);
        zmax = std::max(zmax, data.z[f]);
    }

    // ------------------------------------------------------------------
    // 4. Planarity: the mesh must lie in a plane z = const
    // ------------------------------------------------------------------
    const double extent = std::max(xmax - xmin, ymax - ymin);
    if (!(extent > 0.0)) {
        problems.add("the mesh has no extent in the x-y plane");
    } else if (zmax - zmin > options.planarity_tolerance * extent) {
        problems.add("the mesh is not planar in the x-y plane: z ranges from " + fmt(zmin) + " to " + fmt(zmax) +
                     " (HydraMa2D needs a 2D mesh in the x-y plane)");
    }
    problems.checkpoint();

    // ------------------------------------------------------------------
    // 5. Orientation: strictly positive signed area (counter-clockwise)
    // ------------------------------------------------------------------
    mesh.nelem = static_cast<int>(cells.size() + qcells.size());
    mesh.intma.resize(cells.size());
    mesh.quads.resize(qcells.size());
    for (std::size_t e = 0; e < cells.size(); ++e) {
        std::array<int, 3> n = {new_index[static_cast<std::size_t>(cells[e][0])],
                                new_index[static_cast<std::size_t>(cells[e][1])],
                                new_index[static_cast<std::size_t>(cells[e][2])]};
        const double x0 = mesh.x[static_cast<std::size_t>(n[0])], y0 = mesh.y[static_cast<std::size_t>(n[0])];
        const double x1 = mesh.x[static_cast<std::size_t>(n[1])], y1 = mesh.y[static_cast<std::size_t>(n[1])];
        const double x2 = mesh.x[static_cast<std::size_t>(n[2])], y2 = mesh.y[static_cast<std::size_t>(n[2])];
        const double twice_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        const double l01 = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
        const double l12 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
        const double l20 = (x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2);
        const double longest2 = std::max({l01, l12, l20});
        if (!(std::abs(twice_area) > options.degenerate_tolerance * longest2)) {
            problems.add("triangle " + std::to_string(cell_tags[e]) + " is degenerate (zero area): " +
                         node_label(cells[e][0]) + ", " + node_label(cells[e][1]) + ", " + node_label(cells[e][2]));
            continue;
        }
        if (twice_area < 0.0) {
            std::swap(n[1], n[2]);
            std::swap(cells[e][1], cells[e][2]);
            ++report.reoriented_triangles;
        }
        mesh.intma[e] = n;
    }
    // Quadrilaterals: det J of the bilinear map is affine in (xi, eta), so it
    // is positive on the element iff it is positive at the four corners, where
    // it is a quarter of the cross product of the two corner edges
    // (docs/QUADRILATERAL_ELEMENTS.md, section 4).
    for (std::size_t q = 0; q < qcells.size(); ++q) {
        std::array<int, 4> n{};
        double qx[4], qy[4];
        for (int a = 0; a < 4; ++a) {
            n[static_cast<std::size_t>(a)] = new_index[static_cast<std::size_t>(qcells[q][static_cast<std::size_t>(a)])];
            qx[a] = mesh.x[static_cast<std::size_t>(n[static_cast<std::size_t>(a)])];
            qy[a] = mesh.y[static_cast<std::size_t>(n[static_cast<std::size_t>(a)])];
        }
        double longest2 = 0.0;
        for (int a = 0; a < 4; ++a) {
            const int b = (a + 1) % 4;
            longest2 = std::max(longest2, (qx[b] - qx[a]) * (qx[b] - qx[a]) + (qy[b] - qy[a]) * (qy[b] - qy[a]));
        }
        const std::array<double, 4> corner = q1::corner_jacobians(qx, qy);
        const double tol = 0.25 * options.degenerate_tolerance * longest2;
        int positive = 0, negative = 0;
        for (double c : corner) {
            if (c > tol) ++positive;
            else if (c < -tol) ++negative;
        }
        auto corners = [&]() {
            return node_label(qcells[q][0]) + ", " + node_label(qcells[q][1]) + ", " + node_label(qcells[q][2]) + ", " +
                   node_label(qcells[q][3]);
        };
        if (positive + negative < 4) {
            problems.add("quadrilateral " + std::to_string(qcell_tags[q]) +
                         " is degenerate (a corner angle of 0 or 180 degrees: zero Jacobian determinant): " + corners());
            continue;
        }
        if (positive != 4 && negative != 4) {
            problems.add("quadrilateral " + std::to_string(qcell_tags[q]) +
                         " is not convex or self-intersecting (its Jacobian determinant changes sign): " + corners());
            continue;
        }
        if (negative == 4) { // clockwise: reverse, keeping the first node
            std::swap(n[1], n[3]);
            std::swap(qcells[q][1], qcells[q][3]);
            ++report.reoriented_quadrilaterals;
        }
        mesh.quads[q] = n;
    }
    problems.checkpoint();

    // ------------------------------------------------------------------
    // 6. Edges: conforming, manifold, consistently oriented
    // ------------------------------------------------------------------
    const auto edge_key = [npoin](int a, int b) {
        if (a > b) std::swap(a, b);
        return static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(npoin) + static_cast<std::uint64_t>(b);
    };
    std::unordered_map<std::uint64_t, EdgeRecord> edges;
    edges.reserve(cells.size() * 2 + qcells.size() * 2);
    for (int e = 0; e < mesh.nelem; ++e) {
        const ElementNodes n = mesh.element_nodes(e);
        for (int k = 0; k < n.size(); ++k) {
            const int a = n[k], b = n[(k + 1) % n.size()];
            EdgeRecord& r = edges[edge_key(a, b)];
            ++r.count;
            if (r.count == 1) {
                r.elem = e;
                r.from = a;
                r.to = b;
            } else if (r.count == 2 && r.from == a) {
                // Two counter-clockwise elements traverse a shared edge in
                // opposite directions unless they overlap.
                problems.add((only_triangles ? "triangles " + std::to_string(cell_tags[static_cast<std::size_t>(r.elem)]) +
                                                   " and " + std::to_string(cell_tags[static_cast<std::size_t>(e)])
                                             : element_label(r.elem) + " and " + element_label(e)) +
                             " overlap (both lie on the same side of their common edge " +
                             node_label(file_index_of[static_cast<std::size_t>(a)]) + " - " +
                             node_label(file_index_of[static_cast<std::size_t>(b)]) + ")");
            } else if (r.count == 3) {
                problems.add("edge " + node_label(file_index_of[static_cast<std::size_t>(a)]) + " - " +
                             node_label(file_index_of[static_cast<std::size_t>(b)]) +
                             " is shared by more than two " + (only_triangles ? "triangles" : "elements") +
                             " (non-manifold mesh)");
            }
        }
    }
    problems.checkpoint();

    // ------------------------------------------------------------------
    // 7. Line elements -> boundary sides (merged per edge)
    // ------------------------------------------------------------------
    std::vector<SideRecord> sides;
    std::unordered_map<std::uint64_t, int> side_of_edge;
    std::vector<InteriorLine> interior;
    std::unordered_map<std::uint64_t, int> interior_of_edge;
    for (const LineRecord& l : lines) {
        const int a = new_index[static_cast<std::size_t>(l.a)], b = new_index[static_cast<std::size_t>(l.b)];
        const auto it = (a < 0 || b < 0) ? edges.end() : edges.find(edge_key(a, b));
        if (it == edges.end()) {
            problems.add("line element " + std::to_string(l.tag) + " (" + node_label(l.a) + " - " + node_label(l.b) +
                         ") is not an edge of any " + (only_triangles ? "triangle" : "element"));
            continue;
        }
        const std::uint64_t key = it->first;
        if (it->second.count == 2) {
            auto [pos, added] = interior_of_edge.emplace(key, static_cast<int>(interior.size()));
            if (added) interior.push_back({l.tag, l.a, l.b, {}});
            else ++report.duplicate_elements_merged;
            auto& tags = interior[static_cast<std::size_t>(pos->second)].tags;
            if (l.physical != 0 && std::find(tags.begin(), tags.end(), l.physical) == tags.end()) tags.push_back(l.physical);
            continue;
        }
        auto [pos, added] = side_of_edge.emplace(key, static_cast<int>(sides.size()));
        if (added) sides.push_back({it->second.from, it->second.to, it->second.elem, {}});
        else ++report.duplicate_elements_merged;
        auto& tags = sides[static_cast<std::size_t>(pos->second)].tags;
        if (l.physical != 0 && std::find(tags.begin(), tags.end(), l.physical) == tags.end()) tags.push_back(l.physical);
    }
    problems.checkpoint();

    // ------------------------------------------------------------------
    // 8. Boundary conditions of the physical line groups
    // ------------------------------------------------------------------
    std::set<int> used_tags;
    for (const SideRecord& s : sides) used_tags.insert(s.tags.begin(), s.tags.end());
    for (const InteriorLine& l : interior) used_tags.insert(l.tags.begin(), l.tags.end());

    std::map<int, GroupResolution> groups;
    for (int tag : used_tags) {
        GroupResolution g;
        g.patch.tag = tag;
        g.patch.name = data.physical_name(1, tag);
        const auto ov = options.flag_overrides.find(tag);
        if (ov != options.flag_overrides.end()) {
            g.resolved = true;
            g.patch.flag = ov->second;
            g.patch.type = boundary_type_from_flag(ov->second);
            g.patch.source = ConditionSource::FlagFile;
            g.patch.matched = std::to_string(ov->second);
        } else if (const auto hint = data.boundary_hints.find({1, tag}); hint != data.boundary_hints.end()) {
            // Name combined with the boundary type declared in the file.
            const NameResolution r = resolve_boundary_name(g.patch.name, hint->second.type);
            if (r.resolved()) {
                g.resolved = true;
                g.patch.type = r.type;
                g.patch.flag = solver_flag(r.type).value_or(0);
                g.patch.source = r.from_hint ? ConditionSource::FileType : ConditionSource::Name;
                g.patch.matched = r.from_hint ? hint->second.source : r.matched + " + " + hint->second.source;
            } else if (r.status == NameResolution::Status::Conflict) {
                g.problem = group_label(tag, g.patch.name) + ": the name means " +
                            std::string(to_string(r.candidates.at(0))) + ", but " + hint->second.source + " means " +
                            std::string(to_string(r.candidates.at(1))) + "; make them agree";
            } else {
                g.problem = group_label(tag, g.patch.name) + ": " + r.message(g.patch.name) + " (declared type: " +
                            hint->second.source + ")";
            }
        } else if (!g.patch.name.empty()) {
            const NameResolution r = resolve_boundary_name(g.patch.name);
            if (r.resolved()) {
                g.resolved = true;
                g.patch.type = r.type;
                g.patch.flag = solver_flag(r.type).value_or(0);
                g.patch.matched = r.matched;
            } else {
                g.problem = group_label(tag, g.patch.name) + ": " + r.message(g.patch.name);
            }
        } else {
            g.problem = group_label(tag, "") +
                        ": the group has no name; name it with a boundary keyword (e.g. \"wall\", \"inlet\", "
                        "\"outlet\") or give tag " + std::to_string(tag) + " a flag code in the *.bofl file";
        }
        groups.emplace(tag, std::move(g));
    }
    for (const auto& [tag, code] : options.flag_overrides) {
        if (!used_tags.count(tag))
            report.warnings.push_back("*.bofl entry for physical tag " + std::to_string(tag) +
                                      " matches no boundary line group of the mesh");
    }

    // ------------------------------------------------------------------
    // 9. One condition per boundary side
    // ------------------------------------------------------------------
    std::vector<int> side_flag(sides.size(), 0), side_id(sides.size(), 0);
    std::set<int> reported_groups, needed_unresolved;
    int untagged = 0;
    std::string untagged_example;
    for (std::size_t i = 0; i < sides.size(); ++i) {
        const SideRecord& s = sides[i];
        std::vector<const GroupResolution*> resolved;
        for (int t : s.tags) {
            const GroupResolution& g = groups.at(t);
            if (g.resolved) resolved.push_back(&g);
        }
        const auto where = [&]() {
            return node_label(file_index_of[static_cast<std::size_t>(s.from)]) + " - " +
                   node_label(file_index_of[static_cast<std::size_t>(s.to)]);
        };
        if (resolved.empty()) {
            if (s.tags.empty()) {
                if (untagged++ == 0) untagged_example = where();
            }
            for (int t : s.tags) {
                needed_unresolved.insert(t);
                if (reported_groups.insert(t).second) problems.add(groups.at(t).problem);
            }
            continue;
        }
        const GroupResolution* first = resolved.front();
        for (const GroupResolution* g : resolved) {
            if (g->patch.flag != first->patch.flag || g->patch.type != first->patch.type) {
                problems.add("boundary side " + where() + " is in " + group_label(first->patch.tag, first->patch.name) +
                             " (" + std::string(to_string(first->patch.type)) + ") and in " +
                             group_label(g->patch.tag, g->patch.name) + " (" + std::string(to_string(g->patch.type)) +
                             "), which define different boundary conditions");
                break;
            }
        }
        int tag = first->patch.tag;
        for (const GroupResolution* g : resolved) tag = std::min(tag, g->patch.tag);
        side_id[i] = tag;
        side_flag[i] = first->patch.flag;
    }
    if (untagged > 0) {
        problems.add(std::to_string(untagged) + " boundary line element" + (untagged == 1 ? " is" : "s are") +
                     " in no physical group, e.g. " + untagged_example +
                     "; put every boundary curve in a named physical group");
    }

    // Every boundary edge needs a side.
    int uncovered = 0;
    const EdgeRecord* example = nullptr; // smallest key: deterministic message
    std::uint64_t example_key = 0;
    for (const auto& [key, r] : edges) {
        if (r.count == 1 && !side_of_edge.count(key)) {
            ++uncovered;
            if (!example || key < example_key) {
                example = &r;
                example_key = key;
            }
        }
    }
    if (uncovered > 0) {
        const std::string uncovered_example = node_label(file_index_of[static_cast<std::size_t>(example->from)]) +
                                              " - " + node_label(file_index_of[static_cast<std::size_t>(example->to)]);
        problems.add(std::to_string(uncovered) + " boundary edge" + (uncovered == 1 ? " has" : "s have") +
                     " no line element, e.g. " + uncovered_example +
                     "; put every boundary curve in a named physical group (Physical Curve)");
    }

    // Lines inside the domain: allowed only without a boundary condition.
    for (const InteriorLine& l : interior) {
        const GroupResolution* g = nullptr;
        for (int t : l.tags)
            if (groups.at(t).resolved) g = &groups.at(t);
        if (g) {
            problems.add("line element " + std::to_string(l.tag) + " (" + node_label(l.a) + " - " + node_label(l.b) +
                         ") of " + group_label(g->patch.tag, g->patch.name) + " (" +
                         std::string(to_string(g->patch.type)) +
                         ") lies inside the domain (shared by two elements); internal boundaries are not supported");
        } else {
            ++report.interior_lines_ignored;
        }
    }

    // Patches, and types the solver does not implement.
    for (const SideRecord& side : sides) {
        for (int t : side.tags) {
            GroupResolution& g = groups.at(t);
            if (g.resolved) ++g.patch.sides;
        }
    }
    for (auto& [tag, g] : groups) {
        if (!g.resolved || g.patch.sides == 0) continue;
        const BoundaryType type = g.patch.type;
        if (type == BoundaryType::Custom) {
            report.warnings.push_back(group_label(tag, g.patch.name) + ": flag code " + std::to_string(g.patch.flag) +
                                      " from the *.bofl file is not a known boundary type");
        } else if (!is_implemented(type)) {
            std::string description(boundary_type_info(type).description);
            if (!description.empty() && description.back() == '.') description.pop_back();
            const std::string msg = group_label(tag, g.patch.name) + " is a '" + std::string(to_string(type)) +
                                    "' boundary (" + description + "), which HydraMa2D does not implement yet";
            if (options.allow_unimplemented_boundaries) {
                report.warnings.push_back(msg + "; kept without a boundary condition (flag 0)");
            } else {
                problems.add(msg);
            }
        }
        report.patches.push_back(g.patch);
    }
    for (const auto& [tag, g] : groups) {
        if (!g.resolved && !needed_unresolved.count(tag)) {
            report.warnings.push_back(g.problem + " (ignored: its sides get their condition from other groups" +
                                      std::string(interior.empty() ? "" : " or lie inside the domain") + ")");
        }
    }
    if (report.interior_lines_ignored > 0) {
        report.warnings.push_back(std::to_string(report.interior_lines_ignored) +
                                  " line element(s) inside the domain without a boundary condition ignored");
    }
    problems.checkpoint();

    // ------------------------------------------------------------------
    // 10. Boundary sides of the Mesh
    // ------------------------------------------------------------------
    mesh.nboun = static_cast<int>(sides.size());
    mesh.iside.resize(sides.size());
    for (std::size_t i = 0; i < sides.size(); ++i) {
        BoundarySide& b = mesh.iside[i];
        b.node1 = sides[i].from;
        b.node2 = sides[i].to;
        b.elem = sides[i].elem;
        b.side_id = side_id[i];
        b.flag = side_flag[i];
    }
    for (const auto& [tag, g] : groups) {
        BoundaryGroup bg;
        bg.tag = tag;
        bg.name = g.patch.name;
        for (std::size_t i = 0; i < sides.size(); ++i)
            if (std::find(sides[i].tags.begin(), sides[i].tags.end(), tag) != sides[i].tags.end())
                bg.sides.push_back(static_cast<int>(i));
        if (!bg.sides.empty()) mesh.boundary_groups.push_back(std::move(bg));
    }

    report.triangles = mesh.num_triangles();
    report.quadrilaterals = mesh.num_quads();
    report.boundary_sides = mesh.nboun;
    if (report_out) *report_out = std::move(report);
    return mesh;
}

std::string format_report(const ImportReport& r) {
    std::ostringstream os;
    os << "Mesh format: " << r.format << "\n";
    os << "Mesh import: " << r.triangles << " triangles, ";
    if (r.quadrilaterals > 0) os << r.quadrilaterals << " quadrilaterals, ";
    os << (r.nodes_in_file - r.unused_nodes_dropped) << " nodes, " << r.boundary_sides << " boundary sides\n";
    if (r.unused_nodes_dropped > 0)
        os << "Mesh import: " << r.unused_nodes_dropped << " node(s) not used by any triangle dropped\n";
    if (r.reoriented_triangles > 0)
        os << "Mesh import: " << r.reoriented_triangles << " clockwise triangle(s) reoriented counter-clockwise\n";
    if (r.reoriented_quadrilaterals > 0)
        os << "Mesh import: " << r.reoriented_quadrilaterals
           << " clockwise quadrilateral(s) reoriented counter-clockwise\n";
    if (r.duplicate_elements_merged > 0)
        os << "Mesh import: " << r.duplicate_elements_merged
           << " duplicate element record(s) merged (elements in several physical groups)\n";
    for (const BoundaryPatch& p : r.patches) {
        os << "Boundary " << (p.name.empty() ? "(unnamed)" : "'" + p.name + "'") << " (tag " << p.tag
           << "): " << to_string(p.type) << ", flag " << p.flag << ", " << p.sides << " sides ["
           << (p.source == ConditionSource::FlagFile ? "*.bofl"
                                                      : p.source == ConditionSource::FileType ? "declared type" : "name")
           << (p.source == ConditionSource::Name && p.matched.find(" + ") != std::string::npos ? " + declared type" : "")
           << "]\n";
    }
    for (const std::string& w : r.warnings) os << "WARNING: " << w << "\n";
    return os.str();
}

} // namespace hydrama::mesh
