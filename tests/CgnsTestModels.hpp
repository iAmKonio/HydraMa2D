#pragma once

// CGNS structures (mesh/CgnsModel.hpp) for tests and for the fixture
// generator (tests/data/cgns/generate_fixtures.cpp). No CGNS library needed.
//
// The channel is the mesh of tests/data/su2/channel_regular.su2: 8 x 4 cells
// on [0,4] x [0,1], two triangles per cell with alternating diagonals, vertex
// (i, j) = 1 + 9 j + i (1-based), boundaries inlet (x = 0), outlet (x = 4),
// upper wall (y = 1), lower wall (y = 0), each listed counter-clockwise.
// Every layout below describes that same mesh, so that imports can be
// compared with the SU2 fixture exactly.

#include "Mesh.hpp"
#include "mesh/CgnsModel.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hydrama::test::cgnsmodels {

using namespace hydrama::mesh::cgns;

constexpr int kNx = 8, kNy = 4;

inline std::int64_t vertex(int i, int j) { return 1 + j * (kNx + 1) + i; }

struct ChannelGeometry {
    std::vector<double> x, y;
    std::vector<std::array<std::int64_t, 3>> triangles;                  // counter-clockwise
    std::array<std::vector<std::array<std::int64_t, 2>>, 4> sides;       // inlet, outlet, upper, lower
};

inline ChannelGeometry channel_geometry() {
    ChannelGeometry g;
    for (int j = 0; j <= kNy; ++j) {
        for (int i = 0; i <= kNx; ++i) {
            g.x.push_back(4.0 * i / kNx);
            g.y.push_back(1.0 * j / kNy);
        }
    }
    for (int j = 0; j < kNy; ++j) {
        for (int i = 0; i < kNx; ++i) {
            const std::int64_t a = vertex(i, j), b = vertex(i + 1, j), c = vertex(i + 1, j + 1), d = vertex(i, j + 1);
            if ((i + j) % 2 == 0) {
                g.triangles.push_back({a, b, c});
                g.triangles.push_back({a, c, d});
            } else {
                g.triangles.push_back({a, b, d});
                g.triangles.push_back({b, c, d});
            }
        }
    }
    for (int j = 0; j < kNy; ++j) g.sides[0].push_back({vertex(0, j + 1), vertex(0, j)});
    for (int j = 0; j < kNy; ++j) g.sides[1].push_back({vertex(kNx, j), vertex(kNx, j + 1)});
    for (int i = 0; i < kNx; ++i) g.sides[2].push_back({vertex(i + 1, kNy), vertex(i, kNy)});
    for (int i = 0; i < kNx; ++i) g.sides[3].push_back({vertex(i, 0), vertex(i + 1, 0)});
    return g;
}

struct ChannelSpec {
    bool mixed = false;         // one MIXED section (bars first) instead of TRI_3 + 4 BAR_2 sections
    bool clockwise = false;     // triangles written clockwise
    bool with_z = false;        // PhysicalDimension 3 with CoordinateZ = 0
    int point_set = kElementRange;
    int location = kEdgeCenter; // for PointRange / PointList: kVertex -> BCs on vertices
    std::array<std::string, 4> names = {"inlet", "outlet", "upper_wall", "lower_wall"};
    std::array<int, 4> types = {kBCInflowSubsonic, kBCOutflow, kBCWallViscous, kBCWall};
    std::array<std::string, 4> families = {};   // FamilyName of each BC ("" none)
    std::vector<Family> family_nodes;           // Family_t nodes of the base
    std::string format = "CGNS 3.40 HDF5";
};

inline CgnsFile channel_model(const ChannelSpec& spec = {}) {
    const ChannelGeometry g = channel_geometry();
    Zone zone;
    zone.name = "Channel";
    zone.type = kUnstructured;
    zone.vertices = static_cast<std::int64_t>(g.x.size());
    zone.cells = static_cast<std::int64_t>(g.triangles.size());
    zone.coordinate_names = {"CoordinateX", "CoordinateY"};
    zone.coordinates = {g.x, g.y};
    if (spec.with_z) {
        zone.coordinate_names.push_back("CoordinateZ");
        zone.coordinates.push_back(std::vector<double>(g.x.size(), 0.0));
    }

    auto triangle = [&](const std::array<std::int64_t, 3>& t) {
        return spec.clockwise ? std::array<std::int64_t, 3>{t[0], t[2], t[1]} : t;
    };

    // Element numbering: sections layout: triangles 1..64, then the bars of
    // each boundary; mixed layout: the bars first (1..24), then the triangles.
    std::array<std::pair<std::int64_t, std::int64_t>, 4> bar_range{};
    if (!spec.mixed) {
        Section tris{"Fluid", kTri3, 1, static_cast<std::int64_t>(g.triangles.size()), {}, {}, true};
        for (const auto& t : g.triangles)
            for (std::int64_t v : triangle(t)) tris.connectivity.push_back(v);
        zone.sections.push_back(tris);
        std::int64_t next = tris.end + 1;
        for (int s = 0; s < 4; ++s) {
            Section bars{spec.names[static_cast<std::size_t>(s)] + "_edges", kBar2, next,
                         next + static_cast<std::int64_t>(g.sides[static_cast<std::size_t>(s)].size()) - 1, {}, {}, true};
            for (const auto& e : g.sides[static_cast<std::size_t>(s)]) bars.connectivity.insert(bars.connectivity.end(), e.begin(), e.end());
            bar_range[static_cast<std::size_t>(s)] = {bars.start, bars.end};
            next = bars.end + 1;
            zone.sections.push_back(bars);
        }
    } else {
        Section all{"All", kMixed, 1, 0, {}, {}, true};
        std::int64_t next = 1;
        for (int s = 0; s < 4; ++s) {
            bar_range[static_cast<std::size_t>(s)].first = next;
            for (const auto& e : g.sides[static_cast<std::size_t>(s)]) {
                all.offsets.push_back(static_cast<std::int64_t>(all.connectivity.size()));
                all.connectivity.insert(all.connectivity.end(), {kBar2, e[0], e[1]});
                ++next;
            }
            bar_range[static_cast<std::size_t>(s)].second = next - 1;
        }
        for (const auto& t : g.triangles) {
            all.offsets.push_back(static_cast<std::int64_t>(all.connectivity.size()));
            const auto tt = triangle(t);
            all.connectivity.insert(all.connectivity.end(), {kTri3, tt[0], tt[1], tt[2]});
            ++next;
        }
        all.offsets.push_back(static_cast<std::int64_t>(all.connectivity.size()));
        all.end = next - 1;
        zone.sections.push_back(all);
    }

    for (int s = 0; s < 4; ++s) {
        const std::size_t k = static_cast<std::size_t>(s);
        BoundaryCondition bc;
        bc.name = spec.names[k];
        bc.type = spec.types[k];
        bc.point_set = spec.point_set;
        bc.family = spec.families[k];
        const auto [first, last] = bar_range[k];
        const bool on_vertices = (spec.point_set == kPointRange || spec.point_set == kPointList) && spec.location == kVertex;
        bc.location = (spec.point_set == kPointRange || spec.point_set == kPointList) ? spec.location : kVertex;
        if (on_vertices) {
            // Every vertex of the side, each once, in increasing order.
            for (const auto& e : g.sides[k]) bc.points.insert(bc.points.end(), e.begin(), e.end());
            std::sort(bc.points.begin(), bc.points.end());
            bc.points.erase(std::unique(bc.points.begin(), bc.points.end()), bc.points.end());
            if (spec.point_set == kPointRange) bc.points = {bc.points.front(), bc.points.back()}; // contiguous ids only
        } else if (spec.point_set == kElementRange || spec.point_set == kPointRange) {
            bc.points = {first, last};
        } else {
            for (std::int64_t id = first; id <= last; ++id) bc.points.push_back(id);
        }
        zone.bcs.push_back(bc);
    }

    Base base;
    base.name = "Base";
    base.cell_dimension = 2;
    base.physical_dimension = spec.with_z ? 3 : 2;
    base.families = spec.family_nodes;
    base.zones.push_back(std::move(zone));
    CgnsFile file;
    file.format = spec.format;
    file.bases.push_back(std::move(base));
    return file;
}

// The channel with quadrilaterals in the first `quad_rows` rows of cells and
// triangles above: the mesh of tests/data/su2/mixed_tri_quad.su2 for
// quad_rows = 1 (an inflation layer on the lower wall). Sections "Quads"
// (QUAD_4), "Triangles" (TRI_3) and the boundary bars, ElementRange BCs.
inline CgnsFile channel_model_with_quads(int quad_rows) {
    CgnsFile f = channel_model();
    Zone& z = f.bases[0].zones[0];
    Section quads{"Quads", kQuad4, 1, 0, {}, {}, true};
    Section tris{"Triangles", kTri3, 0, 0, {}, {}, true};
    for (int j = 0; j < kNy; ++j) {
        for (int i = 0; i < kNx; ++i) {
            const std::int64_t a = vertex(i, j), b = vertex(i + 1, j), c = vertex(i + 1, j + 1), d = vertex(i, j + 1);
            if (j < quad_rows) {
                quads.connectivity.insert(quads.connectivity.end(), {a, b, c, d});
            } else if ((i + j) % 2 == 0) {
                tris.connectivity.insert(tris.connectivity.end(), {a, b, c, a, c, d});
            } else {
                tris.connectivity.insert(tris.connectivity.end(), {a, b, d, b, c, d});
            }
        }
    }
    quads.end = static_cast<std::int64_t>(quads.connectivity.size() / 4);
    tris.start = quads.end + 1;
    tris.end = quads.end + static_cast<std::int64_t>(tris.connectivity.size() / 3);
    std::vector<Section> sections;
    for (const Section& sec : {quads, tris})
        if (!sec.connectivity.empty()) sections.push_back(sec); // no empty sections
    if (tris.connectivity.empty()) tris.end = quads.end;
    std::int64_t next = tris.end + 1;
    for (std::size_t s = 1; s < z.sections.size(); ++s) { // boundary bars, renumbered
        Section bars = z.sections[s];
        const std::int64_t shift = bars.start - next;
        bars.start -= shift;
        bars.end -= shift;
        next = bars.end + 1;
        for (BoundaryCondition& bc : z.bcs)
            if (bc.points.size() == 2 && bc.points[0] == bars.start + shift) bc.points = {bars.start, bars.end};
        sections.push_back(bars);
    }
    z.sections = sections;
    z.cells = quads.end + static_cast<std::int64_t>(tris.connectivity.size() / 3);
    return f;
}

// A Mesh (legacy numbering) as CGNS: triangles in section "Fluid" (elements
// 1..nelem), then one BAR_2 section per boundary side id (increasing id,
// sides in mesh order) with an ElementRange BC named by `names` (side id ->
// name, default "bc_<id>") and of type `types` (default BCTypeNull).
inline CgnsFile cgns_model_from_mesh(const Mesh& mesh, const std::map<int, std::string>& names,
                                     const std::map<int, int>& types = {}) {
    Zone zone;
    zone.name = "Zone";
    zone.vertices = mesh.npoin;
    zone.cells = mesh.nelem;
    zone.coordinate_names = {"CoordinateX", "CoordinateY"};
    zone.coordinates = {mesh.x, mesh.y};
    Section tris{"Fluid", kTri3, 1, mesh.nelem, {}, {}, true};
    for (const auto& e : mesh.intma)
        for (int n : e) tris.connectivity.push_back(n + 1);
    zone.sections.push_back(tris);
    std::map<int, std::vector<const BoundarySide*>> by_id;
    for (const auto& s : mesh.iside) by_id[s.side_id].push_back(&s);
    std::int64_t next = mesh.nelem + 1;
    for (const auto& [id, sides] : by_id) {
        const auto name = names.find(id);
        const auto type = types.find(id);
        Section bars{"bars_" + std::to_string(id), kBar2, next, next + static_cast<std::int64_t>(sides.size()) - 1, {}, {}, true};
        for (const BoundarySide* s : sides) bars.connectivity.insert(bars.connectivity.end(), {s->node1 + 1, s->node2 + 1});
        BoundaryCondition bc;
        bc.name = name != names.end() ? name->second : "bc_" + std::to_string(id);
        bc.type = type != types.end() ? type->second : kBCTypeNull;
        bc.point_set = kElementRange;
        bc.points = {bars.start, bars.end};
        zone.bcs.push_back(bc);
        next = bars.end + 1;
        zone.sections.push_back(std::move(bars));
    }
    Base base;
    base.name = "Base";
    base.zones.push_back(std::move(zone));
    CgnsFile file;
    file.format = "CGNS";
    file.bases.push_back(std::move(base));
    return file;
}

} // namespace hydrama::test::cgnsmodels
