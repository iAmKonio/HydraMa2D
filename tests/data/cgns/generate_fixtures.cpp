// Writes the synthetic CGNS fixtures of tests/data/cgns/ (see README.md there).
//
//   cmake --build build --target hydrama2d_cgns_fixtures
//   cd tests/data/cgns && ../../../build/tests/hydrama2d_cgns_fixtures
//
// Build it against the OLDEST supported CGNS (3.4): CGNS 3.x refuses files
// written by CGNS 4.x, while 4.x reads 3.x files.

#include "CgnsTestModels.hpp"
#include "CgnsTestWriter.hpp"

#include <cstdio>
#include <filesystem>

using namespace hydrama::test;
using namespace hydrama::test::cgnsmodels;

namespace {

void write(const CgnsFile& f, const std::string& name, CgnsFileKind kind = CgnsFileKind::Hdf5) {
    write_cgns(f, name, kind);
    std::printf("  %s\n", name.c_str());
}

// Channel with the triangles of one cell replaced by a section of `type`.
CgnsFile channel_with_section(int type, std::vector<std::int64_t> connectivity, const std::string& name) {
    CgnsFile f = channel_model();
    Zone& z = f.bases[0].zones[0];
    const std::int64_t start = z.sections.back().end + 1;
    const std::int64_t npe = static_cast<std::int64_t>(element_type_info(type).nodes);
    const std::int64_t count = npe > 0 ? static_cast<std::int64_t>(connectivity.size()) / npe : 1;
    z.sections.push_back({name, type, start, start + count - 1, std::move(connectivity), {}, true});
    return f;
}

} // namespace

int main() {
    std::printf("CGNS fixtures written with CGNS %d:\n", CGNS_VERSION);

    // --- valid meshes -----------------------------------------------------
    write(channel_model(), "channel_sections.cgns");
    write(channel_model(), "channel_sections_adf.cgns", CgnsFileKind::Adf);

    ChannelSpec vertex_bcs;
    vertex_bcs.mixed = true;
    vertex_bcs.point_set = kPointList;
    vertex_bcs.location = kVertex;
    vertex_bcs.names = {"left", "right", "top", "bottom"};
    vertex_bcs.types = {kBCInflow, kBCOutflow, kBCWall, kBCSymmetryPlane};
    write(channel_model(vertex_bcs), "channel_mixed_vertex_bcs.cgns");

    ChannelSpec families;
    families.clockwise = true;
    families.with_z = true;
    families.point_set = kPointRange;
    families.location = kEdgeCenter;
    families.names = {"BC_1", "BC_2", "BC_3", "BC_4"};
    families.types = {kFamilySpecified, kFamilySpecified, kFamilySpecified, kFamilySpecified};
    families.families = {"Inflow", "Outflow", "Walls", "Walls"};
    families.family_nodes = {{"Inflow", true, kBCInflow}, {"Outflow", true, kBCOutflow}, {"Walls", true, kBCWall}};
    write(channel_model(families), "channel_families_clockwise.cgns");

    // --- meshes HydraMa2D rejects (small ADF files) -------------------------
    {
        CgnsFile f = channel_model();
        f.bases[0].cell_dimension = 3;
        f.bases[0].physical_dimension = 3;
        Zone& z = f.bases[0].zones[0];
        z.coordinate_names.push_back("CoordinateZ");
        z.coordinates.push_back(std::vector<double>(z.coordinates[0].size(), 0.0));
        z.sections.push_back({"Tets", kTetra4, 89, 89, {1, 2, 10, 11}, {}, true});
        write(f, "unsupported_3d.cgns", CgnsFileKind::Adf);
    }
    {
        // Quadrilateral cells (one per channel cell) with the boundary bars.
        CgnsFile f = channel_model();
        Zone& z = f.bases[0].zones[0];
        Section quads{"Quads", kQuad4, 1, kNx * kNy, {}, {}, true};
        for (int j = 0; j < kNy; ++j)
            for (int i = 0; i < kNx; ++i)
                quads.connectivity.insert(quads.connectivity.end(),
                                          {vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)});
        const std::int64_t shift = 64 - quads.end;
        z.sections[0] = quads;
        for (std::size_t s = 1; s < z.sections.size(); ++s) {
            z.sections[s].start -= shift;
            z.sections[s].end -= shift;
        }
        for (BoundaryCondition& bc : z.bcs)
            for (std::int64_t& p : bc.points) p -= shift;
        z.cells = kNx * kNy;
        write(f, "channel_quads.cgns", CgnsFileKind::Adf);
    }
    write(channel_with_section(kTri6, {1, 3, 19, 2, 11, 10}, "Curved"), "unsupported_tri6.cgns", CgnsFileKind::Adf);
    {
        CgnsFile f = channel_model();
        Zone& z = f.bases[0].zones[0];
        // Two polygons: element start offsets 0, 3, 7.
        z.sections.push_back({"Polygons", kNGon, 89, 90, {1, 2, 10, 2, 3, 12, 11}, {0, 3, 7}, true});
        write(f, "unsupported_ngon.cgns", CgnsFileKind::Adf);
    }
    {
        ChannelSpec s;
        s.mixed = true;
        CgnsFile f = channel_model(s);
        Section& all = f.bases[0].zones[0].sections[0];
        all.connectivity.insert(all.connectivity.end(), {kHexa8, 1, 2, 11, 10, 1, 2, 11, 10});
        all.end += 1;
        write(f, "unsupported_mixed_hexa.cgns", CgnsFileKind::Adf);
    }
    {
        CgnsFile f = channel_model();
        Zone second = f.bases[0].zones[0];
        second.name = "Channel2";
        f.bases[0].zones.push_back(second);
        write(f, "unsupported_multizone.cgns", CgnsFileKind::Adf);
    }
    {
        CgnsFile f;
        Base base;
        base.name = "Base";
        Zone z;
        z.name = "Block";
        z.type = kStructured;
        base.zones.push_back(z);
        f.bases.push_back(base);
        write(f, "unsupported_structured.cgns", CgnsFileKind::Adf);
    }
    {
        ChannelSpec s;
        s.names = {"inlet", "outlet", "upper_wall", "lower_wall"};
        s.types = {kBCWall, kBCOutflow, kBCWall, kBCWall}; // "inlet" declared as a wall
        write(channel_model(s), "invalid_name_type_conflict.cgns", CgnsFileKind::Adf);
    }
    return 0;
}
