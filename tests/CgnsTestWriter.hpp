#pragma once

// Writes a CGNS structure (mesh/CgnsModel.hpp) with the CGNS library, in HDF5
// or ADF. Only available in builds with CGNS (HYDRAMA2D_HAVE_CGNS). The files
// it writes are checked with the CGNS project's own validator (cgnscheck),
// see tests/data/cgns/README.md.

#if defined(HYDRAMA2D_HAVE_CGNS) && HYDRAMA2D_HAVE_CGNS

#include "mesh/CgnsModel.hpp"

#include <cgnslib.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace hydrama::test {

class CgnsWriteError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace cgns_writer_detail {

inline void check(int status, const std::string& what) {
    if (status != CG_OK) throw CgnsWriteError("CGNS write error (" + what + "): " + cg_get_error());
}

inline std::vector<cgsize_t> narrow(const std::vector<std::int64_t>& v) {
    return std::vector<cgsize_t>(v.begin(), v.end());
}

// ElementStartOffset of a MIXED section from its inline type codes.
inline std::vector<cgsize_t> mixed_offsets(const mesh::cgns::Section& s) {
    std::vector<cgsize_t> offsets;
    std::size_t pos = 0;
    for (std::int64_t k = s.start; k <= s.end; ++k) {
        offsets.push_back(static_cast<cgsize_t>(pos));
        int npe = 0;
        check(cg_npe(static_cast<CGNS_ENUMT(ElementType_t)>(s.connectivity.at(pos)), &npe), "cg_npe");
        pos += 1 + static_cast<std::size_t>(npe);
    }
    offsets.push_back(static_cast<cgsize_t>(pos));
    return offsets;
}

} // namespace cgns_writer_detail

enum class CgnsFileKind { Hdf5, Adf };

inline void write_cgns(const mesh::cgns::CgnsFile& file, const std::string& path, CgnsFileKind kind = CgnsFileKind::Hdf5) {
    using namespace mesh::cgns;
    using cgns_writer_detail::check;
    using cgns_writer_detail::narrow;

    check(cg_set_file_type(kind == CgnsFileKind::Adf ? CG_FILE_ADF : CG_FILE_HDF5), "file type");
    int fn = 0;
    check(cg_open(path.c_str(), CG_MODE_WRITE, &fn), "open " + path);
    try {
        for (const Base& base : file.bases) {
            int b = 0;
            check(cg_base_write(fn, base.name.c_str(), base.cell_dimension, base.physical_dimension, &b), "base");
            for (const Family& fam : base.families) {
                int f = 0;
                check(cg_family_write(fn, b, fam.name.c_str(), &f), "family");
                if (fam.has_bc) {
                    int fbc = 0;
                    check(cg_fambc_write(fn, b, f, "FamBC", static_cast<CGNS_ENUMT(BCType_t)>(fam.bc_type), &fbc), "family BC");
                }
            }
            for (const Zone& zone : base.zones) {
                int z = 0;
                if (zone.type == kStructured) {
                    // A 3 x 3 structured block (index dimension 2).
                    cgsize_t size[6] = {3, 3, 2, 2, 0, 0};
                    check(cg_zone_write(fn, b, zone.name.c_str(), size, CGNS_ENUMV(Structured), &z), "zone");
                    continue;
                }
                cgsize_t size[3] = {static_cast<cgsize_t>(zone.vertices), static_cast<cgsize_t>(zone.cells), 0};
                check(cg_zone_write(fn, b, zone.name.c_str(), size, CGNS_ENUMV(Unstructured), &z), "zone");
                for (std::size_t c = 0; c < zone.coordinate_names.size(); ++c) {
                    int ci = 0;
                    check(cg_coord_write(fn, b, z, CGNS_ENUMV(RealDouble), zone.coordinate_names[c].c_str(),
                                         zone.coordinates[c].data(), &ci),
                          "coordinates");
                }
                for (const Section& s : zone.sections) {
                    int si = 0;
                    const std::vector<cgsize_t> conn = narrow(s.connectivity);
                    if (s.type == kMixed || s.type == kNGon || s.type == kNFace) {
                        const std::vector<cgsize_t> offsets =
                            s.type == kMixed ? cgns_writer_detail::mixed_offsets(s) : narrow(s.offsets);
                        check(cg_poly_section_write(fn, b, z, s.name.c_str(), static_cast<CGNS_ENUMT(ElementType_t)>(s.type),
                                                    static_cast<cgsize_t>(s.start), static_cast<cgsize_t>(s.end), 0,
                                                    conn.data(), offsets.data(), &si),
                              "section " + s.name);
                    } else {
                        check(cg_section_write(fn, b, z, s.name.c_str(), static_cast<CGNS_ENUMT(ElementType_t)>(s.type),
                                               static_cast<cgsize_t>(s.start), static_cast<cgsize_t>(s.end), 0,
                                               conn.data(), &si),
                              "section " + s.name);
                    }
                }
                for (const BoundaryCondition& bc : zone.bcs) {
                    int bi = 0;
                    const std::vector<cgsize_t> points = narrow(bc.points);
                    check(cg_boco_write(fn, b, z, bc.name.c_str(), static_cast<CGNS_ENUMT(BCType_t)>(bc.type),
                                        static_cast<CGNS_ENUMT(PointSetType_t)>(bc.point_set),
                                        static_cast<cgsize_t>(points.size()), points.data(), &bi),
                          "BC " + bc.name);
                    if (bc.point_set == kPointRange || bc.point_set == kPointList) {
                        check(cg_boco_gridlocation_write(fn, b, z, bi, static_cast<CGNS_ENUMT(GridLocation_t)>(bc.location)),
                              "BC location");
                    }
                    if (!bc.family.empty()) {
                        check(cg_goto(fn, b, "Zone_t", z, "ZoneBC_t", 1, "BC_t", bi, "end"), "BC node");
                        check(cg_famname_write(bc.family.c_str()), "BC family");
                    }
                }
            }
        }
    } catch (...) {
        cg_close(fn);
        cg_set_file_type(CG_FILE_NONE);
        throw;
    }
    check(cg_close(fn), "close");
    cg_set_file_type(CG_FILE_NONE);
}

} // namespace hydrama::test

#endif
