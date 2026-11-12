// CGNS library layer: reads the CgnsModel.hpp structure from a file.
// Compiled with the CGNS library only if HYDRAMA2D_HAVE_CGNS is defined (set
// by CMake when the library is found, see cmake/FindCGNS.cmake); otherwise
// read_cgns_file() reports that CGNS support is not available.

#include "mesh/CgnsReader.hpp"

#include <string>

#if defined(HYDRAMA2D_HAVE_CGNS) && HYDRAMA2D_HAVE_CGNS

#include <cgnslib.h>

#include <cstdio>
#include <filesystem>
#include <limits>
#include <vector>

namespace hydrama::mesh {

using namespace cgns;

// The model uses the codes of the CGNS standard; check them against the library.
#define HYDRAMA2D_SAME(ours, theirs) static_assert(static_cast<int>(ours) == static_cast<int>(CGNS_ENUMV(theirs)))
HYDRAMA2D_SAME(kNode, NODE);
HYDRAMA2D_SAME(kBar2, BAR_2);
HYDRAMA2D_SAME(kTri3, TRI_3);
HYDRAMA2D_SAME(kTri6, TRI_6);
HYDRAMA2D_SAME(kQuad4, QUAD_4);
HYDRAMA2D_SAME(kTetra4, TETRA_4);
HYDRAMA2D_SAME(kHexa8, HEXA_8);
HYDRAMA2D_SAME(kMixed, MIXED);
HYDRAMA2D_SAME(kNGon, NGON_n);
HYDRAMA2D_SAME(kNFace, NFACE_n);
HYDRAMA2D_SAME(kLastElementType, HEXA_125);
HYDRAMA2D_SAME(kBCAxisymmetricWedge, BCAxisymmetricWedge);
HYDRAMA2D_SAME(kBCDegenerateLine, BCDegenerateLine);
HYDRAMA2D_SAME(kBCFarfield, BCFarfield);
HYDRAMA2D_SAME(kBCInflow, BCInflow);
HYDRAMA2D_SAME(kBCOutflowSupersonic, BCOutflowSupersonic);
HYDRAMA2D_SAME(kBCSymmetryPlane, BCSymmetryPlane);
HYDRAMA2D_SAME(kBCWall, BCWall);
HYDRAMA2D_SAME(kBCWallViscousIsothermal, BCWallViscousIsothermal);
HYDRAMA2D_SAME(kFamilySpecified, FamilySpecified);
HYDRAMA2D_SAME(kPointList, PointList);
HYDRAMA2D_SAME(kPointRange, PointRange);
HYDRAMA2D_SAME(kElementRange, ElementRange);
HYDRAMA2D_SAME(kElementList, ElementList);
HYDRAMA2D_SAME(kCellListDonor, CellListDonor);
HYDRAMA2D_SAME(kVertex, Vertex);
HYDRAMA2D_SAME(kCellCenter, CellCenter);
HYDRAMA2D_SAME(kFaceCenter, FaceCenter);
HYDRAMA2D_SAME(kEdgeCenter, EdgeCenter);
HYDRAMA2D_SAME(kStructured, Structured);
HYDRAMA2D_SAME(kUnstructured, Unstructured);
#undef HYDRAMA2D_SAME

namespace {

class CgnsFileHandle {
public:
    explicit CgnsFileHandle(const std::string& path) : path_(path) {
        std::error_code ec;
        if (!std::filesystem::is_regular_file(path, ec)) throw MeshImportError(path + ": cannot open mesh file");
        // Detect ADF / HDF5 and open the file as that type. CGNS 4.x no longer
        // detects the type in cg_open once cg_set_file_type() has been called
        // in the process (CGNS 3.x always detects it).
        int file_type = 0;
        if (cg_is_cgns(path.c_str(), &file_type) != CG_OK)
            throw MeshImportError(path + ": cannot open CGNS file: not a CGNS file (ADF or HDF5)");
        cg_set_file_type(file_type);
        if (cg_open(path.c_str(), CG_MODE_READ, &fn_) != CG_OK)
            throw MeshImportError(path + ": cannot open CGNS file: " + cg_get_error());
    }
    ~CgnsFileHandle() { cg_close(fn_); }
    CgnsFileHandle(const CgnsFileHandle&) = delete;
    CgnsFileHandle& operator=(const CgnsFileHandle&) = delete;

    int fn() const { return fn_; }

    // Throws with the CGNS error message if a library call failed.
    void check(int status, const std::string& what) const {
        if (status != CG_OK) throw MeshImportError(path_ + ": CGNS error while reading " + what + ": " + cg_get_error());
    }

private:
    std::string path_;
    int fn_ = -1;
};

// Sizes from the file are used for allocations: reject absurd values.
std::int64_t checked_count(const CgnsFileHandle& f, cgsize_t n, const std::string& what) {
    if (n < 0 || static_cast<std::int64_t>(n) > std::numeric_limits<int>::max())
        f.check(CG_ERROR, what + " (invalid size " + std::to_string(static_cast<long long>(n)) + ")");
    return static_cast<std::int64_t>(n);
}

std::vector<std::int64_t> widen(const std::vector<cgsize_t>& v) { return std::vector<std::int64_t>(v.begin(), v.end()); }

void read_zone(const CgnsFileHandle& f, int b, int z, Zone& zone) {
    const int fn = f.fn();
    const std::string zlabel = "zone " + std::to_string(z);
    CGNS_ENUMT(ZoneType_t) ztype;
    f.check(cg_zone_type(fn, b, z, &ztype), zlabel);
    zone.type = static_cast<int>(ztype);
    char name[64] = {};
    cgsize_t size[9] = {};
    f.check(cg_zone_read(fn, b, z, name, size), zlabel);
    zone.name = name;
    if (zone.type != kUnstructured) return; // reported by cgns_to_mesh_data

    zone.vertices = checked_count(f, size[0], zlabel + " vertex count");
    zone.cells = checked_count(f, size[1], zlabel + " cell count");
    f.check(cg_ngrids(fn, b, z, &zone.grids), zlabel + " grids");

    int ncoords = 0;
    f.check(cg_ncoords(fn, b, z, &ncoords), zlabel + " coordinates");
    for (int c = 1; c <= ncoords; ++c) {
        CGNS_ENUMT(DataType_t) type;
        char cname[64] = {};
        f.check(cg_coord_info(fn, b, z, c, &type, cname), zlabel + " coordinate info");
        std::vector<double> values(static_cast<std::size_t>(zone.vertices));
        const cgsize_t rmin = 1, rmax = static_cast<cgsize_t>(zone.vertices);
        if (zone.vertices > 0)
            f.check(cg_coord_read(fn, b, z, cname, CGNS_ENUMV(RealDouble), &rmin, &rmax, values.data()),
                    std::string("coordinates ") + cname);
        zone.coordinate_names.push_back(cname);
        zone.coordinates.push_back(std::move(values));
    }

    int nsections = 0;
    f.check(cg_nsections(fn, b, z, &nsections), zlabel + " sections");
    for (int s = 1; s <= nsections; ++s) {
        Section sec;
        char sname[64] = {};
        CGNS_ENUMT(ElementType_t) type;
        cgsize_t start = 0, end = 0;
        int nbndry = 0, parent_flag = 0;
        f.check(cg_section_read(fn, b, z, s, sname, &type, &start, &end, &nbndry, &parent_flag), "section " + std::to_string(s));
        sec.name = sname;
        sec.type = static_cast<int>(type);
        sec.start = start;
        sec.end = end;
        const std::int64_t count = checked_count(f, end - start + 1, "section '" + sec.name + "' element count");
        // Connectivity only where it can be used (the other types are reported).
        const bool wanted = sec.type == kBar2 || sec.type == kTri3 || sec.type == kQuad4 || sec.type == kMixed;
        sec.connectivity_read = wanted;
        if (wanted && count > 0) {
            cgsize_t data_size = 0;
            f.check(cg_ElementDataSize(fn, b, z, s, &data_size), "section '" + sec.name + "' size");
            std::vector<cgsize_t> conn(static_cast<std::size_t>(checked_count(f, data_size, "section '" + sec.name + "' size")));
            if (sec.type == kMixed) {
                // Works with CGNS 3.4 and 4.x files: type codes inline, offsets returned.
                std::vector<cgsize_t> offsets(static_cast<std::size_t>(count) + 1);
                f.check(cg_poly_elements_read(fn, b, z, s, conn.data(), offsets.data(), nullptr),
                        "section '" + sec.name + "' connectivity");
                sec.offsets = widen(offsets);
            } else {
                f.check(cg_elements_read(fn, b, z, s, conn.data(), nullptr), "section '" + sec.name + "' connectivity");
            }
            sec.connectivity = widen(conn);
        }
        zone.sections.push_back(std::move(sec));
    }

    int nbocos = 0;
    f.check(cg_nbocos(fn, b, z, &nbocos), zlabel + " boundary conditions");
    for (int k = 1; k <= nbocos; ++k) {
        BoundaryCondition bc;
        char bname[64] = {};
        CGNS_ENUMT(BCType_t) bctype;
        CGNS_ENUMT(PointSetType_t) ptset;
        CGNS_ENUMT(DataType_t) normal_type;
        CGNS_ENUMT(GridLocation_t) location;
        cgsize_t npnts = 0, normal_size = 0;
        int normal_index[3] = {0, 0, 0}, ndataset = 0;
        const std::string what = "BC " + std::to_string(k);
        f.check(cg_boco_info(fn, b, z, k, bname, &bctype, &ptset, &npnts, normal_index, &normal_size, &normal_type, &ndataset), what);
        f.check(cg_boco_gridlocation_read(fn, b, z, k, &location), what + " grid location");
        bc.name = bname;
        bc.type = static_cast<int>(bctype);
        bc.point_set = static_cast<int>(ptset);
        bc.location = static_cast<int>(location);
        std::vector<cgsize_t> points(static_cast<std::size_t>(checked_count(f, npnts, what + " point count")));
        if (!points.empty()) f.check(cg_boco_read(fn, b, z, k, points.data(), nullptr), what + " points");
        bc.points = widen(points);
        char family[64] = {};
        f.check(cg_goto(fn, b, "Zone_t", z, "ZoneBC_t", 1, "BC_t", k, "end"), what + " node");
        const int status = cg_famname_read(family);
        if (status == CG_OK) bc.family = family;
        else if (status != CG_NODE_NOT_FOUND) f.check(status, what + " family name");
        zone.bcs.push_back(std::move(bc));
    }
}

} // namespace

bool cgns_support_available() { return true; }

std::string cgns_library_version() {
    const int v = CGNS_VERSION; // e.g. 4500
    return std::to_string(v / 1000) + "." + std::to_string((v / 100) % 10) + "." + std::to_string((v / 10) % 10);
}

CgnsFile read_cgns_file(const std::string& path) {
    CgnsFileHandle f(path);
    const int fn = f.fn();
    CgnsFile file;

    float version = 0.0f;
    int file_type = 0;
    f.check(cg_version(fn, &version), "the file version");
    f.check(cg_get_file_type(fn, &file_type), "the file type");
    char vtext[16];
    std::snprintf(vtext, sizeof(vtext), "%.2f", static_cast<double>(version));
    file.format = std::string("CGNS ") + vtext +
                  (file_type == CG_FILE_HDF5 ? " HDF5" : file_type == CG_FILE_ADF || file_type == CG_FILE_ADF2 ? " ADF" : "");

    int nbases = 0;
    f.check(cg_nbases(fn, &nbases), "the bases");
    for (int b = 1; b <= nbases; ++b) {
        Base base;
        char bname[64] = {};
        f.check(cg_base_read(fn, b, bname, &base.cell_dimension, &base.physical_dimension), "base " + std::to_string(b));
        base.name = bname;

        int nfamilies = 0;
        f.check(cg_nfamilies(fn, b, &nfamilies), "the families");
        for (int fam = 1; fam <= nfamilies; ++fam) {
            Family family;
            char fname[64] = {};
            int nfambc = 0, ngeo = 0;
            f.check(cg_family_read(fn, b, fam, fname, &nfambc, &ngeo), "family " + std::to_string(fam));
            family.name = fname;
            if (nfambc > 0) {
                char fbname[64] = {};
                CGNS_ENUMT(BCType_t) bctype;
                f.check(cg_fambc_read(fn, b, fam, 1, fbname, &bctype), "the FamilyBC of family '" + family.name + "'");
                family.has_bc = true;
                family.bc_type = static_cast<int>(bctype);
            }
            base.families.push_back(std::move(family));
        }

        // Only the structure needed for the messages of cgns_to_mesh_data is
        // read from bases / zones that will be rejected anyway.
        int nzones = 0;
        f.check(cg_nzones(fn, b, &nzones), "the zones");
        for (int z = 1; z <= nzones; ++z) {
            Zone zone;
            if (nbases == 1 && nzones == 1 && base.cell_dimension == 2) {
                read_zone(f, b, z, zone);
            } else {
                char zname[64] = {};
                cgsize_t size[9] = {};
                CGNS_ENUMT(ZoneType_t) ztype;
                f.check(cg_zone_type(fn, b, z, &ztype), "zone " + std::to_string(z));
                f.check(cg_zone_read(fn, b, z, zname, size), "zone " + std::to_string(z));
                zone.name = zname;
                zone.type = static_cast<int>(ztype);
            }
            base.zones.push_back(std::move(zone));
        }
        file.bases.push_back(std::move(base));
    }
    return file;
}

} // namespace hydrama::mesh

#else // without the CGNS library

namespace hydrama::mesh {

bool cgns_support_available() { return false; }

std::string cgns_library_version() { return ""; }

cgns::CgnsFile read_cgns_file(const std::string& path) {
    throw MeshImportError(path + ": HydraMa2D was built without CGNS support; install the CGNS library (with HDF5) "
                                 "and reconfigure with CMake (see docs/MESH_IMPORT.md, \"CGNS\")");
}

} // namespace hydrama::mesh

#endif
