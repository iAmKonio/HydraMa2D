#include "IO.hpp"
#include "Mesh.hpp"
#include "Config.hpp"
#include "Solver.hpp"
#include "MeshTestUtils.hpp"
#include "QuadTestUtils.hpp"
#include "IoTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>

#if defined(HYDRAMA2D_HAVE_ZLIB)
#include <zlib.h>
#endif

namespace fs = std::filesystem;
using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Helper to extract an XML attribute value from a tag string: attr="value"
std::string extract_attr(const std::string& tag, const std::string& attr) {
    std::string needle = attr + "=\"";
    size_t pos = tag.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();
    size_t end_pos = tag.find('"', pos);
    if (end_pos == std::string::npos) return "";
    return tag.substr(pos, end_pos - pos);
}

struct VtuDataArrayInfo {
    std::string name;
    std::string type;
    int num_components = 1;
    std::string format;
    uint64_t offset = 0;
};

struct ParsedVtu {
    std::string root_type;
    std::string version;
    std::string byte_order;
    std::string header_type;
    std::string compressor;
    int num_points = 0;
    int num_cells = 0;
    std::vector<VtuDataArrayInfo> arrays;
    size_t underscore_pos = 0;
    std::vector<uint8_t> raw_bytes;

    const VtuDataArrayInfo* find_array(const std::string& name) const {
        for (const auto& a : arrays) {
            if (a.name == name) return &a;
        }
        return nullptr;
    }

    std::vector<uint8_t> get_array_bytes(const std::string& name, size_t expected_uncompressed_size) const {
        const auto* arr = find_array(name);
        if (!arr) return {};
        size_t block_start = underscore_pos + 1 + arr->offset;
        if (block_start >= raw_bytes.size()) return {};

        if (compressor.empty()) {
            if (block_start + sizeof(uint64_t) > raw_bytes.size()) return {};
            uint64_t num_bytes = 0;
            std::memcpy(&num_bytes, &raw_bytes[block_start], sizeof(uint64_t));
            if (num_bytes != expected_uncompressed_size) return {};
            size_t data_start = block_start + sizeof(uint64_t);
            if (data_start + num_bytes > raw_bytes.size()) return {};
            return std::vector<uint8_t>(raw_bytes.begin() + data_start,
                                        raw_bytes.begin() + data_start + num_bytes);
        } else {
#if defined(HYDRAMA2D_HAVE_ZLIB)
            if (block_start + 4 * sizeof(uint64_t) > raw_bytes.size()) return {};
            uint64_t hdr[4];
            std::memcpy(hdr, &raw_bytes[block_start], 4 * sizeof(uint64_t));
            uint64_t num_blocks = hdr[0];
            uint64_t block_size = hdr[1];
            uint64_t last_size  = hdr[2];
            uint64_t comp_size  = hdr[3];

            if (num_blocks != 1 || block_size != expected_uncompressed_size || last_size != expected_uncompressed_size) {
                return {};
            }
            size_t comp_start = block_start + 4 * sizeof(uint64_t);
            if (comp_start + comp_size > raw_bytes.size()) return {};

            std::vector<uint8_t> decomp(expected_uncompressed_size);
            uLong dest_len = static_cast<uLong>(expected_uncompressed_size);
            int ret = ::uncompress(decomp.data(), &dest_len,
                                   &raw_bytes[comp_start], static_cast<uLong>(comp_size));
            if (ret != Z_OK || dest_len != expected_uncompressed_size) return {};
            return decomp;
#else
            return {};
#endif
        }
    }
};

ParsedVtu parse_vtu_file(const fs::path& filepath) {
    ParsedVtu vtu;
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) return vtu;

    in.seekg(0, std::ios::end);
    size_t sz = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);
    vtu.raw_bytes.resize(sz);
    in.read(reinterpret_cast<char*>(vtu.raw_bytes.data()), sz);

    // Locate "<AppendedData encoding=\"raw\">" and following '_'
    std::string content(vtu.raw_bytes.begin(), vtu.raw_bytes.end());
    size_t app_pos = content.find("<AppendedData encoding=\"raw\">");
    if (app_pos == std::string::npos) return vtu;

    size_t under_pos = content.find('_', app_pos);
    if (under_pos == std::string::npos) return vtu;
    vtu.underscore_pos = under_pos;

    // Parse the XML header before '_'
    std::string xml = content.substr(0, under_pos);
    std::istringstream xml_in(xml);
    std::string line;
    while (std::getline(xml_in, line)) {
        if (line.find("<VTKFile") != std::string::npos) {
            vtu.root_type   = extract_attr(line, "type");
            vtu.version     = extract_attr(line, "version");
            vtu.byte_order  = extract_attr(line, "byte_order");
            vtu.header_type = extract_attr(line, "header_type");
            vtu.compressor  = extract_attr(line, "compressor");
        } else if (line.find("<Piece") != std::string::npos) {
            std::string np = extract_attr(line, "NumberOfPoints");
            std::string nc = extract_attr(line, "NumberOfCells");
            if (!np.empty()) vtu.num_points = std::stoi(np);
            if (!nc.empty()) vtu.num_cells  = std::stoi(nc);
        } else if (line.find("<DataArray") != std::string::npos) {
            VtuDataArrayInfo arr;
            arr.name   = extract_attr(line, "Name");
            arr.type   = extract_attr(line, "type");
            arr.format = extract_attr(line, "format");
            std::string nc = extract_attr(line, "NumberOfComponents");
            if (!nc.empty()) arr.num_components = std::stoi(nc);
            std::string off = extract_attr(line, "offset");
            if (!off.empty()) arr.offset = std::stoull(off);
            vtu.arrays.push_back(arr);
        }
    }
    return vtu;
}

// Build a small mixed mesh: 2 triangles and 1 quad (6 nodes total)
Mesh make_mixed_test_mesh() {
    Mesh m;
    m.npoin = 6;
    m.x = {0.0, 1.0, 2.0, 0.0, 1.0, 2.0};
    m.y = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};

    // Triangles: (0, 1, 4) and (0, 4, 3)
    m.intma = {{0, 1, 4}, {0, 4, 3}};
    // Quad: (1, 2, 5, 4)
    m.quads = {{1, 2, 5, 4}};
    m.nelem = 3;
    return m;
}

} // namespace

TEST_CASE("write_vtu: XML structure and uncompressed raw binary appended data", "[vtu][io]") {
    TempDir dir;
    fs::path vtu_path = dir.path() / "mixed.vtu";

    Mesh m = make_mixed_test_mesh();
    std::vector<double> u = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> v = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    std::vector<double> p = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};

    REQUIRE(IO::write_vtu(vtu_path.string(), m, u, v, p, /*compress=*/false));

    ParsedVtu vtu = parse_vtu_file(vtu_path);

    // Verify XML structure attributes
    CHECK(vtu.root_type == "UnstructuredGrid");
    CHECK(vtu.version == "1.0");
    CHECK(vtu.byte_order == "LittleEndian");
    CHECK(vtu.header_type == "UInt64");
    CHECK(vtu.compressor.empty()); // uncompressed
    CHECK(vtu.num_points == 6);
    CHECK(vtu.num_cells == 3);

    // Verify DataArrays present
    CHECK(vtu.find_array("Points") != nullptr);
    CHECK(vtu.find_array("connectivity") != nullptr);
    CHECK(vtu.find_array("offsets") != nullptr);
    CHECK(vtu.find_array("types") != nullptr);
    CHECK(vtu.find_array("velocity") != nullptr);
    CHECK(vtu.find_array("P") != nullptr);

    // Verify Points Float64 (6 * 3 doubles = 144 bytes)
    auto pts_bytes = vtu.get_array_bytes("Points", 6 * 3 * sizeof(double));
    REQUIRE(pts_bytes.size() == 6 * 3 * sizeof(double));
    const double* pts = reinterpret_cast<const double*>(pts_bytes.data());
    for (int i = 0; i < m.npoin; ++i) {
        CHECK(pts[3 * i + 0] == m.x[i]);
        CHECK(pts[3 * i + 1] == m.y[i]);
        CHECK(pts[3 * i + 2] == 0.0);
    }

    // Verify Connectivity Int64: 2 tris (6 ints) + 1 quad (4 ints) = 10 ints (80 bytes)
    auto conn_bytes = vtu.get_array_bytes("connectivity", 10 * sizeof(int64_t));
    REQUIRE(conn_bytes.size() == 10 * sizeof(int64_t));
    const int64_t* conn = reinterpret_cast<const int64_t*>(conn_bytes.data());
    std::vector<int64_t> expected_conn = {0, 1, 4,  0, 4, 3,  1, 2, 5, 4};
    for (size_t i = 0; i < expected_conn.size(); ++i) {
        CHECK(conn[i] == expected_conn[i]);
    }

    // Verify Offsets Int64: 3 cells (3, 6, 10) (24 bytes)
    auto off_bytes = vtu.get_array_bytes("offsets", 3 * sizeof(int64_t));
    REQUIRE(off_bytes.size() == 3 * sizeof(int64_t));
    const int64_t* offsets = reinterpret_cast<const int64_t*>(off_bytes.data());
    CHECK(offsets[0] == 3);
    CHECK(offsets[1] == 6);
    CHECK(offsets[2] == 10);

    // Verify Types UInt8: 3 cells (5, 5, 9) (3 bytes)
    auto types_bytes = vtu.get_array_bytes("types", 3 * sizeof(uint8_t));
    REQUIRE(types_bytes.size() == 3 * sizeof(uint8_t));
    CHECK(types_bytes[0] == 5);
    CHECK(types_bytes[1] == 5);
    CHECK(types_bytes[2] == 9);

    // Verify Velocity Float32 (6 * 3 floats = 72 bytes)
    auto vel_bytes = vtu.get_array_bytes("velocity", 6 * 3 * sizeof(float));
    REQUIRE(vel_bytes.size() == 6 * 3 * sizeof(float));
    const float* vel = reinterpret_cast<const float*>(vel_bytes.data());
    for (int i = 0; i < m.npoin; ++i) {
        CHECK(vel[3 * i + 0] == static_cast<float>(u[i]));
        CHECK(vel[3 * i + 1] == static_cast<float>(v[i]));
        CHECK(vel[3 * i + 2] == 0.0f);
    }

    // Verify Pressure Float32 (6 floats = 24 bytes)
    auto p_bytes = vtu.get_array_bytes("P", 6 * sizeof(float));
    REQUIRE(p_bytes.size() == 6 * sizeof(float));
    const float* p_arr = reinterpret_cast<const float*>(p_bytes.data());
    for (int i = 0; i < m.npoin; ++i) {
        CHECK(p_arr[i] == static_cast<float>(p[i]));
    }
}

TEST_CASE("write_vtu: compressed raw binary appended data with zlib roundtrip", "[vtu][io][zlib]") {
#if !defined(HYDRAMA2D_HAVE_ZLIB)
    SKIP("HydraMa2D was compiled without ZLIB support");
#endif

    TempDir dir;
    fs::path vtu_path = dir.path() / "compressed.vtu";

    Mesh m = make_mixed_test_mesh();
    std::vector<double> u = {1.5, -2.5, 3.5, -4.5, 5.5, -6.5};
    std::vector<double> v = {0.15, -0.25, 0.35, -0.45, 0.55, -0.65};
    std::vector<double> p = {101.3, 202.6, 303.9, 404.2, 505.5, 606.8};
    std::vector<double> T = {300.0, 310.0, 320.0, 330.0, 340.0, 350.0};
    std::vector<double> nut = {1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4};
    std::vector<NamedField> scalars = {{"T", &T}, {"nut", &nut}};

    REQUIRE(IO::write_vtu(vtu_path.string(), m, u, v, p, scalars, /*compress=*/true));

    ParsedVtu vtu = parse_vtu_file(vtu_path);

    // Verify root XML compressor attribute
    CHECK(vtu.root_type == "UnstructuredGrid");
    CHECK(vtu.compressor == "vtkZLibDataCompressor");
    CHECK(vtu.header_type == "UInt64");
    CHECK(vtu.num_points == 6);
    CHECK(vtu.num_cells == 3);

    // Verify all scalar arrays are registered in PointData
    CHECK(vtu.find_array("T") != nullptr);
    CHECK(vtu.find_array("nut") != nullptr);

    // Decompress and verify Points Float64
    auto pts_bytes = vtu.get_array_bytes("Points", 6 * 3 * sizeof(double));
    REQUIRE(pts_bytes.size() == 6 * 3 * sizeof(double));
    const double* pts = reinterpret_cast<const double*>(pts_bytes.data());
    for (int i = 0; i < m.npoin; ++i) {
        CHECK(pts[3 * i + 0] == m.x[i]);
        CHECK(pts[3 * i + 1] == m.y[i]);
        CHECK(pts[3 * i + 2] == 0.0);
    }

    // Decompress and verify Connectivity
    auto conn_bytes = vtu.get_array_bytes("connectivity", 10 * sizeof(int64_t));
    REQUIRE(conn_bytes.size() == 10 * sizeof(int64_t));
    const int64_t* conn = reinterpret_cast<const int64_t*>(conn_bytes.data());
    std::vector<int64_t> expected_conn = {0, 1, 4,  0, 4, 3,  1, 2, 5, 4};
    for (size_t i = 0; i < expected_conn.size(); ++i) {
        CHECK(conn[i] == expected_conn[i]);
    }

    // Decompress and verify Velocity
    auto vel_bytes = vtu.get_array_bytes("velocity", 6 * 3 * sizeof(float));
    REQUIRE(vel_bytes.size() == 6 * 3 * sizeof(float));
    const float* vel = reinterpret_cast<const float*>(vel_bytes.data());
    for (int i = 0; i < m.npoin; ++i) {
        CHECK(vel[3 * i + 0] == static_cast<float>(u[i]));
        CHECK(vel[3 * i + 1] == static_cast<float>(v[i]));
        CHECK(vel[3 * i + 2] == 0.0f);
    }

    // Decompress and verify Pressure
    auto p_bytes = vtu.get_array_bytes("P", 6 * sizeof(float));
    REQUIRE(p_bytes.size() == 6 * sizeof(float));
    const float* p_arr = reinterpret_cast<const float*>(p_bytes.data());
    for (int i = 0; i < m.npoin; ++i) {
        CHECK(p_arr[i] == static_cast<float>(p[i]));
    }

    // Decompress and verify Temperature T
    auto t_bytes = vtu.get_array_bytes("T", 6 * sizeof(float));
    REQUIRE(t_bytes.size() == 6 * sizeof(float));
    const float* t_arr = reinterpret_cast<const float*>(t_bytes.data());
    for (int i = 0; i < m.npoin; ++i) {
        CHECK(t_arr[i] == static_cast<float>(T[i]));
    }

    // Decompress and verify nut
    auto nut_bytes = vtu.get_array_bytes("nut", 6 * sizeof(float));
    REQUIRE(nut_bytes.size() == 6 * sizeof(float));
    const float* nut_arr = reinterpret_cast<const float*>(nut_bytes.data());
    for (int i = 0; i < m.npoin; ++i) {
        CHECK(nut_arr[i] == static_cast<float>(nut[i]));
    }
}

TEST_CASE("write_vtu: file size reduction compared to ASCII VTK (>70% reduction)", "[vtu][io][benchmark]") {
#if !defined(HYDRAMA2D_HAVE_ZLIB)
    SKIP("HydraMa2D was compiled without ZLIB support");
#endif

    TempDir dir;
    fs::path vtk_path  = dir.path() / "case.vtk";
    fs::path vtu_raw   = dir.path() / "case_raw.vtu";
    fs::path vtu_comp  = dir.path() / "case_comp.vtu";

    // Create a 40x40 fine grid mesh (1600 nodes, 3042 triangles)
    Mesh m = make_grid_mesh(GridSpec{0.0, 0.0, 1.0, 1.0, 40, 40});
    std::vector<double> u(m.npoin), v(m.npoin), p(m.npoin), T(m.npoin);
    for (int i = 0; i < m.npoin; ++i) {
        u[i] = std::sin(m.x[i] * 3.14159);
        v[i] = std::cos(m.y[i] * 3.14159);
        p[i] = 100.0 + m.x[i] * m.y[i];
        T[i] = 300.0 + 50.0 * m.x[i];
    }
    std::vector<NamedField> scalars = {{"T", &T}};

    // 1. ASCII VTK
    REQUIRE(IO::write_vtk(vtk_path.string(), m, u, v, p, scalars));
    const auto ascii_size = fs::file_size(vtk_path);

    // 2. Uncompressed VTU
    REQUIRE(IO::write_vtu(vtu_raw.string(), m, u, v, p, scalars, /*compress=*/false));
    const auto raw_size = fs::file_size(vtu_raw);

    // 3. Compressed VTU
    REQUIRE(IO::write_vtu(vtu_comp.string(), m, u, v, p, scalars, /*compress=*/true));
    const auto comp_size = fs::file_size(vtu_comp);

    // Verify uncompressed binary is smaller than ASCII
    CHECK(raw_size < ascii_size);

    // Verify compressed binary is smaller than uncompressed binary
    CHECK(comp_size < raw_size);

    // Compute reduction percentage: 1 - (comp_size / ascii_size)
    const double reduction = 1.0 - (static_cast<double>(comp_size) / static_cast<double>(ascii_size));
    const double reduction_percent = reduction * 100.0;

    // Acceptance criterion: File size reduced by > 70% compared to ASCII .vtk
    CHECK(reduction_percent > 70.0);
}

TEST_CASE("ControlConfig and IO::read_control vtk_format options", "[vtu][config][io]") {
    TempDir dir;

    auto write_control_with = [&](const std::string& extra_line) -> fs::path {
        fs::path p = dir.path() / ("test_" + std::to_string(std::rand()) + ".control");
        std::ofstream out(p);
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Restart from prior run (0 - NO, Not even if values exist, 1 - YES, if values exist)\n0\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Number of Timesteps (Max in case of steady state)\n100\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Transient (0 - OFF, 1 - ON)\n1\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Simulated Time Interval for Output (transient only)\n1.00\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "End Simulation Time (transient only)\n10.0\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Output Controls - Every N iterations\n1000\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Output Controls - Every N CPU seconds\n3600\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Global Minimum Calculated Timestep = -1 OR Local Time Steping = 0 OR Specified Time Step Value = 1\n-1\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Specified Time Step Value (The Actual Value if Activated)\n1.0d-05\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Convergence Check Interval\n100\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Safety Factor for Calculated Local Time Step\n0.6d00\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Theta1 (0.5<=theta1<=1.0). Theta2 Hardcoded as 1.0 for Semi-Implicit\n1.0d00\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Steady State Velocity Convergence (0 - OFF, 1 - ON)\n1\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "L2-Norm\n1.0d-06\n";
        out << "---------------------------------------------------------------------------------------------------\n";
        out << "Run Time Adjustable Option (1 - Yes, if this file modified, values will be re-read, 0 - NO)\n0\n";
        if (!extra_line.empty()) {
            out << extra_line << "\n";
        }
        return p;
    };

    SECTION("Default without vtk_format setting is AsciiVtk") {
        fs::path p = write_control_with("");
        ControlConfig cfg;
        REQUIRE(IO::read_control(p.string(), cfg));
        CHECK(cfg.vtk_format == VtkFormat::AsciiVtk);
    }

    SECTION("vtk_format = ascii_vtk / ascii / vtk") {
        fs::path p = write_control_with("vtk_format = ascii_vtk");
        ControlConfig cfg;
        REQUIRE(IO::read_control(p.string(), cfg));
        CHECK(cfg.vtk_format == VtkFormat::AsciiVtk);

        p = write_control_with("vtk_format: ascii");
        REQUIRE(IO::read_control(p.string(), cfg));
        CHECK(cfg.vtk_format == VtkFormat::AsciiVtk);
    }

    SECTION("vtk_format = binary_vtu / vtu / binary") {
        fs::path p = write_control_with("vtk_format = binary_vtu");
        ControlConfig cfg;
        REQUIRE(IO::read_control(p.string(), cfg));
        CHECK(cfg.vtk_format == VtkFormat::BinaryVtu);

        p = write_control_with("vtk_format vtu");
        REQUIRE(IO::read_control(p.string(), cfg));
        CHECK(cfg.vtk_format == VtkFormat::BinaryVtu);
    }

    SECTION("vtk_format = compressed_vtu / compressed / zlib_vtu") {
        fs::path p = write_control_with("vtk_format = compressed_vtu");
        ControlConfig cfg;
        REQUIRE(IO::read_control(p.string(), cfg));
        CHECK(cfg.vtk_format == VtkFormat::CompressedVtu);

        p = write_control_with("vtk_format = zlib_vtu  # with comment");
        REQUIRE(IO::read_control(p.string(), cfg));
        CHECK(cfg.vtk_format == VtkFormat::CompressedVtu);
    }
}

TEST_CASE("Solver end-to-end output with vtk_format", "[vtu][solver]") {
    TempDir dir;
    copy_benchmark_case(dir.path());

    SimulationConfig cfg;
    cfg.casename = "hinterkante";
    cfg.case_dir = dir.path().string();
    cfg.override_ntime = 1;

    REQUIRE(IO::read_control((dir.path() / "system/hinterkante.control").string(), cfg.control));
    REQUIRE(IO::read_physical((dir.path() / "system/hinterkante.physical").string(), cfg.physical));
    REQUIRE(IO::read_solver((dir.path() / "system/hinterkante.solver").string(), cfg.solver));

    SECTION("Solver writes .vtu when vtk_format is CompressedVtu") {
        cfg.control.vtk_format = VtkFormat::CompressedVtu;
        Solver solver(cfg);
        REQUIRE(solver.initialize());
        solver.run();

        // Check that .vtu was created in results/
        fs::path res_dir = dir.path() / "results";
        bool found_vtu = false;
        for (const auto& entry : fs::directory_iterator(res_dir)) {
            if (entry.path().extension() == ".vtu") {
                found_vtu = true;
                ParsedVtu vtu = parse_vtu_file(entry.path());
                CHECK(vtu.root_type == "UnstructuredGrid");
#if defined(HYDRAMA2D_HAVE_ZLIB)
                CHECK(vtu.compressor == "vtkZLibDataCompressor");
#endif
                CHECK(vtu.num_points > 0);
                CHECK(vtu.num_cells > 0);
            }
        }
        CHECK(found_vtu);
    }

    SECTION("Solver writes legacy .vtk when vtk_format is AsciiVtk") {
        cfg.control.vtk_format = VtkFormat::AsciiVtk;
        Solver solver(cfg);
        REQUIRE(solver.initialize());
        solver.run();

        fs::path res_dir = dir.path() / "results";
        bool found_vtk = false;
        for (const auto& entry : fs::directory_iterator(res_dir)) {
            if (entry.path().extension() == ".vtk") {
                found_vtk = true;
            }
        }
        CHECK(found_vtk);
    }
}

TEST_CASE("write_vtu error handling and edge cases", "[vtu][io][error]") {
    TempDir dir;
    Mesh m = make_mixed_test_mesh();
    std::vector<double> u = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> v = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    std::vector<double> p = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};

    SECTION("Unwritable path returns false") {
        CHECK_FALSE(IO::write_vtu("/non/existent/directory/file.vtu", m, u, v, p));
    }

    SECTION("Mismatched field sizes return false") {
        std::vector<double> short_u = {1.0, 2.0};
        CHECK_FALSE(IO::write_vtu((dir.path() / "bad.vtu").string(), m, short_u, v, p));
    }

    SECTION("Invalid scalar name with whitespace or XML characters returns false") {
        std::vector<double> T = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        CHECK_FALSE(IO::write_vtu((dir.path() / "bad.vtu").string(), m, u, v, p, {{"bad name", &T}}));
        CHECK_FALSE(IO::write_vtu((dir.path() / "bad.vtu").string(), m, u, v, p, {{"bad\"name", &T}}));
        CHECK_FALSE(IO::write_vtu((dir.path() / "bad.vtu").string(), m, u, v, p, {{"bad<name>", &T}}));
    }

    SECTION("Null scalar values pointer returns false") {
        CHECK_FALSE(IO::write_vtu((dir.path() / "bad.vtu").string(), m, u, v, p, {{"T", nullptr}}));
    }
}
