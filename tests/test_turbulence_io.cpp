// Turbulence output: VTK point scalars (IO::write_vtk with extra fields) and
// the dedicated turbulence restart file (IO::write/read_turbulence_restart).

#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "MeshTestUtils.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <random>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;

namespace {

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

const char* const kTriangleVtk =
    "# vtk DataFile Version 3.0\n"
    "Instananeous Variables Output\n"
    "ASCII\n"
    "DATASET UNSTRUCTURED_GRID\n"
    "POINTS 3 double\n"
    "0.0000000000e+00 0.0000000000e+00 0.0\n"
    "2.5000000000e+00 0.0000000000e+00 0.0\n"
    "0.0000000000e+00 -1.0000000000e-03 0.0\n"
    "CELLS 1 4\n"
    "3 0 1 2\n"
    "CELL_TYPES 1\n"
    "5\n"
    "POINT_DATA 3\n"
    "VECTORS velocity float\n"
    "1.0000000e+00 0.0000000e+00 0.0\n"
    "-5.0000000e-01 2.5000000e-01 0.0\n"
    "0.0000000e+00 3.0000000e+00 0.0\n"
    "SCALARS P float 1\n"
    "LOOKUP_TABLE default\n"
    "1.2500000e-01\n"
    "0.0000000e+00\n"
    "-2.0000000e+00\n";

Mesh triangle() { return make_mesh({{0.0, 0.0}, {2.5, 0.0}, {0.0, -1.0e-3}}, {{0, 1, 2}}); }

} // namespace

// =============================================================================
// VTK point scalars
// =============================================================================

TEST_CASE("VTK with extra point scalars, byte for byte", "[io][vtk][turbulence_io]") {
    TempDir dir("vtk-extra");
    const Mesh m = triangle();
    const std::vector<double> nt = {3.0e-3, 0.0, -1.5e-5}, nut = {0.25, 0.0, 0.0}; // 0.25: exact in float
    REQUIRE(IO::write_vtk((dir / "t.vtk").string(), m, {1.0, -0.5, 0.0}, {0.0, 0.25, 3.0}, {0.125, 0.0, -2.0},
                          {{"nu_tilde", &nt}, {"nut", &nut}}));
    const std::string expected = std::string(kTriangleVtk) +
                                 "SCALARS nu_tilde float 1\n"
                                 "LOOKUP_TABLE default\n"
                                 "3.0000000e-03\n"
                                 "0.0000000e+00\n"
                                 "-1.5000000e-05\n"
                                 "SCALARS nut float 1\n"
                                 "LOOKUP_TABLE default\n"
                                 "2.5000000e-01\n"
                                 "0.0000000e+00\n"
                                 "0.0000000e+00\n";
    CHECK(read_text(dir / "t.vtk") == expected);

    const VtkFile f = parse_vtk(dir / "t.vtk");
    REQUIRE(f.extra_scalars.size() == 2);
    CHECK(f.extra_scalars[0].first == "nu_tilde");
    CHECK(f.extra_scalars[1].first == "nut");
    REQUIRE(f.scalar("nut") != nullptr);
    CHECK((*f.scalar("nut"))[0] == 0.25);
    CHECK(f.scalar("k") == nullptr);
}

TEST_CASE("VTK without extra scalars is unchanged", "[io][vtk][turbulence_io][laminar]") {
    TempDir dir("vtk-plain");
    const Mesh m = triangle();
    REQUIRE(IO::write_vtk((dir / "a.vtk").string(), m, {1.0, -0.5, 0.0}, {0.0, 0.25, 3.0}, {0.125, 0.0, -2.0}));
    REQUIRE(IO::write_vtk((dir / "b.vtk").string(), m, {1.0, -0.5, 0.0}, {0.0, 0.25, 3.0}, {0.125, 0.0, -2.0}, {}));
    CHECK(read_text(dir / "a.vtk") == kTriangleVtk);
    CHECK(read_text(dir / "b.vtk") == kTriangleVtk);
    CHECK(parse_vtk(dir / "a.vtk").extra_scalars.empty());
}

TEST_CASE("VTK rejects invalid extra scalars", "[io][vtk][turbulence_io]") {
    CaptureStderr quiet;
    TempDir dir("vtk-bad");
    const Mesh m = triangle();
    const std::vector<double> ok = {1.0, 2.0, 3.0}, short_field = {1.0};
    const std::vector<double> z = {0.0, 0.0, 0.0};
    // Generate an index, not the fields: the pointers must refer to the
    // vectors of the current invocation of this test case.
    const int which = GENERATE(0, 1, 2, 3);
    const NamedField cases[] = {{"short", &short_field}, {"two words", &ok}, {"", &ok}, {"null", nullptr}};
    const NamedField& bad = cases[which];
    CAPTURE(bad.name);
    CHECK_FALSE(IO::write_vtk((dir / "x.vtk").string(), m, z, z, z, {{"fine", &ok}, bad}));
    CHECK_FALSE(fs::exists(dir / "x.vtk")); // validated before the file is created
}

// =============================================================================
// Turbulence restart
// =============================================================================

TEST_CASE("Turbulence restart round-trips model, names and values bit for bit", "[io][restart][turbulence_io]") {
    TempDir dir("turb-restart");
    std::mt19937_64 rng(7);
    std::vector<double> k(200), omega(200);
    for (std::size_t i = 0; i < k.size(); ++i) {
        std::uint64_t bits = rng();
        double x;
        std::memcpy(&x, &bits, sizeof x);
        k[i] = std::isfinite(x) ? x : 1.0 / (1.0 + i);
        omega[i] = std::ldexp(1.0 + 1e-3 * i, static_cast<int>(i) - 100) * (i % 2 ? -1.0 : 1.0);
    }
    k[0] = -0.0;
    k[1] = std::numeric_limits<double>::denorm_min();
    k[2] = std::numeric_limits<double>::max();
    const std::string path = (dir / "case.turb.var").string();
    REQUIRE(IO::write_turbulence_restart(path, "k-omega-sst", {{"k", &k}, {"omega", &omega}}));

    const auto lines = split_lines(read_text(path));
    REQUIRE(lines.size() == 4 + k.size());
    CHECK(lines[0] == "HydraMa2D-turbulence-restart 1");
    CHECK(lines[1] == "model k-omega-sst");
    CHECK(lines[2] == "npoin 200");
    CHECK(lines[3] == "fields 2 k omega");

    TurbulenceRestartData d;
    std::string error;
    REQUIRE(IO::read_turbulence_restart(path, 200, d, error));
    CHECK(error.empty());
    CHECK(d.model == "k-omega-sst");
    CHECK(d.names == std::vector<std::string>{"k", "omega"});
    REQUIRE(d.values.size() == 2);
    int mismatches = 0;
    for (std::size_t i = 0; i < k.size(); ++i)
        if (!same_bits(d.values[0][i], k[i]) || !same_bits(d.values[1][i], omega[i])) ++mismatches;
    CHECK(mismatches == 0);
}

TEST_CASE("Turbulence restart reader rejects invalid files", "[io][restart][turbulence_io]") {
    TempDir dir("turb-restart-bad");
    struct Bad {
        std::string content;
        std::string message;
    };
    const Bad bad = GENERATE(
        Bad{"", "not a turbulence restart file"},
        Bad{"HydraMa2D-turbulence-restart 2\nmodel sa\nnpoin 2\nfields 1 a\n1\n2\n", "not a turbulence restart file"},
        Bad{"HydraMa2D-turbulence-restart 1\nmodl sa\n", "missing model line"},
        Bad{"HydraMa2D-turbulence-restart 1\nmodel sa\nnpoin x\n", "missing npoin line"},
        Bad{"HydraMa2D-turbulence-restart 1\nmodel sa\nnpoin 3\nfields 1 a\n1\n2\n3\n", "file has 3 nodes, mesh has 2"},
        Bad{"HydraMa2D-turbulence-restart 1\nmodel sa\nnpoin 2\nfields 0\n", "missing or invalid fields line"},
        Bad{"HydraMa2D-turbulence-restart 1\nmodel sa\nnpoin 2\nfields 2 a\n", "missing field name"},
        Bad{"HydraMa2D-turbulence-restart 1\nmodel sa\nnpoin 2\nfields 1 a\n1\n", "invalid or missing value for node 2"},
        Bad{"HydraMa2D-turbulence-restart 1\nmodel sa\nnpoin 2\nfields 1 a\n1\nnan\n", "invalid or missing value for node 2"},
        Bad{"HydraMa2D-turbulence-restart 1\nmodel sa\nnpoin 2\nfields 1 a\n1\n2\n3\n", "unexpected trailing data"});
    write_text(dir / "bad.turb.var", bad.content);
    TurbulenceRestartData d;
    std::string error;
    INFO(bad.content);
    CHECK_FALSE(IO::read_turbulence_restart((dir / "bad.turb.var").string(), 2, d, error));
    CHECK_THAT(error, ContainsSubstring(bad.message));

    CHECK_FALSE(IO::read_turbulence_restart((dir / "missing.turb.var").string(), 2, d, error));
    CHECK_THAT(error, ContainsSubstring("cannot open"));
}

TEST_CASE("Turbulence restart writer rejects invalid input", "[io][restart][turbulence_io]") {
    TempDir dir("turb-restart-write");
    const std::vector<double> a = {1.0, 2.0}, b = {1.0};
    const std::string path = (dir / "x.turb.var").string();
    CHECK_FALSE(IO::write_turbulence_restart(path, "sa", {}));
    CHECK_FALSE(IO::write_turbulence_restart(path, "sa", {{"a", &a}, {"b", &b}}));
    CHECK_FALSE(IO::write_turbulence_restart(path, "sa", {{"a b", &a}}));
    CHECK_FALSE(IO::write_turbulence_restart(path, "sa", {{"a", nullptr}}));
    CHECK_FALSE(IO::write_turbulence_restart((dir / "no_dir" / "x").string(), "sa", {{"a", &a}}));
    CHECK(IO::write_turbulence_restart(path, "sa", {{"a", &a}}));
}
