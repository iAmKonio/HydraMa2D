// Boundary naming convention (mesh/BoundaryNaming.hpp): the table itself,
// name normalisation, keyword matching, ambiguity handling, legacy codes, and
// agreement with docs/BOUNDARY_NAMING.md.

#include "BoundaryKind.hpp"
#include "IoTestUtils.hpp"
#include "mesh/BoundaryNaming.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cctype>
#include <map>
#include <set>

using namespace hydrama;
using namespace hydrama::mesh;
using hydrama::test::read_text;
using hydrama::test::split_lines;

namespace {

BoundaryType resolved_type(const std::string& name) {
    const NameResolution r = resolve_boundary_name(name);
    INFO("name '" << name << "': " << r.message(name));
    REQUIRE(r.resolved());
    return r.type;
}

NameResolution::Status status_of(const std::string& name) { return resolve_boundary_name(name).status; }

} // namespace

// =============================================================================
// The table
// =============================================================================

TEST_CASE("Implemented boundary types are exactly the legacy flag codes", "[boundary_naming]") {
    const std::map<BoundaryType, int> expected = {
        {BoundaryType::Wall, 500},   {BoundaryType::WallHot, 501},  {BoundaryType::WallCold, 502},
        {BoundaryType::Inlet, 503},  {BoundaryType::Outlet, 504},   {BoundaryType::Symmetry, 506},
        {BoundaryType::InletProfile, 507}, {BoundaryType::InletParabolic, 508},
    };
    std::set<int> seen_flags;
    for (const BoundaryTypeInfo& info : boundary_type_table()) {
        CAPTURE(info.keyword);
        const auto it = expected.find(info.type);
        if (it != expected.end()) {
            CHECK(is_implemented(info.type));
            CHECK(solver_flag(info.type) == it->second);
            CHECK(boundary_type_from_flag(it->second) == info.type);
            CHECK(seen_flags.insert(info.flag).second);
        } else {
            CHECK_FALSE(is_implemented(info.type));
            CHECK_FALSE(solver_flag(info.type).has_value());
            CHECK(info.flag == 0);
        }
    }
    CHECK(seen_flags.size() == expected.size());
    CHECK(boundary_type_from_flag(505) == BoundaryType::Custom);
    CHECK(boundary_type_from_flag(0) == BoundaryType::Custom);
    CHECK(boundary_type_from_flag(-1) == BoundaryType::Custom);
}

TEST_CASE("The coarse boundary kind agrees with classify_boundary_flag", "[boundary_naming][boundary_kind]") {
    for (const BoundaryTypeInfo& info : boundary_type_table()) {
        CAPTURE(info.keyword);
        if (info.flag != 0) {
            CHECK(to_boundary_kind(info.type) == classify_boundary_flag(info.flag));
        } else {
            CHECK(to_boundary_kind(info.type) == BoundaryKind::Unknown);
        }
    }
    CHECK(to_boundary_kind(BoundaryType::Wall) == BoundaryKind::Wall);
    CHECK(to_boundary_kind(BoundaryType::InletProfile) == BoundaryKind::Inlet);
    CHECK(to_boundary_kind(BoundaryType::InletParabolic) == BoundaryKind::Inlet);
    CHECK(to_boundary_kind(BoundaryType::Symmetry) == BoundaryKind::Symmetry);
}

TEST_CASE("Keywords and aliases are unique and normalised", "[boundary_naming]") {
    std::map<std::string, std::string> owner; // normalised alias -> keyword
    std::set<std::string> keywords;
    for (const BoundaryTypeInfo& info : boundary_type_table()) {
        CAPTURE(info.keyword);
        CHECK(keywords.insert(std::string(info.keyword)).second);
        CHECK(normalize_boundary_name(info.keyword) == info.keyword); // keywords are written normalised
        CHECK_FALSE(info.description.empty());
        if (info.type == BoundaryType::Custom) {
            CHECK(info.aliases.empty());
            continue;
        }
        std::vector<std::string_view> all = info.aliases;
        all.push_back(info.keyword);
        for (std::string_view alias : all) {
            CAPTURE(alias);
            const std::string n = normalize_boundary_name(alias);
            REQUIRE_FALSE(n.empty());
            const auto [it, added] = owner.emplace(n, std::string(info.keyword));
            CHECK((added || it->second == info.keyword));
            if (!added && it->second == info.keyword) FAIL_CHECK("alias listed twice for " << info.keyword);
        }
    }
}

TEST_CASE("Every family is a generic type of the table", "[boundary_naming]") {
    for (const BoundaryTypeInfo& info : boundary_type_table()) {
        CAPTURE(info.keyword);
        const BoundaryTypeInfo& family = boundary_type_info(info.family);
        CHECK(family.family == family.type); // families are not nested
    }
    CHECK(boundary_type_info(BoundaryType::Rotating).family == BoundaryType::Wall);
    CHECK(boundary_type_info(BoundaryType::PressureInlet).family == BoundaryType::Inlet);
    CHECK(boundary_type_info(BoundaryType::Periodic).family == BoundaryType::Periodic);
}

// =============================================================================
// Resolution
// =============================================================================

TEST_CASE("Every keyword and alias resolves to its own type", "[boundary_naming]") {
    for (const BoundaryTypeInfo& info : boundary_type_table()) {
        if (info.type == BoundaryType::Custom) continue;
        CHECK(resolved_type(std::string(info.keyword)) == info.type);
        for (std::string_view alias : info.aliases) CHECK(resolved_type(std::string(alias)) == info.type);
    }
}

TEST_CASE("The required canonical markers", "[boundary_naming]") {
    CHECK(resolved_type("wall") == BoundaryType::Wall);
    CHECK(resolved_type("inlet") == BoundaryType::Inlet);
    CHECK(resolved_type("outlet") == BoundaryType::Outlet);
    CHECK(resolved_type("symmetry") == BoundaryType::Symmetry);
    CHECK(resolved_type("periodic") == BoundaryType::Periodic);
    CHECK(resolved_type("rotating") == BoundaryType::Rotating);
    CHECK(resolved_type("farfield") == BoundaryType::Farfield);
    CHECK(resolved_type("moving_wall") == BoundaryType::MovingWall);
}

TEST_CASE("Normalisation: case, separators, camelCase and digits", "[boundary_naming]") {
    CHECK(normalize_boundary_name("Wall") == "wall");
    CHECK(normalize_boundary_name("  WALL  ") == "wall");
    CHECK(normalize_boundary_name("Velocity-Inlet 2") == "velocity_inlet_2");
    CHECK(normalize_boundary_name("symmetryPlane") == "symmetry_plane");
    CHECK(normalize_boundary_name("noSlipWall") == "no_slip_wall");
    CHECK(normalize_boundary_name("CHTInterface") == "cht_interface");
    CHECK(normalize_boundary_name("wall2") == "wall_2");
    CHECK(normalize_boundary_name("2wall") == "2_wall");
    CHECK(normalize_boundary_name("a..b--c  d") == "a_b_c_d");
    CHECK(normalize_boundary_name("__wall__") == "wall");
    CHECK(normalize_boundary_name("wall_T1") == "wall_t_1");
    CHECK(normalize_boundary_name("") == "");
    CHECK(normalize_boundary_name(" -_ ") == "");
}

TEST_CASE("Keywords are found anywhere in the name, as whole tokens", "[boundary_naming]") {
    CHECK(resolved_type("Wall") == BoundaryType::Wall);
    CHECK(resolved_type("WALLS") == BoundaryType::Wall);
    CHECK(resolved_type("bottom_wall") == BoundaryType::Wall);
    CHECK(resolved_type("wall_top") == BoundaryType::Wall);
    CHECK(resolved_type("upper wall 2") == BoundaryType::Wall);
    CHECK(resolved_type("Inlet-Left") == BoundaryType::Inlet);
    CHECK(resolved_type("inlet2") == BoundaryType::Inlet);
    CHECK(resolved_type("Outlet_01") == BoundaryType::Outlet);
    CHECK(resolved_type("symmetryPlane") == BoundaryType::Symmetry);
    CHECK(resolved_type("Symmetry Plane") == BoundaryType::Symmetry);
    CHECK(resolved_type("velocityInlet") == BoundaryType::Inlet);
    CHECK(resolved_type("cylinderWall") == BoundaryType::Wall);
    CHECK(resolved_type("periodic_left") == BoundaryType::Periodic);
    CHECK(resolved_type("cyclicAMI") == BoundaryType::Periodic);
    CHECK(resolved_type("frontAndBack") == BoundaryType::Empty);

    // Whole tokens only: no substring matches.
    for (const char* name : {"wallace", "walled", "outletter", "symbol", "inletwall", "airfoil", "left", "boundary_1"}) {
        CAPTURE(name);
        CHECK(status_of(name) == NameResolution::Status::NoKeyword);
    }
}

TEST_CASE("A longer keyword wins over the keywords it contains", "[boundary_naming]") {
    CHECK(resolved_type("pressure_inlet") == BoundaryType::PressureInlet);
    CHECK(resolved_type("left_pressure_inlet") == BoundaryType::PressureInlet);
    CHECK(resolved_type("totalPressureInlet") == BoundaryType::PressureInlet);
    CHECK(resolved_type("massFlowInlet") == BoundaryType::MassFlowInlet);
    CHECK(resolved_type("supersonic_inlet") == BoundaryType::SupersonicInlet);
    CHECK(resolved_type("pressure_outlet") == BoundaryType::Outlet);
    CHECK(resolved_type("static_pressure_outlet") == BoundaryType::Outlet);
    CHECK(resolved_type("convective_outlet") == BoundaryType::ConvectiveOutlet);
    CHECK(resolved_type("no_slip_wall") == BoundaryType::Wall);   // not "slip_wall"
    CHECK(resolved_type("noSlip") == BoundaryType::Wall);
    CHECK(resolved_type("slip_wall") == BoundaryType::Symmetry);  // not "wall"
    CHECK(resolved_type("free_slip_wall") == BoundaryType::Symmetry);
    CHECK(resolved_type("symmetry_axis") == BoundaryType::Axis);
    CHECK(resolved_type("inlet_outlet") == BoundaryType::Opening);
    CHECK(resolved_type("moving_wall") == BoundaryType::MovingWall);
    CHECK(resolved_type("rotating_wall") == BoundaryType::Rotating);
    CHECK(resolved_type("inlet_profile") == BoundaryType::InletProfile);
    CHECK(resolved_type("inlet_bfs") == BoundaryType::InletProfile);
    CHECK(resolved_type("inlet_parabolic") == BoundaryType::InletParabolic);
    CHECK(resolved_type("Parabolic-Inlet") == BoundaryType::InletParabolic);
    CHECK(resolved_type("poiseuilleInlet") == BoundaryType::InletParabolic);
    CHECK(resolved_type("fully_developed_inlet_left") == BoundaryType::InletParabolic);
    CHECK(resolved_type("far_field") == BoundaryType::Farfield);
    CHECK(resolved_type("free_stream") == BoundaryType::Farfield);
}

TEST_CASE("A specific type beats the generic type of its family", "[boundary_naming]") {
    CHECK(resolved_type("rotor_wall") == BoundaryType::Rotating);
    CHECK(resolved_type("lid_wall") == BoundaryType::MovingWall);
    CHECK(resolved_type("wall_hot") == BoundaryType::WallHot);
    CHECK(resolved_type("wall_hot_cold") == BoundaryType::WallHot); // "wall_hot" is the longest match
    CHECK(resolved_type("hot_wall_left") == BoundaryType::WallHot);
    CHECK(resolved_type("baffle_wall") == BoundaryType::Baffle);
    CHECK(resolved_type("inlet_supersonic_inflow") == BoundaryType::SupersonicInlet);
    CHECK(resolved_type("axis_symmetry") == BoundaryType::Axis);
    CHECK(resolved_type("outlet_massflow_outlet") == BoundaryType::MassFlowOutlet);
}

TEST_CASE("Names with conflicting keywords are ambiguous", "[boundary_naming]") {
    for (const char* name : {"inlet_wall", "outlet_inlet", "periodic_wall", "symmetry_wall", "hot_wall_cold",
                             "farfield_outlet", "rotor_lid"}) {
        CAPTURE(name);
        const NameResolution r = resolve_boundary_name(name);
        CHECK(r.status == NameResolution::Status::Ambiguous);
        CHECK(r.candidates.size() >= 2);
        CHECK(r.message(name).find("ambiguous") != std::string::npos);
    }
    const NameResolution r = resolve_boundary_name("inlet_wall");
    CHECK(r.candidates == std::vector<BoundaryType>{BoundaryType::Inlet, BoundaryType::Wall});
    CHECK(r.message("inlet_wall").find("inlet and wall") != std::string::npos);
}

TEST_CASE("Numeric names are legacy flag codes", "[boundary_naming]") {
    CHECK(resolved_type("500") == BoundaryType::Wall);
    CHECK(resolved_type("501") == BoundaryType::WallHot);
    CHECK(resolved_type("502") == BoundaryType::WallCold);
    CHECK(resolved_type("503") == BoundaryType::Inlet);
    CHECK(resolved_type("504") == BoundaryType::Outlet);
    CHECK(resolved_type("506") == BoundaryType::Symmetry);
    CHECK(resolved_type("507") == BoundaryType::InletProfile);
    CHECK(resolved_type(" 507 ") == BoundaryType::InletProfile);
    CHECK(resolved_type("508") == BoundaryType::InletParabolic);
    for (const char* name : {"505", "0", "1", "99999999999999999999"}) {
        CAPTURE(name);
        CHECK(status_of(name) == NameResolution::Status::UnknownCode);
    }
    // Digits inside a name are not codes.
    CHECK(resolved_type("wall_503") == BoundaryType::Wall);
}

TEST_CASE("Empty names and messages", "[boundary_naming]") {
    CHECK(status_of("") == NameResolution::Status::Empty);
    CHECK(status_of("  __ ") == NameResolution::Status::Empty);
    const NameResolution r = resolve_boundary_name("airfoil");
    CHECK(r.message("airfoil").find("no boundary keyword") != std::string::npos);
    CHECK(r.message("airfoil").find("airfoil_wall") != std::string::npos); // suggestion
    const NameResolution ok = resolve_boundary_name("bottom_wall");
    CHECK(ok.matched == "wall");
    CHECK(ok.message("bottom_wall").find("wall boundary") != std::string::npos);
}

TEST_CASE("The summary lists every type and alias", "[boundary_naming]") {
    const std::string summary = boundary_naming_summary();
    for (const BoundaryTypeInfo& info : boundary_type_table()) {
        if (info.type == BoundaryType::Custom) continue;
        CAPTURE(info.keyword);
        CHECK(summary.find("  " + std::string(info.keyword)) != std::string::npos);
        for (std::string_view a : info.aliases) CHECK(summary.find(" " + std::string(a)) != std::string::npos);
    }
}

// =============================================================================
// Documentation
// =============================================================================

TEST_CASE("docs/BOUNDARY_NAMING.md lists exactly the table of the code", "[boundary_naming][docs]") {
    // Table rows: | `keyword` | flag or - | `alias`, `alias`, ... | description |
    const std::string doc = read_text(std::string(HYDRAMA2D_SOURCE_DIR) + "/docs/BOUNDARY_NAMING.md");
    auto trim = [](std::string t) {
        const auto b = t.find_first_not_of(" \t");
        if (b == std::string::npos) return std::string();
        return t.substr(b, t.find_last_not_of(" \t") - b + 1);
    };
    // `a`, `b` ... -> {a, b}
    auto quoted = [](const std::string& cell) {
        std::set<std::string> out;
        for (std::size_t open = cell.find('`'); open != std::string::npos; open = cell.find('`', open + 1)) {
            const std::size_t close = cell.find('`', open + 1);
            if (close == std::string::npos) break;
            out.insert(cell.substr(open + 1, close - open - 1));
            open = close;
        }
        return out;
    };
    auto is_keyword = [](const std::string& k) {
        return !k.empty() && std::all_of(k.begin(), k.end(), [](char ch) {
            return std::islower(static_cast<unsigned char>(ch)) || std::isdigit(static_cast<unsigned char>(ch)) || ch == '_';
        });
    };
    auto is_flag = [](const std::string& f) {
        return f == "–" || f == "-" ||
               (!f.empty() && std::all_of(f.begin(), f.end(), [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)); }));
    };
    std::map<std::string, std::pair<std::string, std::set<std::string>>> documented;
    for (const std::string& line : split_lines(doc)) {
        if (line.empty() || line[0] != '|') continue;
        std::vector<std::string> cells; // between the '|' separators
        std::size_t from = 1;
        for (std::size_t bar = line.find('|', from); bar != std::string::npos; bar = line.find('|', from)) {
            cells.push_back(trim(line.substr(from, bar - from)));
            from = bar + 1;
        }
        if (cells.size() < 4 || cells[0].size() < 3 || cells[0].front() != '`' || cells[0].back() != '`') continue;
        const std::string keyword = cells[0].substr(1, cells[0].size() - 2);
        if (!is_keyword(keyword) || !is_flag(cells[1])) continue;
        CHECK(documented.emplace(keyword, std::make_pair(cells[1], quoted(cells[2]))).second);
    }
    REQUIRE_FALSE(documented.empty());

    std::size_t expected_rows = 0;
    for (const BoundaryTypeInfo& info : boundary_type_table()) {
        if (info.type == BoundaryType::Custom) continue;
        ++expected_rows;
        CAPTURE(info.keyword);
        const auto it = documented.find(std::string(info.keyword));
        REQUIRE(it != documented.end());
        CHECK(it->second.first == (info.flag ? std::to_string(info.flag) : std::string("–")));
        const std::set<std::string> aliases(info.aliases.begin(), info.aliases.end());
        CHECK(it->second.second == aliases);
    }
    CHECK(documented.size() == expected_rows);
}
