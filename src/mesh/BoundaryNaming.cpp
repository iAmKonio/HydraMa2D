#include "mesh/BoundaryNaming.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace hydrama::mesh {

namespace {

using T = BoundaryType;

// Keep in sync with docs/BOUNDARY_NAMING.md (checked by
// tests/test_boundary_naming.cpp). Aliases are written in normalised form
// or in any form that normalises to a unique token sequence.
std::vector<BoundaryTypeInfo> make_table() {
    return {
        // ---------------------------------------------------------------
        // Implemented by the solver
        // ---------------------------------------------------------------
        {T::Wall, "wall", flags::kWall, T::Wall,
         "No-slip wall, u = v = 0 (adiabatic in the legacy code table).",
         {"walls", "no_slip", "noslip", "no_slip_wall", "noslip_wall", "viscous_wall", "adiabatic_wall",
          "solid_wall"}},
        {T::WallHot, "wall_hot", flags::kWallHot, T::Wall,
         "No-slip wall tagged as constant temperature T = 1 (there is no energy equation: "
         "the flow sees a no-slip wall).",
         {"hot_wall", "wall_t1", "heated_wall", "isothermal_hot"}},
        {T::WallCold, "wall_cold", flags::kWallCold, T::Wall,
         "No-slip wall tagged as constant temperature T = 0 (there is no energy equation: "
         "the flow sees a no-slip wall).",
         {"cold_wall", "wall_t0", "cooled_wall", "isothermal_cold"}},
        {T::Inlet, "inlet", flags::kInlet, T::Inlet,
         "Velocity inlet, uniform u = 1, v = 0.",
         {"inlets", "inflow", "velocity_inlet", "inlet_velocity", "velocity_inflow", "uniform_inlet"}},
        {T::Outlet, "outlet", flags::kOutlet, T::Outlet,
         "Pressure outlet, p = 0 (velocity: natural / do-nothing condition).",
         {"outlets", "outflow", "pressure_outlet", "outlet_pressure", "exit", "fixed_pressure"}},
        {T::Symmetry, "symmetry", flags::kSymmetry, T::Symmetry,
         "Symmetry line / free-slip wall, u . n = 0.",
         {"sym", "symm", "symmetric", "symmetry_plane", "symmetry_line", "mirror", "slip", "slip_wall",
          "free_slip", "freeslip", "inviscid_wall", "euler_wall"}},
        {T::InletProfile, "inlet_profile", flags::kInletProfile, T::Inlet,
         "Backward-facing-step inflow: fixed 6th-order polynomial u(y), v = 0.",
         {"profile_inlet", "step_inlet", "bfs_inlet", "inlet_bfs"}},
        {T::InletParabolic, "inlet_parabolic", flags::kInletParabolic, T::Inlet,
         "Fully developed (Poiseuille) inflow: parabolic profile along the inward normal, mean velocity 1 "
         "over each connected inlet segment (1.5 at its centre).",
         {"parabolic_inlet", "parabolic_inflow", "poiseuille_inlet", "inlet_poiseuille", "fully_developed_inlet",
          "developed_inlet"}},

        // ---------------------------------------------------------------
        // Recognised, not implemented (the mesh is rejected)
        // ---------------------------------------------------------------
        {T::Periodic, "periodic", 0, T::Periodic,
         "Periodic / cyclic boundary pair.",
         {"cyclic", "cyclic_ami", "periodicity"}},
        {T::Rotating, "rotating", 0, T::Wall,
         "Wall rotating about an axis (prescribed angular velocity).",
         {"rotating_wall", "rotor", "spinning_wall"}},
        {T::MovingWall, "moving_wall", 0, T::Wall,
         "Wall translating with a prescribed tangential velocity (e.g. a cavity lid).",
         {"moving", "sliding_wall", "translating_wall", "lid", "moving_lid"}},
        {T::Farfield, "farfield", 0, T::Farfield,
         "Far-field / free-stream boundary.",
         {"far_field", "freestream", "free_stream"}},
        {T::PressureInlet, "pressure_inlet", 0, T::Inlet,
         "Total (stagnation) pressure inlet.",
         {"total_pressure_inlet", "stagnation_inlet", "stagnation_pressure_inlet", "inlet_total_pressure",
          "pressure_inflow"}},
        {T::MassFlowInlet, "mass_flow_inlet", 0, T::Inlet,
         "Prescribed mass flow rate inlet.",
         {"massflow_inlet", "mass_flow_rate_inlet", "mass_inflow"}},
        {T::MassFlowOutlet, "mass_flow_outlet", 0, T::Outlet,
         "Prescribed mass flow rate outlet.",
         {"massflow_outlet", "mass_flow_rate_outlet", "mass_outflow"}},
        {T::ConvectiveOutlet, "convective_outlet", 0, T::Outlet,
         "Convective (advective) outflow condition.",
         {"advective_outlet", "convective_outflow", "advective_outflow"}},
        {T::NonReflecting, "non_reflecting", 0, T::NonReflecting,
         "Non-reflecting / characteristic boundary (Riemann, Giles).",
         {"nonreflecting", "nrbc", "riemann", "giles", "characteristic"}},
        {T::Opening, "opening", 0, T::Opening,
         "Opening: inflow or outflow depending on the local flow direction.",
         {"inlet_outlet", "inletoutlet", "entrainment", "open_boundary"}},
        {T::SupersonicInlet, "supersonic_inlet", 0, T::Inlet,
         "Supersonic inflow (compressible flow only).",
         {"supersonic_inflow"}},
        {T::SupersonicOutlet, "supersonic_outlet", 0, T::Outlet,
         "Supersonic outflow (compressible flow only).",
         {"supersonic_outflow"}},
        {T::Axis, "axis", 0, T::Symmetry,
         "Axis of an axisymmetric domain (the solver is planar 2D).",
         {"axisymmetric", "axisymmetry", "symmetry_axis", "axis_of_symmetry"}},
        {T::Interface, "interface", 0, T::Interface,
         "Internal interface between zones (fluid / conjugate heat transfer / sliding / mixing plane).",
         {"interior", "internal", "fluid_interface", "cht_interface", "sliding_interface", "mixing_plane",
          "nearfield", "near_field"}},
        {T::Baffle, "baffle", 0, T::Wall,
         "Zero-thickness internal wall.",
         {"baffles", "thin_wall", "internal_wall"}},
        {T::Fan, "fan", 0, T::Fan,
         "Fan / actuator disk: internal pressure-jump interface.",
         {"actuator_disk", "actuator_disc", "intake_fan", "exhaust_fan"}},
        {T::PorousJump, "porous_jump", 0, T::PorousJump,
         "Porous jump / radiator: internal resistance interface.",
         {"radiator"}},
        {T::Empty, "empty", 0, T::Empty,
         "Front / back planes of a one-cell-thick 3D mesh (not a boundary of a 2D mesh).",
         {"front_and_back", "frontandback", "front_back"}},
        {T::Wedge, "wedge", 0, T::Wedge,
         "Wedge planes of an axisymmetric 3D mesh (not a boundary of a 2D mesh).",
         {}},

        {T::Custom, "custom", 0, T::Custom,
         "A flag code from a *.bofl file that is not in this table (kept, with a warning).",
         {}},
    };
}

bool is_alnum(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; }
bool is_upper(char c) { return std::isupper(static_cast<unsigned char>(c)) != 0; }
bool is_lower(char c) { return std::islower(static_cast<unsigned char>(c)) != 0; }
bool is_digit(char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }
bool is_alpha(char c) { return std::isalpha(static_cast<unsigned char>(c)) != 0; }

std::vector<std::string> split_tokens(const std::string& normalized) {
    std::vector<std::string> tokens;
    std::string current;
    for (char c : normalized) {
        if (c == '_') {
            if (!current.empty()) tokens.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    if (!current.empty()) tokens.push_back(current);
    return tokens;
}

struct Pattern {
    std::vector<std::string> tokens;
    std::string text; // normalised alias
    BoundaryType type;
};

// Every keyword and alias of every nameable type, normalised.
const std::vector<Pattern>& patterns() {
    static const std::vector<Pattern> list = [] {
        std::vector<Pattern> out;
        for (const BoundaryTypeInfo& info : boundary_type_table()) {
            if (info.type == T::Custom) continue;
            auto add = [&](std::string_view alias) {
                const std::string n = normalize_boundary_name(alias);
                out.push_back({split_tokens(n), n, info.type});
            };
            add(info.keyword);
            for (std::string_view a : info.aliases) add(a);
        }
        return out;
    }();
    return list;
}

struct Match {
    std::size_t start;
    std::size_t length;
    const Pattern* pattern;
};

} // namespace

const std::vector<BoundaryTypeInfo>& boundary_type_table() {
    static const std::vector<BoundaryTypeInfo> table = make_table();
    return table;
}

const BoundaryTypeInfo& boundary_type_info(BoundaryType type) {
    for (const BoundaryTypeInfo& info : boundary_type_table())
        if (info.type == type) return info;
    throw std::logic_error("boundary_type_info: type missing from the table");
}

std::string_view to_string(BoundaryType type) { return boundary_type_info(type).keyword; }

bool is_implemented(BoundaryType type) { return boundary_type_info(type).flag != 0; }

std::optional<int> solver_flag(BoundaryType type) {
    const int flag = boundary_type_info(type).flag;
    if (flag == 0) return std::nullopt;
    return flag;
}

BoundaryType boundary_type_from_flag(int flag) {
    if (flag != 0) {
        for (const BoundaryTypeInfo& info : boundary_type_table())
            if (info.flag == flag) return info.type;
    }
    return T::Custom;
}

BoundaryKind to_boundary_kind(BoundaryType type) {
    const std::optional<int> flag = solver_flag(type);
    return flag ? classify_boundary_flag(*flag) : BoundaryKind::Unknown;
}

std::string normalize_boundary_name(std::string_view name) {
    std::string out;
    out.reserve(name.size() + 4);
    for (std::size_t i = 0; i < name.size(); ++i) {
        const char c = name[i];
        if (!is_alnum(c)) {
            if (!out.empty() && out.back() != '_') out += '_';
            continue;
        }
        if (i > 0 && !out.empty() && out.back() != '_' && is_alnum(name[i - 1])) {
            const char prev = name[i - 1];
            const char next = i + 1 < name.size() ? name[i + 1] : '\0';
            const bool split = (is_upper(c) && (is_lower(prev) || is_digit(prev))) ||  // noSlip, wall2Top
                               (is_upper(c) && is_upper(prev) && is_lower(next)) ||    // CHTInterface
                               (is_digit(c) && is_alpha(prev)) ||                      // wall2
                               (is_alpha(c) && is_digit(prev));                        // 2wall
            if (split) out += '_';
        }
        out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out;
}

NameResolution resolve_boundary_name(std::string_view name) {
    NameResolution r;
    r.normalized = normalize_boundary_name(name);
    if (r.normalized.empty()) {
        r.status = NameResolution::Status::Empty;
        return r;
    }

    // Rule 2: legacy flag code.
    if (std::all_of(r.normalized.begin(), r.normalized.end(), is_digit)) {
        int code = -1;
        if (r.normalized.size() <= 6) code = std::stoi(r.normalized);
        const BoundaryType type = boundary_type_from_flag(code);
        if (type == T::Custom) {
            r.status = NameResolution::Status::UnknownCode;
            return r;
        }
        r.status = NameResolution::Status::Resolved;
        r.type = type;
        r.matched = r.normalized;
        return r;
    }

    // Rule 3: whole-token matches of every keyword / alias.
    const std::vector<std::string> tokens = split_tokens(r.normalized);
    std::vector<Match> matches;
    for (const Pattern& p : patterns()) {
        const std::size_t len = p.tokens.size();
        if (len == 0 || len > tokens.size()) continue;
        for (std::size_t s = 0; s + len <= tokens.size(); ++s) {
            if (std::equal(p.tokens.begin(), p.tokens.end(), tokens.begin() + static_cast<std::ptrdiff_t>(s))) {
                matches.push_back({s, len, &p});
            }
        }
    }
    if (matches.empty()) {
        r.status = NameResolution::Status::NoKeyword;
        return r;
    }
    // In order of position in the name (candidates are reported in that order).
    std::stable_sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
        return a.start != b.start ? a.start < b.start : a.length > b.length;
    });

    // Discard matches that lie inside a strictly longer match.
    std::vector<Match> maximal;
    for (const Match& m : matches) {
        const bool inside = std::any_of(matches.begin(), matches.end(), [&](const Match& o) {
            return o.length > m.length && o.start <= m.start && m.start + m.length <= o.start + o.length;
        });
        if (!inside) maximal.push_back(m);
    }

    // Rule 4: agree on one type; a specific type beats its generic family.
    std::vector<BoundaryType> types;
    for (const Match& m : maximal)
        if (std::find(types.begin(), types.end(), m.pattern->type) == types.end()) types.push_back(m.pattern->type);

    BoundaryType chosen = types.front();
    if (types.size() == 2) {
        chosen = combine_boundary_types(types[0], types[1]).value_or(T::Custom);
    } else if (types.size() > 2) {
        chosen = T::Custom;
    }
    if (chosen == T::Custom) {
        r.status = NameResolution::Status::Ambiguous;
        r.candidates = types;
        return r;
    }

    // The decisive alias: the longest match of the chosen type.
    const Match* best = nullptr;
    for (const Match& m : maximal)
        if (m.pattern->type == chosen && (!best || m.length > best->length)) best = &m;
    r.status = NameResolution::Status::Resolved;
    r.type = chosen;
    r.matched = best->pattern->text;
    return r;
}

std::optional<BoundaryType> combine_boundary_types(BoundaryType a, BoundaryType b) {
    if (a == b) return a;
    if (boundary_type_info(a).family == b) return a; // a refines b
    if (boundary_type_info(b).family == a) return b; // b refines a
    return std::nullopt;
}

NameResolution resolve_boundary_name(std::string_view name, std::optional<BoundaryType> hint) {
    NameResolution r = resolve_boundary_name(name);
    if (!hint) return r;
    switch (r.status) {
    case NameResolution::Status::Resolved:
        if (const std::optional<BoundaryType> c = combine_boundary_types(r.type, *hint)) {
            r.type = *c;
        } else {
            r.status = NameResolution::Status::Conflict;
            r.candidates = {r.type, *hint};
        }
        return r;
    case NameResolution::Status::Ambiguous: {
        std::vector<BoundaryType> compatible;
        for (BoundaryType t : r.candidates) {
            const std::optional<BoundaryType> c = combine_boundary_types(t, *hint);
            if (c && std::find(compatible.begin(), compatible.end(), *c) == compatible.end()) compatible.push_back(*c);
        }
        if (compatible.size() == 1) {
            r.status = NameResolution::Status::Resolved;
            r.type = compatible.front();
            r.candidates.clear();
        }
        return r;
    }
    case NameResolution::Status::Empty:
    case NameResolution::Status::NoKeyword:
    case NameResolution::Status::UnknownCode:
        r.status = NameResolution::Status::Resolved;
        r.type = *hint;
        r.matched.clear();
        r.from_hint = true;
        return r;
    case NameResolution::Status::Conflict:
        break;
    }
    return r;
}

std::string NameResolution::message(std::string_view name) const {
    std::ostringstream os;
    os << "boundary name '" << name << "'";
    switch (status) {
    case Status::Resolved:
        os << " is a " << to_string(type) << " boundary";
        if (from_hint) os << " (from the boundary type declared in the file)";
        else os << " (matched '" << matched << "')";
        break;
    case Status::Empty:
        os << " contains no letters or digits";
        break;
    case Status::NoKeyword:
        os << " (normalised '" << normalized
           << "') contains no boundary keyword; rename it to include one, e.g. '" << normalized
           << "_wall' (see docs/BOUNDARY_NAMING.md or HydraMa2D --boundary-names), or give its "
              "physical tag a flag code in the *.bofl file";
        break;
    case Status::Ambiguous: {
        os << " (normalised '" << normalized << "') is ambiguous: it matches";
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            os << (i == 0 ? " " : (i + 1 == candidates.size() ? " and " : ", ")) << to_string(candidates[i]);
        }
        os << "; rename it so that it contains only one boundary keyword";
        break;
    }
    case Status::UnknownCode:
        os << " is numeric but not a known flag code (500, 501, 502, 503, 504, 506, 507)";
        break;
    case Status::Conflict:
        os << " (normalised '" << normalized << "') means " << to_string(candidates.at(0))
           << ", but the boundary type declared in the file is " << to_string(candidates.at(1))
           << "; make the name and the declared type agree";
        break;
    }
    return os.str();
}

std::string boundary_naming_summary() {
    std::ostringstream os;
    os << "HydraMa2D boundary names (case-insensitive; a keyword may appear anywhere in the name,\n"
          "e.g. 'bottom_wall', 'Inlet-2', 'symmetryPlane'). See docs/BOUNDARY_NAMING.md.\n\n";
    auto section = [&](bool implemented) {
        os << (implemented ? "Implemented by the solver:\n" : "\nRecognised but not implemented (mesh is rejected):\n");
        for (const BoundaryTypeInfo& info : boundary_type_table()) {
            if (info.type == T::Custom || (info.flag != 0) != implemented) continue;
            os << "  " << info.keyword;
            if (implemented) os << " (flag " << info.flag << ")";
            os << ": " << info.description << "\n";
            if (!info.aliases.empty()) {
                os << "      aliases:";
                for (std::string_view a : info.aliases) os << " " << a;
                os << "\n";
            }
        }
    };
    section(true);
    section(false);
    return os.str();
}

} // namespace hydrama::mesh
