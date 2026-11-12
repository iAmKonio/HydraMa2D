#pragma once

// Boundary naming convention: translation of boundary names (Gmsh physical
// group names today; SU2 markers / CGNS family names later) into boundary
// condition types and HydraMa2D flag codes.
//
// This table is the single source of truth for boundary names. The developer
// documentation (docs/BOUNDARY_NAMING.md) lists the same table, and
// tests/test_boundary_naming.cpp checks that the two agree; `HydraMa2D
// --boundary-names` prints it.
//
// Matching rules (see resolve_boundary_name and docs/BOUNDARY_NAMING.md):
//   1. The name is normalised: lower case, camelCase and letter/digit
//      boundaries split, every other character is a separator "_"
//      ("Velocity-Inlet 2" -> "velocity_inlet_2", "noSlipWall" -> "no_slip_wall").
//   2. A purely numeric name is a legacy flag code ("503" -> inlet).
//   3. Otherwise every keyword or alias is searched as a whole-token sequence
//      in the name ("bottom_wall_2" contains "wall"). A match inside a longer
//      match is discarded ("pressure_inlet" is a pressure inlet, not an inlet).
//   4. The remaining matches must agree on one type. A specific type beats the
//      generic type of its family ("rotor_wall" -> rotating, "hot_wall" ->
//      wall_hot); anything else ("inlet_wall") is ambiguous and rejected.
//
// Some formats also declare a boundary type (CGNS BCType_t). It is a *hint*
// combined with the name by the same family rule (see the overload of
// resolve_boundary_name with a hint): BCWall + "rotor_wall" -> rotating,
// BCWall + "top" -> wall, BCWall + "inlet" -> conflict.

#include "BoundaryKind.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hydrama::mesh {

enum class BoundaryType : unsigned char {
    // Implemented by the solver (legacy flag codes).
    Wall,             // 500
    WallHot,          // 501
    WallCold,         // 502
    Inlet,            // 503
    Outlet,           // 504
    Symmetry,         // 506
    InletProfile,     // 507
    InletParabolic,   // 508
    // Recognised, but not implemented by the solver: a mesh that uses them is
    // rejected (instead of silently treating the boundary as something else).
    Periodic,
    Rotating,
    MovingWall,
    Farfield,
    PressureInlet,
    MassFlowInlet,
    MassFlowOutlet,
    ConvectiveOutlet,
    NonReflecting,
    Opening,
    SupersonicInlet,
    SupersonicOutlet,
    Axis,
    Interface,
    Baffle,
    Fan,
    PorousJump,
    Empty,
    Wedge,
    // A flag code from a *.bofl file that is not in the table (legacy
    // behaviour: accepted with a warning). Never produced from a name.
    Custom,
};

struct BoundaryTypeInfo {
    BoundaryType type;
    std::string_view keyword;            // canonical marker name, e.g. "wall"
    int flag;                            // solver flag code, 0 = not implemented
    BoundaryType family;                 // generic type this type refines (itself if none)
    std::string_view description;
    std::vector<std::string_view> aliases; // alternative names (the keyword is not repeated)
};

// All types in declaration order (Custom last).
const std::vector<BoundaryTypeInfo>& boundary_type_table();
const BoundaryTypeInfo& boundary_type_info(BoundaryType type);

std::string_view to_string(BoundaryType type);  // canonical keyword
bool is_implemented(BoundaryType type);         // true if the solver supports it
std::optional<int> solver_flag(BoundaryType type);

// Type of a legacy flag code (500 -> Wall, ...); Custom for other codes.
BoundaryType boundary_type_from_flag(int flag);

// Coarse classification used by the wall geometry / turbulence code.
// Agrees with classify_boundary_flag(solver_flag(type)) for implemented types;
// Unknown for all others.
BoundaryKind to_boundary_kind(BoundaryType type);

// Rule 1 above. Returns "" for names without letters or digits.
std::string normalize_boundary_name(std::string_view name);

struct NameResolution {
    enum class Status {
        Resolved,    // type is valid
        Empty,       // no letters or digits
        NoKeyword,   // no keyword or alias found in the name
        Ambiguous,   // matches of different types, see candidates
        UnknownCode, // numeric name that is not a known flag code
        Conflict,    // the name and the hint define incompatible types (see candidates)
    };
    Status status = Status::NoKeyword;
    BoundaryType type = BoundaryType::Custom;
    std::string normalized;              // normalised name
    std::string matched;                 // normalised alias / code that decided the type
    std::vector<BoundaryType> candidates; // Ambiguous / Conflict: the competing types
    bool from_hint = false;              // type taken from the hint (the name has no keyword)

    bool resolved() const { return status == Status::Resolved; }
    // Human-readable explanation of a failed resolution.
    std::string message(std::string_view name) const;
};

NameResolution resolve_boundary_name(std::string_view name);

// The same type, or the type that refines the other one (its family is the
// other type), e.g. (wall, rotating) -> rotating; nullopt if incompatible.
std::optional<BoundaryType> combine_boundary_types(BoundaryType a, BoundaryType b);

// Name together with a type declared by the file (e.g. a CGNS BCType):
//   * name resolves to T:  combine(T, hint), or Conflict;
//   * name ambiguous:      the candidate compatible with the hint, if unique;
//   * no usable name (empty, no keyword, unknown code): the hint
//     (from_hint = true).
// Without a hint this is resolve_boundary_name(name).
NameResolution resolve_boundary_name(std::string_view name, std::optional<BoundaryType> hint);

// Plain-text table of every type, flag and alias (HydraMa2D --boundary-names).
std::string boundary_naming_summary();

} // namespace hydrama::mesh
