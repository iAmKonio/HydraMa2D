#pragma once

// Semantic classification of the numeric boundary flag codes (*.bofl files).
//
// This is the single source of truth for "which flag means what" for all new
// code (wall geometry, turbulence boundary conditions). The legacy routines
// (Mesh::determine_wall_nodes, apply_velocity_bcs, setup_pressure_bcs) still
// test the raw codes; tests/test_boundary_kind.cpp checks that both agree.

#include <string_view>

namespace hydrama {

// Known boundary flag codes (see README, "Boundary Flag Codes").
namespace flags {
inline constexpr int kWall = 500;          // adiabatic wall, no-slip
inline constexpr int kWallHot = 501;       // constant temperature wall (T = 1), no-slip
inline constexpr int kWallCold = 502;      // constant temperature wall (T = 0), no-slip
inline constexpr int kInlet = 503;         // prescribed inflow velocity
inline constexpr int kOutlet = 504;        // pressure outlet (p = 0)
inline constexpr int kSymmetry = 506;      // symmetry / slip (u . n = 0)
inline constexpr int kInletProfile = 507;  // backward-facing-step polynomial inflow profile
} // namespace flags

enum class BoundaryKind : unsigned char {
    Unknown,   // unassigned (0) or unrecognised code (e.g. 505)
    Wall,      // no-slip wall: 500, 501, 502
    Inlet,     // prescribed velocity: 503, 507
    Outlet,    // prescribed pressure: 504
    Symmetry,  // slip / symmetry: 506
};

constexpr BoundaryKind classify_boundary_flag(int flag) noexcept {
    switch (flag) {
    case flags::kWall:
    case flags::kWallHot:
    case flags::kWallCold:
        return BoundaryKind::Wall;
    case flags::kInlet:
    case flags::kInletProfile:
        return BoundaryKind::Inlet;
    case flags::kOutlet:
        return BoundaryKind::Outlet;
    case flags::kSymmetry:
        return BoundaryKind::Symmetry;
    default:
        return BoundaryKind::Unknown;
    }
}

constexpr bool is_wall_flag(int flag) noexcept {
    return classify_boundary_flag(flag) == BoundaryKind::Wall;
}

constexpr std::string_view to_string(BoundaryKind kind) noexcept {
    switch (kind) {
    case BoundaryKind::Wall: return "wall";
    case BoundaryKind::Inlet: return "inlet";
    case BoundaryKind::Outlet: return "outlet";
    case BoundaryKind::Symmetry: return "symmetry";
    case BoundaryKind::Unknown: break;
    }
    return "unknown";
}

} // namespace hydrama
