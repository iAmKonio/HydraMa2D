#pragma once

// Turbulence configuration: system/<case>.turbulence
//
// Optional key-value file. If it does not exist the run is laminar and the
// solver behaves exactly as before. Format (one entry per line, '#' starts a
// comment, keys are lower case, enumeration values are case-insensitive,
// numbers accept Fortran 'd' exponents):
//
//   model                  spalart-allmaras   # none | spalart-allmaras | k-epsilon | rng-k-epsilon |
//                                             # realizable-k-epsilon | launder-sharma-k-epsilon | k-omega-sst
//   wall_treatment         low-re             # low-re | wall-function | automatic (default: per model,
//                                             # wall-function for k-epsilon, low-re otherwise)
//   wall_distance          exact              # exact (= aabb-tree) | brute-force
//   gradient_recovery      ppr                # ppr (polynomial preserving) | lumped-projection
//   inlet_intensity        0.05               # turbulence intensity I (fraction of the inflow speed)
//   inlet_length_scale     0.1                # turbulence length scale l (mesh units)
//   inlet_viscosity_ratio  10                 # nu_t / nu at inlets (optional)
//   inlet_nu_tilde_ratio   3                  # SA: nu_tilde / nu at inlets (optional, default 3)
//   inlet_k / inlet_epsilon / inlet_omega     # direct inlet values (optional, override I and l)
//   reference_velocity     1                  # U_ref in k = 1.5 (U_ref I)^2
//   production_limiter     10                 # two-equation models: P_k <= c eps (0 = off)
//   sa_ft2                 on                 # SA: trip term f_t2 on (standard SA) | off (SA-noft2)
//   nut_ratio_max          1e5                # safety cap on nu_t / nu
//   boundary 503 inlet  intensity 0.03 length_scale 0.05 viscosity_ratio 5 nu_tilde_ratio 3
//   boundary 507 inlet  k 1e-3 epsilon 2e-4          # direct values (also: omega)
//   boundary 506 symmetry
//
// `boundary <flag> <kind> [key value]...` overrides how the turbulence
// equations treat a boundary flag (kind: wall | inlet | outlet | symmetry)
// and/or its inlet values; it never changes the velocity / pressure BCs.
// Every key may appear at most once, every flag at most once.

#include "BoundaryKind.hpp"
#include "geometry/FieldOps.hpp"
#include "geometry/WallDistance.hpp"

#include <istream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hydrama {

enum class TurbulenceModelType {
    None,
    SpalartAllmaras,       // SA with negative-SA extension
    KEpsilon,              // standard (Launder-Spalding), wall functions
    RngKEpsilon,
    RealizableKEpsilon,
    LaunderSharmaKEpsilon, // low-Re
    KOmegaSST,             // Menter 2003
    GammaReThetaSST,       // Langtry-Menter (2009) 4-equation transition model
};

enum class WallTreatmentType {
    LowRe,        // resolve the viscous sub-layer (y+ ~ 1)
    WallFunction, // standard log law
    Automatic,    // blended (Spalding / automatic omega)
};

struct TurbulenceBoundaryOverride {
    int flag = 0;
    BoundaryKind kind = BoundaryKind::Unknown;
    std::optional<double> intensity;
    std::optional<double> length_scale;
    std::optional<double> viscosity_ratio;
    std::optional<double> nu_tilde_ratio;
    std::optional<double> k;
    std::optional<double> epsilon;
    std::optional<double> omega;
};

// Inlet turbulence specification after applying defaults and overrides.
struct InletTurbulence {
    double intensity = 0.05;
    double length_scale = 0.1;
    std::optional<double> viscosity_ratio;
    std::optional<double> nu_tilde_ratio;
    std::optional<double> k;       // direct values take precedence over I, l, nu_t/nu
    std::optional<double> epsilon;
    std::optional<double> omega;
};

struct TurbulenceConfig {
    bool present = false; // the .turbulence file exists
    TurbulenceModelType model = TurbulenceModelType::None;
    std::optional<WallTreatmentType> wall_treatment; // unset: model default
    WallDistanceMethod wall_distance = WallDistanceMethod::AabbTree;
    GradientRecovery gradient_recovery = GradientRecovery::PolynomialPreserving;

    InletTurbulence inlet; // defaults for every inlet flag

    double reference_velocity = 1.0; // U_ref for intensity-based inlet k
    double production_limiter = 10.0; // P_k <= c eps (k-epsilon) / c beta* k omega (SST); 0 = off

    bool sa_ft2 = true;
    double nut_ratio_max = 1.0e5;

    std::vector<TurbulenceBoundaryOverride> boundaries;

    bool is_laminar() const noexcept { return model == TurbulenceModelType::None; }

    // Configured wall treatment, or the model default (wall functions for the
    // high-Re k-epsilon variants, low-Re otherwise).
    WallTreatmentType effective_wall_treatment() const noexcept;

    // Override for `flag`, or nullptr.
    const TurbulenceBoundaryOverride* find_boundary(int flag) const noexcept;
    // Boundary kind the turbulence equations use for `flag`.
    BoundaryKind boundary_kind(int flag) const noexcept;
    // Inlet values for `flag` (global defaults merged with the override).
    InletTurbulence inlet_for(int flag) const;
};

std::string_view to_string(TurbulenceModelType model) noexcept;
std::string_view to_string(WallTreatmentType treatment) noexcept;
std::optional<TurbulenceModelType> parse_turbulence_model(std::string_view name);
std::optional<WallTreatmentType> parse_wall_treatment(std::string_view name);

// Parse a .turbulence stream into `config` (which is reset first).
// Returns false and a message "line N: ..." on the first error.
bool parse_turbulence_config(std::istream& in, TurbulenceConfig& config, std::string& error);

// Read system/<case>.turbulence. A missing file is not an error: the config
// is reset to laminar with present = false. Returns false only if the file
// exists but is invalid (message in `error`).
bool read_turbulence_config(const std::string& filepath, TurbulenceConfig& config, std::string& error);

} // namespace hydrama
