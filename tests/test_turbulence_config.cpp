// system/<case>.turbulence parser (TurbulenceConfig), IO::read_turbulence and
// the turbulence model factory.

#include "IO.hpp"
#include "IoTestUtils.hpp"
#include "turbulence/SpalartAllmaras.hpp"
#include "turbulence/TurbulenceConfig.hpp"
#include "turbulence/TurbulenceBC.hpp"
#include "turbulence/TurbulenceFactory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <sstream>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::StartsWith;
using Catch::Matchers::WithinRel;

namespace {

struct Parsed {
    bool ok = false;
    TurbulenceConfig config;
    std::string error;
};

Parsed parse(const std::string& text) {
    Parsed p;
    std::istringstream in(text);
    p.ok = parse_turbulence_config(in, p.config, p.error);
    return p;
}

} // namespace

TEST_CASE("Turbulence config: defaults", "[turbulence_config]") {
    const TurbulenceConfig def;
    CHECK_FALSE(def.present);
    CHECK(def.is_laminar());

    const Parsed p = parse("");
    REQUIRE(p.ok);
    const TurbulenceConfig& c = p.config;
    CHECK(c.present);
    CHECK(c.model == TurbulenceModelType::None);
    CHECK_FALSE(c.wall_treatment.has_value()); // model default
    CHECK(c.effective_wall_treatment() == WallTreatmentType::LowRe);
    CHECK(c.wall_distance == WallDistanceMethod::AabbTree);
    CHECK(c.gradient_recovery == GradientRecovery::PolynomialPreserving);
    CHECK(c.inlet.intensity == 0.05);
    CHECK(c.inlet.length_scale == 0.1);
    CHECK_FALSE(c.inlet.viscosity_ratio.has_value());
    CHECK_FALSE(c.inlet.nu_tilde_ratio.has_value());
    CHECK(c.sa_ft2);
    CHECK(c.nut_ratio_max == 1.0e5);
    CHECK(c.reference_velocity == 1.0);
    CHECK(c.production_limiter == 10.0);
    CHECK_FALSE(c.inlet.k.has_value());
    CHECK_FALSE(c.inlet.epsilon.has_value());
    CHECK_FALSE(c.inlet.omega.has_value());
    CHECK(c.boundaries.empty());
}

TEST_CASE("Turbulence config: wall treatment defaults per model", "[turbulence_config]") {
    TurbulenceConfig c;
    for (auto m : {TurbulenceModelType::SpalartAllmaras, TurbulenceModelType::KOmegaSST,
                   TurbulenceModelType::GammaReThetaSST, TurbulenceModelType::LaunderSharmaKEpsilon,
                   TurbulenceModelType::None}) {
        c.model = m;
        CHECK(c.effective_wall_treatment() == WallTreatmentType::LowRe);
    }
    for (auto m : {TurbulenceModelType::KEpsilon, TurbulenceModelType::RngKEpsilon,
                   TurbulenceModelType::RealizableKEpsilon}) {
        c.model = m;
        CHECK(c.effective_wall_treatment() == WallTreatmentType::WallFunction);
    }
    c.wall_treatment = WallTreatmentType::Automatic;
    CHECK(c.effective_wall_treatment() == WallTreatmentType::Automatic);
}

TEST_CASE("Turbulence config: two-equation keys", "[turbulence_config]") {
    const Parsed p = parse(
        "model k-omega-sst\n"
        "inlet_k 2.5e-3\n"
        "inlet_omega 40\n"
        "inlet_epsilon 1d-3\n"
        "reference_velocity 2.0\n"
        "production_limiter 0\n"
        "boundary 503 inlet k 1e-3 epsilon 2e-4 omega 3\n");
    REQUIRE(p.ok);
    const TurbulenceConfig& c = p.config;
    CHECK(c.inlet.k == 2.5e-3);
    CHECK(c.inlet.omega == 40.0);
    CHECK(c.inlet.epsilon == 1e-3);
    CHECK(c.reference_velocity == 2.0);
    CHECK(c.production_limiter == 0.0);
    const InletTurbulence in = c.inlet_for(503);
    CHECK(in.k == 1e-3);
    CHECK(in.epsilon == 2e-4);
    CHECK(in.omega == 3.0);
    const InletTurbulence other = c.inlet_for(507);
    CHECK(other.k == 2.5e-3);
    CHECK(other.omega == 40.0);
}

TEST_CASE("Two-equation inlet values", "[turbulence_config]") {
    const double nu = 1e-3, cmu = 0.09;
    InletTurbulence in;
    in.intensity = 0.05;
    in.length_scale = 0.2;
    SECTION("intensity and length scale") {
        const TwoEquationInlet v = two_equation_inlet(in, nu, 2.0);
        const double k = 1.5 * 0.1 * 0.1; // U_ref I = 0.1
        CHECK_THAT(v.k, WithinRel(k, 1e-15));
        CHECK_THAT(v.epsilon, WithinRel(std::pow(cmu, 0.75) * std::pow(k, 1.5) / 0.2, 1e-14));
        CHECK_THAT(v.omega, WithinRel(v.epsilon / (cmu * k), 1e-14));
        CHECK_THAT(v.omega, WithinRel(std::sqrt(k) / (std::pow(cmu, 0.25) * 0.2), 1e-14));
    }
    SECTION("viscosity ratio") {
        in.viscosity_ratio = 10.0;
        const TwoEquationInlet v = two_equation_inlet(in, nu, 1.0);
        CHECK_THAT(v.epsilon, WithinRel(cmu * v.k * v.k / (10.0 * nu), 1e-14));
        CHECK_THAT(v.omega, WithinRel(v.k / (10.0 * nu), 1e-14)); // nu_t = k / omega = 10 nu
    }
    SECTION("direct values take precedence") {
        in.viscosity_ratio = 10.0;
        in.k = 0.01;
        in.omega = 5.0;
        TwoEquationInlet v = two_equation_inlet(in, nu, 1.0);
        CHECK(v.k == 0.01);
        CHECK(v.omega == 5.0);
        CHECK_THAT(v.epsilon, WithinRel(cmu * 0.01 * 5.0, 1e-15));
        in.epsilon = 2e-3;
        v = two_equation_inlet(in, nu, 1.0);
        CHECK(v.epsilon == 2e-3);
        CHECK(v.omega == 5.0);
        in.omega.reset();
        v = two_equation_inlet(in, nu, 1.0);
        CHECK_THAT(v.omega, WithinRel(2e-3 / (cmu * 0.01), 1e-14));
    }
    SECTION("zero turbulence stays finite") {
        in.intensity = 0.0;
        const TwoEquationInlet v = two_equation_inlet(in, nu, 1.0);
        CHECK(v.k == 0.0);
        CHECK(v.epsilon > 0.0);
        CHECK(v.omega > 0.0);
    }
}

TEST_CASE("Turbulence config: complete file", "[turbulence_config]") {
    const Parsed p = parse(
        "# HydraMa turbulence settings\r\n"
        "model            Spalart-Allmaras   # case-insensitive value\r\n"
        "\r\n"
        "wall_treatment   low-re\r\n"
        "wall_distance    brute-force\r\n"
        "gradient_recovery Lumped-Projection\r\n"
        "inlet_intensity  0.03\r\n"
        "inlet_length_scale 2.5d-02\r\n"
        "inlet_viscosity_ratio 10\r\n"
        "inlet_nu_tilde_ratio  4.5\r\n"
        "sa_ft2           off\r\n"
        "nut_ratio_max    1.0D+06\r\n"
        "   boundary 503 inlet intensity 0.01 length_scale 0.2 viscosity_ratio 5 nu_tilde_ratio 3\r\n"
        "boundary 506 wall\r\n"
        "boundary 500 SYMMETRY\r\n");
    REQUIRE(p.ok);
    const TurbulenceConfig& c = p.config;
    CHECK(c.model == TurbulenceModelType::SpalartAllmaras);
    CHECK(c.wall_treatment == WallTreatmentType::LowRe);
    CHECK(c.wall_distance == WallDistanceMethod::BruteForce);
    CHECK(c.gradient_recovery == GradientRecovery::LumpedProjection);
    CHECK(c.inlet.intensity == 0.03);
    CHECK(c.inlet.length_scale == 0.025);
    CHECK(c.inlet.viscosity_ratio == 10.0);
    CHECK(c.inlet.nu_tilde_ratio == 4.5);
    CHECK_FALSE(c.sa_ft2);
    CHECK(c.nut_ratio_max == 1.0e6);
    REQUIRE(c.boundaries.size() == 3);
    CHECK(c.boundaries[0].flag == 503);
    CHECK(c.boundaries[0].kind == BoundaryKind::Inlet);
    CHECK(c.boundaries[0].intensity == 0.01);
    CHECK(c.boundaries[0].length_scale == 0.2);
    CHECK(c.boundaries[0].viscosity_ratio == 5.0);
    CHECK(c.boundaries[0].nu_tilde_ratio == 3.0);
    CHECK(c.boundaries[1].kind == BoundaryKind::Wall);
    CHECK(c.boundaries[2].kind == BoundaryKind::Symmetry);

    SECTION("boundary kind: overrides take precedence over the flag classification") {
        CHECK(c.boundary_kind(503) == BoundaryKind::Inlet);
        CHECK(c.boundary_kind(506) == BoundaryKind::Wall);     // overridden
        CHECK(c.boundary_kind(500) == BoundaryKind::Symmetry); // overridden
        CHECK(c.boundary_kind(501) == BoundaryKind::Wall);     // classification
        CHECK(c.boundary_kind(504) == BoundaryKind::Outlet);
        CHECK(c.boundary_kind(507) == BoundaryKind::Inlet);
        CHECK(c.find_boundary(504) == nullptr);
    }

    SECTION("inlet values: per-flag override merged with the global defaults") {
        const InletTurbulence a = c.inlet_for(503);
        CHECK(a.intensity == 0.01);
        CHECK(a.length_scale == 0.2);
        CHECK(a.viscosity_ratio == 5.0);
        CHECK(a.nu_tilde_ratio == 3.0);
        const InletTurbulence b = c.inlet_for(507); // no override
        CHECK(b.intensity == 0.03);
        CHECK(b.length_scale == 0.025);
        CHECK(b.viscosity_ratio == 10.0);
        CHECK(b.nu_tilde_ratio == 4.5);
    }
}

TEST_CASE("Turbulence config: partial inlet override keeps the other defaults", "[turbulence_config]") {
    const Parsed p = parse("inlet_intensity 0.07\nboundary 507 inlet length_scale 0.3\n");
    REQUIRE(p.ok);
    const InletTurbulence in = p.config.inlet_for(507);
    CHECK(in.intensity == 0.07);
    CHECK(in.length_scale == 0.3);
    CHECK_FALSE(in.viscosity_ratio.has_value());
}

TEST_CASE("Turbulence config: model and wall treatment names", "[turbulence_config]") {
    const auto model = GENERATE(TurbulenceModelType::None, TurbulenceModelType::SpalartAllmaras,
                                TurbulenceModelType::KEpsilon, TurbulenceModelType::RngKEpsilon,
                                TurbulenceModelType::RealizableKEpsilon, TurbulenceModelType::LaunderSharmaKEpsilon,
                                TurbulenceModelType::KOmegaSST, TurbulenceModelType::GammaReThetaSST);
    CHECK(parse_turbulence_model(to_string(model)) == model); // canonical names round-trip

    CHECK(parse_turbulence_model("laminar") == TurbulenceModelType::None);
    CHECK(parse_turbulence_model("SA") == TurbulenceModelType::SpalartAllmaras);
    CHECK(parse_turbulence_model("sa-neg") == TurbulenceModelType::SpalartAllmaras);
    CHECK(parse_turbulence_model("SST") == TurbulenceModelType::KOmegaSST);
    CHECK(parse_turbulence_model("gamma-retheta-sst") == TurbulenceModelType::GammaReThetaSST);
    CHECK(parse_turbulence_model("transition-sst") == TurbulenceModelType::GammaReThetaSST);
    CHECK(parse_turbulence_model("langtry-menter") == TurbulenceModelType::GammaReThetaSST);
    CHECK_FALSE(parse_turbulence_model("k-omega").has_value());
    CHECK(to_string(TurbulenceModelType::SpalartAllmaras) == "spalart-allmaras");
    CHECK(to_string(TurbulenceModelType::KOmegaSST) == "k-omega-sst");
    CHECK(to_string(TurbulenceModelType::GammaReThetaSST) == "gamma-re-theta-sst");

    for (auto t : {WallTreatmentType::LowRe, WallTreatmentType::WallFunction, WallTreatmentType::Automatic}) {
        CHECK(parse_wall_treatment(to_string(t)) == t);
    }
    CHECK(parse_wall_treatment("Wall-Function") == WallTreatmentType::WallFunction);
    CHECK_FALSE(parse_wall_treatment("high-re").has_value());
}

TEST_CASE("Turbulence config: boolean switches", "[turbulence_config]") {
    for (const char* on : {"on", "ON", "true", "yes", "1"}) {
        const Parsed p = parse(std::string("sa_ft2 ") + on + "\n");
        REQUIRE(p.ok);
        CHECK(p.config.sa_ft2);
    }
    for (const char* off : {"off", "False", "no", "0"}) {
        const Parsed p = parse(std::string("sa_ft2 ") + off + "\n");
        REQUIRE(p.ok);
        CHECK_FALSE(p.config.sa_ft2);
    }
}

TEST_CASE("Turbulence config: errors report the line and the problem", "[turbulence_config]") {
    struct Bad {
        std::string text;
        std::string message;
    };
    const Bad bad = GENERATE(Bad{"model sa\nfoo 1\n", "line 2: unknown key 'foo'"},
                             Bad{"model sa\nmodel sa\n", "line 2: duplicate key 'model'"},
                             Bad{"model\n", "line 1: 'model' expects 1 value, got 0"},
                             Bad{"model sa sst\n", "line 1: 'model' expects 1 value, got 2"},
                             Bad{"model k-omega\n", "line 1: unknown model 'k-omega'"},
                             Bad{"wall_treatment high-re\n", "unknown wall_treatment 'high-re'"},
                             Bad{"wall_distance poisson\n", "unknown wall_distance 'poisson'"},
                             Bad{"gradient_recovery spr\n", "unknown gradient_recovery 'spr'"},
                             Bad{"inlet_intensity 5%\n", "invalid number '5%'"},
                             Bad{"inlet_intensity -0.1\n", "inlet_intensity must be >= 0"},
                             Bad{"inlet_length_scale 0\n", "inlet_length_scale must be > 0"},
                             Bad{"nut_ratio_max nan\n", "invalid number 'nan'"},
                             Bad{"inlet_viscosity_ratio 1e999\n", "invalid number '1e999'"},
                             Bad{"sa_ft2 maybe\n", "invalid switch 'maybe'"},
                             Bad{"\n\n# c\nboundary 503\n", "line 4: 'boundary' expects"},
                             Bad{"boundary 50x inlet\n", "invalid integer '50x'"},
                             Bad{"boundary 503 outflow\n", "unknown boundary kind 'outflow'"},
                             Bad{"boundary 503 inlet intensity\n", "options must be key value pairs"},
                             Bad{"boundary 503 inlet speed 3\n", "unknown option 'speed'"},
                             Bad{"boundary 503 inlet epsilon 0\n", "epsilon must be > 0"},
                             Bad{"boundary 503 inlet k -1\n", "k must be >= 0"},
                             Bad{"inlet_omega 0\n", "inlet_omega must be > 0"},
                             Bad{"reference_velocity 0\n", "reference_velocity must be > 0"},
                             Bad{"production_limiter -1\n", "production_limiter must be >= 0"},
                             Bad{"boundary 503 inlet intensity 0.1 intensity 0.2\n", "duplicate option 'intensity'"},
                             Bad{"boundary 503 inlet length_scale -1\n", "length_scale must be > 0"},
                             Bad{"boundary 500 wall intensity 0.1\n", "inlet options given for a non-inlet boundary"},
                             Bad{"boundary 503 inlet\nboundary 503 wall\n", "line 2: duplicate boundary entry for flag 503"});
    const Parsed p = parse(bad.text);
    INFO(bad.text);
    CHECK_FALSE(p.ok);
    CHECK_THAT(p.error, ContainsSubstring(bad.message));
    CHECK_THAT(p.error, StartsWith("line "));
}

TEST_CASE("Turbulence config: reading files", "[turbulence_config][io]") {
    TempDir dir("turbconfig");

    SECTION("missing file: laminar, not an error") {
        TurbulenceConfig c;
        c.model = TurbulenceModelType::KOmegaSST; // must be reset
        std::string error = "stale";
        CHECK(read_turbulence_config((dir / "none.turbulence").string(), c, error));
        CHECK(error.empty());
        CHECK_FALSE(c.present);
        CHECK(c.is_laminar());
        CHECK(IO::read_turbulence((dir / "none.turbulence").string(), c));
        CHECK(c.is_laminar());
    }

    SECTION("valid file") {
        write_text(dir / "case.turbulence", "model spalart-allmaras\ninlet_nu_tilde_ratio 5\n");
        TurbulenceConfig c;
        REQUIRE(IO::read_turbulence((dir / "case.turbulence").string(), c));
        CHECK(c.present);
        CHECK(c.model == TurbulenceModelType::SpalartAllmaras);
        CHECK(c.inlet.nu_tilde_ratio == 5.0);
    }

    SECTION("invalid file: false, message names file and line") {
        write_text(dir / "bad.turbulence", "model spalart-allmaras\ninlet_intensity high\n");
        TurbulenceConfig c;
        std::string error;
        CHECK_FALSE(read_turbulence_config((dir / "bad.turbulence").string(), c, error));
        CHECK_THAT(error, ContainsSubstring("bad.turbulence: line 2: invalid number 'high'"));

        CaptureStderr err;
        CHECK_FALSE(IO::read_turbulence((dir / "bad.turbulence").string(), c));
        CHECK_THAT(err.str(), ContainsSubstring("ERROR: *.turbulence FILE HAS INVALID FORMAT"));
    }
}

TEST_CASE("Turbulence model factory", "[turbulence_config][factory]") {
    TurbulenceConfig c;
    CHECK(make_turbulence_model(c) == nullptr); // laminar

    c.model = TurbulenceModelType::SpalartAllmaras;
    const auto model = make_turbulence_model(c);
    REQUIRE(model != nullptr);
    CHECK(model->name() == "spalart-allmaras");
    CHECK(dynamic_cast<SpalartAllmaras*>(model.get()) != nullptr);
    CHECK(is_turbulence_model_available(TurbulenceModelType::SpalartAllmaras));
    CHECK(is_turbulence_model_available(TurbulenceModelType::None));

    SECTION("SA is a low-Re model") {
        c.wall_treatment = WallTreatmentType::WallFunction;
        CHECK_THROWS_WITH(make_turbulence_model(c), ContainsSubstring("supports wall_treatment low-re only"));
    }

    SECTION("models of later phases are reported as not implemented") {
        c.model = GENERATE(TurbulenceModelType::RngKEpsilon, TurbulenceModelType::RealizableKEpsilon);
        CHECK_FALSE(is_turbulence_model_available(c.model));
        CHECK_THROWS_WITH(make_turbulence_model(c), ContainsSubstring("is not implemented yet"));
    }
}

TEST_CASE("Turbulence model factory: two-equation models and wall treatments", "[turbulence_config][factory]") {
    struct Case {
        TurbulenceModelType model;
        std::string name;
    };
    const Case m = GENERATE(Case{TurbulenceModelType::KOmegaSST, "k-omega-sst"},
                            Case{TurbulenceModelType::KEpsilon, "k-epsilon"},
                            Case{TurbulenceModelType::LaunderSharmaKEpsilon, "launder-sharma-k-epsilon"});
    CAPTURE(m.name);
    TurbulenceConfig c;
    c.model = m.model;
    CHECK(is_turbulence_model_available(m.model));
    const auto model = make_turbulence_model(c); // model default wall treatment
    REQUIRE(model != nullptr);
    CHECK(model->name() == m.name);
    CHECK(model->name() == to_string(m.model)); // restart files use the canonical name

    // Explicit, incompatible wall treatments are rejected with a hint.
    c.wall_treatment = m.model == TurbulenceModelType::KEpsilon ? WallTreatmentType::LowRe : WallTreatmentType::WallFunction;
    CHECK_THROWS_WITH(make_turbulence_model(c), ContainsSubstring("supports wall_treatment"));
    if (m.model == TurbulenceModelType::KEpsilon) {
        CHECK_THROWS_WITH(make_turbulence_model(c), ContainsSubstring("use launder-sharma-k-epsilon"));
    }
    c.wall_treatment = WallTreatmentType::Automatic;
    CHECK_THROWS_AS(make_turbulence_model(c), std::invalid_argument);
}
