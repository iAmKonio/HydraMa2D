// Langtry-Menter (2009) gamma-Re_theta transition model coupled with SST.
//
// Closure:
//   - Transition onset correlation Re_theta_t(Tu)
//   - Critical Reynolds number correlation Re_theta_c(Re_thetat_tilde)
//   - Transition length correlation F_length(Re_thetat_tilde)
//   - Transition functions: F_onset1..3, F_onset, F_turb, F_reattach, gamma_sep, F3
//   - Point-implicit source term linearisations and laminar suppression
//
// Model:
//   - Model initialization, names, fields, wall / inlet boundary conditions
//   - Multi-step advance on channel mesh: stability, boundedness, non-negativity

#include "BoundaryConditions.hpp"
#include "TurbulenceTestUtils.hpp"
#include "WallTestUtils.hpp"
#include "turbulence/GammaReThetaSST.hpp"
#include "turbulence/TurbulenceConfig.hpp"
#include "turbulence/TurbulenceFactory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace hydrama;
using namespace hydrama::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;


// =============================================================================
// Transition Correlations
// =============================================================================

TEST_CASE("Transition correlations: Re_theta_t(Tu)", "[transition][correlations]") {
    SECTION("reference values for low turbulence intensity (Tu <= 1.3 %)") {
        // Tu = 0.5%: 1173.51 - 22.63 * 0.5 + 0.1471 / (0.5^2) = 1162.7834
        CHECK_THAT(transition::re_theta_t(0.5), WithinRel(1162.7834, 1e-6));
        // Tu = 1.0%: 1173.51 - 22.63 * 1.0 + 0.1471 / (1.0^2) = 1151.0271
        CHECK_THAT(transition::re_theta_t(1.0), WithinRel(1151.0271, 1e-6));
    }

    SECTION("reference values for high turbulence intensity (Tu > 1.3 %)") {
        // Tu = 2.0%: 331.5 * (2.0 - 0.5658)^(-0.671) = 260.2546
        CHECK_THAT(transition::re_theta_t(2.0), WithinRel(260.2546, 1e-4));
        // Tu = 3.0%: 331.5 * (3.0 - 0.5658)^(-0.671) = 182.4898
        CHECK_THAT(transition::re_theta_t(3.0), WithinRel(182.4898, 1e-4));
    }

    SECTION("limits and clamping") {
        // Tu -> 0 is clamped to Tu = 0.013 % and remains positive finite
        const double re_tt_zero = transition::re_theta_t(0.0);
        CHECK(re_tt_zero > 1000.0);
        CHECK(std::isfinite(re_tt_zero));
        CHECK(transition::re_theta_t(-1.0) == re_tt_zero);

        // Very high Tu is clamped to lower bound of 20.0
        CHECK(transition::re_theta_t(100.0) >= 20.0);
    }

    SECTION("monotonic decrease with increasing turbulence intensity") {
        double prev = transition::re_theta_t(0.1);
        for (double tu = 0.2; tu <= 10.0; tu += 0.2) {
            const double val = transition::re_theta_t(tu);
            CHECK(val < prev);
            prev = val;
        }
    }
}

TEST_CASE("Transition correlations: Re_theta_c(Re_thetat_tilde)", "[transition][correlations]") {
    SECTION("reference values and continuity") {
        // rt = 100: polynomial branch
        CHECK_THAT(transition::re_theta_c(100.0), WithinRel(89.243, 1e-4));

        // rt = 1870: transition point between polynomial and linear branches
        CHECK_THAT(transition::re_theta_c(1870.0), WithinRel(1276.89, 1e-3));

        // rt = 2000: linear branch
        CHECK_THAT(transition::re_theta_c(2000.0), WithinRel(1344.23, 1e-4));

        // Near rt = 1870, the empirical polynomial and linear branches match within ~1.2 (< 0.1 %)
        const double c_minus = transition::re_theta_c(1869.99);
        const double c_plus = transition::re_theta_c(1870.01);
        CHECK_THAT(c_minus, WithinAbs(c_plus, 1.5));
    }

    SECTION("bounds and monotonicity") {
        // Re_theta_c must be positive and <= Re_thetat_tilde
        CHECK(transition::re_theta_c(0.0) >= 1.0);
        CHECK(transition::re_theta_c(50.0) <= 50.0);
        CHECK(transition::re_theta_c(500.0) <= 500.0);
        CHECK(transition::re_theta_c(2500.0) <= 2500.0);

        // Monotonic increase
        double prev = transition::re_theta_c(10.0);
        for (double rt = 20.0; rt <= 3000.0; rt += 50.0) {
            const double val = transition::re_theta_c(rt);
            CHECK(val > prev);
            prev = val;
        }
    }
}

TEST_CASE("Transition correlations: F_length(Re_thetat_tilde)", "[transition][correlations]") {
    SECTION("reference values across piecewise segments") {
        // Segment 1: rt < 400
        CHECK_THAT(transition::f_length(300.0), WithinRel(24.30977, 1e-4));

        // Segment 2: 400 <= rt < 596.5
        CHECK_THAT(transition::f_length(500.0), WithinRel(2.96025, 1e-4));

        // Segment 3: 596.5 <= rt < 1200
        CHECK_THAT(transition::f_length(800.0), WithinRel(0.43895, 1e-4));

        // Segment 4: rt >= 1200
        CHECK_THAT(transition::f_length(1500.0), WithinRel(0.3188, 1e-6));
        CHECK_THAT(transition::f_length(3000.0), WithinRel(0.3188, 1e-6));
    }

    SECTION("lower bound F_length >= 0.1") {
        CHECK(transition::f_length(-10.0) >= 0.1);
        CHECK(transition::f_length(0.0) >= 0.1);
        CHECK(transition::f_length(1199.0) >= 0.1);
    }
}

// =============================================================================
// Transition and Blending Functions
// =============================================================================

TEST_CASE("Transition functions: onset and damping", "[transition][functions]") {
    SECTION("f_onset1, f_onset2, f_onset3, f_onset") {
        const double re_tc = 100.0;

        // Subcritical: Re_V << Re_theta_c -> F_onset = 0
        const double rev_sub = 50.0;
        CHECK(transition::f_onset1(rev_sub, re_tc) < 1.0);
        CHECK(transition::f_onset(rev_sub, re_tc, 0.0) == 0.0);

        // Supercritical: Re_V > 2.193 * Re_theta_c -> F_onset > 0
        const double rev_super = 500.0;
        const double fo1 = transition::f_onset1(rev_super, re_tc);
        CHECK(fo1 > 1.0);
        CHECK(transition::f_onset2(fo1) > 1.0);
        CHECK(transition::f_onset(rev_super, re_tc, 0.0) > 0.0);

        // High turbulent viscosity ratio R_t suppresses onset via F_onset3
        CHECK(transition::f_onset3(0.0) == 1.0);
        CHECK(transition::f_onset3(2.5) == 0.0);
        CHECK(transition::f_onset3(5.0) == 0.0);
    }

    SECTION("f_turb and f_reattach damping") {
        // At Rt = 0, both functions are 1.0
        CHECK(transition::f_turb(0.0) == 1.0);
        CHECK(transition::f_reattach(0.0) == 1.0);

        // As Rt increases, they decay exponentially
        CHECK(transition::f_turb(4.0) < 0.5);
        CHECK(transition::f_turb(10.0) < 1e-4);
        CHECK(transition::f_reattach(20.0) < 0.5);
    }

    SECTION("gamma_sep: separation-induced transition") {
        const double re_tc = 100.0;

        // Attached flow or subcritical Re_V: gamma_sep = 0
        const double rev_sub = 300.0; // < 3.235 * 100 = 323.5
        CHECK(transition::gamma_sep(rev_sub, re_tc, 0.0) == 0.0);

        // Separated shear layer: Re_V > 3.235 * Re_theta_c -> gamma_sep > 0
        const double rev_sep = 600.0;
        const double g_sep = transition::gamma_sep(rev_sep, re_tc, 0.0);
        CHECK(g_sep > 0.0);
        CHECK(g_sep <= 2.0);

        // Large Rt suppresses gamma_sep via f_reattach
        CHECK(transition::gamma_sep(rev_sep, re_tc, 100.0) < 1e-3);
    }

    SECTION("F3 blending modification") {
        // Near wall with small R_y: F3 ~ 1
        CHECK_THAT(transition::f3(1e-4, 1e-4, 1e-3), WithinRel(1.0, 1e-6));

        // Far away with large R_y: F3 ~ 0
        CHECK(transition::f3(1.0, 10.0, 1e-5) < 1e-10);

        // Zero inputs stay safe
        CHECK(transition::f3(0.0, 1.0, 1e-3) == 0.0);
        CHECK(transition::f3(1.0, 0.0, 1e-3) == 0.0);
    }
}

// =============================================================================
// Source Terms and Linearisation
// =============================================================================

TEST_CASE("Transition source terms: laminar suppression and turbulent limit", "[transition][sources]") {
    transition::PointState s;
    s.k = 1e-4;
    s.omega = 50.0;
    s.gamma = 0.0; // fully laminar
    s.re_thetat = 500.0;
    s.nu = 1e-5;
    s.nut = 1e-5;
    s.strain = 10.0;
    s.vorticity = 10.0;
    s.d = 0.01;
    s.u_mag = 1.0;
    s.f1 = 1.0;

    SECTION("laminar flow suppresses turbulent kinetic energy production") {
        const auto t_lam = transition::source(s, 10.0);
        // Intermittency is 0 and rev is small -> gamma_eff = 0
        CHECK(t_lam.gamma_eff == 0.0);
        CHECK(t_lam.p_k_eff == 0.0);
        // Destruction is limited to 0.1 * D_k
        const double d_k_nominal = transition::Constants::beta_star * s.k * s.omega;
        CHECK_THAT(t_lam.d_k_eff, WithinRel(0.1 * d_k_nominal, 1e-12));
    }

    SECTION("turbulent flow restores full production and destruction") {
        s.gamma = 1.0; // fully turbulent
        const auto t_turb = transition::source(s, 10.0);
        CHECK(t_turb.gamma_eff == 1.0);
        CHECK(t_turb.p_k_eff > 0.0);
        const double d_k_nominal = transition::Constants::beta_star * s.k * s.omega;
        CHECK_THAT(t_turb.d_k_eff, WithinRel(d_k_nominal, 1e-12));
    }

    SECTION("point-implicit linearisation stability and diagonal positivity") {
        s.gamma = 0.5;
        const auto t = transition::source(s, 10.0);

        CHECK(t.k.D >= 0.0);
        CHECK(t.log_omega.D >= 0.0);
        CHECK(t.gamma.D >= 0.0);
        CHECK(t.re_thetat.D >= 0.0);

        // Value consistency
        CHECK_THAT(t.k.value(s.k), WithinAbs(t.p_k_eff - t.d_k_eff, 1e-12 * (1.0 + t.p_k_eff)));
        CHECK_THAT(t.gamma.value(s.gamma), WithinAbs(t.p_gamma - t.e_gamma, 1e-12 * (1.0 + t.p_gamma)));
        CHECK_THAT(t.re_thetat.value(s.re_thetat), WithinAbs(t.p_thetat, 1e-12 * (1.0 + std::abs(t.p_thetat))));
    }
}

// =============================================================================
// Model Initialization and Boundary Conditions
// =============================================================================

TEST_CASE("Gamma-Re_theta SST model: fields, names and boundary conditions", "[transition][model]") {
    const double nu = 1e-3, lx = 1.5, ly = 0.5;
    const int nx = 6, ny = 5;
    TurbSetup s(turb_channel(nx, ny, lx, ly), nu, 1e-3);
    const std::vector<double> u(s.mesh.npoin, 1.0), v(s.mesh.npoin, 0.0);

    TurbulenceConfig c = model_config(TurbulenceModelType::GammaReThetaSST);
    c.inlet.intensity = 0.03;
    c.inlet.length_scale = 0.05;

    auto model = make_turbulence_model(c);
    REQUIRE(model != nullptr);
    CHECK(model->name() == "gamma-re-theta-sst");

    model->initialize(s.ctx(), {u, v});

    const auto fields = model->fields();
    REQUIRE(fields.size() == 7);
    CHECK(fields[0].name == "k");
    CHECK(fields[1].name == "omega");
    CHECK(fields[2].name == "nut");
    CHECK(fields[3].name == "gamma");
    CHECK(fields[4].name == "re_thetat");
    CHECK(fields[5].name == "gamma_eff");
    CHECK(fields[6].name == "sst_f1");

    const auto state_fields = model->state_fields();
    REQUIRE(state_fields.size() == 4);
    CHECK(state_fields[0].name == "k");
    CHECK(state_fields[1].name == "log_omega");
    CHECK(state_fields[2].name == "gamma");
    CHECK(state_fields[3].name == "re_thetat");

    auto* trans = dynamic_cast<GammaReThetaSST*>(model.get());
    REQUIRE(trans != nullptr);

    SECTION("wall and inlet values") {
        const double tu_pct = 100.0 * 0.03;
        const double re_tt_expected = transition::re_theta_t(tu_pct);

        int wall_count = 0, inlet_count = 0;
        for (int ip = 0; ip < s.mesh.npoin; ++ip) {
            if (s.walls.is_wall_node(ip)) {
                ++wall_count;
                CHECK(trans->k()[ip] == 0.0);
                CHECK(trans->omega()[ip] > 0.0);
                CHECK(trans->f1()[ip] == 1.0);
            } else if (s.mesh.x[ip] == 0.0) {
                ++inlet_count;
                CHECK(trans->gamma()[ip] == 1.0);
                CHECK_THAT(trans->re_thetat()[ip], WithinRel(re_tt_expected, 1e-4));
            }
        }
        CHECK(wall_count == 2 * (nx + 1));
        CHECK(inlet_count == ny - 1);
    }
}

// =============================================================================
// Time Step Advance on Channel Mesh
// =============================================================================

TEST_CASE("Gamma-Re_theta SST model: multi-step advance stays stable and non-negative", "[transition][advance]") {
    const double nu = 1e-3, lx = 2.0, ly = 0.5;
    const int nx = 8, ny = 6;
    TurbSetup s(turb_channel(nx, ny, lx, ly), nu, 1e-3);
    const std::vector<double> u(s.mesh.npoin, 1.0), v(s.mesh.npoin, 0.0);

    TurbulenceConfig c = model_config(TurbulenceModelType::GammaReThetaSST);
    c.inlet.intensity = 0.02;
    c.inlet.length_scale = 0.05;

    GammaReThetaSST model(c);
    model.initialize(s.ctx(), {u, v});

    for (int step = 0; step < 10; ++step) {
        model.advance(s.ctx(), {u, v});
    }

    // Check all fields are finite and bounded
    for (int ip = 0; ip < s.mesh.npoin; ++ip) {
        CHECK(std::isfinite(model.k()[ip]));
        CHECK(model.k()[ip] >= 0.0);

        CHECK(std::isfinite(model.omega()[ip]));
        CHECK(model.omega()[ip] > 0.0);

        CHECK(std::isfinite(model.gamma()[ip]));
        CHECK(model.gamma()[ip] >= 0.0);
        CHECK(model.gamma()[ip] <= 2.0);

        CHECK(std::isfinite(model.re_thetat()[ip]));
        CHECK(model.re_thetat()[ip] >= 1.0);

        CHECK(std::isfinite(model.nut()[ip]));
        CHECK(model.nut()[ip] >= 0.0);
    }

    const auto res = model.residuals();
    REQUIRE(res.size() == 4);
    for (const auto& r : res) {
        CHECK(std::isfinite(r.relative_change));
    }
}
