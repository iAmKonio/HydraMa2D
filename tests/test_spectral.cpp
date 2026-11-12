// Fast Fourier Transform (FFT), spectral analysis, limit-cycle detection,
// and vortex shedding diagnostics (Strouhal number, mean drag, lift RMS).

#include "post/Spectral.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <vector>

using namespace hydrama::post;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

constexpr double kPi = 3.14159265358979323846;

} // namespace

TEST_CASE("Spectral analysis: FFT frequency and amplitude of pure harmonics", "[post][spectral]") {
    const double dt = 0.01; // 100 Hz sampling rate
    const int N = 1024;
    // Exactly on bin k = 40 (f = 40 / 10.24 = 3.90625 Hz) to verify exact amplitude recovery
    const double f1 = 40.0 / (N * dt);
    const double amp1 = 1.80;

    std::vector<double> time(N), signal(N);
    for (int i = 0; i < N; ++i) {
        time[i] = i * dt;
        signal[i] = amp1 * std::sin(2.0 * kPi * f1 * time[i]);
    }

    const Spectrum spec = compute_spectrum(time, signal, true, true);
    REQUIRE_FALSE(spec.amplitudes.empty());

    // Locate peak in spectrum
    double max_amp = 0.0;
    double peak_freq = 0.0;
    for (std::size_t k = 1; k < spec.amplitudes.size(); ++k) {
        if (spec.amplitudes[k] > max_amp) {
            max_amp = spec.amplitudes[k];
            peak_freq = spec.frequencies[k];
        }
    }

    // Detected peak must match within the frequency resolution df
    CHECK_THAT(peak_freq, WithinAbs(f1, spec.df));
    CHECK_THAT(max_amp, WithinRel(amp1, 0.05)); // within 5% with Hann window
}

TEST_CASE("Shedding analysis: limit-cycle detection, Strouhal number and RMS statistics", "[post][spectral][shedding]") {
    // Synthetic vortex shedding signal:
    // f_s = 0.25 Hz (period T = 4.0 s)
    // U_ref = 1.0 m/s, D = 1.0 m -> Strouhal number St = 0.25
    // Mean drag CD_bar = 3.20, drag oscillation at 2*f_s = 0.5 Hz with amplitude 0.20
    // Lift CL(t) oscillates at f_s = 0.25 Hz with amplitude 1.0
    // CL_rms for pure sine of amplitude A is A / sqrt(2) ~ 0.7071
    const double fs = 0.25;
    const double dt = 0.05;
    const double t_total = 60.0;
    const int N = static_cast<int>(t_total / dt) + 1;

    std::vector<double> t(N), cl(N), cd(N);
    for (int i = 0; i < N; ++i) {
        t[i] = i * dt;
        // Startup transient factor: (1 - exp(-t / 10))
        const double envelope = (t[i] < 15.0) ? (t[i] / 15.0) : 1.0;
        cl[i] = envelope * 1.0 * std::sin(2.0 * kPi * fs * t[i]);
        cd[i] = 3.20 + envelope * 0.20 * std::cos(4.0 * kPi * fs * t[i]);
    }

    // Analyze shedding after transient t >= 20.0
    const SheddingAnalysis analysis = analyze_shedding(t, cl, cd, 1.0, 1.0, 20.0);

    CHECK(analysis.is_periodic == true);
    CHECK(analysis.limit_cycle_converged == true);
    CHECK(analysis.cycles_detected >= 9);
    CHECK_THAT(analysis.period, WithinRel(4.0, 0.005));
    CHECK_THAT(analysis.dominant_frequency, WithinRel(fs, 0.005));
    CHECK_THAT(analysis.strouhal, WithinRel(0.25, 0.005));

    // Drag and lift statistics
    CHECK_THAT(analysis.mean_drag, WithinRel(3.20, 0.01));
    CHECK_THAT(analysis.mean_lift, WithinAbs(0.0, 0.02));
    CHECK_THAT(analysis.cl_rms, WithinRel(1.0 / std::sqrt(2.0), 0.02));
    CHECK_THAT(analysis.cl_amplitude, WithinRel(1.0, 0.02));
    CHECK_THAT(analysis.cd_amplitude, WithinRel(0.20, 0.02));
    CHECK(analysis.period_variation < 0.01); // highly regular cycles

    const std::string report = format_spectral_report(analysis, "cylinder");
    CHECK_THAT(report, ContainsSubstring("AERODYNAMIC & SPECTRAL ANALYSIS SUMMARY: cylinder"));
    CHECK_THAT(report, ContainsSubstring("CONVERGED"));
    CHECK_THAT(report, ContainsSubstring("Strouhal Number (St):"));
    CHECK_THAT(report, ContainsSubstring("0.25"));
}

TEST_CASE("Unbounded circular cylinder benchmark Strouhal number (Re = 100)", "[post][spectral][strouhal]") {
    // For an isolated circular cylinder in unbounded flow at Re = 100,
    // Williamson (1989), Fey et al. (1998) established Strouhal number:
    // St = 0.165 +/- 0.005
    // With U_inf = 1.0, D = 1.0, f_s = 0.165 Hz (period T = 6.06 s).
    const double expected_st = 0.165;
    const double dt = 0.1;
    const double t_total = 100.0;
    const int N = static_cast<int>(t_total / dt) + 1;

    std::vector<double> t(N), cl(N), cd(N);
    for (int i = 0; i < N; ++i) {
        t[i] = i * dt;
        cl[i] = 0.35 * std::sin(2.0 * kPi * expected_st * t[i]);
        cd[i] = 1.35 + 0.05 * std::cos(4.0 * kPi * expected_st * t[i]);
    }

    const SheddingAnalysis analysis = analyze_shedding(t, cl, cd, 1.0, 1.0, 30.0);

    REQUIRE(analysis.limit_cycle_converged);
    CHECK_THAT(analysis.strouhal, WithinAbs(0.165, 0.005));
    CHECK_THAT(analysis.mean_drag, WithinRel(1.35, 0.01));
}
