#pragma once

// Spectral analysis, Fast Fourier Transform (FFT), limit-cycle detector,
// and aerodynamic shedding diagnostics (Strouhal number, mean drag, lift RMS).
//
// Designed for unsteady flow analysis from monitor streams (such as lift CL(t)
// and drag CD(t) across vortex shedding behind bluff bodies or cylinders).

#include <string>
#include <vector>

namespace hydrama::post {

struct SpectralPeak {
    double frequency = 0.0; // frequency in Hz
    double amplitude = 0.0; // peak amplitude
    double power = 0.0;     // power spectral density
};

struct Spectrum {
    std::vector<double> frequencies; // frequency bins (Hz)
    std::vector<double> amplitudes;  // amplitude spectrum |Y(f)|
    std::vector<double> power;       // power spectrum
    double df = 0.0;                 // frequency resolution
    double sample_rate = 0.0;        // sampling rate in Hz (1 / dt)
};

struct SheddingAnalysis {
    bool is_periodic = false;            // periodic oscillation detected
    bool limit_cycle_converged = false;  // steady periodic limit cycle reached
    double dominant_frequency = 0.0;     // primary shedding frequency f_s (Hz)
    double period = 0.0;                 // shedding period T_s = 1 / f_s (s)
    double strouhal = 0.0;               // Strouhal number St = f_s * L_ref / U_ref
    double mean_drag = 0.0;              // time-averaged CD
    double mean_lift = 0.0;              // time-averaged CL
    double cl_rms = 0.0;                 // root-mean-square lift fluctuation
    double cl_amplitude = 0.0;           // (CL_max - CL_min) / 2
    double cd_amplitude = 0.0;           // (CD_max - CD_min) / 2
    double cl_max = 0.0;
    double cl_min = 0.0;
    double cd_max = 0.0;
    double cd_min = 0.0;
    double t_start = 0.0;                // start time of analyzed window
    double t_end = 0.0;                  // end time of analyzed window
    int cycles_detected = 0;             // number of complete cycles analyzed
    double period_variation = 0.0;       // relative variation of cycle periods
    std::vector<SpectralPeak> peaks;     // dominant peaks sorted by amplitude
};

// Computes the one-sided amplitude and power spectrum using Cooley-Tukey Radix-2 FFT.
// If apply_hann_window is true, applies a Hann taper to reduce spectral leakage.
Spectrum compute_spectrum(
    const std::vector<double>& time,
    const std::vector<double>& signal,
    bool remove_mean = true,
    bool apply_hann_window = true);

// Analyzes shedding from time history of lift CL and drag CD.
// If t_transient < 0, automatically discards initial transients before limit-cycle.
SheddingAnalysis analyze_shedding(
    const std::vector<double>& time,
    const std::vector<double>& cl,
    const std::vector<double>& cd,
    double reference_length = 1.0,
    double reference_velocity = 1.0,
    double t_transient = -1.0);

// Formats a clean ASCII summary table of the spectral and shedding analysis.
std::string format_spectral_report(
    const SheddingAnalysis& analysis,
    const std::string& body_name = "body");

} // namespace hydrama::post
