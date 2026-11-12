#include "post/Spectral.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

namespace hydrama::post {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Next power of 2 >= n
std::size_t next_power_of_two(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Cooley-Tukey Radix-2 in-place FFT
void fft_radix2(std::vector<std::complex<double>>& a, bool invert) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = 2.0 * kPi / static_cast<double>(len) * (invert ? -1.0 : 1.0);
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                const std::complex<double> u = a[i + j];
                const std::complex<double> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (std::complex<double>& x : a) {
            x /= static_cast<double>(n);
        }
    }
}

// Resamples an unevenly sampled time series onto a uniform grid
void resample_uniform(const std::vector<double>& t, const std::vector<double>& y,
                      std::size_t num_samples, std::vector<double>& t_out, std::vector<double>& y_out) {
    if (t.size() < 2 || num_samples < 2) return;
    const double t_start = t.front();
    const double t_end = t.back();
    const double dt = (t_end - t_start) / static_cast<double>(num_samples - 1);

    t_out.resize(num_samples);
    y_out.resize(num_samples);

    std::size_t idx = 0;
    for (std::size_t i = 0; i < num_samples; ++i) {
        const double cur_t = t_start + static_cast<double>(i) * dt;
        t_out[i] = cur_t;

        while (idx + 1 < t.size() && t[idx + 1] < cur_t) {
            ++idx;
        }
        if (idx + 1 >= t.size()) {
            y_out[i] = y.back();
        } else {
            const double alpha = (cur_t - t[idx]) / (t[idx + 1] - t[idx]);
            y_out[i] = y[idx] + alpha * (y[idx + 1] - y[idx]);
        }
    }
}

} // namespace

Spectrum compute_spectrum(
    const std::vector<double>& time,
    const std::vector<double>& signal,
    bool remove_mean,
    bool apply_hann_window) {
    if (time.size() < 4 || time.size() != signal.size()) return {};

    const std::size_t N = time.size();
    const double t_span = time.back() - time.front();
    if (t_span <= 0.0) return {};

    // Check if time step is reasonably uniform
    double dt_mean = t_span / static_cast<double>(N - 1);
    bool is_uniform = true;
    for (std::size_t i = 1; i < N; ++i) {
        if (std::abs((time[i] - time[i - 1]) - dt_mean) > 1e-4 * dt_mean) {
            is_uniform = false;
            break;
        }
    }

    std::vector<double> u_time = time;
    std::vector<double> u_signal = signal;
    if (!is_uniform) {
        resample_uniform(time, signal, N, u_time, u_signal);
        dt_mean = (u_time.back() - u_time.front()) / static_cast<double>(N - 1);
    }

    double mean_val = 0.0;
    if (remove_mean) {
        for (double val : u_signal) mean_val += val;
        mean_val /= static_cast<double>(N);
    }

    const std::size_t n_fft = next_power_of_two(N);
    std::vector<std::complex<double>> x(n_fft, std::complex<double>(0.0, 0.0));

    double window_sum = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double w = 1.0;
        if (apply_hann_window) {
            w = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(N - 1)));
        }
        window_sum += w;
        x[i] = (u_signal[i] - mean_val) * w;
    }
    if (window_sum <= 0.0) window_sum = static_cast<double>(N);

    fft_radix2(x, false);

    const std::size_t num_bins = n_fft / 2 + 1;
    Spectrum spec;
    spec.df = 1.0 / (static_cast<double>(n_fft) * dt_mean);
    spec.sample_rate = 1.0 / dt_mean;
    spec.frequencies.resize(num_bins);
    spec.amplitudes.resize(num_bins);
    spec.power.resize(num_bins);

    // Normalization factor for windowed amplitude spectrum
    const double scale = 2.0 / window_sum;

    for (std::size_t k = 0; k < num_bins; ++k) {
        spec.frequencies[k] = static_cast<double>(k) * spec.df;
        const double mag = std::abs(x[k]) * scale;
        spec.amplitudes[k] = mag;
        spec.power[k] = mag * mag;
    }
    // DC component single-sided scale
    spec.amplitudes[0] *= 0.5;
    spec.power[0] *= 0.25;

    return spec;
}

SheddingAnalysis analyze_shedding(
    const std::vector<double>& time,
    const std::vector<double>& cl,
    const std::vector<double>& cd,
    double reference_length,
    double reference_velocity,
    double t_transient) {
    SheddingAnalysis out;
    if (time.size() < 10 || time.size() != cl.size() || time.size() != cd.size()) {
        return out;
    }

    const double t_total = time.back() - time.front();
    if (t_total <= 0.0) return out;

    // Automatic transient cutoff: default to second half of signal if not specified
    double cutoff = t_transient;
    if (cutoff < 0.0) {
        cutoff = time.front() + 0.4 * t_total;
    }

    // Filter points in the analyzed window [cutoff, t_end]
    std::vector<double> win_t, win_cl, win_cd;
    for (std::size_t i = 0; i < time.size(); ++i) {
        if (time[i] >= cutoff) {
            win_t.push_back(time[i]);
            win_cl.push_back(cl[i]);
            win_cd.push_back(cd[i]);
        }
    }
    if (win_t.size() < 8) return out;

    out.t_start = win_t.front();
    out.t_end = win_t.back();

    // 1. Detect zero-crossings on lift CL(t) to identify periodic cycles
    std::vector<double> zero_crossings;
    for (std::size_t i = 1; i < win_t.size(); ++i) {
        if (win_cl[i - 1] < 0.0 && win_cl[i] >= 0.0) {
            const double alpha = -win_cl[i - 1] / (win_cl[i] - win_cl[i - 1]);
            const double t_cross = win_t[i - 1] + alpha * (win_t[i] - win_t[i - 1]);
            zero_crossings.push_back(t_cross);
        }
    }

    // 2. Perform FFT on the lift signal
    const Spectrum spec = compute_spectrum(win_t, win_cl, true, true);
    double fft_peak_freq = 0.0;
    double max_amp = 0.0;
    std::size_t peak_k = 0;

    for (std::size_t k = 1; k < spec.amplitudes.size(); ++k) {
        if (spec.amplitudes[k] > max_amp) {
            max_amp = spec.amplitudes[k];
            peak_k = k;
            fft_peak_freq = spec.frequencies[k];
        }
    }

    // Parabolic interpolation around peak
    if (peak_k > 0 && peak_k + 1 < spec.amplitudes.size()) {
        const double a = spec.amplitudes[peak_k - 1];
        const double b = spec.amplitudes[peak_k];
        const double c = spec.amplitudes[peak_k + 1];
        const double denom = a - 2.0 * b + c;
        if (std::abs(denom) > 1e-14) {
            const double delta = 0.5 * (a - c) / denom;
            fft_peak_freq = (static_cast<double>(peak_k) + delta) * spec.df;
        }
    }

    // 3. Cycle and period analysis
    if (zero_crossings.size() >= 3) {
        std::vector<double> periods;
        for (std::size_t i = 1; i < zero_crossings.size(); ++i) {
            periods.push_back(zero_crossings[i] - zero_crossings[i - 1]);
        }

        const double mean_period = std::accumulate(periods.begin(), periods.end(), 0.0) /
                                   static_cast<double>(periods.size());
        double var = 0.0;
        for (double p : periods) {
            var += (p - mean_period) * (p - mean_period);
        }
        var = std::sqrt(var / static_cast<double>(periods.size()));
        const double rel_var = (mean_period > 0.0) ? (var / mean_period) : 1.0;

        out.is_periodic = true;
        out.cycles_detected = static_cast<int>(periods.size());
        out.period_variation = rel_var;
        out.limit_cycle_converged = (rel_var < 0.02); // less than 2% variation across cycles

        // If zero-crossing period is reliable, use it as primary shedding frequency
        out.period = mean_period;
        out.dominant_frequency = (mean_period > 0.0) ? (1.0 / mean_period) : fft_peak_freq;
    } else {
        out.dominant_frequency = fft_peak_freq;
        out.period = (fft_peak_freq > 0.0) ? (1.0 / fft_peak_freq) : 0.0;
        out.is_periodic = (max_amp > 1e-4);
        out.limit_cycle_converged = false;
    }

    // Strouhal number
    if (reference_velocity > 0.0) {
        out.strouhal = out.dominant_frequency * reference_length / reference_velocity;
    }

    // 4. Mean and RMS statistics over the analyzed window
    out.cl_max = *std::max_element(win_cl.begin(), win_cl.end());
    out.cl_min = *std::min_element(win_cl.begin(), win_cl.end());
    out.cd_max = *std::max_element(win_cd.begin(), win_cd.end());
    out.cd_min = *std::min_element(win_cd.begin(), win_cd.end());

    out.cl_amplitude = 0.5 * (out.cl_max - out.cl_min);
    out.cd_amplitude = 0.5 * (out.cd_max - out.cd_min);

    // Trapezoidal time integration for means
    double dt_sum = 0.0;
    double cl_int = 0.0;
    double cd_int = 0.0;
    for (std::size_t i = 1; i < win_t.size(); ++i) {
        const double dt = win_t[i] - win_t[i - 1];
        dt_sum += dt;
        cl_int += 0.5 * (win_cl[i] + win_cl[i - 1]) * dt;
        cd_int += 0.5 * (win_cd[i] + win_cd[i - 1]) * dt;
    }
    if (dt_sum > 0.0) {
        out.mean_lift = cl_int / dt_sum;
        out.mean_drag = cd_int / dt_sum;

        double cl_sq_int = 0.0;
        for (std::size_t i = 1; i < win_t.size(); ++i) {
            const double dt = win_t[i] - win_t[i - 1];
            const double dev1 = win_cl[i - 1] - out.mean_lift;
            const double dev2 = win_cl[i] - out.mean_lift;
            cl_sq_int += 0.5 * (dev1 * dev1 + dev2 * dev2) * dt;
        }
        out.cl_rms = std::sqrt(cl_sq_int / dt_sum);
    }

    // Collect dominant peaks
    if (!spec.amplitudes.empty()) {
        std::vector<std::size_t> peak_indices;
        for (std::size_t k = 1; k + 1 < spec.amplitudes.size(); ++k) {
            if (spec.amplitudes[k] > spec.amplitudes[k - 1] && spec.amplitudes[k] > spec.amplitudes[k + 1]) {
                peak_indices.push_back(k);
            }
        }
        std::sort(peak_indices.begin(), peak_indices.end(), [&](std::size_t a, std::size_t b) {
            return spec.amplitudes[a] > spec.amplitudes[b];
        });
        const std::size_t top_count = std::min(peak_indices.size(), std::size_t{5});
        for (std::size_t i = 0; i < top_count; ++i) {
            const std::size_t k = peak_indices[i];
            out.peaks.push_back({spec.frequencies[k], spec.amplitudes[k], spec.power[k]});
        }
    }

    return out;
}

std::string format_spectral_report(
    const SheddingAnalysis& a,
    const std::string& body_name) {
    std::ostringstream oss;
    oss << "========================================================================\n"
        << " AERODYNAMIC & SPECTRAL ANALYSIS SUMMARY: " << body_name << "\n"
        << "========================================================================\n"
        << std::fixed << std::setprecision(5)
        << " Analyzed Time Window:          [" << a.t_start << ", " << a.t_end << "] s\n"
        << " Cycles Detected:               " << a.cycles_detected << "\n"
        << " Limit Cycle Status:            " << (a.limit_cycle_converged ? "CONVERGED" : (a.is_periodic ? "PERIODIC" : "UNSTEADY / TRANSIENT")) << "\n"
        << " Dominant Shedding Frequency:   " << a.dominant_frequency << " Hz (Period T = " << a.period << " s)\n"
        << " Strouhal Number (St):          " << a.strouhal << "\n"
        << " Mean Drag Coefficient (CD_bar):" << a.mean_drag << " (Peak-to-Peak: " << (2.0 * a.cd_amplitude) << ")\n"
        << " Mean Lift Coefficient (CL_bar):" << a.mean_lift << " (Peak-to-Peak: " << (2.0 * a.cl_amplitude) << ")\n"
        << " Lift RMS Fluctuations (CL_rms):" << a.cl_rms << "\n";
    if (a.is_periodic && a.cycles_detected > 1) {
        oss << " Cycle Period Variation:        " << std::setprecision(2) << (a.period_variation * 100.0) << " %\n";
    }
    oss << "========================================================================\n";
    return oss.str();
}

} // namespace hydrama::post
