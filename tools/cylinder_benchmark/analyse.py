#!/usr/bin/env python3
"""Benchmark quantities from a HydraMa2D monitor file (results/<case>.monitors.dat).

Steady (Schafer-Turek 2D-1): the last sample, c_D, c_L and the pressure
difference between the probes 'front' and 'back'.

Periodic (2D-2): over the last `--periods` lift periods
  * St = D f / U_mean, f from the upward zero crossings of c_L (linear interpolation);
  * c_D max / mean, c_L max / min (parabolic interpolation of the sampled extrema);
  * dp at t0 + 1/(2 f), t0 = time of the last c_L maximum (definition of the benchmark).

Pure Python (no numpy).

Usage:
    analyse.py MONITORS_FILE [--unsteady] [--periods N] [--pressure-scale S]
"""

import argparse
import sys


def read_monitors(path):
    """Column name -> list of floats."""
    header = None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = line.split()
                continue
            rows.append([float(x) for x in line.split()])
    if header is None or not rows:
        raise ValueError(f"{path}: no data")
    return {name: [r[i] for r in rows] for i, name in enumerate(header)}


def zero_crossings_up(t, y):
    out = []
    for i in range(1, len(y)):
        if y[i - 1] < 0.0 <= y[i]:
            out.append(t[i - 1] + (t[i] - t[i - 1]) * (-y[i - 1]) / (y[i] - y[i - 1]))
    return out


def refine_extremum(t, y, i):
    """Vertex of the parabola through samples i-1, i, i+1 (value, time)."""
    if i <= 0 or i >= len(y) - 1:
        return y[i], t[i]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return y1, t[i]
    s = 0.5 * (y0 - y2) / denom  # offset in samples, |s| <= 1/2 at an extremum
    h = 0.5 * (t[i + 1] - t[i - 1])
    return y1 - 0.25 * (y0 - y2) * s, t[i] + s * h


def interpolate(t, y, time):
    for i in range(1, len(t)):
        if t[i] >= time:
            a = (time - t[i - 1]) / (t[i] - t[i - 1])
            return y[i - 1] + a * (y[i] - y[i - 1])
    return y[-1]


def steady(data, pressure_scale):
    r = {"cd": data["cd"][-1], "cl": data["cl"][-1], "cd_surface": data["cd_surface"][-1],
         "cl_surface": data["cl_surface"][-1], "steps": int(data["step"][-1])}
    # Convergence: change of c_D over the last 100 steps.
    steps = data["step"]
    k = next((i for i in range(len(steps)) if steps[i] >= steps[-1] - 100), 0)
    r["cd_change"] = abs(data["cd"][-1] - data["cd"][k])
    if "p_front" in data and "p_back" in data:
        r["dp"] = (data["p_front"][-1] - data["p_back"][-1]) * pressure_scale
    return r


def periodic(data, periods, pressure_scale):
    t, cl, cd = data["time"], data["cl"], data["cd"]
    up = zero_crossings_up(t, cl)
    if len(up) < periods + 1:
        raise ValueError(f"only {max(len(up) - 1, 0)} lift periods in the file, {periods} needed")
    t_start, t_end = up[-periods - 1], up[-1]
    f = periods / (t_end - t_start)
    window = [i for i in range(len(t)) if t_start <= t[i] <= t_end]
    i0, i1 = window[0], window[-1]

    def extremum(y, sign):
        best = max(range(i0, i1 + 1), key=lambda i: sign * y[i])
        return refine_extremum(t, y, best)

    cl_max, t_clmax = extremum(cl, 1.0)
    cl_min, _ = extremum(cl, -1.0)
    cd_max, _ = extremum(cd, 1.0)
    cd_min, _ = extremum(cd, -1.0)
    # time averages (trapezoidal rule over the window)
    def mean(y):
        s = sum(0.5 * (y[i] + y[i - 1]) * (t[i] - t[i - 1]) for i in range(i0 + 1, i1 + 1))
        return s / (t[i1] - t[i0])
    # the last c_L maximum in the window
    peaks = [i for i in range(max(i0, 1), i1) if cl[i - 1] < cl[i] >= cl[i + 1] and cl[i] > 0.0]
    r = {"st": f, "cd_max": cd_max, "cd_min": cd_min, "cd_mean": mean(cd), "cl_max": cl_max, "cl_min": cl_min,
         "cl_mean": mean(cl), "periods": periods, "t_end": t[-1], "steps": int(data["step"][-1])}
    if "p_front" in data and "p_back" in data and peaks:
        _, t0 = refine_extremum(t, cl, peaks[-1])
        tp = t0 + 0.5 / f
        if tp <= t[-1]:
            r["dp"] = (interpolate(t, data["p_front"], tp) - interpolate(t, data["p_back"], tp)) * pressure_scale
    # period-to-period variation of the lift amplitude: a measure of how periodic the signal is
    amplitudes = []
    for k in range(len(up) - periods - 1, len(up) - 1):
        idx = [i for i in range(len(t)) if up[k] <= t[i] <= up[k + 1]]
        if idx:
            amplitudes.append(max(cl[i] for i in idx))
    if len(amplitudes) > 1:
        r["cl_max_drift"] = max(amplitudes) - min(amplitudes)
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--unsteady", action="store_true")
    p.add_argument("--periods", type=int, default=4)
    p.add_argument("--pressure-scale", type=float, default=1.0,
                   help="factor applied to the pressure difference (e.g. U_mean^2 for dimensional values)")
    a = p.parse_args()
    data = read_monitors(a.file)
    r = periodic(data, a.periods, a.pressure_scale) if a.unsteady else steady(data, a.pressure_scale)
    for k, v in r.items():
        print(f"{k:14s} {v:.6g}" if isinstance(v, float) else f"{k:14s} {v}")


if __name__ == "__main__":
    sys.exit(main())
