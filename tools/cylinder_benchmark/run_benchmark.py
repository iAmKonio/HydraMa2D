#!/usr/bin/env python3
"""Schafer-Turek cylinder benchmark: triangle vs. inflation-layer meshes.

Runs the steady case 2D-1 (Re = 20) and the periodic case 2D-2 (Re = 100) on
meshes written by generate_meshes.py, and compares drag, lift, Strouhal number
and pressure difference with the reference values of Schafer & Turek (1996).
See docs/CYLINDER_BENCHMARK.md.

Non-dimensional units: D = 1, mean inflow velocity 1 (flag 508, parabolic),
so Re = 1 / nu, C = 2 F, t = t_dim U_mean / D and dp = dp_dim / U_mean^2.

Usage:
    run_benchmark.py --bin build/HydraMa2D --meshes MESH_DIR --work WORK_DIR
                     [--cases steady unsteady] [--only tri_L1 bl_L1 ...] [--jobs 4]
                     [--end-time 60]

Each run is a case directory WORK_DIR/<case>_<mesh>/ (kept: logs, monitors,
VTK). The summary is printed and written to WORK_DIR/summary.md.
"""

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True  # no __pycache__ next to the tools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse import periodic, read_monitors, steady  # noqa: E402

# Schafer & Turek (1996): lower / upper bounds of the benchmark results.
# dp is converted to non-dimensional units (dp_dim / U_mean^2: U_mean = 0.2 m/s
# in 2D-1, 1 m/s in 2D-2).
# High-accuracy values of 2D-1 (John & Matthies 2001; also in John 2004):
# c_D = 5.57953523384, c_L = 0.010618948146, dp = 0.11752016697 (dimensional).
PRECISE_STEADY = {"cd": 5.57953523384, "cd_surface": 5.57953523384, "cl": 0.010618948146, "dp": 0.11752016697 / 0.04}

REFERENCE = {
    "steady": {"cd": (5.57, 5.59), "cl": (0.0104, 0.0110), "dp": (0.1172 / 0.04, 0.1176 / 0.04)},
    "unsteady": {"cd_max": (3.22, 3.24), "cl_max": (0.99, 1.01), "st": (0.295, 0.305), "dp": (2.46, 2.50)},
}

CONTROL = """---
Restart from prior run (0 - NO, 1 - YES)
0
---
Number of Timesteps (Max in case of steady state)
{ntime}
---
Transient (0 - OFF, 1 - ON)
{transient}
---
Simulated Time Interval for Output (transient only)
{output_interval}
---
End Simulation Time (transient only)
{end_time}
---
Output Controls - Every N iterations
100000000
---
Output Controls - Every N CPU seconds
360000
---
Global Minimum Calculated Timestep = -1 OR Local Time Steping = 0 OR Specified Time Step Value = 1
{dtfixed}
---
Specified Time Step Value (The Actual Value if Activated)
1.0d-05
---
Convergence Check Interval
{iwrite}
---
Safety Factor for Calculated Local Time Step
0.6d00
---
Theta1 (0.5<=theta1<=1.0)
1.0d00
---
Steady State Velocity Convergence (0 - OFF, 1 - ON)
{vel_check}
---
L2-Norm
{tolerance}
---
Run Time Adjustable Option
0
"""

PHYSICAL = """---
Reynolds Number
{re}
---
pnode - Node in domain for Pressure BC if no other Pressure BCs exist
1
"""

SOLVER = """---
Pressure Field Solver (1 - Conjugate Gradient)
1
---
Conjugate Gradient Convergence Tolerances
Relative Residual
1.0d-08
---
Absolute Residual
1.0d-13
---
method          = cg
preconditioner  = ic0
operator_format = csr
reordering      = rcm
"""

MONITORS = """# Schafer-Turek benchmark: forces on the cylinder, pressure at its front and back
body                cylinder_wall
reference_velocity  1.0
reference_length    1.0
interval            {interval}
probe  front  1.5  2.0
probe  back   2.5  2.0
"""

CASES = {
    # Steady: local time steps until the relative velocity change per step is
    # below 1e-9 (the round-off floor of the iterative pressure solves on the
    # finest meshes is ~1e-10), at most 10000 steps; the analysis reports the
    # change of c_D over the last 100 steps (c_D settles to 1e-9 within about
    # 3000 steps on the finest meshes).
    "steady": dict(re=20.0, transient=0, dtfixed=0, ntime=10000, vel_check=1, iwrite=100, interval=10, tolerance="1.0d-09"),
    "unsteady": dict(re=100.0, transient=1, dtfixed=-1, ntime=100000000, vel_check=0, iwrite=5000, interval=5,
                     tolerance="1.0d-10"),
}


def write_case(directory, mesh_file, case, end_time):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(os.path.join(directory, "constant"))
    os.makedirs(os.path.join(directory, "system"))
    shutil.copy(mesh_file, os.path.join(directory, "constant", "cylinder.msh"))
    c = dict(CASES[case])
    c["end_time"] = end_time
    c["output_interval"] = 2.0 * end_time  # one VTK file at the start, none during the run
    system = os.path.join(directory, "system")
    with open(os.path.join(system, "cylinder.control"), "w") as f:
        f.write(CONTROL.format(**c))
    with open(os.path.join(system, "cylinder.physical"), "w") as f:
        f.write(PHYSICAL.format(**c))
    with open(os.path.join(system, "cylinder.solver"), "w") as f:
        f.write(SOLVER)
    with open(os.path.join(system, "cylinder.monitors"), "w") as f:
        f.write(MONITORS.format(**c))


def mesh_info(log):
    info = {}
    m = re.search(r"Mesh import: (\d+) triangles, (?:(\d+) quadrilaterals, )?(\d+) nodes", log)
    if m:
        info["triangles"] = int(m.group(1))
        info["quads"] = int(m.group(2) or 0)
        info["nodes"] = int(m.group(3))
    m = re.search(r"Time Taken \(s\): ([0-9.eE+-]+)", log)
    if m:
        info["seconds"] = float(m.group(1))
    return info


def run_one(binary, directory):
    start = time.time()
    with open(os.path.join(directory, "run.log"), "w") as log:
        proc = subprocess.run([binary, directory], stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode, time.time() - start


def fmt(value, ref=None):
    if value is None:
        return "-"
    s = f"{value:.4f}" if abs(value) >= 0.1 else f"{value:.5f}"
    if ref:
        lo, hi = ref
        if lo <= value <= hi:
            s += " ✓"
        else:
            mid = 0.5 * (lo + hi)
            s += f" ({100.0 * (value - mid) / abs(mid):+.1f}%)"
    return s


def fmt_error(value, reference):
    if value is None:
        return "-"
    s = f"{value:.4f}" if abs(value) >= 0.1 else f"{value:.5f}"
    return s + f" ({100.0 * (value - reference) / abs(reference):+.2f}%)"


def summarise(work, results):
    lines = []
    for case in ("steady", "unsteady"):
        rows = [r for r in results if r["case"] == case]
        if not rows:
            continue
        ref = REFERENCE[case]
        if case == "steady":
            lines.append("### 2D-1, steady, Re = 20\n")
            lines.append("| mesh | nodes | tri / quad | c_D | c_D surface | c_L | dp | steps | c_D change | time (s) |")
            lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for r in rows:
                q = r.get("quantities", {})
                cells = [fmt_error(q.get(k), PRECISE_STEADY[k]) for k in ("cd", "cd_surface", "cl", "dp")]
                lines.append(f"| {r['mesh']} | {r.get('nodes', '-')} | {r.get('triangles', '-')} / {r.get('quads', '-')} "
                             f"| {' | '.join(cells)} | {q.get('steps', '-')} | {q.get('cd_change', float('nan')):.1e} "
                             f"| {r.get('seconds', 0):.1f} |")
            lines.append("\nIn parentheses: the error relative to the high-accuracy values c_D = 5.579535, "
                         "c_L = 0.0106189, dp = 2.938004 (John & Matthies 2001). Reference intervals of "
                         f"Schafer & Turek (1996): c_D {ref['cd']}, c_L {ref['cl']}, "
                         f"dp {tuple(round(x, 4) for x in ref['dp'])}. c_D change: over the last 100 steps.\n")
        else:
            lines.append("### 2D-2, periodic, Re = 100\n")
            lines.append("| mesh | nodes | c_D max | c_D mean | c_L max | St | dp | steps | time (s) |")
            lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for r in rows:
                q = r.get("quantities", {})
                lines.append(f"| {r['mesh']} | {r.get('nodes', '-')} | {fmt(q.get('cd_max'), ref['cd_max'])} "
                             f"| {fmt(q.get('cd_mean'))} | {fmt(q.get('cl_max'), ref['cl_max'])} "
                             f"| {fmt(q.get('st'), ref['st'])} | {fmt(q.get('dp'), ref['dp'])} "
                             f"| {q.get('steps', '-')} | {r.get('seconds', 0):.1f} |")
            lines.append(f"\nReference (Schafer & Turek 1996): c_D max {ref['cd_max']}, c_L max {ref['cl_max']}, "
                         f"St {ref['st']}, dp {ref['dp']}.\n")
        for r in rows:
            if "error" in r:
                lines.append(f"- {r['mesh']}: {r['error']}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(work, "summary.md"), "w") as f:
        f.write(text)
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", required=True)
    p.add_argument("--meshes", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--cases", nargs="+", default=["steady", "unsteady"], choices=["steady", "unsteady"])
    p.add_argument("--only", nargs="+", help="mesh names (default: every *.msh in --meshes)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--end-time", type=float, default=60.0, help="end time of the periodic case")
    p.add_argument("--periods", type=int, default=4, help="lift periods analysed at the end of the periodic case")
    p.add_argument("--analyse-only", action="store_true", help="do not run, analyse existing results")
    a = p.parse_args()

    binary = os.path.abspath(a.bin)
    meshes = a.only or sorted(f[:-4] for f in os.listdir(a.meshes) if f.endswith(".msh"))
    os.makedirs(a.work, exist_ok=True)
    runs = [(case, m, os.path.join(a.work, f"{case}_{m}")) for case in a.cases for m in meshes]

    if not a.analyse_only:
        for case, m, d in runs:
            write_case(d, os.path.join(a.meshes, m + ".msh"), case, a.end_time)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures = {pool.submit(run_one, binary, d): (case, m) for case, m, d in runs}
            for fut in concurrent.futures.as_completed(futures):
                case, m = futures[fut]
                code, seconds = fut.result()
                print(f"{case:8s} {m:8s} exit {code} ({seconds:.0f} s)", flush=True)

    results = []
    for case, m, d in runs:
        r = {"case": case, "mesh": m}
        try:
            with open(os.path.join(d, "run.log")) as f:
                r.update(mesh_info(f.read()))
            data = read_monitors(os.path.join(d, "results", "cylinder.monitors.dat"))
            r["quantities"] = periodic(data, a.periods, 1.0) if case == "unsteady" else steady(data, 1.0)
        except (OSError, ValueError) as e:
            r["error"] = str(e)
        results.append(r)
    print(summarise(a.work, results))


if __name__ == "__main__":
    main()
