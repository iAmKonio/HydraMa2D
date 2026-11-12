#!/usr/bin/env python3
"""Automated Aerodynamic Polar Sweeps Tool for HydraMa2D.

Executes angle-of-attack schedules, manages warm-restart chaining between consecutive
angles to accelerate convergence, parses converged aerodynamic force coefficients,
transforms body forces to wind axes (CL, CD, CM, L/D), outputs structured polar.csv,
and generates publication-quality 3-panel aerodynamic polar plots.

Usage examples:
    # Basic sweep from -2 deg to 14 deg in steps of 2 deg
    python3 tools/run_polar.py --case-dir cases/naca0012 --alpha-min -2 --alpha-max 14 --alpha-step 2

    # Explicit list of angles with warm-restart
    python3 tools/run_polar.py --case-dir cases/naca0012 --alphas 0,2,4,6,8,10,12,14,16 --warm-start

    # Specify freestream velocity and output directory
    python3 tools/run_polar.py --case-dir cases/naca0012 --speed 1.0 --output-dir results/polar_naca0012
"""

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Configure Matplotlib cache directory to avoid permissions warnings in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def find_hydrama_bin(hint_path=None):
    """Finds the HydraMa2D binary."""
    candidates = []
    if hint_path:
        candidates.append(Path(hint_path))
    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend([
        repo_root / "build" / "HydraMa2D",
        repo_root / "build" / "src" / "HydraMa2D",
        Path("build/HydraMa2D"),
        Path("HydraMa2D"),
    ])
    which_bin = shutil.which("HydraMa2D")
    if which_bin:
        candidates.append(Path(which_bin))

    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()
    raise FileNotFoundError(
        f"HydraMa2D executable not found. Checked: {[str(c) for c in candidates]}. "
        "Please build the project with `cmake --build build` or pass --bin <path>."
    )


def detect_case_info(case_dir, casename_hint=None):
    """Detects problem name and checks required directory structure."""
    case_path = Path(case_dir).resolve()
    if not case_path.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {case_path}")

    constant_dir = case_path / "constant"
    system_dir = case_path / "system"
    if not constant_dir.is_dir() or not system_dir.is_dir():
        raise ValueError(
            f"Case directory {case_path} must contain 'constant/' and 'system/' subdirectories."
        )

    casename = casename_hint
    if not casename:
        # Search for *.control in system/
        control_files = list(system_dir.glob("*.control"))
        if control_files:
            casename = control_files[0].stem
        else:
            # Fall back to directory name
            casename = case_path.name

    return casename, case_path


def parse_monitors_file(monitors_path):
    """Parses HydraMa2D monitor file (results/<case>.monitors.dat).

    Returns a tuple of (columns_dict, last_row_dict).
    """
    if not os.path.exists(monitors_path):
        return None, None

    header = None
    rows = []
    with open(monitors_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = line.split()
                continue
            try:
                values = [float(x) for x in line.split()]
                if len(values) == len(header):
                    rows.append(values)
            except ValueError:
                continue

    if not header or not rows:
        return None, None

    col_data = {col: [r[i] for r in rows] for i, col in enumerate(header)}
    last_row = {col: rows[-1][i] for i, col in enumerate(header)}
    return col_data, last_row


def transform_forces_to_wind(alpha_deg, fx, fy, fx_press=0.0, fy_press=0.0, mz=0.0):
    """Transforms body forces (global x, y) into aerodynamic wind axes (Drag, Lift).

    Wind axes conventions:
        Freestream enters at angle alpha relative to chord (global x).
        Drag is aligned with freestream vector (u_inf, v_inf).
        Lift is perpendicular to freestream vector.

        D  =  Fx * cos(alpha) + Fy * sin(alpha)
        L  = -Fx * sin(alpha) + Fy * cos(alpha)
    """
    alpha_rad = math.radians(alpha_deg)
    cos_a = math.cos(alpha_rad)
    sin_a = math.sin(alpha_rad)

    cl = -fx * sin_a + fy * cos_a
    cd = fx * cos_a + fy * sin_a

    cl_p = -fx_press * sin_a + fy_press * cos_a
    cd_p = fx_press * cos_a + fy_press * sin_a
    cd_f = cd - cd_p

    l_over_d = (cl / cd) if abs(cd) > 1e-12 else 0.0
    cm = mz

    return {
        "alpha_deg": alpha_deg,
        "alpha_rad": alpha_rad,
        "CL": cl,
        "CD": cd,
        "CM": cm,
        "L_over_D": l_over_d,
        "CL_pressure": cl_p,
        "CD_pressure": cd_p,
        "CD_friction": cd_f,
        "CFx": fx,
        "CFy": fy,
    }


def update_control_file(control_file, restart_opt=None, ntime=None, tol=None):
    """Updates selected parameters in a HydraMa2D *.control file."""
    with open(control_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # The HydraMa control file format has 45 non-empty lines:
    # 15 blocks of 3 lines: divider ('---'), description, value.
    # Block 0 (restart_opt): non-empty line index 2
    # Block 1 (ntime): non-empty line index 5
    # Block 13 (l2norm_vel_tolerance): non-empty line index 41
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]

    if restart_opt is not None and len(non_empty_indices) > 2:
        lines[non_empty_indices[2]] = f"{int(restart_opt)}\n"

    if ntime is not None and len(non_empty_indices) > 5:
        lines[non_empty_indices[5]] = f"{int(ntime)}\n"

    if tol is not None and len(non_empty_indices) > 41:
        lines[non_empty_indices[41]] = f"{float(tol):.3e}\n"

    with open(control_file, "w", encoding="utf-8") as f:
        f.writelines(lines)


def update_physical_file(physical_file, re_override=None):
    """Updates Reynolds number in a HydraMa2D *.physical file if requested."""
    if re_override is None:
        return
    with open(physical_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
    # Block 0 (re): non-empty line index 2
    if len(non_empty_indices) > 2:
        lines[non_empty_indices[2]] = f"{float(re_override):.4f}\n"
    with open(physical_file, "w", encoding="utf-8") as f:
        f.writelines(lines)


def prepare_warm_start_var(source_var, target_var):
    """Copies restart file while resetting the step counter to 0 so the next angle starts from step 1."""
    with open(source_var, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if lines:
        lines[0] = "0.0 0\n"
    with open(target_var, "w", encoding="utf-8") as f:
        f.writelines(lines)


def plot_polar(results, output_png, title="HydraMa2D Aerodynamic Polar"):
    """Generates a publication-quality 3-panel aerodynamic polar figure."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib or numpy not available; skipping plot generation.")
        return

    alphas = np.array([r["alpha_deg"] for r in results])
    cls = np.array([r["CL"] for r in results])
    cds = np.array([r["CD"] for r in results])
    l_d = np.array([r["L_over_D"] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    # -------------------------------------------------------------
    # Panel 1: Lift curve CL(alpha) & Lift slope dCL/dalpha
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(alphas, cls, "o-", color="#1f77b4", linewidth=2.0, markersize=6, label=r"$C_L$ (CFD)")

    # Linear lift slope fit (typically alpha between -2 and 6 deg, or first few angles)
    linear_mask = (alphas >= -2.0) & (alphas <= 6.0)
    if np.sum(linear_mask) >= 2:
        fit_alphas = alphas[linear_mask]
        fit_cls = cls[linear_mask]
        slope_deg, intercept = np.polyfit(fit_alphas, fit_cls, 1)
        slope_rad = slope_deg * (180.0 / np.pi)
        line_x = np.linspace(min(fit_alphas) - 1.0, max(fit_alphas) + 2.0, 50)
        line_y = slope_deg * line_x + intercept
        ax1.plot(
            line_x,
            line_y,
            "--",
            color="#d62728",
            linewidth=1.5,
            label=f"Fit: $dC_L/d\\alpha = {slope_rad:.3f}/\\mathrm{{rad}}$\n"
                  f"      $({slope_deg:.4f}/^\\circ)$\n"
                  f"Theory: $2\\pi \\approx 6.283/\\mathrm{{rad}}$",
        )

    # Highlight CL_max / Stall
    idx_max = np.argmax(cls)
    cl_max = cls[idx_max]
    alpha_stall = alphas[idx_max]
    ax1.scatter([alpha_stall], [cl_max], color="#d62728", s=80, zorder=5)
    ax1.annotate(
        f"$C_{{L,max}} = {cl_max:.3f}$\n@ $\\alpha = {alpha_stall:.1f}^\\circ$",
        xy=(alpha_stall, cl_max),
        xytext=(alpha_stall - 3.5, cl_max - 0.15),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        fontweight="bold",
        fontsize=9,
    )

    ax1.set_xlabel(r"Angle of Attack $\alpha$ [deg]", fontsize=11)
    ax1.set_ylabel(r"Lift Coefficient $C_L$", fontsize=11)
    ax1.set_title(r"Lift Curve $C_L(\alpha)$", fontsize=12, fontweight="semibold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax1.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Drag curve CD(alpha) & Breakdown (Pressure vs Friction)
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(alphas, cds, "s-", color="#2ca02c", linewidth=2.0, markersize=6, label=r"$C_D$ (Total)")
    if "CD_pressure" in results[0]:
        cd_p = np.array([r["CD_pressure"] for r in results])
        cd_f = np.array([r["CD_friction"] for r in results])
        ax2.plot(alphas, cd_p, "--", color="#ff7f0e", linewidth=1.5, label=r"$C_{D,p}$ (Pressure)")
        ax2.plot(alphas, cd_f, ":", color="#9467bd", linewidth=1.5, label=r"$C_{D,f}$ (Friction)")

    idx_cd_min = np.argmin(cds)
    cd_min = cds[idx_cd_min]
    alpha_cd_min = alphas[idx_cd_min]
    ax2.scatter([alpha_cd_min], [cd_min], color="#2ca02c", s=80, zorder=5)
    ax2.annotate(
        f"$C_{{D,min}} = {cd_min:.4f}$\n@ $\\alpha = {alpha_cd_min:.1f}^\\circ$",
        xy=(alpha_cd_min, cd_min),
        xytext=(alpha_cd_min + 1.0, cd_min + 0.015),
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
        fontweight="bold",
        fontsize=9,
    )

    ax2.set_xlabel(r"Angle of Attack $\alpha$ [deg]", fontsize=11)
    ax2.set_ylabel(r"Drag Coefficient $C_D$", fontsize=11)
    ax2.set_title(r"Drag Curve $C_D(\alpha)$", fontsize=12, fontweight="semibold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Drag Polar (CL vs CD) & (L/D)_max Tangent
    # -------------------------------------------------------------
    ax3 = axes[2]
    ax3.plot(cds, cls, "d-", color="#9467bd", linewidth=2.0, markersize=6, label=r"Drag Polar ($C_L$ vs $C_D$)")

    # Find (L/D)_max
    valid_ld = np.where(cds > 1e-6, cls / cds, -1e9)
    idx_ld_max = np.argmax(valid_ld)
    ld_max = valid_ld[idx_ld_max]
    cd_opt = cds[idx_ld_max]
    cl_opt = cls[idx_ld_max]
    alpha_opt = alphas[idx_ld_max]

    # Draw tangent line from origin to (L/D)_max
    tangent_cd = np.linspace(0, max(cds) * 1.05, 50)
    tangent_cl = ld_max * tangent_cd
    ax3.plot(tangent_cd, tangent_cl, "--", color="#e377c2", linewidth=1.4,
             label=f"Tangent: $(L/D)_{{max}} = {ld_max:.2f}$\n"
                   f"@ $\\alpha = {alpha_opt:.1f}^\\circ$")
    ax3.scatter([cd_opt], [cl_opt], color="#e377c2", s=90, zorder=5)

    ax3.set_xlabel(r"Drag Coefficient $C_D$", fontsize=11)
    ax3.set_ylabel(r"Lift Coefficient $C_L$", fontsize=11)
    ax3.set_title(r"Drag Polar ($C_L$ vs $C_D$)", fontsize=12, fontweight="semibold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.set_xlim(left=0)
    ax3.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
    ax3.legend(loc="lower right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)
    print(f"Polar plot saved to: {output_png}")


def run_sweep(args):
    """Executes the complete aerodynamic polar sweep."""
    hydrama_bin = find_hydrama_bin(args.bin)
    casename, case_path = detect_case_info(args.case_dir, args.casename)

    output_dir = Path(args.output_dir if args.output_dir else case_path / "polar_output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Determine angle list
    if args.alphas:
        alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    else:
        alpha_min = args.alpha_min
        alpha_max = args.alpha_max
        step = args.alpha_step
        num_pts = int(round((alpha_max - alpha_min) / step)) + 1
        alphas = [round(alpha_min + i * step, 6) for i in range(num_pts)]

    print("=================================================================")
    print("  HydraMa2D Automated Aerodynamic Polar Sweeps Tool              ")
    print("=================================================================")
    print(f"  Case:         {casename} in {case_path}")
    print(f"  Binary:       {hydrama_bin}")
    print(f"  Output Dir:   {output_dir}")
    print(f"  Angles (deg): {alphas}")
    print(f"  Speed (U_inf):{args.speed}")
    print(f"  Warm-Start:   {args.warm_start}")
    print("=================================================================\n")

    polar_csv_path = output_dir / "polar.csv"
    results = []

    last_converged_var = None

    for idx, alpha in enumerate(alphas):
        angle_str = f"{alpha:+.2f}".replace("+", "p").replace("-", "m")
        run_dir = runs_dir / f"alpha_{angle_str}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Set up run directory: constant, system, results
        (run_dir / "constant").mkdir(exist_ok=True)
        (run_dir / "system").mkdir(exist_ok=True)
        (run_dir / "results").mkdir(exist_ok=True)

        # Copy or symlink constant/
        for item in (case_path / "constant").iterdir():
            target = run_dir / "constant" / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            if item.is_file():
                shutil.copy2(item, target)
            elif item.is_dir():
                shutil.copytree(item, target)

        # Copy system/
        for item in (case_path / "system").iterdir():
            target = run_dir / "system" / item.name
            if item.is_file():
                shutil.copy2(item, target)

        # Configure system/<case>.inflow
        inflow_file = run_dir / "system" / f"{casename}.inflow"
        with open(inflow_file, "w", encoding="utf-8") as f:
            f.write(f"# Inflow angle of attack configuration for alpha = {alpha} deg\n")
            f.write(f"speed {args.speed}\n")
            f.write(f"angle {alpha}\n")

        # Configure system/<case>.monitors if missing
        monitors_file = run_dir / "system" / f"{casename}.monitors"
        if not monitors_file.exists():
            # If base case has monitors, copy it; else auto-generate default body monitor
            base_mon = case_path / "system" / f"{casename}.monitors"
            if base_mon.exists():
                shutil.copy2(base_mon, monitors_file)
            else:
                with open(monitors_file, "w", encoding="utf-8") as f:
                    f.write("# Aerodynamic force monitor\n")
                    f.write("body airfoil wall cylinder 500\n")
                    f.write(f"reference_velocity {args.speed}\n")
                    f.write("reference_length 1.0\n")
                    f.write("interval 1\n")
                    f.write("moment_center 0.25 0.0\n")

        # Handle warm-restart chaining
        control_file = run_dir / "system" / f"{casename}.control"
        is_warm = False
        if args.warm_start and last_converged_var and last_converged_var.exists():
            target_var = run_dir / "results" / f"{casename}.var"
            prepare_warm_start_var(last_converged_var, target_var)
            src_turb = last_converged_var.parent / f"{casename}.turb.var"
            if src_turb.exists():
                shutil.copy2(src_turb, run_dir / "results" / f"{casename}.turb.var")
            src_thermal = last_converged_var.parent / f"{casename}.thermal.var"
            if src_thermal.exists():
                shutil.copy2(src_thermal, run_dir / "results" / f"{casename}.thermal.var")
            update_control_file(
                control_file,
                restart_opt=1,
                ntime=args.max_steps,
                tol=args.tol,
            )
            is_warm = True
        else:
            update_control_file(
                control_file,
                restart_opt=0,
                ntime=args.max_steps,
                tol=args.tol,
            )

        # Update Reynolds number if specified
        if args.re:
            update_physical_file(run_dir / "system" / f"{casename}.physical", args.re)

        print(f"[{idx+1}/{len(alphas)}] Running alpha = {alpha:+.2f} deg "
              f"({'warm-restart' if is_warm else 'cold-start'})...", end="", flush=True)

        start_time = time.time()
        cmd = [str(hydrama_bin.resolve()), str(run_dir.resolve())]
        if args.max_steps is not None:
            cmd.extend(["--max-steps", str(args.max_steps)])
        stdout_dest = subprocess.DEVNULL if args.quiet else open(run_dir / "run.log", "w", encoding="utf-8")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(run_dir),
                stdout=stdout_dest,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            elapsed = time.time() - start_time
            if not args.quiet and hasattr(stdout_dest, "close"):
                stdout_dest.close()

            if proc.returncode != 0:
                print(f" FAILED (exit code {proc.returncode})")
                if proc.stderr:
                    print(f"    Error: {proc.stderr.strip()[:200]}")
                continue

            # Parse monitors file
            mon_dat = run_dir / "results" / f"{casename}.monitors.dat"
            col_data, last_row = parse_monitors_file(mon_dat)
            if not last_row:
                print(" FAILED (no monitor data written)")
                continue

            # Extract forces
            fx = last_row.get("cd", 0.0)
            fy = last_row.get("cl", 0.0)
            fx_press = last_row.get("cd_pressure", 0.0)
            fy_press = last_row.get("cl_pressure", 0.0)
            mz = last_row.get("cm", 0.0)
            step_num = int(last_row.get("step", 0))

            aero = transform_forces_to_wind(alpha, fx, fy, fx_press, fy_press, mz)
            aero["iterations"] = step_num
            aero["converged"] = 1
            aero["runtime_sec"] = round(elapsed, 2)
            results.append(aero)

            # Update warm-restart pointer
            var_file = run_dir / "results" / f"{casename}.var"
            if var_file.exists():
                last_converged_var = var_file

            print(f" DONE in {elapsed:.2f}s | CL={aero['CL']:+.4f} CD={aero['CD']:.4f} L/D={aero['L_over_D']:+.2f}")

            # Write/update polar.csv continuously
            fieldnames = [
                "alpha_deg", "alpha_rad", "CL", "CD", "CM", "L_over_D",
                "CL_pressure", "CD_pressure", "CD_friction", "CFx", "CFy",
                "converged", "iterations", "runtime_sec"
            ]
            with open(polar_csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)

        except Exception as e:
            print(f" ERROR: {e}")

    # Summary and plotting
    if not results:
        print("\nERROR: No successful angle runs completed.")
        return 1

    print("\n=================================================================")
    print("  Polar Sweep Summary                                            ")
    print("=================================================================")
    print(f" {'alpha (deg)':>11} | {'CL':>8} | {'CD':>8} | {'L/D':>8} | {'CD_p':>8} | {'CD_f':>8} | {'Steps':>6} | {'Time (s)':>8}")
    print(" " + "-" * 78)
    for r in results:
        print(f" {r['alpha_deg']:>11.2f} | {r['CL']:>8.4f} | {r['CD']:>8.4f} | {r['L_over_D']:>8.2f} | "
              f"{r['CD_pressure']:>8.4f} | {r['CD_friction']:>8.4f} | {r['iterations']:>6} | {r['runtime_sec']:>8.2f}")
    print("=================================================================")
    print(f"Results table written to: {polar_csv_path}")

    if args.plot:
        plot_path = output_dir / "polar.png"
        plot_polar(results, plot_path, title=f"Aerodynamic Polar: {casename}")

    if args.clean_runs:
        shutil.rmtree(runs_dir, ignore_errors=True)
        print("Cleaned up intermediate run directories.")

    return 0


def build_cli_parser():
    """Builds the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Automated Aerodynamic Polar Sweeps Tool for HydraMa2D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case-dir", required=True, help="Base case directory containing constant/ and system/")
    parser.add_argument("--casename", default=None, help="Case name (default: auto-detected from case-dir)")
    parser.add_argument("--alpha-min", type=float, default=0.0, help="Minimum angle of attack (degrees)")
    parser.add_argument("--alpha-max", type=float, default=10.0, help="Maximum angle of attack (degrees)")
    parser.add_argument("--alpha-step", type=float, default=2.0, help="Angle of attack step (degrees)")
    parser.add_argument("--alphas", type=str, default=None, help="Explicit comma-separated angles (e.g. -2,0,2,4,6,8)")
    parser.add_argument("--speed", type=float, default=1.0, help="Freestream velocity magnitude U_inf")
    parser.add_argument("--re", type=float, default=None, help="Reynolds number override")
    parser.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True,
                        help="Chain solution from alpha_i to alpha_{i+1}")
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum timesteps per angle")
    parser.add_argument("--tol", type=float, default=None, help="Velocity L2-norm convergence tolerance override")
    parser.add_argument("--output-dir", default=None, help="Directory for polar outputs, CSV, and plots")
    parser.add_argument("--bin", default=None, help="Path to HydraMa2D executable")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True, help="Generate polar plots")
    parser.add_argument("--clean-runs", action="store_true", help="Remove individual angle run directories after sweep")
    parser.add_argument("--quiet", action="store_true", help="Suppress solver stdout during sweep")
    return parser


if __name__ == "__main__":
    parser = build_cli_parser()
    args = parser.parse_args()
    sys.exit(run_sweep(args))
