#!/usr/bin/env python3
"""Generates a complete NACA 0012 airfoil benchmark case for HydraMa2D.

Creates a 2D mesh (SU2 format) around a NACA 0012 airfoil in an O-grid topology,
with refined boundary layer clustering at the airfoil surface and leading/trailing edges,
and sets up constant/ and system/ configuration files ready for polar sweeps.

Boundary markers:
    - airfoil_wall: no-slip wall (flag 500)
    - inlet: outer inflow boundary (flag 503)
    - outlet: downstream outflow boundary (flag 504)
"""

import argparse
import math
import os
import sys
from pathlib import Path


def naca0012_yt(x, t=0.12):
    """NACA 4-digit half-thickness distribution (closed trailing edge)."""
    return 5.0 * t * (
        0.2969 * math.sqrt(max(0.0, x))
        - 0.1260 * x
        - 0.3516 * (x ** 2)
        + 0.2843 * (x ** 3)
        - 0.1036 * (x ** 4)
    )


def generate_naca0012_su2(output_su2_path, n_foil=96, n_radial=28, r_far=12.0, first_layer=0.015):
    """Generates an O-grid mesh around a NACA 0012 airfoil in SU2 format."""
    if n_foil % 2 != 0:
        n_foil += 1

    foil_x = []
    foil_y = []

    # Parameterize NACA 0012 counter-clockwise from trailing edge upper (beta=0),
    # around leading edge (beta=pi), to trailing edge lower (beta=2*pi).
    foil_x = []
    foil_y = []
    for i in range(n_foil):
        beta = 2.0 * math.pi * i / n_foil
        x = 0.5 * (1.0 + math.cos(beta))
        yt = naca0012_yt(x)
        y = yt if beta <= math.pi else -yt
        foil_x.append(x)
        foil_y.append(y)

    # Radial distribution: smooth geometric stretching
    growth = 1.18
    radial_r = [0.0]
    cur_dr = first_layer
    for j in range(1, n_radial + 1):
        radial_r.append(radial_r[-1] + cur_dr)
        cur_dr *= growth
    scale = r_far / radial_r[-1]
    radial_r = [r * scale for r in radial_r]

    nodes_x = []
    nodes_y = []

    for j in range(n_radial + 1):
        s = j / n_radial
        for i in range(n_foil):
            beta = 2.0 * math.pi * i / n_foil
            fx = foil_x[i]
            fy = foil_y[i]

            # Farfield point along matching ray from chord midpoint (0.5, 0.0)
            ox = 0.5 + r_far * math.cos(beta)
            oy = r_far * math.sin(beta)

            # Transfinite interpolation
            x = (1.0 - s) * fx + s * ox
            y = (1.0 - s) * fy + s * oy
            nodes_x.append(x)
            nodes_y.append(y)

    def node_id(i, j):
        return j * n_foil + (i % n_foil)

    # Counter-clockwise triangle elements in fluid domain
    elements = []
    for j in range(n_radial):
        for i in range(n_foil):
            n00 = node_id(i, j)
            n01 = node_id(i, j + 1)
            n11 = node_id(i + 1, j + 1)
            n10 = node_id(i + 1, j)

            elements.append((n00, n01, n11))
            elements.append((n00, n11, n10))

    # Boundary markers
    # 1. airfoil_wall (j = 0)
    foil_edges = []
    for i in range(n_foil):
        foil_edges.append((node_id(i, 0), node_id(i + 1, 0)))

    # 2. Outer boundary (j = n_radial): inlet (flag 503) and outlet (flag 504)
    # Downstream wake wedge (|beta| < 45 deg or |beta - 2*pi| < 45 deg): outlet
    inlet_edges = []
    outlet_edges = []
    for i in range(n_foil):
        n1 = node_id(i, n_radial)
        n2 = node_id(i + 1, n_radial)
        beta_mid = 2.0 * math.pi * (i + 0.5) / n_foil
        ang = math.atan2(math.sin(beta_mid), math.cos(beta_mid))
        if abs(ang) < math.radians(45.0):
            outlet_edges.append((n1, n2))
        else:
            inlet_edges.append((n1, n2))

    # Write SU2 mesh
    with open(output_su2_path, "w", encoding="utf-8") as f:
        f.write("%\n% NACA 0012 Airfoil Mesh for HydraMa2D\n%\n")
        f.write("NDIME= 2\n")
        f.write(f"NPOIN= {len(nodes_x)}\n")
        for idx, (x, y) in enumerate(zip(nodes_x, nodes_y)):
            f.write(f"{x:.10e} {y:.10e} {idx}\n")

        f.write(f"NELEM= {len(elements)}\n")
        for idx, (n1, n2, n3) in enumerate(elements):
            f.write(f"5 {n1} {n2} {n3} {idx}\n")

        markers = [
            ("airfoil_wall", foil_edges),
            ("inlet", inlet_edges),
            ("outlet", outlet_edges),
        ]
        f.write(f"NMARK= {len(markers)}\n")
        for name, edges in markers:
            f.write(f"MARKER_TAG= {name}\n")
            f.write(f"MARKER_ELEMS= {len(edges)}\n")
            for n1, n2 in edges:
                f.write(f"3 {n1} {n2}\n")

    print(f"Generated mesh: {output_su2_path} ({len(nodes_x)} points, {len(elements)} elements)")


def setup_naca0012_case(case_dir, re=1000.0, speed=1.0, max_steps=500, n_foil=96, n_radial=28, r_far=12.0):
    """Sets up a complete case directory for NACA 0012."""
    case_path = Path(case_dir).resolve()
    const_dir = case_path / "constant"
    system_dir = case_path / "system"
    results_dir = case_path / "results"
    const_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate mesh
    mesh_path = const_dir / "naca0012.su2"
    generate_naca0012_su2(mesh_path, n_foil=n_foil, n_radial=n_radial, r_far=r_far)

    # 2. Control file
    with open(system_dir / "naca0012.control", "w", encoding="utf-8") as f:
        f.write(f"""---
Restart from prior run (0 - NO, 1 - YES)
0
---
Number of Timesteps (Max in case of steady state)
{max_steps}
---
Transient (0 - OFF, 1 - ON)
0
---
Simulated Time Interval for Output (transient only)
1.0
---
End Simulation Time (transient only)
100.0
---
Output Controls - Every N iterations
500
---
Output Controls - Every N CPU seconds
3600
---
Global Minimum Calculated Timestep = -1 OR Local Time Steping = 0 OR Specified Time Step Value = 1
0
---
Specified Time Step Value
1.0e-4
---
Convergence Check Interval
20
---
Safety Factor for Calculated Local Time Step
0.6
---
Theta1 (0.5<=theta1<=1.0). Theta2 Hardcoded as 1.0 for Semi-Implicit
1.0
---
Steady State Velocity Convergence (0 - OFF, 1 - ON)
1
---
L2-Norm
1.0e-5
---
Run Time Adjustable Option
0
""")

    # 3. Physical file
    with open(system_dir / "naca0012.physical", "w", encoding="utf-8") as f:
        f.write(f"""---------------------------------------------------------------------------------------------------
Reynolds Number
{re:.2f}
---------------------------------------------------------------------------------------------------
pnode - Node in domain for Pressure BC if no other Pressure BCs exist
1
""")

    # 4. Solver file
    with open(system_dir / "naca0012.solver", "w", encoding="utf-8") as f:
        f.write("""---------------------------------------------------------------------------------------------------
Pressure Field Solver (1 - Conjugate Gradient, Only PCG has been implemented)
1
---------------------------------------------------------------------------------------------------
Conjugate Gradient Convergence Tolerances - Relative and Absolute Residual. Zero for OFF
Relative Residual. Zero for OFF
1.0e-04
---------------------------------------------------------------------------------------------------
Absolute Residual. Zero for OFF
1.0e-08
---------------------------------------------------------------------------------------------------
Linear Solver Options (optional, one "key = value" per line; omitted keys keep the defaults shown)
method          = gmres
preconditioner  = ilu0
operator_format = csr
reordering      = rcm
gmres_restart   = 50
max_iterations  = 500
""")

    # 5. Inflow file
    with open(system_dir / "naca0012.inflow", "w", encoding="utf-8") as f:
        f.write(f"""# Inflow boundary conditions
speed {speed}
angle 0.0
""")

    # 6. Monitors file
    with open(system_dir / "naca0012.monitors", "w", encoding="utf-8") as f:
        f.write(f"""# Aerodynamic force monitor for NACA 0012
body airfoil_wall
reference_velocity {speed}
reference_length 1.0
interval 1
moment_center 0.25 0.0
""")

    print(f"NACA 0012 case successfully set up at: {case_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NACA 0012 benchmark case for HydraMa2D")
    parser.add_argument("--case-dir", default="cases/naca0012", help="Target case directory")
    parser.add_argument("--re", type=float, default=1000.0, help="Reynolds number")
    parser.add_argument("--speed", type=float, default=1.0, help="Freestream velocity")
    parser.add_argument("--max-steps", type=int, default=500, help="Max iterations")
    parser.add_argument("--n-foil", type=int, default=96, help="Number of surface points along airfoil")
    parser.add_argument("--n-radial", type=int, default=28, help="Number of radial layers in O-grid")
    parser.add_argument("--r-far", type=float, default=12.0, help="Farfield radius in chord lengths")
    args = parser.parse_args()

    setup_naca0012_case(
        args.case_dir,
        re=args.re,
        speed=args.speed,
        max_steps=args.max_steps,
        n_foil=args.n_foil,
        n_radial=args.n_radial,
        r_far=args.r_far,
    )
