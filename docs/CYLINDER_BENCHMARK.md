# Cylinder Benchmark: Triangles vs. Inflation Layers

The laminar "flow around a cylinder" benchmark of Schäfer & Turek (1996) validates HydraMa2D against published drag,
lift, pressure and Strouhal values, on triangle meshes and on the same meshes with a layer of quadrilaterals around
the cylinder (an inflation layer). It answers the question the quadrilateral support was built for: do inflation
layers improve wall quantities, and at what cost?

This document covers:
1. the problem and the reference values;
2. how forces are computed (`include/post/Forces.hpp`);
3. the meshes;
4. the results of the steady case 2D-1 (Re = 20) and the periodic case 2D-2 (Re = 100);
5. conclusions and limitations;
6. how to reproduce the results.

The benchmark uses four features added with it: the parabolic inlet (flag 508, `parabolic_inlet`), force and probe
monitors (`system/<case>.monitors`), the mesh quality report, and `HydraMa2D --check-mesh`.

---

## 1. Problem

Schäfer & Turek define the problem in SI units (D = 0.1 m, ν = 10⁻³ m²/s, ρ = 1 kg/m³). HydraMa2D is
non-dimensional. With the diameter D as the unit of length and the mean inflow velocity Ū as the unit of velocity:

| | Schäfer & Turek | HydraMa2D (D = 1, Ū = 1) |
| :--- | :--- | :--- |
| channel | [0, 2.2] × [0, 0.41] m | [0, 22] × [0, 4.1] |
| cylinder | centre (0.2, 0.2) m, D = 0.1 m | centre (2, 2), radius 0.5 (not centred: 1.5 below, 1.6 above) |
| inflow | U(0, y) = 4 U_m y (H − y) / H², Ū = 2 U_m / 3 | `parabolic_inlet` (flag 508): mean 1, peak 1.5 |
| walls | no slip on the channel walls and the cylinder | `channel_wall`, `cylinder_wall` (500) |
| outflow | "do nothing" | `outlet` (504): p = 0, natural velocity condition |
| Reynolds number | Re = Ū D / ν: 20 (2D-1, Ū = 0.2 m/s), 100 (2D-2, Ū = 1 m/s) | `system/<case>.physical` |
| drag / lift coefficient | c = 2 F / (ρ Ū² D) | c = 2 F |
| pressure difference | Δp = p(0.15, 0.2) − p(0.25, 0.2) in Pa | Δp = p(1.5, 2) − p(2.5, 2) = Δp_SI / Ū² |
| Strouhal number | St = D f / Ū | St = f |

The two pressure points are the front and the rear stagnation points on the cylinder surface. Time is
t = t_SI Ū / D.

**Reference values.** Schäfer & Turek (1996) give intervals that the best results of the participating groups fell
into. For 2D-1, John & Matthies (2001) computed high-accuracy values with higher-order elements on fine meshes; the
errors below are relative to these.

| Case | Quantity | Schäfer & Turek interval | High-accuracy value |
| :--- | :--- | :--- | :--- |
| 2D-1 | c_D | [5.57, 5.59] | 5.57953523384 |
| | c_L | [0.0104, 0.0110] | 0.010618948146 |
| | Δp | [0.1172, 0.1176] Pa → [2.930, 2.940] | 0.11752016697 Pa → 2.93800417 |
| 2D-2 | c_D,max | [3.22, 3.24] | |
| | c_L,max | [0.99, 1.01] | |
| | St | [0.295, 0.305] | |
| | Δp(t₀ + 1/(2f)) | [2.46, 2.50] | |

For 2D-2, t₀ is the time of a lift maximum, so Δp is taken half a period later.

---

## 2. Forces on a body

The force per unit span exerted by the fluid on the body (density 1) is

    F = ∫_body σ n_b ds,      σ = −p I + ν (∇u + ∇uᵀ),

where n_b is the unit normal pointing out of the body into the fluid. `post::surface_force` and
`post::volume_force` evaluate it in two ways.

**Surface traction.** The integral is evaluated along the body sides:
- the pressure is linear along a side, so the pressure part is exact;
- the velocity gradient is that of the adjacent element: constant in a triangle, and at the side midpoint in a
  quadrilateral (exact for its linear variation along the side).

The viscous part is only first-order accurate in the size of the wall-adjacent cell. For Poiseuille flow the
computed wall shear is 6ν(1 − h) instead of 6ν (tested in `[post][forces]`). This is the quantity that depends
directly on the near-wall resolution.

**Volume (reaction) form.** The momentum equation is multiplied by a test function w that equals e_k (k = x, y) on
the body and vanishes on every other boundary, and integrated by parts:

    F_k = − ∫_Ω [ (∂u/∂t + (u·∇)u) · w + ν ∇u : ∇w − p ∇·w ] dΩ.

w is the finite element function that is 1 at the body nodes and 0 elsewhere, so only the first ring of elements
around the body contributes.
- For smooth solutions this form converges like the energy norm squared: Babuška & Miller (1984); Giles, Larson,
  Levenstam & Süli (1997). Schäfer & Turek recommend it, and John & Matthies (2001) and John (2004) used it for the
  reference values.
- The viscous term is written in the form the solver discretises (ν∇u : ∇w, Step 1). On a no-slip wall it equals the
  symmetric form, because the tangential derivatives of u and ∂u_n/∂n vanish there.
- In transient runs the monitors add ∂u/∂t = (uⁿ⁺¹ − uⁿ)/Δt; steady states omit it.
- Integrals use the edge-midpoint rule on triangles and the solver's 2 × 2 Gauss rule on quadrilaterals.
- If the body shared nodes with another boundary (e.g. a plate attached to a channel wall), the traction on that
  boundary within one element of the shared nodes would be included, an O(h) error. The cylinder is isolated.

The monitors write both forms (`cd`, `cl`: volume; `cd_surface`, `cl_surface`, and the pressure part
`cd_pressure`, `cl_pressure`). Tests in `[post][forces]`:
- Couette flow is exact with both forms on distorted triangle, quadrilateral and hybrid meshes.
- Divergence-theorem identities of the volume form on a closed body hold to round-off.
- The forces rotate with the configuration.

---

## 3. Meshes

`tools/cylinder_benchmark/generate_meshes.py` (Gmsh 4.15.2, Frontal-Delaunay, fixed seed) writes two families with
the same boundary discretisation and the same size field. Per level the sizes halve:

| Level | cylinder segments | size at the cylinder | size in the wake (x < 10) | far-field size | first inflation layer |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 0 | 16 | 0.196 | 0.32 | 0.8 | 0.02 |
| 1 | 32 | 0.098 | 0.16 | 0.4 | 0.01 |
| 2 | 64 | 0.049 | 0.08 | 0.2 | 0.005 |
| 3 | 128 | 0.025 | 0.04 | 0.1 | 0.0025 |

- **tri**: triangles only, graded from the cylinder (Gmsh `Distance` + `Threshold`) with a refined wake box.
- **bl**: the same, plus a quadrilateral inflation layer on the cylinder (Gmsh `BoundaryLayer` field with
  `Quads = 1`): first cell height above, growth ratio 1.2, total thickness 0.12 D. That is 6, 9, 12 layers on
  levels 1, 2, 3 (4 on level 0).

| Mesh | nodes | triangles | quadrilaterals | quadrilateral aspect ratio (max / mean) | max skewness (tri / quad) |
| :--- | ---: | ---: | ---: | :--- | :--- |
| tri_L1 | 1 980 | 3 780 | – | – | 0.42 / – |
| bl_L1 | 2 124 | 3 684 | 192 | 9.95 / 7.01 | 0.45 / 0.06 |
| tri_L2 | 7 597 | 14 838 | – | – | 0.45 / – |
| bl_L2 | 7 944 | 14 380 | 576 | 9.90 / 5.59 | 0.41 / 0.03 |
| tri_L3 | 29 598 | 58 487 | – | – | 0.41 / – |
| bl_L3 | 30 417 | 57 053 | 1 536 | 9.86 / 4.54 | 0.43 / 0.02 |

Every element is in the skewness classes *excellent* or *good*. The first-layer cells are about ten times longer
than high. The inflation layer adds 5–7 % nodes. The level-0 meshes (534 / 585 nodes) are test fixtures
(`tests/data/gmsh/schaefer_turek_*_L0.msh`, `[benchmark]` tests).

---

## 4. Results

Solver settings (`tools/cylinder_benchmark/run_benchmark.py`):
- CBS with θ₁ = 1 and safety factor 0.6 on the time step.
- Pressure: CG + IC(0) with RCM on the assembled matrix, relative tolerance 10⁻⁸.
- 2D-1: local time stepping until the relative velocity change per step is below 10⁻⁹ (at most 10 000 steps).
- 2D-2: global time step, t = 0 … 60 from rest, analysed over the last 4 lift periods.

### 4.1 2D-1, steady, Re = 20

Errors in parentheses are relative to the high-accuracy values.

| Mesh | nodes | c_D (volume form) | c_D (surface traction) | c_L | Δp | steps | c_D change, last 100 steps |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tri_L1 | 1 980 | 5.6672 (+1.57 %) | 5.3783 (−3.61 %) | 0.01783 (+67.9 %) | 2.8792 (−2.00 %) | 400 | 1 · 10⁻⁸ |
| bl_L1 | 2 124 | 5.6568 (+1.38 %) | 5.6400 (+1.08 %) | 0.01117 (+5.2 %) | 2.9528 (+0.50 %) | 600 | 1 · 10⁻⁷ |
| tri_L2 | 7 597 | **5.5856** (+0.11 %) | 5.4376 (−2.54 %) | 0.01182 (+11.4 %) | 2.9126 (−0.87 %) | 1 000 | 4 · 10⁻¹⁰ |
| bl_L2 | 7 944 | **5.5897** (+0.18 %) | **5.5800** (+0.008 %) | 0.01208 (+13.8 %) | 2.9425 (+0.15 %) | 1 200 | 3 · 10⁻⁸ |
| tri_L3 | 29 598 | **5.5796** (+0.0005 %) | 5.5186 (−1.09 %) | **0.01097** (+3.3 %) | 2.9243 (−0.47 %) | 10 000 | 2 · 10⁻¹¹ |
| bl_L3 | 30 417 | **5.5804** (+0.015 %) | **5.5752** (−0.078 %) | **0.01064** (+0.21 %) | **2.9391** (+0.039 %) | 10 000 | 2 · 10⁻⁹ |

Bold: inside the Schäfer–Turek interval. Levels 1 and 2 stopped on the 10⁻⁹ criterion; level 3 ran to the 10 000
step limit, at which c_D changes by less than 2 · 10⁻⁹ per 100 steps. The level-3 runs took 47 (tri) and 58 (bl)
minutes with two runs sharing four cores; levels 1 and 2 take seconds.

- **c_D, volume form.** It converges on both families: +0.0005 % (tri) and +0.015 % (bl) on level 3, inside the
  Schäfer–Turek interval from level 2 on. Being superconvergent, this form hides the near-wall resolution. Both
  families give the same accuracy per node.
- **c_D, surface traction.** This is the direct wall-shear integral, and here the inflation layer matters:
  - with the inflation layer, +1.08 %, +0.008 % and −0.078 % on levels 1–3;
  - with triangles only, −3.6 %, −2.5 % and −1.1 %.

  The triangles approach the reference slowly (first order), because the viscous stress of the wall cells is
  under-resolved. The inflation layer resolves it with about 5 % more nodes.
- **Δp.** The front and rear stagnation pressures converge faster with the inflation layer at every level:
  +0.50 / +0.15 / +0.04 % against −2.00 / −0.87 / −0.47 %.
- **c_L.** The lift is about 500 times smaller than the drag and comes from the small asymmetry of the geometry, so
  it is the most sensitive quantity. On level 3 the inflation layer gives +0.2 %, triangles only +3.3 %. Level 2 is
  about +11–14 % for both.

### 4.2 2D-2, periodic, Re = 100

| Mesh | nodes | c_D,max | c_D mean | c_L,max | St | Δp | time steps | Δt | wall time (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tri_L1 | 1 980 | 3.1661 | 3.1330 | 1.0152 | 0.2900 | 2.5038 | 3 575 | 0.0168 | 16 |
| bl_L1 | 2 124 | 3.1898 | 3.1591 | 1.0502 | 0.2944 | **2.4889** | 20 400 | 0.0029 | 90 |
| tri_L2 | 7 597 | 3.2188 | 3.1845 | 1.0390 | **0.2987** | **2.4971** | 8 015 | 0.0075 | 224 |
| bl_L2 | 7 944 | **3.2222** | 3.1895 | 1.0170 | **0.2995** | **2.4875** | 80 805 | 0.00074 | 3 566 |
| tri_L3 | 29 598 | **3.2274** | 3.1939 | 1.0119 | **0.3009** | **2.4905** | 15 280 | 0.0039 | 4 695 |
| bl_L3 | 30 417 | not run: about 320 000 steps (Δt ≈ 1.9 · 10⁻⁴), about a day on this machine | | | | | | | |
| reference | | [3.22, 3.24] | | [0.99, 1.01] | [0.295, 0.305] | [2.46, 2.50] | | | |

Bold: inside the Schäfer–Turek interval. The period-to-period variation of c_L,max over the analysed periods is at
most 0.006 (0.6 % of c_L,max, on tri_L3), so the flow is periodic. The wall times were measured with four runs sharing four cores
and are indicative only.

**Accuracy.** Both families converge towards the reference intervals. At equal level the inflation layer is
more accurate:
- c_D,max: −1.2 % against −2.0 % of the interval midpoint on level 1; inside the interval against just below it on
  level 2.
- St: 0.2944 against 0.2900 on level 1.
- c_L,max: 1.017 against 1.039 on level 2.

bl_L2, with 7 944 nodes, is about as accurate as tri_L3 with 29 598 nodes. The lift amplitude approaches the interval
from above with refinement (1.039 → 1.012 for triangles). The explicit first-order time stepping (θ₁ = 1) and the
p = 0 outlet (instead of "do nothing") may account for the remaining 0.2 %. This was not investigated further.

**Cost.** The CBS time step is Δt = 0.6 min(h² Re / 2, h / |u|) per element, with h the smallest element height
(`compute_timestep`). Transient runs use the global minimum, so the first layer of the inflation mesh (height
0.01 / 2^(L−1)) sets Δt through the diffusive limit, Δt ∝ (first-layer height)² Re:
- Δt is 5.7× smaller than on the triangle mesh on level 1, 10× smaller on level 2, and would be about 20× smaller
  on level 3.
- At equal accuracy (bl_L2 against tri_L3) the inflation-layer mesh needs 3.7× fewer nodes but 5.3× more steps. The
  wall time is similar (3 566 s against 4 695 s).

Steady runs use local time steps, so the thin cells do not slow the others: the steady runs take about the same
number of steps on both families (table 4.1).

---

## 5. Conclusions and limitations

1. **Validation.** On the Schäfer–Turek benchmark, HydraMa2D converges to the published values on triangle,
   quadrilateral-inflation-layer (hybrid) meshes alike.
   - Steady case 2D-1, level 3: c_D within 0.02 % of the high-accuracy value (volume form).
   - Periodic case 2D-2: Strouhal number, c_D,max and Δp inside the reference intervals from level 2 (inflation
     layer) or level 3 (triangles); c_L,max 0.2–0.7 % above the interval.
2. **Inflation layers improve the wall quantities.** With 3–7 % more nodes, on level 3:
   - the drag from the surface traction (the wall-shear integral) has a 14× smaller error (0.08 % against 1.1 %);
   - the stagnation pressure difference has a 12× smaller error (0.04 % against 0.47 %);
   - the lift has a 16× smaller error (0.2 % against 3.3 %).

   On every level the inflation layer gives the better Δp and surface drag. This is the benefit the quadrilateral
   support was built for: resolving the wall-normal gradients with anisotropic cells.
3. **Integral forces: use the volume form.** It is superconvergent and nearly insensitive to the near-wall
   resolution. Its drag agrees with the reference to 0.1–0.2 % from level 2 on both families. The surface traction
   is the right measure for the wall shear itself, and that is where the inflation layer matters.
4. **Explicit time stepping limits thin cells.** Transient runs take their time step from the thinnest cell
   (Δt ∝ h² Re), so a layer of thin cells multiplies the number of steps. Here that cancels the gain in nodes at
   equal accuracy, and bl_L3 was not run (about 320 000 steps). Steady runs with local time stepping do not pay
   this price.

**Limitations.**
- The outlet is p = 0 with the natural velocity condition, not the benchmark's "do nothing" condition. The effect is
  small, but it is not zero.
- The level-3 inflation-layer mesh was not run for 2D-2.
- Wall times were measured on a shared machine.

**Suggested next step.** Treat the wall-normal viscous term implicitly, or use dual time stepping, so that the
diffusive limit h² Re / 2 no longer sets the global time step. This is what makes inflation layers expensive in
transient runs, and it matters more for turbulent cases, whose low-Re wall treatment needs first cells at y+ ≈ 1.

---

## 6. Reproducing the results

```bash
pip install gmsh
python3 tools/cylinder_benchmark/generate_meshes.py /tmp/cyl/meshes --levels 1 2 3
python3 tools/cylinder_benchmark/run_benchmark.py --bin build/HydraMa2D --meshes /tmp/cyl/meshes \
        --work /tmp/cyl/work --cases steady --jobs 2
python3 tools/cylinder_benchmark/run_benchmark.py --bin build/HydraMa2D --meshes /tmp/cyl/meshes \
        --work /tmp/cyl/work --cases unsteady --only tri_L1 bl_L1 tri_L2 bl_L2 tri_L3 --jobs 2
python3 tools/cylinder_benchmark/analyse.py /tmp/cyl/work/unsteady_tri_L2/results/cylinder.monitors.dat --unsteady
```

- Each run is kept in `<work>/<case>_<mesh>/`, with `run.log`, `results/cylinder.monitors.dat` and the VTK files.
- The summary tables above are printed and written to `<work>/summary.md`.
- `--analyse-only` re-analyses existing runs.
- The numbers in this document come from GCC 13.3, Release, Linux/x86-64, Gmsh 4.15.2. Another Gmsh version may
  produce slightly different meshes.

---

## Sources

- M. Schäfer, S. Turek: Benchmark computations of laminar flow around a cylinder. In: E.H. Hirschel (ed.), *Flow
  Simulation with High-Performance Computers II*, Notes on Numerical Fluid Mechanics 52, Vieweg, 1996, 547–566.
- V. John, G. Matthies: Higher-order finite element discretizations in a benchmark problem for incompressible
  flows. *Int. J. Numer. Meth. Fluids* 37 (2001) 885–903.
- V. John: Reference values for drag and lift of a two-dimensional time-dependent flow around a cylinder.
  *Int. J. Numer. Meth. Fluids* 44 (2004) 777–788.
- I. Babuška, A. Miller: The post-processing approach in the finite element method — Part 1: Calculation of
  displacements, stresses and other higher derivatives of the displacements; Part 2: The calculation of stress
  intensity factors. *Int. J. Numer. Meth. Eng.* 20 (1984) 1085–1109, 1111–1129.
- M. Giles, M. Larson, M. Levenstam, E. Süli: Adaptive error control for finite element approximations of the lift
  and drag coefficients in viscous flow. Oxford University Computing Laboratory, Report NA-97/06, 1997.
- O.C. Zienkiewicz, R.L. Taylor, P. Nithiarasu: *The Finite Element Method for Fluid Dynamics*, 7th ed.,
  Butterworth-Heinemann, 2014 (the CBS scheme and its time step limits).
- C. Geuzaine, J.-F. Remacle: Gmsh: a 3-D finite element mesh generator with built-in pre- and post-processing
  facilities. *Int. J. Numer. Meth. Eng.* 79 (2009) 1309–1331 (`BoundaryLayer` field).
- C.J. Stimpson, C.D. Ernst, P. Knupp, P.P. Pébay, D. Thompson: *The Verdict Geometric Quality Library*, Sandia
  report SAND2007-1751, 2007 (quality metrics).
