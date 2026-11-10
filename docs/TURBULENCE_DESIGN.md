# HydraMa 2D – RANS Turbulence Modelling: Design Specification

Status: **approved design**, implementation in progress (see [§10 Roadmap](#10-roadmap-and-status)).

This document specifies how RANS turbulence modelling is added to the CBS
(Characteristic-Based Split) P1–P1 finite element solver. It records the
decisions taken, the architecture, the mathematics of every model, the
boundary conditions, the coupling to the CBS steps and the verification plan.
The references used for algorithms and model formulations are in
[§11](#11-references).

---

## 1. Decisions

| # | Topic | Decision |
|---|---|---|
| D1 | Turbulent stress form | Molecular part keeps the existing Laplacian form (bit-identical laminar results); the eddy-viscosity part uses the **symmetric stress form** `∫ ∇N_i · ν_t (∇u + ∇uᵀ) dΩ`, correct for spatially varying `ν_t`. `-⅔k δ_ij` is absorbed into a modified pressure. |
| D2 | Configuration | New optional key–value file `system/<case>.turbulence`. If absent the run is laminar and all existing cases/tests are unaffected. |
| D3 | Wall functions | Keep the no-slip velocity condition; impose the wall shear stress through the eddy viscosity of the wall-adjacent elements (`ν_eff · u_p / y_p = u_τ²`). `WallTreatment` is a strategy (low-Re / standard log law / automatic). |
| D4 | Spalart–Allmaras | Standard SA with the **negative-SA** extension (SA-neg) for numerical robustness. |
| D5 | k-ε family | Standard, RNG and Realizable variants (high-Re, wall functions) plus the **Launder–Sharma low-Re** damping functions. |
| D6 | k-ω | **Menter SST** (2003 version). |
| D7 | Laminar path | With no turbulence model the solver calls exactly the existing kernels, so laminar results stay bit-for-bit identical and the existing regression suite keeps guarding them. |
| D8 | Tests | Catch2 v3 (pinned 3.4.0). Offline builds use `-DFETCHCONTENT_SOURCE_DIR_CATCH2=<path>`. |
| D9 | Nodal gradients | Polynomial preserving recovery (PPR) as precomputed nodal stencils, default; lumped projection selectable. Chosen after verification showed lumped projection (and SPR) are first order at boundary nodes and on irregular meshes, which made the SA solution first order (§4.4). |
| D10 | Positivity | Point-implicit (Patankar) source update for all models; additionally, for k, ε, ω the negative explicit transport increment is Patankar-weighted (modified Patankar–Euler), guaranteeing non-negative values for any time step. Not used for SA (SA-neg allows negative ν̃). |
| D11 | Time step with turbulence | Diffusive limit with `ν_eff,e = max(ν + ν_t,e, largest diffusivity of the turbulence equations)`; required for the stability of the explicit turbulence equations (measured, §6). |
| D12 | SST transported variable | `ln ω` instead of ω (§7.3): with ω the explicit convection next to the unresolved near-wall ω ∝ 1/y² profile drove ω to 0 at interior nodes (benchmark, step ~10) and ν_t ∝ 1/ω blew up. |
| D13 | Dissipation update | The ε / ε̃ sink `C₂f₂ε²/k` is solved by the exact backward-Euler (Riccati) step, with the lagged production `C₁(ε/k)P_k = C₁C_μf_μkS²` explicit: positive and bounded by the local equilibrium for any Δt. The linearised form overshot by ×14 in one large local time step and triggered a collapse of ε̃ (Launder–Sharma, steady mode). |
| D14 | Wall-function production | The log-law production `τ_w u_τ/(κy)` replaces `ν_t S²` at every node of an element touching the wall (the no-slip linear velocity in these elements overestimates dU/dy by ~ln(Ey⁺)), the node-based analogue of finite-volume wall-cell production. |

---

## 2. Relevant properties of the existing solver

* Non-dimensional incompressible Navier–Stokes, `ν = ani = 1/Re`, linear
  triangles, lumped mass, explicit CBS steps 1 and 3, semi-implicit pressure
  (step 2) solved by an element-by-element Jacobi-PCG.
* Nodal update pattern `φ_i += rhs_i · elcoe2_i`, with
  `elcoe2_i = 1 / Σ_e (M_e,i / Δt_e)`; it already supports local time stepping.
* Viscosity enters in exactly **two places**: the viscous term of
  `compute_step1` (`ldetJ * ani * 0.5`) and the diffusive limit in
  `compute_timestep` (`csafm · h² · Re / 2`). Steps 2 and 3 are unaffected by
  turbulence.
* P1 ⇒ velocity gradients are element-wise constant; for a linear `ν_t` the
  element mean of nodal values integrates `∫ ν_t ∇N_i·∇N_j dΩ` exactly.
* `BoundarySide::elem` gives the element adjacent to every boundary face, so
  the first-cell height `y_p = 2·Area / L_face` is available at no cost.
* `Mesh::wall_node_norm` zeroes normals at nodes classified as "trailing edge"
  (a documented `[characterization]` behaviour that also hits domain corners);
  turbulence code therefore uses its own wall normals (`WallGeometry`).
* In Galerkin FEM a homogeneous Neumann condition is *natural*: outlets (504)
  and symmetry planes (506) need **no code** for `k, ε, ω, ν̃`.
* `hydrama2d_core` is a static library: self-registering factories (static
  registrar objects) would be dropped by the linker ⇒ explicit factory table.
* The EBE PCG storage (nodal diagonal + 3 off-diagonals per element) can hold
  any symmetric P1 operator, e.g. `M/Δt + θ K_ν`, which enables implicit
  diffusion later.

---

## 3. Architecture

```
include/BoundaryKind.hpp          flag -> {Wall, Inlet, Outlet, Symmetry, Unknown}   (single source of truth)
include/geometry/
    SegmentTree.hpp               AABB tree over 2D segments, exact nearest-segment queries
    WallDistance.hpp              WallDistanceCalculator strategy (brute force, AABB tree, [Poisson])
    WallGeometry.hpp              wall faces, inward normals, y_p, wall nodes, node->face CSR, d_wall
    FieldOps.hpp                  element gradients, S, Ω, lumped projection, NodalGradient (PPR),
                                  FlowKinematics
include/numerics/
    ScalarTransport.hpp           fused N-scalar CBS transport kernel
    PointImplicit.hpp             SourceSplit (P - Dφ) and the point-implicit / Patankar update
    DirichletSet.hpp              compiled Dirichlet conditions (builder with override precedence)
include/turbulence/
    TurbulenceConfig.hpp          .turbulence parser and configuration types
    TurbulenceModel.hpp           Strategy interface, FlowView, TurbulenceContext
    TransportModelBase.hpp        CRTP / Template-Method skeleton for N-equation models
    TurbulenceBC.hpp              flag -> Dirichlet sets for turbulence variables
    SpalartAllmaras.hpp           SA + SA-neg (sa:: closure functions and the model)
    TurbulenceFactory.hpp         explicit factory table
    [later] KEpsilon<Variant,Damping>, KOmegaSST, WallTreatment
```

`Mesh` stays a plain data container; all wall-related data lives in
`WallGeometry`, which only needs raw mesh data (coordinates, connectivity,
boundary sides with flags).

### 3.1 Design patterns

| Pattern | Where | Rationale |
|---|---|---|
| Strategy (virtual) | `TurbulenceModel`, `WallDistanceCalculator`, `WallTreatment` | Called once per time step / once per run ⇒ virtual dispatch cost is irrelevant; runtime selection from the config file. |
| CRTP + Template Method | `TransportModelBase<Derived, N>` | Fixed algorithm skeleton for every transport model; model hooks are inlined into the element/node loops. |
| Policy-based design | `KEpsilon<Variant, Damping>` | Variants (Standard, RNG, Realizable) × damping (None, Launder–Sharma) as small `constexpr` policy structs. |
| Factory with explicit table | `make_turbulence_model(cfg)` | Works with static libraries; adding a model is one line. |
| Compiled boundary conditions | `TurbulenceBC` | Flags are resolved once into `(node, value)` lists; per-step application is a tight loop; walls applied last (same corner precedence as `apply_velocity_bcs`). |
| Named field registry | `TurbulenceModel::fields()` | IO (VTK, restart) without changing IO signatures per model. |

### 3.2 Core interfaces (as implemented)

```cpp
struct FlowView    { const std::vector<double>& u; const std::vector<double>& v; };
struct TurbulenceContext { const Mesh& mesh; const WallGeometry& walls; double nu;
                           const std::vector<double>& delte; const std::vector<double>& elcoe2; };

class TurbulenceModel {                        // Strategy
public:
    virtual ~TurbulenceModel() = default;
    virtual std::string_view name() const = 0;
    virtual void initialize(const TurbulenceContext&, const FlowView&) = 0;
    virtual void advance(const TurbulenceContext&, const FlowView&) = 0;   // once per step, after Step 3
    virtual const std::vector<double>& nut() const = 0;              // nodal
    virtual const std::vector<double>& nut_elem() const = 0;         // Step 1 (element means)
    virtual const std::vector<double>& max_diffusivity_elem() const = 0; // Δt limit (max over equations)
    virtual std::vector<FieldRef> fields() = 0;                      // VTK + restart
    virtual std::vector<EquationResidual> residuals() const = 0;
};

template <class Derived, int N>                // N = 1 (SA), 2 (k-ε, SST)
class TransportModelBase : public TurbulenceModel {
    void advance(const TurbulenceContext& c, const FlowView& f) final {
        kinematics_.compute(mesh, recovery_, u, v);      // element ∇u, S, Ω; recovered nodal ∇u
        self().compute_sources(c);                       // nodal P, D from φ^n
        assemble_scalar_transport<N>(...);               // ONE fused element pass, Γ_e of φ^n
        point_implicit_update(...);                      // per equation, Patankar if kPositive[n]
        bcs_[n].apply(phi_[n]);
        self().compute_eddy_viscosity(c);                // ν_t(φ^{n+1}), element means,
        self().compute_diffusivity(c);                   // Γ_e(φ^{n+1}), max over equations
    }
};
```
Every quantity is computed once per step; the diffusivities computed at the
end of a step are those of `φ^n` at the next step.

---

## 4. Geometry kernels and wall metrics (Phase 1)

### 4.1 `BoundaryKind`

| Flag | Kind |
|---|---|
| 500, 501, 502 | `Wall` |
| 503, 507 | `Inlet` |
| 504 | `Outlet` |
| 506 | `Symmetry` |
| anything else (incl. 505, 0) | `Unknown` |

The classifier reproduces exactly the flag sets hard-wired in
`Mesh::determine_wall_nodes`, `apply_velocity_bcs` and `setup_pressure_bcs`;
consistency tests guard against divergence.

### 4.2 `WallGeometry`

Per wall face (a boundary side whose flag classifies as `Wall`):
node ids, adjacent element, opposite node, length `L`, **inward** unit normal
(oriented towards the opposite node, hence independent of the side's node
order) and `y_p = 2·Area / L` (height of the adjacent element over the face).

Per wall node (ascending node id): length-weighted inward normal
`n = Σ L_f n_f / |Σ L_f n_f|` (bisector at corners, zero only at cusps where
the face normals cancel), length-weighted mean `y_p`, and a CSR node → face map.

Per mesh node: exact wall distance `d` and nearest wall face.

### 4.3 Wall distance (implemented)

`d(p) = min_f dist(p, segment_f)` – Euclidean distance to the piecewise-linear
discrete wall (not to wall *nodes*, which overestimates `d` on coarse walls).

* **Point–segment distance** (Ericson 2005, §5.1.2): `t = clamp((p−a)·(b−a)/|b−a|², 0, 1)`,
  `d² = |p − a − t(b−a)|²`; endpoints are handled by exact branches so wall
  nodes get `d = 0` exactly.
* **AABB tree** (bounding-volume hierarchy; Ericson 2005, ch. 6): top-down
  construction, split at the median centroid along the longest axis of the
  centroid bounds (`std::nth_element`, O(N log N)), leaves of ≤ 4 segments,
  nodes and segments stored contiguously in tree order.
* **Query**: depth-first branch-and-bound nearest-neighbour search
  (Friedman, Bentley & Finkel 1977; Roussopoulos, Kelley & Vincent 1995): a
  subtree is pruned when the squared point–box distance ("MINDIST") exceeds
  the best distance found so far; the nearer child is visited first.
* **Warm start**: nodes are queried in mesh order and the previous node's
  nearest face seeds the upper bound (mesh numbering is spatially coherent).
* **Exactness**: pruning uses a tiny round-off slack (`10⁻¹²` relative to the
  wall extent) and ties are broken by the lowest face index, so the tree
  returns **bit-for-bit the same distance and face as a brute-force scan**.
* Cost: build O(N_w log N_w), query ≈ O(log N_w) per node ⇒ O(N log N_w)
  total, once per run (static mesh). Brute force O(N·N_w) is kept as the
  reference implementation for tests.
* **Poisson distance** (planned, Spalding 1994; Tucker 2003): solve
  `∇²φ = −1`, `φ = 0` on walls, `d = √(|∇φ|² + 2φ) − |∇φ|`; reuses the
  existing Laplacian and PCG; for very large or deforming meshes.
* If the mesh has no walls, `d = +∞` and the nearest face is `−1`; models
  must treat this as "no wall influence".
* The distance is the Euclidean distance to the nearest wall, not the
  geodesic distance inside the fluid (standard definition used by SA/SST).
* Custom strategies can be injected (the SA verification test uses a
  "virtual wall" below the mesh).

### 4.4 Velocity kinematics and gradient recovery (implemented)

* Element gradients `∇u_e` are exact for P1; `S = √(2 S_ij S_ij)`,
  `Ω = |∂v/∂x − ∂u/∂y|` per element.
* Nodal gradients (used by the nodal source terms: Ω, |∇ν̃|², later ∇k·∇ω)
  come from a precomputed sparse nodal stencil `∇φ_i = Σ_j G_ij φ_j`:
  * **PPR** (default, Zhang & Naga 2005): least-squares quadratic fit to the
    nodal values of the one-ring patch (two rings at boundary nodes or for
    ill-conditioned patches), differentiated at the node. Exact for
    quadratics on any mesh ⇒ second order everywhere. Falls back to the
    lumped stencil where no fit exists (tiny meshes).
  * **Lumped projection** of the element gradients (area-weighted average),
    second order only on symmetric patches, first order at boundaries and
    on irregular meshes.
  * SPR (Zienkiewicz & Zhu 1992) was evaluated and rejected: for linear
    triangles its centroid samples are superconvergent only on regular
    patterns, so it is first order at boundaries and on irregular meshes.
* Measured (tests): nodal ∂u/∂y order 2.0 (PPR) vs 1.0 (lumped) at boundary
  nodes and on perturbed meshes; the exact SA log-layer solution converges
  at order 1.96 with PPR and 0.91 with the lumped projection.
* Magnitudes S, Ω are computed from the recovered gradient (never by
  recovering magnitudes), so Ω = 0 on a channel centre line.

---

## 5. Numerics shared by all transport models (Phase 2, implemented)

* **Fused scalar transport kernel** – CBS characteristic-Galerkin for `N`
  scalars in one element pass: convection, streamline term `Δt/2 (ū·∇N_i)(ū·∇φ)`
  and variable diffusion `∫ (ν + ν_t/σ) ∇N_i·∇φ`. With `φ = u` and constant
  diffusivity it must reproduce `compute_step1` (test oracle).
* **Point-implicit (Patankar) source update**, nodal, with lumped mass `M_i`
  and nodal time step `Δt_i = M_i elcoe2_i`:
  `φ_i ← (φ_i + elcoe2_i rhs_i + Δt_i P_i) / (1 + Δt_i D_i)` with the source
  split `Q = P − D·φ`, `D ≥ 0` (`SourceSplit::add_linear`: decay implicit,
  growth explicit; `add_value`: sinks Patankar-weighted). Destruction can
  never change the sign of φ.
* **Positivity** (k, ε, ω): if the explicit increment `E = elcoe2 rhs + Δt P`
  is negative, `φ^{n+1} = φ^n / (1 + Δt D − E/φ^n)` (modified Patankar–Euler,
  Burchard et al. 2003) ⇒ `φ^{n+1} ≥ 0` for any Δt. Non-conservative only
  where it is active.
* Velocity kinematics once per step (§4.4).
* Second derivatives (Launder–Sharma `E` term) are zero element-wise for P1:
  they are obtained by applying the recovery stencil to the recovered nodal
  velocity gradient.
* Kernel verification: identical to `compute_step1` for φ = u, v; fused N = 2, 3
  bit-identical to separate passes; manufactured steady diffusion–reaction
  with variable diffusivity converges at second order.

---

## 6. Coupling to the CBS scheme (Phase 3, implemented)

Per time step: `Δt(ν_eff^n)` → Step 1 (`ν_t^n`) → Step 2 → Step 3 → velocity
BCs → `turbulence->advance()` with `u^{n+1}` → `ν_t^{n+1}` (element means).

* **Step 1** – `compute_step1(mesh, ani, nut_elem, ...)`: one element kernel
  templated on `kTurbulent`; the laminar instantiation is the original code.
  The turbulent stress is added after the molecular term in symmetric form
  `rhs_u_i −= A_e ν_t,e [∂N_i/∂x 2∂u/∂x + ∂N_i/∂y (∂u/∂y + ∂v/∂x)]` (and
  analogously for v); elements with `ν_t,e == 0` are skipped, so a zero or
  empty `nut_elem` reproduces the laminar kernel bit for bit. Verified:
  tensor reference, objectivity (rigid rotation ⇒ no stress), transpose
  term (`∂/∂x(ν_t ∂u/∂y)` in the v-momentum), momentum conservation and the
  exact discrete dissipation `Σ u·rhs = −Σ_e A_e ν_t,e S_e²`.
* **Time step** – `compute_timestep(..., nu_eff_elem, ...)` with the per-element
  diffusive limit `csafm h_e² / (2 ν_eff,e)`; bit-identical to the legacy
  overload for `ν_eff ≡ 1/Re`. The solver uses
  `ν_eff,e = max(ν + ν_t,e, max_diffusivity_elem_e)`, i.e. also the
  diffusivity of the turbulence equations (SA: `(ν + ν̃)/σ`). This is
  required for stability: with `ν + ν_t` alone the explicit ν̃ diffusion
  diverged on the benchmark in steady mode within 3 iterations (ν_t/ν hit the
  10⁵ cap). Consequence: in the SA free stream (`ν̃ = 3ν`) the diffusive limit
  is 6× more restrictive than laminar (benchmark: 10 transient steps reach
  t = 0.0149 instead of 0.0513).
* **Solver** – `system/<case>.turbulence` is read in `initialize()`; the
  model comes from the factory (or `Solver::set_turbulence_model()`,
  dependency injection for tests / user models); `WallGeometry` uses the
  configured wall-distance strategy. `turbulence_ == nullptr` ⇒ the original
  laminar code path (also for `model none`).
* **Logging** – per logged step: `Turbulence: <var> change <relative change>
  nut/nu max <…> y+ (first cell) min/mean/max`; steady runs additionally
  require the turbulence residuals to be below the velocity tolerance.
* **Output** – VTK point scalars appended after `P`: the model fields
  (`nu_tilde`, `nut`), `wall_distance`, `yplus` (`d u_τ(nearest face)/ν`,
  `u_τ = √(ν |t·(∇u) n|)` from the wall element); laminar VTK output is
  unchanged.
* **Restart** – `results/<case>.turb.var`:
  ```
  HydraMa2D-turbulence-restart 1
  model spalart-allmaras
  npoin 8956
  fields 1 nu_tilde
  <17 significant digits per value, one line per node>
  ```
  Only the transported variables (`state_fields()`) are stored; derived
  fields are recomputed by `refresh()`. A missing or mismatched file (model,
  field names, node count, format) gives a warning and free-stream
  turbulence, so a turbulent run can start from a laminar solution.
  Restarted runs are bit-identical to uninterrupted ones.
* **Laminar bit-identity** (verified): `.var`, VTK and logs of laminar runs
  are byte-identical to the pre-turbulence binary (transient, steady/local
  Δt, fixed Δt, converged PCG, restart); in the test suite `model none` ≡ no
  file, and a zero-eddy-viscosity model driven through the complete
  turbulent code path reproduces u, v, p and the PCG history bit for bit.
* **Observation (SA, steady, local Δt)**: small negative ν̃ (−0.3 ν) appear
  near the channel walls in the first iterations and vanish by iteration 50
  – the SA-neg branch works as intended.
* **Known limitation (pre-existing)**: `solve_pcg` does not detect non-finite
  residuals; a diverging run spends up to 100000 PCG iterations per step.
  Divergence detection is recommended (Phase 8).

## 7. Models

Notation: `S` strain-rate magnitude, `Ω` vorticity magnitude, `d` wall distance,
`ν` molecular viscosity (`1/Re`). Constants are the published values
(see references); every formula will be unit-tested against hand-computed values.

### 7.1 Spalart–Allmaras with negative-SA extension (SA-neg)
Spalart & Allmaras (1994); Allmaras, Johnson & Spalart (2012).

`ν_t = ν̃ f_v1` (ν̃ ≥ 0), `ν_t = 0` (ν̃ < 0); `χ = ν̃/ν`, `f_v1 = χ³/(χ³ + c_v1³)`.

ν̃ ≥ 0:
```
Dν̃/Dt = c_b1 (1 − f_t2) S̃ ν̃ − (c_w1 f_w − c_b1 f_t2/κ²)(ν̃/d)²
        + (1/σ)[∇·((ν + ν̃)∇ν̃) + c_b2 |∇ν̃|²]
```
`S̄ = ν̃ f_v2/(κ² d²)`, `f_v2 = 1 − χ/(1 + χ f_v1)`;
`S̃ = Ω + S̄` if `S̄ ≥ −c_v2 Ω`, else `S̃ = Ω + Ω(c_v2² Ω + c_v3 S̄)/((c_v3 − 2c_v2)Ω − S̄)`;
`r = min(ν̃/(S̃ κ² d²), 10)`, `g = r + c_w2(r⁶ − r)`, `f_w = g[(1 + c_w3⁶)/(g⁶ + c_w3⁶)]^{1/6}`,
`f_t2 = c_t3 exp(−c_t4 χ²)` (an `ft2 off` option gives SA-noft2).

ν̃ < 0:
```
Dν̃/Dt = c_b1 (1 − c_t3) Ω ν̃ + c_w1 (ν̃/d)²
        + (1/σ)[∇·((ν + ν̃ f_n)∇ν̃) + c_b2 |∇ν̃|²],   f_n = (c_n1 + χ³)/(c_n1 − χ³)
```
Constants: `c_b1 = 0.1355, σ = 2/3, c_b2 = 0.622, κ = 0.41, c_w1 = c_b1/κ² + (1 + c_b2)/σ,
c_w2 = 0.3, c_w3 = 2, c_v1 = 7.1, c_v2 = 0.7, c_v3 = 0.9, c_t3 = 1.2, c_t4 = 0.5, c_n1 = 16`.
The `c_b2|∇ν̃|²` term is evaluated at the nodes from the recovered gradient
(≥ 0 by construction). Sources are evaluated at the nodes (recovered Ω, exact
`d`); positive-branch growth terms are explicit, decay terms implicit; all
negative-branch terms are decay terms towards 0 and therefore implicit.
Diffusivity: element mean of the nodal `(ν + f_n ν̃)/σ` (> 0 always).
BCs: `ν̃ = 0` at walls, `ν̃ = r ν` at inlets (`inlet_nu_tilde_ratio`, else
converted from `inlet_viscosity_ratio` by inverting `χ f_v1(χ)`, else 3),
natural elsewhere; `sa_ft2 off` gives SA-noft2-neg. SA requires
`wall_treatment low-re`.

**Verification** (tests): closure functions against values from an independent
evaluation; log-layer source balance; linearisation consistency (`D ≥ 0`);
exact production ODE in homogeneous shear (with/without `f_t2`); free-stream
preservation; monotone relaxation of negative ν̃ with ν_t = 0; **exact steady
log-layer solution ν̃ = κu_τ(y + y₀) of the full discretisation, second-order
convergence**; smoke runs on the benchmark mesh (Re 200: bounded by the
inflow value, Re 10⁵: growth in the wall shear layers).
Cost: 2.6 ms per step on the 8956-node benchmark (≈ 20 % of a laminar CBS step).

### 7.2 k-ε family: `KEpsilon<Variant, Damping>`
```
Dk/Dt = P_k − ε − D + ∇·[(ν + ν_t/σ_k)∇k]
Dε̃/Dt = C_1 f_1 (ε̃/k) P_k − C_2* f_2 ε̃²/k + E + ∇·[(ν + ν_t/σ_ε)∇ε̃]
ν_t = C_μ f_μ k²/ε̃,   P_k = ν_t S²
```

| Variant | `C_μ` | `C_1` | `C_2` | `σ_k` | `σ_ε` | Notes |
|---|---|---|---|---|---|---|
| Standard (Launder & Spalding 1974) | 0.09 | 1.44 | 1.92 | 1.0 | 1.3 | |
| RNG (Yakhot et al. 1992) | 0.0845 | 1.42 | 1.68 | 0.7194 | 0.7194 | `C_2* = C_2 + C_μ η³(1 − η/η₀)/(1 + βη³)`, `η = Sk/ε`, `η₀ = 4.38`, `β = 0.012` |
| Realizable (Shih et al. 1995) | variable | `max(0.43, η/(η+5))` | 1.9 | 1.0 | 1.2 | ε-equation `C_1 S ε − C_2 ε²/(k + √(νε))`; `C_μ = 1/(A_0 + A_s k U*/ε)`, `A_0 = 4.04`, `A_s = √6 cos φ`, `φ = ⅓ arccos(√6 W)` |

2D note (Realizable): for a traceless symmetric 2×2 strain tensor
`S_ij S_jk S_ki = 0`, hence `W = 0`, `φ = π/6` and `A_s = 3/√2` exactly.

Damping policies:

| Damping | `f_μ` | `f_1` | `f_2` | `D` | `E` | wall BC |
|---|---|---|---|---|---|---|
| None (high-Re, wall functions) | 1 | 1 | 1 | 0 | 0 | wall function (§8) |
| Launder–Sharma (1974) | `exp(−3.4/(1 + R_t/50)²)` | 1 | `1 − 0.3 exp(−R_t²)` | `2ν(∂√k/∂x_j)²` | `2νν_t(∂²U_i/∂x_j∂x_k)²` | `k = 0`, `ε̃ = 0` |

`R_t = k²/(ν ε̃)`; Launder–Sharma uses the Standard constants.

### 7.3 Menter k-ω SST (2003)
Menter (1994); Menter, Kuntz & Langtry (2003).
```
Dk/Dt = P̃_k − β* k ω + ∇·[(ν + σ_k ν_t)∇k],            P̃_k = min(ν_t S², 10 β* k ω)
Dω/Dt = γ S² − β ω² + ∇·[(ν + σ_ω ν_t)∇ω] + 2(1 − F_1) σ_ω2 (1/ω) ∇k·∇ω
ν_t = a_1 k / max(a_1 ω, S F_2)
```
`F_1 = tanh(arg_1⁴)`, `arg_1 = min(max(√k/(β* ω d), 500ν/(d² ω)), 4 σ_ω2 k/(CD_kω d²))`,
`CD_kω = max(2 σ_ω2 ∇k·∇ω / ω, 10⁻¹⁰)`, `F_2 = tanh(arg_2²)`, `arg_2 = max(2√k/(β* ω d), 500ν/(d² ω))`.
Blending `φ = F_1 φ_1 + (1 − F_1) φ_2`; constants `β* = 0.09, a_1 = 0.31`;
set 1: `σ_k1 = 0.85, σ_ω1 = 0.5, β_1 = 0.075, γ_1 = 5/9`;
set 2: `σ_k2 = 1.0, σ_ω2 = 0.856, β_2 = 0.0828, γ_2 = 0.44`.
The exact placement of the production limiter in the ω equation follows the
NASA TMR "SST-2003" definition and is verified in Phase 5.

Safety for all models: tiny positive floors in all ratios (no clipping of the
stored fields), positivity-preserving updates and a cap on `ν_t/ν`
(`nut_ratio_max`).

### 7.4 Two-equation models as implemented (Phases 5–6)

**Menter SST** (`turbulence/KOmegaSST`, model name `k-omega-sst`, low-Re walls):
* transported `k` and `w̃ = ln ω` (D12), state/restart fields `k`, `log_omega`;
  output `k`, `omega`, `nut`, `sst_f1`;
* `Dw̃/Dt = γS²e^−w̃ − βe^w̃ + ∇·(Γ_ω∇w̃) + Γ_ω|∇w̃|² + 2(1−F₁)σ_ω2 ∇k·∇w̃ e^−w̃`
  (identical to the ω equation); exponential terms Newton-linearised
  (`SourceSplit::add_newton`); `CD_kω = max(2σ_ω2 ∇k·∇w̃, 10⁻¹⁰)`;
* F₁, F₂, ν_t from recovered (PPR) nodal gradients of k and w̃ and the exact
  wall distance; F₁ = F₂ = 1 at walls, 0 without walls;
* walls: `k = 0`, `ω = 60ν/(β₁y₁²)` with y₁ the length-weighted first-cell
  height at each wall node; inlets from I/ℓ, ν_t/ν or direct `k`/`omega`;
* production limiter `P̃_k = min(ν_tS², c_lim β*kω)`, `c_lim` = `production_limiter` (10).

**k-ε family** (`turbulence/KEpsilon`, `KEpsilon<Variant, Damping>`):
* `k-epsilon` = `KEpsilon<Standard, NoDamping>` with log-law wall functions
  (`turbulence/WallFunctions`, κ = 0.41, E = 9.8, y⁺_lam = 11.53):
  ε at wall nodes `C_μ^{3/4}k^{3/2}/(κy_p)` (state-dependent BC), ν_t of the
  wall-adjacent elements `ν(y⁺κ/ln(Ey⁺) − 1)` (y⁺ from k), k natural at
  walls, log-law production in the wall-function layer (D14);
* `launder-sharma-k-epsilon` = `KEpsilon<Standard, LaunderSharma>`:
  transported ε̃ (`epsilon_tilde`), f_μ, f₂, `D = 2ν|∇√k|²` (recovered
  gradient of √k), `E = 2νν_t|∇∇u|²` (PPR applied to the recovered velocity
  gradient), walls `k = ε̃ = 0`;
* source limiters: production limiter `P_k ≤ c_lim ε`; Patankar positivity for
  k and ε; exact Riccati update of the ε sink (D13).

**Verification** (unit tests):
* closure functions (F₁, F₂, shear stress limiter, f_μ, f₂, wall functions)
  against an independent Python evaluation; limits and monotonicity;
* linearisations reproduce the sources (`value(φ) = source`, `D ≥ 0`);
* decay of homogeneous isotropic turbulence: standard k-ε
  `k = k₀(1 + (C₂−1)ε₀t/k₀)^(−1/(C₂−1))` and SST `ω = ω₀/(1 + βω₀t)`,
  `k = k₀(1 + βω₀t)^(−β*/β)` with β = β₂ (no walls, F₁ = 0) and β₁ (F₁ = 1);
  model ≡ discrete recurrence (10⁻¹⁰), first-order convergence in Δt;
  Launder–Sharma at high R_t → standard solution, at low R_t (f₂ → 0.7)
  against an RK4 reference;
* exact log-layer solutions (virtual wall): SST `k = u_τ²/√β*`,
  `ω = u_τ/(√β* κ_ω d)` with `κ_ω² = (β₁/β* − γ₁)√β*/σ_ω1`; k-ε
  `k = u_τ²/√C_μ`, `ε = u_τ³/(κ_ε d)` with `κ_ε² = (C₂−C₁)σ_ε√C_μ`: k second
  order, ω/ε order ≈ 1.7 (limited by the natural inflow side boundary of
  the test set-up, measured on 10/20/40/80 cells);
* wall functions reproduce the log-law equilibrium exactly (wall shear
  `u_τ²`, production = dissipation at the wall-function layer);
* benchmark: 10 transient and 30 steady steps per model, VTK fields,
  bit-for-bit restart, reference values; 1000 steady / 2000 transient steps
  without divergence (manual runs).

Observation: on the laminar (Re = 200) benchmark the high-Re k-ε with wall
functions (y⁺ ≈ 4, outside its range) gives ν_t/ν up to ~40 and its steady
iteration does not settle (bounded oscillation); SST, Launder–Sharma and SA
converge with ν_t/ν < 2.

---

## 8. Boundary conditions and wall treatment

| Boundary | ν̃ (SA) | k | ε | ω |
|---|---|---|---|---|
| Inlet (503/507) | `ν̃/ν = 3` (or from `ν_t/ν`) | `1.5 (U I)²` | `C_μ^{3/4} k^{3/2}/ℓ` or `C_μ k²/(r ν)` | `√k/(C_μ^{1/4} ℓ)` or `k/(r ν)` |
| Wall, resolved (low-Re) | 0 | 0 | LS: `ε̃ = 0` | `60 ν/(β_1 y_1²)` |
| Wall, wall function | – | natural (zero flux) | `C_μ^{3/4} k_p^{3/2}/(κ y_p)` | automatic `√(ω_vis² + ω_log²)` |
| Outlet 504 / Symmetry 506 | natural | natural | natural | natural |

`I` turbulence intensity, `ℓ` length scale, `r = ν_t/ν`, `y_1` first-node
wall distance. The BC type is derived from `BoundaryKind`; per-flag overrides
in the config file. No new flag codes.

**WallTreatment strategy** (D3):
* `LowRe` – Dirichlet values above, no modification of `ν_t`.
* `StandardLog` – `u⁺ = ln(E y⁺)/κ` (κ = 0.41, E = 9.8), `u_τ = C_μ^{1/4} √k_p`,
  `ν_t,w = ν (y⁺ κ / ln(E y⁺) − 1)` for `y⁺ > y⁺_lam`, else 0.
* `Automatic` – Spalding's single formula
  `y⁺ = u⁺ + e^{−κB}[e^{κu⁺} − 1 − κu⁺ − (κu⁺)²/2 − (κu⁺)³/6]` (Newton
  inversion) and `ω_vis = 6ν/(β_1 y²)`, `ω_log = u_τ/(√β* κ y)`.

`y_p`, the opposite (sampling) node and the inward normal come from
`WallGeometry`.

### 8.1 Configuration file `system/<case>.turbulence`
```
model            k-omega-sst        # none | spalart-allmaras | k-epsilon | rng-k-epsilon |
                                    # realizable-k-epsilon | launder-sharma-k-epsilon | k-omega-sst
wall_treatment   automatic          # low-re | wall-function | automatic
wall_distance    exact              # exact | brute-force   (poisson planned)
gradient_recovery ppr               # ppr | lumped-projection
inlet_k / inlet_epsilon / inlet_omega   # direct inlet values (optional)
reference_velocity 1                # U_ref in k = 1.5 (U_ref I)^2
production_limiter 10               # P_k <= c eps (k-epsilon), c beta* k omega (SST); 0 = off
inlet_intensity  0.05               # I
inlet_length_scale 0.1              # l
inlet_viscosity_ratio 10            # nu_t/nu (optional)
inlet_nu_tilde_ratio 3              # SA (optional; default 3)
sa_ft2           on                 # SA trip term (off = SA-noft2)
nut_ratio_max    1e5                # cap on nu_t/nu
boundary 503 inlet intensity 0.03 length_scale 0.05 viscosity_ratio 5 nu_tilde_ratio 3
boundary 507 inlet k 1e-3 epsilon 2e-4   # direct values (also omega)
boundary 506 symmetry               # kind override for the turbulence equations only
# wall_treatment: default per model (wall-function for k-epsilon, low-re otherwise)
```
Implemented in `turbulence/TurbulenceConfig` (`IO::read_turbulence`): `#`
comments, CRLF, case-insensitive enumeration values, Fortran `d` exponents;
unknown / duplicate keys and invalid values are errors reported with the line
number; a missing file means laminar.

---

## 9. Efficiency and verification

Efficiency: virtual calls only per step; templates/CRTP in inner loops; one
fused element pass for all transported scalars; one velocity-gradient pass
shared by all models; persistent workspaces (no per-step allocation);
one-off precomputation of wall distance, BC node lists, nodal masses and wall
metrics; gather–compute–scatter kernels ready for element colouring / OpenMP.
Target overhead of a two-equation model: < 30 % per time step (baseline
≈ 13 ms/step on the 8956-node benchmark). Future: implicit diffusion using the
generalised EBE PCG (near-wall `y⁺ ≈ 1` meshes make the explicit viscous limit
restrictive; steady RANS uses local time stepping meanwhile).

Verification (Catch2 tags):

| Tag | Content |
|---|---|
| `[boundary_kind]` | flag table, agreement with legacy flag handling |
| `[wallgeom]` | faces, inward normals, `y_p`, corner bisectors, node→face map, orientation independence |
| `[walldist]` | analytic channel / backward-facing step / benchmark plate distances, tree ≡ brute force (bitwise), Lipschitz & invariance properties, pruning efficiency |
| `[turbulence_config]` | parser: defaults, full file, overrides, 23 error cases with line numbers, file IO, factory |
| `[field_ops]` `[recovery]` | S/Ω of elementary flows, frame invariance, exact linear gradients, projection, PPR exact for quadratics and second order everywhere |
| `[transport]` | equivalence with `compute_step1`, fused ≡ separate, constant preservation, conservation, variable diffusivity, MMS order 2 |
| `[point_implicit]` `[positivity]` | implicit-Euler decay, local Δt, exact Patankar values, positivity for any Δt |
| `[sa]` | closure functions, sources at reference points, linearisation, log-layer balance, BCs, shear ODE, SA-neg relaxation, exact log-layer solution (order 2), benchmark smoke runs |
| `[sst]` `[blending]` `[limiter]` | F1/F2, shear stress limiter, sources and limiter, ln ω linearisation, BCs, decay (β₁/β₂), exact log layer, near-wall ω regression |
| `[kepsilon]` `[damping]` `[wall_function]` | f_μ/f₂, D/E terms, sources and limiter, Riccati update, decay (analytic / RK4), exact log layer, wall-function formulas and log-law consistency |
| `[decay]` | decaying homogeneous turbulence: `k = k_0(1 + (C_2 − 1)ε_0 t/k_0)^{−1/(C_2−1)}` |
| `[positivity]` | `k, ε, ω` positive at large Δt |
| `[turb_bc]` | inlet formulas, wall ω, log-law / Spalding inversion |
| `[coupling]` | Step 1 / Δt overloads: laminar bit-identity, symmetric stress, objectivity, conservation, dissipation |
| `[wall_units]` `[turbulence_io]` | u_τ / y⁺ (Couette), VTK point scalars, turbulence restart format |
| `[solver][turbulence]` | laminar bit-identity (model none, zero-ν_t model), SA benchmark: health, VTK fields, reference values, determinism, bit-for-bit restart, restart from laminar, mismatched restarts, steady mode, configuration errors |

Validation (documented runs): NASA TMR zero-pressure-gradient flat plate
(SA, SST), Driver–Seegmiller backward-facing step (reattachment length);
fully developed channel (Moser, Kim & Mansour 1999) once periodic BCs exist.

---

## 10. Roadmap and status

| Phase | Content | Status |
|---|---|---|
| 0 | Catch2 offline configuration, baseline verification (205 tests) | ✅ done |
| 0 | `BoundaryKind` classifier, Makefile/CMake support for sub-directories | ✅ done |
| 0 | `.turbulence` config reader | ✅ done |
| 0 | named-field VTK output, turbulence restart file | ✅ done (Phase 3) |
| 1 | `WallGeometry`, exact wall distance (AABB tree + brute-force reference), tests | ✅ done |
| 1 | `FieldOps`: gradients, S/Ω, lumped projection, PPR gradient recovery | ✅ done |
| 1 | Poisson wall distance | ⏳ open |
| 2 | Fused scalar transport, point-implicit / Patankar update, compiled scalar BCs | ✅ done |
| 3 | Step 1 / Δt coupling, solver hooks, VTK fields, turbulence restart, laminar bit-identity | ✅ done |
| 4 | SA-neg model (sources, BCs, factory) + verification | ✅ done |
| 4 | SA end-to-end benchmark regression | ✅ done |
| 4 | SA flat-plate validation (needs a flat-plate mesh) | ⏳ |
| 5 | Menter SST (wall-resolved, ln ω) | ✅ done |
| 6 | Standard k-ε + log-law wall functions, Launder–Sharma low-Re k-ε | ✅ done |
| 6 | RNG / Realizable k-ε policies, automatic wall treatment (SST / Spalding) | ⏳ open |
| 7 | Validation runs, documentation, README | ⏳ |
| 8 | Optional: implicit diffusion, OpenMP colouring, periodic BCs, DES, energy equation, PCG divergence detection | ⏳ |

---

## 11. References

Geometry / search algorithms
1. C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2005 –
   §5.1.2 closest point on a segment, §5.1.3 point–AABB distance, ch. 6
   bounding volume hierarchies (top-down median-split construction).
2. J. H. Friedman, J. L. Bentley, R. A. Finkel, "An algorithm for finding best
   matches in logarithmic expected time", *ACM Trans. Math. Software* 3(3),
   209–226, 1977 – branch-and-bound nearest-neighbour search in trees.
3. N. Roussopoulos, S. Kelley, F. Vincent, "Nearest neighbor queries",
   *Proc. ACM SIGMOD*, 71–79, 1995 – MINDIST pruning for bounding-box trees.
4. D. B. Spalding, "Calculation of turbulent heat transfer in cluttered
   spaces", *Proc. 10th Int. Heat Transfer Conference*, Brighton, 1994 –
   Poisson-based wall distance.
5. P. G. Tucker, "Differential equation-based wall distance computation for
   DES and RANS", *J. Comput. Phys.* 190, 229–248, 2003.

Turbulence models
6. P. R. Spalart, S. R. Allmaras, "A one-equation turbulence model for
   aerodynamic flows", *La Recherche Aérospatiale* 1, 5–21, 1994.
7. S. R. Allmaras, F. T. Johnson, P. R. Spalart, "Modifications and
   clarifications for the implementation of the Spalart–Allmaras turbulence
   model", ICCFD7-1902, 2012 – negative SA, `S̃` limiting.
8. B. E. Launder, D. B. Spalding, "The numerical computation of turbulent
   flows", *Comput. Methods Appl. Mech. Eng.* 3(2), 269–289, 1974.
9. B. E. Launder, B. I. Sharma, "Application of the energy-dissipation model
   of turbulence to the calculation of flow near a spinning disc",
   *Letters in Heat and Mass Transfer* 1(2), 131–137, 1974.
10. V. Yakhot, S. A. Orszag, S. Thangam, T. B. Gatski, C. G. Speziale,
    "Development of turbulence models for shear flows by a double expansion
    technique", *Phys. Fluids A* 4(7), 1510–1520, 1992.
11. T.-H. Shih, W. W. Liou, A. Shabbir, Z. Yang, J. Zhu, "A new k-ε eddy
    viscosity model for high Reynolds number turbulent flows",
    *Computers & Fluids* 24(3), 227–238, 1995.
12. F. R. Menter, "Two-equation eddy-viscosity turbulence models for
    engineering applications", *AIAA J.* 32(8), 1598–1605, 1994.
13. F. R. Menter, M. Kuntz, R. Langtry, "Ten years of industrial experience
    with the SST turbulence model", *Turbulence, Heat and Mass Transfer 4*,
    Begell House, 625–632, 2003.
14. D. C. Wilcox, *Turbulence Modeling for CFD*, 3rd ed., DCW Industries, 2006.
15. NASA Langley Turbulence Modeling Resource, https://turbmodels.larc.nasa.gov
    (reference definitions of SA, SA-neg, SST-2003 and verification cases).

Wall treatment / numerics / validation
16. D. B. Spalding, "A single formula for the law of the wall",
    *J. Appl. Mech.* 28(3), 455–458, 1961.
17. F. R. Menter, T. Esch, "Elements of industrial heat transfer
    predictions", 16th Brazilian Congress of Mechanical Engineering (COBEM),
    2001 – automatic near-wall treatment for ω.
18. G. Kalitzin, G. Medic, G. Iaccarino, P. Durbin, "Near-wall behavior of
    RANS turbulence models and implications for wall functions",
    *J. Comput. Phys.* 204, 265–291, 2005.
19. S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere,
    1980 – source-term linearisation (positivity-preserving point-implicit update).
20. O. C. Zienkiewicz, R. L. Taylor, P. Nithiarasu, *The Finite Element
    Method for Fluid Dynamics*, 7th ed., Butterworth-Heinemann, 2014 – CBS scheme.
21. R. D. Moser, J. Kim, N. N. Mansour, "Direct numerical simulation of
    turbulent channel flow up to Re_τ = 590", *Phys. Fluids* 11(4), 943–945, 1999.
22. D. M. Driver, H. L. Seegmiller, "Features of a reattaching turbulent shear
    layer in divergent channel flow", *AIAA J.* 23(2), 163–171, 1985.

Numerics added in Phase 2
23. Z. Zhang, A. Naga, "A new finite element gradient recovery method:
    superconvergence property", *SIAM J. Sci. Comput.* 26(4), 1192–1213, 2005 –
    polynomial preserving recovery (PPR).
24. O. C. Zienkiewicz, J. Z. Zhu, "The superconvergent patch recovery and a
    posteriori error estimates. Part 1: The recovery technique",
    *Int. J. Numer. Methods Eng.* 33, 1331–1364, 1992 – SPR (evaluated, not used).
25. H. Burchard, E. Deleersnijder, A. Meister, "A high-order conservative
    Patankar-type discretisation for stiff systems of production–destruction
    equations", *Appl. Numer. Math.* 47, 1–30, 2003 – (modified) Patankar–Euler.

Two-equation models (Phases 5–6)
26. D. Ilinca, D. Pelletier, "Positivity preservation and adaptive solution
    for the k-ε model of turbulence", *AIAA J.* 36(1), 44–50, 1998 –
    logarithmic turbulence variables in finite elements.
27. F. Bassi, A. Crivellini, S. Rebay, M. Savini, "Discontinuous Galerkin
    solution of the Reynolds-averaged Navier–Stokes and k–ω turbulence model
    equations", *Computers & Fluids* 34, 507–540, 2005 – k–ln(ω) formulation.
28. S. B. Pope, *Turbulent Flows*, Cambridge University Press, 2000 – decaying
    homogeneous turbulence in the k-ε model, log-layer consistency of the
    model constants.
29. OpenFOAM Foundation, `nutkWallFunction` / `epsilonWallFunction`
    boundary conditions (source documentation) – cell-based form of the
    log-law wall functions used here node-wise.

Bibliographic details were compiled without internet access; page numbers and
section numbers should be spot-checked before being quoted in publications.
