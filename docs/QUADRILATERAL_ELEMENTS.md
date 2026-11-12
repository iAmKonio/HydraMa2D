# Quadrilateral Elements

HydraMa2D supports three kinds of mesh:
- meshes of linear triangles (P1);
- meshes of bilinear four-node quadrilaterals (Q1);
- **mixed (hybrid) meshes** of both, typically quadrilateral inflation layers along walls with triangles elsewhere.

This document defines the Gauss–Legendre quadrature, the bilinear element (shape functions and metric derivatives)
and how every term of the solver is evaluated on it. Sources are listed in section 10.

| Code | Content |
| :--- | :--- |
| `include/numerics/Quadrature.hpp`, `src/numerics/Quadrature.cpp` | Gauss–Legendre rules (section 2) |
| `include/numerics/QuadElement.hpp`, `src/numerics/QuadElement.cpp` | Q1 shape functions, Jacobian, element data (sections 3–5) |
| `include/Mesh.hpp`, `src/Mesh.cpp` | Element numbering, per-element geometry (sections 1, 5) |
| `src/Step1.cpp`, `src/Step2.cpp`, `src/Step3.cpp`, `src/numerics/ScalarTransport.cpp` | CBS terms on quadrilaterals (section 6) |
| `src/LinearSolver.cpp`, `src/linalg/EbeOperator.cpp`, `src/linalg/CsrAssembly.cpp` | Pressure operator (section 6.4) |
| `src/mesh/MeshBuilder.cpp`, `src/mesh/MeshCheck.cpp` | Orientation and validity at import (section 4.3) |

---

## 1. Meshes with quadrilaterals

**Element numbering.**
- Elements are numbered **triangles first, then quadrilaterals**: element `e < num_triangles()` is triangle
  `intma[e]`, and element `e >= num_triangles()` is quadrilateral `quads[e - num_triangles()]`.
- `nelem` counts both, so per-element arrays (time steps, eddy viscosity, gradients, `detJ`, `alen_e`) share one
  index space.
- `Mesh::element_nodes(e)` gives the 3 or 4 nodes of any element.

A mesh of triangles only is exactly the original mesh. Every triangle computation keeps its original expressions
and operation order, and the bit-for-bit regression tests guard this.

**Connectivity.**
- Quadrilaterals are stored counter-clockwise, with positive Jacobian determinant everywhere (section 4.3).
- Their edges are straight, so a quadrilateral and a triangle sharing an edge interpolate identically along it. Mixed
  meshes are therefore conforming (C⁰).

**Where meshes come from.**
- Gmsh: element type 3, e.g. `Recombine`, transfinite surfaces, or `BoundaryLayer` fields with `Quads = 1`.
- SU2: VTK type 9.
- CGNS: `QUAD_4`, also inside `MIXED` sections.

See [MESH_IMPORT.md](MESH_IMPORT.md). VTK output writes quadrilaterals as cell type 9 (`VTK_QUAD`).

---

## 2. Gauss–Legendre quadrature

### 2.1 The rule

The n-point Gauss–Legendre rule on [−1, 1] is

```
∫_{-1}^{1} f(x) dx  ≈  Σ_{i=1}^{n} w_i f(x_i).
```

- The nodes x_i are the n roots of the Legendre polynomial P_n.
- The weights are w_i = 2 / ((1 − x_i²) P_n'(x_i)²) (A&S 25.4.29).
- The rule is **exact for every polynomial of degree ≤ 2n − 1**, and no n-point rule is exact for higher degrees.
- For f ∈ C^{2n} the error is (A&S 25.4.29; Davis & Rabinowitz 1984):

  ```
  ∫ f − Q_n(f) = 2^{2n+1} (n!)^4 / ((2n + 1) ((2n)!)^3) · f^{(2n)}(ξ),   ξ ∈ (−1, 1).
  ```

Closed forms (A&S Table 25.4):

| n | ± nodes x_i | weights w_i |
| :---: | :--- | :--- |
| 1 | 0 | 2 |
| 2 | 1/√3 = 0.577350269189626 | 1 |
| 3 | 0; √(3/5) | 8/9; 5/9 |
| 4 | √(3/7 − (2/7)√(6/5)); √(3/7 + (2/7)√(6/5)) | (18 + √30)/36; (18 − √30)/36 |
| 5 | 0; ⅓√(5 − 2√(10/7)); ⅓√(5 + 2√(10/7)) | 128/225; (322 + 13√70)/900; (322 − 13√70)/900 |

### 2.2 Computation

`gauss_legendre(n)` (1 ≤ n ≤ 64) computes the positive half of the nodes by Newton's method on P_n, then mirrors
them (Press et al. 2007, §4.6):

- **Initial guess:** x_i ≈ cos(π (i − ¼) / (n + ½)), i = 1, …, ⌈n/2⌉.
- **Recurrence:** P_n(x) and P_{n−1}(x) come from (k + 1) P_{k+1} = (2k + 1) x P_k − k P_{k−1}, with P_0 = 1 and
  P_1 = x (A&S 22.7.10).
- **Derivative:** P_n'(x) = n (x P_n − P_{n−1}) / (x² − 1) (A&S 22.8.5).
- **Stopping:** iterate until |Δx| ≤ 10⁻¹⁶ max(1, |x|), then evaluate the weight formula at the root.
- The middle node of odd n is set to exactly 0.

The eigenvalue method of Golub & Welsch (1969) is an equivalent alternative.

**Tests** (`tests/test_quadrature.cpp`):
- agreement with the closed forms above (4·10⁻¹⁶) and with A&S Table 25.4 (n = 2, 3, 4, 5, 6, 8);
- symmetric, increasing nodes and positive weights summing to 2 for every n ≤ 64;
- exactness up to degree 2n − 1 for n ≤ 20;
- the exact error term above for degree 2n (n ≤ 8), which shows the degree bound is sharp;
- spectral convergence for eˣ;
- the tensor rule.

### 2.3 Rules on the square

`gauss_legendre_square(n)` is the tensor product: n² points (x_i, x_j) with weights w_i w_j, exact for every
ξ^a η^b with a, b ≤ 2n − 1.

The quadrilateral element uses the **2 × 2 rule**: points (±1/√3, ±1/√3), all weights 1. Its points are ordered
like the element nodes, counter-clockwise:

```
g0 = (−s, −s),  g1 = (s, −s),  g2 = (s, s),  g3 = (−s, s),     s = 1/√3.
```

2 × 2 is the standard *full* integration order for bilinear elements (Hughes 2000, §3.8; Zienkiewicz, Taylor & Zhu
2013, chapter "Shape functions, derivatives, and integration"). Section 5 lists what it integrates exactly.

---

## 3. The bilinear quadrilateral: shape functions

Reference square [−1, 1]², nodes counter-clockwise:

```
  3 (−1, 1) ─────── 2 (1, 1)
      │                 │            N_a(ξ, η) = ¼ (1 + ξ_a ξ)(1 + η_a η),   a = 0, 1, 2, 3
      │                 │
  0 (−1,−1) ─────── 1 (1,−1)         (ξ_a, η_a) = (−1,−1), (1,−1), (1,1), (−1,1)
```

Derivatives:

```
∂N_a/∂ξ = ¼ ξ_a (1 + η_a η),      ∂N_a/∂η = ¼ η_a (1 + ξ_a ξ).
```

Properties (tested in `tests/test_quad_element.cpp`):
- **Kronecker:** N_a(ξ_b, η_b) = δ_ab.
- **Partition of unity:** Σ_a N_a = 1, so Σ_a ∂N_a/∂ξ = Σ_a ∂N_a/∂η = 0.
- **Span:** the space is spanned by {1, ξ, η, ξη}: Σ_a N_a ξ_a = ξ, Σ_a N_a η_a = η, Σ_a N_a ξ_a η_a = ξη.
- **Linear on edges:** on each edge the shape functions are linear (the two nodes of the edge only), which is what
  makes mixed meshes conforming.
- N_a ≥ 0 on the element, so the row-sum lumped mass (section 5.2) is positive.

---

## 4. Metric derivatives

### 4.1 Isoparametric map and Jacobian

The same shape functions map the reference square onto the element with nodes (x_a, y_a):

```
x(ξ, η) = Σ_a N_a(ξ, η) x_a,         y(ξ, η) = Σ_a N_a(ξ, η) y_a.
```

Jacobian matrix and determinant:

```
J = ⎡ x_ξ  x_η ⎤ = ⎡ Σ ∂N_a/∂ξ x_a   Σ ∂N_a/∂η x_a ⎤ ,      det J = x_ξ y_η − x_η y_ξ,
    ⎣ y_ξ  y_η ⎦   ⎣ Σ ∂N_a/∂ξ y_a   Σ ∂N_a/∂η y_a ⎦
```

Special cases:
- **Rectangle h_x × h_y:** J = diag(h_x/2, h_y/2), det J = h_x h_y / 4.
- **Parallelogram:** x = c + A (ξ, η)ᵀ, so J = A is constant.

### 4.2 Physical derivatives (chain rule)

From ∂N/∂ξ = x_ξ ∂N/∂x + y_ξ ∂N/∂y and ∂N/∂η = x_η ∂N/∂x + y_η ∂N/∂y, i.e. ∇_ξ N = Jᵀ ∇_x N:

```
∂N_a/∂x = ( y_η ∂N_a/∂ξ − y_ξ ∂N_a/∂η) / det J,
∂N_a/∂y = (−x_η ∂N_a/∂ξ + x_ξ ∂N_a/∂η) / det J.
```

Integrals transform with dΩ = det J dξ dη.

The element is isoparametric, so **linear functions are reproduced exactly** on any (distorted) element:
Σ_a ∇N_a (c + p x_a + q y_a) = (p, q). This is the basis of the patch test (section 8).

### 4.3 det J is affine; the corner criterion

Write x(ξ, η) = a_0 + a_1 ξ + a_2 η + a_3 ξη (and likewise y with b_k). Then x_ξ = a_1 + a_3 η,
x_η = a_2 + a_3 ξ, y_ξ = b_1 + b_3 η, y_η = b_2 + b_3 ξ, and

```
det J = (a_1 + a_3 η)(b_2 + b_3 ξ) − (a_2 + a_3 ξ)(b_1 + b_3 η)
      = (a_1 b_2 − a_2 b_1) + (a_1 b_3 − a_3 b_1) ξ + (a_3 b_2 − a_2 b_3) η.
```

The ξη terms cancel: **det J is an affine function of (ξ, η)**. Consequences:
1. **det J > 0 on the whole element ⟺ det J > 0 at the four corners,** because the minimum of an affine function
   on the square is at a corner.
2. **At a corner** det J equals a quarter of the cross product of the two edges there:

   ```
   det J(corner a) = ¼ (x_{a+1} − x_a) × (x_{a−1} − x_a).
   ```

   At node 0, J = ½ [x_1 − x_0, x_3 − x_0].
3. **Corners and interior:** det J is the bilinear interpolation of its corner values, with c_0 − c_1 + c_2 − c_3 = 0.
   Also ∫ det J dξdη = 4 det J(0, 0) = area.

All positive corner values mean the quadrilateral is **convex and counter-clockwise**. Hence (`q1::corner_jacobians`,
`build_mesh`):

| Corner values | Element | Import |
| :--- | :--- | :--- |
| all > tol | valid | kept |
| all < −tol | clockwise | reversed (a, d, c, b), first node kept; counted in the report |
| mixed signs | non-convex (reflex corner) or self-intersecting (bow-tie) | error |
| some \|·\| ≤ tol | degenerate: corner angle 0° or 180°, or coincident nodes | error |

Here tol = ¼ · 10⁻¹² · (longest edge)². Checking det J only at the Gauss points is **not** enough: a quadrilateral
with a slightly reflex corner can have det J > 0 at all four Gauss points. The tests include such an element.

---

## 5. Element quantities (2 × 2 Gauss–Legendre)

Computed once per mesh in `q1::compute_geometry` and stored in `Mesh::quad_geom` with, at each Gauss point g:
- the physical weight W_g = w_g det J(g);
- the physical shape function gradients ∇N_a(g) (section 4.2).

The shape function values N_a(g) are the same for every element (`q1::gauss_shape`).

### 5.1 What 2 × 2 integrates exactly

On a general quadrilateral det J is affine and ∇N_a det J = adj(J)ᵀ ∇_ξ N_a is bilinear. Hence:

| Integral | Integrand (in ξ, η) | 2 × 2 exact |
| :--- | :--- | :--- |
| area ∫ 1 dΩ | det J: degree 1 | always |
| lumped mass ∫ N_a dΩ | N_a det J: ≤ 2 per variable | always |
| ∫ ∇N_a dΩ | adj(J)ᵀ ∇_ξ N_a: ≤ 1 per variable | always |
| Laplacian ∫ ∇N_a · ∇N_b dΩ | rational unless J is constant | parallelograms |
| consistent mass, convection ∫ N_a u ∂u/∂x dΩ | ≤ 3 per variable for parallelograms | parallelograms |

On non-affine quadrilaterals the rational integrands are approximated to second order, as is standard for full
integration. The constant-gradient and exact-area properties are what make the patch tests pass on *any*
quadrilateral (section 8).

### 5.2 Quantities

- **Area:** A = Σ_g W_g (exact). `Mesh::detJ[e]` holds 2A for every element, the original meaning for triangles.
- **Row-sum lumped mass:** M_a = ∫ N_a dΩ = Σ_g W_g N_a(g) > 0, with Σ_a M_a = A (Hinton, Rock & Zienkiewicz
  1976). A parallelogram gives A/4 per node. The triangle equivalent is A/3 (`elcoe_e`).
- **Area-averaged gradients:** Ḡ_a = (1/A) ∫ ∇N_a dΩ = (1/A) ∮ N_a n ds, stored in `Mesh::dNkdx[e]`.
  - For a linear field φ, Σ_a Ḡ_a φ_a = ∇φ exactly.
  - These are the "element gradients" of the turbulence models (strain, vorticity, wall shear), replacing the
    constant triangle gradient.
- **Laplacian:** K_ab = Σ_g W_g ∇N_a(g) · ∇N_b(g), stored as 4 diagonal entries and 6 off-diagonal pairs
  (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) (`QuadLaplacian`, `q1::kPairs`).
  - Constants are in its null space and it is positive semi-definite.
  - All four nodes are coupled, including the diagonal pairs (0,2) and (1,3).
  - For a rectangle h_x × h_y (node 0 at the origin):

    ```
    K = h_y/(6 h_x) ⎡ 2 −2 −1  1⎤ + h_x/(6 h_y) ⎡ 2  1 −1 −2⎤       unit square: K = (1/6) ⎡ 4 −1 −2 −1⎤
                    ⎢−2  2  1 −1⎥               ⎢ 1  2 −2 −1⎥                               ⎢−1  4 −1 −2⎥
                    ⎢−1  1  2 −2⎥               ⎢−1 −2  2  1⎥                               ⎢−2 −1  4 −1⎥
                    ⎣ 1 −1 −2  2⎦               ⎣−2 −1  1  2⎦                               ⎣−1 −2 −1  4⎦
    ```

- **Length scale for the time step** (`alen_e`): min over edges of A / L_edge = A / L_max.
  - A rectangle gives min(h_x, h_y); a parallelogram gives its smaller height.
  - **Stability:** with lumped mass, the largest eigenvalue of M_L⁻¹K on a uniform Q1 grid of spacing h is 4/h²,
    reached by the mode (−1)^i. Forward-Euler diffusion is therefore stable for Δt ≤ h²/(2ν). This is exactly the
    existing limit csafm·h²/(2ν) with h = min(h_x, h_y).
  - The triangle rule (minimum height) gives the matching bound h²/(4ν) for right-angled triangles of legs h.
- **Height above a wall edge** (first-cell height y_p, k-ω SST wall values, wall functions):
  - triangles: 2A/L (the height of the opposite vertex);
  - quadrilaterals: the mean distance of the two off-edge nodes from the edge (the cell height of an inflation
    layer).

---

## 6. The CBS scheme on quadrilaterals

Each term is the **same weak form** on both element types. Triangles evaluate it in closed form, exactly as before.
Quadrilaterals use the 2 × 2 rule with u_h = Σ N_a u_a, ∇u_h = Σ ∇N_a u_a at each Gauss point. The CBS
characteristic terms use an element velocity ū:
- triangles: the mean of the three nodal values, which is the value at the centroid;
- quadrilaterals: the mean of the four nodal values, which is the value at (ξ, η) = (0, 0).

References: Zienkiewicz & Codina 1995; Zienkiewicz, Taylor & Nithiarasu 2014, ch. 3.

### 6.1 Step 1 (intermediate velocity), `compute_step1`

```
R_a = − ∫ N_a (u_h·∇) u_h dΩ                       convection (consistent)
      − Δt/2 ∫ (ū·∇N_a) (ū·∇) u_h dΩ               characteristic-Galerkin stabilisation
      − ν ∫ ∇N_a · ∇u_h dΩ                          viscous
      − ∫ ν_t ∇N_a · (∇u_h + ∇u_hᵀ) dΩ              turbulent stress (if a model is active)
```

- **Triangles:** ∫N_a N_b dΩ = A(1+δ_ab)/12 (the `detJ/24` factor), with ∇N and ū constant.
- **Quadrilaterals:** Σ_g W_g [...].

### 6.2 Step 2 (pressure equation right-hand side), `compute_step2`

```
R_a = − ∫ N_a ∇·uⁿ dΩ  +  θ₁ ∫ ∇N_a · (u* − uⁿ) dΩ.
```

### 6.3 Step 3 (velocity correction), `compute_step3`

```
R_a = − ∫ N_a ∇pⁿ⁺¹ dΩ  −  Δt/2 ∫ (ū·∇N_a) ∇pⁿ dΩ.
```

### 6.4 Mass matrix and pressure operator

- **Lumped mass:** the nodal inverse mass `elcoe2` sums M_a/Δt_e over the elements: A/3 for triangles, the row sums
  of section 5.2 for quadrilaterals.
- **Pressure operator:** Δt_e K_ab (section 5.2) in element-by-element storage, blocks of three off-diagonal
  values per block:
  - triangles: 1 block, pairs (0,1), (0,2), (1,2);
  - quadrilaterals: 2 consecutive blocks, pairs (0,1), (0,2), (0,3) | (1,2), (1,3), (2,3).

  `Mesh::num_offdiag_blocks()` = n_tri + 2 n_quad. `EbeOperator`, `P1CsrAssembler` (which also gets
  `add_quad_element_matrix`), `PressureSolver` and the IC(0)/ILU(0)/SSOR preconditioners use this layout. The
  assembled and the element-by-element operator agree to rounding.

### 6.5 Scalar transport (turbulence), `assemble_scalar_transport`

Step 1 for a scalar φ with diffusivity Γ_e: convection, stabilisation with ū, and −∫ Γ ∇N_a · ∇φ dΩ.

### 6.6 Field operations and walls

- **Lumped nodal mass, element → node projection:** M_a per element.
- **Element means:** the value at the element centre.
- **Element gradients:** the area averages of section 5.2.
- **Lumped-projection gradient recovery:** G_ij = Σ_e ∫ N_i ∇N_j dΩ / M_i (2 × 2).
- **PPR recovery:** uses the element node patches of both element types.
- **Wall faces on quadrilaterals:** the inward normal points to the mean of the two off-face nodes, y_p is their mean
  distance, and `WallFace::opposite_node` is −1.

---

## 7. Hybrid meshes in practice

- **Inflation layers:** quadrilateral layers along walls resolve boundary layers with few, high-aspect-ratio cells.
  The first-cell height y_p is the layer thickness, and the wall faces belong to quadrilaterals.
- **Gmsh:**

  ```
  Field[1] = BoundaryLayer; Field[1].CurvesList = {5, 6, 7, 8};
  Field[1].Size = 0.01; Field[1].Ratio = 1.4; Field[1].Thickness = 0.06; Field[1].Quads = 1;
  BoundaryLayer Field = 1;
  ```

  `tests/data/gmsh/cylinder_inflation_4.1_binary.msh` is such a mesh: 318 triangles and 48 quadrilaterals.
- **Validity:** all element types are checked at import. The report lists reoriented (clockwise) triangles and
  quadrilaterals separately.

---

## 8. Verification

| Test file | What is verified |
| :--- | :--- |
| `test_quadrature.cpp` | Section 2: closed forms, A&S tables, exactness degree and error term, tensor rule |
| `test_quad_element.cpp` | Sections 3–5: Kronecker / partition of unity / span; derivatives vs finite differences; Jacobian of rectangles, parallelograms and random distorted quadrilaterals vs finite differences of the map; det J affine (= bilinear interpolation of the corner values, c₀ − c₁ + c₂ − c₃ = 0, area = 4 det J(0,0) = shoelace area); chain rule; linear completeness; corner criterion for clockwise, reflex, bow-tie, degenerate and collapsed elements; exact area, lumped mass (vs 5 × 5 Gauss) and mean gradients (vs boundary integrals); the Laplacian of rectangles in closed form, null space, positivity, and exactness for parallelograms (vs 6 × 6 Gauss) |
| `test_quad_patch.cpp` | **Patch tests** on meshes with singular (irregular, valence ≠ 4) vertices: the MacNeal & Harder (1985) 5-element patch (valence 3), a pentagon of kites (valence 5), a triangle split into three quadrilaterals (valence 3), a hybrid pentagon of quadrilaterals and triangles (valence 7), and perturbed quadrilateral and inflation-layer grids |
| `test_quad_mesh.cpp` | Element numbering, geometry (2A, min(h_x, h_y), masses, heights), `update_lhs` blocks, time steps, wall geometry on quadrilaterals, VTK type 9, importer orientation / validity / duplicates / side orientation, all Gmsh format variants |
| `test_quad_solver.cpp` | Fully developed channel flow on quadrilateral, triangle and inflation-layer meshes; the Gmsh inflation mesh with its boundary conditions; determinism; all four turbulence models on the hybrid mesh |

What the patch tests check on each patch (Irons & Razzaque 1972):
- **Laplacian:** K φ = 0 at interior nodes for linear φ, and a Dirichlet problem with linear boundary data
  reproduces φ exactly.
- **Step 2:** gives −(∇·u) M_i for linear u.
- **Step 3:** gives −∇p M_i for linear p, and the stabilisation vanishes at interior nodes.
- **Step 1:** vanishes for uniform flows; its viscous and eddy-viscous parts vanish for linear velocities.
- **Scalar transport:** equals Step 1.
- **Gradients:** element and recovered gradients are exact for linear fields (PPR also for quadratics).
- **Operators:** the CSR and EBE operators agree.
- **Import:** the importer (clockwise input, sparse tags) gives the same operators.

**Convergence.** −Δu = 2π² sin πx sin πy on smoothly distorted meshes of the unit square (u = 0 on the boundary,
maximum nodal error, N = 8, 16, 32):

| Mesh | Observed order, 8→16 | Observed order, 16→32 |
| :--- | :---: | :---: |
| quadrilaterals | 2.02 | 2.00 |
| inflation layers + triangles | 2.00 | 2.00 |
| triangles | 1.99 | 1.99 |

**Channel flow** (Re = 20, 6 × 1, 48 × 8 cells, steady).
- **Expected outlet profile:** the walls win at the inlet corners, so the discrete inflow flux is Q = 1 − h
  (h = 1/8). The fully developed nodal profile carrying it is u = 6y(1 − y)/(1 + h).
- **Measured maximum deviation:** 0.011 on quadrilaterals, 0.008 on the inflation-layer mesh, 0.007 on triangles.
- **Outlet flux:** within 1% of Q on all three meshes.

---

## 9. Limitations

- **Element order:** quadrilaterals are bilinear (Q1) with straight edges. Higher-order and curved elements are
  rejected at import.
- **Non-affine quadrilaterals:** the Laplacian and the convection term are integrated with the full 2 × 2 rule, which
  is exact for parallelograms and second-order accurate otherwise. Strongly distorted quadrilaterals (corner angles
  near 0° or 180°) are valid but reduce accuracy. Only degenerate and non-convex ones are rejected.
- **Conforming meshes only:** hanging nodes (a node in the middle of an edge of the neighbouring element) are not
  supported; the import reports the open edges around them.
- **Wall geometry:** for a wall face on a quadrilateral, the inward normal distance y_p is the mean distance of the two
  off-face nodes, and `opposite_node` is −1 (there is no single opposite node).

---

## 10. Sources

**Gauss–Legendre quadrature**
1. M. Abramowitz, I. A. Stegun (eds.), *Handbook of Mathematical Functions*, NBS Applied Mathematics Series 55
   (1964).
   - §25.4.29: the Gauss–Legendre formula, weights and error term;
   - Table 25.4: abscissas and weights, used in the tests;
   - §22.7.10: Legendre recurrence;
   - §22.8.5: derivative relation.
2. W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery, *Numerical Recipes: The Art of Scientific
   Computing*, 3rd ed., Cambridge University Press (2007), §4.6 "Gaussian Quadratures and Orthogonal Polynomials"
   (Newton iteration and initial guess).
3. G. H. Golub, J. H. Welsch, "Calculation of Gauss quadrature rules", *Mathematics of Computation* 23 (1969)
   221–230 (eigenvalue method; background).
4. P. J. Davis, P. Rabinowitz, *Methods of Numerical Integration*, 2nd ed., Academic Press (1984): Gauss rules and
   their error.

**Bilinear element, isoparametric mapping, numerical integration**
5. T. J. R. Hughes, *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*, Dover (2000):
   §3.2 (bilinear quadrilateral, Jacobian, positivity) and §3.8 (Gaussian quadrature and integration order).
6. O. C. Zienkiewicz, R. L. Taylor, J. Z. Zhu, *The Finite Element Method: Its Basis and Fundamentals*, 7th ed.,
   Butterworth-Heinemann (2013): chapter "Shape functions, derivatives, and integration" (isoparametric elements,
   numerical integration), and the discussion of the patch test.
7. G. Strang, G. J. Fix, *An Analysis of the Finite Element Method*, Prentice-Hall (1973): patch test, and the effect
   of numerical integration on convergence.

**Patch tests**
8. B. M. Irons, A. Razzaque, "Experience with the patch test for convergence of finite elements", in A. K. Aziz
   (ed.), *The Mathematical Foundations of the Finite Element Method with Applications to Partial Differential
   Equations*, Academic Press (1972) 557–587.
9. R. H. MacNeal, R. L. Harder, "A proposed standard set of problems to test finite element accuracy", *Finite
   Elements in Analysis and Design* 1 (1985) 3–20. Source of the patch geometry: rectangle 0.24 × 0.12 with interior
   nodes (0.04, 0.02), (0.18, 0.03), (0.16, 0.08), (0.08, 0.08).

**Mass lumping**
10. E. Hinton, T. Rock, O. C. Zienkiewicz, "A note on mass lumping and related processes in the finite element
    method", *Earthquake Engineering & Structural Dynamics* 4 (1976) 245–249.

**CBS scheme**
11. O. C. Zienkiewicz, R. Codina, "A general algorithm for compressible and incompressible flow — Part I. The split,
    characteristic-based scheme", *International Journal for Numerical Methods in Fluids* 20 (1995) 869–885.
12. O. C. Zienkiewicz, R. L. Taylor, P. Nithiarasu, *The Finite Element Method for Fluid Dynamics*, 7th ed.,
    Butterworth-Heinemann (2014), ch. 3 (the characteristic-based split algorithm).

**Meshes and output**
13. C. Geuzaine, J.-F. Remacle, "Gmsh: a 3-D finite element mesh generator with built-in pre- and post-processing
    facilities", *International Journal for Numerical Methods in Engineering* 79 (2009) 1309–1331 (the
    `BoundaryLayer` field for the inflation-layer fixture).
14. Kitware, *VTK File Formats* (VTK User's Guide): cell types `VTK_TRIANGLE` = 5, `VTK_QUAD` = 9.
