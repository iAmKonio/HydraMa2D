# HydraMa 2D (C++)

**HydraMa 2D** is an unstructured finite element CFD solver for 2D incompressible fluid flows. It uses the **Characteristic-Based Split (CBS)** scheme on linear 3-node triangular meshes ($P_1 - P_1$).

---

## Features

- **Characteristic-Based Split (CBS) Scheme:**
  - **Step 1:** Explicit intermediate velocity calculation with convective streamline stabilisation and viscous diffusion.
  - **Step 2:** Semi-implicit pressure Poisson solve enforcing mass conservation and avoiding pressure-velocity decoupling without violating the LBB condition.
  - **Step 3:** Explicit velocity correction with pressure gradient and CBS pressure stabilization.
- **Element-By-Element (EBE) PCG Solver:** Matrix-free Jacobi Preconditioned Conjugate Gradient solver requiring only $O(N_{\text{elem}} + N_{\text{poin}})$ memory with no global sparse matrix assembly.
- **Extensible linear solver layer (`hydrama::linalg`):** CG, BiCGSTAB and restarted GMRES(m); matrix-free (EBE) or assembled (CSR) operators; Jacobi / SSOR / IC(0) / ILU(0) preconditioners with reverse Cuthill-McKee reordering, all selectable in `<case>.solver` (see [Linear Solver Architecture](#linear-solver-architecture)).
- **Adaptive Timestepping:** Automatic computation of local and global time steps constrained by convective CFL and diffusive stability limits.
- **Case Structure:** OpenFOAM-style directory structure (`constant/`, `system/`, `results/`).
- **RANS Turbulence Modelling (optional):** Spalart–Allmaras (negative-SA), Menter k-ω SST, standard k-ε with log-law wall functions and Launder–Sharma low-Re k-ε, enabled by `system/<case>.turbulence` (see [Turbulence Modelling](#turbulence-modelling) and `docs/TURBULENCE_DESIGN.md`).
- **Standard Outputs:**
  - ASCII VTK unstructured grids (`results/<case>_<time>.vtk`) for visualization in ParaView; turbulent runs add the turbulence fields, the wall distance and y+.
  - Checkpoint / restart data files (`results/<case>.var`, and `results/<case>.turb.var` for turbulence), written with 17 significant digits so that a restarted run continues bit-for-bit.

---

## Directory Structure

```
HydraMa.dev/
├── CMakeLists.txt                # CMake build configuration
├── Makefile                      # Make build configuration
├── README.md                     # Documentation
├── docs/
│   ├── LINEAR_SOLVER_REFERENCES.md  # Literature behind the linear solver algorithms
│   ├── REVIEWER_GUIDE.md            # Build, test, sanitizer and benchmark instructions
│   └── TURBULENCE_DESIGN.md         # RANS turbulence modelling design specification
├── tools/
│   └── benchmark.sh              # Reproducible benchmark of the solver configurations
├── include/                      # C++ Header files
│   ├── BoundaryConditions.hpp
│   ├── BoundaryKind.hpp          # Flag code -> wall / inlet / outlet / symmetry
│   ├── Config.hpp
│   ├── IO.hpp
│   ├── LinearSolver.hpp          # EBE assembly/mat-vec, legacy solve_pcg() adapter, time step
│   ├── Mesh.hpp
│   ├── PressureSolver.hpp        # Facade: pressure Poisson solve used by the time loop
│   ├── geometry/                 # Wall geometry, wall distance, field operations
│   │   ├── FieldOps.hpp          # Gradients, S, Omega, projection, PPR gradient recovery
│   │   ├── SegmentTree.hpp       # AABB tree, exact nearest-segment queries
│   │   ├── WallDistance.hpp      # Wall distance strategies (AABB tree, brute force)
│   │   └── WallGeometry.hpp      # Wall faces, inward normals, y_p, wall distance
│   ├── numerics/                 # Shared transport numerics
│   │   ├── DirichletSet.hpp      # Compiled Dirichlet conditions
│   │   ├── PointImplicit.hpp     # Point-implicit / Patankar source update
│   │   └── ScalarTransport.hpp   # Fused CBS scalar transport kernel
│   ├── turbulence/               # RANS turbulence models
│   │   ├── KEpsilon.hpp          # k-epsilon family (standard + wall functions, Launder-Sharma)
│   │   ├── KOmegaSST.hpp         # Menter k-omega SST (ln omega)
│   │   ├── SpalartAllmaras.hpp   # SA with negative-SA extension
│   │   ├── TransportModelBase.hpp # CRTP skeleton of transport-equation models
│   │   ├── TurbulenceBC.hpp      # Turbulence boundary conditions
│   │   ├── TurbulenceConfig.hpp  # system/<case>.turbulence parser
│   │   ├── TurbulenceFactory.hpp # Model factory
│   │   ├── TurbulenceModel.hpp   # Model interface
│   │   ├── WallFunctions.hpp     # Log-law wall functions
│   │   └── WallUnits.hpp         # Wall shear stress, friction velocity, y+
│   ├── linalg/                   # Linear algebra layer (namespace hydrama::linalg)
│   │   ├── Linalg.hpp            # Umbrella header
│   │   ├── VectorView.hpp        # Non-owning vector views
│   │   ├── VectorOps.hpp         # Deterministic vector kernels
│   │   ├── LinearOperator.hpp    # Operator interface
│   │   ├── EbeOperator.hpp       # Matrix-free element-by-element operator
│   │   ├── SparsityPattern.hpp   # Shared CSR structure
│   │   ├── CsrMatrix.hpp         # Assembled CSR operator, constrain()
│   │   ├── CsrAssembly.hpp       # P1 element-to-CSR assembly (pattern + scatter map)
│   │   ├── Permutation.hpp       # Permutations, symmetric permutation P A P^T
│   │   ├── Reordering.hpp        # (Reverse) Cuthill-McKee
│   │   ├── ConstrainedOperator.hpp  # Dirichlet decorator
│   │   ├── DirichletConstraints.hpp
│   │   ├── Preconditioner.hpp    # Preconditioner interface
│   │   ├── IdentityPreconditioner.hpp
│   │   ├── JacobiPreconditioner.hpp
│   │   ├── SsorPreconditioner.hpp
│   │   ├── IncompleteCholesky.hpp   # IC(0)
│   │   ├── IncompleteLU.hpp         # ILU(0)
│   │   ├── DiagonalShift.hpp        # Breakdown shift policy of IC(0)/ILU(0)
│   │   ├── ReorderedPreconditioner.hpp  # Preconditioner in another numbering
│   │   ├── ScaledPreconditioner.hpp     # (1/c) M^{-1}: setup reuse for scaled operators
│   │   ├── Factory.hpp           # Name-based creation (catalogues)
│   │   ├── KrylovSolver.hpp      # Krylov base class (common solve driver)
│   │   ├── ConjugateGradient.hpp
│   │   ├── BiCGStab.hpp          # BiCGSTAB (non-symmetric systems)
│   │   ├── Gmres.hpp             # Restarted GMRES(m), Arnoldi + Givens QR
│   │   ├── Convergence.hpp       # Criteria, stop reasons, results, convergence control
│   │   └── IterationMonitor.hpp  # Solve observers
│   ├── Solver.hpp
│   ├── Step1.hpp
│   ├── Step2.hpp
│   └── Step3.hpp
├── src/                          # C++ Source files
│   ├── BoundaryConditions.cpp
│   ├── IO.cpp
│   ├── LinearSolver.cpp
│   ├── Mesh.cpp
│   ├── PressureSolver.cpp
│   ├── geometry/
│   ├── numerics/
│   ├── turbulence/
│   ├── linalg/                   # Implementations of include/linalg/*
│   ├── Solver.cpp
│   ├── Step1.cpp
│   ├── Step2.cpp
│   ├── Step3.cpp
│   └── main.cpp
├── tests/                        # Catch2 unit tests (CMake/CTest only)
│   ├── CMakeLists.txt
│   ├── MeshTestUtils.hpp         # Test mesh builders (triangles, structured grids)
│   ├── LinearSolverTestUtils.hpp # Dense-matrix helpers (reference assembly, direct solve)
│   ├── LinalgTestUtils.hpp       # Test doubles for the linalg interfaces (dense/counting operators, ...)
│   ├── reference/LegacyPcg.hpp   # Frozen copy of the original PCG (bit-for-bit reference)
│   ├── IoTestUtils.hpp           # Temp dirs, stdout capture, case copy, VTK parser
│   ├── test_mesh_area.cpp
│   ├── test_mesh_shape_functions.cpp
│   ├── test_mesh_mass_matrix.cpp
│   ├── test_mesh_face_normals.cpp
│   ├── test_mesh_element_sizes.cpp
│   ├── test_mesh_laplacian.cpp
│   ├── test_mesh_wall_nodes.cpp
│   ├── test_mesh_hinterkante.cpp # Geometry checks on the bundled benchmark mesh
│   ├── test_timestep.cpp
│   ├── test_linear_assembly.cpp
│   ├── test_pcg.cpp
│   ├── test_pressure_bc.cpp
│   ├── test_step1.cpp
│   ├── test_step2.cpp
│   ├── test_step3.cpp
│   ├── test_velocity_bc.cpp
│   ├── test_io.cpp
│   ├── test_solver_regression.cpp # End-to-end runs of the benchmark case
│   ├── test_linalg_vector.cpp
│   ├── test_linalg_interfaces.cpp
│   ├── test_linalg_convergence.cpp
│   ├── test_linalg_monitor.cpp
│   ├── test_linalg_cg.cpp
│   ├── test_linalg_csr.cpp
│   ├── test_linalg_assembly.cpp
│   ├── test_linalg_reordering.cpp
│   ├── test_linalg_assembled_preconditioners.cpp
│   ├── test_linalg_factory.cpp
│   ├── test_solver_config.cpp
│   ├── test_linalg_bicgstab.cpp
│   ├── test_linalg_gmres.cpp
│   ├── test_linalg_nonsymmetric.cpp
│   ├── test_setup_reuse.cpp
│   └── test_pressure_solver.cpp
└── Hinterkante.dev.Re100/        # Example benchmark case (Backward-facing step, Re=200)
    ├── constant/
    │   ├── hinterkante.mesh      # Mesh topology & coordinates
    │   └── hinterkante.bofl      # Boundary flag mapping table
    ├── system/
    │   ├── hinterkante.control   # Timestepping, iterations, tolerances
    │   ├── hinterkante.physical  # Reynolds number & pressure reference node
    │   └── hinterkante.solver    # Pressure solver tolerances and options
    └── results/                  # Generated VTK files and restart data
```

---

## Building

### Requirements
- C++17 compliant compiler (`clang++` or `g++`)
- `make` or `cmake` (>= 3.16)

### Option A: Using `make` (Default)
```bash
make
```
This produces the executable `./HydraMa2D`.

To clean build artifacts:
```bash
make clean
```

### Option B: Using `cmake`
```bash
mkdir build && cd build
cmake ..
make
```

The CMake build is split into two targets:

- `hydrama2d_core` – static library containing all solver code (everything in `src/` except `main.cpp`)
- `HydraMa2D` – the solver executable (`main.cpp` linked against `hydrama2d_core`)

---

## Testing

Unit tests use [Catch2 v3](https://github.com/catchorg/Catch2) and are registered with CTest. They are built by default with CMake (`-DHYDRAMA2D_BUILD_TESTS=OFF` to disable). The `make` build does not build tests.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Each test case is registered as a separate CTest test with a 120 s timeout, so a hanging test (e.g. a solver that no longer converges) fails instead of blocking the run. The solver regression tests work in private temporary directories and never modify the bundled case.

**Getting Catch2:** CMake first looks for an installed Catch2 v3 (e.g. `brew install catch2`, `apt install catch2`, vcpkg or conda; use `-DCMAKE_PREFIX_PATH=<prefix>` for custom locations). If none is found, Catch2 v3.4.0 is downloaded automatically with `FetchContent` (requires network access). For offline builds, point it at a local Catch2 source tree:

```bash
cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_CATCH2=/path/to/Catch2
```

Without network access to GitHub, the Catch2 v3.4.0 binary package of Ubuntu 24.04 can also be unpacked into a user prefix (no root needed):

```bash
curl -O http://archive.ubuntu.com/ubuntu/pool/universe/c/catch2/catch2_3.4.0-1build1_amd64.deb
dpkg-deb -x catch2_3.4.0-1build1_amd64.deb /tmp/catch2 && mkdir -p ~/.local/catch2-3.4.0 && cp -r /tmp/catch2/usr/* ~/.local/catch2-3.4.0/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HOME/.local/catch2-3.4.0
```

**Useful commands:**

```bash
ctest --test-dir build -R "[Nn]ormal"        # run tests whose name matches a (case-sensitive) regex
./build/tests/hydrama2d_tests "[mass]"       # run by Catch2 tag (see --list-tags)
./build/tests/hydrama2d_tests --list-tests   # list all test cases
```

**Current coverage:** every source file. All routines of `src/Mesh.cpp`, `src/LinearSolver.cpp`, `src/BoundaryConditions.cpp`, `src/IO.cpp` and the three CBS steps are unit-tested on analytic test meshes and on the bundled `Hinterkante` mesh; `src/Solver.cpp` is covered by end-to-end regression runs of the benchmark case:

| Tag | Routine | What is verified |
| :--- | :--- | :--- |
| `[area]` | `compute_shape_function_derivatives` | element areas (`detJ`), max/min volume |
| `[shape]` | `compute_shape_function_derivatives` | shape function derivatives |
| `[mass]` | `compute_mass_matrix` | lumped mass, incl. global assembly in `update_lhs` |
| `[normals]` | `compute_face_normals` | boundary face normals and lengths |
| `[size]` | `compute_element_sizes` | minimum element height (time step length scale) |
| `[laplacian]` | `compute_laplacian_stiffness` | element Laplacian, incl. assembly via `update_lhs` / `matrix_vector_mult` |
| `[wall]` | `determine_wall_nodes` | wall node list, wall normals, trailing edge detection |
| `[timestep]` | `compute_timestep` | diffusive/convective limits, fixed / global / local modes |
| `[matvec]` | `matrix_vector_mult` | operator vs. dense reference, symmetry, linearity |
| `[lhs]` | `update_lhs` | assembled operator (uniform and local time steps), symmetry, SPD, inverse lumped mass |
| `[pcg]` | `solve_pcg` | manufactured solutions, direct-solve agreement, Dirichlet nodes, tolerances, residual reporting |
| `[pressure_bc]` | `setup_pressure_bcs` | outlet (504) Dirichlet nodes, reference node fallback |
| `[step1]` | `compute_step1` | convection (exact quadrature), CBS stabilisation, viscous term, free-stream preservation, conservation |
| `[step2]` | `compute_step2` | divergence term, theta1 term, integration-by-parts identity incl. boundary, divergence theorem |
| `[step3]` | `compute_step3` | pressure-gradient term (exact acceleration, integration by parts, net force), old-pressure dt/4 stabilisation (scaling, locality, conservation) |
| `[symmetry]` | `apply_symmetry_bc` | normal component removed, tangential kept, flag selection, corners |
| `[velocity_bc]` | `apply_velocity_bcs` | uniform inflow, step inflow profile, no-slip walls, precedence at corners |
| `[io]` | `IO::*` | Fortran number parsing, config/mesh/flag readers, mesh id validation, bit-exact restart and VTK round trips, error handling |
| `[boundary_kind]` | `classify_boundary_flag` | flag code table, agreement with the legacy wall / inlet / outlet / symmetry handling |
| `[wallgeom]` | `WallGeometry` | wall faces, inward normals (orientation independent), first-cell height `y_p`, corner bisector normals, node-to-face map, input validation |
| `[walldist]` | `WallDistanceCalculator`, `SegmentAabbTree` | exact wall distance vs. analytic channel, backward-facing step and benchmark geometries; AABB tree bit-identical to brute force; 1-Lipschitz; rigid-motion invariance |
| `[turbulence_config]` | `parse_turbulence_config`, `IO::read_turbulence`, `make_turbulence_model` | defaults, complete file, per-flag overrides, error messages with line numbers, missing file = laminar, factory |
| `[field_ops]` `[recovery]` | `FieldOps`, `NodalGradient` | S / Omega of elementary flows and frame invariance, exact gradients, lumped projection, PPR exact for quadratics and second order |
| `[transport]` `[point_implicit]` `[positivity]` | `assemble_scalar_transport`, `point_implicit_update` | reproduces Step 1, fused = separate, conservation, variable diffusivity, second-order MMS, implicit-Euler decay, Patankar positivity |
| `[sa]` | `SpalartAllmaras`, `sa::*` | closure functions, source terms at reference points, log-layer balance, negative SA, BCs, exact shear ODE, exact log-layer solution |
| `[coupling]` | `compute_step1` (with `nut_elem`), `compute_timestep` (with `nu_eff_elem`) | bit-identical to laminar for zero eddy viscosity, symmetric stress vs tensor reference, objectivity, transpose term, conservation |
| `[wall_units]` | `wall_friction_velocity`, `yplus_field`, `first_cell_yplus` | Couette flow, frame and orientation invariance, no walls |
| `[turbulence_io]` | `IO::write_vtk` (point scalars), `IO::write/read_turbulence_restart` | byte-exact VTK, unchanged laminar VTK, bit-exact restart round trip, error handling |
| `[sst]` | `KOmegaSST`, `sst::*` | F1/F2 blending, shear stress limiter, sources and production limiter, ln ω linearisation, wall/inlet BCs, analytical decay, exact log-layer solution |
| `[kepsilon]` | `KEpsilon<...>`, `ke::*`, `wallfn::*` | Launder–Sharma damping, D and E terms, sources and limiter, Riccati update, analytical / RK4 decay, log-law consistency, positivity |
| `[solver][turbulence]` | `Solver` with turbulence | laminar bit-identity, benchmark runs for all models: health, wall/inlet values, VTK fields, reference values, determinism, bit-for-bit restart |
| `[linalg][vector]` | `linalg::Span`, `VectorOps` | views and const-correctness, kernel values, size checks, aliasing, sequential (reproducible) reductions |
| `[linalg][operator]` | `EbeOperator`, `ConstrainedOperator`, `LinearOperator` | bit-for-bit equal to the original mat-vec, dense reference, diagonal, symmetry, live view of coefficients, Dirichlet rows, input validation |
| `[linalg][constraints]` | `DirichletConstraints` | impose / zero / mask, duplicates, validation |
| `[linalg][preconditioner]` | `IdentityPreconditioner`, `JacobiPreconditioner` | inverse diagonal incl. zero guard, pointwise scaling, set-up checks, `update()` default |
| `[linalg][convergence]` | `ConvergenceControl` | every stopping rule with scripted residuals: reference modes (initial / rhs / first iteration), strictness, order, exact solution, cap, non-finite |
| `[linalg][monitor]` | `IterationMonitor`s | residual history, notification order, stream output, groups, observing never changes results |
| `[linalg][cg]` | `ConjugateGradient`, `solve_pcg` | bit-for-bit equal to the frozen original PCG (`[legacy]`), fused = generic path, one mat-vec/preconditioner per iteration, direct-solve agreement, user-defined operators/preconditioners, finite termination, breakdown, NaN, cap, validation, reuse |
| `[linalg][csr]` | `SparsityPattern`, `CsrMatrix`, `constrain()` | validation, queries, bandwidth/profile, triplets, product vs. dense, rectangular, symmetry, constrained matrix |
| `[linalg][assembly]` | `P1CsrAssembler` | pattern = mesh node graph (brute force), scatter map, entries bit-identical to the EBE operator, products vs. EBE and dense reference, generic element assembly, re-assembly, benchmark mesh |
| `[linalg][reordering]` | `Permutation`, `SymmetricPermutation`, (R)CM | P A P^T, plans, scrambled paths/grids, pseudo-peripheral nodes, disconnected graphs, non-symmetric patterns, Liu-Sherman property, benchmark mesh |
| `[linalg][preconditioner]` | SSOR, IC(0), ILU(0), `ReorderedPreconditioner` | closed-form SSOR, `(LU)_ij = a_ij` on the pattern, exactness without fill-in, Kershaw breakdown + shift, ILU(0) = IC(0) for SPD, symmetry/definiteness, structure reuse, CG agreement with dense solves, quality on the benchmark mesh |
| `[linalg][factory]` | `make_preconditioner`, `make_krylov_solver` | catalogues, options, orderings, errors |
| `[linalg][bicgstab]` | `BiCGStab` | direct-solve agreement on upwind/central convection-diffusion with none / Jacobi / ILU(0) / RCM+ILU(0); true reported residual; one step with exact preconditioner or eigenvector rhs; termination within #distinct eigenvalues; pivot, stagnation (ω) and Lanczos (ρ) breakdowns on symbolically constructed systems; work per iteration; stagnation at the attainable accuracy |
| `[linalg][gmres]` | `Gmres` | direct-solve agreement (schemes × preconditioners × restart × MGS/CGS2); minimal residual over the Krylov space vs. dense least squares; monotone history across restarts; finite termination; happy breakdown; full ≤ restarted iterations; GMRES(1) on positive-real matrices; singular systems; work per iteration; cap, NaN, constraints, options |
| `[reuse]` | `ScaledPreconditioner`, `uniform_scaling`, `PressureSolver` | homogeneity `M(cA)^{-1} = (1/c) M(A)^{-1}` of every preconditioner (also with the IC(0) shift), detection of `A = c A_ref`, one numeric setup per uniform-dt sequence, bit-identical results for constant dt, new setup for local dt / geometry changes, new structure for a new mesh, orderings once per structure, `preconditioner_reuse` option, run-log statistics |
| `[linalg][nonsymmetric]` | BiCGSTAB, GMRES | error vs. direct solution as the tolerance tightens; convergence rates ILU(0) < Jacobi < 1; ILU(0) for strong convection; CG / BiCGSTAB / GMRES agree on an SPD system |
| `[solver_options]` | `IO::read_solver`, `pressure_criteria` | optional `key = value` settings, comments/free text, every invalid input, derived criteria |
| `[pressure_solver]` | `PressureSolver` | bit-for-bit equal to the original PCG over multi-step sequences (global and local dt), legacy configuration, binding/re-binding, monitor; all configurations converge to the same pressure, composition, pattern/ordering built once per mesh |
| `[solver]` | `Solver` | 10-step benchmark runs: finite, bounded, converged PCG, BCs, determinism, reference values, bit-for-bit restart, steady mode; every linear solver configuration reproduces the converged reference values (`[configurations]`) |

The `[solver][regression]` tests compare against reference values recorded on Linux/x86-64 (GCC). Values from the bundled configuration are checked with a platform-robust tolerance (1e-3, velocity and time only), because its loose PCG tolerance makes the pressure sensitive to round-off differences between compilers/CPUs (~1 %). A variant with a converged PCG is checked tightly (1e-7). If a deliberate change to the numerics alters these values, update the reference values in `tests/test_solver_regression.cpp`.

Tests tagged **`[characterization]`** pin down *current* behaviour that is questionable (clockwise elements, trailing edge classification of open wall segments, undocumented `dtfixed` values, PCG initial-residual reporting and iteration cap, order-dependent symmetry projection at kinks, non-zero step-profile wall values, lenient/throwing config parsing, restart files from a larger mesh). They are not statements of intended behaviour: if that behaviour is deliberately changed, update these tests accordingly.

---

## Linear Solver Architecture

The pressure Poisson equation (CBS Step 2) is solved through a small linear algebra layer, `hydrama::linalg` (`include/linalg/`, `src/linalg/`), designed so that new Krylov methods, preconditioners and matrix formats can be added without touching the CFD code:

| Component | Role | Pattern |
| :--- | :--- | :--- |
| `LinearOperator` | `y = A x`; `EbeOperator` (matrix-free, element-by-element), `CsrMatrix` (assembled); capability query `assembled()` | Strategy |
| `SparsityPattern` / `P1CsrAssembler` | shared CSR structure of the P1 node graph; element scatter map for O(nelem) re-assembly | |
| `ConstrainedOperator` / `constrain()` | Dirichlet rows removed (`P A`), resp. the matrix `P A P + (I - P)` for assembled preconditioners | Decorator |
| `Permutation` / `reverse_cuthill_mckee()` | bandwidth/profile reducing orderings, `P A P^T` plans | |
| `Preconditioner` | `z = M^{-1} r`; `Identity`, `Jacobi`, `SsorPreconditioner`, `IncompleteCholesky` (IC(0)), `IncompleteLU` (ILU(0)) | Strategy |
| `ReorderedPreconditioner` | builds any assembled preconditioner in another numbering (RCM) | Decorator |
| `ScaledPreconditioner` | `(1/c) M(A_ref)^{-1}` for `A = c A_ref`: setup reuse under uniform time stepping | Decorator |
| `make_preconditioner` / `make_krylov_solver` | name-based creation; catalogues validate configuration files | Factory |
| `KrylovSolver` | non-virtual `solve()`: validation, Dirichlet lifting, `r_0`, convergence control; methods implement `iterate()`: `ConjugateGradient` (SPD), `BiCGStab`, `Gmres` (general systems, right-preconditioned) | Template Method |
| `ConvergenceCriteria` / `ConvergenceControl` | one implementation of the stopping rules for all methods; `SolveResult` with `converged` flag and `StopReason`; `StagnationGuard` against false-convergence loops at the attainable accuracy | |
| `IterationMonitor` | `ResidualHistory`, `StreamMonitor`, `MonitorGroup` | Observer / Composite |
| `PressureSolver` | what `Solver::run()` calls: `update(...)` after `update_lhs()`, then `solve(rhs, p)` | Facade |

Design rules: one virtual call per O(n) operation; solver objects keep their work vectors between time steps (no allocation per solve); all reductions are sequential and index-ordered, so results are reproducible bit for bit; a pointwise preconditioner (Jacobi) is fused into the CG update loop.

Algorithms and literature: see [docs/LINEAR_SOLVER_REFERENCES.md](docs/LINEAR_SOLVER_REFERENCES.md). Build/test/sanitizer/benchmark instructions for reviewers: [docs/REVIEWER_GUIDE.md](docs/REVIEWER_GUIDE.md).

### Pressure solver options (`system/<case>.solver`)

After the ten positional lines, optional settings can be given as `key = value` lines (a line is a setting only if it starts with an identifier followed by `=`; `#` and `!` start comments; unknown keys, invalid values and duplicates are errors):

| Key | Values (default first) | Meaning |
| :--- | :--- | :--- |
| `method` | `cg`, `bicgstab`, `gmres` | Krylov method: CG for SPD systems (the pressure equation); BiCGSTAB / GMRES(m) for general systems |
| `preconditioner` | `jacobi`, `none`, `ssor`, `ic0`, `ilu0` | `ssor`/`ic0`/`ilu0` are built from the assembled, Dirichlet-constrained matrix |
| `operator_format` | `ebe`, `csr` | matrix-free element-by-element or assembled CSR operator in the Krylov iteration |
| `reordering` | `none`, `rcm` | reverse Cuthill-McKee numbering for `ssor`/`ic0`/`ilu0` |
| `max_iterations` | `100000` | iteration cap |
| `residual_reference` | automatic | `first_iteration` (legacy), `initial` (‖r₀‖) or `rhs` (‖b‖); automatic = `first_iteration` for the legacy PCG, `initial` otherwise |
| `ssor_omega` | `1.0` | SSOR relaxation, `0 < ω < 2` (1 = symmetric Gauss-Seidel) |
| `diagonal_shift` | `0.0` | initial shift α of IC(0)/ILU(0) (`A + α diag(A)`; raised automatically on breakdown) |
| `preconditioner_reuse` | `rescale`, `rebuild` | setup reuse for `ssor`/`ic0`/`ilu0` (see below); `rebuild` refactorises every time step |
| `gmres_restart` | `30` | restart length m of GMRES(m) |
| `gmres_orthogonalization` | `mgs`, `cgs2` | Arnoldi orthogonalisation: modified Gram-Schmidt or classical Gram-Schmidt twice |

Example (fastest configuration on the benchmark case):

```
preconditioner = ic0
reordering     = rcm
```

**Setup reuse (`preconditioner_reuse = rescale`, default).** With global/uniform time stepping the pressure matrix is `A = dt K` with a constant `K`, and every preconditioner of the suite is homogeneous: `M(c A) = c M(A)`. The sparsity pattern, element scatter map and RCM ordering are built once per mesh. The numeric factorisation (IC(0)/ILU(0)/SSOR) is computed once and then reused as `(1/c) M(A_ref)^{-1}`. Each update checks the element data (`gstif`, `pdiag`) against the data of the last factorisation: if `A = c A_ref` entrywise to 1e-12, the factorisation is rescaled, otherwise (local time steps, changed geometry) it is rebuilt, and a new mesh also rebuilds pattern and ordering. The run log reports the work done, e.g. ` Pressure solver setup: 1 numeric setups, 199 rescaled reuses, 1 structure builds`. Jacobi (O(n) setup) and the legacy path are always set up from the current values.

Benchmarks (bundled case, 200 steps, Linux/x86-64, GCC 13, `-O3`; `tools/benchmark.sh --suite --repeats 7`, best wall time of the whole run; run-to-run noise on the measurement machine was about ±0.1 s):

| Configuration | Wall time | Pressure iterations | Setup work |
| :--- | :--- | :--- | :--- |
| `jacobi` (legacy, default) | 2.55 s | 16113 | – |
| `ssor` + `rcm`, rebuild | 1.74 s | 3598 | 200 numeric setups |
| `ssor` + `rcm`, rescale | 1.63 s | 3598 | 1 numeric setup, 199 rescaled |
| `ic0` + `rcm`, rebuild | 1.50 s | 2710 | 200 numeric setups |
| `ic0` + `rcm`, rescale | 1.23 s | 2820 | 1 numeric setup, 199 rescaled |
| `ilu0` + `rcm`, rebuild | 1.56 s | 3075 | 200 numeric setups |
| `ilu0` + `rcm`, rescale | **1.11 s** | 2280 | 1 numeric setup, 199 rescaled |

The setup cost per time step drops from 1.12 to 0.13 ms (IC(0)), 1.34 to 0.13 ms (ILU(0)) and 0.55 to 0.11 ms (SSOR). What remains is the O(nelem) check of the element data. Over 200 steps this saves 0.20 s, 0.24 s and 0.09 s respectively. In the bundled case the iteration totals of rebuild and rescale differ because, at the loose bundled tolerance (10 % residual reduction per step), round-off differences change the flow trajectory. At a converged tolerance (abs. 1e-10) both take the same iterations in every step and give fields equal to 1e-11. With a fixed time step (`--fixed-dt 2.5d-03`, `c = 1`) rescale reproduces rebuild bit for bit, and the run time drops by 0.10 s (IC(0)) and 0.27 s (ILU(0)).

At a converged tolerance (10 steps, abs. tol. 1e-10, earlier measurement): iterations in step 1 are jacobi 539 / ssor+rcm 212 / ic0 227 / ic0+rcm 152.

Without reordering, IC(0) saves iterations but not time: the original node numbering (bandwidth 8376) makes its triangular solves cache-unfriendly. RCM reduces the bandwidth to 126.

**Non-symmetric methods.** BiCGSTAB and GMRES(m) are meant for non-symmetric systems (future implicit momentum / transport equations); ILU(0) is their natural preconditioner (on upwind convection-diffusion test operators it cuts the iterations ~10-30x). On the symmetric pressure equation CG remains the method of choice. Restarted GMRES is slow on this ill-conditioned SPD system: with RCM+ILU(0) and relative tolerance 1e-10 it needs 1285 iterations for m = 30 and 158 for m = 300 (no restart), against 161 for CG. Both methods verify convergence with the true residual `b - A x`, so the reported residual is always that of the returned solution. A tolerance below the attainable accuracy (the rounding floor of the true residual, ~1e-13 relative for the pressure matrix) ends with `StopReason::Stagnation` instead of iterating to the cap.

**Legacy PCG:** the configuration used by the solver (`solver_opt` = 1 in `*.solver`; `solver_opt` keeps its original meaning and only affects `compute_timestep`) is CG + Jacobi on the EBE operator with `ConvergenceCriteria::legacy_pcg(relToler, absToler)`: relative tolerance measured against the residual after the first iteration, 100000 iterations, zero-step breakdown guards. It reproduces the original implementation bit for bit (checked against the frozen copy in `tests/reference/LegacyPcg.hpp`). The free function `solve_pcg()` is kept as an adapter. All other configurations default to `ResidualReference::InitialResidual` and stop on breakdown (set `residual_reference` to override).

---

## Turbulence Modelling

Turbulence is enabled by an optional key-value file `system/<case>.turbulence`
(no file, or `model none`, runs exactly the laminar solver). Minimal example:

```
model k-omega-sst
```

| `model` | Model | Transported variables | Wall treatment |
| :--- | :--- | :--- | :--- |
| `spalart-allmaras` | Spalart–Allmaras with negative-SA extension | `nu_tilde` | resolved (low-Re) |
| `k-omega-sst` | Menter k-ω SST (2003), F1/F2 blending, cross diffusion, shear stress limiter | `k`, `log_omega` (ln ω) | resolved (low-Re) |
| `k-epsilon` | Standard k-ε (Launder–Spalding) | `k`, `epsilon` | log-law wall functions |
| `launder-sharma-k-epsilon` | Launder–Sharma low-Re k-ε with damping functions | `k`, `epsilon_tilde` | resolved (low-Re) |

Optional settings (defaults in brackets): inlet turbulence `inlet_intensity`
[0.05] and `inlet_length_scale` [0.1] (k = 1.5 (U_ref I)^2 with
`reference_velocity` [1]), or `inlet_viscosity_ratio` (nu_t/nu), or direct
`inlet_k` / `inlet_epsilon` / `inlet_omega`; SA: `inlet_nu_tilde_ratio` [3],
`sa_ft2` [on]; two-equation models: `production_limiter` [10, 0 = off];
`wall_treatment` [model default]; `wall_distance` [exact | brute-force];
`gradient_recovery` [ppr | lumped-projection]; `nut_ratio_max` [1e5]; and
per-flag overrides such as `boundary 503 inlet k 1e-3 omega 5` or
`boundary 506 symmetry`. Comments start with `#`; unknown keys are errors.
See `docs/TURBULENCE_DESIGN.md` for the models, numerics and verification.

With a turbulence model the solver
- adds the eddy viscosity to Step 1 (symmetric stress form),
- limits the time step with nu_eff = max(nu + nu_t, turbulence diffusivities),
- advances the turbulence equations after Step 3,
- logs per step `Turbulence: <variable> change ... nut/nu max ... y+ (first cell) min/mean/max`,
- writes the model fields as VTK point scalars (SA: `nu_tilde`, `nut`; SST:
  `k`, `omega`, `nut`, `sst_f1`; k-ε: `k`, `epsilon`/`epsilon_tilde`, `nut`)
  plus `wall_distance` and `yplus`, and `results/<case>.turb.var` next to
  `results/<case>.var`.

Restarting (`restart_opt = 1`) continues the turbulence from `<case>.turb.var`;
if it is missing or does not match (model, fields, node count) the turbulence
starts from free-stream values, e.g. to start a turbulent run from a laminar
solution.

Explicit time stepping: the turbulence diffusivity enters the diffusive time
step limit, so turbulent runs take smaller time steps than laminar ones.
For steady problems use local time stepping (`dtfixed = 0`, transient off).

---

## Running a Simulation

### From the Case Directory
```bash
cd Hinterkante.dev.Re100
../HydraMa2D hinterkante
```

### From the Root Directory
```bash
./HydraMa2D Hinterkante.dev.Re100/hinterkante
# or pass the case directory directly:
./HydraMa2D Hinterkante.dev.Re100
```

---

## Boundary Flag Codes

| Flag Code | Description |
| :--- | :--- |
| **500** | Adiabatic wall / prescribed velocity (no-slip if unprescribed) |
| **501** | Constant temperature wall ($T = 1$) with no-slip |
| **502** | Constant temperature wall ($T = 0$) with no-slip |
| **503** | Prescribed inflow velocity ($u = 1.0, v = 0.0$) |
| **504** | Pressure outlet ($p = 0.0$) |
| **506** | Velocity symmetry / slip boundary ($\vec{u} \cdot \vec{n} = 0$) |
| **507** | Backward Facing Step polynomial inflow profile ($v = 0.0$) |

New code classifies these codes through `classify_boundary_flag()` (`include/BoundaryKind.hpp`): 500/501/502 → wall, 503/507 → inlet, 504 → outlet, 506 → symmetry, anything else → unknown.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

