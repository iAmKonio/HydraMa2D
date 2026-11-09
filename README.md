# HydraMa 2D (C++)

**HydraMa 2D** is an unstructured finite element CFD solver for 2D incompressible fluid flows. It uses the **Characteristic-Based Split (CBS)** scheme on linear 3-node triangular meshes ($P_1 - P_1$).

---

## Features

- **Characteristic-Based Split (CBS) Scheme:**
  - **Step 1:** Explicit intermediate velocity calculation with convective streamline stabilisation and viscous diffusion.
  - **Step 2:** Semi-implicit pressure Poisson solve enforcing mass conservation and avoiding pressure-velocity decoupling without violating the LBB condition.
  - **Step 3:** Explicit velocity correction with pressure gradient and CBS pressure stabilization.
- **Element-By-Element (EBE) PCG Solver:** Matrix-free Jacobi Preconditioned Conjugate Gradient solver requiring only $O(N_{\text{elem}} + N_{\text{poin}})$ memory with no global sparse matrix assembly.
- **Adaptive Timestepping:** Automatic computation of local and global time steps constrained by convective CFL and diffusive stability limits.
- **Case Structure:** OpenFOAM-style directory structure (`constant/`, `system/`, `results/`).
- **Standard Outputs:**
  - ASCII VTK unstructured grids (`results/<case>_<time>.vtk`) for visualization in ParaView.
  - Checkpoint / restart data files (`results/<case>.var`), written with 17 significant digits so that a restarted run continues bit-for-bit.

---

## Directory Structure

```
HydraMa.dev/
├── CMakeLists.txt                # CMake build configuration
├── Makefile                      # Make build configuration
├── README.md                     # Documentation
├── include/                      # C++ Header files
│   ├── BoundaryConditions.hpp
│   ├── Config.hpp
│   ├── IO.hpp
│   ├── LinearSolver.hpp
│   ├── Mesh.hpp
│   ├── Solver.hpp
│   ├── Step1.hpp
│   ├── Step2.hpp
│   └── Step3.hpp
├── src/                          # C++ Source files
│   ├── BoundaryConditions.cpp
│   ├── IO.cpp
│   ├── LinearSolver.cpp
│   ├── Mesh.cpp
│   ├── Solver.cpp
│   ├── Step1.cpp
│   ├── Step2.cpp
│   ├── Step3.cpp
│   └── main.cpp
├── tests/                        # Catch2 unit tests (CMake/CTest only)
│   ├── CMakeLists.txt
│   ├── MeshTestUtils.hpp         # Test mesh builders (triangles, structured grids)
│   ├── LinearSolverTestUtils.hpp # Dense-matrix helpers (reference assembly, direct solve)
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
│   └── test_solver_regression.cpp # End-to-end runs of the benchmark case
└── Hinterkante.dev.Re100/        # Example benchmark case (Backward-facing step, Re=200)
    ├── constant/
    │   ├── hinterkante.mesh      # Mesh topology & coordinates
    │   └── hinterkante.bofl      # Boundary flag mapping table
    ├── system/
    │   ├── hinterkante.control   # Timestepping, iterations, tolerances
    │   ├── hinterkante.physical  # Reynolds number & pressure reference node
    │   └── hinterkante.solver    # PCG linear solver tolerances
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
| `[solver]` | `Solver` | 10-step benchmark runs: finite, bounded, converged PCG, BCs, determinism, reference values, bit-for-bit restart, steady mode |

The `[solver][regression]` tests compare against reference values recorded on Linux/x86-64 (GCC). Values from the bundled configuration are checked with a platform-robust tolerance (1e-3, velocity and time only), because its loose PCG tolerance makes the pressure sensitive to round-off differences between compilers/CPUs (~1 %). A variant with a converged PCG is checked tightly (1e-7). If a deliberate change to the numerics alters these values, update the reference values in `tests/test_solver_regression.cpp`.

Tests tagged **`[characterization]`** pin down *current* behaviour that is questionable (clockwise elements, trailing edge classification of open wall segments, undocumented `dtfixed` values, PCG initial-residual reporting and iteration cap, order-dependent symmetry projection at kinks, non-zero step-profile wall values, lenient/throwing config parsing, restart files from a larger mesh). They are not statements of intended behaviour: if that behaviour is deliberately changed, update these tests accordingly.

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
