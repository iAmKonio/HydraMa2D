# Reviewer Guide

How to build, test and verify HydraMa2D, and where to look in the code. All commands are run from the repository
root. Sections 5–7 cover the linear solver layer (`hydrama::linalg`, `PressureSolver`) in detail; the mesh subsystem
is described in [MESH_IMPORT.md](MESH_IMPORT.md), [BOUNDARY_NAMING.md](BOUNDARY_NAMING.md),
[QUADRILATERAL_ELEMENTS.md](QUADRILATERAL_ELEMENTS.md) and [CYLINDER_BENCHMARK.md](CYLINDER_BENCHMARK.md).

## 1. Prerequisites

- C++17 compiler (tested: GCC 13.3 on Linux/x86-64), CMake ≥ 3.16, `make`
- Catch2 v3 for the tests. CMake uses an installed Catch2 or downloads v3.4.0. Without network access, unpack the Ubuntu 24.04 package into a user prefix:

```bash
curl -O http://archive.ubuntu.com/ubuntu/pool/universe/c/catch2/catch2_3.4.0-1build1_amd64.deb
dpkg-deb -x catch2_3.4.0-1build1_amd64.deb /tmp/catch2
mkdir -p ~/.local/catch2-3.4.0 && cp -r /tmp/catch2/usr/* ~/.local/catch2-3.4.0/
```

Then add `-DCMAKE_PREFIX_PATH=$HOME/.local/catch2-3.4.0` to the `cmake` configure commands below. Without network access there is no FetchContent fallback.
- Optional: the CGNS library (3.4 or 4.x) for CGNS input, see section 8.
- Optional, only to regenerate meshes: the Gmsh Python module (`pip install gmsh`).

## 2. Build and run the test suite

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 2>&1 | tee build.log
grep -c warning build.log            # expected: 0 (flags: -O3 -Wall -Wextra -pedantic)
ctest --test-dir build -j4 --output-on-failure
```

Expected without the CGNS library: `100% tests passed, 0 tests failed out of 748`, of which 7 CGNS file tests
are reported as skipped; with CGNS: `out of 754`, one skipped. About 40 s on 4 cores. The tests of the baseline
commit are unchanged and included.

Useful subsets (Catch2 tags, see the coverage table in `README.md`):

```bash
./build/tests/hydrama2d_tests "[legacy]"          # bit-for-bit equality with the original PCG
./build/tests/hydrama2d_tests "[mesh_import],[gmsh],[su2],[cgns]"   # mesh import
./build/tests/hydrama2d_tests "[quad],[quadrature]"                 # quadrilaterals and hybrid meshes
./build/tests/hydrama2d_tests "[post],[mesh_quality],[inlet_parabolic],[check_mesh]"
./build/tests/hydrama2d_tests "[benchmark]"       # Schafer-Turek cylinder, level-0 meshes
./build/tests/hydrama2d_tests "[solver][regression]"
./build/tests/hydrama2d_tests --list-tags
```

The `make` build (no tests) also works: `make CXX=g++ -j4` produces `./HydraMa2D` (`make CGNS=1 ...` with CGNS).

## 3. Sanitizers

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j4 2>&1 | tee build-asan.log
grep -c warning build-asan.log       # expected: 0
ctest --test-dir build-asan -j4 --output-on-failure
```

Expected: the same counts as in section 2, in about 3–4 minutes on 4 cores. Each test has a 120 s CTest timeout.
`-fno-sanitize-recover=all` turns any UBSan finding into a failure, and AddressSanitizer's leak checker is active.

## 4. Bit-for-bit check against the baseline commit

The default settings and every existing configuration must reproduce the original solver exactly on legacy cases.
Inside the test suite this is checked against a frozen copy of the original PCG (`tests/reference/LegacyPcg.hpp`,
tag `[legacy]`) and recorded reference values. End to end, `tools/check_bit_identity.sh` builds the baseline commit
(default: the root commit, `b2a20c2`) in a temporary git worktree and runs both executables on the bundled case:

```bash
tools/check_bit_identity.sh --bin build/HydraMa2D            # 200 steps per configuration
tools/check_bit_identity.sh --bin build/HydraMa2D --steps 1000
```

It compares the restart files (all fields, 17 significant digits; and the turbulence restart file) byte for byte and
the whole log line for line (every pressure residual and iteration count) for 13 configurations: the bundled
settings, steady local time stepping, fixed dt, a converged pressure, SSOR / IC(0) / ILU(0) with RCM, BiCGSTAB and
GMRES with ILU(0), and the four turbulence models. Expected output: 13 × `IDENTICAL`, exit status 0. Only the wall
time, the case directory and the new informational lines (`Mesh formats:` banner, `Mesh quality:`) are filtered
from the logs.

Gmsh, SU2 and CGNS versions of the benchmark mesh reproduce the legacy run bit for bit too; this is part of the test
suite (`[mesh_import][hinterkante]`, `[su2]`, `[cgns][file]`).

## 4b. Cylinder benchmark (triangles vs. inflation layers)

```bash
pip install gmsh                                                        # only to generate the meshes
python3 tools/cylinder_benchmark/generate_meshes.py /tmp/cyl/meshes --levels 1 2 3
python3 tools/cylinder_benchmark/run_benchmark.py --bin build/HydraMa2D \
        --meshes /tmp/cyl/meshes --work /tmp/cyl/work --cases steady --jobs 2
python3 tools/cylinder_benchmark/run_benchmark.py --bin build/HydraMa2D \
        --meshes /tmp/cyl/meshes --work /tmp/cyl/work --cases unsteady --only tri_L1 bl_L1 tri_L2 --jobs 2
```

The steady case (2D-1) takes a few minutes per level-3 mesh; the periodic case (2D-2) from seconds (level 1) to
hours (inflation layers, level 2 and finer: the explicit time step follows the thinnest cell). Results and their
discussion: [CYLINDER_BENCHMARK.md](CYLINDER_BENCHMARK.md).

## 5. Running the solver and configuring the pressure solver

```bash
./build/HydraMa2D Hinterkante.dev.Re100
```

The pressure solver is selected in `Hinterkante.dev.Re100/system/hinterkante.solver`. The first ten non-empty lines are the original positional values (solver option, relative and absolute tolerance). After them, optional `key = value` lines follow. The bundled file lists every key with its default:

```
method          = cg       # cg (SPD) | bicgstab | gmres (general systems)
preconditioner  = jacobi   # none | jacobi | ssor | ic0 | ilu0
operator_format = ebe      # ebe (matrix-free) | csr (assembled)
reordering      = none     # none | rcm (only with ssor, ic0, ilu0)
max_iterations  = 100000
ssor_omega      = 1.0      # 0 < omega < 2 (ssor only)
diagonal_shift  = 0.0      # initial diagonal shift alpha (ic0, ilu0)
preconditioner_reuse = rescale   # rescale: factorise once, rescale for uniform dt | rebuild
gmres_restart   = 30       # GMRES(m) restart length (gmres only)
gmres_orthogonalization = mgs   # mgs | cgs2 (gmres only)
# residual_reference = initial   # first_iteration | initial | rhs
```

Recommended fast configuration for the pressure equation: `preconditioner = ilu0` (or `ic0`) with `reordering = rcm`.

Rules for the settings block:
- `#` and `!` start comments.
- Unknown keys, invalid values, duplicates, and `reordering = rcm` with `jacobi`/`none` make the solver stop at start-up with an `ERROR: *.solver OPTION ...` message.

What the log reports:
- At start-up: `Pressure solver: <method> (<krylov> + <preconditioner>, <operator>)`.
- At the end, for non-legacy configurations: `Pressure solver setup: <n> numeric setups, <m> rescaled reuses, <k> structure builds`.

## 6. Benchmarks

```bash
tools/benchmark.sh --suite --repeats 7                      # bundled case, 200 steps
tools/benchmark.sh --suite --repeats 7 --fixed-dt 2.5d-03   # fixed dt: rebuild and rescale identical
tools/benchmark.sh "preconditioner = ic0" "reordering = rcm"   # one configuration
tools/benchmark.sh --help
```

The script copies the case into a temporary directory, sets the number of steps and the solver settings, runs each configuration `--repeats` times (interleaved in `--suite` mode), and prints the best wall time, the total pressure iterations and the setup statistics. Reference results are in the README section "Pressure solver options". Expect ±0.1 s run-to-run noise on shared machines.

## 7. Where to look

| Topic | Files |
| :--- | :--- |
| Interfaces | `include/linalg/LinearOperator.hpp`, `Preconditioner.hpp`, `KrylovSolver.hpp`, `Convergence.hpp`, `IterationMonitor.hpp` |
| Krylov methods | `src/linalg/ConjugateGradient.cpp`, `BiCGStab.cpp`, `Gmres.cpp` |
| Operators / storage | `EbeOperator`, `CsrMatrix`, `SparsityPattern`, `CsrAssembly` (P1 scatter map), `ConstrainedOperator` |
| Preconditioners | `JacobiPreconditioner`, `SsorPreconditioner`, `IncompleteCholesky`, `IncompleteLU`, `DiagonalShift`, `ReorderedPreconditioner`, `ScaledPreconditioner` |
| Reordering | `Permutation.hpp`, `Reordering.cpp` (RCM, pseudo-peripheral nodes) |
| Factory / configuration | `src/linalg/Factory.cpp`, `src/IO.cpp` (`read_solver_options`), `include/Config.hpp` (`SolverConfig`) |
| CFD integration | `include/PressureSolver.hpp`, `src/PressureSolver.cpp` (composition, setup reuse), `src/Solver.cpp` |
| Frozen reference | `tests/reference/LegacyPcg.hpp` |
| Literature | `docs/LINEAR_SOLVER_REFERENCES.md` |
| Mesh import | `include/mesh/` (`GmshReader`, `Su2Reader`, `CgnsReader`, `MeshBuilder`, `MeshCheck`, `BoundaryNaming`, `MeshQuality`), `IO::read_case_mesh`, `docs/MESH_IMPORT.md` |
| Quadrilaterals | `include/numerics/QuadElement.hpp`, `Quadrature.hpp`, the quadrilateral branches of `Mesh.cpp`, `Step1-3.cpp`, `LinearSolver.cpp`, `docs/QUADRILATERAL_ELEMENTS.md` |
| Forces, probes, benchmark | `include/post/`, `BoundaryConditions.cpp` (`parabolic_inlet_velocities`), `tools/cylinder_benchmark/`, `docs/CYLINDER_BENCHMARK.md` |

## 8. Optional CGNS support

CGNS mesh input is built only if the CGNS library is found (`HYDRAMA2D_WITH_CGNS=AUTO|ON|OFF`, see
`docs/MESH_IMPORT.md`, section 4). Both configurations must build without warnings and pass the suite:

```bash
cmake -S . -B build-nocgns -DHYDRAMA2D_WITH_CGNS=OFF && cmake --build build-nocgns -j4 && ctest --test-dir build-nocgns
cmake -S . -B build-cgns -DHYDRAMA2D_WITH_CGNS=ON [-DCGNS_ROOT=<prefix>] && cmake --build build-cgns -j4 && ctest --test-dir build-cgns
```

Without CGNS the CGNS file tests are reported as *skipped* (Catch2 `SKIP`, `SKIP_RETURN_CODE 4`). With CGNS one
test is skipped: the one that checks the "built without CGNS support" error.
