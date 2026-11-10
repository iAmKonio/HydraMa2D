# Reviewer Guide — Linear Solver Layer

How to build, test and benchmark the linear solver work (`hydrama::linalg`, `PressureSolver`), and where to look in the code. All commands are run from the repository root.

## 1. Prerequisites

- C++17 compiler (tested: GCC 13.3 on Linux/x86-64), CMake ≥ 3.16, `make`
- Catch2 v3 for the tests. CMake uses an installed Catch2 or downloads v3.4.0. Without network access, unpack the Ubuntu 24.04 package into a user prefix:

```bash
curl -O http://archive.ubuntu.com/ubuntu/pool/universe/c/catch2/catch2_3.4.0-1build1_amd64.deb
dpkg-deb -x catch2_3.4.0-1build1_amd64.deb /tmp/catch2
mkdir -p ~/.local/catch2-3.4.0 && cp -r /tmp/catch2/usr/* ~/.local/catch2-3.4.0/
```

Then add `-DCMAKE_PREFIX_PATH=$HOME/.local/catch2-3.4.0` to the `cmake` configure commands below. Without network access there is no FetchContent fallback.

## 2. Build and run the test suite

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4 2>&1 | tee build.log
grep -c warning build.log            # expected: 0 (flags: -O3 -Wall -Wextra -pedantic)
ctest --test-dir build -j4 --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 405`, in about 20 s on 4 cores. The 205 tests of the baseline commit are unchanged and included.

Useful subsets (Catch2 tags, see the coverage table in `README.md`):

```bash
./build/tests/hydrama2d_tests "[legacy]"          # bit-for-bit equality with the original PCG
./build/tests/hydrama2d_tests "[bicgstab],[gmres],[nonsymmetric]"
./build/tests/hydrama2d_tests "[reuse]"           # setup reuse (this last step)
./build/tests/hydrama2d_tests "[solver][regression]"
./build/tests/hydrama2d_tests --list-tags
```

The `make` build (no tests) also works: `make CXX=g++ -j4` produces `./HydraMa2D`.

## 3. Sanitizers

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j4
ctest --test-dir build-asan -j4
```

Expected: 405/405, in about 70 s. Each test has a 120 s CTest timeout. `-fno-sanitize-recover=all` turns any UBSan finding into a failure.

## 4. Bit-for-bit check of the default configuration

The default settings (CG + Jacobi, element-by-element operator) must reproduce the original solver exactly. Inside the test suite this is checked against a frozen copy of the original PCG (`tests/reference/LegacyPcg.hpp`, tag `[legacy]`). For an independent end-to-end check against the baseline commit:

```bash
git worktree add /tmp/hydrama-baseline b71176a
cmake -S /tmp/hydrama-baseline -B /tmp/hydrama-baseline/build -DHYDRAMA2D_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/hydrama-baseline/build -j4
# Run both executables on the bundled case, shortened to 200 steps (the 6th and
# 18th non-empty lines of the control file: number of steps, output interval).
for exe in /tmp/hydrama-baseline/build/HydraMa2D ./build/HydraMa2D; do
  rm -rf /tmp/case-check && cp -r Hinterkante.dev.Re100 /tmp/case-check
  awk '/[^ \t\r]/ { n++; if (n == 6 || n == 18) { print "200"; next } } { print }' \
      Hinterkante.dev.Re100/system/hinterkante.control > /tmp/case-check/system/hinterkante.control
  "$exe" /tmp/case-check | grep -v -e "Time Taken" -e "^Pressure solver: " > "$exe.check.log"
  cp /tmp/case-check/results/hinterkante.var "$exe.check.var"
done
cmp /tmp/hydrama-baseline/build/HydraMa2D.check.var ./build/HydraMa2D.check.var &&
  diff /tmp/hydrama-baseline/build/HydraMa2D.check.log ./build/HydraMa2D.check.log && echo IDENTICAL
git worktree remove --force /tmp/hydrama-baseline
```

Expected output: `IDENTICAL`. The restart file (all fields, 17 significant digits) and the whole log (every pressure residual and iteration count) are compared. The only new log line in the default configuration, `Pressure solver: PCG (CG + jacobi, ebe)`, is filtered out. During development the same check was also run for a converged-pressure variant and for steady local time stepping.

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
