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
  - Checkpoint / restart data files (`results/<case>.var`).

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
