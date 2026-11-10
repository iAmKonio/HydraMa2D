cmake_minimum_required(VERSION 3.16)
project(HydraMa2D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(HYDRAMA2D_BUILD_TESTS "Build the HydraMa2D unit tests (requires Catch2 v3)" ON)

# Compile options shared by the solver library and the executable.
set(HYDRAMA2D_COMPILE_OPTIONS -O3 -Wall -Wextra -pedantic -ffp-contract=off)

# ---------------------------------------------------------------------------
# Core solver library: everything except the program entry point, so that the
# executable and the unit tests link against the same compiled code.
# ---------------------------------------------------------------------------
add_library(hydrama2d_core STATIC
    src/BoundaryConditions.cpp
    src/IO.cpp
    src/LinearSolver.cpp
    src/Mesh.cpp
    src/PressureSolver.cpp
    src/Solver.cpp
    src/Step1.cpp
    src/Step2.cpp
    src/Step3.cpp
    src/linalg/BasicPreconditioners.cpp
    src/linalg/BiCGStab.cpp
    src/linalg/ConjugateGradient.cpp
    src/linalg/ConstrainedOperator.cpp
    src/linalg/Convergence.cpp
    src/linalg/CsrAssembly.cpp
    src/linalg/CsrMatrix.cpp
    src/linalg/DirichletConstraints.cpp
    src/linalg/EbeOperator.cpp
    src/linalg/Factory.cpp
    src/linalg/Gmres.cpp
    src/linalg/IncompleteCholesky.cpp
    src/linalg/IncompleteLU.cpp
    src/linalg/IterationMonitor.cpp
    src/linalg/KrylovSolver.cpp
    src/linalg/LinearOperator.cpp
    src/linalg/Permutation.cpp
    src/linalg/Preconditioner.cpp
    src/linalg/ReorderedPreconditioner.cpp
    src/linalg/ScaledPreconditioner.cpp
    src/linalg/Reordering.cpp
    src/linalg/SparsityPattern.cpp
    src/linalg/SsorPreconditioner.cpp
    src/linalg/VectorOps.cpp
)
target_include_directories(hydrama2d_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(hydrama2d_core PUBLIC cxx_std_17)
target_compile_options(hydrama2d_core PRIVATE ${HYDRAMA2D_COMPILE_OPTIONS})

# ---------------------------------------------------------------------------
# Solver executable
# ---------------------------------------------------------------------------
add_executable(HydraMa2D src/main.cpp)
target_link_libraries(HydraMa2D PRIVATE hydrama2d_core)
target_compile_options(HydraMa2D PRIVATE ${HYDRAMA2D_COMPILE_OPTIONS})

# ---------------------------------------------------------------------------
# Unit tests (Catch2 + CTest)
# ---------------------------------------------------------------------------
if(HYDRAMA2D_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
