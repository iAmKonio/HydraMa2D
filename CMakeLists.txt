cmake_minimum_required(VERSION 3.16)
project(HydraMa2D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(HYDRAMA2D_BUILD_TESTS "Build the HydraMa2D unit tests (requires Catch2 v3)" ON)

# ---------------------------------------------------------------------------
# Optional CGNS mesh input (CGNS library with HDF5), see docs/MESH_IMPORT.md.
#   AUTO (default): used if found; ON: required; OFF: not used.
# Hints: CGNS_ROOT / CMAKE_PREFIX_PATH, CGNS_USE_STATIC_LIBS, HDF5_ROOT.
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(HYDRAMA2D_WITH_CGNS "AUTO" CACHE STRING "CGNS mesh input: AUTO (if found), ON (required) or OFF")
set_property(CACHE HYDRAMA2D_WITH_CGNS PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${HYDRAMA2D_WITH_CGNS}" _hydrama2d_with_cgns)
if(_hydrama2d_with_cgns MATCHES "^(ON|YES|TRUE|Y|1)$")
    set(_hydrama2d_with_cgns ON)
elseif(_hydrama2d_with_cgns MATCHES "^(OFF|NO|FALSE|N|0)$")
    set(_hydrama2d_with_cgns OFF)
elseif(NOT _hydrama2d_with_cgns STREQUAL "AUTO")
    message(FATAL_ERROR "HYDRAMA2D_WITH_CGNS must be AUTO, ON or OFF (got '${HYDRAMA2D_WITH_CGNS}')")
endif()

set(HYDRAMA2D_HAVE_CGNS OFF)
if(_hydrama2d_with_cgns STREQUAL "ON")
    find_package(CGNS 3.3 REQUIRED)
    set(HYDRAMA2D_HAVE_CGNS ON)
elseif(_hydrama2d_with_cgns STREQUAL "AUTO")
    find_package(CGNS 3.3 QUIET)
    set(HYDRAMA2D_HAVE_CGNS ${CGNS_FOUND})
endif()
if(HYDRAMA2D_HAVE_CGNS)
    set(_hydrama2d_cgns_kind "shared")
    if(CGNS_STATIC)
        set(_hydrama2d_cgns_kind "static")
    endif()
    message(STATUS "HydraMa2D CGNS mesh input: enabled (CGNS ${CGNS_VERSION}, ${_hydrama2d_cgns_kind}: ${CGNS_LIBRARY})")
    if(CGNS_RUNTIME_DIRS)
        message(STATUS "HydraMa2D CGNS mesh input: run HydraMa2D with ${CGNS_RUNTIME_DIRS} in the library search "
                       "path (e.g. LD_LIBRARY_PATH); the tests set it themselves")
    endif()
elseif(_hydrama2d_with_cgns STREQUAL "OFF")
    message(STATUS "HydraMa2D CGNS mesh input: disabled (HYDRAMA2D_WITH_CGNS=OFF)")
else()
    message(STATUS "HydraMa2D CGNS mesh input: disabled (CGNS library not found; set CGNS_ROOT or CMAKE_PREFIX_PATH)")
endif()

# Compile options shared by the solver library and the executable.
set(HYDRAMA2D_COMPILE_OPTIONS -O3 -Wall -Wextra -pedantic -ffp-contract=off)

# ---------------------------------------------------------------------------
# Core solver library: everything except the program entry point, so that the
# executable and the unit tests link against the same compiled code.
# ---------------------------------------------------------------------------
add_library(hydrama2d_core STATIC
    src/BoundaryConditions.cpp
    src/CaseMeshCheck.cpp
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
    src/geometry/FieldOps.cpp
    src/geometry/SegmentTree.cpp
    src/geometry/WallDistance.cpp
    src/geometry/WallGeometry.cpp
    src/mesh/BoundaryNaming.cpp
    src/mesh/GmshReader.cpp
    src/mesh/MeshBuilder.cpp
    src/mesh/MeshCheck.cpp
    src/mesh/MeshQuality.cpp
    src/mesh/Su2Reader.cpp
    src/mesh/CgnsReader.cpp
    src/mesh/CgnsTranslate.cpp
    src/numerics/PointImplicit.cpp
    src/numerics/Quadrature.cpp
    src/numerics/QuadElement.cpp
    src/numerics/ScalarTransport.cpp
    src/post/Forces.cpp
    src/post/Monitors.cpp
    src/post/Probes.cpp
    src/post/SkinFriction.cpp
    src/post/Spectral.cpp
    src/turbulence/SpalartAllmaras.cpp
    src/turbulence/TurbulenceBC.cpp
    src/turbulence/TurbulenceConfig.cpp
    src/turbulence/KEpsilon.cpp
    src/turbulence/KOmegaSST.cpp
    src/turbulence/TurbulenceFactory.cpp
    src/turbulence/WallFunctions.cpp
    src/turbulence/WallUnits.cpp
)
target_include_directories(hydrama2d_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(hydrama2d_core PUBLIC cxx_std_17)
target_compile_options(hydrama2d_core PRIVATE ${HYDRAMA2D_COMPILE_OPTIONS})
if(HYDRAMA2D_HAVE_CGNS)
    # Only src/mesh/CgnsReader.cpp uses the library; consumers of the static
    # hydrama2d_core link it through this (link-only) dependency.
    target_link_libraries(hydrama2d_core PRIVATE HydraMa2D::CGNS)
    target_compile_definitions(hydrama2d_core PRIVATE HYDRAMA2D_HAVE_CGNS=1)
endif()

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
