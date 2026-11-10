# ---------------------------------------------------------------------------
# Catch2 v3
#
# 1. Use an installed Catch2 v3 if CMake can find one (apt/Homebrew/vcpkg/
#    conda, or any prefix passed via -DCMAKE_PREFIX_PATH=...).
# 2. Otherwise download the pinned release below with FetchContent.
#    For offline builds point FetchContent at a local source tree with
#    -DFETCHCONTENT_SOURCE_DIR_CATCH2=/path/to/Catch2
# ---------------------------------------------------------------------------
set(HYDRAMA2D_CATCH2_VERSION 3.4.0)

find_package(Catch2 3 CONFIG QUIET)

if(Catch2_FOUND)
    message(STATUS "HydraMa2D tests: using installed Catch2 ${Catch2_VERSION} (${Catch2_DIR})")
else()
    message(STATUS "HydraMa2D tests: Catch2 v3 not found, fetching v${HYDRAMA2D_CATCH2_VERSION}")
    include(FetchContent)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v${HYDRAMA2D_CATCH2_VERSION}
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()

include(Catch)

# ---------------------------------------------------------------------------
# Test executable
# ---------------------------------------------------------------------------
add_executable(hydrama2d_tests
    test_mesh_area.cpp
    test_mesh_shape_functions.cpp
    test_mesh_mass_matrix.cpp
    test_mesh_face_normals.cpp
    test_mesh_element_sizes.cpp
    test_mesh_laplacian.cpp
    test_mesh_wall_nodes.cpp
    test_mesh_hinterkante.cpp
    test_timestep.cpp
    test_linear_assembly.cpp
    test_pcg.cpp
    test_pressure_bc.cpp
    test_step1.cpp
    test_step2.cpp
    test_step3.cpp
    test_velocity_bc.cpp
    test_io.cpp
    test_solver_regression.cpp
    test_linalg_vector.cpp
    test_linalg_interfaces.cpp
    test_linalg_convergence.cpp
    test_linalg_monitor.cpp
    test_linalg_cg.cpp
    test_pressure_solver.cpp
    test_linalg_csr.cpp
    test_linalg_assembly.cpp
    test_linalg_reordering.cpp
    test_linalg_assembled_preconditioners.cpp
    test_linalg_factory.cpp
    test_solver_config.cpp
    test_linalg_bicgstab.cpp
    test_linalg_gmres.cpp
    test_linalg_nonsymmetric.cpp
    test_setup_reuse.cpp
    test_boundary_kind.cpp
    test_wall_geometry.cpp
    test_wall_distance.cpp
    test_turbulence_config.cpp
    test_field_ops.cpp
    test_scalar_transport.cpp
    test_spalart_allmaras.cpp
    test_turbulent_coupling.cpp
    test_wall_units.cpp
    test_turbulence_io.cpp
    test_solver_turbulence.cpp
    test_k_omega_sst.cpp
    test_k_epsilon.cpp
)
target_link_libraries(hydrama2d_tests PRIVATE hydrama2d_core Catch2::Catch2WithMain)
target_compile_options(hydrama2d_tests PRIVATE ${HYDRAMA2D_COMPILE_OPTIONS})

# Location of the bundled benchmark case, used by the real-mesh tests.
target_compile_definitions(hydrama2d_tests PRIVATE
    HYDRAMA2D_CASE_DIR="${PROJECT_SOURCE_DIR}/Hinterkante.dev.Re100"
)

# Register every TEST_CASE as an individual CTest test. The timeout turns a
# hanging test (e.g. a solver that no longer converges) into a failure; the
# slowest test takes ~1 s, so 120 s leaves ample room for sanitizer builds.
catch_discover_tests(hydrama2d_tests PROPERTIES TIMEOUT 120)
