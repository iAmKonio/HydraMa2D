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
    test_boundary_naming.cpp
    test_gmsh_reader.cpp
    test_mesh_builder.cpp
    test_gmsh_fixtures.cpp
    test_mesh_import_regression.cpp
    test_su2_reader.cpp
    test_su2_import.cpp
    test_cgns_translate.cpp
    test_cgns_reader.cpp
    test_quadrature.cpp
    test_quad_element.cpp
    test_quad_mesh.cpp
    test_quad_patch.cpp
    test_quad_solver.cpp
    test_parabolic_inlet.cpp
    test_forces.cpp
    test_probes.cpp
    test_monitors.cpp
    test_mesh_quality.cpp
    test_cylinder_benchmark.cpp
    test_case_mesh_check.cpp
    test_moving_wall.cpp
    test_skin_friction.cpp
    test_spectral.cpp
    test_thermal.cpp
    test_periodic.cpp
    test_inflow_bc.cpp
    test_openmp.cpp
    test_vtu_io.cpp
)
target_link_libraries(hydrama2d_tests PRIVATE hydrama2d_core Catch2::Catch2WithMain)
target_compile_options(hydrama2d_tests PRIVATE ${HYDRAMA2D_COMPILE_OPTIONS})

# Location of the bundled benchmark case (real-mesh tests), of the source tree
# (documentation checks) and of the test data (Gmsh fixtures).
target_compile_definitions(hydrama2d_tests PRIVATE
    HYDRAMA2D_CASE_DIR="${PROJECT_SOURCE_DIR}/Hinterkante.dev.Re100"
    HYDRAMA2D_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    HYDRAMA2D_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

# CGNS: the file-based CGNS tests write meshes with the CGNS library
# (tests/CgnsTestWriter.hpp). Without CGNS they check the "not available"
# path and skip the file tests.
set(_hydrama2d_test_dl_paths "")
if(HYDRAMA2D_HAVE_CGNS)
    target_link_libraries(hydrama2d_tests PRIVATE HydraMa2D::CGNS)
    target_compile_definitions(hydrama2d_tests PRIVATE HYDRAMA2D_HAVE_CGNS=1)
    # A CGNS installation outside the system paths: the dynamic loader must
    # find libcgns and its HDF5 both for the test discovery and for the tests.
    if(CGNS_RUNTIME_DIRS)
        if(CMAKE_VERSION VERSION_LESS 3.22)
            message(WARNING "CGNS is installed in ${CGNS_RUNTIME_DIRS}: add it to the library search path "
                            "(e.g. LD_LIBRARY_PATH) before building and running the tests (CMake < 3.22)")
        else()
            set(_hydrama2d_test_dl_paths DL_PATHS ${CGNS_RUNTIME_DIRS})
        endif()
    endif()

    # Regenerates tests/data/cgns/*.cgns (not built by default):
    #   cmake --build build --target hydrama2d_cgns_fixtures
    #   cd tests/data/cgns && ../../../build/tests/hydrama2d_cgns_fixtures
    add_executable(hydrama2d_cgns_fixtures EXCLUDE_FROM_ALL data/cgns/generate_fixtures.cpp)
    target_include_directories(hydrama2d_cgns_fixtures PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(hydrama2d_cgns_fixtures PRIVATE hydrama2d_core HydraMa2D::CGNS)
    target_compile_definitions(hydrama2d_cgns_fixtures PRIVATE HYDRAMA2D_HAVE_CGNS=1)
    target_compile_options(hydrama2d_cgns_fixtures PRIVATE ${HYDRAMA2D_COMPILE_OPTIONS})
endif()

# Register every TEST_CASE as an individual CTest test. The timeout turns a
# hanging test (e.g. a solver that no longer converges) into a failure; the
# slowest test takes ~1 s, so 120 s leaves ample room for sanitizer builds.
# SKIP_RETURN_CODE: a Catch2 SKIP() (exit code 4, e.g. the CGNS file tests of
# a build without CGNS) is reported as skipped, not as failed.
catch_discover_tests(hydrama2d_tests PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 4 ${_hydrama2d_test_dl_paths})

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME polar_sweeps_tools COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_polar_tools.py)
endif()
