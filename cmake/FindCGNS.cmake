# FindCGNS
# --------
#
# Finds the CGNS library (https://cgns.github.io) for the optional CGNS mesh
# input of HydraMa2D.
#
# Works with
#   * CGNS 4.x, which installs a CMake package (cgns-config.cmake). It is used
#     only to locate the installation: its static target links a target called
#     "hdf5-shared" that the package does not define;
#   * CGNS 3.x, which installs no CMake package: cgnslib.h and libcgns are
#     searched in the usual places.
#
# Hints: CGNS_ROOT or CMAKE_PREFIX_PATH (installation prefix), cgns_DIR
# (CGNS 4 package directory), CGNS_USE_STATIC_LIBS (prefer libcgns.a).
#
# A shared libcgns loads HDF5 itself. A static libcgns built with HDF5
# (CG_BUILD_HDF5 in cgnsconfig.h) also needs HDF5 at link time, which is
# searched with FindHDF5 (hint: HDF5_ROOT); without it CGNS is not usable.
#
# Result variables
#   CGNS_FOUND, CGNS_VERSION (e.g. 4.5.0), CGNS_INCLUDE_DIR, CGNS_LIBRARY,
#   CGNS_STATIC (ON for libcgns.a), CGNS_BUILT_WITH_HDF5,
#   CGNS_RUNTIME_DIRS  directories the dynamic loader must search to run
#                      programs linked with CGNS (only those outside the
#                      system linker paths; empty for a system installation)
# Imported target
#   HydraMa2D::CGNS

include(FindPackageHandleStandardArgs)

# --- 1. Locate the installation -------------------------------------------
set(_cgns_include_hints "")
set(_cgns_library_hints "")
find_package(cgns CONFIG QUIET)
if(cgns_FOUND)
    foreach(_cgns_target CGNS::cgns_shared CGNS::cgns_static)
        if(TARGET ${_cgns_target})
            get_target_property(_cgns_inc ${_cgns_target} INTERFACE_INCLUDE_DIRECTORIES)
            if(_cgns_inc)
                list(APPEND _cgns_include_hints ${_cgns_inc})
            endif()
            foreach(_cgns_cfg LOCATION IMPORTED_LOCATION IMPORTED_LOCATION_NONE IMPORTED_LOCATION_RELEASE
                              IMPORTED_LOCATION_RELWITHDEBINFO IMPORTED_LOCATION_DEBUG)
                get_target_property(_cgns_loc ${_cgns_target} ${_cgns_cfg})
                if(_cgns_loc)
                    get_filename_component(_cgns_loc_dir "${_cgns_loc}" DIRECTORY)
                    list(APPEND _cgns_library_hints "${_cgns_loc_dir}")
                endif()
            endforeach()
        endif()
    endforeach()
endif()

find_path(CGNS_INCLUDE_DIR NAMES cgnslib.h HINTS ${_cgns_include_hints} PATH_SUFFIXES cgns)

if(CGNS_USE_STATIC_LIBS)
    set(_cgns_names ${CMAKE_STATIC_LIBRARY_PREFIX}cgns${CMAKE_STATIC_LIBRARY_SUFFIX} cgns)
else()
    set(_cgns_names cgns cgnsdll)
endif()
find_library(CGNS_LIBRARY NAMES ${_cgns_names} HINTS ${_cgns_library_hints})

# --- 2. Version and configuration ------------------------------------------
set(CGNS_VERSION "")
set(CGNS_BUILT_WITH_HDF5 OFF)
if(CGNS_INCLUDE_DIR AND EXISTS "${CGNS_INCLUDE_DIR}/cgnslib.h")
    file(STRINGS "${CGNS_INCLUDE_DIR}/cgnslib.h" _cgns_version_line REGEX "^#define[ \t]+CGNS_VERSION[ \t]+[0-9]+")
    if(_cgns_version_line MATCHES "CGNS_VERSION[ \t]+([0-9]+)")
        set(_cgns_v "${CMAKE_MATCH_1}")                     # e.g. 4500
        math(EXPR _cgns_major "${_cgns_v} / 1000")
        math(EXPR _cgns_minor "(${_cgns_v} / 100) % 10")
        math(EXPR _cgns_patch "(${_cgns_v} / 10) % 10")
        set(CGNS_VERSION "${_cgns_major}.${_cgns_minor}.${_cgns_patch}")
    endif()
    # HDF5 support: CGNS 3.x defines CG_BUILD_HDF5; CGNS 4.x dropped it and
    # records the HDF5 library it was built with in HDF5_LIBRARY ("" without).
    if(EXISTS "${CGNS_INCLUDE_DIR}/cgnsconfig.h")
        file(STRINGS "${CGNS_INCLUDE_DIR}/cgnsconfig.h" _cgns_hdf5_build REGEX "^#define[ \t]+CG_BUILD_HDF5[ \t]+[0-9]+")
        file(STRINGS "${CGNS_INCLUDE_DIR}/cgnsconfig.h" _cgns_hdf5_lib REGEX "^#define[ \t]+HDF5_LIBRARY[ \t]+\"")
        if(_cgns_hdf5_build)
            if(_cgns_hdf5_build MATCHES "CG_BUILD_HDF5[ \t]+1")
                set(CGNS_BUILT_WITH_HDF5 ON)
            endif()
        elseif(_cgns_hdf5_lib AND NOT _cgns_hdf5_lib MATCHES "HDF5_LIBRARY[ \t]+\"\"")
            set(CGNS_BUILT_WITH_HDF5 ON)
        endif()
    endif()
endif()

# Static library? (On Windows ".lib" is also the import library of a DLL:
# there only CGNS_USE_STATIC_LIBS selects a static libcgns.)
set(CGNS_STATIC OFF)
if(CGNS_LIBRARY)
    get_filename_component(_cgns_ext "${CGNS_LIBRARY}" LAST_EXT)
    if(_cgns_ext STREQUAL CMAKE_STATIC_LIBRARY_SUFFIX AND (NOT WIN32 OR CGNS_USE_STATIC_LIBS))
        set(CGNS_STATIC ON)
    endif()
endif()

# --- 3. HDF5 for a static libcgns --------------------------------------------
set(_cgns_link_ok TRUE)
set(_cgns_hdf5_libraries "")
if(CGNS_LIBRARY AND CGNS_STATIC AND CGNS_BUILT_WITH_HDF5)
    find_package(HDF5 COMPONENTS C QUIET)
    if(HDF5_FOUND)
        set(_cgns_hdf5_libraries ${HDF5_C_LIBRARIES})
    else()
        set(_cgns_link_ok FALSE)
        if(NOT CGNS_FIND_QUIETLY)
            message(STATUS "CGNS: ${CGNS_LIBRARY} is a static library built with HDF5, but HDF5 was not found "
                           "(set HDF5_ROOT)")
        endif()
    endif()
endif()

find_package_handle_standard_args(CGNS
    REQUIRED_VARS CGNS_LIBRARY CGNS_INCLUDE_DIR _cgns_link_ok
    VERSION_VAR CGNS_VERSION)

# --- 4. Imported target and runtime directories -------------------------------
set(CGNS_RUNTIME_DIRS "")
if(CGNS_FOUND)
    if(NOT TARGET HydraMa2D::CGNS)
        add_library(HydraMa2D::CGNS UNKNOWN IMPORTED)
        set_target_properties(HydraMa2D::CGNS PROPERTIES
            IMPORTED_LOCATION "${CGNS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CGNS_INCLUDE_DIR}")
        if(_cgns_hdf5_libraries)
            set_property(TARGET HydraMa2D::CGNS APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${_cgns_hdf5_libraries})
        endif()
    endif()
    # A shared libcgns outside the system paths: the loader must find it and
    # its own dependencies (HDF5, usually installed next to it).
    if(NOT CGNS_STATIC)
        get_filename_component(_cgns_dir "${CGNS_LIBRARY}" DIRECTORY)
        get_filename_component(_cgns_dir "${_cgns_dir}" REALPATH)
        set(_cgns_system_dirs ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES} ${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES})
        set(_cgns_is_system FALSE)
        foreach(_cgns_sys ${_cgns_system_dirs})
            get_filename_component(_cgns_sys "${_cgns_sys}" REALPATH)
            if(_cgns_sys STREQUAL _cgns_dir)
                set(_cgns_is_system TRUE)
            endif()
        endforeach()
        if(NOT _cgns_is_system)
            list(APPEND CGNS_RUNTIME_DIRS "${_cgns_dir}")
        endif()
    endif()
endif()

mark_as_advanced(CGNS_INCLUDE_DIR CGNS_LIBRARY)
