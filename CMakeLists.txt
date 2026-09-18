cmake_minimum_required(VERSION 3.18)

project( SurfATT VERSION 2.1.2 LANGUAGES CXX Fortran)

include(GNUInstallDirs)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

configure_file(
  ${PROJECT_SOURCE_DIR}/include/version.h.in
  ${PROJECT_SOURCE_DIR}/include/version.h
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

option(SURFATT_DEFAULT_RELEASE "Use Release build type when not specified." ON)
if (SURFATT_DEFAULT_RELEASE AND NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release)

option(SURFATT_NATIVE_ARCH "Optimize for the build machine's CPU architecture." OFF)
set(SURFATT_NATIVE_ARCH_FLAG "")
if(SURFATT_NATIVE_ARCH)
  set(SURFATT_NATIVE_ARCH_FLAG "-march=native")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

message(STATUS "Detecting CXX Compiler: ${CMAKE_CXX_COMPILER_ID}")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG ${SURFATT_NATIVE_ARCH_FLAG} -Wno-type-limits")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  # New Intel DPC++/C++ Compiler (icpx) based on LLVM; supports Clang-style warnings.
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG ${SURFATT_NATIVE_ARCH_FLAG} -Wno-tautological-constant-compare -DEIGEN_STRONG_INLINE=inline")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # Classic Intel C++ Compiler (icpc); does NOT support -Wtautological-constant-compare.
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG ${SURFATT_NATIVE_ARCH_FLAG} -diag-disable=10441,11016 -DEIGEN_STRONG_INLINE=inline")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

else ()
  # Fallback defaults for other compilers (e.g., MSVC, native Clang, etc.).
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG ${SURFATT_NATIVE_ARCH_FLAG}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3")
endif()

# Note: Ensure the project enables Fortran support (e.g., project(MyProject LANGUAGES CXX Fortran)) to retrieve this ID.
message(STATUS "Detecting Fortran Compiler: ${CMAKE_Fortran_COMPILER_ID}")

if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -O3 -DNDEBUG ${SURFATT_NATIVE_ARCH_FLAG} -ffixed-line-length-none")
  set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -O0 -g3 -ffixed-line-length-none")

elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
  # New Intel Fortran Compiler (ifx)
  set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -O3 -DNDEBUG ${SURFATT_NATIVE_ARCH_FLAG} -extend-source")
  set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -O0 -g -extend-source")

elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  # Classic Intel Fortran Compiler (ifort)
  set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -O3 -DNDEBUG ${SURFATT_NATIVE_ARCH_FLAG} -diag-disable=10441 -extend-source")
  set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -O0 -g -diag-disable=10441 -extend-source")

else ()
  # Fallback defaults for other Fortran compilers, defaulting to GNU-style formatting.
  set(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -O3 -DNDEBUG ${SURFATT_NATIVE_ARCH_FLAG} -ffixed-line-length-none")
  set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -O0 -g3 -ffixed-line-length-none")
endif()

# ---------------------------------------------------------------------------
# Precision option  (-DUSE_SINGLE_PRECISION=ON to use float)
# ---------------------------------------------------------------------------
option(USE_SINGLE_PRECISION "Use single-precision (float) throughout" OFF)
if (USE_SINGLE_PRECISION)
    add_compile_definitions(USE_SINGLE_PRECISION)
    message(STATUS "Precision: float (single)")
else()
    message(STATUS "Precision: double (default)")
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
option(USE_EXTERNAL_LIBS
    "Use bundled yaml-cpp and spdlog from external_libs submodules."
    OFF
)

# MPI (required)
find_package(MPI REQUIRED)
message(STATUS "MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
message(STATUS "MPI_CXX_INCLUDE_DIRS: ${MPI_CXX_INCLUDE_DIRS}")

# find HDF5 parallel
message(STATUS "Running CMAKE FindHDF5.cmake...")
# set parallel HDF5 default
find_package(HDF5 REQUIRED COMPONENTS C CXX)
if(HDF5_FOUND)
  if(TARGET hdf5::hdf5_cpp-shared AND TARGET hdf5::hdf5-shared)
      set(MY_HDF5_LIBS hdf5::hdf5_cpp-shared hdf5::hdf5-shared)
  elseif(TARGET hdf5::hdf5_cpp AND TARGET hdf5::hdf5)
      set(MY_HDF5_LIBS hdf5::hdf5_cpp hdf5::hdf5)
  elseif(TARGET hdf5::hdf5_cpp-static AND TARGET hdf5::hdf5-static)
      set(MY_HDF5_LIBS hdf5::hdf5_cpp-static hdf5::hdf5-static)
  elseif(TARGET hdf5-shared AND TARGET hdf5_cpp-shared)
      set(MY_HDF5_LIBS hdf5_cpp-shared hdf5-shared)
  elseif(TARGET hdf5 AND TARGET hdf5_cpp)
      set(MY_HDF5_LIBS hdf5_cpp hdf5)
  else()
    set(MY_HDF5_LIBS ${HDF5_LIBRARIES})
    if("${MY_HDF5_LIBS}" STREQUAL "")
        set(MY_HDF5_LIBS ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES})
    endif()
  endif()
  message(STATUS "HDF5_FOUND: ${HDF5_FOUND}")
  message(STATUS "MY_HDF5_LIBS: ${MY_HDF5_LIBS}")
  add_definitions(-DUSE_HDF5)
  include_directories(${HDF5_INCLUDE_DIRS})
endif()

# submodules
add_subdirectory(external_libs)

# yaml-cpp: prefer a system installation, fall back to the bundled submodule.
if (NOT USE_EXTERNAL_LIBS)
    find_package(yaml-cpp QUIET)
endif()

if (yaml-cpp_FOUND AND NOT USE_EXTERNAL_LIBS)
    message(STATUS "Found system yaml-cpp: ${yaml-cpp_DIR}")
    # Normalize to yaml-cpp::yaml-cpp to avoid deprecated target warnings.
    if (NOT TARGET yaml-cpp::yaml-cpp)
        if (TARGET yaml-cpp)
            add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
        else()
            add_library(yaml-cpp::yaml-cpp INTERFACE IMPORTED)
            if (YAML_CPP_INCLUDE_DIR)
                set_property(TARGET yaml-cpp::yaml-cpp APPEND PROPERTY
                    INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE_DIR}")
            endif()
            if (YAML_CPP_LIBRARIES)
                set_property(TARGET yaml-cpp::yaml-cpp APPEND PROPERTY
                    INTERFACE_LINK_LIBRARIES "${YAML_CPP_LIBRARIES}")
            endif()
        endif()
    endif()
else()
    message(STATUS "System yaml-cpp not found (or bundled libraries requested); using external_libs/yaml-cpp")
    initialize_submodule(external_libs/yaml-cpp)
    add_subdirectory(external_libs/yaml-cpp EXCLUDE_FROM_ALL)
endif()

if (NOT TARGET yaml-cpp::yaml-cpp)
    message(FATAL_ERROR "yaml-cpp was found, but no usable CMake target was provided")
endif()

# ------------------------------------------------------
# find spdlog
if (NOT USE_EXTERNAL_LIBS)
    find_package(spdlog QUIET)
endif()

if (spdlog_FOUND AND NOT USE_EXTERNAL_LIBS)
    message(STATUS "Found system spdlog: ${spdlog_DIR}")
else()
    message(STATUS "System spdlog not found (or bundled libraries requested); using external_libs/spdlog")
    initialize_submodule(external_libs/spdlog)
    add_subdirectory(external_libs/spdlog EXCLUDE_FROM_ALL)
endif()

if (TARGET spdlog::spdlog_header_only)
    set(SPDLOG_TARGET spdlog::spdlog_header_only)
elseif(TARGET spdlog::spdlog)
    set(SPDLOG_TARGET spdlog::spdlog)
elseif(TARGET spdlog)
    add_library(spdlog::spdlog ALIAS spdlog)
    set(SPDLOG_TARGET spdlog::spdlog)
else()
    message(FATAL_ERROR "spdlog target not found after discovery")
endif()
message(STATUS "Using spdlog target: ${SPDLOG_TARGET}")

# ---------------------------------------------------------------------------
# Eigen3
# ---------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()
# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

add_subdirectory("src/surfker")
add_subdirectory("src/minpack")
set(PACK_LIBS surfker_obj minpack_obj)

file(GLOB SOURCES "src/*.cpp")
add_library(objs OBJECT ${SOURCES} ${SURFKER_SOURCES})
target_include_directories(objs PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(objs PUBLIC
    Eigen3::Eigen
    yaml-cpp::yaml-cpp
    ${SPDLOG_TARGET}
    MPI::MPI_CXX
    ${MY_HDF5_LIBS}
    ${PACK_LIBS}
)

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
file(GLOB APP_SOURCES "src/*.cxx")

foreach( execsourcefile ${APP_SOURCES} )
    get_filename_component(execname ${execsourcefile} NAME_WE)

    add_executable(${execname} ${execsourcefile})
    
    target_link_libraries(${execname} PRIVATE objs ${PACK_LIBS})

    # eigen
    target_link_libraries(${execname} PRIVATE Eigen3::Eigen)

    # yaml-cpp
    target_link_libraries(${execname} PRIVATE yaml-cpp::yaml-cpp)

    # spdlog
    target_link_libraries(${execname} PRIVATE ${SPDLOG_TARGET})

    # MPI
    target_link_libraries(${execname} PRIVATE MPI::MPI_CXX)

    # HDF5
    target_link_libraries(${execname} PRIVATE ${MY_HDF5_LIBS})

    install(TARGETS ${execname}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endforeach()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
option(BUILD_TESTS "Build test executables" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
