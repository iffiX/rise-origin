cmake_minimum_required(VERSION 3.22...3.30)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)

# we need -fPIC globally (for python module)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set build type to Release by default
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif ()

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

include(cmake/RiseOptions.cmake)
include(cmake/RiseTestOptions.cmake)

# Backend selects enabled languages; resolved before project().
if (RISE_ORIGIN_BACKEND STREQUAL "cuda")
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "${RISE_ORIGIN_CUDA_ARCHITECTURES}")
    endif ()
    set(_rise_languages C CXX CUDA)
elseif (RISE_ORIGIN_BACKEND STREQUAL "cpu")
    set(_rise_languages C CXX)
else ()
    message(FATAL_ERROR "Unknown RISE_ORIGIN_BACKEND '${RISE_ORIGIN_BACKEND}', expected cuda or cpu")
endif ()

project(rise_origin LANGUAGES ${_rise_languages})
include(cmake/RiseSourceUtils.cmake)

################################################################
# Compiler flags
################################################################
# -march=native is GCC/Clang only and invalid for MSVC, so map the requested
# arch onto each compiler's flag. The result is consumed by the build targets.
if (MSVC)
    if (RISE_ORIGIN_CPU_ARCH MATCHES "^/arch:")
        set(RISE_CPU_ARCH_FLAG "${RISE_ORIGIN_CPU_ARCH}")
    elseif (RISE_ORIGIN_CPU_ARCH MATCHES "v4$")
        set(RISE_CPU_ARCH_FLAG "/arch:AVX512")
    elseif (RISE_ORIGIN_CPU_ARCH MATCHES "v3$")
        set(RISE_CPU_ARCH_FLAG "/arch:AVX2")
    else ()
        set(RISE_CPU_ARCH_FLAG "")
    endif ()
else ()
    set(RISE_CPU_ARCH_FLAG "-march=${RISE_ORIGIN_CPU_ARCH}")
endif ()

################################################################
# Dependencies
################################################################
if (BUILD_RISE_ORIGIN OR BUILD_RISE_ORIGIN_TEST)
    include(cmake/RiseDependencies.cmake)
endif ()

################################################################
# Main builds
################################################################
if (BUILD_RISE_ORIGIN)
    message(STATUS "rise origin is configured to be built")
    add_subdirectory(src/rise_origin)
else ()
    message(STATUS "rise origin is not configured to be built")
endif ()

if (BUILD_RISE_ORIGIN_TEST)
    message(STATUS "rise origin tests are configured to be built (backend: ${RISE_ORIGIN_TEST_DEVICE_BACKEND})")
    add_subdirectory(test)
else ()
    message(STATUS "rise origin tests are not configured to be built")
endif ()

################################################################
# Header-only interface for downstream consumers
################################################################
add_library(rise_origin_header INTERFACE)
target_include_directories(rise_origin_header INTERFACE include)

################################################################
# Installation
################################################################
if (BUILD_RISE_ORIGIN)
    include(cmake/RiseInstall.cmake)
endif ()
