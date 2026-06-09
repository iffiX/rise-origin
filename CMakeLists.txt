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
# Dependencies
################################################################
if (BUILD_RISE_ORIGIN OR BUILD_RISE_ORIGIN_TEST)
    include(cmake/RiseDependencies.cmake)
endif ()

################################################################
# Main builds
################################################################
if (BUILD_RISE_ORIGIN)
    message(STATUS "rise_origin is configured to be built")
    add_subdirectory(src/rise_origin)
else ()
    message(STATUS "rise_origin is configured to be not built")
endif ()

if (BUILD_RISE_ORIGIN_TEST)
    message(STATUS "rise_origin_test is configured to be built")
    add_subdirectory(test)
else ()
    message(STATUS "rise_origin_test is configured to be not built")
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
