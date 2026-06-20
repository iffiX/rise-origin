################################################################
# Bundled libraries
################################################################
rise_add_lib_subdirectory(${CMAKE_SOURCE_DIR}/lib/fmt BINARY_DIR ${CMAKE_BINARY_DIR}/lib/fmt)
rise_add_lib_subdirectory(${CMAKE_SOURCE_DIR}/lib/spdlog BINARY_DIR ${CMAKE_BINARY_DIR}/lib/spdlog)
rise_add_lib_subdirectory(${CMAKE_SOURCE_DIR}/lib/hosek-wilkie BINARY_DIR ${CMAKE_BINARY_DIR}/lib/hosek-wilkie)

if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" AND NOT WIN32)
    rise_add_lib_subdirectory(${CMAKE_SOURCE_DIR}/lib/backward-cpp BINARY_DIR ${CMAKE_BINARY_DIR}/lib/backward-cpp)
endif ()

################################################################
# CUDA
################################################################
if (RISE_ORIGIN_BACKEND STREQUAL "cuda")
    # Minimum sm_70 required (32KB parameter size)
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    find_package(CUDAToolkit REQUIRED)
    if (CUDAToolkit_VERSION VERSION_LESS "12.2")
        message(FATAL_ERROR "Minimum CUDA version required is 12.2, found ${CUDAToolkit_VERSION}")
    endif ()
    message(STATUS "Found CUDA ${CUDAToolkit_VERSION}")

    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif ()

################################################################
# Host libraries
################################################################
find_package(HDF5 REQUIRED)
# CMake >=3.30 (CMP0167) defaults find_package(Boost) to config mode, which
# needs BoostConfig.cmake (Boost >=1.70). The manylinux image's boost-devel is
# 1.66 (no config files), so force the classic FindBoost module to locate it.
# Windows conda/vcpkg Boost ships BoostConfig.cmake, so keep config mode there.
if (POLICY CMP0167 AND NOT WIN32)
    cmake_policy(SET CMP0167 OLD)
endif ()
find_package(Boost REQUIRED COMPONENTS filesystem program_options fiber context)
find_package(TBB REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

if (BUILD_RISE_ORIGIN)
    rise_add_lib_subdirectory(${CMAKE_SOURCE_DIR}/lib/pybind11 BINARY_DIR ${CMAKE_BINARY_DIR}/lib/pybind11)
endif ()
