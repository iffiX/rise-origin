cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)

################################################################
# Backend compile options
################################################################
if (RISE_ORIGIN_BACKEND STREQUAL "cuda")
    set(RISE_DEVICE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -maxrregcount=80 --expt-relaxed-constexpr --generate-line-info>)
    set(RISE_DEVICE_LIBS CUDA::cudart)
else ()
    set(RISE_DEVICE_OPTIONS "")
    set(RISE_DEVICE_LIBS "")
endif ()

################################################################
# Sources
################################################################
rise_find_sources(RS_RISE_ORIGIN_CORE_SRC RECURSIVE DIRS core EXTENSIONS cu)
rise_find_sources(RS_RISE_ORIGIN_DEVICE_SRC RECURSIVE DIRS device EXTENSIONS cu)
rise_find_sources(RS_RISE_ORIGIN_RSC_SRC RECURSIVE DIRS rsc EXTENSIONS cpp)
rise_find_sources(RS_RISE_ORIGIN_INTERFACE_SRC RECURSIVE DIRS interface EXTENSIONS cpp)
rise_find_sources(RS_RISE_ORIGIN_PYTHON_SRC RECURSIVE DIRS python EXTENSIONS cpp)

# cpu backend compiles .cu as C++; cuda leaves them to nvcc.
rise_adapt_sources(RS_RISE_ORIGIN_CORE_SRC BACKEND ${RISE_ORIGIN_BACKEND})
rise_adapt_sources(RS_RISE_ORIGIN_DEVICE_SRC BACKEND ${RISE_ORIGIN_BACKEND})

################################################################
# rise_origin_core library
################################################################
add_library(rise_origin_core SHARED
        ${RS_RISE_ORIGIN_CORE_SRC}
        ${RS_RISE_ORIGIN_DEVICE_SRC}
        ${RS_RISE_ORIGIN_RSC_SRC}
        utils/serialization/rs_fast_pack.cpp
        rise_origin.cpp)
target_link_libraries(rise_origin_core PUBLIC
        ${HDF5_LIBRARIES}
        Boost::filesystem
        Boost::program_options
        Boost::fiber
        Boost::context
        TBB::tbb
        fmt::fmt
        spdlog::spdlog
        hosek-wilkie
        libzstd_static
        $<$<BOOL:${MINGW}>:ws2_32>
        ${RISE_DEVICE_LIBS})
target_include_directories(rise_origin_core PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/highfive/include>
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${HDF5_INCLUDE_DIRS})
target_compile_options(rise_origin_core PUBLIC ${RISE_DEVICE_OPTIONS}
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/O2 /fp:fast ${RISE_CPU_ARCH_FLAG}>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3 -ffast-math ${RISE_CPU_ARCH_FLAG}>)

################################################################
# VR session component: OpenXR + Vulkan
################################################################
# The VR component (rs_world_session_component_vr.cu) compiles to nothing unless
# this macro is set, mirroring the device-backend gating, so the default build
# carries no OpenXR or Vulkan dependency.
if (RISE_ORIGIN_BUILD_CORE_WITH_VR)
    target_compile_definitions(rise_origin_core PUBLIC RISE_ORIGIN_BUILD_CORE_WITH_VR)
    target_compile_definitions(rise_origin_core PRIVATE
        RS_VR_HUD_FONT_PATH="${CMAKE_SOURCE_DIR}/lib/imgui/font/rajdhani/Rajdhani-Medium.ttf")
    target_link_libraries(rise_origin_core PUBLIC openxr_loader Vulkan::Vulkan imgui)
endif ()

################################################################
# MCP server session component
################################################################
# The MCP component (rs_world_session_component_mcp.cu) compiles to nothing unless
# this macro is set. It uses the header-only cpp-httplib and stb_image_write in
# lib/, so it links no new library except the Windows socket library.
if (RISE_ORIGIN_BUILD_CORE_WITH_MCP)
    target_compile_definitions(rise_origin_core PUBLIC RISE_ORIGIN_BUILD_CORE_WITH_MCP)
    if (WIN32)
        target_link_libraries(rise_origin_core PUBLIC ws2_32)
    endif ()
endif ()

################################################################
# MSVC + nvcc conformance flags
################################################################
# MSVC needs explicit math constants, no Windows.h min/max macros, no Windows.h
# GDI macros, the standard-conformant preprocessor/lambda processors, and a
# larger object section limit for the templated CUDA TUs.
if (MSVC)
    target_compile_definitions(rise_origin_core PUBLIC
            NOMINMAX NOGDI WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
    target_compile_options(rise_origin_core PUBLIC
            $<$<COMPILE_LANGUAGE:CXX>:/bigobj /utf-8 /EHsc /Zc:preprocessor /Zc:lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj -Xcompiler=/utf-8 -Xcompiler=/Zc:preprocessor -Xcompiler=/Zc:lambda>)
endif ()

################################################################
# Avoiding fatBinData import error caused by lto optimization in pybind11
################################################################
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

################################################################
# rise_origin_instance executable
################################################################
add_executable(rise_origin_instance apps/instance/main.cpp)
target_link_libraries(rise_origin_instance PUBLIC rise_origin_core)

################################################################
# rise_origin_python_core Python module
################################################################
pybind11_add_module(rise_origin_python_core MODULE ${RS_RISE_ORIGIN_PYTHON_SRC})
set_target_properties(rise_origin_python_core PROPERTIES
        OUTPUT_NAME "_core"
        PREFIX ""
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN")
target_link_libraries(rise_origin_python_core PRIVATE rise_origin_core)
target_include_directories(rise_origin_python_core PUBLIC
        ${HDF5_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${PYBIND11_INCLUDE_DIR})
target_include_directories(rise_origin_python_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/python)

################################################################
# backward-cpp for debug builds
################################################################
if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" AND NOT WIN32)
    target_sources(rise_origin_instance PRIVATE ${BACKWARD_ENABLE})
    target_compile_definitions(rise_origin_instance PRIVATE ${BACKWARD_DEFINITIONS})
    target_link_libraries(rise_origin_instance PRIVATE ${BACKWARD_LIBRARIES})

    target_sources(rise_origin_python_core PRIVATE ${BACKWARD_ENABLE})
    target_compile_definitions(rise_origin_python_core PRIVATE ${BACKWARD_DEFINITIONS})
    target_link_libraries(rise_origin_python_core PRIVATE ${BACKWARD_LIBRARIES})
endif ()

set(CMAKE_VERBOSE_MAKEFILE OFF)
