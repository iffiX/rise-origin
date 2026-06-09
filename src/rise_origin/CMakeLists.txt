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
rise_find_sources(RS_RISE_ORIGIN_CORE_SRC      RECURSIVE DIRS core      EXTENSIONS cu)
rise_find_sources(RS_RISE_ORIGIN_DEVICE_SRC    RECURSIVE DIRS device    EXTENSIONS cu)
rise_find_sources(RS_RISE_ORIGIN_RSC_SRC       RECURSIVE DIRS rsc       EXTENSIONS cpp)
rise_find_sources(RS_RISE_ORIGIN_INTERFACE_SRC RECURSIVE DIRS interface EXTENSIONS cpp)
rise_find_sources(RS_RISE_ORIGIN_PYTHON_SRC              DIRS python    EXTENSIONS cpp)

# cpu backend compiles .cu as C++; cuda leaves them to nvcc.
rise_adapt_sources(RS_RISE_ORIGIN_CORE_SRC   BACKEND ${RISE_ORIGIN_BACKEND})
rise_adapt_sources(RS_RISE_ORIGIN_DEVICE_SRC BACKEND ${RISE_ORIGIN_BACKEND})

################################################################
# rise_origin_core library
################################################################
add_library(rise_origin_core
        ${RS_RISE_ORIGIN_CORE_SRC}
        ${RS_RISE_ORIGIN_DEVICE_SRC}
        ${RS_RISE_ORIGIN_RSC_SRC}
        rise_origin.cpp)
target_link_libraries(rise_origin_core PUBLIC
        ${HDF5_LIBRARIES}
        ${Boost_LIBRARIES}
        TBB::tbb
        fmt::fmt
        spdlog::spdlog
        hosek-wilkie
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
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=${RISE_ORIGIN_CPU_ARCH} -ffast-math>)

################################################################
# Avoiding fatBinData import error caused by lto optimization in pybind11
################################################################
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

################################################################
# rise_origin_main executable
################################################################
add_executable(rise_origin_main executables/main.cpp)
target_link_libraries(rise_origin_main PUBLIC rise_origin_core)

################################################################
# rise_origin_python_core Python module
################################################################
pybind11_add_module(rise_origin_python_core MODULE ${RS_RISE_ORIGIN_PYTHON_SRC})
set_target_properties(rise_origin_python_core PROPERTIES
        OUTPUT_NAME "_core"
        PREFIX "")
target_link_libraries(rise_origin_python_core PRIVATE rise_origin_core)
target_include_directories(rise_origin_python_core PUBLIC
        ${HDF5_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${PYBIND11_INCLUDE_DIR})

################################################################
# backward-cpp for debug builds
################################################################
if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    target_sources(rise_origin_main PRIVATE ${BACKWARD_ENABLE})
    target_compile_definitions(rise_origin_main PRIVATE ${BACKWARD_DEFINITIONS})
    target_link_libraries(rise_origin_main PRIVATE ${BACKWARD_LIBRARIES})

    target_sources(rise_origin_python_core PRIVATE ${BACKWARD_ENABLE})
    target_compile_definitions(rise_origin_python_core PRIVATE ${BACKWARD_DEFINITIONS})
    target_link_libraries(rise_origin_python_core PRIVATE ${BACKWARD_LIBRARIES})
endif ()

set(CMAKE_VERBOSE_MAKEFILE OFF)
