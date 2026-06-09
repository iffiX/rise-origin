cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)

################################################################
# CUDA compile options
################################################################
set(CUDA_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr --generate-line-info>)

################################################################
# Main builds
################################################################
file(GLOB RS_RISE_ORIGIN_TEST_SRC test_main.cpp device/data/*.cu device/algorithm/*.cu device/*.cu utils/*.cpp utils/*.cu)
add_executable(rise_origin_test ${RS_RISE_ORIGIN_TEST_SRC})
target_include_directories(rise_origin_test PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/highfive/include>
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${HDF5_INCLUDE_DIRS})
target_link_libraries(rise_origin_test PUBLIC
        gtest
        gtest_main
        ${HDF5_LIBRARIES}
        ${Boost_LIBRARIES}
        TBB::tbb
        fmt::fmt
        spdlog::spdlog
        $<$<BOOL:${MINGW}>:ws2_32>)
target_compile_options(rise_origin_test PUBLIC ${CUDA_OPTIONS})
