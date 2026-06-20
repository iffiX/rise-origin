cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)

################################################################
# Backend compile options
################################################################
if (RISE_ORIGIN_TEST_DEVICE_BACKEND STREQUAL "cuda")
    set(RISE_TEST_DEVICE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr --generate-line-info>)
else ()
    set(RISE_TEST_DEVICE_OPTIONS "")
endif ()

################################################################
# Sources
################################################################
rise_find_sources(RS_RISE_ORIGIN_TEST_SRC
        FILES test_main.cpp
        DIRS device device/data device/algorithm utils/math utils/concurrency utils/serialization
        EXTENSIONS cu cpp)

# cpu backend compiles .cu as C++; cuda leaves them to nvcc.
rise_adapt_sources(RS_RISE_ORIGIN_TEST_SRC BACKEND ${RISE_ORIGIN_TEST_DEVICE_BACKEND})

################################################################
# rise_origin_test executable
################################################################
add_executable(rise_origin_test ${RS_RISE_ORIGIN_TEST_SRC})
target_link_libraries(rise_origin_test PUBLIC
        gtest
        gtest_main
        rise_origin_core)
target_include_directories(rise_origin_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(rise_origin_test PRIVATE
        $<$<STREQUAL:${RISE_ORIGIN_TEST_DEVICE_BACKEND},cuda>:RS_TEST_DEVICE_BACKEND_CUDA>)
target_compile_options(rise_origin_test PUBLIC ${RISE_TEST_DEVICE_OPTIONS})
