cmake_minimum_required(VERSION 3.22)

# Hosek-Wilkie analytic sky model. Vendored verbatim from the canonical
# 2013-02-22 release; BSD-3-Clause (c) 2012-2013 Lukas Hosek and Alexander Wilkie.
# See ArHosekSkyModel.h for the upstream license header.
project(hosek-wilkie LANGUAGES C)

add_library(hosek-wilkie STATIC ArHosekSkyModel.c)

set_target_properties(hosek-wilkie PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_STANDARD 99)

target_include_directories(hosek-wilkie PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>)

# Math library is needed for pow/exp/sin/cos used by the cook + radiance code.
target_link_libraries(hosek-wilkie PUBLIC $<$<NOT:$<BOOL:${WIN32}>>:m>)
