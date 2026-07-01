cmake_minimum_required(VERSION 3.18)
project(imgui LANGUAGES CXX)

find_package(Vulkan REQUIRED)

add_library(imgui STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui_draw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui_widgets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)

target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/backends
)

target_link_libraries(imgui PUBLIC Vulkan::Vulkan)

set_target_properties(imgui PROPERTIES POSITION_INDEPENDENT_CODE ON)
