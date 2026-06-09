# RiseSourceUtils.cmake
# Utilities for collecting and adapting source files across device backends.

# rise_find_sources(<out_var>
#   [RECURSIVE]
#   [FILES  <file1> <file2> ...]
#   [DIRS   <dir1>  <dir2>  ...]
#   [EXTENSIONS <ext1> <ext2> ...]
# )
#
# Collects source files into <out_var> from three optional, composable inputs:
#   FILES      - literal file paths appended verbatim.
#   DIRS + EXTENSIONS
#              - for every (dir, ext) pair runs file(GLOB[_RECURSE] "dir/*.ext").
#   RECURSIVE  - switches the DIRS sweep to file(GLOB_RECURSE ...).
#
# All keyword groups are optional and may be combined freely.
function(rise_find_sources OUT_VAR)
    cmake_parse_arguments(RFS "RECURSIVE" "" "FILES;DIRS;EXTENSIONS" ${ARGN})

    set(_result "")

    # --- literal files ---
    foreach(_f ${RFS_FILES})
        list(APPEND _result "${_f}")
    endforeach()

    # --- glob DIRS x EXTENSIONS ---
    if(RFS_DIRS)
        if(RFS_RECURSIVE)
            set(_glob_mode GLOB_RECURSE)
        else()
            set(_glob_mode GLOB)
        endif()

        foreach(_dir ${RFS_DIRS})
            if(RFS_EXTENSIONS)
                foreach(_ext ${RFS_EXTENSIONS})
                    file(${_glob_mode} _found "${_dir}/*.${_ext}")
                    list(APPEND _result ${_found})
                endforeach()
            else()
                file(${_glob_mode} _found "${_dir}/*")
                list(APPEND _result ${_found})
            endif()
        endforeach()
    endif()

    set(${OUT_VAR} ${_result} PARENT_SCOPE)
endfunction()


# rise_adapt_sources(<list_var> BACKEND <cpu|cuda|hip|sycl>)
#
# Adjusts source-file language properties so that the correct compiler is used
# for each backend without changing the file list itself.
#
#   cpu  - marks every .cu file as LANGUAGE CXX so it is compiled by the C++
#          compiler instead of nvcc.  No CUDA toolchain is required.
#   cuda | hip | sycl - no-op; the files are compiled natively by their toolchain.
function(rise_adapt_sources LIST_VAR)
    cmake_parse_arguments(RAS "" "BACKEND" "" ${ARGN})

    if(NOT RAS_BACKEND)
        message(FATAL_ERROR "rise_adapt_sources: BACKEND argument is required")
    endif()

    if(RAS_BACKEND STREQUAL "cpu")
        foreach(_src ${${LIST_VAR}})
            if(_src MATCHES "\\.cu$")
                set_source_files_properties("${_src}" PROPERTIES LANGUAGE CXX)
            endif()
        endforeach()
    endif()
    # For cuda/hip/sycl the toolchain handles .cu files natively — nothing to do.
endfunction()


# rise_add_lib_subdirectory(<source_dir>
#   [BINARY_DIR <binary_dir>]
#   [EXCLUDE_FROM_ALL]
#   [SYSTEM]
#   [QUIET_DEPRECATION_WARNINGS]
# )
#
# Wrapper around add_subdirectory() that temporarily increases
# CMAKE_MESSAGE_INDENT so nested dependency logs are easier to scan.
function(rise_add_lib_subdirectory SOURCE_DIR)
    if(NOT SOURCE_DIR)
        message(FATAL_ERROR "rise_add_lib_subdirectory: SOURCE_DIR argument is required")
    endif()

    cmake_parse_arguments(RALS
        "EXCLUDE_FROM_ALL;SYSTEM;QUIET_DEPRECATION_WARNINGS"
        "BINARY_DIR"
        ""
        ${ARGN}
    )

    set(_add_subdirectory_args "${SOURCE_DIR}")
    if(RALS_BINARY_DIR)
        list(APPEND _add_subdirectory_args "${RALS_BINARY_DIR}")
    endif()
    if(RALS_EXCLUDE_FROM_ALL)
        list(APPEND _add_subdirectory_args EXCLUDE_FROM_ALL)
    endif()
    if(RALS_SYSTEM)
        list(APPEND _add_subdirectory_args SYSTEM)
    endif()

    if(RALS_QUIET_DEPRECATION_WARNINGS)
        if(DEFINED CMAKE_WARN_DEPRECATED)
            set(_rise_prev_warn_deprecated "${CMAKE_WARN_DEPRECATED}")
            set(_rise_had_warn_deprecated TRUE)
        endif()
        set(CMAKE_WARN_DEPRECATED OFF)

        if(DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
            set(_rise_prev_suppress_dev "${CMAKE_SUPPRESS_DEVELOPER_WARNINGS}")
            set(_rise_had_suppress_dev TRUE)
        endif()
        set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON)
    endif()

    list(APPEND CMAKE_MESSAGE_INDENT ">>  ")
    add_subdirectory(${_add_subdirectory_args})
    list(POP_BACK CMAKE_MESSAGE_INDENT)

    if(RALS_QUIET_DEPRECATION_WARNINGS)
        if(_rise_had_warn_deprecated)
            set(CMAKE_WARN_DEPRECATED "${_rise_prev_warn_deprecated}")
        else()
            unset(CMAKE_WARN_DEPRECATED)
        endif()
        unset(_rise_prev_warn_deprecated)
        unset(_rise_had_warn_deprecated)

        if(_rise_had_suppress_dev)
            set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS "${_rise_prev_suppress_dev}")
        else()
            unset(CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
        endif()
        unset(_rise_prev_suppress_dev)
        unset(_rise_had_suppress_dev)
    endif()
endfunction()
