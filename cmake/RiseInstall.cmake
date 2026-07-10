include(GNUInstallDirs)

################################################################
# rise_origin installation rules
#
# SKBUILD defined  -> wheel build (scikit-build-core from rise_origin/cuda*/)
# SKBUILD undefined -> conda / system install
################################################################

if (DEFINED SKBUILD)
    # Python low-level extension module (wheel layout)
    install(TARGETS rise_origin_python_core
            LIBRARY DESTINATION rise_origin_python_core
            RUNTIME DESTINATION rise_origin_python_core)
else ()
    # Core library
    install(TARGETS rise_origin_core
            EXPORT rise_origin_targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    # Applications
    install(TARGETS rise_origin_instance
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # Public headers
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/rise_origin/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rise_origin
            FILES_MATCHING PATTERN "*.h")

    # Python low-level extension module and type stubs
    if (NOT DEFINED RISE_ORIGIN_PYTHON_INSTALL_DIR)
        execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
                OUTPUT_VARIABLE RISE_ORIGIN_PYTHON_INSTALL_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif ()
    install(TARGETS rise_origin_python_core
            LIBRARY DESTINATION ${RISE_ORIGIN_PYTHON_INSTALL_DIR}/rise_origin_python_core
            RUNTIME DESTINATION ${RISE_ORIGIN_PYTHON_INSTALL_DIR}/rise_origin_python_core)
endif ()
