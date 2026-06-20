# RiseTestOptions.cmake
# Test-specific options for the rise_origin project.

################################################################
# Test device backend
################################################################
set(RISE_ORIGIN_TEST_DEVICE_BACKEND "${RISE_ORIGIN_BACKEND}" CACHE STRING "Device backend for tests: cpu | cuda")
set_property(CACHE RISE_ORIGIN_TEST_DEVICE_BACKEND PROPERTY STRINGS cpu cuda)
