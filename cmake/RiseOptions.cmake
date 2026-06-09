# RiseOptions.cmake
# Centralized build options for rise_origin. Run `cmake -LH` for the full list.
# Included before project() so the backend can choose the enabled languages.

################################################################
# Device backend
################################################################
set(RISE_ORIGIN_BACKEND "cpu" CACHE STRING "Compute backend: cpu | cuda")
set_property(CACHE RISE_ORIGIN_BACKEND PROPERTY STRINGS cpu cuda)

################################################################
# Build targets
################################################################
option(BUILD_RISE_ORIGIN "Build the rise_origin simulator and libraries" ON)
option(BUILD_RISE_ORIGIN_TEST "Build the rise_origin test suite" ON)

################################################################
# CUDA backend settings, used when RISE_ORIGIN_BACKEND=cuda
################################################################
set(RISE_ORIGIN_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA SM architectures, semicolon-separated")

################################################################
# CPU backend settings, used when RISE_ORIGIN_BACKEND=cpu
################################################################
set(RISE_ORIGIN_CPU_ARCH "native" CACHE STRING "CPU -march for the cpu backend")
