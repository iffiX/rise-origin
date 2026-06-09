#!/bin/bash
set -ex

# Parallelize the CMake/nvcc build. Honor a caller-set level (to throttle peak
# RAM on memory-heavy CUDA-13 multi-arch builds), else use all allotted cores.
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${CPU_COUNT:-$(nproc)}}"

# Select the backend matching the cuda_version variant (12.x -> cuda12, 13.x -> cuda13).
BACKEND="${SRC_DIR}/rise_origin/cuda${cuda_version%%.*}"

$PYTHON -m pip install "${BACKEND}" --no-deps --no-build-isolation -v
$PYTHON -m pip install "${SRC_DIR}" --no-deps --no-build-isolation -v
