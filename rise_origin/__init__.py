# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)

from importlib.metadata import PackageNotFoundError, version

try:
    from rise_origin_python_core import *  # noqa: F401,F403
except ImportError as e:
    raise ImportError(
        "rise-origin requires a compute backend. Install one of:\n"
        "    pip install 'rise-origin[cpu]'      # CPU-only\n"
        "    pip install 'rise-origin[cudaX]'    # X = your CUDA major version, e.g. 12, 13\n"
        f"\nOriginal error: {e}"
    ) from e

try:
    __version__ = version("rise-origin")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

del version, PackageNotFoundError
