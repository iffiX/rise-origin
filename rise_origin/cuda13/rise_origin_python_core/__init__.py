# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)

import ctypes
import importlib.util
import os


def _preload(component: str, soname: str) -> None:
    # Load the CUDA runtime into the global symbol scope so _core.so's NEEDED
    # entry resolves. Prefer the pip-installed runtime (nvidia.<component>),
    # then fall back to the system loader (LD_LIBRARY_PATH / ldconfig).
    try:
        spec = importlib.util.find_spec(f"nvidia.{component}")
        if spec and spec.submodule_search_locations:
            so = os.path.join(list(spec.submodule_search_locations)[0], "lib", soname)
            if os.path.exists(so):
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                return
    except Exception:
        pass
    try:
        ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass  # let the _core import below surface the missing-library error


_preload("cu13", "libcudart.so.13")

from ._core import *  # noqa: F401,F403,E402
