# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)

import ctypes
import importlib.util
import os


def _preload(component: str, soname: str, dllname: str) -> None:
    # Load the CUDA runtime so _core's dependency on it resolves. Prefer the
    # pip-installed runtime (nvidia.<component>), then fall back to the system
    # loader. On Windows the runtime lives under nvidia/<component>/bin and the
    # DLL search path must be extended explicitly (os.add_dll_directory).
    if os.name == "nt":
        try:
            spec = importlib.util.find_spec(f"nvidia.{component}")
            if spec and spec.submodule_search_locations:
                bin_dir = os.path.join(list(spec.submodule_search_locations)[0], "bin")
                if os.path.isdir(bin_dir):
                    os.add_dll_directory(bin_dir)
        except Exception:
            pass
        try:
            ctypes.WinDLL(dllname)
        except OSError:
            pass  # let the _core import below surface the missing-library error
        return
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


_preload("cuda_runtime", "libcudart.so.12", "cudart64_12.dll")

from ._core import *  # noqa: F401,F403,E402
