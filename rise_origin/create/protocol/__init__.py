# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The module protocol: settings, the deterministic stream, the decorator.

A module author declares typed settings, decorates a plain function with
``module``, and receives an editor access and validated settings; building
happens only inside ``editor.create``. ``list_modules`` and ``help_module``
answer what is installed and how to call it.
"""

from .types import (
    Bool,
    Bounds,
    Choice,
    Float,
    FloatList,
    VoxelSize,
    Int,
    Settings,
    SettingsError,
    SettingsSchema,
    Str,
    settings,
)
from .logger import Logger, LoggerProgress
from .rng import Rng
from .worker_pool import WorkerPool
from .module import (
    ModuleError,
    ModuleSpec,
    help_module,
    list_modules,
    module,
)

__all__ = [
    "Bool",
    "Bounds",
    "Choice",
    "Float",
    "FloatList",
    "VoxelSize",
    "Int",
    "Settings",
    "SettingsError",
    "SettingsSchema",
    "Str",
    "settings",
    "Logger",
    "LoggerProgress",
    "Rng",
    "WorkerPool",
    "ModuleError",
    "ModuleSpec",
    "help_module",
    "list_modules",
    "module",
]
