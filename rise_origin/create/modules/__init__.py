# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""First-party create modules, one file per real-world kind.

Kind files live at ``modules/<area>/<group>/<kind>.py`` and register
themselves through the ``cr.module`` decorator on import; the registry
imports this package once when a name it does not know is asked for.
"""

from .material.nature import (  # noqa: F401 registers on import
    cloud,
    mineral,
    plant,
)
from .nature.cloud import (  # noqa: F401 registers on import
    cumulonimbus,
    cumulonimbus_field,
    cumulus,
    cumulus_field,
    stratus,
    stratus_field,
)
from .nature.rock import (  # noqa: F401 registers on import
    basalt_columns,
    granite_block,
    granite_boulder,
    limestone_block,
    limestone_boulder,
    marble_block,
    marble_boulder,
    sandstone_block,
    sandstone_boulder,
    slate_block,
    slate_boulder,
)
from .nature.tree import (  # noqa: F401 registers on import
    acacia,
    aspen,
    baobab,
    birch,
    cherry,
    coconut_palm,
    cypress,
    dead_snag,
    ginkgo,
    maple,
    oak,
    pine,
    sequoia,
    spruce,
    sycamore,
    tupelo,
    willow,
)
