# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Internal helpers shared by the ``rsc`` builders."""


def _apply(config, **fields) -> None:
    """Set each non-None field on a config, leaving the engine default otherwise."""
    for name, value in fields.items():
        if value is not None:
            setattr(config, name, value)
