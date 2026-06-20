# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Callback controller: routes records and signals through the world callback."""

from rise_origin import RSC_ControllerCallback

__all__ = ["ControllerCallback"]


class ControllerCallback(RSC_ControllerCallback):
    """Delivers its routed records to, and reads its signals from, the in-process
    world callback.

    Records naming this controller in their ``target_controller_reference`` are
    bundled and passed to the callback each callback step; signals naming this
    controller in their ``source_controller_name`` are driven by what the
    callback returns.

    Args:
        name: Unique controller name referenced by records and signals.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
