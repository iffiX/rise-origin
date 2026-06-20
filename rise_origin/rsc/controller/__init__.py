# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Controller characterizations.

A controller is the I/O endpoint that reads records and drives signals.
The callback controller exchanges them with the in-process world callback.

Example::

    from rise_origin import rsc

    pilot = rsc.controller.ControllerCallback("pilot")
"""

from rise_origin import RSC_ControllerBase

from rise_origin.rsc.controller.callback import ControllerCallback

#: Abstract base of every controller characterization (carries name).
ControllerBase = RSC_ControllerBase

__all__ = ["ControllerBase", "ControllerCallback"]
