# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Signal characterizations.

A signal is a per-step value an entity reads to drive its behavior, produced by
a controller. The mechanical signals inject an external force or torque onto a
structure.

Example::

    from rise_origin import rsc

    push = rsc.signal.Force(
        "push", structure_name="cube", source_controller_reference="pilot",
        body_sid=0, segment_bid=0,
    )
"""

from rise_origin import RSC_SignalBase

from rise_origin.rsc.signal.mechanical import Force, Torque, Transform

#: Abstract base of every signal characterization (carries name + source).
SignalBase = RSC_SignalBase

__all__ = ["SignalBase", "Force", "Torque", "Transform"]
