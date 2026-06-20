# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Simulation characterization.

The simulation config is the single (non-variant) block of a world that sets
global run parameters and per-subsystem behavior: environment, time stepping,
soft- and rigid-body solvers, contact, vision, the instability sentinel, and
control/record cadences. Every sub-config leaves any omitted field at the
engine default.

Example::

    from rise_origin import rsc

    sim = rsc.simulation.Simulation(
        environment=rsc.simulation.Environment(gravity_acceleration=-9.81),
        time=rsc.simulation.Time(dt=1e-3),
        control=rsc.simulation.Control(frequency=50.0),
    )
"""

from rise_origin.rsc.simulation.simulation import (
    Contact,
    Control,
    Environment,
    Record,
    RigidBody,
    Runtime,
    Sentinel,
    Simulation,
    SoftBody,
    Time,
)
from rise_origin.rsc.simulation.vision import (
    Vision,
    VisionDenoiseAtrous,
    VisionDenoiseNone,
    VisionDenoiseSVGF,
    VisionReSTIR,
    VisionTrace,
)

__all__ = [
    "Environment",
    "Time",
    "SoftBody",
    "RigidBody",
    "Contact",
    "VisionTrace",
    "VisionReSTIR",
    "VisionDenoiseNone",
    "VisionDenoiseAtrous",
    "VisionDenoiseSVGF",
    "Vision",
    "Sentinel",
    "Control",
    "Record",
    "Runtime",
    "Simulation",
]
