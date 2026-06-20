# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""MJCF ant — import a MuJoCo model and drive its eight leg hinges.

What you'll learn:

* Load a whole articulated robot straight from a MuJoCo ``.xml`` with
  :meth:`rsc.entity.StructureVoxel.from_mjcf`. Every MuJoCo body is voxelized
  into a rigid segment and every hinge joint becomes a driven hinge constraint.
* Give the imported structure a single rigid material and wire its hinge
  signals to a control callback, since the importer leaves motor torque and the
  driving controller for the caller to set.
* Run an alternating four leg gait from that callback.

The ``ant.xml`` next to this script is the classic Gymnasium ant: a free torso
on four two segment legs, eight hinges in all. The importer skips the free root
joint, so the eight remaining hinges drive the legs.

Run it::

    python examples/external/mjcf_ant/main.py
"""

import math
import os
import sys

from rise_origin import RS_WorldStructureVoxelSignal, rsc

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # noqa: E401,E702 — find sibling _common
import _common

HERE = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.join(HERE, "ant.xml")

# Edge length of one voxel when filling the ant's geoms, in meters.
VOXEL = 0.01

# One rigid material for the whole ant; every imported voxel references it.
MATERIAL = "ant"

BODY_COLOR = _common.PALETTE["deep_blue"]

# Motor torque given to each imported hinge, in newton meters.
HINGE_TORQUE = 4.0

CONTROLLER = "ant#brain"

# Gait shape. Hips steer fore and aft, knees lift a quarter cycle later.
GAIT_HZ = 1.0
HIP_AMPLITUDE = 0.4
KNEE_AMPLITUDE = 0.5

# Imported hinge order follows the MJCF joint order, so the eight signals
# interleave as hip, knee, hip, knee around the four legs.
LEG_NUM = 4


def build():
    """Build the ant world from the bundled MJCF.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    ant = rsc.entity.StructureVoxel.from_mjcf(MJCF_PATH, voxel_size=VOXEL, name="ant")
    ant.material_references.append(MATERIAL)
    ant.signal.source_controller_name = CONTROLLER
    for constraint in ant.constraints:
        constraint.hinge_max_torque = HINGE_TORQUE

    materials = [
        _common.MaterialSpec(
            MATERIAL,
            color=BODY_COLOR,
            kind="rigid",
            roughness=0.12,
            metalness=1.0,
        )
    ]

    return _common.make_world(
        "mjcf_ant",
        materials=materials,
        structures=[ant],
        controllers=[rsc.controller.ControllerCallback(CONTROLLER)],
    )


def control(world_name, time, controllers):
    """Drive an alternating gait: diagonal leg pairs swing out of phase.

    Hips are the even signals, knees the odd ones; each knee lifts a quarter
    cycle after its hip.
    """
    controller = _common.controller_by_name(controllers, CONTROLLER)
    if controller is None:
        return
    wave = 2.0 * math.pi * GAIT_HZ * time
    for signal in controller.signals():
        if not isinstance(signal, RS_WorldStructureVoxelSignal):
            continue
        hinge = signal.hinge_signals().hinge()
        if len(hinge) < 2 * LEG_NUM:
            continue
        for leg in range(LEG_NUM):
            phase = 0.0 if leg in (0, 2) else math.pi
            hinge[2 * leg] = HIP_AMPLITUDE * math.sin(wave + phase)
            hinge[2 * leg + 1] = KNEE_AMPLITUDE * math.sin(wave + phase + math.pi / 2.0)


def main():
    """Build the world and launch it in the selected run mode."""
    _common.launch(build(), control, camera_focus=(0.0, 0.0, 0.5))


if __name__ == "__main__":
    main()
