# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""URDF Franka — import a Panda arm and sweep it through a slow pose.

What you'll learn:

* Load an articulated robot from a ``.urdf`` with
  :meth:`rsc.entity.StructureVoxel.from_urdf`. Each link's collision mesh is
  voxelized into a rigid segment and each joint becomes a constraint, so the
  seven revolute arm joints arrive as hinges and the two finger joints as
  prismatic slides.
* Anchor the base link with a fixed material so the arm has a static frame to
  push against, give the imported joints motor torque, and drive them from a
  control callback.

The ``panda.urdf`` and its collision meshes next to this script come from the
Bullet Franka Panda description.

Run it::

    python examples/external/urdf_franka/main.py
"""

import math
import os
import sys

import rise_origin as rs
from rise_origin import RS_WorldStructureVoxelSignal, rsc

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # noqa: E401,E702 — find sibling _common
import _common

URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panda.urdf")

# Edge length of one voxel when filling each link mesh, in meters.
VOXEL = 0.005

# Two materials: a bolted down base and the movable arm and gripper.
BASE_MATERIAL = "base"
ARM_MATERIAL = "arm"

BASE_COLOR = _common.PALETTE["deep_blue"]
ARM_COLOR = _common.PALETTE["white"]

# Motor limits given to the imported joints.
HINGE_TORQUE = 120.0
PRISMATIC_FORCE = 60.0

# The engine keeps hinge limits within plus or minus pi. One Panda joint lists a
# lower limit a hair past minus pi, so imported hinge limits are clamped to this.
HINGE_LIMIT = math.pi - 1e-3

CONTROLLER = "panda#brain"

# Pose sweep. Each arm hinge eases between a rest angle and a reach angle, and
# the gripper opens and closes once per cycle.
SWEEP_HZ = 0.2
ARM_REACH = 0.6
GRIPPER_OPEN = 0.04


def _root_body_sid(structure):
    """Return the structure scope id of the kinematic root body.

    The root is the one body that drives a joint but is never driven by one,
    found as the body id that is a constraint parent but never a constraint
    child.
    """
    children = {constraint.b_body_sid for constraint in structure.constraints}
    parents = {constraint.a_body_sid for constraint in structure.constraints}
    return next(iter(parents - children))


def build():
    """Build the Franka world from the bundled URDF.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    panda = rsc.entity.StructureVoxel.from_urdf(
        URDF_PATH, voxel_size=VOXEL, name="panda"
    )

    for material in (BASE_MATERIAL, ARM_MATERIAL):
        panda.material_references.append(material)
    root_sid = _root_body_sid(panda)
    for body_sid, body in enumerate(panda.bodies):
        material_sid = 0 if body_sid == root_sid else 1
        voxels = body.material_reference_sid
        for index in range(len(voxels)):
            if voxels[index] != rs.RS_NULL_INDEX:
                voxels[index] = material_sid

    panda.signal.source_controller_name = CONTROLLER
    for constraint in panda.constraints:
        constraint.hinge_max_torque = HINGE_TORQUE
        constraint.prismatic_max_force = PRISMATIC_FORCE
        constraint.hinge_min = max(constraint.hinge_min, -HINGE_LIMIT)
        constraint.hinge_max = min(constraint.hinge_max, HINGE_LIMIT)

    materials = [
        _common.MaterialSpec(
            BASE_MATERIAL,
            color=BASE_COLOR,
            kind="rigid_fixed",
            roughness=0.12,
            metalness=1.0,
        ),
        _common.MaterialSpec(ARM_MATERIAL, color=ARM_COLOR, kind="rigid"),
    ]

    return _common.make_world(
        "urdf_franka",
        materials=materials,
        structures=[panda],
        controllers=[rsc.controller.ControllerCallback(CONTROLLER)],
    )


def control(world_name, time, controllers):
    """Ease every arm hinge between rest and reach, and cycle the gripper."""
    controller = _common.controller_by_name(controllers, CONTROLLER)
    if controller is None:
        return
    phase = 0.5 - 0.5 * math.cos(2.0 * math.pi * SWEEP_HZ * time)
    arm_target = ARM_REACH * phase
    grip_target = GRIPPER_OPEN * phase
    for signal in controller.signals():
        if not isinstance(signal, RS_WorldStructureVoxelSignal):
            continue
        hinge = signal.hinge_signals().hinge()
        for joint in range(len(hinge)):
            hinge[joint] = arm_target
        prismatic = signal.prismatic_signals().prismatic()
        for finger in range(len(prismatic)):
            prismatic[finger] = grip_target


def main():
    """Build the world and launch it in the selected run mode."""
    _common.launch(build(), control, camera_focus=(0.0, 0.0, 0.4))


if __name__ == "__main__":
    main()
