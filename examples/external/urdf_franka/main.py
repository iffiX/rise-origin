# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""URDF Franka — import a Panda arm with its true inertials and texture.

What you'll learn:

* Load an articulated robot from a ``.urdf`` with
  :meth:`rsc.entity.StructureVoxel.from_urdf`, asking it to extract per-link
  materials. Each link's true mass, center of mass, and inertia become a shared
  atlas physical property, and its textured visual color is sampled into per
  voxel optical materials, so the arm keeps its real dynamics and look.
* Take the extracted essences and textures back through ``essence_outputs`` and
  ``resource_outputs`` and hand them to the world.
* Anchor the base by giving the root link's materials a fixed property, give the
  imported joints motor torque, and drive them from a control callback.

The ``panda.urdf`` and its meshes next to this script come from the Bullet
Franka Panda description.

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

# Maximum number of distinct colors sampled from each link's texture.
VISUAL_SIGNATURES = 4

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


def _get_root_materials(structure):
    """Return the material names referenced by the kinematic root body.

    The root is the one body that is a constraint parent but never a constraint
    child.
    """
    children = {constraint.b_body_sid for constraint in structure.constraints}
    parents = {constraint.a_body_sid for constraint in structure.constraints}
    root_sid = next(iter(parents - children))
    return {
        structure.material_references[voxel]
        for voxel in structure.bodies[root_sid].material_reference_sid
        if voxel != rs.RS_NULL_INDEX
    }


def build():
    """Build the Franka world from the bundled URDF.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    essences, resources = [], []
    panda = rsc.entity.StructureVoxel.from_urdf(
        URDF_PATH,
        voxel_size=VOXEL,
        name="panda",
        essence_outputs=essences,
        resource_outputs=resources,
        is_prefer_visual_mesh=True,
        visual_signature_num=VISUAL_SIGNATURES,
    )
    panda.signal.source_controller_reference = CONTROLLER
    for constraint in panda.constraints:
        constraint.hinge_max_torque = HINGE_TORQUE
        constraint.prismatic_max_force = PRISMATIC_FORCE
        constraint.hinge_min = max(constraint.hinge_min, -HINGE_LIMIT)
        constraint.hinge_max = min(constraint.hinge_max, HINGE_LIMIT)

    # Anchor the base: a fixed property turns the root link's atlas materials
    # into anchored rigid bodies.
    for material in _get_root_materials(panda):
        essences.append(
            rsc.essence.MaterialPhysicalPropertyFixed(
                material + "#fixed", material_reference=material
            )
        )

    return _common.make_world(
        "urdf_franka",
        structures=[panda],
        controllers=[rsc.controller.ControllerCallback(CONTROLLER)],
        essences=essences,
        resources=resources,
    )


def control(time, controllers):
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
