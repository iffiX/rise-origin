# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Ant — a spherical torso on four two-segment legs (8 hinges).

What you'll learn:

* Build a solid rigid sphere torso with
  :meth:`rsc.entity.StructureVoxelBody.sphere`, and lay limb segments out as bare
  rigid boxes. The whole ant is fully rigid.
* Splay four legs radially (+X, +Y, -X, -Y) and give each a steering hip
  (vertical axis) and a lifting knee (horizontal axis), for eight driven hinges.
* Drive a simple alternating gait from a control callback.

Run it::

    python examples/creatures/ant.py
"""

import math

from rise_origin import (
    RS_WorldStructureVoxelSignal,
    RSE_StructureVoxelConstraintType,
    rsc,
)

import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # noqa: E401,E702 — find sibling _common
import _common

VOXEL = 0.01

TORSO_RADIUS = 6  # voxels
LEG_THICK = 4  # voxels (square cross-section of every leg segment)
UPPER_LEN = 12  # voxels (horizontal upper leg)
LOWER_LEN = 12  # voxels (vertical lower leg)
GAP = 1  # voxels between torso surface and the upper leg

TORSO_Z = 0.16  # torso centre height (m); feet start just above the floor

# Hinge limits and motor torque.
HIP_MIN, HIP_MAX = -0.6, 0.6
KNEE_MIN, KNEE_MAX = -0.9, 0.9
HINGE_TORQUE = 8.0

BODY_COLOR = _common.PALETTE["deep_blue"]
UPPER_COLOR = _common.PALETTE["blue_green"]
LOWER_COLOR = _common.PALETTE["amber"]

GAIT_HZ = 1.0
HIP_AMPLITUDE = 0.5
KNEE_AMPLITUDE = 0.7

# Each leg's outward direction in the XY plane.
LEG_DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# Material reference order; per-voxel ids index into it.
MATERIAL_REFERENCES = ["torso", "upper", "lower"]
TORSO_ID, UPPER_ID, LOWER_ID = 0, 1, 2

CONTROLLER = "ant#brain"


def _torso_body():
    """A solid rigid sphere torso (one segment)."""
    surf = TORSO_RADIUS * VOXEL
    return rsc.entity.StructureVoxelBody.sphere(
        TORSO_RADIUS,
        material_sid=TORSO_ID,
        voxel_size=VOXEL,
        relative_origin_position=(-surf, -surf, TORSO_Z - surf),
    )


def _box(size, material_id, corner):
    """A bare rigid box of one material at the given structure-local min corner."""
    return rsc.entity.StructureVoxelBody.box(
        size[0],
        size[1],
        size[2],
        material_sid=material_id,
        voxel_size=VOXEL,
        relative_origin_position=corner,
    )


def _hinge(a, b, *, axis, anchor, signal_sid, angle_min, angle_max):
    """A motorized hinge driven by the structure signal ``signal_sid``."""
    return rsc.entity.StructureVoxelConstraint(
        RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_HINGE_JOINT,
        a=a,
        b=b,
        relative_anchor=anchor,
        hinge_relative_axis=axis,
        hinge_min=angle_min,
        hinge_max=angle_max,
        hinge_max_torque=HINGE_TORQUE,
        hinge_signal_sid=signal_sid,
    )


def build():
    """Build the ant world.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    materials = [
        _common.MaterialSpec("torso", color=BODY_COLOR, kind="rigid"),
        _common.MaterialSpec("upper", color=UPPER_COLOR, kind="rigid"),
        _common.MaterialSpec("lower", color=LOWER_COLOR, kind="rigid"),
    ]

    torso = _torso_body()
    bodies = [torso]
    constraints = []

    surf = TORSO_RADIUS * VOXEL
    thick = LEG_THICK * VOXEL
    gap = GAP * VOXEL
    for leg, (dx, dy) in enumerate(LEG_DIRS):
        hip = (dx * surf, dy * surf, TORSO_Z)
        reach = surf + gap + UPPER_LEN * VOXEL
        knee = (dx * reach, dy * reach, TORSO_Z)

        # Upper leg: a horizontal box from the torso surface out to the knee.
        if dx != 0:
            upper_size = (UPPER_LEN, LEG_THICK, LEG_THICK)
            x = hip[0] + gap if dx > 0 else hip[0] - gap - UPPER_LEN * VOXEL
            upper_corner = (x, -thick / 2.0, TORSO_Z - thick / 2.0)
            knee_axis = (0, 1, 0)
        else:
            upper_size = (LEG_THICK, UPPER_LEN, LEG_THICK)
            y = hip[1] + gap if dy > 0 else hip[1] - gap - UPPER_LEN * VOXEL
            upper_corner = (-thick / 2.0, y, TORSO_Z - thick / 2.0)
            knee_axis = (1, 0, 0)

        upper = _box(upper_size, UPPER_ID, upper_corner)
        # Lower leg: a bare rigid box hanging straight down from the knee.
        lower_corner = (
            knee[0] - thick / 2.0,
            knee[1] - thick / 2.0,
            TORSO_Z - LOWER_LEN * VOXEL,
        )
        lower = _box((LEG_THICK, LEG_THICK, LOWER_LEN), LOWER_ID, lower_corner)
        bodies.extend([upper, lower])

        constraints.append(
            _hinge(
                (torso, 0),
                upper,
                axis=(0, 0, 1),  # vertical: hip steers the leg fore/aft
                anchor=hip,
                signal_sid=leg,
                angle_min=HIP_MIN,
                angle_max=HIP_MAX,
            )
        )
        constraints.append(
            _hinge(
                upper,
                lower,
                axis=knee_axis,  # horizontal: knee lifts the foot
                anchor=knee,
                signal_sid=4 + leg,
                angle_min=KNEE_MIN,
                angle_max=KNEE_MAX,
            )
        )

    ant = rsc.entity.StructureVoxel(
        "ant",
        material_references=MATERIAL_REFERENCES,
        bodies=bodies,
        constraints=constraints,
        signal=rsc.entity.StructureVoxelSignal(
            "ant#signal", source_controller_name=CONTROLLER, hinge_signal_num=8
        ),
    )

    return _common.make_world(
        "ant",
        materials=materials,
        structures=[ant],
        controllers=[rsc.controller.ControllerCallback(CONTROLLER)],
    )


def control(world_name, time, controllers):
    """Drive an alternating gait: X-legs and Y-legs swing out of phase.

    Hips (signals 0-3) steer, knees (signals 4-7) lift a quarter-cycle later.
    """
    controller = _common.controller_by_name(controllers, CONTROLLER)
    if controller is None:
        return
    wave = 2.0 * math.pi * GAIT_HZ * time
    for signal in controller.signals():
        if not isinstance(signal, RS_WorldStructureVoxelSignal):
            continue
        hinge = signal.hinge_signals().hinge()
        if len(hinge) < 8:
            continue
        for leg in range(4):
            phase = 0.0 if leg in (0, 2) else math.pi
            hinge[leg] = HIP_AMPLITUDE * math.sin(wave + phase)
            hinge[4 + leg] = KNEE_AMPLITUDE * math.sin(wave + phase + math.pi / 2.0)


def main():
    """Build the world and launch it in the selected run mode."""
    _common.launch(build(), control, camera_focus=(0.0, 0.0, TORSO_Z))


if __name__ == "__main__":
    main()
