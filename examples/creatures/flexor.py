# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Flexor — a muscle-actuated quadruped built from one voxel grid.

A torso with four legs, built as a single body with
:meth:`rsc.entity.StructureVoxelBody.from_grid`. Each leg has two rigid bone
segments (upper + lower) meeting at the knee and joined by a passive hinge, with
a single soft flexor muscle running down one side and spanning the knee.
Contracting the muscle curls the lower bone about the hinge; relaxing lets it
straighten. Bone cores are full-y so their side faces (which swing about the
hinge) are exposed; a small soft pad forms each foot.

Run it::

    python examples/creatures/flexor.py
"""

import math

import numpy as np

from rise_origin import (
    RS_NULL_INDEX,
    RS_WorldStructureVoxelSignal,
    RSE_StructureVoxelConstraintType,
    rsc,
)

import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # noqa: E401,E702 — find sibling _common
import _common

VOXEL = 0.02
GX, GY, GZ = 24, 8, 15

TORSO_X = (3, 21)
TORSO_Z = (10, 14)
LEG_FB = 5  # x: back-skin | bone-core (2) | flexor muscle (2)
LEG_LR = 3  # y thickness
KNEE_Z = 5  # knee gap; lower bone below, upper bone above
MUSCLE_Z = (2, 9)  # z-span of the flexor (bridges the knee)

TORSO_SEG = 0
SHELL_SEG = 40
MUSCLE_SEG = 41

# (corner_x, left_y) of each leg's footprint.
LEGS = [
    (TORSO_X[0], 0),
    (TORSO_X[0], GY - LEG_LR),
    (TORSO_X[1] - LEG_FB, 0),
    (TORSO_X[1] - LEG_FB, GY - LEG_LR),
]

GAIT_HZ = 2
AMPLITUDE = 0.3

# Material reference order; per-voxel ids index into it.
MATERIAL_REFERENCES = ["torso", "upper", "lower", "skin", "muscle"]
TORSO_ID, UPPER_ID, LOWER_ID, SKIN_ID, MUSCLE_ID = 0, 1, 2, 3, 4

CONTROLLER = "flexor#brain"


def build():
    """Build the muscle quadruped.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    materials = [
        _common.MaterialSpec("torso", color=_common.PALETTE["deep_blue"], kind="rigid"),
        _common.MaterialSpec(
            "upper", color=_common.PALETTE["blue_green"], kind="rigid"
        ),
        _common.MaterialSpec("lower", color=_common.PALETTE["amber"], kind="rigid"),
        _common.MaterialSpec(
            "skin", color=_common.PALETTE["sky_blue"], kind="soft", elastic_modulus=2e5
        ),
        _common.MaterialSpec(
            "muscle",
            color=_common.PALETTE["tiger_orange"],
            kind="soft",
            elastic_modulus=3e5,
            min_expansion=-AMPLITUDE,
            max_expansion=0.0,
        ),
    ]

    material = np.full((GX, GY, GZ), RS_NULL_INDEX, dtype=np.int64)
    segment = np.full((GX, GY, GZ), RS_NULL_INDEX, dtype=np.int64)
    expansion = np.full((GX, GY, GZ), RS_NULL_INDEX, dtype=np.int64)

    # Torso: soft shell box with a rigid core one voxel inside.
    material[3:21, :, 10:14] = SKIN_ID
    segment[3:21, :, 10:14] = SHELL_SEG
    material[4:20, 1 : GY - 1, 11:13] = TORSO_ID
    segment[4:20, 1 : GY - 1, 11:13] = TORSO_SEG

    knees = []
    for leg, (x0, y0) in enumerate(LEGS):
        upper, lower = 1 + 2 * leg, 2 + 2 * leg
        ys = slice(y0, y0 + LEG_LR)
        # x layout: back-skin (x0) | bone core (x0+1, x0+2) | flexor (x0+3, x0+4),
        # so the muscle sits directly against the bone core.
        core = slice(x0 + 1, x0 + 3)
        flex = slice(x0 + 3, x0 + 5)

        # Rigid bone cores, full-y (side faces exposed); they meet at the knee.
        material[core, ys, 1 : KNEE_Z + 1] = LOWER_ID
        segment[core, ys, 1 : KNEE_Z + 1] = lower
        material[core, ys, KNEE_Z + 1 : TORSO_Z[0]] = UPPER_ID
        segment[core, ys, KNEE_Z + 1 : TORSO_Z[0]] = upper

        # Single flexor muscle down the flexor side, bridging the knee gap.
        material[flex, ys, MUSCLE_Z[0] : MUSCLE_Z[1]] = MUSCLE_ID
        segment[flex, ys, MUSCLE_Z[0] : MUSCLE_Z[1]] = MUSCLE_SEG
        expansion[flex, ys, MUSCLE_Z[0] : MUSCLE_Z[1]] = leg

        # Small soft foot pad under the bone core.
        material[core, ys, 0:1] = SKIN_ID
        segment[core, ys, 0:1] = SHELL_SEG

        knees.append((x0 + 2.0, y0 + LEG_LR / 2.0))

    body = rsc.entity.StructureVoxelBody.from_grid(
        material != RS_NULL_INDEX,
        material_grid=material,
        segment_grid=segment,
        expansion_signal_grid=expansion,
        voxel_size=VOXEL,
    )

    constraints = []
    for leg, (cx, cy) in enumerate(knees):
        upper, lower = 1 + 2 * leg, 2 + 2 * leg
        hip = (cx * VOXEL, cy * VOXEL, (TORSO_Z[0] - 0.5) * VOXEL)
        constraints.append(
            rsc.entity.StructureVoxelConstraint(
                RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_FIXED_JOINT,
                a_body_sid=0,
                a_segment_bid=TORSO_SEG,
                b_body_sid=0,
                b_segment_bid=upper,
                relative_anchor=hip,
            )
        )
        knee = (cx * VOXEL, cy * VOXEL, (KNEE_Z + 0.5) * VOXEL)
        constraints.append(
            rsc.entity.StructureVoxelConstraint(
                RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_HINGE_JOINT,
                a_body_sid=0,
                a_segment_bid=upper,
                b_body_sid=0,
                b_segment_bid=lower,
                relative_anchor=knee,
                hinge_relative_axis=(0, 1, 0),
                hinge_min=-2.0,
                hinge_max=2.0,
            )
        )

    # Center the whole structure so the body origin stays (0,0,0) and the joint
    # anchors below, computed as voxel*VOXEL, are in the body frame.
    flexor = rsc.entity.StructureVoxel(
        "flexor",
        origin_position=(-GX * VOXEL / 2.0, -GY * VOXEL / 2.0, 0.12),
        orientation=(0.0, 0.07, 0.0, 0.9975),  # slight pitch so it lands unlocked
        material_references=MATERIAL_REFERENCES,
        bodies=[body],
        constraints=constraints,
        signal=rsc.entity.StructureVoxelSignal(
            "flexor#signal", source_controller_reference=CONTROLLER, expansion_signal_num=4
        ),
    )

    return _common.make_world(
        "flexor",
        materials=materials,
        structures=[flexor],
        controllers=[rsc.controller.ControllerCallback(CONTROLLER)],
        # Tight rigid-soft coupling needs a finer constraint step and gentler
        # joint position correction to keep the bonded muscle tissue stable.
        time=rsc.simulation.Time(dt_rigidbody_contact_and_constraint_update=0.0005),
        rigid_body=rsc.simulation.RigidBody(constraint_position_correction_ratio=0.1),
    )


def control(time, controllers):
    """Contract every leg's flexor in unison (curl, then relax back)."""
    controller = _common.controller_by_name(controllers, CONTROLLER)
    if controller is None:
        return
    rate = AMPLITUDE * math.sin(2.0 * math.pi * GAIT_HZ * time)
    for signal in controller.signals():
        if not isinstance(signal, RS_WorldStructureVoxelSignal):
            continue
        expansion = signal.expansion_signals().expansion()
        for leg in range(len(expansion)):
            expansion[leg] = rate


def main():
    """Build the world and launch it in the selected run mode."""
    _common.launch(build(), control, camera_focus=(0.0, 0.0, 0.12))


if __name__ == "__main__":
    main()
