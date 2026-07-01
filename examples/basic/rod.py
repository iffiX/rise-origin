# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Twisted rod: a rainbow soft column wound up by a cylinder motor.

What you'll learn:

* Compose one body from many materials and segments with
  :meth:`rsc.entity.StructureVoxelBody.from_grid`: a ``block`` that pins one end,
  a soft middle quartered into four colors, and a rigid cylinder segment (the
  rotor) on the far end, with the soft voxels bonding to both.
* Add the other half of the motor as a separate fixed cylinder body (the
  stator), anchored in the world.
* Twist the soft rod with a PD-controlled hinge about its long axis: the rotor
  spins against the fixed stator while the block holds the near end, so the four
  colors shear past each other.

Layout along the rod's long axis: block, soft rod, rotor, stator.

Run it::

    python examples/basic/rod.py
    RISE_MODE=record RISE_OUT=rod.mp4 python examples/basic/rod.py
"""

import math
import os
import sys

import numpy as np

from rise_origin import (
    RS_NULL_INDEX,
    RS_WorldStructureVoxelSignal,
    RSE_StructureVoxelConstraintType,
    RSE_StructureVoxelHingeControlMode,
    rsc,
)

# Put the examples root on the path so the sibling _common module is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

VOXEL = 0.02

BLOCK_X = 3
ROD_X = 30
ROTOR_X = 3
STATOR_X = 3
NY, NZ = 6, 6
CYL_RADIUS = 3

AXIS_Z = 0.25

ROTOR_SEG = 5  # rotor segment id (each material region is its own segment)

TWIST = 2.0
TWIST_HZ = 0.1

MOTOR_TORQUE = 300.0  # torque cap (N-m)
MOTOR_KP = 20.0  # PD stiffness (N-m per rad of error)
MOTOR_KD = 0.5  # PD damping (N-m per rad/s)

# 90-degree yaw so the long rod lies broadside to the default camera.
ROD_YAW = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))

CONTROLLER = "rod#brain"

QUAD_COLORS = [
    _common.PALETTE["sky_blue"],
    _common.PALETTE["blue_green"],
    _common.PALETTE["amber"],
    _common.PALETTE["tiger_orange"],
]
BLOCK_COLOR = _common.PALETTE["deep_blue"]
ROTOR_COLOR = (150, 150, 150)
STATOR_COLOR = (90, 90, 90)

# Material reference order for the rod structure; per-voxel ids index into it.
MATERIAL_REFERENCES = ["block", "soft0", "soft1", "soft2", "soft3", "rotor", "stator"]


def _disk_mask():
    """Boolean ``(NY, NZ)`` mask of the inscribed motor-cylinder cross-section."""
    coords = np.arange(NY) + 0.5 - CYL_RADIUS
    yy, zz = np.meshgrid(coords, coords, indexing="ij")
    return (yy * yy + zz * zz) <= float(CYL_RADIUS) ** 2


def build():
    """Build the twisted-rod world.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    materials = [
        _common.MaterialSpec("block", color=BLOCK_COLOR, kind="fixed"),
        *(
            _common.MaterialSpec(
                f"soft{i}", color=color, kind="soft", elastic_modulus=1e5
            )
            for i, color in enumerate(QUAD_COLORS)
        ),
        _common.MaterialSpec("rotor", color=ROTOR_COLOR, kind="rigid"),
        # The stator is an anchored rigid body the hinge reacts against.
        _common.MaterialSpec("stator", color=STATOR_COLOR, kind="rigid_fixed"),
    ]

    main_x = BLOCK_X + ROD_X + ROTOR_X
    mat = np.full((main_x, NY, NZ), RS_NULL_INDEX, dtype=np.int64)

    # Fixed block at the near end.
    mat[:BLOCK_X, :, :] = 0

    # Soft rod, quartered into four colors.
    rod = slice(BLOCK_X, BLOCK_X + ROD_X)
    hy, hz = NY // 2, NZ // 2
    mat[rod, :hy, :hz] = 1
    mat[rod, hy:, :hz] = 2
    mat[rod, :hy, hz:] = 3
    mat[rod, hy:, hz:] = 4

    # Rigid rotor disk at the far end.
    disk = np.repeat(_disk_mask()[None, :, :], ROTOR_X, axis=0)
    rotor_mat = mat[BLOCK_X + ROD_X :, :, :]
    rotor_mat[disk] = 5

    # Each material region is its own segment, so segment id tracks material id.
    seg = mat.copy()

    main_body = rsc.entity.StructureVoxelBody.from_grid(
        mat != RS_NULL_INDEX,
        material_grid=mat,
        segment_grid=seg,
        voxel_size=VOXEL,
        relative_origin_position=(
            -main_x * VOXEL / 2,
            -NY * VOXEL / 2,
            AXIS_Z - NZ * VOXEL / 2,
        ),
    )

    interface_x = main_x * VOXEL / 2
    stator_body = rsc.entity.StructureVoxelBody.cylinder(
        CYL_RADIUS,
        STATOR_X,
        axis="x",
        material_sid=6,
        voxel_size=VOXEL,
        relative_origin_position=(
            interface_x,
            -NY * VOXEL / 2,
            AXIS_Z - NZ * VOXEL / 2,
        ),
    )

    hinge = rsc.entity.StructureVoxelConstraint(
        RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_HINGE_JOINT,
        a_body_sid=0,
        a_segment_bid=ROTOR_SEG,
        b_body_sid=1,
        b_segment_bid=0,
        relative_anchor=(interface_x, 0.0, AXIS_Z),
        hinge_relative_axis=(1.0, 0.0, 0.0),
        hinge_control_mode=(
            RSE_StructureVoxelHingeControlMode.RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION
        ),
        hinge_min=-math.pi,
        hinge_max=math.pi,
        hinge_max_torque=MOTOR_TORQUE,
        hinge_position_control_kp=MOTOR_KP,
        hinge_position_control_kd=MOTOR_KD,
        hinge_signal_sid=0,
    )

    rod_structure = rsc.entity.StructureVoxel(
        "rod",
        orientation=ROD_YAW,
        material_references=MATERIAL_REFERENCES,
        bodies=[main_body, stator_body],
        constraints=[hinge],
        signal=rsc.entity.StructureVoxelSignal(
            "rod#signal", source_controller_reference=CONTROLLER, hinge_signal_num=1
        ),
    )

    return _common.make_world(
        "rod",
        materials=materials,
        structures=[rod_structure],
        controllers=[rsc.controller.ControllerCallback(CONTROLLER)],
    )


def control(time, controllers):
    """Wind the motor hinge back and forth, twisting the rod."""
    controller = _common.controller_by_name(controllers, CONTROLLER)
    if controller is None:
        return
    for signal in controller.signals():
        if isinstance(signal, RS_WorldStructureVoxelSignal):
            hinge = signal.hinge_signals().hinge()
            if len(hinge) > 0:
                hinge[0] = TWIST * math.sin(2.0 * math.pi * TWIST_HZ * time)


def main():
    """Build the world and launch it in the selected run mode."""
    _common.launch(build(), control, camera_focus=(0.0, 0.0, AXIS_Z))


if __name__ == "__main__":
    main()
