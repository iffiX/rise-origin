# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Interactive prismatic-joint test — a 3-segment soft inchworm.

Each segment is an identical soft box (8x6x6 voxels) with a 2x2 rigid rod
along its X axis. Consecutive rods are joined by a prismatic joint that slides
along X, so the two gaps extend/contract on command. Exercises prismatic-joint
stability, basic-position signal dispatch, and the shared interactive harness.

Controls (handled by :mod:`_test_common`)
* ``1`` / ``2`` — hold gap 0 / gap 1 at full extension (signal = 1.0). Add
  ``Shift`` to hold at full contraction (0.0); release returns to neutral (0.5).
* Mouse + ``WASD`` — orbit / pan / move the viewer camera.
* ``T`` — reset the viewer camera (handled by the viewer).
* ``Esc`` — quit.

Run::

    python test/interactive/test_inchworm.py
"""

import numpy as np

import rise_origin as rs
from rise_origin import RS_NULL_INDEX
from rise_origin.create import (
    CameraCreator,
    MaterialCreator,
    StructureCreator,
)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _test_common as common


# --- Geometry ---
VOXEL = 0.01
BOX_X, BOX_Y, BOX_Z = 8, 6, 6
ROD_WIDTH = 2          # rigid-rod cross-section (voxels)
GAP_VOXELS = 4         # X gap between adjacent boxes
ROD_OVERHANG = 2       # rod protrudes this many voxels past each box face
NUM_SEGMENTS = 3
PITCH = (BOX_X + GAP_VOXELS) * VOXEL
TOTAL_X = BOX_X + 2 * ROD_OVERHANG  # body grid extent along X

SOFT_BID = 1
RIGID_BID = 2

# --- Prismatic joint (basic-position mode: signal in [0,1] -> [min, max]) ---
PRISMATIC_MIN = -0.03
PRISMATIC_MAX = 0.03
PRISMATIC_MAX_FORCE = 5.0
PRISMATIC_MAX_SPEED = 1.0

# --- Camera ---
ROBOT_CAMERA_NAME = "worm_camera"
ROBOT_CAMERA_ANCHOR = (-0.04, BOX_Y * VOXEL / 2, 0.16)
ROBOT_CAMERA_ORIENTATION = (0.0, 0.2164, 0.0, 0.9763)  # ~25 deg pitch down


def _segment_grids():
    """Per-voxel ``(materials, segments)`` arrays for one inchworm segment.

    The grid is extended along X by ``ROD_OVERHANG`` so the rigid rod can stick
    past the soft box; voxels outside the soft box and the rod cross-section
    are empty (``None`` materials, ``RS_NULL_INDEX`` segment ids).
    """
    materials = np.full((TOTAL_X, BOX_Y, BOX_Z), None, dtype=object)
    segments = np.full((TOTAL_X, BOX_Y, BOX_Z), RS_NULL_INDEX, dtype=np.int64)
    y_lo = (BOX_Y - ROD_WIDTH) // 2
    z_lo = (BOX_Z - ROD_WIDTH) // 2
    for ix in range(TOTAL_X):
        in_box = ROD_OVERHANG <= ix < ROD_OVERHANG + BOX_X
        for iy in range(BOX_Y):
            for iz in range(BOX_Z):
                is_rod = (
                    y_lo <= iy < y_lo + ROD_WIDTH and z_lo <= iz < z_lo + ROD_WIDTH
                )
                if is_rod:
                    materials[ix, iy, iz] = "rigid"
                    segments[ix, iy, iz] = RIGID_BID
                elif in_box:
                    materials[ix, iy, iz] = "soft"
                    segments[ix, iy, iz] = SOFT_BID
    return materials, segments


def build():
    """Build the inchworm scene and return ``(config, joint_names)``."""
    config = common.make_config()
    config.material_configs.append(
        MaterialCreator.soft("soft", elastic_modulus=3e5, render=(90, 170, 90))
    )
    config.material_configs.append(
        MaterialCreator.rigid("rigid", render=(60, 60, 70))
    )

    sc = StructureCreator(
        "inchworm",
        voxel_size=VOXEL,
        origin_position=(-PITCH, 0.0, 0.02),
    )

    materials, segments = _segment_grids()
    bodies = []
    for seg in range(NUM_SEGMENTS):
        body_x = seg * PITCH - ROD_OVERHANG * VOXEL
        bodies.append(
            sc.add_voxel_grid_body(
                materials,
                segment_bids=segments,
                relative_origin_position=(body_x, 0.0, 0.0),
            )
        )

    # Prismatic joint between adjacent rigid rods, anchored at the gap midpoint
    # at the rod's vertical and lateral centre.
    anchor_y = BOX_Y * VOXEL / 2.0
    anchor_z = BOX_Z * VOXEL / 2.0
    for gap in range(NUM_SEGMENTS - 1):
        anchor_x = gap * PITCH + (BOX_X + GAP_VOXELS / 2.0) * VOXEL
        sc.add_prismatic_joint(
            a=bodies[gap].segment(RIGID_BID),
            b=bodies[gap + 1].segment(RIGID_BID),
            relative_axis=(1, 0, 0),
            relative_anchor=(anchor_x, anchor_y, anchor_z),
            signal_sid=gap,
            distance_min=PRISMATIC_MIN,
            distance_max=PRISMATIC_MAX,
            max_force=PRISMATIC_MAX_FORCE,
            max_speed=PRISMATIC_MAX_SPEED,
        )

    config.structure_configs.append(sc.build())
    config.camera_configs.append(
        CameraCreator.rigid_body_linked(
            ROBOT_CAMERA_NAME,
            structure_name="inchworm",
            attached_to=bodies[0].segment(RIGID_BID),
            relative_anchor=ROBOT_CAMERA_ANCHOR,
            relative_orientation=ROBOT_CAMERA_ORIENTATION,
            image_width=common.ROBOT_CAMERA_SIZE,
            image_height=common.ROBOT_CAMERA_SIZE,
            samples_per_pixel=4,
            field_of_view=60.0,
        )
    )

    joint_names = [f"Gap {i} prismatic" for i in range(NUM_SEGMENTS - 1)]
    return config, joint_names


def main():
    """Open the inchworm in an interactive viewer with the shared HUD."""
    config, joint_names = build()
    controls = common.JointControlState(joint_names)
    for i in range(len(joint_names)):
        controls.as_basic_prismatic(i)
    state = common.HudState()

    def control(sim_id, controllers):
        for ctype, c in controllers:
            if ctype == common.STRUCTURE:
                rec = c.structure_record()
                if rec is not None:
                    state.update_bounds(rec)
                controls.apply_to(c.signal().prismatic_signals())
                for cam in c.camera_records():
                    state.update_robot_frame(cam)

    common.launch(
        config,
        control,
        hud=common.hud_factory(state, controls, robot_camera=True),
        on_event=lambda event, viewer: controls.handle_event(event),
        camera_focus=(0.10, 0.0, 0.05),
        camera_position=(-0.4, -0.3, 0.25),
    )


if __name__ == "__main__":
    main()
