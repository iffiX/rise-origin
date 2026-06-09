# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Interactive expansion-signal test — a single-body muscle arm.

A 13x3x5 voxel arm built as one body with six segments inside it:

* ``shoulder`` (fixed material) — anchored in space.
* ``upper_bone`` / ``upper_skin`` — rigid bone wrapped in soft skin.
* ``muscle`` — soft bicep above the bones, expansion-actuated, bridges the elbow.
* ``lower_bone`` / ``lower_skin`` — rigid forearm + soft skin.

A passive hinge connects the two bones at the elbow. The bicep voxels receive
an expansion signal in RATE mode (signal = d(expansion)/dt). Hold key ``1`` to
contract; release to relax back to rest.

Controls (handled by :mod:`_test_common`)
* ``1`` — hold to contract the bicep; release lets it relax back to its rest
  length (muscle ``min_expansion = -0.5``, ``max_expansion = 0``).
* Mouse + ``WASD`` — orbit / pan / move the viewer camera.
* ``T`` — reset the viewer camera. ``Esc`` — quit.

Run::

    python test/interactive/test_muscle_arm.py
"""

import numpy as np

import rise_origin as rs
from rise_origin import RS_NULL_INDEX
from rise_origin.create import (
    MaterialCreator,
    StructureCreator,
)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _test_common as common


# --- Geometry ---
VOXEL = 0.03
Y_THICKNESS = 3
Z_HEIGHT = 5
X_TOTAL = 13

MOUNT_COLS = (0, 1)
UPPER_COLS = (2, 6)
GAP_COL = 7
FORE_COLS = (8, 12)
MUSCLE_X = (4, 11)
CORE_Z = (1, 2)
TOP_Z = (3, 4)

SEG_FIXED = 1
SEG_UP_RIGID = 2
SEG_UP_SKIN = 3
SEG_MUSCLE = 4
SEG_LO_RIGID = 5
SEG_LO_SKIN = 6

# --- Muscle expansion ---
MUSCLE_MIN_EXPANSION = -0.5   # can contract up to 50%
MUSCLE_MAX_EXPANSION = 0.0    # rest length is the longest
MUSCLE_CONTRACT_RATE = 2.0    # signal applied while key 1 is held
MUSCLE_RELAX_RATE = 2.0       # signal applied while idle (positive -> stretch back)

# --- Hinge limits (passive joint, no signal) ---
HINGE_MIN = -3.0
HINGE_MAX = 3.0


def _role(x: int, z: int):
    """Return ``(material_name, segment_bid, is_muscle)`` for cell ``(x, z)``.

    Returns ``None`` for empty (no-voxel) cells. ``y`` doesn't enter because
    the arm is just extruded along Y.
    """
    core = CORE_Z[0] <= z <= CORE_Z[1]
    bottom = z == 0
    top = TOP_Z[0] <= z <= TOP_Z[1]
    muscle_span = MUSCLE_X[0] <= x <= MUSCLE_X[1]

    if MOUNT_COLS[0] <= x <= MOUNT_COLS[1]:
        return ("fixed", SEG_FIXED, False)

    if UPPER_COLS[0] <= x <= UPPER_COLS[1]:
        if core:
            return ("rigid", SEG_UP_RIGID, False)
        if top and muscle_span:
            return ("muscle", SEG_MUSCLE, True)
        return ("skin", SEG_UP_SKIN, False)

    if x == GAP_COL:
        if top and muscle_span:
            return ("muscle", SEG_MUSCLE, True)
        return None  # gap keeps the bones separate

    if FORE_COLS[0] <= x <= FORE_COLS[1]:
        if core:
            return ("rigid", SEG_LO_RIGID, False)
        if top and muscle_span:
            return ("muscle", SEG_MUSCLE, True)
        return ("skin", SEG_LO_SKIN, False)

    return None


def _arm_grids():
    """Build per-voxel material / segment / expansion-signal grids."""
    materials = np.full((X_TOTAL, Y_THICKNESS, Z_HEIGHT), None, dtype=object)
    segments = np.full((X_TOTAL, Y_THICKNESS, Z_HEIGHT), RS_NULL_INDEX, dtype=np.int64)
    signals = np.full((X_TOTAL, Y_THICKNESS, Z_HEIGHT), RS_NULL_INDEX, dtype=np.int64)
    for x in range(X_TOTAL):
        for z in range(Z_HEIGHT):
            role = _role(x, z)
            if role is None:
                continue
            mat, seg, is_muscle = role
            for y in range(Y_THICKNESS):
                materials[x, y, z] = mat
                segments[x, y, z] = seg
                if is_muscle:
                    signals[x, y, z] = 0
    return materials, segments, signals


def build():
    """Build the muscle-arm scene; returns ``(config, joint_names)``."""
    config = common.make_config()

    config.material_configs.append(
        MaterialCreator.fixed("fixed", render=(60, 60, 60))
    )
    config.material_configs.append(
        MaterialCreator.rigid("rigid", render=(90, 90, 110))
    )
    config.material_configs.append(
        MaterialCreator.soft("skin", elastic_modulus=2e5, render=(210, 180, 160))
    )
    config.material_configs.append(
        MaterialCreator.muscle(
            "muscle",
            min_expansion=MUSCLE_MIN_EXPANSION,
            max_expansion=MUSCLE_MAX_EXPANSION,
            elastic_modulus=3e5,
            render=(200, 70, 70),
        )
    )

    sc = StructureCreator(
        "arm",
        voxel_size=VOXEL,
        origin_position=(-0.2, -0.045, 0.6),
    )

    materials, segments, signals = _arm_grids()
    # Every actuated voxel runs in RATE mode (signal = d(expansion)/dt).
    expansion_modes = np.where(
        signals != RS_NULL_INDEX, 1, 0  # 1 = RATE, 0 = BASIC_POSITION (unused)
    ).astype(np.int64)
    body = sc.add_voxel_grid_body(
        materials,
        segment_bids=segments,
        expansion_signal_sids=signals,
        expansion_control_modes=expansion_modes,
        relative_origin_position=(0.0, 0.0, 0.0),
    )

    # Passive hinge (no signal) at the elbow joins the two bone segments.
    elbow_x = (GAP_COL + 0.5) * VOXEL
    elbow_y = Y_THICKNESS / 2.0 * VOXEL
    elbow_z = (CORE_Z[0] + CORE_Z[1] + 1) / 2.0 * VOXEL
    sc.add_hinge_joint(
        a=body.segment(SEG_UP_RIGID),
        b=body.segment(SEG_LO_RIGID),
        relative_axis=(0.0, 1.0, 0.0),
        relative_anchor=(elbow_x, elbow_y, elbow_z),
        angle_min=HINGE_MIN,
        angle_max=HINGE_MAX,
    )

    config.structure_configs.append(sc.build())

    joint_names = ["Bicep (hold 1 to contract, release to relax)"]
    return config, joint_names


def main():
    """Open the muscle arm in an interactive viewer with the shared HUD."""
    config, joint_names = build()
    controls = common.JointControlState(joint_names)
    # Idle drives the bicep at +RELAX_RATE so it stretches back to rest length;
    # holding key 1 sends -CONTRACT_RATE which contracts it.
    controls.as_custom(
        0,
        neutral=MUSCLE_RELAX_RATE,
        hi=-MUSCLE_CONTRACT_RATE,
        lo=-MUSCLE_CONTRACT_RATE,
    )
    state = common.HudState()

    def control(sim_id, controllers):
        for ctype, c in controllers:
            if ctype == common.STRUCTURE:
                rec = c.structure_record()
                if rec is not None:
                    state.update_bounds(rec)
                signals = c.signal().expansion_signals()
                if signals is not None:
                    rate = controls.signal(0)
                    for i in range(len(signals)):
                        signals[i] = float(rate)

    common.launch(
        config,
        control,
        hud=common.hud_factory(state, controls, robot_camera=False),
        on_event=lambda event, viewer: controls.handle_event(event),
        camera_focus=(0.0, 0.0, 0.6),
        camera_position=(-0.6, -0.5, 0.65),
    )


if __name__ == "__main__":
    main()
