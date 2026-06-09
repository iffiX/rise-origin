# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Interactive force-signal test — push a rigid ball and lift a soft cube.

Two structures share the scene:

* A voxelized **ball** built as a rigid sphere core wrapped in a soft shell.
  The shell gives it real floor contact; a force signal acts on the rigid
  core's local ``X`` / ``Y`` axes and is rotated into world space each frame
  using the core's current orientation.
* A purely soft **cube** lifted along world Z by a single force signal.

Controls (handled by :mod:`_test_common`)
* ``1`` / ``2`` — push the ball along its rigid core's local ``+X`` / ``+Y``
  (``Shift`` reverses).
* ``3`` — lift / press the soft cube along world Z (``Shift`` reverses).
* Release any key to set its force component back to 0.
* Mouse + ``WASD`` — orbit / pan / move the viewer camera.
* ``T`` — reset the viewer camera. ``Esc`` — quit.

Run::

    python test/interactive/test_force.py
"""

from pyquaternion import Quaternion

import rise_origin as rs
from rise_origin.create import (
    MaterialCreator,
    SignalCreator,
    StructureCreator,
)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _test_common as common


VOXEL = 0.01

# Ball: soft shell over a rigid core, both voxelized spheres.
BALL_RADIUS = 5         # outer (soft) radius in voxels
BALL_CORE_RADIUS = 3    # inner (rigid) radius in voxels

# Cube: pure soft box.
CUBE_SIZE = 8

# Force magnitudes (N) at full key press.
BALL_FORCE = 4.0
CUBE_FORCE = 10.0

BALL_NAME = "ball"
CUBE_NAME = "cube"
BALL_SIGNAL = "ball_thrust"
CUBE_SIGNAL = "cube_lift"


def build():
    """Build the scene; returns ``(config, joint_names)``."""
    config = common.make_config()

    config.material_configs.append(
        MaterialCreator.soft("ball_skin", elastic_modulus=3e5, render=(200, 140, 60))
    )
    config.material_configs.append(
        MaterialCreator.rigid("ball_core", render=(70, 70, 80))
    )
    config.material_configs.append(
        MaterialCreator.soft("cube_soft", elastic_modulus=3e5, render=(90, 160, 200))
    )

    # Ball: rigid sphere wrapped in a soft shell.
    ball_sc = StructureCreator(
        BALL_NAME,
        voxel_size=VOXEL,
        origin_position=(0.0, -0.12, 0.1),
    )
    ball_body = ball_sc.add_sphere_body(
        BALL_RADIUS,
        material="ball_core",
        shell=BALL_RADIUS - BALL_CORE_RADIUS,
        shell_material="ball_skin",
        relative_origin_position=(0.0, 0.0, 0.0),
    )
    config.structure_configs.append(ball_sc.build())

    # Cube: pure soft block.
    cube_sc = StructureCreator(
        CUBE_NAME,
        voxel_size=VOXEL,
        origin_position=(0.0, 0.12, 0.1),
    )
    cube_body = cube_sc.add_box_body(
        (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        material="cube_soft",
        relative_origin_position=(0.0, 0.0, 0.0),
    )
    config.structure_configs.append(cube_sc.build())

    # Force signals.
    d = 2 * BALL_RADIUS + 1
    config.signal_configs.append(
        SignalCreator.force(
            BALL_SIGNAL,
            structure_name=BALL_NAME,
            attached_to=ball_body.core,
            voxel_range=(0, 0, 0, d - 1, d - 1, d - 1),
        )
    )
    config.signal_configs.append(
        SignalCreator.force(
            CUBE_SIGNAL,
            structure_name=CUBE_NAME,
            attached_to=cube_body.segment(0),
            voxel_range=(0, 0, 0, CUBE_SIZE - 1, CUBE_SIZE - 1, CUBE_SIZE - 1),
        )
    )

    joint_names = [
        "Ball thrust local-X (rigid)",
        "Ball thrust local-Y (rigid)",
        "Cube lift world-Z (soft)",
    ]
    return config, joint_names


def main():
    """Open the force-signal scene in an interactive viewer with the shared HUD."""
    config, joint_names = build()
    controls = common.JointControlState(joint_names)
    # Three force-magnitude knobs in [-1, 1]; neutral 0 = no force.
    for i in range(len(joint_names)):
        controls.as_custom(i, neutral=0.0, hi=1.0, lo=-1.0)
    state = common.HudState()

    # Latest ball rigid-core orientation, refreshed each step from the structure
    # controller and read by the force controller to rotate local thrust into world.
    ball_orientation = [0.0, 0.0, 0.0, 1.0]

    def control(sim_id, controllers):
        for ctype, c in controllers:
            if ctype == common.STRUCTURE:
                rec = c.structure_record()
                if rec is None:
                    continue
                if c.name() == BALL_NAME:
                    state.update_bounds(rec)
                    ori = rec.rigid_body_orientation()
                    if ori is not None and len(ori) > 0:
                        ball_orientation[:] = [
                            float(ori[0][0]),
                            float(ori[0][1]),
                            float(ori[0][2]),
                            float(ori[0][3]),
                        ]
            elif ctype == common.FORCE:
                force = c.signal().force_signals()
                if force is None or len(force) < 3:
                    continue
                if c.name() == BALL_SIGNAL:
                    local = [
                        controls.signal(0) * BALL_FORCE,
                        controls.signal(1) * BALL_FORCE,
                        0.0,
                    ]
                    q = ball_orientation
                    world = Quaternion(w=q[3], x=q[0], y=q[1], z=q[2]).rotate(local)
                    force[0], force[1], force[2] = (
                        float(world[0]),
                        float(world[1]),
                        float(world[2]),
                    )
                elif c.name() == CUBE_SIGNAL:
                    force[0] = 0.0
                    force[1] = 0.0
                    force[2] = controls.signal(2) * CUBE_FORCE

    common.launch(
        config,
        control,
        hud=common.hud_factory(state, controls, robot_camera=False),
        on_event=lambda event, viewer: controls.handle_event(event),
        camera_focus=(0.0, 0.0, 0.05),
        camera_position=(-0.4, -0.3, 0.25),
    )


if __name__ == "__main__":
    main()
