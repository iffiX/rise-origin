# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Interactive hinge-joint test — a quadruped with 8 hinge-actuated joints.

A rigid-core/soft-shell torso plus four upper + lower legs, all connected by
hinge joints (four hips, four knees). Drives every hinge by keyboard so you can
push each DOF to its limit and watch the solver respond. Mirrors the geometry
of ``examples/creatures/quadruped.py`` but with manual joint control instead of
a baked-in gait.

Controls (handled by :mod:`_test_common`)
* ``1``-``4`` — hip joints (front-left, front-right, back-left, back-right).
* ``5``-``8`` — knee joints (same order).
  Plain key drives the hinge to its positive limit; ``Shift`` drives the
  negative limit; release returns to 0 rad (true neutral).
* Mouse + ``WASD`` — orbit / pan / move the viewer camera.
* ``T`` — reset the viewer camera. ``Esc`` — quit.

Run::

    python test/interactive/test_quadruped.py
"""

import math

import numpy as np

import rise_origin as rs
from rise_origin.create import (
    CameraCreator,
    FloorCreator,
    MaterialCreator,
    StructureCreator,
)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _test_common as common


def _trapezoid_wave_floor(
    x_values: int,
    y_values: int,
    *,
    base_length: int = 10,
    top_length: int = 5,
    height: float = 0.05,
) -> np.ndarray:
    """Generate a 2-D wave heightmap of alternating trapezoidal protrusions/trenches.

    The output is a ``[y_values, x_values]`` float array, tiled in both axes
    with a period of ``2 * base_length`` so it wraps seamlessly when the engine
    repeats it. Each cell of size ``base_length`` is a flat-topped trapezoid;
    cells in a checkerboard pattern flip sign so adjacent tiles form bumps and
    pits in alternation (creating walkable footing for the quadruped to push
    against).

    Args:
        x_values: Heightmap width in samples.
        y_values: Heightmap height in samples.
        base_length: Trapezoid base in samples (full cell pitch).
        top_length: Trapezoid flat-top length in samples.
        height: Peak protrusion/trench height in meters.

    Returns:
        A ``[y_values, x_values]`` float array of heights in meters.
    """
    floor = np.zeros((y_values, x_values))
    center_x = x_values // 2
    center_y = y_values // 2
    half_base = base_length / 2
    half_top = top_length / 2
    slope_band = half_base - half_top
    for x in range(x_values):
        for y in range(y_values):
            x_rel = (x - center_x) % base_length
            y_rel = (y - center_y) % base_length
            phase = (
                1
                if (
                    ((x - center_x) // base_length) % 2
                    == ((y - center_y) // base_length) % 2
                )
                else -1
            )

            in_top_x = half_base - half_top <= x_rel <= half_base + half_top
            in_top_y = half_base - half_top <= y_rel <= half_base + half_top
            if in_top_x and in_top_y:
                value = height
            else:
                if x_rel < slope_band:
                    x_h = (x_rel / slope_band) * height
                elif x_rel <= half_base + half_top:
                    x_h = height
                else:
                    x_h = ((base_length - x_rel) / slope_band) * height
                if y_rel < slope_band:
                    y_h = (y_rel / slope_band) * height
                elif y_rel <= half_base + half_top:
                    y_h = height
                else:
                    y_h = ((base_length - y_rel) / slope_band) * height
                value = min(x_h, y_h)
            floor[y, x] = phase * value
    return floor


# Terrain tile.
FLOOR_X_VALUES = 40
FLOOR_Y_VALUES = 40
FLOOR_SIZE_M = 0.4  # cell pitch = 0.4 / 40 = 0.01 m (matches voxel size)
FLOOR_TRAPEZOID_BASE = 10
FLOOR_TRAPEZOID_TOP = 5
FLOOR_TRAPEZOID_HEIGHT = 0.05  # m


# --- Voxel grids (rigid cores) ---
TORSO = (30, 11, 7)
LEG = (2, 2, 12)

# Soft-shell thickness (voxels) around each rigid core.
TORSO_SHELL = 1
UPPER_SHELL = 1
LOWER_SHELL = 2

# Limb tilt angles about +Y.
UPPER_ANGLE = math.radians(45.0)
LOWER_ANGLE = math.radians(-45.0)

# Lateral offsets (voxels) that splay the legs out to avoid self-collision.
UPPER_LATERAL = 3
LOWER_LATERAL = 7

# Hinge limits and motor torque.
HIP_MIN, HIP_MAX = -0.7, 0.7
KNEE_MIN, KNEE_MAX = -1.5, 1.5
HINGE_TORQUE = 10.0

VOXEL = 0.01

# On-board camera.
ROBOT_CAMERA_NAME = "torso_camera"
ROBOT_CAMERA_ANCHOR = (0.20, 0.0, 0.27)
ROBOT_CAMERA_ORIENTATION = (0.0, 0.2164, 0.0, 0.9763)

BODY_COLOR = (53, 33, 0)
UPPER_COLOR = (102, 153, 153)
LOWER_COLOR = (204, 153, 102)


def _quat_axis_angle(axis, angle):
    """Quaternion ``(x, y, z, w)`` for a rotation of ``angle`` about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2.0
    s = math.sin(half)
    return (
        float(axis[0] * s),
        float(axis[1] * s),
        float(axis[2] * s),
        float(math.cos(half)),
    )


def _rotate(vec, quat):
    """Rotate 3-vector ``vec`` by quaternion ``quat`` given as ``(x, y, z, w)``."""
    vec = np.asarray(vec, dtype=np.float64)
    qv = np.asarray(quat[:3], dtype=np.float64)
    w = quat[3]
    return vec + 2.0 * np.cross(qv, np.cross(qv, vec) + w * vec)


def _full_corner(core_corner, orientation, shell):
    """Min-corner of the full (shelled) body so its rigid core stays put.

    The shell adds ``shell`` voxels on every side; shifting the body origin by
    the rotated shell offset keeps the rigid core at ``core_corner``.
    """
    offset = _rotate((shell * VOXEL, shell * VOXEL, shell * VOXEL), orientation)
    return tuple(np.asarray(core_corner, dtype=np.float64) - offset)


def build():
    """Build the quadruped scene; returns ``(config, joint_names)``."""
    config = common.make_config(control_hz=40.0, record_hz=40.0)

    # Trapezoidal-wave terrain tiled in X and Y.
    wave = _trapezoid_wave_floor(
        FLOOR_X_VALUES,
        FLOOR_Y_VALUES,
        base_length=FLOOR_TRAPEZOID_BASE,
        top_length=FLOOR_TRAPEZOID_TOP,
        height=FLOOR_TRAPEZOID_HEIGHT,
    )
    config.floor_config = FloorCreator.array(
        wave,
        size=FLOOR_SIZE_M,
        is_x_repeat=True,
        is_y_repeat=True,
        render=common.FLOOR,
    )

    for name, color in [
        ("body", BODY_COLOR),
        ("upper", UPPER_COLOR),
        ("lower", LOWER_COLOR),
    ]:
        config.material_configs.append(
            MaterialCreator.rigid(f"{name}_core", render=color)
        )
        config.material_configs.append(
            MaterialCreator.soft(f"{name}_skin", render=color)
        )

    sc = StructureCreator("quadruped", voxel_size=VOXEL)

    torso_size = np.array(TORSO) * VOXEL
    leg_size = np.array(LEG) * VOXEL

    # Torso centred in XY, lifted so the angled legs reach the floor.
    torso_height = torso_size[2] / 2.0 + leg_size[2] * math.cos(UPPER_ANGLE) * 2.0
    torso_corner = (
        -torso_size[0] / 2.0,
        -torso_size[1] / 2.0,
        torso_height - torso_size[2] / 2.0,
    )
    torso = sc.add_box_body(
        TORSO,
        material="body_core",
        shell=TORSO_SHELL,
        shell_material="body_skin",
        relative_origin_position=_full_corner(
            torso_corner, (0.0, 0.0, 0.0, 1.0), TORSO_SHELL
        ),
    )

    # Four legs at the torso's bottom corners. Front legs splay -Y, back legs +Y.
    leg_corners = [
        (leg_size[0] / 2.0, 0.0, True),
        (torso_size[0] - leg_size[0] / 2.0, 0.0, True),
        (leg_size[0] / 2.0, torso_size[1], False),
        (torso_size[0] - leg_size[0] / 2.0, torso_size[1], False),
    ]
    hip_z = torso_size[2] / 2.0

    for leg_idx, (corner_x, corner_y, is_front) in enumerate(leg_corners):
        lateral = -1.0 if is_front else 1.0
        axis = (0.0, 1.0 if is_front else -1.0, 0.0)

        upper_quat = _quat_axis_angle((0.0, 1.0, 0.0), UPPER_ANGLE)
        hip_anchor = (
            torso_corner[0] + corner_x,
            torso_corner[1] + corner_y + lateral * UPPER_LATERAL * VOXEL,
            torso_corner[2] + hip_z,
        )
        # Place the upper leg so its top-centre lands on the hip anchor.
        top_local = np.array([leg_size[0] / 2.0, leg_size[1] / 2.0, leg_size[2]])
        upper_corner = np.array(hip_anchor) - _rotate(top_local, upper_quat)
        upper = sc.add_box_body(
            LEG,
            material="upper_core",
            shell=UPPER_SHELL,
            shell_material="upper_skin",
            relative_origin_position=_full_corner(
                tuple(upper_corner), upper_quat, UPPER_SHELL
            ),
            relative_orientation=upper_quat,
        )
        sc.add_hinge_joint(
            a=torso.core,
            b=upper.core,
            relative_axis=axis,
            relative_anchor=hip_anchor,
            signal_sid=leg_idx,
            angle_min=HIP_MIN,
            angle_max=HIP_MAX,
            max_torque=HINGE_TORQUE,
        )

        # Knee at the bottom-centre of the upper leg.
        knee_local = np.array([leg_size[0] / 2.0, leg_size[1] / 2.0, 0.0])
        knee_anchor = upper_corner + _rotate(knee_local, upper_quat)
        lower_quat = _quat_axis_angle((0.0, 1.0, 0.0), LOWER_ANGLE)
        lower_corner = np.array(knee_anchor) - _rotate(top_local, lower_quat)
        lower_corner[1] += lateral * (LOWER_LATERAL - UPPER_LATERAL) * VOXEL
        lower = sc.add_box_body(
            LEG,
            material="lower_core",
            shell=LOWER_SHELL,
            shell_material="lower_skin",
            relative_origin_position=_full_corner(
                tuple(lower_corner), lower_quat, LOWER_SHELL
            ),
            relative_orientation=lower_quat,
        )
        sc.add_hinge_joint(
            a=upper.core,
            b=lower.core,
            relative_axis=axis,
            relative_anchor=tuple(knee_anchor),
            signal_sid=4 + leg_idx,
            angle_min=KNEE_MIN,
            angle_max=KNEE_MAX,
            max_torque=HINGE_TORQUE,
        )

    config.structure_configs.append(sc.build())
    config.camera_configs.append(
        CameraCreator.rigid_body_linked(
            ROBOT_CAMERA_NAME,
            structure_name="quadruped",
            attached_to=torso.core,
            relative_anchor=ROBOT_CAMERA_ANCHOR,
            relative_orientation=ROBOT_CAMERA_ORIENTATION,
            image_width=common.ROBOT_CAMERA_SIZE,
            image_height=common.ROBOT_CAMERA_SIZE,
            samples_per_pixel=16,
            field_of_view=60.0,
            max_depth=4,
        )
    )

    joint_names = [
        "Hip front-left",
        "Hip front-right",
        "Hip back-left",
        "Hip back-right",
        "Knee front-left",
        "Knee front-right",
        "Knee back-left",
        "Knee back-right",
    ]
    return config, joint_names


def main():
    """Open the quadruped in an interactive viewer with the shared HUD."""
    config, joint_names = build()
    controls = common.JointControlState(joint_names)
    for i in range(4):
        controls.as_hinge(i, hi=HIP_MAX, lo=HIP_MIN)
    for i in range(4, 8):
        controls.as_hinge(i, hi=KNEE_MAX, lo=KNEE_MIN)
    state = common.HudState()

    def control(sim_id, controllers):
        for ctype, c in controllers:
            if ctype == common.STRUCTURE:
                rec = c.structure_record()
                if rec is not None:
                    state.update_bounds(rec)
                controls.apply_to(c.signal().hinge_signals())
                for cam in c.camera_records():
                    state.update_robot_frame(cam)

    common.launch(
        config,
        control,
        hud=common.hud_factory(state, controls, robot_camera=True),
        on_event=lambda event, viewer: controls.handle_event(event),
        camera_focus=(0.0, 0.0, 0.12),
        camera_position=(-0.6, -0.4, 0.4),
    )


if __name__ == "__main__":
    main()
