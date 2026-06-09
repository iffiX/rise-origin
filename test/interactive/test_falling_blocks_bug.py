# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Fuzzy test — random soft+rigid creatures fall from the sky.

Each creature is a two-part articulated structure:

* two identical voxel boxes (rigid core inside a soft shell), drawn from a
  random colour palette,
* connected by a single random joint (hinge or prismatic) that the engine
  drives with a random target every few control steps.

The scene continuously churns: every few callbacks a random batch of creatures
is deleted from the simulation and a fresh batch spawns from the sky. Used to
exercise structure add/delete, rigid+soft interaction, and joint stability
under randomized actuation.

Controls
* No keyboard joint control — the harness drives joints automatically.
* Mouse + ``WASD`` — orbit / pan / move the viewer camera.
* ``T`` — reset the viewer camera. ``Esc`` — quit.

Run::

    python test/interactive/test_falling_blocks.py
"""

import math
import random

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


# --- Population + churn ---
INITIAL_CREATURES = 10  # staggered to avoid interpenetration at spawn
MAX_ALIVE = 60
CHURN_INTERVAL = 5      # callbacks
DELETE_BATCH = 3
SPAWN_BATCH = 3
SIGNAL_REFRESH = 20     # re-randomize joint targets every N callbacks

# --- Spawn region (meters). Random spawns drop into a wide xy box and a height
# range; the initial creatures use a staggered Z to guarantee no interpenetration.
SPAWN_HALF_XY = 1.2
SPAWN_Z_MIN = 1.2
SPAWN_Z_MAX = 4.0
INITIAL_Z_PITCH = 0.8   # spacing along Z for the initial deterministic stack
VOXEL_MIN = 0.03
VOXEL_MAX = 0.06

# --- Creature geometry ---
CORE_VOXELS = 2
SHELL_VOXELS = 1
GAP_VOXELS = 1
SEGMENT_SOFT = 1
SEGMENT_RIGID = 2

# --- Joint params ---
HINGE_MIN = -1.2
HINGE_MAX = 1.2
HINGE_MAX_TORQUE = 0.0
PRISMATIC_HALF_RANGE_VOXELS = 2.0
PRISMATIC_MAX_FORCE = 0.0
PRISMATIC_MAX_SPEED = 1.0

# Hinge-signal randomization range (basic-position mode -> target angle in rad).
HINGE_SIGNAL_MIN = -1.0
HINGE_SIGNAL_MAX = 1.0

# --- Materials ---
DENSITY = 1000.0
POISSONS_RATIO = 0.35
U_STATIC = 1.0
U_DYNAMIC = 0.8
SOFT_ELASTIC_MODULUS = 3e5
RIGID_ELASTIC_MODULUS = 5e8
RIGID_ALBEDO = (60, 60, 70)
SOFT_PALETTE = [
    (200, 80, 80),
    (80, 200, 80),
    (80, 120, 220),
    (220, 200, 80),
    (200, 100, 200),
    (80, 200, 200),
    (230, 140, 60),
    (160, 160, 160),
]

RIGID_NAME = "rigid"


def _soft_names():
    return [f"soft_{i}" for i in range(len(SOFT_PALETTE))]


def _register_materials(config):
    """Attach all soft palette colours + one rigid material to ``config``."""
    for name, albedo in zip(_soft_names(), SOFT_PALETTE):
        config.material_configs.append(
            MaterialCreator.base(
                name,
                is_rigid=False,
                density=DENSITY,
                elastic_modulus=SOFT_ELASTIC_MODULUS,
                poissons_ratio=POISSONS_RATIO,
                u_static=U_STATIC,
                u_dynamic=U_DYNAMIC,
                min_expansion=-0.2,
                max_expansion=0.2,
                render=albedo,
            )
        )
    config.material_configs.append(
        MaterialCreator.base(
            RIGID_NAME,
            is_rigid=True,
            density=DENSITY,
            elastic_modulus=RIGID_ELASTIC_MODULUS,
            poissons_ratio=POISSONS_RATIO,
            u_static=U_STATIC,
            u_dynamic=U_DYNAMIC,
            min_expansion=-0.2,
            max_expansion=0.2,
            render=RIGID_ALBEDO,
        )
    )


def _make_part_arrays(soft_material):
    """Per-voxel ``(materials, segments)`` arrays for one half of a creature.

    A cube of side ``CORE_VOXELS + 2 * SHELL_VOXELS``: rigid core in the centre
    (segment ``RIGID``) wrapped on every face by a soft shell of ``soft_material``
    (segment ``SOFT``).
    """
    total = CORE_VOXELS + 2 * SHELL_VOXELS
    shell = SHELL_VOXELS
    materials = np.full((total, total, total), None, dtype=object)
    segments = np.full((total, total, total), RS_NULL_INDEX, dtype=np.int64)
    for iz in range(total):
        for iy in range(total):
            for ix in range(total):
                in_shell = (
                    ix < shell or ix >= total - shell
                    or iy < shell or iy >= total - shell
                    or iz < shell or iz >= total - shell
                )
                if in_shell:
                    materials[ix, iy, iz] = soft_material
                    segments[ix, iy, iz] = SEGMENT_SOFT
                else:
                    materials[ix, iy, iz] = RIGID_NAME
                    segments[ix, iy, iz] = SEGMENT_RIGID
    return materials, segments, total


def _random_unit_quaternion(rng):
    """Uniform random unit quaternion ``(x, y, z, w)``."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    a = math.sqrt(1.0 - u1)
    b = math.sqrt(u1)
    return (
        a * math.sin(2.0 * math.pi * u2),
        a * math.cos(2.0 * math.pi * u2),
        b * math.sin(2.0 * math.pi * u3),
        b * math.cos(2.0 * math.pi * u3),
    )


def create_creature(name, *, position, orientation, voxel, joint_kind, soft_material):
    """Build one creature: two soft-shell-over-rigid-core parts + 1 joint.

    Args:
        name: Structure name (must be unique within the running scene).
        position: World position of the structure origin (m).
        orientation: World orientation quaternion ``(x, y, z, w)``.
        voxel: Voxel edge length in meters.
        joint_kind: ``"hinge"`` or ``"prismatic"``.
        soft_material: Soft material name (one of ``_soft_names()``).

    Returns:
        A built :class:`RS_StructureConfig` ready for ``add_structure``.
    """
    sc = StructureCreator(
        name,
        voxel_size=voxel,
        origin_position=position,
        orientation=orientation,
    )

    materials, segments, total = _make_part_arrays(soft_material)
    part_a = sc.add_voxel_grid_body(
        materials,
        segment_bids=segments,
        relative_origin_position=(0.0, 0.0, 0.0),
    )
    part_b = sc.add_voxel_grid_body(
        materials,
        segment_bids=segments,
        relative_origin_position=((total + GAP_VOXELS) * voxel, 0.0, 0.0),
    )

    # Joint at the gap midpoint, centred in the part's Y/Z faces.
    cx = (total + GAP_VOXELS / 2.0) * voxel
    cy = total * voxel / 2.0
    cz = total * voxel / 2.0

    if joint_kind == "hinge":
        sc.add_hinge_joint(
            a=part_a.segment(SEGMENT_RIGID),
            b=part_b.segment(SEGMENT_RIGID),
            relative_axis=(0.0, 1.0, 0.0),
            relative_anchor=(cx, cy, cz),
            signal_sid=0,
            angle_min=HINGE_MIN,
            angle_max=HINGE_MAX,
            max_torque=HINGE_MAX_TORQUE,
        )
    else:
        sc.add_prismatic_joint(
            a=part_a.segment(SEGMENT_RIGID),
            b=part_b.segment(SEGMENT_RIGID),
            relative_axis=(1.0, 0.0, 0.0),
            relative_anchor=(cx, cy, cz),
            signal_sid=0,
            distance_min=-PRISMATIC_HALF_RANGE_VOXELS * voxel,
            distance_max=PRISMATIC_HALF_RANGE_VOXELS * voxel,
            max_force=PRISMATIC_MAX_FORCE,
            max_speed=PRISMATIC_MAX_SPEED,
        )
    return sc.build()


def _random_creature(rng, name, *, position=None):
    """Build one creature with random voxel size / orientation / joint.

    Args:
        rng: A ``random.Random`` for reproducible randomness.
        name: Structure name.
        position: World position in meters; defaults to a random point in the
            spawn box. Pass an explicit position for the deterministic initial
            stack so creatures don't interpenetrate.
    """
    voxel = rng.uniform(VOXEL_MIN, VOXEL_MAX)
    if position is None:
        position = (
            rng.uniform(-SPAWN_HALF_XY, SPAWN_HALF_XY),
            rng.uniform(-SPAWN_HALF_XY, SPAWN_HALF_XY),
            rng.uniform(SPAWN_Z_MIN, SPAWN_Z_MAX),
        )
    orientation = _random_unit_quaternion(rng)
    joint_kind = rng.choice(["hinge", "prismatic"])
    soft = rng.choice(_soft_names())
    return create_creature(
        name,
        position=position,
        orientation=orientation,
        voxel=voxel,
        joint_kind=joint_kind,
        soft_material=soft,
    )


def build(rng):
    """Build the initial scene; returns ``config`` and the list of creature names."""
    config = common.make_config()
    _register_materials(config)

    # Deterministic Z-stack so the initial creatures cannot overlap. Random
    # spawns at runtime use the full SPAWN_Z range.
    names = []
    for i in range(INITIAL_CREATURES):
        name = f"creature_{i}"
        position = (
            rng.uniform(-SPAWN_HALF_XY * 0.4, SPAWN_HALF_XY * 0.4),
            rng.uniform(-SPAWN_HALF_XY * 0.4, SPAWN_HALF_XY * 0.4),
            SPAWN_Z_MIN + i * INITIAL_Z_PITCH,
        )
        config.structure_configs.append(
            _random_creature(rng, name, position=position)
        )
        names.append(name)
    return config, names


def main():
    """Run the fuzzy churn test with the shared HUD."""
    rng = random.Random(0)
    config, alive_names = build(rng)
    controls = common.JointControlState([])  # no manual joint control
    state = common.HudState()
    spawn_counter = {"next": 0}
    frame_counter = {"n": 0}

    # The Viewer reference is captured so we can reach the simulation handle for
    # add/delete from inside the per-step callback.
    viewer_holder = {"viewer": None}

    def control(sim_id, controllers):
        frame_counter["n"] += 1
        refresh = frame_counter["n"] % SIGNAL_REFRESH == 0

        viewer = viewer_holder["viewer"]
        handle = viewer.handle if viewer is not None else None
        if handle is not None and frame_counter["n"] % CHURN_INTERVAL == 0:
            n_delete = min(DELETE_BATCH, len(alive_names))
            for name in rng.sample(alive_names, n_delete):
                alive_names.remove(name)
                try:
                    gid = handle.find_structure(name)
                    if gid != RS_NULL_INDEX:
                        handle.delete_structure(gid)
                except Exception as exc:  # noqa: BLE001
                    print(f"delete_structure({name}) failed: {exc}")

            room = MAX_ALIVE - len(alive_names)
            for _ in range(min(SPAWN_BATCH, max(0, room))):
                name = f"spawn_{spawn_counter['next']}"
                spawn_counter["next"] += 1
                try:
                    handle.add_structure(_random_creature(rng, name))
                    alive_names.append(name)
                except Exception as exc:  # noqa: BLE001
                    print(f"add_structure({name}) failed: {exc}")

        for ctype, c in controllers:
            if ctype != common.STRUCTURE:
                continue
            if not refresh:
                continue
            hinge = c.signal().hinge_signals()
            if hinge is not None:
                for i in range(len(hinge)):
                    hinge[i] = rng.uniform(HINGE_SIGNAL_MIN, HINGE_SIGNAL_MAX)
            prismatic = c.signal().prismatic_signals()
            if prismatic is not None:
                for i in range(len(prismatic)):
                    prismatic[i] = rng.uniform(0.0, 1.0)

    rise = rs.RiseOrigin("cuda:0")
    from rise_origin.viewer import Viewer

    viewer = Viewer(
        rise,
        config,
        callback=control,
        hud=common.hud_factory(
            state, controls, robot_camera=False, title="Falling Blocks (auto-churn)"
        ),
        camera_focus=(0.0, 0.0, 0.5),
        camera_position=(-2.0, -1.0, 1.0),
    )
    viewer_holder["viewer"] = viewer
    viewer.run_interactive()
    viewer.wait()


if __name__ == "__main__":
    main()
