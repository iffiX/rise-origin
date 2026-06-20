# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Cube stack: a colorful pyramid of voxelized cubes.

What you'll learn:

* Build many cubes as separate bodies of one
  :class:`rsc.entity.StructureVoxel`, each its own rigid segment, placed with
  plain arithmetic and no joints.
* Stack them into a pyramid: each level is a centered ``s x s`` grid (for 10
  levels the sides are 10, 9, ... 1), each level resting on the one below.
* Tune the rigid-contact solver so flat-faced contact holds the pyramid steady.

Run it::

    python examples/basic/stack.py
    RISE_MODE=record RISE_OUT=stack.mp4 python examples/basic/stack.py
"""

import os
import sys

from rise_origin import rsc

# Put the examples root on the path so the sibling _common module is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _common  # noqa: E402

PILES = 1
VOXEL = 0.033
LEVELS = 10
CUBE = 3  # cube edge (voxels)
GAP = 0  # voxels between neighbours

SIZE = CUBE * VOXEL  # cube edge (m)
STEP = (CUBE + GAP) * VOXEL  # centre spacing within a level, and the level rise


def build():
    """Build the cube-stack world.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    materials = [
        _common.MaterialSpec(f"c{i}", color=color, kind="rigid")
        for i, color in enumerate(_common.PALETTE_CYCLE)
    ]
    n_colors = len(_common.PALETTE_CYCLE)

    structures = []
    for p in range(PILES):
        cube = 0
        bodies = []
        for level in range(LEVELS):
            side = LEVELS - level
            z = level * STEP  # min-corner height: each level sits on the last
            for i in range(side):
                for j in range(side):
                    x = (i - (side - 1) / 2.0) * STEP
                    y = (j - (side - 1) / 2.0) * STEP
                    bodies.append(
                        rsc.entity.StructureVoxelBody.box(
                            CUBE,
                            CUBE,
                            CUBE,
                            voxel_size=VOXEL,
                            material_sid=cube % n_colors,
                            segment_sid=cube,  # each cube its own rigid segment
                            relative_origin_position=(
                                x - SIZE / 2.0,
                                y - SIZE / 2.0,
                                z,
                            ),
                        )
                    )
                    cube += 1
        structures.append(
            rsc.entity.StructureVoxel(
                f"pile{p}",
                origin_position=(p * 10.0, 0.0, 0.0),
                material_references=[f"c{i}" for i in range(n_colors)],
                bodies=bodies,
            )
        )

    return _common.make_world(
        "stack",
        materials=materials,
        structures=structures,
        time=rsc.simulation.Time(
            dt=0.005,
            dt_rigidbody_contact_and_constraint_update=0.005,
            dt_contact_update=0.005,
        ),
        rigid_body=rsc.simulation.RigidBody(
            contact_iterations=20,
            contact_warm_start_damping_z=0.0,
            contact_position_correction_iterations=3,
            contact_position_correction_ratio=0.2,
        ),
    )


def main():
    """Build the world and launch it in the selected run mode (no control)."""
    _common.launch(
        build(),
        None,
        camera_position=(-0.15 * LEVELS, 0.0, 0.1 * LEVELS),
        camera_focus=(0.0, 0.0, 0.12),
    )


if __name__ == "__main__":
    main()
