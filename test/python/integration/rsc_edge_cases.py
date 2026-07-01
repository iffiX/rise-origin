# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Integration tests for rsc edge cases that only surface once a world runs.

Each case builds a full world, hands it to the engine, waits for it to end, and
asserts on the terminal world status. These exercise the kernel build path that
the unit tests over the rsc layer alone cannot reach.
"""

import pytest

import rise_origin as rs
from rise_origin import rsc

VOXEL = 0.02
BOX = 2
GAP = 3

# The connected-body collision mask is 128 bits wide, so a constrained structure
# holds at most 128 rigid bodies.
CONSTRAINED_BODY_CAP = 128


@pytest.mark.sim
@pytest.mark.parametrize(
    "body_num, expected_status",
    [
        (CONSTRAINED_BODY_CAP, rs.RSE_WorldStatus.RSE_WORLD_END_SUCCESS),
        (
            CONSTRAINED_BODY_CAP + 1,
            rs.RSE_WorldStatus.RSE_WORLD_END_FAILED_WITH_INVALID_CONFIG,
        ),
    ],
)
def test_constrained_structure_body_cap(body_num, expected_status):
    """A constrained hinge chain runs at the cap and is rejected one past it.

    Crossing the 128 rigid body cap once dumped core, as the mask was written out
    of bounds while the kernel built the structure. The over-cap structure must
    instead end the world with a clean invalid-config status, the way an
    out-of-range simulation parameter does.

    Args:
        body_num: Number of boxes in the chain, so the number of rigid bodies.
        expected_status: The world status the engine must reach.
    """
    bodies = []
    constraints = []
    for i in range(body_num):
        bodies.append(
            rsc.entity.StructureVoxelBody.box(
                BOX,
                BOX,
                BOX,
                voxel_size=VOXEL,
                relative_origin_position=(i * GAP * VOXEL, 0.0, 0.0),
            )
        )
        if i > 0:
            constraints.append(
                rsc.entity.StructureVoxelConstraint(
                    rs.RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_HINGE_JOINT,
                    a_body_sid=i - 1,
                    a_segment_bid=0,
                    b_body_sid=i,
                    b_segment_bid=0,
                    relative_anchor=((i * GAP - 0.5) * VOXEL, 0.0, 0.0),
                    hinge_relative_axis=(0.0, 0.0, 1.0),
                )
            )

    world = rsc.World(
        "constrained_cap",
        essences=[
            rsc.essence.Material("seg"),
            rsc.essence.MaterialPhysicalPropertyRigid(
                "seg#phys", material_reference="seg", density=1000.0
            ),
        ],
        entities=[
            rsc.entity.StructureVoxel(
                "chain",
                material_references=["seg"],
                bodies=bodies,
                constraints=constraints,
            )
        ],
        simulation=rsc.simulation.Simulation(
            environment=rsc.simulation.Environment(gravity_acceleration=0.0),
            runtime=rsc.simulation.Runtime(max_time=0.05, log_level="warn"),
        ),
    )

    origin = rs.RiseOrigin("cpu")
    handle, _ = origin.hello(world)
    handle.wait_for_end()
    assert handle.is_end() is True
    assert handle.get_status() == expected_status
