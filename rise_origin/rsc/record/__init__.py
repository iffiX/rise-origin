# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Record characterizations.

A record declares state the simulation captures each record step and routes to
a controller (delivered through the world callback) and/or a recorder (written
to disk). The available records are *intersect-region* queries that report,
per record step, which structures currently have voxels inside a user-defined
region.

Example::

    from rise_origin import rsc

    gate = rsc.record.IntersectRegionCuboid(
        "finish_line",
        position=(2.0, 0.0, 0.5),
        size=(0.2, 2.0, 1.0),
        target_controller_reference="referee",
    )
"""

from rise_origin import RSC_RecordBase

from rise_origin.rsc.record.intersect_region import (
    IntersectRegionCuboid,
    IntersectRegionCuboidRigidBodyLinked,
    IntersectRegionSphere,
)

#: Abstract base of every record characterization (carries name + routing).
RecordBase = RSC_RecordBase

__all__ = [
    "RecordBase",
    "IntersectRegionCuboid",
    "IntersectRegionCuboidRigidBodyLinked",
    "IntersectRegionSphere",
]
