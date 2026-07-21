# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Author a world's *Characterization* -- the ``C`` in ``RSC``.

A characterization is the canonical, declarative description of *what exists* in
a world: its resources, essences, entities, records, signals, recorders, and
controllers. The simulation reads this description and derives the consequences;
the characterization itself holds no device memory and runs no physics.

This package provides one thin class per ``RSC_*`` type, named with the prefix
dropped and grouped by axis, so the authoring call mirrors the bound type
one-to-one:

==========================  =====================================
bound type                  builder
==========================  =====================================
``RSC_IntersectRegionSphere``  ``rsc.record.IntersectRegionSphere``
``RSC_StructureVoxel``         ``rsc.entity.StructureVoxel``
``RSC_Material``               ``rsc.essence.Material``
==========================  =====================================

Each builder subclasses its bound type, so instances are accepted anywhere the
bound type is (variants, the world, pickling), and adds an ergonomic
``__init__`` with documented defaults that accepts plain tuples for vectors,
quaternions, and colors.

Pickling any node carries its archive bytes. At the default protocol those bytes
are copied into the pickle stream. To skip that copy on a large scene, pickle
with protocol 5 and collect the archive bytes out of band, which hands the fast
packer's buffer straight to pickle::

    buffers = []
    data = pickle.dumps(world, protocol=5, buffer_callback=buffers.append)
    world = pickle.loads(data, buffers=[b.raw() for b in buffers])

For procedural, composed world generation built on top of these primitives, see
the sibling ``create`` package.
"""

from rise_origin.rsc import (
    simulation,
    resource,
    essence,
    entity,
    record,
    signal,
    recorder,
    controller,
)
from rise_origin import RS_RSCFile
from rise_origin.rsc.world import World

#: Save any RSC node to a .rsct or .rscb file.
save = RS_RSCFile.save
#: Load any RSC node from a .rsct or .rscb file.
load = RS_RSCFile.load
#: Getter of the RSC schema version this build writes.
current_version = RS_RSCFile.current_version
#: Getter of the engine build version.
engine_version = RS_RSCFile.engine_version

__all__ = [
    "simulation",
    "resource",
    "essence",
    "entity",
    "record",
    "signal",
    "recorder",
    "controller",
    "World",
    "save",
    "load",
    "current_version",
    "engine_version",
]
