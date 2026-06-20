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
from rise_origin.rsc.world import World

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
]
