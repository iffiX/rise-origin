# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Voxel-object catalog grouped by category.

Each ``build_*(rng, **kwargs)`` returns a :class:`scene.stamp.Stamp` ready to
be pasted into a :class:`scene.world.World`. Categories:

* :mod:`fixtures` — ceiling strips, doorway frames, signs, hazard markings.
* :mod:`furniture` — chairs, stools, desks, workbenches, bunk beds, tables.
* :mod:`storage` — lockers, crate stacks, floor grates.
* :mod:`architecture` — wall vents, pipe runs, cable bundles, conduits.
* :mod:`equipment` — monitors, server racks, specimen tanks, the reactor core.
* :mod:`decoration` — plants, debris, knocked-out ceilings, door panels.
"""
