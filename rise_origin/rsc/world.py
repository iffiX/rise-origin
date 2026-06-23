# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The top-level world characterization.

A :class:`World` is the complete, declarative description handed to the
simulator: its simulation config plus the resources, essences, entities,
records, signals, recorders, and controllers that populate it. Builder objects
are passed as plain lists and wrapped into their variants automatically, so the
``RSC_*Variant`` types never appear in user code.
"""

from typing import Optional, Sequence

from rise_origin import (
    RSC_ControllerVariant,
    RSC_EntityVariant,
    RSC_EssenceVariant,
    RSC_RecordVariant,
    RSC_RecorderVariant,
    RSC_ResourceVariant,
    RSC_SignalVariant,
    RSC_World,
)

from rise_origin.rsc.simulation import Simulation

__all__ = ["World"]


class World(RSC_World):
    """A complete world description.

    Args:
        name: Unique world name.
        simulation: Simulation config; defaults to a fresh :class:`Simulation`.
        resources: Resource builders (e.g. textures).
        essences: Essence builders (materials and their properties).
        entities: Entity builders (structures, cameras, floors, cosmic).
        records: Record builders (intersect-region queries).
        signals: Signal builders (forces, torques).
        recorders: Recorder builders.
        controllers: Controller builders.
    """

    def __init__(
        self,
        name: str = "world",
        *,
        simulation: Optional[Simulation] = None,
        resources: Sequence = (),
        essences: Sequence = (),
        entities: Sequence = (),
        records: Sequence = (),
        signals: Sequence = (),
        recorders: Sequence = (),
        controllers: Sequence = (),
    ) -> None:
        super().__init__()
        self.name = name
        self.simulation = simulation if simulation is not None else Simulation()
        # The container casters wrap each concrete builder into its variant on assignment.
        self.resources = list(resources)
        self.essences = list(essences)
        self.entities = list(entities)
        self.records = list(records)
        self.signals = list(signals)
        self.recorders = list(recorders)
        self.controllers = list(controllers)
