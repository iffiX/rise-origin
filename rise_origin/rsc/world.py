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
        self._wrap(self.resources, resources, RSC_ResourceVariant)
        self._wrap(self.essences, essences, RSC_EssenceVariant)
        self._wrap(self.entities, entities, RSC_EntityVariant)
        self._wrap(self.records, records, RSC_RecordVariant)
        self._wrap(self.signals, signals, RSC_SignalVariant)
        self._wrap(self.recorders, recorders, RSC_RecorderVariant)
        self._wrap(self.controllers, controllers, RSC_ControllerVariant)

    @staticmethod
    def _wrap(vector, items, variant_cls) -> None:
        """Wrap each builder object into its variant and append to the vector."""
        for item in items:
            vector.append(item if isinstance(item, variant_cls) else variant_cls(item))
