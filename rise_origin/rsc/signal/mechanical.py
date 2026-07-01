# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Mechanical signals applied to a structure.

A signal is a per-step value an entity reads to drive its behavior. Mechanical
signals are produced by the controller named in ``source_controller_reference`` (set
each callback step) and applied to the targeted structure.
"""

from rise_origin import RS_NULL_INDEX, RSC_Force, RSC_Torque, RSC_Transform

from rise_origin.rsc._utils import Bound3ri

__all__ = ["Force", "Torque", "Transform"]


class Force(RSC_Force):
    """An external force applied to a structure's voxels or rigid body.

    The target is either a voxel index range (for soft voxels) or a rigid body
    segment (via ``body_sid`` / ``segment_bid``), within ``structure_name``.

    Args:
        name: Unique signal name.
        structure_name: Name of the structure the force is applied to.
        source_controller_reference: Name of the controller that drives this signal's
            value each callback step.
        voxel_range: Inclusive voxel-index box the force is distributed over.
            ``None`` keeps the engine default (empty range).
        body_sid: Target rigid body sub-id, or ``RS_NULL_INDEX`` for none.
        segment_bid: Target rigid body segment id, or ``RS_NULL_INDEX`` for none.
    """

    def __init__(
        self,
        name: str,
        *,
        structure_name: str,
        source_controller_reference: str = "",
        voxel_range: Bound3ri = None,
        body_sid: int = RS_NULL_INDEX,
        segment_bid: int = RS_NULL_INDEX,
    ) -> None:
        super().__init__()
        self.name = name
        self.source_controller_reference = source_controller_reference
        self.structure_name = structure_name
        if voxel_range is not None:
            self.voxel_range = voxel_range
        self.body_sid = body_sid
        self.segment_bid = segment_bid


class Torque(RSC_Torque):
    """An external torque applied to a structure's rigid body.

    Args:
        name: Unique signal name.
        structure_name: Name of the structure the torque is applied to.
        source_controller_reference: Name of the controller that drives this signal's
            value each callback step.
        body_sid: Target rigid body sub-id, or ``RS_NULL_INDEX`` for none.
        segment_bid: Target rigid body segment id, or ``RS_NULL_INDEX`` for none.
    """

    def __init__(
        self,
        name: str,
        *,
        structure_name: str,
        source_controller_reference: str = "",
        body_sid: int = RS_NULL_INDEX,
        segment_bid: int = RS_NULL_INDEX,
    ) -> None:
        super().__init__()
        self.name = name
        self.source_controller_reference = source_controller_reference
        self.structure_name = structure_name
        self.body_sid = body_sid
        self.segment_bid = segment_bid


class Transform(RSC_Transform):
    """A one-shot transform applied to the whole structure.

    The controller sets a translation, rotation, and pivot each callback step.
    The transform is applied once to every voxel and rigid body of
    ``structure_name`` regardless of material, then reset to identity, so the
    next read shows zero translation and a unit quaternion.

    Args:
        name: Unique signal name.
        structure_name: Name of the structure the transform is applied to.
        source_controller_reference: Name of the controller that drives this signal's
            value each callback step.
    """

    def __init__(
        self,
        name: str,
        *,
        structure_name: str,
        source_controller_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.source_controller_reference = source_controller_reference
        self.structure_name = structure_name
