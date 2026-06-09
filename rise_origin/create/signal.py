# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""One-line builders for :class:`RS_SignalConfig`.

A force signal lets a callback apply an external 3D force to a range of voxels
of a structure each control step. A torque signal applies an external 3D torque
to a single rigid body (torque is undefined for individual soft voxels). Declare
the signal here; read/write its value at runtime through the matching
controller's ``signal().force_signals()`` / ``signal().torque_signals()`` array.

Example::

    from rise_origin.create import SignalCreator, StructureCreator
    sc = StructureCreator("ball", voxel_size=0.01)
    ball = sc.add_box_body(size=(10, 10, 10), material="rigid",
                           relative_origin_position=(0, 0, 0))
    push = SignalCreator.force(
        "push", structure_name="ball",
        attached_to=ball.segment(0),
        voxel_range=(0, 0, 0, 9, 9, 9),
    )
    spin = SignalCreator.torque(
        "spin", structure_name="arm",
        attached_to=arm.segment(0),
    )
"""

from typing import Optional

from rise_origin import (
    RBound3ri,
    RS_ForceSignalConfig,
    RS_TorqueSignalConfig,
    RS_SignalConfig,
    RSE_SignalType,
)
from rise_origin.create.structure import SegmentRef
from rise_origin.create.utils import VoxelRange

__all__ = ["SignalCreator"]


class SignalCreator:
    """Factory of :class:`RS_SignalConfig` presets."""

    @staticmethod
    def force(
        name: str,
        *,
        structure_name: str,
        attached_to: SegmentRef,
        voxel_range: Optional[VoxelRange] = None,
    ) -> RS_SignalConfig:
        """Build a force signal applied to a voxel range of a structure segment.

        Args:
            name: Unique signal name.
            structure_name: Name of the target structure (must be non-empty).
            attached_to: Segment of the structure the force applies to
                (``body.segment(bid)`` from the structure builder).
            voxel_range: Inclusive voxel index box
                ``(min_x, min_y, min_z, max_x, max_y, max_z)`` the force is
                distributed over. ``None`` leaves the engine default range.

        Returns:
            A populated :class:`RS_SignalConfig`.
        """
        signal = RS_SignalConfig()
        signal.name = name
        signal.type = RSE_SignalType.RSE_SIGNAL_FORCE

        cfg = RS_ForceSignalConfig()
        cfg.structure_name = structure_name
        cfg.body_sid = attached_to.body_sid
        cfg.segment_bid = attached_to.segment_bid
        if voxel_range is not None:
            cfg.voxel_range = RBound3ri(*voxel_range)
        signal.config = cfg
        return signal

    @staticmethod
    def torque(
        name: str,
        *,
        structure_name: str,
        attached_to: SegmentRef,
    ) -> RS_SignalConfig:
        """Build a torque signal applied to a rigid body segment of a structure.

        Torque is only defined for rigid bodies; the target segment must be
        rigid (it is undefined for individual soft voxels).

        Args:
            name: Unique signal name.
            structure_name: Name of the target structure (must be non-empty).
            attached_to: Rigid segment of the structure the torque applies to.

        Returns:
            A populated :class:`RS_SignalConfig`.
        """
        signal = RS_SignalConfig()
        signal.name = name
        signal.type = RSE_SignalType.RSE_SIGNAL_TORQUE

        cfg = RS_TorqueSignalConfig()
        cfg.structure_name = structure_name
        cfg.body_sid = attached_to.body_sid
        cfg.segment_bid = attached_to.segment_bid
        signal.config = cfg
        return signal
