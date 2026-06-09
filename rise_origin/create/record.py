# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""One-line builders for :class:`RS_RecordConfig`.

A record selects which per-structure state the simulation exposes (and, if
saving is enabled, writes to disk): voxel state, link state, rigid-body state,
and joint state. Each category has a "range" that picks ``"none"``, ``"all"``,
or (voxel/link only) a random sample of size ``*_sample_num``.

Example::

    from rise_origin.create import RecordCreator
    rec = RecordCreator.structure(
        "robot_record",
        structure_name="quadruped",
        voxel_record_range="all",
        rigid_body_record_range="all",
        joint_record_range="none",
        link_record_range="none",
    )
"""

from rise_origin import (
    RS_RecordConfig,
    RS_StructureRecordConfig,
    RSE_RecordType,
    RSE_StructureJointRecordRange,
    RSE_StructureLinkRecordRange,
    RSE_StructureRigidBodyRecordRange,
    RSE_StructureVoxelRecordRange,
)

__all__ = ["RecordCreator"]


_VOXEL_RANGES = {
    "none": RSE_StructureVoxelRecordRange.RSE_VOXEL_RECORD_RANGE_NONE,
    "all": RSE_StructureVoxelRecordRange.RSE_VOXEL_RECORD_RANGE_ALL,
    "random": RSE_StructureVoxelRecordRange.RSE_VOXEL_RECORD_RANGE_RANDOM,
}
_LINK_RANGES = {
    "none": RSE_StructureLinkRecordRange.RSE_LINK_RECORD_RANGE_NONE,
    "all": RSE_StructureLinkRecordRange.RSE_LINK_RECORD_RANGE_ALL,
    "random": RSE_StructureLinkRecordRange.RSE_LINK_RECORD_RANGE_RANDOM,
}
_RIGID_BODY_RANGES = {
    "none": RSE_StructureRigidBodyRecordRange.RSE_RIGID_BODY_RECORD_RANGE_NONE,
    "all": RSE_StructureRigidBodyRecordRange.RSE_RIGID_BODY_RECORD_RANGE_ALL,
}
_JOINT_RANGES = {
    "none": RSE_StructureJointRecordRange.RSE_JOINT_RECORD_RANGE_NONE,
    "all": RSE_StructureJointRecordRange.RSE_JOINT_RECORD_RANGE_ALL,
}


def _resolve(name: str, value: str, table: dict) -> int:
    """Resolve an enum-literal kwarg string to its underlying int value."""
    if value not in table:
        raise ValueError(
            f"{name} must be one of {sorted(table)}, got {value!r}"
        )
    return int(table[value])


class RecordCreator:
    """Factory of :class:`RS_RecordConfig` presets."""

    @staticmethod
    def structure(
        name: str,
        *,
        structure_name: str,
        voxel_record_range: str = "all",
        rigid_body_record_range: str = "all",
        joint_record_range: str = "none",
        link_record_range: str = "none",
        voxel_sample_num: int = 0,
        link_sample_num: int = 0,
    ) -> RS_RecordConfig:
        """Build a structure record.

        Each ``*_record_range`` kwarg accepts a string literal that maps to the
        underlying ``RSE_Structure*RecordRange`` enum. Voxel and link ranges
        additionally accept ``"random"``, which samples ``voxel_sample_num`` /
        ``link_sample_num`` elements.

        Args:
            name: Unique record name.
            structure_name: Name of the structure to record.
            voxel_record_range: ``"none"`` / ``"all"`` / ``"random"``. Maps to
                ``RS_StructureRecordConfig.voxel_record_range``.
            rigid_body_record_range: ``"none"`` / ``"all"``. Maps to
                ``RS_StructureRecordConfig.rigid_body_record_range``.
            joint_record_range: ``"none"`` / ``"all"``. Maps to
                ``RS_StructureRecordConfig.joint_record_range``.
            link_record_range: ``"none"`` / ``"all"`` / ``"random"``. Maps to
                ``RS_StructureRecordConfig.link_record_range``.
            voxel_sample_num: Sample count used only when
                ``voxel_record_range="random"``.
            link_sample_num: Sample count used only when
                ``link_record_range="random"``.

        Returns:
            A populated :class:`RS_RecordConfig`.
        """
        record = RS_RecordConfig()
        record.name = name
        record.type = RSE_RecordType.RSE_RECORD_STRUCTURE

        cfg = RS_StructureRecordConfig()
        cfg.structure_name = structure_name
        cfg.voxel_record_range = _resolve(
            "voxel_record_range", voxel_record_range, _VOXEL_RANGES
        )
        cfg.rigid_body_record_range = _resolve(
            "rigid_body_record_range", rigid_body_record_range, _RIGID_BODY_RANGES
        )
        cfg.joint_record_range = _resolve(
            "joint_record_range", joint_record_range, _JOINT_RANGES
        )
        cfg.link_record_range = _resolve(
            "link_record_range", link_record_range, _LINK_RANGES
        )
        cfg.voxel_sample_num = voxel_sample_num
        cfg.link_sample_num = link_sample_num
        record.config = cfg
        return record
