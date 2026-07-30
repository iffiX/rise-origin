# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Shared assembly of the rock kinds: sediment pieces in, draft out.

Each rock kind maps its dials onto strata layers, a fracture, and
erosion classes and hands them here; the grid sizing, device choice,
generation schedule, emission, and marshal stay identical across
kinds, so a kind file holds only its character.
"""

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from rise_origin import RQuat3rf, RS_NULL_INDEX, RVec3rf
from rise_origin import create as cr
from rise_origin.create.algorithm import GridSpan
from rise_origin.create.algorithm.sediment import (
    Volume,
    BallErosion,
    ChunkFracture,
    DetailErosion,
    FractureFragment,
    PlaneFracture,
    SeamFracture,
)
from rise_origin.create.utils.torch_device import TorchDevice
from rise_origin.rsc import entity

from ...material.nature import mineral

# Contact stability floor: a stone below this many voxels cannot rest.
MIN_STONE_VOXEL_NUM = 4

# The internal grid must resolve a stone this many cells across so the
# ball measurement reads real shape.
MIN_SIZE_CELL_NUM = 8

# Air margin around the box so wear and emission never touch the rim.
MARGIN_VOXEL_NUM = 2

# Working-set forecast per internal cell across the channel tensors.
BYTES_PER_CELL = 48

# Optical reference lists per mineral material name.
OPTICAL_REFERENCES_BY_NAME = {
    "sandstone_soft": mineral.SANDSTONE_SOFT_OPTICAL_REFERENCES,
    "sandstone_hard": mineral.SANDSTONE_HARD_OPTICAL_REFERENCES,
    "granite": mineral.GRANITE_OPTICAL_REFERENCES,
    "limestone": mineral.LIMESTONE_OPTICAL_REFERENCES,
    "marble": mineral.MARBLE_OPTICAL_REFERENCES,
    "slate": mineral.SLATE_OPTICAL_REFERENCES,
    "basalt": mineral.BASALT_OPTICAL_REFERENCES,
}


class RockBuild:
    """Builds one rock kind's draft from its sediment pieces."""

    @staticmethod
    def make_boulder_grid(
        settings, kind_name: str, size: float, flatness: float, elongation: float
    ):
        """Return the internal grid, device, and box size of one stone.

        Checks the resolution before anything builds so a too-coarse
        stone fails at draft time, then centers a grid holding the
        Zingg-shaped box plus its air margin and picks the device.

        Args:
            settings: The kind's validated settings.
            kind_name: The kind's name, used in the resolution error.
            size: The stone's target diameter (unit: m).
            flatness: How much the stone flattens along z, 0 to 1.
            elongation: How much the stone stretches along x, 0 to 1.

        Returns:
            The internal ``GridSpan``, the torch device, and the
            ``(x, y, z)`` box edge lengths in meters.
        """
        cell_size = settings.voxel_size / settings.wear_subdivision
        if size / cell_size < MIN_SIZE_CELL_NUM:
            raise ValueError(
                f"{kind_name}: size {size} m spans under {MIN_SIZE_CELL_NUM} "
                f"internal cells at voxel_size {settings.voxel_size} m with "
                f"wear_subdivision {settings.wear_subdivision}; raise size, "
                "lower voxel_size, or raise wear_subdivision"
            )
        box_size = (
            size * (1.0 + 0.6 * elongation),
            size,
            size * (1.0 - 0.6 * flatness),
        )
        grid = RockBuild._make_margin_grid(
            box_size, cell_size, settings.wear_subdivision, settings.voxel_size
        )
        device = TorchDevice.pick(grid.get_memory_bytes(BYTES_PER_CELL))
        return grid, device, box_size

    @staticmethod
    def make_box_grid(
        settings,
        kind_name: str,
        box_size: Tuple[float, float, float],
        resolve_size: float,
    ):
        """Return the internal grid and device of an explicit box.

        Checks that the resolved feature spans enough cells, then
        centers a grid holding the box plus its air margin and picks the
        device.

        Args:
            settings: The kind's validated settings.
            kind_name: The kind's name, used in the resolution error.
            box_size: The (x, y, z) box edge lengths (unit: m).
            resolve_size: The feature that must resolve, the smallest
                thing the kind draws (unit: m).

        Returns:
            The internal ``GridSpan`` and the torch device.
        """
        cell_size = settings.voxel_size / settings.wear_subdivision
        if resolve_size / cell_size < MIN_SIZE_CELL_NUM:
            raise ValueError(
                f"{kind_name}: a feature of {resolve_size} m spans under "
                f"{MIN_SIZE_CELL_NUM} internal cells at voxel_size "
                f"{settings.voxel_size} m with wear_subdivision "
                f"{settings.wear_subdivision}; raise it, lower voxel_size, "
                "or raise wear_subdivision"
            )
        grid = RockBuild._make_margin_grid(
            box_size, cell_size, settings.wear_subdivision, settings.voxel_size
        )
        device = TorchDevice.pick(grid.get_memory_bytes(BYTES_PER_CELL))
        return grid, device

    @staticmethod
    def make_boulder_draft(
        editor_access,
        settings,
        *,
        kind_name: str,
        material_names: Tuple[str, ...],
        volume,
        rng,
        size: float,
        flatness: float,
        elongation: float,
        erosion: BallErosion,
        detail: Optional[DetailErosion] = None,
        fragment: Optional[FractureFragment] = None,
        surface_roughness: float = 0.0,
        roughness_feature_size: float = 0.1,
        chunk_num: int = 0,
        chunk_radius: float = 0.0,
    ) -> "cr.Draft":
        """Fracture, erode, and marshal one generated stone volume.

        The kind builds the volume with the generator it wants and
        hands it here; the fracture, wear, emission, and marshal stay
        identical across kinds.

        Args:
            editor_access: The build wall handed to the module body.
            settings: The kind's validated settings.
            kind_name: The draft and entity name, the kind's last name
                part.
            material_names: The mineral material names the stone uses;
                position n becomes the volume material id n + 1.
            volume: The generated stone volume to shape and emit.
            rng: The random stream the fracture draws from.
            size: The stone's target diameter (unit: m).
            flatness: How much the stone flattens along z, 0 to 1.
            elongation: How much the stone stretches along x, 0 to 1.
            erosion: The configured wear the stone ages under.
            detail: The configured surface carving, None for none.
            fragment: The piece to cut the stone to; None cuts the
                default random polytope of the size and shape dials.
            surface_roughness: How far the fracture faces wander in and
                out (unit: m); 0 keeps the cut flat.
            roughness_feature_size: The nodule scale of that wander
                (unit: m).
            chunk_num: How many chunks to knock off after the cut; 0
                keeps the convex block.
            chunk_radius: The reach of one knocked-off chunk (unit: m).
        """
        if fragment is None:
            fragment = PlaneFracture.make_fragment(
                rng.split("fracture"), size, flatness, elongation
            )
        PlaneFracture(
            (fragment,),
            np.zeros((1, 3)),
            rng=rng.split("roughness"),
            surface_roughness=surface_roughness,
            roughness_feature_size=roughness_feature_size,
        ).fracture(volume)
        if chunk_num > 0:
            ChunkFracture(rng.split("chunk"), chunk_num, chunk_radius).fracture(volume)
        for eroser in RockBuild._make_erosion_row(erosion):
            eroser.erode(volume)
        if detail is not None:
            detail.erode(volume)
        coarse = volume.make_coarser(settings.wear_subdivision)
        return RockBuild._make_draft(
            editor_access,
            coarse,
            kind_name=kind_name,
            material_names=material_names,
            voxel_size=settings.voxel_size,
        )

    @staticmethod
    def make_volume_draft(
        editor_access,
        settings,
        *,
        kind_name: str,
        material_names: Tuple[str, ...],
        volume,
        erosion: Optional[BallErosion] = None,
        detail: Optional[DetailErosion] = None,
        is_fixed: bool = True,
        is_loose_dropped: bool = False,
    ) -> "cr.Draft":
        """Erode and marshal a pre-shaped formation volume, no fracture.

        A formation carries its own shape and marks from its generator,
        so no fracture runs; the wear, emission, and marshal are shared,
        and the draft is fixed by default because a formation is scenery.

        Args:
            editor_access: The build wall handed to the module body.
            settings: The kind's validated settings.
            kind_name: The draft and entity name.
            material_names: The mineral material names the formation
                uses; position n becomes the volume material id n + 1.
            volume: The generated formation volume to shape and emit.
            erosion: The configured wear, None for none.
            detail: The configured surface carving, None for none.
            is_fixed: Whether the formation stands as static scenery.
            is_loose_dropped: Whether every solid part but the largest
                falls away at the end; anything a fracture left
                disconnected would have fallen.
        """
        for eroser in RockBuild._make_erosion_row(erosion):
            eroser.erode(volume)
        if detail is not None:
            detail.erode(volume)
        volume.remove_debris()
        coarse = volume.make_coarser(settings.wear_subdivision)
        if is_loose_dropped:
            coarse.remove_loose_parts()
        return RockBuild._make_draft(
            editor_access,
            coarse,
            kind_name=kind_name,
            material_names=material_names,
            voxel_size=settings.voxel_size,
            is_fixed=is_fixed,
        )

    @staticmethod
    def make_block_draft(
        editor_access,
        settings,
        *,
        kind_name: str,
        material_names: Tuple[str, ...],
        volume,
        rng,
        box_size: Tuple[float, float, float],
        seam_spacing: float,
        seam_width: float,
        erosion: Optional[BallErosion] = None,
        detail: Optional[DetailErosion] = None,
    ) -> "cr.Draft":
        """Seam, weather, and marshal one generated block volume.

        The seam cut breaks the filled block into pieces and the wear
        opens their boundaries into cracks, so a block reads as broken
        mass rather than an even box; the marshal is the shared
        formation path with loose parts dropped.

        Args:
            editor_access: The build wall handed to the module body.
            settings: The kind's validated settings.
            kind_name: The draft and entity name.
            material_names: The mineral material names the block uses;
                position n becomes the volume material id n + 1.
            volume: The generated block volume to seam and emit.
            rng: The random stream the seams draw from.
            box_size: The (x, y, z) box edge lengths (unit: m).
            seam_spacing: The nominal seam piece spacing (unit: m).
            seam_width: How wide the seams read (unit: m).
            erosion: The configured wear, None for none.
            detail: The configured surface carving, None for none.
        """
        SeamFracture(rng.split("seams"), box_size, seam_spacing, seam_width).fracture(
            volume
        )
        return RockBuild.make_volume_draft(
            editor_access,
            settings,
            kind_name=kind_name,
            material_names=material_names,
            volume=volume,
            erosion=erosion,
            detail=detail,
            is_fixed=settings.is_fixed,
            is_loose_dropped=True,
        )

    @staticmethod
    def make_wear_rows(material_names: Sequence[str], table: dict) -> Tuple[float, ...]:
        """Return per-material-id rows for a wear table.

        Args:
            material_names: The mineral material names in job order.
            table: A mineral wear table keyed by material name.

        Returns:
            One value per volume material id, with a placeholder air
            row at index 0.
        """
        return (1.0,) + tuple(table[name] for name in material_names)

    @staticmethod
    def _make_erosion_row(erosion) -> tuple:
        """Return the erosions as a row applied in order.

        A kind hands one eroser, a tuple layering several weather
        passes the way the paper composes phenomena, or None.
        """
        if erosion is None:
            return ()
        if isinstance(erosion, tuple):
            return erosion
        return (erosion,)

    @staticmethod
    def _make_margin_grid(
        box_size: Tuple[float, float, float],
        cell_size: float,
        subdivision: int,
        voxel_size: float,
    ) -> GridSpan:
        """Return the centered internal grid holding the box and margin."""
        cell_num = []
        for side in range(3):
            extent = box_size[side] + 2.0 * MARGIN_VOXEL_NUM * voxel_size
            count = int(math.ceil(extent / cell_size / subdivision)) * subdivision
            cell_num.append(count)
        origin = tuple(-count * cell_size / 2.0 for count in cell_num)
        return GridSpan(origin, cell_size, tuple(cell_num))

    @staticmethod
    def _make_draft(
        editor_access,
        coarse,
        *,
        kind_name: str,
        material_names: Tuple[str, ...],
        voxel_size: float,
        is_fixed: bool = False,
    ) -> "cr.Draft":
        """Marshal one emitted volume into a draft, dynamic or fixed."""
        material_class = "static" if is_fixed else "dynamic"
        occupancy = coarse.get_solid_cells().cpu().numpy()
        occupied_num = int(occupancy.sum())
        if occupied_num < MIN_STONE_VOXEL_NUM:
            raise ValueError(
                f"{kind_name}: the stone keeps {occupied_num} voxels, under "
                f"the {MIN_STONE_VOXEL_NUM}-voxel contact floor; raise size "
                "or lower voxel_size"
            )
        material_grid = coarse.material_id.cpu().numpy()
        optical_grid = (
            coarse.optical_mid.cpu().numpy()
            if coarse.optical_mid is not None
            else np.zeros_like(material_grid)
        )
        cells = np.argwhere(occupancy)
        lo = cells.min(axis=0)
        hi = cells.max(axis=0) + 1
        window = tuple(slice(lo[side], hi[side]) for side in range(3))
        occupancy = occupancy[window]
        material_grid = material_grid[window]
        optical_grid = optical_grid[window]
        filled = np.where(
            occupancy,
            material_grid.astype(np.uint32) - 1,
            np.uint32(RS_NULL_INDEX),
        )
        segment_grid = np.where(occupancy, np.uint32(0), np.uint32(RS_NULL_INDEX))
        origin = np.asarray(coarse.grid.origin, dtype=np.float64) + lo * voxel_size
        body = entity.StructureVoxelBody.from_grid(
            filled,
            segment_grid=segment_grid,
            material_optical_reference_mid_grid=optical_grid.astype(np.uint32),
            voxel_size=voxel_size,
            relative_origin_position=RVec3rf(tuple(origin)),
            relative_orientation=RQuat3rf((0.0, 0.0, 0.0, 1.0)),
        )
        material_references = [f"{mineral.MODULE}:{name}" for name in material_names]
        material_optical_references = [
            list(OPTICAL_REFERENCES_BY_NAME[name]) for name in material_names
        ]
        stone = entity.StructureVoxel(
            kind_name,
            material_references=material_references,
            material_optical_references=material_optical_references,
            bodies=[body],
            is_fixed=is_fixed,
        )
        extent = (hi - lo) * voxel_size
        boxes = [cr.Box(tuple(origin), tuple(origin + extent), material_class)]
        requires = [
            cr.MaterialRequire(f"{mineral.MODULE}@^1", name, voxel_size)
            for name in material_names
        ]
        return cr.Draft(
            editor_access,
            kind_name,
            boxes=boxes,
            rscs=[stone],
            requires=requires,
            is_fixed=is_fixed,
        )
