# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Shared assembly of the tree kinds: shape choices in, draft out.

Each tree kind maps its dials onto ``WeberPennParams``, ``TreeBark``,
and ``TreeLeaf`` and hands them here; growth, sweeping, boxes,
requires, and the draft stay identical across kinds, so a kind file
holds only its character.
"""

import math
from typing import Optional, Sequence

import numpy as np

from rise_origin import create as cr
from rise_origin.create.algorithm import (
    TreeBark,
    TreeBodyBudget,
    TreeLeaf,
    TreeVoxelizer,
    WeberPennGraph,
    WeberPennParams,
    WeberPennTree,
)
from rise_origin.rsc import entity

from ...material.nature import plant


class TreeBuild:
    """Builds one tree kind's draft from its shape choices."""

    @staticmethod
    def make_draft(
        editor_access,
        settings,
        *,
        kind_name: str,
        params: WeberPennParams,
        bark: TreeBark,
        leaf: Optional[TreeLeaf],
        leaf_optical_mids: Sequence[int],
        max_leaf_area: float,
        trunk_voxel_size: float,
        canopy_voxel_size: float,
        is_leaf_declared: bool = True,
        shed_fraction: float = 0.0,
        tile_extent: float = 2.2,
    ) -> "cr.Draft":
        """Grow, sweep, and assemble one tree kind's draft.

        Reads the shared dials off the settings: ``seed``, ``lean``,
        and ``is_leaf_soft`` when the kind declares a leaf material.

        Args:
            editor_access: The build wall handed to the module body.
            settings: The kind's validated settings.
            kind_name: The draft and entity name, the kind's last name
                part.
            params: The grown species character.
            bark: The bark detail and wood optical mid mapping.
            leaf: The leaf silhouette; None sweeps a bare tree.
            leaf_optical_mids: The leaf optical mids the season picked;
                empty
                sweeps a bare tree.
            max_leaf_area: The kind's leaf area budget in square
                meters; leaves thin evenly past it at any resolution.
            trunk_voxel_size: Wood cell edge in meters.
            canopy_voxel_size: Leaf cell edge in meters.
            is_leaf_declared: Whether the kind requires a leaf material
                at all; a kind that never grows leaves declares none.
            shed_fraction: Share of the naturally loose leaf parts kept
                unattached.
            tile_extent: Edge length in meters of the world tiles; a
                wide dense crown can coarsen it to stay inside the
                body cap.
        """
        logger = editor_access.logger
        rng = cr.Rng(settings.seed)
        graph = WeberPennTree.grow(rng, params)
        logger.info(
            "grown: %d stems, %d leaves", len(graph.stems), graph.leaf_attaches.shape[0]
        )
        if settings.lean > 0.0:
            TreeBuild._lean_graph(graph, settings.lean, rng.split("lean"))
        specs = TreeVoxelizer(pool=editor_access.pool, logger=logger).make_bodies(
            rng,
            graph,
            trunk_voxel_size=trunk_voxel_size,
            canopy_voxel_size=canopy_voxel_size,
            bark=bark,
            leaf=leaf,
            leaf_optical_mids=leaf_optical_mids,
            shed_fraction=shed_fraction,
            budget=TreeBodyBudget(max_leaf_area=max_leaf_area, tile_extent=tile_extent),
        )
        has_leaf_bodies = any((spec.material_grid == 1).any() for spec in specs)
        bodies = TreeVoxelizer.iterate_structure_bodies(specs)
        trunk_lo, trunk_hi = graph.get_bounds(margin=2.0 * trunk_voxel_size)
        boxes = [cr.Box(tuple(trunk_lo), tuple(trunk_hi), "static")]
        if has_leaf_bodies and leaf is not None:
            boxes.append(TreeBuild._make_leaf_box(graph, leaf, canopy_voxel_size))
        is_canopy_soft = is_leaf_declared and settings.is_leaf_soft and has_leaf_bodies
        leaf_material = plant.LEAF_SOFT if is_canopy_soft else plant.LEAF
        leaf_optics = (
            plant.LEAF_SOFT_OPTICAL_REFERENCES
            if is_canopy_soft
            else plant.LEAF_OPTICAL_REFERENCES
        )
        material_references = [plant.WOOD]
        material_optical_references = [list(plant.WOOD_OPTICAL_REFERENCES)]
        requires = [cr.MaterialRequire(f"{plant.MODULE}@^1", "wood", trunk_voxel_size)]
        if is_leaf_declared:
            material_references.append(leaf_material)
            material_optical_references.append(list(leaf_optics))
            leaf_name = "leaf_soft" if is_canopy_soft else "leaf"
            requires.append(
                cr.MaterialRequire(f"{plant.MODULE}@^1", leaf_name, canopy_voxel_size)
            )
        tree = entity.StructureVoxel(
            kind_name,
            material_references=material_references,
            material_optical_references=material_optical_references,
            bodies=bodies,
            is_fixed=not is_canopy_soft,
        )
        return cr.Draft(
            editor_access,
            kind_name,
            boxes=boxes,
            rscs=[tree],
            requires=requires,
            is_fixed=not is_canopy_soft,
        )

    @staticmethod
    def _lean_graph(graph: WeberPennGraph, lean: float, stream: cr.Rng) -> None:
        """Tilt the whole grown tree by the lean angle."""
        heading = stream.uniform(0.0, 2.0 * math.pi)
        axis = np.array([math.cos(heading), math.sin(heading), 0.0])
        graph.rotate(axis, lean)

    @staticmethod
    def _make_leaf_box(
        graph: WeberPennGraph, leaf: TreeLeaf, voxel_size: float
    ) -> "cr.Box":
        """Return the cover box the leaves can reach."""
        pad = leaf.length + voxel_size
        lo = graph.leaf_attaches.min(axis=0) - pad
        hi = graph.leaf_attaches.max(axis=0) + pad
        return cr.Box(tuple(lo), tuple(hi), "mixed")
