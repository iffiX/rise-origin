# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Shared assembly of the cloud kinds: blob character in, draft out.

A single cloud is one blob field swept into one structure; a field is
several such clouds scattered on a patch, each its own structure so the
scene can move or drop one cloud without touching the rest. The blob
build, the opaque white or rainy shell materials, the sweep, and the
draft stay identical across kinds, so a kind file holds only its blob
character. Clouds are built at their true size and the scene places them.
"""

import math
from typing import List, Tuple

import numpy as np

from rise_origin import RVec3rf, create as cr
from rise_origin.create.algorithm import (
    BlobCloud,
    BlobCloudParams,
    CloudBody,
    CloudSnapshot,
    CloudVoxelizer,
    GridSpan,
)
from rise_origin.rsc import entity

from ...material.nature import cloud

# The density at or above which a cell becomes a cloud voxel; the blob
# field crosses 0.5 at the cloud surface.
ISO = 0.5


class CloudBuild:
    """Builds one cloud kind's draft from its blob character."""

    @staticmethod
    def make_draft(
        editor_access,
        settings,
        *,
        kind_name: str,
        params: BlobCloudParams,
        voxel_size: float,
        is_rainy: bool = False,
    ) -> "cr.Draft":
        """Build one blob cloud as one structure and assemble its draft.

        Args:
            editor_access: The build wall handed to the module body.
            settings: The kind's validated settings.
            kind_name: The draft and structure name.
            params: The blob character shaping the cloud.
            voxel_size: The cell edge (unit: m).
            is_rainy: Whether the cloud takes the translucent rainy
                material instead of opaque white.
        """
        rng = cr.Rng(settings.seed)
        bodies, low, high = CloudBuild._build_cloud(rng, params, voxel_size, is_rainy)
        structure = CloudBuild._make_structure(
            kind_name, bodies, (0.0, 0.0, 0.0), is_rainy
        )
        return CloudBuild._make_draft(
            editor_access, kind_name, [structure], low, high, voxel_size, is_rainy
        )

    @staticmethod
    def make_field_draft(
        editor_access,
        settings,
        *,
        kind_name: str,
        params: BlobCloudParams,
        span: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        cloud_num: int,
        voxel_size: float,
        is_rainy: bool = False,
    ) -> "cr.Draft":
        """Build a scattered field of clouds, one structure each.

        The clouds sit on a jittered grid inside the ``span`` box; each is
        its own structure placed by its origin, so the scene can move or
        drop a single cloud on its own.

        Args:
            editor_access: The build wall handed to the module body.
            settings: The kind's validated settings.
            kind_name: The draft name; each structure adds its index.
            params: The blob character shaping each cloud.
            span: The min and max corners the clouds scatter between
                (unit: m).
            cloud_num: How many clouds the field scatters.
            voxel_size: The cell edge (unit: m).
            is_rainy: Whether the clouds take the translucent rainy
                material instead of opaque white.
        """
        logger = editor_access.logger
        rng = cr.Rng(settings.seed)
        places = CloudBuild._scatter_places(
            rng.split("places"), span[0], span[1], cloud_num
        )
        tasks = [
            (rng.split(f"cloud/{index}"), params, voxel_size, is_rainy)
            for index in range(len(places))
        ]
        pool = editor_access.pool
        if pool is None:
            results = [CloudBuild._build_cloud(*task) for task in tasks]
        else:
            results = pool.starmap(CloudBuild._build_cloud, tasks)
        structures = []
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        for index, (place, result) in enumerate(zip(places, results)):
            bodies, cloud_low, cloud_high = result
            structures.append(
                CloudBuild._make_structure(
                    f"{kind_name}_{index}", bodies, place, is_rainy
                )
            )
            low = np.minimum(low, np.asarray(place) + cloud_low)
            high = np.maximum(high, np.asarray(place) + cloud_high)
        if logger is not None:
            logger.info("field: %d clouds built", len(places))
        return CloudBuild._make_draft(
            editor_access, kind_name, structures, low, high, voxel_size, is_rainy
        )

    @staticmethod
    def _build_cloud(
        rng: "cr.Rng",
        params: BlobCloudParams,
        voxel_size: float,
        is_rainy: bool,
    ) -> Tuple[List[CloudBody], np.ndarray, np.ndarray]:
        """Return one cloud's bodies and its local extent in meters.

        This is the pool task the field build fans out one per cloud, so
        it draws only from its own stream and never touches the session,
        letting whole clouds build in parallel.

        Args:
            rng: The cloud's random stream.
            params: The blob character.
            voxel_size: The cell edge (unit: m).
            is_rainy: Whether the cloud takes the rainy material, whose
                shell count the voxelizer peels the body into.
        """
        eval_step = CloudBuild._make_eval_step(params, voxel_size)
        density, eval_grid = BlobCloud.make_density(rng, params, eval_step)
        snapshot = CloudSnapshot(density=density, grid=eval_grid)
        world_grid = CloudBuild._make_world_grid(eval_grid, voxel_size)
        layer_count = (
            cloud.CLOUD_RAIN_LAYER_COUNT if is_rainy else cloud.CLOUD_LAYER_COUNT
        )
        specs = CloudVoxelizer().make_bodies(
            snapshot,
            world_grid,
            iso=ISO,
            layer_count=layer_count,
        )
        low, high = world_grid.get_extent()
        return specs, np.asarray(low), np.asarray(high)

    @staticmethod
    def _make_structure(
        name: str,
        bodies: List[CloudBody],
        origin: Tuple[float, float, float],
        is_rainy: bool,
    ) -> "entity.StructureVoxel":
        """Return one cloud's structure placed at its field origin.

        Args:
            name: The structure name.
            bodies: The cloud's swept bodies.
            origin: The structure origin (unit: m).
            is_rainy: Whether the cloud takes the rainy shell materials.
        """
        if is_rainy:
            materials = list(cloud.CLOUD_RAIN_LAYER_REFERENCES)
            optical = [list(one) for one in cloud.CLOUD_RAIN_LAYER_OPTICAL_REFERENCES]
        else:
            materials = [cloud.CLOUD]
            optical = [list(cloud.CLOUD_OPTICAL_REFERENCES)]
        return entity.StructureVoxel(
            name,
            origin_position=RVec3rf(tuple(float(v) for v in origin)),
            material_references=materials,
            material_optical_references=optical,
            bodies=CloudVoxelizer.iterate_structure_bodies(bodies),
            is_fixed=True,
        )

    @staticmethod
    def _make_draft(
        editor_access,
        kind_name: str,
        structures: List["entity.StructureVoxel"],
        low: np.ndarray,
        high: np.ndarray,
        voxel_size: float,
        is_rainy: bool,
    ) -> "cr.Draft":
        """Return the draft wrapping the cloud structures and bounds.

        Args:
            editor_access: The build wall handed to the module body.
            kind_name: The draft name.
            structures: The cloud structures, one per cloud.
            low: The field min corner (unit: m).
            high: The field max corner (unit: m).
            voxel_size: The cell edge (unit: m).
            is_rainy: Whether the structures take the rainy shell
                materials, so the requires name each shell instead of the
                opaque white one.
        """
        box = cr.Box(
            (float(low[0]), float(low[1]), float(low[2])),
            (float(high[0]), float(high[1]), float(high[2])),
            "static",
        )
        material_names = cloud.CLOUD_RAIN_LAYER_NAMES if is_rainy else ("cloud",)
        requires = [
            cr.MaterialRequire(f"{cloud.MODULE}@^1", name, voxel_size)
            for name in material_names
        ]
        return cr.Draft(
            editor_access,
            kind_name,
            boxes=[box],
            rscs=structures,
            requires=requires,
            is_fixed=True,
        )

    @staticmethod
    def _scatter_places(
        rng: "cr.Rng",
        low: Tuple[float, float, float],
        high: Tuple[float, float, float],
        cloud_num: int,
    ) -> List[Tuple[float, float, float]]:
        """Return jittered positions for the field's clouds in the box.

        The clouds sit on a jittered grid across the box floor and take a
        uniform height inside the box, so the field spreads evenly in x
        and y and varies in altitude.

        Args:
            rng: The placement random stream.
            low: The box min corner (unit: m).
            high: The box max corner (unit: m).
            cloud_num: How many clouds to place.
        """
        gen = rng.get_numpy_generator()
        side = max(int(math.ceil(math.sqrt(cloud_num))), 1)
        cell_x = (high[0] - low[0]) / side
        cell_y = (high[1] - low[1]) / side
        places = []
        for index in range(cloud_num):
            row, col = divmod(index, side)
            cx = low[0] + (col + 0.5) * cell_x + gen.uniform(-0.35, 0.35) * cell_x
            cy = low[1] + (row + 0.5) * cell_y + gen.uniform(-0.35, 0.35) * cell_y
            cz = gen.uniform(low[2], high[2])
            places.append((float(cx), float(cy), float(cz)))
        return places

    @staticmethod
    def _make_eval_step(params: BlobCloudParams, voxel_size: float) -> float:
        """Return the field cell size that resolves the smallest blobs.

        Args:
            params: The blob character.
            voxel_size: The output voxel size (unit: m).
        """
        nominal = 0.5 * (params.root_radius_min + params.root_radius_max)
        smallest = nominal * params.radius_ratio ** (params.level_num - 1)
        return max(smallest / 2.5, voxel_size)

    @staticmethod
    def _make_world_grid(eval_grid: GridSpan, voxel_size: float) -> GridSpan:
        """Return the output voxel grid over the field extent.

        Args:
            eval_grid: The field evaluation grid.
            voxel_size: The output voxel size (unit: m).
        """
        low, high = eval_grid.get_extent()
        counts = tuple(
            max(int(math.ceil((high[side] - low[side]) / voxel_size)), 1)
            for side in range(3)
        )
        return GridSpan(
            origin=(float(low[0]), float(low[1]), float(low[2])),
            cell_size=voxel_size,
            cell_num=counts,
        )
