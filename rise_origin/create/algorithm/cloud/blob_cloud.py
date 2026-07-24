# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Cumulus cloud shape: Bouthors and Neyret 2004 hierarchical blobs.

The paper's model reproduced as an implicit field. A cloud is a
hierarchy of levels; each level is a set of blobs sitting on the surface
of the level below. A blob is a flattened sphere field ``g`` altered by
three field terms: ``m`` opens furrows between neighbours, ``n`` enlarges
a blob near its base so it does not pop out, and ``o`` flattens the cloud
bottom at a floor height. Each level's surface is the isosurface of the
running field, and the next level's blobs are scattered evenly on it. The
final field, thresholded, is the cloud.

The paper's field functions and its section 5 parameter values are used
without change. Its two genuinely open points, noted where they occur,
are the flattening term's normalization (the printed form mixes a
unitless one with a distance, so the distance is taken relative to the
blob radius, matching every other term in the paper) and the even
surface distribution (the paper reaches for the Witkin and Heckbert 1994
repulsion particle system to reach an even spacing of twice the blob
radius; the same even spacing is reached here by a Poisson-disc pass over
candidate surface points, its stated goal).
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ...protocol.logger import Logger
from ...protocol.rng import Rng
from ..grid_span import GridSpan

__all__ = ["Blob", "BlobCloudParams", "BlobCloud"]

# Candidate surface points sampled per parent blob before thinning.
_CANDIDATES_PER_PARENT = 400

# How near a candidate's field must be to the isosurface to count as on
# the surface.
_SURFACE_BAND = 0.35

# The point chunk the grid field evaluates at once, to bound memory.
_POINT_CHUNK = 60_000


@dataclass(frozen=True)
class Blob:
    """One flattened-sphere particle of the hierarchy.

    Attributes:
        center: The blob center (unit: m).
        radius: The blob radius (unit: m), the paper's r.
        flatten: The flattening in [0, 1], the paper's e.
        level: The hierarchy level the blob belongs to, root is zero.
    """

    center: Tuple[float, float, float]
    radius: float
    flatten: float
    level: int


@dataclass(frozen=True)
class BlobCloudParams:
    """The character of one cumulus cloud, the paper's section 5 values.

    Attributes:
        root_num: How many root blobs, the paper's six.
        root_radius_min: The least root blob radius (unit: m).
        root_radius_max: The most root blob radius (unit: m).
        root_attach_share: The share of the two radii summed a new root
            attaches from its anchor at, below one so the roots overlap.
        root_plane_jitter_scale: The sideways jitter weight in the ground
            plane when growing a new root off its anchor.
        root_rise_scale: The upward rise weight in z when growing a new
            root off its anchor; lower keeps the cluster flatter.
        level_num: How many levels including the root, the paper's three.
        radius_ratio: The radius factor from a level to the next, the
            paper's one third.
        radius_jitter: The per-blob radius spread, the paper's plus or
            minus thirty percent.
        flatten_min: The least blob flattening, the paper's 0.1.
        flatten_max: The most blob flattening, the paper's 1.0.
        furrow: The repulsion offset opening furrows, the paper's epsilon
            of 0.5.
        enlarge: The base-enlargement fraction, the paper's I of 10.
        blend_by_level: The base-enlargement amount per level, the
            paper's b, null for levels zero and one and one for level
            two.
        floor_height: The cloud floor height (unit: m), the paper's h0.
        floor_stiffness: The floor cut stiffness (unit: m), the paper's
            alpha_h.
    """

    root_num: int = 6
    root_radius_min: float = 200.0
    root_radius_max: float = 450.0
    root_attach_share: float = 0.6
    root_plane_jitter_scale: float = 0.5
    root_rise_scale: float = 0.5
    level_num: int = 3
    radius_ratio: float = 1.0 / 3.0
    radius_jitter: float = 0.3
    flatten_min: float = 0.1
    flatten_max: float = 1.0
    furrow: float = 0.5
    enlarge: float = 10.0
    blend_by_level: Tuple[float, ...] = (0.0, 0.0, 1.0)
    floor_height: float = 100.0
    floor_stiffness: float = 100.0


class BlobCloud:
    """Builds one cumulus cloud's density field from blob hierarchy."""

    @staticmethod
    def make_density(
        rng: Rng,
        params: BlobCloudParams,
        eval_step: float,
        logger: Optional[Logger] = None,
    ) -> Tuple[np.ndarray, GridSpan]:
        """Return the cloud density in [0, 1] and the grid it lives on.

        The density is 0.5 at the cloud surface and rises inside, so a
        voxelizer thresholds it at 0.5.

        Args:
            rng: The cloud's random stream.
            params: The cloud character.
            eval_step: The field evaluation cell size (unit: m).
            logger: The session logger, or None for a silent build.
        """
        blobs_by_level = BlobCloud._draw_hierarchy(rng, params)
        if logger is not None:
            logger.info("blobs: %s", " ".join(str(len(b)) for b in blobs_by_level))
        grid = BlobCloud._make_grid(blobs_by_level, eval_step)
        field = BlobCloud._eval_grid(grid, blobs_by_level, params)
        density = np.clip(0.5 + 0.5 * (field - 1.0), 0.0, 1.0).astype(np.float32)
        return density, grid

    @staticmethod
    def _draw_hierarchy(rng: Rng, params: BlobCloudParams) -> List[List[Blob]]:
        """Return the blobs of every level, root first.

        Args:
            rng: The cloud's random stream.
            params: The cloud character.
        """
        root = BlobCloud._draw_root(rng.split("root"), params)
        levels = [root]
        nominal = 0.5 * (params.root_radius_min + params.root_radius_max)
        for level in range(1, params.level_num):
            nominal = nominal * params.radius_ratio
            child = BlobCloud._place_on_surface(
                rng.split(f"level/{level}"), levels, level, nominal, params
            )
            if not child:
                break
            levels.append(child)
        return levels

    @staticmethod
    def _draw_root(rng: Rng, params: BlobCloudParams) -> List[Blob]:
        """Return the root blobs grown into one overlapping cluster.

        The first blob sits at the center on the floor. Each next blob
        picks a random placed blob as its anchor and attaches close
        enough to always overlap it, so the mass stays one connected
        lump. The attach direction sums a repulsion away from every
        placed blob, a sideways jitter in the ground plane, and an
        upright rise, so the cluster spreads and climbs rather than
        piling on one spot.

        Args:
            rng: The root random stream.
            params: The cloud character.
        """
        gen = rng.get_numpy_generator()
        radius = gen.uniform(
            params.root_radius_min, params.root_radius_max, params.root_num
        )
        flatten = gen.uniform(params.flatten_min, params.flatten_max, params.root_num)
        base = params.floor_height + 0.3 * radius
        blobs = [
            Blob(
                (0.0, 0.0, float(base[0])),
                float(radius[0]),
                float(flatten[0]),
                0,
            )
        ]
        for i in range(1, params.root_num):
            anchor = blobs[int(gen.integers(len(blobs)))]
            distance = params.root_attach_share * (anchor.radius + radius[i])
            direction = BlobCloud._make_grow_direction(gen, anchor, blobs, params)
            center = np.asarray(anchor.center) + direction * distance
            center[2] = max(float(center[2]), float(base[i]))
            blobs.append(
                Blob(
                    tuple(float(v) for v in center),
                    float(radius[i]),
                    float(flatten[i]),
                    0,
                )
            )
        return blobs

    @staticmethod
    def _make_grow_direction(
        gen, anchor: Blob, placed: List[Blob], params: BlobCloudParams
    ) -> np.ndarray:
        """Return the unit direction to grow a new blob off its anchor.

        The direction sums a unit repulsion pointing away from every
        placed blob, a sideways jitter in the ground plane weighted by
        the plane jitter scale, and an upright rise in z weighted by the
        rise scale.

        Args:
            gen: The numpy random generator.
            anchor: The placed blob the new blob attaches to.
            placed: Every blob placed so far.
            params: The cloud character.
        """
        anchor_center = np.asarray(anchor.center)
        repel = np.zeros(3)
        for other in placed:
            offset = anchor_center - np.asarray(other.center)
            squared = float(offset @ offset)
            if squared > 1e-6:
                repel += offset / squared
        repel = BlobCloud._normalize(repel)
        jitter = params.root_plane_jitter_scale * BlobCloud._normalize(
            np.array([gen.normal(), gen.normal(), 0.0])
        )
        upright = np.array([0.0, 0.0, params.root_rise_scale * gen.uniform(0.0, 1.0)])
        return BlobCloud._normalize(repel + jitter + upright)

    @staticmethod
    def _place_on_surface(
        rng: Rng,
        levels: List[List[Blob]],
        level: int,
        nominal_radius: float,
        params: BlobCloudParams,
    ) -> List[Blob]:
        """Return the next level's blobs scattered on the running surface.

        Candidate points are sampled in a shell around each parent blob,
        kept where the running field is near its isosurface, and thinned
        to an even Poisson-disc spacing of twice the blob radius, the
        paper's target spacing.

        Args:
            rng: The level's random stream.
            levels: The blobs of every lower level, root first.
            level: The level being placed.
            nominal_radius: The level's nominal blob radius (unit: m).
            params: The cloud character.
        """
        gen = rng.get_numpy_generator()
        parents = levels[level - 1]
        candidates = BlobCloud._make_candidates(gen, parents)
        field = BlobCloud._eval_points(candidates, levels, params)
        on_surface = candidates[np.abs(field - 1.0) < _SURFACE_BAND]
        if on_surface.shape[0] == 0:
            return []
        kept = BlobCloud._make_thin_poisson(gen, on_surface, 2.0 * nominal_radius)
        radius = nominal_radius * gen.uniform(
            1.0 - params.radius_jitter, 1.0 + params.radius_jitter, kept.shape[0]
        )
        flatten = gen.uniform(params.flatten_min, params.flatten_max, kept.shape[0])
        return [
            Blob(
                tuple(float(v) for v in kept[i]),
                float(radius[i]),
                float(flatten[i]),
                level,
            )
            for i in range(kept.shape[0])
        ]

    @staticmethod
    def _make_candidates(gen, parents: List[Blob]) -> np.ndarray:
        """Return shell points sampled around every parent blob.

        Args:
            gen: The numpy random generator.
            parents: The parent level's blobs.
        """
        points = []
        for blob in parents:
            direction = gen.normal(size=(_CANDIDATES_PER_PARENT, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
            reach = blob.radius * gen.uniform(0.9, 1.15, (_CANDIDATES_PER_PARENT, 1))
            points.append(np.asarray(blob.center) + direction * reach)
        return np.concatenate(points, axis=0)

    @staticmethod
    def _make_thin_poisson(gen, points: np.ndarray, min_distance: float) -> np.ndarray:
        """Return an evenly spaced subset no two points closer than a bound.

        A greedy Poisson-disc pass over the points in a shuffled order,
        the even spacing the paper's repulsion particle system reaches.

        Args:
            gen: The numpy random generator.
            points: The candidate points.
            min_distance: The least allowed spacing (unit: m).
        """
        order = gen.permutation(points.shape[0])
        squared = min_distance * min_distance
        kept_list = []
        for index in order:
            point = points[index]
            if kept_list:
                near = np.asarray(kept_list)
                if np.min(np.sum((near - point) ** 2, axis=1)) < squared:
                    continue
            kept_list.append(point)
        return np.asarray(kept_list)

    @staticmethod
    def _make_grid(blobs_by_level: List[List[Blob]], eval_step: float) -> GridSpan:
        """Return the evaluation grid wrapping every blob with a margin.

        Args:
            blobs_by_level: The blobs of every level.
            eval_step: The evaluation cell size (unit: m).
        """
        centers = np.asarray(
            [blob.center for level in blobs_by_level for blob in level]
        )
        radii = np.asarray([blob.radius for level in blobs_by_level for blob in level])
        low = (centers - radii[:, None] * 1.4).min(axis=0)
        high = (centers + radii[:, None] * 1.4).max(axis=0)
        low[2] = min(low[2], 0.0)
        counts = np.maximum(np.ceil((high - low) / eval_step).astype(int), 1)
        return GridSpan(
            origin=(float(low[0]), float(low[1]), float(low[2])),
            cell_size=float(eval_step),
            cell_num=(int(counts[0]), int(counts[1]), int(counts[2])),
        )

    @staticmethod
    def _eval_grid(
        grid: GridSpan, blobs_by_level: List[List[Blob]], params: BlobCloudParams
    ) -> np.ndarray:
        """Return the composed field at every grid cell center.

        Args:
            grid: The evaluation grid.
            blobs_by_level: The blobs of every level.
            params: The cloud character.
        """
        xs, ys, zs = grid.make_axes(None)
        gx, gy, gz = np.meshgrid(
            xs.astype(np.float32),
            ys.astype(np.float32),
            zs.astype(np.float32),
            indexing="ij",
        )
        points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        field = np.empty(points.shape[0], dtype=np.float32)
        for start in range(0, points.shape[0], _POINT_CHUNK):
            stop = start + _POINT_CHUNK
            field[start:stop] = BlobCloud._eval_points(
                points[start:stop], blobs_by_level, params
            )
        return field.reshape(grid.cell_num)

    @staticmethod
    def _eval_points(
        points: np.ndarray, blobs_by_level: List[List[Blob]], params: BlobCloudParams
    ) -> np.ndarray:
        """Return the composed field at loose points, the max over levels.

        Args:
            points: The (n, 3) query points (unit: m).
            blobs_by_level: The blobs of every level.
            params: The cloud character.
        """
        field = np.full(points.shape[0], -np.inf, dtype=np.float32)
        for level, blobs in enumerate(blobs_by_level):
            parents = blobs_by_level[level - 1] if level > 0 else None
            level_field = BlobCloud._eval_level(points, blobs, parents, level, params)
            field = np.maximum(field, level_field)
        return field

    @staticmethod
    def _eval_level(
        points: np.ndarray,
        blobs: Sequence[Blob],
        parents: Optional[Sequence[Blob]],
        level: int,
        params: BlobCloudParams,
    ) -> np.ndarray:
        """Return one level's field, the max of its blob fields.

        Args:
            points: The (n, 3) query points (unit: m).
            blobs: The level's blobs.
            parents: The parent level's blobs, or None for the root.
            level: The level index.
            params: The cloud character.
        """
        if not blobs:
            return np.full(points.shape[0], -np.inf, dtype=np.float32)
        centers = np.asarray([b.center for b in blobs], dtype=np.float32)
        radii = np.asarray([b.radius for b in blobs], dtype=np.float32)
        flatten = np.asarray([b.flatten for b in blobs], dtype=np.float32)
        distance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        base_distance = BlobCloud._get_distance_to_base(points, parents, params)
        # g: the flattened pure sphere field. The flattening shrinks the
        # effective radius away from the base surface; the base distance
        # is taken relative to the blob radius so the printed unitless
        # one and the distance share a scale.
        squash = 1.0 - flatten[None, :] * np.minimum(
            1.0, base_distance[:, None] / radii[None, :]
        )
        effective = np.maximum(radii[None, :] * squash, radii[None, :] * 0.05)
        g = np.exp(1.0 - distance / effective).astype(np.float32)
        # o: the flat cloud bottom cut below the floor height.
        height = points[:, 2]
        floor_cut = np.minimum(
            0.0, (height - params.floor_height) / params.floor_stiffness
        )
        o = g * floor_cut[:, None]
        # n: the base enlargement so a blob does not pop off its parent.
        blend = (
            params.blend_by_level[level] if level < len(params.blend_by_level) else 0.0
        )
        n = (
            blend
            * np.minimum(
                1.0, np.exp(-params.enlarge * base_distance[:, None] / radii[None, :])
            )
            * np.exp(1.0 - distance / radii[None, :])
        ).astype(np.float32)
        # m: the furrow repulsion. Each blob's field drops near every
        # other blob; the sum over all blobs is shared and the self term
        # removed per blob.
        mu = (1.0 - params.furrow - g) * np.minimum(1.0, g * g)
        mu_sum = mu.sum(axis=1, keepdims=True)
        m = mu_sum - mu
        f = g + m + n + o
        return f.max(axis=1).astype(np.float32)

    @staticmethod
    def _get_distance_to_base(
        points: np.ndarray,
        parents: Optional[Sequence[Blob]],
        params: BlobCloudParams,
    ) -> np.ndarray:
        """Return the distance from each point to the base surface.

        The base surface of the root is the floor plane; of a higher
        level it is the union of the parent blob spheres, whose distance
        is the nearest parent blob's surface distance.

        Args:
            points: The (n, 3) query points (unit: m).
            parents: The parent level's blobs, or None for the root.
            params: The cloud character.
        """
        if not parents:
            return np.abs(points[:, 2] - params.floor_height).astype(np.float32)
        centers = np.asarray([b.center for b in parents], dtype=np.float32)
        radii = np.asarray([b.radius for b in parents], dtype=np.float32)
        distance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        return np.abs(distance - radii[None, :]).min(axis=1).astype(np.float32)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """Return the unit vector, or zeros if its length is negligible.

        Args:
            vector: The vector to normalize.
        """
        length = float(np.linalg.norm(vector))
        if length < 1e-9:
            return np.zeros_like(vector)
        return vector / length
