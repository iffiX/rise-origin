# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The shared tree voxelizer: a grown graph into a few voxel bodies.

Every tree kind hands its grown :class:`WeberPennGraph` here with data
choices only, and gets engine-ready bodies back; no kind writes voxel
code. The emission splits by what each part of the tree is:

- Thick limbs (trunk and majors) sweep as one oriented body per stem
  chain, so their cylinders stay smooth at any angle; boxes hug the
  spine, so little empty space is stored. Bark is carved surface
  detail in one of four families, furrows, bands, plates, or flaking
  patches, cut into the swept radius; the wood optical mids follow the
  detail, crevices dark and crests light, with moss fading up from
  the ground.
- Thin twigs, leaf holders, and leaves land on one world-aligned
  lattice, deduplicated globally (a cell is claimed once, first leaf
  wins), and pack into disjoint space tiles trimmed to their content, so
  nothing overlaps and no giant sparse grid is stored. Every twig
  stamps its one voxel center line, face connected at any angle.

Leaves stamp as one-voxel-thick silhouette patches; thick limbs block
leaf cells so foliage never grows into them. Each leaf-carrying stem
drops small fixed holder blobs of wood where its leaves attach,
overlapping the swept limb, so every leaf is welded to a world-anchored
base and a soft canopy hangs instead of falling.

TODO: a dead snag's limbs may later become individual jointed rigid
bodies so pushing the tree snaps branches off; breakable joints with
some springiness at the joints, once the engine grows them, make that
possible.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import cc3d
import numpy as np

from rise_origin import RQuat3rf, RS_NULL_INDEX, RVec3rf
from rise_origin.rsc import entity

from ...protocol.logger import Logger
from ...protocol.rng import Rng
from ...protocol.worker_pool import WorkerPool
from ..field import ValueNoiseField
from .weber_penn import WeberPennGraph

__all__ = [
    "TreeBark",
    "TreeBarkFurrows",
    "TreeBarkBands",
    "TreeBarkPlates",
    "TreeBarkPatches",
    "TreeLeaf",
    "TreeBodyBudget",
    "TreeBody",
    "TreeVoxelizer",
]

# Cell packing offset for world-lattice keys; cell coordinates are int16
# and stay far inside +-2**15, so three 16 bit fields fill one int64 key.
_KEY_OFFSET = 1 << 15
_KEY_SHIFT = 16

# Fixed stems per twig worker shard; content-derived, never worker-count
# derived, so any pool size rasterizes identical shards.
_TWIG_SHARD_STEMS = 512


@dataclass(frozen=True)
class TreeBarkFurrows:
    """Bark detail of ridges running along the limb.

    The classic furrowed bark of oaks and maples: crests and crevices
    follow the limb direction and wander slowly along it.

    Attributes:
        depth: How deep the furrows cut into the radius, in meters;
            a voxel size coarser than the depth skips the carve, so
            coarse trees keep a clean surface.
        width: Distance between furrows around the limb, in meters.
        wobble_length: How far along the limb a furrow wanders before
            bending, in meters.
    """

    depth: float = 0.032
    width: float = 0.05
    wobble_length: float = 0.6


@dataclass(frozen=True)
class TreeBarkBands:
    """Bark detail of ridges running around the limb.

    Full rings read as palm scars or cherry bands; low coverage breaks
    the rings into the short dark dashes of a birch trunk.

    Attributes:
        depth: How deep the bands cut into the radius, in meters; a
            voxel size coarser than the depth skips the carve.
        length: Distance between bands along the limb, in meters.
        wobble: How far a band waves up and down around the limb, as a
            fraction of the band spacing.
        coverage: Fraction of each ring kept; 1 keeps whole rings and
            small values leave scattered dashes.
    """

    depth: float = 0.016
    length: float = 0.2
    wobble: float = 0.3
    coverage: float = 1.0


@dataclass(frozen=True)
class TreeBarkPlates:
    """Bark detail of a staggered plate mosaic over the limb surface.

    Grooves outline rows of plates; half stagger weaves the diamond
    pattern of a palm trunk, and shift slides each column of plates up
    or down so the rows stop lining up, which reads as pine and spruce
    scales.

    Attributes:
        depth: How deep the grooves cut into the radius, in meters; a
            voxel size coarser than the depth skips the carve.
        width: Plate size around the limb, in meters.
        length: Plate size along the limb, in meters.
        stagger: Row offset as a fraction of the plate width, 0 to 0.5.
        jitter: How unevenly the plates stand out, 0 to 1.
        shift: Per-column offset of the rows as a fraction of the plate
            length; 0 keeps level rings, higher values break them into
            scales.
    """

    depth: float = 0.024
    width: float = 0.08
    length: float = 0.12
    stagger: float = 0.5
    jitter: float = 0.6
    shift: float = 0.0


@dataclass(frozen=True)
class TreeBarkPatches:
    """Bark detail of flaking patch layers, like a sycamore trunk.

    A smooth field cut at nested thresholds steps the surface into
    patch levels, each peeled to a different depth with its own
    optical mid.

    Attributes:
        depth: Radius step between patch levels, in meters; a voxel
            size coarser than the step keeps the patch colors on a
            smooth surface.
        cell: Patch feature size in meters.
        optical_mid_levels: Ordered optical mids of the levels, deepest
            first;
            the count sets how many layers show.
    """

    depth: float = 0.016
    cell: float = 0.3
    optical_mid_levels: Sequence[int] = (3, 0, 4)


@dataclass(frozen=True)
class TreeBark:
    """The trunk surface detail and how the wood optical mids follow
    it.

    Attributes:
        detail: The bark detail family: furrows, bands, plates, or
            patches.
        grain_amplitude: Fine per-cell roughness added on top of the
            detail, in meters, so coarse voxels carry little of it.
        moss_coverage: Fraction of low bark carrying moss, 0 for none.
        moss_height: Height in meters the moss fades out by.
        moss_cell: Moss patch feature size in meters.
        peel_coverage: Fraction of bark peeled away to expose the wood
            under it, 0 for none; peeled spots take ``optical_mid_peel``.
        peel_cell: Peel spot feature size in meters.
        optical_mid_base: Wood optical mid of plain bark.
        optical_mid_shade: Wood optical mid speckled into deep grain.
        optical_mid_moss: Wood optical mid of moss patches.
        optical_mid_crevice: Wood optical mid where the detail cuts deepest.
        optical_mid_light: Wood optical mid on the detail's crests.
        optical_mid_peel: Wood optical mid of peeled spots.
    """

    detail: Union[TreeBarkFurrows, TreeBarkBands, TreeBarkPlates, TreeBarkPatches] = (
        TreeBarkFurrows()
    )
    grain_amplitude: float = 0.01
    moss_coverage: float = 0.0
    moss_height: float = 2.5
    moss_cell: float = 0.35
    peel_coverage: float = 0.0
    peel_cell: float = 0.45
    optical_mid_base: int = 0
    optical_mid_shade: int = 1
    optical_mid_moss: int = 2
    optical_mid_crevice: int = 3
    optical_mid_light: int = 4
    optical_mid_peel: int = 0


@dataclass(frozen=True)
class TreeLeaf:
    """The per-kind leaf silhouette, stamped one voxel thick.

    Attributes:
        outline: The silhouette name: ``"ellipse"``, ``"hex"``,
            ``"tee"``, ``"maple_star"``, ``"fan"``, or ``"strip"``.
        length: Leaf length from its attach point, in meters.
        width_ratio: Leaf width as a fraction of the length.
    """

    outline: str = "hex"
    length: float = 0.18
    width_ratio: float = 0.8


@dataclass(frozen=True)
class TreeBodyBudget:
    """How the voxelizer bounds body count and voxel spend.

    Attributes:
        thick_radius_voxels: Stems at least this many voxels in mean
            radius sweep as their own oriented bodies; thinner stems
            join the world tiles.
        max_oriented_bodies: Most oriented stem bodies one tree may
            emit; past it the radius bar rises until the thickest stems
            fit.
        tile_extent: Edge length in meters of the world tiles twigs and
            leaves pack into.
        max_body_cells: Most grid cells one body may span. Body grids
            ship dense, so their memory follows grid volume; a fine
            voxel size halves the tile edge and splits long limb
            chains until every body fits, trading body count for a
            bounded build. Default sizes stay under it untouched, so
            their body counts keep the declared caps. A limb span
            never splits below its own padded thickness, so a very
            small budget may keep one thick piece over it.
        max_leaf_area: Most leaf face area in square meters the whole
            tree may stamp; when the grown foliage would pass it,
            leaves thin evenly across the crown to fit, so foliage
            stays the same at every canopy voxel size. None never
            thins.
        blocker_bleed_voxels: Extra radius in leaf voxels around thick
            limbs that leaf cells keep clear of.
        holder_radius_voxels: Radius of the fixed wood blob dropped at
            each leaf attach point, in leaf voxels.
    """

    thick_radius_voxels: float = 0.75
    max_oriented_bodies: int = 40
    tile_extent: float = 2.2
    max_body_cells: int = 8_000_000
    max_leaf_area: Optional[float] = None
    blocker_bleed_voxels: float = 1.0
    holder_radius_voxels: float = 1.2


@dataclass
class TreeBody:
    """One voxel body of a swept tree.

    Attributes:
        occupancy: (X, Y, Z) bool grid in the body's own frame.
        material_grid: (X, Y, Z) int grid of structure material indices,
            meaningful under ``occupancy``; wood carries 0 and leaves
            carry 1.
        optical_mid_grid: (X, Y, Z) int grid of optical mids, each
            indexing its cell's material optical list.
        origin: The grid's min corner in tree space.
        orientation: Body rotation as an (x, y, z, w) quaternion turning
            the body frame into tree space.
        voxel_size: Cell edge in meters.
    """

    occupancy: np.ndarray
    material_grid: np.ndarray
    optical_mid_grid: np.ndarray
    origin: np.ndarray
    orientation: np.ndarray
    voxel_size: float


class TreeVoxelizer:
    """Sweeps grown tree graphs into few voxel bodies.

    Args:
        pool: The shared worker pool the heavy sweep stages run on;
            None runs every stage in this process. Output is byte
            identical either way: work shards are fixed by content,
            results merge in task order, and no random stream crosses
            a process.
        logger: The logger sweep stages report progress through; None
            sweeps silently.
    """

    def __init__(
        self, pool: Optional[WorkerPool] = None, logger: Optional[Logger] = None
    ) -> None:
        self._pool = pool
        self._logger = logger

    def make_bodies(
        self,
        rng: Rng,
        graph: WeberPennGraph,
        *,
        trunk_voxel_size: float,
        canopy_voxel_size: float,
        bark: TreeBark,
        leaf: Optional[TreeLeaf] = None,
        leaf_optical_mids: Sequence[int] = (),
        shed_fraction: float = 0.0,
        budget: TreeBodyBudget = TreeBodyBudget(),
    ) -> List[TreeBody]:
        """Sweep a grown tree into wood and leaf bodies.

        Args:
            rng: The random stream for bark phases, grain, and the leaf
                optical mid draws.
            graph: The grown tree.
            trunk_voxel_size: Wood cell edge in meters.
            canopy_voxel_size: Leaf cell edge in meters.
            bark: The bark detail and wood optical mid mapping.
            leaf: The leaf silhouette; None grows a bare tree even if
                the graph carries leaves.
            leaf_optical_mids: The leaf optical mids leaves dapple over,
                one whole mid per leaf; empty grows a bare tree.
            shed_fraction: Share of the naturally loose leaf parts kept
                unattached, 0 to 1; the rest weld back or delete. A
                soft tree sheds the kept parts when the world runs; a
                fixed tree holds them in place.
            budget: The body and voxel budget.
        """
        positions = np.vstack(graph.positions)
        radii = np.asarray(graph.radii, dtype=float)
        thick_ids = TreeVoxelizer._pick_thick_stems(
            graph, radii, trunk_voxel_size, budget
        )
        grain_salt = np.uint32(rng.split("grain").integers(0, 2**31))
        moss = (
            ValueNoiseField(rng.split("moss"), bark.moss_cell)
            if bark.moss_coverage > 0.0
            else None
        )
        peel = (
            ValueNoiseField(rng.split("peel"), bark.peel_cell)
            if bark.peel_coverage > 0.0
            else None
        )
        chain_draws = [TreeVoxelizer._draw_chain_phases(rng, sid) for sid in thick_ids]
        progress = (
            self._logger.make_progress("sweeping limbs", len(thick_ids))
            if self._logger is not None
            else None
        )
        bodies: List[TreeBody] = []
        if self._pool is not None and len(thick_ids) > 1:
            spans_per_chain = self._pool.starmap(
                TreeVoxelizer._run_chain_task,
                [
                    (
                        positions[graph.stems[sid].node_ids],
                        radii[graph.stems[sid].node_ids].astype(float),
                        graph.stems[sid].is_broken,
                        phase,
                        wobble_phases,
                        trunk_voxel_size,
                        bark,
                        grain_salt,
                        moss,
                        peel,
                        budget.max_body_cells,
                    )
                    for sid, (phase, wobble_phases) in zip(thick_ids, chain_draws)
                ],
            )
            for spans in spans_per_chain:
                bodies.extend(
                    TreeVoxelizer._make_span_body(*span, trunk_voxel_size)
                    for span in spans
                )
                if progress is not None:
                    progress.step()
        else:
            for sid, (phase, wobble_phases) in zip(thick_ids, chain_draws):
                stem = graph.stems[sid]
                TreeVoxelizer._sweep_chain_span(
                    points=positions[stem.node_ids],
                    point_radii=radii[stem.node_ids].astype(float),
                    is_broken_tip=stem.is_broken,
                    arc_offset=0.0,
                    voxel_size=trunk_voxel_size,
                    bark=bark,
                    phase=phase,
                    wobble_phases=wobble_phases,
                    grain_salt=grain_salt,
                    moss=moss,
                    peel=peel,
                    max_body_cells=budget.max_body_cells,
                    bodies=bodies,
                )
                if progress is not None:
                    progress.step()
        if progress is not None:
            progress.finish()
        thick_set = set(thick_ids)
        stem_spines = [
            (
                positions[stem.node_ids],
                radii[np.asarray(stem.node_ids)].astype(float),
            )
            for sid, stem in enumerate(graph.stems)
            if sid not in thick_set and len(stem.node_ids) >= 2
        ]
        if self._pool is not None and len(stem_spines) > _TWIG_SHARD_STEMS:
            shards = [
                stem_spines[begin : begin + _TWIG_SHARD_STEMS]
                for begin in range(0, len(stem_spines), _TWIG_SHARD_STEMS)
            ]
            shard_cells = [
                cells
                for cells in self._pool.starmap(
                    TreeVoxelizer._rasterize_stem_cells,
                    [(shard, trunk_voxel_size) for shard in shards],
                )
                if cells.shape[0]
            ]
            twig_cells = (
                np.unique(np.vstack(shard_cells), axis=0)
                if shard_cells
                else np.zeros((0, 3), dtype=np.int16)
            )
        else:
            twig_cells = TreeVoxelizer._rasterize_stem_cells(
                stem_spines, trunk_voxel_size
            )
        if self._logger is not None:
            self._logger.info(
                "twigs: %d stems, %d cells", len(stem_spines), twig_cells.shape[0]
            )
        twig_optical_mids = TreeVoxelizer._make_twig_optical_mids(
            twig_cells, grain_salt, bark
        )
        if trunk_voxel_size != canopy_voxel_size:
            bodies.extend(
                TreeVoxelizer._emit_tiles(
                    twig_cells,
                    np.zeros(twig_cells.shape[0], dtype=np.uint8),
                    twig_optical_mids,
                    TreeVoxelizer._make_tile_keys(
                        twig_cells,
                        trunk_voxel_size,
                        TreeVoxelizer._get_tile_extent(trunk_voxel_size, budget),
                    ),
                    trunk_voxel_size,
                )
            )
            twig_cells = np.zeros((0, 3), dtype=np.int16)
            twig_optical_mids = np.zeros(0, dtype=np.uint8)
        has_leaves = (
            leaf is not None
            and len(leaf_optical_mids) > 0
            and graph.leaf_attaches.shape[0] > 0
        )
        if not has_leaves:
            if twig_cells.shape[0]:
                bodies.extend(
                    TreeVoxelizer._emit_tiles(
                        twig_cells,
                        np.zeros(twig_cells.shape[0], dtype=np.uint8),
                        twig_optical_mids,
                        TreeVoxelizer._make_tile_keys(
                            twig_cells,
                            trunk_voxel_size,
                            TreeVoxelizer._get_tile_extent(trunk_voxel_size, budget),
                        ),
                        trunk_voxel_size,
                    )
                )
            return bodies
        leaf_cells, leaf_looks, leaf_tiles, holder_cells, holder_tiles = (
            TreeVoxelizer._stamp_leaves(
                rng,
                graph,
                positions,
                radii,
                thick_ids,
                twig_cells,
                canopy_voxel_size,
                leaf,
                np.asarray(leaf_optical_mids, dtype=np.uint8),
                budget,
            )
        )
        cells = [leaf_cells, holder_cells]
        materials = [
            np.ones(leaf_cells.shape[0], dtype=np.uint8),
            np.zeros(holder_cells.shape[0], dtype=np.uint8),
        ]
        optical_mids = [
            leaf_looks,
            np.full(holder_cells.shape[0], bark.optical_mid_base, dtype=np.uint8),
        ]
        tiles = [leaf_tiles, holder_tiles]
        if twig_cells.shape[0]:
            # A holder blob can land on a twig cell; the holder copy rides
            # its leaf's tile, so the twig copy must yield or two bodies
            # would claim one world cell.
            keep_twigs = ~TreeVoxelizer._is_member(
                TreeVoxelizer._pack_keys(twig_cells),
                TreeVoxelizer._pack_keys(holder_cells),
            )
            twig_cells = twig_cells[keep_twigs]
            twig_optical_mids = twig_optical_mids[keep_twigs]
        if twig_cells.shape[0]:
            cells.append(twig_cells)
            materials.append(np.zeros(twig_cells.shape[0], dtype=np.uint8))
            optical_mids.append(twig_optical_mids)
            tiles.append(
                TreeVoxelizer._make_tile_keys(
                    twig_cells,
                    canopy_voxel_size,
                    TreeVoxelizer._get_tile_extent(
                        canopy_voxel_size, budget, reach=leaf.length
                    ),
                )
            )
        all_cells = np.vstack(cells)
        tile_bodies = TreeVoxelizer._emit_tiles(
            all_cells,
            np.concatenate(materials),
            np.concatenate(optical_mids),
            np.concatenate(tiles),
            canopy_voxel_size,
        )
        TreeVoxelizer._settle_loose_leaf_parts(
            tile_bodies,
            shed_fraction,
            np.sort(TreeVoxelizer._pack_keys(all_cells)),
            self._pool,
        )
        bodies.extend(body for body in tile_bodies if body.occupancy.any())
        if self._logger is not None:
            self._logger.info("swept: %d bodies", len(bodies))
        return bodies

    @staticmethod
    def iterate_structure_bodies(bodies: Sequence[TreeBody]):
        """Yield engine voxel bodies for swept tree bodies, one at a time.

        Material indices double as segment ids, so wood and leaves land
        in separate rigid segments and a soft canopy welds to its fixed
        holders. Each swept body's grids are cleared as it converts and
        each engine body yields right away, so feeding this straight
        into an entity keeps at most one body standing twice; read
        anything you need off the swept bodies before this call.

        Args:
            bodies: The swept bodies to convert and consume.
        """
        for body in bodies:
            filled = np.where(
                body.occupancy,
                body.material_grid.astype(np.uint32),
                np.uint32(RS_NULL_INDEX),
            )
            converted = entity.StructureVoxelBody.from_grid(
                filled,
                segment_grid=filled,
                material_optical_reference_mid_grid=body.optical_mid_grid.astype(
                    np.uint32
                ),
                voxel_size=body.voxel_size,
                relative_origin_position=RVec3rf(tuple(body.origin)),
                relative_orientation=RQuat3rf(tuple(body.orientation)),
            )
            body.occupancy = np.zeros((0, 0, 0), dtype=bool)
            body.material_grid = np.zeros((0, 0, 0), dtype=np.uint8)
            body.optical_mid_grid = np.zeros((0, 0, 0), dtype=np.uint8)
            yield converted

    # ------------------------------------------------------------------
    # wood: oriented limb sweeps and twig rasterization
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_thick_stems(
        graph: WeberPennGraph,
        radii: np.ndarray,
        voxel_size: float,
        budget: TreeBodyBudget,
    ) -> List[int]:
        """Return the stems thick enough to sweep as oriented bodies."""
        bar = budget.thick_radius_voxels * voxel_size
        means = np.array(
            [
                float(radii[stem.node_ids].mean()) if len(stem.node_ids) >= 2 else 0.0
                for stem in graph.stems
            ]
        )
        thick = [
            sid
            for sid in range(len(graph.stems))
            if means[sid] >= bar and len(graph.stems[sid].node_ids) >= 2
        ]
        if len(thick) > budget.max_oriented_bodies:
            ranked = sorted(thick, key=lambda sid: -means[sid])
            thick = sorted(ranked[: budget.max_oriented_bodies])
        return thick

    @staticmethod
    def _draw_chain_phases(rng: Rng, sid: int) -> Tuple[float, np.ndarray]:
        """Draw one chain's bark phase and wobble phases."""
        generator = rng.split(f"bark/{sid}").get_numpy_generator()
        phase = generator.uniform(0.0, 2.0 * math.pi)
        wobble_phases = generator.uniform(0.0, 2.0 * math.pi, 2)
        return phase, wobble_phases

    @staticmethod
    def _run_chain_task(
        points: np.ndarray,
        point_radii: np.ndarray,
        is_broken_tip: bool,
        phase: float,
        wobble_phases: np.ndarray,
        voxel_size: float,
        bark: TreeBark,
        grain_salt: np.uint32,
        moss: Optional[ValueNoiseField],
        peel: Optional[ValueNoiseField],
        max_body_cells: int,
    ) -> List[tuple]:
        """Sweep one chain on a worker and return its spans as cells.

        Spans return as (origin, orientation, shape, cells, optical
        mids) instead of dense grids; the moss and peel fields travel
        as plain field instances.
        """
        bodies: List[TreeBody] = []
        TreeVoxelizer._sweep_chain_span(
            points=points,
            point_radii=point_radii,
            is_broken_tip=is_broken_tip,
            arc_offset=0.0,
            voxel_size=voxel_size,
            bark=bark,
            phase=phase,
            wobble_phases=wobble_phases,
            grain_salt=grain_salt,
            moss=moss,
            peel=peel,
            max_body_cells=max_body_cells,
            bodies=bodies,
        )
        spans = []
        for body in bodies:
            cells = np.argwhere(body.occupancy).astype(np.int32)
            mids = body.optical_mid_grid[cells[:, 0], cells[:, 1], cells[:, 2]]
            spans.append(
                (
                    body.origin,
                    body.orientation,
                    body.occupancy.shape,
                    cells,
                    mids,
                )
            )
        return spans

    @staticmethod
    def _make_span_body(
        origin: np.ndarray,
        orientation: np.ndarray,
        shape: tuple,
        cells: np.ndarray,
        optical_mids: np.ndarray,
        voxel_size: float,
    ) -> TreeBody:
        """Rebuild one oriented span body from its worker cells."""
        occupancy = np.zeros(shape, dtype=bool)
        material_grid = np.zeros(shape, dtype=np.uint8)
        optical_mid_grid = np.zeros(shape, dtype=np.uint8)
        occupancy[cells[:, 0], cells[:, 1], cells[:, 2]] = True
        optical_mid_grid[cells[:, 0], cells[:, 1], cells[:, 2]] = optical_mids
        return TreeBody(
            occupancy,
            material_grid,
            optical_mid_grid,
            origin,
            orientation,
            voxel_size,
        )

    @staticmethod
    def _sweep_chain_span(
        *,
        points: np.ndarray,
        point_radii: np.ndarray,
        is_broken_tip: bool,
        arc_offset: float,
        voxel_size: float,
        bark: TreeBark,
        phase: float,
        wobble_phases: np.ndarray,
        grain_salt: np.uint32,
        moss,
        peel,
        max_body_cells: int,
        bodies: List[TreeBody],
    ) -> None:
        """Sweep one spine span, splitting while its box passes the budget.

        A span over the budget splits at its middle spine point; a
        single over-budget segment gains an interpolated midpoint
        first. A span no longer than its own padded thickness stops
        splitting, since its box cannot shrink further.
        """
        anchor = points[0]
        chord = points[-1] - anchor
        rotation, quaternion = TreeVoxelizer._make_frame(chord)
        spine = (points - anchor) @ rotation
        grain_pad = bark.grain_amplitude
        pad = float(point_radii.max()) + grain_pad + voxel_size
        local_min = spine.min(axis=0) - pad
        local_max = spine.max(axis=0) + pad
        counts = np.maximum(
            np.ceil((local_max - local_min) / voxel_size).astype(int), 1
        )
        vectors = np.diff(spine, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        arcs = np.concatenate([[0.0], np.cumsum(lengths)])
        is_over = int(np.prod(counts)) > max_body_cells
        if is_over and points.shape[0] == 2 and float(arcs[-1]) > 2.0 * pad:
            points = np.vstack([points[0], 0.5 * (points[0] + points[1]), points[1]])
            point_radii = np.array(
                [
                    point_radii[0],
                    0.5 * (point_radii[0] + point_radii[1]),
                    point_radii[1],
                ]
            )
            spine = (points - anchor) @ rotation
            arcs = np.concatenate([[0.0], [0.5 * arcs[-1]], [arcs[-1]]])
        if is_over and points.shape[0] >= 3:
            middle = points.shape[0] // 2
            shared = dict(
                voxel_size=voxel_size,
                bark=bark,
                phase=phase,
                wobble_phases=wobble_phases,
                grain_salt=grain_salt,
                moss=moss,
                peel=peel,
                max_body_cells=max_body_cells,
                bodies=bodies,
            )
            TreeVoxelizer._sweep_chain_span(
                points=points[: middle + 1],
                point_radii=point_radii[: middle + 1],
                is_broken_tip=False,
                arc_offset=arc_offset,
                **shared,
            )
            TreeVoxelizer._sweep_chain_span(
                points=points[middle:],
                point_radii=point_radii[middle:],
                is_broken_tip=is_broken_tip,
                arc_offset=arc_offset + float(arcs[middle]),
                **shared,
            )
            return
        occupancy = np.zeros(tuple(counts), dtype=bool)
        optical_mid_grid = np.zeros(tuple(counts), dtype=np.uint8)
        for k in range(vectors.shape[0]):
            if lengths[k] < 1e-9:
                continue
            TreeVoxelizer._stamp_wood_segment(
                occupancy,
                optical_mid_grid,
                local_min,
                voxel_size,
                a=spine[k],
                b=spine[k + 1],
                radius_a=max(float(point_radii[k]), 0.5 * voxel_size),
                radius_b=max(float(point_radii[k + 1]), 0.5 * voxel_size),
                arc_start=arc_offset + float(arcs[k]),
                bark=bark,
                phase=phase,
                wobble_phases=wobble_phases,
                grain_salt=grain_salt,
                moss=moss,
                peel=peel,
                anchor=anchor,
                rotation=rotation,
                is_ragged_end=is_broken_tip and k == vectors.shape[0] - 1,
            )
        if not occupancy.any():
            return
        origin = anchor + rotation @ local_min
        bodies.append(
            TreeBody(
                occupancy,
                np.zeros(tuple(counts), dtype=np.uint8),
                optical_mid_grid,
                origin,
                quaternion,
                voxel_size,
            )
        )

    @staticmethod
    def _stamp_wood_segment(
        occupancy: np.ndarray,
        optical_mid_grid: np.ndarray,
        local_min: np.ndarray,
        voxel_size: float,
        *,
        a: np.ndarray,
        b: np.ndarray,
        radius_a: float,
        radius_b: float,
        arc_start: float,
        bark: TreeBark,
        phase: float,
        wobble_phases: np.ndarray,
        grain_salt: np.uint32,
        moss,
        peel,
        anchor: np.ndarray,
        rotation: np.ndarray,
        is_ragged_end: bool = False,
    ) -> None:
        """Stamp one tapered capsule with bark detail into a chain grid.

        A ragged end spikes the swept radius over the last stretch of
        the segment with per-cell speckle, so a snapped stem shows a
        splintered break face instead of a smooth cap.
        """
        length = float(np.linalg.norm(b - a))
        axis = (b - a) / length
        grain_pad = bark.grain_amplitude
        reach = max(radius_a, radius_b) + grain_pad + voxel_size
        lo_cell = np.maximum(
            np.floor((np.minimum(a, b) - reach - local_min) / voxel_size).astype(int),
            0,
        )
        hi_cell = np.minimum(
            np.ceil((np.maximum(a, b) + reach - local_min) / voxel_size).astype(int),
            np.array(occupancy.shape),
        )
        if np.any(lo_cell >= hi_cell):
            return
        window = tuple(slice(lo_cell[i], hi_cell[i]) for i in range(3))
        xs = local_min[0] + (np.arange(lo_cell[0], hi_cell[0]) + 0.5) * voxel_size
        ys = local_min[1] + (np.arange(lo_cell[1], hi_cell[1]) + 0.5) * voxel_size
        zs = local_min[2] + (np.arange(lo_cell[2], hi_cell[2]) + 0.5) * voxel_size
        px = xs[:, None, None] - a[0]
        py = ys[None, :, None] - a[1]
        pz = zs[None, None, :] - a[2]
        along = np.clip(px * axis[0] + py * axis[1] + pz * axis[2], 0.0, length)
        vx = px - along * axis[0]
        vy = py - along * axis[1]
        vz = pz - along * axis[2]
        distance = np.sqrt(vx * vx + vy * vy + vz * vz)
        radius = radius_a + (radius_b - radius_a) * (along / length)
        side = TreeVoxelizer._make_side(axis)
        other = np.cross(axis, side)
        angle = np.arctan2(
            vx * other[0] + vy * other[1] + vz * other[2],
            vx * side[0] + vy * side[1] + vz * side[2],
        )
        arc = arc_start + along
        signal = TreeVoxelizer._make_detail_signal(
            bark.detail,
            angle=angle,
            arc=arc,
            radius=(radius_a + radius_b) / 2.0,
            phase=phase,
            wobble_phases=wobble_phases,
        )
        depth = bark.detail.depth if bark.detail.depth >= voxel_size else 0.0
        cut = -0.5 * depth * (1.0 - signal)
        ix = np.arange(lo_cell[0], hi_cell[0], dtype=np.uint32)[:, None, None]
        iy = np.arange(lo_cell[1], hi_cell[1], dtype=np.uint32)[None, :, None]
        iz = np.arange(lo_cell[2], hi_cell[2], dtype=np.uint32)[None, None, :]
        grain = TreeVoxelizer._hash_cells(ix, iy, iz, grain_salt)
        threshold = np.maximum(radius + cut + grain * grain_pad, voxel_size * 0.72)
        if is_ragged_end:
            splinter_span = max(1.5 * radius_b, 3.0 * voxel_size)
            ragged = along > length - splinter_span
            spikes = 0.15 + 0.85 * (np.broadcast_to(grain, ragged.shape) * 0.5 + 0.5)
            threshold = np.where(ragged, threshold * spikes, threshold)
        keep = distance <= threshold
        if not keep.any():
            return
        fresh = keep & ~occupancy[window]
        occupancy[window] |= keep
        if not fresh.any():
            return
        signal_fresh = np.broadcast_to(signal, fresh.shape)[fresh]
        grain_fresh = np.broadcast_to(grain, fresh.shape)[fresh]
        if isinstance(bark.detail, TreeBarkPatches):
            levels = np.asarray(bark.detail.optical_mid_levels, dtype=np.uint8)
            steps = max(levels.shape[0] - 1, 1)
            picked = np.clip(
                np.round((signal_fresh + 1.0) * 0.5 * steps).astype(int),
                0,
                levels.shape[0] - 1,
            )
            optical_mids = levels[picked]
        else:
            optical_mids = np.full(
                int(fresh.sum()), bark.optical_mid_base, dtype=np.uint8
            )
            optical_mids[signal_fresh > 0.55] = bark.optical_mid_light
            optical_mids[signal_fresh < -0.55] = bark.optical_mid_crevice
        optical_mids[grain_fresh < -0.7] = bark.optical_mid_shade
        if moss is not None or peel is not None:
            fx, fy, fz = np.nonzero(fresh)
            local_points = np.stack([xs[fx], ys[fy], zs[fz]], axis=1)
            world_points = anchor + local_points @ rotation.T
            if peel is not None:
                peeled = peel(world_points) * 0.5 + 0.5 < bark.peel_coverage
                optical_mids[peeled] = bark.optical_mid_peel
            if moss is not None:
                low = world_points[:, 2] < bark.moss_height
                if low.any():
                    fade = 1.0 - world_points[low, 2] / bark.moss_height
                    grown = (
                        moss(world_points[low]) * 0.5 + 0.5 < bark.moss_coverage * fade
                    )
                    mossy = np.zeros(optical_mids.shape, dtype=bool)
                    mossy[low] = grown
                    optical_mids[mossy] = bark.optical_mid_moss
        view = optical_mid_grid[window]
        view[fresh] = optical_mids

    @staticmethod
    def _make_detail_signal(
        detail,
        *,
        angle: np.ndarray,
        arc: np.ndarray,
        radius: float,
        phase: float,
        wobble_phases: np.ndarray,
    ) -> np.ndarray:
        """Return the bark detail height signal in [-1, 1] per cell.

        The signal reads over the limb surface coordinates: the angle
        around the limb and the arc length along it. 1 is a crest at
        the full swept radius, -1 the deepest cut; the optical mid mapping
        keys off the same values. A new detail family is one dataclass
        and one branch here.
        """
        turns = 2.0 * math.pi
        salt = np.uint32(abs(int(phase * 1e7)) % (2**31))
        if isinstance(detail, TreeBarkBands):
            wave = detail.wobble * np.sin(2.0 * angle + wobble_phases[0])
            signal = np.cos(arc * turns / detail.length + phase + wave * turns)
            if detail.coverage < 1.0:
                rows = np.floor(arc / detail.length + 0.5).astype(np.int64)
                sectors = max(3, int(round(turns * radius / (2.0 * detail.length))))
                around = np.floor((angle + math.pi) / turns * sectors).astype(np.int64)
                dashes = (
                    TreeVoxelizer._hash_cells(
                        rows.astype(np.uint32),
                        around.astype(np.uint32),
                        np.uint32(1) + np.zeros_like(rows, dtype=np.uint32),
                        salt,
                    )
                    * 0.5
                    + 0.5
                ) < detail.coverage
                signal = np.where(dashes, signal, 0.2)
            return signal
        if isinstance(detail, TreeBarkPlates):
            columns = max(3, int(round(turns * radius / detail.width)))
            plain_column = np.floor((angle + math.pi) / turns * columns).astype(
                np.int64
            )
            slid = (
                TreeVoxelizer._hash_cells(
                    (plain_column % columns).astype(np.uint32),
                    np.zeros_like(plain_column, dtype=np.uint32),
                    np.uint32(4) + np.zeros_like(plain_column, dtype=np.uint32),
                    salt,
                )
                * 0.5
                + 0.5
            )
            long = arc / detail.length + detail.shift * slid
            rows = np.floor(long).astype(np.int64)
            around = (angle + math.pi) / turns * columns + detail.stagger * rows
            column = np.floor(around).astype(np.int64)
            side_frac = around - column
            long_frac = long - rows
            edge = np.minimum(
                np.minimum(side_frac, 1.0 - side_frac),
                np.minimum(long_frac, 1.0 - long_frac),
            )
            height = TreeVoxelizer._hash_cells(
                (column % columns).astype(np.uint32),
                rows.astype(np.uint32),
                np.uint32(2) + np.zeros_like(column, dtype=np.uint32),
                salt,
            )
            plate = np.clip(0.4 + detail.jitter * 0.6 * height, -0.8, 1.0)
            return np.where(edge < 0.14, -1.0, plate)
        if isinstance(detail, TreeBarkPatches):
            levels = max(len(detail.optical_mid_levels), 1)
            pu = (angle + math.pi) * radius / detail.cell
            pv = arc / detail.cell
            iu = np.floor(pu).astype(np.int64)
            iv = np.floor(pv).astype(np.int64)
            fu = pu - iu
            fv = pv - iv
            fu = fu * fu * (3.0 - 2.0 * fu)
            fv = fv * fv * (3.0 - 2.0 * fv)
            field = np.zeros(np.broadcast(pu, pv).shape)
            for du in (0, 1):
                for dv in (0, 1):
                    corner = (
                        TreeVoxelizer._hash_cells(
                            (iu + du).astype(np.uint32),
                            (iv + dv).astype(np.uint32),
                            np.uint32(3) + np.zeros_like(iu, dtype=np.uint32),
                            salt,
                        )
                        * 0.5
                        + 0.5
                    )
                    weight = (fu if du else 1.0 - fu) * (fv if dv else 1.0 - fv)
                    field = field + corner * weight
            picked = np.clip(np.floor(field * levels).astype(int), 0, levels - 1)
            if levels == 1:
                return np.zeros_like(field)
            return -1.0 + 2.0 * picked / (levels - 1)
        wobble = 0.9 * np.sin(arc * turns / detail.wobble_length + wobble_phases[0])
        wobble = wobble + 0.5 * np.sin(
            arc * 2.3 * turns / detail.wobble_length + wobble_phases[1]
        )
        furrow_count = max(3, int(round(turns * radius / detail.width)))
        return np.cos(angle * furrow_count + phase + wobble)

    @staticmethod
    def _rasterize_stem_cells(
        stem_spines: List[Tuple[np.ndarray, np.ndarray]],
        voxel_size: float,
    ) -> np.ndarray:
        """Return the world-lattice cells the given stems pass through.

        Every segment stamps its one voxel center line, face connected
        at any angle, so a thin twig is exactly one voxel wide and
        never hangs by a corner; a segment thicker than a voxel adds
        its capsule cells around the line. Takes prepared (spine
        points, point radii) pairs, so a worker shard runs the same
        code on its slice of stems.
        """
        pieces = []
        for spine, point_radii in stem_spines:
            for k in range(spine.shape[0] - 1):
                radius_a = float(point_radii[k])
                radius_b = float(point_radii[k + 1])
                pieces.append(
                    TreeVoxelizer._rasterize_line_cells(
                        spine[k], spine[k + 1], voxel_size
                    )
                )
                if max(radius_a, radius_b) >= 0.87 * voxel_size:
                    pieces.append(
                        TreeVoxelizer._rasterize_capsule_cells(
                            spine[k],
                            spine[k + 1],
                            radius_a,
                            radius_b,
                            voxel_size,
                        )
                    )
        if not pieces:
            return np.zeros((0, 3), dtype=np.int16)
        return np.unique(np.vstack(pieces), axis=0)

    @staticmethod
    def _rasterize_line_cells(
        a: np.ndarray, b: np.ndarray, voxel_size: float
    ) -> np.ndarray:
        """Return the face-connected one voxel line between two points.

        Samples the segment finer than a voxel, drops repeated cells,
        and bridges any leftover diagonal step one axis at a time, so
        consecutive cells always share a face.
        """
        length = float(np.linalg.norm(b - a))
        steps = max(int(math.ceil(length / (0.4 * voxel_size))), 1)
        points = a + (b - a) * np.linspace(0.0, 1.0, steps + 1)[:, None]
        cells = np.floor(points / voxel_size).astype(np.int64)
        keep = np.ones(cells.shape[0], dtype=bool)
        keep[1:] = np.any(cells[1:] != cells[:-1], axis=1)
        cells = cells[keep]
        chain = [cells[0]]
        for k in range(1, cells.shape[0]):
            target = cells[k]
            step = chain[-1].copy()
            for axis in range(3):
                while step[axis] != target[axis]:
                    step = step.copy()
                    step[axis] += 1 if target[axis] > step[axis] else -1
                    chain.append(step)
        return np.array(chain, dtype=np.int16)

    @staticmethod
    def _rasterize_capsule_cells(
        a: np.ndarray,
        b: np.ndarray,
        radius_a: float,
        radius_b: float,
        voxel_size: float,
    ) -> np.ndarray:
        """Return the world-lattice cells one tapered capsule covers."""
        length = float(np.linalg.norm(b - a))
        reach = max(radius_a, radius_b)
        lo = np.floor((np.minimum(a, b) - reach) / voxel_size).astype(np.int64)
        hi = np.ceil((np.maximum(a, b) + reach) / voxel_size).astype(np.int64) + 1
        xs = (np.arange(lo[0], hi[0]) + 0.5) * voxel_size
        ys = (np.arange(lo[1], hi[1]) + 0.5) * voxel_size
        zs = (np.arange(lo[2], hi[2]) + 0.5) * voxel_size
        px = xs[:, None, None] - a[0]
        py = ys[None, :, None] - a[1]
        pz = zs[None, None, :] - a[2]
        if length < 1e-9:
            distance = np.sqrt(px * px + py * py + pz * pz)
            keep = distance <= reach
        else:
            axis = (b - a) / length
            along = np.clip(px * axis[0] + py * axis[1] + pz * axis[2], 0.0, length)
            vx = px - along * axis[0]
            vy = py - along * axis[1]
            vz = pz - along * axis[2]
            radius = radius_a + (radius_b - radius_a) * (along / length)
            keep = vx * vx + vy * vy + vz * vz <= radius * radius
        kx, ky, kz = np.nonzero(keep)
        return np.stack([kx + lo[0], ky + lo[1], kz + lo[2]], axis=1).astype(np.int16)

    @staticmethod
    def _make_twig_optical_mids(
        cells: np.ndarray, grain_salt: np.uint32, bark: TreeBark
    ) -> np.ndarray:
        """Return per-cell twig optical mids, base bark with shade speckle."""
        optical_mids = np.full(cells.shape[0], bark.optical_mid_base, dtype=np.uint8)
        if cells.shape[0]:
            speckle = TreeVoxelizer._hash_cells(
                cells[:, 0].astype(np.uint32),
                cells[:, 1].astype(np.uint32),
                cells[:, 2].astype(np.uint32),
                grain_salt,
            )
            optical_mids[speckle < -0.5] = bark.optical_mid_shade
        return optical_mids

    # ------------------------------------------------------------------
    # leaves: stamping onto the world lattice
    # ------------------------------------------------------------------

    @staticmethod
    def _stamp_leaves(
        rng: Rng,
        graph: WeberPennGraph,
        positions: np.ndarray,
        radii: np.ndarray,
        thick_ids: Sequence[int],
        twig_cells: np.ndarray,
        voxel_size: float,
        leaf: TreeLeaf,
        leaf_optical_mids: np.ndarray,
        budget: TreeBodyBudget,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stamp every leaf onto the world lattice, first leaf wins.

        Returns leaf cells with their optical mids and tile keys, then the
        holder cells with their tile keys. Leaf cells keep clear of
        thick limbs, holders, and twig cells already on the same
        lattice. Every cell of a leaf and of its holder carries the
        tile key of the leaf's attach point: bodies do not join, so a
        leaf split across two tile bodies would be two physical
        pieces, and whole-leaf ownership is what makes per-body
        connectivity the real hanging truth the shedding pass reads.
        """
        keep = TreeVoxelizer._thin_leaves(graph, leaf, budget.max_leaf_area)
        attaches = graph.leaf_attaches[keep]
        length_axes = graph.leaf_length_axes[keep]
        width_axes = graph.leaf_width_axes[keep]
        leaf_tile_keys = TreeVoxelizer._make_tile_keys(
            np.floor(attaches / voxel_size).astype(np.int16),
            voxel_size,
            TreeVoxelizer._get_tile_extent(voxel_size, budget, reach=leaf.length),
        )
        holder_cells, holder_owners = TreeVoxelizer._make_holder_cells(
            attaches, voxel_size, budget.holder_radius_voxels
        )
        blocked = TreeVoxelizer._rasterize_blockers(
            graph, positions, radii, thick_ids, voxel_size, budget
        )
        taken = np.unique(
            np.concatenate(
                [
                    TreeVoxelizer._pack_keys(holder_cells),
                    blocked,
                    (
                        TreeVoxelizer._pack_keys(twig_cells)
                        if twig_cells.shape[0]
                        else np.zeros(0, dtype=np.int64)
                    ),
                ]
            )
        )
        generator = rng.split("leaf_look").get_numpy_generator()
        count = attaches.shape[0]
        mids = leaf_optical_mids[
            np.minimum(
                (generator.uniform(0.0, 1.0, count) * len(leaf_optical_mids)).astype(
                    int
                ),
                len(leaf_optical_mids) - 1,
            )
        ]
        template, stalk_count = TreeVoxelizer._make_leaf_template(leaf, voxel_size)
        # Stalk samples of every leaf go ahead of every blade sample, so
        # first-wins deduplication never robs a leaf of its anchor path.
        cell_blocks = []
        leaf_blocks = []
        for part in (template[:stalk_count], template[stalk_count:]):
            points = (
                attaches[:, None, :]
                + part[None, :, 0, None] * length_axes[:, None, :]
                + part[None, :, 1, None] * width_axes[:, None, :]
            ).reshape(-1, 3)
            block = np.floor(points / voxel_size).astype(np.int16)
            owners = np.repeat(np.arange(count), part.shape[0])
            above = points[:, 2] >= voxel_size
            cell_blocks.append(block[above])
            leaf_blocks.append(owners[above])
        cells = np.vstack(cell_blocks)
        leaf_of = np.concatenate(leaf_blocks)
        keys = TreeVoxelizer._pack_keys(cells)
        _unique, first = np.unique(keys, return_index=True)
        cells = cells[first]
        leaf_of = leaf_of[first]
        keys = keys[first]
        free = ~TreeVoxelizer._is_member(keys, taken)
        return (
            cells[free],
            mids[leaf_of[free]],
            leaf_tile_keys[leaf_of[free]],
            holder_cells,
            leaf_tile_keys[holder_owners],
        )

    @staticmethod
    def _thin_leaves(
        graph: WeberPennGraph,
        leaf: TreeLeaf,
        max_leaf_area: Optional[float],
    ) -> np.ndarray:
        """Return the mask of leaves kept under the leaf area budget.

        Estimates each leaf's face area from its outline and keeps an
        even deterministic stride of leaves across the whole crown when
        the total would pass the budget, so a tree carries the same
        foliage at every canopy voxel size.
        """
        count = graph.leaf_attaches.shape[0]
        keep = np.ones(count, dtype=bool)
        if max_leaf_area is None or count == 0:
            return keep
        fill_by_outline = {
            "ellipse": 0.8,
            "hex": 0.95,
            "tee": 0.8,
            "maple_star": 0.8,
            "fan": 0.7,
            "strip": 2.2,
        }
        fill = fill_by_outline.get(leaf.outline, 0.8)
        area_per_leaf = max(leaf.length * leaf.length * leaf.width_ratio * fill, 1e-9)
        fraction = max_leaf_area / (count * area_per_leaf)
        if fraction >= 1.0:
            return keep
        ranks = np.floor(np.arange(count) * fraction)
        return np.concatenate([[True], ranks[1:] > ranks[:-1]])

    @staticmethod
    def _make_holder_cells(
        attaches: np.ndarray, voxel_size: float, holder_radius_voxels: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the fixed wood blob cells at every leaf attach point.

        Returns the cells and the leaf each cell belongs to; a cell
        shared by neighboring blobs goes to the first leaf, so its body
        follows that leaf's tile.
        """
        if not attaches.shape[0]:
            return np.zeros((0, 3), dtype=np.int16), np.zeros(0, dtype=np.int32)
        offsets = TreeVoxelizer._make_ball_offsets(holder_radius_voxels)
        centers = np.floor(attaches / voxel_size).astype(np.int16)
        cells = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        owners = np.repeat(
            np.arange(attaches.shape[0], dtype=np.int32), offsets.shape[0]
        )
        keys = TreeVoxelizer._pack_keys(cells)
        _unique, first = np.unique(keys, return_index=True)
        return cells[first], owners[first]

    @staticmethod
    def _rasterize_blockers(
        graph: WeberPennGraph,
        positions: np.ndarray,
        radii: np.ndarray,
        thick_ids: Sequence[int],
        voxel_size: float,
        budget: TreeBodyBudget,
    ) -> np.ndarray:
        """Return sorted keys of cells leaves keep clear of.

        Thick limbs block with a bleed margin; leaf-carrying twigs are
        not blockers, since leaves grow on them.
        """
        bleed = budget.blocker_bleed_voxels * voxel_size
        pieces = []
        for sid in thick_ids:
            stem = graph.stems[sid]
            spine = positions[stem.node_ids]
            for k in range(len(stem.node_ids) - 1):
                pieces.append(
                    TreeVoxelizer._rasterize_capsule_cells(
                        spine[k],
                        spine[k + 1],
                        float(radii[stem.node_ids[k]]) + bleed,
                        float(radii[stem.node_ids[k + 1]]) + bleed,
                        voxel_size,
                    )
                )
        if not pieces:
            return np.zeros(0, dtype=np.int64)
        return np.unique(TreeVoxelizer._pack_keys(np.vstack(pieces)))

    @staticmethod
    def _make_leaf_template(
        leaf: TreeLeaf, voxel_size: float
    ) -> Tuple[np.ndarray, int]:
        """Return (S, 2) sample offsets covering the leaf silhouette.

        Offsets are (along, across) meters from the attach point; the
        silhouette extends from the attach point along the length axis.
        Every outline keeps a stalk strip at the attach point, so a
        stamped leaf always touches its holder; the stalk samples come
        first and their count returns alongside, so stamping can give
        them claim priority.
        """
        spacing = 0.55 * voxel_size
        half_width = 0.5 * leaf.length * leaf.width_ratio
        u = np.arange(0.0, leaf.length + spacing, spacing)
        v = np.arange(-half_width, half_width + spacing, spacing)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        u_unit = uu / leaf.length
        v_unit = vv / max(half_width, 1e-9)
        if leaf.outline == "ellipse":
            keep = (2.0 * u_unit - 1.0) ** 2 + v_unit**2 <= 1.0
        elif leaf.outline == "tee":
            keep = (np.abs(v_unit) <= 0.34) & (u_unit <= 0.72)
            keep |= (u_unit >= 0.55) & (u_unit <= 1.0) & (np.abs(v_unit) <= 1.0)
        elif leaf.outline == "maple_star":
            cu = u_unit - 0.55
            cv = v_unit * 0.55
            radius = np.sqrt(cu**2 + cv**2)
            theta = np.arctan2(cv, cu)
            lobes = np.abs(np.cos(2.5 * theta))
            keep = radius <= 0.2 + 0.45 * lobes**1.4
        elif leaf.outline == "fan":
            spread = np.sqrt((u_unit - 0.15) ** 2 + (v_unit * 0.5) ** 2)
            keep = (np.abs(v_unit) <= 1.7 * (u_unit - 0.1)) & (spread <= 0.85)
        elif leaf.outline == "strip":
            keep = np.abs(v_unit) <= 1.0
        else:
            keep = (np.abs(v_unit) <= 1.0) & (
                np.abs(v_unit) + 1.6 * np.abs(2.0 * u_unit - 1.0) <= 1.8
            )
        stalk = (np.abs(v_unit) <= 0.25) & (u_unit <= 0.3)
        keep |= stalk
        stalk_offsets = np.stack([uu[keep & stalk], vv[keep & stalk]], axis=1)
        blade_offsets = np.stack([uu[keep & ~stalk], vv[keep & ~stalk]], axis=1)
        return (
            np.vstack([stalk_offsets, blade_offsets]),
            stalk_offsets.shape[0],
        )

    # ------------------------------------------------------------------
    # shedding: naturally loose leaf parts stay or settle by the dial
    # ------------------------------------------------------------------

    @staticmethod
    def _settle_loose_leaf_parts(
        bodies: List[TreeBody],
        shed_fraction: float,
        occupied_keys: np.ndarray,
        pool: Optional[WorkerPool],
    ) -> None:
        """Settle every naturally loose leaf part across the tile bodies.

        Voxelized leaves fragment naturally: a tilted leaf can step
        diagonally and a leaf can lose contested cells to a neighbor,
        so some leaf parts end up face-disconnected from every anchor.
        This pass finds them all, keeps an even deterministic share of
        ``shed_fraction`` loose, so a soft tree sheds them when the
        world runs while a fixed tree holds them in place, and settles
        the rest: a one-cell gap welds back, a far orphan deletes.
        At 0 every leaf part is anchored; at 1 every naturally loose
        part stays loose.

        Args:
            bodies: The tile bodies, changed in place.
            shed_fraction: Share of the loose parts kept, 0 to 1.
            occupied_keys: Sorted keys of every packed lattice cell, so
                weld cells never land where another body holds a voxel.
            pool: The worker pool the per-body search runs on; None
                searches in this process.
        """
        leafy = [
            body for body in bodies if (body.material_grid == 1)[body.occupancy].any()
        ]
        if pool is not None and len(leafy) > 1:
            handle = pool.make_shared_array(occupied_keys)
            try:
                choices_per_body = pool.starmap(
                    TreeVoxelizer._run_shed_task,
                    [
                        (
                            np.argwhere(body.occupancy).astype(np.int32),
                            body.material_grid[body.occupancy],
                            body.optical_mid_grid[body.occupancy],
                            body.occupancy.shape,
                            body.origin,
                            body.voxel_size,
                            handle,
                        )
                        for body in leafy
                    ],
                )
            finally:
                pool.free_shared_array(handle[0])
            found = [
                (body, choices)
                for body, choices in zip(leafy, choices_per_body)
                if choices
            ]
        else:
            found = []
            for body in leafy:
                labels, anchored, strays = TreeVoxelizer._find_stray_labels(body)
                if strays:
                    found.append(
                        (
                            body,
                            [
                                TreeVoxelizer._find_stray_choice(
                                    body, labels, anchored, label, occupied_keys
                                )
                                for label in strays
                            ],
                        )
                    )
        total = sum(len(choices) for _body, choices in found)
        if total == 0:
            return
        kept = TreeVoxelizer._pick_kept_strays(total, shed_fraction)
        welded: set = set()
        index = 0
        for body, choices in found:
            for choice in choices:
                if not kept[index]:
                    TreeVoxelizer._apply_stray_choice(body, *choice, welded)
                index += 1

    @staticmethod
    def _run_shed_task(
        filled_cells: np.ndarray,
        materials: np.ndarray,
        optical_mids: np.ndarray,
        shape: tuple,
        origin: np.ndarray,
        voxel_size: float,
        occupied_handle: tuple,
    ) -> List[tuple]:
        """Find one tile body's loose part choices on a worker."""
        occupied_keys = WorkerPool.read_shared_array(occupied_handle)
        occupancy = np.zeros(shape, dtype=bool)
        material_grid = np.zeros(shape, dtype=np.uint8)
        optical_mid_grid = np.zeros(shape, dtype=np.uint8)
        occupancy[filled_cells[:, 0], filled_cells[:, 1], filled_cells[:, 2]] = True
        material_grid[filled_cells[:, 0], filled_cells[:, 1], filled_cells[:, 2]] = (
            materials
        )
        optical_mid_grid[filled_cells[:, 0], filled_cells[:, 1], filled_cells[:, 2]] = (
            optical_mids
        )
        body = TreeBody(
            occupancy,
            material_grid,
            optical_mid_grid,
            origin,
            np.array([0.0, 0.0, 0.0, 1.0]),
            voxel_size,
        )
        labels, anchored, strays = TreeVoxelizer._find_stray_labels(body)
        return [
            TreeVoxelizer._find_stray_choice(
                body, labels, anchored, label, occupied_keys
            )
            for label in strays
        ]

    @staticmethod
    def _find_stray_labels(
        body: TreeBody,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Return a tile's component labels, anchored ones, and strays.

        Components come from one 6-connectivity cc3d pass over the tile.
        A component is anchored when it holds at least one wood cell,
        since fixed wood is anchored in world space and soft leaves weld
        to it; every other component is a loose leaf part. Whole-leaf
        tile ownership makes this per-body answer the physical one.
        """
        labels = cc3d.connected_components(body.occupancy, connectivity=6)
        wood = body.occupancy & (body.material_grid == 0)
        anchored = np.unique(labels[wood])
        present = np.unique(labels[body.occupancy])
        strays = sorted(set(present.tolist()) - set(anchored.tolist()) - {0})
        return labels, anchored, strays

    @staticmethod
    def _pick_kept_strays(count: int, shed_fraction: float) -> np.ndarray:
        """Return which loose parts stay loose, an even stride.

        The choice is positional over the deterministic part order
        (bodies in tile order, labels ascending), so the same tree
        sheds the same parts on every build.
        """
        held = int(round(min(max(shed_fraction, 0.0), 1.0) * count))
        if held <= 0:
            return np.zeros(count, dtype=bool)
        if held >= count:
            return np.ones(count, dtype=bool)
        ranks = np.floor(np.arange(count) * held / count)
        return np.concatenate([[True], ranks[1:] > ranks[:-1]])

    @staticmethod
    def _find_stray_choice(
        body: TreeBody,
        labels: np.ndarray,
        anchored: np.ndarray,
        label: int,
        occupied_keys: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Return one loose part's cells and its ordered weld choices.

        Most loose parts sit corner to corner with an anchored cell (a
        tilted leaf stepping diagonally), so filling one empty cell
        that face-touches both sides welds the part back. The weld cell
        must also be free on the world lattice, since a cell empty in
        this grid can hold another body's voxel.

        Returns:
            The part cells, the candidate weld cells in pick order,
            their world keys, and the part's optical mid.
        """
        part = labels == label
        xs, ys, zs = np.nonzero(part)
        lo = np.maximum(np.array([xs.min(), ys.min(), zs.min()]) - 2, 0)
        hi = np.minimum(
            np.array([xs.max(), ys.max(), zs.max()]) + 3,
            np.array(body.occupancy.shape),
        )
        window = tuple(slice(lo[i], hi[i]) for i in range(3))
        part_window = part[window]
        part_cells = np.stack([xs, ys, zs], axis=1).astype(np.int32)
        part_mid = int(body.optical_mid_grid[window][part_window][0])
        anchored_window = np.isin(labels[window], anchored) & body.occupancy[window]
        no_choice = np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int64)
        if not anchored_window.any():
            return part_cells, no_choice[0], no_choice[1], part_mid
        bridges = (
            TreeVoxelizer._get_touch_faces(part_window)
            & TreeVoxelizer._get_touch_faces(anchored_window)
            & ~body.occupancy[window]
        )
        bx, by, bz = np.nonzero(bridges)
        if not bx.shape[0]:
            return part_cells, no_choice[0], no_choice[1], part_mid
        origin_cell = np.round(body.origin / body.voxel_size).astype(np.int64)
        candidates = np.stack([bx, by, bz], axis=1) + lo + origin_cell
        keys = TreeVoxelizer._pack_keys(candidates)
        free = np.nonzero(~TreeVoxelizer._is_member(keys, occupied_keys))[0]
        candidate_cells = (np.stack([bx, by, bz], axis=1)[free] + lo).astype(np.int32)
        return part_cells, candidate_cells, keys[free], part_mid

    @staticmethod
    def _apply_stray_choice(
        body: TreeBody,
        part_cells: np.ndarray,
        candidate_cells: np.ndarray,
        candidate_keys: np.ndarray,
        part_mid: int,
        welded: set,
    ) -> None:
        """Weld one loose part at its first unclaimed choice, or delete.

        The claims set spans every body, since two tile bodies at a
        border can bridge into the same free world cell. A part with no
        choice left, a leaf that lost its whole stalk, deletes instead;
        that costs the crown about a percent of leaf cells.
        """
        for pick in range(candidate_keys.shape[0]):
            key = int(candidate_keys[pick])
            if key in welded:
                continue
            cell = candidate_cells[pick]
            body.occupancy[cell[0], cell[1], cell[2]] = True
            body.material_grid[cell[0], cell[1], cell[2]] = 1
            body.optical_mid_grid[cell[0], cell[1], cell[2]] = part_mid
            welded.add(key)
            return
        cells = part_cells
        body.occupancy[cells[:, 0], cells[:, 1], cells[:, 2]] = False
        body.material_grid[cells[:, 0], cells[:, 1], cells[:, 2]] = 0
        body.optical_mid_grid[cells[:, 0], cells[:, 1], cells[:, 2]] = 0

    @staticmethod
    def _get_touch_faces(mask: np.ndarray) -> np.ndarray:
        """Return the cells sharing a face with a mask, mask included."""
        touched = mask.copy()
        touched[1:, :, :] |= mask[:-1, :, :]
        touched[:-1, :, :] |= mask[1:, :, :]
        touched[:, 1:, :] |= mask[:, :-1, :]
        touched[:, :-1, :] |= mask[:, 1:, :]
        touched[:, :, 1:] |= mask[:, :, :-1]
        touched[:, :, :-1] |= mask[:, :, 1:]
        return touched

    # ------------------------------------------------------------------
    # finalize: packing lattice cells into engine grids
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_tiles(
        cells: np.ndarray,
        materials: np.ndarray,
        optical_mids: np.ndarray,
        tile_keys: np.ndarray,
        voxel_size: float,
    ) -> List[TreeBody]:
        """Pack lattice cells into tile bodies grouped by their keys.

        Occupied cells stay disjoint through the global claim; the tile
        boxes may overlap a little, because leaf cells ride their
        leaf's attach tile instead of their own coordinates. Bodies do
        not join, so a leaf split across two bodies would be two
        physical pieces. Each grid trims to its content.
        """
        if not cells.shape[0]:
            return []
        order = np.argsort(tile_keys, kind="stable")
        cells = cells[order]
        materials = materials[order]
        optical_mids = optical_mids[order]
        tile_keys = tile_keys[order]
        bodies = []
        starts = np.concatenate(
            [[0], np.nonzero(np.diff(tile_keys))[0] + 1, [tile_keys.shape[0]]]
        )
        for begin, end in zip(starts[:-1], starts[1:]):
            part = cells[begin:end]
            lo = part.min(axis=0)
            span = part.max(axis=0) - lo + 1
            occupancy = np.zeros(tuple(span), dtype=bool)
            material_grid = np.zeros(tuple(span), dtype=np.uint8)
            optical_mid_grid = np.zeros(tuple(span), dtype=np.uint8)
            local = part - lo
            occupancy[local[:, 0], local[:, 1], local[:, 2]] = True
            material_grid[local[:, 0], local[:, 1], local[:, 2]] = materials[begin:end]
            optical_mid_grid[local[:, 0], local[:, 1], local[:, 2]] = optical_mids[
                begin:end
            ]
            bodies.append(
                TreeBody(
                    occupancy,
                    material_grid,
                    optical_mid_grid,
                    lo.astype(float) * voxel_size,
                    np.array([0.0, 0.0, 0.0, 1.0]),
                    voxel_size,
                )
            )
        return bodies

    @staticmethod
    def _make_tile_keys(
        cells: np.ndarray, voxel_size: float, tile_extent: float
    ) -> np.ndarray:
        """Return the uint32 tile key of each lattice cell.

        Tile indices pack into 10 bits per axis around a 512 offset, so
        a tree must stay inside 512 tiles of the origin on every axis
        and a tile edge must stay inside 1024 cells; both hold by far
        for real trees and are hard asserted since the cell budget
        behind the tile edge is configurable.
        """
        if not cells.shape[0]:
            return np.zeros(0, dtype=np.uint32)
        tile_cells = max(int(round(tile_extent / voxel_size)), 1)
        assert tile_cells <= 1024, f"tile edge of {tile_cells} cells over 1024"
        tiles = np.floor_divide(cells, tile_cells) + 512
        assert (
            int(tiles.min()) >= 0 and int(tiles.max()) < 1024
        ), "tile index outside the uint32 key range"
        tiles = tiles.astype(np.uint32)
        return (tiles[:, 0] << 20) | (tiles[:, 1] << 10) | tiles[:, 2]

    @staticmethod
    def _get_tile_extent(
        voxel_size: float, budget: TreeBodyBudget, reach: float = 0.0
    ) -> float:
        """Return the tile edge kept inside the body cell budget.

        A full tile's cell count grows with the cube of the edge over
        the voxel size, so a fine voxel size halves the edge until a
        tile fits ``max_body_cells``; default sizes keep the declared
        extent and body count. ``reach`` widens the counted edge on
        both sides for content that belongs to a tile but hangs past
        its border, the way a leaf owned by its attach tile does.
        """
        extent = budget.tile_extent
        while (
            (extent + 2.0 * reach) / voxel_size
        ) ** 3 > budget.max_body_cells and extent > voxel_size:
            extent *= 0.5
        return extent

    # ------------------------------------------------------------------
    # shared small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_ball_offsets(radius_voxels: float) -> np.ndarray:
        """Return integer cell offsets within a radius of the center."""
        reach = int(math.ceil(radius_voxels))
        span = np.arange(-reach, reach + 1)
        ox, oy, oz = np.meshgrid(span, span, span, indexing="ij")
        inside = ox**2 + oy**2 + oz**2 <= radius_voxels**2
        return np.stack([ox[inside], oy[inside], oz[inside]], axis=1).astype(np.int16)

    @staticmethod
    def _pack_keys(cells: np.ndarray) -> np.ndarray:
        """Pack lattice cell coordinates into one int64 key each.

        Coordinates must fit int16; the pack hard asserts the range
        since the offset headroom is what keeps keys unique.
        """
        if not cells.shape[0]:
            return np.zeros(0, dtype=np.int64)
        assert (
            int(cells.min()) >= -_KEY_OFFSET and int(cells.max()) < _KEY_OFFSET
        ), "cell coordinate outside the int16 key range"
        shifted = cells.astype(np.int64) + _KEY_OFFSET
        return (
            (shifted[:, 0] << (2 * _KEY_SHIFT))
            | (shifted[:, 1] << _KEY_SHIFT)
            | shifted[:, 2]
        )

    @staticmethod
    def _is_member(keys: np.ndarray, sorted_keys: np.ndarray) -> np.ndarray:
        """Return which keys appear in a sorted key array."""
        if not sorted_keys.shape[0]:
            return np.zeros(keys.shape[0], dtype=bool)
        found = np.searchsorted(sorted_keys, keys)
        found = np.minimum(found, sorted_keys.shape[0] - 1)
        return sorted_keys[found] == keys

    @staticmethod
    def _hash_cells(
        ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, salt: np.uint32
    ) -> np.ndarray:
        """Return deterministic per-cell speckle in [-1, 1]."""
        with np.errstate(over="ignore"):
            h = (
                ix * np.uint32(73856093)
                ^ iy * np.uint32(19349663)
                ^ iz * np.uint32(83492791)
                ^ salt
            )
            h = h ^ (h >> np.uint32(13))
            h = h * np.uint32(0x5BD1E995)
            h = h ^ (h >> np.uint32(15))
        return h.astype(np.float64) / float(2**31) - 1.0

    @staticmethod
    def _make_side(direction: np.ndarray) -> np.ndarray:
        """Return a unit vector normal to a direction."""
        helper = (
            np.array([1.0, 0.0, 0.0])
            if abs(direction[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        side = np.array(
            [
                direction[1] * helper[2] - direction[2] * helper[1],
                direction[2] * helper[0] - direction[0] * helper[2],
                direction[0] * helper[1] - direction[1] * helper[0],
            ]
        )
        return side / np.linalg.norm(side)

    @staticmethod
    def _make_frame(direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return a body frame whose z axis runs along a direction.

        Returns:
            ``(rotation, quaternion)``: the 3x3 matrix whose columns are
            the frame axes in tree space, and the same rotation as an
            (x, y, z, w) quaternion.
        """
        norm = float(np.linalg.norm(direction))
        z_axis = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        x_axis = TreeVoxelizer._make_side(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / s,
                    (rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[1, 0] - rotation[0, 1]) / s,
                    0.25 * s,
                ]
            )
        else:
            major = int(np.argmax(np.diag(rotation)))
            minor = (major + 1) % 3
            last = (major + 2) % 3
            s = (
                math.sqrt(
                    1.0
                    + rotation[major, major]
                    - rotation[minor, minor]
                    - rotation[last, last]
                )
                * 2.0
            )
            quaternion = np.zeros(4)
            quaternion[major] = 0.25 * s
            quaternion[minor] = (rotation[minor, major] + rotation[major, minor]) / s
            quaternion[last] = (rotation[last, major] + rotation[major, last]) / s
            quaternion[3] = (rotation[last, minor] - rotation[minor, last]) / s
        return rotation, quaternion / np.linalg.norm(quaternion)
