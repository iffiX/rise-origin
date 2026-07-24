# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Weber-Penn trees: the parametric tree model of Weber and Penn (1995).

The model follows the paper. Every stem carries its own frame, the
relative coordinate system of paper 4.1, with the side axis ground
parallel at spawn; curvature, curve back, the random curve variation,
and the upward attraction of deep stems all bend about that axis.
Splits pitch each clone by the declination compensated split angle,
redistribute that pitch back over the remaining segments, and swing
every stem of the fork, the continuing one included, around the tree
axis by the paper's declination weighted magnitude; a stem that split
or was born as a clone halves its propensity to split again, and
fractional split rates diffuse by the paper's Floyd-Steinberg error
value (paper 4.2). Children sit at evenly spaced stations along the
parent, placed by the down-angle and rotate tables; the trunk spawns
its full ``branches`` count while deeper stems scale the count by
their length or position, so long low branches carry the most children
(paper 4.3). Lengths follow the shape envelope equations, radii the
ratio power law capped at the parent's spawn point radius, plus taper
and flare (paper 4.4). Leaves are the last level of recursion, hung on
the deepest stems by the same machinery with density ``leaves *
shape_ratio(4, offset)`` loading the outside of the crown, faces bent
up and outward (paper 4.5 and 4.9).

Not implemented from the paper: the pruning envelope, helical stems
(negative ``curve_v``), taper values past 1, lobes, and wind sway.
Extensions beyond the paper: ``ground_clearance`` truncates any stem
that descends into a band above the ground, ``max_stems`` caps how
many stems a heavily splitting tree spawns, ``break_fraction`` snaps
stems short, ``whorl_branches`` rings children at each station,
``leaf_begin`` bares the inner span of leaf stems, and ``flare_height``
keeps the base flare short. This file grows the tree graph
only; sweeping it into voxel bodies is ``voxelizer.py``. Every
random draw flows through a handed-in :class:`Rng` on stable split
paths, so the same seed always grows the same tree.

Reference paper: tmp-docs/create-look-design/nature/tree/ref-papers/
weber-penn-1995-creation-and-rendering-of-realistic-trees.pdf
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ...protocol.rng import Rng

__all__ = [
    "WeberPennParams",
    "WeberPennStem",
    "WeberPennGraph",
    "WeberPennTree",
]


@dataclass(frozen=True)
class WeberPennParams:
    """The character of one tree species, in the paper's vocabulary.

    Per-level sequences index by stem level (0 is the trunk); a level past
    the end of a sequence reads its last entry. Leaves read the entry at
    index ``levels``. Angles are degrees.

    Attributes:
        shape: Crown envelope id shaping child lengths: 0 conical,
            1 spherical, 2 hemispherical, 3 cylindrical, 4 tapered
            cylindrical, 5 flame, 6 inverse conical, 7 tend flame.
        base_size: Fraction of the trunk kept bare of branches.
        scale: Overall tree size in meters.
        scale_v: Random size variation in meters.
        levels: Stem levels grown; 1 is a bare trunk.
        ratio: Trunk radius per meter of trunk length.
        ratio_power: Child radius follows
            ``(child length / parent length) ** ratio_power``.
        flare: Trunk base widening; 0 none.
        flare_height: Height in meters the base flare spans; the flare
            stays short and wide no matter how tall the tree grows.
        base_splits: Splits at the first trunk segment; 0 keeps one stem.
        length: Per-level relative stem length.
        length_v: Per-level random length variation.
        taper: Per-level radius loss toward the stem tip, 0 to 1.
        seg_splits: Per-level average splits per segment, error diffused.
        split_angle: Per-level split pitch away from the stem.
        split_angle_v: Per-level random split pitch variation.
        curve_res: Per-level segments per stem.
        curve: Per-level total pitch a stem curves by.
        curve_back: Per-level opposite pitch of the stem's second half.
        curve_v: Per-level random curvature variation.
        down_angle: Per-level child pitch away from the parent; the entry
            at ``levels`` pitches the leaves.
        down_angle_v: Per-level variation; negative distributes the pitch
            along the parent instead of drawing it randomly.
        rotate: Per-level child spiral step around the parent; the entry
            at ``levels`` spirals the leaves.
        rotate_v: Per-level variation; a negative rotate walks the
            spawn line by 180 + rotate per child, alternating children
            around the parent.
        branches: Per-level count of child spawn points along a parent
            stem.
        whorl_branches: Per-level children grown together at each spawn
            point, spread evenly around the parent; 1 grows single
            branches, higher counts ring them the way conifers do.
        break_fraction: Per-level fraction of a stem's length kept
            before it snaps off; 0 grows the full stem. A broken stem
            is marked, ends in a ragged tip, and grows nothing past the
            break, which is how a dead snag loses its top.
        leaves: Most leaves one deepest stem carries; 0 grows a bare
            tree. Density scales by ``shape_ratio(4, offset)``, loading
            the crown's outside.
        leaf_begin: Fraction of a stem bare of leaves from its base, the
            leaf-level analog of ``base_size``; higher values push the
            foliage to the outer crown and expose the branch structure.
        attraction_up: Bend of deep stems toward vertical; negative
            weeps them down.
        ground_clearance: Height in meters of a band above the ground no
            stem may descend into; a stem reaching down into it is
            truncated there and spawns nothing beyond, and no leaf hangs
            inside it.
        max_stems: Most stems the whole tree may spawn; splitting and
            branching stop cleanly at the cap.
    """

    shape: int = 0
    base_size: float = 0.3
    scale: float = 8.0
    scale_v: float = 0.0
    levels: int = 3
    ratio: float = 0.015
    ratio_power: float = 1.2
    flare: float = 0.6
    flare_height: float = 0.3
    base_splits: int = 0
    length: Sequence[float] = (1.0, 0.3, 0.6, 0.4)
    length_v: Sequence[float] = (0.0, 0.0, 0.1, 0.0)
    taper: Sequence[float] = (1.0, 1.0, 1.0, 1.0)
    seg_splits: Sequence[float] = (0.0, 0.0, 0.0, 0.0)
    split_angle: Sequence[float] = (0.0, 0.0, 0.0, 0.0)
    split_angle_v: Sequence[float] = (0.0, 0.0, 0.0, 0.0)
    curve_res: Sequence[int] = (8, 5, 4, 3)
    curve: Sequence[float] = (0.0, -40.0, -40.0, 0.0)
    curve_back: Sequence[float] = (0.0, 0.0, 0.0, 0.0)
    curve_v: Sequence[float] = (20.0, 50.0, 75.0, 0.0)
    down_angle: Sequence[float] = (0.0, 60.0, 45.0, 45.0)
    down_angle_v: Sequence[float] = (0.0, -50.0, 10.0, 10.0)
    rotate: Sequence[float] = (0.0, 140.0, 140.0, 77.0)
    rotate_v: Sequence[float] = (0.0, 0.0, 0.0, 0.0)
    branches: Sequence[int] = (1, 30, 25, 12)
    whorl_branches: Sequence[int] = (1, 1, 1, 1)
    break_fraction: Sequence[float] = (0.0, 0.0, 0.0, 0.0)
    leaves: int = 0
    leaf_begin: float = 0.0
    attraction_up: float = 0.5
    ground_clearance: float = 0.5
    max_stems: int = 4096

    def get_at_level(self, values: Sequence, level: int):
        """Return a per-level entry, the last one covering deeper levels.

        Args:
            values: The per-level sequence read.
            level: The stem level asked for.
        """
        return values[min(level, len(values) - 1)]


@dataclass
class WeberPennStem:
    """One stem chain of a grown tree.

    Attributes:
        node_ids: The spine as graph node indices, base first; every
            consecutive pair is one swept segment.
        level: The stem's level; 0 is the trunk.
        parent_stem_id: The stem this one grew off, -1 for the trunk.
        is_broken: Whether the stem snapped off short of its full
            length; the sweep gives such tips a ragged break face.
    """

    node_ids: List[int]
    level: int
    parent_stem_id: int
    is_broken: bool = False


@dataclass
class WeberPennGraph:
    """A grown tree as nodes, stem chains, and leaf frames.

    Attributes:
        positions: Node positions in tree space, trunk base at the origin.
        radii: Stem radius at each node, in meters.
        stems: Every stem chain; index is the stem id.
        leaf_attaches: (L, 3) leaf attach points in tree space.
        leaf_length_axes: (L, 3) unit axes each leaf extends along.
        leaf_width_axes: (L, 3) unit axes across each leaf face.
        leaf_stem_ids: (L,) the stem id each leaf hangs on.
    """

    positions: List[np.ndarray] = field(default_factory=list)
    radii: List[float] = field(default_factory=list)
    stems: List[WeberPennStem] = field(default_factory=list)
    leaf_attaches: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    leaf_length_axes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    leaf_width_axes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    leaf_stem_ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )

    def get_bounds(self, margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Return the (min corner, max corner) of every node.

        Args:
            margin: Extra distance added on every side beyond the largest
                stem radius.
        """
        stacked = np.vstack(self.positions)
        pad = max(self.radii) + margin
        return stacked.min(axis=0) - pad, stacked.max(axis=0) + pad

    def rotate(self, axis: np.ndarray, angle: float) -> None:
        """Rotate the whole grown tree about an axis through the origin.

        Positions, leaf attach points, and leaf axes all turn together,
        so a leaned tree keeps its leaves facing with it.

        Args:
            axis: The unit rotation axis.
            angle: The rotation angle in radians.
        """
        cos = math.cos(angle)
        sin = math.sin(angle)

        def turn(rows: np.ndarray) -> np.ndarray:
            cross = np.cross(np.broadcast_to(axis, rows.shape), rows)
            along = rows @ axis
            return rows * cos + cross * sin + np.outer(along, axis) * (1.0 - cos)

        turned = turn(np.vstack(self.positions))
        self.positions = [turned[i] for i in range(turned.shape[0])]
        if self.leaf_attaches.shape[0]:
            self.leaf_attaches = turn(self.leaf_attaches)
            self.leaf_length_axes = turn(self.leaf_length_axes)
            self.leaf_width_axes = turn(self.leaf_width_axes)


class WeberPennTree:
    """Grows Weber-Penn tree graphs; the voxelizer sweeps them."""

    @staticmethod
    def grow(rng: Rng, params: WeberPennParams) -> WeberPennGraph:
        """Grow one tree from the origin, trunk along +z.

        Splits clone a stem's remaining segments as their own chains, and
        every chain keeps spawning children at its own stations, so
        heavily split trees read fuller. The trunk's side axis takes a
        random horizontal turn, so the first fork plane varies by seed.

        Args:
            rng: The random stream; stable split paths make the same seed
                grow the same tree.
            params: The species character.
        """
        stream = rng.split("tree")
        scale_tree = params.scale + stream.uniform(-1.0, 1.0) * params.scale_v
        trunk_length = scale_tree * (
            params.get_at_level(params.length, 0)
            + stream.uniform(-1.0, 1.0) * params.get_at_level(params.length_v, 0)
        )
        trunk_radius = trunk_length * params.ratio
        graph = WeberPennGraph()
        graph.positions.append(np.zeros(3))
        graph.radii.append(trunk_radius)
        leaf_rows: List[Tuple] = []
        roll = math.radians(stream.split("roll").uniform(0.0, 360.0))
        WeberPennTree._grow_stem(
            stream.split("trunk"),
            graph,
            params,
            leaf_rows,
            split_errors={},
            level=0,
            parent_stem_id=-1,
            base_index=0,
            heading=np.array([0.0, 0.0, 1.0]),
            side=np.array([math.cos(roll), math.sin(roll), 0.0]),
            stem_length=trunk_length,
            base_radius=trunk_radius,
            parent_length=trunk_length,
            scale_tree=scale_tree,
            offset_fraction=1.0,
        )
        WeberPennTree._apply_flare(graph, params, trunk_length)
        if leaf_rows:
            graph.leaf_attaches = np.vstack([rows[0] for rows in leaf_rows])
            graph.leaf_length_axes = np.vstack([rows[1] for rows in leaf_rows])
            graph.leaf_width_axes = np.vstack([rows[2] for rows in leaf_rows])
            graph.leaf_stem_ids = np.concatenate([rows[3] for rows in leaf_rows])
        return graph

    # ------------------------------------------------------------------
    # private static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grow_stem(
        stream: Rng,
        graph: WeberPennGraph,
        params: WeberPennParams,
        leaf_rows: List[Tuple],
        *,
        split_errors: Dict[int, float],
        level: int,
        parent_stem_id: int,
        base_index: int,
        heading: np.ndarray,
        side: np.ndarray,
        stem_length: float,
        base_radius: float,
        parent_length: float,
        scale_tree: float,
        offset_fraction: float,
        segments_done: int = 0,
        azimuth_state: float = 0.0,
        pitch_return: float = 0.0,
        is_clone: bool = False,
    ) -> None:
        """Grow one stem chain from a node, splitting and spawning.

        The stem carries its own frame, the paper's relative coordinate
        system: ``heading`` is the z axis and ``side`` the x axis,
        ground parallel at spawn, so curvature bends in the stem's own
        plane. Children and leaves sit at evenly spaced stations along
        the stem, interpolated inside segments. ``pitch_return`` spreads
        a split pitch over the remaining segments in the reverse
        direction, and a stem that split or was born as a clone halves
        its propensity to split again (paper 4.2).
        """
        if len(graph.stems) >= params.max_stems:
            return
        stem_id = len(graph.stems)
        graph.stems.append(WeberPennStem([base_index], level, parent_stem_id))
        curve_res = int(params.get_at_level(params.curve_res, level))
        segment_length = stem_length / curve_res
        taper = params.get_at_level(params.taper, level)
        whorl = max(int(params.get_at_level(params.whorl_branches, level + 1)), 1)
        stations = WeberPennTree._make_child_stations(
            params, level, stem_length, parent_length, offset_fraction, scale_tree
        )
        leaf_stations = (
            WeberPennTree._make_leaf_stations(params, stem_length, offset_fraction)
            if level + 1 >= params.levels and params.leaves > 0
            else np.zeros(0)
        )
        split_factor = 0.5 if is_clone else 1.0
        break_at = params.get_at_level(params.break_fraction, level)
        last_segment = curve_res
        if break_at > 0.0:
            drawn = break_at * (0.7 + 0.6 * stream.split("break").uniform(0.0, 1.0))
            last_segment = min(max(int(round(curve_res * drawn)), 1), curve_res)
            graph.stems[stem_id].is_broken = last_segment < curve_res
        position = graph.positions[base_index].copy()
        azimuth = azimuth_state
        done_length = segments_done * segment_length
        station_index = int(np.searchsorted(stations, done_length))
        leaf_index = int(np.searchsorted(leaf_stations, done_length))
        leaf_starts: List[np.ndarray] = []
        leaf_headings: List[np.ndarray] = []
        leaf_sides: List[np.ndarray] = []
        up = np.array([0.0, 0.0, 1.0])
        for segment in range(segments_done, last_segment):
            heading, side = WeberPennTree._bend_segment(
                stream.split(f"bend/{segment}"),
                params,
                level,
                heading,
                side,
                segment,
                pitch_return,
            )
            next_position = position + heading * segment_length
            if (
                next_position[2] < params.ground_clearance
                and next_position[2] < position[2]
            ):
                break
            done = (segment + 1) / curve_res
            radius = max(base_radius * (1.0 - taper * done), 1e-4)
            graph.positions.append(next_position.copy())
            graph.radii.append(radius)
            next_index = len(graph.positions) - 1
            graph.stems[stem_id].node_ids.append(next_index)
            end_length = (segment + 1) * segment_length
            while (
                station_index < stations.shape[0]
                and stations[station_index] <= end_length
            ):
                offset = float(stations[station_index])
                attach_position = position + heading * (
                    offset - segment * segment_length
                )
                attach_radius = max(
                    base_radius * (1.0 - taper * offset / stem_length), 1e-4
                )
                graph.positions.append(attach_position)
                graph.radii.append(attach_radius)
                attach_index = len(graph.positions) - 1
                azimuth = WeberPennTree._advance_azimuth(
                    stream.split(f"turn/{station_index}"),
                    params,
                    level + 1,
                    azimuth,
                )
                for member in range(whorl):
                    WeberPennTree._spawn_child(
                        stream.split(f"child/{station_index}/{member}"),
                        graph,
                        params,
                        leaf_rows,
                        split_errors=split_errors,
                        level=level,
                        parent_stem_id=stem_id,
                        attach_index=attach_index,
                        parent_heading=heading,
                        parent_side=side,
                        parent_length=stem_length,
                        parent_base_radius=base_radius,
                        offset=offset,
                        attach_radius=attach_radius,
                        scale_tree=scale_tree,
                        azimuth=azimuth + member * 360.0 / whorl,
                    )
                station_index += 1
            while (
                leaf_index < leaf_stations.shape[0]
                and leaf_stations[leaf_index] <= end_length
            ):
                offset = float(leaf_stations[leaf_index])
                leaf_index += 1
                leaf_starts.append(
                    position + heading * (offset - segment * segment_length)
                )
                leaf_headings.append(heading)
                leaf_sides.append(side)
            splits = WeberPennTree._draw_split_count(
                params, split_errors, level, segment, split_factor
            )
            if splits > 0 and segment < curve_res - 1:
                declination = math.degrees(math.acos(min(max(heading[2], -1.0), 1.0)))
                remaining = max(curve_res - (segment + 1), 1)
                for clone in range(splits):
                    clone_stream = stream.split(f"clone/{segment}/{clone}")
                    angle = max(
                        params.get_at_level(params.split_angle, level)
                        + clone_stream.uniform(-1.0, 1.0)
                        * params.get_at_level(params.split_angle_v, level)
                        - declination,
                        0.0,
                    )
                    clone_heading = WeberPennTree._rotate_about(
                        heading, side, math.radians(angle)
                    )
                    swing = WeberPennTree._draw_split_swing(clone_stream, declination)
                    WeberPennTree._grow_stem(
                        clone_stream,
                        graph,
                        params,
                        leaf_rows,
                        split_errors=split_errors,
                        level=level,
                        parent_stem_id=stem_id,
                        base_index=next_index,
                        heading=WeberPennTree._rotate_about(clone_heading, up, swing),
                        side=WeberPennTree._rotate_about(side, up, swing),
                        stem_length=stem_length,
                        base_radius=base_radius,
                        parent_length=parent_length,
                        scale_tree=scale_tree,
                        offset_fraction=offset_fraction,
                        segments_done=segment + 1,
                        azimuth_state=azimuth,
                        pitch_return=pitch_return - angle / remaining,
                        is_clone=True,
                    )
                swing = WeberPennTree._draw_split_swing(
                    stream.split(f"spread/{segment}"), declination
                )
                heading = WeberPennTree._rotate_about(heading, up, swing)
                side = WeberPennTree._rotate_about(side, up, swing)
                split_factor = 0.5
            position = next_position
        if leaf_starts:
            WeberPennTree._spawn_stem_leaves(
                stream.split("leaves"),
                params,
                leaf_rows,
                stem_id=stem_id,
                starts=np.vstack(leaf_starts),
                headings=np.vstack(leaf_headings),
                sides=np.vstack(leaf_sides),
            )

    @staticmethod
    def _spawn_stem_leaves(
        stream: Rng,
        params: WeberPennParams,
        leaf_rows: List[Tuple],
        *,
        stem_id: int,
        starts: np.ndarray,
        headings: np.ndarray,
        sides: np.ndarray,
    ) -> None:
        """Place one stem's leaf frames in a single batched pass.

        Leaves pitch and spiral with the ``levels`` entries of the
        down-angle and rotate tables, exactly like child stems, using
        the stem frame rows of their stations, and their width axes
        bend the faces up and outward (paper 4.9).
        """
        generator = stream.get_numpy_generator()
        count = starts.shape[0]
        leaf_level = params.levels
        attaches = starts
        keep = attaches[:, 2] >= params.ground_clearance
        if not keep.any():
            return
        down_v = params.get_at_level(params.down_angle_v, leaf_level)
        down = params.get_at_level(params.down_angle, leaf_level) + (
            generator.uniform(-1.0, 1.0, count) * down_v if down_v >= 0.0 else down_v
        )
        rotate = params.get_at_level(params.rotate, leaf_level)
        rotate_v = params.get_at_level(params.rotate_v, leaf_level)
        steps = generator.uniform(-1.0, 1.0, count) * rotate_v
        if rotate >= 0.0:
            azimuths = np.cumsum(steps + rotate)
        else:
            # A negative rotate walks the spawn line by 180 + rotate per
            # leaf, so leaves alternate around the stem while the line
            # slowly turns (paper 4.3).
            azimuths = np.cumsum(steps + 180.0 + rotate)
        length_axes = WeberPennTree._rotate_rows(
            headings, sides, np.radians(np.broadcast_to(down, (count,)))
        )
        length_axes = WeberPennTree._rotate_rows(
            length_axes, headings, np.radians(azimuths)
        )
        norms = np.linalg.norm(length_axes, axis=1)
        keep &= norms > 1e-9
        if not keep.any():
            return
        length_axes = length_axes / np.maximum(norms, 1e-9)[:, None]
        outward = attaches.copy()
        outward[:, 2] = 0.0
        outward_norms = np.linalg.norm(outward, axis=1)
        target_normals = np.zeros((count, 3))
        target_normals[:, 2] = 1.0
        has_outward = outward_norms > 1e-6
        target_normals[has_outward] += (
            outward[has_outward] / outward_norms[has_outward][:, None]
        )
        width_axes = np.cross(target_normals, length_axes)
        width_norms = np.linalg.norm(width_axes, axis=1)
        flat = width_norms < 1e-9
        if flat.any():
            width_axes[flat] = WeberPennTree._make_side_rows(length_axes[flat])
            width_norms[flat] = 1.0
        width_axes = width_axes / width_norms[:, None]
        leaf_rows.append(
            (
                attaches[keep],
                length_axes[keep],
                width_axes[keep],
                np.full(int(keep.sum()), stem_id, dtype=np.int64),
            )
        )

    @staticmethod
    def _bend_segment(
        stream: Rng,
        params: WeberPennParams,
        level: int,
        heading: np.ndarray,
        side: np.ndarray,
        segment: int,
        pitch_return: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a segment's frame after curvature and attraction.

        Everything rotates about the stem's own side axis, the paper's
        x axis: the curve and curve back share, the random curve
        variation, the split pitch return, and for deep stems the
        upward attraction ``attraction_up * declination *
        cos(orientation) / curve_res``, where the orientation of the
        frame's y axis against the sky projects the pull into the bend
        plane (paper 4.8). A positive curve bends the stem away from
        its parent the way the down angle did.
        """
        curve_res = int(params.get_at_level(params.curve_res, level))
        curve = params.get_at_level(params.curve, level)
        curve_back = params.get_at_level(params.curve_back, level)
        curve_v = params.get_at_level(params.curve_v, level)
        if curve_back and segment >= curve_res // 2:
            delta = curve_back / max(curve_res - curve_res // 2, 1)
        elif curve_back:
            delta = curve / max(curve_res // 2, 1)
        else:
            delta = curve / curve_res
        delta += stream.uniform(-1.0, 1.0) * curve_v / curve_res
        delta += pitch_return
        if level >= 2 and params.attraction_up:
            sky = WeberPennTree._cross(heading, side)[2]
            declination = math.acos(min(max(heading[2], -1.0), 1.0))
            delta -= math.degrees(
                params.attraction_up
                * declination
                * min(max(sky, -1.0), 1.0)
                / curve_res
            )
        turned = WeberPennTree._rotate_about(heading, side, math.radians(delta))
        norm = math.sqrt(turned[0] ** 2 + turned[1] ** 2 + turned[2] ** 2)
        if norm > 1e-9:
            return turned / norm, side
        return np.array([0.0, 0.0, 1.0]), side

    @staticmethod
    def _draw_split_count(
        params: WeberPennParams,
        split_errors: Dict[int, float],
        level: int,
        segment: int,
        split_factor: float,
    ) -> int:
        """Return how many clones a segment spawns, error diffused.

        Fractional split rates spread evenly by the paper's
        Floyd-Steinberg style error value, one per level, instead of
        random rounding, which clusters splits. ``split_factor`` halves
        the rate of stems that already cloned or are clones.
        """
        if level == 0 and segment == 0 and params.base_splits:
            return int(params.base_splits)
        if level == 0 and segment == 0:
            return 0
        average = params.get_at_level(params.seg_splits, level) * split_factor
        if average <= 0.0:
            return 0
        error = split_errors.get(level, 0.0)
        count = int(round(average + error))
        split_errors[level] = error + average - count
        return max(count, 0)

    @staticmethod
    def _draw_split_swing(stream: Rng, declination: float) -> float:
        """Return a fork member's turn around the tree axis, in radians.

        The paper spreads every stem of a split, the continuing stem
        included, by rotating it about the vertical axis through the
        split point by ``20 + 0.75 * (30 + |declination - 90|) *
        random**2`` degrees with a random sign, so near vertical stems
        may swing far around while near horizontal ones diverge gently
        (paper 4.2).
        """
        spin = stream.uniform(0.0, 1.0)
        swing = 20.0 + 0.75 * (30.0 + abs(declination - 90.0)) * spin * spin
        if stream.uniform(0.0, 1.0) < 0.5:
            swing = -swing
        return math.radians(swing)

    @staticmethod
    def _make_child_stations(
        params: WeberPennParams,
        level: int,
        stem_length: float,
        parent_length: float,
        offset_fraction: float,
        scale_tree: float,
    ) -> np.ndarray:
        """Return the offsets in meters where a stem spawns children.

        The trunk spawns its full ``branches`` count beyond the bare
        base; a first level branch scales the count by its length
        against the longest possible sibling, and deeper stems scale it
        down along the parent, so long low branches carry the most
        children (paper 4.3). Stations spread evenly over the span.
        """
        if level + 1 >= params.levels:
            return np.zeros(0)
        maximum = int(params.get_at_level(params.branches, level + 1))
        if maximum <= 0:
            return np.zeros(0)
        if level == 0:
            count = maximum
            begin = min(params.base_size * scale_tree, stem_length)
        elif level == 1:
            longest = params.get_at_level(params.length, 1)
            ratio = stem_length / max(parent_length * longest, 1e-6)
            count = int(round(maximum * (0.2 + 0.8 * min(ratio, 1.0))))
            begin = 0.0
        else:
            count = int(round(maximum * (1.0 - 0.5 * offset_fraction)))
            begin = 0.0
        if count <= 0:
            return np.zeros(0)
        span = stem_length - begin
        return begin + (np.arange(count) + 0.5) * (span / count)

    @staticmethod
    def _make_leaf_stations(
        params: WeberPennParams, stem_length: float, offset_fraction: float
    ) -> np.ndarray:
        """Return the offsets in meters where a deepest stem hangs leaves.

        The count is ``leaves * shape_ratio(4, offset)``, loading the
        outside of the crown (paper 4.5); ``leaf_begin`` keeps the
        inner span bare the way ``base_size`` does for the trunk.
        """
        count = int(
            round(params.leaves * WeberPennTree._shape_ratio(4, offset_fraction))
        )
        if count <= 0:
            return np.zeros(0)
        begin = min(params.leaf_begin, 1.0) * stem_length
        span = stem_length - begin
        return begin + (np.arange(count) + 0.5) * (span / count)

    @staticmethod
    def _advance_azimuth(
        stream: Rng, params: WeberPennParams, child_level: float, azimuth: float
    ) -> float:
        """Return the next child spawn azimuth after the rotate step.

        A positive rotate spirals the spawn points around the parent; a
        negative one walks the spawn line by 180 + rotate per child, so
        children alternate around the parent while the line slowly
        turns (paper 4.3).
        """
        rotate = params.get_at_level(params.rotate, child_level)
        rotate_v = params.get_at_level(params.rotate_v, child_level)
        if rotate >= 0.0:
            return azimuth + rotate + stream.uniform(-1.0, 1.0) * rotate_v
        return azimuth + 180.0 + rotate + stream.uniform(-1.0, 1.0) * rotate_v

    @staticmethod
    def _spawn_child(
        stream: Rng,
        graph: WeberPennGraph,
        params: WeberPennParams,
        leaf_rows: List[Tuple],
        *,
        split_errors: Dict[int, float],
        level: int,
        parent_stem_id: int,
        attach_index: int,
        parent_heading: np.ndarray,
        parent_side: np.ndarray,
        parent_length: float,
        parent_base_radius: float,
        offset: float,
        attach_radius: float,
        scale_tree: float,
        azimuth: float,
    ) -> None:
        """Grow one child stem off a parent station at a given azimuth.

        The child frame turns the parent's side axis around the parent
        by the azimuth, then pitches away from the parent about that
        turned axis by the down angle, so a child off a vertical parent
        starts with its side axis parallel to the ground (paper 4.1).
        The child radius follows the paper's ratio power law from the
        parent's base radius, capped by the parent's radius at the
        spawn point.
        """
        child_level = level + 1
        length_base = params.base_size * scale_tree
        ratio_done = (parent_length - offset) / max(
            parent_length - (length_base if level == 0 else 0.0), 1e-6
        )
        length_factor = params.get_at_level(
            params.length, child_level
        ) + stream.uniform(-1.0, 1.0) * params.get_at_level(
            params.length_v, child_level
        )
        if level == 0:
            child_length = (
                parent_length
                * length_factor
                * WeberPennTree._shape_ratio(params.shape, ratio_done)
            )
        else:
            child_length = length_factor * (parent_length - 0.6 * offset)
        if child_length <= 0.0:
            return
        down_v = params.get_at_level(params.down_angle_v, child_level)
        if down_v >= 0.0:
            down = (
                params.get_at_level(params.down_angle, child_level)
                + stream.uniform(-1.0, 1.0) * down_v
            )
        else:
            down = params.get_at_level(params.down_angle, child_level) + down_v * (
                1.0 - 2.0 * WeberPennTree._shape_ratio(0, ratio_done)
            )
        child_side = WeberPennTree._rotate_about(
            parent_side, parent_heading, math.radians(azimuth)
        )
        child_heading = WeberPennTree._rotate_about(
            parent_heading, child_side, math.radians(down)
        )
        child_radius = min(
            attach_radius,
            parent_base_radius
            * (child_length / max(parent_length, 1e-6)) ** params.ratio_power,
        )
        WeberPennTree._grow_stem(
            stream.split("stem"),
            graph,
            params,
            leaf_rows,
            split_errors=split_errors,
            level=child_level,
            parent_stem_id=parent_stem_id,
            base_index=attach_index,
            heading=child_heading,
            side=child_side,
            stem_length=child_length,
            base_radius=child_radius,
            parent_length=parent_length,
            scale_tree=scale_tree,
            offset_fraction=min(offset / max(parent_length, 1e-6), 1.0),
        )

    @staticmethod
    def _apply_flare(
        graph: WeberPennGraph, params: WeberPennParams, trunk_length: float
    ) -> None:
        """Widen the trunk base radii by the paper's flare curve.

        The paper flares the bottom eighth of the trunk, which reads as
        a tall cone on big trees; a real root flare is short and
        wide, so the zone height caps at ``flare_height``. The sweep
        interpolates radius linearly between nodes, so a node is
        inserted at the zone top when the trunk's first segment is
        coarser than the zone; otherwise the flare would spread over
        the whole segment.
        """
        if not params.flare:
            return
        zone = min(0.125 * trunk_length, params.flare_height)
        trunk = graph.stems[0]
        if len(trunk.node_ids) >= 2:
            base = graph.positions[trunk.node_ids[0]]
            second = graph.positions[trunk.node_ids[1]]
            span = second - base
            length = float(np.linalg.norm(span))
            if length > zone * 1.5:
                part = zone / length
                graph.positions.append(base + span * part)
                first_radius = graph.radii[trunk.node_ids[0]]
                second_radius = graph.radii[trunk.node_ids[1]]
                graph.radii.append(first_radius + (second_radius - first_radius) * part)
                trunk.node_ids.insert(1, len(graph.positions) - 1)
        for index, position in enumerate(graph.positions):
            done = position[2] / max(zone, 1e-6)
            if done < 0.0 or done > 1.0:
                continue
            y = max(0.0, 1.0 - done)
            graph.radii[index] *= params.flare * (100.0**y - 1.0) / 100.0 + 1.0

    @staticmethod
    def _shape_ratio(shape: int, done: float) -> float:
        """Return the paper's crown envelope ratio at a trunk fraction."""
        done = min(max(done, 0.0), 1.0)
        if shape == 0:
            return 0.2 + 0.8 * done
        if shape == 1:
            return 0.2 + 0.8 * math.sin(math.pi * done)
        if shape == 2:
            return 0.2 + 0.8 * math.sin(0.5 * math.pi * done)
        if shape == 3:
            return 1.0
        if shape == 4:
            return 0.5 + 0.5 * done
        if shape == 5:
            return done / 0.7 if done <= 0.7 else (1.0 - done) / 0.3
        if shape == 6:
            return 1.0 - 0.8 * done
        return 0.5 + 0.5 * done / 0.7 if done <= 0.7 else 0.5 + 0.5 * (1.0 - done) / 0.3

    @staticmethod
    def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the cross product of two 3-vectors without wrapper cost."""
        return np.array(
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        )

    @staticmethod
    def _make_side_rows(directions: np.ndarray) -> np.ndarray:
        """Return unit vectors normal to each row direction."""
        helpers = np.zeros_like(directions)
        first = np.abs(directions[:, 0]) < 0.9
        helpers[first, 0] = 1.0
        helpers[~first, 1] = 1.0
        sides = np.cross(directions, helpers)
        return sides / np.linalg.norm(sides, axis=1)[:, None]

    @staticmethod
    def _rotate_about(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Return a vector rotated about a unit axis by an angle."""
        cos = math.cos(angle)
        sin = math.sin(angle)
        cross = WeberPennTree._cross(axis, vector)
        along = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2]
        return vector * cos + cross * sin + axis * (along * (1.0 - cos))

    @staticmethod
    def _rotate_rows(
        vectors: np.ndarray, axes: np.ndarray, angles: np.ndarray
    ) -> np.ndarray:
        """Return each row vector rotated about its row axis by its angle."""
        cos = np.cos(angles)[:, None]
        sin = np.sin(angles)[:, None]
        cross = np.cross(axes, vectors)
        along = np.sum(axes * vectors, axis=1)[:, None]
        return vectors * cos + cross * sin + axes * (along * (1.0 - cos))
