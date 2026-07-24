# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The scene index: the two level spatial index the editor keeps over drafts.

The index is pure bookkeeping over plain values: entries are keyed by the
caller's integer uids and hold cover boxes at root-frame poses, so the index
never sees or builds Draft objects, and every answer speaks uids the caller
translates back to its own nodes. A draft in the index is either in its
initial unrealized state, present only as abstract cover boxes, or in a
realized state with actual voxels; in the realized case scene index blocks
are added to its entry so probing can be exact instead of box-accurate.

The frame vocabulary is root; "world" only ever means the compiled rsc World.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import fcl
import numpy as np

from rise_origin import RS_NULL_INDEX, RSC_StructureVoxel

from ..data.box import Box
from ..data.pose import Pose
from ..data.result import Distance

__all__ = [
    "SceneIndexError",
    "SceneIndexOverlapHit",
    "SceneIndexCellHit",
    "SceneIndexBlock",
    "SceneIndex",
]

_MIN_EXTENT = 1e-6


class SceneIndexError(RuntimeError):
    """An index registration that breaks the caller's own invariant."""


@dataclass(frozen=True)
class SceneIndexOverlapHit:
    """One cover box pair found by an overlap query.

    Attributes:
        uid: The registered entry whose box overlaps the queried cover.
        box_index: Which box of that entry's cover overlaps.
        own_box_index: Which box of the queried cover it overlaps with.
    """

    uid: int
    box_index: int
    own_box_index: int


@dataclass(frozen=True)
class SceneIndexCellHit:
    """One realized voxel found at a probed root-frame point.

    Attributes:
        uid: The registered entry whose voxel holds the point.
        material_reference: The material name of that voxel.
        is_fixed_structure: Whether the owning structure is permanently
            fixed in the engine.
    """

    uid: int
    material_reference: str
    is_fixed_structure: bool


class SceneIndexBlock:
    """One realized voxel body of an entry, viewed in the root frame.

    A block wraps one structure voxel body: it holds the body's root-frame
    pose, decodes the body's flat material sid array into an ``[i, j, k]``
    indexed grid on first use, and answers voxel questions at exact cell
    accuracy.

    Args:
        uid: The owning entry's uid.
        structure: The rsc structure voxel entity the body belongs to.
        body: The structure voxel body this block wraps.
        root_pose: Root-frame pose of the body's grid min corner.
    """

    def __init__(
        self,
        uid: int,
        structure: RSC_StructureVoxel,
        body: object,
        root_pose: Pose,
    ) -> None:
        self.uid = uid
        self.structure = structure
        self.body = body
        self.root_pose = root_pose
        self.voxel_size = float(body.voxel_size)
        self.dims = (int(body.x_voxels), int(body.y_voxels), int(body.z_voxels))
        self._rotation = np.asarray(
            root_pose.orientation.to_rotation_matrix().to_list(), dtype=float
        )
        position = root_pose.relative_position
        self._translation = np.array([position.x, position.y, position.z])
        self._material_sid: Optional[np.ndarray] = None
        self._occupied: Optional[np.ndarray] = None

    def get_material_sid(self) -> np.ndarray:
        """Return the per-voxel material sid grid indexed ``[i, j, k]``.

        The engine's flat arrays are x fastest, so the decode is a
        ``(nz, ny, nx)`` reshape followed by a transpose.
        """
        if self._material_sid is None:
            nx, ny, nz = self.dims
            flat = np.asarray(self.body.material_reference_sid, dtype=np.int64)
            self._material_sid = flat.reshape(nz, ny, nx).transpose(2, 1, 0)
        return self._material_sid

    def get_occupied_indices(self) -> np.ndarray:
        """Return the ``(n, 3)`` integer indices of occupied voxels."""
        if self._occupied is None:
            self._occupied = np.argwhere(self.get_material_sid() != RS_NULL_INDEX)
        return self._occupied

    def get_occupied_positions(self) -> np.ndarray:
        """Return the ``(n, 3)`` root-frame positions of occupied voxel centers."""
        centers = (self.get_occupied_indices() + 0.5) * self.voxel_size
        return centers @ self._rotation.T + self._translation

    def get_material_sid_at_point(
        self, position: Union[Sequence[float], np.ndarray]
    ) -> Optional[int]:
        """Return the material sid of the voxel holding a root-frame point.

        Args:
            position: The probed point in the root frame.

        Returns:
            The material sid of the voxel holding the point, or None when
            the point falls outside the body's grid or in an empty cell.
        """
        local = self._rotation.T @ (
            np.asarray(position, dtype=float) - self._translation
        )
        index = np.floor(local / self.voxel_size).astype(int)
        nx, ny, nz = self.dims
        if np.any(index < 0) or index[0] >= nx or index[1] >= ny or index[2] >= nz:
            return None
        sid = int(self.get_material_sid()[index[0], index[1], index[2]])
        return None if sid == RS_NULL_INDEX else sid


class SceneIndexEntry:
    """One registered uid: its cover, its pose, and its realized blocks."""

    def __init__(self, boxes: Tuple[Box, ...], root_pose: Pose) -> None:
        self.boxes = boxes
        self.root_pose = root_pose
        self.entities: Optional[Tuple[object, ...]] = None
        self.objects: List[Tuple[fcl.CollisionObject, object]] = []
        self.blocks: List[SceneIndexBlock] = []


class SceneIndex:
    """A two level spatial index over placed drafts.

    A draft in the index is either in its initial unrealized state, present
    only as abstract cover boxes, or in a realized state with actual
    voxels; in the realized case scene index blocks are added to its entry
    so probing can be exact instead of box-accurate. The two levels work as
    follows:

    - The box level holds every entry's cover boxes. ``insert`` registers
      one entry per uid, turning each cover box into an fcl collision
      object for exact oriented overlap tests and keeping flat arrays of
      every box's root-frame axis-aligned bounds for vectorized ray, range,
      and bound queries. Overlap, ray, nearby, bound, and distance
      questions are all answered here, so unrealized drafts are fully
      queryable from their boxes alone.
    - The voxel level holds :class:`SceneIndexBlock` views over actual
      voxel bodies, attached by ``realize`` once a draft's rsc entities
      exist. Cell probes first select candidate entries at the box level
      and only then look voxels up inside the candidates' blocks, so box
      questions never pay for voxel data.

    Every query answers in the root frame and speaks uids only; the caller
    translates uids back to its own nodes. The ``include`` and ``exclude``
    uid sets scope any query to a subtree without building a second index.
    """

    def __init__(self) -> None:
        self._manager = fcl.DynamicAABBTreeCollisionManager()
        self._manager.setup()
        self._entries: Dict[int, SceneIndexEntry] = {}
        self._owners: Dict[int, Tuple[int, int]] = {}
        self._flat_lowers: Optional[np.ndarray] = None
        self._flat_uppers: Optional[np.ndarray] = None
        self._flat_uids: Optional[np.ndarray] = None
        self._flat_box_indices: Optional[np.ndarray] = None
        self._is_flat_dirty = True

    def insert(self, uid: int, boxes: Sequence[Box], root_pose: Pose) -> None:
        """Register a new uid's cover boxes at its root-frame pose.

        Each box becomes one fcl collision object and the entry starts
        unrealized, holding no blocks. The caller owns the placement
        invariant and moves entries with :meth:`refit`.

        Args:
            uid: Key of the new entry; an already registered uid is a
                loud error.
            boxes: The cover boxes to register, in the frame ``root_pose``
                maps to root.
            root_pose: Root-frame pose of the cover boxes.
        """
        if uid in self._entries:
            raise SceneIndexError(f"uid {uid} is already registered")
        entry = SceneIndexEntry(tuple(boxes), root_pose)
        self._register_objects(uid, entry)
        self._entries[uid] = entry
        self._is_flat_dirty = True

    def refit(self, uid: int, root_pose: Pose) -> None:
        """Move a registered uid to a new root-frame pose.

        The fcl objects re-register at the new pose and any realized blocks
        rebuild there, so both levels answer at the new place.

        Args:
            uid: The registered entry to move.
            root_pose: The new root-frame pose to place it at.
        """
        entry = self._get_entry(uid)
        self._unregister_objects(entry)
        entry.root_pose = root_pose
        self._register_objects(uid, entry)
        if entry.entities is not None:
            entry.blocks = SceneIndex._make_blocks(uid, entry.entities, root_pose)
        self._is_flat_dirty = True

    def realize(self, uid: int, entities: Sequence[object]) -> None:
        """Attach the realized voxel level of a registered uid.

        Builds one block per body at the entry's pose and replaces any
        blocks attached before.

        Args:
            uid: The registered entry to attach the voxel level to.
            entities: The realized rsc entities; each structure voxel
                entity is read into one block per body.
        """
        entry = self._get_entry(uid)
        entry.entities = tuple(entities)
        entry.blocks = SceneIndex._make_blocks(uid, entry.entities, entry.root_pose)

    def remove(self, uid: int) -> None:
        """Drop a registered uid entirely, cover boxes and blocks alike.

        Args:
            uid: The registered entry to drop.
        """
        entry = self._get_entry(uid)
        self._unregister_objects(entry)
        del self._entries[uid]
        self._is_flat_dirty = True

    def get_blocks(self, uid: int) -> List[SceneIndexBlock]:
        """Return the blocks of one uid, empty when unrealized or unknown.

        Args:
            uid: The entry whose blocks to return.
        """
        entry = self._entries.get(uid)
        return [] if entry is None else list(entry.blocks)

    def query_overlaps(
        self,
        boxes: Sequence[Box],
        pose: Pose,
        include: Optional[Set[int]] = None,
        exclude: Optional[Set[int]] = None,
    ) -> List[SceneIndexOverlapHit]:
        """Return every registered box overlapping the queried cover.

        Runs one exact oriented collision test per queried box through fcl,
        so rotated covers report true overlaps rather than bound overlaps.

        Args:
            boxes: The queried cover, in the frame ``pose`` maps to root.
            pose: Root-frame pose of the queried cover.
            include: Keep only entries with these uids when given.
            exclude: Drop entries with these uids.
        """
        hits = []
        seen = set()
        for own_index, box in enumerate(boxes):
            obj, _geometry = SceneIndex._make_object(box, pose)
            request = fcl.CollisionRequest(
                num_max_contacts=1000000, enable_contact=False
            )
            data = fcl.CollisionData(request=request)
            self._manager.collide(obj, data, fcl.defaultCollisionCallback)
            for contact in data.result.contacts:
                owner = self._owners.get(id(contact.o1)) or self._owners.get(
                    id(contact.o2)
                )
                if owner is None:
                    continue
                uid, box_index = owner
                if include is not None and uid not in include:
                    continue
                if exclude is not None and uid in exclude:
                    continue
                key = (uid, box_index, own_index)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(SceneIndexOverlapHit(uid, box_index, own_index))
        return hits

    def query_ray(
        self,
        origin: Union[Sequence[float], np.ndarray],
        direction: Union[Sequence[float], np.ndarray],
        distance: Optional[float] = None,
        include: Optional[Set[int]] = None,
        exclude: Optional[Set[int]] = None,
    ) -> Optional[Tuple[int, float, np.ndarray]]:
        """Return the first box a root-frame ray hits, or None.

        The ray runs a vectorized slab test on the boxes' root-frame
        axis-aligned bounds, so a rotated cover answers at the accuracy of
        its bounds rather than its true faces.

        Args:
            origin: Ray start point in the root frame.
            direction: Unit ray direction in the root frame.
            distance: Farthest accepted travel; None means unbounded.
            include: Keep only entries with these uids when given.
            exclude: Drop entries with these uids.

        Returns:
            The hit as (uid, travel from the origin, outward axis-aligned
            face normal of the hit bound), or None when nothing is hit.
        """
        self._refresh_flat()
        if self._flat_lowers is None or self._flat_lowers.shape[0] == 0:
            return None
        o = np.asarray(origin, dtype=float)
        d = np.asarray(direction, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_low = (self._flat_lowers - o) / d
            t_high = (self._flat_uppers - o) / d
        t_near = np.fmin(t_low, t_high)
        t_far = np.fmax(t_low, t_high)
        parallel_outside = (np.abs(d) < 1e-12) & (
            (o < self._flat_lowers) | (o > self._flat_uppers)
        )
        t_near = np.where(np.abs(d) < 1e-12, -np.inf, t_near)
        t_far = np.where(np.abs(d) < 1e-12, np.inf, t_far)
        t_enter = t_near.max(axis=1)
        t_exit = t_far.min(axis=1)
        limit = np.inf if distance is None else float(distance)
        valid = (
            (t_exit >= np.maximum(t_enter, 0.0))
            & (t_enter <= limit)
            & ~parallel_outside.any(axis=1)
        )
        if include is not None:
            valid &= np.isin(self._flat_uids, list(include))
        if exclude is not None:
            valid &= ~np.isin(self._flat_uids, list(exclude))
        if not valid.any():
            return None
        t_hit = np.where(t_enter > 0.0, t_enter, 0.0)
        t_hit = np.where(valid, t_hit, np.inf)
        best = int(np.argmin(t_hit))
        axis = int(np.argmax(t_near[best]))
        normal = np.zeros(3)
        normal[axis] = -np.sign(d[axis]) if abs(d[axis]) > 1e-12 else 1.0
        return int(self._flat_uids[best]), float(t_hit[best]), normal

    def query_nearby(
        self,
        source_boxes: Sequence[Box],
        source_pose: Pose,
        distance: float,
        num: Optional[int] = None,
        include: Optional[Set[int]] = None,
        exclude: Optional[Set[int]] = None,
    ) -> List[Tuple[int, Distance]]:
        """Return uids within range of a source cover, sorted ascending.

        Distances are per-axis gaps between root-frame axis-aligned bounds
        of the box pairs, taking the minimizing pair per uid, so a rotated
        cover answers at the accuracy of its bounds.

        Args:
            source_boxes: The cover measured from, in the frame
                ``source_pose`` maps to root; empty means a point source.
            source_pose: Root-frame pose of the source cover.
            distance: Keep only entries within this range, in meters.
            num: Keep only the nearest so many entries when given.
            include: Keep only entries with these uids when given.
            exclude: Drop entries with these uids.

        Returns:
            (uid, distance) pairs sorted by the scalar distance, where the
            distance carries the per-axis gaps of the minimizing box pair.
        """
        self._refresh_flat()
        if self._flat_lowers is None or self._flat_lowers.shape[0] == 0:
            return []
        best: Dict[int, Tuple[float, np.ndarray]] = {}
        for box in source_boxes if source_boxes else [Box((0, 0, 0), (0, 0, 0))]:
            lo, hi = box.get_aabb_in_root(source_pose)
            gaps = np.maximum(
                0.0, np.maximum(lo - self._flat_uppers, self._flat_lowers - hi)
            )
            euclid = np.linalg.norm(gaps, axis=1)
            for flat_index in np.nonzero(euclid <= distance)[0]:
                uid = int(self._flat_uids[flat_index])
                if include is not None and uid not in include:
                    continue
                if exclude is not None and uid in exclude:
                    continue
                value = float(euclid[flat_index])
                if uid not in best or value < best[uid][0]:
                    best[uid] = (value, gaps[flat_index])
        results = []
        for uid, (value, gap) in best.items():
            results.append(
                (uid, Distance(float(gap[0]), float(gap[1]), float(gap[2]), value))
            )
        results.sort(key=lambda item: item[1].euclidean)
        if num is not None:
            results = results[:num]
        return results

    def get_bounds_in_root(
        self, uids: Sequence[int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the union of the given uids' root-frame bounds, or None.

        Args:
            uids: The registered uids whose bounds to union.

        Returns:
            The (lower, upper) corner pair over every cover box of the
            given uids, or None when none of them is registered.
        """
        self._refresh_flat()
        if self._flat_lowers is None or self._flat_lowers.shape[0] == 0:
            return None
        keep = np.isin(self._flat_uids, list(uids))
        if not keep.any():
            return None
        return (
            self._flat_lowers[keep].min(axis=0),
            self._flat_uppers[keep].max(axis=0),
        )

    def get_cell_hits_at_point(
        self, position: Union[Sequence[float], np.ndarray]
    ) -> List[SceneIndexCellHit]:
        """Return every realized voxel holding a root-frame point.

        The probe descends the box level first: only entries whose cover
        bounds contain the point have their blocks asked, so unrealized or
        far away entries cost nothing.

        Args:
            position: The probed point in the root frame.

        Returns:
            One hit per realized voxel whose cell holds the point; several
            hits mean several bodies occupy the same spot.
        """
        self._refresh_flat()
        if self._flat_lowers is None or self._flat_lowers.shape[0] == 0:
            return []
        point = np.asarray(position, dtype=float)
        inside = np.all(
            (self._flat_lowers <= point) & (point <= self._flat_uppers), axis=1
        )
        candidates = {int(uid) for uid in self._flat_uids[inside]}
        hits = []
        for uid in sorted(candidates):
            for block in self._entries[uid].blocks:
                sid = block.get_material_sid_at_point(point)
                if sid is None:
                    continue
                references = list(block.structure.material_references)
                hits.append(
                    SceneIndexCellHit(
                        uid,
                        references[sid] if sid < len(references) else "",
                        bool(block.structure.is_fixed),
                    )
                )
        return hits

    def compute_distance(
        self, uids_a: Sequence[int], uids_b: Sequence[int]
    ) -> Distance:
        """Return the closest approach between two registered uid sets.

        A pruned traversal: box pairs order by their axis-aligned lower
        bound distances and refine through exact oriented fcl distance only
        while the bound can still beat the best exact value found. Both
        sides answer infinity when either holds no registered boxes.

        Args:
            uids_a: The first side's registered uids.
            uids_b: The other side's registered uids.

        Returns:
            The exact distance of the minimizing box pair as the scalar,
            with that pair's axis-aligned gaps as the components; an
            overlapping pair answers zero.
        """
        self._refresh_flat()
        if self._flat_lowers is None or self._flat_lowers.shape[0] == 0:
            return Distance(np.inf, np.inf, np.inf, np.inf)
        rows_a = np.nonzero(np.isin(self._flat_uids, list(uids_a)))[0]
        rows_b = np.nonzero(np.isin(self._flat_uids, list(uids_b)))[0]
        if rows_a.size == 0 or rows_b.size == 0:
            return Distance(np.inf, np.inf, np.inf, np.inf)
        gaps = np.maximum(
            0.0,
            np.maximum(
                self._flat_lowers[rows_a][:, None, :]
                - self._flat_uppers[rows_b][None, :, :],
                self._flat_lowers[rows_b][None, :, :]
                - self._flat_uppers[rows_a][:, None, :],
            ),
        )
        bounds = np.linalg.norm(gaps, axis=2)
        order = np.argsort(bounds, axis=None)
        best_value = np.inf
        best_pair = None
        for position in order:
            i, j = np.unravel_index(position, bounds.shape)
            if bounds[i, j] >= best_value:
                break
            row_a, row_b = int(rows_a[i]), int(rows_b[j])
            value = max(0.0, float(fcl.distance(*self._get_flat_objects(row_a, row_b))))
            if value < best_value:
                best_value = value
                best_pair = (row_a, row_b)
        if best_pair is None:
            return Distance(np.inf, np.inf, np.inf, np.inf)
        row_a, row_b = best_pair
        pair_gaps = np.maximum(
            0.0,
            np.maximum(
                self._flat_lowers[row_a] - self._flat_uppers[row_b],
                self._flat_lowers[row_b] - self._flat_uppers[row_a],
            ),
        )
        return Distance(
            float(pair_gaps[0]), float(pair_gaps[1]), float(pair_gaps[2]), best_value
        )

    def _get_entry(self, uid: int) -> SceneIndexEntry:
        """Return the entry of a registered uid, or raise."""
        entry = self._entries.get(uid)
        if entry is None:
            raise SceneIndexError(f"uid {uid} is not registered")
        return entry

    def _register_objects(self, uid: int, entry: SceneIndexEntry) -> None:
        """Build and register the fcl objects of an entry's cover."""
        entry.objects = []
        for index, box in enumerate(entry.boxes):
            obj, geometry = SceneIndex._make_object(box, entry.root_pose)
            self._owners[id(geometry)] = (uid, index)
            entry.objects.append((obj, geometry))
            self._manager.registerObject(obj)

    def _unregister_objects(self, entry: SceneIndexEntry) -> None:
        """Drop an entry's fcl objects from the manager and the owner map."""
        for obj, geometry in entry.objects:
            self._manager.unregisterObject(obj)
            self._owners.pop(id(geometry), None)
        entry.objects = []

    def _get_flat_objects(
        self, row_a: int, row_b: int
    ) -> Tuple[fcl.CollisionObject, fcl.CollisionObject, object, object]:
        """Return the stored fcl objects of two flat rows with a request pair."""
        uid_a = int(self._flat_uids[row_a])
        uid_b = int(self._flat_uids[row_b])
        object_a = self._entries[uid_a].objects[int(self._flat_box_indices[row_a])][0]
        object_b = self._entries[uid_b].objects[int(self._flat_box_indices[row_b])][0]
        return object_a, object_b, fcl.DistanceRequest(), fcl.DistanceResult()

    def _refresh_flat(self) -> None:
        """Rebuild the flat root-bound arrays when marked dirty."""
        if not self._is_flat_dirty:
            return
        lowers, uppers, uids, box_indices = [], [], [], []
        for uid, entry in self._entries.items():
            for index, box in enumerate(entry.boxes):
                lo, hi = box.get_aabb_in_root(entry.root_pose)
                lowers.append(lo)
                uppers.append(hi)
                uids.append(uid)
                box_indices.append(index)
        if lowers:
            self._flat_lowers = np.asarray(lowers)
            self._flat_uppers = np.asarray(uppers)
            self._flat_uids = np.asarray(uids)
            self._flat_box_indices = np.asarray(box_indices)
        else:
            self._flat_lowers = None
            self._flat_uppers = None
            self._flat_uids = None
            self._flat_box_indices = None
        self._is_flat_dirty = False

    @staticmethod
    def _make_blocks(
        uid: int, entities: Tuple[object, ...], root_pose: Pose
    ) -> List[SceneIndexBlock]:
        """Build the voxel block views of one entry's structure entities."""
        blocks = []
        for entity in entities:
            if not isinstance(entity, RSC_StructureVoxel):
                continue
            entity_pose = Pose(entity.origin_position, entity.orientation)
            for body in entity.bodies:
                body_pose = Pose(
                    body.relative_origin_position, body.relative_orientation
                )
                pose = root_pose.compose(entity_pose.compose(body_pose))
                blocks.append(SceneIndexBlock(uid, entity, body, pose))
        return blocks

    @staticmethod
    def _make_object(box: Box, pose: Pose) -> Tuple[fcl.CollisionObject, object]:
        """Build the fcl collision object of one posed cover box.

        Returns the object together with its geometry, whose identity keys
        the owner map, because contacts report geometries, not objects.
        """
        lo = np.array([box.min.x, box.min.y, box.min.z])
        hi = np.array([box.max.x, box.max.y, box.max.z])
        extent = np.maximum(hi - lo, _MIN_EXTENT)
        center = (lo + hi) * 0.5
        rotation = np.asarray(
            pose.orientation.to_rotation_matrix().to_list(), dtype=float
        )
        position = pose.relative_position
        translation = np.array([position.x, position.y, position.z]) + rotation @ center
        geometry = fcl.Box(extent[0], extent[1], extent[2])
        obj = fcl.CollisionObject(geometry, fcl.Transform(rotation, translation))
        return obj, geometry
