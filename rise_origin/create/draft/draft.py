# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The Draft: the thing being made, one recursive tree class.

A node holds an author-chosen local frame, a box cover as the authoring-time
spatial truth, declared material requires, plain rscs, and loose tags
and annotations. A query's scope and frame are the draft it is called on:
it sees exactly that draft's subtree and speaks that draft's frame.
Placement re-parents, so the tree is the whole
dependency structure and moving a node carries its subtree by frame
composition.

Drafts never touch the Editor: every node holds only an editor access that
allocates uids, declares requires, creates modules, loads cached rscs, and
answers gated scene reads for nodes placed in the live session. Session
mutations go only through the editor's own verbs; a detached tree attaches
structurally and answers its queries through a scratch index. Effective
fixedness is a node's own flag or any ancestor's, fixing direction only.
"""

import fnmatch
import math
from typing import (
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from rise_origin import (
    RQuat3rf,
    RSC_EntityBase,
    RSC_FloorArray,
    RSC_FloorBase,
    RSC_FloorFlat,
    RSC_StructureVoxel,
    RVec3rf,
)

from ..constants import CONTACT_OVERLAP_EPSILON, FACE_PLANE_EPSILON
from ..data.annotation import (
    Annotation,
    AnnotationNote,
    AnnotationPath,
    AnnotationPoint,
    AnnotationRegion,
    AnnotationView,
)
from ..data.box import AABB, Box
from ..data.dir import Dir
from ..data.identity import Identity
from ..data.material import MaterialRequire
from ..data.path import Path
from ..data.pose import Pose
from ..data.result import (
    AlongJitter,
    Distance,
    Filters,
    FloorSample,
    GridJitter,
    LeanResult,
    NearbyHit,
    Overlap,
    PlacementReport,
    RayHit,
    RingJitter,
    SurfaceAnchor,
)
from ..editor.index import SceneIndex
from .draft_plan_view import DraftPlanView, DraftPlanViewBox

__all__ = ["DraftError", "DraftScopeError", "Draft"]

_DIR_VECTORS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
_ANNOTATION_KINDS = {
    "point": AnnotationPoint,
    "region": AnnotationRegion,
    "path": AnnotationPath,
    "view": AnnotationView,
    "note": AnnotationNote,
}


class DraftError(RuntimeError):
    """A Draft operation that cannot be applied."""


class DraftScopeError(DraftError):
    """A node argument that lies outside this draft's subtree."""


class Draft:
    """A node of the build tree: the thing being made.

    Args:
        editor_access: The access wall of the owning editor; the only door a
            node or module body ever holds.
        name: Node name; sibling names uniquify on insert.
        uid: Adopt this uid instead of allocating a fresh one; None
            allocates through the access. Journal replay passes recorded
            uids so rebuilt nodes keep their identity; adopting a uid the
            session never allocated breaks uid uniqueness.
        boxes: Cover boxes with material class tags, the authoring truth,
            in this node's local frame.
        rscs: Plain rsc objects this node carries: entities, essences, and
            resources alike.
        requires: Declared material requires of this node's own content.
        is_fixed: Whether this node's content stands permanently fixed; any
            fixed ancestor fixes the subtree below it.
        tags: Loose search strings.
        annotations: Named 3D data in this node's local frame.
    """

    def __init__(
        self,
        editor_access,
        name: str,
        *,
        uid: Optional[int] = None,
        boxes: Sequence[Box] = (),
        rscs: Sequence[object] = (),
        requires: Sequence[MaterialRequire] = (),
        is_fixed: bool = False,
        tags: Sequence[str] = (),
        annotations: Optional[Mapping[str, Annotation]] = None,
    ) -> None:
        self.editor_access = editor_access
        self.uid: int = editor_access.allocate_uid() if uid is None else int(uid)
        self.name = name
        self.parent: Optional["Draft"] = None
        self.children: List["Draft"] = []
        self.relative_pose = Pose()
        self.identity: Optional[Identity] = None
        self.is_deterministic = True
        self.boxes: List[Box] = list(boxes)
        self.rscs: List[object] = list(rscs)
        self.requires: List[MaterialRequire] = list(requires)
        self.is_fixed = bool(is_fixed)
        self.tags: Set[str] = set(tags)
        self.annotations: Dict[str, Annotation] = dict(annotations or {})
        self.has_floor = any(isinstance(rsc, RSC_FloorBase) for rsc in rscs)
        self.is_realized = True

    def __repr__(self) -> str:
        return f"Draft('{self.name}', uid={self.uid})"

    # ------------------------------------------------------------------
    # tree and frames
    # ------------------------------------------------------------------

    def get_root(self) -> "Draft":
        """Return the top-most ancestor of this node."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def iterate_subtree_nodes(self) -> Iterator["Draft"]:
        """Yield the subtree depth first, this draft included."""
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def find_node_by_name(self, name: str) -> "Draft":
        """Return the first depth-first node of the given name in the subtree.

        Args:
            name: Name to match against nodes in this draft's subtree.
        """
        for node in self.iterate_subtree_nodes():
            if node.name == name:
                return node
        raise DraftScopeError(f"no node named '{name}' under '{self.name}'")

    def get_pose(self, thing: Union["Draft", str]) -> Pose:
        """Return a sub node's pose in this draft's frame.

        Args:
            thing: The sub node to pose, given directly or by name; it must
                sit inside this draft's subtree.
        """
        node = self._resolve_node_in_scope(thing)
        return self._get_node_pose_in_self(node)

    def get_bounds(
        self,
        node: Union["Draft", str, None] = None,
        rotation: Optional[RQuat3rf] = None,
    ) -> AABB:
        """Return cover bounds in this draft's frame.

        Args:
            node: The sub node whose subtree is bounded, given directly or
                by name; ``None`` bounds the whole subtree of this draft.
            rotation: Bounds the subtree's cover under this hypothetical
                orientation at its current position, taking the union over
                the rotated cover boxes rather than rotating one fat box;
                ``None`` keeps the current orientation.
        """
        target = self if node is None else self._resolve_node_in_scope(node)
        items = self._get_node_cover_boxes(target, self)
        if not items:
            raise DraftError(f"'{target.name}' has no cover boxes to bound")
        if rotation is not None:
            base = self._get_node_pose_in_self(target)
            override = Pose(base.relative_position, rotation)
            local_items = self._get_node_cover_boxes(target, target)
            items = [(box, override.compose(pose)) for box, pose in local_items]
        lowers, uppers = [], []
        for box, pose in items:
            lo, hi = box.get_aabb_in_root(pose)
            lowers.append(lo)
            uppers.append(hi)
        return AABB(
            tuple(np.min(np.asarray(lowers), axis=0)),
            tuple(np.max(np.asarray(uppers), axis=0)),
        )

    def get_supporters(self, thing: Union["Draft", str]) -> List["Draft"]:
        """Return the parent chain of a sub node, nearest first, up to this
        draft.

        Args:
            thing: The sub node whose parent chain is returned, given
                directly or by name.
        """
        node = self._resolve_node_in_scope(thing)
        chain = []
        walker = node.parent
        while walker is not None:
            chain.append(walker)
            if walker is self:
                break
            walker = walker.parent
        return chain

    def get_dependents(self, thing: Union["Draft", str]) -> List["Draft"]:
        """Return every descendant of a sub node, depth first.

        Args:
            thing: The sub node whose descendants are returned, given
                directly or by name.
        """
        node = self._resolve_node_in_scope(thing)
        return [child for child in node.iterate_subtree_nodes() if child is not node]

    def get_effective_is_fixed(self) -> bool:
        """Return whether this node or any ancestor is fixed."""
        walker: Optional["Draft"] = self
        while walker is not None:
            if walker.is_fixed:
                return True
            walker = walker.parent
        return False

    # ------------------------------------------------------------------
    # API category 1: geometric queries and checks
    # ------------------------------------------------------------------

    def get_floor_at(self, x: float, y: float) -> Optional[FloorSample]:
        """Sample the floor under a point in this draft's frame, or None.

        Only the floor's own node and the root can answer: the floor child
        samples itself, the root asks its floor child, and every other
        draft answers None.

        Args:
            x: X of the sample point in this draft's frame.
            y: Y of the sample point in this draft's frame.
        """
        floor_node = self._find_floor_node()
        if floor_node is None:
            return None
        floor_entity = next(
            rsc for rsc in floor_node.rscs if isinstance(rsc, RSC_FloorBase)
        )
        point_in_root = self._get_node_pose_in_root(self).apply(
            (float(x), float(y), 0.0)
        )
        node_pose = self._get_node_pose_in_root(floor_node)
        local = Draft._make_array(node_pose.invert().apply(point_in_root))
        sample = (
            Draft._sample_floor_flat(floor_entity, local)
            if isinstance(floor_entity, RSC_FloorFlat)
            else Draft._sample_floor_array(floor_entity, local)
        )
        if sample is None:
            return None
        height_local, normal_local, slope_deg = sample
        surface_in_root = node_pose.apply(
            (float(local[0]), float(local[1]), height_local)
        )
        self_pose = self._get_node_pose_in_root(self)
        surface_in_self = self_pose.invert().apply(surface_in_root)
        normal_in_root = Draft._make_rotation(node_pose.orientation) @ normal_local
        normal_in_self = Draft._make_rotation(self_pose.orientation).T @ normal_in_root
        return FloorSample(float(surface_in_self.z), tuple(normal_in_self), slope_deg)

    def cast_ray(
        self,
        from_position,
        direction: Dir,
        distance: Optional[float] = None,
    ) -> Optional[RayHit]:
        """Return the first cover blocker along a ray in this draft's frame.

        Args:
            from_position: Ray origin in this draft's frame.
            direction: Ray direction, an axis name or a free vector in this
                draft's frame.
            distance: Longest travel to search along the ray; ``None``
                searches without a limit.
        """
        hit = self._cast_ray_in_root(from_position, direction, distance)
        if hit is None:
            return None
        node, t, _normal = hit
        return RayHit(t, node)

    def get_bound_surface_at(
        self,
        from_position,
        direction: Dir,
        distance: Optional[float] = None,
    ) -> Optional[SurfaceAnchor]:
        """Return where a ray in this draft's frame meets declared matter.

        Args:
            from_position: Ray origin in this draft's frame.
            direction: Ray direction, an axis name or a free vector in this
                draft's frame.
            distance: Longest travel to search along the ray; ``None``
                searches without a limit.
        """
        hit = self._cast_ray_in_root(from_position, direction, distance)
        if hit is None:
            return None
        node, t, normal_in_root = hit
        self_pose = self._get_node_pose_in_root(self)
        direction_in_self = Draft._resolve_dir(direction)
        origin = np.asarray(from_position, dtype=float)
        position = origin + direction_in_self * t
        normal_in_self = Draft._make_rotation(self_pose.orientation).T @ normal_in_root
        return SurfaceAnchor(tuple(position), tuple(normal_in_self), node)

    def get_distance(self, a: Union["Draft", str], b: Union["Draft", str]) -> Distance:
        """Return the closest approach between two sub nodes' covers, by axis.

        When one argument nests inside the other, the outer side drops the
        inner subtree, so the answer measures the matter around it rather
        than the thing against itself.

        Args:
            a: One sub node of the pair, given directly or by name.
            b: The other sub node of the pair, given directly or by name.

        Returns:
            The closest approach, whose axis components are root-frame gaps
            of the minimizing box pair.
        """
        node_a = self._resolve_node_in_scope(a)
        node_b = self._resolve_node_in_scope(b)
        uids_a, uids_b = Draft._make_disjoint_uid_sets(node_a, node_b)
        return self._compute_index_distance(uids_a, uids_b)

    def get_gap(
        self, a: Union["Draft", str], b: Union["Draft", str], direction: Dir
    ) -> float:
        """Return the travel of sub node ``a`` along a direction until it
        first meets sub node ``b``.

        Args:
            a: The sub node that travels, given directly or by name.
            b: The sub node it travels toward until first contact, given
                directly or by name.
            direction: An axis-name direction runs on axis intervals and
                reports penetration as a negative value; a free vector
                direction sweeps the boxes and reports an existing overlap
                as zero.

        Returns:
            The travel until first contact; infinity means the two slide
            past forever.
        """
        node_a = self._resolve_node_in_scope(a)
        node_b = self._resolve_node_in_scope(b)
        items_a, items_b = self._get_disjoint_node_cover_boxes(node_a, node_b)
        if not items_a or not items_b:
            raise DraftError("both nodes need cover boxes for a gap")
        if isinstance(direction, str):
            return Draft._compute_axis_gap(items_a, items_b, direction)
        return Draft._compute_sweep_gap(items_a, items_b, direction)

    def get_nearby(
        self,
        from_thing: Union["Draft", str, AABB],
        distance: float = 10.0,
        num: Optional[int] = None,
        filters: Optional[Filters] = None,
    ) -> List[NearbyHit]:
        """Return what is placed around a sub node or a box in this draft's
        frame.

        Args:
            from_thing: The sub node or box measured from, in this draft's
                frame; a sub node given directly or by name, or an ``AABB``.
            distance: Keep only nodes within this range, in meters.
            num: Keep only the nearest so many nodes when given.
            filters: Keep only nodes passing these predicates when given.

        Returns:
            What is placed around the source, its distance components taken
            from the root-frame axis gaps of the minimizing box pair.
        """
        self_pose = self._get_node_pose_in_root(self)
        if isinstance(from_thing, AABB):
            source_boxes = [Box(from_thing.min, from_thing.max)]
            source_pose = self_pose
            exclude: Set[int] = set()
        else:
            node = self._resolve_node_in_scope(from_thing)
            source_items = self._get_node_cover_boxes(node, node)
            node_pose = self._get_node_pose_in_root(node)
            node_position = Draft._make_array(node_pose.relative_position)
            source_boxes = []
            for box, pose in source_items:
                lo, hi = box.get_aabb_in_root(node_pose.compose(pose))
                source_boxes.append(Box(lo - node_position, hi - node_position))
            source_pose = Pose(node_pose.relative_position)
            exclude = {n.uid for n in node.iterate_subtree_nodes()}
        include = self._make_scope_uids()
        allowed = self._apply_filters(filters)
        if allowed is not None:
            include = allowed if include is None else (include & allowed)
        hits = self._query_nearby(
            source_boxes,
            source_pose,
            distance,
            num=num,
            include=include,
            exclude=exclude,
        )
        nodes = self._make_nodes_by_uid()
        return [NearbyHit(nodes[uid], value) for uid, value in hits]

    def get_surrounding(
        self,
        point,
        distance: float = 10.0,
        num: Optional[int] = None,
        filters: Optional[Filters] = None,
    ) -> List[NearbyHit]:
        """Return what is placed around a raw point in this draft's frame.

        Args:
            point: The point measured from, in this draft's frame.
            distance: Keep only nodes within this range, in meters.
            num: Keep only the nearest so many nodes when given.
            filters: Keep only nodes passing these predicates when given.
        """
        point_in_root = self._get_node_pose_in_root(self).apply(point)
        region = AABB(point_in_root, point_in_root)
        hits = Draft.get_nearby(self.get_root(), region, distance, None, filters)
        include = self._make_scope_uids()
        if include is not None:
            hits = [hit for hit in hits if hit.node.uid in include]
        return hits if num is None else hits[:num]

    def get_plan_view(self, min_bound, max_bound) -> DraftPlanView:
        """Return the plan view of a box of space in this draft's frame.

        This draft's frame must not be rotated against the root, since a
        plan view is gravity aligned.

        Args:
            min_bound: Low corner of the box of space, in this draft's
                frame.
            max_bound: High corner of the box of space, in this draft's
                frame.
        """
        self_pose = self._get_node_pose_in_root(self)
        if abs(float(self_pose.orientation.w) - 1.0) > 1e-9:
            raise DraftError("plan view needs an unrotated draft")
        lo = np.asarray(min_bound, dtype=float)
        hi = np.asarray(max_bound, dtype=float)
        band_top = float(hi[2])
        items = []
        for node in self.iterate_subtree_nodes():
            if not node.boxes:
                continue
            pose = self._get_node_pose_in_self(node)
            for box in node.boxes:
                box_lo, box_hi = box.get_aabb_in_root(pose)
                if np.any(box_lo > hi) or np.any(box_hi < lo):
                    continue
                items.append(
                    DraftPlanViewBox(
                        node=node,
                        min_xy=(float(box_lo[0]), float(box_lo[1])),
                        max_xy=(float(box_hi[0]), float(box_hi[1])),
                        z_interval=(float(box_lo[2]), float(box_hi[2])),
                        is_cut=bool(box_lo[2] <= band_top <= box_hi[2]),
                    )
                )
        return DraftPlanView(items, tuple(lo), tuple(hi))

    def check_placement_validity(self, thing: Union["Draft", str]) -> PlacementReport:
        """Report every cover overlap between a placed sub node and the scope.

        Overlaps classify by the boxes' material classes: static with static
        is a legal embedding, anything dynamic is a violation, and a mixed
        box defers to the voxel gate at realization (pending).

        Args:
            thing: The placed sub node to check, given directly or by name.
        """
        node = self._resolve_node_in_scope(thing)
        nodes = self._make_nodes_by_uid()
        root = self.get_root()
        subtree_uids = {n.uid for n in node.iterate_subtree_nodes()}
        include = self._make_scope_uids()
        violations, embeddings, pending = [], [], []
        for own in node.iterate_subtree_nodes():
            if not own.boxes:
                continue
            own_pose = root._get_node_pose_in_self(own)
            hits = self._query_overlaps(
                own.boxes, own_pose, include=include, exclude=subtree_uids
            )
            for hit in hits:
                other = nodes[hit.uid]
                own_box = own.boxes[hit.own_box_index]
                other_box = other.boxes[hit.box_index]
                other_pose = root._get_node_pose_in_self(other)
                own_lo, own_hi = own_box.get_aabb_in_root(own_pose)
                other_lo, other_hi = other_box.get_aabb_in_root(other_pose)
                widths = np.minimum(own_hi, other_hi) - np.maximum(own_lo, other_lo)
                widths = np.maximum(widths, 0.0)
                if float(widths.min()) <= CONTACT_OVERLAP_EPSILON:
                    continue
                axis = int(np.argmin(widths))
                translation = np.zeros(3)
                own_center = (own_lo + own_hi) * 0.5
                other_center = (other_lo + other_hi) * 0.5
                sign = 1.0 if own_center[axis] >= other_center[axis] else -1.0
                translation[axis] = sign * widths[axis]
                overlap = Overlap(
                    other,
                    tuple(widths),
                    tuple(translation),
                    (own_box.material_class, other_box.material_class),
                )
                classes = {own_box.material_class, other_box.material_class}
                if "mixed" in classes:
                    pending.append(overlap)
                elif "dynamic" in classes:
                    violations.append(overlap)
                else:
                    embeddings.append(overlap)
        return PlacementReport(
            not violations, tuple(violations), tuple(embeddings), tuple(pending)
        )

    # ------------------------------------------------------------------
    # API category 2: solvers
    # ------------------------------------------------------------------

    def compute_snap_pose(
        self,
        from_thing: "Draft",
        from_face: Dir,
        from_relative_point,
        target: "Draft",
        target_face: Dir,
        target_relative_point,
        rotation: float = 0.0,
        gap: float = 0.0,
    ) -> Pose:
        """Solve the pose aligning a point on one face to a point on another.

        Face selectors are in each node's own local frame; relative points
        are offsets from the face center along its ``(u, v)`` basis, with
        ``(0, 0)`` meaning the center.

        Args:
            from_thing: The mover, aligned onto the target; it may be
                detached.
            from_face: Which face of the mover carries the aligned point.
            from_relative_point: The aligned point on the mover's face.
            target: The sub node the mover aligns onto.
            target_face: Which face of the target carries the meeting point.
            target_relative_point: The meeting point on the target's face.
            rotation: Spins the mover about the shared normal, in degrees.
            gap: Leaves clearance along the target normal.

        Returns:
            The aligning pose in this draft's frame; whether it fits is
            judged by :meth:`check_placement_validity`.
        """
        target_node = self._resolve_node_in_scope(target)
        f_center, f_normal, f_u, f_v = Draft._get_face_frame(from_thing, from_face)
        t_center, t_normal, t_u, t_v = Draft._get_face_frame(target_node, target_face)
        target_pose = self._get_node_pose_in_self(target_node)
        t_rotation = Draft._make_rotation(target_pose.orientation)
        n_t = t_rotation @ t_normal
        u_t = t_rotation @ t_u
        anchor_t = Draft._make_array(
            target_pose.apply(
                t_center
                + t_u * float(target_relative_point[0])
                + t_v * float(target_relative_point[1])
            )
        )
        spin = Draft._make_matrix(
            Draft._make_quat_from_axis_angle(n_t, math.radians(rotation))
        )
        u_goal = spin @ u_t
        n_goal = -n_t
        v_goal = np.cross(n_goal, u_goal)
        basis_from = np.column_stack([f_u, f_v, f_normal])
        basis_goal = np.column_stack([u_goal, v_goal, n_goal])
        rotation_matrix = basis_goal @ basis_from.T
        orientation = Draft._make_quat_from_matrix(rotation_matrix)
        anchor_f = (
            f_center
            + f_u * float(from_relative_point[0])
            + f_v * float(from_relative_point[1])
        )
        position = anchor_t + n_t * float(gap) - rotation_matrix @ anchor_f
        return Pose(tuple(position), orientation)

    def compute_facing_pose(
        self,
        from_thing: Union["Draft", str],
        from_dir: Dir,
        target: Union["Draft", str],
    ) -> Pose:
        """Solve the upright rotation turning a node's named side toward a
        target.

        Position is unchanged, the turn is about this draft's +z only.

        Args:
            from_thing: The mover being turned, given directly or by name.
            from_dir: Names which local direction of the mover should face
                the target; the system never guesses a front.
            target: The sub node turned toward, given directly or by name.
        """
        node = self._resolve_node_in_scope(from_thing)
        target_node = self._resolve_node_in_scope(target)
        node_pose = self._get_node_pose_in_self(node)
        target_bounds = self.get_bounds(target_node)
        aim = (
            Draft._make_array(target_bounds.min) + Draft._make_array(target_bounds.max)
        ) * 0.5 - Draft._make_array(node_pose.relative_position)
        aim[2] = 0.0
        if np.linalg.norm(aim) < 1e-9:
            raise DraftError("target is directly above or below; facing is undefined")
        current = Draft._make_rotation(node_pose.orientation) @ Draft._resolve_dir(
            from_dir
        )
        current[2] = 0.0
        if np.linalg.norm(current) < 1e-9:
            raise DraftError("from_dir points straight up or down; facing is undefined")
        yaw = math.atan2(aim[1], aim[0]) - math.atan2(current[1], current[0])
        orientation = Draft._multiply_quat(
            Draft._make_quat_about_z(yaw), Draft._get_quat_parts(node_pose.orientation)
        )
        return Pose(node_pose.relative_position, orientation)

    def compute_along_poses(
        self,
        thing: "Draft",
        thing_tangent_dir: Dir,
        path: Path,
        spacing: Optional[float] = None,
        num: Optional[int] = None,
        jitter: Optional[AlongJitter] = None,
        seed: int = 0,
    ) -> List[Pose]:
        """Solve upright poses along a path for copies of a thing.

        Exactly one of ``spacing`` or ``num`` must be given. ``min_radius``
        filleting is not applied yet; Catmull-Rom smoothing already rounds
        interior corners.

        Args:
            thing: The thing the poses place copies of along the path.
            thing_tangent_dir: Names which local direction of the thing
                rides tangent to the path.
            path: The path to ride, its waypoints in this draft's frame.
            spacing: Distance between consecutive poses along the path.
            num: Number of poses spread along the path.
            jitter: Uniform jitter ranges applied to each pose when given.
            seed: Seed for the jitter random stream.

        Returns:
            The upright poses along the path, in this draft's frame.
        """
        del thing
        if (spacing is None) == (num is None):
            raise DraftError("give exactly one of spacing or num")
        points, lengths = path.resample()
        total = float(lengths[-1])
        if num is not None:
            if num == 1:
                stations = [total * 0.5]
            else:
                stations = [total * i / (num - 1) for i in range(num)]
        else:
            stations = list(np.arange(0.0, total + 1e-9, float(spacing)))
        from ..protocol.rng import Rng

        stream = Rng(seed, "along")
        poses = []
        tangent_local = Draft._resolve_dir(thing_tangent_dir)
        for index, station in enumerate(stations):
            s = station
            wobble = stream.split(str(index))
            if jitter is not None:
                s = float(
                    np.clip(s + wobble.uniform(-jitter.along, jitter.along), 0.0, total)
                )
            position, tangent = Path.sample_at_station(points, lengths, s)
            yaw = math.atan2(tangent[1], tangent[0]) - math.atan2(
                tangent_local[1], tangent_local[0]
            )
            if jitter is not None:
                lateral = wobble.uniform(-jitter.lateral, jitter.lateral)
                normal = np.array([-tangent[1], tangent[0], 0.0])
                position = position + normal * lateral
                yaw += math.radians(wobble.uniform(-jitter.yaw_deg, jitter.yaw_deg))
            poses.append(Pose(tuple(position), Draft._make_quat_about_z(yaw)))
        return poses

    def compute_grid_poses(
        self,
        thing: "Draft",
        thing_facing_dir: Dir,
        region: AABB,
        x_spacing: Optional[float] = None,
        num_x: Optional[int] = None,
        y_spacing: Optional[float] = None,
        num_y: Optional[int] = None,
        facing_yaw: float = 0.0,
        jitter: Optional[GridJitter] = None,
        seed: int = 0,
    ) -> List[Pose]:
        """Solve one-facing lattice poses inside a containment box.

        Per axis exactly one of spacing or num must be given.

        Args:
            thing: The thing the poses place copies of in the box.
            thing_facing_dir: Names which local direction of the thing is
                aimed at ``facing_yaw``.
            region: The containment box the lattice fills, in this draft's
                frame.
            x_spacing: Spacing along x, fitting as many poses as possible
                with the block centered.
            num_x: Count along x, dividing the x extent cell-centered.
            y_spacing: Spacing along y, fitting as many poses as possible
                with the block centered.
            num_y: Count along y, dividing the y extent cell-centered.
            facing_yaw: The yaw every pose faces, in degrees in this draft's
                frame.
            jitter: Uniform jitter ranges, clamped so nothing leaves the
                box, when given.
            seed: Seed for the jitter random stream.
        """
        del thing
        lo = Draft._make_array(region.min)
        hi = Draft._make_array(region.max)
        xs = Draft._space_axis(lo[0], hi[0], x_spacing, num_x, "x")
        ys = Draft._space_axis(lo[1], hi[1], y_spacing, num_y, "y")
        facing_local = Draft._resolve_dir(thing_facing_dir)
        if np.linalg.norm(facing_local[:2]) < 1e-9:
            raise DraftError("thing_facing_dir points straight up or down")
        yaw = math.radians(facing_yaw) - math.atan2(facing_local[1], facing_local[0])
        from ..protocol.rng import Rng

        stream = Rng(seed, "grid")
        poses = []
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                px, py, cell_yaw = x, y, yaw
                if jitter is not None:
                    wobble = stream.split(f"{i},{j}")
                    px = float(
                        np.clip(x + wobble.uniform(-jitter.x, jitter.x), lo[0], hi[0])
                    )
                    py = float(
                        np.clip(y + wobble.uniform(-jitter.y, jitter.y), lo[1], hi[1])
                    )
                    cell_yaw += math.radians(
                        wobble.uniform(-jitter.yaw_deg, jitter.yaw_deg)
                    )
                poses.append(
                    Pose((px, py, float(lo[2])), Draft._make_quat_about_z(cell_yaw))
                )
        return poses

    def compute_ring_poses(
        self,
        thing: "Draft",
        thing_facing_dir: Dir,
        center,
        radius: float,
        num: int,
        start_angle: float = 0.0,
        arc: float = 360.0,
        is_facing_inside: bool = True,
        jitter: Optional[RingJitter] = None,
        seed: int = 0,
    ) -> List[Pose]:
        """Solve poses on a circle or arc, faced toward or away from its
        center.

        Args:
            thing: The thing the poses place copies of on the ring.
            thing_facing_dir: Names which local direction of the thing aims
                along the radius.
            center: Center of the circle, in this draft's frame.
            radius: Radius of the circle, in meters.
            num: Number of poses spread around the ring.
            start_angle: Angle of the first pose, in degrees.
            arc: Angular span the poses cover, in degrees; a full circle
                never duplicates the seam point.
            is_facing_inside: Face the poses toward the center when true,
                away when false.
            jitter: Uniform jitter ranges applied to each pose when given.
            seed: Seed for the jitter random stream.
        """
        del thing
        if num < 1:
            raise DraftError("num must be at least 1")
        facing_local = Draft._resolve_dir(thing_facing_dir)
        if np.linalg.norm(facing_local[:2]) < 1e-9:
            raise DraftError("thing_facing_dir points straight up or down")
        center_point = np.asarray(center, dtype=float)
        from ..protocol.rng import Rng

        stream = Rng(seed, "ring")
        poses = []
        for index in range(num):
            theta = math.radians(start_angle + index * arc / num)
            r = float(radius)
            tangential = 0.0
            yaw_extra = 0.0
            if jitter is not None:
                wobble = stream.split(str(index))
                r += wobble.uniform(-jitter.radial, jitter.radial)
                tangential = wobble.uniform(-jitter.tangential, jitter.tangential)
                yaw_extra = math.radians(
                    wobble.uniform(-jitter.yaw_deg, jitter.yaw_deg)
                )
            theta += tangential / max(r, 1e-9)
            position = center_point + np.array(
                [r * math.cos(theta), r * math.sin(theta), 0.0]
            )
            radial = (
                center_point - position if is_facing_inside else position - center_point
            )
            yaw = math.atan2(radial[1], radial[0]) - math.atan2(
                facing_local[1], facing_local[0]
            )
            poses.append(
                Pose(tuple(position), Draft._make_quat_about_z(yaw + yaw_extra))
            )
        return poses

    def compute_lean_pose(
        self,
        thing: "Draft",
        thing_base_dir: Dir,
        thing_lean_dir: Dir,
        base: Union["Draft", str],
        base_face: Dir,
        lean: Union["Draft", str],
        lean_face: Dir,
        angle: float,
        at: float = 0.0,
    ) -> LeanResult:
        """Solve the two-contact rest of a thing between a base and a lean
        support.

        The result guarantees correct touching, not that the thing stays.

        Args:
            thing: The thing rested between the two supports.
            thing_base_dir: Names which local direction of the thing rests
                on the base.
            thing_lean_dir: Names which local direction of the thing leans
                on the lean support.
            base: The base support sub node, given directly or by name.
            base_face: Which face of the base the thing rests on.
            lean: The lean support sub node, given directly or by name.
            lean_face: Which face of the lean support the thing leans on.
            angle: The tilt from the base plane, in degrees; it is the
                author's choice, not a solution.
            at: Slides the rest along the fold line.
        """
        base_node = self._resolve_node_in_scope(base)
        lean_node = self._resolve_node_in_scope(lean)
        b_center, b_normal, _bu, _bv = Draft._get_face_frame(base_node, base_face)
        l_center, l_normal, _lu, _lv = Draft._get_face_frame(lean_node, lean_face)
        base_pose = self._get_node_pose_in_self(base_node)
        lean_pose = self._get_node_pose_in_self(lean_node)
        n_b = Draft._make_rotation(base_pose.orientation) @ b_normal
        n_l = Draft._make_rotation(lean_pose.orientation) @ l_normal
        if abs(float(np.dot(n_b, n_l))) > 1e-6:
            raise DraftError("lean supports must be perpendicular in this version")
        base_dir = Draft._resolve_dir(thing_base_dir)
        lean_dir = Draft._resolve_dir(thing_lean_dir)
        stand = Draft._make_quat_between(base_dir, -n_b)
        leaned_now = Draft._make_matrix(stand) @ lean_dir
        toward = -n_l
        spin_from = leaned_now - n_b * float(np.dot(leaned_now, n_b))
        spin_to = toward - n_b * float(np.dot(toward, n_b))
        if np.linalg.norm(spin_from) < 1e-9 or np.linalg.norm(spin_to) < 1e-9:
            spin = (0.0, 0.0, 0.0, 1.0)
        else:
            spin_angle = math.atan2(
                float(np.dot(np.cross(spin_from, spin_to), n_b)),
                float(np.dot(spin_from, spin_to)),
            )
            spin = Draft._make_quat_from_axis_angle(n_b, spin_angle)
        fold_axis = np.cross(n_b, -n_l)
        tilt = Draft._make_quat_from_axis_angle(fold_axis, math.radians(90.0 - angle))
        orientation = Draft._multiply_quat(tilt, Draft._multiply_quat(spin, stand))
        corners = Draft._get_node_cover_corners(thing)
        rotated = corners @ Draft._make_matrix(orientation).T
        base_plane = float(np.dot(Draft._make_array(base_pose.apply(b_center)), n_b))
        lean_plane = float(np.dot(Draft._make_array(lean_pose.apply(l_center)), -n_l))
        heights = rotated @ n_b
        lowest = int(np.argmin(heights))
        reaches = rotated @ (-n_l)
        furthest = int(np.argmax(reaches))
        slide_up = base_plane - float(heights[lowest])
        slide_in = lean_plane - float(reaches[furthest])
        fold_dir = fold_axis / np.linalg.norm(fold_axis)
        position = n_b * slide_up + (-n_l) * slide_in + fold_dir * float(at)
        contact_base = rotated[lowest] + position
        contact_lean = rotated[furthest] + position
        return LeanResult(
            Pose(tuple(position), orientation),
            (tuple(contact_base), tuple(contact_lean)),
            float(angle),
        )

    def compute_mirrored_poses(
        self, nodes: Sequence["Draft"], plane_point, plane_normal: Dir
    ) -> List[Pose]:
        """Solve the mirrored layout of sub nodes across a plane.

        Positions reflect exactly; orientations rebuild as proper rotations,
        so an asymmetric thing stays same-handed. Geometry never mirrors
        here; a truly mirrored thing must come from its maker. Nothing
        moves: the editor's mirror verb applies the returned poses.

        Args:
            nodes: The sub nodes whose layout mirrors.
            plane_point: A point on the mirror plane, in this draft's frame.
            plane_normal: The normal of the mirror plane, in this draft's
                frame.

        Returns:
            One mirrored pose per node, in this draft's frame and in the
            given node order.
        """
        normal = Draft._resolve_dir(plane_normal)
        point = np.asarray(plane_point, dtype=float)
        results = []
        for node in nodes:
            target = self._resolve_node_in_scope(node)
            pose = self._get_node_pose_in_self(target)
            p = Draft._make_array(pose.relative_position)
            reflected = p - 2.0 * float(np.dot(p - point, normal)) * normal
            matrix = Draft._make_rotation(pose.orientation)
            x_axis = matrix[:, 0] - 2.0 * float(np.dot(matrix[:, 0], normal)) * normal
            z_axis = matrix[:, 2] - 2.0 * float(np.dot(matrix[:, 2], normal)) * normal
            z_axis = z_axis / np.linalg.norm(z_axis)
            x_axis = x_axis - z_axis * float(np.dot(x_axis, z_axis))
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            orientation = Draft._make_quat_from_matrix(
                np.column_stack([x_axis, y_axis, z_axis])
            )
            results.append(Pose(tuple(reflected), orientation))
        return results

    # ------------------------------------------------------------------
    # API category 3: modification operations
    # ------------------------------------------------------------------

    def insert(
        self, child: "Draft", pose: Pose, parent: Optional["Draft"] = None
    ) -> None:
        """Attach a detached draft at a pose in this draft's frame.

        Structural linking only, for detached trees such as the ones module
        bodies compose: no journal and no scene index exist here. A tree
        placed in the live session refuses this method, since session
        mutations go only through the editor verbs; a floor child is always
        refused, because floors carry session rules only the editor can
        promise.

        Args:
            child: The detached draft to attach.
            pose: Where in space the child lands, in this draft's frame.
            parent: The sub node to attach the child under; defaults to this
                draft itself.
        """
        if self.editor_access.is_session_node(self):
            raise DraftScopeError(
                "session mutations go through the editor; call the editor verbs"
            )
        if child.parent is not None:
            raise DraftError(f"'{child.name}' is already placed; remove it first")
        if child is self or child is self.get_root():
            raise DraftError("cannot insert a node into itself")
        if child.editor_access is not self.editor_access:
            raise DraftError(
                f"'{child.name}' belongs to a different editor; recreate it here"
            )
        if child.has_floor:
            raise DraftError(
                f"'{child.name}' carries a floor; floors enter through an editor"
            )
        anchor = self._resolve_node_in_scope(parent) if parent is not None else self
        Draft._attach_child(self, anchor, child, pose)

    # ------------------------------------------------------------------
    # lifecycle and serialization
    # ------------------------------------------------------------------

    def bake_entities(self) -> List[object]:
        """Return entity copies posed in this draft's frame, names path-prefixed.

        The editor's compile and preview end here: compile bakes the root
        and preview bakes any subtree in its own frame. Structure fixedness
        bakes as the entity's own flag or the node's effective fixedness; a
        floor anywhere but the identity pose is an authoring error caught
        here.
        """
        baked = []
        seen_names = set()
        for node in self.iterate_subtree_nodes():
            entities = [p for p in node.rscs if isinstance(p, RSC_EntityBase)]
            if not entities:
                continue
            pose = self._get_node_pose_in_self(node)
            path = Draft._make_path_name(self, node)
            for entity in entities:
                name = f"{path}/{entity.name}" if path else entity.name
                if name in seen_names:
                    raise DraftError(f"duplicate entity name '{name}' at compile")
                seen_names.add(name)
                overrides = {"name": name}
                if hasattr(entity, "origin_position"):
                    entity_pose = pose.compose(
                        Pose(entity.origin_position, entity.orientation)
                    )
                    overrides["origin_position"] = entity_pose.relative_position
                    overrides["orientation"] = entity_pose.orientation
                elif isinstance(entity, RSC_FloorBase) and not pose.is_identity():
                    raise DraftError(
                        f"floor '{entity.name}' must sit at the identity pose"
                    )
                if isinstance(entity, RSC_StructureVoxel):
                    overrides["is_fixed"] = bool(entity.is_fixed) or (
                        node.get_effective_is_fixed()
                    )
                baked.append(entity.shallow_copy(**overrides))
        return baked

    def realize(self) -> None:
        """Fill missing rscs through the cache, per unrealized identity.

        Content only: an identity node whose ``is_realized`` is false loads
        its rscs through the editor access under its own identity, the
        loaded rscs transplant onto the placed subtree in matching walk
        order, and every member marks realized, nested identities included,
        because stored entries are always complete. The scene index learns
        about realized voxels through the editor's own realize verb, which
        compile always runs first.
        """
        for node in self.iterate_subtree_nodes():
            if node.identity is None or node.is_realized:
                continue
            members = list(node.iterate_subtree_nodes())
            loaded = node.editor_access.load_rscs(node.identity)
            if len(loaded) != len(members):
                raise DraftError(
                    f"cached tree of {node.identity.module} no longer matches "
                    f"the placed subtree of '{node.name}'"
                )
            for member, rscs in zip(members, loaded):
                if not member.rscs and rscs:
                    member.rscs = list(rscs)
                    member.has_floor = any(
                        isinstance(rsc, RSC_FloorBase) for rsc in rscs
                    )
                member.is_realized = True

    def to_json(self, save_rsc, is_identity_content_saved: bool = True) -> dict:
        """Serialize the subtree into its draft json.

        Args:
            save_rsc: Stores one rsc object and returns its file index.
            is_identity_content_saved: When false, module-born content, any
                node at or under a deterministic identity, skips rsc saving,
                because the identity alone rebuilds the same bytes; content
                under a non-deterministic identity saves regardless.
        """
        return self._make_json(save_rsc, is_identity_content_saved, False)

    @staticmethod
    def from_json(
        editor_access,
        draft_json: dict,
        load_rsc=None,
        is_uid_restored: bool = False,
    ) -> "Draft":
        """Rebuild a detached subtree from its draft json.

        Args:
            editor_access: The access wall the rebuilt nodes hold.
            draft_json: The draft json to rebuild from.
            load_rsc: Loads one rsc object by its file index; without it the
                rebuilt nodes stay json-only.
            is_uid_restored: When true, every node adopts its recorded uid,
                which is what journal replay needs; the default allocates
                fresh uids, which is what a cache hit needs.
        """
        rscs: List[object] = []
        stored_rscs = draft_json["rscs"]
        if stored_rscs is not None and load_rsc is not None:
            for index in stored_rscs:
                rscs.append(load_rsc(index))
        node = Draft(
            editor_access,
            draft_json["name"],
            uid=draft_json["uid"] if is_uid_restored else None,
            boxes=[Box.from_json(box) for box in draft_json["boxes"]],
            rscs=rscs,
            requires=[
                MaterialRequire.from_json(require) for require in draft_json["requires"]
            ],
            is_fixed=draft_json["is_fixed"],
            tags=draft_json["tags"],
            annotations={
                name: Draft._make_annotation_from_json(data)
                for name, data in draft_json["annotations"].items()
            },
        )
        node.relative_pose = Pose.from_json(draft_json["pose"])
        if draft_json["identity"] is not None:
            node.identity = Identity.from_json(draft_json["identity"])
        node.has_floor = draft_json["has_floor"]
        node.is_deterministic = draft_json.get("is_deterministic", True)
        node.is_realized = draft_json["is_realized"] and (
            stored_rscs is None or (load_rsc is not None and stored_rscs != [])
        )
        for child_json in draft_json["children"]:
            child = Draft.from_json(
                editor_access, child_json, load_rsc, is_uid_restored
            )
            child.parent = node
            node.children.append(child)
        return node

    @staticmethod
    def make_sibling_name(anchor: "Draft", name: str) -> str:
        """Return the name, numbered when a sibling already carries it.

        Every attach path shares this rule, so entity path names stay
        unique below one parent.

        Args:
            anchor: The parent whose children the name must not collide
                with.
            name: The wanted name.
        """
        held = {child.name for child in anchor.children}
        if name not in held:
            return name
        counter = 2
        while f"{name}_{counter}" in held:
            counter += 1
        return f"{name}_{counter}"

    # ------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------

    def _resolve_node_in_scope(self, thing: Union["Draft", str]) -> "Draft":
        """Resolve a node argument and require it inside this draft's subtree."""
        if isinstance(thing, str):
            return self.find_node_by_name(thing)
        walker: Optional["Draft"] = thing
        while walker is not None:
            if walker is self:
                return thing
            walker = walker.parent
        raise DraftScopeError(f"'{thing.name}' is outside the subtree of '{self.name}'")

    def _get_node_pose_in_root(self, node: "Draft") -> Pose:
        """Return a node's pose in its root frame."""
        chain = []
        walker: Optional["Draft"] = node
        while walker is not None:
            chain.append(walker)
            walker = walker.parent
        pose = Pose()
        for link in reversed(chain):
            pose = pose.compose(link.relative_pose)
        return pose

    def _get_node_pose_in_self(self, node: "Draft") -> Pose:
        """Return a node's pose in this draft's frame."""
        if node is self:
            return Pose()
        return (
            self._get_node_pose_in_root(self)
            .invert()
            .compose(self._get_node_pose_in_root(node))
        )

    def _get_node_cover_boxes(
        self,
        node: "Draft",
        frame_owner: "Draft",
        exclude: Optional[Set[int]] = None,
    ) -> List[Tuple[Box, Pose]]:
        """Return the subtree's cover boxes posed in a given node's frame."""
        items = []
        for child in node.iterate_subtree_nodes():
            if not child.boxes:
                continue
            if exclude is not None and child.uid in exclude:
                continue
            pose = frame_owner._get_node_pose_in_self(child)
            items.extend((box, pose) for box in child.boxes)
        return items

    def _get_disjoint_node_cover_boxes(
        self, node_a: "Draft", node_b: "Draft"
    ) -> Tuple[List[Tuple[Box, Pose]], List[Tuple[Box, Pose]]]:
        """Return the two covers with each side's own subtree kept apart.

        When one argument nests inside the other, the outer side drops the
        inner subtree, so the answer measures the matter around it rather
        than the thing against itself.
        """
        uids_a = {n.uid for n in node_a.iterate_subtree_nodes()}
        uids_b = {n.uid for n in node_b.iterate_subtree_nodes()}
        exclude_from_a = uids_b if node_b.uid in uids_a else None
        exclude_from_b = uids_a if node_a.uid in uids_b else None
        items_a = self._get_node_cover_boxes(node_a, self, exclude=exclude_from_a)
        items_b = self._get_node_cover_boxes(node_b, self, exclude=exclude_from_b)
        return items_a, items_b

    def _query_overlaps(self, boxes, pose, include=None, exclude=None):
        """Answer an overlap query through the session or a scratch index."""
        if self.editor_access.is_session_node(self):
            return self.editor_access.query_overlaps(
                self, boxes, pose, include=include, exclude=exclude
            )
        return self._make_scratch_index().query_overlaps(
            boxes, pose, include=include, exclude=exclude
        )

    def _query_ray(self, origin, direction, distance=None, include=None, exclude=None):
        """Answer a ray query through the session or a scratch index."""
        if self.editor_access.is_session_node(self):
            return self.editor_access.query_ray(
                self,
                origin,
                direction,
                distance=distance,
                include=include,
                exclude=exclude,
            )
        return self._make_scratch_index().query_ray(
            origin, direction, distance=distance, include=include, exclude=exclude
        )

    def _query_nearby(
        self, source_boxes, source_pose, distance, num=None, include=None, exclude=None
    ):
        """Answer a range query through the session or a scratch index."""
        if self.editor_access.is_session_node(self):
            return self.editor_access.query_nearby(
                self,
                source_boxes,
                source_pose,
                distance,
                num=num,
                include=include,
                exclude=exclude,
            )
        return self._make_scratch_index().query_nearby(
            source_boxes,
            source_pose,
            distance,
            num=num,
            include=include,
            exclude=exclude,
        )

    def _compute_index_distance(self, uids_a, uids_b) -> Distance:
        """Answer a distance query through the session or a scratch index."""
        if self.editor_access.is_session_node(self):
            return self.editor_access.compute_distance(self, uids_a, uids_b)
        return self._make_scratch_index().compute_distance(uids_a, uids_b)

    def _make_scratch_index(self) -> SceneIndex:
        """Build a scene index over this draft's whole root subtree."""
        root = self.get_root()
        index = SceneIndex()
        for node in root.iterate_subtree_nodes():
            if node.boxes:
                index.insert(node.uid, node.boxes, root._get_node_pose_in_self(node))
        return index

    def _make_nodes_by_uid(self) -> Dict[int, "Draft"]:
        """Return the uid map over this draft's whole root subtree."""
        return {node.uid: node for node in self.get_root().iterate_subtree_nodes()}

    def _make_scope_uids(self) -> Optional[Set[int]]:
        """Return this draft's subtree uids to include, or None at the root."""
        if self.parent is None:
            return None
        return {node.uid for node in self.iterate_subtree_nodes()}

    def _apply_filters(self, filters: Optional[Filters]) -> Optional[Set[int]]:
        """Return the uids passing the filters, or None when unfiltered."""
        if filters is None:
            return None
        allowed = set()
        under_uids = (
            {node.uid for node in filters.under.iterate_subtree_nodes()}
            if filters.under is not None
            else None
        )
        for node in self.iterate_subtree_nodes():
            if filters.module is not None:
                full = node.identity.module if node.identity else ""
                short = full.split(".", 1)[-1] if full else ""
                if not (
                    fnmatch.fnmatch(full, filters.module)
                    or fnmatch.fnmatch(short, filters.module)
                ):
                    continue
            if filters.tags and not set(filters.tags) <= node.tags:
                continue
            if (
                filters.is_fixed is not None
                and node.get_effective_is_fixed() != filters.is_fixed
            ):
                continue
            if under_uids is not None and node.uid not in under_uids:
                continue
            if node.name in filters.exclude:
                continue
            allowed.add(node.uid)
        return allowed

    def _cast_ray_in_root(
        self, from_position, direction: Dir, distance: Optional[float]
    ) -> Optional[Tuple["Draft", float, np.ndarray]]:
        """Cast a ray given in this draft's frame through the root index."""
        self_pose = self._get_node_pose_in_root(self)
        origin = Draft._make_array(self_pose.apply(from_position))
        ray_dir = Draft._make_rotation(self_pose.orientation) @ Draft._resolve_dir(
            direction
        )
        hit = self._query_ray(
            origin, ray_dir, distance=distance, include=self._make_scope_uids()
        )
        if hit is None:
            return None
        uid, t, normal = hit
        return self._make_nodes_by_uid()[uid], t, normal

    def _find_floor_node(self) -> Optional["Draft"]:
        """Return the floor node this draft may see, or None.

        The floor's own node answers itself and a root answers its floor
        child, which is always an immediate child by the editor's rules.
        """
        if self.has_floor:
            return self
        if self.parent is None:
            for child in self.children:
                if child.has_floor:
                    return child
        return None

    def _make_json(
        self,
        save_rsc,
        is_identity_content_saved: bool,
        is_inside_identity: bool,
        is_identity_deterministic: bool = True,
    ) -> dict:
        """Serialize one node with its children into the draft json shape."""
        if self.identity is not None:
            is_identity_deterministic = self.is_deterministic
        is_module_content = is_inside_identity or self.identity is not None
        stored_rscs = None
        if self.rscs and (
            is_identity_content_saved
            or not is_module_content
            or not is_identity_deterministic
        ):
            if save_rsc is None:
                raise DraftError(
                    f"'{self.name}' carries rscs but no rsc store was given"
                )
            stored_rscs = [save_rsc(rsc) for rsc in self.rscs]
        elif self.rscs:
            stored_rscs = []
        return {
            "uid": self.uid,
            "name": self.name,
            "pose": self.relative_pose.to_json(),
            "boxes": [box.to_json() for box in self.boxes],
            "requires": [require.to_json() for require in self.requires],
            "tags": sorted(self.tags),
            "annotations": {
                name: annotation.to_json()
                for name, annotation in sorted(self.annotations.items())
            },
            "is_fixed": self.is_fixed,
            "has_floor": self.has_floor,
            "is_realized": self.is_realized,
            "identity": None if self.identity is None else self.identity.to_json(),
            "is_deterministic": self.is_deterministic,
            "rscs": stored_rscs,
            "children": [
                child._make_json(
                    save_rsc,
                    is_identity_content_saved,
                    is_module_content,
                    is_identity_deterministic,
                )
                for child in self.children
            ],
        }

    # ------------------------------------------------------------------
    # private static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_child(
        frame_owner: "Draft", anchor: "Draft", child: "Draft", pose: Pose
    ) -> Pose:
        """Link a child under an anchor at a pose in the frame owner's frame."""
        parent_pose = (
            frame_owner._get_node_pose_in_self(anchor)
            if anchor is not frame_owner
            else Pose()
        )
        relative = parent_pose.invert().compose(pose)
        child.name = Draft.make_sibling_name(anchor, child.name)
        child.parent = anchor
        child.relative_pose = relative
        anchor.children.append(child)
        return relative

    @staticmethod
    def _make_annotation_from_json(annotation_json: dict) -> Annotation:
        """Rebuild one annotation, dispatched by its json type field."""
        cls = _ANNOTATION_KINDS.get(annotation_json["type"])
        if cls is None:
            raise DraftError(f"unknown annotation type '{annotation_json['type']}'")
        return cls.from_json(annotation_json)

    @staticmethod
    def _make_disjoint_uid_sets(
        node_a: "Draft", node_b: "Draft"
    ) -> Tuple[Set[int], Set[int]]:
        """Return the two subtree uid sets with nesting kept apart."""
        uids_a = {n.uid for n in node_a.iterate_subtree_nodes()}
        uids_b = {n.uid for n in node_b.iterate_subtree_nodes()}
        if node_b.uid in uids_a:
            uids_a -= uids_b
        if node_a.uid in uids_b:
            uids_b -= uids_a
        return uids_a, uids_b

    @staticmethod
    def _make_path_name(root: "Draft", node: "Draft") -> str:
        """Return the node's path name below the root."""
        parts = []
        walker: Optional["Draft"] = node
        while walker is not None and walker is not root:
            parts.append(walker.name)
            walker = walker.parent
        return "/".join(reversed(parts))

    @staticmethod
    def _make_array(vector) -> np.ndarray:
        """Return the numpy form of a bound or sequence 3-vector."""
        if isinstance(vector, RVec3rf):
            return np.array([vector.x, vector.y, vector.z])
        return np.asarray(vector, dtype=float)

    @staticmethod
    def _make_rotation(orientation: RQuat3rf) -> np.ndarray:
        """Return the 3x3 rotation matrix of a bound quaternion."""
        return np.asarray(orientation.to_rotation_matrix().to_list(), dtype=float)

    @staticmethod
    def _resolve_dir(direction: Dir) -> np.ndarray:
        """Return the unit vector of a direction, resolving axis names."""
        if isinstance(direction, str):
            if direction not in _DIR_VECTORS:
                raise DraftError(f"unknown direction name '{direction}'")
            return np.asarray(_DIR_VECTORS[direction])
        vector = Draft._make_array(direction)
        length = float(np.linalg.norm(vector))
        if length < 1e-12:
            raise DraftError("cannot use a zero-length direction")
        return vector / length

    @staticmethod
    def _get_quat_parts(orientation) -> Tuple[float, float, float, float]:
        """Return the (x, y, z, w) parts of a bound or tuple quaternion."""
        if isinstance(orientation, RQuat3rf):
            return (
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
        return tuple(float(part) for part in orientation)

    @staticmethod
    def _make_matrix(orientation) -> np.ndarray:
        """Return the 3x3 rotation matrix of a tuple or bound quaternion."""
        x, y, z, w = Draft._get_quat_parts(orientation)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    @staticmethod
    def _multiply_quat(a, b) -> Tuple[float, float, float, float]:
        """Return ``a * b``, the rotation applying ``b`` first and ``a`` second."""
        ax, ay, az, aw = Draft._get_quat_parts(a)
        bx, by, bz, bw = Draft._get_quat_parts(b)
        return (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def _make_quat_about_z(angle_rad: float) -> Tuple[float, float, float, float]:
        """Return the rotation about +z by the given angle."""
        half = 0.5 * angle_rad
        return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))

    @staticmethod
    def _make_quat_from_axis_angle(
        axis, angle_rad: float
    ) -> Tuple[float, float, float, float]:
        """Return the rotation about a unit axis by the given angle."""
        unit = Draft._resolve_dir(axis)
        half = 0.5 * angle_rad
        s = float(np.sin(half))
        return (unit[0] * s, unit[1] * s, unit[2] * s, float(np.cos(half)))

    @staticmethod
    def _make_quat_between(from_dir, to_dir) -> Tuple[float, float, float, float]:
        """Return the smallest rotation mapping one direction onto another."""
        a = Draft._resolve_dir(from_dir)
        b = Draft._resolve_dir(to_dir)
        dot = float(np.dot(a, b))
        if dot > 1.0 - 1e-12:
            return (0.0, 0.0, 0.0, 1.0)
        if dot < -1.0 + 1e-12:
            pick = (
                np.array([1.0, 0.0, 0.0])
                if abs(a[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            side = np.cross(a, pick)
            side = side / np.linalg.norm(side)
            return (float(side[0]), float(side[1]), float(side[2]), 0.0)
        axis = np.cross(a, b)
        quat = np.array([axis[0], axis[1], axis[2], 1.0 + dot])
        quat /= np.linalg.norm(quat)
        return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    @staticmethod
    def _make_quat_from_matrix(matrix: np.ndarray) -> Tuple[float, float, float, float]:
        """Return the quaternion of a proper 3x3 rotation matrix."""
        m = np.asarray(matrix, dtype=float)
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        quat = np.array([x, y, z, w])
        quat /= np.linalg.norm(quat)
        return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    @staticmethod
    def _get_face_frame(
        node: "Draft", direction: Dir
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (center, normal, u, v) of the extremal cover plane, node local."""
        normal = Draft._resolve_dir(direction)
        corners = Draft._get_node_cover_corners(node)
        projections = corners @ normal
        top = float(projections.max())
        on_plane = corners[projections >= top - FACE_PLANE_EPSILON]
        if abs(float(normal[2])) >= 0.99:
            u = np.array([1.0, 0.0, 0.0])
        else:
            u = np.cross(np.array([0.0, 0.0, 1.0]), normal)
            u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        u_values = on_plane @ u
        v_values = on_plane @ v
        center = (
            normal * top
            + u * float((u_values.min() + u_values.max()) * 0.5)
            + v * float((v_values.min() + v_values.max()) * 0.5)
        )
        return center, normal, u, v

    @staticmethod
    def _get_node_cover_corners(node: "Draft") -> np.ndarray:
        """Return the node-local corners of every cover box in the subtree."""
        corners = []
        for child in node.iterate_subtree_nodes():
            if not child.boxes:
                continue
            pose = node._get_node_pose_in_self(child)
            matrix = Draft._make_rotation(pose.orientation)
            offset = Draft._make_array(pose.relative_position)
            for box in child.boxes:
                corners.append(box.get_corners() @ matrix.T + offset)
        if not corners:
            raise DraftError(f"'{node.name}' has no cover boxes")
        return np.vstack(corners)

    @staticmethod
    def _compute_axis_gap(
        a_items: List[Tuple[Box, Pose]],
        b_items: List[Tuple[Box, Pose]],
        direction: str,
    ) -> float:
        """Return the axis-interval gap between two covers."""
        axis = {"x": 0, "y": 1, "z": 2}[direction[1]]
        sign = 1.0 if direction[0] == "+" else -1.0
        others = [i for i in range(3) if i != axis]
        best = np.inf
        for a_box, a_pose in a_items:
            a_lo, a_hi = a_box.get_aabb_in_root(a_pose)
            for b_box, b_pose in b_items:
                b_lo, b_hi = b_box.get_aabb_in_root(b_pose)
                has_shadow = all(
                    a_lo[i] < b_hi[i] - CONTACT_OVERLAP_EPSILON
                    and b_lo[i] < a_hi[i] - CONTACT_OVERLAP_EPSILON
                    for i in others
                )
                if not has_shadow:
                    continue
                if sign > 0:
                    a_interval = (a_lo[axis], a_hi[axis])
                    b_interval = (b_lo[axis], b_hi[axis])
                else:
                    a_interval = (-a_hi[axis], -a_lo[axis])
                    b_interval = (-b_hi[axis], -b_lo[axis])
                if b_interval[1] <= a_interval[0]:
                    continue
                best = min(best, b_interval[0] - a_interval[1])
        return float(best)

    @staticmethod
    def _compute_sweep_gap(
        a_items: List[Tuple[Box, Pose]],
        b_items: List[Tuple[Box, Pose]],
        direction,
    ) -> float:
        """Return the swept first-contact travel along a free direction.

        A SAT-style sweep on box pairs: both boxes project on the face
        normals and edge cross products, and the first contact comes from
        interval arithmetic. Exact and deterministic; a pair already
        overlapping reports zero.
        """
        unit = Draft._resolve_dir(direction)
        best = np.inf
        for a_box, a_pose in a_items:
            for b_box, b_pose in b_items:
                travel = Draft._sweep_box_pair(a_box, a_pose, b_box, b_pose, unit)
                if travel is not None:
                    best = min(best, travel)
        return float(best)

    @staticmethod
    def _sweep_box_pair(
        a_box: Box, a_pose: Pose, b_box: Box, b_pose: Pose, unit: np.ndarray
    ) -> Optional[float]:
        """Return the first-contact travel of one moving box pair, or None."""
        a_rotation = Draft._make_rotation(a_pose.orientation)
        b_rotation = Draft._make_rotation(b_pose.orientation)
        a_min = Draft._make_array(a_box.min)
        a_max = Draft._make_array(a_box.max)
        b_min = Draft._make_array(b_box.min)
        b_max = Draft._make_array(b_box.max)
        a_half = (a_max - a_min) * 0.5
        b_half = (b_max - b_min) * 0.5
        a_center = Draft._make_array(a_pose.apply((a_min + a_max) * 0.5))
        b_center = Draft._make_array(b_pose.apply((b_min + b_max) * 0.5))
        axes = [a_rotation[:, i] for i in range(3)]
        axes += [b_rotation[:, i] for i in range(3)]
        for i in range(3):
            for j in range(3):
                cross = np.cross(a_rotation[:, i], b_rotation[:, j])
                length = np.linalg.norm(cross)
                if length > 1e-9:
                    axes.append(cross / length)
        t_enter = -np.inf
        t_exit = np.inf
        for axis in axes:
            r_a = float(np.abs(a_rotation.T @ axis) @ a_half)
            r_b = float(np.abs(b_rotation.T @ axis) @ b_half)
            d = float(np.dot(b_center - a_center, axis))
            s = float(np.dot(unit, axis))
            radius = r_a + r_b
            if abs(s) < 1e-12:
                if abs(d) > radius:
                    return None
                continue
            t0 = (d - radius) / s
            t1 = (d + radius) / s
            if t0 > t1:
                t0, t1 = t1, t0
            t_enter = max(t_enter, t0)
            t_exit = min(t_exit, t1)
            if t_enter > t_exit:
                return None
        if t_exit < 0.0:
            return None
        return max(t_enter, 0.0)

    @staticmethod
    def _space_axis(
        low: float,
        high: float,
        spacing: Optional[float],
        num: Optional[int],
        label: str,
    ) -> List[float]:
        """Return lattice stations along one axis of a containment box."""
        if (spacing is None) == (num is None):
            raise DraftError(f"give exactly one of {label}_spacing or num_{label}")
        extent = high - low
        if num is not None:
            step = extent / num
            return [low + (i + 0.5) * step for i in range(num)]
        count = max(1, int(math.floor(extent / spacing)) + 1)
        while (count - 1) * spacing > extent:
            count -= 1
        margin = (extent - (count - 1) * spacing) * 0.5
        return [low + margin + i * spacing for i in range(count)]

    @staticmethod
    def _sample_floor_flat(
        entity: RSC_FloorFlat, local: np.ndarray
    ) -> Optional[Tuple[float, np.ndarray, float]]:
        """Sample a flat floor at a local point."""
        del local
        return float(entity.height), np.array([0.0, 0.0, 1.0]), 0.0

    @staticmethod
    def _sample_floor_array(
        entity: RSC_FloorArray, local: np.ndarray
    ) -> Optional[Tuple[float, np.ndarray, float]]:
        """Bilinearly sample a height-field floor at a local point."""
        x_size = float(entity.x_size)
        y_size = float(entity.y_size)
        x_values = int(entity.x_values)
        y_values = int(entity.y_values)
        grid = np.asarray(entity.height, dtype=float).reshape(y_values, x_values)
        u = (float(local[0]) + x_size * 0.5) / x_size
        v = (float(local[1]) + y_size * 0.5) / y_size
        if entity.is_x_repeat:
            u = u % 1.0
        elif u < 0.0 or u > 1.0:
            return None
        if entity.is_y_repeat:
            v = v % 1.0
        elif v < 0.0 or v > 1.0:
            return None
        fx = u * (x_values - 1)
        fy = v * (y_values - 1)
        ix = int(np.clip(math.floor(fx), 0, x_values - 2))
        iy = int(np.clip(math.floor(fy), 0, y_values - 2))
        bx = fx - ix
        by = fy - iy
        height = (
            grid[iy, ix] * (1 - bx) * (1 - by)
            + grid[iy, ix + 1] * bx * (1 - by)
            + grid[iy + 1, ix] * (1 - bx) * by
            + grid[iy + 1, ix + 1] * bx * by
        )
        step_x = x_size / (x_values - 1)
        step_y = y_size / (y_values - 1)
        dzdx = (grid[iy, min(ix + 1, x_values - 1)] - grid[iy, ix]) / step_x
        dzdy = (grid[min(iy + 1, y_values - 1), ix] - grid[iy, ix]) / step_y
        normal = np.array([-dzdx, -dzdy, 1.0])
        normal = normal / np.linalg.norm(normal)
        slope = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
        return float(height), normal, slope
