# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The Editor: one session, one tree, one door to the engine.

An editor owns everything a session needs: an exclusively locked session
folder, the write-through journal the session rebuilds from, the scene
index, the declared material table, a handle on the shared realization
cache, and the editor-born root draft. Users and agents hold the editor,
whose verbs are the only door to session mutations; drafts and module
bodies hold only its access wall, which carries the build abilities and
the gated session reads.
"""

import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import requests

from rise_origin import (
    RS_WorldTaskDataFailed,
    RSC_CosmicBackgroundBase,
    RSC_EntityBase,
    RSC_EssenceBase,
    RSC_FloorBase,
    RSC_MaterialPhysicalPropertyRigid,
    RSC_MaterialPhysicalPropertySoft,
    RSC_ResourceBase,
    RSC_SimulationVisionDenoiseNone,
    RSC_StructureVoxel,
    RSE_WorldTaskState,
)
from rise_origin.rsc.entity import CosmicBackgroundHosekWilkie, FloorFlat
from rise_origin.rsc.essence import (
    Material,
    MaterialOpticalPropertySimpleMicrofacet,
    MaterialPhysicalPropertyFixed,
)
from rise_origin.rsc.resource import TextureGrid
from rise_origin.rsc.simulation import Simulation, VisionDenoiseSVGF
from rise_origin.rsc.world import World

from ..constants import (
    CONTACT_OVERLAP_EPSILON,
    PREVIEW_GRID_LINE_COLOR,
    PREVIEW_GRID_LINE_WIDTH,
    PREVIEW_GRID_SPACING,
    PREVIEW_GROUND_FILL_COLOR,
    PREVIEW_SKY_GROUND_COLOR,
    PREVIEW_SKY_INTENSITY,
    PREVIEW_SKY_SUN_COLOR,
    RIGID_CONSTRAINT_DT_PER_VOXEL_SIZE,
    SOFT_INTEGRATION_FREQUENCY_MULTIPLIER,
)
from ..data.identity import Identity
from ..data.material import MaterialRequire
from ..data.module_reference import ModuleReference
from ..data.pose import Pose
from ..data.result import Overlap, PlacementReport
from ..data.update import (
    CompileMemo,
    CompileMemoEntity,
    WorldUpdate,
    WorldUpdateInstruction,
)
from ..draft.draft import Draft
from ..protocol.module import ModuleError, ModuleSpec, registry
from ..protocol.logger import Logger
from ..protocol.types import Settings
from ..protocol.worker_pool import WorkerPool
from .cache import RealizationCache
from .folder_manager import FolderManager
from .index import SceneIndex
from .journal import EditorJournal
from .material import (
    EditorMaterialDataTable,
    EditorMaterialError,
    EditorMaterialRequireTable,
)

__all__ = ["EditorError", "EditorAccess", "Editor"]


class EditorError(RuntimeError):
    """An editor operation that cannot be applied."""


class EditorAccess:
    """The only thing drafts and module bodies hold into an editor.

    The access owns nothing and carries two categories of abilities:

    - Editor general access: build abilities every holder may use.
    - Editor scene query: session reads that take the asking node and
      refuse one outside the live session tree, so module bodies, which
      only hold detached drafts, never see the scene and stay pure.

    Session mutations are not here; they go through the editor's verbs.
    """

    def __init__(self, editor: "Editor") -> None:
        self._editor = editor

    # ------------------------------------------------------------------
    # Editor general access
    # ------------------------------------------------------------------

    def allocate_uid(self) -> int:
        """Return a fresh node uid of the owning editor's session."""
        return self._editor._allocate_uid()

    def create(self, name: str, settings: Optional[dict] = None) -> Draft:
        """Create a module draft through the owning editor.

        Args:
            name: The module reference to create, carrying an optional
                ``@`` version specifier.
            settings: The settings for this module; None fills the schema
                defaults in.
        """
        return self._editor.create(name, settings)

    @property
    def pool(self) -> "WorkerPool":
        """The owning editor's shared worker pool."""
        return self._editor.pool

    @property
    def logger(self) -> Logger:
        """The logger of the module under creation.

        Named by the four-field module name while a module body runs,
        so every line carries the module as its prefix; outside any
        build it is the editor's own logger.
        """
        if self._editor._building:
            return Logger(self._editor._building[-1])
        return self._editor.logger

    def load_rscs(self, identity: Identity) -> List[List[object]]:
        """Load the cached rscs of an identity, one list per subtree node.

        Args:
            identity: The identity whose cached rscs load.
        """
        return self._editor._load_rscs(identity)

    # ------------------------------------------------------------------
    # Editor scene query
    # ------------------------------------------------------------------

    def is_session_node(self, node: Draft) -> bool:
        """Return whether a node sits in the live session tree.

        Args:
            node: The draft whose membership is asked.
        """
        return node.get_root() is self._editor.root

    def query_overlaps(self, node: Draft, boxes, pose, include=None, exclude=None):
        """Answer a session overlap query for a placed node.

        Args:
            node: The asking draft; must sit in the live session tree.
            boxes: The queried cover, in the frame ``pose`` maps to root.
            pose: Root-frame pose of the queried cover.
            include: Keep only entries with these uids when given.
            exclude: Drop entries with these uids.
        """
        self._check_session_node(node)
        return self._editor._index.query_overlaps(
            boxes, pose, include=include, exclude=exclude
        )

    def query_ray(
        self, node: Draft, origin, direction, distance=None, include=None, exclude=None
    ):
        """Answer a session ray query for a placed node.

        Args:
            node: The asking draft; must sit in the live session tree.
            origin: Ray start point in the root frame.
            direction: Unit ray direction in the root frame.
            distance: Farthest accepted travel; None means unbounded.
            include: Keep only entries with these uids when given.
            exclude: Drop entries with these uids.
        """
        self._check_session_node(node)
        return self._editor._index.query_ray(
            origin, direction, distance=distance, include=include, exclude=exclude
        )

    def query_nearby(
        self,
        node: Draft,
        source_boxes,
        source_pose,
        distance,
        num=None,
        include=None,
        exclude=None,
    ):
        """Answer a session range query for a placed node.

        Args:
            node: The asking draft; must sit in the live session tree.
            source_boxes: The cover measured from, in the frame
                ``source_pose`` maps to root; empty means a point source.
            source_pose: Root-frame pose of the source cover.
            distance: Keep only entries within this range, in meters.
            num: Keep only the nearest so many entries when given.
            include: Keep only entries with these uids when given.
            exclude: Drop entries with these uids.
        """
        self._check_session_node(node)
        return self._editor._index.query_nearby(
            source_boxes,
            source_pose,
            distance,
            num=num,
            include=include,
            exclude=exclude,
        )

    def compute_distance(self, node: Draft, uids_a, uids_b):
        """Answer a session distance query for a placed node.

        Args:
            node: The asking draft; must sit in the live session tree.
            uids_a: The first side's registered uids.
            uids_b: The other side's registered uids.
        """
        self._check_session_node(node)
        return self._editor._index.compute_distance(uids_a, uids_b)

    def _check_session_node(self, node: Draft) -> None:
        """Refuse a session read from a node outside the live session tree."""
        if node.get_root() is not self._editor.root:
            raise EditorError(
                f"'{node.name}' is not placed in the session; detached drafts "
                "answer queries from their own tree"
            )


class Editor:
    """The working context of one editing session.

    Construction allocates (or adopts) a session folder, locks it
    exclusively for this editor's lifetime, and starts the write-through
    journal, so the session survives the process: :meth:`open` rebuilds
    the live editor from the folder alone and :meth:`fork` branches it
    into a new folder. The editor verbs are the only door to session
    mutations; drafts hold the access wall and nothing else.

    Args:
        name: World and root node name.
        title: Free description of what the session is about; stored in
            the journal head, not required to be unique.
        material_requires: Requires declared up front; more may follow
            through :meth:`require` until the simulation freezes.
        session_dir: Session folder to use; None allocates a fresh
            uuid-named folder under the sessions home.
        cache: Realization cache to build through; None opens the default
            shared cache.
    """

    def __init__(
        self,
        name: str,
        *,
        title: str = "",
        material_requires: Sequence[MaterialRequire] = (),
        session_dir: Optional[str] = None,
        cache: Optional[RealizationCache] = None,
        worker_processes: Optional[int] = None,
        log_level: Optional[str] = "info",
    ) -> None:
        if session_dir is None:
            session_dir = FolderManager.make_session_folder()
        lock = FolderManager.lock_session_folder(session_dir)
        try:
            journal = EditorJournal(
                session_dir,
                head={
                    "name": name,
                    "title": title,
                    "material_requires": [
                        require.to_json() for require in material_requires
                    ],
                },
            )
        except BaseException:
            lock.release()
            raise
        self._build(
            name,
            title,
            material_requires,
            session_dir,
            cache,
            lock,
            journal,
            worker_processes=worker_processes,
            log_level=log_level,
        )

    @classmethod
    def open(cls, session_dir: str, *, cache: Optional[RealizationCache] = None):
        """Rebuild the live editor of an existing session folder.

        The journal alone rebuilds the session: the head restores the
        name, the title, and the declared requires; every surviving op
        replays into nodes carrying their recorded uids; module-born
        content stays unloaded until realize. The folder locks exclusively
        exactly as a fresh session does.

        Args:
            session_dir: The session folder to rebuild the editor from.
            cache: Realization cache to build through; None opens the
                default shared cache.
        """
        lock = FolderManager.lock_session_folder(session_dir)
        try:
            journal, head = EditorJournal.resume(session_dir)
        except BaseException:
            lock.release()
            raise
        editor = cls.__new__(cls)
        editor._build(
            head["name"],
            head.get("title", ""),
            [
                MaterialRequire.from_json(require)
                for require in head.get("material_requires", [])
            ],
            session_dir,
            cache,
            lock,
            journal,
        )
        editor._replay(journal.get_entries())
        return editor

    def __enter__(self) -> "Editor":
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    @property
    def root(self) -> Draft:
        """The editor-born root draft of this session."""
        return self._root

    def close(self) -> None:
        """Release the session folder for the next editor.

        The journal is write-through, so nothing needs flushing; the
        held resources are the exclusive session lock and the worker
        pool. A closed editor must not be used again; open the folder
        instead.
        """
        self.pool.close()
        self._session_lock.release()
        self.logger.info("session '%s' closed", self.name)
        if self._log_handler is not None:
            logging.getLogger("rise_origin").removeHandler(self._log_handler)
            self._log_handler = None

    def require(self, *material_requires: MaterialRequire) -> None:
        """Add to the session's declared material requires.

        Declarations add freely until the simulation freezes and
        persist in the journal, so a reopened session keeps them; a frozen
        session refuses new declarations because the frozen step was
        resolved over the declared set.

        Args:
            material_requires: The requires to add to the declared set.
        """
        if not material_requires:
            return
        self._require_table.add_requires(material_requires)
        self._journal.append_require(
            [require.to_json() for require in material_requires]
        )

    def set_simulation_rsc_frozen(
        self, is_frozen: bool, base: Optional[Simulation] = None
    ) -> None:
        """Freeze or unfreeze the session's simulation configuration.

        Once frozen, the simulation configuration is no longer changeable
        and the declared material requires are pinned; a later create
        whose content breaks the pin faults at creation instead of at
        compile. Unfreezing lifts both. Frozen state is session state,
        not journal content: a reopened session starts unfrozen.

        Args:
            is_frozen: Whether the session freezes.
            base: The base the pinned configuration updates; freeze only.
        """
        self._frozen_simulation = None
        self._require_table.unfreeze()
        if is_frozen:
            self._frozen_simulation = self.get_simulation_rsc(base)
            self._require_table.freeze()

    def check_placement_validity(self, child: Draft, pose: Pose) -> PlacementReport:
        """Report every cover overlap a detached draft would make at a pose.

        The pre-insert twin of :meth:`Draft.check_placement_validity`: the
        child stays detached and nothing mutates; its cover poses against
        the scene at the given root-frame pose and every overlap
        classifies exactly as it would after an insert. Static with
        static is a legal embedding, anything dynamic is a violation,
        and a mixed box defers to the voxel gate at compile.

        Args:
            child: The detached draft whose placement is asked.
            pose: The root-frame pose the child would be inserted at.
        """
        if child is self._root or child.parent is not None:
            raise EditorError(
                f"'{child.name}' is already placed; ask the root's own "
                "check_placement_validity instead"
            )
        if child.editor_access is not self._access:
            raise EditorError(
                f"'{child.name}' belongs to a different editor; recreate it here"
            )
        violations, embeddings, pending = [], [], []
        for own in child.iterate_subtree_nodes():
            if not own.boxes:
                continue
            own_pose = pose.compose(child.get_pose(own))
            hits = self._index.query_overlaps(own.boxes, own_pose)
            for hit in hits:
                other = self._nodes_by_uid[hit.uid]
                own_box = own.boxes[hit.own_box_index]
                other_box = other.boxes[hit.box_index]
                other_pose = self._root.get_pose(other)
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

    def create(self, name: str, settings: Optional[dict] = None) -> Draft:
        """Create the draft of a module, lazily through the realization cache.

        The full sequence, in order:

        1. The reference parses and the module resolves: the text behind
           ``@`` is a version specifier and the newest matching registered
           version wins.
        2. Settings validate against the schema. Note ``seed`` is a setting
           every module carries; pass it like any other setting, and an
           omitted seed defaults to zero. The module, its version, and
           the canonical settings json form the identity.
        3. The cache resolves the identity. On a hit the draft rebuilds
           from the stored json alone: boxes, requires, tags, annotations,
           and identity, with no rsc objects and no blob file reads. On a
           miss the body runs with this editor's access and the validated
           settings; nested creates that hit the cache fill from their own
           entries, the result is honesty checked (material references
           covered by requires or in-tree essences, body voxel sizes among
           the declared sizes, constraints inside their own structure),
           stored complete, and returned fully populated. A body whose own
           creates cycle back into this module is refused.
        4. In a frozen session, the draft is held against the pinned
           simulation: a require the frozen table does not cover, or a
           material demanding a smaller step than the frozen dt, raises
           here at creation rather than at compile.

        What each cache outcome means:

        1. Hit: no rscs load at all. The draft comes back declared-only,
           boxes and structure without content, and stays so until
           realize.
        2. Miss: the module function itself is called. The returned tree
           is fully populated, nested content included.
        3. Both compile identically, because compile realizes first.
        4. Rsc files evicted between the hit and realization rebuild
           through the body; a deterministic module rebuilds the same
           bytes, a non-deterministic one a new valid artifact. A
           deterministic body may not create a non-deterministic module;
           that create refuses and asks the outer module to declare
           is_deterministic=False.

        Nesting composes the same way. A house module's body creates a
        wall assembly module, itself nesting wall and light leaves, and
        adds a hand-made window leaf::

            @cr.module(name="acme.town.building.house", version="1.0.0")
            def house(editor_access, settings):
                \"\"\"A house.\"\"\"
                root = cr.Draft(editor_access, "house")
                assembly = editor_access.create("acme.town.part.wall_assembly")
                root.insert(assembly, cr.Pose())
                window = cr.Draft(editor_access, "window", rscs=[...])
                root.insert(window, cr.Pose((1.0, 0.0, 2.0)))
                return root

        A miss of the house runs this body: when the nested assembly
        create hits the cache, its subtree fills from the assembly's own
        entry before the house entry stores, so the house entry is
        complete. A later house hit loads nothing; one realize fills
        wall, light, and window from the house entry alone.

        Single create, place one thing::

            maple = editor.create("rise_origin.nature.tree.maple",
                                  {"height": 8.0})
            editor.insert(maple, cr.Pose((3.0, 0.0, 0.0)))

        Repeated create is the way to duplicate: the same settings hit the
        cache, and a varied seed gives each instance its own look::

            poses = editor.root.compute_along_poses(path, num=20)
            for seed, pose in enumerate(poses):
                maple = editor.create("rise_origin.nature.tree.maple",
                                      {"seed": seed})
                editor.insert(maple, pose)

        Args:
            name: The module reference to create, carrying an optional
                ``@`` version specifier.
            settings: The settings for this module; None fills the schema
                defaults in.
        """
        spec = registry.resolve(ModuleReference.from_text(name))
        if spec.name in self._creating:
            raise ModuleError(f"dependency cycle through '{spec.name}'")
        if (
            not spec.is_deterministic
            and self._creating_specs
            and self._creating_specs[-1].is_deterministic
        ):
            raise ModuleError(
                f"'{self._creating_specs[-1].name}' is declared deterministic "
                f"but creates non-deterministic '{spec.name}'; declare "
                "is_deterministic=False on it"
            )
        self._creating.add(spec.name)
        self._creating_specs.append(spec)
        self._building.append(spec.name)
        try:
            validated = spec.schema.validate(settings)
            identity = Identity(spec.name, spec.version, validated.to_json())
            hit = self.cache.resolve(identity, need="json")
            if hit is not None:
                Logger(spec.name).debug("cache hit")
                draft = Draft.from_json(self._access, hit.draft_json)
            else:
                draft = self._build_module_draft(spec, identity, validated)
        finally:
            self._building.pop()
            self._creating_specs.pop()
            self._creating.discard(spec.name)
        self._check_frozen_creation(draft)
        return draft

    def insert(self, child: Draft, pose: Pose, parent: Optional[Draft] = None) -> None:
        """Insert a detached draft into the session at a root-frame pose.

        The insert journals with its full declared content, registers the
        subtree in the scene index, and uniquifies sibling names. Only
        this path may parent a floor: one per session, an immediate child
        of the root, at the identity pose. An unrealized non-deterministic
        node realizes here first, so the journal records the exact built
        content a rebuild could not reproduce.

        Note: no overlap check runs here, deliberately. Cover boxes
        over-approximate and static content legally embeds, so a hard
        gate would refuse correct scenes; ask the editor's
        check_placement_validity before placing or the root's after, and
        whether things work eventually depends on the simulation
        stability.

        Args:
            child: The detached draft to insert.
            pose: The root-frame pose to place the child at.
            parent: The sub node to insert under, naming where in the
                structure; None inserts under the root.
        """
        if child.parent is not None:
            raise EditorError(f"'{child.name}' is already placed; remove it first")
        if child is self._root:
            raise EditorError("cannot insert the root into itself")
        if child.editor_access is not self._access:
            raise EditorError(
                f"'{child.name}' belongs to a different editor; recreate it here"
            )
        for node in child.iterate_subtree_nodes():
            if (
                node.identity is not None
                and not node.is_deterministic
                and not node.is_realized
            ):
                node.realize()
        anchor = self._root if parent is None else self._resolve_session_node(parent)
        if child.has_floor:
            if anchor is not self._root:
                raise EditorError("a floor may only be an immediate child of the root")
            if self._floor_uid is not None:
                raise EditorError("the session already has a floor")
            if not pose.is_identity():
                raise EditorError("a floor must be inserted at the identity pose")
        relative = self._root.get_pose(anchor).invert().compose(pose)
        child.name = Draft.make_sibling_name(anchor, child.name)
        child.parent = anchor
        child.relative_pose = relative
        anchor.children.append(child)
        self._register_subtree(child)
        if child.has_floor:
            self._floor_uid = child.uid
        self._journal.append(
            "insert",
            {
                "parent": anchor.uid,
                "pose": relative.to_json(),
                "draft": child.to_json(
                    self._journal.save_rsc, is_identity_content_saved=False
                ),
            },
            {"remove": child.uid},
            created=tuple(node.uid for node in child.iterate_subtree_nodes()),
            touched=(anchor.uid,),
        )

    def remove(self, node: Union[Draft, str]) -> Draft:
        """Detach a sub node's subtree from the session and return it.

        The journal records the removal with enough content to reverse it.

        Args:
            node: The sub node to detach, by draft or by name.

        Returns:
            The detached subtree, still fully usable and re-insertable.
        """
        target = self._resolve_session_node(node)
        if target.parent is None:
            raise EditorError(f"'{target.name}' is not placed")
        parent = target.parent
        args = {
            "node": target.uid,
            "parent": parent.uid,
            "pose": target.relative_pose.to_json(),
            "draft": target.to_json(
                self._journal.save_rsc, is_identity_content_saved=False
            ),
        }
        self._unregister_subtree(target)
        self._journal.append(
            "remove",
            args,
            {"reinsert": args},
            touched=tuple(
                [parent.uid] + [n.uid for n in target.iterate_subtree_nodes()]
            ),
        )
        parent.children.remove(target)
        target.parent = None
        return target

    def move(self, node: Union[Draft, str], pose: Pose) -> None:
        """Set a sub node's frame to an absolute root-frame pose.

        Never incremental and never re-parenting; floors refuse to move.

        Args:
            node: The sub node to move, by draft or by name.
            pose: The absolute root-frame pose to set the node's frame to;
                absolute, because absolute poses are what solvers emit,
                replay cleanly, and give the journal a true inverse.
        """
        target = self._resolve_session_node(node)
        if target is self._root:
            raise EditorError("cannot move the root")
        if target.has_floor:
            raise EditorError("a floor never moves; remove and reinsert terrain")
        previous = target.relative_pose
        relative = self._root.get_pose(target.parent).invert().compose(pose)
        target.relative_pose = relative
        self._refit_subtree(target)
        self._journal.append(
            "move",
            {"node": target.uid, "pose": relative.to_json()},
            {"node": target.uid, "pose": previous.to_json()},
            touched=(target.uid,),
        )

    def mirror(self, nodes: Sequence[Draft], plane_point, plane_normal) -> List[Pose]:
        """Mirror the layout of sub nodes across a root-frame plane.

        The poses come from :meth:`Draft.compute_mirrored_poses`: positions
        reflect exactly and orientations rebuild as proper rotations;
        geometry itself never mirrors.

        Args:
            nodes: The sub nodes whose layout mirrors.
            plane_point: A point on the mirror plane, in root frame.
            plane_normal: The normal of the mirror plane, in root frame.

        Returns:
            The new root-frame pose of each mirrored node, in order.
        """
        poses = self._root.compute_mirrored_poses(nodes, plane_point, plane_normal)
        targets = [self._resolve_session_node(node) for node in nodes]
        previous = [target.relative_pose for target in targets]
        for target, pose in zip(targets, poses):
            parent_pose = (
                self._root.get_pose(target.parent)
                if target.parent is not None
                else Pose()
            )
            target.relative_pose = parent_pose.invert().compose(pose)
            self._refit_subtree(target)
        self._journal.append(
            "mirror",
            {
                "nodes": [target.uid for target in targets],
                "poses": [target.relative_pose.to_json() for target in targets],
            },
            {
                "nodes": [target.uid for target in targets],
                "poses": [pose.to_json() for pose in previous],
            },
            touched=tuple(target.uid for target in targets),
        )
        return poses

    def undo(self, num: int = 1) -> None:
        """Reverse the last journaled ops, newest first.

        Args:
            num: How many ops to reverse.
        """
        for entry in self._journal.undo(num):
            self._apply_inverse(entry)

    def retract(self, entry_id: int) -> None:
        """Reverse one older journal entry when nothing later depends on it.

        Args:
            entry_id: The id of the journal entry to reverse.
        """
        self._apply_inverse(self._journal.retract(entry_id))

    def realize(self) -> None:
        """Fill missing module-born rscs and refresh the index's voxel level.

        Content fills through :meth:`Draft.realize`, driven by each node's
        ``is_realized``: a realized node is skipped, and a node that came
        declared-only from the cache is unrealized, so its rscs load and
        fill here. The block sweep then attaches every realized
        structure's voxels to its scene index entry, so exact-voxel
        probes see what the cache loaded.
        """
        self._root.realize()
        for node in self._root.iterate_subtree_nodes():
            entities = [rsc for rsc in node.rscs if isinstance(rsc, RSC_EntityBase)]
            if entities and node.boxes:
                self._index.realize(node.uid, entities)

    def get_simulation_rsc(
        self,
        base: Optional[Simulation] = None,
        node: Optional[Draft] = None,
    ) -> Simulation:
        """Resolve the simulation configuration of the session.

        Returns a deep copy of the base with the time step minimized over
        the base's own step and every stability rule the content demands:
        the soft rule ``h / (multiplier * wave speed)`` with the pressure
        wave speed of each soft material, over the declared requires and
        the in-tree soft essences at their voxel sizes, and the rigid
        constraint rule over the smallest voxel size when anything rigid
        is present. Sub-update steps fill to the
        resolved step where unset; every other field keeps the base. A
        frozen session answers a copy of its pin. Override in a derived
        editor to change the policy.

        Args:
            base: Simulation to update a copy of; None starts from the
                engine defaults.
            node: Resolve over this subtree only, as preview does; None
                resolves the whole session with its declared requires.
        """
        if self._frozen_simulation is not None and base is None and node is None:
            return self._frozen_simulation.deep_copy()
        simulation = base.deep_copy() if base is not None else Simulation()
        target = self._root if node is None else node
        candidates = self._make_dt_candidates(target)
        if simulation.time.dt is not None:
            candidates.append(float(simulation.time.dt))
        if candidates:
            dt = min(candidates)
            simulation.time.dt = dt
            if simulation.time.dt_rigidbody_contact_and_constraint_update is None:
                simulation.time.dt_rigidbody_contact_and_constraint_update = dt
            if simulation.time.dt_contact_update is None:
                simulation.time.dt_contact_update = dt
        return simulation

    def compile(self, memo: Optional[CompileMemo] = None) -> WorldUpdate:
        """Compile the session into the update a consumer is missing.

        The steps in order: realize the tree through the cache, resolve
        materials into an ordered emission, run the embedding gate over
        pending mixed overlaps, bake entities into root-frame copies, and
        diff against the memo. Unseen essences, resources, and entities
        add; an entity whose content changed deletes and re-adds; an
        entity that only moved emits a move; a vanished entity deletes.
        Deletes come first, then adds with essences and resources ahead of
        entities, then moves.

        Args:
            memo: The record of what a consumer already holds; it advances
                to what was emitted. None compiles against the editor's own
                memo, so successive compiles yield increments, while a fresh
                CompileMemo yields the full emission.
        """
        memo = self._memo if memo is None else memo
        self.realize()
        data = self._resolve_materials(self._root)
        essences = data.get_essences()
        resources = data.get_resources()
        self._check_embedding_gate(data.get_material_physical_kinds())
        current: Dict[str, Tuple[object, str, Pose]] = {}
        for entity in self._root.bake_entities():
            fingerprint, pose = Editor._make_entity_fingerprint(entity)
            current[entity.name] = (entity, fingerprint, pose)
        instructions = []
        for name, held in memo.entities.items():
            if name not in current or current[name][1] != held.fingerprint:
                instructions.append(WorldUpdateInstruction("delete", entity_name=name))
        for essence in essences:
            if essence.name not in memo.essence_names:
                instructions.append(WorldUpdateInstruction("add", rsc=essence))
        for resource in resources:
            if resource.name not in memo.resource_names:
                instructions.append(WorldUpdateInstruction("add", rsc=resource))
        for name, (entity, fingerprint, pose) in current.items():
            held = memo.entities.get(name)
            if held is None or held.fingerprint != fingerprint:
                instructions.append(WorldUpdateInstruction("add", rsc=entity))
            elif held.pose.to_json() != pose.to_json():
                instructions.append(
                    WorldUpdateInstruction("move", entity_name=name, pose=pose)
                )
        memo.entities = {
            name: CompileMemoEntity(fingerprint, pose)
            for name, (_entity, fingerprint, pose) in current.items()
        }
        memo.essence_names |= {essence.name for essence in essences}
        memo.resource_names |= {resource.name for resource in resources}
        memo.journal_position = self._journal.get_cursor()
        return WorldUpdate(tuple(instructions))

    def save(self, path: str) -> None:
        """Compile everything fresh and save the whole world to one file.

        The world carries the session name, the resolved simulation, and
        every essence, resource, and entity of the tree.

        Args:
            path: The file path to save the world to.
        """
        update = self.compile(memo=CompileMemo())
        essences, resources, entities = [], [], []
        for instruction in update.instructions:
            if instruction.op != "add":
                continue
            if isinstance(instruction.rsc, RSC_EssenceBase):
                essences.append(instruction.rsc)
            elif isinstance(instruction.rsc, RSC_ResourceBase):
                resources.append(instruction.rsc)
            else:
                entities.append(instruction.rsc)
        world = World(
            self.name,
            simulation=self.get_simulation_rsc(),
            essences=essences,
            resources=resources,
            entities=entities,
        )
        world.save(path)

    def preview(
        self,
        node: Optional[Draft] = None,
        *,
        device: Optional[str] = None,
        is_interactive: bool = True,
        duration: Optional[float] = None,
        viewer_constructor_kwargs: Optional[dict] = None,
    ) -> None:
        """Show one subtree in a viewer over a scratch engine instance.

        Realizes and compiles the subtree alone into a temporary world
        with its resolved simulation and materials and runs it on a
        separate engine instance under a viewer; this editor keeps running
        unaffected. An interactive preview opens paused for inspection and
        the space bar toggles the world between paused and running; a
        headless preview plays its duration, since a paused world would
        never end. A detached draft previews standing in its own frame, so
        a freshly created thing shows at the origin. Content missing a
        cosmic background gets a default Hosek-Wilkie sky, and content
        missing a floor gets a grid ground plane, so a bare draft reads;
        content bringing its own keeps it. A resolved simulation carrying
        no denoiser gets SVGF, so a preview stands clean at the few
        samples per pixel a live view affords; one naming its own denoiser
        keeps it. A side thread asks the live
        handle for the world's real voxel numbers and prints them once,
        so every preview reports what the engine actually allocated.

        Args:
            node: The sub node or detached draft to show; None shows the
                whole session.
            device: Compute device such as ``"cuda:0"``; None takes the
                engine default.
            is_interactive: Open the live window and block until it
                closes; False runs headless instead.
            duration: Simulated seconds of a headless run; interactive
                runs ignore it.
            viewer_constructor_kwargs: Additional keyword arguments passed
                through to the ``Viewer`` constructor; None passes none.
        """
        target = self._root if node is None else node
        self.logger.info("preview: realizing '%s'", target.name)
        target.realize()
        data = self._resolve_materials(target)
        simulation = self.get_simulation_rsc(node=target)
        simulation.runtime.is_paused_on_start = is_interactive
        if isinstance(simulation.vision.denoise, RSC_SimulationVisionDenoiseNone):
            simulation.vision.denoise = VisionDenoiseSVGF()
        self.logger.info("preview: compiling world")
        entities = target.bake_entities()
        extra_entities, extra_essences, extra_resources = (
            Editor._make_preview_environment(entities)
        )
        world = World(
            target.name,
            simulation=simulation,
            essences=data.get_essences() + extra_essences,
            resources=data.get_resources() + extra_resources,
            entities=entities + extra_entities,
        )
        from rise_origin.runtime.viewer import Viewer

        viewer_kwargs = dict(viewer_constructor_kwargs or {})
        if is_interactive and "hud" not in viewer_kwargs:
            viewer_kwargs["hud"] = Editor._make_preview_pause_hud()
        viewer = Viewer(
            f"{self.name}_preview",
            instance=(device, world),
            **viewer_kwargs,
        )
        stop = threading.Event()
        probe = threading.Thread(
            target=Editor._print_preview_voxel_count,
            args=(viewer, stop),
            daemon=True,
        )
        probe.start()
        self.logger.info(
            "preview: engine starting on %s (%d entities)",
            device or "default device",
            len(entities) + len(extra_entities),
        )
        try:
            if is_interactive:
                viewer.run_interactive()
            else:
                viewer.run_headless(duration)
                viewer.wait()
        finally:
            stop.set()
            probe.join()
            self.logger.info("preview closed")

    def apply_world_update(
        self,
        update: WorldUpdate,
        target,
        *,
        token: Optional[str] = None,
        timeout_s: float = 60.0,
    ) -> None:
        """Apply a compiled update to a running world.

        Instructions apply in their compiled order: deletes, then adds
        with essences and resources ahead of entities, then moves. A move
        re-adds the found entity at its new pose and keeps its frozen
        state, since a running world has no pose mutator. A world handle
        applies everything inside one modification transaction; the MCP
        road applies tool call by tool call, so a running world may step
        between instructions. A paused world holds every task pending
        until it resumes, which surfaces here as a timeout.

        Args:
            update: The world update to apply, as compile returned it.
            target: A live world handle, or the MCP address of a running
                engine instance, such as ``"http://127.0.0.1:8765/mcp"``.
            token: The bearer token of the MCP interface; None sends
                none.
            timeout_s: Seconds to wait on each task before giving up.
        """
        if isinstance(target, str):
            self._apply_update_to_mcp(update, target, token, timeout_s)
        else:
            self._apply_update_to_handle(update, target, timeout_s)

    def fork(self, title: Optional[str] = None) -> "Editor":
        """Branch this session into a fresh folder and return its editor.

        The journal and its blobs copy into a new session folder and
        replay into a live editor, so both histories stay linear: the fork
        deviates freely and this session is untouched. Uids replay to
        their recorded values, so both editors name the shared past the
        same way.

        Args:
            title: The fork's session title; None keeps this session's.
        """
        new_dir = FolderManager.make_session_folder()
        source = FolderManager.get_session_journal_path(self.session_dir)
        with open(source, "r", encoding="utf-8") as reader:
            lines = reader.readlines()
        if title is not None and lines:
            head_line = json.loads(lines[0])
            head_line["head"]["title"] = title
            lines[0] = json.dumps(head_line, sort_keys=True) + "\n"
        target = FolderManager.get_session_journal_path(new_dir)
        with open(target, "w", encoding="utf-8") as writer:
            writer.writelines(lines)
        blobs = FolderManager.get_session_blobs_path(self.session_dir)
        if os.path.isdir(blobs):
            shutil.copytree(blobs, FolderManager.get_session_blobs_path(new_dir))
        return Editor.open(new_dir, cache=self.cache)

    def _build(
        self,
        name: str,
        title: str,
        material_requires: Sequence[MaterialRequire],
        session_dir: str,
        cache: Optional[RealizationCache],
        lock,
        journal: EditorJournal,
        worker_processes: Optional[int] = None,
        log_level: Optional[str] = "info",
    ) -> None:
        """Wire the organs of one session; the ctor and open both end here."""
        self.name = name
        self.title = title
        self.session_dir = session_dir
        self.cache = cache if cache is not None else RealizationCache()
        self.pool = WorkerPool(worker_processes)
        self.logger = Logger("rise_origin.create.editor")
        self._building: List[str] = []
        self._log_handler = Editor._ensure_log_handler(log_level)
        self._session_lock = lock
        self._journal = journal
        self._index = SceneIndex()
        self._require_table = EditorMaterialRequireTable(material_requires)
        self._access = EditorAccess(self)
        self._next_uid = 0
        self._root = Draft(self._access, name)
        self._nodes_by_uid: Dict[int, Draft] = {self._root.uid: self._root}
        self._floor_uid: Optional[int] = None
        self._memo = CompileMemo()
        self._frozen_simulation: Optional[Simulation] = None
        self._creating: Set[str] = set()
        self._creating_specs: List[ModuleSpec] = []

    def _replay(self, entries) -> None:
        """Apply journal entries onto the fresh root with their recorded uids.

        Entries replay structurally through a recorded-uid table that every
        replayed insert extends, the uid counter resumes past the largest
        recorded uid, and the finished tree registers into the session
        organs in one sweep.
        """
        nodes_by_uid = {self._root.uid: self._root}
        for entry in entries:
            self._replay_entry(entry, nodes_by_uid)
        self._next_uid = max(nodes_by_uid) + 1
        self._register_subtree(self._root)
        for child in self._root.children:
            if child.has_floor:
                self._floor_uid = child.uid

    def _replay_entry(self, entry, nodes_by_uid: Dict[int, Draft]) -> None:
        """Replay one journal entry onto the rebuilt tree.

        ``nodes_by_uid`` maps recorded uids to rebuilt nodes; the caller
        seeds it with the root and every replayed insert extends it, since
        later entries reference the nodes earlier entries created.
        """
        if entry.op == "insert":
            node = Draft.from_json(
                self._access,
                entry.args["draft"],
                self._journal.load_rsc,
                is_uid_restored=True,
            )
            parent = nodes_by_uid[entry.args["parent"]]
            node.parent = parent
            node.relative_pose = Pose.from_json(entry.args["pose"])
            parent.children.append(node)
            for child in node.iterate_subtree_nodes():
                nodes_by_uid[child.uid] = child
        elif entry.op == "remove":
            node = nodes_by_uid[entry.args["node"]]
            node.parent.children.remove(node)
            node.parent = None
        elif entry.op == "move":
            nodes_by_uid[entry.args["node"]].relative_pose = Pose.from_json(
                entry.args["pose"]
            )
        elif entry.op == "mirror":
            for uid, pose in zip(entry.args["nodes"], entry.args["poses"]):
                nodes_by_uid[uid].relative_pose = Pose.from_json(pose)

    def _apply_inverse(self, entry) -> None:
        """Apply one journal entry's inverse onto the live tree."""
        if entry.op == "insert":
            node = self._nodes_by_uid[entry.inverse["remove"]]
            self._unregister_subtree(node)
            node.parent.children.remove(node)
            node.parent = None
        elif entry.op == "remove":
            args = entry.inverse["reinsert"]
            parent = self._nodes_by_uid[args["parent"]]
            child = Draft.from_json(
                self._access,
                args["draft"],
                self._journal.load_rsc,
                is_uid_restored=True,
            )
            child.parent = parent
            child.relative_pose = Pose.from_json(args["pose"])
            parent.children.append(child)
            self._register_subtree(child)
            if child.has_floor:
                self._floor_uid = child.uid
        elif entry.op == "move":
            node = self._nodes_by_uid[entry.inverse["node"]]
            node.relative_pose = Pose.from_json(entry.inverse["pose"])
            self._refit_subtree(node)
        elif entry.op == "mirror":
            for uid, pose in zip(entry.inverse["nodes"], entry.inverse["poses"]):
                node = self._nodes_by_uid[uid]
                node.relative_pose = Pose.from_json(pose)
                self._refit_subtree(node)

    def _resolve_session_node(self, thing: Union[Draft, str]) -> Draft:
        """Resolve a node argument and require it inside the session tree."""
        if isinstance(thing, str):
            return self._root.find_node_by_name(thing)
        if thing.get_root() is not self._root:
            raise EditorError(f"'{thing.name}' is not placed in this session")
        return thing

    def _register_subtree(self, node: Draft) -> None:
        """Register a placed subtree into the uid map and the scene index."""
        for child in node.iterate_subtree_nodes():
            self._nodes_by_uid[child.uid] = child
            if child.boxes:
                self._index.insert(child.uid, child.boxes, self._root.get_pose(child))
                entities = [
                    rsc for rsc in child.rscs if isinstance(rsc, RSC_EntityBase)
                ]
                if entities:
                    self._index.realize(child.uid, entities)

    def _unregister_subtree(self, node: Draft) -> None:
        """Drop a placed subtree from the uid map and the scene index."""
        for child in node.iterate_subtree_nodes():
            if child.boxes:
                self._index.remove(child.uid)
            self._nodes_by_uid.pop(child.uid, None)
            if self._floor_uid == child.uid:
                self._floor_uid = None

    def _refit_subtree(self, node: Draft) -> None:
        """Update a moved subtree's poses in the scene index."""
        for child in node.iterate_subtree_nodes():
            if child.boxes:
                self._index.refit(child.uid, self._root.get_pose(child))

    def _allocate_uid(self) -> int:
        """Return the next uid of this session."""
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def _build_module_draft(
        self, spec: ModuleSpec, identity: Identity, validated: Settings
    ) -> Draft:
        """Run a module body on a cache miss, check it, store it, return it.

        The draft realizes before the check and the store: when the body
        created a nested module and got a cache hit, that subtree is still
        empty here, so realize loads its rscs from the nested module's own
        cache entry. Every stored entry then holds full content, and a
        miss always returns a fully populated tree.
        """
        logger = Logger(spec.name)
        logger.info("building (seed %s)", validated["seed"])
        started = time.monotonic()
        draft = spec.fn(self._access, validated)
        if not isinstance(draft, Draft):
            raise ModuleError(f"{spec.name} returned {type(draft).__name__}")
        if draft.parent is not None:
            raise ModuleError(f"{spec.name} returned a placed Draft")
        build_ms = (time.monotonic() - started) * 1000.0
        logger.info("built in %.1fs", build_ms / 1000.0)
        draft.identity = identity
        draft.is_deterministic = spec.is_deterministic
        draft.realize()
        Editor._check_build_honesty(spec, draft, validated)
        collected: Dict[int, object] = {}

        def store_rsc(rsc: object) -> int:
            index = len(collected)
            collected[index] = rsc
            return index

        draft_json = draft.to_json(store_rsc, is_identity_content_saved=True)
        self.cache.store(identity, draft_json, collected, build_ms)
        return draft

    def _check_frozen_creation(self, draft: Draft) -> None:
        """Hold a created draft against the pinned simulation, or pass."""
        if self._frozen_simulation is None:
            return
        uncovered = [
            require
            for node in draft.iterate_subtree_nodes()
            for require in node.requires
            if not self._require_table.has_require(require)
        ]
        if uncovered:
            listing = ", ".join(
                "("
                + require.get_world_material_reference()
                + ", "
                + ", ".join(f"{size:g} m" for size in require.voxel_size)
                + ")"
                for require in uncovered
            )
            raise EditorError(
                f"the simulation is frozen and '{draft.name}' requires "
                "undeclared material and voxel size pairs: " + listing
            )
        dt = self._frozen_simulation.time.dt
        if dt is None:
            return
        candidates = self._make_dt_candidates(draft)
        if candidates and min(candidates) < float(dt) - 1e-12:
            raise EditorError(
                f"'{draft.name}' needs a step of {min(candidates):g} s below "
                f"the frozen {float(dt):g} s"
            )

    def _resolve_materials(self, node: Draft) -> EditorMaterialDataTable:
        """Gather and satisfy every material of one tree.

        The tree's own essences and resources register first. Requires
        then group by module, and a module whose required materials are
        all registered in-tree asks for nothing. Everything else forms one
        constraint set: the version texts of the requires plus the
        concrete versions of placed module drafts. The registry pins one
        version per module over that set, dependencies included, so pair
        and optical-only modules enter the world through plain dependency
        declarations. Every pinned module builds once, placed modules
        stand as placed, and each required material must end registered
        under its world name.

        Args:
            node: The tree resolved: the root at compile, any subtree at
                preview.

        Returns:
            The data table compile emits from.
        """
        data = EditorMaterialDataTable()
        Editor._register_tree_data(data, node)
        constraints: Dict[str, List[Optional[str]]] = {}
        wanted_by_module: Dict[str, Set[str]] = {}
        for name, group in sorted(Editor._gather_requires(node).items()):
            wanted = {require.get_world_material_reference() for require in group}
            if all(data.has_material(reference) for reference in wanted):
                continue
            texts = constraints.setdefault(name, [])
            for require in group:
                if require.module_reference.version_text not in texts:
                    texts.append(require.module_reference.version_text)
            wanted_by_module[name] = wanted
        placed = set()
        for child in node.iterate_subtree_nodes():
            if child.identity is None:
                continue
            reference = child.identity.get_reference()
            placed.add(reference.get_name())
            texts = constraints.setdefault(reference.get_name(), [])
            if reference.version_text not in texts:
                texts.append(reference.version_text)
        if not constraints:
            return data
        try:
            pins = registry.resolve_pins(constraints)
        except ModuleError as error:
            raise EditorMaterialError(str(error)) from error
        for name in sorted(pins):
            if name in placed:
                continue
            draft = self._create_realized(f"{name}@{pins[name].version}")
            Editor._register_tree_data(data, draft)
        for name in sorted(wanted_by_module):
            for reference in sorted(wanted_by_module[name]):
                if not data.has_material(reference):
                    raise EditorMaterialError(
                        f"material module '{name}' carries no Material "
                        f"named '{reference}'"
                    )
        return data

    def _create_realized(self, name: str) -> Draft:
        """Create a module draft and realize it, for material resolution."""
        draft = self.create(name)
        draft.realize()
        return draft

    def _load_rscs(self, identity: Identity) -> List[List[object]]:
        """Return the stored rscs of an identity, one list per subtree node.

        Lists follow the cached tree's depth-first walk. An entry whose
        rsc files were evicted rebuilds through the module body and stores
        again; a deterministic module rebuilds the same bytes, and a
        non-deterministic one rebuilds a new valid artifact that no
        session depends on, because the journal records placed content.
        """
        hit = self.cache.resolve(identity, need="rscs")
        if hit is None or hit.rscs is None:
            return self._rebuild_rscs(identity)
        return Editor._get_node_rscs_from_json(hit.draft_json, hit.rscs)

    def _rebuild_rscs(self, identity: Identity) -> List[List[object]]:
        """Rebuild an identity's rscs through its module body."""
        spec = registry.resolve(identity.get_reference())
        validated = spec.schema.validate(json.loads(identity.settings_json))
        draft = self._build_module_draft(spec, identity, validated)
        return [list(node.rscs) for node in draft.iterate_subtree_nodes()]

    def _make_dt_candidates(self, node: Draft) -> List[float]:
        """Return every stability step the content in reach demands.

        Requires contribute through their own material's property
        essences at their declared voxel sizes, read from the module
        version the world pins; in-tree soft essences contribute at
        their own node's smallest body size; anything rigid contributes
        the linear constraint rule at the smallest size in reach.
        """
        candidates: List[float] = []
        smallest: Optional[float] = None
        is_rigid_present = False
        requires = [
            require
            for child in node.iterate_subtree_nodes()
            for require in child.requires
        ]
        if node is self._root:
            requires.extend(self._require_table.get_requires())
        pins = self._make_dt_pins(requires, node)
        seen: Set[Tuple[str, Tuple[float, ...]]] = set()
        for require in requires:
            world_reference = require.get_world_material_reference()
            key = (world_reference, require.voxel_size)
            if key in seen:
                continue
            seen.add(key)
            for size in require.voxel_size:
                smallest = size if smallest is None else min(smallest, size)
            essences = self._get_material_module_essences(require, pins)
            if essences is None:
                continue
            for essence in essences:
                if (
                    isinstance(essence, RSC_MaterialPhysicalPropertyRigid)
                    and essence.material_reference == world_reference
                ):
                    is_rigid_present = True
                if (
                    isinstance(essence, RSC_MaterialPhysicalPropertySoft)
                    and essence.material_reference == world_reference
                ):
                    modulus = float(essence.elastic_modulus)
                    density = float(essence.density)
                    if modulus > 0 and density > 0:
                        for size in require.voxel_size:
                            candidates.append(Editor._make_soft_step(size, essence))
        for child in node.iterate_subtree_nodes():
            sizes = [
                float(body.voxel_size)
                for entity in child.rscs
                if isinstance(entity, RSC_StructureVoxel)
                for body in entity.bodies
            ]
            if sizes:
                smallest = min(sizes) if smallest is None else min(smallest, min(sizes))
            for essence in child.rscs:
                if isinstance(essence, RSC_MaterialPhysicalPropertyRigid):
                    is_rigid_present = True
                if isinstance(essence, RSC_MaterialPhysicalPropertySoft) and sizes:
                    modulus = float(essence.elastic_modulus)
                    density = float(essence.density)
                    if modulus > 0 and density > 0:
                        candidates.append(Editor._make_soft_step(min(sizes), essence))
        if is_rigid_present and smallest is not None:
            candidates.append(RIGID_CONSTRAINT_DT_PER_VOXEL_SIZE * smallest)
        return candidates

    @staticmethod
    def _make_soft_step(voxel_size: float, essence) -> float:
        """Return the stable step of one soft essence at a voxel size.

        The bound follows the engine's softbody task: the voxel size over
        the pressure wave speed, divided by the multiplier. A degenerate
        Poisson ratio falls back to the plain sqrt(E / density) speed.
        """
        modulus = float(essence.elastic_modulus)
        density = float(essence.density)
        poisson = float(essence.poissons_ratio)
        if 0.0 <= poisson < 0.5:
            factor = (1.0 - poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        else:
            factor = 1.0
        wave_speed = math.sqrt(modulus * factor / density)
        return voxel_size / (SOFT_INTEGRATION_FREQUENCY_MULTIPLIER * wave_speed)

    def _make_dt_pins(
        self, requires: Sequence[MaterialRequire], node: Draft
    ) -> Dict[str, ModuleSpec]:
        """Pin module versions for dt the way compile pins them.

        The constraint set joins the requires' version texts with the
        concrete versions of the tree's placed module drafts, so the dt
        rules read the same module versions compile will emit. Modules
        that are missing or match no version drop out silently, and a
        version conflict answers no pins at all: both fault at compile
        with full context, and dt just reads no module essences here.

        Args:
            requires: The requires in reach of the dt resolution.
            node: The tree whose placed module drafts join the
                constraints.
        """
        constraints: Dict[str, List[Optional[str]]] = {}
        for require in requires:
            texts = constraints.setdefault(require.module_reference.get_name(), [])
            if require.module_reference.version_text not in texts:
                texts.append(require.module_reference.version_text)
        for child in node.iterate_subtree_nodes():
            if child.identity is None:
                continue
            reference = child.identity.get_reference()
            texts = constraints.setdefault(reference.get_name(), [])
            if reference.version_text not in texts:
                texts.append(reference.version_text)
        resolvable = {}
        for name, texts in constraints.items():
            try:
                for text in texts:
                    registry.resolve(
                        ModuleReference.from_text(
                            name if text is None else f"{name}@{text}"
                        )
                    )
            except KeyError:
                continue
            resolvable[name] = texts
        try:
            return registry.resolve_pins(resolvable)
        except ModuleError:
            return {}

    def _get_material_module_essences(
        self, require: MaterialRequire, pins: Dict[str, ModuleSpec]
    ) -> Optional[List[object]]:
        """Return a require's module essences at the world pin, or None.

        Absence is not an error here; unresolved materials fault at
        compile with the full missing list.

        Args:
            require: The require whose module essences are read.
            pins: The pinned specs of the dt resolution.
        """
        spec = pins.get(require.module_reference.get_name())
        if spec is None:
            return None
        draft = self._create_realized(f"{spec.name}@{spec.version}")
        essences = []
        for node in draft.iterate_subtree_nodes():
            essences.extend(
                rsc for rsc in node.rscs if isinstance(rsc, RSC_EssenceBase)
            )
        return essences

    def _check_embedding_gate(self, physical_kinds: Dict[str, Set[str]]) -> None:
        """Run the voxel gate over every pending mixed overlap, once.

        A voxel violates when it co-occupies a cell with truly dynamic
        matter, or is itself dynamic and co-occupies anything; static with
        static passes as a legal embedding. Work is bounded to the mixed
        boxes' own voxels.
        """
        problems = []
        for node in self._root.iterate_subtree_nodes():
            mixed = [box for box in node.boxes if box.material_class == "mixed"]
            if not mixed:
                continue
            pose = self._root.get_pose(node)
            subtree_uids = {n.uid for n in node.iterate_subtree_nodes()}
            for box in mixed:
                hits = self._index.query_overlaps([box], pose, exclude=subtree_uids)
                candidates = {hit.uid for hit in hits}
                if not candidates:
                    continue
                box_lo, box_hi = box.get_aabb_in_root(pose)
                for leaf in node.iterate_subtree_nodes():
                    is_leaf_fixed = leaf.get_effective_is_fixed()
                    for block in self._index.get_blocks(leaf.uid):
                        positions = block.get_occupied_positions()
                        keep = np.all(positions >= box_lo, axis=1) & np.all(
                            positions <= box_hi, axis=1
                        )
                        if not keep.any():
                            continue
                        sids = block.get_material_sid()[
                            tuple(block.get_occupied_indices()[keep].T)
                        ]
                        references = list(block.structure.material_references)
                        is_block_fixed = bool(block.structure.is_fixed) or is_leaf_fixed
                        for position, sid in zip(positions[keep], sids):
                            is_own_dynamic = Editor._check_dynamic(
                                references[int(sid)],
                                is_block_fixed,
                                physical_kinds,
                            )
                            for cell in self._index.get_cell_hits_at_point(position):
                                if (
                                    cell.uid in subtree_uids
                                    or cell.uid not in candidates
                                ):
                                    continue
                                other = self._nodes_by_uid[cell.uid]
                                is_other_dynamic = Editor._check_dynamic(
                                    cell.material_reference,
                                    cell.is_fixed_structure
                                    or other.get_effective_is_fixed(),
                                    physical_kinds,
                                )
                                if is_own_dynamic or is_other_dynamic:
                                    problems.append(
                                        f"'{node.name}' voxel at "
                                        f"{tuple(position)} co-occupies with "
                                        f"'{other.name}'"
                                    )
        if problems:
            raise EditorError("embedding gate failed:\n" + "\n".join(problems[:20]))

    def _apply_update_to_handle(
        self, update: WorldUpdate, handle, timeout_s: float
    ) -> None:
        """Apply an update through a live world handle, atomically.

        Moved entities read and re-pose first, since a synchronous read
        inside an open modification transaction deadlocks against the
        world loop. Every mutation then submits inside one transaction;
        the tasks are awaited together, and any failure raises with the
        engine's error text.
        """
        moves: Dict[str, Tuple[object, bool]] = {}
        for instruction in update.instructions:
            if instruction.op != "move":
                continue
            entity = handle.find_entity(instruction.entity_name).value.deep_copy()
            is_frozen = handle.is_entity_frozen(instruction.entity_name)
            entity.origin_position = instruction.pose.relative_position
            entity.orientation = instruction.pose.orientation
            moves[instruction.entity_name] = (entity, is_frozen)
        tasks: List[Tuple[str, object]] = []
        handle.begin_modification_transaction()
        try:
            for instruction in update.instructions:
                if instruction.op == "add":
                    rsc = instruction.rsc
                    if isinstance(rsc, RSC_EssenceBase):
                        tasks.append(("add_essence", handle.add_essence(rsc)))
                    elif isinstance(rsc, RSC_ResourceBase):
                        tasks.append(("add_resource", handle.add_resource(rsc)))
                    else:
                        tasks.append(("add_entity", handle.add_entity(rsc)))
                elif instruction.op == "delete":
                    tasks.append(
                        ("remove_entity", handle.remove_entity(instruction.entity_name))
                    )
                else:
                    entity, is_frozen = moves[instruction.entity_name]
                    tasks.append(
                        ("remove_entity", handle.remove_entity(instruction.entity_name))
                    )
                    tasks.append(("add_entity", handle.add_entity(entity)))
                    if is_frozen:
                        tasks.append(
                            (
                                "freeze_entity",
                                handle.freeze_entity(instruction.entity_name),
                            )
                        )
        finally:
            handle.end_modification_transaction()
        for name, task in tasks:
            state = task.wait_for(timeout_s)
            if state == RSE_WorldTaskState.RSE_WORLD_TASK_PENDING:
                raise EditorError(
                    f"'{name}' stayed pending past {timeout_s:g} s; a paused "
                    "world applies tasks only after it resumes"
                )
            data = task.get()
            if isinstance(data, RS_WorldTaskDataFailed):
                raise EditorError(f"'{name}' failed: {data.error}")

    def _apply_update_to_mcp(
        self,
        update: WorldUpdate,
        address: str,
        token: Optional[str],
        timeout_s: float,
    ) -> None:
        """Apply an update through a world's MCP interface, tool by tool.

        Each instruction becomes one built-in MCP tool call, so this road
        works against any running instance without server-side setup; the
        world may step between instructions, since the interface offers
        no cross-call transaction.
        """
        for instruction in update.instructions:
            if instruction.op == "add":
                rsc = instruction.rsc
                if isinstance(rsc, RSC_EssenceBase):
                    tool = "add_essence"
                elif isinstance(rsc, RSC_ResourceBase):
                    tool = "add_resource"
                else:
                    tool = "add_entity"
                Editor._call_mcp_tool(
                    address,
                    token,
                    tool,
                    {"rsct": Editor._make_rsct_payload(rsc)},
                    timeout_s,
                )
            elif instruction.op == "delete":
                Editor._call_mcp_tool(
                    address,
                    token,
                    "remove_entity",
                    {"name": instruction.entity_name},
                    timeout_s,
                )
            else:
                name = instruction.entity_name
                found = Editor._call_mcp_tool(
                    address, token, "find_entity", {"name": name}, timeout_s
                )
                is_frozen = Editor._call_mcp_tool(
                    address, token, "is_entity_frozen", {"name": name}, timeout_s
                )
                position = instruction.pose.relative_position
                orientation = instruction.pose.orientation
                found["data"]["origin_position"] = [
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                ]
                found["data"]["orientation"] = [
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                ]
                Editor._call_mcp_tool(
                    address, token, "remove_entity", {"name": name}, timeout_s
                )
                Editor._call_mcp_tool(
                    address, token, "add_entity", {"rsct": found}, timeout_s
                )
                if is_frozen is True:
                    Editor._call_mcp_tool(
                        address, token, "freeze_entity", {"name": name}, timeout_s
                    )

    @staticmethod
    def _gather_requires(node: Draft) -> Dict[str, List[MaterialRequire]]:
        """Return one tree's requires grouped by module name.

        One module builds once per compile, so its requires resolve
        together; groups keep declaration order.
        """
        groups: Dict[str, List[MaterialRequire]] = {}
        for child in node.iterate_subtree_nodes():
            for require in child.requires:
                groups.setdefault(require.module_reference.get_name(), []).append(
                    require
                )
        return groups

    @staticmethod
    def _register_tree_data(data: EditorMaterialDataTable, node: Draft) -> None:
        """Register one tree's essences and resources into a data table."""
        for child in node.iterate_subtree_nodes():
            for rsc in child.rscs:
                if isinstance(rsc, RSC_EssenceBase):
                    data.add_essence(rsc)
                elif isinstance(rsc, RSC_ResourceBase):
                    data.add_resource(rsc)

    @staticmethod
    def _check_build_honesty(
        spec: ModuleSpec, draft: Draft, settings: Settings
    ) -> None:
        """Verify a built draft against its declared requires and settings.

        Every structure's material references must come from Material
        essences the tree carries or from its declared requires; every
        body's voxel size must appear among the declared require sizes or
        the module's voxel_size setting, unless the structure references
        in-tree materials only; every constraint must reference bodies of
        its own structure.
        """
        from rise_origin import RSC_Material

        provided = set()
        declared_names = set()
        declared_sizes = set()
        for node in draft.iterate_subtree_nodes():
            for require in node.requires:
                declared_names.add(require.get_world_material_reference())
                for size in require.voxel_size:
                    declared_sizes.add(MaterialRequire.round_voxel_size(size))
            for rsc in node.rscs:
                if isinstance(rsc, RSC_Material):
                    provided.add(rsc.name)
        if "voxel_size" in spec.schema.fields:
            value = settings["voxel_size"]
            sizes = (value,) if isinstance(value, (int, float)) else tuple(value)
            declared_sizes |= {MaterialRequire.round_voxel_size(size) for size in sizes}
        for node in draft.iterate_subtree_nodes():
            for entity in node.rscs:
                if not isinstance(entity, RSC_StructureVoxel):
                    continue
                references = list(entity.material_references)
                for reference in references:
                    if reference not in provided and reference not in declared_names:
                        raise ModuleError(
                            f"{spec.name} references material '{reference}' "
                            "it neither requires nor provides"
                        )
                is_private = all(reference in provided for reference in references)
                body_count = len(entity.bodies)
                for body in entity.bodies:
                    size = MaterialRequire.round_voxel_size(float(body.voxel_size))
                    if size not in declared_sizes and not is_private:
                        raise ModuleError(
                            f"{spec.name} builds voxels at {size:g} m without "
                            "declaring that size"
                        )
                for constraint in entity.constraints:
                    if (
                        int(constraint.a_body_sid) >= body_count
                        or int(constraint.b_body_sid) >= body_count
                    ):
                        raise ModuleError(
                            f"a constraint in '{entity.name}' references a "
                            "body outside its own structure"
                        )

    @staticmethod
    def _check_dynamic(
        material_reference: str,
        is_fixed_structure: bool,
        physical_kinds: Dict[str, Set[str]],
    ) -> bool:
        """Return whether a voxel is truly dynamic matter."""
        if is_fixed_structure:
            return False
        kinds = physical_kinds.get(material_reference, set())
        return "fixed" not in kinds

    @staticmethod
    def _get_node_rscs_from_json(
        draft_json: dict, rscs: Dict[int, object]
    ) -> List[List[object]]:
        """Return loaded rscs grouped per node in the json's walk order."""
        lists = []
        stack = [draft_json]
        while stack:
            current = stack.pop()
            indices = current["rscs"] or []
            lists.append([rscs[index] for index in indices])
            stack.extend(reversed(current["children"]))
        return lists

    @staticmethod
    def _make_entity_fingerprint(entity: object) -> Tuple[str, Pose]:
        """Return an entity's pose-free content hash with its baked pose.

        Posed entities hash a copy standing at the identity pose, so a
        pure move keeps its fingerprint and diffs into a move instruction
        rather than a delete and re-add.
        """
        if hasattr(entity, "origin_position"):
            pose = Pose(entity.origin_position, entity.orientation)
            neutral = entity.shallow_copy(
                origin_position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)
            )
        else:
            pose = Pose()
            neutral = entity
        handle, path = tempfile.mkstemp(suffix=".rscb")
        os.close(handle)
        try:
            neutral.save(path)
            with open(path, "rb") as saved:
                digest = hashlib.sha256(saved.read()).hexdigest()
        finally:
            os.remove(path)
        return digest, pose

    @staticmethod
    def _ensure_log_handler(log_level: Optional[str]):
        """Make sure the session console handler exists, or leave logging alone.

        Attaches one compact handler to the ``rise_origin`` logger when
        the level asks for one and no real handler sits there yet, so
        plain scripts read the build story while sessions under a
        configured logging setup stay untouched.

        Args:
            log_level: The level name to log at; None attaches nothing.
        """
        if log_level is None:
            return None
        root = logging.getLogger("rise_origin")
        root.setLevel(getattr(logging, log_level.upper()))
        for handler in root.handlers:
            if not isinstance(handler, logging.NullHandler):
                return None
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root.addHandler(handler)
        return handler

    @staticmethod
    def _print_preview_voxel_count(viewer, stop: threading.Event) -> None:
        """Poll the preview handle and print the world's real voxel numbers.

        The viewer blocks its caller, so this runs on a side thread. It
        waits for the handle to boot, reads the static and dynamic voxel
        numbers the executor publishes on its control steps, prints them
        once, and exits; closing the preview ends the wait.

        Args:
            viewer: The preview viewer owning the world handle.
            stop: Set when the preview closes, ending the wait.
        """
        while not stop.is_set():
            handle = viewer.handle
            if handle is not None and (handle.is_running() or handle.is_paused()):
                static_num = handle.get_info("world_static_voxel_num")
                dynamic_num = handle.get_info("world_dynamic_voxel_num")
                if static_num or dynamic_num:
                    Logger("rise_origin.create.editor").info(
                        "preview voxels: static %s, dynamic %s",
                        static_num or 0,
                        dynamic_num or 0,
                    )
                    return
            stop.wait(0.2)

    @staticmethod
    def _make_preview_pause_hud():
        """Return the viewer hud toggling pause on the space bar.

        An interactive preview opens paused for inspection; this hud is
        the way back out, switching the world between paused and running
        on the space bar's rising edge and naming the state on screen.
        """
        from PySide6 import QtCore, QtGui
        from PySide6.QtCore import Qt

        state = {"was_down": False}

        def hud(painter, viewer) -> None:
            is_down = viewer.is_key_down(Qt.Key.Key_Space)
            if is_down and not state["was_down"] and viewer.handle is not None:
                if viewer.handle.is_paused():
                    viewer.handle.resume()
                else:
                    viewer.handle.pause()
            state["was_down"] = is_down
            if viewer.handle is None:
                return
            font = QtGui.QFont()
            font.setPointSize(11)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(240, 240, 240))
            label = (
                "paused - space resumes"
                if viewer.handle.is_paused()
                else "running - space pauses"
            )
            painter.drawText(
                QtCore.QRect(16, 16, 320, 24), Qt.AlignmentFlag.AlignLeft, label
            )

        return hud

    @staticmethod
    def _make_preview_environment(
        entities: List[object],
    ) -> Tuple[List[object], List[object], List[object]]:
        """Return the default sky and floor a preview world is missing.

        Content carrying its own cosmic background or floor keeps it;
        only the missing pieces inject, under ``preview.`` names.
        """
        extra_entities: List[object] = []
        extra_essences: List[object] = []
        extra_resources: List[object] = []
        if not any(isinstance(entity, RSC_CosmicBackgroundBase) for entity in entities):
            extra_entities.append(
                CosmicBackgroundHosekWilkie(
                    "preview.sky",
                    sun_color=Editor._make_byte_color(PREVIEW_SKY_SUN_COLOR),
                    ground_color=Editor._make_byte_color(PREVIEW_SKY_GROUND_COLOR),
                    intensity=PREVIEW_SKY_INTENSITY,
                )
            )
        if not any(isinstance(entity, RSC_FloorBase) for entity in entities):
            extra_resources.append(
                TextureGrid(
                    "preview.ground#grid",
                    fill_color=PREVIEW_GROUND_FILL_COLOR,
                    line_color=PREVIEW_GRID_LINE_COLOR,
                    x_spacing=PREVIEW_GRID_SPACING,
                    y_spacing=PREVIEW_GRID_SPACING,
                    line_width=PREVIEW_GRID_LINE_WIDTH,
                )
            )
            extra_essences.extend(
                [
                    Material("preview.ground"),
                    MaterialPhysicalPropertyFixed(
                        "preview.ground#fixed",
                        material_reference="preview.ground",
                    ),
                    MaterialOpticalPropertySimpleMicrofacet(
                        "preview.ground#optic",
                        material_reference="preview.ground",
                        albedo_texture_reference="preview.ground#grid",
                    ),
                ]
            )
            extra_entities.append(
                FloorFlat("preview.floor", material_reference="preview.ground")
            )
        return extra_entities, extra_essences, extra_resources

    @staticmethod
    def _make_byte_color(color: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Return an RGB 0-1 float color as a 0-255 byte triple.

        Args:
            color: The color as three 0-1 floats.
        """
        return tuple(round(channel * 255) for channel in color)

    @staticmethod
    def _make_rsct_payload(rsc: object) -> dict:
        """Return an rsc object as the MCP ``{"type", "data"}`` json payload."""
        handle, path = tempfile.mkstemp(suffix=".rsct")
        os.close(handle)
        try:
            rsc.save(path)
            with open(path, "r", encoding="utf-8") as saved:
                stored = json.load(saved)
        finally:
            os.remove(path)
        return {"type": stored["header"]["type"], "data": stored["body"]}

    @staticmethod
    def _call_mcp_tool(
        address: str,
        token: Optional[str],
        name: str,
        arguments: dict,
        timeout_s: float,
    ):
        """Call one MCP tool, poll a pending result, return the inner value."""
        started = time.monotonic()
        inner = Editor._post_mcp_tool_call(address, token, name, arguments, timeout_s)
        while isinstance(inner, dict) and inner.get("status") == "pending":
            if time.monotonic() - started > timeout_s:
                raise EditorError(
                    f"MCP tool '{name}' stayed pending past {timeout_s:g} s; a "
                    "paused world applies tasks only after it resumes"
                )
            time.sleep(0.05)
            inner = Editor._post_mcp_tool_call(
                address, token, "get_result", {"id": inner["id"]}, timeout_s
            )
        if isinstance(inner, dict) and inner.get("ok") is False:
            raise EditorError(f"MCP tool '{name}' failed: {inner.get('error')}")
        return inner

    @staticmethod
    def _post_mcp_tool_call(
        address: str,
        token: Optional[str],
        name: str,
        arguments: dict,
        timeout_s: float,
    ):
        """POST one tools/call and return its double-wrapped inner result."""
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        response = requests.post(
            address, json=payload, headers=headers, timeout=timeout_s
        )
        response.raise_for_status()
        result = response.json().get("result", {})
        content = result.get("content", [])
        text = content[0].get("text", "") if content else ""
        if result.get("isError"):
            raise EditorError(f"MCP tool '{name}' failed: {text}")
        try:
            return json.loads(text)
        except ValueError:
            return text
