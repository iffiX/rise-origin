# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Query inputs and results: filters, distances, jitters, and reports.

Results keep Draft references because they are statements about nodes.
Coordinates are in the frame of the draft whose method produced the value,
unless the method says otherwise.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

from rise_origin import RVec3rf

from .box import MaterialClass
from .pose import Pose

if TYPE_CHECKING:
    from ..draft.draft import Draft

__all__ = [
    "Filters",
    "Distance",
    "AlongJitter",
    "GridJitter",
    "RingJitter",
    "FloorSample",
    "RayHit",
    "SurfaceAnchor",
    "NearbyHit",
    "Overlap",
    "PlacementReport",
    "LeanResult",
]


@dataclass(frozen=True)
class Filters:
    """The cheap predicate set every range query accepts.

    Attributes:
        module: Module name prefix pattern such as
            ``"rise_origin.nature.tree.*"``.
        tags: Tags a node must all carry.
        is_fixed: Keep only effectively fixed or only free nodes when set.
        under: Keep only nodes inside this subtree.
        exclude: Node names to drop.
    """

    module: Optional[str] = None
    tags: Tuple[str, ...] = ()
    is_fixed: Optional[bool] = None
    under: Optional["Draft"] = None
    exclude: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Distance:
    """Per-axis gaps of the minimizing box pair plus the scalar.

    The components skip the shadow test, so they are not the three
    :meth:`Draft.get_gap` values.

    Attributes:
        x: The gap along x; zero when the boxes overlap on this axis.
        y: The gap along y; zero when the boxes overlap on this axis.
        z: The gap along z; zero when the boxes overlap on this axis.
        euclidean: The scalar distance between the boxes.
    """

    x: float
    y: float
    z: float
    euclidean: float


@dataclass(frozen=True)
class AlongJitter:
    """Uniform pose jitter ranges for :meth:`Draft.compute_along_poses`.

    Attributes:
        along: The jitter range along the path, in meters.
        lateral: The jitter range sideways from the path, in meters.
        yaw_deg: The yaw jitter range in degrees.
    """

    along: float = 0.0
    lateral: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class GridJitter:
    """Uniform pose jitter ranges for :meth:`Draft.compute_grid_poses`.

    Attributes:
        x: The jitter range along the lattice x axis, in meters.
        y: The jitter range along the lattice y axis, in meters.
        yaw_deg: The yaw jitter range in degrees.
    """

    x: float = 0.0
    y: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class RingJitter:
    """Uniform pose jitter ranges for :meth:`Draft.compute_ring_poses`.

    Attributes:
        radial: The jitter range toward and away from the ring center, in
            meters.
        tangential: The jitter range along the ring, in meters.
        yaw_deg: The yaw jitter range in degrees.
    """

    radial: float = 0.0
    tangential: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class FloorSample:
    """The floor height, normal, and grade at one ``(x, y)`` point.

    Attributes:
        height: The floor height at the point, in meters.
        normal: The floor unit normal at the point.
        slope_deg: The floor grade at the point, in degrees.
    """

    height: float
    normal: RVec3rf
    slope_deg: float

    def __post_init__(self) -> None:
        if not isinstance(self.normal, RVec3rf):
            object.__setattr__(self, "normal", RVec3rf(self.normal))


@dataclass(frozen=True)
class RayHit:
    """The first cover box a ray meets.

    Attributes:
        distance: The distance from the ray start to the box, in meters.
        blocker: The node whose cover box the ray meets.
    """

    distance: float
    blocker: "Draft"


@dataclass(frozen=True)
class SurfaceAnchor:
    """Where a ray meets declared matter, as a placement anchor.

    Attributes:
        position: The point where the ray meets the matter.
        normal: The outward unit normal at that point.
        node: The node whose matter the ray meets.
    """

    position: RVec3rf
    normal: RVec3rf
    node: "Draft"

    def __post_init__(self) -> None:
        if not isinstance(self.position, RVec3rf):
            object.__setattr__(self, "position", RVec3rf(self.position))
        if not isinstance(self.normal, RVec3rf):
            object.__setattr__(self, "normal", RVec3rf(self.normal))


@dataclass(frozen=True)
class NearbyHit:
    """One range query result, sorted ascending by distance.

    Attributes:
        node: The found node.
        distance: The distance from the queried thing to the node.
    """

    node: "Draft"
    distance: Distance


@dataclass(frozen=True)
class Overlap:
    """One cover overlap between the checked thing and another node.

    Attributes:
        other: The overlapping node.
        depth: Per-axis overlap widths of the deepest box pair.
        translation: The minimal translation that clears the overlap.
        material_class_pair: The two boxes' declared material classes.
    """

    other: "Draft"
    depth: RVec3rf
    translation: RVec3rf
    material_class_pair: Tuple[MaterialClass, MaterialClass]

    def __post_init__(self) -> None:
        if not isinstance(self.depth, RVec3rf):
            object.__setattr__(self, "depth", RVec3rf(self.depth))
        if not isinstance(self.translation, RVec3rf):
            object.__setattr__(self, "translation", RVec3rf(self.translation))


@dataclass(frozen=True)
class PlacementReport:
    """The no-intersection rule as one detailed report.

    Attributes:
        is_ok: Whether the placement passes the rule.
        violations: Hard overlaps that break the rule.
        embeddings: Legal static co-occupation overlaps.
        pending: Overlaps involving a mixed box, resolved once at
            realization.
    """

    is_ok: bool
    violations: Tuple[Overlap, ...] = ()
    embeddings: Tuple[Overlap, ...] = ()
    pending: Tuple[Overlap, ...] = ()


@dataclass(frozen=True)
class LeanResult:
    """A solved two-contact rest pose with its contact points.

    Attributes:
        pose: The solved rest pose.
        contacts: The base contact point and the lean contact point.
        tilt: The tilt from the base plane, in degrees.
    """

    pose: Pose
    contacts: Tuple[RVec3rf, RVec3rf]
    tilt: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "contacts",
            tuple(
                point if isinstance(point, RVec3rf) else RVec3rf(point)
                for point in self.contacts
            ),
        )
