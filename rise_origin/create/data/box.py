# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Boxes: the cover box, the axis-aligned bound, and the selected face.

A cover box is the author's statement of occupied space in the owning node's
local frame. The frame vocabulary is root for the whole create layer; "world"
only ever means the compiled rsc World.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Tuple

import numpy as np

from rise_origin import RVec2rf, RVec3rf

from .pose import Pose

if TYPE_CHECKING:
    from ..draft.draft import Draft

__all__ = ["MaterialClass", "Box", "AABB", "Face"]

MaterialClass = Literal["static", "dynamic", "mixed"]


@dataclass(frozen=True)
class Box:
    """One cover box with its declared material class.

    A voxel is static when it belongs to a fixed structure or carries a fixed
    physical material; a box tagged ``static`` must contain only static
    voxels, one tagged ``dynamic`` only dynamic voxels, and ``mixed`` defers
    the judgement to the voxel gate at realization.

    Attributes:
        min: Lower corner in the owning node's local frame, in meters.
        max: Upper corner in the owning node's local frame, in meters.
        material_class: The declared class of the matter inside.
    """

    min: RVec3rf
    max: RVec3rf
    material_class: MaterialClass = "static"

    def __post_init__(self) -> None:
        if not isinstance(self.min, RVec3rf):
            object.__setattr__(self, "min", RVec3rf(self.min))
        if not isinstance(self.max, RVec3rf):
            object.__setattr__(self, "max", RVec3rf(self.max))

    def get_corners(self) -> np.ndarray:
        """Return the eight corners of this box as an ``(8, 3)`` array."""
        lo = np.array([self.min.x, self.min.y, self.min.z])
        hi = np.array([self.max.x, self.max.y, self.max.z])
        return np.array(
            [
                [x, y, z]
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            ]
        )

    def get_aabb_in_root(self, pose: Pose) -> Tuple[np.ndarray, np.ndarray]:
        """Return the root-frame axis-aligned bounds of this box under a pose.

        Args:
            pose: The pose mapping this box's frame to the root frame.

        Returns:
            The root-frame min corner and max corner of the box.
        """
        corners = self.get_corners()
        rotation = np.asarray(
            pose.orientation.to_rotation_matrix().to_list(), dtype=float
        )
        position = np.array(
            [
                pose.relative_position.x,
                pose.relative_position.y,
                pose.relative_position.z,
            ]
        )
        in_root = corners @ rotation.T + position
        return in_root.min(axis=0), in_root.max(axis=0)

    def to_json(self) -> dict:
        """Return the json form of this box."""
        return {
            "min": [self.min.x, self.min.y, self.min.z],
            "max": [self.max.x, self.max.y, self.max.z],
            "class": self.material_class,
        }

    @staticmethod
    def from_json(box_json: dict) -> "Box":
        """Return the box of its json form.

        Args:
            box_json: The json form of a box, as :meth:`to_json` returns it.
        """
        return Box(
            RVec3rf(box_json["min"]), RVec3rf(box_json["max"]), box_json["class"]
        )


@dataclass(frozen=True)
class AABB:
    """An axis-aligned box, in meters, in the frame of the draft whose
    method takes or returns it.

    Attributes:
        min: The lower corner.
        max: The upper corner.
    """

    min: RVec3rf
    max: RVec3rf

    def __post_init__(self) -> None:
        if not isinstance(self.min, RVec3rf):
            object.__setattr__(self, "min", RVec3rf(self.min))
        if not isinstance(self.max, RVec3rf):
            object.__setattr__(self, "max", RVec3rf(self.max))


@dataclass(frozen=True)
class Face:
    """The extremal plane of a node cover in a local direction.

    Attributes:
        center: Face center, in the frame of the draft that returned it.
        normal: Outward unit normal, in the same frame as the center.
        rect: Extents of the face plane along its ``(u, v)`` basis.
        node: The Draft the face belongs to.
    """

    center: RVec3rf
    normal: RVec3rf
    rect: RVec2rf
    node: "Draft"

    def __post_init__(self) -> None:
        if not isinstance(self.center, RVec3rf):
            object.__setattr__(self, "center", RVec3rf(self.center))
        if not isinstance(self.normal, RVec3rf):
            object.__setattr__(self, "normal", RVec3rf(self.normal))
        if not isinstance(self.rect, RVec2rf):
            object.__setattr__(self, "rect", RVec2rf(self.rect))
