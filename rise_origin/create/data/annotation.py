# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The annotation family: named 3D data in the owning node's local frame.

Annotations inform and are never load-bearing: no matching, no snapping, no
checking. Geometric payloads ride the node's frame, so moving the node moves
its annotations with it. The json round trip here is shared by the journal
and the cache.
"""

from dataclasses import dataclass
from typing import Optional

from rise_origin import RQuat3rf, RVec3rf

from .box import AABB
from .path import Path
from .pose import Pose

__all__ = [
    "Annotation",
    "AnnotationPoint",
    "AnnotationRegion",
    "AnnotationPath",
    "AnnotationView",
    "AnnotationNote",
]


class Annotation:
    """The base of every annotation kind."""

    def to_json(self) -> dict:
        """Return the json form of this annotation."""
        raise NotImplementedError

    @staticmethod
    def from_json(annotation_json: dict) -> "Annotation":
        """Return the annotation of its json form.

        Args:
            annotation_json: The json form of an annotation, as
                :meth:`to_json` returns it.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class AnnotationPoint(Annotation):
    """A named point, with an optional orientation, in the node's local frame.

    Attributes:
        position: The point position.
        orientation: The orientation at the point, or None for no
            orientation.
    """

    position: RVec3rf
    orientation: Optional[RQuat3rf] = None

    def __post_init__(self) -> None:
        if not isinstance(self.position, RVec3rf):
            object.__setattr__(self, "position", RVec3rf(self.position))
        if self.orientation is not None and not isinstance(self.orientation, RQuat3rf):
            object.__setattr__(self, "orientation", RQuat3rf(self.orientation))

    def to_json(self) -> dict:
        """Return the json form of this annotation."""
        orientation = self.orientation
        return {
            "type": "point",
            "position": [self.position.x, self.position.y, self.position.z],
            "orientation": (
                None
                if orientation is None
                else [orientation.x, orientation.y, orientation.z, orientation.w]
            ),
        }

    @staticmethod
    def from_json(annotation_json: dict) -> "AnnotationPoint":
        """Return the annotation of its json form.

        Args:
            annotation_json: The json form of a point annotation, as
                :meth:`to_json` returns it.
        """
        orientation = annotation_json["orientation"]
        return AnnotationPoint(
            RVec3rf(annotation_json["position"]),
            None if orientation is None else RQuat3rf(orientation),
        )


@dataclass(frozen=True)
class AnnotationRegion(Annotation):
    """A named box volume in the node's local frame.

    Attributes:
        box: The axis-aligned box of the volume.
    """

    box: AABB

    def to_json(self) -> dict:
        """Return the json form of this annotation."""
        return {
            "type": "region",
            "min": [self.box.min.x, self.box.min.y, self.box.min.z],
            "max": [self.box.max.x, self.box.max.y, self.box.max.z],
        }

    @staticmethod
    def from_json(annotation_json: dict) -> "AnnotationRegion":
        """Return the annotation of its json form.

        Args:
            annotation_json: The json form of a region annotation, as
                :meth:`to_json` returns it.
        """
        return AnnotationRegion(AABB(annotation_json["min"], annotation_json["max"]))


@dataclass(frozen=True)
class AnnotationPath(Annotation):
    """A named curve in the node's local frame.

    Attributes:
        path: The curve of this annotation.
    """

    path: Path

    def to_json(self) -> dict:
        """Return the json form of this annotation."""
        return {"type": "path", **self.path.to_json()}

    @staticmethod
    def from_json(annotation_json: dict) -> "AnnotationPath":
        """Return the annotation of its json form.

        Args:
            annotation_json: The json form of a path annotation, as
                :meth:`to_json` returns it.
        """
        return AnnotationPath(
            Path(
                tuple(RVec3rf(point) for point in annotation_json["waypoints"]),
                annotation_json["smooth"],
                annotation_json["min_radius"],
            )
        )


@dataclass(frozen=True)
class AnnotationView(Annotation):
    """A named camera pose with a lens hint, in the node's local frame.

    Attributes:
        pose: The camera pose.
        fov_deg: The field of view in degrees.
    """

    pose: Pose
    fov_deg: float = 60.0

    def to_json(self) -> dict:
        """Return the json form of this annotation."""
        return {"type": "view", "pose": self.pose.to_json(), "fov_deg": self.fov_deg}

    @staticmethod
    def from_json(annotation_json: dict) -> "AnnotationView":
        """Return the annotation of its json form.

        Args:
            annotation_json: The json form of a view annotation, as
                :meth:`to_json` returns it.
        """
        return AnnotationView(
            Pose.from_json(annotation_json["pose"]), annotation_json["fov_deg"]
        )


@dataclass(frozen=True)
class AnnotationNote(Annotation):
    """Free text attached to the node.

    Attributes:
        text: The note text.
    """

    text: str

    def to_json(self) -> dict:
        """Return the json form of this annotation."""
        return {"type": "note", "text": self.text}

    @staticmethod
    def from_json(annotation_json: dict) -> "AnnotationNote":
        """Return the annotation of its json form.

        Args:
            annotation_json: The json form of a note annotation, as
                :meth:`to_json` returns it.
        """
        return AnnotationNote(annotation_json["text"])
