# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The pose: a position and orientation pair in one draft's frame.

Composition follows the frame rule: ``parent.compose(child)`` expresses a
child pose, given in the parent's frame, in the parent's own outer frame.
The math delegates to the engine-bound vector and quaternion types.
"""

from dataclasses import dataclass, field

from rise_origin import RQuat3rf, RVec3rf

__all__ = ["Pose"]


@dataclass(frozen=True)
class Pose:
    """A position and orientation in the frame of the draft whose method
    takes or returns it.

    Attributes:
        relative_position: Position ``(x, y, z)`` in meters.
        orientation: Unit quaternion ``(x, y, z, w)``.
    """

    relative_position: RVec3rf = field(default_factory=RVec3rf.zero)
    orientation: RQuat3rf = field(default_factory=RQuat3rf.unit)

    def __post_init__(self) -> None:
        if not isinstance(self.relative_position, RVec3rf):
            object.__setattr__(
                self, "relative_position", RVec3rf(self.relative_position)
            )
        if not isinstance(self.orientation, RQuat3rf):
            object.__setattr__(self, "orientation", RQuat3rf(self.orientation))

    def compose(self, child: "Pose") -> "Pose":
        """Express a child pose, given in this pose's frame, in the outer frame.

        Args:
            child: The pose to express, given in this pose's frame.
        """
        position = self.relative_position + self.orientation.rotate_vec3d(
            child.relative_position
        )
        return Pose(position, self.orientation * child.orientation)

    def invert(self) -> "Pose":
        """Return the pose mapping the outer frame back into this pose's frame."""
        inverse = self.orientation.inverse()
        return Pose(-inverse.rotate_vec3d(self.relative_position), inverse)

    def is_identity(self) -> bool:
        """Return whether this pose is the identity within float32 tolerance.

        The tolerance admits the engine's quaternion renormalization,
        which maps an exact unit quaternion to w = 0.99999994 once.
        """
        position = self.relative_position
        return (
            abs(position.x) < 1e-6
            and abs(position.y) < 1e-6
            and abs(position.z) < 1e-6
            and abs(float(self.orientation.w) - 1.0) < 1e-6
        )

    def apply(self, point) -> RVec3rf:
        """Map a point through this pose.

        Args:
            point: The point to map, given in this pose's frame.
        """
        if not isinstance(point, RVec3rf):
            point = RVec3rf(point)
        return self.relative_position + self.orientation.rotate_vec3d(point)

    def to_json(self) -> dict:
        """Return the json form of this pose."""
        position = self.relative_position
        orientation = self.orientation
        return {
            "p": [position.x, position.y, position.z],
            "q": [orientation.x, orientation.y, orientation.z, orientation.w],
        }

    @staticmethod
    def from_json(pose_json: dict) -> "Pose":
        """Return the pose of its json form.

        Args:
            pose_json: The json form of a pose, as :meth:`to_json` returns it.
        """
        return Pose(RVec3rf(pose_json["p"]), RQuat3rf(pose_json["q"]))
