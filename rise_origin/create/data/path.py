# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The path: waypoints on a curve with centripetal Catmull-Rom smoothing.

Waypoints lie on the curve itself. The path owns its own sampling math, so
solvers ask the path instead of carrying private curve helpers.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from rise_origin import RVec3rf

__all__ = ["Path"]


@dataclass(frozen=True)
class Path:
    """Waypoints on a curve with centripetal Catmull-Rom smoothing.

    ``smooth="none"`` keeps the straight polyline; ``min_radius`` rounds
    corners that would bend tighter and is not applied yet.

    Attributes:
        waypoints: Points on the curve, in the frame of the draft whose
            method takes the path.
        smooth: The smoothing mode.
        min_radius: The smallest corner radius, in meters.
    """

    waypoints: Tuple[RVec3rf, ...]
    smooth: Literal["none", "catmull_rom"] = "catmull_rom"
    min_radius: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "waypoints",
            tuple(
                point if isinstance(point, RVec3rf) else RVec3rf(point)
                for point in self.waypoints
            ),
        )

    def resample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return a dense polyline and its cumulative arc lengths.

        Returns:
            The ``(n, 3)`` polyline points and the ``(n,)`` cumulative
            lengths, ready for :meth:`sample_at_station`.
        """
        waypoints = np.array([[p.x, p.y, p.z] for p in self.waypoints])
        if waypoints.shape[0] < 2:
            raise ValueError("a path needs at least two waypoints")
        if self.smooth == "none" or waypoints.shape[0] == 2:
            points = waypoints
        else:
            points = Path._smooth_catmull_rom(waypoints)
        deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
        lengths = np.concatenate([[0.0], np.cumsum(deltas)])
        return points, lengths

    def to_json(self) -> dict:
        """Return the json form of this path."""
        return {
            "waypoints": [[p.x, p.y, p.z] for p in self.waypoints],
            "smooth": self.smooth,
            "min_radius": self.min_radius,
        }

    @staticmethod
    def from_json(path_json: dict) -> "Path":
        """Return the path of its json form."""
        return Path(
            tuple(RVec3rf(point) for point in path_json["waypoints"]),
            path_json["smooth"],
            path_json["min_radius"],
        )

    @staticmethod
    def sample_at_station(
        points: np.ndarray, lengths: np.ndarray, station: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the position and unit tangent at an arc-length station.

        Args:
            points: The ``(n, 3)`` dense polyline from :meth:`resample`.
            lengths: The ``(n,)`` cumulative arc lengths from :meth:`resample`.
            station: The arc-length distance along the polyline to sample at.
        """
        s = float(np.clip(station, 0.0, lengths[-1]))
        index = int(np.searchsorted(lengths, s, side="right") - 1)
        index = min(index, points.shape[0] - 2)
        span = lengths[index + 1] - lengths[index]
        blend = 0.0 if span <= 0 else (s - lengths[index]) / span
        position = points[index] * (1 - blend) + points[index + 1] * blend
        low = max(index - 1, 0)
        high = min(index + 2, points.shape[0] - 1)
        tangent = points[high] - points[low]
        norm = np.linalg.norm(tangent[:2])
        if norm < 1e-12:
            tangent = points[index + 1] - points[index]
        return position, tangent / max(np.linalg.norm(tangent), 1e-12)

    @staticmethod
    def _smooth_catmull_rom(waypoints: np.ndarray, samples: int = 16) -> np.ndarray:
        """Return the centripetal Catmull-Rom polyline through the waypoints."""
        padded = np.vstack(
            [
                waypoints[0] * 2 - waypoints[1],
                waypoints,
                waypoints[-1] * 2 - waypoints[-2],
            ]
        )
        points = []
        for i in range(1, padded.shape[0] - 2):
            p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
            t = np.zeros(4)
            for j, (a, b) in enumerate(((p0, p1), (p1, p2), (p2, p3))):
                t[j + 1] = t[j] + max(np.linalg.norm(b - a) ** 0.5, 1e-9)
            queries = np.linspace(t[1], t[2], samples, endpoint=False)
            for tq in queries:
                a1 = (t[1] - tq) / (t[1] - t[0]) * p0 + (tq - t[0]) / (t[1] - t[0]) * p1
                a2 = (t[2] - tq) / (t[2] - t[1]) * p1 + (tq - t[1]) / (t[2] - t[1]) * p2
                a3 = (t[3] - tq) / (t[3] - t[2]) * p2 + (tq - t[2]) / (t[3] - t[2]) * p3
                b1 = (t[2] - tq) / (t[2] - t[0]) * a1 + (tq - t[0]) / (t[2] - t[0]) * a2
                b2 = (t[3] - tq) / (t[3] - t[1]) * a2 + (tq - t[1]) / (t[3] - t[1]) * a3
                point = (t[2] - tq) / (t[2] - t[1]) * b1 + (tq - t[1]) / (
                    t[2] - t[1]
                ) * b2
                points.append(point)
        points.append(waypoints[-1])
        return np.asarray(points)
