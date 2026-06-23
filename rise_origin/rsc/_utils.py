# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Internal tuple-to-vector coercion helpers shared by the ``rsc`` builders.

The bound ``RVec3rf`` / ``RQuat3rf`` / ``RColor3b`` / ``RColor3f`` / ``RBound3ri``
types cannot be assigned a raw Python tuple, so every ``rsc`` class accepts
either the bound type or a plain tuple and funnels it through these coercers.

Alias names drop the bound type's leading ``R`` but keep its precision suffix
(``Vec3rf``, ``Color3b`` vs a future ``Color3f``) so they never collide.
"""

from typing import Sequence, Tuple, Union

from rise_origin import (
    RBound3ri,
    RColor3b,
    RColor3f,
    RQuat3rf,
    RMat3rf,
    RVec3rf,
    VecFloat,
)

#: 3D float vector / position in meters: ``(x, y, z)`` or an ``RVec3rf``.
Vec3rf = Union[RVec3rf, Tuple[float, float, float]]

#: 3x3 row-major float matrix: a sequence of three ``(x, y, z)`` rows, a flat
#: nine-element sequence, or an ``RMat3rf``.
Mat3rf = Union[RMat3rf, Sequence[Sequence[float]], Sequence[float]]

#: Orientation quaternion in ``(x, y, z, w)`` order, or an ``RQuat3rf``.
Quat3rf = Union[RQuat3rf, Tuple[float, float, float, float]]

#: Byte RGB color, 0-255 per channel: ``(r, g, b)`` or an ``RColor3b``.
Color3b = Union[RColor3b, Tuple[int, int, int]]

#: Float RGB color, 0-1 per channel: ``(r, g, b)`` or an ``RColor3f``.
Color3f = Union[RColor3f, Tuple[float, float, float]]

#: Inclusive signed voxel-index box: ``(min_x, min_y, min_z, max_x, max_y, max_z)``
#: or an ``RBound3ri``.
Bound3ri = Union[RBound3ri, Tuple[int, int, int, int, int, int]]


def _mat3rf(value: Mat3rf) -> RMat3rf:
    """Coerce a 3x3 or flat nine-element sequence into an ``RMat3rf``."""
    if isinstance(value, RMat3rf):
        return value
    rows = list(value)
    if len(rows) == 9 and not hasattr(rows[0], "__len__"):
        rows = [rows[0:3], rows[3:6], rows[6:9]]
    matrix = RMat3rf()
    matrix.from_list([VecFloat([float(x) for x in row]) for row in rows])
    return matrix


def _apply(config, **fields) -> None:
    """Set each non-None field on a config, leaving the engine default otherwise."""
    for name, value in fields.items():
        if value is not None:
            setattr(config, name, value)
