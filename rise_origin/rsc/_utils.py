# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Internal tuple-to-vector coercion helpers shared by the ``rsc`` builders.

The bound ``RVec3rf`` / ``RQuat3rf`` / ``RColor3b`` / ``RColor3f`` / ``RBound3ri``
types cannot be assigned a raw Python tuple, so every ``rsc`` class accepts
either the bound type or a plain tuple and funnels it through these coercers.

Alias names drop the bound type's leading ``R`` but keep its precision suffix
(``Vec3rf``, ``Color3b`` vs a future ``Color3f``) so they never collide.
"""

from typing import Tuple, Union

from rise_origin import RBound3ri, RColor3b, RColor3f, RQuat3rf, RVec3rf

#: 3D float vector / position in meters: ``(x, y, z)`` or an ``RVec3rf``.
Vec3rf = Union[RVec3rf, Tuple[float, float, float]]

#: Orientation quaternion in ``(x, y, z, w)`` order, or an ``RQuat3rf``.
Quat3rf = Union[RQuat3rf, Tuple[float, float, float, float]]

#: Byte RGB color, 0-255 per channel: ``(r, g, b)`` or an ``RColor3b``.
Color3b = Union[RColor3b, Tuple[int, int, int]]

#: Float RGB color, 0-1 per channel: ``(r, g, b)`` or an ``RColor3f``.
Color3f = Union[RColor3f, Tuple[float, float, float]]

#: Inclusive signed voxel-index box: ``(min_x, min_y, min_z, max_x, max_y, max_z)``
#: or an ``RBound3ri``.
Bound3ri = Union[RBound3ri, Tuple[int, int, int, int, int, int]]


def _vec3rf(value: Vec3rf) -> RVec3rf:
    """Coerce a 3-tuple into an ``RVec3rf`` (pass through if already one)."""
    return value if isinstance(value, RVec3rf) else RVec3rf(*value)


def _quat3rf(value: Quat3rf) -> RQuat3rf:
    """Coerce a 4-tuple into an ``RQuat3rf`` (pass through if already one)."""
    return value if isinstance(value, RQuat3rf) else RQuat3rf(*value)


def _color3b(value: Color3b) -> RColor3b:
    """Coerce a 3-tuple into an ``RColor3b`` (pass through if already one)."""
    return value if isinstance(value, RColor3b) else RColor3b(*value)


def _color3f(value: Color3f) -> RColor3f:
    """Coerce a 3-tuple into an ``RColor3f`` (pass through if already one)."""
    return value if isinstance(value, RColor3f) else RColor3f(*value)


def _bound3ri(value: Bound3ri) -> RBound3ri:
    """Coerce a 6-tuple into an ``RBound3ri`` (pass through if already one)."""
    return value if isinstance(value, RBound3ri) else RBound3ri(*value)


def _fill(target, values) -> None:
    """Append each value into a bound (non-list-assignable) vector, in order."""
    for value in values:
        target.append(value)


def _apply(config, **fields) -> None:
    """Set each non-None field on a config, leaving the engine default otherwise."""
    for name, value in fields.items():
        if value is not None:
            setattr(config, name, value)
