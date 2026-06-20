# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The :class:`Stamp` voxel-block abstraction.

A :class:`Stamp` is a sub-grid of materials (object array of name strings or
``None``) that can be composed via slice writes, rotated about Z, mirrored,
and pasted into a :class:`scene.world.World` at any anchor.

Object builders return :class:`Stamp` instances; biome furnishers compose them
into cell-sized layouts. The Stamp does not store segment ids — those are looked
up via :data:`scene.materials.SEG_OF` at paste time, so renaming a material
never invalidates a stored stamp.
"""

from __future__ import annotations

import functools
from typing import Callable, Tuple

import numpy as np


PasteMode = str  # "overlay" | "replace"


class Stamp:
    """A small (X, Y, Z) voxel block.

    Args:
        shape: ``(nx, ny, nz)`` extent in voxels of *whichever resolution this
            stamp will eventually be pasted into*. Stamps are resolution-agnostic
            in their own right.
    """

    __slots__ = ("materials",)

    def __init__(self, shape: Tuple[int, int, int]) -> None:
        self.materials = np.full(shape, None, dtype=object)

    # ------------------------------------------------------------------ shape

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.materials.shape)  # type: ignore[return-value]

    @property
    def nx(self) -> int: return self.materials.shape[0]

    @property
    def ny(self) -> int: return self.materials.shape[1]

    @property
    def nz(self) -> int: return self.materials.shape[2]

    # ------------------------------------------------------------------ writes

    def set(self, x: int, y: int, z: int, material: str) -> None:
        """Place one voxel of ``material`` at ``(x, y, z)``. Out-of-bounds is a no-op."""
        if 0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz:
            self.materials[x, y, z] = material

    def fill(self, slc: Tuple, material: str) -> None:
        """Fill a region with ``material``.

        ``slc`` is a tuple of three ``slice`` (or ``int``) objects, indexing
        into ``self.materials`` directly.
        """
        self.materials[slc] = material

    def clear(self, slc: Tuple) -> None:
        """Knock out voxels in ``slc`` back to ``None`` (carve holes / damage)."""
        self.materials[slc] = None

    # ----------------------------------------------------------------- queries

    def is_empty(self) -> bool:
        return not np.any(self.materials != None)  # noqa: E711

    def occupied_count(self) -> int:
        return int(np.count_nonzero(self.materials != None))  # noqa: E711

    # ----------------------------------------------------------- transformations

    def rotate_z(self, k: int) -> "Stamp":
        """Return a new stamp rotated ``k * 90`` degrees about the Z axis.

        Rotates within the XY plane; Z stays as Z. The returned stamp has the
        appropriately-permuted shape (after k=1, X and Y dimensions swap).
        """
        k = int(k) % 4
        if k == 0:
            out = Stamp(self.shape)
            out.materials = self.materials.copy()
            return out
        rotated = np.rot90(self.materials, k=k, axes=(0, 1))
        out = Stamp(rotated.shape)  # type: ignore[arg-type]
        out.materials = np.ascontiguousarray(rotated)
        return out

    def mirror_x(self) -> "Stamp":
        """Return a new stamp mirrored along the X axis."""
        out = Stamp(self.shape)
        out.materials = np.flip(self.materials, axis=0).copy()
        return out

    def mirror_y(self) -> "Stamp":
        """Return a new stamp mirrored along the Y axis."""
        out = Stamp(self.shape)
        out.materials = np.flip(self.materials, axis=1).copy()
        return out

    def materialize(self) -> "Stamp":
        """Mark this stamp as fully built. Returns ``self``.

        Logically signals that the stamp is now immutable from the catalog's
        point of view (callers should treat it as read-only). The real work
        is done by :func:`stamp_cached`, which calls ``materialize()`` once
        per cached variant; subsequent calls hand out the same instance.

        Note that ``Stamp.paste_into`` only reads the materials array, and
        ``rotate_z`` / ``mirror_x`` / ``mirror_y`` all return *fresh* stamps,
        so sharing the same ``Stamp`` across many placements is safe.
        """
        return self

    # ------------------------------------------------------------------ paste

    def paste_into(
        self,
        world_mats: np.ndarray,
        world_segs: np.ndarray,
        ax: int, ay: int, az: int,
        seg_of: dict,
        mode: PasteMode = "overlay",
    ) -> int:
        """Paste this stamp into the world arrays at anchor ``(ax, ay, az)``.

        Args:
            world_mats: (NX, NY, NZ) object-array of material names to write into.
            world_segs: (NX, NY, NZ) int array of segment ids matching ``world_mats``.
            ax, ay, az: world anchor (lower-corner) for the stamp's origin.
            seg_of: material name -> segment id (typically :data:`scene.materials.SEG_OF`).
            mode:
              * ``"overlay"`` — only write where the world is currently empty (``None``).
              * ``"replace"`` — write unconditionally (used for damage knock-outs etc).

        Returns:
            The number of voxels actually written.
        """
        sx, sy, sz = self.shape
        # Crop the stamp to the world bounds.
        wx, wy, wz = world_mats.shape
        x0, y0, z0 = max(0, ax), max(0, ay), max(0, az)
        x1, y1, z1 = min(wx, ax + sx), min(wy, ay + sy), min(wz, az + sz)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            return 0
        sx0, sy0, sz0 = x0 - ax, y0 - ay, z0 - az
        sx1, sy1, sz1 = sx0 + (x1 - x0), sy0 + (y1 - y0), sz0 + (z1 - z0)

        src = self.materials[sx0:sx1, sy0:sy1, sz0:sz1]
        present = src != None  # noqa: E711

        if mode == "overlay":
            empty = world_mats[x0:x1, y0:y1, z0:z1] == None  # noqa: E711
            mask = present & empty
        elif mode == "replace":
            mask = present
        else:
            raise ValueError(f"unknown paste mode {mode!r}")

        if not np.any(mask):
            return 0

        # Apply.
        world_view_m = world_mats[x0:x1, y0:y1, z0:z1]
        world_view_s = world_segs[x0:x1, y0:y1, z0:z1]
        # The materials array is `object`; index with the boolean mask and write.
        names = src[mask]
        world_view_m[mask] = names
        # Map each name to its segment id and write into world_segs.
        # Vectorised map via dictionary lookup.
        unique, inverse = np.unique(names, return_inverse=True)
        try:
            seg_lookup = np.fromiter((seg_of[n] for n in unique), dtype=np.int64,
                                     count=len(unique))
        except KeyError as e:
            raise KeyError(f"material name not in seg_of: {e.args[0]!r}") from None
        world_view_s[mask] = seg_lookup[inverse]
        return int(mask.sum())


# ============================================================================
# @stamp_cached decorator (build-time perf)
# ============================================================================


def _freeze_kwargs(kwargs: dict) -> tuple:
    """Build a hashable cache key from kwargs that may contain unhashables.

    Lists / tuples / dicts are coerced through ``tuple`` recursively; numpy
    scalars cast to plain Python. Falls back to ``repr`` for any other type.
    """
    def freeze(v):
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        if isinstance(v, (list, tuple)):
            return tuple(freeze(x) for x in v)
        if isinstance(v, dict):
            return tuple(sorted((k, freeze(vv)) for k, vv in v.items()))
        if isinstance(v, np.generic):
            return v.item()
        return repr(v)
    return tuple(sorted((k, freeze(v)) for k, v in kwargs.items()))


def stamp_cached(n_variants: int = 8) -> Callable:
    """Wrap a ``build_*(rng, **kwargs) -> Stamp`` so it caches up to
    ``n_variants`` pre-built instances per (fn, kwargs) cache key.

    On the first ``n_variants`` calls the wrapper builds a fresh stamp using
    its own internal RNG (seeded once at module load) and caches it. On
    subsequent calls it returns one of the cached stamps chosen by the
    caller's ``rng``, giving each placement a stable but visually-varied
    appearance.

    Safe because :meth:`Stamp.paste_into` only reads the materials array,
    and rotation / mirror helpers all return fresh stamps — sharing a single
    ``Stamp`` instance across many placements is correct.

    Set ``n_variants=0`` (or omit the decorator) for builders whose voxel
    content *must* be unique per call — e.g. random-walk specimens or debris
    piles.
    """
    if n_variants <= 0:
        # No-op pass-through.
        def identity(fn): return fn
        return identity

    def wrap(fn):
        cache: dict = {}
        # One RNG per decorated builder so variants stay deterministic
        # across runs but unique per cache key.
        builder_rng = np.random.default_rng(
            hash(getattr(fn, "__qualname__", fn.__name__)) & 0xFFFFFFFF
        )

        @functools.wraps(fn)
        def wrapper(rng, **kwargs):
            key = _freeze_kwargs(kwargs)
            variants = cache.setdefault(key, [])
            if len(variants) < n_variants:
                fresh = fn(builder_rng, **kwargs).materialize()
                variants.append(fresh)
            # Pick by the caller's rng so placements look RNG-driven even
            # though we hand out cached instances.
            idx = int(rng.integers(0, len(variants))) if len(variants) > 1 else 0
            return variants[idx]

        wrapper._cache = cache  # exposed for tests / stats
        return wrapper
    return wrap


# ============================================================================
# Free-standing helpers used by many builders
# ============================================================================


def disc_offsets(radius: int) -> list:
    """Integer (dx, dy) offsets inside a filled disc of the given radius."""
    r = int(radius)
    return [
        (dx, dy)
        for dx in range(-r, r + 1)
        for dy in range(-r, r + 1)
        if dx * dx + dy * dy <= r * r
    ]


def ring_offsets(radius: int) -> list:
    """Integer (dx, dy) offsets forming a 1-voxel-thick discrete ring (annulus
    of inner radius r-1, outer r). Use for valve wheels, gauge faces."""
    r = int(radius)
    inner_sq = (r - 1) * (r - 1)
    outer_sq = r * r
    return [
        (dx, dy)
        for dx in range(-r, r + 1)
        for dy in range(-r, r + 1)
        if inner_sq < dx * dx + dy * dy <= outer_sq
    ]


def octagon_offsets(half_side: int) -> list:
    """Integer (dx, dy) offsets inside an axis-aligned octagon — a square of
    half-side ``r`` with the 4 corner ``r//2 × r//2`` quadrants knocked out.
    Used for "rounded" pipe and reactor cross-sections."""
    r = int(half_side)
    cut = r // 2
    out = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            # Corner cut: skip if both |dx| and |dy| are in the cut zone closest to a corner.
            if abs(dx) > r - cut and abs(dy) > r - cut:
                continue
            out.append((dx, dy))
    return out
