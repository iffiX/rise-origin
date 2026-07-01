# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Voxel-structure entities and their sub-characterizations.

A :class:`StructureVoxel` is a deformable/rigid body built from a voxel lattice.
It is composed of one or more :class:`StructureVoxelBody` blocks (each a dense
voxel grid with per-voxel material/segment/signal assignments),
:class:`StructureVoxelConstraint` joints between rigid segments, and an embedded
record and signal declaration.

These are the faithful, low-level characterization classes: per-voxel data is
given as flat row-major arrays. Procedural shape generators (spheres, lattices,
limbs) live in the ``create`` package and emit these classes.
"""

import math
import os
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from rise_origin import (
    RS_NULL_INDEX,
    RSC_StructureVoxel,
    RSC_StructureVoxelBody,
    RSC_StructureVoxelConstraint,
    RSC_StructureVoxelRecord,
    RSC_StructureVoxelSignal,
    RSE_StructureVoxelConstraintType,
    RSE_StructureVoxelHingeControlMode,
    RSE_StructureVoxelPrismaticControlMode,
    RSE_StructureVoxelRecordJointRange,
    RSE_StructureVoxelRecordLinkRange,
    RSE_StructureVoxelRecordRigidBodyRange,
    RSE_StructureVoxelRecordVoxelRange,
)

from rise_origin.rsc._utils import Quat3rf, Vec3rf

__all__ = [
    "VoxelSizeSpec",
    "StructureVoxelBody",
    "StructureVoxelConstraint",
    "StructureVoxelRecord",
    "StructureVoxelSignal",
    "StructureVoxel",
]

#: Per-body voxel resolution for a multi-body importer: one ``float`` applied to
#: every body, or a ``dict`` mapping each source part name to its voxel edge
#: length in meters. In the ``dict`` form the ``"*"`` key sets the default for
#: any part not listed explicitly.
VoxelSizeSpec = Union[float, Dict[str, float]]

# Generalized-winding-number threshold separating inside (~1) from outside (~0).
_INSIDE_THRESHOLD = 0.5


def _voxelize_mesh(
    mesh, voxel_size: float
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Voxelize a ``trimesh.Trimesh`` into a solid occupancy grid.

    Robust to non-watertight meshes via the generalized winding number, which
    stays near 1 inside the surface and near 0 outside regardless of holes,
    flipped normals, or non-manifold edges.

    Args:
        mesh: The triangle mesh to voxelize (a ``trimesh.Trimesh``).
        voxel_size: Edge length of one voxel, in meters.

    Returns:
        A ``(occupancy, min_corner)`` pair: ``occupancy`` is an ``(X, Y, Z)``
        boolean array of filled cells, and ``min_corner`` is the world-space
        position of cell ``(0, 0, 0)``'s minimum corner, in meters.
    """
    import igl

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # One voxel of padding so the surface never sits on the grid boundary.
    lo = vertices.min(axis=0) - voxel_size
    hi = vertices.max(axis=0) + voxel_size
    nx, ny, nz = np.maximum(1, np.ceil((hi - lo) / voxel_size).astype(np.int64))

    xs = lo[0] + (np.arange(nx) + 0.5) * voxel_size
    ys = lo[1] + (np.arange(ny) + 0.5) * voxel_size
    zs = lo[2] + (np.arange(nz) + 0.5) * voxel_size
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    query = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).astype(
        np.float64
    )

    winding = igl.fast_winding_number(vertices, faces, query)
    occupancy = (np.abs(winding) >= _INSIDE_THRESHOLD).reshape(nx, ny, nz)

    min_corner = (float(lo[0]), float(lo[1]), float(lo[2]))
    return occupancy, min_corner


class StructureVoxelBody(RSC_StructureVoxelBody):
    """A dense voxel grid block within a structure.

    Voxels fill an ``x_voxels`` by ``y_voxels`` by ``z_voxels`` lattice. The
    per-voxel arrays are row-major, ordered X fastest then Y then Z, with length
    ``x_voxels * y_voxels * z_voxels`` when provided.

    Args:
        relative_origin_position: Block origin relative to the structure origin,
            ``(x, y, z)`` in meters.
        relative_orientation: Block orientation relative to the structure,
            ``(x, y, z, w)``.
        x_voxels: Voxel count along X.
        y_voxels: Voxel count along Y.
        z_voxels: Voxel count along Z.
        material_reference_sid: Per-voxel material reference id; use
            ``RS_NULL_INDEX`` for an empty voxel.
        segment_bid: Per-voxel rigid segment id; ``RS_NULL_INDEX`` if
            unspecified.
        expansion_signal_sid: Per-voxel expansion signal id; ``RS_NULL_INDEX``
            for no expansion signal.
        expansion_control_mode: Per-expansion-signal control mode
            (``RSE_StructureExpansionControlMode`` values); empty defaults each
            to basic-position control.
        link: Per-voxel 6-bit face-link mask (bit 0 = +X, 1 = -X, 2 = +Y,
            3 = -Y, 4 = +Z, 5 = -Z); empty links all adjacent faces.
        voxel_size: Edge length of one voxel, in meters.
    """

    # Engine voxel flattening: X varies fastest, then Y, then Z.
    _ENGINE_FLATTEN_ORDER = (2, 1, 0)

    def __init__(
        self,
        *,
        relative_origin_position: Vec3rf = (0.0, 0.0, 0.0),
        relative_orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
        x_voxels: int = 0,
        y_voxels: int = 0,
        z_voxels: int = 0,
        material_reference_sid: Sequence[int] = (),
        segment_bid: Sequence[int] = (),
        expansion_signal_sid: Sequence[int] = (),
        expansion_control_mode: Sequence[int] = (),
        link: Sequence[int] = (),
        voxel_size: float = 0.0,
    ) -> None:
        super().__init__()
        self.relative_origin_position = relative_origin_position
        self.relative_orientation = relative_orientation
        self.x_voxels = x_voxels
        self.y_voxels = y_voxels
        self.z_voxels = z_voxels
        self.material_reference_sid = material_reference_sid
        self.segment_bid = segment_bid
        self.expansion_signal_sid = expansion_signal_sid
        if expansion_control_mode:
            self.expansion_control_mode = expansion_control_mode
        self.link = link
        self.voxel_size = voxel_size

    @classmethod
    def box(
        cls,
        x_voxels: int,
        y_voxels: int,
        z_voxels: int,
        *,
        material_sid: int = 0,
        segment_sid: int = 0,
        voxel_size: float,
        relative_origin_position: Vec3rf = (0.0, 0.0, 0.0),
        relative_orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
    ) -> "StructureVoxelBody":
        """A solid rectangular block of voxels.

        Args:
            x_voxels: Voxel count along X.
            y_voxels: Voxel count along Y.
            z_voxels: Voxel count along Z.
            material_sid: Material reference id for every voxel.
            segment_sid: Rigid segment id for every voxel.
            voxel_size: Edge length of one voxel, in meters.
            relative_origin_position: Min-corner relative to the structure
                origin, ``(x, y, z)`` in meters.
            relative_orientation: Body orientation ``(x, y, z, w)``.

        Returns:
            A populated :class:`StructureVoxelBody`.
        """
        occ = np.ones((x_voxels, y_voxels, z_voxels), dtype=bool)
        return cls.from_grid(
            occ,
            material_sid=material_sid,
            segment_sid=segment_sid,
            voxel_size=voxel_size,
            relative_origin_position=relative_origin_position,
            relative_orientation=relative_orientation,
        )

    @classmethod
    def sphere(
        cls,
        radius_voxels: int,
        *,
        material_sid: int = 0,
        segment_sid: int = 0,
        voxel_size: float,
        relative_origin_position: Vec3rf = (0.0, 0.0, 0.0),
        relative_orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
    ) -> "StructureVoxelBody":
        """A solid sphere of the given voxel radius.

        Args:
            radius_voxels: Sphere radius in voxels (grid is ``2*radius`` per side).
            material_sid: Material reference id for every voxel.
            segment_sid: Rigid segment id for every voxel.
            voxel_size: Edge length of one voxel, in meters.
            relative_origin_position: Min-corner relative to the structure
                origin, ``(x, y, z)`` in meters.
            relative_orientation: Body orientation ``(x, y, z, w)``.

        Returns:
            A populated :class:`StructureVoxelBody`.
        """
        return cls.from_grid(
            cls._make_sphere_mask(radius_voxels),
            material_sid=material_sid,
            segment_sid=segment_sid,
            voxel_size=voxel_size,
            relative_origin_position=relative_origin_position,
            relative_orientation=relative_orientation,
        )

    @classmethod
    def cylinder(
        cls,
        radius_voxels: int,
        height_voxels: int,
        *,
        axis: str = "z",
        material_sid: int = 0,
        segment_sid: int = 0,
        voxel_size: float,
        relative_origin_position: Vec3rf = (0.0, 0.0, 0.0),
        relative_orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
    ) -> "StructureVoxelBody":
        """A solid cylinder of ``height_voxels`` along ``axis``.

        Args:
            radius_voxels: Cross-section radius in voxels.
            height_voxels: Length along ``axis`` in voxels.
            axis: Cylinder axis, one of ``"x"`` / ``"y"`` / ``"z"``.
            material_sid: Material reference id for every voxel.
            segment_sid: Rigid segment id for every voxel.
            voxel_size: Edge length of one voxel, in meters.
            relative_origin_position: Min-corner relative to the structure
                origin, ``(x, y, z)`` in meters.
            relative_orientation: Body orientation ``(x, y, z, w)``.

        Returns:
            A populated :class:`StructureVoxelBody`.
        """
        return cls.from_grid(
            cls._make_cylinder_mask(radius_voxels, height_voxels, axis),
            material_sid=material_sid,
            segment_sid=segment_sid,
            voxel_size=voxel_size,
            relative_origin_position=relative_origin_position,
            relative_orientation=relative_orientation,
        )

    @classmethod
    def capsule(
        cls,
        radius_voxels: int,
        height_voxels: int,
        *,
        axis: str = "z",
        material_sid: int = 0,
        segment_sid: int = 0,
        voxel_size: float,
        relative_origin_position: Vec3rf = (0.0, 0.0, 0.0),
        relative_orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
    ) -> "StructureVoxelBody":
        """A solid capsule: a cylinder of ``height_voxels`` with hemispherical caps.

        The caps extend ``radius_voxels`` beyond each end, so the total span
        along ``axis`` is ``height_voxels + 2 * radius_voxels``.

        Args:
            radius_voxels: Cross-section and cap radius in voxels.
            height_voxels: Length of the straight cylindrical section in voxels.
            axis: Capsule axis, one of ``"x"`` / ``"y"`` / ``"z"``.
            material_sid: Material reference id for every voxel.
            segment_sid: Rigid segment id for every voxel.
            voxel_size: Edge length of one voxel, in meters.
            relative_origin_position: Min-corner relative to the structure
                origin, ``(x, y, z)`` in meters.
            relative_orientation: Body orientation ``(x, y, z, w)``.

        Returns:
            A populated :class:`StructureVoxelBody`.
        """
        return cls.from_grid(
            cls._make_capsule_mask(radius_voxels, height_voxels, axis),
            material_sid=material_sid,
            segment_sid=segment_sid,
            voxel_size=voxel_size,
            relative_origin_position=relative_origin_position,
            relative_orientation=relative_orientation,
        )

    @classmethod
    def from_grid(
        cls,
        occupancy,
        *,
        material_sid: int = 0,
        segment_sid: int = 0,
        voxel_size: float,
        material_grid=None,
        segment_grid=None,
        expansion_signal_grid=None,
        relative_origin_position: Vec3rf = (0.0, 0.0, 0.0),
        relative_orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
    ) -> "StructureVoxelBody":
        """Build a body from an occupancy grid (the kernel every generator uses).

        Occupied cells take ``material_sid`` / ``segment_sid``; empty cells are
        left unfilled (``RS_NULL_INDEX``). Pass ``material_grid`` / ``segment_grid``
        to assign these per voxel instead, overriding the scalar defaults.

        Args:
            occupancy: ``(X, Y, Z)`` boolean array; truthy cells are present.
            material_sid: Uniform material reference id for present voxels.
            segment_sid: Uniform rigid segment id for present voxels.
            voxel_size: Edge length of one voxel, in meters.
            material_grid: Optional ``(X, Y, Z)`` per-voxel material id grid;
                overrides ``material_sid``.
            segment_grid: Optional ``(X, Y, Z)`` per-voxel segment id grid;
                overrides ``segment_sid``.
            expansion_signal_grid: Optional ``(X, Y, Z)`` per-voxel expansion
                signal id grid; ``RS_NULL_INDEX`` for voxels with no expansion.
            relative_origin_position: Min-corner of the grid relative to the
                structure origin, ``(x, y, z)`` in meters.
            relative_orientation: Body orientation ``(x, y, z, w)``.

        Returns:
            A populated :class:`StructureVoxelBody`.
        """
        occ = np.asarray(occupancy, dtype=bool)
        mats = (
            np.asarray(material_grid)
            if material_grid is not None
            else np.where(occ, material_sid, RS_NULL_INDEX)
        )
        segs = (
            np.asarray(segment_grid)
            if segment_grid is not None
            else np.where(occ, segment_sid, RS_NULL_INDEX)
        )
        exp = (
            np.asarray(expansion_signal_grid)
            if expansion_signal_grid is not None
            else None
        )

        # Trim empty borders: crop to the tight box of voxels that carry data so
        # the dense arrays do not store the surrounding empty space, then shift
        # the origin so the kept voxels stay in the same place.
        present = occ | (mats != RS_NULL_INDEX) | (segs != RS_NULL_INDEX)
        if present.any():
            present_x, present_y, present_z = np.nonzero(present)
            x0, y0, z0 = int(present_x.min()), int(present_y.min()), int(present_z.min())
            x1 = int(present_x.max()) + 1
            y1 = int(present_y.max()) + 1
            z1 = int(present_z.max()) + 1
            box = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
            mats = mats[box]
            segs = segs[box]
            if exp is not None:
                exp = exp[box]
            shift = cls._rotate_vector(
                relative_orientation,
                (x0 * voxel_size, y0 * voxel_size, z0 * voxel_size),
            )
            base = np.asarray(relative_origin_position, dtype=np.float64) + shift
            relative_origin_position = (float(base[0]), float(base[1]), float(base[2]))

        nx, ny, nz = mats.shape
        expansion = cls._flatten(exp) if exp is not None else ()
        return cls(
            relative_origin_position=relative_origin_position,
            relative_orientation=relative_orientation,
            x_voxels=int(nx),
            y_voxels=int(ny),
            z_voxels=int(nz),
            material_reference_sid=cls._flatten(mats),
            segment_bid=cls._flatten(segs),
            expansion_signal_sid=expansion,
            voxel_size=voxel_size,
        )

    @classmethod
    def from_sdf(
        cls,
        fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        *,
        bounds: Tuple[Vec3rf, Vec3rf],
        voxel_size: float,
        material_sid: int = 0,
        segment_sid: int = 0,
        relative_orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
    ) -> "StructureVoxelBody":
        """A body filled where a signed-distance function is non-positive.

        ``fn`` is evaluated, vectorized, at every voxel center inside ``bounds``;
        cells where it returns ``<= 0`` are present. This subsumes analytic
        shapes (ellipsoid, torus, cone), CSG via ``min`` / ``max`` of distances,
        and swept tubes (distance to a polyline) without a dedicated generator.

        Args:
            fn: Signed-distance function taking three ``(X, Y, Z)`` coordinate
                arrays (body-local meters) and returning a distance array;
                negative inside the surface.
            bounds: ``(min_corner, max_corner)`` of the sampled box, each
                ``(x, y, z)`` in body-local meters.
            voxel_size: Edge length of one voxel, in meters.
            material_sid: Material reference id for present voxels.
            segment_sid: Rigid segment id for present voxels.
            relative_orientation: Body orientation ``(x, y, z, w)``.

        Returns:
            A populated :class:`StructureVoxelBody`.
        """
        (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds
        nx = max(1, int(round((max_x - min_x) / voxel_size)))
        ny = max(1, int(round((max_y - min_y) / voxel_size)))
        nz = max(1, int(round((max_z - min_z) / voxel_size)))
        xs = min_x + (np.arange(nx) + 0.5) * voxel_size
        ys = min_y + (np.arange(ny) + 0.5) * voxel_size
        zs = min_z + (np.arange(nz) + 0.5) * voxel_size
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        occ = np.asarray(fn(gx, gy, gz)) <= 0.0
        return cls.from_grid(
            occ,
            material_sid=material_sid,
            segment_sid=segment_sid,
            voxel_size=voxel_size,
            relative_origin_position=(min_x, min_y, min_z),
            relative_orientation=relative_orientation,
        )

    @classmethod
    def from_mesh(
        cls,
        mesh,
        *,
        voxel_size: float,
        material_sid: int = 0,
        segment_sid: int = 0,
        relative_orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
    ) -> "StructureVoxelBody":
        """A body voxelized from an in-memory triangle mesh.

        The mesh is voxelized into a solid via OpenVDB, which is robust to the
        non-watertight input common in art and CAD-tessellated assets. The body's
        ``relative_origin_position`` is taken from the voxel grid's world origin.
        Loading a mesh from a file is the caller's job (e.g. ``trimesh.load``);
        this is the shared voxelization kernel every structure-level importer
        feeds into.

        Args:
            mesh: The triangle mesh to voxelize (a ``trimesh.Trimesh``).
            voxel_size: Edge length of one voxel, in meters (the voxelization
                pitch).
            material_sid: Material reference id for present voxels.
            segment_sid: Rigid segment id for present voxels.
            relative_orientation: Body orientation ``(x, y, z, w)``.

        Returns:
            A populated :class:`StructureVoxelBody`.
        """
        occupancy, min_corner = _voxelize_mesh(mesh, voxel_size)
        return cls.from_grid(
            occupancy,
            material_sid=material_sid,
            segment_sid=segment_sid,
            voxel_size=voxel_size,
            relative_origin_position=min_corner,
            relative_orientation=relative_orientation,
        )

    @staticmethod
    def _flatten(grid) -> np.ndarray:
        """Flatten an (X, Y, Z) grid into the engine's X-fastest voxel order."""
        order = StructureVoxelBody._ENGINE_FLATTEN_ORDER
        return np.asarray(grid).transpose(order).ravel(order="C")

    @staticmethod
    def _rotate_vector(orientation, vector) -> np.ndarray:
        """Rotate a 3-vector by a quaternion given as (x, y, z, w)."""
        quat = np.asarray(orientation, dtype=np.float64)
        vec = np.asarray(vector, dtype=np.float64)
        q_vec, q_w = quat[:3], quat[3]
        crossed = 2.0 * np.cross(q_vec, vec)
        return vec + q_w * crossed + np.cross(q_vec, crossed)

    @staticmethod
    def _make_sphere_mask(radius: int) -> np.ndarray:
        """Solid-sphere occupancy of the given voxel radius, shape (2r, 2r, 2r)."""
        n = 2 * radius
        coords = np.arange(n) + 0.5 - radius
        gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
        return (gx * gx + gy * gy + gz * gz) <= float(radius) ** 2

    @staticmethod
    def _make_cylinder_mask(radius: int, length: int, axis: str) -> np.ndarray:
        """Solid-cylinder occupancy of ``length`` voxels along ``axis`` (x/y/z)."""
        n = 2 * radius
        coords = np.arange(n) + 0.5 - radius
        if axis == "z":
            gx, gy = np.meshgrid(coords, coords, indexing="ij")
            disk = (gx * gx + gy * gy) <= float(radius) ** 2
            return np.repeat(disk[:, :, None], length, axis=2)
        if axis == "y":
            gx, gz = np.meshgrid(coords, coords, indexing="ij")
            disk = (gx * gx + gz * gz) <= float(radius) ** 2
            return np.repeat(disk[:, None, :], length, axis=1)
        if axis == "x":
            gy, gz = np.meshgrid(coords, coords, indexing="ij")
            disk = (gy * gy + gz * gz) <= float(radius) ** 2
            return np.repeat(disk[None, :, :], length, axis=0)
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")

    @staticmethod
    def _make_capsule_mask(radius: int, length: int, axis: str) -> np.ndarray:
        """Solid-capsule occupancy: a ``length``-voxel cylinder with hemispherical caps."""
        n = 2 * radius
        total = length + 2 * radius
        perp = np.arange(n) + 0.5 - radius
        along = np.arange(total) + 0.5 - total / 2.0
        half = length / 2.0
        if axis == "z":
            gp, gq, ga = np.meshgrid(perp, perp, along, indexing="ij")
        elif axis == "y":
            gp, ga, gq = np.meshgrid(perp, along, perp, indexing="ij")
        elif axis == "x":
            ga, gp, gq = np.meshgrid(along, perp, perp, indexing="ij")
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        axial_excess = np.clip(np.abs(ga) - half, 0.0, None)
        return (gp * gp + gq * gq + axial_excess * axial_excess) <= float(radius) ** 2


class StructureVoxelConstraint(RSC_StructureVoxelConstraint):
    """A joint between two rigid segments of a structure.

    ``type`` selects the joint kind; only the matching field group (hinge or
    prismatic) is meaningful for the actuated joints.

    The two segments are addressed by their integer ids within the owning
    structure: ``a_body_sid`` / ``a_segment_bid`` and ``b_body_sid`` /
    ``b_segment_bid``.

    Args:
        type: Joint type (``RSE_StructureVoxelConstraintType``).
        is_collision_enabled: Whether the two joined bodies are allowed to
            collide with each other. Defaults to ``False``.
        a_body_sid: Structure-scope id of the first body.
        a_segment_bid: Segment id within the first body.
        b_body_sid: Structure-scope id of the second body.
        b_segment_bid: Segment id within the second body.
        relative_anchor: Joint anchor in structure-local coordinates,
            ``(x, y, z)`` in meters.
        hinge_control_mode: Hinge actuation mode
            (``RSE_StructureVoxelHingeControlMode``).
        hinge_relative_axis: Hinge axis; positive rotation follows the
            right-hand rule about it.
        hinge_min: Minimum hinge angle in radians.
        hinge_max: Maximum hinge angle in radians.
        hinge_position_control_kp: Proportional gain (PD position mode only).
        hinge_position_control_kd: Derivative gain (PD position mode only).
        hinge_max_speed: Maximum hinge speed in rad/s.
        hinge_max_torque: Maximum motor torque in N-m.
        hinge_signal_sid: Structure-scope signal id driving the hinge.
        prismatic_control_mode: Prismatic actuation mode
            (``RSE_StructureVoxelPrismaticControlMode``).
        prismatic_relative_axis: Translation axis of the prismatic joint.
        prismatic_min: Minimum prismatic distance in meters.
        prismatic_max: Maximum prismatic distance in meters.
        prismatic_position_control_kp: Proportional gain (PD position mode only).
        prismatic_position_control_kd: Derivative gain (PD position mode only).
        prismatic_max_speed: Maximum prismatic speed in m/s.
        prismatic_max_force: Maximum motor force in N.
        prismatic_signal_sid: Structure-scope signal id driving the prismatic
            joint.
    """

    def __init__(
        self,
        type: int,
        *,
        is_collision_enabled: bool = False,
        a_body_sid: int = RS_NULL_INDEX,
        a_segment_bid: int = RS_NULL_INDEX,
        b_body_sid: int = RS_NULL_INDEX,
        b_segment_bid: int = RS_NULL_INDEX,
        relative_anchor: Vec3rf = (0.0, 0.0, 0.0),
        hinge_control_mode: int = (
            RSE_StructureVoxelHingeControlMode.RSE_STRUCTURE_VOXEL_HINGE_CONTROL_BASIC_POSITION
        ),
        hinge_relative_axis: Vec3rf = (0.0, 0.0, 0.0),
        hinge_min: float = -math.pi,
        hinge_max: float = math.pi,
        hinge_position_control_kp: float = 0.0,
        hinge_position_control_kd: float = 0.0,
        hinge_max_speed: float = 10.0,
        hinge_max_torque: float = 0.0,
        hinge_signal_sid: int = RS_NULL_INDEX,
        prismatic_control_mode: int = (
            RSE_StructureVoxelPrismaticControlMode.RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_BASIC_POSITION
        ),
        prismatic_relative_axis: Vec3rf = (0.0, 0.0, 0.0),
        prismatic_min: float = -1.0,
        prismatic_max: float = 1.0,
        prismatic_position_control_kp: float = 0.0,
        prismatic_position_control_kd: float = 0.0,
        prismatic_max_speed: float = 10.0,
        prismatic_max_force: float = 0.0,
        prismatic_signal_sid: int = RS_NULL_INDEX,
    ) -> None:
        super().__init__()
        self.type = type
        self.is_collision_enabled = is_collision_enabled
        self.a_body_sid = a_body_sid
        self.a_segment_bid = a_segment_bid
        self.b_body_sid = b_body_sid
        self.b_segment_bid = b_segment_bid
        self.relative_anchor = relative_anchor
        self.hinge_control_mode = hinge_control_mode
        self.hinge_relative_axis = hinge_relative_axis
        self.hinge_min = hinge_min
        self.hinge_max = hinge_max
        self.hinge_position_control_kp = hinge_position_control_kp
        self.hinge_position_control_kd = hinge_position_control_kd
        self.hinge_max_speed = hinge_max_speed
        self.hinge_max_torque = hinge_max_torque
        self.hinge_signal_sid = hinge_signal_sid
        self.prismatic_control_mode = prismatic_control_mode
        self.prismatic_relative_axis = prismatic_relative_axis
        self.prismatic_min = prismatic_min
        self.prismatic_max = prismatic_max
        self.prismatic_position_control_kp = prismatic_position_control_kp
        self.prismatic_position_control_kd = prismatic_position_control_kd
        self.prismatic_max_speed = prismatic_max_speed
        self.prismatic_max_force = prismatic_max_force
        self.prismatic_signal_sid = prismatic_signal_sid


class StructureVoxelRecord(RSC_StructureVoxelRecord):
    """Selects which per-structure state to sample, and how to route it.

    Each ``*_record_range`` picks none, all, or (voxel/link only) a random
    sample of size ``*_sample_num``.

    Args:
        name: Record name (unique within the structure).
        voxel_record_range: Voxel sampling range
            (``RSE_StructureVoxelRecordVoxelRange``).
        link_record_range: Link sampling range
            (``RSE_StructureVoxelRecordLinkRange``).
        rigid_body_record_range: Rigid-body sampling range
            (``RSE_StructureVoxelRecordRigidBodyRange``).
        joint_record_range: Joint sampling range
            (``RSE_StructureVoxelRecordJointRange``).
        voxel_sample_num: Voxel count sampled when the voxel range is random.
        link_sample_num: Link count sampled when the link range is random.
        target_controller_reference: Controller that receives this record via
            the world callback. Empty disables controller delivery.
        target_recorder_reference: Recorder that persists this record. Empty
            disables recorder delivery.
    """

    def __init__(
        self,
        name: str = "default",
        *,
        voxel_record_range: int = (
            RSE_StructureVoxelRecordVoxelRange.RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_NONE
        ),
        link_record_range: int = (
            RSE_StructureVoxelRecordLinkRange.RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_NONE
        ),
        rigid_body_record_range: int = (
            RSE_StructureVoxelRecordRigidBodyRange.RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_NONE
        ),
        joint_record_range: int = (
            RSE_StructureVoxelRecordJointRange.RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_NONE
        ),
        voxel_sample_num: int = 0,
        link_sample_num: int = 0,
        target_controller_reference: str = "",
        target_recorder_reference: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.voxel_record_range = voxel_record_range
        self.link_record_range = link_record_range
        self.rigid_body_record_range = rigid_body_record_range
        self.joint_record_range = joint_record_range
        self.voxel_sample_num = voxel_sample_num
        self.link_sample_num = link_sample_num
        self.target_controller_reference = target_controller_reference
        self.target_recorder_reference = target_recorder_reference


class StructureVoxelSignal(RSC_StructureVoxelSignal):
    """Declares how many actuation signals the structure reads from a controller.

    Args:
        name: Signal name (unique within the structure).
        source_controller_reference: Controller that drives these signals each
            callback step.
        expansion_signal_num: Number of voxel-expansion signals.
        hinge_signal_num: Number of hinge-joint signals.
        prismatic_signal_num: Number of prismatic-joint signals.
    """

    def __init__(
        self,
        name: str = "default",
        *,
        source_controller_reference: str = "",
        expansion_signal_num: int = 0,
        hinge_signal_num: int = 0,
        prismatic_signal_num: int = 0,
    ) -> None:
        super().__init__()
        self.name = name
        self.source_controller_reference = source_controller_reference
        self.expansion_signal_num = expansion_signal_num
        self.hinge_signal_num = hinge_signal_num
        self.prismatic_signal_num = prismatic_signal_num


class StructureVoxel(RSC_StructureVoxel):
    """A voxel-lattice structure entity.

    Args:
        name: Unique entity name.
        origin_position: Initial world origin ``(x, y, z)`` in meters.
        orientation: Initial world orientation ``(x, y, z, w)``.
        material_references: Material names referenced by this structure's
            per-voxel ``material_reference_sid`` indices, in id order.
        is_fixed: If true, the structure is static and never moves.
        bodies: The voxel-grid blocks composing the structure.
        constraints: The joints between rigid segments of the structure.
        record: Embedded record. ``None`` keeps the default record.
        signal: Embedded signal declaration. ``None`` keeps the default signal.
    """

    # Dict key selecting the fallback voxel size for unlisted parts.
    _VOXEL_SIZE_DEFAULT_KEY = "*"

    # Surface color samples drawn per surface voxel, so each voxel's color is the
    # majority over its patch rather than one aliased point on a thin feature.
    _VOXEL_COLOR_SAMPLE_NUM = 12

    # Plain white clay (albedo, roughness, metalness) for the geometry-only look.
    _WHITE_SIGNATURE = ((0.8, 0.8, 0.8), 0.7, 0.0)

    # URDF joint type -> constraint type (unsupported types are skipped).
    _URDF_JOINT_TYPE = {
        "revolute": RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_HINGE_JOINT,
        "continuous": RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_HINGE_JOINT,
        "prismatic": RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT,
        "fixed": RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_FIXED_JOINT,
    }

    def __init__(
        self,
        name: str,
        *,
        origin_position: Vec3rf = (0.0, 0.0, 0.0),
        orientation: Quat3rf = (0.0, 0.0, 0.0, 1.0),
        material_references: Sequence[str] = (),
        is_fixed: bool = False,
        bodies: Sequence[StructureVoxelBody] = (),
        constraints: Sequence[StructureVoxelConstraint] = (),
        record: Optional[StructureVoxelRecord] = None,
        signal: Optional[StructureVoxelSignal] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.origin_position = origin_position
        self.orientation = orientation
        self.material_references.extend(material_references)
        self.is_fixed = is_fixed
        self.bodies = list(bodies)
        self.constraints = list(constraints)
        if record is not None:
            self.record = record
        if signal is not None:
            self.signal = signal

    @classmethod
    def from_mesh(
        cls,
        source,
        *,
        voxel_size: float,
        name: str = "mesh",
        material_sid: int = 0,
    ) -> "StructureVoxel":
        """A single-body structure voxelized from one triangle mesh.

        Args:
            source: A mesh file path (any format ``trimesh`` reads) or an
                in-memory ``trimesh.Trimesh``.
            voxel_size: Edge length of one voxel, in meters.
            name: Unique structure name.
            material_sid: Material reference id for every voxel.

        Returns:
            A single-body :class:`StructureVoxel`.
        """
        import trimesh

        mesh = (
            source
            if isinstance(source, trimesh.Trimesh)
            else trimesh.load(source, force="mesh")
        )
        body = StructureVoxelBody.from_mesh(
            mesh, voxel_size=voxel_size, material_sid=material_sid
        )
        return cls(name, bodies=[body])

    @classmethod
    def from_vox(
        cls,
        path: str,
        *,
        voxel_size: float,
        name: str = "voxel",
    ) -> "StructureVoxel":
        """A single-body structure loaded from a MagicaVoxel ``.vox`` file.

        The first model in the file is used. Each voxel's palette index becomes
        its material reference id (0-based), so distinct colors map to distinct
        materials the caller can name via the structure's ``material_references``.

        Args:
            path: Filesystem path to the ``.vox`` file.
            voxel_size: Physical edge length of one voxel, in meters.
            name: Unique structure name.

        Returns:
            A single-body :class:`StructureVoxel`.
        """
        (size_x, size_y, size_z), voxels = cls._parse_vox(path)
        material_grid = np.full((size_x, size_y, size_z), RS_NULL_INDEX, dtype=np.int64)
        xs, ys, zs, colors = voxels[:, 0], voxels[:, 1], voxels[:, 2], voxels[:, 3]
        material_grid[xs, ys, zs] = colors.astype(np.int64) - 1  # palette is 1-based
        occupancy = material_grid != RS_NULL_INDEX
        body = StructureVoxelBody.from_grid(
            occupancy, material_grid=material_grid, voxel_size=voxel_size
        )
        return cls(name, bodies=[body])

    @classmethod
    def from_urdf(
        cls,
        path: str,
        *,
        voxel_size: VoxelSizeSpec,
        name: Optional[str] = None,
        mesh_dir: Optional[str] = None,
        essence_outputs: Optional[list] = None,
        resource_outputs: Optional[list] = None,
        is_prefer_visual_mesh: bool = True,
        is_mimic_visual_signature: bool = True,
        visual_signature_num: int = 8,
    ) -> "StructureVoxel":
        """A multi-body articulated structure loaded from a URDF robot.

        Each link's geometry is voxelized into a rigid body placed at the link's
        pose (zero configuration); each joint becomes a constraint wiring its
        parent and child bodies (revolute/continuous -> hinge, prismatic ->
        prismatic, fixed -> fixed). Unsupported joint types and links without
        geometry are skipped.

        When ``essence_outputs`` is given, each link gets its own materials: the
        link's true mass, center of mass, and inertia become a shared atlas
        physical property, and its visual color and roughness/metalness become
        optical properties. The ``Material`` and property essences are appended
        to ``essence_outputs`` and the color textures to ``resource_outputs``, so
        the caller can hand both to the world. Otherwise every voxel keeps
        material id 0 and the caller supplies a material.

        Args:
            path: Filesystem path to the ``.urdf`` file.
            voxel_size: One ``float`` for every link, or a ``dict`` mapping link
                name to its voxel edge length in meters; in the ``dict`` form the
                ``"*"`` key sets the default for unlisted links.
            name: Unique structure name. Defaults to the robot's URDF name.
            mesh_dir: Directory the link mesh paths resolve against. Defaults to
                the directory holding the ``.urdf`` file.
            essence_outputs: List that receives the per-link ``Material`` and
                physical and optical property essences. ``None`` keeps every
                voxel at material id 0.
            resource_outputs: List that receives the per-link color textures the
                optical essences reference. Required when ``essence_outputs`` is
                given.
            is_prefer_visual_mesh: Voxelize the visual geometry when present,
                falling back to the collision geometry; collision first if false.
            is_mimic_visual_signature: Sample each link's texture into per-voxel
                colors when true; give every link one plain white clay material
                when false, for a geometry-only look.
            visual_signature_num: Maximum number of distinct visual signatures
                (color plus roughness and metalness) sampled per link.

        Returns:
            A multi-body :class:`StructureVoxel` with joints wired.
        """
        import yourdfpy

        is_extracting_materials = essence_outputs is not None
        if is_extracting_materials and resource_outputs is None:
            raise ValueError(
                "resource_outputs is required when essence_outputs is given"
            )

        base_dir = mesh_dir or os.path.dirname(os.path.abspath(path))
        resolve = cls._get_urdf_mesh_resolver(base_dir)
        robot = yourdfpy.URDF.load(
            path,
            build_scene_graph=True,
            load_meshes=False,
            load_collision_meshes=False,
            filename_handler=resolve,
        )
        base = robot.base_link
        structure_name = name or robot.robot.name

        bodies = []
        body_of_link = {}
        material_references = []
        for link_name, link in robot.link_map.items():
            mesh = cls._get_urdf_link_mesh(link, resolve, is_prefer_visual_mesh)
            if mesh is None:
                continue
            transform = np.asarray(
                robot.get_transform(frame_to=link_name, frame_from=base)
            )
            placed = mesh.copy()
            placed.apply_transform(transform)
            link_voxel_size = cls._resolve_size(voxel_size, link_name)

            if not is_extracting_materials:
                body = StructureVoxelBody.from_mesh(placed, voxel_size=link_voxel_size)
                bodies.append(body)
                body_of_link[link_name] = body
                continue

            occupancy, min_corner = _voxelize_mesh(placed, link_voxel_size)
            occupied_min = np.argwhere(occupancy).min(axis=0)
            com_corner = np.asarray(min_corner) + occupied_min * link_voxel_size
            inertial = cls._get_urdf_inertial(robot, link_name, base)
            if is_mimic_visual_signature:
                palette, index_grid = cls._get_voxel_signatures(
                    placed,
                    occupancy,
                    min_corner,
                    link_voxel_size,
                    cls._get_urdf_signature(robot, link),
                    visual_signature_num,
                )
            else:
                palette = [cls._WHITE_SIGNATURE]
                index_grid = np.zeros(occupancy.shape, dtype=np.int64)
            base_id = len(material_references)
            material_references.extend(
                cls._make_signature_materials(
                    f"{structure_name}/{link_name}",
                    inertial,
                    com_corner,
                    palette,
                    essence_outputs,
                    resource_outputs,
                )
            )
            material_grid = np.where(occupancy, base_id + index_grid, RS_NULL_INDEX)
            body = StructureVoxelBody.from_grid(
                occupancy,
                material_grid=material_grid,
                voxel_size=link_voxel_size,
                relative_origin_position=min_corner,
            )
            bodies.append(body)
            body_of_link[link_name] = body

        hinge_type = RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_HINGE_JOINT
        prismatic_type = (
            RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT
        )
        # Geometry-less links carry no body, so bridge a joint to the nearest
        # ancestor link that does, keeping the chain connected through them.
        parent_link_of = {j.child: j.parent for j in robot.joint_map.values()}
        constraints = []
        hinge_count = 0
        prismatic_count = 0
        for joint in robot.joint_map.values():
            constraint_type = cls._URDF_JOINT_TYPE.get(joint.type)
            if constraint_type is None:
                continue
            if joint.child not in body_of_link:
                continue
            parent_link = joint.parent
            while parent_link not in body_of_link and parent_link in parent_link_of:
                parent_link = parent_link_of[parent_link]
            if parent_link not in body_of_link:
                continue
            parent_body = body_of_link[parent_link]
            child_body = body_of_link[joint.child]
            child_transform = np.asarray(
                robot.get_transform(frame_to=joint.child, frame_from=base)
            )
            anchor = tuple(float(v) for v in child_transform[:3, 3])
            if constraint_type == hinge_type:
                axis = tuple(
                    float(v) for v in child_transform[:3, :3] @ np.asarray(joint.axis)
                )
                lower, upper = (
                    (joint.limit.lower, joint.limit.upper)
                    if joint.limit is not None
                    else (-math.pi, math.pi)
                )
                constraints.append(
                    StructureVoxelConstraint(
                        constraint_type,
                        a_body_sid=bodies.index(parent_body),
                        a_segment_bid=0,
                        b_body_sid=bodies.index(child_body),
                        b_segment_bid=0,
                        relative_anchor=anchor,
                        hinge_relative_axis=axis,
                        hinge_min=lower,
                        hinge_max=upper,
                        hinge_signal_sid=hinge_count,
                    )
                )
                hinge_count += 1
            elif constraint_type == prismatic_type:
                axis = tuple(
                    float(v) for v in child_transform[:3, :3] @ np.asarray(joint.axis)
                )
                lower, upper = (
                    (joint.limit.lower, joint.limit.upper)
                    if joint.limit is not None
                    else (-1.0, 1.0)
                )
                constraints.append(
                    StructureVoxelConstraint(
                        constraint_type,
                        a_body_sid=bodies.index(parent_body),
                        a_segment_bid=0,
                        b_body_sid=bodies.index(child_body),
                        b_segment_bid=0,
                        relative_anchor=anchor,
                        prismatic_relative_axis=axis,
                        prismatic_min=lower,
                        prismatic_max=upper,
                        prismatic_signal_sid=prismatic_count,
                    )
                )
                prismatic_count += 1
            else:
                constraints.append(
                    StructureVoxelConstraint(
                        constraint_type,
                        a_body_sid=bodies.index(parent_body),
                        a_segment_bid=0,
                        b_body_sid=bodies.index(child_body),
                        b_segment_bid=0,
                        relative_anchor=anchor,
                    )
                )

        signal = StructureVoxelSignal(
            structure_name + "_signal",
            hinge_signal_num=hinge_count,
            prismatic_signal_num=prismatic_count,
        )
        return cls(
            structure_name,
            material_references=material_references,
            bodies=bodies,
            constraints=constraints,
            signal=signal,
        )

    @classmethod
    def from_mjcf(
        cls,
        path: str,
        *,
        voxel_size: VoxelSizeSpec,
        name: str = "mjcf",
        essence_outputs: Optional[list] = None,
        resource_outputs: Optional[list] = None,
        is_prefer_visual_mesh: bool = True,
        is_mimic_visual_signature: bool = True,
        visual_signature_num: int = 8,
    ) -> "StructureVoxel":
        """A multi-body articulated structure loaded from a MuJoCo MJCF model.

        Each MuJoCo body's geoms are voxelized into a rigid body placed at the
        body's pose (zero configuration); each joint becomes a constraint wiring
        its body to its parent (hinge -> hinge, slide -> prismatic, ball ->
        ball-and-socket), and a jointless body is welded to its parent. Free
        joints, plane geoms, and bodies without geoms are skipped.

        When ``essence_outputs`` is given, each body gets its own materials: its
        mass, center of mass, and inertia become a shared atlas physical
        property, and its color and roughness/metalness become optical
        properties, appended to ``essence_outputs`` and ``resource_outputs``.
        Otherwise every voxel keeps material id 0 and the caller supplies a
        material.

        Args:
            path: Filesystem path to the ``.xml`` MJCF file.
            voxel_size: One ``float`` for every body, or a ``dict`` mapping body
                name to its voxel edge length in meters; in the ``dict`` form the
                ``"*"`` key sets the default for unlisted bodies.
            name: Unique structure name.
            essence_outputs: List that receives the per-body ``Material`` and
                physical and optical property essences. ``None`` keeps every
                voxel at material id 0.
            resource_outputs: List that receives the per-body color textures the
                optical essences reference. Required when ``essence_outputs`` is
                given.
            is_prefer_visual_mesh: Voxelize the visual geoms (no contact) when
                present, falling back to the collision geoms; collision first if
                false.
            is_mimic_visual_signature: Sample each body's color into per-voxel
                signatures when true; give every body one plain white clay
                material when false, for a geometry-only look.
            visual_signature_num: Maximum number of distinct visual signatures
                sampled per body.

        Returns:
            A multi-body :class:`StructureVoxel` with joints wired.
        """
        import mujoco

        is_extracting_materials = essence_outputs is not None
        if is_extracting_materials and resource_outputs is None:
            raise ValueError(
                "resource_outputs is required when essence_outputs is given"
            )

        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        bodies = []
        body_of_id = {}
        material_references = []
        for body_id in range(1, model.nbody):  # body 0 is the world
            mesh, geom_ids = cls._get_mjcf_body_mesh(
                model, data, body_id, is_prefer_visual_mesh
            )
            if mesh is None:
                continue
            body_name = model.body(body_id).name or f"body{body_id}"
            body_voxel_size = cls._resolve_size(voxel_size, body_name)

            if not is_extracting_materials:
                body = StructureVoxelBody.from_mesh(mesh, voxel_size=body_voxel_size)
                bodies.append(body)
                body_of_id[body_id] = body
                continue

            occupancy, min_corner = _voxelize_mesh(mesh, body_voxel_size)
            occupied_min = np.argwhere(occupancy).min(axis=0)
            com_corner = np.asarray(min_corner) + occupied_min * body_voxel_size
            inertial = cls._get_mjcf_inertial(model, data, body_id)
            if is_mimic_visual_signature:
                palette, index_grid = cls._get_voxel_signatures(
                    mesh,
                    occupancy,
                    min_corner,
                    body_voxel_size,
                    cls._get_mjcf_signature(model, geom_ids),
                    visual_signature_num,
                )
            else:
                palette = [cls._WHITE_SIGNATURE]
                index_grid = np.zeros(occupancy.shape, dtype=np.int64)
            base_id = len(material_references)
            material_references.extend(
                cls._make_signature_materials(
                    f"{name}/{body_name}",
                    inertial,
                    com_corner,
                    palette,
                    essence_outputs,
                    resource_outputs,
                )
            )
            material_grid = np.where(occupancy, base_id + index_grid, RS_NULL_INDEX)
            body = StructureVoxelBody.from_grid(
                occupancy,
                material_grid=material_grid,
                voxel_size=body_voxel_size,
                relative_origin_position=min_corner,
            )
            bodies.append(body)
            body_of_id[body_id] = body

        hinge_type = RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_HINGE_JOINT
        prismatic_type = (
            RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT
        )
        ball_type = (
            RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_BALL_AND_SOCKET_JOINT
        )
        fixed_type = RSE_StructureVoxelConstraintType.RSE_STRUCTURE_VOXEL_FIXED_JOINT
        constraints = []
        hinge_count = 0
        prismatic_count = 0
        for child_id in range(1, model.nbody):
            if child_id not in body_of_id:
                continue
            # Nearest ancestor body, skipping any geometry-less link in between.
            parent_id = int(model.body_parentid[child_id])
            while parent_id != 0 and parent_id not in body_of_id:
                parent_id = int(model.body_parentid[parent_id])
            if parent_id not in body_of_id:
                continue
            joint_ids = [
                j for j in range(model.njnt) if int(model.jnt_bodyid[j]) == child_id
            ]
            if any(
                int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE for j in joint_ids
            ):
                continue
            parent_body = body_of_id[parent_id]
            child_body = body_of_id[child_id]
            child_world = cls._make_pose_matrix(
                data.xpos[child_id], data.xquat[child_id]
            )

            # In the absence of joints the body is welded to its parent.
            if not joint_ids:
                constraints.append(
                    StructureVoxelConstraint(
                        fixed_type,
                        a_body_sid=bodies.index(parent_body),
                        a_segment_bid=0,
                        b_body_sid=bodies.index(child_body),
                        b_segment_bid=0,
                        relative_anchor=tuple(float(v) for v in data.xpos[child_id]),
                    )
                )
                continue

            joint_id = joint_ids[0]
            joint_type = int(model.jnt_type[joint_id])
            anchor = tuple(
                float(v)
                for v in (child_world @ np.append(model.jnt_pos[joint_id], 1.0))[:3]
            )
            axis = tuple(
                float(v)
                for v in child_world[:3, :3] @ np.asarray(model.jnt_axis[joint_id])
            )
            limited = bool(model.jnt_limited[joint_id])
            lower = float(model.jnt_range[joint_id][0])
            upper = float(model.jnt_range[joint_id][1])
            if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                constraints.append(
                    StructureVoxelConstraint(
                        hinge_type,
                        a_body_sid=bodies.index(parent_body),
                        a_segment_bid=0,
                        b_body_sid=bodies.index(child_body),
                        b_segment_bid=0,
                        relative_anchor=anchor,
                        hinge_relative_axis=axis,
                        hinge_min=lower if limited else -math.pi,
                        hinge_max=upper if limited else math.pi,
                        hinge_signal_sid=hinge_count,
                    )
                )
                hinge_count += 1
            elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                constraints.append(
                    StructureVoxelConstraint(
                        prismatic_type,
                        a_body_sid=bodies.index(parent_body),
                        a_segment_bid=0,
                        b_body_sid=bodies.index(child_body),
                        b_segment_bid=0,
                        relative_anchor=anchor,
                        prismatic_relative_axis=axis,
                        prismatic_min=lower if limited else -1.0,
                        prismatic_max=upper if limited else 1.0,
                        prismatic_signal_sid=prismatic_count,
                    )
                )
                prismatic_count += 1
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                constraints.append(
                    StructureVoxelConstraint(
                        ball_type,
                        a_body_sid=bodies.index(parent_body),
                        a_segment_bid=0,
                        b_body_sid=bodies.index(child_body),
                        b_segment_bid=0,
                        relative_anchor=anchor,
                    )
                )

        signal = StructureVoxelSignal(
            name + "_signal",
            hinge_signal_num=hinge_count,
            prismatic_signal_num=prismatic_count,
        )
        return cls(
            name,
            material_references=material_references,
            bodies=bodies,
            constraints=constraints,
            signal=signal,
        )

    @classmethod
    def from_gltf(
        cls,
        path: str,
        *,
        voxel_size: VoxelSizeSpec,
        name: str = "gltf",
        essence_outputs: Optional[list] = None,
        resource_outputs: Optional[list] = None,
        is_mimic_visual_signature: bool = True,
        visual_signature_num: int = 8,
    ) -> "StructureVoxel":
        """A structure voxelized from a glTF / GLB scene.

        Each mesh node becomes its own rigid body voxelized at the node
        transform; the bodies are unconnected, since glTF carries no joints (its
        skeleton and skin are bones and per-vertex weights for mesh deformation,
        not kinematic constraints, and are not interpreted). glTF has no inertials
        either, so the engine derives each body's mass from its voxels.

        When ``essence_outputs`` is given, each node gets its own materials: its
        texture color is sampled into per-voxel optical materials over a rigid
        physical property, appended to ``essence_outputs`` and
        ``resource_outputs``. Otherwise every voxel keeps material id 0 and the
        caller supplies a material.

        Args:
            path: Filesystem path to the ``.gltf`` / ``.glb`` file.
            voxel_size: One ``float`` for every node, or a ``dict`` mapping node
                name to its voxel edge length in meters; in the ``dict`` form the
                ``"*"`` key sets the default for unlisted nodes.
            name: Unique structure name.
            essence_outputs: List that receives the per-node ``Material`` and
                physical and optical property essences. ``None`` keeps every
                voxel at material id 0.
            resource_outputs: List that receives the per-node color textures the
                optical essences reference. Required when ``essence_outputs`` is
                given.
            is_mimic_visual_signature: Sample each node's texture into per-voxel
                signatures when true; give every node one plain white clay
                material when false, for a geometry-only look.
            visual_signature_num: Maximum number of distinct visual signatures
                sampled per node.

        Returns:
            A :class:`StructureVoxel` with one body per mesh node.
        """
        import trimesh

        is_extracting_materials = essence_outputs is not None
        if is_extracting_materials and resource_outputs is None:
            raise ValueError(
                "resource_outputs is required when essence_outputs is given"
            )

        scene = trimesh.load(path, force="scene")
        bodies = []
        material_references = []
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]
            mesh = scene.geometry[geometry_name].copy()
            mesh.apply_transform(transform)
            node_voxel_size = cls._resolve_size(voxel_size, node_name)

            if not is_extracting_materials:
                bodies.append(
                    StructureVoxelBody.from_mesh(mesh, voxel_size=node_voxel_size)
                )
                continue

            occupancy, min_corner = _voxelize_mesh(mesh, node_voxel_size)
            if is_mimic_visual_signature:
                palette, index_grid = cls._get_voxel_signatures(
                    mesh,
                    occupancy,
                    min_corner,
                    node_voxel_size,
                    cls._get_gltf_signature(mesh),
                    visual_signature_num,
                )
            else:
                palette = [cls._WHITE_SIGNATURE]
                index_grid = np.zeros(occupancy.shape, dtype=np.int64)
            base_id = len(material_references)
            material_references.extend(
                cls._make_signature_materials(
                    f"{name}/{node_name}",
                    None,
                    None,
                    palette,
                    essence_outputs,
                    resource_outputs,
                )
            )
            material_grid = np.where(occupancy, base_id + index_grid, RS_NULL_INDEX)
            bodies.append(
                StructureVoxelBody.from_grid(
                    occupancy,
                    material_grid=material_grid,
                    voxel_size=node_voxel_size,
                    relative_origin_position=min_corner,
                )
            )
        return cls(name, material_references=material_references, bodies=bodies)

    @classmethod
    def from_step(
        cls,
        path: str,
        *,
        voxel_size: VoxelSizeSpec,
        name: str = "step",
        tessellation_tolerance: float = 1e-3,
    ) -> "StructureVoxel":
        """A structure voxelized from a STEP CAD file.

        Each solid in the (possibly assembly) STEP is tessellated to a mesh and
        voxelized into its own unconnected rigid body; STEP carries no joints, so
        the bodies are left unwired (add joints afterward with the ``a`` / ``b``
        body refs). Only geometry is imported: per-solid color and appearance are
        not read, so the caller supplies materials. STEP carries no inertials, so
        the engine derives each body's mass from its voxels.

        Args:
            path: Filesystem path to the ``.step`` / ``.stp`` file.
            voxel_size: One ``float`` for every solid, or a ``dict`` mapping the
                solid's index (as ``"0"``, ``"1"``, ...) to its voxel edge length
                in meters; in the ``dict`` form the ``"*"`` key sets the default.
            name: Unique structure name.
            tessellation_tolerance: B-rep tessellation tolerance, in meters.

        Returns:
            A :class:`StructureVoxel` with one body per STEP solid.
        """
        import cadquery
        import trimesh

        solids = cadquery.importers.importStep(path).solids().vals()
        bodies = []
        for index, solid in enumerate(solids):
            vertices, faces = solid.tessellate(tessellation_tolerance)
            mesh = trimesh.Trimesh(
                vertices=np.asarray([(v.x, v.y, v.z) for v in vertices]),
                faces=np.asarray(faces),
            )
            body = StructureVoxelBody.from_mesh(
                mesh, voxel_size=cls._resolve_size(voxel_size, str(index))
            )
            bodies.append(body)
        return cls(name, bodies=bodies)

    @staticmethod
    def _parse_vox(path: str):
        """Parse the first model of a MagicaVoxel .vox file into (size, voxels)."""
        import struct

        with open(path, "rb") as handle:
            data = handle.read()
        if data[:4] != b"VOX ":
            raise ValueError(f"{path!r} is not a MagicaVoxel .vox file")
        size = None
        cursor = 8  # past the 'VOX ' magic and the version int32
        while cursor + 12 <= len(data):
            chunk_id = data[cursor : cursor + 4]
            content_len = struct.unpack_from("<i", data, cursor + 4)[0]
            content = data[cursor + 12 : cursor + 12 + content_len]
            if chunk_id == b"SIZE":
                size = struct.unpack_from("<iii", content, 0)
            elif chunk_id == b"XYZI":
                count = struct.unpack_from("<i", content, 0)[0]
                voxels = np.frombuffer(
                    content[4 : 4 + count * 4], dtype=np.uint8
                ).reshape(count, 4)
                return size, voxels
            cursor += 12 + content_len  # descend into chunk children, which follow
        raise ValueError(f"{path!r} contains no voxel data")

    @staticmethod
    def _resolve_size(voxel_size: VoxelSizeSpec, name: str) -> float:
        """Resolve a per-body voxel size from a scalar or a name-keyed dict."""
        if isinstance(voxel_size, dict):
            if name in voxel_size:
                return voxel_size[name]
            default_key = StructureVoxel._VOXEL_SIZE_DEFAULT_KEY
            if default_key in voxel_size:
                return voxel_size[default_key]
            raise KeyError(
                f"voxel_size has no entry for {name!r} and no {default_key!r} default"
            )
        return voxel_size

    @staticmethod
    def _get_urdf_mesh_resolver(mesh_dir: str) -> Callable[[str], str]:
        """Return a handler that resolves a URDF mesh path against ``mesh_dir``.

        The whole path after any ``package://`` directive is kept and joined with
        the mesh directory, the convention Bullet and pybullet descriptions use.
        """

        def resolve(fname: str) -> str:
            candidate = fname.split("://", 1)[1] if "://" in fname else fname
            if not os.path.isabs(candidate):
                candidate = os.path.join(mesh_dir, candidate)
            return candidate

        return resolve

    @staticmethod
    def _get_urdf_link_mesh(link, resolve, is_prefer_visual: bool):
        """Return a link's preferred mesh, falling back to the other geometry set.

        Tries visual or collision geometry first per ``is_prefer_visual``, then
        the other, so a link missing the preferred geometry, or whose preferred
        mesh files are absent, still yields a body.
        """
        primary = link.visuals if is_prefer_visual else link.collisions
        fallback = link.collisions if is_prefer_visual else link.visuals
        for items in (primary, fallback):
            try:
                mesh = StructureVoxel._get_urdf_geometries_mesh(items, resolve)
            except Exception:
                mesh = None
            if mesh is not None:
                return mesh
        return None

    @staticmethod
    def _get_urdf_geometries_mesh(items, resolve: Callable[[str], str]):
        """Combine a list of URDF visual or collision geometries into one trimesh.

        Returns ``None`` when the list holds no usable geometry. Both visual and
        collision items expose ``geometry`` and ``origin``, so the same builder
        serves either.
        """
        import trimesh

        parts = []
        for item in items:
            geometry = item.geometry
            if geometry.box is not None:
                primitive = trimesh.creation.box(extents=tuple(geometry.box.size))
            elif geometry.cylinder is not None:
                primitive = trimesh.creation.cylinder(
                    radius=geometry.cylinder.radius, height=geometry.cylinder.length
                )
            elif geometry.sphere is not None:
                primitive = trimesh.creation.icosphere(radius=geometry.sphere.radius)
            elif geometry.mesh is not None:
                primitive = trimesh.load(resolve(geometry.mesh.filename), force="mesh")
                if geometry.mesh.scale is not None:
                    primitive.apply_scale(geometry.mesh.scale)
            else:
                continue
            if item.origin is not None:
                primitive = primitive.copy()
                primitive.apply_transform(np.asarray(item.origin))
            parts.append(primitive)
        if not parts:
            return None
        return trimesh.util.concatenate(parts)

    @staticmethod
    def _get_urdf_inertial(robot, link_name, base):
        """Return a link's ``(mass, com, inertia)`` in the structure frame, or None.

        ``com`` is the center of mass and ``inertia`` the 3x3 tensor about it,
        both transformed from the link frame into the structure frame at the zero
        configuration.
        """
        inertial = getattr(robot.link_map[link_name], "inertial", None)
        if inertial is None:
            return None
        link_transform = np.asarray(
            robot.get_transform(frame_to=link_name, frame_from=base)
        )
        com_transform = link_transform @ np.asarray(inertial.origin)
        com = com_transform[:3, 3]
        rotation = com_transform[:3, :3]
        inertia = rotation @ np.asarray(inertial.inertia) @ rotation.T
        return float(inertial.mass), com, inertia

    @staticmethod
    def _get_urdf_signature(robot, link):
        """Return a link's ``(albedo, roughness, metalness)`` visual signature.

        URDF materials carry only a base color, so roughness defaults to a matte
        value and metalness to zero. Named materials are resolved through the
        robot's material map.
        """
        albedo = (0.7, 0.7, 0.7)
        named = {material.name: material for material in robot.robot.materials}
        for visual in link.visuals:
            material = visual.material
            if material is None:
                continue
            if material.color is None and material.name in named:
                material = named[material.name]
            if material.color is not None:
                rgba = material.color.rgba
                albedo = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
                break
        return albedo, 0.7, 0.0

    @staticmethod
    def _get_mjcf_body_mesh(model, data, body_id, is_prefer_visual):
        """Return a body's preferred geom mesh and the geom ids it used, or None.

        Geoms split into visual-only (no contact) and collision sets; the
        preferred set is tried first and the other as a fallback, and the merged
        mesh is placed at the body's world pose.
        """
        import trimesh

        visual, collision = [], []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            is_visual = (
                int(model.geom_contype[geom_id]) == 0
                and int(model.geom_conaffinity[geom_id]) == 0
            )
            (visual if is_visual else collision).append(geom_id)
        for geom_ids in (
            (visual, collision) if is_prefer_visual else (collision, visual)
        ):
            parts = [
                StructureVoxel._make_mjcf_geom_mesh(model, geom_id)
                for geom_id in geom_ids
            ]
            parts = [part for part in parts if part is not None]
            if not parts:
                continue
            mesh = trimesh.util.concatenate(parts)
            mesh.apply_transform(
                StructureVoxel._make_pose_matrix(
                    data.xpos[body_id], data.xquat[body_id]
                )
            )
            return mesh, geom_ids
        return None, []

    @staticmethod
    def _get_mjcf_inertial(model, data, body_id):
        """Return a body's ``(mass, com, inertia)`` in the structure frame, or None.

        MuJoCo already resolves the world-frame center of mass and principal
        inertia frame, so the tensor about the center of mass follows from
        rotating the diagonal inertia into that frame.
        """
        mass = float(model.body_mass[body_id])
        if mass <= 0:
            return None
        com = np.asarray(data.xipos[body_id])
        rotation = np.asarray(data.ximat[body_id]).reshape(3, 3)
        inertia = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
        return mass, com, inertia

    @staticmethod
    def _get_mjcf_signature(model, geom_ids):
        """Return a body's ``(albedo, roughness, metalness)`` visual signature.

        Color comes from the body's material or geom rgba; a material's shininess
        and reflectance map to roughness and metalness, with a matte default
        otherwise.
        """
        for geom_id in geom_ids:
            material_id = int(model.geom_matid[geom_id])
            if material_id >= 0:
                rgba = model.mat_rgba[material_id]
                roughness = float(1.0 - model.mat_shininess[material_id])
                metalness = float(model.mat_reflectance[material_id])
                return (
                    (float(rgba[0]), float(rgba[1]), float(rgba[2])),
                    roughness,
                    metalness,
                )
            rgba = model.geom_rgba[geom_id]
            return (float(rgba[0]), float(rgba[1]), float(rgba[2])), 0.7, 0.0
        return (0.7, 0.7, 0.7), 0.7, 0.0

    @staticmethod
    def _get_gltf_signature(mesh):
        """Return a mesh's ``(albedo, roughness, metalness)`` PBR fallback.

        Used when the mesh carries no texture; the base color and the roughness
        and metalness factors come from its glTF material, with a matte default
        otherwise.
        """
        material = getattr(getattr(mesh, "visual", None), "material", None)
        albedo = (0.7, 0.7, 0.7)
        roughness = 0.7
        metalness = 0.0
        if material is not None:
            base = getattr(material, "baseColorFactor", None)
            if base is None:
                base = getattr(material, "main_color", None)
            if base is not None:
                base = np.asarray(base, dtype=np.float64)
                if base.max() > 1.0:
                    base = base / 255.0
                albedo = (float(base[0]), float(base[1]), float(base[2]))
            if getattr(material, "roughnessFactor", None) is not None:
                roughness = float(material.roughnessFactor)
            if getattr(material, "metallicFactor", None) is not None:
                metalness = float(material.metallicFactor)
        return albedo, roughness, metalness

    @staticmethod
    def _get_mesh_texture(mesh):
        """Return a textured mesh's ``(uv, image)``, or None when it has none.

        Only a real texture image with vertex UVs counts; an untextured or
        primitive mesh returns None so the caller keeps the material's flat color.
        """
        visual = getattr(mesh, "visual", None)
        uv = getattr(visual, "uv", None)
        material = getattr(visual, "material", None)
        image = getattr(material, "image", None) if material is not None else None
        if image is None and material is not None:
            image = getattr(material, "baseColorTexture", None)
        if uv is None or image is None:
            return None
        return np.asarray(uv), image

    @staticmethod
    def _get_image_palette(image, count):
        """Return up to ``count`` median-cut palette colors of a texture, 0-1 RGB.

        The palette is read from a downsampled copy, which is far cheaper than
        quantizing the full image and gives the same dominant colors. Snapping
        samples to this palette keeps anti-aliased edges crisp.
        """
        thumbnail = image.convert("RGB")
        thumbnail.thumbnail((256, 256))
        quantized = thumbnail.quantize(colors=count)
        indices = sorted(index for _, index in quantized.getcolors())
        palette = np.asarray(quantized.getpalette(), dtype=np.float64).reshape(-1, 3)
        return palette[indices] / 255.0

    @staticmethod
    def _get_surface_samples(mesh, uv, image, count):
        """Draw ``count`` area-uniform surface points with their texture colors.

        Points are spread across the mesh in proportion to face area, and the
        texture is read at each point's barycentric-interpolated UV, so the
        samples represent the whole visible surface rather than the vertices.
        """
        import trimesh

        points, face = mesh.sample(count, return_index=True)
        barycentric = trimesh.triangles.points_to_barycentric(
            mesh.triangles[face], points
        )
        point_uv = (barycentric[:, :, None] * uv[mesh.faces[face]]).sum(axis=1)
        colors = np.asarray(trimesh.visual.uv_to_color(point_uv, image))
        return points, colors[:, :3].astype(np.float64) / 255.0

    @staticmethod
    def _get_voxel_signatures(
        mesh, occupancy, min_corner, voxel_size, default_signature, signature_num
    ):
        """Return a body's signature palette and its per-voxel palette-index grid.

        The palette lists up to ``signature_num`` distinct ``(albedo, roughness,
        metalness)`` signatures the body's voxels reference, ordered most common
        first; the grid holds each voxel's palette index. The texture is first
        quantized to that many colors, then a textured mesh paints each voxel with
        the majority color of the surface samples that fall in it, leaving hidden
        interior voxels at the dominant signature 0; otherwise one flat signature
        covers the body.
        """
        _, roughness, metalness = default_signature
        texture = StructureVoxel._get_mesh_texture(mesh)
        if texture is None or signature_num <= 1:
            return [default_signature], np.zeros(occupancy.shape, dtype=np.int64)
        uv, image = texture
        palette_colors = StructureVoxel._get_image_palette(image, signature_num)
        sample_num = max(
            int(mesh.area / (voxel_size * voxel_size))
            * StructureVoxel._VOXEL_COLOR_SAMPLE_NUM,
            signature_num,
        )
        points, colors = StructureVoxel._get_surface_samples(
            mesh, uv, image, sample_num
        )

        # Snap each sample to its nearest palette color, then keep only the
        # palette colors this body actually uses, most common first.
        distances = ((colors[:, None, :] - palette_colors[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        order = np.argsort(-np.bincount(labels, minlength=len(palette_colors)))
        order = order[np.bincount(labels, minlength=len(palette_colors))[order] > 0]
        palette_colors = palette_colors[order]
        remap = np.zeros(len(distances[0]), dtype=np.int64)
        remap[order] = np.arange(len(order))
        labels = remap[labels]

        voxel = np.clip(
            ((points - np.asarray(min_corner)) / voxel_size).astype(np.int64),
            0,
            np.asarray(occupancy.shape) - 1,
        )
        flat = np.ravel_multi_index(voxel.T, occupancy.shape)
        unique_voxels, inverse = np.unique(flat, return_inverse=True)
        votes = np.zeros((len(unique_voxels), len(palette_colors)), dtype=np.int64)
        np.add.at(votes, (inverse, labels), 1)
        index_grid = np.zeros(occupancy.shape, dtype=np.int64)
        index_grid.flat[unique_voxels] = votes.argmax(axis=1)

        palette = [
            ((float(red), float(green), float(blue)), roughness, metalness)
            for red, green, blue in palette_colors
        ]
        return palette, index_grid

    @staticmethod
    def _make_signature_materials(
        name, inertial, com_corner, palette, essence_outputs, resource_outputs
    ):
        """Append one material per signature to the out-lists and return the names.

        Every signature of a body shares the same atlas (or rigid) physical
        property, so the voxels fuse into one rigid segment, while each gets its
        own albedo and roughness/metalness texture.
        """
        from rise_origin.rsc import essence, resource

        names = []
        is_multi = len(palette) > 1
        for index, (albedo, roughness, metalness) in enumerate(palette):
            material_name = f"{name}#{index}" if is_multi else name
            names.append(material_name)
            essence_outputs.append(essence.Material(material_name))
            if inertial is not None:
                mass, com, inertia = inertial
                com_local = com - com_corner
                essence_outputs.append(
                    essence.MaterialPhysicalPropertyAtlas(
                        material_name + "#phys",
                        material_reference=material_name,
                        mass=mass,
                        com_position=(
                            float(com_local[0]),
                            float(com_local[1]),
                            float(com_local[2]),
                        ),
                        inertia=inertia,
                    )
                )
            else:
                essence_outputs.append(
                    essence.MaterialPhysicalPropertyRigid(
                        material_name + "#phys",
                        material_reference=material_name,
                        density=1000.0,
                    )
                )
            albedo_texture = material_name + "#albedo"
            resource_outputs.append(
                resource.TextureConstant(albedo_texture, value=tuple(albedo))
            )
            roughness_metalness_texture = material_name + "#rm"
            resource_outputs.append(
                resource.TextureConstant(
                    roughness_metalness_texture, value=(roughness, metalness, 0.0)
                )
            )
            essence_outputs.append(
                essence.MaterialOpticalPropertySimpleMicrofacet(
                    material_name + "#optic",
                    material_reference=material_name,
                    albedo_texture_reference=albedo_texture,
                    roughness_metalness_texture_reference=roughness_metalness_texture,
                )
            )
        return names

    @staticmethod
    def _make_pose_matrix(position, quat_wxyz) -> np.ndarray:
        """Compose a position and a ``(w, x, y, z)`` quaternion into a 4x4 transform."""
        import trimesh

        transform = trimesh.transformations.quaternion_matrix(np.asarray(quat_wxyz))
        transform[:3, 3] = position
        return transform

    @staticmethod
    def _make_mjcf_geom_mesh(model, geom_id: int):
        """Build a MuJoCo geom's trimesh in body-local frame, or None to skip."""
        import mujoco
        import trimesh

        geom_type = int(model.geom_type[geom_id])
        size = model.geom_size[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            primitive = trimesh.creation.box(extents=2.0 * np.asarray(size))
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            primitive = trimesh.creation.icosphere(radius=size[0])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            primitive = trimesh.creation.capsule(height=2.0 * size[1], radius=size[0])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            primitive = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            data_id = int(model.geom_dataid[geom_id])
            vert_start = int(model.mesh_vertadr[data_id])
            vert_count = int(model.mesh_vertnum[data_id])
            face_start = int(model.mesh_faceadr[data_id])
            face_count = int(model.mesh_facenum[data_id])
            primitive = trimesh.Trimesh(
                vertices=np.asarray(
                    model.mesh_vert[vert_start : vert_start + vert_count]
                ),
                faces=np.asarray(model.mesh_face[face_start : face_start + face_count]),
            )
        else:
            return None
        primitive.apply_transform(
            StructureVoxel._make_pose_matrix(
                model.geom_pos[geom_id], model.geom_quat[geom_id]
            )
        )
        return primitive
