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

from rise_origin.rsc._utils import Quat3rf, Vec3rf, _fill, _quat3rf, _vec3rf

__all__ = [
    "BodyRef",
    "VoxelSizeSpec",
    "StructureVoxelBody",
    "StructureVoxelConstraint",
    "StructureVoxelRecord",
    "StructureVoxelSignal",
    "StructureVoxel",
]

#: A reference to one rigid segment of a body, used by constraint endpoints and
#: camera / region links: a :class:`StructureVoxelBody` (its segment 0), or a
#: ``(body, segment_bid)`` tuple to pick a non-default segment. Resolved to ids
#: by identity at structure assembly, so no manual id bookkeeping is needed.
BodyRef = Union["StructureVoxelBody", Tuple["StructureVoxelBody", int]]

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
        self.relative_origin_position = _vec3rf(relative_origin_position)
        self.relative_orientation = _quat3rf(relative_orientation)
        self.x_voxels = x_voxels
        self.y_voxels = y_voxels
        self.z_voxels = z_voxels
        _fill(self.material_reference_sid, material_reference_sid)
        _fill(self.segment_bid, segment_bid)
        _fill(self.expansion_signal_sid, expansion_signal_sid)
        if expansion_control_mode:
            self.expansion_control_mode = list(expansion_control_mode)
        _fill(self.link, link)
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
            cls._sphere_mask(radius_voxels),
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
            cls._cylinder_mask(radius_voxels, height_voxels, axis),
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
            cls._capsule_mask(radius_voxels, height_voxels, axis),
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
        nx, ny, nz = occ.shape
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
        expansion = (
            cls._flatten(np.asarray(expansion_signal_grid)).tolist()
            if expansion_signal_grid is not None
            else ()
        )
        return cls(
            relative_origin_position=relative_origin_position,
            relative_orientation=relative_orientation,
            x_voxels=int(nx),
            y_voxels=int(ny),
            z_voxels=int(nz),
            material_reference_sid=cls._flatten(mats).tolist(),
            segment_bid=cls._flatten(segs).tolist(),
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
    def _sphere_mask(radius: int) -> np.ndarray:
        """Solid-sphere occupancy of the given voxel radius, shape (2r, 2r, 2r)."""
        n = 2 * radius
        coords = np.arange(n) + 0.5 - radius
        gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
        return (gx * gx + gy * gy + gz * gz) <= float(radius) ** 2

    @staticmethod
    def _cylinder_mask(radius: int, length: int, axis: str) -> np.ndarray:
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
    def _capsule_mask(radius: int, length: int, axis: str) -> np.ndarray:
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

    @staticmethod
    def _resolve_body_link(attached_to: "BodyRef") -> Tuple[str, int, int]:
        """Resolve a body ref into ``(structure_name, body_sid, segment_bid)``.

        The body must already be added to a :class:`StructureVoxel` (which stamps
        it with its structure name and body id); cameras and rigid-body-linked
        regions read those stamps to wire themselves without id bookkeeping.
        """
        body, segment_bid = (
            attached_to if isinstance(attached_to, tuple) else (attached_to, 0)
        )
        try:
            return body._structure_name, body._body_sid, segment_bid
        except AttributeError as error:
            raise ValueError(
                "attached_to body must be added to a StructureVoxel before linking"
            ) from error


class StructureVoxelConstraint(RSC_StructureVoxelConstraint):
    """A joint between two rigid segments of a structure.

    ``type`` selects the joint kind; only the matching field group (hinge or
    prismatic) is meaningful for the actuated joints.

    The two segments can be addressed two ways. Pass ``a`` / ``b`` as the
    :class:`StructureVoxelBody` objects themselves (or ``(body, segment_bid)``
    tuples) and the owning :class:`StructureVoxel` resolves them to ids by
    identity at assembly -- no manual id bookkeeping. Or set the
    ``a_body_sid`` / ``a_segment_bid`` / ``b_body_sid`` / ``b_segment_bid``
    integers directly. Object refs, when given, win.

    Args:
        type: Joint type (``RSE_StructureVoxelConstraintType``).
        is_collision_enabled: Whether the two joined bodies are allowed to
            collide with each other. Defaults to ``False``.
        a: First segment as a body object or ``(body, segment_bid)`` tuple;
            resolved at structure assembly. ``None`` uses the ``a_*`` integers.
        b: Second segment as a body object or ``(body, segment_bid)`` tuple;
            resolved at structure assembly. ``None`` uses the ``b_*`` integers.
        a_body_sid: Body id of the first segment (ignored when ``a`` is given).
        a_segment_bid: Segment id within the first body.
        b_body_sid: Body id of the second segment (ignored when ``b`` is given).
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
        a: Optional[BodyRef] = None,
        b: Optional[BodyRef] = None,
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
        # Pending body-object endpoints; StructureVoxel resolves them by identity.
        self._a_ref = a
        self._b_ref = b
        self.a_body_sid = a_body_sid
        self.a_segment_bid = a_segment_bid
        self.b_body_sid = b_body_sid
        self.b_segment_bid = b_segment_bid
        self.relative_anchor = _vec3rf(relative_anchor)
        self.hinge_control_mode = hinge_control_mode
        self.hinge_relative_axis = _vec3rf(hinge_relative_axis)
        self.hinge_min = hinge_min
        self.hinge_max = hinge_max
        self.hinge_position_control_kp = hinge_position_control_kp
        self.hinge_position_control_kd = hinge_position_control_kd
        self.hinge_max_speed = hinge_max_speed
        self.hinge_max_torque = hinge_max_torque
        self.hinge_signal_sid = hinge_signal_sid
        self.prismatic_control_mode = prismatic_control_mode
        self.prismatic_relative_axis = _vec3rf(prismatic_relative_axis)
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
        source_controller_name: Controller that drives these signals each
            callback step.
        expansion_signal_num: Number of voxel-expansion signals.
        hinge_signal_num: Number of hinge-joint signals.
        prismatic_signal_num: Number of prismatic-joint signals.
    """

    def __init__(
        self,
        name: str = "default",
        *,
        source_controller_name: str = "",
        expansion_signal_num: int = 0,
        hinge_signal_num: int = 0,
        prismatic_signal_num: int = 0,
    ) -> None:
        super().__init__()
        self.name = name
        self.source_controller_name = source_controller_name
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
        self.origin_position = _vec3rf(origin_position)
        self.orientation = _quat3rf(orientation)
        _fill(self.material_references, material_references)
        self.is_fixed = is_fixed

        # Stamp each body with its assembly id so object refs (constraint
        # endpoints, camera/region links) resolve by identity.
        bodies = list(bodies)
        index = {id(body): i for i, body in enumerate(bodies)}
        for i, body in enumerate(bodies):
            body._body_sid = i
            body._structure_name = name

        # Resolve constraint endpoints given as body objects into ids.
        constraints = list(constraints)
        for constraint in constraints:
            if getattr(constraint, "_a_ref", None) is not None:
                constraint.a_body_sid, constraint.a_segment_bid = (
                    self._resolve_endpoint(constraint._a_ref, index)
                )
            if getattr(constraint, "_b_ref", None) is not None:
                constraint.b_body_sid, constraint.b_segment_bid = (
                    self._resolve_endpoint(constraint._b_ref, index)
                )

        _fill(self.bodies, bodies)
        _fill(self.constraints, constraints)
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
    ) -> "StructureVoxel":
        """A multi-body articulated structure loaded from a URDF robot.

        Each link's collision geometry is voxelized into a rigid body placed at
        the link's pose (zero configuration); each joint becomes a constraint
        wiring its parent and child bodies (revolute/continuous -> hinge,
        prismatic -> prismatic, fixed -> fixed). Unsupported joint types and
        links without collision geometry are skipped. Only collision meshes are
        loaded.

        Args:
            path: Filesystem path to the ``.urdf`` file.
            voxel_size: One ``float`` for every link, or a ``dict`` mapping link
                name to its voxel edge length in meters; in the ``dict`` form the
                ``"*"`` key sets the default for unlisted links.
            name: Unique structure name. Defaults to the robot's URDF name.
            mesh_dir: Directory the link mesh paths resolve against. Defaults to
                the directory holding the ``.urdf`` file.

        Returns:
            A multi-body :class:`StructureVoxel` with joints wired.
        """
        import yourdfpy

        base_dir = mesh_dir or os.path.dirname(os.path.abspath(path))
        resolve = cls._make_mesh_resolver(base_dir)
        robot = yourdfpy.URDF.load(
            path,
            build_scene_graph=True,
            load_meshes=False,
            load_collision_meshes=False,
            filename_handler=resolve,
        )
        base = robot.base_link

        bodies = []
        body_of_link = {}
        for link_name, link in robot.link_map.items():
            mesh = cls._link_collision_mesh(link, resolve)
            if mesh is None:
                continue
            transform = np.asarray(
                robot.get_transform(frame_to=link_name, frame_from=base)
            )
            placed = mesh.copy()
            placed.apply_transform(transform)
            body = StructureVoxelBody.from_mesh(
                placed, voxel_size=cls._resolve_size(voxel_size, link_name)
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
                        a=parent_body,
                        b=child_body,
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
                        a=parent_body,
                        b=child_body,
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
                        a=parent_body,
                        b=child_body,
                        relative_anchor=anchor,
                    )
                )

        structure_name = name or robot.robot.name
        signal = StructureVoxelSignal(
            structure_name + "_signal",
            hinge_signal_num=hinge_count,
            prismatic_signal_num=prismatic_count,
        )
        return cls(
            structure_name, bodies=bodies, constraints=constraints, signal=signal
        )

    @classmethod
    def from_mjcf(
        cls,
        path: str,
        *,
        voxel_size: VoxelSizeSpec,
        name: str = "mjcf",
    ) -> "StructureVoxel":
        """A multi-body articulated structure loaded from a MuJoCo MJCF model.

        Each MuJoCo body's geoms are voxelized into a rigid body placed at the
        body's pose (zero configuration); each joint becomes a constraint wiring
        its body to its parent (hinge -> hinge, slide -> prismatic, ball ->
        ball-and-socket). Free joints, plane geoms, and bodies without geoms are
        skipped.

        Args:
            path: Filesystem path to the ``.xml`` MJCF file.
            voxel_size: One ``float`` for every body, or a ``dict`` mapping body
                name to its voxel edge length in meters; in the ``dict`` form the
                ``"*"`` key sets the default for unlisted bodies.
            name: Unique structure name.

        Returns:
            A multi-body :class:`StructureVoxel` with joints wired.
        """
        import mujoco
        import trimesh

        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        bodies = []
        body_of_id = {}
        for body_id in range(1, model.nbody):  # body 0 is the world
            parts = [
                cls._mjcf_geom_mesh(model, geom_id)
                for geom_id in range(model.ngeom)
                if model.geom_bodyid[geom_id] == body_id
            ]
            parts = [part for part in parts if part is not None]
            if not parts:
                continue
            mesh = trimesh.util.concatenate(parts)
            mesh.apply_transform(
                cls._pose_matrix(data.xpos[body_id], data.xquat[body_id])
            )
            body = StructureVoxelBody.from_mesh(
                mesh,
                voxel_size=cls._resolve_size(voxel_size, model.body(body_id).name),
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
            child_world = cls._pose_matrix(data.xpos[child_id], data.xquat[child_id])

            # In the absence of joints the body is welded to its parent.
            if not joint_ids:
                constraints.append(
                    StructureVoxelConstraint(
                        fixed_type,
                        a=parent_body,
                        b=child_body,
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
                        a=parent_body,
                        b=child_body,
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
                        a=parent_body,
                        b=child_body,
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
                        a=parent_body,
                        b=child_body,
                        relative_anchor=anchor,
                    )
                )

        signal = StructureVoxelSignal(
            name + "_signal",
            hinge_signal_num=hinge_count,
            prismatic_signal_num=prismatic_count,
        )
        return cls(name, bodies=bodies, constraints=constraints, signal=signal)

    @classmethod
    def from_gltf(
        cls,
        path: str,
        *,
        voxel_size: VoxelSizeSpec,
        name: str = "gltf",
    ) -> "StructureVoxel":
        """A structure voxelized from a glTF / GLB scene.

        Each mesh node in the scene becomes a rigid body voxelized at its node
        transform; a single-mesh file yields one body. Skeleton/skin animation is
        not interpreted -- the meshes are imported as static rigid bodies.

        Args:
            path: Filesystem path to the ``.gltf`` / ``.glb`` file.
            voxel_size: One ``float`` for every node, or a ``dict`` mapping node
                name to its voxel edge length in meters; in the ``dict`` form the
                ``"*"`` key sets the default for unlisted nodes.
            name: Unique structure name.

        Returns:
            A :class:`StructureVoxel` with one body per mesh node.
        """
        import trimesh

        scene = trimesh.load(path, force="scene")
        bodies = []
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]
            mesh = scene.geometry[geometry_name].copy()
            mesh.apply_transform(transform)
            body = StructureVoxelBody.from_mesh(
                mesh, voxel_size=cls._resolve_size(voxel_size, node_name)
            )
            bodies.append(body)
        return cls(name, bodies=bodies)

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
        voxelized into its own rigid body; STEP carries no kinematics, so the
        bodies are unconnected (wire joints afterward with the ``a`` / ``b`` body
        refs). Requires the ``[cad]`` extra (OpenCASCADE via cadquery).

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
        try:
            import cadquery
        except ImportError as error:
            raise ImportError(
                "STEP import requires the [cad] extra: pip install 'rise-origin[cad]'"
            ) from error
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
    def _resolve_endpoint(ref: BodyRef, index: dict) -> Tuple[int, int]:
        """Resolve a constraint endpoint body ref to ``(body_sid, segment_bid)``."""
        body, segment_bid = ref if isinstance(ref, tuple) else (ref, 0)
        body_sid = index.get(id(body))
        if body_sid is None:
            raise ValueError(
                "constraint endpoint references a body not in this structure's bodies"
            )
        return body_sid, segment_bid

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
    def _make_mesh_resolver(mesh_dir: str) -> Callable[[str], str]:
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
    def _link_collision_mesh(link, resolve: Callable[[str], str]):
        """Combine a URDF link's collision geometries into one trimesh, or None."""
        import trimesh

        parts = []
        for collision in link.collisions:
            geometry = collision.geometry
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
            if collision.origin is not None:
                primitive = primitive.copy()
                primitive.apply_transform(np.asarray(collision.origin))
            parts.append(primitive)
        if not parts:
            return None
        return trimesh.util.concatenate(parts)

    @staticmethod
    def _pose_matrix(position, quat_wxyz) -> np.ndarray:
        """Compose a position and a ``(w, x, y, z)`` quaternion into a 4x4 transform."""
        import trimesh

        transform = trimesh.transformations.quaternion_matrix(np.asarray(quat_wxyz))
        transform[:3, 3] = position
        return transform

    @staticmethod
    def _mjcf_geom_mesh(model, geom_id: int):
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
            StructureVoxel._pose_matrix(
                model.geom_pos[geom_id], model.geom_quat[geom_id]
            )
        )
        return primitive
