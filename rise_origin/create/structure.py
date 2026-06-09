# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""A builder for :class:`RS_StructureConfig` (voxel bodies + joints).

A *structure* is one simulated object (a robot, a creature, a prop). It is
built from *bodies* — axis-aligned voxel grids — joined by *constraints*
(hinge, prismatic, fixed, ball joints). Voxels reference *materials* by name
and are grouped into rigid *segments* that joints attach to.

``StructureCreator`` removes the bookkeeping the raw API forces on you:

* registering material names and resolving their per-voxel indices,
* filling the per-voxel ``material_reference_sid`` / ``segment_bid`` /
  ``expansion_signal_sid`` arrays (including the ``RS_NULL_INDEX``
  sentinel for empty voxels),
* assigning ``body_sid`` in insertion order,
* counting actuation signals,
* extending the opaque pybind vectors in place.

Joints attach to a *segment of a body*, not to a body. The builder returns a
:class:`BodyRef` from each ``add_*_body`` call; ask it for a
:class:`SegmentRef` via ``.segment(segment_bid)`` and pass that to the joint
methods. Composite (shell + core) bodies return :class:`CompositeBodyRef`,
which exposes the core and shell segments by name.

Example::

    from rise_origin.create import StructureCreator

    sc = StructureCreator("hopper", voxel_size=0.01)
    base = sc.add_box_body(size=(6, 6, 3), material="rigid",
                           relative_origin_position=(0.0, 0.0, 0.1))
    foot = sc.add_box_body(size=(4, 4, 6), material="rigid",
                           relative_origin_position=(0.01, 0.01, 0.0))
    sc.add_prismatic_joint(
        a=base.segment(0), b=foot.segment(0),
        relative_axis=(0, 0, 1),
        relative_anchor=(0.03, 0.03, 0.1),
        signal_sid=0,
    )
    structure = sc.build()

Naming rule: kwarg names mirror the underlying C++ field names. Inside a
per-joint method (e.g. ``add_hinge_joint``), the redundant joint-type prefix
(``hinge_``) is dropped, since the method already disambiguates the type;
otherwise the kwarg name matches C++ verbatim.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from rise_origin import (
    RS_NULL_INDEX,
    RQuat3rf,
    RS_StructureBodyConfig,
    RS_StructureConstraintConfig,
    RS_StructureConfig,
    RS_StructureTendonConfig,
    RVec3rf,
    RSE_StructureConstraintType,
    RSE_StructureExpansionControlMode,
    RSE_StructureHingeJointControlMode,
    RSE_StructurePrismaticJointControlMode,
)
from rise_origin.create.utils import Quat, Vec3

__all__ = [
    "StructureCreator",
    "BodyRef",
    "SegmentRef",
    "CompositeBodyRef",
]

# String names accepted by the helpers' control_mode argument, mapped to the
# control-mode enums. The strings appear in user-facing kwargs only.
_EXPANSION_CONTROL_MODES = {
    "position": RSE_StructureExpansionControlMode.RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION,
    "rate": RSE_StructureExpansionControlMode.RSE_STRUCTURE_EXPANSION_CONTROL_MODE_RATE,
}
_HINGE_CONTROL_MODES = {
    "basic": RSE_StructureHingeJointControlMode.RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_BASIC_POSITION,
    "pd": RSE_StructureHingeJointControlMode.RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_PD_POSITION,
    "torque": RSE_StructureHingeJointControlMode.RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_TORQUE,
    "velocity": RSE_StructureHingeJointControlMode.RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_VELOCITY,
}
_PRISMATIC_CONTROL_MODES = {
    "basic": RSE_StructurePrismaticJointControlMode.RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_BASIC_POSITION,
    "pd": RSE_StructurePrismaticJointControlMode.RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_PD_POSITION,
    "force": RSE_StructurePrismaticJointControlMode.RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_FORCE,
    "velocity": RSE_StructurePrismaticJointControlMode.RSE_STRUCTURE_PRISMATIC_JOINT_CONTROL_MODE_VELOCITY,
}

# Composite-body segment ids. Shell voxels live in segment 0, core voxels in 1.
_SHELL_SEGMENT_BID = 0
_CORE_SEGMENT_BID = 1


def _unit(axis: Vec3) -> Vec3:
    """Return ``axis`` normalized to unit length.

    Args:
        axis: A 3-vector ``(x, y, z)``.

    Returns:
        The unit-length version of ``axis``.

    Raises:
        ValueError: If ``axis`` has zero length.
    """
    vec = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("axis must be non-zero")
    vec = vec / norm
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def _sphere_mask(radius: int) -> np.ndarray:
    """Build a solid-sphere occupancy grid of the given voxel radius.

    Args:
        radius: Sphere radius in voxels.

    Returns:
        A ``(2*radius, 2*radius, 2*radius)`` boolean array, ``True`` inside.
    """
    n = 2 * radius
    coords = np.arange(n) + 0.5 - radius
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    return (gx * gx + gy * gy + gz * gz) <= float(radius) ** 2


def _cylinder_mask(radius: int, length: int, axis: str) -> np.ndarray:
    """Build a solid-cylinder occupancy grid.

    Args:
        radius: Cross-section radius in voxels.
        length: Length along ``axis`` in voxels.
        axis: Cylinder axis, one of ``"x"``, ``"y"``, ``"z"``.

    Returns:
        A boolean occupancy array oriented so the circular cross-section lies in
        the plane perpendicular to ``axis``.
    """
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


@dataclass(frozen=True)
class SegmentRef:
    """A reference to one rigid segment of one body.

    What joints attach to. Constructed via :meth:`BodyRef.segment` or returned
    as a named property of :class:`CompositeBodyRef`.
    """

    body_sid: int
    segment_bid: int


@dataclass(frozen=True)
class BodyRef:
    """A reference to one body added to a :class:`StructureCreator`.

    Use :meth:`segment` to get a :class:`SegmentRef` that joint methods accept.
    """

    body_sid: int

    def segment(self, segment_bid: int = 0) -> SegmentRef:
        """Return a reference to one rigid segment of this body.

        Args:
            segment_bid: Segment id within the body (defaults to 0).

        Returns:
            A :class:`SegmentRef` pinning this body and the chosen segment.
        """
        return SegmentRef(self.body_sid, segment_bid)


@dataclass(frozen=True)
class CompositeBodyRef(BodyRef):
    """A body with a rigid core wrapped in a soft shell.

    Exposes the two segments by name so joints attach to ``body.core`` /
    ``body.shell`` without the caller needing to know segment ids.
    """

    @property
    def core(self) -> SegmentRef:
        """Reference to the rigid core segment."""
        return SegmentRef(self.body_sid, _CORE_SEGMENT_BID)

    @property
    def shell(self) -> SegmentRef:
        """Reference to the soft shell segment."""
        return SegmentRef(self.body_sid, _SHELL_SEGMENT_BID)


class StructureCreator:
    """Incremental builder for a :class:`RS_StructureConfig`.

    Add bodies with the geometry methods, connect them with the joint methods,
    then call :meth:`build`.
    """

    def __init__(
        self,
        name: str,
        voxel_size: float,
        *,
        origin_position: Vec3 = (0.0, 0.0, 0.0),
        orientation: Quat = (0.0, 0.0, 0.0, 1.0),
        is_fixed: bool = False,
    ):
        """Create an empty structure builder.

        Args:
            name: Unique structure name (referenced by records, signals, cameras).
            voxel_size: Edge length of one voxel in meters (> 0).
            origin_position: World position of the structure origin ``(x, y, z)``.
            orientation: World orientation quaternion ``(x, y, z, w)``.
            is_fixed: If ``True``, the whole structure is pinned in the world.
        """
        self._name = name
        self._voxel_size = voxel_size
        self._origin_position = origin_position
        self._orientation = orientation
        self._is_fixed = is_fixed

        self._material_order: List[str] = []
        self._material_index = {}
        self._bodies: List[RS_StructureBodyConfig] = []
        self._constraints: List[RS_StructureConstraintConfig] = []
        self._tendons: List[RS_StructureTendonConfig] = []
        self._hinge_signal_num = 0
        self._prismatic_signal_num = 0
        self._expansion_signal_num = 0

    ################################################################
    # Materials
    ################################################################

    def _material_ref(self, name: str) -> int:
        """Resolve a material name to its index, registering it on first use."""
        if name not in self._material_index:
            self._material_index[name] = len(self._material_order)
            self._material_order.append(name)
        return self._material_index[name]

    ################################################################
    # Geometry
    ################################################################

    def _emit_grids(
        self,
        material_grid: np.ndarray,
        segment_grid: np.ndarray,
        *,
        relative_origin_position: Vec3,
        relative_orientation: Quat,
        expansion_signal_sid_grid: Optional[np.ndarray] = None,
        expansion_control_mode_grid: Optional[np.ndarray] = None,
    ) -> int:
        """Append a body from prebuilt per-voxel index grids (lowest level).

        Args:
            material_grid: ``(X, Y, Z)`` int grid of material indices, with
                ``RS_NULL_INDEX`` for empty voxels.
            segment_grid: ``(X, Y, Z)`` int grid of segment ids, ``RS_NULL_INDEX``
                for empty voxels.
            relative_origin_position: Minimum-corner position (structure-relative meters).
            relative_orientation: Body orientation quaternion ``(x, y, z, w)``.
            expansion_signal_sid_grid: Optional ``(X, Y, Z)`` int grid of
                expansion signal SIDs (``RS_NULL_INDEX`` where unactuated).
            expansion_control_mode_grid: Optional ``(X, Y, Z)`` int grid of
                expansion control modes (one per voxel).

        Returns:
            The ``body_sid`` of the appended body (insertion order).
        """
        nx, ny, nz = material_grid.shape
        body = RS_StructureBodyConfig()
        body.body_sid = len(self._bodies)
        body.relative_origin_position = RVec3rf(*relative_origin_position)
        body.relative_orientation = RQuat3rf(*relative_orientation)
        body.x_voxels = int(nx)
        body.y_voxels = int(ny)
        body.z_voxels = int(nz)

        # The engine flattens voxels with X fastest, then Y, then Z, so emit in
        # (z, y, x) order. transpose((2, 1, 0)) + C-order ravel does exactly that.
        order = (2, 1, 0)
        body.material_reference_sid.extend(
            material_grid.transpose(order).ravel(order="C").tolist()
        )
        body.segment_bid.extend(segment_grid.transpose(order).ravel(order="C").tolist())
        if expansion_signal_sid_grid is not None:
            body.expansion_signal_sid.extend(
                expansion_signal_sid_grid.transpose(order).ravel(order="C").tolist()
            )
        if expansion_control_mode_grid is not None:
            body.expansion_control_mode.extend(
                RSE_StructureExpansionControlMode(int(v))
                for v in expansion_control_mode_grid.transpose(order).ravel(order="C")
            )

        self._bodies.append(body)
        return body.body_sid

    def _emit_body(
        self,
        occupancy: np.ndarray,
        *,
        material: str,
        relative_origin_position: Vec3,
        relative_orientation: Quat,
        segment_bid: int,
        core_mask: Optional[np.ndarray] = None,
        shell_material: Optional[str] = None,
        expansion_signal_sid: Optional[int] = None,
        expansion_control_mode: str = "position",
    ) -> Union[BodyRef, CompositeBodyRef]:
        """Build one body from an occupancy grid (used by the shape helpers).

        Args:
            occupancy: ``(X, Y, Z)`` boolean array; ``True`` voxels are present.
            material: Material name for present voxels (or the core, with a shell).
            relative_origin_position: Minimum-corner position of the grid
                (structure-relative m).
            relative_orientation: Body orientation quaternion ``(x, y, z, w)``.
            segment_bid: Segment id for a uniform body.
            core_mask: Optional ``(X, Y, Z)`` boolean mask of core voxels; voxels
                in ``occupancy`` but not ``core_mask`` use ``shell_material``.
            shell_material: Material name for the soft shell (requires ``core_mask``).
            expansion_signal_sid: If set, all present voxels carry this expansion
                signal id.
            expansion_control_mode: ``"position"`` (default; signal is the target
                expansion ratio) or ``"rate"`` (signal is d(expansion)/dt).

        Returns:
            A :class:`CompositeBodyRef` when a shell+core was built; otherwise
            a :class:`BodyRef`.
        """
        occupancy = np.asarray(occupancy, dtype=bool)

        is_composite = core_mask is not None and shell_material is not None
        if is_composite:
            core_mask = np.asarray(core_mask, dtype=bool) & occupancy
            shell_mask = occupancy & ~core_mask
            material_grid = np.full(occupancy.shape, RS_NULL_INDEX, dtype=np.int64)
            material_grid[core_mask] = self._material_ref(material)
            material_grid[shell_mask] = self._material_ref(shell_material)
            segment_grid = np.full(occupancy.shape, RS_NULL_INDEX, dtype=np.int64)
            segment_grid[core_mask] = _CORE_SEGMENT_BID
            segment_grid[shell_mask] = _SHELL_SEGMENT_BID
        else:
            mat_idx = self._material_ref(material)
            material_grid = np.where(occupancy, mat_idx, RS_NULL_INDEX).astype(np.int64)
            segment_grid = np.where(occupancy, segment_bid, RS_NULL_INDEX).astype(np.int64)

        expansion_signal_sid_grid = None
        expansion_control_mode_grid = None
        if expansion_signal_sid is not None:
            expansion_signal_sid_grid = np.where(
                occupancy, expansion_signal_sid, RS_NULL_INDEX
            ).astype(np.int64)
            self._expansion_signal_num = max(
                self._expansion_signal_num, expansion_signal_sid + 1
            )
            assert expansion_control_mode in _EXPANSION_CONTROL_MODES, (
                f"expansion_control_mode must be one of "
                f"{sorted(_EXPANSION_CONTROL_MODES)}, got {expansion_control_mode!r}"
            )
            mode_value = int(_EXPANSION_CONTROL_MODES[expansion_control_mode])
            expansion_control_mode_grid = np.full(
                occupancy.shape, mode_value, dtype=np.int64
            )

        body_sid = self._emit_grids(
            material_grid,
            segment_grid,
            relative_origin_position=relative_origin_position,
            relative_orientation=relative_orientation,
            expansion_signal_sid_grid=expansion_signal_sid_grid,
            expansion_control_mode_grid=expansion_control_mode_grid,
        )
        if is_composite:
            return CompositeBodyRef(body_sid=body_sid)
        return BodyRef(body_sid=body_sid)

    def add_voxel_grid_body(
        self,
        materials: np.ndarray,
        *,
        relative_origin_position: Vec3,
        relative_orientation: Quat = (0.0, 0.0, 0.0, 1.0),
        segment_bids: Optional[np.ndarray] = None,
        expansion_signal_sids: Optional[np.ndarray] = None,
        expansion_control_modes: Optional[np.ndarray] = None,
    ) -> BodyRef:
        """Add a body from explicit per-voxel arrays (the low-level interface).

        This is the most general builder: it lets you mix materials and segments
        voxel-by-voxel in one body — e.g. a rigid core, a soft shell, and a band
        of muscle voxels — which the shape helpers cannot express.

        Args:
            materials: ``(X, Y, Z)`` object array of material names; use ``None``
                or ``""`` for an empty (absent) voxel.
            relative_origin_position: Minimum-corner position (structure-relative
                meters).
            relative_orientation: Body orientation quaternion ``(x, y, z, w)``.
            segment_bids: Optional ``(X, Y, Z)`` int array of segment ids for
                present voxels; defaults to segment ``0`` everywhere.
            expansion_signal_sids: Optional ``(X, Y, Z)`` int array of expansion
                signal ids (use a negative value or ``RS_NULL_INDEX`` for
                unactuated voxels). ``None`` means no expansion actuation.
            expansion_control_modes: Optional ``(X, Y, Z)`` int array of expansion
                control mode values (one per voxel; use the
                ``RSE_StructureExpansionControlMode`` enum). If ``None`` and
                ``expansion_signal_sids`` is set, every voxel defaults to
                ``BASIC_POSITION``.

        Returns:
            A :class:`BodyRef`. Call ``.segment(bid)`` to address a specific
            segment when connecting joints.
        """
        materials = np.asarray(materials, dtype=object)
        present = np.array(
            [m is not None and m != "" for m in materials.ravel()]
        ).reshape(materials.shape)

        material_grid = np.full(materials.shape, RS_NULL_INDEX, dtype=np.int64)
        for name in {m for m in materials.ravel() if m is not None and m != ""}:
            material_grid[materials == name] = self._material_ref(name)

        if segment_bids is None:
            segment_grid = np.where(present, 0, RS_NULL_INDEX).astype(np.int64)
        else:
            segment_grid = np.where(
                present, np.asarray(segment_bids, dtype=np.int64), RS_NULL_INDEX
            ).astype(np.int64)

        expansion_signal_sid_grid = None
        expansion_control_mode_grid = None
        if expansion_signal_sids is not None:
            sids = np.asarray(expansion_signal_sids, dtype=np.int64)
            expansion_signal_sid_grid = np.where(present & (sids >= 0), sids, RS_NULL_INDEX)
            actuated = expansion_signal_sid_grid[expansion_signal_sid_grid != RS_NULL_INDEX]
            if actuated.size > 0:
                self._expansion_signal_num = max(
                    self._expansion_signal_num, int(actuated.max()) + 1
                )
            if expansion_control_modes is not None:
                modes = np.asarray(expansion_control_modes, dtype=np.int64)
                expansion_control_mode_grid = np.where(
                    expansion_signal_sid_grid != RS_NULL_INDEX,
                    modes,
                    int(RSE_StructureExpansionControlMode.RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION),
                ).astype(np.int64)

        body_sid = self._emit_grids(
            material_grid,
            segment_grid,
            relative_origin_position=relative_origin_position,
            relative_orientation=relative_orientation,
            expansion_signal_sid_grid=expansion_signal_sid_grid,
            expansion_control_mode_grid=expansion_control_mode_grid,
        )
        return BodyRef(body_sid=body_sid)

    def add_box_body(
        self,
        size: Vec3,
        material: str,
        *,
        relative_origin_position: Vec3,
        relative_orientation: Quat = (0.0, 0.0, 0.0, 1.0),
        segment_bid: int = 0,
        shell: int = 0,
        shell_material: Optional[str] = None,
        expansion_signal_sid: Optional[int] = None,
        expansion_control_mode: str = "position",
    ) -> Union[BodyRef, CompositeBodyRef]:
        """Add a solid box body.

        With ``shell > 0`` and ``shell_material`` set, the box gets a soft shell
        of that thickness around a rigid core; ``size`` is then the **core**
        size and the full body is ``size + 2*shell`` per axis. The returned
        :class:`CompositeBodyRef` exposes ``.core`` and ``.shell`` segment refs.

        Args:
            size: Core size in voxels ``(nx, ny, nz)``.
            material: Material name (the core material when a shell is present).
            relative_origin_position: Minimum-corner position (structure-relative
                meters).
            relative_orientation: Body orientation quaternion ``(x, y, z, w)``.
            segment_bid: Segment id when there is no shell.
            shell: Soft-shell thickness in voxels (0 = no shell).
            shell_material: Material name for the shell (required if ``shell > 0``).
            expansion_signal_sid: Optional expansion signal id for soft actuation.
            expansion_control_mode: ``"position"`` or ``"rate"`` — interpretation
                of the expansion signal value.

        Returns:
            A :class:`BodyRef` for uniform bodies, or a :class:`CompositeBodyRef`
            when a shell + core was built.
        """
        nx, ny, nz = (int(round(v)) for v in size)
        if shell > 0 and shell_material is not None:
            total = (nx + 2 * shell, ny + 2 * shell, nz + 2 * shell)
            occupancy = np.ones(total, dtype=bool)
            core_mask = np.zeros(total, dtype=bool)
            core_mask[shell:-shell, shell:-shell, shell:-shell] = True
            return self._emit_body(
                occupancy,
                material=material,
                relative_origin_position=relative_origin_position,
                relative_orientation=relative_orientation,
                segment_bid=segment_bid,
                core_mask=core_mask,
                shell_material=shell_material,
                expansion_signal_sid=expansion_signal_sid,
                expansion_control_mode=expansion_control_mode,
            )
        occupancy = np.ones((nx, ny, nz), dtype=bool)
        return self._emit_body(
            occupancy,
            material=material,
            relative_origin_position=relative_origin_position,
            relative_orientation=relative_orientation,
            segment_bid=segment_bid,
            expansion_signal_sid=expansion_signal_sid,
            expansion_control_mode=expansion_control_mode,
        )

    def add_sphere_body(
        self,
        radius: int,
        material: str,
        *,
        relative_origin_position: Vec3,
        relative_orientation: Quat = (0.0, 0.0, 0.0, 1.0),
        segment_bid: int = 0,
        shell: int = 0,
        shell_material: Optional[str] = None,
        expansion_signal_sid: Optional[int] = None,
        expansion_control_mode: str = "position",
    ) -> Union[BodyRef, CompositeBodyRef]:
        """Add a solid sphere body (``2*radius`` voxels per axis).

        Args:
            radius: Sphere radius in voxels.
            material: Material name (the core material when a shell is present).
            relative_origin_position: Minimum-corner position (structure-relative
                meters).
            relative_orientation: Body orientation quaternion ``(x, y, z, w)``.
            segment_bid: Segment id when there is no shell.
            shell: Soft-shell thickness in voxels (0 = no shell).
            shell_material: Material name for the shell (required if ``shell > 0``).
            expansion_signal_sid: Optional expansion signal id for soft actuation.
            expansion_control_mode: ``"position"`` or ``"rate"``.

        Returns:
            A :class:`BodyRef`, or :class:`CompositeBodyRef` when a shell+core was built.
        """
        occupancy = _sphere_mask(radius)
        core_mask = None
        if shell > 0 and shell_material is not None:
            core_mask = _sphere_mask(radius)
            inner = _sphere_mask(max(radius - shell, 0))
            pad = radius - max(radius - shell, 0)
            full = np.zeros_like(occupancy)
            if inner.shape[0] > 0:
                full[
                    pad : pad + inner.shape[0],
                    pad : pad + inner.shape[1],
                    pad : pad + inner.shape[2],
                ] = inner
            core_mask = full
        return self._emit_body(
            occupancy,
            material=material,
            relative_origin_position=relative_origin_position,
            relative_orientation=relative_orientation,
            segment_bid=segment_bid,
            core_mask=core_mask,
            shell_material=shell_material if core_mask is not None else None,
            expansion_signal_sid=expansion_signal_sid,
            expansion_control_mode=expansion_control_mode,
        )

    def add_cylinder_body(
        self,
        radius: int,
        length: int,
        material: str,
        *,
        axis: str = "z",
        relative_origin_position: Vec3,
        relative_orientation: Quat = (0.0, 0.0, 0.0, 1.0),
        segment_bid: int = 0,
        expansion_signal_sid: Optional[int] = None,
        expansion_control_mode: str = "position",
    ) -> BodyRef:
        """Add a solid cylinder body.

        Args:
            radius: Cross-section radius in voxels.
            length: Length along ``axis`` in voxels.
            material: Material name for all voxels.
            axis: Cylinder axis, one of ``"x"``, ``"y"``, ``"z"``.
            relative_origin_position: Minimum-corner position (structure-relative
                meters).
            relative_orientation: Body orientation quaternion ``(x, y, z, w)``.
            segment_bid: Segment id for all voxels.
            expansion_signal_sid: Optional expansion signal id for soft actuation.
            expansion_control_mode: ``"position"`` or ``"rate"``.

        Returns:
            A :class:`BodyRef`.
        """
        occupancy = _cylinder_mask(radius, length, axis)
        result = self._emit_body(
            occupancy,
            material=material,
            relative_origin_position=relative_origin_position,
            relative_orientation=relative_orientation,
            segment_bid=segment_bid,
            expansion_signal_sid=expansion_signal_sid,
            expansion_control_mode=expansion_control_mode,
        )
        return result  # always BodyRef (no shell path)

    ################################################################
    # Joints
    ################################################################

    def add_hinge_joint(
        self,
        a: SegmentRef,
        b: SegmentRef,
        *,
        relative_axis: Vec3,
        relative_anchor: Vec3,
        signal_sid: Optional[int] = None,
        control_mode: str = "basic",
        angle_min: float = -np.pi,
        angle_max: float = np.pi,
        max_torque: float = 0.0,
        max_speed: float = 10.0,
        position_control_kp: float = 0.0,
        position_control_kd: float = 0.0,
    ) -> None:
        """Connect two segments with a 1-DOF hinge (revolute) joint.

        Inside this method the redundant ``hinge_`` prefix is dropped: kwarg
        ``foo`` maps to C++ ``hinge_foo``. ``relative_anchor`` is shared across
        joint types so it keeps its full name.

        Args:
            a: First segment (the joint's ``a`` side).
            b: Second segment (the joint's ``b`` side).
            relative_axis: Rotation axis (structure-relative); normalized
                internally. Maps to C++ ``hinge_relative_axis``.
            relative_anchor: Joint anchor point (structure-relative meters).
            signal_sid: Signal id to drive this hinge, or ``None`` for a passive
                joint. Maps to C++ ``hinge_signal_sid``.
            control_mode: Motor control mode. One of ``"basic"``, ``"pd"``,
                ``"torque"``, ``"velocity"``. Maps to C++ ``hinge_control_mode``.
            angle_min: Minimum angle in radians (``>= -pi``). Maps to ``hinge_min``.
            angle_max: Maximum angle in radians (``<= pi``). Maps to ``hinge_max``.
            max_torque: Maximum motor torque (N-m, ``>= 0``). Maps to
                ``hinge_max_torque``.
            max_speed: Maximum joint speed (rad/s, ``>= 0``). Maps to
                ``hinge_max_speed``.
            position_control_kp: Proportional gain for ``control_mode="pd"``
                (``>= 0``). Maps to ``hinge_position_control_kp``.
            position_control_kd: Derivative gain for ``control_mode="pd"``
                (``>= 0``). Maps to ``hinge_position_control_kd``.
        """
        assert control_mode in _HINGE_CONTROL_MODES, (
            f"control_mode must be one of {sorted(_HINGE_CONTROL_MODES)}, "
            f"got {control_mode!r}"
        )
        c = RS_StructureConstraintConfig()
        c.type = RSE_StructureConstraintType.RSE_HINGE_JOINT
        c.a_body_sid = a.body_sid
        c.a_segment_bid = a.segment_bid
        c.b_body_sid = b.body_sid
        c.b_segment_bid = b.segment_bid
        c.relative_anchor = RVec3rf(*relative_anchor)
        c.hinge_relative_axis = RVec3rf(*_unit(relative_axis))
        c.hinge_min = angle_min
        c.hinge_max = angle_max
        c.hinge_max_torque = max_torque
        c.hinge_max_speed = max_speed
        c.hinge_position_control_kp = position_control_kp
        c.hinge_position_control_kd = position_control_kd
        c.hinge_control_mode = int(_HINGE_CONTROL_MODES[control_mode])
        if signal_sid is not None:
            c.hinge_signal_sid = signal_sid
            self._hinge_signal_num = max(self._hinge_signal_num, signal_sid + 1)
        self._constraints.append(c)

    def add_prismatic_joint(
        self,
        a: SegmentRef,
        b: SegmentRef,
        *,
        relative_axis: Vec3,
        relative_anchor: Vec3,
        signal_sid: Optional[int] = None,
        control_mode: str = "basic",
        distance_min: float = -1.0,
        distance_max: float = 1.0,
        max_force: float = 0.0,
        max_speed: float = 10.0,
        position_control_kp: float = 0.0,
        position_control_kd: float = 0.0,
    ) -> None:
        """Connect two segments with a 1-DOF prismatic (sliding) joint.

        Inside this method the redundant ``prismatic_`` prefix is dropped: kwarg
        ``foo`` maps to C++ ``prismatic_foo``. ``relative_anchor`` is shared
        across joint types so it keeps its full name.

        Args:
            a: First segment (the joint's ``a`` side).
            b: Second segment (the joint's ``b`` side).
            relative_axis: Slide axis (structure-relative); normalized internally.
                Maps to C++ ``prismatic_relative_axis``.
            relative_anchor: Joint anchor point (structure-relative meters).
            signal_sid: Signal id to drive this joint, or ``None``. Maps to
                ``prismatic_signal_sid``.
            control_mode: Motor control mode. One of ``"basic"``, ``"pd"``,
                ``"force"``, ``"velocity"``. Maps to ``prismatic_control_mode``.
            distance_min: Minimum slide distance in meters. Maps to ``prismatic_min``.
            distance_max: Maximum slide distance in meters. Maps to ``prismatic_max``.
            max_force: Maximum motor force (N, ``>= 0``). Maps to
                ``prismatic_max_force``.
            max_speed: Maximum joint speed (m/s, ``>= 0``). Maps to
                ``prismatic_max_speed``.
            position_control_kp: Proportional gain for ``control_mode="pd"``.
                Maps to ``prismatic_position_control_kp``.
            position_control_kd: Derivative gain for ``control_mode="pd"``.
                Maps to ``prismatic_position_control_kd``.
        """
        assert control_mode in _PRISMATIC_CONTROL_MODES, (
            f"control_mode must be one of {sorted(_PRISMATIC_CONTROL_MODES)}, "
            f"got {control_mode!r}"
        )
        c = RS_StructureConstraintConfig()
        c.type = RSE_StructureConstraintType.RSE_PRISMATIC_JOINT
        c.a_body_sid = a.body_sid
        c.a_segment_bid = a.segment_bid
        c.b_body_sid = b.body_sid
        c.b_segment_bid = b.segment_bid
        c.relative_anchor = RVec3rf(*relative_anchor)
        c.prismatic_relative_axis = RVec3rf(*_unit(relative_axis))
        c.prismatic_min = distance_min
        c.prismatic_max = distance_max
        c.prismatic_max_force = max_force
        c.prismatic_max_speed = max_speed
        c.prismatic_position_control_kp = position_control_kp
        c.prismatic_position_control_kd = position_control_kd
        c.prismatic_control_mode = int(_PRISMATIC_CONTROL_MODES[control_mode])
        if signal_sid is not None:
            c.prismatic_signal_sid = signal_sid
            self._prismatic_signal_num = max(self._prismatic_signal_num, signal_sid + 1)
        self._constraints.append(c)

    def add_fixed_joint(
        self, a: SegmentRef, b: SegmentRef, *, relative_anchor: Vec3
    ) -> None:
        """Weld two segments together with a fixed joint.

        Args:
            a: First segment.
            b: Second segment.
            relative_anchor: Weld anchor point (structure-relative meters).
        """
        c = RS_StructureConstraintConfig()
        c.type = RSE_StructureConstraintType.RSE_FIXED_JOINT
        c.a_body_sid = a.body_sid
        c.a_segment_bid = a.segment_bid
        c.b_body_sid = b.body_sid
        c.b_segment_bid = b.segment_bid
        c.relative_anchor = RVec3rf(*relative_anchor)
        self._constraints.append(c)

    def add_ball_and_socket_joint(
        self, a: SegmentRef, b: SegmentRef, *, relative_anchor: Vec3
    ) -> None:
        """Connect two segments with a 3-DOF ball-and-socket joint.

        Args:
            a: First segment.
            b: Second segment.
            relative_anchor: Joint anchor point (structure-relative meters).
        """
        c = RS_StructureConstraintConfig()
        c.type = RSE_StructureConstraintType.RSE_BALL_AND_SOCKET_JOINT
        c.a_body_sid = a.body_sid
        c.a_segment_bid = a.segment_bid
        c.b_body_sid = b.body_sid
        c.b_segment_bid = b.segment_bid
        c.relative_anchor = RVec3rf(*relative_anchor)
        self._constraints.append(c)

    ################################################################
    # Build
    ################################################################

    def build(self) -> RS_StructureConfig:
        """Assemble and return the :class:`RS_StructureConfig`.

        Returns:
            A structure config with all bodies, joints, materials, and signal
            counts populated. Append it to ``config.structure_configs``.
        """
        structure = RS_StructureConfig()
        structure.name = self._name
        structure.is_fixed = self._is_fixed
        structure.voxel_size = self._voxel_size
        structure.origin_position = RVec3rf(*self._origin_position)
        structure.orientation = RQuat3rf(*self._orientation)

        for material_name in self._material_order:
            structure.material_references.append(material_name)
        for body in self._bodies:
            structure.bodies.append(body)
        for constraint in self._constraints:
            structure.constraints.append(constraint)
        for tendon in self._tendons:
            structure.tendons.append(tendon)

        structure.hinge_signal_num = self._hinge_signal_num
        structure.prismatic_signal_num = self._prismatic_signal_num
        structure.expansion_signal_num = self._expansion_signal_num
        return structure
