# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Simulation configuration: global parameters and per-subsystem sub-configs.

Every field defaults to ``None`` and is only written when supplied, so an
omitted argument keeps the engine's built-in default.
"""

from typing import Optional

from rise_origin import (
    RS_MathExpr,
    RSC_Simulation,
    RSC_SimulationContact,
    RSC_SimulationControl,
    RSC_SimulationEnvironment,
    RSC_SimulationRecord,
    RSC_SimulationRigidBody,
    RSC_SimulationRuntime,
    RSC_SimulationSentinel,
    RSC_SimulationSoftBody,
    RSC_SimulationTime,
)

from rise_origin.rsc._utils import _apply
from rise_origin.rsc.simulation.vision import Vision

__all__ = [
    "Environment",
    "Time",
    "SoftBody",
    "RigidBody",
    "Contact",
    "Sentinel",
    "Control",
    "Record",
    "Runtime",
    "Simulation",
]


class Environment(RSC_SimulationEnvironment):
    """World environment.

    Args:
        gravity_acceleration: Gravitational acceleration along -Z, in m/s^2.
    """

    def __init__(self, *, gravity_acceleration: Optional[float] = None) -> None:
        super().__init__()
        _apply(self, gravity_acceleration=gravity_acceleration)


class Time(RSC_SimulationTime):
    """Time-stepping cadence.

    The ``dt_*`` periods are optional: leave one ``None`` and the engine derives
    it from ``dt_ratio`` and that subsystem's stability limit.

    Args:
        dt: Fixed base time step in seconds; ``None`` lets the engine choose.
        dt_ratio: Multiplier on the automatically chosen ``dt``, must be > 0.
        dt_update_interval: Steps between re-derivations of the adaptive dt.
        dt_softbody_and_rigidbody_update: Soft/rigid-body update period (s).
        dt_rigidbody_contact_and_constraint_update: Rigid contact/constraint
            update period (s).
        dt_contact_update: Contact broad-phase update period (s).
        dt_mechanical_signal_update: Mechanical-signal update period (s).
        dt_sentinel_update: Sentinel-check period (s).
        is_aligned_to_wallclock: When True, delay the running world so simulation
            time does not outrun wall-clock time.
    """

    def __init__(
        self,
        *,
        dt: Optional[float] = None,
        dt_ratio: Optional[float] = None,
        dt_update_interval: Optional[int] = None,
        dt_softbody_and_rigidbody_update: Optional[float] = None,
        dt_rigidbody_contact_and_constraint_update: Optional[float] = None,
        dt_contact_update: Optional[float] = None,
        dt_mechanical_signal_update: Optional[float] = None,
        dt_sentinel_update: Optional[float] = None,
        is_aligned_to_wallclock: bool = False,
    ) -> None:
        super().__init__()
        _apply(
            self,
            dt=dt,
            dt_ratio=dt_ratio,
            dt_update_interval=dt_update_interval,
            dt_softbody_and_rigidbody_update=dt_softbody_and_rigidbody_update,
            dt_rigidbody_contact_and_constraint_update=dt_rigidbody_contact_and_constraint_update,
            dt_contact_update=dt_contact_update,
            dt_mechanical_signal_update=dt_mechanical_signal_update,
            dt_sentinel_update=dt_sentinel_update,
            is_aligned_to_wallclock=is_aligned_to_wallclock,
        )


class SoftBody(RSC_SimulationSoftBody):
    """Soft-body solver settings.

    Args:
        internal_damping_z: Internal (link) damping ratio.
        contact_damping_z: Contact damping ratio.
        global_damping_z: Global velocity damping ratio.
        integration_frequency_multiplier: Divisor on the stiffest natural angular
            frequency when auto-computing the soft-body step, must be > 0.
        link_max_compression: Maximum compressive strain magnitude for a soft
            link, in the range [0, 1).
        friction_max_penetration_ratio: Share of a voxel the friction cone may read as
            pressed together, in the range (0, 1].
    """

    def __init__(
        self,
        *,
        internal_damping_z: Optional[float] = None,
        contact_damping_z: Optional[float] = None,
        global_damping_z: Optional[float] = None,
        integration_frequency_multiplier: Optional[float] = None,
        link_max_compression: Optional[float] = None,
        friction_max_penetration_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()
        _apply(
            self,
            internal_damping_z=internal_damping_z,
            contact_damping_z=contact_damping_z,
            global_damping_z=global_damping_z,
            integration_frequency_multiplier=integration_frequency_multiplier,
            link_max_compression=link_max_compression,
            friction_max_penetration_ratio=friction_max_penetration_ratio,
        )


class RigidBody(RSC_SimulationRigidBody):
    """Rigid-body constraint and contact solver settings.

    Args:
        constraint_iterations: Constraint solver iteration count.
        constraint_position_correction_ratio: Position-correction fraction
            applied per constraint iteration.
        constraint_warm_start_damping_z: Warm-start damping ratio for
            constraints.
        constraint_compliance_initial: Initial constraint compliance.
        constraint_compliance_min_threshold: Lower clamp on adaptive compliance.
        constraint_compliance_max_threshold: Upper clamp on adaptive compliance.
        constraint_compliance_auto_adjust_watt_threshold: Power threshold above
            which compliance auto-adjusts.
        constraint_compliance_auto_adjust_step: Compliance auto-adjust step size.
        contact_iterations: Contact solver iteration count.
        contact_position_correction_iterations: Contact position-correction
            iteration count.
        contact_manifold_size: Contacts one rigid body pair keeps, the deepest
            plus the farthest from it, in the range [1, 8].
        contact_position_correction_ratio: Contact position-correction fraction.
        contact_warm_start_damping_z: Warm-start damping ratio for contacts.
        contact_slop: Allowed contact penetration before correction, in meters.
        linear_damping_z: Linear velocity damping ratio.
        angular_damping_z: Angular velocity damping ratio.
    """

    def __init__(
        self,
        *,
        constraint_iterations: Optional[int] = None,
        constraint_position_correction_ratio: Optional[float] = None,
        constraint_warm_start_damping_z: Optional[float] = None,
        constraint_compliance_initial: Optional[float] = None,
        constraint_compliance_min_threshold: Optional[float] = None,
        constraint_compliance_max_threshold: Optional[float] = None,
        constraint_compliance_auto_adjust_watt_threshold: Optional[float] = None,
        constraint_compliance_auto_adjust_step: Optional[float] = None,
        contact_iterations: Optional[int] = None,
        contact_position_correction_iterations: Optional[int] = None,
        contact_manifold_size: Optional[int] = None,
        contact_position_correction_ratio: Optional[float] = None,
        contact_warm_start_damping_z: Optional[float] = None,
        contact_slop: Optional[float] = None,
        linear_damping_z: Optional[float] = None,
        angular_damping_z: Optional[float] = None,
    ) -> None:
        super().__init__()
        _apply(
            self,
            constraint_iterations=constraint_iterations,
            constraint_position_correction_ratio=constraint_position_correction_ratio,
            constraint_warm_start_damping_z=constraint_warm_start_damping_z,
            constraint_compliance_initial=constraint_compliance_initial,
            constraint_compliance_min_threshold=constraint_compliance_min_threshold,
            constraint_compliance_max_threshold=constraint_compliance_max_threshold,
            constraint_compliance_auto_adjust_watt_threshold=constraint_compliance_auto_adjust_watt_threshold,
            constraint_compliance_auto_adjust_step=constraint_compliance_auto_adjust_step,
            contact_iterations=contact_iterations,
            contact_position_correction_iterations=contact_position_correction_iterations,
            contact_manifold_size=contact_manifold_size,
            contact_position_correction_ratio=contact_position_correction_ratio,
            contact_warm_start_damping_z=contact_warm_start_damping_z,
            contact_slop=contact_slop,
            linear_damping_z=linear_damping_z,
            angular_damping_z=angular_damping_z,
        )


class Contact(RSC_SimulationContact):
    """Contact broad-phase BVH maintenance cadence.

    Args:
        bvh_rebuild_sub_interval: Steps between full contact-BVH rebuilds.
        bvh_refit_sub_interval: Steps between contact-BVH refits.
    """

    def __init__(
        self,
        *,
        bvh_rebuild_sub_interval: Optional[int] = None,
        bvh_refit_sub_interval: Optional[int] = None,
    ) -> None:
        super().__init__()
        _apply(
            self,
            bvh_rebuild_sub_interval=bvh_rebuild_sub_interval,
            bvh_refit_sub_interval=bvh_refit_sub_interval,
        )


class Sentinel(RSC_SimulationSentinel):
    """Numerical-instability watchdog.

    Args:
        is_enabled: Whether the sentinel halts the run on instability.
        degenerate_min_dt: Smallest global time step treated as normal (s).
    """

    def __init__(
        self,
        *,
        is_enabled: Optional[bool] = None,
        degenerate_min_dt: Optional[float] = None,
    ) -> None:
        super().__init__()
        _apply(self, is_enabled=is_enabled, degenerate_min_dt=degenerate_min_dt)


class Control(RSC_SimulationControl):
    """Controller-callback cadence.

    Args:
        frequency: Controller callback frequency in Hz.
    """

    def __init__(self, *, frequency: Optional[float] = None) -> None:
        super().__init__()
        _apply(self, frequency=frequency)


class Record(RSC_SimulationRecord):
    """Record cadence.

    Args:
        frequency: Record sampling frequency in Hz.
    """

    def __init__(self, *, frequency: Optional[float] = None) -> None:
        super().__init__()
        _apply(self, frequency=frequency)


class Runtime(RSC_SimulationRuntime):
    """Run-level limits and output settings.

    Args:
        is_paused_on_start: Whether the world starts paused.
        max_time: Maximum simulated time in seconds (0 for unbounded).
        record_buffer_size: Record ring-buffer capacity.
        is_save_record: Whether records are persisted to disk.
        compute_accuracy: Compute float type (``RSE_FloatType`` value).
        memory_accuracy: Storage float type (``RSE_FloatType`` value).
        log_level: Engine log verbosity.
    """

    def __init__(
        self,
        *,
        is_paused_on_start: Optional[bool] = None,
        max_time: Optional[float] = None,
        record_buffer_size: Optional[int] = None,
        is_save_record: Optional[bool] = None,
        compute_accuracy: Optional[int] = None,
        memory_accuracy: Optional[int] = None,
        log_level: Optional[str] = None,
    ) -> None:
        super().__init__()
        _apply(
            self,
            is_paused_on_start=is_paused_on_start,
            max_time=max_time,
            record_buffer_size=record_buffer_size,
            is_save_record=is_save_record,
            compute_accuracy=compute_accuracy,
            memory_accuracy=memory_accuracy,
            log_level=log_level,
        )


class Simulation(RSC_Simulation):
    """Top-level simulation config aggregating all subsystem sub-configs.

    Args:
        seed: Random seed for the run.
        stop_condition: An :class:`RS_MathExpr` over recorded quantities; the run
            ends when it evaluates truthy.
        environment: World environment config.
        time: Time-stepping config.
        soft_body: Soft-body solver config.
        rigid_body: Rigid-body solver config.
        contact: Contact broad-phase config.
        vision: Vision (path-tracing) config.
        sentinel: Instability-watchdog config.
        control: Controller-cadence config.
        record: Record-cadence config.
        runtime: Run-level limits config.
    """

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        stop_condition: Optional[RS_MathExpr] = None,
        environment: Optional[Environment] = None,
        time: Optional[Time] = None,
        soft_body: Optional[SoftBody] = None,
        rigid_body: Optional[RigidBody] = None,
        contact: Optional[Contact] = None,
        vision: Optional[Vision] = None,
        sentinel: Optional[Sentinel] = None,
        control: Optional[Control] = None,
        record: Optional[Record] = None,
        runtime: Optional[Runtime] = None,
    ) -> None:
        super().__init__()
        _apply(
            self,
            seed=seed,
            stop_condition=stop_condition,
            environment=environment,
            time=time,
            soft_body=soft_body,
            rigid_body=rigid_body,
            contact=contact,
            vision=vision,
            sentinel=sentinel,
            control=control,
            record=record,
            runtime=runtime,
        )
