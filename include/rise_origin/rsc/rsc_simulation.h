// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSC_SIMULATION_H
#define RSC_SIMULATION_H

#include <string>

#include "rise_origin/utils/data/rs_optional.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/math/rs_float.h"
#include "rise_origin/utils/math/rs_math_expr.h"
#include "rise_origin/utils/rs_def.h"
#include "rsc_base.h"

namespace rise {

namespace rsc {

/**
 * @brief World environment settings.
 */
class RSC_SimulationEnvironment : public RSC_Base {
   public:
    /// Gravitational acceleration along world -Z (unit: m/s^2).
    RFloat gravity_acceleration = -9.81;

    bool operator==(const RSC_SimulationEnvironment& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Time-stepping settings.
 *
 * The whole simulation advances with one fixed time step `dt`. Each `dt_*` below is the largest
 * time step its subsystem can tolerate, not how often it actually runs: the subsystem runs once
 * every `max(1, floor(dt_sub_task / dt))` base steps, so its real period is that step count times
 * `dt`. A value left empty takes its subsystem default, scaled by `dt_ratio`.
 */
class RSC_SimulationTime : public RSC_Base {
   public:
    /// Base time step (unit: s). If empty, it is chosen automatically as the largest step that
    /// keeps every subsystem stable.
    RS_Optional<RFloat> dt;

    /// Multiplier on the automatically chosen `dt` actually used, must be > 0. Values below 1 trade
    /// speed for extra stability margin; values above 1 take a larger, faster but less stable step.
    /// Ignored when `dt` is set explicitly.
    RFloat dt_ratio = 1;

    /// Number of base steps between automatic recomputations of `dt`. 0 computes `dt` once at start
    /// and keeps it fixed for the rest of the run.
    RSize dt_update_interval = 10;

    /// Largest time step (s) tolerated between updates that advance the soft body and couple it
    /// with rigid bodies.
    RS_Optional<RFloat> dt_softbody_and_rigidbody_update;

    /// Largest time step (s) tolerated between rigid-body contact resolution and constraint solves.
    RS_Optional<RFloat> dt_rigidbody_contact_and_constraint_update;

    /// Largest time step (s) tolerated between broadphase contact detection passes.
    RS_Optional<RFloat> dt_contact_update;

    /// Largest time step (s) tolerated between mechanical signal updates.
    RS_Optional<RFloat> dt_mechanical_signal_update;

    /// Largest time step (s) tolerated between sentinel (divergence) checks.
    RS_Optional<RFloat> dt_sentinel_update;

    /// When true, the running world is delayed so simulation time does not outrun wall-clock time.
    bool is_aligned_to_wallclock = false;

    bool operator==(const RSC_SimulationTime& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Soft-body solver settings.
 */
class RSC_SimulationSoftBody : public RSC_Base {
   public:
    /// Internal (structural) damping ratio for soft links, [0, 1] of critical.
    RFloat internal_damping_z = 0.5;

    /// Contact damping ratio for soft voxels, [0, 1] of critical.
    RFloat contact_damping_z = 0.2;

    /// Global (environmental) damping ratio for soft voxels, [0, 1].
    RFloat global_damping_z = 0.001;

    /// Divisor applied to the stiffest natural angular frequency when auto-computing the soft-body
    /// step: dt = 1 / (integration_frequency_multiplier * omega). Larger values give smaller, more
    /// stable steps. Must be > 0; values below ~2 approach the lattice stability limit. The default
    /// is tuned for a braced 3-D voxel lattice, whose bending, torsion and diagonal modes raise the
    /// governing frequency above the single-bond value.
    RFloat integration_frequency_multiplier = 3.5;

    /// Maximum compressive strain magnitude for a soft link; strain is floored at
    /// -link_max_compression to keep heavily compressed links valid. Range [0, 1).
    RFloat link_max_compression = 0.8;

    /// Share of a voxel the friction cone may read as pressed together. Range (0, 1].
    RFloat friction_max_penetration_ratio = 0.1;

    bool operator==(const RSC_SimulationSoftBody& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Rigid-body solver settings.
 */
class RSC_SimulationRigidBody : public RSC_Base {
   public:
    /// Number of PGS iterations per constraint solve.
    RSize constraint_iterations = 15;

    /// Fraction of positional constraint error fed back as bias velocity, [0, 1].
    RFloat constraint_position_correction_ratio = 0.1;

    /// Warm-start damping for the constraint solver, [0, 1]; 1 disables warm start.
    RFloat constraint_warm_start_damping_z = 0.1;

    /// Initial per-joint stiffness factor, [0, 1]; 1 = hard PGS.
    RFloat constraint_stiffness_initial = 1;

    /// Minimum allowed per-joint stiffness after auto-adjust, [0, 1].
    RFloat constraint_stiffness_min_threshold = 5e-2;

    /// Maximum allowed per-joint stiffness after auto-adjust, [0, 1].
    RFloat constraint_stiffness_max_threshold = 1;

    /// Coupling-power threshold (watts) below which per-joint stiffness recovers and
    /// above which it softens.
    RFloat constraint_stiffness_auto_adjust_watt_threshold = 1;

    /// Multiplicative step applied to per-joint stiffness each slow rigid update, [0, 1].
    RFloat constraint_stiffness_auto_adjust_step = 1e-1;

    /// Number of velocity-solve iterations for the contact solver.
    RSize contact_iterations = 20;

    /// Number of position-correction iterations for the contact solver (Box2D NGS style).
    RSize contact_position_correction_iterations = 3;

    /// Contacts one rigid body pair keeps, the deepest plus the farthest from it. Range [1, 8].
    RSize contact_manifold_size = 4;

    /// Fraction of penetration (beyond contact_slop) removed per position-correction iteration.
    RFloat contact_position_correction_ratio = 0.3;

    /// Warm-start damping for the contact solver, [0, 1]; 0 = full warm start, 1 = disabled.
    RFloat contact_warm_start_damping_z = 0.0;

    /// Penetration slop allowed before contact correction kicks in (meters).
    RFloat contact_slop = 1e-4;

    /// Linear velocity damping for rigid bodies, as a decay rate (per second).
    RFloat linear_damping_z = 0.01;

    /// Angular velocity damping for rigid bodies, as a decay rate (per second).
    RFloat angular_damping_z = 0.01;

    bool operator==(const RSC_SimulationRigidBody& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Contact detection / broadphase scheduling settings.
 */
class RSC_SimulationContact : public RSC_Base {
   public:
    /// Sub-interval (in contact detection passes) between contact BVH rebuilds.
    RSize bvh_rebuild_sub_interval = 4;

    /// Sub-interval (in contact detection passes) between contact BVH refits.
    RSize bvh_refit_sub_interval = 1;

    bool operator==(const RSC_SimulationContact& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Vision tracer settings.
 */
class RSC_SimulationVisionTrace : public RSC_Base {
   public:
    /// Indirect-only firefly clamp cap; primary hits at depth 0 unclamped.
    RFloat firefly_cap = 8.0;

    /// Depth at which Russian roulette starts terminating paths.
    RSize russian_roulette_start_depth = 3;

    bool operator==(const RSC_SimulationVisionTrace& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief ReSTIR direct- and global-illumination resampler settings.
 */
class RSC_SimulationVisionReSTIR : public RSC_Base {
   public:
    /// Master enable for the ReSTIR DI/GI resamplers; when false, basic path tracing with NEE.
    bool is_enabled = false;

    /// Max world-space distance (m) between previous and current shading point before the
    /// temporal source reservoir is rejected as stale.
    RFloat disocclusion_position_threshold = 0.005;

    /// Cosine threshold on normal alignment for temporal merge and spatial reuse gates.
    RFloat disocclusion_normal_threshold = 0.906;

    /// Max length of the deterministic specular chain walked by the info pass (PTUD).
    RSize max_specular_chain = 4;

    /// Max temporal samples retained per direct illumination reservoir.
    RSize di_m_cap = 20;

    /// Neighboring pixels resampled by the direct illumination spatial reuse pass.
    RSize di_spatial_neighbors = 10;

    /// Pixel radius of the direct illumination spatial reuse disk.
    RFloat di_spatial_radius = 10.0;

    /// Whether each indirect illumination candidate computes a direct lighting estimate at the
    /// bounce hit point.
    bool is_gi_bounce_nee_enabled = true;

    /// Max temporal samples retained per indirect illumination reservoir.
    RSize gi_m_cap = 30;

    /// Neighboring pixels resampled by the indirect illumination spatial reuse pass.
    RSize gi_spatial_neighbors = 6;

    /// Pixel radius of the indirect illumination spatial reuse disk.
    RFloat gi_spatial_radius = 10.0;

    /// Frame interval at which the indirect illumination reservoir's bounce ray is re-validated.
    RSize gi_validation_period = 6;

    bool operator==(const RSC_SimulationVisionReSTIR& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Disabled denoiser.
 */
class RSC_SimulationVisionDenoiseNone : public RSC_Base {
   public:
    bool operator==(const RSC_SimulationVisionDenoiseNone& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Edge-aware a-trous wavelet denoiser. Constant sigma, spatial only.
 */
class RSC_SimulationVisionDenoiseAtrous : public RSC_Base {
   public:
    /// Number of a-trous wavelet iterations.
    RSize atrous_iterations = 2;

    /// Per-pixel pass-count floor for the a-trous wavelet, applied alongside the roughness^2 cap.
    RSize atrous_min_passes = 1;

    /// Edge-stopping luminance-similarity scale.
    RFloat sigma_color = 1.5;

    /// Edge-stopping normal-similarity exponent.
    RFloat sigma_normal = 128.0;

    /// Edge-stopping depth-similarity scale.
    RFloat sigma_depth = 0.1;

    bool operator==(const RSC_SimulationVisionDenoiseAtrous& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Spatiotemporal Variance-Guided Filter denoiser.
 */
class RSC_SimulationVisionDenoiseSVGF : public RSC_Base {
   public:
    /// Number of a-trous wavelet iterations.
    RSize atrous_iterations = 2;

    /// Per-pixel pass-count floor for the a-trous wavelet, applied alongside the roughness^2 cap.
    RSize atrous_min_passes = 1;

    /// Edge-stopping luminance-similarity scale.
    RFloat sigma_color = 1.5;

    /// Edge-stopping normal-similarity exponent.
    RFloat sigma_normal = 128.0;

    /// Edge-stopping depth-similarity scale.
    RFloat sigma_depth = 0.1;

    /// Temporal EMA weight for color history.
    RFloat temporal_alpha = 0.2;

    /// Temporal EMA weight for moment (variance) history.
    RFloat temporal_alpha_moments = 0.2;

    /// Effective-history-length floor for the temporal EMA, applied alongside the roughness^2 cap.
    RFloat temporal_min_history = 2;

    bool operator==(const RSC_SimulationVisionDenoiseSVGF& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

using RSC_SimulationVisionDenoiseVariant = RS_Variant<
    RSC_SimulationVisionDenoiseNone,
    RSC_SimulationVisionDenoiseAtrous,
    RSC_SimulationVisionDenoiseSVGF>;

/**
 * @brief Vision pipeline settings.
 */
class RSC_SimulationVision : public RSC_Base {
   public:
    /// Trace settings.
    RSC_SimulationVisionTrace trace;

    /// ReSTIR resampler settings.
    RSC_SimulationVisionReSTIR restir;

    /// Denoiser settings.
    RSC_SimulationVisionDenoiseVariant denoise;

    bool operator==(const RSC_SimulationVision& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Sentinel (divergence watchdog) settings.
 */
class RSC_SimulationSentinel : public RSC_Base {
   public:
    /// Master enable for the sentinel; when false, divergence is neither detected nor recovered
    /// from.
    bool is_enabled = true;

    /// Smallest global time step treated as normal; below this the step is reported degenerate and
    /// the sentinel takes over.
    RFloat degenerate_min_dt = 1e-7;

    bool operator==(const RSC_SimulationSentinel& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Default control frequency.
 */
class RSC_SimulationControl : public RSC_Base {
   public:
    /// Default control loop frequency (unit: Hz).
    RFloat frequency = 20;

    bool operator==(const RSC_SimulationControl& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Default record frequency.
 */
class RSC_SimulationRecord : public RSC_Base {
   public:
    /// Default record frequency (unit: Hz).
    RFloat frequency = 20;

    bool operator==(const RSC_SimulationRecord& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Runtime settings such as buffers, accuracy, logging.
 */
class RSC_SimulationRuntime : public RSC_Base {
   public:
    /// Whether the world starts paused.
    bool is_paused_on_start = false;

    /// Hard cap on simulated time (seconds).
    RFloat max_time = 100000000;

    /// Number of frames buffered on device before being flushed to the recorder.
    RSize record_buffer_size = 5;

    /// Whether to persist simulation records.
    bool is_save_record = true;

    /// Compute precision (RSE_FLOAT_32 or RSE_FLOAT_64).
    RSE_FloatType compute_accuracy = RSE_FLOAT_32;

    /// Memory storage precision (RSE_FLOAT_32 or RSE_FLOAT_64).
    RSE_FloatType memory_accuracy = RSE_FLOAT_32;

    /// Log level ("err", "warn", "info", "debug", "trace").
    std::string log_level = "info";

    bool operator==(const RSC_SimulationRuntime& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

/**
 * @brief Top-level simulation settings.
 */
class RSC_Simulation : public RSC_Base {
   public:
    /// Random seed.
    uint64_t seed = 0x123456789abcdef;

    /// Stop condition expression. Simulation halts when this evaluates true.
    RS_MathExpr stop_condition;

    // Sub-field settings, note we are not using rsc ref here, no need.
    RSC_SimulationEnvironment environment;
    RSC_SimulationTime time;
    RSC_SimulationSoftBody soft_body;
    RSC_SimulationRigidBody rigid_body;
    RSC_SimulationContact contact;
    RSC_SimulationVision vision;
    RSC_SimulationSentinel sentinel;
    RSC_SimulationControl control;
    RSC_SimulationRecord record;
    RSC_SimulationRuntime runtime;

    bool operator==(const RSC_Simulation& other) const;

   protected:
    void _validate(RS_RSCValidator& validator) const override;
    void _read(const RS_RSCFileTreeReader& tree) override;
    void _write(RS_RSCFileTreeWriter& tree) const override;
    void _deserialize(RS_RSCArchive& archive) override;
    void _serialize(RS_RSCArchive& archive) const override;
    RSCRef<RSC_Base> _shallowCopy() const override { return RSCRef<RSC_Base>::fromThis(this); }
};

}  // namespace rsc

}  // namespace rise

#endif  // RSC_SIMULATION_H
