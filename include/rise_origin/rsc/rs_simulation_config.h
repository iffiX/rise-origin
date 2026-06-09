// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_CONFIG_H
#define RS_SIMULATION_CONFIG_H

#include <string>

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_float.h"
#include "rise_origin/utils/rs_math_expr.h"
#include "rs_base_config.h"

namespace rise {

RS_ENUM(
    RSE_VisionDenoiseType,
    unsigned int,
    (RSE_VISION_DENOISE_NONE, 0),
    (RSE_VISION_DENOISE_ATROUS, 1),
    (RSE_VISION_DENOISE_SVGF, 2)
)

class RS_SimulationConfig : public RS_BaseConfig {
   public:
    /***************************************************************
     * Seed
     ***************************************************************/
    /**
     * Seed for the simulation's random number generator.
     *
     * Determines the reproducibility of any stochastic behaviour (e.g. random
     * material/process noise). Any 64-bit value is valid; reuse the same seed to
     * reproduce a run exactly.
     */
    uint64_t seed = 0x123456789abcdef;

    /***************************************************************
     * Stop condition
     ***************************************************************/
    /**
     * Stop condition of the simulation, as a math expression (e.g. "t > 10").
     *
     * Evaluated each step; the simulation ends when it becomes true. Empty means
     * no stop condition (runs until externally stopped).
     */
    RS_MathExpr stop_condition;

    /***************************************************************
     * Runtime (cannot be modified after simulation starts)
     ***************************************************************/
    /// Interval (in physics steps) between integration time-step recomputations.
    RSize runtime_dt_update_interval = 10;

    /// Interval (in physics steps) between rigid body subsystem updates.
    RSize runtime_rigidbody_update_interval = 10;

    /// Interval (in physics steps) between contact detection updates.
    RSize runtime_contact_update_interval = 10;

    /// Sub-interval (in contact updates) between contact BVH rebuilds.
    RSize runtime_contact_bvh_rebuild_sub_interval = 4;

    /// Sub-interval (in contact updates) between contact BVH refits.
    RSize runtime_contact_bvh_refit_sub_interval = 1;

    /// Interval (in physics steps) between mechanical signal updates.
    RSize runtime_mechanical_signal_update_interval = 10;

    /// Interval (in physics steps) between divergence (NaN/blow-up) checks.
    RSize runtime_divergence_check_interval = 100;

    /// Number of frames buffered on device before being flushed to the recorder.
    RSize runtime_record_buffer_size = 5;

    /// Hard cap on simulated time (seconds).
    RFloat runtime_max_time = 1000000;

    /// Whether to persist simulation records.
    bool runtime_save_record = true;

    /// Compute accuracy.
    RSE_FloatType runtime_simulation_compute_accuracy = RSE_FLOAT_32;

    /// Memory storage accuracy.
    RSE_FloatType runtime_simulation_memory_accuracy = RSE_FLOAT_32;

    /// Log level: "err", "warn", "info", "debug", or "trace".
    std::string runtime_log_level = "info";

    /***************************************************************
     * Integration
     ***************************************************************/
    /**
     * Manually specified integration time step, in seconds.
     *
     * Range: >= 0. A value of 0 (the default) means "auto": the engine uses the
     * stability-recommended time step computed from material stiffness, mass and
     * voxel size. Set a positive value only to override that recommendation; too
     * large a step will make stiff materials/contacts blow up.
     */
    RFloat integration_time_step = 0;

    /**
     * Fraction of the recommended time step to actually use.
     *
     * Range: (0, 1]. 1 uses the full recommended step; 0.1 uses one tenth of it
     * (smaller = more stable but slower). Only meaningful when @ref
     * integration_time_step is 0 (auto). Lower this if a simulation is unstable
     * near the stability limit.
     */
    RFloat integration_dt_frac = 1;

    /***************************************************************
     * Soft body
     ***************************************************************/
    /**
     * Internal (structural) damping ratio for soft links.
     *
     * Range: [0, 1] as a fraction of critical damping. Damps relative motion
     * within a soft body (bond vibrations); the coefficient scales with the
     * link's mass/stiffness so the ratio is consistent across materials. 0 = no
     * internal damping (springy/oscillatory), 1 = critically damped.
     */
    RFloat softbody_internal_damping_z = 0.5;

    /**
     * Contact damping ratio for soft voxels.
     *
     * Range: [0, 1] as a fraction of critical damping. Controls how much normal
     * contact velocity is dissipated, i.e. bounciness on impact (0 = elastic,
     * 1 = critically damped / no bounce). Scales with mass/stiffness so it is
     * density-independent.
     */
    RFloat softbody_contact_damping_z = 0.2;

    /**
     * Global (environmental / air-drag) damping ratio for soft voxels.
     *
     * Range: [0, 1]. Applied as a velocity-proportional drag whose coefficient
     * scales with the voxel's mass/stiffness. Note this is NOT the same physical
     * quantity as the rigid body's @ref rigidbody_linear_damping_z (a bare
     * velocity decay rate), so equal numeric values do not produce equal drag
     * between soft and rigid bodies.
     */
    RFloat softbody_global_damping_z = 0.001;

    /***************************************************************
     * Rigid body
     ***************************************************************/
    /**
     * Number of Projected Gauss-Seidel (PGS) iterations per constraint solve.
     *
     * Range: >= 0. More iterations make joints/constraints stiffer and converge
     * better at higher CPU/GPU cost. Typical values are 5-20; 0 effectively
     * disables constraint solving.
     */
    RSize rigidbody_constraint_iterations = 10;

    /**
     * Position-correction ratio applied each constraint solver call.
     *
     * Range: [0, 1]. Fraction of the positional constraint error fed back as a
     * bias velocity to pull joints back together. 0 = no correction (joints may
     * drift apart), higher = stiffer correction but more energy/jitter. ~0.1 is
     * a common stable choice.
     */
    RFloat rigidbody_constraint_position_correction_ratio = 0.1;

    /**
     * Warm-start damping for the constraint solver.
     *
     * Range: [0, 1]. Scales how much of the previous solve's accumulated impulse
     * is reused to warm-start the next solve. 1 disables warm start (start from
     * zero each step); 0 uses the full previous lambda (fastest convergence but
     * can be less stable).
     */
    RFloat rigidbody_constraint_warm_start_damping_z = 0.1;

    /**
     * Initial per-joint compliance factor.
     *
     * Range: [0, 1]. New joints start at this value; 1 = hard PGS, lower = compliant.
     * The runtime auto-adjusts each joint's compliance every slow rigid update; see
     * @ref rigidbody_constraint_compliance_auto_adjust_watt_threshold.
     */
    RFloat rigidbody_constraint_compliance_initial = 1;

    /**
     * Minimum allowed per-joint compliance after auto-adjust.
     *
     * Range: [0, 1]. Floor for the adaptation; prevents the constraint from fully
     * releasing under heavy soft-coupling load.
     */
    RFloat rigidbody_constraint_compliance_min_threshold = 5e-2;

    /**
     * Maximum allowed per-joint compliance after auto-adjust.
     *
     * Range: [0, 1]. Ceiling for the adaptation; 1 means full hard PGS.
     */
    RFloat rigidbody_constraint_compliance_max_threshold = 1;

    /**
     * Power threshold above which per-joint compliance softens.
     *
     * Units: watts (J/s). Each slow rigid update, the runtime sums the cumulative
     * coupling work delivered to the joint's two bodies and divides by the slow
     * rigid-step interval to get a power signal. Above this threshold the compliance
     * is scaled by `(1 - step)`; below it the compliance is scaled by `(1 + step)`.
     * Using power (not energy) keeps tuning invariant to changes in the slow
     * rigid-step interval.
     */
    RFloat rigidbody_constraint_compliance_auto_adjust_watt_threshold = 1;

    /**
     * Multiplicative step applied to per-joint compliance each slow rigid update.
     *
     * Range: [0, 1]. The compliance is multiplied by `(1 - step)` when the power
     * signal exceeds the watt threshold, or by `(1 + step)` otherwise, then clamped
     * to [@ref rigidbody_constraint_compliance_min_threshold,
     *      @ref rigidbody_constraint_compliance_max_threshold].
     */
    RFloat rigidbody_constraint_compliance_auto_adjust_step = 1e-2;

    /**
     * Number of velocity-solve iterations for the rigid body contact solver.
     *
     * Range: >= 0. More iterations resolve contact velocity / friction more
     * accurately at higher cost.
     */
    RSize rigidbody_contact_iterations = 20;

    /**
     * Number of position-correction iterations for the rigid body contact solver.
     * The contact position correction idea is borrowed from Box2D NGS.
     *
     * Range: >= 0. Sweeps that push interpenetrating bodies apart directly on
     * positions, after the velocity solve. Small number is preferred.
     */
    RSize rigidbody_contact_position_correction_iterations = 3;

    /**
     * Position-correction ratio for the rigid body contact solver.
     *
     * Range: [0, 1]. Fraction of penetration (beyond @ref rigidbody_contact_slop)
     * removed per position-correction iteration; see @ref
     * rigidbody_constraint_position_correction_ratio for the analogous joint term.
     */
    RFloat rigidbody_contact_position_correction_ratio = 0.4;

    /**
     * Warm-start damping for the contact solver.
     *
     * Range: [0, 1]. Scales the previous-step per-voxel `J^T*lambda` seed by
     * `(1 - rigidbody_contact_warm_start_damping_z)`. 0 = full warm start; 1 = disabled.
     */
    RFloat rigidbody_contact_warm_start_damping_z = 0.0;

    /**
     * Restitution (bounciness) of rigid body contacts.
     *
     * Range: [0, 1]. 0 = fully inelastic (no bounce), 1 = elastic.
     */
    RFloat rigidbody_contact_restitution = 0;

    /**
     * Penetration slop allowed before contact correction kicks in (meters).
     *
     * Range: >= 0. A small tolerance that prevents jitter from tiny residual
     * penetrations; only depth beyond the slop is corrected.
     */
    RFloat rigidbody_contact_slop = 1e-4;

    /**
     * Linear velocity damping for rigid bodies, as a decay rate (per second).
     *
     * Range: [0, 1]. Applied as momentum *= (1 - rigidbody_linear_damping_z * dt),
     * i.e. an environmental drag independent of mass: terminal velocity ~ g /
     * rigidbody_linear_damping_z. 0 = no drag (free fall). This is a different
     * model from the soft @ref softbody_global_damping_z; see that field's note.
     */
    RFloat rigidbody_linear_damping_z = 0.01;

    /**
     * Angular velocity damping for rigid bodies, as a decay rate (per second).
     *
     * Range: [0, 1]. Applied as angular_momentum *= (1 - rigidbody_angular_damping_z * dt).
     * 0 = no rotational drag.
     */
    RFloat rigidbody_angular_damping_z = 0.01;

    /***************************************************************
     * Vision
     ***************************************************************/
    /**
     * Indirect-only firefly clamp cap; primary hits at depth 0 unclamped.
     *
     * Range: > 0. Low values suppress fireflies from rare bright sample paths
     * but bias the indirect bounce luminance downwards; high values let
     * fireflies through.
     */
    RFloat vision_trace_firefly_cap = 8.0;

    /**
     * Depth at which Russian roulette starts terminating paths.
     *
     * Range: >= 0. Paths above this depth get a survival probability based on
     * throughput max; rejected paths terminate. Lower values cut deep paths
     * earlier (cheaper, more variance), higher values keep more depth (slower,
     * smoother).
     */
    RSize vision_trace_russian_roulette_start_depth = 3;

    /**
     * Master enable for the ReSTIR direct-illumination and global-illumination
     * resamplers. When false, the executor uses basic path tracing with NEE
     * lighting.
     */
    bool vision_is_restir_enabled = true;

    /**
     * Maximum world-space distance between the previous frame's stored shading
     * point and the current frame's shading point before the temporal source
     * reservoir is rejected as stale. Used by both DI and GI temporal gates.
     *
     * Range: > 0, in meters. Tight enough that motion-vector reprojection
     * round-snapping to a neighboring pixel fails the gate when the neighbor
     * world point differs from the current one — avoids ghost stripes under
     * camera rotation. Larger values let the gate accept off-by-one reuse;
     * smaller values produce more noise on newly-disoccluded regions that
     * the reservoirs then rebuild within a few frames.
     * 
     * For complex illuniation scenes relaxing this may help.
     */
    RFloat vision_restir_disocclusion_position_threshold = 0.005;

    /**
     * Cosine threshold on the normal-alignment check of the multi-field
     * disocclusion gate for temporal merge and the geometric similarity gate
     * for spatial reuse. Used by both DI and GI gates. A previous-frame or
     * neighbor reservoir is accepted only when its stored normal's dot product
     * with the current pixel's normal exceeds this value.
     *
     * Range: (0, 1]. Default `cos(25 deg) = 0.906`; raise toward 1 to be
     * stricter (cleaner silhouettes, more rejected reuse), lower it for more
     * permissive reuse (more variance reduction, more bleed across normal
     * discontinuities).
     */
    RFloat vision_restir_disocclusion_normal_threshold = 0.906;

    /**
     * Maximum length of the deterministic specular chain walked from each
     * primary hit by the ReSTIR info pass before treating the current hit as
     * the chain endpoint in PTUD (Path Through Until Diffuse).
     *
     * Range: >= 1. Higher values let longer specular chains (e.g. mirror
     * through glass into glass) keep ReSTIR's variance reduction at the
     * underlying diffuse endpoint; lower values truncate sooner and may
     * darken a very reflective hall-of-mirrors scene's corners.
     */
    RSize vision_restir_max_specular_chain = 4;

    /**
     * Maximum number of temporal samples retained by each direct
     * illumination reservoir before older samples are decayed.
     *
     * Range: >= 1, in samples. Higher values smooth direct illumination
     * across more frames but slow the response to lighting and geometry
     * changes.
     *
     * Bitterli et al. 2020 §5.
     */
    RSize vision_restir_di_m_cap = 20;

    /**
     * Total number of neighboring pixels resampled by the spatial reuse
     * pass for the direct illumination reservoir.
     *
     * Range: >= 0, in samples. 0 disables spatial reuse entirely.
     */
    RSize vision_restir_di_spatial_neighbors = 10;

    /**
     * Pixel radius of the uniform disk from which spatial reuse neighbors
     * are sampled for the direct illumination reservoir.
     *
     * Range: > 0, in pixels. Wider radius shares evidence across more of
     * the image at the cost of more rejected neighbors at silhouettes and
     * depth discontinuities.
     *
     * Ouyang et al. 2021 §5.
     */
    RFloat vision_restir_di_spatial_radius = 10.0;

    /**
     * Whether each indirect illumination candidate computes a direct
     * lighting estimate at the bounce hit point. Without it, only sky and
     * emissive hits contribute to the indirect bounce.
     */
    bool vision_restir_gi_is_bounce_nee_enabled = true;

    /**
     * Maximum number of temporal samples retained by each indirect
     * illumination reservoir before older samples are decayed.
     *
     * Range: >= 1, in samples. Higher values smooth indirect light across
     * more frames but slow the response to lighting and geometry changes.
     *
     * Bitterli et al. 2020 §5.
     */
    RSize vision_restir_gi_m_cap = 30;

    /**
     * Total number of neighboring pixels resampled by the spatial reuse
     * pass for the indirect illumination reservoir.
     *
     * Range: >= 0, in samples. 0 disables spatial reuse entirely.
     */
    RSize vision_restir_gi_spatial_neighbors = 6;

    /**
     * Pixel radius of the uniform disk from which spatial reuse neighbors
     * are sampled for the indirect illumination reservoir.
     *
     * Range: > 0, in pixels. Smaller than the direct illumination default
     * because indirect bounce evidence is more sensitive to shading point
     * divergence across neighbors.
     *
     * Ouyang et al. 2021 §5.
     */
    RFloat vision_restir_gi_spatial_radius = 10.0;

    /**
     * Frame interval at which the indirect illumination reservoir's
     * stored bounce ray is re-traced from the current shading point to
     * check whether the geometry along the path has changed. Stale
     * samples are dropped before the temporal merge.
     *
     * Range: >= 0, in frames. 0 disables validation. Smaller intervals
     * react faster to scene changes at proportional cost.
     *
     * Ouyang et al. 2021 §4.3, default 6.
     */
    RSize vision_restir_gi_validation_period = 6;

    /**
     * Denoise filter type.
     *
     * RSE_VISION_DENOISE_NONE skips the denoise pass entirely.
     * RSE_VISION_DENOISE_ATROUS runs the edge-aware a-trous wavelet filter.
     * RSE_VISION_DENOISE_SVGF runs the Spatiotemporal Variance-Guided Filter
     * (Schied et al. 2017): temporal accumulation + moment-based variance
     * estimate driving a variance-guided a-trous wavelet.
     */
    RSE_VisionDenoiseType vision_denoise_type = RSE_VISION_DENOISE_NONE;

    /**
     * Number of a-trous wavelet iterations.
     *
     * Range: >= 0. Each pass doubles the sample step (1, 2, 4, ...), so total
     * coverage is 4 * (2^N - 1) + 1 pixels per axis. Larger = smoother but more
     * compute.
     */
    RSize vision_denoise_atrous_iterations = 3;

    /**
     * Per-pixel pass-count floor for the a-trous wavelet, applied alongside
     * the roughness^2 cap.
     *
     * Range: >= 0. 0 = smooth surfaces (mirror, glass) skip filtering entirely
     * (passthrough); 1 = smooth surfaces still get at least one pass.
     */
    RSize vision_denoise_atrous_min_passes = 1;

    /**
     * Edge-stopping luminance-similarity scale.
     *
     * Range: > 0. Smaller = stricter edge (preserves more luminance variation),
     * larger = looser (more smoothing across radiance changes).
     */
    RFloat vision_denoise_sigma_color = 1.5;

    /**
     * Edge-stopping normal-similarity exponent.
     *
     * Range: > 0. The weight is pow(max(0, dot(n_c, n_neighbor)), sigma_normal).
     * Larger = stricter edge (face boundaries stay crisp).
     */
    RFloat vision_denoise_sigma_normal = 128.0;

    /**
     * Edge-stopping depth-similarity scale.
     *
     * Range: > 0. The weight is exp(-(d_diff^2) / sigma_depth^2). Smaller =
     * stricter edge (preserves silhouettes), larger = looser.
     */
    RFloat vision_denoise_sigma_depth = 0.1;

    /**
     * SVGF temporal EMA weight for color history.
     *
     * Range: (0, 1]. Used only when @ref vision_denoise_type is
     * RSE_VISION_DENOISE_SVGF. The blend is
     *     blended = (1 - eff_alpha) * prev + eff_alpha * current
     * with eff_alpha = max(temporal_alpha, 1 / capped_history_length), so
     * smaller values give longer effective history. Bootstrap term keeps the
     * first few frames from over-relying on a near-empty history.
     */
    RFloat vision_denoise_temporal_alpha = 0.2;

    /**
     * SVGF temporal EMA weight for first/second luminance moments (variance).
     *
     * Range: (0, 1]. Same blend shape as @ref vision_denoise_temporal_alpha
     * but applied to (m1, m2). Variance is then derived as max(0, m2 - m1^2)
     * once the history length reaches the 4-frame threshold; below that, a
     * 7x7 spatial luminance variance over the blended canvas is used instead.
     */
    RFloat vision_denoise_temporal_alpha_moments = 0.2;

    /**
     * SVGF effective-history-length floor, applied alongside the roughness^2
     * cap on the temporal EMA.
     *
     * Range: >= 1. The cap is capped_len = max(temporal_min_history,
     * roughness^2 * new_len), so 1.0 = smooth surfaces (mirror, glass) get
     * no reuse (eff_alpha = 1, single-frame); 1.5 = smooth surfaces get
     * ~67% new + 33% prev; values > 2 produce visible long-tail ghosting on
     * low-roughness pixels because primary-surface motion vectors do not
     * track reflected content.
     */
    RFloat vision_denoise_temporal_min_history = 2;

    /***************************************************************
     * Control
     ***************************************************************/
    /**
     * Frequency (Hz) at which the user control callback is invoked.
     *
     * Range: > 0. Independent of the physics time step; the engine runs the
     * control callback every 1 / control_frequency seconds of simulated time.
     */
    RFloat control_frequency = 20;

    /***************************************************************
     * Record
     ***************************************************************/
    /**
     * Frequency (Hz) at which recording / camera capture is performed.
     *
     * Range: > 0. As with @ref control_frequency, expressed in simulated time.
     */
    RFloat record_frequency = 20;

    /**
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void _validate() const override;

    /**
     * @throw RS_InvalidConfigurationException When the configuration is invalid.
     */
    void _read(const RS_ConfigPropertyTree &config_tree) override;

    template <typename Serializer>
    void serialize(Serializer &s) {
        s("seed", seed);
        s("stop_condition", stop_condition);
        s("runtime_dt_update_interval", runtime_dt_update_interval);
        s("runtime_rigidbody_update_interval", runtime_rigidbody_update_interval);
        s("runtime_contact_update_interval", runtime_contact_update_interval);
        s("runtime_contact_bvh_rebuild_sub_interval", runtime_contact_bvh_rebuild_sub_interval);
        s("runtime_contact_bvh_refit_sub_interval", runtime_contact_bvh_refit_sub_interval);
        s("runtime_mechanical_signal_update_interval", runtime_mechanical_signal_update_interval);
        s("runtime_divergence_check_interval", runtime_divergence_check_interval);
        s("runtime_record_buffer_size", runtime_record_buffer_size);
        s("runtime_max_time", runtime_max_time);
        s("runtime_save_record", runtime_save_record);
        s("runtime_simulation_compute_accuracy", runtime_simulation_compute_accuracy);
        s("runtime_simulation_memory_accuracy", runtime_simulation_memory_accuracy);
        s("runtime_log_level", runtime_log_level);
        s("integration_time_step", integration_time_step);
        s("integration_dt_frac", integration_dt_frac);
        s("softbody_internal_damping_z", softbody_internal_damping_z);
        s("softbody_contact_damping_z", softbody_contact_damping_z);
        s("softbody_global_damping_z", softbody_global_damping_z);
        s("rigidbody_constraint_iterations", rigidbody_constraint_iterations);
        s("rigidbody_constraint_position_correction_ratio",
          rigidbody_constraint_position_correction_ratio);
        s("rigidbody_constraint_warm_start_damping_z", rigidbody_constraint_warm_start_damping_z);
        s("rigidbody_constraint_compliance_initial", rigidbody_constraint_compliance_initial);
        s("rigidbody_constraint_compliance_min_threshold",
          rigidbody_constraint_compliance_min_threshold);
        s("rigidbody_constraint_compliance_max_threshold",
          rigidbody_constraint_compliance_max_threshold);
        s("rigidbody_constraint_compliance_auto_adjust_watt_threshold",
          rigidbody_constraint_compliance_auto_adjust_watt_threshold);
        s("rigidbody_constraint_compliance_auto_adjust_step",
          rigidbody_constraint_compliance_auto_adjust_step);
        s("rigidbody_contact_iterations", rigidbody_contact_iterations);
        s("rigidbody_contact_position_correction_iterations",
          rigidbody_contact_position_correction_iterations);
        s("rigidbody_contact_position_correction_ratio",
          rigidbody_contact_position_correction_ratio);
        s("rigidbody_contact_warm_start_damping_z", rigidbody_contact_warm_start_damping_z);
        s("rigidbody_contact_restitution", rigidbody_contact_restitution);
        s("rigidbody_contact_slop", rigidbody_contact_slop);
        s("rigidbody_linear_damping_z", rigidbody_linear_damping_z);
        s("rigidbody_angular_damping_z", rigidbody_angular_damping_z);
        s("vision_trace_firefly_cap", vision_trace_firefly_cap);
        s("vision_trace_russian_roulette_start_depth", vision_trace_russian_roulette_start_depth);
        s("vision_is_restir_enabled", vision_is_restir_enabled);
        s("vision_restir_disocclusion_position_threshold",
          vision_restir_disocclusion_position_threshold);
        s("vision_restir_disocclusion_normal_threshold",
          vision_restir_disocclusion_normal_threshold);
        s("vision_restir_max_specular_chain", vision_restir_max_specular_chain);
        s("vision_restir_di_m_cap", vision_restir_di_m_cap);
        s("vision_restir_di_spatial_neighbors", vision_restir_di_spatial_neighbors);
        s("vision_restir_di_spatial_radius", vision_restir_di_spatial_radius);
        s("vision_restir_gi_is_bounce_nee_enabled", vision_restir_gi_is_bounce_nee_enabled);
        s("vision_restir_gi_m_cap", vision_restir_gi_m_cap);
        s("vision_restir_gi_spatial_neighbors", vision_restir_gi_spatial_neighbors);
        s("vision_restir_gi_spatial_radius", vision_restir_gi_spatial_radius);
        s("vision_restir_gi_validation_period", vision_restir_gi_validation_period);
        s("vision_denoise_type", vision_denoise_type);
        s("vision_denoise_atrous_iterations", vision_denoise_atrous_iterations);
        s("vision_denoise_atrous_min_passes", vision_denoise_atrous_min_passes);
        s("vision_denoise_sigma_color", vision_denoise_sigma_color);
        s("vision_denoise_sigma_normal", vision_denoise_sigma_normal);
        s("vision_denoise_sigma_depth", vision_denoise_sigma_depth);
        s("vision_denoise_temporal_alpha", vision_denoise_temporal_alpha);
        s("vision_denoise_temporal_alpha_moments", vision_denoise_temporal_alpha_moments);
        s("vision_denoise_temporal_min_history", vision_denoise_temporal_min_history);
        s("control_frequency", control_frequency);
        s("record_frequency", record_frequency);
    }
};

}  // namespace rise
#endif  // RS_SIMULATION_CONFIG_H
