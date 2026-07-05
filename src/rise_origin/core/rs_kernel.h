// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_H
#define RS_KERNEL_H

#include <mutex>
#include <unordered_map>

#include "rise_origin/device/data/rs_value.h"
#include "rise_origin/logic/vision/rs_vision_denoise_m.h"
#include "rise_origin/utils/interchange/rs_json.h"
#include "rise_origin/utils/math/rs_math_expr.h"
#include "rs_context.h"

namespace rise {

/**
 * Subsystem operations whose elapsed time is tracked on the kernel, so device kernels can read the
 * true time since the operation last ran as its effective dt.
 */
RS_ENUM(
    RSE_KernelUpdateOperation,
    RIndex,
    (RSE_KERNEL_UPDATE_OPERATION_SOFT_BODY, 0),
    (RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_CONSTRAINT_AND_CONTACT, 1),
    (RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_INTEGRATE, 2),
    (RSE_KERNEL_UPDATE_OPERATION_MAX, 3)
)

/**
 * Runtime states are maintained by host and device separately
 */
struct RS_KernelRuntimeStates {
    RBigFloat dt = 0;  // actual time step

    RBigFloat time = 0;  // current time of the simulation in seconds

    RBigIndex step = 0;

    RState last_update_stamped = 0;

    // Per-operation last-run time and elapsed time since, indexed by RSE_KernelUpdateOperation.
    RBigFloat last_update_time[RSE_KERNEL_UPDATE_OPERATION_MAX] = {0};
    RBigFloat last_update_passed_time[RSE_KERNEL_UPDATE_OPERATION_MAX] = {0};
};

struct RS_KernelHostData {
    RS_KernelRuntimeStates runtime_states;

    // Simulation stop condition.
    RS_MathExpr stop_condition;

    // Metrics
    RS_JSON metrics;

    // Event time marker
    std::unordered_map<std::string, RBigFloat> event_time;

    // Event counter
    std::unordered_map<std::string, RBigSize> event_count;

    // For determining PGS steps
    RSize constraint_joint_max_colors = 0;

    // Max per-structure counts (for block-based PGS launch sizing)
    RSize max_constraint_joint_per_structure = 0;
    RSize max_rigid_body_per_structure = 0;
};

/**
 * Note:
 * memory management is handled by RS_KernelManager since some
 * creation information is complex, not needed for kernel execution,
 * and it would be messy to put them in #host_data
 * kernel execution logic is implemented by global functions called by
 * RS_Kernel*Executor::doTimeStep
 */
template <typename Context = RS_AnyContext>
struct alignas(8) RS_KernelBase {
    static_assert(sizeof(Context) <= sizeof(RS_AnyContext), "Context is too big");
    /*
     * Note: the following update functions are designed to be called separately
     * on host and device on their corresponding copy of the kernel
     */
    void updateTimeOnHost() const {
        host_data->runtime_states.step++;
        host_data->runtime_states.time += host_data->runtime_states.dt;
    }

    void updateDtOnHost(RFloat dt) const { host_data->runtime_states.dt = dt; }

    void updateOperationTimeOnHost(RSE_KernelUpdateOperation operation) const {
        auto& states = host_data->runtime_states;
        RState mask = RState(1) << operation;
        RBigFloat passed;
        if (!(states.last_update_stamped & mask)) {
            passed = states.dt;
            states.last_update_stamped |= mask;
        } else {
            passed = states.time - states.last_update_time[operation];
        }
        states.last_update_passed_time[operation] = passed;
        states.last_update_time[operation] = states.time;
    }

    HOST_DEVICE_INLINE_QUALIFIER void updateTimeOnDevice() {
        runtime_states.get().step++;
        runtime_states.get().time += runtime_states.get().dt;
    }

    HOST_DEVICE_INLINE_QUALIFIER void updateDtOnDevice(RFloat dt) { runtime_states.get().dt = dt; }

    HOST_DEVICE_INLINE_QUALIFIER void updateOperationTimeOnDevice(
        RSE_KernelUpdateOperation operation
    ) {
        auto& states = runtime_states.get();
        RState mask = RState(1) << operation;
        RBigFloat passed;
        if (!(states.last_update_stamped & mask)) {
            passed = states.dt;
            states.last_update_stamped |= mask;
        } else {
            passed = states.time - states.last_update_time[operation];
        }
        states.last_update_passed_time[operation] = passed;
        states.last_update_time[operation] = states.time;
    }

    [[nodiscard]] bool isStopConditionMet() const {
        // An unset/empty stop_condition means "never stop on condition"
        if (host_data->stop_condition.empty()) return false;
        if (host_data->stop_condition[0].op == RSE_ME_END) return false;
        std::unordered_map<std::string, RFloat> variables;
        for (auto& [key, value] : host_data->metrics.items()) {
            variables[key] = value.template get<RFloat>();
        }
        variables["t"] = host_data->runtime_states.time;
        return RS_MathExprExecutor::eval(variables, host_data->stop_condition) > 0;
    }

    [[nodiscard]] RBigFloat getHostTime() const { return host_data->runtime_states.time; }

    [[nodiscard]] RBigIndex getHostStep() const { return host_data->runtime_states.step; }

    [[nodiscard]] RBigFloat getHostDt() const { return host_data->runtime_states.dt; }

    [[nodiscard]] RBigFloat getHostTimeSinceLastOperation(
        RSE_KernelUpdateOperation operation
    ) const {
        return host_data->runtime_states.last_update_passed_time[operation];
    }

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER RBigFloat getDeviceTime() const {
        return runtime_states.get().time;
    }

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER RBigIndex getDeviceStep() const {
        return runtime_states.get().step;
    }

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER RBigFloat getDeviceDt() const {
        return runtime_states.get().dt;
    }

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER RBigFloat
    getDeviceTimeSinceLastOperation(RSE_KernelUpdateOperation operation) const {
        return runtime_states.get().last_update_passed_time[operation];
    }

    [[nodiscard]] RBigFloat getTimeSinceLastExecutionOfEvent(const std::string& event_name) const {
        auto it = host_data->event_time.find(event_name);
        if (it == host_data->event_time.end()) return host_data->runtime_states.time;
        return host_data->runtime_states.time - it->second;
    }

    [[nodiscard]] RBigSize getExecutionCountOfEvent(const std::string& event_name) const {
        auto it = host_data->event_count.find(event_name);
        if (it == host_data->event_count.end()) return 0;
        return it->second;
    }

    void executeEvent(const std::string& event_name) const {
        host_data->event_time[event_name] = host_data->runtime_states.time;
        host_data->event_count[event_name]++;
    }

    void clearExecutionCountOfEvent(const std::string& event_name) const {
        host_data->event_count[event_name] = 0;
    }

    void terminate() { is_terminated = true; }

    void pause() { is_paused = true; }

    void resume() { is_paused = false; }

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER bool isTerminated() const { return is_terminated; }

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER bool isPaused() const { return is_paused; }

    template <typename NewContext>
    RS_KernelBase<NewContext>& castAsContextType() {
        return *reinterpret_cast<RS_KernelBase<NewContext>*>(this);
    }

    /**
     * Runtime data (constant data, set by host, kernel read only)
     */
    bool is_terminated = false;
    bool is_paused = false;

    // Simulation characterization.
    uint64_t seed = 0x123456789abcdef;

    bool time_is_aligned_to_wallclock = false;

    RFloat softbody_internal_damping_z = 0;
    RFloat softbody_contact_damping_z = 0;
    RFloat softbody_global_damping_z = 0;
    RFloat softbody_integration_frequency_multiplier = 3.5;
    RFloat softbody_link_max_compression = 0.8;

    RSize rigidbody_constraint_iterations = 0;
    RFloat rigidbody_constraint_position_correction_ratio = 0;
    RFloat rigidbody_constraint_warm_start_damping_z = 1;
    RFloat rigidbody_constraint_stiffness_initial = 1;
    RFloat rigidbody_constraint_stiffness_min_threshold = 0;
    RFloat rigidbody_constraint_stiffness_max_threshold = 1;
    RFloat rigidbody_constraint_stiffness_auto_adjust_watt_threshold = 0;
    RFloat rigidbody_constraint_stiffness_auto_adjust_step = 0;
    RSize rigidbody_contact_iterations = 0;
    RSize rigidbody_contact_position_correction_iterations = 0;
    RFloat rigidbody_contact_position_correction_ratio = 0;
    RFloat rigidbody_contact_warm_start_damping_z = 0;
    RFloat rigidbody_contact_slop = 0;
    RFloat rigidbody_linear_damping_z = 0;
    RFloat rigidbody_angular_damping_z = 0;

    RFloat vision_trace_firefly_cap = 8;
    RSize vision_trace_russian_roulette_start_depth = 3;

    bool vision_is_restir_enabled = false;
    RFloat vision_restir_disocclusion_position_threshold = 0;
    RFloat vision_restir_disocclusion_normal_threshold = 0;
    RSize vision_restir_max_specular_chain = 0;
    RSize vision_restir_di_m_cap = 0;
    RSize vision_restir_di_spatial_neighbors = 0;
    RFloat vision_restir_di_spatial_radius = 0;
    bool vision_restir_gi_is_bounce_nee_enabled = false;
    RSize vision_restir_gi_m_cap = 0;
    RSize vision_restir_gi_spatial_neighbors = 0;
    RFloat vision_restir_gi_spatial_radius = 0;
    RSize vision_restir_gi_validation_period = 0;

    RSE_VisionDenoiseType vision_denoise_type = RSE_VISION_DENOISE_NONE;
    RSize vision_denoise_atrous_iterations = 0;
    RSize vision_denoise_atrous_min_passes = 0;
    RFloat vision_denoise_sigma_color = 0.0;
    RFloat vision_denoise_sigma_normal = 0.0;
    RFloat vision_denoise_sigma_depth = 0.0;
    RFloat vision_denoise_temporal_alpha = 0.0;
    RFloat vision_denoise_temporal_alpha_moments = 0.0;
    RFloat vision_denoise_temporal_min_history = 0.0;

    RFloat control_frequency = 0;
    RFloat record_frequency = 0;

    RFloat world_grav_acc = 0;

    // Simulation flags
    // True when the emissive light set changed and the previous reservoirs may index a removed
    // light. The ReSTIR resample skips temporal and spatial reuse for one frame so a stale
    // reservoir light index is never dereferenced.
    bool vision_is_restir_reservoir_stale = false;

    /**
     * Runtime states
     */
    device::RS_Value<RS_KernelRuntimeStates> runtime_states;

    /**
     * Host data
     * This is always a pointer pointing to host memory
     */
    RS_KernelHostData* host_data = nullptr;

    using ContextType = Context;

    // Data structure that holds info for voxels, links, materials
    // It is always at the end, so we can cast the RS_AnyContext to desired context
    RSE_FloatType ctx_accuracy = RSE_FLOAT_32;
    Context ctx;
};

using RS_Kernel = RS_KernelBase<>;

template <typename Function>
void castKernelPtrAndCall(RS_Kernel* kernel, Function function) {
    static_assert(sizeof(RS_Context<RSE_FLOAT_32>) <= sizeof(RS_AnyContext), "Context is too big");
    static_assert(sizeof(RS_Context<RSE_FLOAT_64>) <= sizeof(RS_AnyContext), "Context is too big");
    switch (kernel->ctx_accuracy) {
        case RSE_FLOAT_32:
            function(&kernel->castAsContextType<RS_Context<RSE_FLOAT_32>>());
            break;
        case RSE_FLOAT_64:
            function(&kernel->castAsContextType<RS_Context<RSE_FLOAT_64>>());
            break;
        default:
            RS_THROW(RS_InvalidArgumentException("Invalid kernel type"));
    }
}

}  // namespace rise

#endif  // RS_KERNEL_H
