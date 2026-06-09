// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_H
#define RS_KERNEL_H

#include <mutex>
#include <unordered_map>

#include "rise_origin/device/data/rs_value.h"
#include "rise_origin/rsc/rs_floor_config.h"
#include "rise_origin/rsc/rs_simulation_config.h"
#include "rise_origin/rsc/rs_world_config.h"
#include "rise_origin/utils/rs_json.h"
#include "rise_origin/utils/rs_math_expr.h"
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
    RBigFloat time_step = 0;

    RBigFloat dt = 0;  // time_step * integration_dt_frac, actual time step

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

    // Synchronization mutex
    std::recursive_mutex sync_mutex;

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

    void updateTimeStepOnHost(RFloat time_step) const {
        host_data->runtime_states.time_step = time_step;
        host_data->runtime_states.dt = integration_dt_frac * time_step;
    }

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

    HOST_DEVICE_INLINE_QUALIFIER void updateTimeStepOnDevice(RFloat time_step) {
        runtime_states.get().time_step = time_step;
        runtime_states.get().dt = integration_dt_frac * time_step;
    }

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
        if (host_data->stop_condition[0].op == meEND) return false;
        auto variables = host_data->metrics;
        variables["t"] = host_data->runtime_states.time;
        return RS_MathExprExecutor::eval(variables, host_data->stop_condition) > 0;
    }

    [[nodiscard]] RBigFloat getHostTime() const { return host_data->runtime_states.time; }

    [[nodiscard]] RBigIndex getHostStep() const { return host_data->runtime_states.step; }

    [[nodiscard]] RBigFloat getHostDt() const { return host_data->runtime_states.dt; }

    [[nodiscard]] RBigFloat getHostTimeStep() const { return host_data->runtime_states.time_step; }

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

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER RBigFloat getDeviceTimeStep() const {
        return runtime_states.get().time_step;
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

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER bool isModified() const { return is_modified; }

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER bool isAutoTimeStep() const {
        return integration_time_step == 0;
    }

    [[nodiscard]] std::recursive_mutex& getSyncMutex() const { return host_data->sync_mutex; }

    void lockSyncMutex() const { host_data->sync_mutex.lock(); }

    [[nodiscard]] bool tryLockSyncMutex() const { return host_data->sync_mutex.try_lock(); }

    void unlockSyncMutex() const { host_data->sync_mutex.unlock(); }

    void markModified() { is_modified = true; }

    void clearModified() { is_modified = false; }

    template <typename NewContext>
    RS_KernelBase<NewContext>& castAsContextType() {
        return *reinterpret_cast<RS_KernelBase<NewContext>*>(this);
    }

    /**
     * Runtime data (constant data, set by host, kernel read only)
     */
    // Unique id assigned by user
    RIndex id = RS_NULL_INDEX;

    bool is_terminated = false;
    bool is_paused = false;
    bool is_modified = false;

    // Simulation config.
    uint64_t seed = 0x123456789abcdef;

    RFloat integration_time_step = 0;
    RFloat integration_dt_frac = 0;

    RFloat softbody_internal_damping_z = 0;
    RFloat softbody_contact_damping_z = 0;
    RFloat softbody_global_damping_z = 0;

    RSize rigidbody_constraint_iterations = 0;
    RFloat rigidbody_constraint_position_correction_ratio = 0;
    RFloat rigidbody_constraint_warm_start_damping_z = 1;
    RFloat rigidbody_constraint_compliance_initial = 1;
    RFloat rigidbody_constraint_compliance_min_threshold = 0;
    RFloat rigidbody_constraint_compliance_max_threshold = 1;
    RFloat rigidbody_constraint_compliance_auto_adjust_watt_threshold = 0;
    RFloat rigidbody_constraint_compliance_auto_adjust_step = 0;
    RSize rigidbody_contact_iterations = 0;
    RSize rigidbody_contact_position_correction_iterations = 0;
    RFloat rigidbody_contact_position_correction_ratio = 0;
    RFloat rigidbody_contact_warm_start_damping_z = 0;
    RFloat rigidbody_contact_restitution = 0;
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

    // World config.
    RFloat world_grav_acc = 0;
    RSE_WorldRenderType world_render_type = RSE_WORLD_RENDER_CONSTANT_COLOR;
    RColor3f world_color = RColor3f::zero();
    RColor3f world_horizon_color = RColor3f::zero();
    RVec3f world_sun_direction = RVec3f(0, 0, 1);
    RColor3f world_sun_color = RColor3f::zero();
    RFloat world_sun_angular_radius = 0;
    RFloat world_sky_intensity = 0;
    // Hosek-Wilkie cooked state: 3 channels x 9 coefficients per channel.
    RFloat world_hw_config[27] = {0};
    RFloat world_hw_radiance[3] = {0, 0, 0};

    // Floor config.
    RSE_FloorRenderType floor_render_type = RSE_FLOOR_RENDER_OPAQUE;
    RSE_FloorDataType floor_data_type = RSE_FLOOR_DATA_NONE;
    RColor3f floor_color = RColor3f::zero();
    RColor3f floor_line_color = RColor3f::zero();
    RFloat floor_grid_x_spacing = 1;
    RFloat floor_grid_y_spacing = 1;
    RFloat floor_grid_line_width = 0.05;
    RFloat floor_roughness = 1;
    RFloat floor_metalness = 0;
    RFloat floor_ior = 1.5;

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
