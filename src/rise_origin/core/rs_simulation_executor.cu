// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <device/data/rs_array.h>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <utility>

#include "rise_origin/device/rs_constant_memory.h"
#include "rise_origin/logic/mesh/rs_mesh_triangle_c.h"
#include "rise_origin/logic/metric/rs_basic_metric_c.h"
#include "rise_origin/logic/ray/rs_ray_sensor_c.h"
#include "rise_origin/logic/ray/rs_ray_sensor_ray_c.h"
#include "rise_origin/logic/record/rs_basic_record_c.h"
#include "rise_origin/logic/record/rs_camera_record_c.h"
#include "rise_origin/logic/record/rs_intersect_region_c.h"
#include "rise_origin/logic/record/rs_ray_sensor_record_c.h"
#include "rise_origin/logic/record/rs_structure_record_c.h"
#include "rise_origin/logic/rigid/rs_constraint_c.h"
#include "rise_origin/logic/rigid/rs_contact_c.h"
#include "rise_origin/logic/rigid/rs_rigid_body_c.h"
#include "rise_origin/logic/rigid/rs_rigid_body_surface_voxel_c.h"
#include "rise_origin/logic/signal/rs_force_c.h"
#include "rise_origin/logic/signal/rs_torque_c.h"
#include "rise_origin/logic/signal/rs_transform_c.h"
#include "rise_origin/logic/soft/rs_link_c.h"
#include "rise_origin/logic/soft/rs_voxel_c.h"
#include "rise_origin/logic/utils/rs_launch_c.h"
#include "rise_origin/logic/vision/rs_camera_c.h"
#include "rise_origin/logic/vision/rs_vision_denoise_c.h"
#include "rise_origin/logic/vision/rs_vision_ray_c.h"
#include "rise_origin/utils/meta/rs_constexpr.h"
#include "rise_origin/utils/rs_conf.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_iterators.h"
#include "rs_simulation.h"
#include "rs_simulation_executor.h"

using namespace std;
using namespace fmt;
using namespace rise::logic;
using namespace rise::device;

namespace rise {

/*****************************************************************************
 * Private compute functions
 *****************************************************************************/
class RSCF_UpdateKernelTime : public RS_ComputeFunction<RSCF_UpdateKernelTime> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run() {
        RS_ConstantMemory::as<RS_Kernel>()->updateTimeOnDevice();
    };
};

class RSCF_UpdateKernelDt : public RS_ComputeFunction<RSCF_UpdateKernelDt> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RFloat dt) {
        RS_ConstantMemory::as<RS_Kernel>()->updateDtOnDevice(dt);
    };
};

class RSCF_UpdateKernelOperationTime : public RS_ComputeFunction<RSCF_UpdateKernelOperationTime> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RSE_KernelUpdateOperation operation) {
        RS_ConstantMemory::as<RS_Kernel>()->updateOperationTimeOnDevice(operation);
    };
};

static auto ensure_vector_size = [](auto &vector, RSize size) {
    if (vector.capacity() < size) {
        vector.resize(size, false);
    }
    vector.occupy(size);
};

/*****************************************************************************
 * RS_SimulationExecutor implementation
 *****************************************************************************/
RS_SimulationExecutor::RS_SimulationExecutor(
    RS_MemoryProvider *device_provider,
    RS_MemoryProvider *host_provider,
    RS_WorldCallback *callback
)
    : _device_provider(device_provider), _host_provider(host_provider), _callback(callback) {
    if (spdlog::get("root")) {
        // Only init logger if root logger is created
        _logger = spdlog::get("root")->clone("simulation executor");
    }

    castProviderPtrAndCall(device_provider, [&](auto p_ptr) {
        _reduce = RS_Reduce(p_ptr);
        _reduce_by_group = RS_ReduceByGroup(p_ptr);
        _scan = RS_Scan(p_ptr);
        _sort = RS_RadixSortPairs(p_ptr);
    });
}

void RS_SimulationExecutor::init(RS_Simulation *simulation) {
    _simulation = simulation;
    _kernel_manager = simulation->kernel_manager.get();
    _kernel = &_kernel_manager->kernel;

    // Cache the static execution settings once, at init. Later user edits to the simulation
    // characterization do not change how this run executes. The per-subsystem run intervals
    // are not cached here; they are dynamic, recomputed by _runDtUpdate from the current
    // global time step.
    auto sim_rsc = _kernel_manager->getSimulationRSC();
    _time = sim_rsc.time;
    _dt_update_interval = sim_rsc.time.dt_update_interval;
    _contact_bvh_rebuild_sub_interval = sim_rsc.contact.bvh_rebuild_sub_interval;
    _contact_bvh_refit_sub_interval = sim_rsc.contact.bvh_refit_sub_interval;
    _sentinel_enabled = sim_rsc.sentinel.is_enabled;
    _simulation_compute_accuracy = sim_rsc.runtime.compute_accuracy;
}

void RS_SimulationExecutor::free() {
    _simulation = nullptr;
    _kernel = nullptr;
    _kernel_manager = nullptr;

    for (auto &handle_it : _handles) {
        handle_it.second.free();
    }
    _handles.clear();

    _reduce_input_buffer.free();
    _reduce_output_buffer.free();
    _reduce.free();
    _reduce_by_group.free();
    _scan.free();
    _sort.free();
}

bool RS_SimulationExecutor::step() {
    bool is_dt_normal = true;
    _runDtUpdate(is_dt_normal);
    while (!is_dt_normal) {
        bool is_hard_abort = false;
        RSize purged = _runSentinel(is_hard_abort);
        if (is_hard_abort || purged == 0) {
            _simulation->postMessage(
                RSE_WORLD_MESSAGE_ERROR, "world aborted: unrecoverable degenerate time step"
            );
            _kernel->terminate();
            return false;
        }
        _runDtUpdate(is_dt_normal);
    }

    if (_kernel->getTimeSinceLastExecutionOfEvent("print_time_and_dt") >= 0.1) {
        _kernel->executeEvent("print_time_and_dt");
        RS_LOG(_logger, info, "Time: {}s, dt: {}s.", _kernel->getHostTime(), _kernel->getHostDt());
    }

    if (_kernel->getHostStep() % _softbody_and_rigidbody_update_interval == 0) _runSoftbodyUpdate();
    if (_kernel->getHostStep() % _rigidbody_contact_and_constraint_update_interval == 0)
        _runRigidbodyContactAndConstraintUpdate();
    if (_kernel->getHostStep() % _softbody_and_rigidbody_update_interval == 0)
        _runRigidbodyCoupling();

    _runAdvanceTime();

    bool is_record_event_due =
        _kernel->getTimeSinceLastExecutionOfEvent("record") >= 1.0 / _kernel->record_frequency;
    bool is_callback_event_due =
        _kernel->getTimeSinceLastExecutionOfEvent("callback") >= 1.0 / _kernel->control_frequency;

    if (is_record_event_due || is_callback_event_due) {
        _runRecordUpdate();
    }
    if (is_record_event_due) {
        _kernel->executeEvent("record");
        _runRecorderRecordSaving();
    }
    if (is_callback_event_due) {
        _kernel->executeEvent("callback");
        _runControllerRecordSaving();
    }

    if (_sentinel_enabled && _kernel->getHostStep() % _sentinel_update_interval == 0) {
        bool is_hard_abort = false;
        _runSentinel(is_hard_abort);
        if (is_hard_abort) {
            _simulation->postMessage(
                RSE_WORLD_MESSAGE_ERROR, "world aborted: sentinel purge ceiling exceeded"
            );
            _kernel->terminate();
            return false;
        }
    }

    if (_kernel->getHostStep() % _contact_update_interval == 0) _runContactUpdate();
    if (_kernel->getHostStep() % _mechanical_signal_update_interval == 0)
        _runMechanicalSignalUpdate();
    if (is_callback_event_due) {
        _performCallback();
    }

    return true;
}

RFloat RS_SimulationExecutor::_runDtUpdate(bool &is_dt_normal) {
    RFloat dt = 0;
    is_dt_normal = true;
    _runWithCriticalDataAccessGuard([&](auto *k_ptr, auto *p_ptr, auto, bool was_modified) {
        using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;

        bool due = !_has_computed_dt || was_modified ||
                   (_dt_update_interval > 0 && k_ptr->getHostStep() % _dt_update_interval == 0);
        if (!due) {
            dt = RFloat(k_ptr->getHostDt());
            return;
        }

        const RFloat dt_ratio = _time.dt_ratio;

        // Each subsystem's needed dt: a specified field is used as-is; an empty (auto) field takes
        // its default scaled by dt_ratio.

        /****************************************************************
         * Softbody and rigidbody coupling dt
         ****************************************************************/
        // Auto = dt_ratio x the soft stable step, set by the largest natural frequency sqrt(k/m)
        // over links and voxels: dt = 1 / (integration_frequency_multiplier * max_frequency).
        RFloat freq_soft_dt = RS_NumTraits<RFloat>::highest();
        auto link_handles = _getLaunchHandles("link");
        auto dynamic_voxel_handles = _getLaunchHandles("dynamic_voxel");
        _resizeReduceBufferIfNeeded(
            RS_MAX(link_handles.size(), dynamic_voxel_handles.size()) * sizeof(RFloat), 0
        );
        if (link_handles.size() > 0) {
            p_ptr->template run<
                RSCF_LaunchSetOrGetWithHandles<RSCF_GetLinkFrequencyMetric, KernelType>>(
                {link_handles.size(), RS_AUTO_BLOCK_SIZE, false}, link_handles, _reduce_input_buffer
            );
            RFloat link_max_freq = _reduce.reduce<RFloat, RS_ReduceMaxOp>(
                _reduce_input_buffer, link_handles.size(), 0
            );
            if (link_max_freq > 0)
                freq_soft_dt = RS_MIN(
                    freq_soft_dt,
                    1 / (k_ptr->softbody_integration_frequency_multiplier * sqrt(link_max_freq))
                );
        }
        if (dynamic_voxel_handles.size() > 0) {
            p_ptr->template run<
                RSCF_LaunchSetOrGetWithHandles<RSCF_GetVoxelFrequencyMetric, KernelType>>(
                {dynamic_voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false},
                dynamic_voxel_handles,
                _reduce_input_buffer
            );
            RFloat voxel_max_freq = _reduce.reduce<RFloat, RS_ReduceMaxOp>(
                _reduce_input_buffer, dynamic_voxel_handles.size(), 0
            );
            if (voxel_max_freq > 0)
                freq_soft_dt = RS_MIN(
                    freq_soft_dt,
                    1 / (k_ptr->softbody_integration_frequency_multiplier * sqrt(voxel_max_freq))
                );
        }
        // With no soft bodies (freq still highest) fall back to dt_ratio x 0.005s.
        RFloat needed_softbody_and_rigidbody =
            !_time.dt_softbody_and_rigidbody_update.empty()
                ? RFloat(_time.dt_softbody_and_rigidbody_update.get())
                : (freq_soft_dt < RS_NumTraits<RFloat>::highest() ? dt_ratio * freq_soft_dt
                                                                  : dt_ratio * RFloat(0.005));

        /****************************************************************
         * Rigidbody contact and constraint dt
         ****************************************************************/
        RFloat needed_rigidbody_contact_and_constraint =
            !_time.dt_rigidbody_contact_and_constraint_update.empty()
                ? RFloat(_time.dt_rigidbody_contact_and_constraint_update.get())
                : dt_ratio * RFloat(0.005);

        /****************************************************************
         * Contact broadphase dt
         ****************************************************************/
        RFloat needed_contact = !_time.dt_contact_update.empty()
                                    ? RFloat(_time.dt_contact_update.get())
                                    : dt_ratio * RFloat(0.005);

        /****************************************************************
         * Mechanical signal dt
         ****************************************************************/
        RFloat needed_mechanical_signal = !_time.dt_mechanical_signal_update.empty()
                                              ? RFloat(_time.dt_mechanical_signal_update.get())
                                              : dt_ratio * RFloat(0.005);

        /****************************************************************
         * Sentinel dt
         ****************************************************************/
        RFloat needed_sentinel = !_time.dt_sentinel_update.empty()
                                     ? RFloat(_time.dt_sentinel_update.get())
                                     : dt_ratio * RFloat(0.1);

        /****************************************************************
         * Gather global dt
         ****************************************************************/
        if (!_time.dt.empty()) {
            dt = RFloat(_time.dt.get());
        } else {
            dt = std::min({
                needed_softbody_and_rigidbody,
                needed_rigidbody_contact_and_constraint,
                needed_contact,
                needed_mechanical_signal,
                needed_sentinel,
            });
        }

        // Degenerate dt: report not-normal and let step() drive the sentinel.
        if (!(dt >= RFloat(1e-7))) {
            is_dt_normal = false;
            dt = 0;
            return;
        }

        p_ptr->template run<RSCF_UpdateKernelDt>({1, 1, false}, dt);
        k_ptr->updateDtOnHost(dt);

        /****************************************************************
         * Snap each subsystem to a base-step interval
         ****************************************************************/
        _softbody_and_rigidbody_update_interval = _snapInterval(needed_softbody_and_rigidbody, dt);
        _rigidbody_contact_and_constraint_update_interval =
            _snapInterval(needed_rigidbody_contact_and_constraint, dt);
        _contact_update_interval = _snapInterval(needed_contact, dt);
        _mechanical_signal_update_interval = _snapInterval(needed_mechanical_signal, dt);
        _sentinel_update_interval = _snapInterval(needed_sentinel, dt);

        _has_computed_dt = true;

        /****************************************************************
         * Vailidity check
         ****************************************************************/
        // A manual time.dt coarser than a subsystem's stability dt or the control/record period is
        // honored faithfully but flagged. Throttled to avoid spam.
        if (k_ptr->getTimeSinceLastExecutionOfEvent("warn_coarse_dt") >= 1.0) {
            bool warned = false;
            auto warn_if_coarse = [&](const char *what, RFloat needed) {
                if (dt > needed) {
                    RS_LOG(
                        _logger,
                        warn,
                        "dt {}s exceeds task [{}] stability dt {}s.",
                        dt,
                        what,
                        needed
                    );
                    warned = true;
                }
            };
            warn_if_coarse("softbody_and_rigidbody", needed_softbody_and_rigidbody);
            warn_if_coarse(
                "rigidbody_contact_and_constraint", needed_rigidbody_contact_and_constraint
            );
            warn_if_coarse("contact", needed_contact);
            warn_if_coarse("mechanical_signal", needed_mechanical_signal);
            warn_if_coarse("sentinel", needed_sentinel);
            if (k_ptr->control_frequency > 0 && dt > 1.0 / k_ptr->control_frequency) {
                RS_LOG(
                    _logger,
                    warn,
                    "dt {}s exceeds control period {}s.",
                    dt,
                    1.0 / k_ptr->control_frequency
                );
                warned = true;
            }
            if (k_ptr->record_frequency > 0 && dt > 1.0 / k_ptr->record_frequency) {
                RS_LOG(
                    _logger,
                    warn,
                    "dt {}s exceeds record period {}s.",
                    dt,
                    1.0 / k_ptr->record_frequency
                );
                warned = true;
            }
            if (warned) k_ptr->executeEvent("warn_coarse_dt");
        }
    });
    return dt;
}

RSize RS_SimulationExecutor::_runSentinel(bool &is_hard_abort) {
    is_hard_abort = false;
    RSize purged = 0;

    // Structures flagged for purge, maps (structure gid -> reason string).
    std::unordered_map<RIndex, std::string> offenders;

    // The purge mutates manager maps, so take command execution rather than just data access.
    _runWithCommandExecutionGuard([&](auto *k_ptr, auto *p_ptr, auto, bool) {
        using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;
        auto &ctx = k_ptr->ctx;

        // Note: for reduction should use full voxel handles to correspond to layout group sizes
        auto voxel_handles = _getLaunchHandles("voxel");
        auto link_handles = _getLaunchHandles("link");

        // Inactive or empty structures contribute no launch handles, so the grouped reductions
        // below must run over the active, non-empty segments only.
        std::vector<RSize> voxel_active_sizes;
        std::vector<RIndex> voxel_active_ids;
        std::vector<RSize> link_active_sizes;
        std::vector<RIndex> link_active_ids;
        for (auto [size, gid, is_active] :
             zip(ctx.host_data->structure_voxel_layout.getSegmentSizes(),
                 ctx.host_data->structure_voxel_layout.getSegmentIds(),
                 ctx.host_data->structure_voxel_layout.getSegmentIsActive())) {
            if (is_active && size > 0) {
                voxel_active_sizes.push_back(size);
                voxel_active_ids.push_back(gid);
            }
        }
        for (auto [size, gid, is_active] :
             zip(ctx.host_data->structure_link_layout.getSegmentSizes(),
                 ctx.host_data->structure_link_layout.getSegmentIds(),
                 ctx.host_data->structure_link_layout.getSegmentIsActive())) {
            if (is_active && size > 0) {
                link_active_sizes.push_back(size);
                link_active_ids.push_back(gid);
            }
        }

        /****************************************************************
         * Voxel NaN check
         ****************************************************************/
        if (voxel_handles.size() > 0) {
            _resizeReduceBufferIfNeeded(voxel_handles.size() * sizeof(RBound3f), 0);
            p_ptr->template run<
                RSCF_LaunchSetOrGetWithHandles<RSCF_GetIsVoxelNanMetric, KernelType>>(
                {voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false},
                voxel_handles,
                _reduce_input_buffer
            );
            auto result = _reduce_by_group.reduce<bool, RS_ReduceOrOp>(
                _reduce_input_buffer, voxel_active_sizes, false
            );
            for (RSize seg = 0; seg < result.size(); ++seg) {
                if (voxel_active_sizes[seg] > 0 && result[seg])
                    offenders.emplace(voxel_active_ids[seg], "nan voxel");
            }
        }

        /****************************************************************
         * Voxel frequency check
         ****************************************************************/
        if (voxel_handles.size() > 0) {
            _resizeReduceBufferIfNeeded(voxel_handles.size() * sizeof(RBound3f), 0);
            p_ptr->template run<
                RSCF_LaunchSetOrGetWithHandles<RSCF_GetVoxelFrequencyMetric, KernelType>>(
                {voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false},
                voxel_handles,
                _reduce_input_buffer
            );
            auto result = _reduce_by_group.reduce<RFloat, RS_ReduceMaxOp>(
                _reduce_input_buffer, voxel_active_sizes, RFloat(0)
            );
            for (RSize seg = 0; seg < result.size(); ++seg) {
                RFloat f = result[seg];
                RFloat dt = 1 / (k_ptr->softbody_integration_frequency_multiplier * sqrt(f));
                if (voxel_active_sizes[seg] > 0 && (!std::isfinite(f) || dt < RFloat(1e-7)))
                    offenders.emplace(
                        voxel_active_ids[seg],
                        format("degenerate voxel frequency {} (dt {}s)", f, dt)
                    );
            }
        }

        /****************************************************************
         * Link frequency check
         ****************************************************************/
        if (link_handles.size() > 0) {
            _resizeReduceBufferIfNeeded(link_handles.size() * sizeof(RFloat), 0);
            p_ptr->template run<
                RSCF_LaunchSetOrGetWithHandles<RSCF_GetLinkFrequencyMetric, KernelType>>(
                {link_handles.size(), RS_AUTO_BLOCK_SIZE, false}, link_handles, _reduce_input_buffer
            );
            auto result = _reduce_by_group.reduce<RFloat, RS_ReduceMaxOp>(
                _reduce_input_buffer, link_active_sizes, RFloat(0)
            );
            for (RSize seg = 0; seg < result.size(); ++seg) {
                RFloat f = result[seg];
                RFloat dt = 1 / (k_ptr->softbody_integration_frequency_multiplier * sqrt(f));
                if (link_active_sizes[seg] > 0 && (!std::isfinite(f) || dt < RFloat(1e-7)))
                    offenders.emplace(
                        link_active_ids[seg], format("degenerate link frequency {} (dt {}s)", f, dt)
                    );
            }
        }

        /****************************************************************
         * Voxel bound check
         ****************************************************************/
        if (voxel_handles.size() > 0) {
            const RFloat range = RFloat(RS_WORLD_MAX_RANGE);
            RBound3f world_box(-range, -range, -range, range, range, range);
            _resizeReduceBufferIfNeeded(voxel_handles.size() * sizeof(RBound3f), 0);
            p_ptr->template run<
                RSCF_LaunchSetOrGetWithHandles<RSCF_GetVoxelBoundMetric, KernelType>>(
                {voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false},
                voxel_handles,
                _reduce_input_buffer
            );
            auto result = _reduce_by_group.reduce<RBound3f, RS_ReduceBoundUnionOp>(
                _reduce_input_buffer, voxel_active_sizes, RBound3f::lowest()
            );
            for (RSize seg = 0; seg < result.size(); ++seg) {
                RBound3f bound = result[seg];
                if (voxel_active_sizes[seg] > 0 && bound.isValid() &&
                    !world_box.isContaining(bound))
                    offenders.emplace(
                        voxel_active_ids[seg],
                        format(
                            "out of world bounds min ({}, {}, {}) max ({}, {}, {})",
                            bound.min_x,
                            bound.min_y,
                            bound.min_z,
                            bound.max_x,
                            bound.max_y,
                            bound.max_z
                        )
                    );
            }
        }

        /****************************************************************
         * Purge
         ****************************************************************/
        for (auto &[gid, reason] : offenders) {
            RS_LOG(
                _logger,
                warn,
                "Sentinel found invalid structure with gid: {}, reason: {}",
                gid,
                reason
            );
            std::string name = _kernel_manager->purgeStructure(gid);
            _simulation->postMessage(
                RSE_WORLD_MESSAGE_WARNING, format("structure {} purged: {}", name, reason)
            );
            if (++purged > 100) {
                is_hard_abort = true;
                break;
            }
        }

        _rebuildStaticVoxelBVHIfNeeded();
        _rebuildDynamicVoxelBVH();
    });
    return purged;
}

void RS_SimulationExecutor::_runAdvanceTime() {
    _runWithCriticalDataAccessGuard([&](auto *k_ptr, auto *p_ptr, auto, bool) {
        p_ptr->template run<RSCF_UpdateKernelTime>({1, 1, false});
        k_ptr->updateTimeOnHost();
    });
}

void RS_SimulationExecutor::_runSoftbodyUpdate() {
    _runWithCriticalDataAccessGuard([&](auto *k_ptr, auto *p_ptr, auto acc_const, bool) {
        static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
        using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;

        // Stamp the soft-body operation time so soft kernels read the true soft_dt
        // (>= the base dt when the user runs soft body every N>1 base steps).
        p_ptr->template run<RSCF_UpdateKernelOperationTime>(
            {1, 1, false}, RSE_KERNEL_UPDATE_OPERATION_SOFT_BODY
        );
        k_ptr->updateOperationTimeOnHost(RSE_KERNEL_UPDATE_OPERATION_SOFT_BODY);

        auto link_handles = _getLaunchHandles("link");
        p_ptr->template run<RSCF_LaunchWithHandles<
            typename RS_LinkCompute<ComputeAccuracy>::RSCF_UpdateBernoulliLink,
            KernelType>>({link_handles.size(), RS_AUTO_BLOCK_SIZE, false}, link_handles);

        auto dynamic_voxel_handles = _getLaunchHandles("dynamic_voxel");
        p_ptr->template run<RSCF_LaunchWithHandles<
            typename RS_VoxelCompute<ComputeAccuracy>::RSCF_UpdateVoxelContact,
            KernelType>>(
            {dynamic_voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false}, dynamic_voxel_handles
        );
        p_ptr->template run<RSCF_LaunchWithHandles<
            typename RS_VoxelCompute<ComputeAccuracy>::RSCF_UpdateVoxel,
            KernelType>>(
            {dynamic_voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false}, dynamic_voxel_handles
        );
    });
}

void RS_SimulationExecutor::_runRigidbodyCoupling() {
    _runWithCriticalDataAccessGuard([&](auto *k_ptr, auto *p_ptr, auto acc_const, bool) {
        static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
        using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;
        auto &ctx = k_ptr->ctx;

        // Stamp the rigid-body operation time so the rigid kernels read the true rigid_dt.
        p_ptr->template run<RSCF_UpdateKernelOperationTime>(
            {1, 1, false}, RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_INTEGRATE
        );
        k_ptr->updateOperationTimeOnHost(RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_INTEGRATE);

        if (ctx.structure_rigid_body_layout.size() == 0) return;
        auto rigid_body_surface_voxel_handles = _getLaunchHandles("rigid_body_surface_voxel");

        // Gather each surface voxel's soft coupling correction and reduce onto its body.
        const std::vector<RSize> reduction_group_sizes =
            ctx.host_data->structure_rigid_body_surface_voxel_to_rigid_body_reduction_layout
                .getInputSegmentSizes();
        _resizeReduceBufferIfNeeded(
            ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_input_layout.size() *
                sizeof(RS_CouplingCorrection),
            reduction_group_sizes.size() * sizeof(RS_CouplingCorrection)
        );
        p_ptr->template run<RSCF_LaunchSetOrGetWithLayout<
            typename RS_RigidBodySurfaceVoxelCompute<
                ComputeAccuracy>::RSCF_GetRigidBodyCouplingCorrectionFromVoxel,
            KernelType>>(
            {ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_input_layout.size(),
             RS_AUTO_BLOCK_SIZE,
             false},
            ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_input_layout,
            _reduce_input_buffer
        );
        _reduce_by_group.reduce<RS_CouplingCorrection, RS_ReduceCouplingCorrectionOp>(
            _reduce_input_buffer,
            _reduce_output_buffer,
            reduction_group_sizes,
            RS_CouplingCorrection::zero()
        );
        p_ptr->template run<RSCF_LaunchSetOrGetWithLayout<
            typename RS_RigidBodyCompute<
                ComputeAccuracy>::RSCF_ApplyCouplingCorrectionAndUpdateRigidBody,
            KernelType>>(
            {ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_output_layout.size(),
             RS_AUTO_BLOCK_SIZE,
             false},
            ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_output_layout,
            _reduce_output_buffer
        );

        // Push integrated pose to surface voxels so next soft step sees moving anchors.
        p_ptr->template run<RSCF_LaunchWithHandles<
            typename RS_RigidBodySurfaceVoxelCompute<
                ComputeAccuracy>::RSCF_UpdateVoxelFromRigidBody,
            KernelType>>(
            {rigid_body_surface_voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false},
            rigid_body_surface_voxel_handles
        );
    });
}

void RS_SimulationExecutor::_runRigidbodyContactAndConstraintUpdate() {
    _runWithCriticalDataAccessGuard([&](auto *k_ptr, auto *p_ptr, auto acc_const, bool) {
        static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
        using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;
        auto &ctx = k_ptr->ctx;

        // Stamp the rigid-body slow constraint and contact solve time.
        p_ptr->template run<RSCF_UpdateKernelOperationTime>(
            {1, 1, false}, RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_CONSTRAINT_AND_CONTACT
        );
        k_ptr->updateOperationTimeOnHost(
            RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_CONSTRAINT_AND_CONTACT
        );

        if (ctx.structure_rigid_body_layout.size() == 0) return;
        auto rigid_body_handles = _getLaunchHandles("rigid_body");
        auto rigid_body_surface_voxel_handles = _getLaunchHandles("rigid_body_surface_voxel");
        auto dynamic_rigid_body_handles = _getLaunchHandles("dynamic_rigid_body");

        // Apply gravity impulse before any solver so the solver sees gravity-loaded lm.
        p_ptr->template run<RSCF_LaunchWithHandles<
            typename RS_RigidBodyCompute<ComputeAccuracy>::RSCF_ApplyGravity,
            KernelType>>(
            {dynamic_rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false},
            dynamic_rigid_body_handles
        );

        // Constraint joints solve, using block-shared PGS, one block per structure.
        if (ctx.structure_constraint_joint_layout.size() > 0) {
            auto constraint_joint_handles = _getLaunchHandles("constraint_joint");
            p_ptr->template run<RSCF_LaunchWithHandles<
                typename RS_ConstraintCompute<ComputeAccuracy>::RSCF_UpdateConstraintJoint,
                KernelType>>(
                {constraint_joint_handles.size(), RS_AUTO_BLOCK_SIZE, false},
                constraint_joint_handles
            );

            using SharedRBState = typename RS_ConstraintCompute<
                ComputeAccuracy>::RSCF_PGSUpdateJoint::SharedRigidBodyState;
            RSize num_blocks = ctx.host_data->structure_constraint_joint_block_range_layout.size();
            RSize max_cj_per_structure = k_ptr->host_data->max_constraint_joint_per_structure;
            RSize max_rb_per_structure = k_ptr->host_data->max_rigid_body_per_structure;
            RSize block_size = RS_ALIGN(max_cj_per_structure, 32);
            RSize total_threads = block_size * num_blocks;
            RBigSize shared_bytes = RBIG_SIZE(max_rb_per_structure) * sizeof(SharedRBState);

            p_ptr->template run<RSCF_LaunchDirect<
                typename RS_ConstraintCompute<ComputeAccuracy>::RSCF_PGSUpdateJoint,
                KernelType>>(
                {total_threads, block_size, true, shared_bytes},
                ctx.structure_constraint_joint_block_ranges,
                k_ptr->host_data->constraint_joint_max_colors
            );

            // Zero the per-body coupling-work accumulator just consumed by stiffness adaptation.
            p_ptr->template run<RSCF_LaunchWithHandles<
                typename RS_RigidBodyCompute<ComputeAccuracy>::RSCF_ResetRigidBodyCouplingWork,
                KernelType>>(
                {dynamic_rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false},
                dynamic_rigid_body_handles
            );
        }

        // Manifold-reduced flat rigid contact.
        const RSize rbsv_num = rigid_body_surface_voxel_handles.size();
        if (rbsv_num > 0) {
            using CC = RS_ContactCompute<ComputeAccuracy>;
            const RSize slot_num = rbsv_num * RSize(3);
            ensure_vector_size(ctx.structure_rigid_contact_flags, slot_num);
            ensure_vector_size(ctx.structure_rigid_contact_prefix_sums, slot_num);

            // Count live contact slots, prefix sum, then emit compacted into the flat list.
            p_ptr->template run<
                RSCF_LaunchSetOrGetWithHandles<typename CC::RSCF_CountRigidContacts, KernelType>>(
                {rbsv_num, RS_AUTO_BLOCK_SIZE, false},
                rigid_body_surface_voxel_handles,
                ctx.structure_rigid_contact_flags.data
            );
            _scan.template inclusiveSum<RIndex, RIndex>(
                ctx.structure_rigid_contact_flags, ctx.structure_rigid_contact_prefix_sums
            );
            RIndex total_valid = 0;
            ctx.structure_rigid_contact_prefix_sums.readSlice(&total_valid, 1, slot_num - 1, true);

            if (total_valid > 0) {
                ensure_vector_size(ctx.structure_rigid_contacts, total_valid);
                ensure_vector_size(ctx.structure_rigid_sorted_contacts, total_valid);
                ensure_vector_size(ctx.structure_rigid_contact_keys, total_valid);
                ensure_vector_size(ctx.structure_rigid_sorted_contact_keys, total_valid);
                ensure_vector_size(ctx.structure_rigid_contact_indices, total_valid);
                ensure_vector_size(ctx.structure_rigid_sorted_contact_indices, total_valid);

                p_ptr->template run<RSCF_LaunchSetOrGetWithHandles<
                    typename CC::RSCF_EmitRigidContacts,
                    KernelType>>(
                    {rbsv_num, RS_AUTO_BLOCK_SIZE, false},
                    rigid_body_surface_voxel_handles,
                    ctx.structure_rigid_contact_flags.data,
                    ctx.structure_rigid_contacts,
                    ctx.structure_rigid_contact_prefix_sums.data
                );

                // Drop reciprocal emissions: sort by canonical voxel-pair key, keep first of run.
                p_ptr->template run<
                    RSCF_LaunchDirect<typename CC::RSCF_GetRigidContactVoxelPairKeys, KernelType>>(
                    {total_valid, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_rigid_contacts,
                    ctx.structure_rigid_contact_keys.data,
                    ctx.structure_rigid_contact_indices.data
                );
                _sort.template sort<logic::RS_RigidContactKey, RIndex>(
                    ctx.structure_rigid_contact_keys,
                    ctx.structure_rigid_contact_indices,
                    ctx.structure_rigid_sorted_contact_keys,
                    ctx.structure_rigid_sorted_contact_indices
                );
                p_ptr->template run<
                    RSCF_LaunchDirect<typename CC::RSCF_ReorderRigidContacts, KernelType>>(
                    {total_valid, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_rigid_contacts,
                    ctx.structure_rigid_sorted_contacts,
                    ctx.structure_rigid_sorted_contact_indices.data
                );
                p_ptr->template run<
                    RSCF_LaunchDirect<typename CC::RSCF_FlagRigidContactKeyRunStart, KernelType>>(
                    {total_valid, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_rigid_sorted_contact_keys.data,
                    ctx.structure_rigid_contact_flags.data,
                    RIndex(total_valid)
                );
                _scan.template inclusiveSum<RIndex, RIndex>(
                    ctx.structure_rigid_contact_flags, ctx.structure_rigid_contact_prefix_sums
                );
                RIndex total_unique = 0;
                ctx.structure_rigid_contact_prefix_sums.readSlice(
                    &total_unique, 1, total_valid - 1, true
                );
                p_ptr->template run<
                    RSCF_LaunchDirect<typename CC::RSCF_CompactSelectedRigidContacts, KernelType>>(
                    {total_valid, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_rigid_sorted_contacts,
                    ctx.structure_rigid_contacts,
                    ctx.structure_rigid_contact_flags.data,
                    ctx.structure_rigid_contact_prefix_sums.data
                );
                total_valid = total_unique;

                // Sort by owner-pair key and reduce to a manifold per body pair.
                p_ptr->template run<
                    RSCF_LaunchDirect<typename CC::RSCF_GetRigidContactKeys, KernelType>>(
                    {total_valid, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_rigid_contacts,
                    ctx.structure_rigid_contact_keys.data,
                    ctx.structure_rigid_contact_indices.data
                );
                _sort.template sort<logic::RS_RigidContactKey, RIndex>(
                    ctx.structure_rigid_contact_keys,
                    ctx.structure_rigid_contact_indices,
                    ctx.structure_rigid_sorted_contact_keys,
                    ctx.structure_rigid_sorted_contact_indices
                );
                p_ptr->template run<
                    RSCF_LaunchDirect<typename CC::RSCF_ReorderRigidContacts, KernelType>>(
                    {total_valid, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_rigid_contacts,
                    ctx.structure_rigid_sorted_contacts,
                    ctx.structure_rigid_sorted_contact_indices.data
                );
                // Group boundaries: flag owner-pair run starts, scan to group ids, compact starts.
                p_ptr->template run<
                    RSCF_LaunchDirect<typename CC::RSCF_FlagRigidContactKeyRunStart, KernelType>>(
                    {total_valid, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_rigid_sorted_contact_keys.data,
                    ctx.structure_rigid_contact_flags.data,
                    RIndex(total_valid)
                );
                _scan.template inclusiveSum<RIndex, RIndex>(
                    ctx.structure_rigid_contact_flags, ctx.structure_rigid_contact_prefix_sums
                );
                RIndex group_num = 0;
                ctx.structure_rigid_contact_prefix_sums.readSlice(
                    &group_num, 1, total_valid - 1, true
                );
                p_ptr->template run<RSCF_LaunchDirect<
                    typename CC::RSCF_CompactRigidManifoldGroupStart,
                    KernelType>>(
                    {total_valid, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_rigid_contact_flags.data,
                    ctx.structure_rigid_contact_prefix_sums.data,
                    ctx.structure_rigid_contact_indices.data
                );

                // Read group start offsets to host, derive per-group lengths for reduce-by-group.
                std::vector<RIndex> group_starts(group_num);
                ctx.structure_rigid_contact_indices.readSlice(
                    group_starts.data(), group_num, 0, true
                );
                std::vector<RSize> group_len(group_num);
                for (RIndex g = 0; g < group_num; ++g) {
                    RIndex end = (g + 1 < group_num) ? group_starts[g + 1] : total_valid;
                    group_len[g] = end - group_starts[g];
                }

                ensure_vector_size(ctx.structure_rigid_manifold_deepest_in, total_valid);
                ensure_vector_size(ctx.structure_rigid_manifold_deepest_out, group_num);
                ensure_vector_size(ctx.structure_rigid_manifold_farthest_in, total_valid);
                ensure_vector_size(ctx.structure_rigid_manifold_farthest_out, group_num);

                // Deepest contact per group: the manifold anchor.
                p_ptr->template run<
                    RSCF_LaunchDirect<typename CC::RSCF_GetRigidManifoldDepth, KernelType>>(
                    {total_valid, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_rigid_sorted_contacts,
                    ctx.structure_rigid_manifold_deepest_in.data
                );
                _reduce_by_group
                    .reduce<logic::RS_RigidManifoldDeepest, logic::RS_ReduceRigidManifoldDeepestOp>(
                        ctx.structure_rigid_manifold_deepest_in.data,
                        ctx.structure_rigid_manifold_deepest_out.data,
                        group_len,
                        logic::RS_RigidManifoldDeepest::empty()
                    );
                ctx.structure_rigid_contact_flags.fillSlice(RIndex(0), total_valid, 0, false);
                p_ptr->template run<
                    RSCF_LaunchDirect<typename CC::RSCF_SetRigidManifoldDeepest, KernelType>>(
                    {group_num, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_rigid_manifold_deepest_out.data,
                    ctx.structure_rigid_contact_flags.data
                );

                // Top-three contacts farthest from the anchor per group.
                p_ptr->template run<
                    RSCF_LaunchDirect<typename CC::RSCF_GetRigidManifoldDistance, KernelType>>(
                    {total_valid, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_rigid_sorted_contacts,
                    ctx.structure_rigid_contact_prefix_sums.data,
                    ctx.structure_rigid_manifold_deepest_out.data,
                    ctx.structure_rigid_manifold_farthest_in.data
                );
                _reduce_by_group.reduce<
                    logic::RS_RigidManifoldFarthest,
                    logic::RS_ReduceRigidManifoldFarthestOp>(
                    ctx.structure_rigid_manifold_farthest_in.data,
                    ctx.structure_rigid_manifold_farthest_out.data,
                    group_len,
                    logic::RS_RigidManifoldFarthest::empty()
                );
                p_ptr->template run<
                    RSCF_LaunchDirect<typename CC::RSCF_SetRigidManifoldFarthest, KernelType>>(
                    {group_num, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_rigid_manifold_farthest_out.data,
                    ctx.structure_rigid_manifold_deepest_out.data,
                    ctx.structure_rigid_contact_flags.data
                );

                _scan.template inclusiveSum<RIndex, RIndex>(
                    ctx.structure_rigid_contact_flags, ctx.structure_rigid_contact_prefix_sums
                );
                RIndex total_selected = 0;
                ctx.structure_rigid_contact_prefix_sums.readSlice(
                    &total_selected, 1, total_valid - 1, true
                );

                if (total_selected > 0) {
                    p_ptr->template run<RSCF_LaunchDirect<
                        typename CC::RSCF_CompactSelectedRigidContacts,
                        KernelType>>(
                        {total_valid, RS_AUTO_BLOCK_SIZE, false},
                        ctx.structure_rigid_sorted_contacts,
                        ctx.structure_rigid_contacts,
                        ctx.structure_rigid_contact_flags.data,
                        ctx.structure_rigid_contact_prefix_sums.data
                    );

                    // Per-body contact count, the normalizer input.
                    p_ptr->template run<RSCF_LaunchWithHandles<
                        typename CC::RSCF_ResetRigidBodyContactCount,
                        KernelType>>(
                        {rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false}, rigid_body_handles
                    );
                    p_ptr->template run<RSCF_LaunchDirect<
                        typename CC::RSCF_AccumulateRigidContactCount,
                        KernelType>>(
                        {total_selected, RS_AUTO_BLOCK_SIZE, false}, ctx.structure_rigid_contacts
                    );

                    // Build constraint data, seed warm-start lambda, and apply it.
                    p_ptr->template run<
                        RSCF_LaunchDirect<typename CC::RSCF_JacobiBuildRigidContact, KernelType>>(
                        {total_selected, RS_AUTO_BLOCK_SIZE, false}, ctx.structure_rigid_contacts
                    );

                    // Velocity-solve iterations.
                    const RSize contact_iterations = k_ptr->rigidbody_contact_iterations;
                    for (RSize contact_it = 0; contact_it < contact_iterations; ++contact_it) {
                        p_ptr->template run<RSCF_LaunchDirect<
                            typename CC::RSCF_JacobiUpdateRigidContact,
                            KernelType>>(
                            {total_selected, RS_AUTO_BLOCK_SIZE, false},
                            ctx.structure_rigid_contacts
                        );
                    }

                    // Persist warm-start lambda back on the source voxels.
                    p_ptr->template run<RSCF_LaunchDirect<
                        typename CC::RSCF_WritebackRigidContactLambda,
                        KernelType>>(
                        {total_selected, RS_AUTO_BLOCK_SIZE, false}, ctx.structure_rigid_contacts
                    );

                    // Split-impulse position correction: snapshot com, then iterate, applying
                    // directly to com and orientation. Velocity is not touched.
                    p_ptr->template run<RSCF_LaunchWithHandles<
                        typename CC::RSCF_SnapshotPositionCorrectionCom,
                        KernelType>>(
                        {rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false}, rigid_body_handles
                    );
                    const RSize position_correction_iterations =
                        k_ptr->rigidbody_contact_position_correction_iterations;
                    for (RSize contact_it = 0; contact_it < position_correction_iterations;
                         ++contact_it) {
                        p_ptr->template run<RSCF_LaunchDirect<
                            typename CC::RSCF_SolveRigidContactPositionCorrection,
                            KernelType>>(
                            {total_selected, RS_AUTO_BLOCK_SIZE, false},
                            ctx.structure_rigid_contacts
                        );
                    }
                }
            }
        }

        // Damp momentum at the slow-step rate after all solvers.
        // Note: Pose integration and surface-voxel sync are handled by the
        // _runRigidbodyCoupling which fires every base step.
        p_ptr->template run<RSCF_LaunchWithHandles<
            typename RS_RigidBodyCompute<ComputeAccuracy>::RSCF_ApplyDamping,
            KernelType>>(
            {dynamic_rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false},
            dynamic_rigid_body_handles
        );
    });
}

void RS_SimulationExecutor::_runRecordUpdate() {
    _runWithCriticalDataAccessGuard([&](auto *, auto *, auto, bool) {
        _updateVisionAndRay();
        _updateIntersectRegionRecord();
    });
}

void RS_SimulationExecutor::_runRecorderRecordSaving() {
    _runWithCriticalDataAccessGuard([&](auto *, auto *, auto, bool) { _storeRecorderRecords(); });
}

void RS_SimulationExecutor::_runControllerRecordSaving() {
    _runWithCriticalDataAccessGuard([&](auto *, auto *, auto, bool) { _storeControllerRecords(); });
}

void RS_SimulationExecutor::_runContactUpdate() {
    RSize sub_step = _kernel->getHostStep() / _contact_update_interval;
    bool is_rebuild_needed = (sub_step % _contact_bvh_rebuild_sub_interval == 0);
    bool is_refit_needed = !is_rebuild_needed && (sub_step % _contact_bvh_refit_sub_interval == 0);
    bool is_discovery_needed = is_rebuild_needed || is_refit_needed;

    _runWithCriticalDataAccessGuard([&](auto *k_ptr, auto *p_ptr, auto acc_const, bool) {
        static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
        using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;

        _rebuildStaticVoxelBVHIfNeeded();
        _rebuildFloorTriangleBVHIfNeeded();
        if (is_rebuild_needed) {
            _rebuildDynamicVoxelBVH();
        } else if (is_refit_needed) {
            _refitDynamicVoxelBVH();
        }

        if (!is_discovery_needed) return;

        auto &ctx = k_ptr->ctx;
        RSize leaf_count = ctx.dynamic_voxel_bvh.object_identifiers.size();
        if (leaf_count == 0) return;

        p_ptr->template run<RSCF_LaunchDirect<
            typename RS_VoxelCompute<ComputeAccuracy>::RSCF_FindContactVoxels,
            KernelType>>(
            {leaf_count, RS_AUTO_BLOCK_SIZE, false}, ctx.dynamic_voxel_bvh.object_identifiers
        );
    });
}

void RS_SimulationExecutor::_runMechanicalSignalUpdate() {
    _runWithCriticalDataAccessGuard([&](auto *k_ptr, auto *p_ptr, auto acc_const, bool) {
        static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
        using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;
        auto &ctx = k_ptr->ctx;

        // Force signals: rigid and soft.
        if (ctx.force_rigid.size() > 0) {
            auto dynamic_rigid_body_handles = _getLaunchHandles("dynamic_rigid_body");
            p_ptr->template run<RSCF_LaunchWithHandles<
                typename RS_ForceCompute<ComputeAccuracy>::RSCF_ApplyRigidForceSignal,
                KernelType>>(
                {dynamic_rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false},
                dynamic_rigid_body_handles
            );

            p_ptr->template run<RSCF_LaunchDirect<
                typename RS_ForceCompute<ComputeAccuracy>::RSCF_UpdateRigidForceTime,
                KernelType>>({ctx.force_rigid.size(), RS_AUTO_BLOCK_SIZE, false});
        }
        if (ctx.force_soft.size() > 0) {
            auto dynamic_voxel_handles = _getLaunchHandles("dynamic_voxel");
            p_ptr->template run<RSCF_LaunchWithHandles<
                typename RS_ForceCompute<ComputeAccuracy>::RSCF_ApplySoftForceSignal,
                KernelType>>(
                {dynamic_voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false}, dynamic_voxel_handles
            );

            p_ptr->template run<RSCF_LaunchDirect<
                typename RS_ForceCompute<ComputeAccuracy>::RSCF_UpdateSoftForceTime,
                KernelType>>({ctx.force_soft.size(), RS_AUTO_BLOCK_SIZE, false});
        }

        // Torque signals: rigid only.
        if (ctx.torque_rigid.size() > 0) {
            auto dynamic_rigid_body_handles = _getLaunchHandles("dynamic_rigid_body");
            p_ptr->template run<RSCF_LaunchWithHandles<
                typename RS_TorqueCompute<ComputeAccuracy>::RSCF_ApplyRigidTorqueSignal,
                KernelType>>(
                {dynamic_rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false},
                dynamic_rigid_body_handles
            );

            p_ptr->template run<RSCF_LaunchDirect<
                typename RS_TorqueCompute<ComputeAccuracy>::RSCF_UpdateRigidTorqueTime,
                KernelType>>({ctx.torque_rigid.size(), RS_AUTO_BLOCK_SIZE, false});
        }

        // Transform signals: whole structure, applied once then reset to identity.
        if (ctx.transform_mem.size() > 0) {
            auto voxel_handles = _getLaunchHandles("voxel");
            p_ptr->template run<RSCF_LaunchWithHandles<
                typename RS_TransformCompute<ComputeAccuracy>::RSCF_ApplyVoxelTransformSignal,
                KernelType>>({voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false}, voxel_handles);

            auto rigid_body_handles = _getLaunchHandles("rigid_body");
            p_ptr->template run<RSCF_LaunchWithHandles<
                typename RS_TransformCompute<ComputeAccuracy>::RSCF_ApplyRigidBodyTransformSignal,
                KernelType>>(
                {rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false}, rigid_body_handles
            );

            p_ptr->template run<RSCF_LaunchDirect<
                typename RS_TransformCompute<ComputeAccuracy>::RSCF_ResetTransformSignal,
                KernelType>>({ctx.transform_mem.size(), RS_AUTO_BLOCK_SIZE, false});

            // The transforms are applied, so flag a static rebuild for any that moved static
            // voxels and clear each transform host signal.
            _kernel_manager->finalizeTransformSignals();
        }

        // Rebuild the static voxel BVH now if a transform moved static voxels; a no-op otherwise.
        _rebuildStaticVoxelBVHIfNeeded();
    });
}

void RS_SimulationExecutor::_performCallback() {
    if (!_callback) return;
    _runWithCommandExecutionGuard([&](auto *k_ptr, auto *, auto, bool) {
        // Complete in-flight operations so the timer measures the copy.
        _device_provider->sync();

        RS_Timer timer;
        _kernel_manager->syncControllerRecords();

        RS_LOG(
            _logger, debug, "Callback sync controller record takes: {}.", timer.elapsedTimeString()
        );
        timer.reset();

        auto names = _simulation->getControllerNames();
        auto by_name = _kernel_manager->getControllerSignalsAndRecords(names);

        vector<RS_WorldController> controllers;
        controllers.reserve(by_name.size());
        for (auto &[name, bundle] : by_name) {
            RS_WorldController controller;
            controller.name = name;
            controller.records = std::move(bundle.records);
            controller.signals = std::move(bundle.signals);
            controllers.emplace_back(std::move(controller));
        }

        if (_callback) (*_callback)(k_ptr->getHostTime(), controllers);

        RS_LOG(_logger, debug, "Callback call takes: {}.", timer.elapsedTimeString());
        timer.reset();

        _kernel_manager->syncControllerSignals();

        RS_LOG(
            _logger, debug, "Callback sync controller signal takes: {}.", timer.elapsedTimeString()
        );
        timer.reset();
    });
}

void RS_SimulationExecutor::_updateVisionAndRay() {
    _rebuildStaticVoxelBVHIfNeeded();
    _rebuildFloorTriangleBVHIfNeeded();
    _rebuildDynamicVoxelBVH();

    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            auto &ctx = k_ptr->ctx;
            using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;

            // SVGF needs last frame's aux for the disocclusion gate: ping-pong it
            // here so the info trace refills camera_aux_* while camera_prev_aux_*
            // holds the previous snapshot. Swapped before the sync below so the
            // pointers reach the device this frame.
            if (k_ptr->vision_denoise_type == RSE_VISION_DENOISE_SVGF &&
                ctx.camera_info_ray_layout.size() > 0) {
                exchange(ctx.camera_aux_normal, ctx.camera_prev_aux_normal);
                exchange(ctx.camera_aux_depth_z, ctx.camera_prev_aux_depth_z);
                exchange(ctx.camera_aux_object_key, ctx.camera_prev_aux_object_key);
            }

            p_ptr->template run<
                RSCF_LaunchWithLayout<RS_CameraCompute<>::RSCF_UpdateCamera, KernelType>>(
                {ctx.camera_layout.size(), RS_AUTO_BLOCK_SIZE, false}, ctx.camera_layout
            );

            p_ptr->template run<
                RSCF_LaunchWithLayout<RS_RaySensorCompute<>::RSCF_UpdateRaySensor, KernelType>>(
                {ctx.ray_sensor_layout.size(), RS_AUTO_BLOCK_SIZE, false}, ctx.ray_sensor_layout
            );

            p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                RS_CameraCompute<>::RSCF_ClearCameraCanvas,
                KernelType>>(
                {ctx.camera_rgb_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                ctx.camera_rgb_record_layout
            );

            // Light list is shared by ReSTIR resample and basic-mode NEE.
            RSize emissive_num = ctx.structure_emissive_voxel_handles.size();
            ensure_vector_size(ctx.light_weight, emissive_num + 1);
            ensure_vector_size(ctx.light_cdf, emissive_num + 1);

            // Sycnrhonize kernel context with updated aux buffers and light data.
            _kernel_manager->markKernelModified();

            float sun_weight = float(ctx.cosmic_background_vision_mem.sun_color.luminance()) *
                               float(ctx.cosmic_background_vision_mem.intensity);
            ctx.light_weight.data.copyFrom(
                RS_ConstMemoryPtr(&sun_weight, sizeof(float)), sizeof(float)
            );

            _kernel_manager->syncKernelIfNeeded();
            if (emissive_num > 0) {
                p_ptr->template run<RSCF_LaunchDirect<
                    RS_VisionRayCompute<>::RSCF_WriteEmissiveLightWeights,
                    KernelType>>({emissive_num, RS_AUTO_BLOCK_SIZE, false});
            }
            _scan.template inclusiveSum<float, float>(ctx.light_weight, ctx.light_cdf);

            if (k_ptr->vision_is_restir_enabled) {
                p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                    RS_VisionRayCompute<>::RSCF_TraceCameraInfoRaysRestir,
                    KernelType>>(
                    {ctx.camera_info_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_info_ray_layout
                );

                // Temporal reuse, merges last frame's prev reservoir.
                p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                    RS_VisionRayCompute<>::RSCF_ResampleReservoirDI,
                    KernelType>>(
                    {ctx.camera_info_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_info_ray_layout
                );

                // Ouyang 2021 Alg 4: single spatial pass per pixel
                // that reads only from the temporal buffer and writes to the
                // spatial buffer. The kernel iterates K neighbor picks
                // internally; with K=0 it pass-through-copies self.
                p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                    RS_VisionRayCompute<>::RSCF_SpatialReuseReservoirDI,
                    KernelType>>(
                    {ctx.camera_info_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_info_ray_layout
                );

                p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                    RS_VisionRayCompute<>::RSCF_ResampleReservoirGI,
                    KernelType>>(
                    {ctx.camera_info_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_info_ray_layout
                );

                p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                    RS_VisionRayCompute<>::RSCF_SpatialReuseReservoirGI,
                    KernelType>>(
                    {ctx.camera_info_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_info_ray_layout
                );

                // The resample above rebuilt the reservoirs against the current light set, so
                // reuse is safe again next frame.
                _kernel_manager->clearVisionRestirReservoirStale();

                // Restir RGB is depth-0 only (chain throughput + reservoirs
                // already encode direct + 1-bounce indirect); launched per pixel.
                p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                    RS_VisionRayCompute<>::RSCF_TraceCameraRGBRaysRestir,
                    KernelType>>(
                    {ctx.camera_info_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_info_ray_layout
                );
            } else {
                p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                    RS_VisionRayCompute<>::RSCF_TraceCameraInfoRaysBasic,
                    KernelType>>(
                    {ctx.camera_info_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_info_ray_layout
                );

                p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                    RS_VisionRayCompute<>::RSCF_TraceCameraRGBRaysBasic,
                    KernelType>>(
                    {ctx.camera_rgb_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_rgb_ray_layout
                );
            }

            // Denoise.
            const auto denoise_type = k_ptr->vision_denoise_type;
            const RSize denoise_iterations = k_ptr->vision_denoise_atrous_iterations;
            const bool denoise_active = (denoise_type == RSE_VISION_DENOISE_ATROUS ||
                                         denoise_type == RSE_VISION_DENOISE_SVGF) &&
                                        denoise_iterations > 0 &&
                                        ctx.camera_info_ray_layout.size() > 0;
            if (denoise_active) {
                if (denoise_type == RSE_VISION_DENOISE_SVGF) {
                    // Temporal accumulate + variance estimate, then ping-pong
                    // history pairs so next frame's "prev" sees this frame.
                    p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                        RS_VisionDenoiseCompute<>::RSCF_SVGFTemporalAccumulate,
                        KernelType>>(
                        {ctx.camera_info_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                        ctx.camera_info_ray_layout
                    );
                    p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                        RS_VisionDenoiseCompute<>::RSCF_SVGFVarianceEstimate,
                        KernelType>>(
                        {ctx.camera_info_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                        ctx.camera_info_ray_layout
                    );
                    exchange(
                        ctx.camera_denoiser_color_history, ctx.camera_prev_denoiser_color_history
                    );
                    exchange(
                        ctx.camera_denoiser_moment_history, ctx.camera_prev_denoiser_moment_history
                    );
                    exchange(
                        ctx.camera_denoiser_history_length, ctx.camera_prev_denoiser_history_length
                    );
                }
                for (RSize pass = 0; pass < denoise_iterations; ++pass) {
                    if (denoise_type == RSE_VISION_DENOISE_SVGF) {
                        p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                            RS_VisionDenoiseCompute<>::RSCF_SVGFAtrousFilter,
                            KernelType>>(
                            {ctx.camera_info_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                            ctx.camera_info_ray_layout,
                            ctx.camera_current_rgb_canvas,
                            ctx.camera_next_rgb_canvas,
                            ctx.camera_denoiser_variance,
                            ctx.camera_next_denoiser_variance,
                            pass
                        );
                        exchange(ctx.camera_denoiser_variance, ctx.camera_next_denoiser_variance);
                    } else {
                        p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                            RS_VisionDenoiseCompute<>::RSCF_AtrousFilter,
                            KernelType>>(
                            {ctx.camera_info_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                            ctx.camera_info_ray_layout,
                            ctx.camera_current_rgb_canvas,
                            ctx.camera_next_rgb_canvas,
                            pass
                        );
                    }
                    exchange(ctx.camera_current_rgb_canvas, ctx.camera_next_rgb_canvas);
                }
            }

            // Ping-pong all temporal reservoirs at end of frame so the next
            // frame's "prev" sees this frame's data.
            if (k_ptr->vision_is_restir_enabled) {
                exchange(ctx.camera_prev_reservoir_di, ctx.camera_current_reservoir_di);
                exchange(ctx.camera_prev_reservoir_gi, ctx.camera_current_reservoir_gi);
                _kernel_manager->markKernelModified();
                _kernel_manager->syncKernelIfNeeded();
            }
        });
    });
}

void RS_SimulationExecutor::_updateIntersectRegionRecord() {
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            selectAccuracy(_simulation_compute_accuracy, [&](auto acc_const) {
                static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
                using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;
                using IRC = RS_IntersectRegionCompute<ComputeAccuracy>;
                auto &ctx = k_ptr->ctx;

                RSize region_num = ctx.intersect_regions.size();
                if (region_num == 0) return;

                // Materialize the region BVH leaves from the live source mem and rebuild it.
                ctx.intersect_region_bvh.resize(region_num, false);
                p_ptr->template run<
                    RSCF_LaunchDirect<typename IRC::RSCF_UpdateIntersectRegion, KernelType>>(
                    {region_num, RS_AUTO_BLOCK_SIZE, false},
                    ctx.intersect_region_bvh.object_identifiers,
                    ctx.intersect_region_bvh.leaf_bounds
                );
                ctx.intersect_region_bvh.rebuild();
                _kernel_manager->markKernelModified();
                _kernel_manager->syncKernelIfNeeded();

                auto voxel_handles = _getLaunchHandles("voxel");
                RSize voxel_num = voxel_handles.size();
                if (voxel_num == 0) {
                    _kernel_manager->setIntersectRegionRecords({});
                    return;
                }

                // Pass 1: count region memberships per voxel, then exclusive-scan to offsets.
                ensure_vector_size(ctx.intersect_region_counts, voxel_num);
                ensure_vector_size(ctx.intersect_region_offsets, voxel_num);
                p_ptr->template run<RSCF_LaunchSetOrGetWithHandles<
                    typename IRC::RSCF_CountIntersectRegionMembership,
                    KernelType>>(
                    {voxel_num, RS_AUTO_BLOCK_SIZE, false},
                    voxel_handles,
                    ctx.intersect_region_counts.data
                );
                _scan.template exclusiveSum<RSize, RSize>(
                    ctx.intersect_region_counts, ctx.intersect_region_offsets
                );

                RSize last_offset = 0;
                RSize last_count = 0;
                ctx.intersect_region_offsets.readSlice(&last_offset, 1, voxel_num - 1, false);
                ctx.intersect_region_counts.readSlice(&last_count, 1, voxel_num - 1, true);
                RSize total = last_offset + last_count;
                if (total == 0) {
                    _kernel_manager->setIntersectRegionRecords({});
                    return;
                }

                // Pass 2: emit one (record_gid, structure_gid) pair per membership.
                ensure_vector_size(ctx.intersect_region_pairs, total);
                p_ptr->template run<RSCF_LaunchSetOrGetWithHandles<
                    typename IRC::RSCF_WriteIntersectRegionMembership,
                    KernelType>>(
                    {voxel_num, RS_AUTO_BLOCK_SIZE, false},
                    voxel_handles,
                    ctx.intersect_region_offsets.data,
                    ctx.intersect_region_pairs
                );

                // Condense: sort record-gid-major, flag run starts, scan, compact to unique set.
                ensure_vector_size(ctx.intersect_region_keys, total);
                ensure_vector_size(ctx.intersect_region_sorted_keys, total);
                ensure_vector_size(ctx.intersect_region_indices, total);
                ensure_vector_size(ctx.intersect_region_sorted_indices, total);
                ensure_vector_size(ctx.intersect_region_flags, total);
                ensure_vector_size(ctx.intersect_region_prefix_sums, total);

                p_ptr->template run<
                    RSCF_LaunchDirect<logic::RSCF_GetIntersectRegionPairKeys, KernelType>>(
                    {total, RS_AUTO_BLOCK_SIZE, false},
                    ctx.intersect_region_pairs,
                    ctx.intersect_region_keys,
                    ctx.intersect_region_indices
                );
                _sort.template sort<RBigIndex, RIndex>(
                    ctx.intersect_region_keys,
                    ctx.intersect_region_indices,
                    ctx.intersect_region_sorted_keys,
                    ctx.intersect_region_sorted_indices
                );
                p_ptr->template run<
                    RSCF_LaunchDirect<logic::RSCF_FlagIntersectRegionPairRunStart, KernelType>>(
                    {total, RS_AUTO_BLOCK_SIZE, false},
                    ctx.intersect_region_sorted_keys,
                    ctx.intersect_region_flags,
                    RIndex(total)
                );
                _scan.template inclusiveSum<RIndex, RIndex>(
                    ctx.intersect_region_flags, ctx.intersect_region_prefix_sums
                );
                RIndex unique_num = 0;
                ctx.intersect_region_prefix_sums.readSlice(&unique_num, 1, total - 1, true);
                if (unique_num == 0) {
                    _kernel_manager->setIntersectRegionRecords({});
                    return;
                }

                ensure_vector_size(ctx.intersect_region_unique_pairs, unique_num);
                p_ptr->template run<
                    RSCF_LaunchDirect<logic::RSCF_CompactIntersectRegionPairs, KernelType>>(
                    {total, RS_AUTO_BLOCK_SIZE, false},
                    ctx.intersect_region_pairs,
                    ctx.intersect_region_sorted_indices,
                    ctx.intersect_region_flags,
                    ctx.intersect_region_prefix_sums,
                    ctx.intersect_region_unique_pairs
                );

                // Deliver the record-gid-major unique set to the owning managers.
                std::vector<logic::RS_IntersectRegionPair> host_pairs(unique_num);
                ctx.intersect_region_unique_pairs.readSlice(host_pairs.data(), unique_num, 0, true);
                _kernel_manager->setIntersectRegionRecords(host_pairs);
            });
        });
    });
}

void RS_SimulationExecutor::_storeRecorderRecords() {
    if (_kernel_manager->getKernelRecordBufferFrameNum() == 0) return;
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;
            auto &ctx = k_ptr->ctx;

            /****************************************************************
             * Structure records
             ****************************************************************/
            if (ctx.host_data->structure_record_layout.size() != 0) {
                p_ptr->template run<
                    RSCF_LaunchUpdateRootRecord<RSCF_UpdateStructureRootRecord, KernelType>>(
                    {ctx.structure_records.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_recorder_buffers,
                    ctx.structure_records,
                    ctx.structure_voxel_record_layout,
                    ctx.structure_link_record_layout,
                    ctx.structure_rigid_body_record_layout,
                    ctx.structure_joint_record_layout
                );

                // voxel
                p_ptr->template run<RSCF_LaunchStoreRecordWithRecordHandles<
                    RSCF_StoreStructureVoxelRecord,
                    KernelType>>(
                    {ctx.structure_voxel_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_records,
                    ctx.structure_voxel_record_handles,
                    ctx.structure_voxel_record_layout
                );

                // link
                p_ptr->template run<RSCF_LaunchStoreRecordWithRecordHandles<
                    RSCF_StoreStructureLinkRecord,
                    KernelType>>(
                    {ctx.structure_link_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_records,
                    ctx.structure_link_record_handles,
                    ctx.structure_link_record_layout
                );

                // rigid body
                p_ptr->template run<RSCF_LaunchStoreRecordWithRecordHandles<
                    RSCF_StoreStructureRigidBodyRecord,
                    KernelType>>(
                    {ctx.structure_rigid_body_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_records,
                    ctx.structure_rigid_body_record_handles,
                    ctx.structure_rigid_body_record_layout
                );

                // Joint
                // Note: for begin and end, we need to pass joint_record_layout because they
                // store frame size. But for actually store the record, we need to pass
                // structure_constraint_joint_layout because worker access per constraint joint data
                // and write to joint records
                p_ptr->template run<RSCF_LaunchStoreRecordWithRecordHandles<
                    RSCF_StoreStructureJointRecord,
                    KernelType>>(
                    {ctx.structure_constraint_joint_record_layout.size(),
                     RS_AUTO_BLOCK_SIZE,
                     false},
                    ctx.structure_records,
                    ctx.structure_constraint_joint_record_handles,
                    ctx.structure_constraint_joint_record_layout
                );
            }

            /****************************************************************
             * Camera records
             ****************************************************************/
            if (ctx.host_data->camera_record_layout.size() != 0) {
                p_ptr->template run<
                    RSCF_LaunchUpdateRootRecord<RSCF_UpdateCameraRootRecord, KernelType>>(
                    {ctx.camera_records.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_recorder_buffers,
                    ctx.camera_records,
                    ctx.camera_rgb_record_layout,
                    ctx.camera_depth_record_layout,
                    ctx.camera_normal_record_layout,
                    ctx.camera_motion_record_layout,
                    ctx.camera_entity_info_record_layout,
                    ctx.camera_state_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraRGBRecord, KernelType>>(
                    {ctx.camera_rgb_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_rgb_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraDepthRecord, KernelType>>(
                    {ctx.camera_depth_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_depth_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraNormalRecord, KernelType>>(
                    {ctx.camera_normal_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_normal_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraMotionRecord, KernelType>>(
                    {ctx.camera_motion_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_motion_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraEntityInfoRecord, KernelType>>(
                    {ctx.camera_entity_info_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_entity_info_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraStateRecord, KernelType>>(
                    {ctx.camera_state_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_state_record_layout
                );
            }

            /****************************************************************
             * Ray sensor records
             ****************************************************************/
            if (ctx.host_data->ray_sensor_record_layout.size() != 0) {
                p_ptr->template run<
                    RSCF_LaunchUpdateRootRecord<RSCF_UpdateRaySensorRootRecord, KernelType>>(
                    {ctx.ray_sensor_records.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.ray_sensor_recorder_buffers,
                    ctx.ray_sensor_records,
                    ctx.ray_sensor_hit_record_layout,
                    ctx.ray_sensor_entity_info_record_layout,
                    ctx.ray_sensor_entity_detail_record_layout,
                    ctx.ray_sensor_material_record_layout,
                    ctx.ray_sensor_normal_record_layout,
                    ctx.ray_sensor_velocity_record_layout,
                    ctx.ray_sensor_state_record_layout
                );

                p_ptr->template run<RSCF_LaunchStoreRecordWithLayout<
                    RS_RaySensorRayCompute<>::RSCF_TraceRaySensorRays,
                    KernelType>>(
                    {ctx.ray_sensor_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.ray_sensor_records,
                    ctx.ray_sensor_ray_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreRaySensorStateRecord, KernelType>>(
                    {ctx.ray_sensor_state_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.ray_sensor_records,
                    ctx.ray_sensor_state_record_layout
                );
            }
        });
    });
}

void RS_SimulationExecutor::_storeControllerRecords() {
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;
            auto &ctx = k_ptr->ctx;

            /****************************************************************
             * Structure records
             ****************************************************************/
            if (ctx.host_data->structure_record_layout.size() != 0) {
                p_ptr->template run<
                    RSCF_LaunchUpdateRootRecord<RSCF_UpdateStructureRootRecord, KernelType>>(
                    {ctx.structure_records.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_controller_buffers,
                    ctx.structure_records,
                    ctx.structure_voxel_record_layout,
                    ctx.structure_link_record_layout,
                    ctx.structure_rigid_body_record_layout,
                    ctx.structure_joint_record_layout
                );

                // voxel
                p_ptr->template run<RSCF_LaunchStoreRecordWithRecordHandles<
                    RSCF_StoreStructureVoxelRecord,
                    KernelType>>(
                    {ctx.structure_voxel_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_records,
                    ctx.structure_voxel_record_handles,
                    ctx.structure_voxel_record_layout
                );

                // link
                p_ptr->template run<RSCF_LaunchStoreRecordWithRecordHandles<
                    RSCF_StoreStructureLinkRecord,
                    KernelType>>(
                    {ctx.structure_link_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_records,
                    ctx.structure_link_record_handles,
                    ctx.structure_link_record_layout
                );

                // rigid body
                p_ptr->template run<RSCF_LaunchStoreRecordWithRecordHandles<
                    RSCF_StoreStructureRigidBodyRecord,
                    KernelType>>(
                    {ctx.structure_rigid_body_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_records,
                    ctx.structure_rigid_body_record_handles,
                    ctx.structure_rigid_body_record_layout
                );

                // Joint
                // Note: for begin and end, we need to pass joint_record_layout because they
                // store frame size. But for actually store the record, we need to pass
                // structure_constraint_joint_layout because worker access per constraint joint data
                // and write to joint records
                p_ptr->template run<RSCF_LaunchStoreRecordWithRecordHandles<
                    RSCF_StoreStructureJointRecord,
                    KernelType>>(
                    {ctx.structure_constraint_joint_record_layout.size(),
                     RS_AUTO_BLOCK_SIZE,
                     false},
                    ctx.structure_records,
                    ctx.structure_constraint_joint_record_handles,
                    ctx.structure_constraint_joint_record_layout
                );
            }

            /****************************************************************
             * Camera records
             ****************************************************************/
            if (ctx.host_data->camera_record_layout.size() != 0) {
                p_ptr->template run<
                    RSCF_LaunchUpdateRootRecord<RSCF_UpdateCameraRootRecord, KernelType>>(
                    {ctx.camera_records.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_controller_buffers,
                    ctx.camera_records,
                    ctx.camera_rgb_record_layout,
                    ctx.camera_depth_record_layout,
                    ctx.camera_normal_record_layout,
                    ctx.camera_motion_record_layout,
                    ctx.camera_entity_info_record_layout,
                    ctx.camera_state_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraRGBRecord, KernelType>>(
                    {ctx.camera_rgb_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_rgb_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraDepthRecord, KernelType>>(
                    {ctx.camera_depth_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_depth_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraNormalRecord, KernelType>>(
                    {ctx.camera_normal_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_normal_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraMotionRecord, KernelType>>(
                    {ctx.camera_motion_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_motion_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraEntityInfoRecord, KernelType>>(
                    {ctx.camera_entity_info_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_entity_info_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraStateRecord, KernelType>>(
                    {ctx.camera_state_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_state_record_layout
                );
            }

            /****************************************************************
             * Ray sensor records
             ****************************************************************/
            if (ctx.host_data->ray_sensor_record_layout.size() != 0) {
                p_ptr->template run<
                    RSCF_LaunchUpdateRootRecord<RSCF_UpdateRaySensorRootRecord, KernelType>>(
                    {ctx.ray_sensor_records.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.ray_sensor_controller_buffers,
                    ctx.ray_sensor_records,
                    ctx.ray_sensor_hit_record_layout,
                    ctx.ray_sensor_entity_info_record_layout,
                    ctx.ray_sensor_entity_detail_record_layout,
                    ctx.ray_sensor_material_record_layout,
                    ctx.ray_sensor_normal_record_layout,
                    ctx.ray_sensor_velocity_record_layout,
                    ctx.ray_sensor_state_record_layout
                );

                p_ptr->template run<RSCF_LaunchStoreRecordWithLayout<
                    RS_RaySensorRayCompute<>::RSCF_TraceRaySensorRays,
                    KernelType>>(
                    {ctx.ray_sensor_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.ray_sensor_records,
                    ctx.ray_sensor_ray_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreRaySensorStateRecord, KernelType>>(
                    {ctx.ray_sensor_state_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.ray_sensor_records,
                    ctx.ray_sensor_state_record_layout
                );
            }
        });
    });
}

void RS_SimulationExecutor::_rebuildStaticVoxelBVHIfNeeded() {
    // Static geometry rebuilds when the partition membership changed or a transform moved it.
    _resolveVoxelMobilityPartitionIfNeeded();
    if (!_is_static_bvh_dirty && !_kernel_manager->isStaticVoxelMoved()) return;
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        _buildVoxelBVH(k_ptr->ctx.static_voxel_bvh, _handles.at("static_voxel"));
    });
    _is_static_bvh_dirty = false;
    _kernel_manager->clearStaticVoxelMoved();
}

void RS_SimulationExecutor::_rebuildDynamicVoxelBVH() {
    _resolveVoxelMobilityPartitionIfNeeded();
    _resolveVoxelBVHDynamicPartition();
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        _buildVoxelBVH(k_ptr->ctx.dynamic_voxel_bvh, _handles.at("bvh_dynamic_voxel"));
    });
}

void RS_SimulationExecutor::_refitDynamicVoxelBVH() {
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        _refitVoxelBVH(k_ptr->ctx.dynamic_voxel_bvh);
    });
}

void RS_SimulationExecutor::_rebuildFloorTriangleBVHIfNeeded() {
    if (!_kernel_manager->isFloorTriangleModified()) return;

    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;
            auto &ctx = k_ptr->ctx;

            RSize triangle_count = ctx.mesh_triangles.size();
            if (triangle_count == 0) {
                ctx.floor_triangle_bvh.resize(0);
                _kernel_manager->clearFloorTriangleModified();
                return;
            }

            ctx.floor_triangle_bvh.resize(triangle_count, false);

            p_ptr->template run<RSCF_LaunchDirect<
                RS_MeshTriangleCompute::RSCF_FillTriangleBVHLeafData,
                KernelType>>(
                {triangle_count, RS_AUTO_BLOCK_SIZE, false},
                ctx.mesh_triangles,
                ctx.floor_triangle_bvh.object_identifiers,
                ctx.floor_triangle_bvh.leaf_bounds
            );

            ctx.floor_triangle_bvh.rebuild();

            _kernel_manager->markKernelModified();
            _kernel_manager->syncKernelIfNeeded();

            _kernel_manager->clearFloorTriangleModified();
        });
    });
}

void RS_SimulationExecutor::_buildVoxelBVH(
    logic::RS_VoxelBVH &bvh,
    device::RS_Vector<RHandle> &handles
) {
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            selectAccuracy(_simulation_compute_accuracy, [&](auto acc_const) {
                static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
                using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;

                RSize num_active = handles.size();
                bvh.resize(num_active, false);
                if (num_active > 0) {
                    p_ptr->template run<RSCF_LaunchDirect<
                        typename RS_VoxelCompute<ComputeAccuracy>::RSCF_FillVoxelBVHLeafData,
                        KernelType>>(
                        {num_active, RS_AUTO_BLOCK_SIZE, false},
                        handles,
                        bvh.object_identifiers,
                        bvh.leaf_bounds
                    );
                    bvh.rebuild();
                }
                _kernel_manager->markKernelModified();
                _kernel_manager->syncKernelIfNeeded();
            });
        });
    });
}

void RS_SimulationExecutor::_refitVoxelBVH(logic::RS_VoxelBVH &bvh) {
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            selectAccuracy(_simulation_compute_accuracy, [&](auto acc_const) {
                static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
                using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;

                RSize leaf_count = bvh.object_identifiers.size();
                if (leaf_count == 0) return;

                p_ptr->template run<RSCF_LaunchDirect<
                    typename RS_VoxelCompute<ComputeAccuracy>::RSCF_RefitVoxelBVHLeafData,
                    KernelType>>(
                    {leaf_count, RS_AUTO_BLOCK_SIZE, false}, bvh.object_identifiers, bvh.leaf_bounds
                );

                bvh.refit();
            });
        });
    });
}

template <typename Body>
void RS_SimulationExecutor::_runWithCriticalDataAccessGuard(Body &&body) {
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            selectAccuracy(_simulation_compute_accuracy, [&](auto acc_const) {
                // Take critical data access so no commit can mutate the context while we launch.
                RS_KernelManagerCriticalDataAccessGuard guard(_kernel_manager);
                bool was_modified = _kernel_manager->isKernelModified();
                _kernel_manager->syncKernelIfNeeded();
                body(k_ptr, p_ptr, acc_const, was_modified);
            });
        });
    });
}

template <typename Body>
void RS_SimulationExecutor::_runWithCommandExecutionGuard(Body &&body) {
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            selectAccuracy(_simulation_compute_accuracy, [&](auto acc_const) {
                // Pause command execution so no command runs while we touch manager state.
                RS_KernelManagerCommandExecutionGuard guard(_kernel_manager);
                bool was_modified = _kernel_manager->isKernelModified();
                _kernel_manager->syncKernelIfNeeded();
                body(k_ptr, p_ptr, acc_const, was_modified);
            });
        });
    });
}

RS_Vector<RHandle> RS_SimulationExecutor::_getLaunchHandles(const std::string &name) {
    RS_Vector<RHandle> indices;
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            auto &ctx = k_ptr->ctx;
            if ((name == "voxel" || name == "link" || name == "rigid_body" ||
                 name == "rigid_body_surface_voxel" || name == "constraint_joint") &&
                _handles.find(name) == _handles.end()) {
                _handles.insert({name, RS_Vector<RHandle>(p_ptr)});
                _handle_version_ids[name] = 0;
            }
            if (name == "voxel") {
                if (ctx.host_data->structure_voxel_layout.versionID() !=
                    _handle_version_ids.at(name)) {
                    ensure_vector_size(
                        _handles["voxel"], ctx.host_data->structure_voxel_layout.size()
                    );
                    p_ptr->template run<RSCF_ResolveThreadHandleFromLayout>(
                        {ctx.structure_voxel_layout.size()},
                        ctx.structure_voxel_layout,
                        _handles["voxel"]
                    );
                    _handle_version_ids[name] = ctx.host_data->structure_voxel_layout.versionID();
                }
                indices = _handles["voxel"];
            } else if (name == "dynamic_voxel" || name == "static_voxel") {
                // The dynamic and static subsets are derived from the full voxel set.
                _resolveVoxelMobilityPartitionIfNeeded();
                indices = _handles.at(name);
            } else if (name == "bvh_dynamic_voxel") {
                // The dynamic BVH subset is derived from the current dynamic subset.
                _resolveVoxelMobilityPartitionIfNeeded();
                _resolveVoxelBVHDynamicPartition();
                indices = _handles.at(name);
            } else if (name == "link") {
                if (ctx.host_data->structure_link_layout.versionID() !=
                    _handle_version_ids.at(name)) {
                    ensure_vector_size(
                        _handles["link"], ctx.host_data->structure_link_layout.size()
                    );
                    p_ptr->template run<RSCF_ResolveThreadHandleFromLayout>(
                        {ctx.structure_link_layout.size()},
                        ctx.structure_link_layout,
                        _handles["link"]
                    );
                    _handle_version_ids[name] = ctx.host_data->structure_link_layout.versionID();
                }
                indices = _handles["link"];
            } else if (name == "rigid_body") {
                if (ctx.host_data->structure_rigid_body_layout.versionID() !=
                    _handle_version_ids.at(name)) {
                    ensure_vector_size(
                        _handles["rigid_body"], ctx.host_data->structure_rigid_body_layout.size()
                    );
                    p_ptr->template run<RSCF_ResolveThreadHandleFromLayout>(
                        {ctx.structure_rigid_body_layout.size()},
                        ctx.structure_rigid_body_layout,
                        _handles["rigid_body"]
                    );
                    _handle_version_ids[name] =
                        ctx.host_data->structure_rigid_body_layout.versionID();
                }
                indices = _handles["rigid_body"];
            } else if (name == "dynamic_rigid_body") {
                // The mobile subset is derived from the full rigid body set.
                _resolveRigidBodyMobilityPartitionIfNeeded();
                indices = _handles.at(name);
            } else if (name == "rigid_body_surface_voxel") {
                if (ctx.host_data->structure_rigid_body_surface_voxel_layout.versionID() !=
                    _handle_version_ids.at(name)) {
                    ensure_vector_size(
                        _handles["rigid_body_surface_voxel"],
                        ctx.host_data->structure_rigid_body_surface_voxel_layout.size()
                    );
                    p_ptr->template run<RSCF_ResolveThreadHandleFromLayout>(
                        {ctx.structure_rigid_body_surface_voxel_layout.size()},
                        ctx.structure_rigid_body_surface_voxel_layout,
                        _handles["rigid_body_surface_voxel"]
                    );
                    _handle_version_ids[name] =
                        ctx.host_data->structure_rigid_body_surface_voxel_layout.versionID();
                }
                indices = _handles["rigid_body_surface_voxel"];
            } else if (name == "constraint_joint") {
                if (ctx.host_data->structure_constraint_joint_layout.versionID() !=
                    _handle_version_ids.at(name)) {
                    ensure_vector_size(
                        _handles["constraint_joint"],
                        ctx.host_data->structure_constraint_joint_layout.size()
                    );
                    p_ptr->template run<RSCF_ResolveThreadHandleFromLayout>(
                        {ctx.structure_constraint_joint_layout.size()},
                        ctx.structure_constraint_joint_layout,
                        _handles["constraint_joint"]
                    );
                    _handle_version_ids[name] =
                        ctx.host_data->structure_constraint_joint_layout.versionID();
                }
                indices = _handles["constraint_joint"];

            } else {
                RS_THROW(RS_InvalidArgumentException("Invalid index name {}", name));
            }
        });
    });
    return indices;
}

bool RS_SimulationExecutor::_resolveVoxelMobilityPartitionIfNeeded() {
    bool recomputed = false;
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            selectAccuracy(_simulation_compute_accuracy, [&](auto acc_const) {
                static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
                using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;
                auto &ctx = k_ptr->ctx;

                // Physics iterates the dynamic subset, the static BVH consumes the static subset;
                // the dynamic BVH subset is derived from the dynamic subset on every rebuild.
                if (_handles.find("dynamic_voxel") == _handles.end()) {
                    _handles.insert({"dynamic_voxel", RS_Vector<RHandle>(p_ptr)});
                    _handles.insert({"static_voxel", RS_Vector<RHandle>(p_ptr)});
                    _handle_version_ids["voxel_partition"] = 0;
                }

                // Recompute when the layout topology changed or any voxel's mobility flipped.
                uint64_t version = ctx.host_data->structure_voxel_layout.versionID();
                bool dirty = _kernel_manager->isVoxelPartitionModified() ||
                             _handle_version_ids.at("voxel_partition") != version;
                if (!dirty) return;

                auto voxel_handles = _getLaunchHandles("voxel");
                RSize num_voxels = voxel_handles.size();

                if (num_voxels == 0) {
                    _handles["dynamic_voxel"].occupy(0);
                    _handles["static_voxel"].occupy(0);
                } else {
                    ensure_vector_size(ctx.structure_voxel_dynamic_flags, num_voxels);
                    ensure_vector_size(ctx.structure_voxel_dynamic_prefix_sums, num_voxels);
                    ensure_vector_size(ctx.structure_voxel_static_flags, num_voxels);
                    ensure_vector_size(ctx.structure_voxel_static_prefix_sums, num_voxels);

                    // One classify pass writes both flag arrays.
                    p_ptr->template run<RSCF_LaunchSetOrGetWithHandles<
                        logic::RSCF_GetVoxelMobilityFlags,
                        KernelType>>(
                        {num_voxels, RS_AUTO_BLOCK_SIZE, false},
                        voxel_handles,
                        ctx.structure_voxel_dynamic_flags,
                        ctx.structure_voxel_static_flags
                    );

                    auto compact = [&](auto &flags, auto &prefix_sums, const char *key) {
                        _scan.template inclusiveSum<RIndex, RIndex>(flags, prefix_sums);
                        RIndex count = 0;
                        prefix_sums.readSlice(&count, 1, num_voxels - 1, true);
                        ensure_vector_size(_handles[key], count);
                        if (count > 0) {
                            p_ptr->template run<RSCF_LaunchSetOrGetWithHandles<
                                logic::RSCF_CompactVoxelHandles,
                                KernelType>>(
                                {num_voxels, RS_AUTO_BLOCK_SIZE, false},
                                voxel_handles,
                                flags,
                                prefix_sums,
                                _handles[key]
                            );
                        }
                    };
                    compact(
                        ctx.structure_voxel_dynamic_flags,
                        ctx.structure_voxel_dynamic_prefix_sums,
                        "dynamic_voxel"
                    );
                    compact(
                        ctx.structure_voxel_static_flags,
                        ctx.structure_voxel_static_prefix_sums,
                        "static_voxel"
                    );
                }

                _handle_version_ids["voxel_partition"] = version;
                _kernel_manager->clearVoxelPartitionModified();
                _is_static_bvh_dirty = true;
                recomputed = true;
            });
        });
    });
    return recomputed;
}

void RS_SimulationExecutor::_resolveVoxelBVHDynamicPartition() {
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            selectAccuracy(_simulation_compute_accuracy, [&](auto acc_const) {
                static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
                using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;
                auto &ctx = k_ptr->ctx;

                if (_handles.find("bvh_dynamic_voxel") == _handles.end()) {
                    _handles.insert({"bvh_dynamic_voxel", RS_Vector<RHandle>(p_ptr)});
                }

                auto &dynamic_handles = _handles.at("dynamic_voxel");
                RSize count_dynamic = dynamic_handles.size();
                if (count_dynamic == 0) {
                    _handles["bvh_dynamic_voxel"].occupy(0);
                    return;
                }

                ensure_vector_size(ctx.structure_voxel_bvh_dynamic_flags, count_dynamic);
                ensure_vector_size(ctx.structure_voxel_bvh_dynamic_prefix_sums, count_dynamic);

                // Re-evaluate exposure over the dynamic subset so a voxel exposed since the last
                // rebuild joins the dynamic BVH.
                p_ptr->template run<RSCF_LaunchSetOrGetWithHandles<
                    logic::RSCF_GetVoxelBVHDynamicFlags,
                    KernelType>>(
                    {count_dynamic, RS_AUTO_BLOCK_SIZE, false},
                    dynamic_handles,
                    ctx.structure_voxel_bvh_dynamic_flags
                );

                _scan.template inclusiveSum<RIndex, RIndex>(
                    ctx.structure_voxel_bvh_dynamic_flags,
                    ctx.structure_voxel_bvh_dynamic_prefix_sums
                );
                RIndex count = 0;
                ctx.structure_voxel_bvh_dynamic_prefix_sums.readSlice(
                    &count, 1, count_dynamic - 1, true
                );
                ensure_vector_size(_handles["bvh_dynamic_voxel"], count);
                if (count > 0) {
                    p_ptr->template run<RSCF_LaunchSetOrGetWithHandles<
                        logic::RSCF_CompactVoxelHandles,
                        KernelType>>(
                        {count_dynamic, RS_AUTO_BLOCK_SIZE, false},
                        dynamic_handles,
                        ctx.structure_voxel_bvh_dynamic_flags,
                        ctx.structure_voxel_bvh_dynamic_prefix_sums,
                        _handles["bvh_dynamic_voxel"]
                    );
                }
            });
        });
    });
}

bool RS_SimulationExecutor::_resolveRigidBodyMobilityPartitionIfNeeded() {
    bool recomputed = false;
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            using KernelType = std::remove_pointer_t<std::remove_reference_t<decltype(k_ptr)>>;
            auto &ctx = k_ptr->ctx;

            // Physics iterates the mobile subset, so static and frozen bodies cost no threads; the
            // full set still serves the transform and the contact anchors.
            if (_handles.find("dynamic_rigid_body") == _handles.end()) {
                _handles.insert({"dynamic_rigid_body", RS_Vector<RHandle>(p_ptr)});
                _handle_version_ids["rigid_body_partition"] = 0;
            }

            // Recompute when the layout topology changed or any body's mobility flipped.
            uint64_t version = ctx.host_data->structure_rigid_body_layout.versionID();
            bool dirty = _kernel_manager->isRigidBodyPartitionModified() ||
                         _handle_version_ids.at("rigid_body_partition") != version;
            if (!dirty) return;

            auto rigid_body_handles = _getLaunchHandles("rigid_body");
            RSize num_rigid_bodies = rigid_body_handles.size();

            if (num_rigid_bodies == 0) {
                _handles["dynamic_rigid_body"].occupy(0);
            } else {
                ensure_vector_size(ctx.structure_rigid_body_dynamic_flags, num_rigid_bodies);
                ensure_vector_size(ctx.structure_rigid_body_dynamic_prefix_sums, num_rigid_bodies);

                p_ptr->template run<RSCF_LaunchSetOrGetWithHandles<
                    logic::RSCF_GetRigidBodyMobilityFlags,
                    KernelType>>(
                    {num_rigid_bodies, RS_AUTO_BLOCK_SIZE, false},
                    rigid_body_handles,
                    ctx.structure_rigid_body_dynamic_flags
                );

                _scan.template inclusiveSum<RIndex, RIndex>(
                    ctx.structure_rigid_body_dynamic_flags,
                    ctx.structure_rigid_body_dynamic_prefix_sums
                );
                RIndex count = 0;
                ctx.structure_rigid_body_dynamic_prefix_sums.readSlice(
                    &count, 1, num_rigid_bodies - 1, true
                );
                ensure_vector_size(_handles["dynamic_rigid_body"], count);
                if (count > 0) {
                    p_ptr->template run<RSCF_LaunchSetOrGetWithHandles<
                        logic::RSCF_CompactRigidBodyHandles,
                        KernelType>>(
                        {num_rigid_bodies, RS_AUTO_BLOCK_SIZE, false},
                        rigid_body_handles,
                        ctx.structure_rigid_body_dynamic_flags,
                        ctx.structure_rigid_body_dynamic_prefix_sums,
                        _handles["dynamic_rigid_body"]
                    );
                }
            }

            _handle_version_ids["rigid_body_partition"] = version;
            _kernel_manager->clearRigidBodyPartitionModified();
            recomputed = true;
        });
    });
    return recomputed;
}

void RS_SimulationExecutor::_resizeReduceBufferIfNeeded(
    RBigSize needed_input_size,
    RBigSize needed_output_size
) {
    // Each buffer grows independently so the per-segment output is not over-allocated to the
    // per-element input size. Allocates even at size 0 so the buffer stays bound to the provider.
    if (_reduce_input_buffer.mem_provider == nullptr ||
        _reduce_input_buffer.size() < needed_input_size) {
        _reduce_input_buffer.free();
        _reduce_input_buffer = _device_provider->allocate(needed_input_size, true);
    }
    if (_reduce_output_buffer.mem_provider == nullptr ||
        _reduce_output_buffer.size() < needed_output_size) {
        _reduce_output_buffer.free();
        _reduce_output_buffer = _device_provider->allocate(needed_output_size, true);
    }
}

RSize RS_SimulationExecutor::_snapInterval(RFloat needed, RFloat time_step) {
    if (!(time_step > RFloat(0)) || !(needed > RFloat(0))) {
        return 1;
    }
    RFloat steps = std::floor(needed / time_step);
    // Clamp to keep the cast finite for unbounded (absent) subsystems.
    steps = RS_MIN(steps, RFloat(1e9));
    return steps >= RFloat(1) ? static_cast<RSize>(steps) : RSize(1);
}

}  // namespace rise
