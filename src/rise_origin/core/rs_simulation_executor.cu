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
#include "rise_origin/logic/record/rs_basic_record_c.h"
#include "rise_origin/logic/rigid/rs_constraint_c.h"
#include "rise_origin/logic/rigid/rs_contact_c.h"
#include "rise_origin/logic/rigid/rs_rigid_body_c.h"
#include "rise_origin/logic/rigid/rs_rigid_body_surface_voxel_c.h"
#include "rise_origin/logic/signal/rs_force_c.h"
#include "rise_origin/logic/signal/rs_torque_c.h"
#include "rise_origin/logic/soft/rs_link_c.h"
#include "rise_origin/logic/soft/rs_voxel_c.h"
#include "rise_origin/logic/vision/rs_camera_c.h"
#include "rise_origin/logic/vision/rs_vision_denoise_c.h"
#include "rise_origin/logic/vision/rs_vision_ray_c.h"
#include "rise_origin/utils/rs_constexpr.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_iterators.h"
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

class RSCF_UpdateKernelTimeStep : public RS_ComputeFunction<RSCF_UpdateKernelTimeStep> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RFloat time_step) {
        RS_ConstantMemory::as<RS_Kernel>()->updateTimeStepOnDevice(time_step);
    };
};

class RSCF_UpdateKernelOperationTime : public RS_ComputeFunction<RSCF_UpdateKernelOperationTime> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(RSE_KernelUpdateOperation operation) {
        RS_ConstantMemory::as<RS_Kernel>()->updateOperationTimeOnDevice(operation);
    };
};

// In this way, Launch* has all the attributes marked by ComputeFunction
template <typename ComputeFunction, typename Kernel>
class RSCF_LaunchDirect : public RS_ComputeFunction<ComputeFunction> {
   public:
    using RS_ComputeFunction<ComputeFunction>::index1D;

    template <typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void run(Args... args) {
        ComputeFunction::run(RS_ConstantMemory::as<Kernel>(), args...);
    }
};

template <typename ComputeFunction, typename Kernel>
class RSCF_LaunchWithHandles : public RS_ComputeFunction<ComputeFunction> {
   public:
    using RS_ComputeFunction<ComputeFunction>::index1D;

    template <typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_Array<RHandle, 1> handles, Args... args) {
        ComputeFunction::run(RS_ConstantMemory::as<Kernel>(), handles[index1D()], args...);
    }
};

template <typename ComputeFunction, typename Kernel>
class RSCF_LaunchWithLayout
    : public RS_ComputeFunction<RSCF_LaunchWithLayout<ComputeFunction, Kernel>> {
   public:
    using RS_ComputeFunction<RSCF_LaunchWithLayout>::index1D;

    template <typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_DeviceLayout layout, Args... args) {
        RIndex i = index1D();
        auto &ctx = RS_ConstantMemory::as<Kernel>()->ctx;
        auto search_result = layout.searchForBelongedSegment(i);
        ComputeFunction::run(RS_ConstantMemory::as<Kernel>(), search_result.global_index, args...);
    }
};

template <typename ComputeFunction, typename Kernel>
class RSCF_LaunchWithLayoutPerSegment
    : public RS_ComputeFunction<RSCF_LaunchWithLayoutPerSegment<ComputeFunction, Kernel>> {
   public:
    using RS_ComputeFunction<RSCF_LaunchWithLayoutPerSegment>::index1D;

    template <typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_DeviceLayout layout, Args... args) {
        RIndex i = index1D();
        auto &ctx = RS_ConstantMemory::as<Kernel>()->ctx;
        auto search_result = layout.searchForBelongedSegment(i);
        ComputeFunction::run(
            RS_ConstantMemory::as<Kernel>(),
            search_result.segment_index,
            search_result.in_segment_index,
            args...
        );
    }
};

template <typename SetOrGetFunction, typename Kernel>
class RSCF_LaunchSetOrGetWithHandles
    : public RS_ComputeFunction<RSCF_LaunchSetOrGetWithHandles<SetOrGetFunction, Kernel>> {
   public:
    using RS_ComputeFunction<RSCF_LaunchSetOrGetWithHandles>::index1D;

    template <typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_Array<RHandle, 1> handles, RS_MemoryPtr buffer, Args... args) {
        RIndex i = index1D();
        if (i >= handles.size()) return;

        SetOrGetFunction::run(RS_ConstantMemory::as<Kernel>(), buffer, handles[i], i, args...);
    }
};

template <typename SetOrGetFunction, typename Kernel>
class RSCF_LaunchSetOrGetWithLayout
    : public RS_ComputeFunction<RSCF_LaunchSetOrGetWithLayout<SetOrGetFunction, Kernel>> {
   public:
    using RS_ComputeFunction<RSCF_LaunchSetOrGetWithLayout>::index1D;

    template <typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_DeviceLayout layout, RS_MemoryPtr buffer, Args... args) {
        RIndex i = index1D();
        if (i >= layout.size()) return;

        auto &ctx = RS_ConstantMemory::as<Kernel>()->ctx;
        auto search_result = layout.searchForBelongedSegment(i);

        if (search_result.segment_id != RS_NULL_INDEX) {
            // Get target location from output layout and set the reduced value
            SetOrGetFunction::run(
                RS_ConstantMemory::as<Kernel>(),
                buffer,                      // Buffer
                search_result.global_index,  // Target handle in the kernel
                i,                           // Set/Get index in the buffer
                args...
            );
        }
    }
};

template <typename StoreRecordFunction, typename Kernel>
class RSCF_LaunchStoreRecordWithRecordHandles
    : public RS_ComputeFunction<
          RSCF_LaunchStoreRecordWithRecordHandles<StoreRecordFunction, Kernel>> {
   public:
    using RS_ComputeFunction<RSCF_LaunchStoreRecordWithRecordHandles>::index1D;

    template <typename RootRecord>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RS_Vector<RootRecord> root_records,
        RS_Vector<RS_Array<RIndex, 1>> record_handles,
        RS_DeviceLayout handle_layout
    ) {
        RIndex i = index1D();
        auto &ctx = RS_ConstantMemory::as<Kernel>()->ctx;
        auto search_result = handle_layout.searchForBelongedSegment(i);
        StoreRecordFunction::run(
            RS_ConstantMemory::as<Kernel>(),
            root_records[search_result.segment_index],
            record_handles[search_result.segment_index][search_result.in_segment_index],
            search_result.in_segment_index
        );
    }
};

template <typename StoreRecordFunction, typename Kernel>
class RSCF_LaunchStoreRecordWithLayout
    : public RS_ComputeFunction<RSCF_LaunchStoreRecordWithLayout<StoreRecordFunction, Kernel>> {
   public:
    using RS_ComputeFunction<RSCF_LaunchStoreRecordWithLayout>::index1D;

    template <typename RootRecord>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_Vector<RootRecord> root_records, RS_DeviceLayout layout) {
        RIndex i = index1D();
        auto &ctx = RS_ConstantMemory::as<Kernel>()->ctx;
        auto search_result = layout.searchForBelongedSegment(i);
        StoreRecordFunction::run(
            RS_ConstantMemory::as<Kernel>(),
            root_records[search_result.segment_index],
            search_result.segment_index,
            search_result.in_segment_index
        );
    }
};

template <typename Kernel>
class RSCF_LaunchUpdateRootRecord : public RS_ComputeFunction<RSCF_LaunchUpdateRootRecord<Kernel>> {
   public:
    using RS_ComputeFunction<RSCF_LaunchUpdateRootRecord>::index1D;

    template <typename Buffer, typename RootRecord, typename... Layout>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(RS_Vector<Buffer> buffers, RS_Vector<RootRecord> root_records, Layout... handle_layouts) {
        RIndex i = index1D();
        auto &ctx = RS_ConstantMemory::as<Kernel>()->ctx;
        RSCF_UpdateRootRecord::run(
            RS_ConstantMemory::as<Kernel>(),
            buffers[i],
            root_records[i],
            handle_layouts.segmentSize(i)...
        );
    }
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
    RS_SimulationCallback callback
)
    : _device_provider(device_provider),
      _host_provider(host_provider),
      _callback(std::move(callback)) {
    if (spdlog::get("root")) {
        // Only init logger if root logger is created
        _logger = spdlog::get("root")->clone("sequential executor");
    }

    castProviderPtrAndCall(device_provider, [&](auto p_ptr) {
        _reduce = RS_Reduce(p_ptr);
        _reduce_by_group = RS_ReduceByGroup(p_ptr);
        _scan = RS_Scan(p_ptr);
        _sort = RS_RadixSortPairs(p_ptr);
    });
}

void RS_SimulationExecutor::init(RIndex id, RS_Kernel *kernel, RS_KernelManager *kernel_manager) {
    _id = id;
    _kernel = kernel;
    _kernel_manager = kernel_manager;

    // Cache the runtime_* execution settings once, at init. Later user edits to the
    // simulation config do not change how this run executes.
    auto sim_config = kernel_manager->getSimulationConfig();
    _dt_update_interval = sim_config.runtime_dt_update_interval;
    _rigidbody_update_interval = sim_config.runtime_rigidbody_update_interval;
    _contact_update_interval = sim_config.runtime_contact_update_interval;
    _contact_bvh_rebuild_sub_interval = sim_config.runtime_contact_bvh_rebuild_sub_interval;
    _contact_bvh_refit_sub_interval = sim_config.runtime_contact_bvh_refit_sub_interval;
    _mechanical_signal_update_interval = sim_config.runtime_mechanical_signal_update_interval;
    _divergence_check_interval = sim_config.runtime_divergence_check_interval;
    _simulation_compute_accuracy = sim_config.runtime_simulation_compute_accuracy;

    lock_guard lock(kernel->getSyncMutex());
    RS_ConstantMemory::write(kernel, _device_provider->device, sizeof(RS_Kernel));
    kernel->clearModified();
}

void RS_SimulationExecutor::free() {
    _id = RS_NULL_INDEX;
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
    bool result = true;

    // Auto dt selection. A degenerate recommended dt aborts the simulation phases for this step
    // but still lets the kernel-manager phases below run, matching the original behaviour.
    if (_runDtUpdate()) {
        if (_kernel->getTimeSinceLastExecutionOfEvent("print_time_and_dt") >= 0.1) {
            _kernel->executeEvent("print_time_and_dt");
            RS_LOG(
                _logger, info, "time: {}s, dt: {}s.", _kernel->getHostTime(), _kernel->getHostDt()
            );
        }
        _runSoftbodyUpdate();
        if (_kernel->getHostStep() % _rigidbody_update_interval == 0) _runRigidbodyUpdate();
        _runRigidbodyCoupling();

        _advanceTime();

        if (_kernel->getTimeSinceLastExecutionOfEvent("record") >=
            1.0 / _kernel->record_frequency) {
            _kernel->executeEvent("record");
            _runRecordSaving();
        }
        if (_kernel->getTimeSinceLastExecutionOfEvent("callback") >=
            1.0 / _kernel->control_frequency) {
            _runControllerRecordSaving();
        }
        if (_kernel->getHostStep() % _divergence_check_interval == 0 && !_runDivergenceCheck())
            result = false;
    } else {
        result = false;
    }

    if (_kernel->getHostStep() % _contact_update_interval == 0) _runContactUpdate();
    if (_kernel->getHostStep() % _mechanical_signal_update_interval == 0)
        _runMechanicalSignalUpdate();
    if (_kernel->getTimeSinceLastExecutionOfEvent("callback") >= 1.0 / _kernel->control_frequency) {
        _kernel->executeEvent("callback");
        _performCallback();
    }

    return result;
}

bool RS_SimulationExecutor::_runDtUpdate() {
    bool result = true;
    _runGuarded([&](auto *k_ptr, auto *p_ptr, auto, bool was_modified) {
        using KernelType = std::remove_pointer_t<decltype(k_ptr)>;
        if (!k_ptr->isAutoTimeStep()) return;
        if (!(was_modified || k_ptr->getHostStep() % _dt_update_interval == 0)) return;

        // Largest natural frequency sqrt(k/m) in the simulation sets the stable dt:
        // dt = 1 / (2*pi * max_frequency), taken over links and voxels.
        RFloat time_step = RS_NumTraits<RFloat>::highest();
        auto link_handles = _getLaunchHandles("link");
        auto voxel_handles = _getLaunchHandles("voxel");
        _resizeReduceBufferIfNeeded(
            RS_MAX(link_handles.size(), voxel_handles.size()) * sizeof(RFloat)
        );

        if (link_handles.size() > 0) {
            p_ptr->template run<
                RSCF_LaunchSetOrGetWithHandles<RSCF_GetLinkFrequencyMetric, KernelType>>(
                {link_handles.size(), RS_AUTO_BLOCK_SIZE, false}, link_handles, _reduce_input_buffer
            );
            // Divide 2, sometimes link max freq is too constrictive
            RFloat link_max_freq = _reduce.reduce<RFloat, RS_ReduceMaxOp>(
                                       _reduce_input_buffer, link_handles.size(), 0
                                   ) /
                                   2;
            if (link_max_freq > 0)
                time_step = RS_MIN(time_step, 1 / (RS_2PI * sqrt(link_max_freq)));
        }
        if (voxel_handles.size() > 0) {
            p_ptr->template run<
                RSCF_LaunchSetOrGetWithHandles<RSCF_GetVoxelFrequencyMetric, KernelType>>(
                {voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false},
                voxel_handles,
                _reduce_input_buffer
            );
            RFloat voxel_max_freq = _reduce.reduce<RFloat, RS_ReduceMaxOp>(
                _reduce_input_buffer, voxel_handles.size(), 0
            );
            if (voxel_max_freq > 0)
                time_step = RS_MIN(time_step, 1 / (RS_2PI * sqrt(voxel_max_freq)));
        }
        // Default time step with empty simulation
        if (time_step == RS_NumTraits<RFloat>::highest()) time_step = 1e-3;

        if (time_step < 1e-9) {
            result = false;
            return;
        }
        p_ptr->template run<RSCF_UpdateKernelTimeStep>({1, 1, false}, time_step);
        k_ptr->updateTimeStepOnHost(time_step);
    });
    return result;
}

void RS_SimulationExecutor::_runSoftbodyUpdate() {
    _runGuarded([&](auto *k_ptr, auto *p_ptr, auto acc_const, bool) {
        static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
        using KernelType = std::remove_pointer_t<decltype(k_ptr)>;

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

        auto voxel_handles = _getLaunchHandles("voxel");
        p_ptr->template run<RSCF_LaunchWithHandles<
            typename RS_VoxelCompute<ComputeAccuracy>::RSCF_UpdateVoxelContact,
            KernelType>>({voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false}, voxel_handles);
        p_ptr->template run<RSCF_LaunchWithHandles<
            typename RS_VoxelCompute<ComputeAccuracy>::RSCF_UpdateVoxel,
            KernelType>>({voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false}, voxel_handles);
    });
}

void RS_SimulationExecutor::_runRigidbodyCoupling() {
    _runGuarded([&](auto *k_ptr, auto *p_ptr, auto acc_const, bool) {
        static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
        using KernelType = std::remove_pointer_t<decltype(k_ptr)>;
        auto &ctx = k_ptr->ctx;

        // Stamp the rigid-body operation time so the rigid kernels read the true rigid_dt.
        p_ptr->template run<RSCF_UpdateKernelOperationTime>(
            {1, 1, false}, RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_INTEGRATE
        );
        k_ptr->updateOperationTimeOnHost(RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_INTEGRATE);

        if (ctx.structure_rigid_body_layout.size() == 0) return;
        auto rigid_body_surface_voxel_handles = _getLaunchHandles("rigid_body_surface_voxel");

        // Gather each surface voxel's soft coupling correction and reduce onto its body.
        _resizeReduceBufferIfNeeded(
            ctx.structure_rigid_body_surface_voxel_to_rigid_body_reduction_input_layout.size() *
            sizeof(RS_CouplingCorrection)
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
            ctx.host_data->structure_rigid_body_surface_voxel_to_rigid_body_reduction_layout
                .getInputSegmentSizes(),
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

void RS_SimulationExecutor::_runRigidbodyUpdate() {
    _runGuarded([&](auto *k_ptr, auto *p_ptr, auto acc_const, bool) {
        static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
        using KernelType = std::remove_pointer_t<decltype(k_ptr)>;
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

        // Apply gravity impulse before any solver so the solver sees gravity-loaded lm.
        p_ptr->template run<RSCF_LaunchWithHandles<
            typename RS_RigidBodyCompute<ComputeAccuracy>::RSCF_ApplyGravity,
            KernelType>>(
            {rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false}, rigid_body_handles
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
            RSize shared_bytes = max_rb_per_structure * sizeof(SharedRBState);

            p_ptr->template run<RSCF_LaunchDirect<
                typename RS_ConstraintCompute<ComputeAccuracy>::RSCF_PGSUpdateJoint,
                KernelType>>(
                {total_threads, block_size, true, shared_bytes},
                ctx.structure_constraint_joint_block_ranges,
                k_ptr->host_data->constraint_joint_max_colors
            );

            // Zero the per-body coupling-work accumulator just consumed by compliance adaptation.
            p_ptr->template run<RSCF_LaunchWithHandles<
                typename RS_RigidBodyCompute<ComputeAccuracy>::RSCF_ResetRigidBodyCouplingWork,
                KernelType>>(
                {rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false}, rigid_body_handles
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
            {rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false}, rigid_body_handles
        );
    });
}

void RS_SimulationExecutor::_advanceTime() {
    _runGuarded([&](auto *k_ptr, auto *p_ptr, auto, bool) {
        p_ptr->template run<RSCF_UpdateKernelTime>({1, 1, false});
        k_ptr->updateTimeOnHost();
    });
}

void RS_SimulationExecutor::_runRecordSaving() {
    _runGuarded([&](auto *, auto *, auto, bool) {
        _updateLightAndVision();
        _storeRecorderRecords();
    });
}

void RS_SimulationExecutor::_runControllerRecordSaving() {
    _runGuarded([&](auto *, auto *, auto, bool) { _storeControllerRecords(); });
}

bool RS_SimulationExecutor::_runDivergenceCheck() {
    bool ok = true;
    _runGuarded([&](auto *k_ptr, auto *p_ptr, auto, bool) {
        using KernelType = std::remove_pointer_t<decltype(k_ptr)>;
        auto voxel_handles = _getLaunchHandles("voxel");
        p_ptr->template run<RSCF_LaunchSetOrGetWithHandles<RSCF_GetIsVoxelNanMetric, KernelType>>(
            {voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false}, voxel_handles, _reduce_input_buffer
        );
        if (_reduce.reduce<bool, RS_ReduceOrOp>(_reduce_input_buffer, voxel_handles.size(), false))
            ok = false;
    });
    return ok;
}

void RS_SimulationExecutor::_runContactUpdate() {
    RSize sub_step = _kernel->getHostStep() / _contact_update_interval;
    bool is_rebuild_needed = (sub_step % _contact_bvh_rebuild_sub_interval == 0);
    bool is_refit_needed = !is_rebuild_needed && (sub_step % _contact_bvh_refit_sub_interval == 0);
    bool is_discovery_needed = is_rebuild_needed || is_refit_needed;

    _runGuarded([&](auto *k_ptr, auto *p_ptr, auto acc_const, bool) {
        static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
        using KernelType = std::remove_pointer_t<decltype(k_ptr)>;

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
    _runGuarded([&](auto *k_ptr, auto *p_ptr, auto acc_const, bool) {
        static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
        using KernelType = std::remove_pointer_t<decltype(k_ptr)>;
        auto &ctx = k_ptr->ctx;

        // Force signals: rigid and soft.
        if (ctx.force_rigid.size() > 0) {
            auto rigid_body_handles = _getLaunchHandles("rigid_body");
            p_ptr->template run<RSCF_LaunchWithHandles<
                typename RS_ForceCompute<ComputeAccuracy>::RSCF_ApplyRigidForceSignal,
                KernelType>>(
                {rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false}, rigid_body_handles
            );

            p_ptr->template run<RSCF_LaunchDirect<
                typename RS_ForceCompute<ComputeAccuracy>::RSCF_UpdateRigidForceTime,
                KernelType>>({ctx.force_rigid.size(), RS_AUTO_BLOCK_SIZE, false});
        }
        if (ctx.force_soft.size() > 0) {
            auto voxel_handles = _getLaunchHandles("voxel");
            p_ptr->template run<RSCF_LaunchWithHandles<
                typename RS_ForceCompute<ComputeAccuracy>::RSCF_ApplySoftForceSignal,
                KernelType>>({voxel_handles.size(), RS_AUTO_BLOCK_SIZE, false}, voxel_handles);

            p_ptr->template run<RSCF_LaunchDirect<
                typename RS_ForceCompute<ComputeAccuracy>::RSCF_UpdateSoftForceTime,
                KernelType>>({ctx.force_soft.size(), RS_AUTO_BLOCK_SIZE, false});
        }

        // Torque signals: rigid only.
        if (ctx.torque_rigid.size() > 0) {
            auto rigid_body_handles = _getLaunchHandles("rigid_body");
            p_ptr->template run<RSCF_LaunchWithHandles<
                typename RS_TorqueCompute<ComputeAccuracy>::RSCF_ApplyRigidTorqueSignal,
                KernelType>>(
                {rigid_body_handles.size(), RS_AUTO_BLOCK_SIZE, false}, rigid_body_handles
            );

            p_ptr->template run<RSCF_LaunchDirect<
                typename RS_TorqueCompute<ComputeAccuracy>::RSCF_UpdateRigidTorqueTime,
                KernelType>>({ctx.torque_rigid.size(), RS_AUTO_BLOCK_SIZE, false});
        }
    });
}

void RS_SimulationExecutor::_performCallback() {
    if (_callback) {
        RS_KernelManagerGuard guard(_kernel_manager);

        _kernel_manager->syncControllerRecords();

        vector<pair<RSE_SimulationControllerType, RS_Any>> controllers;
        for (auto &structure_controller : _kernel_manager->getStructureControllers()) {
            controllers.emplace_back(RSE_SIMULATION_STRUCTURE_CONTROLLER, structure_controller);
        }
        for (auto &camera_controller : _kernel_manager->getCameraControllers()) {
            controllers.emplace_back(RSE_SIMULATION_CAMERA_CONTROLLER, camera_controller);
        }
        for (auto &force_controller : _kernel_manager->getForceControllers()) {
            controllers.emplace_back(RSE_SIMULATION_FORCE_CONTROLLER, force_controller);
        }
        for (auto &torque_controller : _kernel_manager->getTorqueControllers()) {
            controllers.emplace_back(RSE_SIMULATION_TORQUE_CONTROLLER, torque_controller);
        }

        _callback(_id, controllers);

        _kernel_manager->syncControllerSignals();
    }
}

void RS_SimulationExecutor::_updateLightAndVision() {
    _rebuildStaticVoxelBVHIfNeeded();
    _rebuildFloorTriangleBVHIfNeeded();
    _rebuildDynamicVoxelBVH();

    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            auto &ctx = k_ptr->ctx;
            using KernelType = std::remove_pointer_t<decltype(k_ptr)>;

            p_ptr->template run<
                RSCF_LaunchWithLayout<RS_CameraCompute<>::RSCF_UpdateCamera, KernelType>>(
                {ctx.camera_layout.size(), RS_AUTO_BLOCK_SIZE, false}, ctx.camera_layout
            );

            p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                RS_CameraCompute<>::RSCF_ClearCameraCanvas,
                KernelType>>(
                {ctx.camera_pixel_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                ctx.camera_pixel_record_layout
            );

            // Light list is shared by ReSTIR resample and basic-mode NEE.
            RSize emissive_num = ctx.structure_emissive_voxel_handles.size();
            ensure_vector_size(ctx.light_weight, emissive_num + 1);
            ensure_vector_size(ctx.light_cdf, emissive_num + 1);
            float sun_weight =
                float(k_ptr->world_sun_color.luminance()) * float(k_ptr->world_sky_intensity);
            ctx.light_weight.data.copyFrom(
                RS_ConstMemoryPtr(&sun_weight, sizeof(float)), sizeof(float)
            );
            if (emissive_num > 0) {
                p_ptr->template run<RS_VisionRayCompute<>::RSCF_WriteEmissiveLightWeights>(
                    {emissive_num, RS_AUTO_BLOCK_SIZE, false}, *k_ptr
                );
            }
            _scan.template inclusiveSum<float, float>(ctx.light_weight, ctx.light_cdf);

            // Synchronize context for vector resize.
            p_ptr->sync();
            RS_ConstantMemory::write(k_ptr, _device_provider->device, sizeof(RS_Kernel));

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
            const bool denoise_active =
                (denoise_type == RSE_VISION_DENOISE_ATROUS ||
                 denoise_type == RSE_VISION_DENOISE_SVGF) &&
                denoise_iterations > 0 && ctx.camera_info_ray_layout.size() > 0;
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
                        ctx.camera_denoiser_color_history,
                        ctx.camera_prev_denoiser_color_history
                    );
                    exchange(
                        ctx.camera_denoiser_moment_history,
                        ctx.camera_prev_denoiser_moment_history
                    );
                    exchange(
                        ctx.camera_denoiser_history_length,
                        ctx.camera_prev_denoiser_history_length
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
                            pass
                        );
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
                if (denoise_type == RSE_VISION_DENOISE_SVGF) {
                    p_ptr->template run<RSCF_LaunchWithLayoutPerSegment<
                        RS_VisionDenoiseCompute<>::RSCF_SVGFCopyAux,
                        KernelType>>(
                        {ctx.camera_info_ray_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                        ctx.camera_info_ray_layout
                    );
                }
            }

            // Ping-pong all temporal reservoirs at end of frame so the next
            // frame's "prev" sees this frame's data.
            if (k_ptr->vision_is_restir_enabled) {
                exchange(ctx.camera_prev_reservoir_di, ctx.camera_current_reservoir_di);
                exchange(ctx.camera_prev_reservoir_gi, ctx.camera_current_reservoir_gi);
                p_ptr->sync();
                RS_ConstantMemory::write(
                    k_ptr, _device_provider->device, sizeof(RS_Kernel)
                );
            }
        });
    });
}

void RS_SimulationExecutor::_storeRecorderRecords() {
    if (_kernel_manager->getKernelRecordBufferFrameNum() == 0) return;
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            using KernelType = std::remove_pointer_t<decltype(k_ptr)>;
            auto &ctx = k_ptr->ctx;

            /****************************************************************
             * Structure records
             ****************************************************************/
            if (ctx.host_data->structure_record_layout.size() != 0) {
                p_ptr->template run<RSCF_LaunchUpdateRootRecord<KernelType>>(
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
                p_ptr->template run<RSCF_LaunchUpdateRootRecord<KernelType>>(
                    {ctx.camera_records.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_recorder_buffers,
                    ctx.camera_records,
                    ctx.camera_pixel_record_layout,
                    ctx.camera_state_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraPixelRecord, KernelType>>(
                    {ctx.camera_pixel_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_pixel_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraStateRecord, KernelType>>(
                    {ctx.camera_state_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_state_record_layout
                );
            }
        });
    });
}

void RS_SimulationExecutor::_storeControllerRecords() {
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            using KernelType = std::remove_pointer_t<decltype(k_ptr)>;
            auto &ctx = k_ptr->ctx;

            /****************************************************************
             * Structure records
             ****************************************************************/
            if (ctx.host_data->structure_record_layout.size() != 0) {
                p_ptr->template run<RSCF_LaunchUpdateRootRecord<KernelType>>(
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
                p_ptr->template run<RSCF_LaunchUpdateRootRecord<KernelType>>(
                    {ctx.camera_records.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_controller_buffers,
                    ctx.camera_records,
                    ctx.camera_pixel_record_layout,
                    ctx.camera_state_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraPixelRecord, KernelType>>(
                    {ctx.camera_pixel_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_pixel_record_layout
                );

                p_ptr->template run<
                    RSCF_LaunchStoreRecordWithLayout<RSCF_StoreCameraStateRecord, KernelType>>(
                    {ctx.camera_state_record_layout.size(), RS_AUTO_BLOCK_SIZE, false},
                    ctx.camera_records,
                    ctx.camera_state_record_layout
                );
            }
        });
    });
}

void RS_SimulationExecutor::_rebuildStaticVoxelBVHIfNeeded() {
    auto &spatial_manager = _kernel_manager->getKernelSpatialManager();
    if (!spatial_manager.isStaticVoxelModified()) return;

    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            selectAccuracy(_simulation_compute_accuracy, [&](auto acc_const) {
                static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
                using KernelType = std::remove_pointer_t<decltype(k_ptr)>;
                auto &ctx = k_ptr->ctx;

                RSize num_voxels = ctx.structure_voxels.size();
                if (num_voxels == 0) {
                    ctx.static_voxel_bvh.resize(0);
                    spatial_manager.clearStaticVoxelModified();
                    return;
                }

                ensure_vector_size(ctx.structure_voxel_bvh_flags, num_voxels);
                ensure_vector_size(ctx.structure_voxel_bvh_prefix_sums, num_voxels);

                p_ptr->template run<RSCF_LaunchDirect<
                    typename RS_VoxelCompute<ComputeAccuracy>::RSCF_GetStaticVoxelFlag,
                    KernelType>>(
                    {num_voxels, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_voxel_bvh_flags.data,
                    num_voxels
                );

                _scan.template inclusiveSum<RIndex, RIndex>(
                    ctx.structure_voxel_bvh_flags, ctx.structure_voxel_bvh_prefix_sums
                );

                RIndex num_active = 0;
                ctx.structure_voxel_bvh_prefix_sums.readSlice(&num_active, 1, num_voxels - 1, true);

                if (num_active == 0) {
                    ctx.static_voxel_bvh.resize(0);
                    spatial_manager.clearStaticVoxelModified();
                    return;
                }

                ensure_vector_size(ctx.structure_voxel_bvh_active_handles, num_active);

                p_ptr->template run<RSCF_LaunchDirect<
                    typename RS_VoxelCompute<ComputeAccuracy>::RSCF_CompactStaticVoxels,
                    KernelType>>(
                    {num_voxels, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_voxel_bvh_flags.data,
                    num_voxels,
                    ctx.structure_voxel_bvh_prefix_sums.data,
                    ctx.structure_voxel_bvh_active_handles.data
                );

                ctx.static_voxel_bvh.resize(num_active);

                p_ptr->template run<RSCF_LaunchDirect<
                    typename RS_VoxelCompute<ComputeAccuracy>::RSCF_FillVoxelBVHLeafData,
                    KernelType>>(
                    {num_active, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_voxel_bvh_active_handles,
                    ctx.static_voxel_bvh.object_identifiers,
                    ctx.static_voxel_bvh.leaf_bounds
                );

                ctx.static_voxel_bvh.rebuild();

                RS_ConstantMemory::write(k_ptr, _device_provider->device, sizeof(RS_Kernel));

                spatial_manager.clearStaticVoxelModified();
            });
        });
    });
}

void RS_SimulationExecutor::_rebuildDynamicVoxelBVH() {
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            selectAccuracy(_simulation_compute_accuracy, [&](auto acc_const) {
                static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
                using KernelType = std::remove_pointer_t<decltype(k_ptr)>;
                auto &ctx = k_ptr->ctx;

                if (ctx.host_data->structure_voxel_layout.size() == 0) {
                    ctx.dynamic_voxel_bvh.resize(0);
                    return;
                }

                auto voxel_handles = _getLaunchHandles("voxel");
                RSize num_voxels = voxel_handles.size();

                ensure_vector_size(ctx.structure_voxel_bvh_flags, num_voxels);
                ensure_vector_size(ctx.structure_voxel_bvh_prefix_sums, num_voxels);

                p_ptr->template run<RSCF_LaunchSetOrGetWithHandles<
                    typename RS_VoxelCompute<ComputeAccuracy>::RSCF_GetDynamicVoxelFlag,
                    KernelType>>(
                    {num_voxels, RS_AUTO_BLOCK_SIZE, false},
                    voxel_handles,
                    ctx.structure_voxel_bvh_flags.data
                );

                _scan.template inclusiveSum<RIndex, RIndex>(
                    ctx.structure_voxel_bvh_flags, ctx.structure_voxel_bvh_prefix_sums
                );

                RIndex num_active = 0;
                ctx.structure_voxel_bvh_prefix_sums.readSlice(&num_active, 1, num_voxels - 1, true);

                if (num_active == 0) {
                    ctx.dynamic_voxel_bvh.resize(0);
                    return;
                }

                ensure_vector_size(ctx.structure_voxel_bvh_active_handles, num_active);

                p_ptr->template run<RSCF_LaunchSetOrGetWithHandles<
                    typename RS_VoxelCompute<ComputeAccuracy>::RSCF_CompactDynamicVoxels,
                    KernelType>>(
                    {num_voxels, RS_AUTO_BLOCK_SIZE, false},
                    voxel_handles,
                    ctx.structure_voxel_bvh_flags.data,
                    ctx.structure_voxel_bvh_prefix_sums.data,
                    ctx.structure_voxel_bvh_active_handles.data
                );

                ctx.dynamic_voxel_bvh.resize(num_active);

                p_ptr->template run<RSCF_LaunchDirect<
                    typename RS_VoxelCompute<ComputeAccuracy>::RSCF_FillVoxelBVHLeafData,
                    KernelType>>(
                    {num_active, RS_AUTO_BLOCK_SIZE, false},
                    ctx.structure_voxel_bvh_active_handles,
                    ctx.dynamic_voxel_bvh.object_identifiers,
                    ctx.dynamic_voxel_bvh.leaf_bounds
                );

                ctx.dynamic_voxel_bvh.rebuild();

                RS_ConstantMemory::write(k_ptr, _device_provider->device, sizeof(RS_Kernel));
            });
        });
    });
}

void RS_SimulationExecutor::_refitDynamicVoxelBVH() {
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            selectAccuracy(_simulation_compute_accuracy, [&](auto acc_const) {
                static constexpr RSE_FloatType ComputeAccuracy = acc_const.value;
                using KernelType = std::remove_pointer_t<decltype(k_ptr)>;
                auto &ctx = k_ptr->ctx;

                RSize leaf_count = ctx.dynamic_voxel_bvh.object_identifiers.size();
                if (leaf_count == 0) return;

                p_ptr->template run<RSCF_LaunchDirect<
                    typename RS_VoxelCompute<ComputeAccuracy>::RSCF_RefitVoxelBVHLeafData,
                    KernelType>>(
                    {leaf_count, RS_AUTO_BLOCK_SIZE, false},
                    ctx.dynamic_voxel_bvh.object_identifiers,
                    ctx.dynamic_voxel_bvh.leaf_bounds
                );

                ctx.dynamic_voxel_bvh.refit();
            });
        });
    });
}

void RS_SimulationExecutor::_rebuildFloorTriangleBVHIfNeeded() {
    auto &spatial_manager = _kernel_manager->getKernelSpatialManager();
    if (!spatial_manager.isFloorTriangleModified()) return;

    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            using KernelType = std::remove_pointer_t<decltype(k_ptr)>;
            auto &ctx = k_ptr->ctx;

            RSize triangle_count = ctx.mesh_triangles.size();
            if (triangle_count == 0) {
                ctx.floor_triangle_bvh.resize(0);
                spatial_manager.clearFloorTriangleModified();
                return;
            }

            ctx.floor_triangle_bvh.resize(triangle_count);

            p_ptr->template run<RSCF_LaunchDirect<
                RS_MeshTriangleCompute::RSCF_FillTriangleBVHLeafData,
                KernelType>>(
                {triangle_count, RS_AUTO_BLOCK_SIZE, false},
                ctx.mesh_triangles,
                ctx.floor_triangle_bvh.object_identifiers,
                ctx.floor_triangle_bvh.leaf_bounds
            );

            ctx.floor_triangle_bvh.rebuild();

            RS_ConstantMemory::write(k_ptr, _device_provider->device, sizeof(RS_Kernel));

            spatial_manager.clearFloorTriangleModified();
        });
    });
}

template <typename Body>
void RS_SimulationExecutor::_runGuarded(Body &&body) {
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            selectAccuracy(_simulation_compute_accuracy, [&](auto acc_const) {
                // Pause the kernel manager so it cannot mutate kernel data while we launch.
                RS_KernelManagerGuard guard(_kernel_manager);
                bool was_modified = k_ptr->isModified();
                if (was_modified) {
                    p_ptr->sync();
                    RS_ConstantMemory::write(k_ptr, _device_provider->device, sizeof(RS_Kernel));
                    k_ptr->clearModified();
                }
                body(k_ptr, p_ptr, acc_const, was_modified);
            });
        });
    });
}

RS_Array<RIndex, 1> RS_SimulationExecutor::_getLaunchHandles(const std::string &name) {
    RS_Array<RIndex, 1> indices;
    castKernelPtrAndCall(_kernel, [&](auto k_ptr) {
        castProviderPtrAndCall(_device_provider, [&](auto p_ptr) {
            auto &ctx = k_ptr->ctx;
            if ((name == "voxel" || name == "link" || name == "rigid_body" ||
                 name == "rigid_body_surface_voxel" || name == "constraint_joint") &&
                _handles.find(name) == _handles.end()) {
                _handles.insert({name, {{0}, p_ptr}});
                _handle_version_ids[name] = 0;
            }
            if (name == "voxel") {
                if (ctx.host_data->structure_voxel_layout.versionID() !=
                    _handle_version_ids.at(name)) {
                    _handles["voxel"].resize(ctx.host_data->structure_voxel_layout.size(), false);
                    p_ptr->template run<RSCF_ResolveThreadHandleFromLayout>(
                        {ctx.structure_voxel_layout.size()},
                        ctx.structure_voxel_layout,
                        _handles["voxel"]
                    );
                    _handle_version_ids[name] = ctx.host_data->structure_voxel_layout.versionID();
                }
                indices = _handles["voxel"];
            } else if (name == "link") {
                if (ctx.host_data->structure_link_layout.versionID() !=
                    _handle_version_ids.at(name)) {
                    _handles["link"].resize(ctx.host_data->structure_link_layout.size(), false);
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
                    _handles["rigid_body"].resize(
                        ctx.host_data->structure_rigid_body_layout.size(), false
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
            } else if (name == "rigid_body_surface_voxel") {
                if (ctx.host_data->structure_rigid_body_surface_voxel_layout.versionID() !=
                    _handle_version_ids.at(name)) {
                    _handles["rigid_body_surface_voxel"].resize(
                        ctx.host_data->structure_rigid_body_surface_voxel_layout.size(), false
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
                    _handles["constraint_joint"].resize(
                        ctx.host_data->structure_constraint_joint_layout.size(), false
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

void RS_SimulationExecutor::_resizeReduceBufferIfNeeded(RSize needed_size) {
    if (_reduce_input_buffer.mem_provider == nullptr || _reduce_input_buffer.size() < needed_size) {
        // Just allocate same amount of memory for both input and output, simpler
        // Output size is always smaller or equal to input size
        _reduce_input_buffer.free();
        _reduce_output_buffer.free();

        // Will allocate even if its 0 (then it's associated with mem provider)
        _reduce_input_buffer = _device_provider->allocate(needed_size, true);
        _reduce_output_buffer = _device_provider->allocate(needed_size, true);
    }
}

}  // namespace rise
