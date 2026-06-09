// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_EXECUTOR_H
#define RS_SIMULATION_EXECUTOR_H

#include <spdlog/spdlog.h>

#include "rise_origin/core/rs_simulation_callback.h"
#include "rise_origin/device/algorithm/rs_reduce.h"
#include "rise_origin/device/algorithm/rs_scan.h"
#include "rise_origin/device/algorithm/rs_sort.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rs_kernel.h"
#include "rs_kernel_manager.h"

namespace rise {

class RS_SimulationExecutor {
   public:
    RS_SimulationExecutor(
        RS_MemoryProvider *device_provider,
        RS_MemoryProvider *host_provider,
        RS_SimulationCallback callback
    );

    void init(RIndex id, RS_Kernel *kernel, RS_KernelManager *kernel_manager);
    void free();

    bool step();

   private:
    RS_MemoryProvider *_device_provider = nullptr;
    RS_MemoryProvider *_host_provider = nullptr;
    RS_SimulationCallback _callback;
    RSize _dt_update_interval = RS_MAX_SIZE;
    RSize _rigidbody_update_interval = RS_MAX_SIZE;
    RSize _contact_update_interval = RS_MAX_SIZE;
    RSize _contact_bvh_rebuild_sub_interval = RS_MAX_SIZE;
    RSize _contact_bvh_refit_sub_interval = RS_MAX_SIZE;
    RSize _mechanical_signal_update_interval = RS_MAX_SIZE;
    RSize _divergence_check_interval = RS_MAX_SIZE;
    RSE_FloatType _simulation_compute_accuracy = RSE_FLOAT_32;
    std::shared_ptr<spdlog::logger> _logger;

    // Kernel
    RIndex _id = RS_NULL_INDEX;
    RS_Kernel *_kernel = nullptr;
    RS_KernelManager *_kernel_manager = nullptr;

    // Handles
    std::unordered_map<std::string, device::RS_Array<RIndex, 1>> _handles;
    std::unordered_map<std::string, uint64_t> _handle_version_ids;

    // Reduce setup
    RS_MemoryPtr _reduce_input_buffer;
    RS_MemoryPtr _reduce_output_buffer;
    device::RS_Reduce _reduce;
    device::RS_ReduceByGroup _reduce_by_group;

    // Prefix sum setup
    device::RS_Scan _scan;

    // Sort setup
    device::RS_RadixSortPairs _sort;

    // step() phases. Each pauses the kernel manager and re-syncs the device kernel if the host
    // marked it modified, then launches its kernels (no shared outer lock). _run* phases route
    // through _runGuarded; the manager-touching _update* phases take the guard themselves.
    [[nodiscard]] bool _runDtUpdate();
    void _runSoftbodyUpdate();
    void _runRigidbodyCoupling();
    void _runRigidbodyUpdate();
    void _advanceTime();
    void _runRecordSaving();
    void _runControllerRecordSaving();
    [[nodiscard]] bool _runDivergenceCheck();
    void _runContactUpdate();
    void _runMechanicalSignalUpdate();
    void _performCallback();

    // Launch helpers invoked inside a guarded phase (no guard of their own).
    void _updateLightAndVision();
    void _storeRecorderRecords();
    void _storeControllerRecords();
    void _rebuildStaticVoxelBVHIfNeeded();
    void _rebuildDynamicVoxelBVH();
    void _refitDynamicVoxelBVH();
    void _rebuildFloorTriangleBVHIfNeeded();

    // Common utilities.
    template <typename Body>
    void _runGuarded(Body &&body);
    device::RS_Array<RHandle, 1> _getLaunchHandles(const std::string &name);
    void _resizeReduceBufferIfNeeded(RSize needed_size);

    // void _updateMetrics();
};

}  // namespace rise

#endif  // RS_SIMULATION_EXECUTOR_H
