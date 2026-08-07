// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_EXECUTOR_H
#define RS_SIMULATION_EXECUTOR_H

#include <spdlog/spdlog.h>

#include <chrono>

#include "rise_origin/core/rs_world_callback.h"
#include "rise_origin/device/algorithm/rs_reduce.h"
#include "rise_origin/device/algorithm/rs_scan.h"
#include "rise_origin/device/algorithm/rs_sort.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/rsc/rsc_simulation.h"
#include "rs_kernel.h"
#include "rs_kernel_manager.h"

namespace rise {

class RS_Simulation;

class RS_SimulationExecutor {
   public:
    /**
     * @brief Construct the executor with the device and host memory providers and the world
     * callback.
     */
    RS_SimulationExecutor(
        RS_MemoryProvider *device_provider,
        RS_MemoryProvider *host_provider,
        RS_WorldCallback *callback
    );

    /**
     * @brief Snapshot the simulation characterization and set up the kernel and per-run state.
     */
    void init(RS_Simulation *simulation);

    /**
     * @brief Release all per-run state acquired in init.
     */
    void free();

    /**
     * @brief Advance the world by one base time step, running dynamics, sensing, and callbacks.
     */
    bool step();

    /**
     * @brief Run one tick with time and dynamics frozen while sensing and callbacks keep running.
     */
    void stepPaused();

   private:
    RS_MemoryProvider *_device_provider = nullptr;
    RS_MemoryProvider *_host_provider = nullptr;
    RS_WorldCallback *_callback = nullptr;

    // Cached static execution settings, snapshotted at init. Later user edits to the simulation
    // characterization do not change how this run executes.
    rsc::RSC_SimulationTime _time;
    RSize _dt_update_interval = RS_MAX_SIZE;
    RSize _contact_bvh_rebuild_sub_interval = RS_MAX_SIZE;
    RSize _contact_bvh_refit_sub_interval = RS_MAX_SIZE;
    bool _sentinel_enabled = true;
    RSE_FloatType _simulation_compute_accuracy = RSE_FLOAT_32;

    // Dynamic per-subsystem run intervals (base-step counts), recomputed by _runDtUpdate whenever
    // the global time step is recomputed. interval_i = max(1, floor(needed_i / time_step)).
    bool _has_computed_dt = false;
    RSize _softbody_and_rigidbody_update_interval = 1;
    RSize _rigidbody_contact_and_constraint_update_interval = 1;
    RSize _contact_update_interval = 1;
    RSize _mechanical_signal_update_interval = 1;
    RSize _sentinel_update_interval = 1;

    // World load: smoothed wall-clock time spent per second of advanced sim time, measured over the
    // interval between two control callbacks.
    bool _is_world_load_measured = false;
    bool _has_world_load = false;
    RBigFloat _world_load = 0;
    std::chrono::steady_clock::time_point _world_load_measured_wallclock_time;
    RBigFloat _world_load_measured_sim_time = 0;

    // Wall-clock pacing anchor.
    bool _is_wallclock_anchored = false;
    std::chrono::steady_clock::time_point _wallclock_anchor_time;
    RBigFloat _wallclock_anchor_sim_time = 0;

    std::shared_ptr<spdlog::logger> _logger;

    // Kernel
    RS_Simulation *_simulation = nullptr;
    RS_Kernel *_kernel = nullptr;
    RS_KernelManager *_kernel_manager = nullptr;

    // Handles
    std::unordered_map<std::string, device::RS_Vector<RHandle>> _handles;
    std::unordered_map<std::string, uint64_t> _handle_version_ids;

    // Set on every voxel-partition recompute, cleared when the static voxel BVH is rebuilt.
    bool _is_static_contact_bvh_dirty = true;
    bool _is_static_ray_bvh_dirty = true;

    // Reduce setup
    RS_MemoryPtr _reduce_input_buffer;
    RS_MemoryPtr _reduce_output_buffer;
    device::RS_Reduce _reduce;
    device::RS_ReduceByGroup _reduce_by_group;

    // Prefix sum setup
    device::RS_Scan _scan;
    device::RS_Vector<RIndex> _prefix_sum_flags_0;
    device::RS_Vector<RIndex> _prefix_sum_flags_1;
    device::RS_Vector<RIndex> _prefix_sum_result_0;

    // Sort setup
    device::RS_RadixSortPairs _sort;
    device::RS_SortPairs _merge_sort;

    // step() phases. Physics/data phases route through _runWithCriticalDataAccessGuard; the
    // callback and sentinel route through _runWithCommandExecutionGuard.
    RFloat _runDtUpdate(bool &is_dt_normal);
    RSize _runSentinel(bool &is_hard_abort);
    void _runAdvanceTime();
    void _runSoftbodyUpdate();
    void _runRigidbodyCoupling();
    void _runRigidbodyContactAndConstraintUpdate();
    void _runRecordUpdate();
    void _runRecorderRecordSaving();
    void _runControllerRecordSaving();
    void _runContactUpdate();
    void _runMechanicalSignalDynamicsUpdate();
    bool _runMechanicalSignalTransformUpdate();
    void _performCallback();
    void _measureWorldLoad();
    void _syncExecutorInfo();
    void _paceToWallclock();

    // Launch helpers invoked inside a guarded phase (no guard of their own).
    void _updateVisionAndRay();
    void _updateIntersectRegionRecord();
    void _storeRecorderRecords();
    void _storeControllerRecords();
    void _rebuildStaticVoxelContactBVHIfNeeded();
    void _rebuildStaticVoxelRayBVHIfNeeded();
    void _rebuildDynamicVoxelContactBVH();
    void _rebuildDynamicVoxelRayBVH();
    void _refitDynamicVoxelContactBVH();
    void _refitDynamicVoxelRayBVH();
    void _rebuildFloorTriangleBVHIfNeeded();
    template <typename VoxelBVH>
    void _buildVoxelBVH(VoxelBVH &bvh, device::RS_Vector<RHandle> &handles);
    template <typename VoxelBVH>
    void _refitVoxelBVH(VoxelBVH &bvh);

    // Common utilities.
    template <typename Body>
    void _runWithCriticalDataAccessGuard(Body &&body);
    template <typename Body>
    void _runWithCommandExecutionGuard(Body &&body);
    device::RS_Vector<RHandle> _getLaunchHandles(const std::string &name);
    bool _resolveVoxelMobilityPartitionIfNeeded();
    template <typename ActiveFlagsComputeFunction>
    void _resolveVoxelActivePartition(const char *static_key, const char *dynamic_key);
    void _resolveVoxelContactActivePartition();
    void _resolveVoxelRayActivePartition();
    bool _resolveRigidBodyMobilityPartitionIfNeeded();
    bool _resolveRigidBodyBoundVoxelSurfacePartitionIfNeeded();
    void _resizeReduceBufferIfNeeded(RBigSize needed_input_size, RBigSize needed_output_size);

    /**
     * @brief Base-step count a subsystem runs on: max(1, floor(needed / time_step)).
     * @details Clamped to keep the cast finite for unbounded (absent) subsystems.
     */
    static RSize _snapInterval(RFloat needed, RFloat time_step);
};

}  // namespace rise

#endif  // RS_SIMULATION_EXECUTOR_H
