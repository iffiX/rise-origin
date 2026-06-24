// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_MANAGER_H
#define RS_KERNEL_MANAGER_H

#include <spdlog/spdlog.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <queue>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kernel_manager/rs_kernel_camera_perspective_manager.h"
#include "kernel_manager/rs_kernel_cosmic_background_manager.h"
#include "kernel_manager/rs_kernel_floor_manager.h"
#include "kernel_manager/rs_kernel_force_manager.h"
#include "kernel_manager/rs_kernel_gid_registration.h"
#include "kernel_manager/rs_kernel_intersect_region_manager.h"
#include "kernel_manager/rs_kernel_material_manager.h"
#include "kernel_manager/rs_kernel_spatial_manager.h"
#include "kernel_manager/rs_kernel_structure_voxel_manager.h"
#include "kernel_manager/rs_kernel_texture_manager.h"
#include "kernel_manager/rs_kernel_torque_manager.h"
#include "rise_origin/core/rs_world_controller.h"
#include "rise_origin/core/rs_world_handle.h"
#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/rsc/rsc_world.h"
#include "rise_origin/utils/data/rs_any.h"
#include "rise_origin/utils/memory/rs_memory.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_logging.h"
#include "rs_kernel.h"

namespace rise {

/**
 * @brief Which manager kind a staged commit targets; selects the map the flush dispatches to.
 */
RS_ENUM(
    RSE_KernelManagerType,
    RIndex,
    (RSE_KERNEL_MANAGER_STRUCTURE_VOXEL, 0),
    (RSE_KERNEL_MANAGER_CAMERA_PERSPECTIVE, 1),
    (RSE_KERNEL_MANAGER_FLOOR, 2),
    (RSE_KERNEL_MANAGER_COSMIC_BACKGROUND, 3),
    (RSE_KERNEL_MANAGER_FORCE, 4),
    (RSE_KERNEL_MANAGER_TORQUE, 5),
    (RSE_KERNEL_MANAGER_INTERSECT_REGION, 6)
)

/**
 * @brief Per-controller I/O bundle assembled by the kernel manager (keyed by name).
 */
struct RS_KernelControllerBundle {
    std::vector<RS_WorldRecordVariant> records;
    std::vector<RS_WorldSignalVariant> signals;
};

/**
 * @brief Per-recorder structure-voxel record bundle (keyed by recorder name).
 */
struct RS_KernelRecorderStructureVoxelBundle {
    std::unordered_map<std::string, std::vector<RS_WorldStructureVoxelRecord>> structure_records;
};

/**
 * @brief Per-recorder camera-perspective record bundle (keyed by recorder name).
 */
struct RS_KernelRecorderCameraPerspectiveBundle {
    std::unordered_map<std::string, std::vector<RS_WorldCameraPerspectiveRecord>> camera_records;
};

/**
 * @brief Per-recorder intersect-region record bundle (keyed by recorder name).
 */
struct RS_KernelRecorderIntersectRegionBundle {
    std::unordered_map<std::string, std::vector<RS_WorldIntersectRegionRecord>>
        intersect_region_records;
};

/**
 * @brief A built manager staged for commit at the transaction flush. Skipped if its manager has
 * since been removed from its map.
 */
struct RS_KernelPendingCommit {
    RSE_KernelManagerType type;
    std::string name;
};

class RS_KernelManager {
   public:
    std::atomic<bool> is_kernel_inited = {false};
    RS_Kernel kernel;

    RS_KernelManager();

    void init(
        RS_MemoryProvider *host_provider,
        RS_MemoryProvider *device_provider,
        RSize record_buffer_frame_num,
        std::function<void(const RS_WorldMessage &)> message_sink = {}
    );
    void free();
    void initKernel(RSE_FloatType context_mem_accuracy);
    void freeKernel();

    /**
     * @brief Pause/resume command execution. Implies critical data access; take either this or
     * critical data access, never both.
     */
    void pauseCommandExecution();
    void resumeCommandExecution();

    /**
     * @brief Acquire/release exclusive access to the kernel data (where the context lives).
     */
    void acquireCriticalDataAccess();
    void releaseCriticalDataAccess();

    /*****************************************************************
     * Kernel Management APIs
     *****************************************************************/
    /**
     * @brief Mark the host kernel as modified so the next sync pushes it to the device.
     */
    void markKernelModified();

    /**
     * @brief Whether the host kernel has been modified since the last sync.
     */
    [[nodiscard]] bool isKernelModified() const;

    /**
     * @brief If modified, drain the device stream and push the kernel struct to device, clearing
     * the modified flag.
     */
    void syncKernelIfNeeded();

    /*****************************************************************
     * Simulation execution APIs
     *****************************************************************/
    void syncRecorderRecords();
    void waitForRecorderRecordsSync();
    std::unordered_map<std::string, RS_KernelRecorderStructureVoxelBundle>
    getRecorderStructureVoxelRecords();
    std::unordered_map<std::string, RS_KernelRecorderCameraPerspectiveBundle>
    getRecorderCameraPerspectiveRecords();
    std::unordered_map<std::string, RS_KernelRecorderIntersectRegionBundle>
    getRecorderIntersectRegionRecords();

    void syncControllerRecords();
    void syncControllerSignals();

    /**
     * @brief Collect every controller's records and signals, keyed by controller name, including
     * an empty bundle for each registered name so idle controllers still appear.
     */
    std::unordered_map<std::string, RS_KernelControllerBundle> getControllerSignalsAndRecords(
        const std::unordered_set<std::string> &controller_names
    );

    /**
     * @brief Route the record-gid-major unique membership pairs to the owning region managers.
     * A single host scan finds each record's contiguous run and hands it to that manager.
     */
    void setIntersectRegionRecords(const std::vector<logic::RS_IntersectRegionPair> &pairs);

    /**
     * @brief Synchronously remove a structure and every dependent signal, controller,
     * and camera etc.
     */
    std::string purgeStructure(RIndex structure_gid);

    /**
     * @brief Mark the ReSTIR reservoir history stale after the emissive light set changed, so the
     * next resample skips reuse. Sets the kernel flag and marks the kernel modified.
     */
    void markVisionRestirReservoirStale();

    /**
     * @brief Whether the ReSTIR reservoir history is stale and reuse must be skipped this frame.
     */
    [[nodiscard]] bool isVisionRestirReservoirStale() const;

    /**
     * @brief Clear the ReSTIR reservoir stale flag once a fresh-only resample has rebuilt the
     * history. Sets the kernel flag and marks the kernel modified.
     */
    void clearVisionRestirReservoirStale();

    /**
     * @brief Mark the voxel partition modified so the next step rebuilds the dynamic voxel BVH.
     */
    void markVoxelPartitionModified();

    /**
     * @brief Whether the voxel partition changed since the last dynamic voxel BVH rebuild.
     */
    [[nodiscard]] bool isVoxelPartitionModified() const;

    /**
     * @brief Clear the voxel partition modified flag after a dynamic voxel BVH rebuild.
     */
    void clearVoxelPartitionModified();

    /**
     * @brief Mark the floor triangles modified so the next step rebuilds the floor triangle BVH.
     */
    void markFloorTriangleModified();

    /**
     * @brief Whether the floor triangles changed since the last floor triangle BVH rebuild.
     */
    [[nodiscard]] bool isFloorTriangleModified() const;

    /**
     * @brief Clear the floor triangle modified flag after a floor triangle BVH rebuild.
     */
    void clearFloorTriangleModified();

    /*****************************************************************
     * Handle APIs
     *****************************************************************/
    void pauseKernel(bool sync = false);
    void resumeKernel(bool sync = false);
    void terminateKernel(bool sync = false);

    void beginModificationTransaction(bool sync = false);
    void endModificationTransaction(bool sync = false);

    void setSimulationRSC(
        const rsc::RSC_Simulation &simulation,
        RIndex message_id = RS_NULL_INDEX,
        bool sync = false
    );
    const rsc::RSC_Simulation &getSimulationRSC() const;

    void addResource(
        const rsc::RSC_ResourceVariant &resource,
        RIndex message_id = RS_NULL_INDEX,
        bool sync = false
    );
    void updateResource(
        const std::string &name,
        const rsc::RSC_ResourceVariant &resource,
        RIndex message_id = RS_NULL_INDEX,
        bool sync = false
    );
    void
    removeResource(const std::string &name, RIndex message_id = RS_NULL_INDEX, bool sync = false);
    rsc::RSC_ResourceVariant findResource(const std::string &name);

    void addEssence(
        const rsc::RSC_EssenceVariant &essence,
        RIndex message_id = RS_NULL_INDEX,
        bool sync = false
    );
    void updateEssence(
        const std::string &name,
        const rsc::RSC_EssenceVariant &essence,
        RIndex message_id = RS_NULL_INDEX,
        bool sync = false
    );
    void
    removeEssence(const std::string &name, RIndex message_id = RS_NULL_INDEX, bool sync = false);
    rsc::RSC_EssenceVariant findEssence(const std::string &name);

    void addEntity(
        const rsc::RSC_EntityVariant &entity,
        RIndex message_id = RS_NULL_INDEX,
        bool sync = false
    );
    void
    removeEntity(const std::string &name, RIndex message_id = RS_NULL_INDEX, bool sync = false);
    void
    freezeEntity(const std::string &name, RIndex message_id = RS_NULL_INDEX, bool sync = false);
    void
    unfreezeEntity(const std::string &name, RIndex message_id = RS_NULL_INDEX, bool sync = false);
    bool isEntityFrozen(const std::string &name);
    rsc::RSC_EntityVariant findEntity(const std::string &name);

    void addRecord(
        const rsc::RSC_RecordVariant &record,
        RIndex message_id = RS_NULL_INDEX,
        bool sync = false
    );
    void
    removeRecord(const std::string &name, RIndex message_id = RS_NULL_INDEX, bool sync = false);
    rsc::RSC_RecordVariant findRecord(const std::string &name);

    void addSignal(
        const rsc::RSC_SignalVariant &signal,
        RIndex message_id = RS_NULL_INDEX,
        bool sync = false
    );
    void
    removeSignal(const std::string &name, RIndex message_id = RS_NULL_INDEX, bool sync = false);
    rsc::RSC_SignalVariant findSignal(const std::string &name);

    rsc::RSC_World getWorld();

    /****************************************************************
     * Data manager APIs
     ****************************************************************/
    RS_MemoryProvider *getDeviceProvider() const;
    RS_MemoryProvider *getHostProvider() const;
    std::shared_ptr<logging::logger> getLogger() const;
    std::shared_ptr<RS_ThreadPool> getThreadPool() const;
    RSize getKernelRecordBufferFrameNum() const;

    // Counters / gid allocators
    RIndex getEntityCounter() const;
    RIndex getEssenceCounter() const;
    RIndex getResourceCounter() const;
    RIndex getRecordCounter() const;
    RIndex getSignalCounter() const;

    RIndex registerEntity(const std::string &name);
    RIndex registerEssence(const std::string &name);
    RIndex registerResource(const std::string &name);
    RIndex registerRecord(const std::string &name);
    RIndex registerSignal(const std::string &name);

    void deregisterEntity(const std::string &name);
    void deregisterEssence(const std::string &name);
    void deregisterResource(const std::string &name);
    void deregisterRecord(const std::string &name);
    void deregisterSignal(const std::string &name);

    // Kernel data managers — singletons on top, name-keyed maps below.
    RS_KernelSpatialManager &getKernelSpatialManager();
    RS_KernelTextureManager &getKernelTextureManager();
    RS_KernelMaterialManager &getKernelMaterialManager();
    const std::unique_ptr<RS_KernelCosmicBackgroundManager> &
    getKernelCosmicBackgroundManager() const;
    const std::unique_ptr<RS_KernelFloorManager> &getKernelFloorManager() const;

    std::unordered_map<std::string, std::unique_ptr<RS_KernelStructureVoxelManager>> &
    getKernelStructureVoxelManagers();
    std::unordered_map<std::string, std::unique_ptr<RS_KernelCameraPerspectiveManager>> &
    getKernelCameraPerspectiveManagers();
    std::unordered_map<std::string, std::unique_ptr<RS_KernelForceManager>> &
    getKernelForceManagers();
    std::unordered_map<std::string, std::unique_ptr<RS_KernelTorqueManager>> &
    getKernelTorqueManagers();
    std::unordered_map<std::string, std::unique_ptr<RS_KernelIntersectRegionManager>> &
    getKernelIntersectRegionManagers();

    /**
     * @brief Whether a kernel context critical zone is currently entered but not exited.
     */
    bool isInKernelDataCriticalZone() const;

    /**
     * @brief Mark the start of a non-local kernel context mutation.
     * @details A sub-manager calls this immediately before its first ctx mutation and
     * exitKernelDataCriticalZone() immediately after the last; exit is only reached on success, so
     * an exception escaping the bracket leaves the depth nonzero, signalling an unrecoverable
     * failure. The depth counter supports nested ctx regions.
     */
    void enterKernelDataCriticalZone();

    /**
     * @brief Exit the entered kernel context critical zone.
     */
    void exitKernelDataCriticalZone();

   private:
    RS_MemoryProvider *_device_provider = nullptr;
    RS_MemoryProvider *_host_provider = nullptr;
    std::shared_ptr<logging::logger> _logger;
    std::shared_ptr<RS_ThreadPool> _thread_pool;
    std::unique_ptr<std::thread> _kernel_synchronization_thread;
    std::queue<std::tuple<RIndex, std::string, RS_Any>> _command_queue;
    std::mutex _command_queue_mutex;
    std::condition_variable _command_queue_cv;

    std::recursive_mutex _command_execution_mutex;

    std::recursive_mutex _kernel_data_mutex;

    bool _is_transaction_open = false;
    std::vector<RS_KernelPendingCommit> _pending_commits;

    std::unordered_map<std::string, std::function<void(RS_Any, RIndex)>> _methods;

    std::function<void(const RS_WorldMessage &)> _message_sink;

    /// Messages queued by the running command, drained to the sink once its commit is done.
    std::vector<RS_WorldMessage> _pending_messages;

    // Whether the host kernel has been modified since the last syncKernelIfNeeded().
    bool _is_kernel_modified = false;

    RSize _kernel_data_critical_zone_depth = 0;

    RSize _kernel_record_buffer_frame_num = 0;

    /// Active simulation sub-characterization; kernel constants are derived from this.
    rsc::RSC_Simulation _simulation_rsc;

    /// Per-axis gid counters for device-side management.
    RIndex _entity_counter = 0;
    RIndex _essence_counter = 0;
    RIndex _resource_counter = 0;
    RIndex _record_counter = 0;
    RIndex _signal_counter = 0;

    std::unordered_map<std::string, RIndex> _entity_gids;
    std::unordered_map<std::string, RIndex> _essence_gids;
    std::unordered_map<std::string, RIndex> _resource_gids;
    std::unordered_map<std::string, RIndex> _record_gids;
    std::unordered_map<std::string, RIndex> _signal_gids;

    /// Kernel data managers — singletons.
    RS_KernelSpatialManager _kernel_spatial_manager;
    RS_KernelTextureManager _kernel_texture_manager;
    RS_KernelMaterialManager _kernel_material_manager;
    std::unique_ptr<RS_KernelCosmicBackgroundManager> _kernel_cosmic_background_manager;
    std::unique_ptr<RS_KernelFloorManager> _kernel_floor_manager;

    /// Kernel data managers — name-keyed.
    std::unordered_map<std::string, std::unique_ptr<RS_KernelStructureVoxelManager>>
        _kernel_structure_voxel_managers;
    std::unordered_map<std::string, std::unique_ptr<RS_KernelCameraPerspectiveManager>>
        _kernel_camera_perspective_managers;
    std::unordered_map<std::string, std::unique_ptr<RS_KernelForceManager>> _kernel_force_managers;
    std::unordered_map<std::string, std::unique_ptr<RS_KernelTorqueManager>>
        _kernel_torque_managers;
    std::unordered_map<std::string, std::unique_ptr<RS_KernelIntersectRegionManager>>
        _kernel_intersect_region_managers;

    void _kernelSynchronizationThreadFunc();

    void _pauseKernel();
    void _resumeKernel();
    void _terminateKernel();

    void _beginModificationTransaction();
    void _endModificationTransaction();

    void _configureLayoutGrowthPolicies(RS_ContextHostData *host_data);

    void _setSimulationRSC(const rsc::RSC_Simulation &simulation, RIndex message_id);

    void _addResource(const rsc::RSC_ResourceVariant &resource, RIndex message_id);
    void _updateResource(
        const std::string &name,
        const rsc::RSC_ResourceVariant &resource,
        RIndex message_id
    );
    void _removeResource(const std::string &name, RIndex message_id);

    void _addEssence(const rsc::RSC_EssenceVariant &essence, RIndex message_id);
    void _updateEssence(
        const std::string &name,
        const rsc::RSC_EssenceVariant &essence,
        RIndex message_id
    );
    void _removeEssence(const std::string &name, RIndex message_id);

    void _addEntity(const rsc::RSC_EntityVariant &entity, RIndex message_id);
    void _removeEntity(const std::string &name, RIndex message_id);
    void _freezeEntity(const std::string &name, RIndex message_id);
    void _unfreezeEntity(const std::string &name, RIndex message_id);

    void _addRecord(const rsc::RSC_RecordVariant &record, RIndex message_id);
    void _removeRecord(const std::string &name, RIndex message_id);

    void _addSignal(const rsc::RSC_SignalVariant &signal, RIndex message_id);
    void _removeSignal(const std::string &name, RIndex message_id);

    /**
     * @brief Stage a built manager for commit; commits immediately when no transaction is open.
     */
    void _stageCommit(RSE_KernelManagerType type, const std::string &name);

    /**
     * @brief Drop a staged commit for a manager that was removed before the flush.
     */
    void _retractPendingCommit(RSE_KernelManagerType type, const std::string &name);

    /**
     * @brief Flush all staged commits to the kernel data.
     */
    void _flushPendingCommits();

    /**
     * @brief Flush the staged commit for the given manager to the kernel data if it is still
     * present in its map.
     */
    void _flushOneCommit(RSE_KernelManagerType type, const std::string &name);

    /**
     * @brief Queue an info message for the running command; skipped when message_id is null.
     */
    void _queueInfo(RIndex message_id, RS_WorldMessageData data);
};

/**
 * @brief RAII guard that pauses command execution for its lifetime.
 */
class RS_KernelManagerCommandExecutionGuard {
   public:
    RS_KernelManagerCommandExecutionGuard(RS_KernelManager *kernel_manager)
        : _kernel_manager(kernel_manager) {
        kernel_manager->pauseCommandExecution();
    }

    ~RS_KernelManagerCommandExecutionGuard() { _kernel_manager->resumeCommandExecution(); }

   private:
    RS_KernelManager *_kernel_manager;
};

/**
 * @brief RAII guard that holds exclusive kernel data access for its lifetime.
 */
class RS_KernelManagerCriticalDataAccessGuard {
   public:
    RS_KernelManagerCriticalDataAccessGuard(RS_KernelManager *kernel_manager)
        : _kernel_manager(kernel_manager) {
        kernel_manager->acquireCriticalDataAccess();
    }

    ~RS_KernelManagerCriticalDataAccessGuard() { _kernel_manager->releaseCriticalDataAccess(); }

   private:
    RS_KernelManager *_kernel_manager;
};

}  // namespace rise

#endif  // RS_KERNEL_MANAGER_H
