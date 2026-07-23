// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_MANAGER_H
#define RS_KERNEL_MANAGER_H

#include <spdlog/spdlog.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kernel_manager/rs_kernel_camera_perspective_manager.h"
#include "kernel_manager/rs_kernel_camera_stereo_manager.h"
#include "kernel_manager/rs_kernel_cosmic_background_manager.h"
#include "kernel_manager/rs_kernel_floor_manager.h"
#include "kernel_manager/rs_kernel_force_manager.h"
#include "kernel_manager/rs_kernel_gid_registration.h"
#include "kernel_manager/rs_kernel_intersect_region_manager.h"
#include "kernel_manager/rs_kernel_material_manager.h"
#include "kernel_manager/rs_kernel_ray_sensor_manager.h"
#include "kernel_manager/rs_kernel_spatial_manager.h"
#include "kernel_manager/rs_kernel_structure_voxel_manager.h"
#include "kernel_manager/rs_kernel_texture_manager.h"
#include "kernel_manager/rs_kernel_torque_manager.h"
#include "kernel_manager/rs_kernel_transform_manager.h"
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
    (RSE_KERNEL_MANAGER_CAMERA_STEREO, 2),
    (RSE_KERNEL_MANAGER_RAY_SENSOR, 3),
    (RSE_KERNEL_MANAGER_FLOOR, 4),
    (RSE_KERNEL_MANAGER_COSMIC_BACKGROUND, 5),
    (RSE_KERNEL_MANAGER_FORCE, 6),
    (RSE_KERNEL_MANAGER_TORQUE, 7),
    (RSE_KERNEL_MANAGER_TRANSFORM, 8),
    (RSE_KERNEL_MANAGER_INTERSECT_REGION, 9)
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
 * @brief Per-recorder camera-stereo record bundle (keyed by recorder name).
 */
struct RS_KernelRecorderCameraStereoBundle {
    std::unordered_map<std::string, std::vector<RS_WorldCameraStereoRecord>> camera_records;
};

/**
 * @brief Per-recorder ray-sensor record bundle (keyed by recorder name).
 */
struct RS_KernelRecorderRaySensorBundle {
    std::unordered_map<std::string, std::vector<RS_WorldRaySensorRecord>> ray_sensor_records;
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
        std::function<void(RSE_WorldMessageLevel, const std::string &)> message_sink = {}
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
    std::unordered_map<std::string, RS_KernelRecorderCameraStereoBundle>
    getRecorderCameraStereoRecords();
    std::unordered_map<std::string, RS_KernelRecorderRaySensorBundle> getRecorderRaySensorRecords();
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
     * @brief After transform signals are applied, raise the static voxel moved flag for any that
     *        moved static voxels and reset each transform host signal to identity.
     */
    void finalizeTransformSignals();

    /**
     * @brief Route the record-gid-major unique membership pairs to the owning region managers.
     * A single host scan finds each record's contiguous run and hands it to that manager.
     */
    void setIntersectRegionRecords(const std::vector<logic::RS_IntersectRegionPair> &pairs);

    /**
     * @brief Gids of every structure that can move a voxel, rebuilt only when the structure set
     * changes. The sentinel reads it to pick each structure's world range.
     */
    const std::unordered_set<RIndex> &getDynamicStructureGIDs();

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
     * @brief Mark the rigid body partition modified so the next step rebuilds the dynamic subset.
     */
    void markRigidBodyPartitionModified();

    /**
     * @brief Whether the rigid body partition changed since the last dynamic subset rebuild.
     */
    [[nodiscard]] bool isRigidBodyPartitionModified() const;

    /**
     * @brief Clear the rigid body partition modified flag after a dynamic subset rebuild.
     */
    void clearRigidBodyPartitionModified();

    /**
     * @brief Mark static voxels moved so the next step rebuilds the static voxel BVH.
     */
    void markStaticVoxelMoved();

    /**
     * @brief Whether static voxels moved since the last static voxel BVH rebuild.
     */
    [[nodiscard]] bool isStaticVoxelMoved() const;

    /**
     * @brief Clear the static voxel moved flag after a static voxel BVH rebuild.
     */
    void clearStaticVoxelMoved();

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
    std::string getInfo(const std::string &info_name);

    void pauseKernel(bool sync = false);
    void resumeKernel(bool sync = false);
    void terminateKernel(bool sync = false);

    void beginModificationTransaction(bool sync = false);
    void endModificationTransaction(bool sync = false);

    RS_WorldTask setSimulationRSC(const rsc::RSC_Simulation &simulation, bool sync = false);
    const rsc::RSC_Simulation &getSimulationRSC() const;

    RS_WorldTask addResource(const rsc::RSC_ResourceVariant &resource, bool sync = false);
    RS_WorldTask updateResource(
        const std::string &name,
        const rsc::RSC_ResourceVariant &resource,
        bool sync = false
    );
    RS_WorldTask removeResource(const std::string &name, bool sync = false);
    rsc::RSC_ResourceVariant findResource(const std::string &name);

    RS_WorldTask addEssence(const rsc::RSC_EssenceVariant &essence, bool sync = false);
    RS_WorldTask updateEssence(
        const std::string &name,
        const rsc::RSC_EssenceVariant &essence,
        bool sync = false
    );
    RS_WorldTask removeEssence(const std::string &name, bool sync = false);
    rsc::RSC_EssenceVariant findEssence(const std::string &name);

    RS_WorldTask addEntity(const rsc::RSC_EntityVariant &entity, bool sync = false);
    RS_WorldTask removeEntity(const std::string &name, bool sync = false);
    RS_WorldTask freezeEntity(const std::string &name, bool sync = false);
    RS_WorldTask unfreezeEntity(const std::string &name, bool sync = false);
    bool isEntityFrozen(const std::string &name);
    rsc::RSC_EntityVariant findEntity(const std::string &name);
    RIndex findEntityGID(const std::string &name);
    std::string findEntityName(RIndex gid);

    RS_WorldTask addRecord(const rsc::RSC_RecordVariant &record, bool sync = false);
    RS_WorldTask removeRecord(const std::string &name, bool sync = false);
    rsc::RSC_RecordVariant findRecord(const std::string &name);

    RS_WorldTask addSignal(const rsc::RSC_SignalVariant &signal, bool sync = false);
    RS_WorldTask removeSignal(const std::string &name, bool sync = false);
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
    std::unordered_map<std::string, std::unique_ptr<RS_KernelCameraStereoManager>> &
    getKernelCameraStereoManagers();
    std::unordered_map<std::string, std::unique_ptr<RS_KernelRaySensorManager>> &
    getKernelRaySensorManagers();
    std::unordered_map<std::string, std::unique_ptr<RS_KernelForceManager>> &
    getKernelForceManagers();
    std::unordered_map<std::string, std::unique_ptr<RS_KernelTorqueManager>> &
    getKernelTorqueManagers();
    std::unordered_map<std::string, std::unique_ptr<RS_KernelTransformManager>> &
    getKernelTransformManagers();
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
    using TaskPromise = std::shared_ptr<std::promise<RS_WorldTaskData>>;

    // RAII scope raising the command execution depth for one command, released on every exit path.
    struct CommandExecutionScope {
        RS_KernelManager *manager;
        CommandExecutionScope(RS_KernelManager *manager) : manager(manager) {
            manager->_acquireCommandExecution();
        }
        ~CommandExecutionScope() { manager->_releaseCommandExecution(); }
    };

    RS_MemoryProvider *_device_provider = nullptr;
    RS_MemoryProvider *_host_provider = nullptr;
    std::shared_ptr<logging::logger> _logger;
    std::shared_ptr<RS_ThreadPool> _thread_pool;
    std::unique_ptr<std::thread> _kernel_synchronization_thread;
    std::queue<std::tuple<TaskPromise, std::string, RS_Any>> _command_queue;
    std::mutex _command_queue_mutex;
    std::condition_variable _command_queue_cv;

    // Finders and the callback raise the pause count; the sync thread and transactions raise the
    // execution depth. Readers wait while a command executes; the sync thread waits while paused.
    std::mutex _command_execution_mutex;
    std::condition_variable _command_execution_cv;
    RSize _command_pause_count = 0;
    RSize _command_execution_depth = 0;

    std::recursive_mutex _kernel_data_mutex;

    bool _is_transaction_open = false;
    std::vector<RS_KernelPendingCommit> _pending_commits;

    std::unordered_map<std::string, std::function<void(RS_Any, TaskPromise)>> _methods;

    /// Message sink where output are given to
    std::function<void(RSE_WorldMessageLevel, const std::string &)> _message_sink;

    /// Task results built by the running command, delivered to their promises once its commit is
    /// done.
    std::vector<std::pair<TaskPromise, RS_WorldTaskData>> _pending_results;

    // Whether the host kernel has been modified since the last syncKernelIfNeeded().
    bool _is_kernel_modified = false;

    RSize _kernel_data_critical_zone_depth = 0;

    RSize _kernel_record_buffer_frame_num = 0;

    /// Active simulation sub-characterization; kernel constants are derived from this.
    rsc::RSC_Simulation _simulation_rsc;

    /// Per-axis gid counters for device-side management.
    RIndex _entity_counter = RS_WORLD_RSC_ENTITY_RESERVED_GID_NUM;
    RIndex _essence_counter = 0;
    RIndex _resource_counter = 0;
    RIndex _record_counter = 0;
    RIndex _signal_counter = 0;

    std::unordered_map<std::string, RIndex> _entity_gids;
    std::unordered_map<RIndex, std::string> _entity_names;
    std::unordered_map<std::string, RIndex> _essence_gids;
    std::unordered_map<std::string, RIndex> _resource_gids;
    std::unordered_map<std::string, RIndex> _record_gids;
    std::unordered_map<std::string, RIndex> _signal_gids;

    /// Cached gids of movable structures, rebuilt from the structure managers when they change.
    bool _is_dynamic_structure_gids_dirty = true;
    std::unordered_set<RIndex> _dynamic_structure_gids;

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
    std::unordered_map<std::string, std::unique_ptr<RS_KernelCameraStereoManager>>
        _kernel_camera_stereo_managers;
    std::unordered_map<std::string, std::unique_ptr<RS_KernelRaySensorManager>>
        _kernel_ray_sensor_managers;
    std::unordered_map<std::string, std::unique_ptr<RS_KernelForceManager>> _kernel_force_managers;
    std::unordered_map<std::string, std::unique_ptr<RS_KernelTorqueManager>>
        _kernel_torque_managers;
    std::unordered_map<std::string, std::unique_ptr<RS_KernelTransformManager>>
        _kernel_transform_managers;
    std::unordered_map<std::string, std::unique_ptr<RS_KernelIntersectRegionManager>>
        _kernel_intersect_region_managers;

    void _kernelSynchronizationThreadFunc();

    RS_WorldTask _enqueueCommand(std::string command, RS_Any arg);

    void _acquireCommandExecution();
    void _releaseCommandExecution();

    void _pauseKernel();
    void _resumeKernel();
    void _terminateKernel();

    void _beginModificationTransaction();
    void _endModificationTransaction();

    void _configureLayoutGrowthPolicies(RS_ContextHostData *host_data);

    void _setSimulationRSC(const rsc::RSC_Simulation &simulation, TaskPromise task);

    void _addResource(const rsc::RSC_ResourceVariant &resource, TaskPromise task);
    void _updateResource(
        const std::string &name,
        const rsc::RSC_ResourceVariant &resource,
        TaskPromise task
    );
    void _removeResource(const std::string &name, TaskPromise task);

    void _addEssence(const rsc::RSC_EssenceVariant &essence, TaskPromise task);
    void _updateEssence(
        const std::string &name,
        const rsc::RSC_EssenceVariant &essence,
        TaskPromise task
    );
    void _removeEssence(const std::string &name, TaskPromise task);

    void _addEntity(const rsc::RSC_EntityVariant &entity, TaskPromise task);
    void _removeEntity(const std::string &name, TaskPromise task);
    void _freezeEntity(const std::string &name, TaskPromise task);
    void _unfreezeEntity(const std::string &name, TaskPromise task);

    void _addRecord(const rsc::RSC_RecordVariant &record, TaskPromise task);
    void _removeRecord(const std::string &name, TaskPromise task);

    void _addSignal(const rsc::RSC_SignalVariant &signal, TaskPromise task);
    void _removeSignal(const std::string &name, TaskPromise task);

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
     * @brief Stage the running command's task result, delivered once its commit is done; skipped
     * when task is null.
     */
    void _queueResult(TaskPromise task, RS_WorldTaskData data);
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
