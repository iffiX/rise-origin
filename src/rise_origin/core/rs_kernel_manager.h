// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_MANAGER_H
#define RS_KERNEL_MANAGER_H

#include <spdlog/spdlog.h>

#include <any>
#include <condition_variable>
#include <map>
#include <queue>
#include <thread>
#include <unordered_map>

#include "kernel_data_manager/rs_kernel_camera_manager.h"
#include "kernel_data_manager/rs_kernel_floor_manager.h"
#include "kernel_data_manager/rs_kernel_force_manager.h"
#include "kernel_data_manager/rs_kernel_material_manager.h"
#include "kernel_data_manager/rs_kernel_spatial_manager.h"
#include "kernel_data_manager/rs_kernel_structure_manager.h"
#include "kernel_data_manager/rs_kernel_structure_record_manager.h"
#include "kernel_data_manager/rs_kernel_torque_manager.h"
#include "kernel_data_manager/rs_kernel_world_manager.h"
#include "rise_origin/core/rs_simulation_controller.h"
#include "rise_origin/core/rs_simulation_signal.h"
#include "rise_origin/rsc/rs_config.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_memory.h"
#include "rs_kernel.h"

namespace rise {

class RS_KernelManager {
   public:
    std::atomic<bool> is_kernel_inited = {false};
    RS_Kernel kernel;

    RS_KernelManager();

    void init(
        RS_MemoryProvider *host_provider,
        RS_MemoryProvider *device_provider,
        RSize record_buffer_frame_num
    );
    void free();
    void initKernel(RIndex kernel_id, RSE_FloatType context_mem_accuracy);
    void freeKernel();

    void pauseCommandExecution();
    void resumeCommandExecution();

    /*****************************************************************
     * Simulation execution APIs
     *****************************************************************/
    void syncRecorderRecords();
    void waitForRecorderRecordsSync();
    std::vector<std::pair<std::string, std::vector<RS_SimulationStructureRecord>>>
    getRecorderStructureRecords();
    std::vector<std::pair<std::string, std::vector<RS_SimulationCameraRecord>>>
    getRecorderCameraRecords();

    void syncControllerRecords();
    void syncControllerSignals();
    std::vector<RS_SimulationStructureController> getStructureControllers();
    std::vector<RS_SimulationCameraController> getCameraControllers();
    std::vector<RS_SimulationForceController> getForceControllers();
    std::vector<RS_SimulationTorqueController> getTorqueControllers();

    /*****************************************************************
     * Handle APIs
     *****************************************************************/
    void pauseKernel(bool sync = false);
    void resumeKernel(bool sync = false);
    void terminateKernel(bool sync = false);

    void beginModificationTransaction(bool sync = false);
    void endModificationTransaction(bool sync = false);
    void setSimulationConfig(const RS_SimulationConfig &config, bool sync = false);
    void setWorld(const RS_WorldConfig &config, bool sync = false);
    void setFloor(const RS_FloorConfig &config, bool sync = false);

    void addMaterial(const RS_MaterialConfig &config, bool sync = false);
    void addRecord(const RS_RecordConfig &config, bool sync = false);
    void addSignal(const RS_SignalConfig &config, bool sync = false);
    void addStructure(const RS_StructureConfig &config, bool sync = false);
    void addCamera(const RS_CameraConfig &config, bool sync = false);

    void updateMaterial(const RS_MaterialConfig &config, bool sync = false);

    void deleteFloor(bool sync = false);
    void deleteMaterial(RIndex id, bool sync = false);
    void deleteRecord(RIndex id, bool sync = false);
    void deleteSignal(RIndex id, bool sync = false);
    void deleteStructure(RIndex id, bool sync = false);
    void deleteCamera(RIndex id, bool sync = false);

    RIndex findMaterial(const std::string &name);
    RIndex findRecord(const std::string &name);
    RIndex findSignal(const std::string &name);
    RIndex findStructure(const std::string &name);
    RIndex findCamera(const std::string &name);

    RS_Config getConfig();
    RS_SimulationConfig getSimulationConfig();
    RS_FloorConfig getFloorConfig();
    RS_WorldConfig getWorldConfig();
    std::vector<RS_MaterialConfig> getMaterialConfigs();
    std::vector<RS_RecordConfig> getRecordConfigs();
    std::vector<RS_SignalConfig> getSignalConfigs();
    std::vector<RS_StructureConfig> getStructureConfigs();
    std::vector<RS_CameraConfig> getCameraConfigs();

    /****************************************************************
     * Data manager APIs
     ****************************************************************/
    RS_MemoryProvider *getDeviceProvider() const;
    RS_MemoryProvider *getHostProvider() const;
    std::shared_ptr<logging::logger> getLogger() const;
    std::shared_ptr<RS_ThreadPool> getThreadPool() const;
    RSize getKernelRecordBufferFrameNum() const;

    // Counters
    RIndex getMaterialCounter() const;
    RIndex getRecordCounter() const;
    RIndex getSignalCounter() const;
    RIndex getStructureCounter() const;
    RIndex getCameraCounter() const;

    // Name maps
    const std::unordered_map<std::string, RIndex> &getMaterialNameToGid() const;
    const std::unordered_map<std::string, RIndex> &getRecordNameToGid() const;
    const std::unordered_map<std::string, RIndex> &getSignalNameToGid() const;
    const std::unordered_map<std::string, RIndex> &getStructureNameToGid() const;
    const std::unordered_map<std::string, RIndex> &getCameraNameToGid() const;

    // Kernel data managers
    RS_KernelSpatialManager &getKernelSpatialManager();
    const std::unique_ptr<RS_KernelFloorManager> &getKernelFloorManager() const;
    const std::unordered_map<RIndex, RS_KernelMaterialManager> &getKernelMaterialManagers() const;
    const std::unordered_map<RIndex, RS_KernelStructureRecordManager> &
    getKernelStructureRecordManagers() const;
    const std::unordered_map<RIndex, RS_KernelForceManager> &getKernelForceManagers() const;
    const std::unordered_map<RIndex, RS_KernelTorqueManager> &getKernelTorqueManagers() const;
    const std::unordered_map<RIndex, RS_KernelStructureManager> &getKernelStructureManagers() const;
    const std::unordered_map<RIndex, RS_KernelCameraManager> &getKernelCameraManagers() const;

   private:
    RS_MemoryProvider *_device_provider = nullptr;
    RS_MemoryProvider *_host_provider = nullptr;
    std::shared_ptr<logging::logger> _logger;
    std::shared_ptr<RS_ThreadPool> _thread_pool;
    std::unique_ptr<std::thread> _kernel_synchronization_thread;
    std::queue<std::tuple<std::string, RS_Any>> _command_queue;
    std::mutex _command_queue_mutex;
    std::condition_variable _command_queue_cv;

    std::atomic<int> _command_exec_lock = {0};
    std::atomic<int> _command_exec_pause_count = {0};

    std::unordered_map<std::string, std::function<void(RS_Any)>> _methods;

    RSize _kernel_record_buffer_frame_num = 0;

    RS_SimulationConfig _simulation_config;

    /// Name maps
    std::unordered_map<std::string, RIndex> _material_name_to_gid;
    std::unordered_map<std::string, RIndex> _record_name_to_gid;
    std::unordered_map<std::string, RIndex> _signal_name_to_gid;
    std::unordered_map<std::string, RIndex> _structure_name_to_gid;
    std::unordered_map<std::string, RIndex> _camera_name_to_gid;

    /// Relation maps
    std::unordered_map<RIndex, RSE_RecordType> _record_gid_to_record_type;
    std::unordered_map<RIndex, RIndex> _structure_gid_to_structure_record_gid;
    std::unordered_map<RIndex, std::vector<RIndex>> _structure_gid_to_rigid_body_camera_gids;

    /// Counters
    RIndex _material_counter = 0;
    RIndex _record_counter = 0;
    RIndex _signal_counter = 0;
    RIndex _structure_counter = 0;
    RIndex _camera_counter = 0;

    /// Kernel data managers
    RS_KernelSpatialManager _kernel_spatial_manager;
    std::unique_ptr<RS_KernelFloorManager> _kernel_floor_manager;
    std::unique_ptr<RS_KernelWorldManager> _kernel_world_manager;
    std::unordered_map<RIndex, RS_KernelMaterialManager> _kernel_material_managers;
    std::unordered_map<RIndex, RS_KernelStructureRecordManager> _kernel_structure_record_managers;
    std::unordered_map<RIndex, RS_KernelForceManager> _kernel_force_managers;
    std::unordered_map<RIndex, RS_KernelTorqueManager> _kernel_torque_managers;
    std::unordered_map<RIndex, RS_KernelStructureManager> _kernel_structure_managers;
    std::unordered_map<RIndex, RS_KernelCameraManager> _kernel_camera_managers;

    void _kernelSynchronizationThreadFunc();

    RIndex _allocateMaterialGid();
    RIndex _allocateRecordGid();
    RIndex _allocateSignalGid();
    RIndex _allocateStructureGid();
    RIndex _allocateCameraGid();

    void _pauseKernel();
    void _resumeKernel();
    void _terminateKernel();

    void _beginModificationTransaction();
    void _endModificationTransaction();

    void _setSimulationConfig(const RS_SimulationConfig &config);
    void _setWorld(const RS_WorldConfig &config);
    void _setFloor(const RS_FloorConfig &config);

    void _addMaterial(const RS_MaterialConfig &config);
    void _addRecord(const RS_RecordConfig &config);
    void _addSignal(const RS_SignalConfig &config);
    void _addStructure(const RS_StructureConfig &config);
    void _addCamera(const RS_CameraConfig &config);

    void _updateMaterial(const RS_MaterialConfig &config);

    void _deleteFloor();
    void _deleteWorld();
    void _deleteMaterial(RIndex id);
    void _deleteRecord(RIndex id);
    void _deleteSignal(RIndex id);
    void _deleteStructure(RIndex id);
    void _deleteCamera(RIndex id);
};

class RS_KernelManagerGuard {
   public:
    RS_KernelManagerGuard(RS_KernelManager *kernel_manager) : _kernel_manager(kernel_manager) {
        kernel_manager->pauseCommandExecution();
    }

    ~RS_KernelManagerGuard() { _kernel_manager->resumeCommandExecution(); }

   private:
    RS_KernelManager *_kernel_manager;
};
}  // namespace rise

#endif  // RS_KERNEL_MANAGER_H