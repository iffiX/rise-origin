// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_HANDLE_H
#define RS_SIMULATION_HANDLE_H

#include "rise_origin/rsc/rs_config.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_simulation_def.h"

namespace rise {

/**
 * Abstract base class for simulation handles
 */
class RS_SimulationHandle {
   public:
    virtual ~RS_SimulationHandle() = default;

    // State Management
    virtual bool isPending() = 0;
    virtual bool isRunning() = 0;
    virtual bool isPaused() = 0;
    virtual bool isEnd() = 0;
    virtual void pause() = 0;
    virtual void resume() = 0;
    virtual void terminate() = 0;
    virtual RSE_SimulationStatus getStatus() = 0;
    virtual bool waitForRunning() = 0;
    virtual void waitForEnd() = 0;

    // Configuration Management
    virtual void beginModificationTransaction() = 0;
    virtual void endModificationTransaction() = 0;
    virtual void setSimulationConfig(const RS_SimulationConfig &config) = 0;
    virtual void setWorld(const RS_WorldConfig &config) = 0;
    virtual void setFloor(const RS_FloorConfig &config) = 0;

    // Entity Management
    // throws RS_InvalidConfigurationException if the configuration is invalid
    virtual void addMaterial(const RS_MaterialConfig &config) = 0;
    virtual void addRecord(const RS_RecordConfig &config) = 0;
    virtual void addSignal(const RS_SignalConfig &config) = 0;
    virtual void addStructure(const RS_StructureConfig &config) = 0;
    virtual void addCamera(const RS_CameraConfig &config) = 0;
    virtual void updateMaterial(const RS_MaterialConfig &config) = 0;
    virtual void deleteFloor() = 0;
    virtual void deleteMaterial(RIndex id) = 0;
    virtual void deleteRecord(RIndex id) = 0;
    virtual void deleteSignal(RIndex id) = 0;
    virtual void deleteStructure(RIndex id) = 0;
    virtual void deleteCamera(RIndex id) = 0;

    // Querying
    virtual RIndex findMaterial(const std::string &name) = 0;
    virtual RIndex findRecord(const std::string &name) = 0;
    virtual RIndex findSignal(const std::string &name) = 0;
    virtual RIndex findStructure(const std::string &name) = 0;
    virtual RIndex findCamera(const std::string &name) = 0;
    virtual RS_Config getConfig() = 0;
    virtual RS_SimulationConfig getSimulationConfig() = 0;
    virtual RS_FloorConfig getFloorConfig() = 0;
    virtual std::vector<RS_MaterialConfig> getMaterialConfigs() = 0;
    virtual std::vector<RS_RecordConfig> getRecordConfigs() = 0;
    virtual std::vector<RS_SignalConfig> getSignalConfigs() = 0;
    virtual std::vector<RS_StructureConfig> getStructureConfigs() = 0;
    virtual std::vector<RS_CameraConfig> getCameraConfigs() = 0;
};

}  // namespace rise

#endif  // RS_SIMULATION_HANDLE_H
