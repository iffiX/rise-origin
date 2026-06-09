// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_LOCAL_HANDLE_H
#define RS_SIMULATION_LOCAL_HANDLE_H

#include "rise_origin/core/rs_simulation_handle.h"
#include "rs_simulation.h"

namespace rise {

class RS_SimulationLocalHandle : public RS_SimulationHandle {
   public:
    explicit RS_SimulationLocalHandle(const std::shared_ptr<RS_Simulation> &simulation);

    // State Management
    bool isPending() override;
    bool isRunning() override;
    bool isPaused() override;
    bool isEnd() override;
    void pause() override;
    void resume() override;
    void terminate() override;
    RSE_SimulationStatus getStatus() override;
    bool waitForRunning() override;
    void waitForEnd() override;

    // Configuration Management
    void beginModificationTransaction() override;
    void endModificationTransaction() override;
    void setSimulationConfig(const RS_SimulationConfig &config) override;
    void setWorld(const RS_WorldConfig &config) override;
    void setFloor(const RS_FloorConfig &config) override;

    // Entity Management
    void addMaterial(const RS_MaterialConfig &config) override;
    void addRecord(const RS_RecordConfig &config) override;
    void addSignal(const RS_SignalConfig &config) override;
    void addStructure(const RS_StructureConfig &config) override;
    void addCamera(const RS_CameraConfig &config) override;
    void updateMaterial(const RS_MaterialConfig &config) override;
    void deleteFloor() override;
    void deleteMaterial(RIndex id) override;
    void deleteRecord(RIndex id) override;
    void deleteSignal(RIndex id) override;
    void deleteStructure(RIndex id) override;
    void deleteCamera(RIndex id) override;

    // Querying
    RIndex findMaterial(const std::string &name) override;
    RIndex findRecord(const std::string &name) override;
    RIndex findSignal(const std::string &name) override;
    RIndex findStructure(const std::string &name) override;
    RIndex findCamera(const std::string &name) override;
    RS_Config getConfig() override;
    RS_SimulationConfig getSimulationConfig() override;
    RS_FloorConfig getFloorConfig() override;
    std::vector<RS_MaterialConfig> getMaterialConfigs() override;
    std::vector<RS_RecordConfig> getRecordConfigs() override;
    std::vector<RS_SignalConfig> getSignalConfigs() override;
    std::vector<RS_StructureConfig> getStructureConfigs() override;
    std::vector<RS_CameraConfig> getCameraConfigs() override;

   private:
    std::shared_ptr<RS_Simulation> _simulation;
};

}  // namespace rise

#endif  // RS_SIMULATION_LOCAL_HANDLE_H
