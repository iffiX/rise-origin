// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_LOCAL_HANDLE_H
#define RS_WORLD_LOCAL_HANDLE_H

#include "rise_origin/core/rs_world_handle.h"
#include "rs_simulation.h"

namespace rise {

class RS_WorldLocalHandle : public RS_WorldHandle {
   public:
    explicit RS_WorldLocalHandle(const std::shared_ptr<RS_Simulation> &simulation);

    // Identity
    std::string name() override;

    // State
    bool isPending() override;
    bool isRunning() override;
    bool isPaused() override;
    bool isEnd() override;
    void pause() override;
    void resume() override;
    void terminate() override;
    RSE_WorldStatus getStatus() override;
    bool waitForRunning() override;
    void waitForEnd() override;

    // Messages
    bool hasMessage() override;
    std::vector<RS_WorldMessage> getMessages() override;
    RS_WorldMessage waitForMessage(
        RIndex message_id = RS_NULL_INDEX,
        RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_ANY_LEVEL
    ) override;

    // Modification transactions
    void beginModificationTransaction() override;
    void endModificationTransaction() override;

    // Root
    rsc::RSC_World getWorld() override;
    RIndex setSimulation(const rsc::RSC_Simulation &simulation) override;

    // Resource
    RIndex addResource(const rsc::RSC_ResourceVariant &resource) override;
    RIndex removeResource(const std::string &name) override;
    rsc::RSC_ResourceVariant findResource(const std::string &name) override;

    // Essence
    RIndex addEssence(const rsc::RSC_EssenceVariant &essence) override;
    RIndex updateEssence(const std::string &name, const rsc::RSC_EssenceVariant &essence) override;
    RIndex removeEssence(const std::string &name) override;
    rsc::RSC_EssenceVariant findEssence(const std::string &name) override;

    // Entity
    RIndex addEntity(const rsc::RSC_EntityVariant &entity) override;
    RIndex removeEntity(const std::string &name) override;
    RIndex freezeEntity(const std::string &name) override;
    RIndex unfreezeEntity(const std::string &name) override;
    rsc::RSC_EntityVariant findEntity(const std::string &name) override;
    bool isEntityFrozen(const std::string &name) override;

    // Record
    RIndex addRecord(const rsc::RSC_RecordVariant &record) override;
    RIndex removeRecord(const std::string &name) override;
    rsc::RSC_RecordVariant findRecord(const std::string &name) override;

    // Signal
    RIndex addSignal(const rsc::RSC_SignalVariant &signal) override;
    RIndex removeSignal(const std::string &name) override;
    rsc::RSC_SignalVariant findSignal(const std::string &name) override;

    // Recorder
    RIndex addRecorder(const rsc::RSC_RecorderVariant &recorder) override;
    RIndex removeRecorder(const std::string &name) override;
    rsc::RSC_RecorderVariant findRecorder(const std::string &name) override;

    // Controller
    RIndex addController(const rsc::RSC_ControllerVariant &controller) override;
    RIndex removeController(const std::string &name) override;
    rsc::RSC_ControllerVariant findController(const std::string &name) override;

   private:
    std::shared_ptr<RS_Simulation> _simulation;
};

}  // namespace rise

#endif  // RS_WORLD_LOCAL_HANDLE_H
