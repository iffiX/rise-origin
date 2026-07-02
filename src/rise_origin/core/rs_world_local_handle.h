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

    // Info
    std::string getInfo(const std::string &info_name) override;

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
    RBigIndex getNewestMessageIndex() override;
    RBigIndex getOldestMessageIndex() override;
    std::vector<RS_WorldMessage> getMessagesMatching(
        RSize max_count = RS_MAX_SIZE,
        RBigIndex oldest_index = 0,
        RBigIndex newest_index = RS_NULL_BIG_INDEX,
        RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_INFO
    ) override;
    RS_WorldMessage waitForMessage(RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_INFO) override;
    RSize getMessageBufferSize() override;
    void setMessageBufferSize(RSize max_messages) override;

    // Modification transactions
    void beginModificationTransaction() override;
    void endModificationTransaction() override;

    // Root
    rsc::RSC_World getWorld() override;
    RS_WorldTask setSimulation(const rsc::RSC_Simulation &simulation) override;

    // Resource
    RS_WorldTask addResource(const rsc::RSC_ResourceVariant &resource) override;
    RS_WorldTask removeResource(const std::string &name) override;
    rsc::RSC_ResourceVariant findResource(const std::string &name) override;

    // Essence
    RS_WorldTask addEssence(const rsc::RSC_EssenceVariant &essence) override;
    RS_WorldTask
    updateEssence(const std::string &name, const rsc::RSC_EssenceVariant &essence) override;
    RS_WorldTask removeEssence(const std::string &name) override;
    rsc::RSC_EssenceVariant findEssence(const std::string &name) override;

    // Entity
    RS_WorldTask addEntity(const rsc::RSC_EntityVariant &entity) override;
    RS_WorldTask removeEntity(const std::string &name) override;
    RS_WorldTask freezeEntity(const std::string &name) override;
    RS_WorldTask unfreezeEntity(const std::string &name) override;
    rsc::RSC_EntityVariant findEntity(const std::string &name) override;
    bool isEntityFrozen(const std::string &name) override;
    RIndex findEntityGID(const std::string &name) override;
    std::string findEntityName(RIndex gid) override;

    // Record
    RS_WorldTask addRecord(const rsc::RSC_RecordVariant &record) override;
    RS_WorldTask removeRecord(const std::string &name) override;
    rsc::RSC_RecordVariant findRecord(const std::string &name) override;

    // Signal
    RS_WorldTask addSignal(const rsc::RSC_SignalVariant &signal) override;
    RS_WorldTask removeSignal(const std::string &name) override;
    rsc::RSC_SignalVariant findSignal(const std::string &name) override;

    // Recorder
    RS_WorldTask addRecorder(const rsc::RSC_RecorderVariant &recorder) override;
    RS_WorldTask removeRecorder(const std::string &name) override;
    rsc::RSC_RecorderVariant findRecorder(const std::string &name) override;

    // Controller
    RS_WorldTask addController(const rsc::RSC_ControllerVariant &controller) override;
    RS_WorldTask removeController(const std::string &name) override;
    rsc::RSC_ControllerVariant findController(const std::string &name) override;

   private:
    std::shared_ptr<RS_Simulation> _simulation;
};

}  // namespace rise

#endif  // RS_WORLD_LOCAL_HANDLE_H
