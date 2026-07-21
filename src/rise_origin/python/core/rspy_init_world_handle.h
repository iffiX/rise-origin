// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_INIT_WORLD_HANDLE_H
#define RSPY_INIT_WORLD_HANDLE_H

#include "common/rspy_common.h"

/*****************************************************************
 * World handle wrapper.
 *****************************************************************/

class RSPY_WorldHandle {
   public:
    std::shared_ptr<RS_WorldHandle> handle;
    explicit RSPY_WorldHandle(std::shared_ptr<RS_WorldHandle> handle) : handle(std::move(handle)) {}

    // Info
    [[nodiscard]] std::string get_info(const std::string &info_name) const {
        return handle->getInfo(info_name);
    }

    // Lifecycle
    [[nodiscard]] bool is_pending() const { return handle->isPending(); }
    [[nodiscard]] bool is_running() const { return handle->isRunning(); }
    [[nodiscard]] bool is_paused() const { return handle->isPaused(); }
    [[nodiscard]] bool is_end() const { return handle->isEnd(); }
    void pause() const { handle->pause(); }
    void resume() const { handle->resume(); }
    void terminate() const { handle->terminate(); }
    [[nodiscard]] RSE_WorldStatus get_status() const { return handle->getStatus(); }

    bool wait_for_running() const {
        // Release the GIL before blocking on the simulation thread.
        py::gil_scoped_release release;
        return handle->waitForRunning();
    }

    void wait_for_end() const {
        py::gil_scoped_release release;
        handle->waitForEnd();
    }

    [[nodiscard]] RBigIndex get_newest_message_index() const {
        return handle->getNewestMessageIndex();
    }

    [[nodiscard]] RBigIndex get_oldest_message_index() const {
        return handle->getOldestMessageIndex();
    }

    [[nodiscard]] std::vector<RS_WorldMessage> get_messages_matching(
        RSize max_count = RS_MAX_SIZE,
        RBigIndex oldest_index = 0,
        RBigIndex newest_index = RS_NULL_BIG_INDEX,
        RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_INFO
    ) const {
        return handle->getMessagesMatching(max_count, oldest_index, newest_index, level);
    }

    RS_WorldMessage wait_for_message(RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_INFO) const {
        py::gil_scoped_release release;
        return handle->waitForMessage(level);
    }

    [[nodiscard]] RSize get_message_buffer_size() const { return handle->getMessageBufferSize(); }

    void set_message_buffer_size(RSize max_messages) const {
        handle->setMessageBufferSize(max_messages);
    }

    void begin_modification_transaction() const { handle->beginModificationTransaction(); }
    void end_modification_transaction() const { handle->endModificationTransaction(); }

    [[nodiscard]] rsc::RSC_World get_world() const { return handle->getWorld(); }

    RS_WorldTask set_simulation(const rsc::RSC_Simulation &simulation) const {
        return handle->setSimulation(simulation);
    }

    // Resource
    RS_WorldTask add_resource(const rsc::RSC_ResourceVariant &resource) const {
        return handle->addResource(resource);
    }
    RS_WorldTask remove_resource(const std::string &name) const {
        return handle->removeResource(name);
    }
    [[nodiscard]] rsc::RSC_ResourceVariant find_resource(const std::string &name) const {
        return handle->findResource(name);
    }

    // Essence
    RS_WorldTask add_essence(const rsc::RSC_EssenceVariant &essence) const {
        return handle->addEssence(essence);
    }
    RS_WorldTask
    update_essence(const std::string &name, const rsc::RSC_EssenceVariant &essence) const {
        return handle->updateEssence(name, essence);
    }
    RS_WorldTask remove_essence(const std::string &name) const {
        return handle->removeEssence(name);
    }
    [[nodiscard]] rsc::RSC_EssenceVariant find_essence(const std::string &name) const {
        return handle->findEssence(name);
    }

    // Entity
    RS_WorldTask add_entity(const rsc::RSC_EntityVariant &entity) const {
        return handle->addEntity(entity);
    }
    RS_WorldTask remove_entity(const std::string &name) const { return handle->removeEntity(name); }
    RS_WorldTask freeze_entity(const std::string &name) const { return handle->freezeEntity(name); }
    RS_WorldTask unfreeze_entity(const std::string &name) const {
        return handle->unfreezeEntity(name);
    }
    [[nodiscard]] rsc::RSC_EntityVariant find_entity(const std::string &name) const {
        return handle->findEntity(name);
    }
    [[nodiscard]] bool is_entity_frozen(const std::string &name) const {
        return handle->isEntityFrozen(name);
    }

    // Record
    RS_WorldTask add_record(const rsc::RSC_RecordVariant &record) const {
        return handle->addRecord(record);
    }
    RS_WorldTask remove_record(const std::string &name) const { return handle->removeRecord(name); }
    [[nodiscard]] rsc::RSC_RecordVariant find_record(const std::string &name) const {
        return handle->findRecord(name);
    }

    // Signal
    RS_WorldTask add_signal(const rsc::RSC_SignalVariant &signal) const {
        return handle->addSignal(signal);
    }
    RS_WorldTask remove_signal(const std::string &name) const { return handle->removeSignal(name); }
    [[nodiscard]] rsc::RSC_SignalVariant find_signal(const std::string &name) const {
        return handle->findSignal(name);
    }

    // Recorder
    RS_WorldTask add_recorder(const rsc::RSC_RecorderVariant &recorder) const {
        return handle->addRecorder(recorder);
    }
    RS_WorldTask remove_recorder(const std::string &name) const {
        return handle->removeRecorder(name);
    }
    [[nodiscard]] rsc::RSC_RecorderVariant find_recorder(const std::string &name) const {
        return handle->findRecorder(name);
    }

    // Controller
    RS_WorldTask add_controller(const rsc::RSC_ControllerVariant &controller) const {
        return handle->addController(controller);
    }
    RS_WorldTask remove_controller(const std::string &name) const {
        return handle->removeController(name);
    }
    [[nodiscard]] rsc::RSC_ControllerVariant find_controller(const std::string &name) const {
        return handle->findController(name);
    }
};

/// Bind world messages, tasks, and the world handle.
void initRSPYWorldHandle(py::module &m);

#endif  // RSPY_INIT_WORLD_HANDLE_H
