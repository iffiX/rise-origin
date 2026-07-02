// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_HANDLE_H
#define RS_WORLD_HANDLE_H

#include <chrono>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include "rise_origin/rsc/rsc_world.h"
#include "rise_origin/utils/data/rs_variant.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_world_def.h"

namespace rise {

/**
 * @brief Severity of a world message.
 */
RS_ENUM(
    RSE_WorldMessageLevel,
    RIndex,
    (RSE_WORLD_MESSAGE_INFO, 0),
    (RSE_WORLD_MESSAGE_WARNING, 1),
    (RSE_WORLD_MESSAGE_ERROR, 2)
);

/**
 * @brief One entry in a world's message log.
 * @details Entries live in a bounded ring keyed by a monotonic index and are never removed by
 * reads.
 */
struct RS_WorldMessage {
    RBigIndex index = RS_NULL_BIG_INDEX;
    std::chrono::system_clock::time_point time;
    RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_INFO;
    std::string text;
};

/**
 * @brief Task result carrying the gid a world add operation assigned.
 */
struct RS_WorldTaskDataCreated {
    RIndex gid = RS_NULL_INDEX;
    std::string name;
};

/**
 * @brief Task result for a command that succeeded without producing an object.
 */
struct RS_WorldTaskDataSuccess {};

/**
 * @brief Task result for a command that failed.
 */
struct RS_WorldTaskDataFailed {
    std::string error;
};

/**
 * @brief Result body carried by a world task.
 */
using RS_WorldTaskData =
    RS_Variant<RS_WorldTaskDataCreated, RS_WorldTaskDataSuccess, RS_WorldTaskDataFailed>;

/**
 * @brief Completion of a single deferred world command.
 * @details Copyable. The outcome is delivered as data, so get never throws for a finished command.
 */
class RS_WorldTask {
    /**
     * @brief Shared future over the command result.
     */
    std::shared_future<RS_WorldTaskData> _data;

   public:
    RS_WorldTask() = default;
    explicit RS_WorldTask(std::shared_future<RS_WorldTaskData> data) : _data(std::move(data)) {}

    /**
     * @brief Completion state without blocking.
     */
    [[nodiscard]] RSE_WorldTaskState state() const {
        if (!isFinished()) return RSE_WORLD_TASK_PENDING;
        return _data.get().is<RS_WorldTaskDataFailed>() ? RSE_WORLD_TASK_FAIL
                                                        : RSE_WORLD_TASK_SUCCESS;
    }

    /**
     * @brief Whether the command has finished without blocking.
     */
    [[nodiscard]] bool isFinished() const {
        return _data.valid() &&
               _data.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    }

    /**
     * @brief Whether this task refers to a command.
     */
    [[nodiscard]] bool isValid() const { return _data.valid(); }

    /**
     * @brief Block until the command finishes.
     */
    void wait() const {
        if (_data.valid()) _data.wait();
    }

    /**
     * @brief Block until the command finishes or the timeout elapses.
     * @return The completion state, or RSE_WORLD_TASK_PENDING on timeout.
     */
    template <class Rep, class Period>
    RSE_WorldTaskState waitFor(const std::chrono::duration<Rep, Period>& timeout) const {
        if (!_data.valid() || _data.wait_for(timeout) != std::future_status::ready)
            return RSE_WORLD_TASK_PENDING;
        return _data.get().is<RS_WorldTaskDataFailed>() ? RSE_WORLD_TASK_FAIL
                                                        : RSE_WORLD_TASK_SUCCESS;
    }

    /**
     * @brief Block until the command finishes and return its result.
     */
    [[nodiscard]] RS_WorldTaskData get() const { return _data.get(); }

    /**
     * @brief Make a task already finished with the given result.
     */
    static RS_WorldTask makeFinished(RS_WorldTaskData data) {
        std::promise<RS_WorldTaskData> promise;
        promise.set_value(std::move(data));
        return RS_WorldTask(promise.get_future().share());
    }
};

/**
 * @brief Handle to a running world.
 * @details Exposes lifecycle control and name-keyed add/update/remove/find
 * operations over the six RSC category axes (entity, essence, resource,
 * controller, signal, recorder). An atomic sequence of mutations can be
 * framed by wrapping with begin/endModificationTransaction.
 */
class RS_WorldHandle {
   public:
    virtual ~RS_WorldHandle() = default;

    /*****************************************************************
     * Identity
     *****************************************************************/
    /**
     * @brief Name of the world this handle controls.
     */
    virtual std::string name() = 0;

    /*****************************************************************
     * Info
     *****************************************************************/
    /**
     * @brief Query a named piece of world or system info as a string.
     * @details One key resolves to one source. device_memory_used / device_memory_capacity and
     * host_memory_used / host_memory_capacity report bytes; device_name reports the compute device
     * as type:id such as cuda:0. An unknown key returns an empty string.
     */
    virtual std::string getInfo(const std::string& info_name) = 0;

    /*****************************************************************
     * State
     *****************************************************************/
    /**
     * @brief True while the world is uninitialized or initializing.
     */
    virtual bool isPending() = 0;

    /**
     * @brief True while the world is running.
     */
    virtual bool isRunning() = 0;

    /**
     * @brief True while the world is paused.
     */
    virtual bool isPaused() = 0;

    /**
     * @brief True once the world has ended (terminated or errored).
     */
    virtual bool isEnd() = 0;

    /**
     * @brief Request the world pause.
     */
    virtual void pause() = 0;

    /**
     * @brief Request the world resume.
     */
    virtual void resume() = 0;

    /**
     * @brief Request the world terminate.
     */
    virtual void terminate() = 0;

    /**
     * @brief Current world status.
     */
    virtual RSE_WorldStatus getStatus() = 0;

    /**
     * @brief Block until the world is running; false if it ended first.
     */
    virtual bool waitForRunning() = 0;

    /**
     * @brief Block until the world ends.
     */
    virtual void waitForEnd() = 0;

    /*****************************************************************
     * Messages
     *****************************************************************/
    /**
     * @brief Index of the newest message in the log, or RS_NULL_BIG_INDEX when empty.
     */
    virtual RBigIndex getNewestMessageIndex() = 0;

    /**
     * @brief Index of the oldest retained message; a cursor below it has missed entries.
     */
    virtual RBigIndex getOldestMessageIndex() = 0;

    /**
     * @brief Return up to max_count messages in [oldest_index, newest_index) at severity >= level.
     * @details Reads never remove. Results run oldest to newest; when capped, the newest are kept.
     */
    virtual std::vector<RS_WorldMessage> getMessagesMatching(
        RSize max_count = RS_MAX_SIZE,
        RBigIndex oldest_index = 0,
        RBigIndex newest_index = RS_NULL_BIG_INDEX,
        RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_INFO
    ) = 0;

    /**
     * @brief Block until the next message at severity >= level, then return it.
     */
    virtual RS_WorldMessage waitForMessage(
        RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_INFO
    ) = 0;

    /**
     * @brief Number of entries the message log retains.
     */
    virtual RSize getMessageBufferSize() = 0;

    /**
     * @brief Set the number of entries the message log retains; shrinking evicts the oldest.
     */
    virtual void setMessageBufferSize(RSize max_messages) = 0;

    /*****************************************************************
     * Modification transactions
     *****************************************************************/
    /**
     * @brief Open a modification transaction framing subsequent mutations.
     */
    virtual void beginModificationTransaction() = 0;

    /**
     * @brief Close the current modification transaction.
     */
    virtual void endModificationTransaction() = 0;

    /*****************************************************************
     * Root
     *****************************************************************/
    /**
     * @brief Snapshot the world as an RSC_World characterization.
     */
    virtual rsc::RSC_World getWorld() = 0;

    /**
     * @brief Apply a new simulation sub-characterization.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask setSimulation(const rsc::RSC_Simulation& simulation) = 0;

    /*****************************************************************
     * Resource
     *****************************************************************/
    /**
     * @brief Add a resource.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask addResource(const rsc::RSC_ResourceVariant& resource) = 0;

    /**
     * @brief Remove the named resource.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask removeResource(const std::string& name) = 0;

    /**
     * @brief Re-emit the named resource's characterization.
     */
    virtual rsc::RSC_ResourceVariant findResource(const std::string& name) = 0;

    /*****************************************************************
     * Essence
     *****************************************************************/
    /**
     * @brief Add an essence.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask addEssence(const rsc::RSC_EssenceVariant& essence) = 0;
    /**
     * @brief Update the named essence.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask
    updateEssence(const std::string& name, const rsc::RSC_EssenceVariant& essence) = 0;

    /**
     * @brief Remove the named essence.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask removeEssence(const std::string& name) = 0;

    /**
     * @brief Re-emit the named essence's characterization.
     */
    virtual rsc::RSC_EssenceVariant findEssence(const std::string& name) = 0;

    /*****************************************************************
     * Entity
     *****************************************************************/
    /**
     * @brief Add an entity.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask addEntity(const rsc::RSC_EntityVariant& entity) = 0;
    /**
     * @brief Remove the named entity.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask removeEntity(const std::string& name) = 0;

    /**
     * @brief Freeze the named entity so it stops simulating or rendering while staying present.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask freezeEntity(const std::string& name) = 0;

    /**
     * @brief Unfreeze the named entity so it resumes simulating or rendering.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask unfreezeEntity(const std::string& name) = 0;

    /**
     * @brief Re-emit the named entity's characterization.
     */
    virtual rsc::RSC_EntityVariant findEntity(const std::string& name) = 0;

    /**
     * @brief Whether the named entity is currently frozen.
     */
    virtual bool isEntityFrozen(const std::string& name) = 0;

    /**
     * @brief Resolve an entity name to its global id, or RS_NULL_INDEX if absent.
     */
    virtual RIndex findEntityGID(const std::string& name) = 0;

    /**
     * @brief Resolve a global id to its entity name, or empty if absent.
     */
    virtual std::string findEntityName(RIndex gid) = 0;

    /*****************************************************************
     * Record
     *****************************************************************/
    /**
     * @brief Add a record.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask addRecord(const rsc::RSC_RecordVariant& record) = 0;

    /**
     * @brief Remove the named record.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask removeRecord(const std::string& name) = 0;

    /**
     * @brief Re-emit the named record's characterization.
     */
    virtual rsc::RSC_RecordVariant findRecord(const std::string& name) = 0;

    /*****************************************************************
     * Signal
     *****************************************************************/
    /**
     * @brief Add a signal.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask addSignal(const rsc::RSC_SignalVariant& signal) = 0;

    /**
     * @brief Remove the named signal.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask removeSignal(const std::string& name) = 0;

    /**
     * @brief Re-emit the named signal's characterization.
     */
    virtual rsc::RSC_SignalVariant findSignal(const std::string& name) = 0;

    /*****************************************************************
     * Recorder
     *****************************************************************/
    /**
     * @brief Add a recorder.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask addRecorder(const rsc::RSC_RecorderVariant& recorder) = 0;

    /**
     * @brief Remove the named recorder.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask removeRecorder(const std::string& name) = 0;

    /**
     * @brief Re-emit the named recorder's characterization.
     */
    virtual rsc::RSC_RecorderVariant findRecorder(const std::string& name) = 0;

    /*****************************************************************
     * Controller
     *****************************************************************/
    /**
     * @brief Add a controller.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask addController(const rsc::RSC_ControllerVariant& controller) = 0;

    /**
     * @brief Remove the named controller.
     * @return Task tracking the command.
     */
    virtual RS_WorldTask removeController(const std::string& name) = 0;

    /**
     * @brief Re-emit the named controller's characterization.
     */
    virtual rsc::RSC_ControllerVariant findController(const std::string& name) = 0;
};

}  // namespace rise

#endif  // RS_WORLD_HANDLE_H
