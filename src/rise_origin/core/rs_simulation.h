// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_H
#define RS_SIMULATION_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rise_origin/core/recorder/rs_simulation_recorder.h"
#include "rise_origin/core/rs_world_def.h"
#include "rise_origin/core/rs_world_handle.h"
#include "rise_origin/rsc/rs_rsc_validator.h"
#include "rise_origin/rsc/rsc_world.h"
#include "rise_origin/utils/math/rs_float.h"

namespace rise {

class RS_KernelManager;

class RS_Simulation {
   public:
    std::mutex status_mutex;
    std::condition_variable status_cv;
    RSE_WorldStatus status{RSE_WORLD_UNINITIALIZED};
    std::string status_info;
    std::string name;
    RSE_FloatType simulation_memory_accuracy = RSE_FLOAT_32;

    RS_RSCValidator validator;
    std::shared_ptr<RS_KernelManager> kernel_manager;
    std::future<void> recorder_sync_future{};

    /// Serializes handle mutations and spans a begin/endModificationTransaction so its commands
    /// stay contiguous. Recursive so a mutator nests in an open transaction; reads and waits skip
    /// it.
    std::recursive_mutex api_mutex;

    /// Active recorders, keyed by name.
    std::unordered_map<
        std::string,
        std::pair<rsc::RSC_RecorderVariant, std::shared_ptr<RS_SimulationRecorder>>>
        recorders;
    std::mutex recorders_mutex;

    /// Registered callback controllers, keyed by name.
    std::unordered_map<std::string, rsc::RSC_ControllerVariant> controllers;
    std::mutex controllers_mutex;

    /// Bounded append-only message log; reads never remove and oldest entries evict past capacity.
    /// Guarded by message_mutex.
    std::mutex message_mutex;
    std::condition_variable message_cv;
    std::deque<RS_WorldMessage> messages;
    RBigIndex message_next_index = 0;
    RSize message_capacity = RS_WORLD_MESSAGE_BUFFER_DEFAULT_SIZE;

    RS_Simulation() = default;

    /*****************************************************************
     * State
     *****************************************************************/
    /**
     * @brief Read the current status and status info atomically.
     */
    [[nodiscard]] std::pair<RSE_WorldStatus, std::string> getStatusAndStatusInfo() {
        std::lock_guard lock(status_mutex);
        return std::make_pair(status, status_info);
    }

    /**
     * @brief Set the status and status info and wake all status waiters.
     */
    void
    setStatusAndStatusInfo(const RSE_WorldStatus new_status, const std::string &new_status_info) {
        std::lock_guard lock(status_mutex);
        status = new_status;
        status_info = new_status_info;
        status_cv.notify_all();
    }

    /**
     * @brief Block until the status reaches expected_status or the world ends; true if it matched.
     */
    bool waitForStatus(const RSE_WorldStatus &expected_status) {
        std::unique_lock lock(status_mutex);
        status_cv.wait(lock, [this, expected_status] {
            return status == expected_status || isWorldEnd(status);
        });
        return expected_status == status;
    }

    /**
     * @brief Block until the world is live, running or paused, or ends; true while it is live.
     */
    bool waitForLive() {
        std::unique_lock lock(status_mutex);
        status_cv.wait(lock, [this] {
            return status == RSE_WORLD_RUNNING || status == RSE_WORLD_PAUSED || isWorldEnd(status);
        });
        return status == RSE_WORLD_RUNNING || status == RSE_WORLD_PAUSED;
    }

    /**
     * @brief Block until the world ends.
     */
    void waitForEnd() {
        std::unique_lock lock(status_mutex);
        status_cv.wait(lock, [this] { return isWorldEnd(status); });
    }

    /*****************************************************************
     * Messages
     *****************************************************************/

    /**
     * @brief Index of the newest message in the log, or RS_NULL_BIG_INDEX when empty.
     */
    RBigIndex getNewestMessageIndex() {
        std::lock_guard lock(message_mutex);
        if (messages.empty()) return RS_NULL_BIG_INDEX;
        return messages.back().index;
    }

    /**
     * @brief Index of the oldest retained message, or RS_NULL_BIG_INDEX when empty.
     * @details A reader whose cursor is below this index has missed evicted entries.
     */
    RBigIndex getOldestMessageIndex() {
        std::lock_guard lock(message_mutex);
        if (messages.empty()) return RS_NULL_BIG_INDEX;
        return messages.front().index;
    }

    /**
     * @brief Return up to max_count messages in [oldest_index, newest_index) at severity >= level.
     * @details Reads never remove. Results run oldest to newest; when capped, the newest are kept.
     */
    std::vector<RS_WorldMessage> getMessagesMatching(
        RSize max_count = RS_MAX_SIZE,
        RBigIndex oldest_index = 0,
        RBigIndex newest_index = RS_NULL_BIG_INDEX,
        RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_INFO
    ) {
        std::lock_guard lock(message_mutex);
        std::vector<RS_WorldMessage> result;
        for (const auto &message : messages) {
            if (message.index < oldest_index) continue;
            if (message.index >= newest_index) break;
            if (message.level < level) continue;
            result.push_back(message);
        }
        if (result.size() > max_count)
            result.erase(result.begin(), result.end() - static_cast<std::ptrdiff_t>(max_count));
        return result;
    }

    /**
     * @brief Block until the next message at severity >= level posted after this call, then return
     * it.
     */
    RS_WorldMessage waitForMessage(RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_INFO) {
        std::unique_lock lock(message_mutex);
        RBigIndex since = message_next_index;
        while (true) {
            for (const auto &message : messages) {
                if (message.index < since) continue;
                if (message.level >= level) return message;
            }
            message_cv.wait(lock);
        }
    }

    /**
     * @brief Number of entries the message log retains.
     */
    RSize getMessageBufferSize() {
        std::lock_guard lock(message_mutex);
        return message_capacity;
    }

    /**
     * @brief Set the number of entries the message log retains; shrinking evicts the oldest.
     */
    void setMessageBufferSize(RSize max_messages) {
        std::lock_guard lock(message_mutex);
        message_capacity = max_messages;
        while (messages.size() > message_capacity) messages.pop_front();
    }

    /**
     * @brief Append a message at the given level and wake all message waiters.
     */
    void postMessage(RSE_WorldMessageLevel level, std::string text) {
        {
            std::lock_guard lock(message_mutex);
            if (message_next_index == RS_NULL_BIG_INDEX)
                RS_THROW(RS_BaseException("World message log index space exhausted"));
            messages.push_back(
                {message_next_index++, std::chrono::system_clock::now(), level, std::move(text)}
            );
            while (messages.size() > message_capacity) messages.pop_front();
        }
        message_cv.notify_all();
    }

    /*****************************************************************
     * Recorder sync
     *****************************************************************/
    /**
     * @brief Store the future tracking an in-flight recorder flush.
     */
    void setRecorderSyncFuture(std::future<void> &&new_future) {
        recorder_sync_future = std::move(new_future);
    }

    /**
     * @brief Block until any in-flight recorder flush completes.
     */
    void waitForRecorderSync() {
        if (recorder_sync_future.valid()) recorder_sync_future.wait();
    }

    /*****************************************************************
     * Recorder ops
     *****************************************************************/

    /**
     * @brief Construct a recorder from the variant, call start(), push.
     */
    void addRecorder(const rsc::RSC_RecorderVariant &recorder);

    /**
     * @brief Wait for any in-flight sync, call finish() on the matching recorder, erase.
     */
    void removeRecorder(const std::string &name);

    /**
     * @brief Re-emit the originating variant for the named recorder.
     */
    rsc::RSC_RecorderVariant findRecorder(const std::string &name);

    /*****************************************************************
     * Controller ops
     *****************************************************************/

    /**
     * @brief Register a callback controller from the variant.
     */
    void addController(const rsc::RSC_ControllerVariant &controller);

    /**
     * @brief Unregister the named controller.
     */
    void removeController(const std::string &name);

    /**
     * @brief Re-emit the originating variant for the named controller.
     */
    rsc::RSC_ControllerVariant findController(const std::string &name);

    /**
     * @brief Snapshot the set of registered controller names.
     */
    std::unordered_set<std::string> getControllerNames();
};

}  // namespace rise

#endif  // RS_SIMULATION_H
