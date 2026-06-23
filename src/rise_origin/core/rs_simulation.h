// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_H
#define RS_SIMULATION_H

#include <atomic>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rise_origin/core/recorder/rs_simulation_recorder.h"
#include "rise_origin/core/rs_world_handle.h"
#include "rise_origin/core/rs_world_def.h"
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

    /// Active recorders, keyed by name.
    std::unordered_map<
        std::string,
        std::pair<rsc::RSC_RecorderVariant, std::shared_ptr<RS_SimulationRecorder>>>
        recorders;
    std::mutex recorders_mutex;

    /// Registered callback controllers, keyed by name.
    std::unordered_map<std::string, rsc::RSC_ControllerVariant> controllers;
    std::mutex controllers_mutex;

    /// Async message channel.
    std::mutex message_mutex;
    std::condition_variable message_cv;
    std::vector<RS_WorldMessage> messages;
    std::atomic<RIndex> message_id_counter{0};

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
    void setStatusAndStatusInfo(
        const RSE_WorldStatus new_status,
        const std::string &new_status_info
    ) {
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
     * @brief Report whether one or more world messages are waiting to be drained.
     * @return True when getMessages() would return at least one message.
     */
    bool hasMessage() {
        std::lock_guard lock(message_mutex);
        return !messages.empty();
    }

    /**
     * @brief Drain and return all currently queued world messages.
     * @details Messages are returned in posting order and removed from the queue atomically. Once
     * drained, a message cannot satisfy a later waitForMessage() call.
     * @return The queued messages.
     */
    std::vector<RS_WorldMessage> getMessages() {
        std::lock_guard lock(message_mutex);
        std::vector<RS_WorldMessage> result;
        result.swap(messages);
        return result;
    }

    /**
     * @brief Allocate a fresh message id for a mutator about to issue work.
     */
    RIndex allocateMessageId() {
        RIndex issued = message_id_counter.load(std::memory_order_relaxed);
        RIndex next;
        do {
            next = (issued >= RS_WORLD_MESSAGE_ID_ROTATE_THRESHOLD) ? 0 : issued + 1;
        } while (!message_id_counter.compare_exchange_weak(issued, next, std::memory_order_relaxed));
        return issued;
    }

    /**
     * @brief Append a message and wake all message waiters.
     */
    void postMessage(const RS_WorldMessage &message) {
        {
            std::lock_guard lock(message_mutex);
            messages.push_back(message);
        }
        message_cv.notify_all();
    }

    /**
     * @brief Append a plain-text message and wake all message waiters.
     */
    void postMessage(RIndex id, RSE_WorldMessageLevel level, std::string text) {
        postMessage({id, level, RS_WorldMessageData{RS_WorldMessageText{std::move(text)}}});
    }

    /**
     * @brief Block until a matching message arrives.
     * @details An id-specific wait scans messages already posted (the outcome may have arrived
     * first); an any-message wait (message_id == RS_NULL_INDEX) only considers messages posted after
     * this call. RSE_WORLD_MESSAGE_ANY_LEVEL matches any severity; a specific level matches
     * severity >= level. Messages drained by getMessages() are no longer available to this call.
     */
    RS_WorldMessage waitForMessage(
        RIndex message_id = RS_NULL_INDEX,
        RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_ANY_LEVEL
    ) {
        std::unique_lock lock(message_mutex);
        auto matches = [&](const RS_WorldMessage &m) {
            if (message_id != RS_NULL_INDEX && m.id != message_id) return false;
            if (level != RSE_WORLD_MESSAGE_ANY_LEVEL && m.level < level) return false;
            return true;
        };
        RSize next = (message_id == RS_NULL_INDEX) ? messages.size() : 0;
        RS_WorldMessage result;
        message_cv.wait(lock, [&] {
            for (; next < messages.size(); ++next) {
                if (matches(messages[next])) {
                    result = messages[next];
                    return true;
                }
            }
            return false;
        });
        return result;
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
