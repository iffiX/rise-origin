// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_BARRIER_H
#define RS_BARRIER_H

#include <condition_variable>

#include "rise_origin/utils/rs_def.h"

namespace rise {
/**
 * @brief Class representing a host side barrier for synchronizing multiple threads. This barrier
 * allows multiple threads to synchronize and wait until all threads reach a certain point.
 */
class RS_Barrier {
   public:
    /**
     * @brief Constructs a barrier with a specified number of threads.
     *
     * @param thread_num The number of threads to synchronize.
     */
    explicit RS_Barrier(RSize thread_num) : _thread_num(thread_num), _remain_num(thread_num) {
        assert(_thread_num != 0);
    }

    RS_Barrier(const RS_Barrier &barrier) = delete;
    RS_Barrier(RS_Barrier &&barrier) = delete;
    RS_Barrier &operator=(const RS_Barrier &barrier) = delete;
    RS_Barrier &operator=(RS_Barrier &&barrier) = delete;

    ~RS_Barrier() noexcept {
        // assert(_remain_num == 0);
    }

    /**
     * @brief Waits until all threads reach the barrier.
     *
     * This function blocks the calling thread until all threads reach the barrier.
     */
    void wait() {
        std::unique_lock lock(_mutex);

        assert(_thread_num != 0);

        if (--_remain_num == 0) {
            // last thread
            _remain_num = _thread_num;
            _condition.notify_all();
        } else {
            _condition.wait(lock);
        }
    }

   private:
    /// The total number of threads that need to arrive at the barrier.
    RSize _thread_num;

    /// How many threads are left to arrive before the barrier is considered "full."
    std::atomic<RSize> _remain_num;

    /// Mutex for controlling access to the barrier.
    std::mutex _mutex;

    /// Condition variable for signaling when all threads reach the barrier.
    std::condition_variable _condition;

#ifdef FRIEND_TEST
    FRIEND_TEST(RSBarrierTest, Constructor);
#endif
};

}  // namespace rise
#endif  // RS_BARRIER_H
