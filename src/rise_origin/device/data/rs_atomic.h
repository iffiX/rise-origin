// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_ATOMIC_H
#define RS_ATOMIC_H

#include "rise_origin/device/platform/rs_platform.h"

namespace rise {

namespace device {

template <typename T>
using RS_Atomic = rs_device_std::atomic<T>;

/**
 * @brief Spinlock implementation.
 * @note Spinlock is implemented using an atomic unsigned int. Its size is guaranteed to be
 * sizeof(unsigned int).
 */
struct RS_SpinLock {
#ifndef __CUDA_ARCH__
    rs_device_std::atomic<unsigned int> state{0};

    /**
     * @brief Clears the spinlock by resetting its state to 0.
     */
    HOST_DEVICE_INLINE_QUALIFIER void clear() {
        state.store(0, rs_device_std::memory_order_relaxed);
    }

    /**
     * @brief Acquires the spinlock.
     * @note Busy-waits until the lock is available.
     */
    HOST_DEVICE_INLINE_QUALIFIER void lock() {
        unsigned int expected = 0;
        while (!state.compare_exchange_weak(expected, 1, rs_device_std::memory_order_acquire)) {
            expected = 0;
        }
    }

    /**
     * @brief Attempts to acquire the spinlock without blocking.
     * @return true if the lock was successfully acquired, false otherwise.
     */
    HOST_DEVICE_INLINE_QUALIFIER bool try_lock() {
        unsigned int expected = 0;
        return state.compare_exchange_weak(expected, 1, rs_device_std::memory_order_acquire);
    }

    /**
     * @brief Releases the spinlock.
     */
    HOST_DEVICE_INLINE_QUALIFIER void unlock() {
        // Make sure all previous writes are visible
        rs_device_std::atomic_thread_fence(rs_device_std::memory_order_seq_cst);
        state.store(0, rs_device_std::memory_order_release);
    }
#else
    unsigned int state = 0;

    /**
     * @brief Clears the spinlock by resetting its state to 0.
     */
    DEVICE_INLINE_QUALIFIER void clear() { state = 0; }

    /**
     * @brief Acquires the spinlock.
     * @note Busy-waits using CUDA atomicCAS until the lock is available.
     */
    DEVICE_INLINE_QUALIFIER void lock() {
        while (atomicCAS(&state, 0, 1) != 0) {
        }
    }

    /**
     * @brief Attempts to acquire the spinlock without blocking.
     * @return true if the lock was acquired, false otherwise.
     */
    DEVICE_INLINE_QUALIFIER bool try_lock() { return atomicCAS(&state, 0, 1) == 0; }

    /**
     * @brief Releases the spinlock.
     */
    DEVICE_INLINE_QUALIFIER void unlock() {
        // Make sure all previous writes are visible
        __threadfence();
        atomicExch(&state, 0);
    }
#endif
};

/**
 * @brief TicketLock implementation.
 * @note TicketLock is implemented using two atomic unsigned int counters: served_ticket and
 * next_ticket. Both counters are initialized to 0. This lock ensures fairness by serving threads in
 * FIFO order.
 */
struct RS_TicketLock {
#ifndef __CUDA_ARCH__
    rs_device_std::atomic<unsigned int> served_ticket{0};
    rs_device_std::atomic<unsigned int> next_ticket{0};

    /**
     * @brief Clears the ticket lock by resetting both served_ticket and next_ticket to 0.
     */
    HOST_DEVICE_INLINE_QUALIFIER void clear() {
        served_ticket.store(0, rs_device_std::memory_order_relaxed);
        next_ticket.store(0, rs_device_std::memory_order_relaxed);
    }

    /**
     * @brief Acquires the ticket lock.
     * @note Each thread obtains a unique ticket and waits until its ticket is served.
     */
    HOST_DEVICE_INLINE_QUALIFIER void lock() {
        unsigned int my_ticket = next_ticket.fetch_add(1, rs_device_std::memory_order_relaxed);
        while (served_ticket.load(rs_device_std::memory_order_acquire) != my_ticket) {
            // busy-wait until our ticket is served
        }
    }

    /**
     * @brief Attempts to acquire the ticket lock without blocking.
     * @return true if the lock was acquired, false otherwise.
     */
    HOST_DEVICE_INLINE_QUALIFIER bool try_lock() {
        unsigned int current_served = served_ticket.load(rs_device_std::memory_order_relaxed);
        unsigned int current_next = next_ticket.load(rs_device_std::memory_order_relaxed);
        if (current_served == current_next) {
            if (next_ticket.compare_exchange_strong(
                    current_next, current_next + 1, rs_device_std::memory_order_acquire
                )) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Releases the ticket lock.
     * @note Increments the served_ticket counter to allow the next waiting thread to proceed.
     */
    HOST_DEVICE_INLINE_QUALIFIER void unlock() {
        // Make sure all previous writes are visible
        rs_device_std::atomic_thread_fence(rs_device_std::memory_order_seq_cst);
        served_ticket.fetch_add(1, rs_device_std::memory_order_release);
    }
#else
    unsigned int served_ticket = 0;
    unsigned int next_ticket = 0;

    /**
     * @brief Clears the ticket lock by resetting both served_ticket and next_ticket to 0.
     */
    DEVICE_INLINE_QUALIFIER void clear() {
        served_ticket = 0;
        next_ticket = 0;
    }

    /**
     * @brief Acquires the ticket lock.
     * @note Each thread obtains a ticket using atomicAdd and busy-waits until its ticket is served.
     */
    DEVICE_INLINE_QUALIFIER void lock() {
        unsigned int my_ticket = atomicAdd(&next_ticket, 1);
        while (atomicCAS(&served_ticket, my_ticket, my_ticket) != my_ticket) {
            // busy-wait until our ticket is served
        }
    }

    /**
     * @brief Attempts to acquire the ticket lock without blocking.
     * @return true if the lock was acquired, false otherwise.
     */
    DEVICE_INLINE_QUALIFIER bool try_lock() {
        unsigned int current_served = served_ticket;
        unsigned int current_next = next_ticket;
        if (current_served == current_next) {
            if (atomicCAS(&next_ticket, current_next, current_next + 1) == current_next)
                return true;
        }
        return false;
    }

    /**
     * @brief Releases the ticket lock.
     * @note Increments the served_ticket counter after ensuring all writes are visible.
     */
    DEVICE_INLINE_QUALIFIER void unlock() {
        // Make sure all previous writes are visible
        __threadfence();
        atomicAdd(&served_ticket, 1);
    }
#endif
};

}  // namespace device

}  // namespace rise

#endif  // RS_ATOMIC_H
