#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "rise_origin/utils/concurrency/rs_thread_pool.h"

using namespace rise;
using namespace std::chrono_literals;

/*****************************************************************
 * Test Summary:
 *
 * RS_ThreadPool runs submitted work on a fixed set of worker
 * threads. Tasks can be detached when no result is needed, or
 * submitted to get a future for the result. Loops are split into
 * blocks and spread across the threads.
 *
 * - Detached Work
 *   detachTask, detachLoop, and a mix of both
 *
 * - Waiting
 *   wait, waitFor with a timeout, waitUntil with a time point
 *
 * - Submitted Work
 *   submitTask returning a future with a result
 *****************************************************************/

/*****************************************************************
 * Test cases for detached task and loop execution
 *
 * Test cases:
 * 1. Detach many small tasks and count them.
 * 2. Detach a loop and count every index.
 * 3. Mix detached tasks and loops.
 *****************************************************************/

TEST(RS_SynchronizedThreadPoolTest, BasicTaskExecution) {
    RS_ThreadPool pool;
    std::atomic<int> counter{0};
    const int num_tasks = 10;

    for (int i = 0; i < num_tasks; i++) {
        pool.detachTask([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
    }

    pool.wait();
    EXPECT_EQ(counter.load(), num_tasks);
}

TEST(RS_SynchronizedThreadPoolTest, DetachLoopTest) {
    RS_ThreadPool pool;
    const int loop_end = 100;
    std::atomic<int> sum{0};

    pool.detachLoop(0, loop_end, [&sum](int index) {
        sum.fetch_add(1, std::memory_order_relaxed);
    });

    pool.wait();
    EXPECT_EQ(sum.load(), loop_end);
}

TEST(RS_SynchronizedThreadPoolTest, MultipleTasksTest) {
    RS_ThreadPool pool;
    std::atomic<int> counter{0};
    const int num_tasks = 50;

    for (int i = 0; i < num_tasks; i++) {
        if (i % 2 == 0) {
            // Simple task increments counter by one.
            pool.detachTask([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
        } else {
            // Loop task runs ten times, each one increments counter.
            pool.detachLoop(i, i + 10, [&counter](int /*index*/) {
                counter.fetch_add(1, std::memory_order_relaxed);
            });
        }
    }

    pool.wait();
    // Even tasks: 25 tasks times 1 is 25.
    // Odd tasks: 25 tasks times 10 is 250.
    // Total expected counter is 25 plus 250 is 275.
    EXPECT_EQ(counter.load(), 275);
}

/*****************************************************************
 * Test cases for waiting with a timeout
 *
 * Test cases:
 * 1. waitFor returns false when the task is still running.
 * 2. waitUntil returns false when the time point is reached first.
 *****************************************************************/

TEST(RS_SynchronizedThreadPoolTest, WaitForTimeoutTest) {
    RS_ThreadPool pool;
    std::atomic<bool> is_task_executed{false};

    // Submit a task that sleeps for 500 ms before setting the flag.
    pool.detachTask([&is_task_executed]() {
        std::this_thread::sleep_for(500ms);
        is_task_executed = true;
    });

    // Wait for only 100 ms, the task should not finish in that time.
    bool is_finished = pool.waitFor(100ms);
    EXPECT_FALSE(is_finished);

    // Wait for the task to complete.
    pool.wait();
    EXPECT_TRUE(is_task_executed.load());
}

TEST(RS_SynchronizedThreadPoolTest, WaitUntilTest) {
    RS_ThreadPool pool;
    std::atomic<int> counter{0};

    // Submit a task that increments the counter five times with a 100 ms gap.
    pool.detachTask([&counter]() {
        for (int i = 0; i < 5; i++) {
            std::this_thread::sleep_for(100ms);
            counter.fetch_add(1, std::memory_order_relaxed);
        }
    });

    auto timeout_time = std::chrono::steady_clock::now() + 300ms;
    bool is_finished = pool.waitUntil(timeout_time);
    // The task should not finish within 300 ms since it needs about 500 ms total.
    EXPECT_FALSE(is_finished);

    // Now wait until the task is fully finished.
    pool.wait();
    EXPECT_EQ(counter.load(), 5);
}

/*****************************************************************
 * Test cases for submitted work with a future
 *
 * Test cases:
 * 1. submitTask returns a future that yields the task result.
 *****************************************************************/

TEST(RS_SynchronizedThreadPoolTest, SubmitTaskExecution) {
    RS_ThreadPool pool;
    const int num_tasks = 10;
    std::vector<int> results(num_tasks, 0);

    for (int i = 0; i < num_tasks; i++) {
        EXPECT_EQ(
            pool.submitTask([&results, i]() {
                    results[i] = i;
                    return i;
                })
                .get(),
            i
        );
    }

    pool.wait();
    for (int i = 0; i < num_tasks; i++) {
        EXPECT_EQ(results[i], i);
    }
}
