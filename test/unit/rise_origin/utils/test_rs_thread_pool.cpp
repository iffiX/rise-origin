#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "rise_origin/utils/rs_thread_pool.h"

using namespace rise;
using namespace std::chrono_literals;

// Test basic task execution using detachTask.
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

// Test parallel loop execution using detachLoop.
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

// Test submitting a mix of detachTask and detachLoop tasks.
TEST(RS_SynchronizedThreadPoolTest, MultipleTasksTest) {
    RS_ThreadPool pool;
    std::atomic<int> counter{0};
    const int num_tasks = 50;

    for (int i = 0; i < num_tasks; i++) {
        if (i % 2 == 0) {
            // Simple task increments counter by 1.
            pool.detachTask([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
        } else {
            // Loop task runs 10 iterations, each incrementing counter.
            pool.detachLoop(i, i + 10, [&counter](int /*index*/) {
                counter.fetch_add(1, std::memory_order_relaxed);
            });
        }
    }

    pool.wait();
    // Even tasks: 25 tasks * 1 = 25
    // Odd tasks: 25 tasks * 10 = 250
    // Total expected counter = 25 + 250 = 275
    EXPECT_EQ(counter.load(), 275);
}

// Test the waitFor method with a timeout that expires before the task completes.
TEST(RS_SynchronizedThreadPoolTest, WaitForTimeoutTest) {
    RS_ThreadPool pool;
    std::atomic<bool> task_executed{false};

    // Submit a task that sleeps for 500 ms before setting task_executed.
    pool.detachTask([&task_executed]() {
        std::this_thread::sleep_for(500ms);
        task_executed = true;
    });

    // Wait for only 100 ms; the task should not finish in that time.
    bool finished = pool.waitFor(100ms);
    EXPECT_FALSE(finished);

    // Wait for the task to complete.
    pool.wait();
    EXPECT_TRUE(task_executed.load());
}

// Test the waitUntil method by setting a timeout point.
TEST(RS_SynchronizedThreadPoolTest, WaitUntilTest) {
    RS_ThreadPool pool;
    std::atomic<int> counter{0};

    // Submit a task that increments the counter 5 times with a 100 ms delay between increments.
    pool.detachTask([&counter]() {
        for (int i = 0; i < 5; i++) {
            std::this_thread::sleep_for(100ms);
            counter.fetch_add(1, std::memory_order_relaxed);
        }
    });

    auto timeout_time = std::chrono::steady_clock::now() + 300ms;
    bool finished = pool.waitUntil(timeout_time);
    // The task should not finish within 300ms since it needs ~500ms total.
    EXPECT_FALSE(finished);

    // Now wait until the task is fully finished.
    pool.wait();
    EXPECT_EQ(counter.load(), 5);
}
