#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

#include "rise_origin/utils/concurrency/rs_barrier.h"

using namespace rise;

/*****************************************************************
 * Test Summary:
 *
 * RS_Barrier lets a fixed number of threads meet at a common
 * point. Every thread blocks in wait until the last one arrives,
 * then all of them are released together. The barrier resets on
 * its own so it can be used again for the next round.
 *
 * - Construction
 *   Build a barrier for a given thread count
 *
 * - Single and Multi Thread Wait
 *   One thread passes through, many threads meet and release
 *
 * - Reuse
 *   The same barrier serves many rounds in a row
 *
 * - Stress
 *   Random delays, high thread count, and rapid reuse
 *****************************************************************/

/*****************************************************************
 * Test cases for RS_Barrier construction and basic wait
 *
 * Test cases:
 * 1. Construct with several thread counts.
 * 2. A single thread passes through right away.
 * 3. Many threads meet at the barrier and all finish.
 *****************************************************************/

TEST(RS_BarrierTest, Constructor) {
    RS_Barrier barrier(5);
    EXPECT_NO_THROW(RS_Barrier(1));
    EXPECT_NO_THROW(RS_Barrier(100));
}

TEST(RS_BarrierTest, SingleThreadWait) {
    RS_Barrier barrier(1);

    // A single thread should pass through right away.
    auto start_time = std::chrono::steady_clock::now();
    barrier.wait();
    auto end_time = std::chrono::steady_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    EXPECT_LT(elapsed.count(), 100);
}

TEST(RS_BarrierTest, BasicMultiThreadWait) {
    const int num_threads = 5;
    RS_Barrier barrier(num_threads);

    std::atomic<int> threads_started{0};
    std::atomic<int> threads_finished{0};
    std::vector<std::thread> threads;

    auto start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&barrier, &threads_started, &threads_finished]() {
            threads_started++;
            barrier.wait();
            threads_finished++;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::steady_clock::now();

    EXPECT_EQ(threads_started.load(), num_threads);
    EXPECT_EQ(threads_finished.load(), num_threads);

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    EXPECT_LT(elapsed.count(), 1000);
}

/*****************************************************************
 * Test cases for RS_Barrier reuse across rounds
 *
 * Test cases:
 * 1. The same barrier serves a fixed number of rounds.
 *****************************************************************/

TEST(RS_BarrierTest, BarrierReuse) {
    const int num_threads = 3;
    const int num_cycles = 5;
    RS_Barrier barrier(num_threads);

    std::atomic<int> total_completions{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&barrier, &total_completions, num_cycles]() {
            for (int cycle = 0; cycle < num_cycles; ++cycle) {
                barrier.wait();
                total_completions++;
                // Small delay between rounds.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(total_completions.load(), num_threads * num_cycles);
}

/*****************************************************************
 * Test cases for RS_Barrier under stress
 *
 * Test cases:
 * 1. Random delays before each thread reaches the barrier.
 * 2. A high thread count meets at one barrier.
 * 3. Rapid reuse over many rounds.
 *****************************************************************/

TEST(RS_BarrierStressTest, RandomDelayStress) {
    const int num_threads = 8;
    RS_Barrier barrier(num_threads);

    std::atomic<int> threads_finished{0};
    std::vector<std::thread> threads;
    std::random_device rd;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&barrier, &threads_finished, seed = rd()]() {
            std::mt19937 gen(seed);
            std::uniform_int_distribution<int> dis(1, 50);

            // Random delay before reaching the barrier.
            std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen)));

            barrier.wait();
            threads_finished++;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(threads_finished.load(), num_threads);
}

TEST(RS_BarrierStressTest, HighThreadCount) {
    const int num_threads = 50;
    RS_Barrier barrier(num_threads);

    std::atomic<int> threads_finished{0};
    std::vector<std::thread> threads;

    auto start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&barrier, &threads_finished]() {
            barrier.wait();
            threads_finished++;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::steady_clock::now();

    EXPECT_EQ(threads_finished.load(), num_threads);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    EXPECT_LT(elapsed.count(), 5000);
}

TEST(RS_BarrierStressTest, RapidBarrierReuse) {
    const int num_threads = 6;
    const int num_rapid_cycles = 100;
    RS_Barrier barrier(num_threads);

    std::atomic<int> total_completions{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&barrier, &total_completions, num_rapid_cycles]() {
            for (int cycle = 0; cycle < num_rapid_cycles; ++cycle) {
                barrier.wait();
                total_completions++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(total_completions.load(), num_threads * num_rapid_cycles);
}
