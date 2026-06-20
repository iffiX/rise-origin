// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_THREAD_POOL_H
#define RS_THREAD_POOL_H

/**
 * @note Adapted from BS::thread_pool (https://github.com/bshoshany/thread-pool)
 */

#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"

namespace rise {

RS_MAKE_EXCEPTION(RS_ThreadPoolDeadLockException, RS_RuntimeException);

/**
 * @brief A helper class to facilitate waiting for and/or getting the results of
 * multiple futures at once.
 *
 * @tparam T The return type of the futures.
 */
template <typename T>
class RS_MultiFuture : public std::vector<std::future<T>> {
   public:
    // Inherit all constructors from the base class `std::vector`.
    using std::vector<std::future<T>>::vector;

    // The copy constructor and copy assignment operator are deleted. The elements stored in a
    // `RS_MultiFuture` are futures, which cannot be copied.
    RS_MultiFuture(const RS_MultiFuture&) = delete;
    RS_MultiFuture& operator=(const RS_MultiFuture&) = delete;

    // The move constructor and move assignment operator are defaulted.
    RS_MultiFuture(RS_MultiFuture&&) = default;
    RS_MultiFuture& operator=(RS_MultiFuture&&) = default;

    /**
     * @brief Get the results from all the futures stored in this `RS_MultiFuture`, rethrowing any
     * stored exceptions.
     *
     * @return If the futures return `void`, this function returns `void` as well. Otherwise, it
     * returns a vector containing the results.
     */
    [[nodiscard]] std::conditional_t<std::is_void_v<T>, void, std::vector<T>> get() {
        if constexpr (std::is_void_v<T>) {
            for (std::future<T>& future : *this) future.get();
            return;
        } else {
            std::vector<T> results;
            results.reserve(this->size());
            for (std::future<T>& future : *this) results.push_back(future.get());
            return results;
        }
    }

    /**
     * @brief Check how many of the futures stored in this `RS_MultiFuture` are ready.
     *
     * @return The number of ready futures.
     */
    [[nodiscard]] size_t readyCount() const {
        size_t count = 0;
        for (const std::future<T>& future : *this) {
            if (future.wait_for(std::chrono::duration<double>::zero()) == std::future_status::ready)
                ++count;
        }
        return count;
    }

    /**
     * @brief Check if all the futures stored in this `RS_MultiFuture` are valid.
     *
     * @return `true` if all futures are valid, `false` if at least one of the futures is not valid.
     */
    [[nodiscard]] bool isValid() const {
        bool is_valid = true;
        for (const std::future<T>& future : *this) is_valid = is_valid && future.valid();
        return is_valid;
    }

    /**
     * @brief Wait for all the futures stored in this `RS_MultiFuture`.
     */
    void wait() const {
        for (const std::future<T>& future : *this) future.wait();
    }

    /**
     * @brief Wait for all the futures stored in this `RS_MultiFuture`, but stop waiting after the
     * specified duration has passed. This function first waits for the first future for the desired
     * duration. If that future is ready before the duration expires, this function waits for the
     * second future for whatever remains of the duration. It continues similarly until the duration
     * expires.
     *
     * @tparam Rep An arithmetic type representing the number of ticks to wait.
     * @tparam Period An `std::ratio` representing the length of each tick in seconds.
     * @param duration The amount of time to wait.
     * @return `true` if all futures have been waited for before the duration expired, `false`
     * otherwise.
     */
    template <typename Rep, typename Period>
    bool waitFor(const std::chrono::duration<Rep, Period>& duration) const {
        const std::chrono::time_point<std::chrono::steady_clock> start_time =
            std::chrono::steady_clock::now();
        for (const std::future<T>& future : *this) {
            future.wait_for(duration - (std::chrono::steady_clock::now() - start_time));
            if (duration < std::chrono::steady_clock::now() - start_time) return false;
        }
        return true;
    }

    /**
     * @brief Wait for all the futures stored in this `RS_MultiFuture`, but stop waiting after the
     * specified time point has been reached. This function first waits for the first future until
     * the desired time point. If that future is ready before the time point is reached, this
     * function waits for the second future until the desired time point. It continues similarly
     * until the time point is reached.
     *
     * @tparam Clock The type of the clock used to measure time.
     * @tparam Duration An `std::chrono::duration` type used to indicate the time point.
     * @param timeout_time The time point at which to stop waiting.
     * @return `true` if all futures have been waited for before the time point was reached, `false`
     * otherwise.
     */
    template <typename Clock, typename Duration>
    bool waitUntil(const std::chrono::time_point<Clock, Duration>& timeout_time) const {
        for (const std::future<T>& future : *this) {
            future.wait_until(timeout_time);
            if (timeout_time < std::chrono::steady_clock::now()) return false;
        }
        return true;
    }
};

/**
 * @brief A helper class to divide a range into blocks.
 */
class RS_ThreadLoopBlocks {
   public:
    /**
     * @brief Divider indices in [start, end) into approximately similar size of blocks.
     * The basic idea is to assign the quotient of (end - start) / block_num to each block,
     * then for remainder, we assign 1 to the first remainder number of blocks.
     *
     * @param start The first index in the range.
     * @param end The index after the last index in the range.
     * @param block_num The desired number of blocks to divide the range into.
     */
    RS_ThreadLoopBlocks(RIndex start, RIndex end, RSize block_num)
        : _start(start), _end(end), _block_num(block_num) {
        if (end > start) {
            RSize size = end - start;
            _block_size = RS_FLOOR_DIV(size, block_num);
            _remainder = size - block_num * _block_size;
        }
    }

    /**
     * @brief Get the first element index of a block.
     *
     * @param block_index The block index.
     * @return The first element index.
     */
    [[nodiscard]] RIndex start(RIndex block_index) const {
        return _start + block_index * _block_size +
               (block_index < _remainder ? block_index : _remainder);
    }

    /**
     * @brief Get the element index after the last of a block.
     *
     * @param block_index The block index.
     * @return The index after the last index.
     */
    [[nodiscard]] RIndex end(RIndex block_index) const {
        return RS_MIN(start(block_index + 1), _end);
    }

    /// @brief Get number of blocks
    [[nodiscard]] RSize size() const { return _block_num; }

   private:
    RIndex _start = RS_NULL_INDEX;
    RIndex _end = RS_NULL_INDEX;
    RSize _block_num = 0;
    RSize _block_size = 0;
    RSize _remainder = 0;
};

/**
 * @brief Thread pool.
 *
 */
class RS_ThreadPool {
   public:
    struct RS_SynchronizedThreadPoolWorker {
        inline static thread_local RIndex index = RS_NULL_INDEX;
        inline static thread_local RS_ThreadPool* pool = nullptr;
        inline static thread_local void* data = nullptr;
    };

    /**
     * @brief Construct a new thread pool. The number of threads will be the total number of
     * hardware threads available.
     */
    RS_ThreadPool() : RS_ThreadPool(0, [] {}, [] {}) {}

    /**
     * @brief Construct a new thread pool with the specified number of threads.
     *
     * @param thread_num The number of threads to use.
     */
    explicit RS_ThreadPool(const RSize thread_num) : RS_ThreadPool(thread_num, [] {}, [] {}) {}

    /**
     * @brief Construct a new thread pool with the specified initialization function and free
     * function.
     *
     * @param init_task An initialization function to run in each thread before it starts to execute
     * any submitted tasks.
     * @param free_task A free function to run in each thread before it terminates.
     */
    explicit
    RS_ThreadPool(const std::function<void()>& init_task, const std::function<void()>& free_task)
        : RS_ThreadPool(0, init_task, free_task) {}

    /**
     * @brief Construct a new thread pool with the specified number of threads and initialization
     * function.
     *
     * @param thread_num The number of threads to use.
     * @param init_task An initialization function to run in each thread before it starts to execute
     * any submitted tasks.
     * @param free_task A free function to run in each thread before it terminates.
     */
    RS_ThreadPool(
        const RSize thread_num,
        const std::function<void()>& init_task,
        const std::function<void()>& free_task
    )
        : _threads(std::make_unique<std::thread[]>(_determineThreadNum(thread_num))),
          _thread_num(_determineThreadNum(thread_num)) {
        _createThreads(init_task, free_task);
    }

    // The thread pool uses a mutex, which cannot be copied or moved.
    RS_ThreadPool(const RS_ThreadPool&) = delete;
    RS_ThreadPool(RS_ThreadPool&&) = delete;
    RS_ThreadPool& operator=(const RS_ThreadPool&) = delete;
    RS_ThreadPool& operator=(RS_ThreadPool&&) = delete;

    /**
     * @brief Destruct the thread pool. Waits for all tasks to complete, then destroys all threads.
     * Note that if the pool is paused, then any tasks still in the queue will never be executed.
     */
    ~RS_ThreadPool() {
        wait();
        _destroyThreads();
    }

    /**
     * @brief Get the number of threads in the pool.
     *
     * @return The number of threads.
     */
    [[nodiscard]] RSize size() const { return _thread_num; }

    /**
       * @brief Get the number of tasks currently waiting in the queue to be executed by the
       threads.
       *
       * @return The number of queued tasks.
       */
    [[nodiscard]] RSize getTaskQueuedNum() const { return _task_queue.size(); }

    /**
     * @brief Get the number of tasks currently being executed by the threads.
     * @note Include init/free tasks.
     * @return The number of running tasks.
     */
    [[nodiscard]] RSize getTaskRunningNum() const { return _task_running_num; }

    /**
     * @brief Get the total number of tasks waiting in the queue or currently being executed
     * by the threads.
     * @note Include init/free tasks.
     * @return The total number of tasks.
     */
    [[nodiscard]] RSize getTaskTotalNum() const { return _task_running_num + _task_queue.size(); }

    /**
     * @brief Clear remaining tasks in the queue. will not affect the running tasks.
     */
    void clear() {
        std::lock_guard lock(_task_mutex);
        while (!_task_queue.empty()) _task_queue.pop();
    }

    /**
     * @brief Submit an asynchronous task with no arguments and no return value into the task queue.
     *
     * @tparam Function The type of the function.
     * @param task The function to push.
     */
    template <typename Function>
    void detachTask(Function&& task) {
        {
            // After submitting the task, we need to release lock
            // to handle it over to worker threads
            std::lock_guard lock(_task_mutex);
            _task_queue.emplace(std::forward<Function>(task));
        }
        _task_available_cv.notify_one();
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block
     * separately to the queue.
     * @note Does not return a `future`, so the user must use `wait()` or some other method to
     * ensure that the loop finishes executing.
     *
     * @tparam Function The type of the function to loop through.
     * @param start The start index in the loop.
     * @param end The end index in the loop. Indices in [start, end) will be executed.
     * @param loop A function that will be called once per index, signature should be
     * (RIndex idx) -> void {...}
     * @param block_num The maximum number of blocks to split the loop into. The default is 0,
     * which means the number of blocks will be equal to the number of threads in the pool. If
     * Indices is less than the number of threads in the pool, only threads of the number of
     * indices will be used.
     */
    template <typename Function>
    void detachLoop(RIndex start, RIndex end, Function&& loop, RSize block_num = 0) {
        if (end > start) {
            if (block_num == 0) block_num = RS_MIN(_thread_num, end - start);
            RS_ThreadLoopBlocks blocks(start, end, block_num);

            for (RIndex block = 0; block < block_num; ++block) {
                RIndex block_start = blocks.start(block);
                RIndex block_end = blocks.end(block);
                if (block_end > block_start)
                    detachTask([loop_ = std::forward<Function>(loop), block_start, block_end] {
                        for (RIndex i = block_start; i < block_end; ++i) loop_(i);
                    });
            }
        }
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block
     * separately to the queue. Similar to detachLoop, but block function allow users to have local
     * variables in the block function, like a local cache in the block for loop.
     * @note Does not return a `future`, so the user must use `wait()` or some other method to
     * ensure that the loop finishes executing.
     *
     * @tparam Function The type of the function to loop through.
     * @param start The start index in the loop.
     * @param end The end index in the loop. Indices in [start, end) will be executed.
     * @param block_func A function that will be called once per block, should iterate through given
     * block range [block_start, block_end), signature should be
     * (RIndex block_start, RIndex block_end) -> void {...}
     * @param block_num The maximum number of blocks to split the loop into. The default is 0,
     * which means the number of blocks will be equal to the number of threads in the pool. If
     * Indices is less than the number of threads in the pool, only threads of the number of
     * indices will be used.
     */
    template <typename Function>
    void detachBlocks(RIndex start, RIndex end, Function&& block_func, RSize block_num = 0) {
        if (end > start) {
            if (block_num == 0) block_num = RS_MIN(_thread_num, end - start);
            RS_ThreadLoopBlocks blocks(start, end, block_num);

            for (RIndex block = 0; block < block_num; ++block) {
                RIndex block_start = blocks.start(block);
                RIndex block_end = blocks.end(block);
                if (block_end > block_start)
                    detachTask([block_func_ = std::forward<Function>(block_func),
                                block_start,
                                block_end] { block_func_(block_start, block_end); });
            }
        }
    }

    /**
     * @brief Submit an asynchronous task with no arguments into the task queue.
     * Returns a std::future<return_type> which can be used to get the returned value.
     *
     * @tparam Function The type of the function.
     * @param task The function to push.
     * @return A future to be used later to wait for the function to finish
     * executing and/or obtain result.
     */
    template <typename Function, typename ReturnType = std::invoke_result_t<std::decay_t<Function>>>
    [[nodiscard]] std::future<ReturnType> submitTask(Function&& task) {
        const std::shared_ptr<std::promise<ReturnType>> task_promise =
            std::make_shared<std::promise<ReturnType>>();
        detachTask([task = std::forward<Function>(task), task_promise] {
            try {
                if constexpr (std::is_void_v<ReturnType>) {
                    task();
                    task_promise->set_value();
                } else {
                    task_promise->set_value(task());
                }
            } catch (...) {
                try {
                    task_promise->set_exception(std::current_exception());
                } catch (...) {
                }
            }
        });
        return task_promise->get_future();
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block
     * separately to the queue. Returns a multi future containing the futures for all the blocks,
     * which can be waited on.
     *
     * @tparam Function The type of the function to loop through.
     * @param start The start index in the loop.
     * @param end The end index in the loop. Indices in [start, end) will be executed.
     * @param loop A function that will be called once per index, signature should be
     * (RIndex idx) -> void {...}
     * @param block_num The maximum number of blocks to split the loop into. The default is 0,
     * which means the number of blocks will be equal to the number of threads in the pool. If
     * Indices is less than the number of threads in the pool, only threads of the number of
     * indices will be used.
     */
    template <typename Function>
    [[nodiscard]] RS_MultiFuture<void>
    submitLoop(RIndex start, RIndex end, Function&& loop, RSize block_num = 0) {
        if (end > start) {
            if (block_num == 0) block_num = RS_MIN(_thread_num, end - start);
            RS_ThreadLoopBlocks blocks(start, end, block_num);

            RS_MultiFuture<void> future;
            future.reserve(block_num);

            for (RIndex block = 0; block < block_num; ++block) {
                RIndex block_start = blocks.start(block);
                RIndex block_end = blocks.end(block);
                if (block_end > block_start) {
                    future.push_back(
                        submitTask([loop_ = std::forward<Function>(loop), block_start, block_end] {
                            for (RIndex i = block_start; i < block_end; ++i) loop_(i);
                        })
                    );
                }
            }
            return future;
        }
        return {};
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block
     * separately to the queue. Returns a multi future containing the futures for all the blocks,
     * which can be waited on. Similar to submitLoop, but block function allow users to have local
     * variables in the block function, like a local cache in the block for loop.
     *
     * @tparam Function The type of the function to loop through.
     * @param start The start index in the loop.
     * @param end The end index in the loop. Indices in [start, end) will be executed.
     * @param block_func A function that will be called once per block, should iterate through given
     * block range [block_start, block_end), signature should be
     * (RIndex block_start, RIndex block_end) -> void {...}
     * @param block_num The maximum number of blocks to split the loop into. The default is 0,
     * which means the number of blocks will be equal to the number of threads in the pool. If
     * Indices is less than the number of threads in the pool, only threads of the number of
     * indices will be used.
     */
    template <typename Function>
    [[nodiscard]] RS_MultiFuture<void>
    submitBlocks(RIndex start, RIndex end, Function&& block_func, RSize block_num = 0) {
        if (end > start) {
            if (block_num == 0) block_num = RS_MIN(_thread_num, end - start);
            RS_ThreadLoopBlocks blocks(start, end, block_num);

            RS_MultiFuture<void> future;
            future.reserve(block_num);

            for (RIndex block = 0; block < block_num; ++block) {
                RIndex block_start = blocks.start(block);
                RIndex block_end = blocks.end(block);
                if (block_end > block_start) {
                    future.push_back(submitTask([block_func = std::forward<Function>(block_func),
                                                 block_start,
                                                 block_end] { block_func(block_start, block_end); })
                    );
                }
            }
            return future;
        }
        return {};
    }

    /**
     * @brief Submit a sequence of tasks enumerated by indices to the queue.
     * Returns a `multi_future` that contains the futures for all the tasks which
     * can be used to get returned values of all tasks
     *
     * @tparam Function The type of the function to loop through.
     * @param start The start index in the loop.
     * @param end The end index in the loop. Indices in [start, end) will be executed.
     * @param sequence A function that will be called once per index, signature should be
     * (RIndex idx) -> any {...}
     * @return A `multi_future` that can be used to wait for all the tasks to finish. If the
     * sequence function returns a value, the `multi_future` can also be used to obtain the
     * values returned by each task.
     */
    template <
        typename Function,
        typename ReturnType = std::invoke_result_t<std::decay_t<Function>, RIndex>>
    [[nodiscard]] RS_MultiFuture<ReturnType>
    submitTaskSequence(RIndex start, RIndex end, Function&& sequence) {
        if (end > start) {
            RS_MultiFuture<ReturnType> future;
            future.reserve(end - start);
            for (RIndex i = start; i < end; ++i)
                future.push_back(submitTask([sequence = std::forward<Function>(sequence), i] {
                    return sequence(i);
                }));
            return future;
        }
        return {};
    }

    /**
     * @brief Wait for tasks to be completed. This function waits for all tasks, both those that
     * are currently running in the threads and those that are still waiting in the queue.
     *
     * @throws RS_ThreadPoolDeadlockException if called from within a thread of the same pool,
     * which would result in a deadlock.
     */
    void wait() {
        if (RS_SynchronizedThreadPoolWorker::pool == this)
            RS_THROW(RS_ThreadPoolDeadLockException());
        std::unique_lock lock(_task_mutex);
        _is_waiting = true;
        _task_done_cv.wait(lock, [this] { return _task_running_num == 0 && _task_queue.empty(); });
        _is_waiting = false;
    }

    /**
     * @brief Wait for tasks to be completed, but stop waiting after the specified duration
     has
     * passed.
     *
     * @tparam Rep An arithmetic type representing the number of ticks to wait.
     * @tparam Period An `std::ratio` representing the length of each tick in seconds.
     * @param duration The amount of time to wait.
     * @return `true` if all tasks finished running, `false` if the duration expired but some
     tasks
     * are still running.
     *
     * @throws RS_ThreadPoolDeadlockException if called from within a thread of the same
     pool,
     * which would result in a deadlock.
     */
    template <typename Rep, typename Period>
    bool waitFor(const std::chrono::duration<Rep, Period>& duration) {
        if (RS_SynchronizedThreadPoolWorker::pool == this)
            RS_THROW(RS_ThreadPoolDeadLockException());
        std::unique_lock lock(_task_mutex);
        _is_waiting = true;
        const bool status = _task_done_cv.wait_for(lock, duration, [this] {
            return _task_running_num == 0 && _task_queue.empty();
        });
        _is_waiting = false;
        return status;
    }

    /**
     * @brief Wait for tasks to be completed, but stop waiting after the specified time point
     has
     * been reached.
     *
     * @tparam Clock The type of the clock used to measure time.
     * @tparam Duration An `std::chrono::duration` type used to indicate the time point.
     * @param timeout_time The time point at which to stop waiting.
     * @return `true` if all tasks finished running, `false` if the time point was reached
     but some
     * tasks are still running.
     *
     * @throws RS_ThreadPoolDeadlockException if called from within a thread of the same
     pool,
     * which would result in a deadlock.
     */
    template <typename Clock, typename Duration>
    bool waitUntil(const std::chrono::time_point<Clock, Duration>& timeout_time) {
        if (RS_SynchronizedThreadPoolWorker::pool == this)
            RS_THROW(RS_ThreadPoolDeadLockException());
        std::unique_lock lock(_task_mutex);
        _is_waiting = true;
        const bool status = _task_done_cv.wait_until(lock, timeout_time, [this] {
            return _task_running_num == 0 && _task_queue.empty();
        });
        _is_waiting = false;
        return status;
    }

   private:
    /**
     * @brief A mutex to ensure atomic access to pool task queue for both external callers
     * and worker threads
     */
    std::mutex _task_mutex = {};

    /// @brief A condition variable to notify `worker()` that a new task has become available.
    std::condition_variable _task_available_cv = {};

    /// @brief A condition variable to notify `wait()` that the tasks are done.
    std::condition_variable _task_done_cv = {};

    /// @brief Task queue
    std::queue<std::function<void()>> _task_queue = {};

    /// @brief The number of tasks currently running.
    RSize _task_running_num = 0;

    /// @brief A smart pointer to manage the memory allocated for the threads.
    std::unique_ptr<std::thread[]> _threads = nullptr;

    /// @brief The number of threads in the pool.
    RSize _thread_num = 0;

    /**
     * @brief A flag indicating that `wait()` is active and expects to be notified whenever a
     * task is done.
     */
    bool _is_waiting = false;

    /**
     * @brief A flag indicating to the workers to keep running. When set to `false`, the workers
     * terminate permanently.
     */
    bool _is_running = false;

    /**
     * @brief Create the threads in the pool and assign a worker to each thread.
     *
     * @param init_task An initialization function to run in each thread before it starts to
     * execute any submitted tasks.
     * @param free_task A free function to run in each thread before it terminates.
     */
    void
    _createThreads(const std::function<void()>& init_task, const std::function<void()>& free_task) {
        {
            std::lock_guard lock(_task_mutex);
            _is_running = true;
        }
        _task_running_num = _thread_num;
        for (RIndex i = 0; i < _thread_num; ++i) {
            _threads[i] = std::thread(&RS_ThreadPool::_worker, this, i, init_task, free_task);
        }
    }

    /**
     * @brief Destroy the threads in the pool.
     */
    void _destroyThreads() {
        {
            // After setting the _is_running to false, we need to release lock
            // to handle it over to worker threads
            std::lock_guard lock(_task_mutex);
            _is_running = false;
        }
        _task_available_cv.notify_all();
        for (RIndex i = 0; i < _thread_num; ++i) {
            _threads[i].join();
        }
    }

    /**
     * @brief Determine how many threads the pool should have.
     *
     * @param thread_num If `thread_num` is a positive number, then the pool will be created
     * with this number of threads. If it is 0, then the pool will be created with the total
     * number of hardware threads available, as obtained from
     * `std::thread::hardware_concurrency()`. If the latter returns zero for some reason, then
     * the pool will be created with just one thread.
     * @return The number of threads to use for constructing the pool.
     */
    [[nodiscard]] static RSize _determineThreadNum(const RSize thread_num) {
        if (thread_num > 0) return thread_num;
        if (std::thread::hardware_concurrency() > 0) return std::thread::hardware_concurrency();
        return 1;
    }

    /**
     * @brief A worker function to be assigned to each thread in the pool.
     *
     * @param index The index of this thread.
     * @param init_task An initialization function to run in this thread before it starts to
     * execute any submitted tasks.
     * @param free_task A free function to run in each thread before it terminates.
     */
    void _worker(
        const RIndex index,
        const std::function<void()>& init_task,
        const std::function<void()>& free_task
    ) {
        RS_SynchronizedThreadPoolWorker::index = index;
        RS_SynchronizedThreadPoolWorker::pool = this;
        init_task();

        std::unique_lock lock(_task_mutex);
        while (true) {
            --_task_running_num;
            lock.unlock();
            if (_is_waiting && _task_running_num == 0 && _task_queue.empty())
                _task_done_cv.notify_all();

            lock.lock();
            _task_available_cv.wait(lock, [this] { return !_is_running || !_task_queue.empty(); });
            if (!_is_running) {
                lock.unlock();
                break;
            }
            auto task = std::move(_task_queue.front());
            _task_queue.pop();
            ++_task_running_num;
            lock.unlock();

            task();

            lock.lock();
        }
        lock.lock();
        ++_task_running_num;
        lock.unlock();
        free_task();
        lock.lock();
        --_task_running_num;
        lock.unlock();
        RS_SynchronizedThreadPoolWorker::index = RS_NULL_INDEX;
        RS_SynchronizedThreadPoolWorker::pool = nullptr;
    }
};

}  // namespace rise

#endif  // RS_THREAD_POOL_H
