// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CPU_H
#define RS_CPU_H

#include <boost/core/demangle.hpp>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <string>

#include "rise_origin/device/platform/rs_exec_trace.h"
#include "rise_origin/utils/concurrency/rs_cpu_block.h"
#include "rise_origin/utils/concurrency/rs_thread_pool.h"
#include "rise_origin/utils/memory/rs_memory.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_env_var.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/time/rs_timer.h"

namespace rise {

namespace device {

class RS_MemoryCPUProvider final : public RS_MemoryProvider {
   public:
    std::shared_ptr<std::recursive_mutex> mutex = nullptr;
    std::shared_ptr<RS_ThreadPool> pool = nullptr;
    RS_MemoryPtr shared_mem;
    RS_MemoryPtr stack_mem;
    std::vector<RS_CPUBlock> cpu_blocks;
    std::vector<RS_CPUGridBlock> host_grid_blocks;

    RS_MemoryCPUProvider() {};

    RS_MemoryCPUProvider(RSize thread_num) { init(thread_num); }

    void init(RSize thread_num) {
        if (mutex) {
            RS_THROW(RS_ReinitializedException("CPU provider is already initialized"));
        }

        type = RSE_MEMORY_PROVIDER_TYPE_CPU;
        device = {RSE_DEVICE_TYPE_CPU, 0};

        mutex = std::make_shared<std::recursive_mutex>();
        if (thread_num == 0) {
            pool = std::make_shared<RS_ThreadPool>();
        } else {
            pool = std::make_shared<RS_ThreadPool>(thread_num);
        }
        shared_mem = RS_MemoryPtr();
        stack_mem = RS_MemoryPtr();
    };

    void free() override {
        // Not initialized, no op
        if (!mutex) return;

        mutex->lock();
        // Prevent further execution and wait for all tasks to finish
        pool->wait();

        // Free memory before resetting pool (since free them will wait for pool to finish)
        shared_mem.free();
        stack_mem.free();

        pool.reset();
        cpu_blocks.clear();
        host_grid_blocks.clear();

        mutex->unlock();
        mutex.reset();
        device = {RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX};
    }

    RS_MemoryPtr allocate(RBigSize size, bool sync = true) override {
        if (!mutex) RS_THROW(RS_UninitializedException("CPU provider is not initialized"));

        if (size == 0) return {nullptr, 0, 0, device, this};

        std::lock_guard<std::recursive_mutex> lock(*mutex);
        // Make sure malloc task is executed after all previous tasks
        pool->wait();
        RS_Timer exec_timer;
        void *ptr = std::malloc(size);

        if (!ptr)
            RS_THROW(
                RS_DeviceMemoryException("Failed to allocate device memory on the CPU provider")
            );

        if (isExecLogging())
            execTracePrint(__FILE__, __LINE__, format("malloc(size={})", size), &exec_timer);

        return {ptr, size, 0, device, this};
    }

    RS_MemoryPtr allocateHost(RBigSize size, bool sync = true) override {
        if (!mutex) RS_THROW(RS_UninitializedException("CPU provider is not initialized"));

        if (size == 0) return {nullptr, 0, 0, device, this};

        std::lock_guard<std::recursive_mutex> lock(*mutex);
        // Make sure malloc task is executed after all previous tasks
        pool->wait();

        RS_Timer exec_timer;
        void *ptr = std::malloc(size);
        if (!ptr)
            RS_THROW(RS_HostMemoryException("Failed to allocate host memory on the CPU provider"));

        if (isExecLogging())
            execTracePrint(__FILE__, __LINE__, format("malloc(size={})", size), &exec_timer);

        return {ptr, size, 0, device, this};
    }

    void deallocate(const RS_MemoryPtr &ptr, bool sync = true) override {
        if (!mutex) RS_THROW(RS_UninitializedException("CPU provider is not initialized"));

        if (ptr.empty()) return;

        if (ptr.provider() != this)
            RS_THROW(RS_HostExecutionException("Cannot free memory allocated by another provider"));

        std::lock_guard<std::recursive_mutex> lock(*mutex);
        // Make sure free task is executed after all previous tasks
        pool->wait();
        RS_Timer exec_timer;
        std::free(ptr.data);

        if (isExecLogging())
            execTracePrint(
                __FILE__,
                __LINE__,
                format("free(ptr={})", static_cast<void *>(ptr.data)),
                &exec_timer
            );
    }

    void set(
        const RS_MemoryPtr &ptr,
        int value,
        RBigSize size,
        RBigSize offset = 0,
        bool sync = true
    ) override {
        if (!mutex) RS_THROW(RS_UninitializedException("CPU provider is not initialized"));

        if (size == 0) return;

        if (ptr.provider() != this)
            RS_THROW(RS_HostExecutionException("Cannot set memory allocated by another provider"));

        std::lock_guard<std::recursive_mutex> lock(*mutex);
        // Make sure memset task is executed after all previous tasks
        pool->wait();
        RS_Timer exec_timer;
        std::memset((ptr + offset).data, value, size);

        if (isExecLogging())
            execTracePrint(
                __FILE__,
                __LINE__,
                format(
                    "memset(ptr={}, value={}, count={})",
                    static_cast<void *>((ptr + offset).data),
                    value,
                    size
                ),
                &exec_timer
            );
    }

    void copy(
        const RS_MemoryPtr &dst,
        const RS_ConstMemoryPtr &src,
        RBigSize size,
        bool sync = true,
        bool ordered = true
    ) override {
        if (!mutex) RS_THROW(RS_UninitializedException("CPU provider is not initialized"));

        if (size == 0) return;

        if (dst.provider() != this && src.provider() != this)
            RS_THROW(
                RS_HostExecutionException("One end of the copy must be allocated by this provider")
            );

        if (dst.device().type != RSE_DEVICE_TYPE_CPU || src.device().type != RSE_DEVICE_TYPE_CPU)
            RS_THROW(RS_HostExecutionException(
                "Cannot perform copy between dst device type {} index {} and src device type "
                "{} "
                "index {}",
                enumToString(dst.device().type),
                dst.device().id,
                enumToString(src.device().type),
                src.device().id
            ));

        std::lock_guard<std::recursive_mutex> lock(*mutex);
        // Make sure memcpy task is executed after all previous tasks
        pool->wait();
        RS_Timer exec_timer;
        std::memcpy(dst.data, src.data, size);

        if (isExecLogging())
            execTracePrint(
                __FILE__,
                __LINE__,
                format(
                    "memcpy(dst={}, src={}, count={})",
                    static_cast<void *>(dst.data),
                    static_cast<const void *>(src.data),
                    size
                ),
                &exec_timer
            );
    }

    void
    sync(bool sync_main = true, bool sync_copy_to = true, bool sync_copy_from = true) override {
        if (!mutex) RS_THROW(RS_UninitializedException("CPU provider is not initialized"));

        std::lock_guard<std::recursive_mutex> lock(*mutex);
        RS_Timer exec_timer;
        pool->wait();

        if (isExecLogging()) execTracePrint(__FILE__, __LINE__, "sync()", &exec_timer);
    }

    template <typename ComputeFunction, bool Debug = false, typename... Args>
    void run(RS_MemoryProviderRunConfig config, Args... args) {
        if (!mutex) RS_THROW(RS_UninitializedException("CPU provider is not initialized"));

        if (!config.validate())
            RS_THROW(RS_InvalidArgumentException("Invalid config: {}", config.toString()));

        if (config.thread_num == 0) return;

        ComputeFunction::validate();

        config.block_size = config.block_size == RS_AUTO_BLOCK_SIZE
                                ? RS_DEVICE_CPU_DEFAULT_BLOCK_SIZE
                                : config.block_size;

        config.shared_mem_size = config.shared_mem_size == 0
                                     ? ComputeFunction::sharedMemLayoutSize() * config.block_size
                                     : config.shared_mem_size;

        if constexpr (!ComputeFunction::isCPUSupported()) {
            RS_THROW(
                RS_DeviceExecutionException("Compute function is not supported by the CPU provider")
            );
        } else {
            RSize block_num = RS_CEIL_DIV(config.thread_num, config.block_size);

            std::lock_guard<std::recursive_mutex> lock(*mutex);
            pool->wait();  // make sure all previous tasks are finished
            RSize parallel_block_num = pool->size();

            if (config.shared_mem_size * parallel_block_num > shared_mem.size()) {
                shared_mem.free();
                shared_mem = allocate(config.shared_mem_size * parallel_block_num, true);
            }

            if (RS_DEVICE_CPU_FIBER_STACK_SIZE * config.block_size * parallel_block_num >
                stack_mem.size()) {
                stack_mem.free();
                stack_mem = allocate(
                    RS_DEVICE_CPU_FIBER_STACK_SIZE * config.block_size * parallel_block_num, true
                );
            }

            if (parallel_block_num > cpu_blocks.size()) {
                cpu_blocks.resize(parallel_block_num);
            }

            RS_Timer exec_timer;
            if (isExecLogging()) {
                // Open the line before submission so a crash still shows the kernel.
                execTraceBegin(
                    __FILE__,
                    __LINE__,
                    format(
                        "{}<<<block_num={}, block_size={}, smem={}>>>",
                        boost::core::demangle(typeid(ComputeFunction).name()),
                        block_num,
                        config.block_size,
                        config.shared_mem_size
                    )
                );
                exec_timer.reset();
            }

            pool->detachLoop(
                0,
                block_num,
                [=](RSize b) {
                    RIndex worker_thread_index =
                        RS_ThreadPool::RS_SynchronizedThreadPoolWorker::index;
                    if (config.shared_mem_size > 0) {
                        cpu_blocks[worker_thread_index].run<ComputeFunction, Args...>(
                            b,
                            config.block_size,
                            RS_MIN(config.block_size, config.thread_num - b * config.block_size),
                            RS_DEVICE_CPU_FIBER_STACK_SIZE,
                            (stack_mem + worker_thread_index * config.block_size *
                                             RS_DEVICE_CPU_FIBER_STACK_SIZE)
                                .ptr(),
                            (shared_mem + worker_thread_index * config.shared_mem_size).ptr(),
                            args...
                        );
                    } else {
                        cpu_blocks[worker_thread_index].run<ComputeFunction, Args...>(
                            b,
                            config.block_size,
                            RS_MIN(config.block_size, config.thread_num - b * config.block_size),
                            RS_DEVICE_CPU_FIBER_STACK_SIZE,
                            (stack_mem + worker_thread_index * config.block_size *
                                             RS_DEVICE_CPU_FIBER_STACK_SIZE)
                                .ptr(),
                            static_cast<void *>(nullptr),
                            args...
                        );
                    }
                },
                RS_DEVICE_CPU_THREAD_POOL_TASK_TO_THREAD_RATIO * pool->size()
            );
            if (config.sync || RS_EnvironmentVariables::getInstance().isExecForceSync())
                pool->wait();

            if (isExecLogging()) execTraceEnd(exec_timer);
        }
    }

    template <typename ComputeFunction, bool Debug = false, typename... Args>
    void runGrid(RS_MemoryProviderRunGridConfig config, Args... args) {
        if (!mutex) RS_THROW(RS_UninitializedException("CPU provider is not initialized"));

        if (!config.validate())
            RS_THROW(RS_InvalidArgumentException("Invalid config: {}", config.toString()));

        RSize total_threads_per_block =
            config.block_dim_x * config.block_dim_y * config.block_dim_z;
        RSize total_blocks = config.grid_dim_x * config.grid_dim_y * config.grid_dim_z;

        if (total_threads_per_block == 0 || total_blocks == 0) return;

        ComputeFunction::validate();

        config.shared_mem_size =
            config.shared_mem_size == 0
                ? ComputeFunction::sharedMemLayoutSize() * total_threads_per_block
                : config.shared_mem_size;

        if constexpr (!ComputeFunction::isCPUSupported()) {
            RS_THROW(
                RS_DeviceExecutionException("Compute function is not supported by the CPU provider")
            );
        } else {
            std::lock_guard<std::recursive_mutex> lock(*mutex);
            pool->wait();  // make sure all previous tasks are finished
            RSize parallel_block_num = pool->size();

            if (config.shared_mem_size * parallel_block_num > shared_mem.size()) {
                shared_mem.free();
                shared_mem = allocate(config.shared_mem_size * parallel_block_num, true);
            }

            if (RS_DEVICE_CPU_FIBER_STACK_SIZE * total_threads_per_block * parallel_block_num >
                stack_mem.size()) {
                stack_mem.free();
                stack_mem = allocate(
                    RS_DEVICE_CPU_FIBER_STACK_SIZE * total_threads_per_block * parallel_block_num,
                    true
                );
            }

            if (parallel_block_num > host_grid_blocks.size()) {
                host_grid_blocks.resize(parallel_block_num);
            }

            RS_Timer exec_timer;
            if (isExecLogging()) {
                // Open the line before submission so a crash still shows the kernel.
                execTraceBegin(
                    __FILE__,
                    __LINE__,
                    format(
                        "{}<<<grid=({}, {}, {}), block=({}, {}, {}), smem={}>>>",
                        boost::core::demangle(typeid(ComputeFunction).name()),
                        config.grid_dim_x,
                        config.grid_dim_y,
                        config.grid_dim_z,
                        config.block_dim_x,
                        config.block_dim_y,
                        config.block_dim_z,
                        config.shared_mem_size
                    )
                );
                exec_timer.reset();
            }

            pool->detachLoop(
                0,
                total_blocks,
                [=](RSize linear_block_idx) {
                    RIndex worker_thread_index =
                        RS_ThreadPool::RS_SynchronizedThreadPoolWorker::index;

                    // Decompose linear block index into 3D block indices
                    RIndex bz = linear_block_idx % config.grid_dim_z;
                    RIndex by = (linear_block_idx / config.grid_dim_z) % config.grid_dim_y;
                    RIndex bx = linear_block_idx / (config.grid_dim_y * config.grid_dim_z);

                    if (config.shared_mem_size > 0) {
                        host_grid_blocks[worker_thread_index].runGrid<ComputeFunction, Args...>(
                            config.grid_dim_x,
                            config.grid_dim_y,
                            config.grid_dim_z,
                            config.block_dim_x,
                            config.block_dim_y,
                            config.block_dim_z,
                            bx,
                            by,
                            bz,
                            RS_DEVICE_CPU_FIBER_STACK_SIZE,
                            (stack_mem + worker_thread_index * total_threads_per_block *
                                             RS_DEVICE_CPU_FIBER_STACK_SIZE)
                                .ptr(),
                            (shared_mem + worker_thread_index * config.shared_mem_size).ptr(),
                            args...
                        );
                    } else {
                        host_grid_blocks[worker_thread_index].runGrid<ComputeFunction, Args...>(
                            config.grid_dim_x,
                            config.grid_dim_y,
                            config.grid_dim_z,
                            config.block_dim_x,
                            config.block_dim_y,
                            config.block_dim_z,
                            bx,
                            by,
                            bz,
                            RS_DEVICE_CPU_FIBER_STACK_SIZE,
                            (stack_mem + worker_thread_index * total_threads_per_block *
                                             RS_DEVICE_CPU_FIBER_STACK_SIZE)
                                .ptr(),
                            static_cast<void *>(nullptr),
                            args...
                        );
                    }
                },
                RS_DEVICE_CPU_THREAD_POOL_TASK_TO_THREAD_RATIO * pool->size()
            );
            if (config.sync || RS_EnvironmentVariables::getInstance().isExecForceSync())
                pool->wait();

            if (isExecLogging()) execTraceEnd(exec_timer);
        }
    }

    template <typename... Args>
    void capture(const char *key, Args &&...args) {
        auto all = std::forward_as_tuple(std::forward<Args>(args)...);
        std::get<sizeof...(Args) - 1>(all)();
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_CPU_H
