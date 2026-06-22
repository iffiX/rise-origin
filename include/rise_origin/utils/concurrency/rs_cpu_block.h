// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CPU_BLOCK_H
#define RS_CPU_BLOCK_H

#include <boost/context/continuation_fcontext.hpp>
#include <tuple>
#include <vector>

#include "rise_origin/utils/data/rs_any.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

class RS_CPUBlock {
   public:
    struct RS_CPUFiber {
        RIndex fiber_index = 0;
        boost::context::detail::fcontext_t ctx{};
        bool is_finished = false;
        RS_CPUBlock* block = nullptr;
    };

    struct RS_CPUFiberTransferInfo {
        bool is_from_host = false;
        RS_CPUFiber* from_fiber = nullptr;
        RS_CPUFiber* to_fiber = nullptr;
    };

    // The main context of the “block thread” so fibers can jump back
    boost::context::detail::fcontext_t main_ctx{};

    RS_CPUFiberTransferInfo transfer_info{};

    bool block_is_finished = false;

    bool block_has_switched = false;

    bool block_has_error = false;

    // The current block index
    RIndex block_index = RS_NULL_INDEX;

    // Number of fibers (a.k.a. blockDim.x on CPU).
    RSize block_size = 0;

    RSize active_thread_num = 0;

    char* block_shared_mem = nullptr;

    // The array of virtual fibers (each fiber corresponds to a GPU thread)
    std::vector<RS_CPUFiber> fibers;

    // The block arguments
    RS_Any block_args;

    RS_CPUBlock() = default;

    /**
     * Allocate or resize a single buffer for all fiber stacks,
     * then set up the offset for each fiber.
     * @param block_index The block index
     * @param block_size  The number of “virtual threads” in this block
     * @param active_thread_num The number of active threads in this block
     * @param stack_size  The size of each fiber’s stack
     * @param block_stacks The buffer holding all fiber stacks of this block
     * @param block_shared_mem The buffer holding all shared memory of this block
     */
    template <typename ComputeFunction, typename... Args>
    void run(
        RIndex block_index,
        RSize block_size,
        RSize active_thread_num,
        RSize stack_size,
        void* block_stacks,
        void* block_shared_mem,
        Args... args
    ) {
        block_is_finished = false;
        block_has_switched = false;
        block_has_error = false;
        this->block_index = block_index;
        this->block_size = block_size;
        this->active_thread_num = active_thread_num;
        this->block_shared_mem = static_cast<char*>(block_shared_mem);
        block_args = std::make_tuple(args...);

        if constexpr (ComputeFunction::isSyncBlockRequired()) {
            if (fibers.size() < block_size) fibers.resize(block_size);

            // Assign each fiber an offset in host_stacks
            for (RSize i = 0; i < active_thread_num; ++i) {
                fibers[i].fiber_index = i;
                auto stack_top = static_cast<char*>(block_stacks) + (i + 1) * stack_size;
                fibers[i].ctx = boost::context::detail::make_fcontext(
                    stack_top, stack_size, &fiberEntryFunction<ComputeFunction, Args...>
                );
                fibers[i].is_finished = false;
                fibers[i].block = this;
            }

            // Clear main ctx and run the block
            main_ctx = {};
            transfer_info = {};

            // Run fiber 0 first, if no switch happens, switch to for loop for better performance
            if (active_thread_num > 0) {
                transfer_info.is_from_host = true;
                transfer_info.from_fiber = nullptr;
                transfer_info.to_fiber = &fibers[0];
                setThreadLocalStorage<ComputeFunction>(
                    this, static_cast<char*>(block_shared_mem), 0
                );
                auto t = boost::context::detail::jump_fcontext(fibers[0].ctx, &transfer_info);

                // When we return here, it must be from a fiber, and never from ourself (main
                // thread)
                auto* t_info_ptr = static_cast<RS_CPUFiberTransferInfo*>(t.data);
                RIndex from_fiber_index = t_info_ptr->from_fiber->fiber_index;
                fibers[from_fiber_index].ctx = t.fctx;

                if (block_has_error)
                    RS_THROW(RS_HostExecutionException("An error is encountered in a fiber"));
                else {
                    if (block_has_switched) {
                        for (RIndex i = 1; i < active_thread_num; ++i) {
                            if (!fibers[i].is_finished) {
                                setThreadLocalStorage<ComputeFunction>(
                                    this, static_cast<char*>(block_shared_mem), i
                                );
                                transfer_info.is_from_host = true;
                                transfer_info.from_fiber = nullptr;
                                transfer_info.to_fiber = &fibers[i];

                                auto t = boost::context::detail::jump_fcontext(
                                    fibers[i].ctx, &transfer_info
                                );

                                // When we return here, it must be from a fiber, and never from
                                // ourself (main thread)
                                t_info_ptr = static_cast<RS_CPUFiberTransferInfo*>(t.data);
                                from_fiber_index = t_info_ptr->from_fiber->fiber_index;
                                fibers[from_fiber_index].ctx = t.fctx;
                            }
                            // An error is encountered in one of the fibers => break
                            if (block_is_finished) {
                                if (block_has_error)
                                    RS_THROW(RS_HostExecutionException(
                                        "An error is encountered in a fiber"
                                    ));
                                break;
                            }
                        }
                    } else {
                        for (RIndex i = 1; i < active_thread_num; ++i) {
                            setThreadLocalStorage<ComputeFunction>(
                                this, static_cast<char*>(block_shared_mem), i
                            );
                            ComputeFunction::run(args...);
                        }
                    }
                    block_is_finished = true;
                }
            } else {
                block_is_finished = true;
            }
        } else {
            for (RIndex i = 0; i < active_thread_num; ++i) {
                setThreadLocalStorage<ComputeFunction>(
                    this, static_cast<char*>(block_shared_mem), i
                );
                ComputeFunction::run(args...);
            }
            block_is_finished = true;
        }
    }

    /*****************************************************************
     * Fiber Entry Function
     *
     * Called the first time we jump_fcontext() into a particular fiber.
     * We store the newly resumed fcontext and run the user’s compute function.
     *****************************************************************/
    template <typename ComputeFunction, typename... Args>
    static void fiberEntryFunction(boost::context::detail::transfer_t transfer) {
        // Unpack our start info
        auto* t_info_ptr = static_cast<RS_CPUFiberTransferInfo*>(transfer.data);
        RS_CPUBlock* block = t_info_ptr->to_fiber->block;
        RIndex fiber_index = t_info_ptr->to_fiber->fiber_index;

        // Preserver the context of the fiber/main thread where we jumped from
        if (t_info_ptr->is_from_host) {
            block->main_ctx = transfer.fctx;
        } else {
            RIndex from_fiber_index = t_info_ptr->from_fiber->fiber_index;
            block->fibers[from_fiber_index].ctx = transfer.fctx;
        }

        // TODO: better try catch handling with error details? For now, just let error propagate
        // try {
        std::apply(
            [&](auto&&... unpacked_args) {
                ComputeFunction::run(std::forward<decltype(unpacked_args)>(unpacked_args)...);
            },
            block->block_args.as<std::tuple<Args...>>()
        );
        // } catch (...) {
        //     // Mark error
        //     block->block_is_finished = true;
        //     block->block_has_error = true;
        // }

        // Mark fiber done
        block->fibers[fiber_index].is_finished = true;

        // Switch back to the block’s main context
        // (the OS thread that manages all these fibers)
        t_info_ptr->is_from_host = false;
        t_info_ptr->from_fiber = t_info_ptr->to_fiber;
        t_info_ptr->to_fiber = nullptr;

        auto t = boost::context::detail::jump_fcontext(block->main_ctx, t_info_ptr);

        // Although we should not return here, handle the returned fcontext
        t_info_ptr = static_cast<RS_CPUFiberTransferInfo*>(t.data);
        block = t_info_ptr->to_fiber->block;
        if (t_info_ptr->is_from_host) {
            block->main_ctx = t.fctx;
        } else {
            RIndex from_fiber_index = t_info_ptr->from_fiber->fiber_index;
            block->fibers[from_fiber_index].ctx = t.fctx;
        }
    }

    /*****************************************************************
     * Barrier-Like Sync: syncThreadsInBlockImpl
     *
     * All “virtual threads” in the block share one OS thread,
     *
     * Once you’re the last fiber to arrive, you reset the barrier and
     * optionally jump to fiber 0 (or keep going).
     *****************************************************************/
    template <typename ComputeFunction>
    void syncThreadsInBlockImpl(RIndex fiber_index) {
        _barrier_count++;
        block_has_switched = true;
        if (active_thread_num != block_size) {
            RS_THROW(RS_HostExecutionException(
                "The block size configuration is not equal to the number of active threads,"
                "this will cause a deadlock on GPU side"
            ));
        }
        if (_barrier_count < static_cast<int>(block_size)) {
            // Not all have arrived => jump to the next fiber in round-robin fashion
            int next_fiber_idx = (fiber_index + 1) % block_size;
            setThreadLocalStorage<ComputeFunction>(this, block_shared_mem, next_fiber_idx);

            transfer_info.is_from_host = false;
            transfer_info.from_fiber = &fibers[fiber_index];
            transfer_info.to_fiber = &fibers[next_fiber_idx];

            auto t =
                boost::context::detail::jump_fcontext(fibers[next_fiber_idx].ctx, &transfer_info);

            auto* t_info_ptr = static_cast<RS_CPUFiberTransferInfo*>(t.data);

            if (t_info_ptr->is_from_host) {
                main_ctx = t.fctx;
            } else {
                RIndex from_fiber_index = t_info_ptr->from_fiber->fiber_index;
                fibers[from_fiber_index].ctx = t.fctx;
            }

            // When we come back here, all threads have arrived at the barrier
            // so we can proceed after the sync.
        } else {
            // This is the last fiber to arrive => reset barrier
            _barrier_count = 0;
            // Jump to fiber 0 (or continue if fiber_index==0)
            if (fiber_index != 0) {
                setThreadLocalStorage<ComputeFunction>(this, block_shared_mem, 0);

                transfer_info.is_from_host = false;
                transfer_info.from_fiber = &fibers[fiber_index];
                transfer_info.to_fiber = &fibers[0];

                auto t = boost::context::detail::jump_fcontext(fibers[0].ctx, &transfer_info);

                auto* t_info_ptr = static_cast<RS_CPUFiberTransferInfo*>(t.data);

                if (t_info_ptr->is_from_host) {
                    main_ctx = t.fctx;
                } else {
                    RIndex from_fiber_index = t_info_ptr->from_fiber->fiber_index;
                    fibers[from_fiber_index].ctx = t.fctx;
                }
            }
            // If fiber_index == 0, we just continue from here
        }
    }

   private:
    RSize _barrier_count = 0;

    template <typename ComputeFunction>
    static void
    setThreadLocalStorage(RS_CPUBlock* block, char* block_shared_mem, RIndex fiber_index) {
        ComputeFunction::setIndicesAndBlockSize(
            /*blockDimX=*/block->block_size,
            /*blockDimY=*/1,
            /*blockDimZ=*/1,
            /*blockIdxX=*/block->block_index,
            /*blockIdxY=*/0,
            /*blockIdxZ=*/0,
            /*threadIdxX=*/fiber_index,
            /*threadIdxY=*/0,
            /*threadIdxZ=*/0
        );

        if (block_shared_mem != nullptr) {
            ComputeFunction::setSharedMem(block_shared_mem);
        }

        ComputeFunction::setBlock(block);
    }
};

class RS_CPUGridBlock {
   public:
    struct RS_CPUFiber {
        RIndex fiber_index = 0;
        boost::context::detail::fcontext_t ctx{};
        bool is_finished = false;
        RS_CPUGridBlock* block = nullptr;

        // Pre-calculated 3D thread coordinates to avoid runtime division/modulo
        RIndex thread_idx_x = 0;
        RIndex thread_idx_y = 0;
        RIndex thread_idx_z = 0;
    };

    struct RS_CPUFiberTransferInfo {
        bool is_from_host = false;
        RS_CPUFiber* from_fiber = nullptr;
        RS_CPUFiber* to_fiber = nullptr;
    };

    // The main context of the "block thread" so fibers can jump back
    boost::context::detail::fcontext_t main_ctx{};

    RS_CPUFiberTransferInfo transfer_info{};

    bool block_is_finished = false;

    bool block_has_switched = false;

    bool block_has_error = false;

    // The 3D grid dimensions
    RSize grid_dim_x = 0;
    RSize grid_dim_y = 0;
    RSize grid_dim_z = 0;

    // The 3D block dimensions
    RSize block_dim_x = 0;
    RSize block_dim_y = 0;
    RSize block_dim_z = 0;

    // The current 3D block indices
    RIndex block_idx_x = 0;
    RIndex block_idx_y = 0;
    RIndex block_idx_z = 0;

    // Total number of threads in the block
    RSize total_threads = 0;

    char* block_shared_mem = nullptr;

    // The array of virtual fibers (each fiber corresponds to a GPU thread)
    std::vector<RS_CPUFiber> fibers;

    // The block arguments
    RS_Any block_args;

    RS_CPUGridBlock() = default;

    /**
     * Run a grid block with 3D block and thread indices
     * @param grid_dim_x, grid_dim_y, grid_dim_z The 3D Grid dimensions
     * @param block_dim_x, block_dim_y, block_dim_z The 3D block dimensions
     * @param block_idx_x, block_idx_y, block_idx_z The 3D block indices
     * @param stack_size  The size of each fiber's stack
     * @param block_stacks The buffer holding all fiber stacks of this block
     * @param block_shared_mem The buffer holding all shared memory of this block
     */
    template <typename ComputeFunction, typename... Args>
    void runGrid(
        RSize grid_dim_x,
        RSize grid_dim_y,
        RSize grid_dim_z,
        RSize block_dim_x,
        RSize block_dim_y,
        RSize block_dim_z,
        RIndex block_idx_x,
        RIndex block_idx_y,
        RIndex block_idx_z,
        RSize stack_size,
        void* block_stacks,
        void* block_shared_mem,
        Args... args
    ) {
        block_is_finished = false;
        block_has_switched = false;
        block_has_error = false;

        // Recompute if block shape is different
        bool needs_recompute_thread_coords = this->block_dim_x != block_dim_x ||
                                             this->block_dim_y != block_dim_y ||
                                             this->block_dim_z != block_dim_z;
        this->grid_dim_x = grid_dim_x;
        this->grid_dim_y = grid_dim_y;
        this->grid_dim_z = grid_dim_z;
        this->block_idx_x = block_idx_x;
        this->block_idx_y = block_idx_y;
        this->block_idx_z = block_idx_z;
        this->block_dim_x = block_dim_x;
        this->block_dim_y = block_dim_y;
        this->block_dim_z = block_dim_z;
        this->total_threads = block_dim_x * block_dim_y * block_dim_z;
        this->block_shared_mem = static_cast<char*>(block_shared_mem);
        block_args = std::make_tuple(args...);

        if constexpr (ComputeFunction::isSyncBlockRequired()) {
            if (fibers.size() < total_threads) fibers.resize(total_threads);

            // Assign each fiber an offset in block_stacks and pre-calculate 3D thread coordinates
            for (RSize i = 0; i < total_threads; ++i) {
                fibers[i].fiber_index = i;
                auto stack_top = static_cast<char*>(block_stacks) + (i + 1) * stack_size;
                fibers[i].ctx = boost::context::detail::make_fcontext(
                    stack_top, stack_size, &fiberEntryFunction<ComputeFunction, Args...>
                );
                fibers[i].is_finished = false;
                fibers[i].block = this;
            }

            if (needs_recompute_thread_coords) {
                RSize y_mul_z = block_dim_y * block_dim_z;
                for (RSize x = 0; x < block_dim_x; ++x) {
                    for (RSize y = 0; y < block_dim_y; ++y) {
                        for (RSize z = 0; z < block_dim_z; ++z) {
                            RIndex idx = x * y_mul_z + y * block_dim_z + z;
                            fibers[idx].thread_idx_x = x;
                            fibers[idx].thread_idx_y = y;
                            fibers[idx].thread_idx_z = z;
                        }
                    }
                }
            }

            // Clear main ctx and run the block
            main_ctx = {};
            transfer_info = {};

            // Run fiber 0 first, if no switch happens, switch to for loop for better performance
            if (total_threads > 0) {
                transfer_info.is_from_host = true;
                transfer_info.from_fiber = nullptr;
                transfer_info.to_fiber = &fibers[0];
                setThreadLocalStorage<ComputeFunction>(
                    this, static_cast<char*>(block_shared_mem), 0, 0, 0
                );
                auto t = boost::context::detail::jump_fcontext(fibers[0].ctx, &transfer_info);

                // When we return here, it must be from a fiber, and never from ourself (main
                // thread)
                auto* t_info_ptr = static_cast<RS_CPUFiberTransferInfo*>(t.data);
                RIndex from_fiber_index = t_info_ptr->from_fiber->fiber_index;
                fibers[from_fiber_index].ctx = t.fctx;

                if (block_has_error)
                    RS_THROW(RS_HostExecutionException("An error is encountered in a fiber"));
                else {
                    if (block_has_switched) {
                        for (RIndex i = 1; i < total_threads; ++i) {
                            if (!fibers[i].is_finished) {
                                setThreadLocalStorage<ComputeFunction>(
                                    this,
                                    static_cast<char*>(block_shared_mem),
                                    fibers[i].thread_idx_x,
                                    fibers[i].thread_idx_y,
                                    fibers[i].thread_idx_z
                                );
                                transfer_info.is_from_host = true;
                                transfer_info.from_fiber = nullptr;
                                transfer_info.to_fiber = &fibers[i];

                                t = boost::context::detail::jump_fcontext(
                                    fibers[i].ctx, &transfer_info
                                );

                                // When we return here, it must be from a fiber, and never from
                                // ourself (main thread)
                                t_info_ptr = static_cast<RS_CPUFiberTransferInfo*>(t.data);
                                from_fiber_index = t_info_ptr->from_fiber->fiber_index;
                                fibers[from_fiber_index].ctx = t.fctx;
                            }

                            // An error is encountered in one of the fibers => break
                            if (block_is_finished) {
                                if (block_has_error)
                                    RS_THROW(RS_HostExecutionException(
                                        "An error is encountered in a fiber"
                                    ));
                                break;
                            }
                        }
                    } else {
                        for (RIndex i = 1; i < total_threads; ++i) {
                            setThreadLocalStorage<ComputeFunction>(
                                this,
                                static_cast<char*>(block_shared_mem),
                                fibers[i].thread_idx_x,
                                fibers[i].thread_idx_y,
                                fibers[i].thread_idx_z
                            );
                            ComputeFunction::run(args...);
                        }
                    }
                    block_is_finished = true;
                }
            } else {
                block_is_finished = true;
            }
        } else {
            for (RIndex x = 0; x < block_dim_x; ++x) {
                for (RIndex y = 0; y < block_dim_y; ++y) {
                    for (RIndex z = 0; z < block_dim_z; ++z) {
                        setThreadLocalStorage<ComputeFunction>(
                            this, static_cast<char*>(block_shared_mem), x, y, z
                        );
                        ComputeFunction::run(args...);
                    }
                }
            }
            block_is_finished = true;
        }
    }

    template <typename ComputeFunction, typename... Args>
    static void fiberEntryFunction(boost::context::detail::transfer_t transfer) {
        // Unpack our start info
        auto* t_info_ptr = static_cast<RS_CPUFiberTransferInfo*>(transfer.data);
        RS_CPUGridBlock* block = t_info_ptr->to_fiber->block;
        RIndex fiber_index = t_info_ptr->to_fiber->fiber_index;

        // Preserve the context of the fiber/main thread where we jumped from
        if (t_info_ptr->is_from_host) {
            block->main_ctx = transfer.fctx;
        } else {
            RIndex from_fiber_index = t_info_ptr->from_fiber->fiber_index;
            block->fibers[from_fiber_index].ctx = transfer.fctx;
        }

        std::apply(
            [&](auto&&... unpacked_args) {
                ComputeFunction::run(std::forward<decltype(unpacked_args)>(unpacked_args)...);
            },
            block->block_args.as<std::tuple<Args...>>()
        );

        // Mark fiber done
        block->fibers[fiber_index].is_finished = true;

        // Switch back to the block's main context
        t_info_ptr->is_from_host = false;
        t_info_ptr->from_fiber = t_info_ptr->to_fiber;
        t_info_ptr->to_fiber = nullptr;

        auto t = boost::context::detail::jump_fcontext(block->main_ctx, t_info_ptr);

        // Although we should not return here, handle the returned fcontext
        t_info_ptr = static_cast<RS_CPUFiberTransferInfo*>(t.data);
        block = t_info_ptr->to_fiber->block;
        if (t_info_ptr->is_from_host) {
            block->main_ctx = t.fctx;
        } else {
            RIndex from_fiber_index = t_info_ptr->from_fiber->fiber_index;
            block->fibers[from_fiber_index].ctx = t.fctx;
        }
    }

    template <typename ComputeFunction>
    void syncThreadsInBlockImpl(RIndex fiber_index) {
        _barrier_count++;
        block_has_switched = true;
        if (total_threads != (block_dim_x * block_dim_y * block_dim_z)) {
            RS_THROW(RS_HostExecutionException(
                "The block dimension configuration is not equal to the number of total threads,"
                "this will cause a deadlock on GPU side"
            ));
        }
        if (_barrier_count < static_cast<int>(total_threads)) {
            // Not all have arrived => jump to the next fiber in round-robin fashion
            RIndex next_fiber_idx = (fiber_index + 1) % total_threads;

            setThreadLocalStorage<ComputeFunction>(
                this,
                block_shared_mem,
                fibers[next_fiber_idx].thread_idx_x,
                fibers[next_fiber_idx].thread_idx_y,
                fibers[next_fiber_idx].thread_idx_z
            );

            transfer_info.is_from_host = false;
            transfer_info.from_fiber = &fibers[fiber_index];
            transfer_info.to_fiber = &fibers[next_fiber_idx];

            auto t =
                boost::context::detail::jump_fcontext(fibers[next_fiber_idx].ctx, &transfer_info);

            auto* t_info_ptr = static_cast<RS_CPUFiberTransferInfo*>(t.data);

            if (t_info_ptr->is_from_host) {
                main_ctx = t.fctx;
            } else {
                RIndex from_fiber_index = t_info_ptr->from_fiber->fiber_index;
                fibers[from_fiber_index].ctx = t.fctx;
            }

            // When we come back here, all threads have arrived at the barrier
            // so we can proceed after the sync.
        } else {
            // This is the last fiber to arrive => reset barrier
            _barrier_count = 0;
            // Jump to fiber 0 or continue if fiber_index==0
            if (fiber_index != 0) {
                setThreadLocalStorage<ComputeFunction>(this, block_shared_mem, 0, 0, 0);

                transfer_info.is_from_host = false;
                transfer_info.from_fiber = &fibers[fiber_index];
                transfer_info.to_fiber = &fibers[0];

                auto t = boost::context::detail::jump_fcontext(fibers[0].ctx, &transfer_info);

                auto* t_info_ptr = static_cast<RS_CPUFiberTransferInfo*>(t.data);

                if (t_info_ptr->is_from_host) {
                    main_ctx = t.fctx;
                } else {
                    RIndex from_fiber_index = t_info_ptr->from_fiber->fiber_index;
                    fibers[from_fiber_index].ctx = t.fctx;
                }
            }
            // If fiber_index == 0, we just continue from here
        }
    }

   private:
    RSize _barrier_count = 0;

    template <typename ComputeFunction>
    static void setThreadLocalStorage(
        RS_CPUGridBlock* block,
        char* block_shared_mem,
        RIndex thread_idx_x,
        RIndex thread_idx_y,
        RIndex thread_idx_z
    ) {
        ComputeFunction::setIndicesAndBlockSize(
            /*blockDimX=*/block->block_dim_x,
            /*blockDimY=*/block->block_dim_y,
            /*blockDimZ=*/block->block_dim_z,
            /*blockIdxX=*/block->block_idx_x,
            /*blockIdxY=*/block->block_idx_y,
            /*blockIdxZ=*/block->block_idx_z,
            /*threadIdxX=*/thread_idx_x,
            /*threadIdxY=*/thread_idx_y,
            /*threadIdxZ=*/thread_idx_z
        );

        if (block_shared_mem != nullptr) {
            ComputeFunction::setSharedMem(block_shared_mem);
        }

        ComputeFunction::setBlock(block);
    }
};

}  // namespace rise

#endif  // RS_CPU_BLOCK_H
