// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_REDUCE_H
#define RS_REDUCE_H

#include <algorithm>
#include <numeric>

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/device/rs_cooperative_group.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_search.h"

#define RS_REDUCE_MAX_GROUP_REDUCE_LEVEL 5

namespace rise {

namespace device {

/**
 * @brief Host block reduce: each block serially reduces its slice of the input.
 * @param d_in Input array
 * @param d_out One reduced value per block
 * @param d_in_len Number of input elements
 * @param init_value Initial value for padding (out-of-range slots)
 * @param block_elem_num Elements per block (2 * block size)
 */
template <typename T, typename ReduceOp>
class RSCF_CPUBlockReduce : public RS_ComputeFunction<RSCF_CPUBlockReduce<T, ReduceOp>> {
   public:
    using IsCudaSupported = std::false_type;

    using BaseType = RS_ComputeFunction<RSCF_CPUBlockReduce>;
    using BaseType::blockDimX;
    using BaseType::blockIdxX;
    using BaseType::index1D;

    HOST_INLINE_QUALIFIER static void
    run(const T *d_in, T *d_out, RSize d_in_len, T init_value, RSize block_elem_num) {
        auto op = ReduceOp();
        RIndex i = index1D();
        RIndex elem_offset = i * block_elem_num;
        T value = init_value;
        for (RSize j = 0, max = RS_MIN(d_in_len - elem_offset, block_elem_num); j < max; j++) {
            value = op(value, d_in[elem_offset + j]);
        }
        d_out[i] = value;
    }
};

/**
 * @brief Block reduce: each block reduces 2 * block size elements in shared memory.
 * @param d_in Input array
 * @param d_out One reduced value per block
 * @param d_in_len Number of input elements
 * @param init_value Initial value for padding (out-of-range slots)
 */
template <typename T, typename ReduceOp>
class RSCF_BlockReduce : public RS_ComputeFunction<RSCF_BlockReduce<T, ReduceOp>> {
   public:
    using IsCPUSupported = std::false_type;

    using BaseType = RS_ComputeFunction<RSCF_BlockReduce>;
    using BaseType::blockDimX;
    using BaseType::blockIdxX;
    using BaseType::sharedMem;
    using BaseType::syncThreadsInBlock;
    using BaseType::threadIdxX;

    // One element per thread
    struct SharedMemLayout {
        T value;
    };

    DEVICE_INLINE_QUALIFIER static void run(const T *d_in, T *d_out, RSize d_in_len, T init_value) {
        auto op = ReduceOp();
        T *s_out = reinterpret_cast<T *>(sharedMem());

        RIndex elem_offset = blockIdxX() * blockDimX() * 2 + threadIdxX();
        RIndex local_tid = threadIdxX();

        // Fold both halves during load, padding out-of-range slots with init_value
        T value = init_value;
        if (elem_offset < d_in_len) {
            value = d_in[elem_offset];
            if (elem_offset + blockDimX() < d_in_len)
                value = op(value, d_in[elem_offset + blockDimX()]);
        }
        s_out[local_tid] = value;

        syncThreadsInBlock();

        // Shared tree while more than one group of partials remains
        constexpr RSize sync_group_size = RS_SynchronizedGroup<>::GroupSize;
        unsigned int s = blockDimX() / 2;
        for (; s >= sync_group_size; s >>= 1) {
            if (local_tid < s) s_out[local_tid] = op(s_out[local_tid], s_out[local_tid + s]);
            syncThreadsInBlock();
        }

        // Finish the last group of partials inside one synchronized group
        if constexpr (sync_group_size > 1 && sizeof(T) % sizeof(uint32_t) == 0) {
            if (blockDimX() >= sync_group_size) {
                auto sync_group = getCurrentSynchronizedGroup();
                if (local_tid < sync_group_size) {
                    value = s_out[local_tid];
                    for (int off = sync_group_size / 2; off > 0; off >>= 1)
                        value =
                            op(value, sync_group.shfl(value, static_cast<int>(local_tid) ^ off));
                    if (local_tid == 0) d_out[blockIdxX()] = value;
                }
                return;
            }
        }

        // Shared-mem fallback: sub-group blocks and sub-word types
        for (; s > 0; s >>= 1) {
            if (local_tid < s) s_out[local_tid] = op(s_out[local_tid], s_out[local_tid + s]);
            syncThreadsInBlock();
        }
        if (local_tid == 0) d_out[blockIdxX()] = s_out[0];
    }
};

/**
 * @brief Group-aware host block reduce. Gathers the packed input and pads on the fly,
 * removing the need for a separate align pass.
 * @param d_in   "Packed" input array
 * @param d_out  One reduced value per block
 * @param d_in_sizes Input group sizes
 * @param d_in_sizes_psum Input group offsets (i.e. prefix sum of input group sizes,
 * starting with 0)
 * @param d_out_sizes_psum Prefix sum of output group sizes aligned to 2 * block size
 * @param group_num Number of groups
 * @param init_value Initial value for padding
 * @param block_elem_num Elements per block (2 * block size)
 */
template <typename T, typename ReduceOp>
class RSCF_HostGroupBlockReduce
    : public RS_ComputeFunction<RSCF_HostGroupBlockReduce<T, ReduceOp>> {
   public:
    using IsCudaSupported = std::false_type;

    using BaseType = RS_ComputeFunction<RSCF_HostGroupBlockReduce>;
    using BaseType::blockDimX;
    using BaseType::blockIdxX;
    using BaseType::index1D;

    HOST_INLINE_QUALIFIER static void run(
        const T *d_in,
        T *d_out,
        const RSize *d_in_sizes,
        const RSize *d_in_sizes_psum,
        const RSize *d_out_sizes_psum,
        RSize group_num,
        T init_value,
        RSize block_elem_num
    ) {
        auto op = ReduceOp();
        RIndex i = index1D();
        RIndex block_base = i * block_elem_num;

        // Whole block lies in one group
        auto group = RSCF_BinaryPrefixSearch::run(block_base, d_out_sizes_psum, group_num);
        T value = init_value;
        if (group.index != RS_NULL_INDEX) {
            RSize group_size = d_in_sizes[group.index];
            RSize in_offset = d_in_sizes_psum[group.index];
            for (RSize j = 0; j < block_elem_num; j++) {
                RIndex local = group.local_index + j;
                if (local < group_size) value = op(value, d_in[in_offset + local]);
            }
        }
        d_out[i] = value;
    }
};

/**
 * @brief Group-aware block reduce. Gathers the packed input and pads on the fly,
 * removing the need for a separate align pass.
 * @param d_in "Packed" input array
 * @param d_out One reduced value per block
 * @param d_in_sizes Input group sizes
 * @param d_in_sizes_psum Input group offsets (i.e. prefix sum of input group sizes,
 * starting with 0)
 * @param d_out_sizes_psum Prefix sum of output group sizes aligned to 2 * block size
 * @param group_num Number of groups
 * @param init_value Initial value for padding
 */
template <typename T, typename ReduceOp>
class RSCF_GroupBlockReduce : public RS_ComputeFunction<RSCF_GroupBlockReduce<T, ReduceOp>> {
   public:
    using IsCPUSupported = std::false_type;

    using BaseType = RS_ComputeFunction<RSCF_GroupBlockReduce>;
    using BaseType::blockDimX;
    using BaseType::blockIdxX;
    using BaseType::sharedMem;
    using BaseType::syncThreadsInBlock;
    using BaseType::threadIdxX;

    // One element per thread
    struct SharedMemLayout {
        T value;
    };

    DEVICE_INLINE_QUALIFIER static void run(
        const T *d_in,
        T *d_out,
        const RSize *d_in_sizes,
        const RSize *d_in_sizes_psum,
        const RSize *d_out_sizes_psum,
        RSize group_num,
        T init_value
    ) {
        auto op = ReduceOp();
        T *s_out = reinterpret_cast<T *>(sharedMem());

        RIndex local_tid = threadIdxX();
        // Whole block lies in one group (block-uniform search)
        RIndex block_base = blockIdxX() * blockDimX() * 2;
        auto group = RSCF_BinaryPrefixSearch::run(block_base, d_out_sizes_psum, group_num);

        // Gather + fold both halves during load, padding out-of-range slots
        T value = init_value;
        if (group.index != RS_NULL_INDEX) {
            RSize group_size = d_in_sizes[group.index];
            RSize in_offset = d_in_sizes_psum[group.index];
            RIndex local = group.local_index + local_tid;
            if (local < group_size) value = d_in[in_offset + local];
            if (local + blockDimX() < group_size)
                value = op(value, d_in[in_offset + local + blockDimX()]);
        }
        s_out[local_tid] = value;

        syncThreadsInBlock();

        // Shared tree while more than one group of partials remains
        constexpr RSize sync_group_size = RS_SynchronizedGroup<>::GroupSize;
        unsigned int s = blockDimX() / 2;
        for (; s >= sync_group_size; s >>= 1) {
            if (local_tid < s) s_out[local_tid] = op(s_out[local_tid], s_out[local_tid + s]);
            syncThreadsInBlock();
        }

        // Finish the last group of partials inside one synchronized group
        if constexpr (sync_group_size > 1 && sizeof(T) % sizeof(uint32_t) == 0) {
            if (blockDimX() >= sync_group_size) {
                auto sync_group = getCurrentSynchronizedGroup();
                if (local_tid < sync_group_size) {
                    value = s_out[local_tid];
                    for (int off = sync_group_size / 2; off > 0; off >>= 1)
                        value =
                            op(value, sync_group.shfl(value, static_cast<int>(local_tid) ^ off));
                    if (local_tid == 0) d_out[blockIdxX()] = value;
                }
                return;
            }
        }

        // Shared-mem fallback: sub-group blocks and sub-word types
        for (; s > 0; s >>= 1) {
            if (local_tid < s) s_out[local_tid] = op(s_out[local_tid], s_out[local_tid + s]);
            syncThreadsInBlock();
        }
        if (local_tid == 0) d_out[blockIdxX()] = s_out[0];
    }
};

class RS_Reduce {
   public:
    /*****************************************************************
     * Data
     *****************************************************************/
    RS_MemoryPtr host_buffer{};
    RS_MemoryPtr reduce_buffer1{};
    RS_MemoryPtr reduce_buffer2{};
    RS_MemoryProvider *mem_provider = nullptr;
    /*****************************************************************
     * Methods
     *****************************************************************/

    RS_Reduce() = default;

    explicit RS_Reduce(RS_MemoryProvider *provider) { init(provider); }

    void init(RS_MemoryProvider *provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);
        host_buffer = provider->allocateHost(0, true);
        reduce_buffer1 = provider->allocate(0, true);
        reduce_buffer2 = provider->allocate(0, true);
        mem_provider = provider;
    }

    void free() {
        host_buffer.free();
        reduce_buffer1.free();
        reduce_buffer2.free();
        mem_provider = nullptr;
    }

    /**
     * @brief Reduce values.
     * @tparam T
     * @tparam ReduceOp Reduce operator.
     * @param in Input device memory to be reduced of size element number, has to be on the same
     * device as where the reducer is initialized.
     * @param in_len Input element number.
     * @param init_value Initial value for threads that doesn't correspond to a valid element.
     * @param block_size Block size used for block reduction.
     * @note reduce implicitly synchronizes execution before returning results.
     */
    template <typename ReduceOp, typename T>
    T reduce(
        const T *in,
        RSize in_len,
        T init_value = 0,
        RSize block_size = RS_DEVICE_MAX_BLOCK_SIZE
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        return reduce<T, ReduceOp>(
            RS_ConstMemoryPtr(in, sizeof(T) * in_len, mem_provider->device, mem_provider),
            in_len,
            init_value,
            block_size
        );
    }

    /**
     * @brief Reduce values.
     * @tparam T
     * @tparam ReduceOp Reduce operator.
     * @param array Input array to be reduced of size element number, has to be on the same
     * device as where the reducer is initialized.
     * @param init_value Initial value for threads that doesn't correspond to a valid element.
     * @param block_size Block size used for block reduction.
     * @note reduce implicitly synchronizes execution before returning results.
     */
    template <typename ReduceOp, typename T>
    T reduce(RS_Array<T, 1> array, T init_value = 0, RSize block_size = RS_DEVICE_MAX_BLOCK_SIZE) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            array.device() == device(), "Array device must be the same as the reducer device"
        );
        return reduce<T, ReduceOp>(array.data, array.size(), init_value, block_size);
    }

    /**
     * @brief Reduce values.
     * @tparam T
     * @tparam ReduceOp Reduce operator.
     * @param vector Input vector to be reduced of size element number, has to be on the same
     * device as where the reducer is initialized.
     * @param init_value Initial value for threads that doesn't correspond to a valid element.
     * @param block_size Block size used for block reduction.
     * @note reduce implicitly synchronizes execution before returning results.
     */
    template <typename ReduceOp, typename T>
    T reduce(RS_Vector<T> vector, T init_value = 0, RSize block_size = RS_DEVICE_MAX_BLOCK_SIZE) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            vector.device() == device(), "Vector device must be the same as the reducer device"
        );
        return reduce<T, ReduceOp>(vector.data(), vector.size(), init_value, block_size);
    }

    /**
     * @brief Reduce values.
     * @tparam T
     * @tparam ReduceOp
     * @param in Input device memory to be reduced of size element number, has to be on the same
     * device as where the reducer is initialized.
     * @param in_len Input element number.
     * @param init_value Initial value for threads that doesn't correspond to a valid element.
     * @param block_size Block size used for block reduction.
     * @note reduce implicitly synchronizes execution before returning results.
     */
    template <typename T, typename ReduceOp, typename MemoryPtr>
    std::enable_if_t<
        std::is_same_v<MemoryPtr, RS_MemoryPtr> || std::is_same_v<MemoryPtr, RS_ConstMemoryPtr>,
        T>
    reduce(
        MemoryPtr in,
        RSize in_len,
        T init_value = 0,
        RSize block_size = RS_DEVICE_MAX_BLOCK_SIZE
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in.device() == device(), "Input memory device must be the same as the reducer device"
        );

        if (in_len == 0) {
            return init_value;
        }

        if (host_buffer.size() < sizeof(T)) {
            host_buffer.free();
            host_buffer = mem_provider->allocateHost(sizeof(T), true);
        }

        // Compute max block size and take shared memory limit into account
        while (block_size * sizeof(T) > RS_DEVICE_SHARED_MEMORY_MAX_SIZE) {
            block_size /= 2;
            if (block_size < 1) {
                // Unlikely to happen
                RS_THROW(RS_HostExecutionException("Shared memory limit exceeded"));
            }
        }

        // Size buffers after block size is final (smaller block => more blocks)
        if (RBigSize buffer_size = _getBufferSize(sizeof(T), in_len, block_size);
            reduce_buffer1.size() < buffer_size) {
            reduce_buffer1.free();
            reduce_buffer2.free();
            reduce_buffer1 = mem_provider->allocate(buffer_size, true);
            reduce_buffer2 = mem_provider->allocate(buffer_size, true);
        }

        *static_cast<T *>(host_buffer.ptr()) = init_value;
        castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
            _reduce<T, ReduceOp>(
                p_ptr, {in}, reduce_buffer1, reduce_buffer2, in_len, init_value, block_size
            );
        });
        return *static_cast<T *>(host_buffer.ptr());
    }

    [[nodiscard]] RS_MemoryProvider *provider() const { return mem_provider; }

    [[nodiscard]] RS_DeviceLocation device() const {
        return mem_provider ? mem_provider->device
                            : RS_DeviceLocation(RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX);
    }

   private:
    static RBigSize _getBufferSize(RSize elem_size, RSize elem_num, RSize block_size) {
        // One reduced value per block; the level 0 output bounds both buffers
        RSize max_elems_per_block = block_size * 2;

        return RBIG_SIZE(elem_size) * RS_CEIL_DIV(elem_num, max_elems_per_block);
    }

    template <typename T, typename ReduceOp, typename MemoryProvider>
    void _reduce(
        MemoryProvider *mem_provider,
        const RS_ConstMemoryPtr &in,
        const RS_MemoryPtr &buffer1,
        const RS_MemoryPtr &buffer2,
        RSize in_len,
        T init_value,
        RSize block_size
    ) {
        RSize max_elems_per_block = block_size * 2;
        RSize block_num = RS_CEIL_DIV(in_len, max_elems_per_block);

        // Perform block level reduce
        if (mem_provider->device.type == RSE_DEVICE_TYPE_CPU) {
            mem_provider->template run<RSCF_CPUBlockReduce<T, ReduceOp>>(
                {block_num, 1, true},
                static_cast<const T *>(in.ptr()),
                static_cast<T *>(buffer1.ptr()),
                in_len,
                init_value,
                max_elems_per_block
            );
        } else {
            mem_provider->template run<RSCF_BlockReduce<T, ReduceOp>>(
                {block_num * block_size, block_size, false},
                static_cast<const T *>(in.ptr()),
                static_cast<T *>(buffer1.ptr()),
                in_len,
                init_value
            );
        }

        if (in_len < max_elems_per_block) {
            // The result is ready in out since there is only 1 block
            host_buffer.copyFrom(buffer1, sizeof(T));
        } else
            // Reduce recursively
            _reduce<T, ReduceOp>(
                mem_provider,
                RS_ConstMemoryPtr(buffer1),
                buffer2,
                buffer1,
                block_num,
                init_value,
                block_size
            );
    }
};

class RS_ReduceByGroup {
   public:
    /*****************************************************************
     * Data
     *****************************************************************/
    RSize group_num = 0;
    RSize level_num = 0;
    RSize group_block_size = 0;
    RS_MemoryPtr host_buffer{};
    RS_MemoryPtr host_sizes_buffer{};
    RS_MemoryPtr sizes_buffer{};
    RS_MemoryPtr reduce_buffer1{};
    RS_MemoryPtr reduce_buffer2{};
    RS_MemoryProvider *mem_provider = nullptr;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_ReduceByGroup() = default;

    explicit RS_ReduceByGroup(RS_MemoryProvider *provider) { init(provider); }

    void init(RS_MemoryProvider *provider) {
        mem_provider = provider;
        group_num = 0;
        level_num = 0;
        host_buffer = provider->allocateHost(0, true);
        host_sizes_buffer = provider->allocateHost(0, true);
        sizes_buffer = provider->allocate(0, true);
        reduce_buffer1 = provider->allocate(0, true);
        reduce_buffer2 = provider->allocate(0, true);
        mem_provider = provider;
    }

    void free() {
        group_num = 0;
        level_num = 0;
        host_buffer.free();
        host_sizes_buffer.free();
        sizes_buffer.free();
        reduce_buffer1.free();
        reduce_buffer2.free();
        mem_provider = nullptr;
    }

    /**
     * @brief Reduce variable groups with variable sizes of values, by group. Read results back as
     * std::vector.
     * @tparam T
     * @tparam ReduceOp
     * @param in Input device memory to be reduced of size element number.
     * @param group_len Input element number for each group.
     * @param init_value Initial value for threads that doesn't correspond to a valid element.
     * @param block_size Block size used for block reduction.
     * @note reduce implicitly synchronizes execution before returning results.
     */
    template <typename ReduceOp, typename T>
    std::vector<T> reduce(
        const T *in,
        const std::vector<RSize> &group_len,
        T init_value = 0,
        RSize block_size = RS_AUTO_BLOCK_SIZE
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);

        RSize total_len = 0;
        for (auto g_len : group_len) total_len += g_len;
        return reduce<T, ReduceOp>(
            RS_ConstMemoryPtr(in, sizeof(T) * total_len, mem_provider->device, mem_provider),
            group_len,
            init_value,
            block_size
        );
    }

    /**
     * @brief Reduce variable groups with variable sizes of values, by group. Results stored to
     * device output memory. Does not synchronize.
     * @tparam T
     * @tparam ReduceOp
     * @param in Input device memory to be reduced of size element number.
     * @param out Output device memory to store results of size group number.
     * @param group_len Input element number for each group.
     * @param init_value Initial value for threads that doesn't correspond to a valid element.
     * @param block_size Block size used for block reduction.
     * @note reduce implicitly synchronizes execution before returning results.
     */
    template <typename ReduceOp, typename T>
    void reduce(
        const T *in,
        T *out,
        const std::vector<RSize> &group_len,
        T init_value = 0,
        RSize block_size = RS_AUTO_BLOCK_SIZE
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);

        RSize total_len = 0;
        for (auto g_len : group_len) total_len += g_len;
        reduce<T, ReduceOp>(
            RS_ConstMemoryPtr(in, sizeof(T) * total_len, mem_provider->device, mem_provider),
            RS_MemoryPtr(out, sizeof(T) * group_len.size(), mem_provider->device, mem_provider),
            group_len,
            init_value,
            block_size
        );
    }

    /**
     * @brief Reduce variable groups with variable sizes of values, by group. Read results back as
     * std::vector.
     * @tparam T
     * @tparam ReduceOp
     * @param array Input array to be reduced of size element number, has to be on the same
     * device as where the reducer is initialized.
     * @param group_len Input element number for each group.
     * @param init_value Initial value for threads that doesn't correspond to a valid element.
     * @param block_size Block size used for block reduction.
     * @note reduce implicitly synchronizes execution before returning results.
     */
    template <typename ReduceOp, typename T>
    std::vector<T> reduce(
        RS_Array<T, 1> array,
        const std::vector<RSize> &group_len,
        T init_value = 0,
        RSize block_size = RS_AUTO_BLOCK_SIZE
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            array.device() == device(), "Array device must be the same as the reducer device"
        );
        return reduce<T, ReduceOp>(array.data, group_len, init_value, block_size);
    }

    /**
     * @brief Reduce variable groups with variable sizes of values, by group. Read results back as
     * std::vector.
     * @tparam T
     * @tparam ReduceOp
     * @param vector Input vector to be reduced of size element number, has to be on the same
     * device as where the reducer is initialized.
     * @param group_len Input element number for each group.
     * @param init_value Initial value for threads that doesn't correspond to a valid element.
     * @param block_size Block size used for block reduction.
     * @note reduce implicitly synchronizes execution before returning results.
     */
    template <typename ReduceOp, typename T>
    std::vector<T> reduce(
        RS_Vector<T> vector,
        const std::vector<RSize> &group_len,
        T init_value = 0,
        RSize block_size = RS_AUTO_BLOCK_SIZE
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            vector.device() == device(), "Vector device must be the same as the reducer device"
        );
        return reduce<T, ReduceOp>(vector.data, group_len, init_value, block_size);
    }

    /**
     * @brief Reduce variable groups with variable sizes of values, by group. Results stored to
     * device output memory. Does not synchronize.
     * @tparam T
     * @tparam ReduceOp
     * @param array Input array to be reduced of size element number, has to be on the same
     * device as where the reducer is initialized.
     * @param array_out Output array for storing the result. The output array will be resized
     * to number of groups.
     * @param group_len Input element number for each group.
     * @param init_value Initial value for threads that doesn't correspond to a valid element.
     * @param block_size Block size used for block reduction.
     */
    template <typename ReduceOp, typename T>
    void reduce(
        RS_Array<T, 1> array,
        RS_Array<T, 1> array_out,
        const std::vector<RSize> &group_len,
        T init_value = 0,
        RSize block_size = RS_AUTO_BLOCK_SIZE
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            array.device() == device(), "Array device must be the same as the reducer device"
        );
        RS_DATA_ASSERT_ARGUMENT(
            array_out.device() == device(),
            "Output array device must be the same as the reducer device"
        );
        array_out.resize({group_len.size()}, /* copy */ false);
        reduce<T, ReduceOp>(array.data, array_out.data, group_len, init_value, block_size);
    }

    /**
     * @brief Reduce variable groups with variable sizes of values, by group. Read results back as
     * std::vector.
     * @tparam T
     * @tparam ReduceOp
     * @param in Input device memory to be reduced of size element number.
     * @param group_len Input element number for each group.
     * @param init_value Initial value for threads that doesn't correspond to a valid element.
     * @param block_size Block size used for block reduction.
     * @note reduce implicitly synchronizes execution before returning results.
     */
    template <typename T, typename ReduceOp, typename MemoryPtr>
    std::vector<T> reduce(
        MemoryPtr in,
        const std::vector<RSize> &group_len,
        T init_value = 0,
        RSize block_size = RS_AUTO_BLOCK_SIZE
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in.device() == device(), "Input memory device must be the same as the reducer device"
        );
        if (group_len.empty()) {
            return {};
        }

        if (std::all_of(group_len.begin(), group_len.end(), [](auto len) { return len == 0; })) {
            std::vector<T> result(group_len.size(), init_value);
            return result;
        }

        if (host_buffer.size() < sizeof(T) * group_len.size()) {
            host_buffer.free();
            host_buffer = mem_provider->allocateHost(sizeof(T) * group_len.size(), true);
        }

        if (block_size == RS_AUTO_BLOCK_SIZE) {
            RSize group_mean_len =
                RS_CEIL_DIV(std::reduce(group_len.begin(), group_len.end()), group_len.size());
            block_size = min2PowerAbove(group_mean_len, RS_DEVICE_MAX_BLOCK_SIZE);

            // Compute max block size and take shared memory limit into account
            while (block_size * sizeof(T) > RS_DEVICE_SHARED_MEMORY_MAX_SIZE) {
                block_size /= 2;
                if (block_size < 1) {
                    // Unlikely to happen
                    RS_THROW(RS_HostExecutionException("Shared memory limit exceeded"));
                }
            }
        }

        // Size buffers after block size is final (smaller block => more blocks)
        if (RBigSize buffer_size = _getReduceByGroupBufferSize(sizeof(T), group_len, block_size);
            reduce_buffer1.size() < buffer_size) {
            reduce_buffer1.free();
            reduce_buffer2.free();
            reduce_buffer1 = mem_provider->allocate(buffer_size, true);
            reduce_buffer2 = mem_provider->allocate(buffer_size, true);
        }

        _updateReduceSizes(block_size, group_len);

        for (size_t i = 0; i < group_len.size(); i++) host_buffer.get<T>(i) = init_value;

        std::vector<T> result;

        if (level_num != 0) {
            castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
                _reduceByBlockSize<T, ReduceOp>(p_ptr, block_size, {in}, host_buffer, init_value);
            });
            for (RSize i = 0; i < group_num; i++) result.emplace_back(host_buffer.get<T>(i));
        }

        return std::move(result);
    }

    /**
     * @brief Reduce variable groups with variable sizes of values, by group. Results stored to
     * device output memory. Does not synchronize.
     * @tparam T
     * @tparam ReduceOp
     * @param in Input device memory to be reduced of size element number.
     * @param out Output device memory to store results of size group number.
     * @param group_len Input element number for each group.
     * @param init_value Initial value for threads that doesn't correspond to a valid element
     * @param block_size Block size used for block reduction.
     */
    template <typename T, typename ReduceOp, typename MemoryPtr>
    void reduce(
        MemoryPtr in,
        RS_MemoryPtr out,
        const std::vector<RSize> &group_len,
        T init_value = 0,
        RSize block_size = RS_AUTO_BLOCK_SIZE
    ) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            in.device() == device(), "Input memory device must be the same as the reducer device"
        );
        RS_DATA_ASSERT_ARGUMENT(
            out.device() == device(), "Output memory device must be the same as the reducer device"
        );
        if (group_len.empty()) {
            return;
        }

        if (std::all_of(group_len.begin(), group_len.end(), [](auto len) { return len == 0; })) {
            std::vector<T> result(group_len.size(), init_value);
            return;
        }

        if (host_buffer.size() < sizeof(T) * group_len.size()) {
            host_buffer.free();
            host_buffer = mem_provider->allocateHost(sizeof(T) * group_len.size(), true);
        }

        if (block_size == RS_AUTO_BLOCK_SIZE) {
            RSize group_mean_len =
                RS_CEIL_DIV(std::reduce(group_len.begin(), group_len.end()), group_len.size());

            block_size = min2PowerAbove(group_mean_len, RS_DEVICE_MAX_BLOCK_SIZE);

            // Compute max block size and take shared memory limit into account
            while (block_size * sizeof(T) > RS_DEVICE_SHARED_MEMORY_MAX_SIZE) {
                block_size /= 2;
                if (block_size < 1) {
                    // Unlikely to happen
                    RS_THROW(RS_HostExecutionException("Shared memory limit exceeded"));
                }
            }
        }

        // Size buffers after block size is final (smaller block => more blocks)
        if (RBigSize buffer_size = _getReduceByGroupBufferSize(sizeof(T), group_len, block_size);
            reduce_buffer1.size() < buffer_size) {
            reduce_buffer1.free();
            reduce_buffer2.free();
            reduce_buffer1 = mem_provider->allocate(buffer_size, true);
            reduce_buffer2 = mem_provider->allocate(buffer_size, true);
        }

        _updateReduceSizes(block_size, group_len);
        if (level_num != 0) {
            castProviderPtrAndCall(mem_provider, [&](auto p_ptr) {
                _reduceByBlockSize<T, ReduceOp>(p_ptr, block_size, {in}, out, init_value);
            });
        }
    }

    [[nodiscard]] RS_MemoryProvider *provider() const { return mem_provider; }

    [[nodiscard]] RS_DeviceLocation device() const {
        return mem_provider ? mem_provider->device
                            : RS_DeviceLocation(RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX);
    }

   private:
    static RBigSize _getReduceByGroupBufferSize(
        RSize elem_size,
        const std::vector<RSize> &group_elem_num,
        RSize block_size
    ) {
        RSize total = 0;

        // One reduced value per block; the level 0 output bounds both buffers
        RSize max_elems_per_block = block_size * 2;

        for (auto elem_num : group_elem_num) total += RS_CEIL_DIV(elem_num, max_elems_per_block);
        return RBIG_SIZE(elem_size) * total;
    }

    void _updateReduceSizes(RSize block_size, const std::vector<RSize> &group_sizes) {
        // Check whether sizes are the same as last time. The stored offsets are aligned to the
        // block size, so a different block size must recompute them even on the same groups.
        bool is_same = false;
        if (group_sizes.size() == group_num && block_size == group_block_size) {
            is_same = true;
            for (RSize i = 0; i < group_sizes.size(); i++) {
                if (group_sizes[i] != host_sizes_buffer.get<RSize>(i)) {
                    is_same = false;
                    break;
                }
            }
        }
        if (is_same) return;

        // Re-compute sizes
        RSize max_elems_per_block = block_size * 2;
        level_num = 1;
        group_num = group_sizes.size();
        group_block_size = block_size;
        bool is_all_reduced = true;

        // Store order: current group sizes, input group offsets, output group offsets
        RBigSize needed_host_sizes_bytes =
            sizeof(RSize) * group_sizes.size() * 3 * RS_REDUCE_MAX_GROUP_REDUCE_LEVEL;
        if (host_sizes_buffer.size() < needed_host_sizes_bytes) {
            host_sizes_buffer = mem_provider->allocateHost(needed_host_sizes_bytes, true);
            sizes_buffer = mem_provider->allocate(needed_host_sizes_bytes, true);
            host_sizes_buffer.set(0);
            sizes_buffer.set(0);
        }

        auto h_sizes_ptr = host_sizes_buffer.as<RSize>();
        auto current_group_sizes = group_sizes;

        // The basic idea is, every turn of reduction, the size of group is kept at 1,
        // or divided by max_elems_per_block (block_size * 2), when all groups are reduced
        // to size 1, reduction is finished.
        for (; level_num <= RS_REDUCE_MAX_GROUP_REDUCE_LEVEL; level_num++) {
            // Store current group sizes
            for (auto group_size : current_group_sizes) {
                *h_sizes_ptr = group_size;
                h_sizes_ptr++;
            }

            // Compute input group offsets
            RSize offset = 0;
            for (auto group_size : current_group_sizes) {
                *h_sizes_ptr = offset;
                offset += group_size;
                h_sizes_ptr++;
            }

            // Compute output group sizes and prefix sums
            RSize psum = 0;
            for (auto group_size : current_group_sizes) {
                psum += RS_ALIGN(group_size, max_elems_per_block);
                *h_sizes_ptr = psum;
                h_sizes_ptr++;
            }
            // Update group sizes
            is_all_reduced = true;
            for (auto &group_size : current_group_sizes) {
                group_size = RS_CEIL_DIV(group_size, max_elems_per_block);
                if (group_size > 1) is_all_reduced = false;
            }
            if (is_all_reduced) break;
        }
        if (!is_all_reduced) {
            level_num = 0;
            group_num = 0;
            group_block_size = 0;
            RS_THROW(RS_HostExecutionException("Not all elements are reduced"));
        } else
            sizes_buffer.copyFrom(host_sizes_buffer, host_sizes_buffer.size(), true);
    }

    template <typename T, typename ReduceOp, typename MemoryProvider>
    void _reduceByBlockSize(
        MemoryProvider *mem_provider,
        RSize block_size,
        const RS_ConstMemoryPtr &in,
        const RS_MemoryPtr &out,
        T init_value
    ) {
        switch (block_size) {
            case 1:
                _reduce<T, ReduceOp, 1>(
                    mem_provider, in, out, reduce_buffer1, reduce_buffer2, 0, init_value
                );
                break;
            case 2:
                _reduce<T, ReduceOp, 2>(
                    mem_provider, in, out, reduce_buffer1, reduce_buffer2, 0, init_value
                );
                break;
            case 4:
                _reduce<T, ReduceOp, 4>(
                    mem_provider, in, out, reduce_buffer1, reduce_buffer2, 0, init_value
                );
                break;
            case 8:
                _reduce<T, ReduceOp, 8>(
                    mem_provider, in, out, reduce_buffer1, reduce_buffer2, 0, init_value
                );
                break;
            case 16:
                _reduce<T, ReduceOp, 16>(
                    mem_provider, in, out, reduce_buffer1, reduce_buffer2, 0, init_value
                );
                break;
            case 32:
                _reduce<T, ReduceOp, 32>(
                    mem_provider, in, out, reduce_buffer1, reduce_buffer2, 0, init_value
                );
                break;
            case 64:
                _reduce<T, ReduceOp, 64>(
                    mem_provider, in, out, reduce_buffer1, reduce_buffer2, 0, init_value
                );
                break;
            case 128:
                _reduce<T, ReduceOp, 128>(
                    mem_provider, in, out, reduce_buffer1, reduce_buffer2, 0, init_value
                );
                break;
            case 256:
                _reduce<T, ReduceOp, 256>(
                    mem_provider, in, out, reduce_buffer1, reduce_buffer2, 0, init_value
                );
                break;
            case 512:
                _reduce<T, ReduceOp, 512>(
                    mem_provider, in, out, reduce_buffer1, reduce_buffer2, 0, init_value
                );
                break;
            default:
                _reduce<T, ReduceOp, 1024>(
                    mem_provider, in, out, reduce_buffer1, reduce_buffer2, 0, init_value
                );
                break;
        }
    }

    template <typename T, typename ReduceOp, RSize BlockSize, typename MemoryProvider>
    void _reduce(
        MemoryProvider *mem_provider,
        const RS_ConstMemoryPtr &in,
        const RS_MemoryPtr &out,
        const RS_MemoryPtr &buffer1,
        const RS_MemoryPtr &buffer2,
        RSize level,
        T init_value
    ) {
        RSize sizes_offset = level * group_num * 3;
        // in_len is the last prefix sum of the aligned output group sizes
        RSize in_len = host_sizes_buffer.get<RSize>(sizes_offset + group_num * 3 - 1);

        // Group-aware block reduce: gather + pad fused into the load
        RSize max_elems_per_block = BlockSize * 2;
        RSize reduce_block_num = RS_CEIL_DIV(in_len, max_elems_per_block);

        if (mem_provider->device.type == RSE_DEVICE_TYPE_CPU) {
            mem_provider->template run<RSCF_HostGroupBlockReduce<T, ReduceOp>>(
                {reduce_block_num, 1, true},
                static_cast<const T *>(in.ptr()),
                static_cast<T *>(buffer1.ptr()),
                sizes_buffer.as<RSize>() + sizes_offset,
                sizes_buffer.as<RSize>() + sizes_offset + group_num,
                sizes_buffer.as<RSize>() + sizes_offset + group_num * 2,
                group_num,
                init_value,
                max_elems_per_block
            );
        } else {
            mem_provider->template run<RSCF_GroupBlockReduce<T, ReduceOp>>(
                {reduce_block_num * BlockSize, BlockSize, false},
                static_cast<const T *>(in.ptr()),
                static_cast<T *>(buffer1.ptr()),
                sizes_buffer.as<RSize>() + sizes_offset,
                sizes_buffer.as<RSize>() + sizes_offset + group_num,
                sizes_buffer.as<RSize>() + sizes_offset + group_num * 2,
                group_num,
                init_value
            );
        }

        if (level + 1 == level_num) {
            if (out != buffer1) {
                out.copyFrom(buffer1, sizeof(T) * group_num, true);
            }
        } else
            // Reduce recursively, swapping buffers to avoid aliasing
            _reduce<T, ReduceOp, BlockSize>(
                mem_provider, buffer1, out, buffer2, buffer1, level + 1, init_value
            );
    }
};

struct RS_ReduceMaxOp {
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) {
        return RS_MAX(a, b);
    }
};

struct RS_ReduceMinOp {
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) {
        return RS_MIN(a, b);
    }
};

struct RS_ReduceSumOp {
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) {
        return a + b;
    }
};

struct RS_ReduceOrOp {
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) {
        return a || b;
    }
};

struct RS_ReduceAndOp {
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) {
        return a && b;
    }
};

struct RS_ReduceVecMaxOp {
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) {
        return a.max(b);
    }
};

struct RS_ReduceVecMinOp {
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) {
        return a.min(b);
    }
};

struct RS_ReduceVecSumOp {
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) {
        return a + b;
    }
};

struct RS_ReduceBoundUnionOp {
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) {
        return a.unionWith(b);
    }
};

struct RS_ReduceBoundIntersectionOp {
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER T operator()(const T &a, const T &b) {
        return a.intersectionWith(b);
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_REDUCE_H
