// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_COMPUTE_KERNEL_H
#define RS_COMPUTE_KERNEL_H

#include <map>
#include <string>

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/concurrency/rs_cpu_block.h"

#define RS_COMPUTE_FUNCTION_CHECK_HAS(something)                                       \
    template <typename T, typename = void>                                             \
    struct Has##something : std::false_type {};                                        \
    template <typename T>                                                              \
    struct Has##something<T, std::void_t<typename T::something>> : std::true_type {}

#define RS_COMPUTE_FUNCTION_CHECK_IS_OR_FALSE(something)                               \
    template <typename T, typename = void>                                             \
    struct ValueOf##something : std::false_type {};                                    \
    template <typename T>                                                              \
    struct ValueOf##something<T, std::void_t<typename T::something>> : T::something {}

#define RS_COMPUTE_FUNCTION_CHECK_IS_OR_TRUE(something)                                \
    template <typename T, typename = void>                                            \
    struct ValueOf##something : std::true_type {};                                     \
    template <typename T>                                                              \
    struct ValueOf##something<T, std::void_t<typename T::something>> : T::something {}

// Does not check templates
#define RS_COMPUTE_FUNCTION_CHECK_METHOD(method)                                        \
    template <typename T, typename = void>                                              \
    struct HasMethod##method : std::false_type {};                                      \
    template <typename T>                                                               \
    struct HasMethod##method<T, std::void_t<decltype(&T::method)>> : std::true_type {}; \
    static constexpr bool HasMethod##method##_V = HasMethod##method<DerivedComputeFunction>::value;

namespace rise {

namespace device {

/**
 * @brief Non-template base class to hold all the thread-local storage
 *        and the unified interface for retrieving block/thread indices.
 *        This ensures we have only one set of TLS for all derived compute functions.
 */
class RS_ComputeFunctionBase {
   public:
#ifdef __CUDA_ARCH__
    /*****************************************************************
     * Device-side implementations
     *****************************************************************/
    DEVICE_INLINE_QUALIFIER static RSize blockDimX() { return blockDim.x; }
    DEVICE_INLINE_QUALIFIER static RSize blockDimY() { return blockDim.y; }
    DEVICE_INLINE_QUALIFIER static RSize blockDimZ() { return blockDim.z; }

    DEVICE_INLINE_QUALIFIER static RIndex blockIdxX() { return blockIdx.x; }
    DEVICE_INLINE_QUALIFIER static RIndex blockIdxY() { return blockIdx.y; }
    DEVICE_INLINE_QUALIFIER static RIndex blockIdxZ() { return blockIdx.z; }

    DEVICE_INLINE_QUALIFIER static RIndex threadIdxX() { return threadIdx.x; }
    DEVICE_INLINE_QUALIFIER static RIndex threadIdxY() { return threadIdx.y; }
    DEVICE_INLINE_QUALIFIER static RIndex threadIdxZ() { return threadIdx.z; }

    DEVICE_INLINE_QUALIFIER static std::byte* sharedMem() {
        extern __shared__ std::byte sharedMemPtr[];
        return sharedMemPtr;
    }

    static void setIndicesAndBlockSize(
        RSize /*blockDimX*/,
        RSize /*blockDimY*/,
        RSize /*blockDimZ*/,
        RIndex /*blockIdxX*/,
        RIndex /*blockIdxY*/,
        RIndex /*blockIdxZ*/,
        RIndex /*threadIdxX*/,
        RIndex /*threadIdxY*/,
        RIndex /*threadIdxZ*/
    ) {
        // No-op on device side
    }

    static void setBlock(RS_CPUBlock* /*b*/) {
        // No-op on device side
    }

    static void setSharedMem(void* /*ptr*/) {
        // No-op on device side
    }

#else
    /*****************************************************************
     * Host-side thread-local storage
     *****************************************************************/
    inline static thread_local RIndex block_idx_x = 0;
    inline static thread_local RIndex block_idx_y = 0;
    inline static thread_local RIndex block_idx_z = 0;

    inline static thread_local RIndex thread_idx_x = 0;
    inline static thread_local RIndex thread_idx_y = 0;
    inline static thread_local RIndex thread_idx_z = 0;

    inline static thread_local RSize block_dim_x = 1;
    inline static thread_local RSize block_dim_y = 1;
    inline static thread_local RSize block_dim_z = 1;

    inline static thread_local RS_CPUBlock* block = nullptr;
    inline static thread_local void* shared_mem_ptr = nullptr;

    static RSize blockDimX() { return block_dim_x; }
    static RSize blockDimY() { return block_dim_y; }
    static RSize blockDimZ() { return block_dim_z; }

    static RIndex blockIdxX() { return block_idx_x; }
    static RIndex blockIdxY() { return block_idx_y; }
    static RIndex blockIdxZ() { return block_idx_z; }

    static RIndex threadIdxX() { return thread_idx_x; }
    static RIndex threadIdxY() { return thread_idx_y; }
    static RIndex threadIdxZ() { return thread_idx_z; }

    static std::byte* sharedMem() {
        assert(shared_mem_ptr != nullptr);
        return static_cast<std::byte*>(shared_mem_ptr);
    }

    static void setIndicesAndBlockSize(
        RSize blockDimX,
        RSize blockDimY,
        RSize blockDimZ,
        RIndex blockIdxX,
        RIndex blockIdxY,
        RIndex blockIdxZ,
        RIndex threadIdxX,
        RIndex threadIdxY,
        RIndex threadIdxZ
    ) {
        block_dim_x = blockDimX;
        block_dim_y = blockDimY;
        block_dim_z = blockDimZ;

        block_idx_x = blockIdxX;
        block_idx_y = blockIdxY;
        block_idx_z = blockIdxZ;

        thread_idx_x = threadIdxX;
        thread_idx_y = threadIdxY;
        thread_idx_z = threadIdxZ;
    }

    static void setBlock(RS_CPUBlock* b) { block = b; }

    static void setSharedMem(void* ptr) { shared_mem_ptr = ptr; }
#endif
};

template <typename DerivedComputeFunction>
class RS_ComputeFunction : public RS_ComputeFunctionBase {
   public:
    RS_COMPUTE_FUNCTION_CHECK_HAS(SharedMemLayout);

    RS_COMPUTE_FUNCTION_CHECK_IS_OR_TRUE(IsCudaSupported);
    RS_COMPUTE_FUNCTION_CHECK_IS_OR_TRUE(IsCPUSupported);

    RS_COMPUTE_FUNCTION_CHECK_IS_OR_FALSE(IsSyncBlockRequired);

    /*****************************************************************
     * Attribute query methods
     *****************************************************************/
    static constexpr size_t sharedMemLayoutSize() {
        if constexpr (HasSharedMemLayout<DerivedComputeFunction>::value) {
            return sizeof(typename DerivedComputeFunction::SharedMemLayout);
        } else {
            return 0;
        }
    }

    static constexpr bool isCudaSupported() {
        return ValueOfIsCudaSupported<DerivedComputeFunction>::value;
    }
    static constexpr bool isCPUSupported() {
        return ValueOfIsCPUSupported<DerivedComputeFunction>::value;
    }

    static constexpr bool isSyncBlockRequired() {
        return ValueOfIsSyncBlockRequired<DerivedComputeFunction>::value;
    }

    // Note: compute function runner must invoke this function, we cannot put
    // static_assert outside since at the time of initialization of:
    // class A : public RS_ComputeFunction<A> {};
    // the DerivedComputeFunction will not be initialized yet.
    static constexpr void validate() {
        static_assert(
            ValueOfIsCudaSupported<DerivedComputeFunction>::value ||
                ValueOfIsCPUSupported<DerivedComputeFunction>::value,
            "Compute function must be either IsCudaSupported or IsCPUSupported."
        );
    }

    static std::map<std::string, std::string> getProperties() {
        return {
            {"SharedMemLayoutSize", std::to_string(sharedMemLayoutSize())},
            {"IsCudaSupported", std::to_string(isCudaSupported())},
            {"IsCPUSupported", std::to_string(isCPUSupported())},
            {"IsSyncBlockRequired", std::to_string(isSyncBlockRequired())}
        };
    }

    /*****************************************************************
     * Unified interface
     *****************************************************************/
#ifdef __CUDA_ARCH__
    DEVICE_INLINE_QUALIFIER static void syncThreadsInBlock() { __syncthreads(); }

    DEVICE_INLINE_QUALIFIER static void threadfence() { __threadfence(); }
#else
    static void syncThreadsInBlock() {
        // Defer to the RS_CPUBlock's sync
        assert(block != nullptr && "RS_CPUBlock pointer must be set before syncing.");
        block->syncThreadsInBlockImpl<RS_ComputeFunction>(thread_idx_x);
    }

    static void threadfence() {
        // No-op on host side
    }
#endif

    HOST_DEVICE_INLINE_QUALIFIER static RIndex index1D() {
        return blockIdxX() * blockDimX() + threadIdxX();
    }

    HOST_DEVICE_INLINE_QUALIFIER static RIndex index2D() {
        return (blockIdxY() * blockDimY() + threadIdxY()) * blockDimX() + threadIdxX();
    }

    HOST_DEVICE_INLINE_QUALIFIER static RIndex index3D() {
        return ((blockIdxZ() * blockDimZ() + threadIdxZ()) * blockDimY() + threadIdxY()) *
                   blockDimX() +
               threadIdxX();
    }

    HOST_DEVICE_INLINE_QUALIFIER static RIndex threadIndex1D() { return threadIdxX(); }

    HOST_DEVICE_INLINE_QUALIFIER static RIndex threadIndex2D() {
        return threadIdxY() * blockDimX() + threadIdxX();
    }

    HOST_DEVICE_INLINE_QUALIFIER static RIndex threadIndex3D() {
        return (threadIdxZ() * blockDimY() + threadIdxY()) * blockDimX() + threadIdxX();
    }

    HOST_DEVICE_INLINE_QUALIFIER static RSize blockSize1D() { return blockDimX(); }

    HOST_DEVICE_INLINE_QUALIFIER static RSize blockSize2D() { return blockDimX() * blockDimY(); }

    HOST_DEVICE_INLINE_QUALIFIER static RSize blockSize3D() {
        return blockDimX() * blockDimY() * blockDimZ();
    }

    template <typename StructType, typename MemberType>
    HOST_DEVICE_INLINE_QUALIFIER static constexpr size_t offsetOf(MemberType StructType::*member) {
        return reinterpret_cast<std::size_t>(&(static_cast<StructType*>(nullptr)->*member));
    }

    /**
     * @brief access shared memory at a specific member and index
     */
    template <typename StructType, typename MemberType>
    HOST_DEVICE_INLINE_QUALIFIER static MemberType&
    sharedMemAt(MemberType StructType::*member, RSize block_size, RIndex index) {
        static_assert(
            HasSharedMemLayout<DerivedComputeFunction>::value,
            "Compute kernel must define a valid SharedMemLayout."
        );
        size_t member_offset = offsetOf(member) * block_size;
        size_t index_offset = index * sizeof(MemberType);
        return *reinterpret_cast<MemberType*>(sharedMem() + member_offset + index_offset);
    }

    template <typename StructType, typename MemberType>
    HOST_DEVICE_INLINE_QUALIFIER static MemberType& sharedMemAt1D(MemberType StructType::*member) {
        static_assert(
            HasSharedMemLayout<DerivedComputeFunction>::value,
            "Compute kernel must define a valid SharedMemLayout."
        );
        size_t member_offset = offsetOf(member) * blockSize1D();
        size_t index_offset = threadIndex1D() * sizeof(MemberType);
        return *reinterpret_cast<MemberType*>(sharedMem() + member_offset + index_offset);
    }

    template <typename StructType, typename MemberType>
    HOST_DEVICE_INLINE_QUALIFIER static MemberType& sharedMemAt2D(MemberType StructType::*member) {
        static_assert(
            HasSharedMemLayout<DerivedComputeFunction>::value,
            "Compute kernel must define a valid SharedMemLayout."
        );
        size_t member_offset = offsetOf(member) * blockSize2D();
        size_t index_offset = threadIndex2D() * sizeof(MemberType);
        return *reinterpret_cast<MemberType*>(sharedMem() + member_offset + index_offset);
    }

    template <typename StructType, typename MemberType>
    HOST_DEVICE_INLINE_QUALIFIER static MemberType& sharedMemAt3D(MemberType StructType::*member) {
        static_assert(
            HasSharedMemLayout<DerivedComputeFunction>::value,
            "Compute kernel must define a valid SharedMemLayout."
        );
        size_t member_offset = offsetOf(member) * blockSize3D();
        size_t index_offset = threadIndex3D() * sizeof(MemberType);
        return *reinterpret_cast<MemberType*>(sharedMem() + member_offset + index_offset);
    }
    /**
     * @note Child classes must implement this, either as method or templated method
     */
    // DEVICE_INLINE_QUALIFIER void run(...) { ... }
    // or HOST_DEVICE_INLINE_QUALIFIER void run(...) { ... }
};

}  // namespace device

}  // namespace rise

#endif  // RS_COMPUTE_KERNEL_H