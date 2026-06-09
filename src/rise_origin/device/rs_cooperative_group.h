// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_COOPERATIVE_GROUP_H
#define RS_COOPERATIVE_GROUP_H

#ifdef __CUDACC__
#include <cooperative_groups.h>
#endif

namespace rise {

namespace device {

#ifdef __CUDA_ARCH__
namespace cg = cooperative_groups;

/**
 * @brief A synchronized group where all thread in the group are executing instructions
 * in lockstep, corresponds to warp in CUDA.
 */
template <RSize GroupSize_ = CUDA_MAX_WARP_SIZE>
class RS_SynchronizedGroup : public cg::thread_block_tile<GroupSize_> {
   public:
    using BaseGroup = cg::thread_block_tile<GroupSize_>;
    constexpr static RSize GroupSize = GroupSize_;

    template <typename ParentType>
    DEVICE_INLINE_QUALIFIER explicit RS_SynchronizedGroup(ParentType group)
        : cg::thread_block_tile<GroupSize_>(cg::tiled_partition<GroupSize_>(group)) {}

    /// @brief Return the elected thread rank in the group, -1 is invalid
    DEVICE_INLINE_QUALIFIER int elect(int predicate) {
        if constexpr (GroupSize_ > 1) {
            unsigned int group_mask = this->ballot(predicate);
            return group_mask > 0 ? __ffs(static_cast<int>(group_mask)) - 1 : -1;
        } else
            return static_cast<int>(predicate > 0) - 1;
    }

    DEVICE_INLINE_QUALIFIER int random_elect(int predicate, int seed) {
        if constexpr (GroupSize_ > 1) {
            unsigned int group_mask = this->ballot(predicate);

            unsigned int shift = seed & 31;
            unsigned int rotated_mask = (group_mask >> shift) | (group_mask << (32 - shift));
            int bit_index = __ffs(static_cast<int>(rotated_mask)) - 1;

            return group_mask > 0 ? ((bit_index + static_cast<int>(shift)) & 31) : -1;
        } else
            return static_cast<int>(predicate > 0) - 1;
    }

    template <typename ValueType>
    DEVICE_INLINE_QUALIFIER ValueType shfl(ValueType value, int src_rank) {
        static_assert(
            sizeof(ValueType) % 4 == 0, "Unsupported value type, size must be multiple of 4"
        );
        if constexpr (sizeof(ValueType) == 4 || sizeof(ValueType) == 8) {
            return BaseGroup::shfl(value, src_rank);
        } else if constexpr (sizeof(ValueType) % 8 == 0) {
            ValueType result;
            auto result_ptr = reinterpret_cast<uint64_t *>(&result),
                 value_ptr = reinterpret_cast<uint64_t *>(&value);
            for (int i = 0; i < sizeof(ValueType) / 8; i++) {
                *result_ptr = reinterpret_cast<uint64_t>(BaseGroup::shfl(*value_ptr, src_rank));
                ++result_ptr;
                ++value_ptr;
            }
            return result;
        } else if constexpr (sizeof(ValueType) % 4 == 0) {
            ValueType result;
            auto result_ptr = reinterpret_cast<uint32_t *>(&result),
                 value_ptr = reinterpret_cast<uint32_t *>(&value);
            for (int i = 0; i < sizeof(ValueType) / 4; i++) {
                *result_ptr = reinterpret_cast<uint32_t>(BaseGroup::shfl(*value_ptr, src_rank));
                ++result_ptr;
                ++value_ptr;
            }
            return result;
        }
    }
};

template <RSize GroupSize = CUDA_MAX_WARP_SIZE>
DEVICE_INLINE_QUALIFIER RS_SynchronizedGroup<GroupSize> getCurrentSynchronizedGroup() {
    return RS_SynchronizedGroup<GroupSize>(cg::this_thread_block());
}

#else

template <RSize GroupSize_ = 1>
class RS_SynchronizedGroup {
   public:
    static_assert(GroupSize_ == 1, "For host, only group size 1 is supported");
    constexpr static RSize GroupSize = 1;
    HOST_INLINE_QUALIFIER unsigned int thread_rank() { return 0; }
    HOST_INLINE_QUALIFIER void sync() {}
    HOST_INLINE_QUALIFIER unsigned int ballot(int predicate) { return predicate > 0; }
    /// @brief Return the elected thread rank in the group, -1 is invalid
    HOST_INLINE_QUALIFIER int elect(int predicate) { return static_cast<int>(predicate > 0) - 1; }
    HOST_INLINE_QUALIFIER int random_elect(int predicate, int seed) {
        return static_cast<int>(predicate > 0) - 1;
    }
    HOST_INLINE_QUALIFIER int any(int predicate) { return predicate > 0; }
    HOST_INLINE_QUALIFIER int all(int predicate) { return predicate > 0; }
    template <typename ValueType>
    HOST_INLINE_QUALIFIER ValueType shfl(ValueType value, int src_rank) {
        return value;
    }
};

template <RSize GroupSize = 1>
HOST_INLINE_QUALIFIER RS_SynchronizedGroup<GroupSize> getCurrentSynchronizedGroup() {
    return RS_SynchronizedGroup<GroupSize>();
}

#endif

}  // namespace device

}  // namespace rise

#endif  // RS_COOPERATIVE_GROUP_H
