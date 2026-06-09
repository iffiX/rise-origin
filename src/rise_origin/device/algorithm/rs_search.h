// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SEARCH_H
#define RS_SEARCH_H

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace device {

/**
 * @brief Search for the prefix sum the index belongs to.
 * @param index The index.
 * @param prefix_sum Prefix sum array,
 * the first element is the size of the first element,
 * the last element is the size of all elements.
 * Eg:
 * group size = [1, 2, 3, 4, 5],
 * prefix sum = [1, 3, 6, 10, 15]
 * @param num The number of prefix sums.
 * @return SearchResult, index is the prefix sum index, local_index is the index
 * within the group, if input index is larger than the last element of the prefix sum, return
 * {RS_NULL_INDEX, RS_NULL_INDEX}.
 */
class RSCF_BinaryPrefixSearch : public RS_ComputeFunction<RSCF_BinaryPrefixSearch> {
   public:
    struct SearchResult {
        RIndex index;
        RIndex local_index;
    };

    HOST_DEVICE_INLINE_QUALIFIER static SearchResult
    run(RIndex index, const RSize *prefix_sum, RSize num) {
        if (prefix_sum[num - 1] <= index) {
            return {RS_NULL_INDEX, RS_NULL_INDEX};
        }
        if (num == 1) {
            return {0, index};
        }
        // At least 2 groups, and gindex is within max boundary
        RIndex i_min = 0, i_max = num - 1;
        while (i_max > i_min) {
            RIndex i_mid = (i_min + i_max) / 2;
            RSize low = prefix_sum[i_mid];
            bool has_low = low <= index;
            bool has_high = prefix_sum[i_mid + 1] > index;
            if (has_low && has_high) {
                return {i_mid + 1, index - low};
            } else if (has_low) {
                i_min = i_mid;
            } else {
                i_max = i_mid;
                if (i_max == 0) return {0, index};
            }
        }
        return {RS_NULL_INDEX, RS_NULL_INDEX};
    }
};

class RSCF_BinaryPrefixSearchWithMemPtr : public RSCF_BinaryPrefixSearch {
   public:
    template <typename MemoryPtr>
    HOST_DEVICE_INLINE_QUALIFIER static SearchResult run(RIndex index, MemoryPtr ptr) {
        return RSCF_BinaryPrefixSearch::run(index, ptr.ptr(), ptr.size / sizeof(RSize));
    }
};

class RSCF_BinaryPrefixSearchWithArray1DLike : public RSCF_BinaryPrefixSearch {
   public:
    template <typename Array1DLike>
    HOST_DEVICE_INLINE_QUALIFIER static SearchResult run(RIndex index, Array1DLike array) {
        return RSCF_BinaryPrefixSearch::run(index, array.ptr(), array.size());
    }
};

/**
 * @brief Search for the cdf bucket the query value belongs to.
 * @param r The query value.
 * @param cdf Monotonically increasing prefix-sum (cumulative distribution) array.
 * Eg:
 * weights = [0.5, 1.0, 2.5, 1.0],
 * cdf     = [0.5, 1.5, 4.0, 5.0]
 * @param num The number of cdf entries.
 * @return Index of the first cdf entry strictly greater than r (std::upper_bound
 * semantics). Out-of-range (r >= cdf[num - 1]) clamps to the last bucket, so a
 * query scaled by cdf[num - 1] always maps to a valid index. Returns
 * RS_NULL_INDEX only when num == 0.
 */
template <typename FloatT>
class RSCF_BinaryPrefixSearchFloat
    : public RS_ComputeFunction<RSCF_BinaryPrefixSearchFloat<FloatT>> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static RIndex run(FloatT r, const FloatT *cdf, RSize num) {
        if (num == 0) return RS_NULL_INDEX;
        if (num == 1 || r < cdf[0]) return 0;
        if (r >= cdf[num - 1]) return num - 1;
        RIndex i_min = 0, i_max = num - 1;
        while (i_max - i_min > 1) {
            RIndex i_mid = (i_min + i_max) / 2;
            if (cdf[i_mid] > r) {
                i_max = i_mid;
            } else {
                i_min = i_mid;
            }
        }
        return i_max;
    }
};

template <typename FloatT>
class RSCF_BinaryPrefixSearchFloatWithMemPtr : public RSCF_BinaryPrefixSearchFloat<FloatT> {
   public:
    template <typename MemoryPtr>
    HOST_DEVICE_INLINE_QUALIFIER static RIndex run(FloatT r, MemoryPtr ptr) {
        return RSCF_BinaryPrefixSearchFloat<FloatT>::run(
            r, ptr.template as<FloatT>(), ptr.size / sizeof(FloatT)
        );
    }
};

template <typename FloatT>
class RSCF_BinaryPrefixSearchFloatWithArray1DLike : public RSCF_BinaryPrefixSearchFloat<FloatT> {
   public:
    template <typename Array1DLike>
    HOST_DEVICE_INLINE_QUALIFIER static RIndex run(FloatT r, Array1DLike array) {
        return RSCF_BinaryPrefixSearchFloat<FloatT>::run(r, array.ptr(), array.size());
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_SEARCH_H