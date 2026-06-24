// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_LAYOUT_H
#define RS_LAYOUT_H

#include <vector>

#include "rise_origin/device/algorithm/rs_search.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace device {

class RS_DeviceLayout {
   public:
    struct SegmentInfo {
        RSize size;
        RSize offset;
        RIndex segment_id;

        HOST_DEVICE_INLINE_QUALIFIER SegmentInfo(RSize size, RSize offset, RIndex segment_id)
            : size(size), offset(offset), segment_id(segment_id) {}
    };

    struct SegmentSearchResult {
        RIndex segment_id;
        RIndex segment_index;
        RIndex in_segment_index;
        RSize global_index;

        HOST_DEVICE_INLINE_QUALIFIER SegmentSearchResult(
            RIndex segment_id,
            RIndex segment_index,
            RIndex in_segment_index,
            RSize global_index
        )
            : segment_id(segment_id),
              segment_index(segment_index),
              in_segment_index(in_segment_index),
              global_index(global_index) {}
    };

    /*****************************************************************
     * Data
     *****************************************************************/
    uint64_t version_id = 0;
    RSize total_size = 0;
    RS_Vector<SegmentInfo> segments;
    RS_Vector<RSize> segment_size_prefix_sums;

    RS_DeviceLayout() = default;

    explicit RS_DeviceLayout(RS_MemoryProvider* provider) { init(provider); }

    RS_DeviceLayout(const RS_DeviceLayout& array) = default;

    void init(RS_MemoryProvider* provider) {
        version_id = 0;
        total_size = 0;
        segments.init(provider);
        segment_size_prefix_sums.init(provider);
    }

    void free() {
        total_size = 0;
        segments.free();
        segment_size_prefix_sums.free();
    }

    HOST_DEVICE_INLINE_QUALIFIER uint64_t versionID() const { return version_id; }

    HOST_DEVICE_INLINE_QUALIFIER RSize size() const { return total_size; }

    HOST_DEVICE_INLINE_QUALIFIER RSize segmentNum() const { return segments.size(); }

    HOST_DEVICE_INLINE_QUALIFIER RSize segmentSize(RIndex index) const {
        assert(index < segments.size());
        return segments[index].size;
    }

    HOST_DEVICE_INLINE_QUALIFIER RSize segmentOffset(RIndex index) const {
        assert(index < segments.size());
        return segments[index].offset;
    }

    HOST_DEVICE_INLINE_QUALIFIER SegmentSearchResult
    searchForBelongedSegment(RIndex thread_index, RIndex start_segment_index = 0) const {
        auto local_search_result = RSCF_BinaryPrefixSearchWithArray1DLike::run(
            thread_index +
                (start_segment_index == 0 ? 0 : segment_size_prefix_sums[start_segment_index - 1]),
            segment_size_prefix_sums
        );
        if (local_search_result.index == RS_NULL_INDEX)
            return {RS_NULL_INDEX, RS_NULL_INDEX, RS_NULL_INDEX, RS_NULL_INDEX};
        auto& segment_info = segments[local_search_result.index];
        return {
            segment_info.segment_id,
            local_search_result.index,
            local_search_result.local_index,
            local_search_result.local_index + segment_info.offset
        };
    }
};

class RSCF_ResolveThreadHandleFromLayout
    : public RS_ComputeFunction<RSCF_ResolveThreadHandleFromLayout> {
   public:
    template <typename Array1DLike>
    HOST_DEVICE_INLINE_QUALIFIER static void run(RS_DeviceLayout layout, Array1DLike handles) {
        RIndex i = index1D();
        auto search_result = layout.searchForBelongedSegment(i);
        handles[i] = search_result.global_index;
    }
};

}  // namespace device

/**
 * @brief Computes a layout's over-allocated reserved capacity from its packed total capacity.
 *
 * The reserve grows at 2x below double_below_threshold, is exact above exact_above_threshold, and
 * fades linearly between the two.
 */
class RS_LayoutGrowthPolicy {
   public:
    RSize double_below_threshold = (1u << 20);
    RSize exact_above_threshold = (16u << 20);
    RSize minimum_capacity = 16384;

    RS_LayoutGrowthPolicy() = default;

    /**
     * @brief Constructs a growth policy.
     *
     * @param double_below_threshold Packed total at or below which the reserve grows at 2x.
     * @param exact_above_threshold Packed total at or above which the reserve is exact.
     * @param minimum_capacity Lower bound on the reserve while non-empty.
     */
    RS_LayoutGrowthPolicy(
        RSize double_below_threshold,
        RSize exact_above_threshold,
        RSize minimum_capacity = 16384
    );

    /**
     * @brief Returns the growth multiplier for a packed total capacity.
     *
     * @param total_capacity Current packed total capacity.
     * @return Multiplier in [1, 2].
     */
    [[nodiscard]] double getMultiplier(RSize total_capacity) const;

    /**
     * @brief Returns the reserve for the current packed total, re-targeting only outside the band
     *        [total_capacity, total_capacity * multiplier].
     *
     * @param total_capacity Current packed total capacity.
     * @param old_reserved_capacity Previously stored reserved capacity.
     * @return New reserved capacity.
     */
    [[nodiscard]] RSize
    getReservedCapacity(RSize total_capacity, RSize old_reserved_capacity) const;
};

class RS_CompactLayout {
   public:
    /**
     * @brief Constructs a compact layout with growth policy thresholds.
     *
     * @param double_below_threshold Packed total at or below which the reserve grows at 2x.
     * @param exact_above_threshold Packed total at or above which the reserve is exact.
     * @param minimum_capacity Lower bound on the reserve while non-empty.
     */
    RS_CompactLayout(
        RSize double_below_threshold = (1u << 20),
        RSize exact_above_threshold = (16u << 20),
        RSize minimum_capacity = 16384
    );

    /**
     * @brief Replaces the growth policy thresholds.
     *
     * @param double_below_threshold Packed total at or below which the reserve grows at 2x.
     * @param exact_above_threshold Packed total at or above which the reserve is exact.
     * @param minimum_capacity Lower bound on the reserve while non-empty.
     */
    void setGrowthPolicy(
        RSize double_below_threshold,
        RSize exact_above_threshold,
        RSize minimum_capacity
    );

    void clear();

    void add(RSize size, RIndex segment_id, bool is_active = true);

    void setSegmentIsActive(RIndex index, bool is_active);

    void remove(RIndex index);

    [[nodiscard]] uint64_t versionID() const;

    [[nodiscard]] RSize segmentNum() const;

    [[nodiscard]] RSize segmentSize(RIndex index) const;

    [[nodiscard]] RSize segmentOffset(RIndex index) const;

    [[nodiscard]] bool segmentIsActive(RIndex index) const;

    [[nodiscard]] RSize size() const;

    [[nodiscard]] RSize capacity() const;

    [[nodiscard]] RIndex find(RIndex segment_id) const;

    [[nodiscard]] const std::vector<RSize>& getSegmentSizes() const;

    [[nodiscard]] const std::vector<RSize>& getSegmentOffsets() const;

    [[nodiscard]] const std::vector<RIndex>& getSegmentIds() const;

    [[nodiscard]] const std::vector<bool>& getSegmentIsActive() const;

    void syncDeviceLayout(device::RS_DeviceLayout& device_layout) const;

   private:
    uint64_t _version_id = 0;
    RSize _total_active_size = 0;
    RSize _total_reserved_capacity = 0;
    RS_LayoutGrowthPolicy _growth;
    std::vector<RSize> _segment_sizes;
    std::vector<RSize> _segment_offsets;
    std::vector<RSize> _segment_size_prefix_sums;

    std::vector<RIndex> _segment_indices;
    std::vector<bool> _segment_is_active;
    std::unordered_map<RIndex, RIndex> _segment_id_to_index;

    void _updateSegmentOffsetsAndSizePrefixSums(RIndex update_start_index);
    void _refreshReservedCapacity();
};

class RS_PooledLayout {
   public:
    /**
     * @brief Constructs a pooled layout with growth policy thresholds.
     *
     * @param double_below_threshold Packed total at or below which the reserve grows at 2x.
     * @param exact_above_threshold Packed total at or above which the reserve is exact.
     * @param minimum_capacity Lower bound on the reserve while non-empty.
     */
    RS_PooledLayout(
        RSize double_below_threshold = (1u << 20),
        RSize exact_above_threshold = (16u << 20),
        RSize minimum_capacity = 16384
    );

    /**
     * @brief Replaces the growth policy thresholds.
     *
     * @param double_below_threshold Packed total at or below which the reserve grows at 2x.
     * @param exact_above_threshold Packed total at or above which the reserve is exact.
     * @param minimum_capacity Lower bound on the reserve while non-empty.
     */
    void setGrowthPolicy(
        RSize double_below_threshold,
        RSize exact_above_threshold,
        RSize minimum_capacity
    );

    void clear();

    void add(RSize size, RIndex segment_id, bool is_active = true);

    void setSegmentIsActive(RIndex index, bool is_active);

    void remove(RIndex index);

    [[nodiscard]] uint64_t versionID() const;

    [[nodiscard]] RSize segmentNum() const;

    [[nodiscard]] RSize segmentSize(RIndex index) const;

    [[nodiscard]] RSize segmentOffset(RIndex index) const;

    [[nodiscard]] bool segmentIsActive(RIndex index) const;

    [[nodiscard]] RSize size() const;

    [[nodiscard]] RSize capacity() const;

    [[nodiscard]] RIndex find(RIndex segment_id) const;

    [[nodiscard]] const std::vector<RSize>& getSegmentSizes() const;

    [[nodiscard]] const std::vector<RSize>& getSegmentOffsets() const;

    [[nodiscard]] const std::vector<RIndex>& getSegmentIds() const;

    [[nodiscard]] const std::vector<bool>& getSegmentIsActive() const;

    void syncDeviceLayout(device::RS_DeviceLayout& device_layout) const;

   private:
    uint64_t _version_id = 0;
    RSize _total_active_size = 0;
    RSize _total_capacity = 0;
    RSize _total_reserved_capacity = 0;
    RS_LayoutGrowthPolicy _growth;
    std::vector<RSize> _segment_sizes;
    std::vector<RSize> _segment_offsets;
    std::vector<RIndex> _segment_indices;
    std::vector<bool> _segment_is_active;
    std::unordered_map<RIndex, RIndex> _segment_id_to_index;

    void _updateSegmentIDToIndex();
    void _refreshReservedCapacity();
};

class RS_ReductionLayout {
   public:
    // Represents a mapping from input segment to output segment
    struct Association {
        RIndex input_segment_id;   // ID in input layout
        RSize input_offset;        // Offset within input segment
        RSize input_size;          // Size of input data to process
        RIndex output_segment_id;  // ID in output layout
        RSize output_offset;       // Offset within output segment

        Association(
            RIndex input_segment_id,
            RSize input_offset,
            RSize input_size,
            RIndex output_segment_id,
            RSize output_offset
        )
            : input_segment_id(input_segment_id),
              input_offset(input_offset),
              input_size(input_size),
              output_segment_id(output_segment_id),
              output_offset(output_offset) {}
    };

    RS_ReductionLayout() = default;

    // Associate with input/output layouts - clears existing associations
    void associateInputLayout(const RS_PooledLayout& input_layout);
    void associateInputLayout(const RS_CompactLayout& input_layout);
    void associateOutputLayout(const RS_PooledLayout& output_layout);
    void associateOutputLayout(const RS_CompactLayout& output_layout);

    void clear();

    // Create an association between input and output segments using segment_ids
    void associate(
        RIndex input_segment_id,
        RSize input_offset,
        RSize input_size,
        RIndex output_segment_id,
        RSize output_offset
    );

    // Remove associations
    void removeAssociationsForInputSegment(RIndex input_segment_id);
    void removeAssociationsForOutputSegment(RIndex output_segment_id);

    [[nodiscard]] uint64_t versionID() const;

    // For querying associations. We will do a lazy evaluation when needed
    const std::vector<Association>& getAssociations();
    std::vector<RIndex> getAssociationsForInputSegment(RIndex input_segment_id);
    std::vector<RIndex> getAssociationsForOutputSegment(RIndex output_segment_id);

    // Get sizes for reduction operations
    std::vector<RSize> getInputSegmentSizes();

    // Sync device layouts for kernel execution - triggers update if needed
    void syncDeviceInputLayout(device::RS_DeviceLayout& device_layout);
    void syncDeviceOutputLayout(device::RS_DeviceLayout& device_layout);

   private:
    uint64_t _local_version_id = 0;

    // References to source layouts
    const RS_PooledLayout* _input_pooled_layout = nullptr;
    const RS_CompactLayout* _input_compact_layout = nullptr;
    const RS_PooledLayout* _output_pooled_layout = nullptr;
    const RS_CompactLayout* _output_compact_layout = nullptr;

    // All associations - store by segment_id, not index
    std::vector<Association> _associations;

    // Maps for faster lookup
    std::unordered_map<RIndex, std::vector<RIndex>> _input_segment_id_to_associations;
    std::unordered_map<RIndex, std::vector<RIndex>> _output_segment_id_to_associations;

    // Lazily update layouts when needed
    uint64_t _cached_version_id = 0;
    std::vector<device::RS_DeviceLayout::SegmentInfo> _input_segments;
    std::vector<RSize> _input_prefix_sums;
    std::vector<device::RS_DeviceLayout::SegmentInfo> _output_segments;
    std::vector<RSize> _output_prefix_sums;
    RSize _total_input_size = 0;
    RSize _total_output_size = 0;

    // Lazily update all the cached layout information
    void _updateLayouts();

    // Helper for finding segment indices by id
    RIndex _findInputSegmentIndex(RIndex segment_id) const;
    bool _isInputSegmentActive(RIndex segment_index) const;
    RIndex _findOutputSegmentIndex(RIndex segment_id) const;
    bool _isOutputSegmentActive(RIndex segment_index) const;

    uint64_t _getInputVersionID() const;
    uint64_t _getOutputVersionID() const;
};

}  // namespace rise

#endif  // RS_LAYOUT_H
