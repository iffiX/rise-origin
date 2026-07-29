// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <fmt/format.h>

#include <cmath>

#include "rise_origin/utils/rs_exception.h"
#include "rise_origin/utils/rs_iterators.h"
#include "rs_layout.h"

using namespace std;

namespace rise {

RS_LayoutGrowthPolicy::RS_LayoutGrowthPolicy(
    RSize double_below_threshold,
    RSize growth_elements,
    RSize minimum_capacity
)
    : double_below_threshold(double_below_threshold),
      growth_elements(growth_elements),
      minimum_capacity(minimum_capacity) {}

RSize RS_LayoutGrowthPolicy::getReservedCapacity(
    RSize total_capacity,
    RSize old_reserved_capacity
) const {
    if (total_capacity == 0) return 0;
    RSize target = (total_capacity <= double_below_threshold) ? (total_capacity * 2)
                                                              : (total_capacity + growth_elements);
    target = RS_MAX(target, minimum_capacity);
    // Re-target only when the old reserve is outside [total_capacity, target], else keep it.
    return (old_reserved_capacity < total_capacity || old_reserved_capacity > target)
               ? target
               : old_reserved_capacity;
}

RS_CompactLayout::RS_CompactLayout(
    RSize double_below_threshold,
    RSize growth_elements,
    RSize minimum_capacity
)
    : _growth(double_below_threshold, growth_elements, minimum_capacity) {}

void RS_CompactLayout::setGrowthPolicy(
    RSize double_below_threshold,
    RSize growth_elements,
    RSize minimum_capacity
) {
    _growth = RS_LayoutGrowthPolicy(double_below_threshold, growth_elements, minimum_capacity);
}

void RS_CompactLayout::clear() {
    _version_id++;
    _total_active_size = 0;
    _segment_sizes.clear();
    _segment_offsets.clear();
    _segment_size_prefix_sums.clear();

    _segment_indices.clear();
    _segment_is_active.clear();
    _segment_id_to_index.clear();
    _refreshReservedCapacity();
}

void RS_CompactLayout::add(RSize size, RIndex segment_id, bool is_active) {
    _version_id++;
    _segment_sizes.push_back(size);
    _segment_indices.push_back(segment_id);
    _segment_is_active.push_back(is_active);
    _segment_id_to_index[segment_id] = _segment_sizes.size() - 1;
    if (is_active) {
        _total_active_size += size;
    }
    _updateSegmentOffsetsAndSizePrefixSums(_segment_sizes.size() - 1);
    _refreshReservedCapacity();
}

void RS_CompactLayout::setSegmentIsActive(RIndex index, bool is_active) {
    if (index >= _segment_is_active.size()) {
        RS_THROW(RS_OutOfBoundsException("Index out of bounds"));
    }
    _version_id++;
    if (is_active && !_segment_is_active[index]) {
        _total_active_size += _segment_sizes[index];
    } else if (!is_active && _segment_is_active[index]) {
        _total_active_size -= _segment_sizes[index];
    }
    _segment_is_active[index] = is_active;
}

void RS_CompactLayout::remove(RIndex index) {
    if (index >= _segment_indices.size()) {
        RS_THROW(RS_OutOfBoundsException("Index out of bounds"));
    } else {
        _version_id++;
        if (_segment_is_active[index]) {
            _total_active_size -= _segment_sizes[index];
        }
        _segment_sizes.erase(_segment_sizes.begin() + index);
        _segment_indices.erase(_segment_indices.begin() + index);
        _segment_is_active.erase(_segment_is_active.begin() + index);
        // rebuild segment index map
        _segment_id_to_index.clear();
        for (RIndex i = 0; i < _segment_indices.size(); i++) {
            _segment_id_to_index[_segment_indices[i]] = i;
        }
        _updateSegmentOffsetsAndSizePrefixSums(index);
        _refreshReservedCapacity();
    }
}

uint64_t RS_CompactLayout::versionID() const { return _version_id; }

RSize RS_CompactLayout::segmentNum() const { return _segment_sizes.size(); }

RSize RS_CompactLayout::segmentSize(RIndex index) const {
    if (index >= _segment_indices.size()) {
        RS_THROW(RS_OutOfBoundsException("Index out of bounds"));
    } else {
        return _segment_sizes[index];
    }
}

RSize RS_CompactLayout::segmentOffset(RIndex index) const {
    if (index >= _segment_size_prefix_sums.size()) {
        RS_THROW(RS_OutOfBoundsException("Index out of bounds"));
    }
    return _segment_offsets[index];
}

bool RS_CompactLayout::segmentIsActive(RIndex index) const {
    if (index >= _segment_is_active.size()) {
        RS_THROW(RS_OutOfBoundsException("Index out of bounds"));
    }
    return _segment_is_active[index];
}

RSize RS_CompactLayout::size() const { return _total_active_size; }

RSize RS_CompactLayout::capacity() const { return _total_reserved_capacity; }

RIndex RS_CompactLayout::find(RIndex segment_id) const {
    if (auto it = _segment_id_to_index.find(segment_id); it != _segment_id_to_index.end()) {
        return it->second;
    }
    return RS_NULL_INDEX;
}

const std::vector<RSize>& RS_CompactLayout::getSegmentSizes() const { return _segment_sizes; }

const std::vector<RSize>& RS_CompactLayout::getSegmentOffsets() const { return _segment_offsets; }

const std::vector<RIndex>& RS_CompactLayout::getSegmentIds() const { return _segment_indices; }

const std::vector<bool>& RS_CompactLayout::getSegmentIsActive() const { return _segment_is_active; }

void RS_CompactLayout::syncDeviceLayout(device::RS_DeviceLayout& device_layout) const {
    if (device_layout.segments.capacity() < segmentNum()) {
        // reserve more
        device_layout.segments.resize(segmentNum() * 2, false);
        device_layout.segment_size_prefix_sums.resize(segmentNum() * 2, false);
    }
    vector<device::RS_DeviceLayout::SegmentInfo> tmp_segments;
    vector<RSize> tmp_segment_size_prefix_sums;
    // Inactive segments still cross, sized zero, so a segment keeps the same index on both sides.
    for (auto&& [size, offset, segment_id, is_active] :
         zip(_segment_sizes, _segment_offsets, _segment_indices, _segment_is_active)) {
        RSize active_size = is_active ? size : 0;
        tmp_segments.emplace_back(active_size, offset, segment_id, is_active);
        tmp_segment_size_prefix_sums.push_back(
            (tmp_segment_size_prefix_sums.empty() ? 0 : tmp_segment_size_prefix_sums.back()) +
            active_size
        );
    }
    device_layout.version_id = _version_id;
    device_layout.total_size = size();
    device_layout.segments.clear();
    device_layout.segments.writeSlice(tmp_segments.data(), tmp_segments.size(), 0);
    device_layout.segment_size_prefix_sums.clear();
    device_layout.segment_size_prefix_sums.writeSlice(
        tmp_segment_size_prefix_sums.data(), tmp_segment_size_prefix_sums.size(), 0
    );
}

void RS_CompactLayout::_updateSegmentOffsetsAndSizePrefixSums(RIndex update_start_index) {
    _segment_offsets.resize(_segment_sizes.size());
    _segment_size_prefix_sums.resize(_segment_sizes.size());
    for (RIndex i = update_start_index; i < _segment_sizes.size(); i++) {
        _segment_offsets[i] = i == 0 ? 0 : _segment_size_prefix_sums[i - 1];
        _segment_size_prefix_sums[i] = _segment_offsets[i] + _segment_sizes[i];
    }
}

void RS_CompactLayout::_refreshReservedCapacity() {
    RSize total_capacity = _segment_size_prefix_sums.empty() ? 0 : _segment_size_prefix_sums.back();
    _total_reserved_capacity =
        _growth.getReservedCapacity(total_capacity, _total_reserved_capacity);
}

RS_PooledLayout::RS_PooledLayout(
    RSize double_below_threshold,
    RSize growth_elements,
    RSize minimum_capacity
)
    : _growth(double_below_threshold, growth_elements, minimum_capacity) {}

void RS_PooledLayout::setGrowthPolicy(
    RSize double_below_threshold,
    RSize growth_elements,
    RSize minimum_capacity
) {
    _growth = RS_LayoutGrowthPolicy(double_below_threshold, growth_elements, minimum_capacity);
}

void RS_PooledLayout::clear() {
    _version_id++;
    _total_active_size = 0;
    _total_capacity = 0;
    _segment_sizes.clear();
    _segment_offsets.clear();
    _segment_indices.clear();
    _segment_is_active.clear();
    _segment_id_to_index.clear();
    _refreshReservedCapacity();
}

void RS_PooledLayout::add(RSize size, RIndex segment_id, bool is_active) {
    _version_id++;
    /// Check if there is space before the first segment
    if (!_segment_offsets.empty() && size <= _segment_offsets[0]) {
        _segment_offsets.insert(_segment_offsets.begin(), 0);
        _segment_sizes.insert(_segment_sizes.begin(), size);
        _segment_indices.insert(_segment_indices.begin(), segment_id);
        _segment_is_active.insert(_segment_is_active.begin(), is_active);
        _total_active_size += is_active ? size : 0;

        // Segment id to index map is invalid, rebuild
        _updateSegmentIDToIndex();
        return;
    }
    /// Search for empty areas between segments
    for (RIndex segment = 0; segment < _segment_sizes.size(); segment++) {
        /// If we find space at the end of the layout
        if (segment == _segment_sizes.size() - 1) {
            if (_segment_offsets[segment] + _segment_sizes[segment] + size < _total_capacity) {
                /// At the end of the layout, there is an empty space of at least size before the
                /// current max allocated size
                _segment_offsets.push_back(_segment_offsets[segment] + _segment_sizes[segment]);
                _segment_sizes.push_back(size);
                _segment_indices.push_back(segment_id);
                _segment_is_active.push_back(is_active);
                _total_active_size += is_active ? size : 0;

                // Only need to insert new segment id to index map, no need to rebuild
                _segment_id_to_index[segment_id] = _segment_sizes.size() - 1;
                return;
            }
        } else if (_segment_offsets[segment + 1] - _segment_offsets[segment] -
                       _segment_sizes[segment] >=
                   size) {
            /// Between current segmentA, at offsetA with sizeA and the next segmentB at offsetB,
            /// there is an empty space of at least "size"
            RSize new_segment_offset = _segment_offsets[segment] + _segment_sizes[segment];
            _segment_offsets.insert(_segment_offsets.begin() + segment + 1, new_segment_offset);
            _segment_sizes.insert(_segment_sizes.begin() + segment + 1, size);
            _segment_indices.insert(_segment_indices.begin() + segment + 1, segment_id);
            _segment_is_active.insert(_segment_is_active.begin() + segment + 1, is_active);
            _total_active_size += is_active ? size : 0;
            // Segment id to index map is invalid, rebuild
            _updateSegmentIDToIndex();
            return;
        }
    }
    /// No empty area found, append to the end
    _segment_offsets.push_back(_total_capacity);
    _segment_sizes.push_back(size);
    _segment_indices.push_back(segment_id);
    _segment_is_active.push_back(is_active);
    _segment_id_to_index[segment_id] = _segment_sizes.size() - 1;
    _total_active_size += is_active ? size : 0;
    _total_capacity += size;
    _refreshReservedCapacity();
}

void RS_PooledLayout::setSegmentIsActive(RIndex index, bool is_active) {
    if (index >= _segment_is_active.size()) {
        RS_THROW(RS_OutOfBoundsException("Index out of bounds"));
    }
    _version_id++;
    if (is_active && !_segment_is_active[index]) {
        _total_active_size += _segment_sizes[index];
    } else if (!is_active && _segment_is_active[index]) {
        _total_active_size -= _segment_sizes[index];
    }
    _segment_is_active[index] = is_active;
}

void RS_PooledLayout::remove(RIndex index) {
    if (index >= _segment_indices.size()) {
        RS_THROW(RS_OutOfBoundsException("Index out of bounds"));
    } else {
        _version_id++;
        if (_segment_is_active[index]) {
            _total_active_size -= _segment_sizes[index];
        }
        _segment_offsets.erase(_segment_offsets.begin() + index);
        _segment_sizes.erase(_segment_sizes.begin() + index);
        _segment_indices.erase(_segment_indices.begin() + index);
        _segment_is_active.erase(_segment_is_active.begin() + index);

        // Segment id to index map is invalid, rebuild
        _updateSegmentIDToIndex();
    }
}

uint64_t RS_PooledLayout::versionID() const { return _version_id; }

RSize RS_PooledLayout::segmentNum() const { return _segment_sizes.size(); }

RSize RS_PooledLayout::segmentSize(RIndex index) const {
    if (index >= _segment_indices.size()) {
        RS_THROW(RS_OutOfBoundsException("Index out of bounds"));
    } else {
        return _segment_sizes[index];
    }
}

RSize RS_PooledLayout::segmentOffset(RIndex index) const {
    if (index >= _segment_indices.size()) {
        RS_THROW(RS_OutOfBoundsException("Index out of bounds"));
    } else {
        return _segment_offsets[index];
    }
}

bool RS_PooledLayout::segmentIsActive(RIndex index) const {
    if (index >= _segment_indices.size()) {
        RS_THROW(RS_OutOfBoundsException("Index out of bounds"));
    }
    return _segment_is_active[index];
}

RSize RS_PooledLayout::size() const { return _total_active_size; }

RSize RS_PooledLayout::capacity() const { return _total_reserved_capacity; }

RIndex RS_PooledLayout::find(RIndex segment_id) const {
    if (auto it = _segment_id_to_index.find(segment_id); it != _segment_id_to_index.end()) {
        return it->second;
    }
    return RS_NULL_INDEX;
}

const std::vector<RSize>& RS_PooledLayout::getSegmentSizes() const { return _segment_sizes; }

const std::vector<RSize>& RS_PooledLayout::getSegmentOffsets() const { return _segment_offsets; }

const std::vector<RIndex>& RS_PooledLayout::getSegmentIds() const { return _segment_indices; }

const std::vector<bool>& RS_PooledLayout::getSegmentIsActive() const { return _segment_is_active; }

void RS_PooledLayout::syncDeviceLayout(device::RS_DeviceLayout& device_layout) const {
    if (device_layout.segments.capacity() < segmentNum()) {
        // reserve more
        device_layout.segments.resize(segmentNum() * 2, false);
        device_layout.segment_size_prefix_sums.resize(segmentNum() * 2, false);
    }
    vector<device::RS_DeviceLayout::SegmentInfo> tmp_segments;
    vector<RSize> tmp_segment_size_prefix_sums;
    // Inactive segments still cross, sized zero, so a segment keeps the same index on both sides.
    for (auto&& [size, offset, segment_id, is_active] :
         zip(_segment_sizes, _segment_offsets, _segment_indices, _segment_is_active)) {
        RSize active_size = is_active ? size : 0;
        tmp_segments.emplace_back(active_size, offset, segment_id, is_active);
        tmp_segment_size_prefix_sums.push_back(
            (tmp_segment_size_prefix_sums.empty() ? 0 : tmp_segment_size_prefix_sums.back()) +
            active_size
        );
    }
    device_layout.version_id = _version_id;
    device_layout.total_size = size();
    device_layout.segments.clear();
    device_layout.segments.writeSlice(tmp_segments.data(), tmp_segments.size(), 0);
    device_layout.segment_size_prefix_sums.clear();
    device_layout.segment_size_prefix_sums.writeSlice(
        tmp_segment_size_prefix_sums.data(), tmp_segment_size_prefix_sums.size(), 0
    );
}

void RS_PooledLayout::_updateSegmentIDToIndex() {
    _segment_id_to_index.clear();
    for (RIndex i = 0; i < _segment_sizes.size(); i++) {
        _segment_id_to_index[_segment_indices[i]] = i;
    }
}

void RS_PooledLayout::_refreshReservedCapacity() {
    _total_reserved_capacity =
        _growth.getReservedCapacity(_total_capacity, _total_reserved_capacity);
}

void RS_ReductionLayout::associateInputLayout(const RS_PooledLayout& input_layout) {
    _input_pooled_layout = &input_layout;
    _input_compact_layout = nullptr;
    // Clear associations when changing layouts
    clear();
}

void RS_ReductionLayout::associateInputLayout(const RS_CompactLayout& input_layout) {
    _input_compact_layout = &input_layout;
    _input_pooled_layout = nullptr;
    // Clear associations when changing layouts
    clear();
}

void RS_ReductionLayout::associateOutputLayout(const RS_PooledLayout& output_layout) {
    _output_pooled_layout = &output_layout;
    _output_compact_layout = nullptr;
    // Clear associations when changing layouts
    clear();
}

void RS_ReductionLayout::associateOutputLayout(const RS_CompactLayout& output_layout) {
    _output_compact_layout = &output_layout;
    _output_pooled_layout = nullptr;
    // Clear associations when changing layouts
    clear();
}

void RS_ReductionLayout::clear() {
    _local_version_id++;
    _associations.clear();
    _input_segment_id_to_associations.clear();
    _output_segment_id_to_associations.clear();
    _total_input_size = 0;
    _total_output_size = 0;
}

void RS_ReductionLayout::associate(
    RIndex input_segment_id,
    RSize input_offset,
    RSize input_size,
    RIndex output_segment_id,
    RSize output_offset
) {
    // Check that we have associated layouts
    if (!_input_pooled_layout && !_input_compact_layout) {
        RS_THROW(RS_HostExecutionException("No input layout associated"));
    }
    if (!_output_pooled_layout && !_output_compact_layout) {
        RS_THROW(RS_HostExecutionException("No output layout associated"));
    }

    // Validate input segment exists using segment_id (not index)
    RIndex input_segment_index = _findInputSegmentIndex(input_segment_id);
    if (input_segment_index == RS_NULL_INDEX) {
        RS_THROW(RS_OutOfBoundsException("Input segment ID not found in layout"));
    }

    // Validate output segment exists using segment_id (not index)
    RIndex output_segment_index = _findOutputSegmentIndex(output_segment_id);
    if (output_segment_index == RS_NULL_INDEX) {
        RS_THROW(RS_OutOfBoundsException("Output segment ID not found in layout"));
    }

    // Validate input offset and size
    RSize input_segment_size;
    if (_input_pooled_layout) {
        input_segment_size = _input_pooled_layout->segmentSize(input_segment_index);
    } else {
        input_segment_size = _input_compact_layout->segmentSize(input_segment_index);
    }

    if (input_offset + input_size > input_segment_size) {
        RS_THROW(RS_OutOfBoundsException("Input range exceeds segment size"));
    }

    // Validate output offset
    RSize output_segment_size;
    if (_output_pooled_layout) {
        output_segment_size = _output_pooled_layout->segmentSize(output_segment_index);
    } else {
        output_segment_size = _output_compact_layout->segmentSize(output_segment_index);
    }

    if (output_offset >= output_segment_size) {
        RS_THROW(RS_OutOfBoundsException("Output range exceeds segment size"));
    }

    _local_version_id++;

    // Create and store the association with segment_ids
    RIndex association_index = _associations.size();
    _associations.emplace_back(
        input_segment_id, input_offset, input_size, output_segment_id, output_offset
    );

    // Update lookup maps with segment_ids
    _input_segment_id_to_associations[input_segment_id].push_back(association_index);
    _output_segment_id_to_associations[output_segment_id].push_back(association_index);

    // Don't rebuild associations now - do it lazily when needed
}

void RS_ReductionLayout::removeAssociationsForInputSegment(RIndex input_segment_id) {
    auto it = _input_segment_id_to_associations.find(input_segment_id);
    if (it == _input_segment_id_to_associations.end()) {
        return;
    }

    _local_version_id++;

    // Get all associations for this input segment
    auto& assoc_indices = it->second;

    // Mark all associated elements for removal
    for (RIndex assoc_idx : assoc_indices) {
        auto& assoc = _associations[assoc_idx];

        // Remove this association from output segment map
        auto& output_assocs = _output_segment_id_to_associations[assoc.output_segment_id];
        output_assocs.erase(
            std::remove(output_assocs.begin(), output_assocs.end(), assoc_idx), output_assocs.end()
        );

        // Mark as invalid by setting to NULL_INDEX
        assoc.input_segment_id = RS_NULL_INDEX;
    }

    // Remove from input map
    _input_segment_id_to_associations.erase(it);

    // Don't rebuild associations now - do it lazily when needed
}

void RS_ReductionLayout::removeAssociationsForOutputSegment(RIndex output_segment_id) {
    auto it = _output_segment_id_to_associations.find(output_segment_id);
    if (it == _output_segment_id_to_associations.end()) {
        return;  // No associations for this segment
    }

    _local_version_id++;

    // Get all associations for this output segment
    auto& assoc_indices = it->second;

    // Mark all associated elements for removal
    for (RIndex assoc_idx : assoc_indices) {
        auto& assoc = _associations[assoc_idx];

        // Remove this association from input segment map
        auto& input_assocs = _input_segment_id_to_associations[assoc.input_segment_id];
        input_assocs.erase(
            std::remove(input_assocs.begin(), input_assocs.end(), assoc_idx), input_assocs.end()
        );

        // Mark as invalid by setting to NULL_INDEX
        assoc.input_segment_id = RS_NULL_INDEX;
    }

    // Remove from output map
    _output_segment_id_to_associations.erase(it);

    // Don't rebuild associations now - do it lazily when needed
}

uint64_t RS_ReductionLayout::versionID() const {
    return _local_version_id + _getInputVersionID() + _getOutputVersionID();
}

const std::vector<RS_ReductionLayout::Association>& RS_ReductionLayout::getAssociations() {
    if (_cached_version_id != versionID()) {
        _updateLayouts();
    }
    return _associations;
}

std::vector<RIndex> RS_ReductionLayout::getAssociationsForInputSegment(RIndex input_segment_id) {
    if (_cached_version_id != versionID()) {
        _updateLayouts();
    }
    auto it = _input_segment_id_to_associations.find(input_segment_id);
    if (it != _input_segment_id_to_associations.end()) {
        return it->second;
    }
    return {};
}

std::vector<RIndex> RS_ReductionLayout::getAssociationsForOutputSegment(RIndex output_segment_id) {
    if (_cached_version_id != versionID()) {
        _updateLayouts();
    }
    auto it = _output_segment_id_to_associations.find(output_segment_id);
    if (it != _output_segment_id_to_associations.end()) {
        return it->second;
    }
    return {};
}

std::vector<RSize> RS_ReductionLayout::getInputSegmentSizes() {
    if (_cached_version_id != versionID()) {
        _updateLayouts();
    }

    std::vector<RSize> sizes;
    for (const auto& segment : _input_segments) {
        sizes.push_back(segment.size);
    }
    return sizes;
}

void RS_ReductionLayout::syncDeviceInputLayout(device::RS_DeviceLayout& device_layout) {
    if (_cached_version_id != versionID()) {
        _updateLayouts();
    }

    // Update the device layout
    if (device_layout.segments.capacity() < _input_segments.size()) {
        // reserve more
        device_layout.segments.resize(_input_segments.size() * 2, false);
        device_layout.segment_size_prefix_sums.resize(_input_segments.size() * 2, false);
    }

    device_layout.total_size = _total_input_size;
    device_layout.segments.clear();
    device_layout.segments.writeSlice(_input_segments.data(), _input_segments.size(), 0);
    device_layout.segment_size_prefix_sums.clear();
    device_layout.segment_size_prefix_sums.writeSlice(
        _input_prefix_sums.data(), _input_prefix_sums.size(), 0
    );
}

void RS_ReductionLayout::syncDeviceOutputLayout(device::RS_DeviceLayout& device_layout) {
    if (_cached_version_id != versionID()) {
        _updateLayouts();
    }

    // Update the device layout
    if (device_layout.segments.capacity() < _output_segments.size()) {
        // reserve more
        device_layout.segments.resize(_output_segments.size() * 2, false);
        device_layout.segment_size_prefix_sums.resize(_output_segments.size() * 2, false);
    }

    device_layout.total_size = _total_output_size;
    device_layout.segments.clear();
    device_layout.segments.writeSlice(_output_segments.data(), _output_segments.size(), 0);
    device_layout.segment_size_prefix_sums.clear();
    device_layout.segment_size_prefix_sums.writeSlice(
        _output_prefix_sums.data(), _output_prefix_sums.size(), 0
    );
}

void RS_ReductionLayout::_updateLayouts() {
    // Reset all cached data
    _input_segments.clear();
    _input_prefix_sums.clear();
    _output_segments.clear();
    _output_prefix_sums.clear();
    _total_input_size = 0;
    _total_output_size = 0;

    // First, filter out invalid associations
    std::vector<Association> valid_associations;
    for (const auto& assoc : _associations) {
        // Check if both input and output segments still exist
        RIndex input_segment_index = _findInputSegmentIndex(assoc.input_segment_id);
        RIndex output_segment_index = _findOutputSegmentIndex(assoc.output_segment_id);

        if (input_segment_index == RS_NULL_INDEX || output_segment_index == RS_NULL_INDEX) {
            continue;  // Skip associations with missing segments
        }

        valid_associations.push_back(assoc);
    }

    // Replace associations with only valid ones
    _associations.swap(valid_associations);

    // Rebuild the lookup maps
    _input_segment_id_to_associations.clear();
    _output_segment_id_to_associations.clear();
    for (RIndex assoc_idx = 0; assoc_idx < _associations.size(); assoc_idx++) {
        const auto& assoc = _associations[assoc_idx];
        _input_segment_id_to_associations[assoc.input_segment_id].push_back(assoc_idx);
        _output_segment_id_to_associations[assoc.output_segment_id].push_back(assoc_idx);
    }

    // Build new segments based on current layouts
    for (const auto& assoc : _associations) {
        // Find current indices for segment_ids
        RIndex input_segment_index = _findInputSegmentIndex(assoc.input_segment_id);
        RIndex output_segment_index = _findOutputSegmentIndex(assoc.output_segment_id);

        if (!_isInputSegmentActive(input_segment_index) ||
            !_isOutputSegmentActive(output_segment_index)) {
            // Skip associations with inactive segments
            continue;
        }

        // Get the global offset for this input segment
        RSize input_global_offset;
        if (_input_pooled_layout) {
            input_global_offset = _input_pooled_layout->segmentOffset(input_segment_index);
        } else {
            input_global_offset = _input_compact_layout->segmentOffset(input_segment_index);
        }

        // Get the global offset for this output segment
        RSize output_global_offset;
        if (_output_pooled_layout) {
            output_global_offset = _output_pooled_layout->segmentOffset(output_segment_index);
        } else {
            output_global_offset = _output_compact_layout->segmentOffset(output_segment_index);
        }

        // Create segment info for this input
        _input_segments.emplace_back(
            assoc.input_size,                          // Size to process
            input_global_offset + assoc.input_offset,  // Global offset in the input layout
            assoc.input_segment_id                     // Original segment ID
        );

        // Create segment info for this output
        _output_segments.emplace_back(
            1,                                           // Size is always 1 for reduction output
            output_global_offset + assoc.output_offset,  // Global offset in the output layout
            assoc.output_segment_id                      // Original segment ID
        );

        // Update sizes
        _total_input_size += assoc.input_size;
        _total_output_size += 1;
    }

    // Calculate input prefix sums
    RSize current_input_offset = 0;
    for (const auto& segment : _input_segments) {
        current_input_offset += segment.size;
        _input_prefix_sums.push_back(current_input_offset);
    }

    // Calculate output prefix sums
    RSize current_output_offset = 0;
    for (const auto& segment : _output_segments) {
        current_output_offset += segment.size;
        _output_prefix_sums.push_back(current_output_offset);
    }

    _cached_version_id = versionID();
}

RIndex RS_ReductionLayout::_findInputSegmentIndex(RIndex segment_id) const {
    if (_input_pooled_layout) return _input_pooled_layout->find(segment_id);
    if (_input_compact_layout) return _input_compact_layout->find(segment_id);
    return RS_NULL_INDEX;
}

bool RS_ReductionLayout::_isInputSegmentActive(RIndex segment_index) const {
    if (_input_pooled_layout) return _input_pooled_layout->segmentIsActive(segment_index);
    if (_input_compact_layout) return _input_compact_layout->segmentIsActive(segment_index);
    return false;
}

RIndex RS_ReductionLayout::_findOutputSegmentIndex(RIndex segment_id) const {
    if (_output_pooled_layout) return _output_pooled_layout->find(segment_id);
    if (_output_compact_layout) return _output_compact_layout->find(segment_id);
    return RS_NULL_INDEX;
}

bool RS_ReductionLayout::_isOutputSegmentActive(RIndex segment_index) const {
    if (_output_pooled_layout) return _output_pooled_layout->segmentIsActive(segment_index);
    if (_output_compact_layout) return _output_compact_layout->segmentIsActive(segment_index);
    return false;
}

uint64_t RS_ReductionLayout::_getInputVersionID() const {
    if (_input_pooled_layout) return _input_pooled_layout->versionID();
    if (_input_compact_layout) return _input_compact_layout->versionID();
    return 0;
}

uint64_t RS_ReductionLayout::_getOutputVersionID() const {
    if (_output_pooled_layout) return _output_pooled_layout->versionID();
    if (_output_compact_layout) return _output_compact_layout->versionID();
    return 0;
}

}  // namespace rise