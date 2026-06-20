// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_LAUNCH_C_H
#define RS_LAUNCH_C_H

#include "rise_origin/core/rs_layout.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_value.h"
#include "rise_origin/device/data/rs_vector.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/device/rs_constant_memory.h"
#include "rise_origin/logic/record/rs_basic_record_c.h"

namespace rise {

namespace logic {

/**
 * @brief Launch a compute function directly, passing only the constant-memory kernel.
 * @details The compute function computes its own thread index and reads the kernel via
 * RS_ConstantMemory::as<Kernel>(); no per-launch kernel copy is made.
 *
 * @param args forwarded to ComputeFunction::run after the kernel pointer.
 */
template <typename ComputeFunction, typename Kernel>
class RSCF_LaunchDirect : public device::RS_ComputeFunction<ComputeFunction> {
   public:
    using device::RS_ComputeFunction<ComputeFunction>::index1D;

    template <typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void run(Args... args) {
        ComputeFunction::run(device::RS_ConstantMemory::as<Kernel>(), args...);
    }
};

/**
 * @brief Launch a compute function once per resolved handle.
 *
 * @param handles thread-indexed handle array.
 * @param args    forwarded to ComputeFunction::run after the kernel and this thread's handle.
 */
template <typename ComputeFunction, typename Kernel>
class RSCF_LaunchWithHandles : public device::RS_ComputeFunction<ComputeFunction> {
   public:
    using device::RS_ComputeFunction<ComputeFunction>::index1D;

    template <typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(device::RS_Vector<RHandle> handles, Args... args) {
        ComputeFunction::run(device::RS_ConstantMemory::as<Kernel>(), handles[index1D()], args...);
    }
};

/**
 * @brief Launch a compute function once per layout element, resolving its global handle.
 *
 * @param layout device layout mapping thread index to a global handle.
 * @param args   forwarded to ComputeFunction::run after the kernel and the resolved handle.
 */
template <typename ComputeFunction, typename Kernel>
class RSCF_LaunchWithLayout
    : public device::RS_ComputeFunction<RSCF_LaunchWithLayout<ComputeFunction, Kernel>> {
   public:
    using device::RS_ComputeFunction<RSCF_LaunchWithLayout>::index1D;

    template <typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void run(device::RS_DeviceLayout layout, Args... args) {
        RIndex i = index1D();
        auto &ctx = device::RS_ConstantMemory::as<Kernel>()->ctx;
        auto search_result = layout.searchForBelongedSegment(i);
        ComputeFunction::run(
            device::RS_ConstantMemory::as<Kernel>(), search_result.global_index, args...
        );
    }
};

/**
 * @brief Launch a compute function once per layout element, passing its segment and in-segment
 * indices.
 *
 * @param layout device layout mapping thread index to a segment and in-segment index.
 * @param args   forwarded to ComputeFunction::run after the kernel and the two indices.
 */
template <typename ComputeFunction, typename Kernel>
class RSCF_LaunchWithLayoutPerSegment
    : public device::RS_ComputeFunction<RSCF_LaunchWithLayoutPerSegment<ComputeFunction, Kernel>> {
   public:
    using device::RS_ComputeFunction<RSCF_LaunchWithLayoutPerSegment>::index1D;

    template <typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void run(device::RS_DeviceLayout layout, Args... args) {
        RIndex i = index1D();
        auto &ctx = device::RS_ConstantMemory::as<Kernel>()->ctx;
        auto search_result = layout.searchForBelongedSegment(i);
        ComputeFunction::run(
            device::RS_ConstantMemory::as<Kernel>(),
            search_result.segment_index,
            search_result.in_segment_index,
            args...
        );
    }
};

/**
 * @brief Launch a set-or-get compute function once per resolved handle, with a scratch buffer.
 *
 * @param handles thread-indexed handle array.
 * @param buffer  per-thread scratch buffer passed to the compute function.
 * @param args    forwarded to SetOrGetFunction::run after the kernel, buffer, handle and index.
 */
template <typename SetOrGetFunction, typename Kernel>
class RSCF_LaunchSetOrGetWithHandles
    : public device::RS_ComputeFunction<RSCF_LaunchSetOrGetWithHandles<SetOrGetFunction, Kernel>> {
   public:
    using device::RS_ComputeFunction<RSCF_LaunchSetOrGetWithHandles>::index1D;

    template <typename Buffer, typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(device::RS_Vector<RHandle> handles, Buffer buffer, Args... args) {
        RIndex i = index1D();
        if (i >= handles.size()) return;

        SetOrGetFunction::run(
            device::RS_ConstantMemory::as<Kernel>(), handles[i], i, buffer, args...
        );
    }
};

/**
 * @brief Launch a set-or-get compute function once per layout element, with a scratch buffer.
 *
 * @param layout device layout mapping thread index to a global handle.
 * @param buffer per-thread scratch buffer passed to the compute function.
 * @param args   forwarded to SetOrGetFunction::run after the kernel, buffer, handle and index.
 */
template <typename SetOrGetFunction, typename Kernel>
class RSCF_LaunchSetOrGetWithLayout
    : public device::RS_ComputeFunction<RSCF_LaunchSetOrGetWithLayout<SetOrGetFunction, Kernel>> {
   public:
    using device::RS_ComputeFunction<RSCF_LaunchSetOrGetWithLayout>::index1D;

    template <typename Buffer, typename... Args>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(device::RS_DeviceLayout layout, Buffer buffer, Args... args) {
        RIndex i = index1D();
        if (i >= layout.size()) return;

        auto &ctx = device::RS_ConstantMemory::as<Kernel>()->ctx;
        auto search_result = layout.searchForBelongedSegment(i);

        if (search_result.segment_id != RS_NULL_INDEX) {
            SetOrGetFunction::run(
                device::RS_ConstantMemory::as<Kernel>(),
                search_result.global_index,  // Target handle in the kernel
                i,                           // Set/Get index in the buffer
                buffer,                      // Buffer
                args...
            );
        }
    }
};

/**
 * @brief Launch a store-record compute function once per record handle.
 *
 * @param root_records   per-segment root records.
 * @param record_handles per-segment record handle arrays.
 * @param handle_layout  device layout mapping thread index to a segment and in-segment index.
 */
template <typename StoreRecordFunction, typename Kernel>
class RSCF_LaunchStoreRecordWithRecordHandles
    : public device::RS_ComputeFunction<
          RSCF_LaunchStoreRecordWithRecordHandles<StoreRecordFunction, Kernel>> {
   public:
    using device::RS_ComputeFunction<RSCF_LaunchStoreRecordWithRecordHandles>::index1D;

    template <typename RootRecord>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        device::RS_Vector<RootRecord> root_records,
        device::RS_Vector<device::RS_Array<RIndex, 1>> record_handles,
        device::RS_DeviceLayout handle_layout
    ) {
        RIndex i = index1D();
        auto &ctx = device::RS_ConstantMemory::as<Kernel>()->ctx;
        auto search_result = handle_layout.searchForBelongedSegment(i);
        StoreRecordFunction::run(
            device::RS_ConstantMemory::as<Kernel>(),
            root_records[search_result.segment_index],
            record_handles[search_result.segment_index][search_result.in_segment_index],
            search_result.in_segment_index
        );
    }
};

/**
 * @brief Launch a store-record compute function once per layout element.
 *
 * @param root_records per-segment root records.
 * @param layout       device layout mapping thread index to a segment and in-segment index.
 */
template <typename StoreRecordFunction, typename Kernel>
class RSCF_LaunchStoreRecordWithLayout
    : public device::RS_ComputeFunction<RSCF_LaunchStoreRecordWithLayout<StoreRecordFunction, Kernel>> {
   public:
    using device::RS_ComputeFunction<RSCF_LaunchStoreRecordWithLayout>::index1D;

    template <typename RootRecord>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(device::RS_Vector<RootRecord> root_records, device::RS_DeviceLayout layout) {
        RIndex i = index1D();
        auto &ctx = device::RS_ConstantMemory::as<Kernel>()->ctx;
        auto search_result = layout.searchForBelongedSegment(i);
        StoreRecordFunction::run(
            device::RS_ConstantMemory::as<Kernel>(),
            root_records[search_result.segment_index],
            search_result.segment_index,
            search_result.in_segment_index
        );
    }
};

/**
 * @brief Launch the root-record update once per buffer, supplying each handle layout's segment size.
 *
 * @param buffers        per-segment record buffers.
 * @param root_records   per-segment root records.
 * @param handle_layouts the per-element layouts whose segment sizes drive the update.
 */
template <typename Kernel>
class RSCF_LaunchUpdateRootRecord
    : public device::RS_ComputeFunction<RSCF_LaunchUpdateRootRecord<Kernel>> {
   public:
    using device::RS_ComputeFunction<RSCF_LaunchUpdateRootRecord>::index1D;

    template <typename Buffer, typename RootRecord, typename... Layout>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        device::RS_Vector<Buffer> buffers,
        device::RS_Vector<RootRecord> root_records,
        Layout... handle_layouts
    ) {
        RIndex i = index1D();
        auto &ctx = device::RS_ConstantMemory::as<Kernel>()->ctx;
        RSCF_UpdateRootRecord::run(
            device::RS_ConstantMemory::as<Kernel>(),
            buffers[i],
            root_records[i],
            handle_layouts.segmentSize(i)...
        );
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_LAUNCH_C_H
