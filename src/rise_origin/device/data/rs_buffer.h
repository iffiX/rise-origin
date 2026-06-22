// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_BUFFER_H
#define RS_BUFFER_H

#include <cassert>

#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_data_util.h"
#include "rs_value.h"

namespace rise {

namespace device {

/// @brief RS_BufferIndex is a stack-like index for push/pop of "chunks" inside a contiguous buffer.
class RS_BufferIndex {
   public:
    /*****************************************************************
     * Data
     *****************************************************************/

    /// @brief Memory pointer holding a single RSize for the current number of chunks (stack
    /// size).
    RS_MemoryPtr chunk_num;

    /// @brief Memory pointer for an array (max_chunk_num) of RBigSize chunk offsets.
    RS_MemoryPtr chunk_data_offset;

    /// @brief Memory pointer for an array (max_chunk_num) of RBigSize chunk sizes.
    RS_MemoryPtr chunk_data_size;

    /// @brief The contiguous data region to which these offsets refer.
    RS_MemoryPtr root_data;

    /// @brief Maximum number of chunks that can be pushed.
    RSize max_chunk_num = 0;

    /// @brief Maximum total data size, in bytes.
    RBigSize max_data_size = 0;

    /// @brief True if this index is allocated in device memory (otherwise host memory).
    bool is_on_device = false;

    /*****************************************************************
     * Methods
     *****************************************************************/

    /// @brief Default constructor.
    RS_BufferIndex() = default;

    /**
     * @brief Construct a new RS_BufferIndex and initialize it.
     *
     * @param provider      Memory provider.
     * @param max_chunk_num Maximum number of chunks.
     * @param max_data_size Maximum total data bytes.
     * @param is_on_device  Whether the index is allocated on device.
     */
    RS_BufferIndex(
        RS_MemoryProvider* provider,
        RSize max_chunk_num,
        RBigSize max_data_size,
        bool is_on_device
    ) {
        init(provider, max_chunk_num, max_data_size, is_on_device);
    }

    /**
     * @brief Allocate chunk_num, chunk_data_offset, chunk_data_size in the specified memory
     * space.
     *
     * @param provider      Memory provider.
     * @param max_chunk_num Maximum number of chunks.
     * @param max_data_size Maximum total data bytes.
     * @param is_on_device  Whether the index is allocated on device.
     */
    void init(
        RS_MemoryProvider* provider,
        RSize max_chunk_num,
        RBigSize max_data_size,
        bool is_on_device
    ) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        this->max_chunk_num = max_chunk_num;
        this->max_data_size = max_data_size;
        this->is_on_device = is_on_device;

        // 1) chunk_num: a single RSize
        if (is_on_device) {
            chunk_num = provider->allocate(sizeof(RSize), false);
        } else {
            chunk_num = provider->allocateHost(sizeof(RSize), false);
        }
        // Set chunk_num = 0
        chunk_num.set(0, chunk_num.size(), 0, true);

        // 2) chunk_data_offset & chunk_data_size: each has max_chunk_num elements
        RBigSize array_bytes = RBIG_SIZE(max_chunk_num) * sizeof(RBigSize);
        if (is_on_device) {
            chunk_data_offset = provider->allocate(array_bytes, false);
            chunk_data_size = provider->allocate(array_bytes, false);
        } else {
            chunk_data_offset = provider->allocateHost(array_bytes, false);
            chunk_data_size = provider->allocateHost(array_bytes, false);
        }
        // Initialize them to 0
        if (array_bytes > 0) {
            chunk_data_offset.set(0, array_bytes, 0, true);
            chunk_data_size.set(0, array_bytes, 0, true);
        }
    }

    /**
     * @brief Free this index's memory.
     */
    void free() {
        chunk_num.free();
        chunk_data_offset.free();
        chunk_data_size.free();

        // Root data is set from outside, just clear it
        root_data = RS_MemoryPtr();

        max_chunk_num = 0;
        max_data_size = 0;
        is_on_device = false;
    }

    /**
     * @brief Copy chunk data from another buffer index.
     *
     * @param other  The buffer index to copy from.
     */
    void copyChunkData(const RS_BufferIndex& other, bool sync = true) {
        RS_DATA_EXPECT_INIT(chunk_num.provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(other.chunk_num.provider() != nullptr);
        RS_DATA_EXPECT_ARGUMENT(
            max_chunk_num == other.max_chunk_num,
            "this->max_chunk_num={}, "
            "other.max_chunk_num={}",
            max_chunk_num,
            other.max_chunk_num
        );
        chunk_num.copyFrom(other.chunk_num, sizeof(RSize), sync);
        chunk_data_offset.copyFrom(other.chunk_data_offset, other.chunk_data_offset.size(), sync);
        chunk_data_size.copyFrom(other.chunk_data_size, other.chunk_data_size.size(), sync);
    }

    /**
     * @brief Set the root data pointer.
     *
     * @param root_data The root data pointer.
     */
    void setRootData(RS_MemoryPtr root_data) {
        RS_DATA_EXPECT_INIT(chunk_num.provider() != nullptr);
        this->root_data = root_data;
    }

    /**
     * @brief Return how many chunks are currently pushed.
     */
    HOST_DEVICE_INLINE_QUALIFIER RSize chunkNum() const { return chunk_num.get<RSize>(0); }

    /**
     * @brief Returns the total bytes used by all chunks (sum of their sizes).
     */
    HOST_DEVICE_INLINE_QUALIFIER RBigSize usedDataSize() const {
        RSize cn = chunkNum();
        if (cn == 0) return 0;
        // last chunk => index (cn - 1)
        RBigSize last_offset = chunk_data_offset.get<RBigSize>(cn - 1);
        RBigSize last_size = chunk_data_size.get<RBigSize>(cn - 1);
        return last_offset + last_size;
    }

    /**
     * @brief Whether a new chunk of size 'chunk_size' can be pushed without overflow.
     */
    HOST_DEVICE_INLINE_QUALIFIER bool canPushChunk(RBigSize chunk_size) const {
        RSize cn = chunkNum();
        if (cn >= max_chunk_num) {
            return false;
        }
        return (usedDataSize() + chunk_size <= max_data_size);
    }

    /**
     * @brief Push a new chunk, incrementing the stack count, and
     *        storing offset & size for the top chunk.
     *
     * @param chunk_size Number of bytes for this new chunk
     */
    HOST_DEVICE_INLINE_QUALIFIER void pushChunk(RBigSize chunk_size) {
        RSize cn = chunkNum();
        assert(canPushChunk(chunk_size) && "Cannot push chunk, no capacity");

        // offset for new chunk
        RBigSize offset =
            (cn == 0)
                ? 0
                : (chunk_data_offset.get<RBigSize>(cn - 1) + chunk_data_size.get<RBigSize>(cn - 1));

        chunk_data_offset.get<RBigSize>(cn) = offset;
        chunk_data_size.get<RBigSize>(cn) = chunk_size;

        chunk_num.get<RSize>(0) = cn + 1;
    }

    /**
     * @brief Pop the top chunk off the stack (decrement chunk_num by 1).
     *        Does nothing if empty.
     */
    HOST_DEVICE_INLINE_QUALIFIER void popChunk() {
        RSize cn = chunkNum();
        if (cn == 0) return;
        chunk_num.get<RSize>(0) = cn - 1;
    }

    /**
     * @brief Get a chunk from the index.
     *
     * @param chunk_index The index of the chunk to get.
     * @return A pointer pointing to the start of the chunk with size of the chunk.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_MemoryPtr getChunk(RIndex chunk_index) const {
        assert(chunk_index < chunkNum() && "Chunk index out of bounds");
        RBigSize offset = chunk_data_offset.get<RBigSize>(chunk_index);
        RBigSize size = chunk_data_size.get<RBigSize>(chunk_index);
        auto ptr = root_data + offset;
        ptr.bytes = size;
        return ptr;
    }

    /**
     * @brief Clear the index from host side.
     */
    void clearHost(bool sync = true) {
        RS_DATA_EXPECT_INIT(chunk_num.provider() != nullptr);
        chunk_num.set(0, sizeof(RSize), 0, sync);
    }

    /**
     * @brief Clear the index from device side.
     */
    HOST_DEVICE_INLINE_QUALIFIER void clearDevice() {
        for (RSize i = 0; i < max_chunk_num; i++) {
            chunk_data_offset.get<RBigSize>(i) = 0;
            chunk_data_size.get<RBigSize>(i) = 0;
        }
        chunk_num.get<RSize>(0) = 0;
    }
};

/**
 * @brief RS_DoubleBuffer is a double-buffered container with two host
 *        regions (A/B) and two device regions (A/B). 'is_a_active'
 *        indicates which pair (A or B) is active.
 *
 *        The user can push or pop chunks on host or device.
 *        push returns an RS_MemoryPtr to the newly allocated region
 *        (zero-copy). The user writes data directly to that region.
 *
 *        The inactive side will be copied after they are swapped with
 *        the active side, which is done by calling swapAndCopy(). After
 *        swapping, the active side can be accessed individually by host
 *        and device, with inactive sides being copied in background.
 */
class RS_DoubleBuffer {
   public:
    /*****************************************************************
     * Data
     *****************************************************************/

    /// @brief Memory provider.
    RS_MemoryProvider* mem_provider = nullptr;

    /// @brief Host buffer A
    RS_MemoryPtr host_a;

    /// @brief Device buffer A
    RS_MemoryPtr device_a;

    /// @brief Host buffer B
    RS_MemoryPtr host_b;

    /// @brief Device buffer B
    RS_MemoryPtr device_b;

    /// @brief Host index for buffer A
    RS_BufferIndex host_index_a;

    /// @brief Device index for buffer A
    RS_BufferIndex device_index_a;

    /// @brief Host index for buffer B
    RS_BufferIndex host_index_b;

    /// @brief Device index for buffer B
    RS_BufferIndex device_index_b;

    /// @brief Device side flag storing which side is active (true=>A, false=>B).
    RS_Value<bool> is_a_active_device;

    /// @brief Host side flag storing which side is active (true=>A, false=>B).
    bool is_a_active_host = true;

    /*****************************************************************
     * Methods
     *****************************************************************/

    /// @brief Default constructor.
    RS_DoubleBuffer() = default;

    /**
     * @brief Construct and initialize the double-buffer with the given capacities.
     *
     * @param max_chunk_num  Maximum chunks per buffer.
     * @param max_data_size  Maximum bytes per buffer.
     * @param provider       Memory provider.
     */
    RS_DoubleBuffer(RSize max_chunk_num, RBigSize max_data_size, RS_MemoryProvider* provider) {
        init(max_chunk_num, max_data_size, provider);
    }

    /**
     * @brief Allocate host/device buffers A/B and initialize indexes and is_a_active.
     *
     * @param max_chunk_num  Maximum chunks per buffer
     * @param max_data_size  Maximum bytes per buffer
     * @param provider       Memory provider
     */
    void init(RSize max_chunk_num, RBigSize max_data_size, RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        this->mem_provider = provider;

        // 1) Allocate data buffers
        host_a = provider->allocateHost(max_data_size, true);
        host_b = provider->allocateHost(max_data_size, true);
        device_a = provider->allocate(max_data_size, true);
        device_b = provider->allocate(max_data_size, true);

        // 2) Initialize indexes
        host_index_a.init(provider, max_chunk_num, max_data_size, /*is_on_device=*/false);
        device_index_a.init(provider, max_chunk_num, max_data_size, /*is_on_device=*/true);
        host_index_b.init(provider, max_chunk_num, max_data_size, /*is_on_device=*/false);
        device_index_b.init(provider, max_chunk_num, max_data_size, /*is_on_device=*/true);

        // Link their data pointers
        host_index_a.setRootData(host_a);
        device_index_a.setRootData(device_a);
        host_index_b.setRootData(host_b);
        device_index_b.setRootData(device_b);

        // 3) Initialize is_a_active => true
        is_a_active_device.init(provider);
        is_a_active_device.fill(true);
        is_a_active_host = true;
    }

    /**
     * @brief Free all buffer memory and indexes.
     */
    void free() {
        mem_provider = nullptr;

        host_a.free();
        host_b.free();
        device_a.free();
        device_b.free();

        host_index_a.free();
        device_index_a.free();
        host_index_b.free();
        device_index_b.free();

        is_a_active_device.free();
        is_a_active_host = true;
    }

    /**
     * @brief Swap which buffer is active (A <-> B), while copying the inactive side
     *        (host->device or device->host). Also clears the new active side.
     * @note The buffer index copy is always synchronous and ordered. But the buffer data copy
     *       can be synchronous/asynchronous and ordered/unordered.
     *
     * @param device_to_host  If false, copies from inactive host buffer to inactive device buffer.
     *                        If true, copies from inactive device buffer to inactive host buffer.
     * @param sync            Whether the buffer data copy is synchronous.
     * @param ordered         Whether the buffer data copy is ordered.
     */
    void swapAndCopy(bool device_to_host, bool sync = true, bool ordered = true) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        bool new_is_a_active = !is_a_active_host;
        is_a_active_device.write(&new_is_a_active, false);
        is_a_active_host = new_is_a_active;

        RS_BufferIndex& new_h_index_active = (new_is_a_active ? host_index_a : host_index_b);
        RS_BufferIndex& new_d_index_active = (new_is_a_active ? device_index_a : device_index_b);

        // Clear the new active side that will be written to
        if (device_to_host) {
            // Device is the producer, host is the consumer
            new_d_index_active.clearHost(false);
        } else {
            // Host is the producer, device is the consumer
            new_h_index_active.clearHost(false);
        }

        RS_MemoryPtr& new_h_inactive = (new_is_a_active ? host_b : host_a);
        RS_MemoryPtr& new_d_inactive = (new_is_a_active ? device_b : device_a);

        RS_BufferIndex& new_h_index_inactive = (new_is_a_active ? host_index_b : host_index_a);
        RS_BufferIndex& new_d_index_inactive = (new_is_a_active ? device_index_b : device_index_a);

        if (device_to_host) {
            new_h_index_inactive.copyChunkData(new_d_index_inactive, true);
            RBigSize sync_size = new_h_index_inactive.usedDataSize();
            if (sync_size > 0) {
                new_h_inactive.copyFrom(new_d_inactive, sync_size, sync, ordered);
            }
        } else {
            new_d_index_inactive.copyChunkData(new_h_index_inactive, true);
            RBigSize sync_size = new_h_index_inactive.usedDataSize();
            if (sync_size > 0) {
                new_d_inactive.copyFrom(new_h_inactive, sync_size, sync, ordered);
            }
        }
    }

    /**
     * @brief Wait for the copy to complete.
     */
    void waitForCopyFinish() const {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        mem_provider->sync(false, true, true);
    }

    /**
     * @brief Whether the active host buffer is empty.
     */
    bool isHostEmpty() {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_BufferIndex& h_index = (is_a_active_host ? host_index_a : host_index_b);
        return h_index.chunkNum() == 0;
    }

    /**
     * @brief Allocate a chunk on the active host side (A or B), returning
     *        an RS_MemoryPtr to the start of that region.  The caller
     *        may write data directly to this pointer.
     *
     * @param chunk_size  Number of bytes requested.
     * @return RS_MemoryPtr pointing to the newly allocated region,
     *         or an empty (null) pointer if push fails.
     */
    RS_MemoryPtr pushHost(RBigSize chunk_size) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_BufferIndex& h_index = (is_a_active_host ? host_index_a : host_index_b);
        RS_MemoryPtr& h_buffer = (is_a_active_host ? host_a : host_b);

        if (!h_index.canPushChunk(chunk_size)) {
            return {};
        }
        RBigSize offset = h_index.usedDataSize();
        h_index.pushChunk(chunk_size);

        auto mem_ptr = h_buffer + offset;
        mem_ptr.bytes = chunk_size;
        return mem_ptr;
    }

    /**
     * @brief Pop (remove) the last chunk from the active host side.
     */
    void popHost() {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_BufferIndex& h_index = (is_a_active_host ? host_index_a : host_index_b);
        h_index.popChunk();
    }

    /**
     * @brief Get the number of chunks in the active host buffer.
     */
    RSize getHostNum() {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_BufferIndex& h_index = (is_a_active_host ? host_index_a : host_index_b);
        return h_index.chunkNum();
    }

    /**
     * @brief Get a chunk from the active host buffer.
     *
     * @param chunk_index The index of the chunk to get.
     * @return A pointer pointing to the start of the chunk with size of the chunk.
     */
    RS_MemoryPtr getHostChunk(RIndex chunk_index) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_BufferIndex& h_index = (is_a_active_host ? host_index_a : host_index_b);
        return h_index.getChunk(chunk_index);
    }

    /**
     * @brief Clear the active host buffer.
     */
    void clearHost() {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        RS_BufferIndex& h_index = (is_a_active_host ? host_index_a : host_index_b);
        h_index.clearHost();
    }

    /**
     * @brief Whether the active device buffer is empty.
     */
    HOST_DEVICE_INLINE_QUALIFIER bool isDeviceEmpty() {
        RS_BufferIndex& d_index = (is_a_active_device.get() ? device_index_a : device_index_b);
        return d_index.chunkNum() == 0;
    }

    /**
     * @brief Allocate a chunk on the active device side (A or B).
     *        This returns an RS_MemoryPtr to the start of the new
     *        chunk in device memory.  The caller (host or device)
     *        can then write data to that pointer.
     *
     * @param chunk_size  Bytes requested
     * @return RS_MemoryPtr (device pointer) or an empty pointer if fail
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_MemoryPtr pushDevice(RBigSize chunk_size) {
        bool is_a_active = is_a_active_device.get();

        RS_BufferIndex& d_index = (is_a_active ? device_index_a : device_index_b);
        RS_MemoryPtr d_buffer = (is_a_active ? device_a : device_b);

        if (!d_index.canPushChunk(chunk_size)) {
            return {};
        }

        RBigSize offset = d_index.usedDataSize();
        d_index.pushChunk(chunk_size);

        auto mem_ptr = d_buffer + offset;
        mem_ptr.bytes = chunk_size;

        return mem_ptr;
    }

    /**
     * @brief Pop the last chunk from the active device side (A or B).
     */
    HOST_DEVICE_INLINE_QUALIFIER void popDevice() {
        bool is_a_active = is_a_active_device.get();
        RS_BufferIndex& d_index = (is_a_active ? device_index_a : device_index_b);
        d_index.popChunk();
    }

    /**
     * @brief Clear the active device buffer.
     */
    HOST_DEVICE_INLINE_QUALIFIER void clearDevice() {
        bool is_a_active = is_a_active_device.get();
        RS_BufferIndex& d_index = (is_a_active ? device_index_a : device_index_b);
        d_index.clearDevice();
    }

    /**
     * @brief Get the number of chunks in the active device buffer.
     */
    HOST_DEVICE_INLINE_QUALIFIER RSize getDeviceNum() {
        bool is_a_active = is_a_active_device.get();
        RS_BufferIndex& d_index = (is_a_active ? device_index_a : device_index_b);
        return d_index.chunkNum();
    }

    /**
     * @brief Get a chunk from the active device buffer.
     *
     * @param chunk_index The index of the chunk to get.
     * @return A pointer pointing to the start of the chunk with size of the chunk.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_MemoryPtr getDeviceChunk(RIndex chunk_index) {
        bool is_a_active = is_a_active_device.get();
        RS_BufferIndex& d_index = (is_a_active ? device_index_a : device_index_b);
        return d_index.getChunk(chunk_index);
    }
};

/**
 * @brief RS_DirectBuffer is a single-pair buffer container with host
 *        and device regions. Synchronization between buffers is always
 *        immediate through sync() method.
 */
class RS_DirectBuffer {
   public:
    /*****************************************************************
     * Data
     *****************************************************************/

    /// @brief Memory provider.
    RS_MemoryProvider* mem_provider = nullptr;

    /// @brief Host buffer
    RS_MemoryPtr host;

    /// @brief Device buffer
    RS_MemoryPtr device;

    /// @brief Maximum total data size
    RBigSize max_data_size = 0;

    /*****************************************************************
     * Methods
     *****************************************************************/

    /// @brief Default constructor.
    RS_DirectBuffer() = default;

    /**
     * @brief Construct and initialize the direct buffer with given capacities.
     *
     * @param max_data_size  Maximum bytes per buffer.
     * @param provider       Memory provider.
     */
    RS_DirectBuffer(RBigSize max_data_size, RS_MemoryProvider* provider) {
        init(max_data_size, provider);
    }

    /**
     * @brief Allocate host/device buffers and initialize indexes.
     *
     * @param max_data_size  Maximum bytes per buffer
     * @param provider       Memory provider
     */
    void init(RBigSize max_data_size, RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        mem_provider = provider;

        // Allocate data buffers
        host = provider->allocateHost(max_data_size, true);
        device = provider->allocate(max_data_size, true);
        this->max_data_size = max_data_size;
    }

    /**
     * @brief Free all buffer memory.
     */
    void free() {
        mem_provider = nullptr;
        host.free();
        device.free();
        max_data_size = 0;
    }

    /**
     * @brief Synchronize buffers immediately in specified direction.
     *
     * @param device_to_host  If true: device->host copy, else host->device copy
     */
    void sync(bool device_to_host) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        if (device_to_host) {
            host.copyFrom(device, max_data_size, true, true);
        } else {
            device.copyFrom(host, max_data_size, true, true);
        }
    }

    /**
     * @brief Whether the host buffer is empty. Always true.
     */
    bool isHostEmpty() {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        return true;
    }

    /**
     * @brief Allocate a chunk on host side. If the chunk size is larger
     * than the max data size, return an empty pointer. Otherwise, return
     * the host pointer.
     *
     * @param chunk_size  Number of bytes requested.
     * @return RS_MemoryPtr pointing to newly allocated region,
     *         or empty pointer if push fails.
     */
    RS_MemoryPtr pushHost(RBigSize chunk_size) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        if (chunk_size > max_data_size) {
            return {};
        }
        return host;
    }

    /**
     * @brief Pop (remove) the last chunk from host side. Always empty.
     */
    void popHost() { RS_DATA_EXPECT_INIT(mem_provider != nullptr); }

    /**
     * @brief Get number of chunks in host buffer. Always 0.
     */
    RSize getHostNum() {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        return 0;
    }

    /**
     * @brief Get a chunk from host buffer.
     *
     * @param chunk_index  Index of the chunk to get.
     * @return Pointer to start of chunk with size.
     */
    RS_MemoryPtr getHostChunk(RIndex chunk_index) {
        RS_DATA_EXPECT_INIT(mem_provider != nullptr);
        if (chunk_index != 0) {
            RS_THROW(
                RS_OutOfBoundsException("Invalid chunk index {} for host buffer", chunk_index)
            );
        }
        return host;
    }

    /**
     * @brief Clear host buffer.
     */
    void clearHost() { RS_DATA_EXPECT_INIT(mem_provider != nullptr); }

    /**
     * @brief Whether device buffer is empty.
     */
    HOST_DEVICE_INLINE_QUALIFIER bool isDeviceEmpty() { return false; }

    /**
     * @brief Allocate a chunk on device side.
     *
     * @param chunk_size  Bytes requested
     * @return RS_MemoryPtr (device pointer) or empty pointer if fail
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_MemoryPtr pushDevice(RBigSize chunk_size) {
        if (chunk_size > max_data_size) {
            return {};
        }
        return device;
    }

    /**
     * @brief Pop last chunk from device side.
     */
    HOST_DEVICE_INLINE_QUALIFIER void popDevice() {}

    /**
     * @brief Clear device buffer.
     */
    HOST_DEVICE_INLINE_QUALIFIER void clearDevice() {}

    /**
     * @brief Get number of chunks in device buffer.
     */
    HOST_DEVICE_INLINE_QUALIFIER RSize getDeviceNum() { return 0; }

    /**
     * @brief Get a chunk from device buffer.
     *
     * @param chunk_index  Index of the chunk to get.
     * @return Pointer to start of chunk with size.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_MemoryPtr getDeviceChunk(RIndex chunk_index) {
        assert(chunk_index == 0);
        return device;
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_BUFFER_H