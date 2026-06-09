// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_MEMPOOL_H
#define RS_MEMPOOL_H

#include <vector>

#include "rise_origin/device/rs_dev_util.h"
#include "rs_array.h"
#include "rs_bitmask.h"
#include "rs_data_util.h"
namespace rise {

namespace device {

/**
 * @brief RS_MemArrayPool is an Array based memory pool. Getting chunk
 * pointer is much faster than RS_MemPool since we don't retrieve the pointer
 * from memory, but just compute with index and the base memory pointer stub.
 */
template <RSize ChunkBytes_>
class RS_MemArrayPool {
   public:
    static constexpr RSize ChunkBytes = ChunkBytes_;
    struct Chunk {
        char data[ChunkBytes_];
    };
    /*****************************************************************
     * Data
     *****************************************************************/
    RS_DivMod<RSize> chunk_num_divmod;
    RS_BitMask<> chunk_mask;
    RS_Array<Chunk, 1> chunks;

    /*****************************************************************
     * Methods
     *****************************************************************/
    /// @brief Default constructor.
    RS_MemArrayPool() = default;

    /**
     * @brief Constructor that initializes an empty memory pool.
     * @param provider Memory provider.
     */
    explicit RS_MemArrayPool(RS_MemoryProvider* provider) { init(provider); }

    /**
     * @brief Constructor that initializes the memory pool with the given number of chunks and
     *        memory provider.
     * @param chunks Number of chunks to initialize the memory pool with.
     * @param provider Memory provider.
     */
    RS_MemArrayPool(RSize chunks, RS_MemoryProvider* provider) { init(chunks, provider); }

    /**
     * @brief Initializes an empty memory pool.
     * @param provider Memory provider.
     */
    void init(RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        chunk_num_divmod = RS_DivMod<RSize>();
        chunk_mask.init(provider);
        chunks.init(provider);
    }

    /**
     * @brief Initializes a memory pool with the given number of chunks and memory provider.
     * @param chunks Number of chunks to initialize the memory pool with.
     * @param provider Memory provider.
     */
    void init(RSize chunks, RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        chunk_num_divmod = RS_DivMod<RSize>(chunks == 0 ? 1 : chunks);
        chunk_mask.init(chunks, provider);
        this->chunks.init(chunks, provider);
    }

    /**
     * @brief Frees all memory regions in the pool.
     * @warning It will free the data, all related lists will be affected.
     */
    void free() {
        chunk_num_divmod = RS_DivMod<RSize>();
        chunk_mask.free();
        chunks.free();
    }

    /**
     * @brief Resizes the memory pool to the given number of chunks.
     * @param new_size New size of the memory pool.
     * @param copy Whether to copy the data from the old memory pool to the new one.
     */
    void resize(RSize new_size, bool copy = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        chunk_num_divmod.setDivisor(new_size);
        chunk_mask.resize(new_size, copy);
        chunks.resize({new_size}, copy);
    }

    /**
     * @brief Clears the memory pool and releases all allocated chunks.
     * @param sync Whether operation is synchronous.
     */
    void clear(bool sync = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        chunk_mask.fill(false, sync);
    }

    /**
     * @brief Copies the memory pool.
     * @param mem_array_pool Memory pool to be copied.
     */
    void copy(const RS_MemArrayPool& mem_array_pool) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(mem_array_pool.provider() != nullptr);

        if (this == &mem_array_pool) return;
        chunk_num_divmod = mem_array_pool.chunk_num_divmod;
        chunk_mask.copy(mem_array_pool.chunk_mask);
        chunks.copy(mem_array_pool.chunks);
    }

    /**
     * @brief Moves the memory pool to the specified provider and returns a new memory pool.
     * @param provider Memory provider.
     * @return New memory pool.
     */
    RS_MemArrayPool move(RS_MemoryProvider* provider) const {
        RS_DATA_EXPECT_INIT(this->provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        RS_MemArrayPool mem_array_pool;
        mem_array_pool.chunk_num_divmod = chunk_num_divmod;
        mem_array_pool.chunk_mask = chunk_mask.move(provider);
        mem_array_pool.chunks = chunks.move(provider);
        return mem_array_pool;
    }

    /// @brief Returns the memory provider.
    [[nodiscard]] RS_MemoryProvider* provider() const { return chunks.provider(); }

    /// @brief Returns the device location.
    [[nodiscard]] RS_DeviceLocation device() const { return chunks.device(); }

    /// @brief Returns the number of allocated chunks.
    [[nodiscard]] RSize getAllocatedNum() const {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        auto host_chunk_mask = chunk_mask.move(&RS_MemoryDefaultProvider::getInstance());
        RSize count = host_chunk_mask.count();
        host_chunk_mask.free();
        return count;
    }

    /// @brief Returns the ratio of allocated chunks versus total chunks.
    [[nodiscard]] RFloat getAllocatedRatio() const {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        return static_cast<RFloat>(getAllocatedNum()) / static_cast<RFloat>(chunks.size());
    }

    /// @brief Acquire a chunk from the memory pool.
    /// @param last_chunk_seed The index of the last chunk index allocated.
    /// @param thread_seed The seed of the thread.
    HOST_DEVICE_INLINE_QUALIFIER RIndex
    acquire(RIndex last_chunk_seed = RS_NULL_INDEX, RIndex thread_seed = RS_NULL_INDEX) {
        // Compute partition index using knuth multiplicative hash function
        // 3 is added to prevent both seeds are 0 / NULL_INDEX and cause initial hash to be 0
        RIndex hash = (3 + thread_seed + last_chunk_seed) * 2654435761U;
        RIndex start_index = chunk_num_divmod.mod(hash);

        // Search forward for an available chunk
        RIndex index = start_index;
        while (index < chunks.size()) {
            index = chunk_mask.findNext<false>(index);
            if (index == RS_NULL_INDEX) {
                break;
            }
            if (chunk_mask.lockAtomic(index)) {
                return index;
            }
            // Not available, search next
        }

        // Search backward if forward search failed
        if (start_index > 0) {
            index = start_index - 1;
            while (index < chunks.size()) {
                index = chunk_mask.findPrev<false>(index);
                if (index == RS_NULL_INDEX) {
                    break;
                }
                if (chunk_mask.lockAtomic(index)) {
                    return index;
                }
                // Not available, search next
            }
        }

        // No chunk is available
        return RS_NULL_INDEX;
    }

    /// @brief Releases a chunk back to the memory pool.
    HOST_DEVICE_INLINE_QUALIFIER void release(RIndex chunk) {
        if (chunk > 0 && chunk < chunks.size()) {
            chunk_mask.setOffAtomic(chunk);
        }
    }

    HOST_DEVICE_INLINE_QUALIFIER void* getChunkPtr(RIndex chunk) const {
        if (chunk == RS_NULL_INDEX) return nullptr;
        assert(chunk < chunks.size());
        return (chunks.data + sizeof(Chunk) * chunk).data;
    }

    /// @brief Returns the number of chunks in the memory pool.
    HOST_DEVICE_INLINE_QUALIFIER RSize size() const { return chunks.size(); }

    /// @brief Returns whether the memory pool is empty.
    HOST_DEVICE_INLINE_QUALIFIER constexpr bool empty() const { return chunks.empty(); }

    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const { return sizeof(*this); }

    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const {
        return chunk_mask.extraBytes() + chunks.extraBytes();
    }
};

template <RSize ChunkBytes_>
class RS_MemPool {
   public:
    static constexpr RSize ChunkBytes = ChunkBytes_;
    /*****************************************************************
     * Data
     *****************************************************************/
    RSize chunk_num = 0;
    RSize memory_region_num = 0;
    RS_DivMod<RSize> chunk_num_divmod;
    RS_BitMask<> chunk_mask;
    RS_Array<void*, 1> chunk_ptrs;
    std::vector<RS_MemoryPtr>* host_memory_regions = nullptr;

    /*****************************************************************
     * Methods
     *****************************************************************/
    /// @brief Default constructor.
    RS_MemPool() = default;

    /// @brief Constructor that initializes an empty memory pool.
    explicit RS_MemPool(RS_MemoryProvider* provider) { init(provider); }

    /// @brief Constructor that initializes the memory pool with the given number of chunks and
    /// memory provider.
    RS_MemPool(RSize chunks, RS_MemoryProvider* provider) { init(chunks, provider); }

    /// @brief Initializes an empty memory pool.
    void init(RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        chunk_num = 0;
        memory_region_num = 0;
        chunk_num_divmod = RS_DivMod<RSize>();
        chunk_mask.init(provider);
        chunk_ptrs.init(provider);
        host_memory_regions = new std::vector<RS_MemoryPtr>();
    }

    /// @brief Initializes a memory pool with the given number of chunks and memory provider.
    void init(RSize chunks, RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        chunk_num = 0;
        memory_region_num = 0;
        chunk_num_divmod = RS_DivMod<RSize>();
        chunk_mask.init(provider);
        chunk_ptrs.init(provider);
        host_memory_regions = new std::vector<RS_MemoryPtr>();
        grow(chunks);
    }

    /// @brief Grows the memory pool by allocating new chunks.
    /// @param new_chunks The number of new chunks to allocate.
    void grow(RSize new_chunks) {
        RS_DATA_EXPECT_INIT(host_memory_regions != nullptr);

        if (new_chunks == 0) return;

        RS_MemoryProvider* mem_provider = provider();
        RSize old_chunk_num = chunk_num;
        chunk_num += new_chunks;
        chunk_num_divmod.setDivisor(chunk_num);

        // Resize chunk_mask and chunk_ptrs
        chunk_mask.resize(chunk_num);
        chunk_ptrs.resize({chunk_num});

        // Allocate new memory using the provider
        size_t region_size = new_chunks * ChunkBytes;
        RS_MemoryPtr mem_ptr = mem_provider->allocate(region_size, /*sync=*/true);

        // Store the new memory region
        host_memory_regions->push_back(mem_ptr);
        memory_region_num++;

        // Store pointers
        void* base_ptr = mem_ptr.data;
        auto new_ptrs = new void*[new_chunks];
        for (RSize i = 0; i < new_chunks; ++i) {
            void* chunk_ptr = static_cast<std::byte*>(base_ptr) + i * ChunkBytes;
            new_ptrs[i] = chunk_ptr;
        }
        chunk_ptrs.writeSlice(new_ptrs, new_chunks, old_chunk_num);
        delete[] new_ptrs;
    }

    /// @brief Frees all memory regions in the pool.
    void free() {
        if (!host_memory_regions) return;

        // Free all memory regions
        for (auto& region : *host_memory_regions) {
            region.free();
        }

        // Reset chunk_num and free other resources
        chunk_num = 0;
        memory_region_num = 0;
        chunk_num_divmod = RS_DivMod<RSize>();
        chunk_mask.free();
        chunk_ptrs.free();
        delete host_memory_regions;
        host_memory_regions = nullptr;
    }

    /// @brief Clears the memory pool and releases all allocated chunks.
    void clear(bool sync = true) {
        RS_DATA_EXPECT_INIT(host_memory_regions != nullptr);

        chunk_mask.fill(false, sync);
    }

    /// @brief Copies the memory pool from another memory pool. For simplicity, it just
    /// frees the current memory pool and moves the other memory pool to the current provider.
    /// @note The copied memory pool is joined as one contiguous memory region.
    void copy(const RS_MemPool& mem_pool) {
        RS_DATA_EXPECT_INIT(host_memory_regions != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(mem_pool.host_memory_regions != nullptr);

        if (&mem_pool == this) return;
        RS_MemoryProvider* mem_provider = provider();
        free();
        auto moved_mem_pool = mem_pool.move(mem_provider);
        chunk_num = moved_mem_pool.chunk_num;
        memory_region_num = moved_mem_pool.memory_region_num;
        chunk_num_divmod = moved_mem_pool.chunk_num_divmod;
        chunk_mask = moved_mem_pool.chunk_mask;
        chunk_ptrs = moved_mem_pool.chunk_ptrs;
        host_memory_regions = new std::vector<RS_MemoryPtr>(*moved_mem_pool.host_memory_regions);
    }

    /// @brief Moves the memory pool to the specified provider and returns a new memory pool.
    /// @note The returned memory pool is joined as one contiguous memory region.
    RS_MemPool move(RS_MemoryProvider* provider) const {
        RS_DATA_EXPECT_INIT(host_memory_regions != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        if (chunk_num == 0) return RS_MemPool(provider);

        RS_MemPool mem_pool;

        // Move basic data members
        mem_pool.chunk_num = chunk_num;
        mem_pool.memory_region_num = 1;  // One new contiguous memory region
        mem_pool.chunk_num_divmod = chunk_num_divmod;

        // Move chunk_mask
        mem_pool.chunk_mask = chunk_mask.move(provider);

        // Allocate new chunk_ptrs
        mem_pool.chunk_ptrs = RS_Array<void*, 1>(chunk_num, provider);

        // Allocate one new memory region
        RSize total_size = chunk_num * ChunkBytes;
        RS_MemoryPtr new_mem_ptr = provider->allocate(total_size, /*sync=*/true);

        // Update chunk_ptrs
        auto new_chunk_ptrs = new void*[chunk_num];
        for (RSize i = 0; i < chunk_num; ++i) {
            new_chunk_ptrs[i] = static_cast<std::byte*>(new_mem_ptr.data) + i * ChunkBytes;
        }
        mem_pool.chunk_ptrs.write(new_chunk_ptrs);
        delete[] new_chunk_ptrs;

        // Update memory_regions as one contiguous memory region
        mem_pool.host_memory_regions = new std::vector<RS_MemoryPtr>();
        mem_pool.host_memory_regions->push_back(new_mem_ptr);

        RSize offset = 0;
        for (auto& region : *host_memory_regions) {
            region.copyTo(new_mem_ptr + offset, region.size());
            offset += region.size();
        }

        return mem_pool;
    }

    [[nodiscard]] RS_MemoryProvider* provider() const { return chunk_mask.provider(); }

    [[nodiscard]] RS_DeviceLocation device() const { return chunk_mask.device(); }

    /// @brief Returns the number of allocated chunks.
    [[nodiscard]] RSize getAllocatedNum() const {
        RS_DATA_EXPECT_INIT(host_memory_regions != nullptr);

        auto host_chunk_mask = chunk_mask.move(&RS_MemoryDefaultProvider::getInstance());
        RSize count = host_chunk_mask.count();
        host_chunk_mask.free();
        return count;
    }

    /// @brief Returns the ratio of allocated chunks versus total chunks.
    [[nodiscard]] RFloat getAllocatedRatio() const {
        RS_DATA_EXPECT_INIT(host_memory_regions != nullptr);

        return static_cast<RFloat>(getAllocatedNum()) / static_cast<RFloat>(chunk_num);
    }

    /// @brief Acquire a chunk from the memory pool.
    /// @param last_chunk_seed The index of the last chunk index allocated.
    /// @param thread_seed The seed of the thread.
    HOST_DEVICE_INLINE_QUALIFIER RIndex
    acquire(RIndex last_chunk_seed = RS_NULL_INDEX, RIndex thread_seed = RS_NULL_INDEX) {
        // Compute partition index using knuth multiplicative hash function
        // 3 is added to prevent both seeds are 0 / NULL_INDEX and cause initial hash to be 0
        RIndex hash = (3 + thread_seed + last_chunk_seed) * 2654435761U;
        RIndex start_index = chunk_num_divmod.mod(hash);

        // Search forward for an available chunk
        RIndex index = start_index;
        while (index < chunk_num) {
            index = chunk_mask.findNext<false>(index);
            if (index == RS_NULL_INDEX) {
                break;
            }
            if (chunk_mask.lockAtomic(index)) {
                return index;
            }
            // Not available, search next
        }

        // Search backward if forward search failed
        if (start_index > 0) {
            index = start_index - 1;
            while (index < chunk_num) {
                index = chunk_mask.findPrev<false>(index);
                if (index == RS_NULL_INDEX) {
                    break;
                }
                if (chunk_mask.lockAtomic(index)) {
                    return index;
                }
                // Not available, search next
            }
        }

        // No chunk is available
        return RS_NULL_INDEX;
    }

    /// @brief Releases a chunk back to the memory pool.
    HOST_DEVICE_INLINE_QUALIFIER void release(RIndex chunk) {
        if (chunk > 0 && chunk < chunk_num) {
            chunk_mask.setOffAtomic(chunk);
        }
    }

    HOST_DEVICE_INLINE_QUALIFIER void* getChunkPtr(RIndex chunk) const {
        if (chunk == RS_NULL_INDEX) return nullptr;
        if (chunk < chunk_num) return chunk_ptrs.get(chunk);
        return nullptr;
    }

    /// @brief Returns the number of chunks in the memory pool.
    HOST_DEVICE_INLINE_QUALIFIER RSize size() const { return chunk_num; }

    /// @brief Returns whether the memory pool is empty.
    HOST_DEVICE_INLINE_QUALIFIER constexpr bool empty() const { return chunk_num == 0; }

    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const { return sizeof(*this); }

    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const {
        return chunk_mask.extraBytes() + chunk_ptrs.extraBytes();
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_MEMPOOL_H
