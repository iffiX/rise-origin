// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_LIST_H
#define RS_LIST_H

#include "rise_origin/device/rs_static_construct.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_atomic.h"
#include "rs_mempool.h"
#include "rs_value.h"

namespace rise {

namespace device {

class RSCF_ListRebuild : public RS_ComputeFunction<RSCF_ListRebuild> {
   public:
    template <typename List>
    HOST_DEVICE_INLINE_QUALIFIER static void run(List list) {
        list.header.get().rebuild(&list.mem_pool);
    }
};

/**
 * @brief RS_ListChunk is a chunk of a list.
 *
 * @tparam T The type of elements stored.
 * @tparam ChunkSize The number of elements per chunk (must be power-of-two).
 */
template <typename T, RSize ChunkSize = 32>
class RS_ListChunk {
   public:
    using ObjectType = T;
    static_assert(isPowerOfTwo(ChunkSize), "ChunkSize must be power of two");

    RS_ListChunk* prev = nullptr;
    RS_ListChunk* next = nullptr;
    RIndex next_index = RS_NULL_INDEX;
    RIndex mem_pool_index = RS_NULL_INDEX;
    T data[ChunkSize];
};

/**
 * @brief RS_ListHeader is a header for a list of chunks.
 *
 * @tparam T The type of elements stored.
 * @tparam ChunkSize The number of elements per chunk (must be power-of-two).
 * @tparam MemPoolType The template for the memory pool used to allocate chunks.
 */
template <typename T, RSize ChunkSize = 32, template <RSize> typename MemPoolType = RS_MemArrayPool>
class RS_ListHeader {
   public:
    using ObjectType = T;
    using ChunkType = RS_ListChunk<T, ChunkSize>;
    using MemPool = MemPoolType<sizeof(ChunkType)>;
    static constexpr RSize ChunkMask = ChunkSize - 1;
    static constexpr RSize ChunkShift = logTwoOf(ChunkSize);

    RSize elem_num = 0;
    RIndex head_index = RS_NULL_INDEX;
    ChunkType* head = nullptr;
    ChunkType* tail = nullptr;

    RS_ListHeader() = default;

    /// @brief Returns the number of elements in the list.
    HOST_DEVICE_INLINE_QUALIFIER RSize size() const { return elem_num; }

    /// @brief Returns whether the list is empty.
    HOST_DEVICE_INLINE_QUALIFIER bool empty() const { return elem_num == 0; }

    /// @brief Access operator for the list.
    HOST_DEVICE_INLINE_QUALIFIER T& operator[](RIndex index) { return get(index); }

    /// @brief Const access operator for the list.
    HOST_DEVICE_INLINE_QUALIFIER const T& operator[](RIndex index) const { return get(index); }

    /// @brief Returns the reference to the element at the specified index.
    HOST_DEVICE_INLINE_QUALIFIER T& get(RIndex index) {
        assert(index < elem_num);
        ChunkType* chunk = findChunk(index);
        return chunk->data[index & ChunkMask];
    }

    /// @brief Returns the const reference to the element at the specified index.
    HOST_DEVICE_INLINE_QUALIFIER const T& get(RIndex index) const {
        assert(index < elem_num);
        ChunkType* chunk = findChunk(index);
        return chunk->data[index & ChunkMask];
    }

    /// @brief Sets the value at the specified index.
    HOST_DEVICE_INLINE_QUALIFIER void set(RIndex index, const T& value) {
        assert(index < elem_num);
        ChunkType* chunk = findChunk(index);
        chunk->data[index & ChunkMask] = value;
    }

    /**
     * @brief Finds the chunk containing the element at the specified index.
     * @param index The index of the element to find.
     * @return The chunk containing the element.
     */
    HOST_DEVICE_INLINE_QUALIFIER ChunkType* findChunk(RIndex index) const {
        assert(index < elem_num);

        ChunkType* chunk = nullptr;
        RIndex chunk_index = index >> ChunkShift;

        if (index < (elem_num >> 1)) {
            // Search from head
            chunk = head;
            for (RIndex i = 0; i < chunk_index; ++i) {
                chunk = chunk->next;
            }
        } else {
            // Search from tail
            chunk = tail;
            // (elem_num + ChunkMask) >> ChunkShift is equivalent of RS_CEIL_DIV(elem_num,
            // ChunkSize)
            for (RIndex i = ((elem_num + ChunkMask) >> ChunkShift) - 1; i > chunk_index; --i) {
                chunk = chunk->prev;
            }
        }

        return chunk;
    }

    /**
     * @brief Appends an element to the end of the list.
     * @param mem_pool The memory pool to use.
     * @param value The value to append.
     * @param initial_seed The initial seed for the allocation of the first chunk.
     * @return True if the append succeeds, false otherwise.
     */
    HOST_DEVICE_INLINE_QUALIFIER bool
    append(MemPool* mem_pool, const T& value, RIndex initial_seed = RS_NULL_INDEX) {
        if (!(elem_num & ChunkMask)) {
            // All positions are filled in last chunk, allocate new chunk
            RIndex chunk_idx =
                mem_pool->acquire(tail == nullptr ? initial_seed : tail->mem_pool_index);
            if (chunk_idx == RS_NULL_INDEX) {
                return false;
            }
            auto chunk = reinterpret_cast<ChunkType*>(mem_pool->getChunkPtr(chunk_idx));
            chunk->next = nullptr;
            chunk->next_index = RS_NULL_INDEX;
            chunk->mem_pool_index = chunk_idx;
            // data[] is not initialized

            if (tail) {
                tail->next_index = chunk_idx;
                tail->next = chunk;
                chunk->prev = tail;
                tail = chunk;
            } else {
                head = chunk;
                tail = chunk;
                head_index = chunk_idx;
                chunk->prev = nullptr;
            }
        }
        tail->data[elem_num & ChunkMask] = value;
        ++elem_num;
        return true;
    }

    /**
     * @brief Removes the last element from the list.
     * @param mem_pool The memory pool to use.
     */
    HOST_DEVICE_INLINE_QUALIFIER void pop(MemPool* mem_pool) {
        assert(elem_num > 0);
        --elem_num;
        if (!(elem_num & ChunkMask)) {
            // At the start of a chunk, so deallocate the current tail chunk
            ChunkType* old_tail = tail;
            tail = tail->prev;
            if (tail) {
                tail->next = nullptr;
            } else {
                head = nullptr;
            }
            mem_pool->release(old_tail->mem_pool_index);
        }
        if (elem_num == 0) {
            head_index = RS_NULL_INDEX;
            head = nullptr;
            tail = nullptr;
        }
    }

    /**
     * @brief Clears the list and releases all allocated chunks.
     * @param mem_pool The memory pool to use.
     */
    HOST_DEVICE_INLINE_QUALIFIER void clear(MemPool* mem_pool) {
        ChunkType* current = head;
        while (current != nullptr) {
            ChunkType* next = current->next;
            mem_pool->release(current->mem_pool_index);
            current = next;
        }
        head_index = RS_NULL_INDEX;
        head = nullptr;
        tail = nullptr;
        elem_num = 0;
    }

    /**
     * @brief Rebuild pointers for chunks in chain using their chunk index.
     * This needs to be called after the mem pool is moved.
     * @param new_mem_pool The new memory pool to use.
     */
    HOST_DEVICE_INLINE_QUALIFIER void rebuild(MemPool* new_mem_pool) {
        if (head_index == RS_NULL_INDEX) {
            return;
        }
        RSize chunk_num = (elem_num + (ChunkSize - 1)) >> ChunkShift;
        if (chunk_num > 0) {
            head = static_cast<ChunkType*>(new_mem_pool->getChunkPtr(head_index));
            ChunkType* current_chunk_ptr = nullptr;
            ChunkType* next_chunk_ptr = head;
            for (RSize i = 0; i < chunk_num; i++) {
                next_chunk_ptr->prev = current_chunk_ptr;
                current_chunk_ptr = next_chunk_ptr;
                next_chunk_ptr = static_cast<ChunkType*>(
                    new_mem_pool->getChunkPtr(current_chunk_ptr->next_index)
                );
                current_chunk_ptr->next = next_chunk_ptr;
            }
            tail = current_chunk_ptr;
        }
    }
};

template <typename ListHeaderType, RSize ListNum>
class RS_ListView {
   public:
    using ObjectType = typename ListHeaderType::ObjectType;
    using MemPool = typename ListHeaderType::MemPool;
    ListHeaderType* lists[ListNum] = {nullptr};
    RSize list_prefix_sizes[ListNum] = {0};

    RS_ListView() = default;

    /**
     * @brief Constructs an RS_ListView with the given lists organized by list headers.
     * @param lists The lists to be included in the view.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_ListView(
        rs_device_std::initializer_list<ListHeaderType*> lists
    ) {
        assert(lists.size() == ListNum);
        RIndex i = 0;
        for (auto list : lists) {
            this->lists[i++] = list;
        }
        _computePrefixSizes();
    }

    /**
     * @brief Returns the size of the view.
     * @return The size of the view.
     */
    HOST_DEVICE_INLINE_QUALIFIER RSize size() const { return list_prefix_sizes[ListNum - 1]; }

    /// @brief Returns the reference to the element at the specified index.
    HOST_DEVICE_INLINE_QUALIFIER ObjectType& operator[](RIndex index) {
        assert(index < size());
        for (RIndex list_index = 0; list_index < ListNum - 1; list_index++) {
            if (index < list_prefix_sizes[list_index]) {
                return lists[list_index]->get(index);
            }
            index -= list_prefix_sizes[list_index];
        }
        return lists[ListNum - 1]->get(index);
    }

    /// @brief Returns the const reference to the element at the specified index.
    HOST_DEVICE_INLINE_QUALIFIER const ObjectType& operator[](RIndex index) const {
        return operator[](index);
    }

    /// @brief Returns the reference to the element at the specified index.
    HOST_DEVICE_INLINE_QUALIFIER ObjectType& get(RIndex index) { return operator[](index); }

    /// @brief Returns the const reference to the element at the specified index.
    HOST_DEVICE_INLINE_QUALIFIER const ObjectType& get(RIndex index) const {
        return operator[](index);
    }

    /// @brief Sets the element at the specified index.
    HOST_DEVICE_INLINE_QUALIFIER void set(RIndex index, const ObjectType& value) {
        operator[](index) = value;
    }

    /**
     * @brief Removes the last element from the specified list.
     * @param mem_pool The memory pool to use.
     * @param list_index The index of the list to pop from.
     */
    HOST_DEVICE_INLINE_QUALIFIER void popFromList(MemPool* mem_pool, RIndex list_index) {
        assert(list_index < ListNum);
        if (lists[list_index] != nullptr) {
            lists[list_index]->pop(mem_pool);
            _computePrefixSizes();
        }
    }

    /**
     * @brief Appends an element to the specified list.
     * @param mem_pool The memory pool to use.
     * @param list_index The index of the list to append to.
     * @param value The value to append.
     */
    HOST_DEVICE_INLINE_QUALIFIER void
    appendToList(MemPool* mem_pool, RIndex list_index, const ObjectType& value) {
        assert(list_index < ListNum);
        if (lists[list_index] != nullptr) {
            lists[list_index]->append(mem_pool, value);
            _computePrefixSizes();
        }
    }

    /**
     * @brief Clears all lists.
     * @param mem_pool The memory pool to use.
     */
    HOST_DEVICE_INLINE_QUALIFIER void clear(MemPool* mem_pool) {
        for (RIndex i = 0; i < ListNum; i++) {
            if (lists[i] != nullptr) lists[i]->clear(mem_pool);
            list_prefix_sizes[i] = 0;
        }
    }

   private:
    HOST_DEVICE_INLINE_QUALIFIER void _computePrefixSizes() {
        if constexpr (ListNum > 0) {
            RSize prefix_size = lists[0] == nullptr ? 0 : lists[0]->size();
            loop<0, ListNum - 1>([&](auto i_const) {
                static constexpr RIndex i = i_const.value;
                list_prefix_sizes[i] = prefix_size;
                prefix_size = lists[i + 1] == nullptr ? prefix_size
                                                      : prefix_size + lists[i + 1]->size();
            });
            list_prefix_sizes[ListNum - 1] = prefix_size;
        }
    }
};

/**
 * @brief RS_List is a dynamic list container which can be dynamically appended and popped
 * on the device. It is implemented as a linked list of chunks. It is also possible to
 * construct individual list chunks managed by the list header from multiple threads and
 * then merged into a single list.
 *
 * @tparam T The type of elements stored.
 * @tparam ListChunkSize The number of elements per chunk (must be power-of-two).
 * @tparam MemPoolType The template for the memory pool used to allocate chunks.
 */
template <typename T, RSize ListChunkSize = 32, template <RSize> typename MemPoolType = RS_MemPool>
class RS_List {
   public:
    using ObjectType = T;
    using ListHeaderType = RS_ListHeader<T, ListChunkSize, MemPoolType>;
    using ListChunkType = typename ListHeaderType::ChunkType;
    // The memory pool type used to allocate chunks (each chunk is sizeof(ChunkType) bytes)
    using MemPool = MemPoolType<sizeof(ListChunkType)>;

    class RS_ListAccessor {
       public:
        RSize offset = 0;
        ListHeaderType header;

        HOST_DEVICE_INLINE_QUALIFIER
        RS_ListAccessor(RSize offset, ListHeaderType header) : offset(offset), header(header) {}

        HOST_DEVICE_INLINE_QUALIFIER T& operator[](RIndex index) { return header[index + offset]; }

        HOST_DEVICE_INLINE_QUALIFIER const T& operator[](RIndex index) const {
            return header[index + offset];
        }

        HOST_DEVICE_INLINE_QUALIFIER T& get(RIndex index) { return header[index + offset]; }

        HOST_DEVICE_INLINE_QUALIFIER const T& get(RIndex index) const {
            return header[index + offset];
        }

        HOST_DEVICE_INLINE_QUALIFIER void set(RIndex index, const T& value) {
            header.set(index + offset, value);
        }

        HOST_DEVICE_INLINE_QUALIFIER RSize size() const { return header.size() - offset; }
    };

    /*****************************************************************
     * Data
     *****************************************************************/
    /**
     * @brief The header that manages the list’s chunks.
     * @note When directly accessed, the mem_pool pointer is not valid. It is always
     *       valid when accessed through the list methods.
     */
    RS_Value<ListHeaderType> header;

    /**
     * @brief The spinlock used to synchronize access to the list.
     */
    RS_Value<RS_SpinLock> lock;

    /**
     * @brief The memory pool used to allocate chunks.
     */
    MemPool mem_pool;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_List() = default;

    /**
     * @brief Constructs an RS_List using the given memory provider.
     * @param provider Memory provider.
     */
    explicit RS_List(RS_MemoryProvider* provider) { init(provider); }

    /**
     * @brief Constructs an RS_List using the given memory provider and initializes the
     *        memory pool with the given number of chunks.
     * @param chunks Number of chunks to initialize the memory pool with.
     * @param provider Memory provider.
     */
    RS_List(RSize chunks, RS_MemoryProvider* provider) { init(chunks, provider); }

    /**
     * @brief Copy constructor.
     * @warning It is a shallow copy which only copies mempool handle and header.
     * @param list RS_List to be copied.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_List(const RS_List& list)
        : header(list.header), lock(list.lock), mem_pool(list.mem_pool) {}

    /**
     * @brief Initializes the list with the given memory provider.
     * @param provider Memory provider.
     */
    void init(RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        header.init(ListHeaderType(), provider);
        lock.init(provider);
        unsigned int lock_init = 0;
        lock.write(reinterpret_cast<RS_SpinLock*>(&lock_init), true);
        mem_pool.init(provider);
    }

    /**
     * @brief Initializes the list with the given memory provider and initializes the
     *        memory pool with the given number of chunks.
     * @param chunks Number of chunks to initialize the memory pool with.
     * @param provider Memory provider.
     */
    void init(RSize chunks, RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        header.init(ListHeaderType(), provider);
        lock.init(provider);
        unsigned int lock_init = 0;
        lock.write(reinterpret_cast<RS_SpinLock*>(&lock_init), true);
        mem_pool.init(chunks, provider);
    }

    /**
     * @brief Frees the GPU memory associated with the list.
     * @warning It will free the data, all related lists will be affected.
     */
    void free() {
        header.free();
        lock.free();
        mem_pool.free();
    }

    /**
     * @brief Grows the memory pool by the given number of chunks.
     * @note It is only available when the memory pool type is RS_MemPool.
     * @param new_chunks Number of chunks to grow the memory pool by.
     */
    template <
        typename U = MemPoolType<sizeof(ListChunkType)>,
        typename = std::enable_if_t<std::is_same_v<U, RS_MemPool<sizeof(ListChunkType)>>>>
    void grow(RSize new_chunks) {
        RS_DATA_EXPECT_INIT(header.provider() != nullptr);
        mem_pool.grow(new_chunks);
    }

    /**
     * @brief Resizes the memory pool to the given number of chunks.
     * @param new_size New size of the memory pool.
     * @param copy Whether to copy the data from the old memory pool to the new one.
     */
    template <
        typename U = MemPoolType<sizeof(ListChunkType)>,
        typename = std::enable_if_t<std::is_same_v<U, RS_MemArrayPool<sizeof(ListChunkType)>>>>
    void resize(RSize new_size, bool copy = true) {
        RS_DATA_EXPECT_INIT(header.provider() != nullptr);

        mem_pool.resize(new_size, copy);
    }

    /**
     * @brief Clears the list and releases all allocated chunks.
     * @param sync Whether operation is synchronous.
     */
    void clear(bool sync = true) {
        RS_DATA_EXPECT_INIT(header.provider() != nullptr);

        header.fill(ListHeaderType(), sync);
        mem_pool.clear(sync);
    }

    /**
     * @brief Copies the list.
     * @param list RS_List to be copied.
     */
    void copy(const RS_List& list) {
        RS_DATA_EXPECT_INIT(header.provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(list.header.provider() != nullptr);

        header.copy(list.header);
        mem_pool.copy(list.mem_pool);
        castProviderPtrAndCall(provider(), [&](auto p_ptr) {
            p_ptr->template run<RSCF_ListRebuild>({1}, *this);
        });
    }

    /**
     * @brief Moves the list to the specified provider and returns a new list.
     * @param provider Memory provider.
     * @return New list.
     */
    RS_List move(RS_MemoryProvider* provider) const {
        RS_DATA_EXPECT_INIT(header.provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        RS_List new_list;
        new_list.header = header.move(provider);
        new_list.mem_pool = mem_pool.move(provider);
        castProviderPtrAndCall(provider, [&](auto p_ptr) {
            p_ptr->template run<RSCF_ListRebuild>({1}, new_list);
        });
        return new_list;
    }

    /// @brief Returns the memory provider.
    [[nodiscard]] RS_MemoryProvider* provider() const { return header.provider(); }

    /// @brief Returns the device location.
    [[nodiscard]] RS_DeviceLocation device() const { return header.device(); }

    /// @brief Returns the number of elements in the list.
    HOST_DEVICE_INLINE_QUALIFIER RSize size() const {
#ifdef __CUDA_ARCH__
        return header.get().size();
#else
        ListHeaderType host_header;
        header.read(&host_header, true);
        return host_header.size();
#endif
    }

    /// @brief Returns whether the list is empty.
    HOST_DEVICE_INLINE_QUALIFIER bool empty() const {
#ifdef __CUDA_ARCH__
        return header.get().empty();
#else
        ListHeaderType host_header;
        header.read(&host_header, true);
        return host_header.empty();
#endif
    }

    /// @brief Returns the base size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RBigSize baseBytes() const { return sizeof(*this); }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RBigSize extraBytes() const {
        return header.extraBytes() + lock.extraBytes() + mem_pool.extraBytes();
    }

    /**
     * @brief Access operator for the list.
     * @note It is generally recommended to use the accessor to access the value,
     *       as it caches the header and is more efficient.
     * @param index Index of the element to access.
     * @return Reference to the element at the given index.
     */
    HOST_DEVICE_INLINE_QUALIFIER T& operator[](RIndex index) { return header.get()[index]; }

    /**
     * @brief Const access operator for the list.
     * @note It is generally recommended to use the accessor to access the value,
     *       as it caches the header and is more efficient.
     * @param index Index of the element.
     * @return Const reference to the element at the given index.
     */
    HOST_DEVICE_INLINE_QUALIFIER const T& operator[](RIndex index) const {
        return header.get()[index];
    }

    /**
     * @brief Returns the reference to the element at the specified index.
     * @note It is generally recommended to use the accessor to access the value,
     *       as it caches the header and is more efficient.
     * @param index Index of the element.
     * @return Reference to the element at the given index.
     */
    HOST_DEVICE_INLINE_QUALIFIER T& get(RIndex index) { return header.get()[index]; }

    /**
     * @brief Returns the const reference to the element at the specified index.
     * @note It is generally recommended to use the accessor to access the value,
     *       as it caches the header and is more efficient.
     * @param index Index of the element.
     * @return Const reference to the element at the given index.
     */
    HOST_DEVICE_INLINE_QUALIFIER const T& get(RIndex index) const { return header.get()[index]; }

    /**
     * @brief Sets the value at the specified index.
     * @note It is generally recommended to use the accessor to set the value,
     *       as it caches the header and is more efficient.
     * @param index Index of the element.
     * @param value New value to set.
     */
    template <bool Lock = true>
    HOST_DEVICE_INLINE_QUALIFIER void set(RIndex index, const T& value) {
        header.get().set(index, value);
    }

    /**
     * @brief Slices the list. By default, it slices the entire list.
     *
     * @param begin The index of the first element to access.
     * @param end The index of the last element to access.
     * @return Accessor to the list.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_ListAccessor
    slice(RIndex begin = 0, RIndex end = RS_NULL_INDEX) {
        RSize size = this->size();
        if (end == RS_NULL_INDEX) {
            end = size;
        }
        begin = RS_MIN(begin, size);
        end = RS_MAX(begin, RS_MIN(end, size));
        if (end > begin) {
            ListHeaderType sliced_header = header.get();
            RSize offset = begin & ListHeaderType::ChunkMask;

            auto begin_chunk_ptr = sliced_header.findChunk(begin);
            auto end_chunk_ptr = sliced_header.findChunk(end);

            // For sliced header, we skip the `offset` elements in the first chunk.
            // The total number of elements is `end - begin + offset`.
            sliced_header.head_index = begin_chunk_ptr->mem_pool_index;
            sliced_header.head = begin_chunk_ptr;
            sliced_header.tail = end_chunk_ptr;
            sliced_header.elem_num = end - begin + offset;

            return RS_ListAccessor(offset, sliced_header);
        } else {
            return RS_ListAccessor(0, ListHeaderType());
        }
    }

    /**
     * @brief Appends an element to the end of the list.
     *
     * @tparam Lock Whether to lock the list during operation.
     * @param value The element to append.
     * @param initial_seed Optional seed used during allocation of the first chunk.
     * @return True if the append succeeds, false otherwise.
     */
    template <bool Lock = true>
    HOST_DEVICE_INLINE_QUALIFIER bool append(const T& value, RIndex initial_seed = RS_NULL_INDEX) {
        if constexpr (Lock) {
            lock.get().lock();
        }
        bool result = header.get().append(&mem_pool, value, initial_seed);
        if constexpr (Lock) {
            lock.get().unlock();
        }
        return result;
    }

    /**
     * @brief Removes the last element from the list.
     *
     * @tparam Lock Whether to lock the list during operation.
     */
    template <bool Lock = true>
    HOST_DEVICE_INLINE_QUALIFIER void pop() {
        if constexpr (Lock) {
            lock.get().lock();
        }
        header.get().pop(&mem_pool);
        if constexpr (Lock) {
            lock.get().unlock();
        }
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_LIST_H
