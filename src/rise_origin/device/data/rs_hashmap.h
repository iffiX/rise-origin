// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_HASHMAP_H
#define RS_HASHMAP_H

#include <algorithm>
#include <cassert>
#include <execution>
#include <unordered_set>

#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/device/rs_cooperative_group.h"
#include "rs_array.h"
#include "rs_atomic.h"
#include "rs_hash.h"
#include "rs_list.h"

namespace rise {

namespace device {

template <typename Hasher, typename KeyType, typename BucketKeyArray>
class RSCF_HashSetRehashKey
    : public RS_ComputeFunction<RSCF_HashSetRehashKey<Hasher, KeyType, BucketKeyArray>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_HashSetRehashKey>;
    using BaseType::index1D;
    using BucketKeysType = typename BucketKeyArray::ObjectType;
    static constexpr RSize BucketSize = sizeof(BucketKeysType) / sizeof(KeyType);

    HOST_DEVICE_INLINE_QUALIFIER static void
    run(BucketKeyArray in_key_array, BucketKeyArray out_key_array) {
        if (index1D() < BucketSize * in_key_array.size()) {
            RSize bucket_num = in_key_array.size();
            RIndex bucket_index = index1D() / BucketSize;
            RIndex bucket_local_index = index1D() % BucketSize;
            KeyType key = in_key_array[bucket_index].keys[bucket_local_index];
            if (!Hasher::isNullKey(key)) {
                auto hash = Hasher()(key);
                out_key_array[hash & (bucket_num - 1)].keys[bucket_local_index] = key;
            }
        }
    }
};

template <typename Hasher, typename KeyType, typename BucketKeyArray, typename BuckeyValueArray>
class RSCF_HashMapRehashKey
    : public RS_ComputeFunction<
          RSCF_HashMapRehashKey<Hasher, KeyType, BucketKeyArray, BuckeyValueArray>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_HashMapRehashKey>;
    using BaseType::index1D;
    using BucketKeysType = typename BucketKeyArray::ObjectType;
    static constexpr RSize BucketSize = sizeof(BucketKeysType) / sizeof(KeyType);

    HOST_DEVICE_INLINE_QUALIFIER static void run(
        BucketKeyArray in_key_array,
        BuckeyValueArray in_value_array,
        BucketKeyArray out_key_array,
        BuckeyValueArray out_value_array
    ) {
        if (index1D() < BucketSize * in_key_array.size()) {
            RSize bucket_num = in_key_array.size();
            RIndex bucket_index = index1D() / BucketSize;
            RIndex bucket_local_index = index1D() % BucketSize;
            KeyType key = in_key_array[bucket_index].keys[bucket_local_index];
            if (!Hasher::isNullKey(key)) {
                auto hash = Hasher()(key);
                out_key_array[hash & (bucket_num - 1)].keys[bucket_local_index] = key;
                out_value_array[hash & (bucket_num - 1)].values[bucket_local_index] =
                    in_value_array[bucket_index].values[bucket_local_index];
            }
        }
    }
};

template <typename ValueType, typename BucketValueListHeaderArray>
class RSCF_HashMultiMapRebuild
    : public RS_ComputeFunction<RSCF_HashMultiMapRebuild<ValueType, BucketValueListHeaderArray>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_HashMultiMapRebuild>;
    using BaseType::index1D;
    static constexpr RSize BucketSize = sizeof(BucketValueListHeaderArray) / sizeof(ValueType);

    template <typename MemPool>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(BucketValueListHeaderArray value_list_header_array, MemPool mem_pool) {
        if (index1D() < BucketSize * value_list_header_array.size()) {
            RSize bucket_num = value_list_header_array.size();
            RIndex bucket_index = index1D() / BucketSize;
            RIndex bucket_local_index = index1D() % BucketSize;
            value_list_header_array[bucket_index].values[bucket_local_index].rebuild(&mem_pool);
        }
    }
};

/**
 * @brief RS_HashSet is a hash set on the device. The storage is similar to @class RS_HashFlatMap
 * @note: The behavior of mix calling insert() and erase() and contains() in the same kernel with
 * multiple parallel threads is undefined.
 * @tparam KeyType_ The key type of the hash set.
 * @tparam HashKeyType The key type of the hash function.
 * @tparam GenericHasher The hash function to use.
 * @tparam BucketSize_ The number of elements in each bucket.
 */
template <
    typename KeyType_,
    typename HashKeyType = uint32_t,
    template <typename K, typename V, RIndex HashType> typename GenericHasher = RS_GenericHasher,
    RSize BucketSize_ = 16>
class RS_HashSet {
   public:
    using KeyType = KeyType_;
    template <RIndex HashType>
    using Hasher = GenericHasher<KeyType, HashKeyType, HashType>;

    constexpr static RSize BucketSize = BucketSize_;

    static_assert(isPowerOfTwo(BucketSize), "BucketSize must be power of two");

    /*****************************************************************
     * Data
     *****************************************************************/
    struct BucketKeys {
        KeyType keys[BucketSize];
    };

    struct Table {
        RS_Array<BucketKeys, 1> bucket_keys;
        RS_Array<RS_SpinLock, 1> bucket_locks;
    } tables[2];

    RSize bucket_num_per_table = 0;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_HashSet() = default;

    /**
     * @brief Constructs an empty hashset with the given memory provider.
     */
    explicit RS_HashSet(RS_MemoryProvider *provider) { init(provider); }

    /**
     * @brief Constructs an empty hashset with the given maximum entry number and memory provider.
     */
    RS_HashSet(RSize max_entry_num, RS_MemoryProvider *provider) { init(max_entry_num, provider); }

    /**
     * @brief Initializes an empty hash set with the given memory provider.
     */
    void init(RS_MemoryProvider *provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        bucket_num_per_table = 0;
        tables[0].bucket_keys.init(provider);
        tables[0].bucket_locks.init(provider);
        tables[1].bucket_keys.init(provider);
        tables[1].bucket_locks.init(provider);
    }

    /**
     * @brief Initializes a hash set with the given maximum entry number and memory provider.
     */
    void init(RSize max_entry_num, RS_MemoryProvider *provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        bucket_num_per_table =
            nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, BucketSize), BucketSize));
        tables[0].bucket_keys.init(bucket_num_per_table, provider);
        tables[0].bucket_locks.init(bucket_num_per_table, provider);
        tables[1].bucket_keys.init(bucket_num_per_table, provider);
        tables[1].bucket_locks.init(bucket_num_per_table, provider);
        clear();
    }

    /**
     * @brief Frees the GPU memory associated with the hash set.
     * @warning It will free the data, all related hash sets will be affected.
     */
    void free() {
        bucket_num_per_table = 0;
        tables[0].bucket_keys.free();
        tables[0].bucket_locks.free();
        tables[1].bucket_keys.free();
        tables[1].bucket_locks.free();
    }

    /**
     * @brief Resize the hash set to the new size.
     *
     * @param new_max_entry_num The new size of the hash set.
     * @param copy Whether to copy the data to the new hash set.
     */
    void resize(RSize new_max_entry_num, bool copy = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        RSize old_bucket_num = bucket_num_per_table;
        bucket_num_per_table =
            nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(new_max_entry_num / 2, BucketSize), BucketSize));

        tables[0].bucket_keys.resize({bucket_num_per_table}, copy);
        tables[1].bucket_keys.resize({bucket_num_per_table}, copy);
        tables[0].bucket_locks.resize({bucket_num_per_table}, false);
        tables[1].bucket_locks.resize({bucket_num_per_table}, false);

        if (copy) {
            // Clear locks
            tables[0].bucket_locks.data.set(0, tables[0].bucket_locks.data.size());
            tables[1].bucket_locks.data.set(0, tables[1].bucket_locks.data.size());

            // Clear new part if new size is larger than old size
            if (bucket_num_per_table > old_bucket_num) {
                RSize key_clear_size = (bucket_num_per_table - old_bucket_num) * sizeof(BucketKeys);
                RSize key_clear_offset = old_bucket_num * sizeof(BucketKeys);
                tables[0].bucket_keys.data.set(0, key_clear_size, key_clear_offset);
                tables[1].bucket_keys.data.set(0, key_clear_size, key_clear_offset);
            }

            // Rehash to make everything consistent
            castProviderPtrAndCall(provider(), [&](auto p_ptr) { _rehash(p_ptr); });
        } else {
            clear();
        }
    }

    /**
     * @brief Clear entries in the hash set.
     */
    void clear(bool sync = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        // Memset on array memory is faster then call fill() on array
        tables[0].bucket_keys.data.set(0, tables[0].bucket_keys.data.size(), 0, sync);
        tables[0].bucket_locks.data.set(0, tables[0].bucket_locks.data.size(), 0, sync);
        tables[1].bucket_keys.data.set(0, tables[1].bucket_keys.data.size(), 0, sync);
        tables[1].bucket_locks.data.set(0, tables[1].bucket_locks.data.size(), 0, sync);
    }

    /**
     * @brief Copies data from another RS_HashSet. Resize as needed.
     *
     * @param hash_set RS_HashSet to copy data from.
     */
    void copy(const RS_HashSet &hash_set) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(hash_set.provider() != nullptr);

        bucket_num_per_table = hash_set.bucket_num_per_table;
        tables[0].bucket_keys.copy(hash_set.tables[0].bucket_keys);
        tables[0].bucket_locks.copy(hash_set.tables[0].bucket_locks);
        tables[1].bucket_keys.copy(hash_set.tables[1].bucket_keys);
        tables[1].bucket_locks.copy(hash_set.tables[1].bucket_locks);
    }

    /**
     * @brief Moves data to the specified MemoryProvider and returns a new RS_HashSet.
     */
    RS_HashSet move(RS_MemoryProvider *provider) const {
        RS_DATA_EXPECT_INIT(this->provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        RS_HashSet new_set;
        new_set.bucket_num_per_table = bucket_num_per_table;
        new_set.tables[0].bucket_keys = tables[0].bucket_keys.move(provider);
        new_set.tables[0].bucket_locks = tables[0].bucket_locks.move(provider);
        new_set.tables[1].bucket_keys = tables[1].bucket_keys.move(provider);
        new_set.tables[1].bucket_locks = tables[1].bucket_locks.move(provider);
        return new_set;
    }

    [[nodiscard]] RS_MemoryProvider *provider() const { return tables[0].bucket_keys.provider(); }

    [[nodiscard]] RS_DeviceLocation device() const { return tables[0].bucket_keys.device(); }

    HOST_DEVICE_INLINE_QUALIFIER constexpr RSize size() const {
        return bucket_num_per_table * 2 * BucketSize;
    }

    /**
     * @brief Returns the number of unique keys in the hash set.
     */
    HOST_DEVICE_INLINE_QUALIFIER RSize count() const {
        RSize non_null_keys = 0;
        RSize duplicate_keys = 0;
        for (RSize i = 0; i < bucket_num_per_table; i++) {
            for (RSize j = 0; j < BucketSize; j++) {
                KeyType key = tables[0].bucket_keys[i].keys[j];
                if (!Hasher<0>::isNullKey(key)) {
                    non_null_keys++;
                    HashKeyType other_table_hash = Hasher<1>()(key);
                    RSize other_table_bucket_id = other_table_hash & (bucket_num_per_table - 1);
                    for (RSize k = 0; k < BucketSize; k++) {
                        if (tables[1].bucket_keys[other_table_bucket_id].keys[k] == key) {
                            duplicate_keys++;
                            break;
                        }
                    }
                }
            }
        }
        for (RSize i = 0; i < bucket_num_per_table; i++) {
            for (RSize j = 0; j < BucketSize; j++) {
                KeyType key = tables[1].bucket_keys[i].keys[j];
                if (!Hasher<0>::isNullKey(key)) {
                    non_null_keys++;
                }
            }
        }
        return non_null_keys - duplicate_keys;
    }

    HOST_DEVICE_INLINE_QUALIFIER constexpr bool empty() const { return bucket_num_per_table == 0; }

    /// @brief Returns the base size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const { return sizeof(*this); }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const {
        return tables[0].bucket_keys.extraBytes() + tables[0].bucket_locks.extraBytes() +
               tables[1].bucket_keys.extraBytes() + tables[1].bucket_locks.extraBytes();
    }

    /**
     * @brief Inserts a key into the hash set.
     * @param key The key to insert.
     * @return true if the insertion was successful and all entries are placed correctly,
     * false if eviction of an old key happens.
     * @tparam GroupSize The number of threads in the group (warp), if GroupSize is larger than 1,
     * make sure all threads in the group are calling into this function with same arguments.
     * @tparam MaxEvictNum The maximum number of elements to evict.
     */
    template <RSize GroupSize = 1, RSize MaxEvictNum = 16>
    HOST_DEVICE_INLINE_QUALIFIER bool insert(KeyType key) {
        if (bucket_num_per_table < 1) return false;
        static_assert(isPowerOfTwo(GroupSize), "GroupSize must be power of two");
        auto group = getCurrentSynchronizedGroup<RS_MIN(GroupSize, BucketSize)>();
        constexpr RSize ActualGroupSize = RS_MIN(GroupSize, BucketSize);
        constexpr RSize GroupIter = BucketSize / ActualGroupSize;

        group.sync();
        assert(Hasher<0>::isValidKey(key));

        // Start with the initial table
        // Add 1 since key = 0 means the cell in bucket is empty
        key = key + 1;
        uint32_t lane_id = group.thread_rank();

        HashKeyType hash = Hasher<0>()(key);
        int is_second_table = hash & 1;

        for (int i = 0; i < MaxEvictNum; i++) {
            hash = is_second_table ? Hasher<1>()(key) : Hasher<0>()(key);

            RIndex bucket_id = hash & (bucket_num_per_table - 1);
            BucketKeys *bucket_keys = &tables[is_second_table].bucket_keys[bucket_id];
            _lockBucket(is_second_table, bucket_id, group);

            // Check whether key already exists in current table
            // Note: key may exist in either table at the same time
            KeyType probe_keys[GroupIter];
#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                probe_keys[g] = bucket_keys->keys[g * ActualGroupSize + lane_id];
            }

#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                if (int leader = group.elect(probe_keys[g] == key); leader != -1) {
#ifdef RS_DEBUG_HASHMAP
                    if (group.thread_rank() == 0)
                        printf(
                            "Group %d found key %d\n",
                            (int)((blockDim.x * blockIdx.x + threadIdx.x) / GroupSize),
                            (int)(key - 1)
                        );
#endif
                    _unlockBucket(is_second_table, bucket_id, group);
                    return true;
                }
            }

#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                // If key doesn't exist in current table yet, find empty location to insert key
                if (int leader = group.elect(Hasher<0>::isNullKey(probe_keys[g])); leader != -1) {
                    if (lane_id == leader) {
                        RIndex value_local_index = g * ActualGroupSize + lane_id;
                        bucket_keys->keys[value_local_index] = key;
                    }
#ifdef RS_DEBUG_HASHMAP
                    if (group.thread_rank() == 0)
                        printf(
                            "Group %d created key %d\n",
                            (int)((blockDim.x * blockIdx.x + threadIdx.x) / GroupSize),
                            (int)(key - 1)
                        );
#endif
                    _unlockBucket(is_second_table, bucket_id, group);
                    return true;
                }
            }

            // Probe failed, evict a random key to the other table
            // below is equal to modulo GroupSize (also works for GroupSize == 1)
            int evict_local_index = hash & (BucketSize - 1);
            int evict_lane = evict_local_index & (ActualGroupSize - 1);
            int evict_iter = evict_local_index / ActualGroupSize;

            if (lane_id == evict_lane) {
                KeyType evict_key = probe_keys[evict_iter];
#ifdef RS_DEBUG_HASHMAP
                printf(
                    "Thread %d inserted key %d, evicted key %d\n",
                    (int)blockDim.x * blockIdx.x + threadIdx.x,
                    (int)(key - 1),
                    (int)(evict_key - 1)
                );
#endif
                bucket_keys->keys[evict_local_index] = key;

                key = evict_key;
            }
            _unlockBucket(is_second_table, bucket_id, group);
            // Synchronize evicted key within cooperative group
            key = group.shfl(key, evict_lane);
            is_second_table = !is_second_table;
        }
#ifdef RS_DEBUG_HASHMAP
        if (group.thread_rank() == 0)
            printf(
                "Group %d failed to insert %d\n",
                (int)((blockDim.x * blockIdx.x + threadIdx.x) / GroupSize),
                (int)(key - 1)
            );
#endif
        return false;
    }

    /**
     * @brief Erase key from hash set.
     * @param key The key to erase.
     * @return true if key is erased, false otherwise.
     * @tparam GroupSize The number of threads in the group (warp), if GroupSize is larger than 1,
     * make sure all threads in the group are calling into this function with same arguments.
     */
    template <RSize GroupSize = 1>
    HOST_DEVICE_INLINE_QUALIFIER bool erase(KeyType key) {
        if (bucket_num_per_table < 1) return false;
        static_assert(isPowerOfTwo(GroupSize), "GroupSize must be power of two");
        auto group = getCurrentSynchronizedGroup<RS_MIN(GroupSize, BucketSize)>();
        constexpr RSize ActualGroupSize = RS_MIN(GroupSize, BucketSize);
        constexpr RSize GroupIter = BucketSize / ActualGroupSize;

        group.sync();
        assert(Hasher<0>::isValidKey(key));

        // Check the first table
        key = key + 1;
        uint32_t lane_id = group.thread_rank();

        HashKeyType hash = Hasher<0>()(key);
        RIndex bucket_id = hash & (bucket_num_per_table - 1);
        int found_key = 0;
        BucketKeys *bucket_keys = &tables[0].bucket_keys[bucket_id];
        _lockBucket(0, bucket_id, group);
#pragma unroll
        for (RSize i = 0; i < GroupIter; i++) {
            if constexpr (ActualGroupSize == 1) {
                if (bucket_keys->keys[i] == key) {
                    found_key = 1;
                    Hasher<0>::setNullKey(bucket_keys->keys[i]);
                }
            } else {
                unsigned int offset = i * ActualGroupSize;
                if (int leader = group.elect(bucket_keys->keys[offset + lane_id] == key);
                    leader != -1) {
                    found_key = 1;
                    if (lane_id == leader)
                        Hasher<0>::setNullKey(bucket_keys->keys[offset + leader]);
                }
            }
        }
        _unlockBucket(0, bucket_id, group);
        // Check the second table
        hash = Hasher<1>()(key);
        bucket_id = hash & (bucket_num_per_table - 1);
        bucket_keys = &tables[1].bucket_keys[bucket_id];

        _lockBucket(1, bucket_id, group);
#pragma unroll
        for (RSize i = 0; i < GroupIter; i++) {
            if constexpr (ActualGroupSize == 1) {
                if (bucket_keys->keys[i] == key) {
                    found_key = 1;
                    Hasher<0>::setNullKey(bucket_keys->keys[i]);
                }
            } else {
                unsigned int offset = i * ActualGroupSize;
                if (int leader = group.elect(bucket_keys->keys[offset + lane_id] == key);
                    leader != -1) {
                    found_key = 1;
                    if (lane_id == leader)
                        Hasher<0>::setNullKey(bucket_keys->keys[offset + leader]);
                }
            }
        }
        _unlockBucket(1, bucket_id, group);
        if constexpr (ActualGroupSize == 1) {
            return found_key;
        } else {
            return group.any(found_key);
        }
    }

    /**
     * @brief Returns if a key is contained in hash set.
     * @param key The key to find.
     * @return True if contains, false otherwise.
     * @tparam GroupSize The number of threads in the group (warp), if GroupSize is larger than 1,
     * make sure all threads in the group are calling into this function with same arguments.
     */
    template <RSize GroupSize = 1>
    HOST_DEVICE_INLINE_QUALIFIER bool contains(KeyType key) {
        if (bucket_num_per_table < 1) return false;
        static_assert(isPowerOfTwo(GroupSize), "GroupSize must be power of two");
        auto group = getCurrentSynchronizedGroup<RS_MIN(GroupSize, BucketSize)>();
        constexpr RSize ActualGroupSize = RS_MIN(GroupSize, BucketSize);
        constexpr RSize GroupIter = BucketSize / ActualGroupSize;

        group.sync();
        assert(Hasher<0>::isValidKey(key));

        // Check the first table
        // Add 1 since key = 0 means the cell in bucket is empty
        key = key + 1;
        uint32_t lane_id = group.thread_rank();

        HashKeyType hash = Hasher<0>()(key);
        RIndex bucket_id = hash & (bucket_num_per_table - 1);
        BucketKeys *bucket_keys = &tables[0].bucket_keys[bucket_id];

#pragma unroll
        for (RSize i = 0; i < GroupIter; i++) {
            if constexpr (ActualGroupSize == 1) {
                if (bucket_keys->keys[i] == key) {
                    return true;
                }
            } else {
                unsigned int offset = i * ActualGroupSize;
                if (int leader = group.elect(bucket_keys->keys[offset + lane_id] == key);
                    leader != -1) {
                    return true;
                }
            }
        }

        // Check the second table
        hash = Hasher<1>()(key);
        bucket_id = hash & (bucket_num_per_table - 1);
        bucket_keys = &tables[1].bucket_keys[bucket_id];

#pragma unroll
        for (RSize i = 0; i < GroupIter; i++) {
            if constexpr (ActualGroupSize == 1) {
                if (bucket_keys->keys[i] == key) {
                    return true;
                }
            } else {
                unsigned int offset = i * ActualGroupSize;
                if (int leader = group.elect(bucket_keys->keys[offset + lane_id] == key);
                    leader != -1) {
                    return true;
                }
            }
        }
        return false;
    }

   protected:
    /// @brief Rehash keys in the hashmap.
    template <typename MemoryProvider>
    void _rehash(MemoryProvider *mem_provider) {
        if (bucket_num_per_table < 1) return;
        auto tmp_bucket_keys = RS_Array<BucketKeys, 1>({bucket_num_per_table}, mem_provider);
        tmp_bucket_keys.copy(tables[0].bucket_keys);
        mem_provider
            ->template run<RSCF_HashSetRehashKey<Hasher<0>, KeyType, RS_Array<BucketKeys, 1>>>(
                {bucket_num_per_table * BucketSize}, tmp_bucket_keys, tables[0].bucket_keys
            );
        tmp_bucket_keys.copy(tables[1].bucket_keys);
        mem_provider
            ->template run<RSCF_HashSetRehashKey<Hasher<1>, KeyType, RS_Array<BucketKeys, 1>>>(
                {bucket_num_per_table * BucketSize}, tmp_bucket_keys, tables[1].bucket_keys
            );
        tmp_bucket_keys.free();
    }

    template <typename SynchronizedGroup>
    HOST_DEVICE_INLINE_QUALIFIER void
    _lockBucket(bool is_second_table, uint32_t bucket_id, SynchronizedGroup &group) {
        if (group.thread_rank() == 0) {
#ifdef RS_DEBUG_HASHMAP
            printf(
                "Table %d Bucket %d begin locking by thread %d\n",
                (int)is_second_table,
                (int)bucket_id,
                (int)blockDim.x * blockIdx.x + threadIdx.x
            );
#endif
            tables[is_second_table].bucket_locks[bucket_id].lock();
#ifdef RS_DEBUG_HASHMAP
            printf(
                "Table %d Bucket %d locked by thread %d\n",
                (int)is_second_table,
                (int)bucket_id,
                (int)blockDim.x * blockIdx.x + threadIdx.x
            );
#endif
        }
        // Important to sync after lock, make sure all group are ready to perform actions
        group.sync();
    }

    template <typename SynchronizedGroup>
    HOST_DEVICE_INLINE_QUALIFIER void
    _unlockBucket(uint32_t is_second_table, uint32_t bucket_id, SynchronizedGroup &group) {
        // Important to sync before unlock, make sure all writes by the group are finished
        group.sync();
        if (group.thread_rank() == 0) {
            tables[is_second_table].bucket_locks[bucket_id].unlock();
#ifdef RS_DEBUG_HASHMAP
            printf(
                "Table %d Bucket %d unlocked by thread %d\n",
                (int)is_second_table,
                (int)bucket_id,
                (int)blockDim.x * blockIdx.x + threadIdx.x
            );
#endif
        }
    }
};

/**
 * @brief RS_HashFlatMap is a flat hash map on the device. The value is stored in the hash table
 * instead of being stored in a memory pool. Ideal for small values less than or equal to pointer
 * size.
 * @note: The behavior of mix calling insert() and erase() and find() in the same kernel with
 * multiple parallel threads is undefined.
 * @note: The eventual value returned value of find() after calling insert() using the same key
 * in multiple threads is unordered. However, one value is guaranteed to be returned if at least
 * one insert is successful.
 * @tparam KeyType_ The key type of the hash map.
 * @tparam ValueType_ The value type of the hash map.
 * @tparam HashKeyType The key type of the hash function.
 * @tparam GenericHasher The hash function to use.
 * @tparam BucketSize_ The number of elements in each bucket.
 */
template <
    typename KeyType_,
    typename ValueType_,
    typename HashKeyType = uint32_t,
    template <typename K, typename V, RIndex HashType> typename GenericHasher = RS_GenericHasher,
    RSize BucketSize_ = 16>
class RS_HashFlatMap {
   public:
    using KeyType = KeyType_;
    using ValueType = ValueType_;
    template <RIndex HashType>
    using Hasher = GenericHasher<KeyType, HashKeyType, HashType>;

    constexpr static RSize BucketSize = BucketSize_;

    static_assert(isPowerOfTwo(BucketSize), "BucketSize must be power of two");

    /*****************************************************************
     * Data
     *****************************************************************/
    struct BucketKeys {
        KeyType keys[BucketSize];
    };

    struct BucketValues {
        ValueType values[BucketSize];
    };

    struct Table {
        RS_Array<BucketKeys, 1> bucket_keys;
        RS_Array<BucketValues, 1> bucket_values;
        RS_Array<RS_SpinLock, 1> bucket_locks;
    } tables[2];

    RSize bucket_num_per_table = 0;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_HashFlatMap() = default;

    /**
     * @brief Construct an empty hash map with the given provider.
     */
    explicit RS_HashFlatMap(RS_MemoryProvider *provider) { init(provider); }

    /**
     * @brief Construct a hash map with the given maximum entry number and memory provider.
     */
    RS_HashFlatMap(RSize max_entry_num, RS_MemoryProvider *provider) {
        init(max_entry_num, provider);
    }

    /**
     * @brief Initializes an empty hash map with the given memory provider.
     */
    void init(RS_MemoryProvider *provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);
        bucket_num_per_table = 0;
        tables[0].bucket_keys.init(provider);
        tables[0].bucket_values.init(provider);
        tables[0].bucket_locks.init(provider);
        tables[1].bucket_keys.init(provider);
        tables[1].bucket_values.init(provider);
        tables[1].bucket_locks.init(provider);
    }

    /**
     * @brief Initializes the hash map with the given maximum entry number and memory provider.
     */
    void init(RSize max_entry_num, RS_MemoryProvider *provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        bucket_num_per_table =
            nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, BucketSize), BucketSize));
        tables[0].bucket_keys.init(bucket_num_per_table, provider);
        tables[0].bucket_values.init(bucket_num_per_table, provider);
        tables[0].bucket_locks.init(bucket_num_per_table, provider);
        tables[1].bucket_keys.init(bucket_num_per_table, provider);
        tables[1].bucket_values.init(bucket_num_per_table, provider);
        tables[1].bucket_locks.init(bucket_num_per_table, provider);
        clear();
    }

    /**
     * @brief Frees the GPU memory associated with the hash map.
     * @warning It will free the data, all related hash maps will be affected.
     */
    void free() {
        bucket_num_per_table = 0;
        tables[0].bucket_keys.free();
        tables[0].bucket_values.free();
        tables[0].bucket_locks.free();
        tables[1].bucket_keys.free();
        tables[1].bucket_values.free();
        tables[1].bucket_locks.free();
    }

    /**
     * @brief Resize the hash map to the new size.
     *
     * @param new_max_entry_num The new size of the hash map.
     * @param copy Whether to copy the data to the new hash map.
     */
    void resize(RSize new_max_entry_num, bool copy = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        RSize old_bucket_num = bucket_num_per_table;
        bucket_num_per_table =
            nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(new_max_entry_num / 2, BucketSize), BucketSize));

        tables[0].bucket_keys.resize({bucket_num_per_table}, copy);
        tables[0].bucket_values.resize({bucket_num_per_table}, copy);
        tables[1].bucket_keys.resize({bucket_num_per_table}, copy);
        tables[1].bucket_values.resize({bucket_num_per_table}, copy);
        tables[0].bucket_locks.resize({bucket_num_per_table}, false);
        tables[1].bucket_locks.resize({bucket_num_per_table}, false);

        if (copy) {
            // Clear locks
            tables[0].bucket_locks.data.set(0, tables[0].bucket_locks.data.size());
            tables[1].bucket_locks.data.set(0, tables[1].bucket_locks.data.size());

            // Clear new part if new size is larger than old size
            if (bucket_num_per_table > old_bucket_num) {
                RSize key_clear_size = (bucket_num_per_table - old_bucket_num) * sizeof(BucketKeys);
                RSize key_clear_offset = old_bucket_num * sizeof(BucketKeys);
                RSize value_clear_size =
                    (bucket_num_per_table - old_bucket_num) * sizeof(BucketValues);
                RSize value_clear_offset = old_bucket_num * sizeof(BucketValues);
                tables[0].bucket_keys.data.set(0, key_clear_size, key_clear_offset);
                tables[0].bucket_values.data.set(0, value_clear_size, value_clear_offset);
                tables[1].bucket_keys.data.set(0, key_clear_size, key_clear_offset);
                tables[1].bucket_values.data.set(0, value_clear_size, value_clear_offset);
            }
            // Rehash to make everything consistent
            castProviderPtrAndCall(provider(), [&](auto p_ptr) { _rehash(p_ptr); });
        } else {
            clear();
        }
    }

    /**
     * @brief Clear entries in the hash map.
     */
    void clear(bool sync = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        tables[0].bucket_keys.data.set(0, tables[0].bucket_keys.data.size(), 0, sync);
        tables[0].bucket_values.data.set(0, tables[0].bucket_values.data.size(), 0, sync);
        tables[0].bucket_locks.data.set(0, tables[0].bucket_locks.data.size(), 0, sync);
        tables[1].bucket_keys.data.set(0, tables[1].bucket_keys.data.size(), 0, sync);
        tables[1].bucket_values.data.set(0, tables[1].bucket_values.data.size(), 0, sync);
        tables[1].bucket_locks.data.set(0, tables[1].bucket_locks.data.size(), 0, sync);
    }

    /**
     * @brief Copies data from another RS_HashFlatMap. Resize as needed.
     *
     * @param hash_flat_map RS_HashFlatMap to copy data from.
     */
    void copy(const RS_HashFlatMap &hash_flat_map) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(hash_flat_map.provider() != nullptr);

        bucket_num_per_table = hash_flat_map.bucket_num_per_table;
        tables[0].bucket_keys.copy(hash_flat_map.tables[0].bucket_keys);
        tables[0].bucket_values.copy(hash_flat_map.tables[0].bucket_values);
        tables[0].bucket_locks.copy(hash_flat_map.tables[0].bucket_locks);
        tables[1].bucket_keys.copy(hash_flat_map.tables[1].bucket_keys);
        tables[1].bucket_values.copy(hash_flat_map.tables[1].bucket_values);
        tables[1].bucket_locks.copy(hash_flat_map.tables[1].bucket_locks);
    }

    /**
     * @brief Moves data to the specified MemoryProvider and returns a new RS_HashFlatMap.
     */
    RS_HashFlatMap move(RS_MemoryProvider *provider) const {
        RS_DATA_EXPECT_INIT(this->provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        RS_HashFlatMap new_map;
        new_map.bucket_num_per_table = bucket_num_per_table;
        new_map.tables[0].bucket_keys = tables[0].bucket_keys.move(provider);
        new_map.tables[0].bucket_values = tables[0].bucket_values.move(provider);
        new_map.tables[0].bucket_locks = tables[0].bucket_locks.move(provider);
        new_map.tables[1].bucket_keys = tables[1].bucket_keys.move(provider);
        new_map.tables[1].bucket_values = tables[1].bucket_values.move(provider);
        new_map.tables[1].bucket_locks = tables[1].bucket_locks.move(provider);
        return new_map;
    }

    [[nodiscard]] RS_MemoryProvider *provider() const { return tables[0].bucket_keys.provider(); }

    [[nodiscard]] RS_DeviceLocation device() const { return tables[0].bucket_keys.device(); }

    HOST_DEVICE_INLINE_QUALIFIER constexpr RSize size() const {
        return bucket_num_per_table * 2 * BucketSize;
    }

    /**
     * @brief Returns the number of unique keys in the hash map.
     */
    HOST_DEVICE_INLINE_QUALIFIER RSize count() const {
        RSize non_null_keys = 0;
        RSize duplicate_keys = 0;
        for (RSize i = 0; i < bucket_num_per_table; i++) {
            for (RSize j = 0; j < BucketSize; j++) {
                KeyType key = tables[0].bucket_keys[i].keys[j];
                if (!Hasher<0>::isNullKey(key)) {
                    non_null_keys++;
                    HashKeyType other_table_hash = Hasher<1>()(key);
                    RSize other_table_bucket_id = other_table_hash & (bucket_num_per_table - 1);
                    for (RSize k = 0; k < BucketSize; k++) {
                        if (tables[1].bucket_keys[other_table_bucket_id].keys[k] == key) {
                            duplicate_keys++;
                            break;
                        }
                    }
                }
            }
        }
        for (RSize i = 0; i < bucket_num_per_table; i++) {
            for (RSize j = 0; j < BucketSize; j++) {
                KeyType key = tables[1].bucket_keys[i].keys[j];
                if (!Hasher<0>::isNullKey(key)) {
                    non_null_keys++;
                }
            }
        }
        return non_null_keys - duplicate_keys;
    }

    HOST_DEVICE_INLINE_QUALIFIER constexpr bool empty() const { return bucket_num_per_table == 0; }

    /// @brief Returns the base size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const { return sizeof(*this); }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const {
        return tables[0].bucket_keys.extraBytes() + tables[0].bucket_values.extraBytes() +
               tables[0].bucket_locks.extraBytes() + tables[1].bucket_keys.extraBytes() +
               tables[1].bucket_values.extraBytes() + tables[1].bucket_locks.extraBytes();
    }

    /**
     * @brief Inserts a key-value pair into the hash map.
     * @param key The key to insert.
     * @param value The value to insert.
     * @return true if the insertion was successful and all entries are placed correctly,
     * false if eviction of an old key happens.
     * @tparam GroupSize The number of threads in the group (warp), if GroupSize is larger than 1,
     * make sure all threads in the group are calling into this function with same arguments.
     * @tparam MaxEvictNum The maximum number of elements to evict.
     */
    template <RSize GroupSize = 1, RSize MaxEvictNum = 16>
    HOST_DEVICE_INLINE_QUALIFIER bool insert(KeyType key, ValueType value) {
        if (bucket_num_per_table < 1) return false;
        static_assert(isPowerOfTwo(GroupSize), "GroupSize must be power of two");
        auto group = getCurrentSynchronizedGroup<RS_MIN(GroupSize, BucketSize)>();
        constexpr RSize ActualGroupSize = RS_MIN(GroupSize, BucketSize);
        constexpr RSize GroupIter = BucketSize / ActualGroupSize;

        group.sync();
        assert(Hasher<0>::isValidKey(key));

        // Start with the initial table
        // Add 1 since key = 0 means the cell in bucket is empty
        key = key + 1;
        uint32_t lane_id = group.thread_rank();

        HashKeyType hash = Hasher<0>()(key);
        int is_second_table = hash & 1;

        for (int i = 0; i < MaxEvictNum; i++) {
            hash = is_second_table ? Hasher<1>()(key) : Hasher<0>()(key);

            RIndex bucket_id = hash & (bucket_num_per_table - 1);
            BucketKeys *bucket_keys = &tables[is_second_table].bucket_keys[bucket_id];
            BucketValues *bucket_values = &tables[is_second_table].bucket_values[bucket_id];
            _lockBucket(is_second_table, bucket_id, group);

            // Check whether key already exists in current table
            // Note: key may exist in either table at the same time
            KeyType probe_keys[GroupIter];
#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                probe_keys[g] = bucket_keys->keys[g * ActualGroupSize + lane_id];
            }

#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                if (int leader = group.elect(probe_keys[g] == key); leader != -1) {
                    if (lane_id == leader) {
                        RIndex value_local_index = g * ActualGroupSize + lane_id;
                        bucket_values->values[value_local_index] = value;
                    }
#ifdef RS_DEBUG_HASHMAP
                    if (group.thread_rank() == 0)
                        printf(
                            "Group %d insert to existing key %d\n",
                            (int)((blockDim.x * blockIdx.x + threadIdx.x) / GroupSize),
                            (int)(key - 1)
                        );
#endif
                    _unlockBucket(is_second_table, bucket_id, group);
                    return true;
                }
            }

#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                // If key doesn't exist in current table yet, find empty location to insert key
                if (int leader = group.elect(Hasher<0>::isNullKey(probe_keys[g])); leader != -1) {
                    if (lane_id == leader) {
                        RIndex value_local_index = g * ActualGroupSize + lane_id;
                        bucket_keys->keys[value_local_index] = key;
                        bucket_values->values[value_local_index] = value;
                    }
#ifdef RS_DEBUG_HASHMAP
                    if (group.thread_rank() == 0)
                        printf(
                            "Group %d created key %d\n",
                            (int)((blockDim.x * blockIdx.x + threadIdx.x) / GroupSize),
                            (int)(key - 1)
                        );
#endif
                    _unlockBucket(is_second_table, bucket_id, group);
                    return true;
                }
            }

            // Probe failed, evict a random key to the other table
            // below is equal to modulo GroupSize (also works for GroupSize == 1)
            int evict_local_index = hash & (BucketSize - 1);
            int evict_lane = evict_local_index & (ActualGroupSize - 1);
            int evict_iter = evict_local_index / ActualGroupSize;

            if (lane_id == evict_lane) {
                KeyType evict_key = probe_keys[evict_iter];
                ValueType evict_value = bucket_values->values[evict_local_index];
#ifdef RS_DEBUG_HASHMAP
                printf(
                    "Thread %d inserted key %d, evicted key %d\n",
                    (int)blockDim.x * blockIdx.x + threadIdx.x,
                    (int)(key - 1),
                    (int)(evict_key - 1)
                );
#endif
                bucket_keys->keys[evict_local_index] = key;
                bucket_values->values[evict_local_index] = value;

                key = evict_key;
                value = evict_value;
            }
            _unlockBucket(is_second_table, bucket_id, group);
            // Synchronize evicted key within cooperative group
            key = group.shfl(key, evict_lane);
            value = group.shfl(value, evict_lane);
            is_second_table = !is_second_table;
        }
#ifdef RS_DEBUG_HASHMAP
        if (group.thread_rank() == 0)
            printf(
                "Group %d failed to insert %d\n",
                (int)((blockDim.x * blockIdx.x + threadIdx.x) / GroupSize),
                (int)(key - 1)
            );
#endif
        return false;
    }

    /**
     * @brief Erase key from hash map.
     * @param key The key to erase.
     * @return true if key is erased, false otherwise.
     * @tparam GroupSize The number of threads in the group (warp), if GroupSize is larger than 1,
     * make sure all threads in the group are calling into this function with same arguments.
     */
    template <RSize GroupSize = 1>
    HOST_DEVICE_INLINE_QUALIFIER bool erase(KeyType key) {
        if (bucket_num_per_table < 1) return false;
        static_assert(isPowerOfTwo(GroupSize), "GroupSize must be power of two");
        auto group = getCurrentSynchronizedGroup<RS_MIN(GroupSize, BucketSize)>();
        constexpr RSize ActualGroupSize = RS_MIN(GroupSize, BucketSize);
        constexpr RSize GroupIter = BucketSize / ActualGroupSize;

        group.sync();
        assert(Hasher<0>::isValidKey(key));

        // Check the first table
        key = key + 1;
        uint32_t lane_id = group.thread_rank();

        HashKeyType hash = Hasher<0>()(key);
        RIndex bucket_id = hash & (bucket_num_per_table - 1);
        int found_key = 0;
        BucketKeys *bucket_keys = &tables[0].bucket_keys[bucket_id];
        _lockBucket(0, bucket_id, group);
#pragma unroll
        for (RSize i = 0; i < GroupIter; i++) {
            if constexpr (ActualGroupSize == 1) {
                if (bucket_keys->keys[i] == key) {
                    found_key = 1;
                    Hasher<0>::setNullKey(bucket_keys->keys[i]);
                }
            } else {
                unsigned int offset = i * ActualGroupSize;
                if (int leader = group.elect(bucket_keys->keys[offset + lane_id] == key);
                    leader != -1) {
                    found_key = 1;
                    if (lane_id == leader)
                        Hasher<0>::setNullKey(bucket_keys->keys[offset + leader]);
                }
            }
        }
        _unlockBucket(0, bucket_id, group);
        // Check the second table
        hash = Hasher<1>()(key);
        bucket_id = hash & (bucket_num_per_table - 1);
        bucket_keys = &tables[1].bucket_keys[bucket_id];

        _lockBucket(1, bucket_id, group);
#pragma unroll
        for (RSize i = 0; i < GroupIter; i++) {
            if constexpr (ActualGroupSize == 1) {
                if (bucket_keys->keys[i] == key) {
                    found_key = 1;
                    Hasher<0>::setNullKey(bucket_keys->keys[i]);
                }
            } else {
                unsigned int offset = i * ActualGroupSize;
                if (int leader = group.elect(bucket_keys->keys[offset + lane_id] == key);
                    leader != -1) {
                    found_key = 1;
                    if (lane_id == leader)
                        Hasher<0>::setNullKey(bucket_keys->keys[offset + leader]);
                }
            }
        }
        _unlockBucket(1, bucket_id, group);
        if constexpr (ActualGroupSize == 1) {
            return found_key;
        } else {
            return group.any(found_key);
        }
    }

    /**
     * @brief Find key in hash map.
     * @param key The key to find.
     * @return The pointer to the value of the key if found, nullptr otherwise.
     * @tparam GroupSize The number of threads in the group (warp), if GroupSize is larger than 1,
     * make sure all threads in the group are calling into this function with same arguments.
     */
    template <RSize GroupSize = 1>
    HOST_DEVICE_INLINE_QUALIFIER ValueType *find(KeyType key) {
        if (bucket_num_per_table < 1) return nullptr;
        static_assert(isPowerOfTwo(GroupSize), "GroupSize must be power of two");
        auto group = getCurrentSynchronizedGroup<RS_MIN(GroupSize, BucketSize)>();
        constexpr RSize ActualGroupSize = RS_MIN(GroupSize, BucketSize);
        constexpr RSize GroupIter = BucketSize / ActualGroupSize;

        group.sync();
        assert(Hasher<0>::isValidKey(key));

        // Check the first table
        // Add 1 since key = 0 means the cell in bucket is empty
        key = key + 1;
        uint32_t lane_id = group.thread_rank();

        HashKeyType hash = Hasher<0>()(key);
        RIndex bucket_id = hash & (bucket_num_per_table - 1);
        BucketKeys *bucket_keys = &tables[0].bucket_keys[bucket_id];

#pragma unroll
        for (RSize i = 0; i < GroupIter; i++) {
            if constexpr (ActualGroupSize == 1) {
                if (bucket_keys->keys[i] == key) {
                    return &tables[0].bucket_values[bucket_id].values[i];
                }
            } else {
                unsigned int offset = i * ActualGroupSize;
                if (int leader = group.elect(bucket_keys->keys[offset + lane_id] == key);
                    leader != -1) {
                    ValueType *result = nullptr;
                    if (lane_id == leader) {
                        result = &tables[0].bucket_values[bucket_id].values[offset + leader];
                    }
                    result = group.shfl(result, leader);
                    return result;
                }
            }
        }

        // Check the second table
        hash = Hasher<1>()(key);
        bucket_id = hash & (bucket_num_per_table - 1);
        bucket_keys = &tables[1].bucket_keys[bucket_id];

#pragma unroll
        for (RSize i = 0; i < GroupIter; i++) {
            if constexpr (ActualGroupSize == 1) {
                if (bucket_keys->keys[i] == key) {
                    return &tables[1].bucket_values[bucket_id].values[i];
                }
            } else {
                unsigned int offset = i * ActualGroupSize;
                if (int leader = group.elect(bucket_keys->keys[offset + lane_id] == key);
                    leader != -1) {
                    ValueType *result = nullptr;
                    if (lane_id == leader) {
                        result = &tables[1].bucket_values[bucket_id].values[offset + leader];
                    }
                    result = group.shfl(result, leader);
                    return result;
                }
            }
        }
        return nullptr;
    }

   protected:
    /// @brief Rehash keys in the hashmap.
    template <typename MemoryProvider>
    void _rehash(MemoryProvider *mem_provider) {
        if (bucket_num_per_table < 1) return;
        auto tmp_bucket_keys = RS_Array<BucketKeys, 1>({bucket_num_per_table}, mem_provider);
        auto tmp_bucket_values = RS_Array<BucketValues, 1>({bucket_num_per_table}, mem_provider);
        tmp_bucket_keys.copy(tables[0].bucket_keys);
        tmp_bucket_values.copy(tables[0].bucket_values);
        mem_provider->template run<RSCF_HashMapRehashKey<
            Hasher<0>,
            KeyType,
            RS_Array<BucketKeys, 1>,
            RS_Array<BucketValues, 1>>>(
            {bucket_num_per_table * BucketSize},
            tmp_bucket_keys,
            tmp_bucket_values,
            tables[0].bucket_keys,
            tables[0].bucket_values
        );
        tmp_bucket_keys.copy(tables[1].bucket_keys);
        tmp_bucket_values.copy(tables[1].bucket_values);
        mem_provider->template run<RSCF_HashMapRehashKey<
            Hasher<1>,
            KeyType,
            RS_Array<BucketKeys, 1>,
            RS_Array<BucketValues, 1>>>(
            {bucket_num_per_table * BucketSize},
            tmp_bucket_keys,
            tmp_bucket_values,
            tables[1].bucket_keys,
            tables[1].bucket_values
        );
        tmp_bucket_keys.free();
        tmp_bucket_values.free();
    }

    template <typename SynchronizedGroup>
    HOST_DEVICE_INLINE_QUALIFIER void
    _lockBucket(bool is_second_table, uint32_t bucket_id, SynchronizedGroup &group) {
        if (group.thread_rank() == 0) {
#ifdef RS_DEBUG_HASHMAP
            printf(
                "Table %d Bucket %d begin locking by thread %d\n",
                (int)is_second_table,
                (int)bucket_id,
                (int)blockDim.x * blockIdx.x + threadIdx.x
            );
#endif
            tables[is_second_table].bucket_locks[bucket_id].lock();
#ifdef RS_DEBUG_HASHMAP
            printf(
                "Table %d Bucket %d locked by thread %d\n",
                (int)is_second_table,
                (int)bucket_id,
                (int)blockDim.x * blockIdx.x + threadIdx.x
            );
#endif
        }
        // Important to sync after lock, make sure all group are ready to perform actions
        group.sync();
    }

    template <typename SynchronizedGroup>
    HOST_DEVICE_INLINE_QUALIFIER void
    _unlockBucket(uint32_t is_second_table, uint32_t bucket_id, SynchronizedGroup &group) {
        // Important to sync before unlock, make sure all writes by the group are finished
        group.sync();
        if (group.thread_rank() == 0) {
            tables[is_second_table].bucket_locks[bucket_id].unlock();
#ifdef RS_DEBUG_HASHMAP
            printf(
                "Table %d Bucket %d unlocked by thread %d\n",
                (int)is_second_table,
                (int)bucket_id,
                (int)blockDim.x * blockIdx.x + threadIdx.x
            );
#endif
        }
    }
};

/**
 * @brief RS_HashStableFlatMap is a stable flat hash map on the device. The insertion strategy is
 * based on the simple power of two strategy described in https://arxiv.org/pdf/2108.07232v3, once
 * the key is successfully inserted, the pointer of the value is guaranteed to be stable.
 * Additionally, this map also guarantees that there is always only one key in both tables through
 * ordering of the table insertion for each key.
 * @note: The behavior of mix calling insert() and erase() or erase() and find() in the same kernel
 * with multiple parallel threads is undefined. However, since the hash map is stable, therefore
 * first find() to check if a key is inserted, then insert() if it is not found is safe, and it
 * will avoid the cost of locking in insert().
 * @note: The eventual value returned value of find() after calling insert() using the same key
 * in multiple threads is unordered. However, one value is guaranteed to be returned if at least
 * one insert is successful.
 * @tparam KeyType_ The key type of the hash map.
 * @tparam ValueType_ The value type of the hash map.
 * @tparam HashKeyType The key type of the hash function.
 * @tparam GenericHasher The hash function to use.
 * @tparam BucketSize_ The number of elements in each bucket.
 */
template <
    typename KeyType_,
    typename ValueType_,
    typename HashKeyType = uint32_t,
    template <typename K, typename V, RIndex HashType> typename GenericHasher = RS_GenericHasher,
    RSize BucketSize_ = 16>
class RS_HashStableFlatMap
    : public RS_HashFlatMap<KeyType_, ValueType_, HashKeyType, GenericHasher, BucketSize_> {
   public:
    using KeyType = KeyType_;
    using ValueType = ValueType_;
    using BaseMapType =
        RS_HashFlatMap<KeyType_, ValueType_, HashKeyType, GenericHasher, BucketSize_>;
    template <RIndex HashType>
    using Hasher = GenericHasher<KeyType, HashKeyType, HashType>;
    using BucketKeys = typename BaseMapType::BucketKeys;
    using BucketValues = typename BaseMapType::BucketValues;

    constexpr static RSize BucketSize = BucketSize_;

    /*****************************************************************
     * Data
     *****************************************************************/
    using BaseMapType::bucket_num_per_table;
    using BaseMapType::tables;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_HashStableFlatMap() = default;

    /**
     * @brief Constructs an empty stable flat hash map with the given provider.
     */
    explicit RS_HashStableFlatMap(RS_MemoryProvider *provider) : BaseMapType(provider) {}

    /**
     * @brief Constructs a stable flat hash map with the given maximum entry number and
     * memory provider.
     */
    RS_HashStableFlatMap(RSize max_entry_num, RS_MemoryProvider *provider)
        : BaseMapType(max_entry_num, provider) {}

    // init(), free(), resize(), clear(), copy(), move(), size(), count(), empty()
    // baseBytes(), extraBytes(), erase(), find() are inherited

    /**
     * @brief Inserts a key-value pair into the hash map.
     * @param key The key to insert.
     * @param value The value to insert.
     * @return True if the insertion was successful. False otherwise.
     * @tparam GroupSize The number of threads in the group (warp), if GroupSize is larger than 1,
     * make sure all threads in the group are calling into this function with same arguments.
     */
    template <RSize GroupSize = 1>
    HOST_DEVICE_INLINE_QUALIFIER bool insert(KeyType key, ValueType value) {
        bool exists;
        return insertUnique<GroupSize>(key, value, exists, true) != nullptr;
    }

    /**
     * @brief Inserts a key-value pair into the hash map, if the key already exists, overwrite/not
     * overwrite it depending on user input. If not overwrite, the pointer returned will be point to
     * the existing value. Otherwise, the old value will be overwritten and the pointer returned
     * will be pointed to the newly inserted value.
     * @param key The key to insert.
     * @param value The value to insert.
     * @param exists Returns whether the key already exists.
     * @param overwrite Whether to overwrite the value or not.
     * @return A pointer to the newly inserted / existing value if the insertion was successful.
     * Nullptr if pointer doesn't exist and insertion failed.
     * @tparam GroupSize The number of threads in the group (warp), if GroupSize is larger than 1,
     * make sure all threads in the group are calling into this function with same arguments.
     */
    template <RSize GroupSize = 1>
    HOST_DEVICE_INLINE_QUALIFIER ValueType *
    insertUnique(KeyType key, ValueType value, bool &exists, bool overwrite = false) {
        if (bucket_num_per_table < 1) return nullptr;
        static_assert(isPowerOfTwo(GroupSize), "GroupSize must be power of two");
        auto group = getCurrentSynchronizedGroup<RS_MIN(GroupSize, BucketSize)>();
        constexpr RSize ActualGroupSize = RS_MIN(GroupSize, BucketSize);
        constexpr RSize GroupIter = BucketSize / ActualGroupSize;

        group.sync();
        assert(Hasher<0>::isValidKey(key));

        // Start with the initial table
        // Add 1 since key = 0 means the cell in bucket is empty
        key = key + 1;
        uint32_t lane_id = group.thread_rank();
        HashKeyType hash = Hasher<0>()(key);
        int is_second_table = hash & 1;
        exists = false;

        for (int i = 0; i < 2; ++i) {
            hash = is_second_table ? Hasher<1>()(key) : Hasher<0>()(key);
            RIndex bucket_id = hash & (bucket_num_per_table - 1);
            BucketKeys *bucket_keys = &tables[is_second_table].bucket_keys[bucket_id];
            BucketValues *bucket_values = &tables[is_second_table].bucket_values[bucket_id];
            _lockBucket(is_second_table, bucket_id, group);

            // Check whether key already exists in current table
            KeyType probe_keys[GroupIter];
#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                probe_keys[g] = bucket_keys->keys[g * ActualGroupSize + lane_id];
            }

#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                if (int leader = group.elect(probe_keys[g] == key); leader != -1) {
                    RIndex value_local_index = g * ActualGroupSize + leader;
                    if (overwrite && lane_id == leader) {
                        bucket_values->values[value_local_index] = value;
                    }
#ifdef RS_DEBUG_HASHMAP
                    if (group.thread_rank() == 0)
                        printf(
                            "Group %d insert to existing key %d\n",
                            (int)((blockDim.x * blockIdx.x + threadIdx.x) / GroupSize),
                            (int)(key - 1)
                        );
#endif
                    _unlockBucket(is_second_table, bucket_id, group);
                    exists = true;
                    return &bucket_values->values[value_local_index];
                }
            }

#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                // If key doesn't exist in current table yet, find empty location to insert key
                if (int leader = group.elect(Hasher<0>::isNullKey(probe_keys[g])); leader != -1) {
                    RIndex value_local_index = g * ActualGroupSize + leader;
                    if (lane_id == leader) {
                        bucket_keys->keys[value_local_index] = key;
                        bucket_values->values[value_local_index] = value;
                    }
#ifdef RS_DEBUG_HASHMAP
                    if (group.thread_rank() == 0)
                        printf(
                            "Group %d created key %d\n",
                            (int)((blockDim.x * blockIdx.x + threadIdx.x) / GroupSize),
                            (int)(key - 1)
                        );
#endif
                    _unlockBucket(is_second_table, bucket_id, group);
                    return &bucket_values->values[value_local_index];
                }
            }
            _unlockBucket(is_second_table, bucket_id, group);
            is_second_table = !is_second_table;
        }
        return nullptr;
    }

   protected:
    using BaseMapType::_lockBucket;
    using BaseMapType::_unlockBucket;
};

/**
 * @brief RS_HashMap is a hash map implemented on RS_HashFlatMap. The value is stored in a
 * dedicated array, and the base flat map stores index of the value as its value. This
 * implementation reduces the cost of eviction in insert() compared to move the value
 * around directly.
 * @note: The behavior of mix calling insert() and erase() and find() in the same kernel with
 * multiple parallel threads is undefined.
 * @note: The eventual value returned value of find() after calling insert() using the same key
 * in multiple threads is unordered. However, one value is guaranteed to be returned if at least
 * one insert is successful.
 */
template <
    typename KeyType_,
    typename ValueType_,
    typename HashKeyType = uint32_t,
    template <typename K, typename V, RIndex HashType> typename GenericHasher = RS_GenericHasher,
    RSize BucketSize_ = 16>
class RS_HashMap
    : public RS_HashFlatMap<KeyType_, RIndex, HashKeyType, GenericHasher, BucketSize_> {
   public:
    using KeyType = KeyType_;
    using ValueType = ValueType_;
    using BaseMapType = RS_HashFlatMap<KeyType_, RIndex, HashKeyType, GenericHasher, BucketSize_>;
    template <RIndex HashType>
    using Hasher = GenericHasher<KeyType, HashKeyType, HashType>;
    using BucketKeys = typename BaseMapType::BucketKeys;
    using BucketValues = typename BaseMapType::BucketValues;

    constexpr static RSize BucketSize = BucketSize_;

    /*****************************************************************
     * Data
     *****************************************************************/
    using BaseMapType::bucket_num_per_table;
    using BaseMapType::tables;

    // The tables and bucket_num_per_table are in the base class
    RS_Array<ValueType_, 1> values;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_HashMap() = default;

    /**
     * @brief Constructs an empty hash map with the given provider.
     */
    explicit RS_HashMap(RS_MemoryProvider *provider) : BaseMapType(provider), values(0, provider) {}

    /**
     * @brief Constructs a hash map with the given maximum entry number and provider.
     */
    RS_HashMap(RSize max_entry_num, RS_MemoryProvider *provider)
        : BaseMapType(max_entry_num, provider),
          values(bucket_num_per_table * 2 * BucketSize, provider) {}

    /**
     * @brief Initializes an empty hash map with the given provider.
     */
    void init(RS_MemoryProvider *provider) {
        BaseMapType::init(provider);
        values.init(0, provider);
    }

    /**
     * @brief Initialized the hash map with the given maximum entry number and provider.
     */
    void init(RSize max_entry_num, RS_MemoryProvider *provider) {
        BaseMapType::init(max_entry_num, provider);
        values.init(bucket_num_per_table * 2 * BucketSize, provider);
    }

    void free() {
        BaseMapType::free();
        values.free();
    }

    void resize(RSize new_max_entry_num, bool copy = true) {
        RS_DATA_EXPECT_INIT(this->provider() != nullptr);

        BaseMapType::resize(new_max_entry_num, copy);
        values.resize({bucket_num_per_table * 2 * BucketSize}, copy);
    }

    void clear(bool sync = true) {
        RS_DATA_EXPECT_INIT(this->provider() != nullptr);

        BaseMapType::clear(sync);
    }

    void copy(const RS_HashMap &hash_map) {
        RS_DATA_EXPECT_INIT(this->provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(hash_map.provider() != nullptr);

        BaseMapType::copy(hash_map);
        values.copy(hash_map.values);
    }

    RS_HashMap move(RS_MemoryProvider *provider) const {
        RS_DATA_EXPECT_INIT(this->provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        RS_HashMap new_map;
        auto tmp_map = BaseMapType::move(provider);
        new_map.bucket_num_per_table = tmp_map.bucket_num_per_table;
        new_map.tables[0].bucket_keys = tmp_map.tables[0].bucket_keys;
        new_map.tables[0].bucket_values = tmp_map.tables[0].bucket_values;
        new_map.tables[0].bucket_locks = tmp_map.tables[0].bucket_locks;
        new_map.tables[1].bucket_keys = tmp_map.tables[1].bucket_keys;
        new_map.tables[1].bucket_values = tmp_map.tables[1].bucket_values;
        new_map.tables[1].bucket_locks = tmp_map.tables[1].bucket_locks;
        new_map.values = values.move(provider);
        return new_map;
    }

    // size() inherited

    // count() inherited

    // empty() inherited

    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const {
        return BaseMapType::baseBytes() + values.baseBytes();
    }

    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const {
        return BaseMapType::extraBytes() + values.extraBytes();
    }

    /**
     * @brief Insert key-value pair into hash map.
     * @param key The key to insert.
     * @param value The value to insert.
     * @return true if the insertion was successful and all entries are placed correctly,
     * false if all keys in start bucket are being relocated and cannot insert, or eviction of an
     * old key happens.
     * @tparam GroupSize The number of threads in the group (warp), if GroupSize is larger than 1,
     * make sure all threads in the group are calling into this function with same arguments.
     * @tparam MaxEvictNum The maximum number of elements to evict.
     */
    template <RSize GroupSize = 1, RSize MaxEvictNum = 16>
    HOST_DEVICE_INLINE_QUALIFIER bool insert(KeyType key, const ValueType &value) {
        if (bucket_num_per_table < 1) return false;
        static_assert(isPowerOfTwo(GroupSize), "GroupSize must be power of two");
        auto group = getCurrentSynchronizedGroup<RS_MIN(GroupSize, BucketSize)>();
        constexpr RSize ActualGroupSize = RS_MIN(GroupSize, BucketSize);
        constexpr RSize GroupIter = BucketSize / ActualGroupSize;

        group.sync();
        assert(Hasher<0>::isValidKey(key));

        // Start with the initial table
        // Add 1 since key = 0 means the cell in bucket is empty
        key = key + 1;
        uint32_t lane_id = group.thread_rank();

        HashKeyType hash = Hasher<0>()(key);
        int is_second_table = hash & 1;

        hash = is_second_table ? Hasher<1>()(key) : Hasher<0>()(key);

        RIndex bucket_id = hash & (bucket_num_per_table - 1);
        BucketKeys *bucket_keys = &tables[is_second_table].bucket_keys[bucket_id];
        BucketValues *bucket_values = &tables[is_second_table].bucket_values[bucket_id];
        _lockBucket(is_second_table, bucket_id, group);

        KeyType probe_keys[GroupIter];
#pragma unroll
        for (RSize g = 0; g < GroupIter; g++) {
            probe_keys[g] = bucket_keys->keys[g * ActualGroupSize + lane_id];
        }

#pragma unroll
        for (RSize g = 0; g < GroupIter; g++) {
            if (int leader = group.elect(probe_keys[g] == key); leader != -1) {
                if (lane_id == leader) {
                    RIndex value_local_index = g * ActualGroupSize + lane_id;
                    RIndex value_index =
                        (is_second_table * bucket_num_per_table + bucket_id) * BucketSize +
                        value_local_index;
                    bucket_values->values[value_local_index] = value_index;
                    values[value_index] = value;
                }
                _unlockBucket(is_second_table, bucket_id, group);
                return true;
            }
        }

#pragma unroll
        for (RSize g = 0; g < GroupIter; g++) {
            // If key doesn't exist in current table yet, find empty location to insert key
            if (int leader = group.elect(Hasher<0>::isNullKey(probe_keys[g])); leader != -1) {
                if (lane_id == leader) {
                    RIndex value_local_index = g * ActualGroupSize + lane_id;
                    RIndex value_index =
                        (is_second_table * bucket_num_per_table + bucket_id) * BucketSize +
                        value_local_index;
                    bucket_keys->keys[value_local_index] = key;
                    bucket_values->values[value_local_index] = value_index;
                    values[value_index] = value;
                }
                _unlockBucket(is_second_table, bucket_id, group);
                return true;
            }
        }

        // Now we have to evict key and make space for new key
        int evict_local_index = -1, evict_iter, evict_lane = -1;
        for (evict_iter = 0; evict_iter < GroupIter; evict_iter++) {
            int idx = evict_iter * ActualGroupSize + lane_id;
            if (evict_lane = group.random_elect(
                    bucket_values->values[idx] != RS_NULL_INDEX, static_cast<int>(hash)
                );
                evict_lane != -1) {
                evict_local_index = idx;
                break;
            }
        }
        if (evict_local_index == -1) {
            // All values in current bucket are being relocated, fail
            // Very unlikely to happen
            _unlockBucket(is_second_table, bucket_id, group);
            return false;
        }

        RIndex *start_value_index = nullptr;
        RIndex start_bucket_id = bucket_id;
        int start_is_second_table = is_second_table;

        KeyType evict_key;
        RIndex evict_value_index = RS_NULL_INDEX;
        RIndex write_value_index = RS_NULL_INDEX;

        int write_leader;

        if (lane_id == evict_lane) {
            evict_key = probe_keys[evict_iter];
            evict_value_index = bucket_values->values[evict_local_index];

            // Write new key, and reserve value index
            bucket_keys->keys[evict_local_index] = key;
            start_value_index = &bucket_values->values[evict_local_index];
            *start_value_index = RS_NULL_INDEX;
        }
        // Unlock bucket to prevent deadlock
        _unlockBucket(is_second_table, bucket_id, group);

        // Synchronize evicted key within cooperative group
        start_value_index = group.shfl(start_value_index, evict_lane);
        evict_key = group.shfl(evict_key, evict_lane);
        evict_value_index = group.shfl(evict_value_index, evict_lane);
        is_second_table = !is_second_table;

        for (int i = 0; i < MaxEvictNum; i++) {
            hash = is_second_table ? Hasher<1>()(evict_key) : Hasher<0>()(evict_key);

            bucket_id = hash & (bucket_num_per_table - 1);
            bucket_keys = &tables[is_second_table].bucket_keys[bucket_id];
            bucket_values = &tables[is_second_table].bucket_values[bucket_id];
            _lockBucket(is_second_table, bucket_id, group);

#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                probe_keys[g] = bucket_keys->keys[g * ActualGroupSize + lane_id];
            }

#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                if (write_leader = group.elect(probe_keys[g] == evict_key); write_leader != -1) {
                    if (lane_id == write_leader) {
                        RIndex value_local_index = g * ActualGroupSize + lane_id;
                        write_value_index = bucket_values->values[value_local_index];
                        bucket_values->values[value_local_index] = evict_value_index;
                    }
                    _unlockBucket(is_second_table, bucket_id, group);
                    goto insert_all_success;
                }
            }

#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                // If key doesn't exist in current table yet, find empty location to insert key
                if (write_leader = group.elect(Hasher<0>::isNullKey(probe_keys[g]));
                    write_leader != -1) {
                    if (lane_id == write_leader) {
                        RIndex value_local_index = g * ActualGroupSize + lane_id;
                        write_value_index =
                            (is_second_table * bucket_num_per_table + bucket_id) * BucketSize +
                            value_local_index;
                        bucket_keys->keys[value_local_index] = evict_key;
                        bucket_values->values[value_local_index] = evict_value_index;
                    }
                    _unlockBucket(is_second_table, bucket_id, group);
                    goto insert_all_success;
                }
            }

            // Probe failed, evict a random key to the other table
            for (evict_iter = 0; evict_iter < GroupIter; evict_iter++) {
                int idx = evict_iter * ActualGroupSize + lane_id;
                if (evict_lane = group.random_elect(
                        bucket_values->values[idx] != RS_NULL_INDEX, static_cast<int>(hash)
                    );
                    evict_lane != -1) {
                    evict_local_index = idx;
                    break;
                }
            }
            if (evict_local_index == -1) {
                // All values in current bucket are being relocated, fail
                // Very unlikely to happen
                _unlockBucket(is_second_table, bucket_id, group);
                goto insert_all_failed;
            }

            if (lane_id == evict_lane) {
                KeyType new_evict_key = probe_keys[evict_iter];
                RIndex new_evict_value_index = bucket_values->values[evict_local_index];
                bucket_keys->keys[evict_local_index] = evict_key;
                bucket_values->values[evict_local_index] = evict_value_index;

                evict_key = new_evict_key;
                evict_value_index = new_evict_value_index;
            }
            _unlockBucket(is_second_table, bucket_id, group);
            // Synchronize evicted key within cooperative group
            evict_key = group.shfl(evict_key, evict_lane);
            evict_value_index = group.shfl(evict_value_index, evict_lane);
            is_second_table = !is_second_table;
        }

    insert_all_failed:
        _lockBucket(start_is_second_table, start_bucket_id, group);
        if (lane_id == 0) {
            // Overwrite the last evicted value
            *start_value_index = evict_value_index;
            values[evict_value_index] = value;
        }
        _unlockBucket(start_is_second_table, start_bucket_id, group);
        return false;

    insert_all_success:
        _lockBucket(start_is_second_table, start_bucket_id, group);
        if (lane_id == write_leader) {
            *start_value_index = write_value_index;
            values[write_value_index] = value;
        }
        _unlockBucket(start_is_second_table, start_bucket_id, group);
        return true;
    }

    // erase() inherited

    template <RSize GroupSize = 1>
    HOST_DEVICE_INLINE_QUALIFIER ValueType *find(KeyType key) {
        if (bucket_num_per_table < 1) return nullptr;
        RIndex *value_index = BaseMapType::template find<GroupSize>(key);
        return value_index == nullptr ? nullptr : &values[*value_index];
    }

   protected:
    using BaseMapType::_lockBucket;
    using BaseMapType::_rehash;
    using BaseMapType::_unlockBucket;
};

/**
 * @brief RS_HashMultiMap is a hash map implemented like RS_HashFlatMap. The values are stored in
 * a @class RS_ListHeader.
 * @note: The behavior of mix calling insert() and erase() and find() in the same kernel with
 * multiple parallel threads is undefined.
 * @note: The eventual value returned value of find() after calling insert() using the same key
 * in multiple threads is unordered. However, one value is guaranteed to be returned if at least
 * one insert is successful.
 */
template <
    typename KeyType_,
    typename ValueType_,
    typename HashKeyType = uint32_t,
    template <typename K, typename V, RIndex HashType> typename GenericHasher = RS_GenericHasher,
    template <RSize> typename MemPoolType = RS_MemPool,
    RSize BucketSize_ = 16,
    RSize ListChunkSize_ = 32>
class RS_HashMultiMap {
   public:
    using KeyType = KeyType_;
    using ValueType = ValueType_;
    using ListHeaderType = RS_ListHeader<ValueType, ListChunkSize_, MemPoolType>;
    using ListChunkType = typename ListHeaderType::ChunkType;
    using ListViewType = RS_ListView<ListHeaderType, 2>;
    template <RIndex HashType>
    using Hasher = GenericHasher<KeyType, HashKeyType, HashType>;

    constexpr static RSize BucketSize = BucketSize_;
    constexpr static RSize ListChunkSize = ListChunkSize_;

    /*****************************************************************
     * Data
     *****************************************************************/
    struct BucketKeys {
        KeyType keys[BucketSize];
    };

    struct BucketValues {
        ListHeaderType values[BucketSize];
    };

    struct Table {
        RS_Array<BucketKeys, 1> bucket_keys;
        RS_Array<BucketValues, 1> bucket_values;
        RS_Array<RS_SpinLock, 1> bucket_locks;
    } tables[2];
    RSize bucket_num_per_table = 0;

    // The tables and bucket_num_per_table are in the base class
    MemPoolType<sizeof(ListChunkType)> value_chunk_pool;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_HashMultiMap() = default;

    /**
     * @brief Constructs an empty hash multi map with the given memory provider.
     */
    explicit RS_HashMultiMap(RS_MemoryProvider *provider) { init(provider); }

    /**
     * @brief Constructs a hash multi map with the given max_entry_num, initial_pool_entry_num and
     * memory provider.
     */
    RS_HashMultiMap(
        RSize max_entry_num,
        RSize initial_pool_entry_num,
        RS_MemoryProvider *provider
    ) {
        init(max_entry_num, initial_pool_entry_num, provider);
    }

    void init(RS_MemoryProvider *provider) {
        bucket_num_per_table = 0;
        tables[0].bucket_keys.init(provider);
        tables[0].bucket_values.init(provider);
        tables[0].bucket_locks.init(provider);
        tables[1].bucket_keys.init(provider);
        tables[1].bucket_values.init(provider);
        tables[1].bucket_locks.init(provider);
        value_chunk_pool.init(provider);
    }

    void init(RSize max_entry_num, RSize initial_pool_entry_num, RS_MemoryProvider *provider) {
        bucket_num_per_table =
            nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(max_entry_num / 2, BucketSize), BucketSize));
        tables[0].bucket_keys.init(bucket_num_per_table, provider);
        tables[0].bucket_values.init(bucket_num_per_table, provider);
        tables[0].bucket_locks.init(bucket_num_per_table, provider);
        tables[1].bucket_keys.init(bucket_num_per_table, provider);
        tables[1].bucket_values.init(bucket_num_per_table, provider);
        tables[1].bucket_locks.init(bucket_num_per_table, provider);
        value_chunk_pool.init(initial_pool_entry_num, provider);
        clear();
    }

    void free() {
        bucket_num_per_table = 0;
        tables[0].bucket_keys.free();
        tables[0].bucket_values.free();
        tables[0].bucket_locks.free();
        tables[1].bucket_keys.free();
        tables[1].bucket_values.free();
        tables[1].bucket_locks.free();
        value_chunk_pool.free();
    }

    void resize(RSize new_max_entry_num, RSize grow_or_resize_pool_entry_num, bool copy = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        RSize old_bucket_num = bucket_num_per_table;
        bucket_num_per_table =
            nextPowerOfTwo(RS_CEIL_DIV(RS_MAX(new_max_entry_num / 2, BucketSize), BucketSize));

        tables[0].bucket_keys.resize({bucket_num_per_table}, copy);
        tables[0].bucket_values.resize({bucket_num_per_table}, copy);
        tables[1].bucket_keys.resize({bucket_num_per_table}, copy);
        tables[1].bucket_values.resize({bucket_num_per_table}, copy);
        tables[0].bucket_locks.resize({bucket_num_per_table}, false);
        tables[1].bucket_locks.resize({bucket_num_per_table}, false);

        if (copy) {
            // Clear locks
            tables[0].bucket_locks.data.set(0, tables[0].bucket_locks.data.size());
            tables[1].bucket_locks.data.set(0, tables[1].bucket_locks.data.size());

            // Clear new part if new size is larger than old size
            if (bucket_num_per_table > old_bucket_num) {
                auto init_new_values = new BucketValues[bucket_num_per_table - old_bucket_num];
                for (RSize i = old_bucket_num; i < bucket_num_per_table; i++) {
                    for (RSize j = 0; j < BucketSize; j++) {
                        init_new_values[i - old_bucket_num].values[j] = {};
                    }
                }
                RSize key_clear_size = (bucket_num_per_table - old_bucket_num) * sizeof(BucketKeys);
                RSize key_clear_offset = old_bucket_num * sizeof(BucketKeys);
                tables[0].bucket_keys.data.set(0, key_clear_size, key_clear_offset);
                tables[0].bucket_values.writeSlice(
                    init_new_values, bucket_num_per_table - old_bucket_num, old_bucket_num
                );
                tables[1].bucket_keys.data.set(0, key_clear_size, key_clear_offset);
                tables[1].bucket_values.writeSlice(
                    init_new_values, bucket_num_per_table - old_bucket_num, old_bucket_num
                );
                delete[] init_new_values;
            }

            // Rehash to make everything consistent
            castProviderPtrAndCall(provider(), [&](auto p_ptr) { _rehash(p_ptr); });
        } else {
            clear();
        }

        if constexpr (std::is_same_v<
                          MemPoolType<sizeof(ListChunkType)>,
                          RS_MemPool<sizeof(ListChunkType)>>) {
            value_chunk_pool.grow(grow_or_resize_pool_entry_num);
        } else {
            value_chunk_pool.resize(grow_or_resize_pool_entry_num, copy);
        }
    }

    /**
     * @brief Clear entries in the hash map.
     */
    void clear(bool sync = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        BucketValues value_lists;
        for (RSize i = 0; i < BucketSize; i++) {
            value_lists.values[i] = {};
        }
        tables[0].bucket_keys.data.set(0, tables[0].bucket_keys.data.size(), 0, sync);
        tables[0].bucket_values.fill(value_lists, sync);
        tables[0].bucket_locks.data.set(0, tables[0].bucket_locks.data.size(), 0, sync);
        tables[1].bucket_keys.data.set(0, tables[1].bucket_keys.data.size(), 0, sync);
        tables[1].bucket_values.fill(value_lists, sync);
        tables[1].bucket_locks.data.set(0, tables[1].bucket_locks.data.size(), 0, sync);

        value_chunk_pool.clear(sync);
    }

    void copy(const RS_HashMultiMap &hash_multi_map) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(hash_multi_map.provider() != nullptr);

        bucket_num_per_table = hash_multi_map.bucket_num_per_table;
        tables[0].bucket_keys.copy(hash_multi_map.tables[0].bucket_keys);
        tables[0].bucket_values.copy(hash_multi_map.tables[0].bucket_values);
        tables[0].bucket_locks.copy(hash_multi_map.tables[0].bucket_locks);
        tables[1].bucket_keys.copy(hash_multi_map.tables[1].bucket_keys);
        tables[1].bucket_values.copy(hash_multi_map.tables[1].bucket_values);
        tables[1].bucket_locks.copy(hash_multi_map.tables[1].bucket_locks);
        value_chunk_pool.copy(hash_multi_map.value_chunk_pool);

        // Rebuild all lists
        castProviderPtrAndCall(provider(), [&](auto p_ptr) { _rebuild(p_ptr); });
    }

    RS_HashMultiMap move(RS_MemoryProvider *provider) const {
        RS_DATA_EXPECT_INIT(this->provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        RS_HashMultiMap new_map;
        new_map.bucket_num_per_table = bucket_num_per_table;
        new_map.tables[0].bucket_keys = tables[0].bucket_keys.move(provider);
        new_map.tables[0].bucket_values = tables[0].bucket_values.move(provider);
        new_map.tables[0].bucket_locks = tables[0].bucket_locks.move(provider);
        new_map.tables[1].bucket_keys = tables[1].bucket_keys.move(provider);
        new_map.tables[1].bucket_values = tables[1].bucket_values.move(provider);
        new_map.tables[1].bucket_locks = tables[1].bucket_locks.move(provider);
        new_map.value_chunk_pool = value_chunk_pool.move(provider);

        castProviderPtrAndCall(provider, [&](auto p_ptr) { new_map._rebuild(p_ptr); });
        return new_map;
    }

    [[nodiscard]] RS_MemoryProvider *provider() const { return tables[0].bucket_keys.provider(); }

    [[nodiscard]] RS_DeviceLocation device() const { return tables[0].bucket_keys.device(); }

    HOST_DEVICE_INLINE_QUALIFIER constexpr RSize size() const {
        return bucket_num_per_table * 2 * BucketSize;
    }

    /**
     * @brief Returns the number of unique keys in the hash map.
     */
    HOST_DEVICE_INLINE_QUALIFIER RSize count() const {
        RSize non_null_keys = 0;
        RSize duplicate_keys = 0;
        for (RSize i = 0; i < bucket_num_per_table; i++) {
            for (RSize j = 0; j < BucketSize; j++) {
                KeyType key = tables[0].bucket_keys[i].keys[j];
                if (!Hasher<0>::isNullKey(key)) {
                    non_null_keys++;
                    HashKeyType other_table_hash = Hasher<1>()(key);
                    RSize other_table_bucket_id = other_table_hash & (bucket_num_per_table - 1);
                    for (RSize k = 0; k < BucketSize; k++) {
                        if (tables[1].bucket_keys[other_table_bucket_id].keys[k] == key) {
                            duplicate_keys++;
                            break;
                        }
                    }
                }
            }
        }
        for (RSize i = 0; i < bucket_num_per_table; i++) {
            for (RSize j = 0; j < BucketSize; j++) {
                KeyType key = tables[1].bucket_keys[i].keys[j];
                if (!Hasher<0>::isNullKey(key)) {
                    non_null_keys++;
                }
            }
        }
        return non_null_keys - duplicate_keys;
    }

    HOST_DEVICE_INLINE_QUALIFIER constexpr bool empty() const { return bucket_num_per_table == 0; }

    /// @brief Returns the base size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const {
        return sizeof(bucket_num_per_table) + tables[0].bucket_keys.data.baseBytes() +  //
               tables[0].bucket_values.data.baseBytes() +                               //
               tables[0].bucket_locks.data.baseBytes() +                                //
               tables[1].bucket_keys.data.baseBytes() +                                 //
               tables[1].bucket_values.data.baseBytes() +                               //
               tables[1].bucket_locks.data.baseBytes() + value_chunk_pool.baseBytes();
    }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const {
        return tables[0].bucket_keys.extraBytes() + tables[0].bucket_values.extraBytes() +
               tables[0].bucket_locks.extraBytes() + tables[1].bucket_keys.extraBytes() +
               tables[1].bucket_values.extraBytes() + tables[1].bucket_locks.extraBytes() +
               value_chunk_pool.extraBytes();
    }

    /**
     * @brief Insert key-value pair into hash map.
     * @param key The key to insert.
     * @param value The value to insert.
     * @return true if the insertion was successful and all entries are placed correctly,
     * false if all keys in start bucket are being relocated and cannot insert, or eviction of an
     * old key happens.
     * @tparam GroupSize The number of threads in the group (warp), if GroupSize is larger than 1,
     * make sure all threads in the group are calling into this function with same arguments.
     * @tparam MaxEvictNum The maximum number of elements to evict.
     */
    template <RSize GroupSize = 1, RSize MaxEvictNum = 16>
    HOST_DEVICE_INLINE_QUALIFIER bool insert(KeyType key, const ValueType &value) {
        if (bucket_num_per_table < 1) return false;
        static_assert(isPowerOfTwo(GroupSize), "GroupSize must be power of two");
        auto group = getCurrentSynchronizedGroup<RS_MIN(GroupSize, BucketSize)>();
        constexpr RSize ActualGroupSize = RS_MIN(GroupSize, BucketSize);
        constexpr RSize GroupIter = BucketSize / ActualGroupSize;

        group.sync();
        assert(Hasher<0>::isValidKey(key));

        bool append_success = false;

        // Start with the initial table
        // Add 1 since key = 0 means the cell in bucket is empty
        key = key + 1;
        uint32_t lane_id = group.thread_rank();

        ListHeaderType value_list;
        HashKeyType hash = Hasher<0>()(key);
        int is_second_table = hash & 1;

        for (int i = 0; i < MaxEvictNum; i++) {
            hash = is_second_table ? Hasher<1>()(key) : Hasher<0>()(key);

            RIndex bucket_id = hash & (bucket_num_per_table - 1);
            BucketKeys *bucket_keys = &tables[is_second_table].bucket_keys[bucket_id];
            BucketValues *bucket_values = &tables[is_second_table].bucket_values[bucket_id];
            _lockBucket(is_second_table, bucket_id, group);

            // Check whether key already exists in current table
            // Note: key may exist in either table at the same time
            KeyType probe_keys[GroupIter];
#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                probe_keys[g] = bucket_keys->keys[g * ActualGroupSize + lane_id];
            }

#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                if (int leader = group.elect(probe_keys[g] == key); leader != -1) {
                    if (lane_id == leader) {
                        RIndex value_local_index = g * ActualGroupSize + lane_id;
                        if (i == 0) {
                            auto &existing_value_list = bucket_values->values[value_local_index];
                            append_success = existing_value_list.append(
                                &value_chunk_pool, value, static_cast<RIndex>(hash)
                            );
                        } else
                            bucket_values->values[value_local_index] = value_list;
                    }
#ifdef RS_DEBUG_HASHMAP
                    if (group.thread_rank() == 0)
                        printf(
                            "Group %d insert to existing key %d\n",
                            (int)((blockDim.x * blockIdx.x + threadIdx.x) / GroupSize),
                            (int)(key - 1)
                        );
#endif
                    append_success = group.shfl(static_cast<int>(append_success), leader);
                    _unlockBucket(is_second_table, bucket_id, group);
                    return append_success;
                }
            }

#pragma unroll
            for (RSize g = 0; g < GroupIter; g++) {
                // If key doesn't exist in current table yet, find empty location to insert key
                if (int leader = group.elect(Hasher<0>::isNullKey(probe_keys[g])); leader != -1) {
                    if (lane_id == leader) {
                        if (i == 0)
                            append_success = value_list.append(
                                &value_chunk_pool, value, static_cast<RIndex>(hash)
                            );
                        RIndex value_local_index = g * ActualGroupSize + lane_id;
                        bucket_keys->keys[value_local_index] = key;
                        bucket_values->values[value_local_index] = value_list;
                    }
#ifdef RS_DEBUG_HASHMAP
                    if (group.thread_rank() == 0)
                        printf(
                            "Group %d created key %d\n",
                            (int)((blockDim.x * blockIdx.x + threadIdx.x) / GroupSize),
                            (int)(key - 1)
                        );
#endif
                    append_success = group.shfl(static_cast<int>(append_success), leader);
                    _unlockBucket(is_second_table, bucket_id, group);
                    return append_success;
                }
            }

            // Probe failed, evict a random key to the other table
            // below is equal to modulo GroupSize (also works for GroupSize == 1)
            int evict_local_index = hash & (BucketSize - 1);
            int evict_lane = evict_local_index & (ActualGroupSize - 1);
            int evict_iter = evict_local_index / ActualGroupSize;

            if (lane_id == evict_lane) {
                if (i == 0)
                    append_success =
                        value_list.append(&value_chunk_pool, value, static_cast<RIndex>(hash));
                KeyType evict_key = probe_keys[evict_iter];
                ListHeaderType evict_value_list = bucket_values->values[evict_local_index];
#ifdef RS_DEBUG_HASHMAP
                printf(
                    "Thread %d inserted key %d, evicted key %d\n",
                    (int)blockDim.x * blockIdx.x + threadIdx.x,
                    (int)(key - 1),
                    (int)(evict_key - 1)
                );
#endif
                bucket_keys->keys[evict_local_index] = key;
                bucket_values->values[evict_local_index] = value_list;

                key = evict_key;
                value_list = evict_value_list;
            }
            append_success = group.shfl(static_cast<int>(append_success), evict_lane);
            _unlockBucket(is_second_table, bucket_id, group);
            // Synchronize evicted key within cooperative group
            key = group.shfl(key, evict_lane);
            value_list.elem_num = group.shfl(value_list.elem_num, evict_lane);
            value_list.head_index = group.shfl(value_list.head_index, evict_lane);
            value_list.head = group.shfl(value_list.head, evict_lane);
            value_list.tail = group.shfl(value_list.tail, evict_lane);

            is_second_table = !is_second_table;
        }
#ifdef RS_DEBUG_HASHMAP
        if (group.thread_rank() == 0)
            printf(
                "Group %d failed to insert %d\n",
                (int)((blockDim.x * blockIdx.x + threadIdx.x) / GroupSize),
                (int)(key - 1)
            );
#endif
        // Definitely failed even if append is successful since we have to evict a key.
        value_list.clear(&value_chunk_pool);
        return false;
    }

    template <RSize GroupSize = 1>
    HOST_DEVICE_INLINE_QUALIFIER bool erase(KeyType key) {
        if (bucket_num_per_table < 1) return false;
        static_assert(isPowerOfTwo(GroupSize), "GroupSize must be power of two");
        auto group = getCurrentSynchronizedGroup<RS_MIN(GroupSize, BucketSize)>();
        constexpr RSize ActualGroupSize = RS_MIN(GroupSize, BucketSize);
        constexpr RSize GroupIter = BucketSize / ActualGroupSize;

        group.sync();
        assert(Hasher<0>::isValidKey(key));

        // Check the first table
        key = key + 1;
        uint32_t lane_id = group.thread_rank();

        HashKeyType hash = Hasher<0>()(key);
        RIndex bucket_id = hash & (bucket_num_per_table - 1);
        int found_key = 0;
        BucketKeys *bucket_keys = &tables[0].bucket_keys[bucket_id];
        BucketValues *bucket_values = &tables[0].bucket_values[bucket_id];
        _lockBucket(0, bucket_id, group);
#pragma unroll
        for (RSize i = 0; i < GroupIter; i++) {
            if constexpr (ActualGroupSize == 1) {
                if (bucket_keys->keys[i] == key) {
                    found_key = 1;
                    Hasher<0>::setNullKey(bucket_keys->keys[i]);
                    auto &value_list = bucket_values->values[i];
                    value_list.clear();
                }
            } else {
                unsigned int offset = i * ActualGroupSize;
                if (int leader = group.elect(bucket_keys->keys[offset + lane_id] == key);
                    leader != -1) {
                    found_key = 1;
                    if (lane_id == leader) {
                        Hasher<0>::setNullKey(bucket_keys->keys[offset + leader]);
                        auto &value_list = bucket_values->values[offset + leader];
                        value_list.clear();
                    }
                }
            }
        }
        _unlockBucket(0, bucket_id, group);
        // Check the second table
        hash = Hasher<1>()(key);
        bucket_id = hash & (bucket_num_per_table - 1);
        bucket_keys = &tables[1].bucket_keys[bucket_id];

        _lockBucket(1, bucket_id, group);
#pragma unroll
        for (RSize i = 0; i < GroupIter; i++) {
            if constexpr (ActualGroupSize == 1) {
                if (bucket_keys->keys[i] == key) {
                    found_key = 1;
                    Hasher<0>::setNullKey(bucket_keys->keys[i]);
                    auto &value_list = bucket_values->values[i];
                    value_list.clear(&value_chunk_pool);
                }
            } else {
                unsigned int offset = i * ActualGroupSize;
                if (int leader = group.elect(bucket_keys->keys[offset + lane_id] == key);
                    leader != -1) {
                    found_key = 1;
                    if (lane_id == leader) {
                        Hasher<0>::setNullKey(bucket_keys->keys[offset + leader]);
                        auto &value_list = bucket_values->values[offset + leader];
                        value_list.clear(&value_chunk_pool);
                    }
                }
            }
        }
        _unlockBucket(1, bucket_id, group);
        if constexpr (ActualGroupSize == 1) {
            return found_key;
        } else {
            return group.any(found_key);
        }
    }

    /**
     * @brief Find key in hash map.
     * @param key The key to find.
     * @return A list view containing values, if there aren't any values, the list view is empty.
     * @tparam GroupSize The number of threads in the group (warp), if GroupSize is larger than 1,
     * make sure all threads in the group are calling into this function with same arguments.
     */
    template <RSize GroupSize = 1>
    HOST_DEVICE_INLINE_QUALIFIER ListViewType find(KeyType key) {
        if (bucket_num_per_table < 1) return {nullptr, nullptr};
        static_assert(isPowerOfTwo(GroupSize), "GroupSize must be power of two");
        auto group = getCurrentSynchronizedGroup<RS_MIN(GroupSize, BucketSize)>();
        constexpr RSize ActualGroupSize = RS_MIN(GroupSize, BucketSize);
        constexpr RSize GroupIter = BucketSize / ActualGroupSize;

        group.sync();
        assert(Hasher<0>::isValidKey(key));

        ListHeaderType *first_list = nullptr, *second_list = nullptr;

        // Check the first table
        // Add 1 since key = 0 means the cell in bucket is empty
        key = key + 1;
        uint32_t lane_id = group.thread_rank();

        HashKeyType hash = Hasher<0>()(key);
        RIndex bucket_id = hash & (bucket_num_per_table - 1);
        BucketKeys *bucket_keys = &tables[0].bucket_keys[bucket_id];

#pragma unroll
        for (RSize i = 0; i < GroupIter; i++) {
            if constexpr (ActualGroupSize == 1) {
                if (bucket_keys->keys[i] == key) {
                    first_list = &tables[0].bucket_values[bucket_id].values[i];
                    break;
                }
            } else {
                unsigned int offset = i * ActualGroupSize;
                if (int leader = group.elect(bucket_keys->keys[offset + lane_id] == key);
                    leader != -1) {
                    if (lane_id == leader) {
                        first_list = &tables[0].bucket_values[bucket_id].values[offset + leader];
                    }
                    first_list = group.shfl(first_list, leader);
                    break;
                }
            }
        }

        // Check the second table
        hash = Hasher<1>()(key);
        bucket_id = hash & (bucket_num_per_table - 1);
        bucket_keys = &tables[1].bucket_keys[bucket_id];

#pragma unroll
        for (RSize i = 0; i < GroupIter; i++) {
            if constexpr (ActualGroupSize == 1) {
                if (bucket_keys->keys[i] == key) {
                    second_list = &tables[1].bucket_values[bucket_id].values[i];
                    break;
                }
            } else {
                unsigned int offset = i * ActualGroupSize;
                if (int leader = group.elect(bucket_keys->keys[offset + lane_id] == key);
                    leader != -1) {
                    if (lane_id == leader) {
                        second_list = &tables[1].bucket_values[bucket_id].values[offset + leader];
                    }
                    second_list = group.shfl(second_list, leader);
                    break;
                }
            }
        }
        return {first_list, second_list};
    }

   protected:
    /// @brief Rehash keys in the hashmap.
    template <typename MemoryProvider>
    void _rehash(MemoryProvider *mem_provider) {
        if (bucket_num_per_table < 1) return;
        auto tmp_bucket_keys = RS_Array<BucketKeys, 1>({bucket_num_per_table}, mem_provider);
        auto tmp_bucket_values = RS_Array<BucketValues, 1>({bucket_num_per_table}, mem_provider);
        tmp_bucket_keys.copy(tables[0].bucket_keys);
        tmp_bucket_values.copy(tables[0].bucket_values);
        mem_provider->template run<RSCF_HashMapRehashKey<
            Hasher<0>,
            KeyType,
            RS_Array<BucketKeys, 1>,
            RS_Array<BucketValues, 1>>>(
            {bucket_num_per_table * BucketSize},
            tmp_bucket_keys,
            tmp_bucket_values,
            tables[0].bucket_keys,
            tables[0].bucket_values
        );
        tmp_bucket_keys.copy(tables[1].bucket_keys);
        tmp_bucket_values.copy(tables[1].bucket_values);
        mem_provider->template run<RSCF_HashMapRehashKey<
            Hasher<1>,
            KeyType,
            RS_Array<BucketKeys, 1>,
            RS_Array<BucketValues, 1>>>(
            {bucket_num_per_table * BucketSize},
            tmp_bucket_keys,
            tmp_bucket_values,
            tables[1].bucket_keys,
            tables[1].bucket_values
        );
        tmp_bucket_keys.free();
        tmp_bucket_values.free();
    }

    template <typename MemoryProvider>
    void _rebuild(MemoryProvider *provider) {
        if (bucket_num_per_table < 1) return;
        provider->template run<RSCF_HashMultiMapRebuild<ListHeaderType, RS_Array<BucketValues, 1>>>(
            {bucket_num_per_table * BucketSize}, tables[0].bucket_values, value_chunk_pool
        );
        provider->template run<RSCF_HashMultiMapRebuild<ListHeaderType, RS_Array<BucketValues, 1>>>(
            {bucket_num_per_table * BucketSize}, tables[1].bucket_values, value_chunk_pool
        );
    }

    template <typename SynchronizedGroup>
    HOST_DEVICE_INLINE_QUALIFIER void
    _lockBucket(bool is_second_table, uint32_t bucket_id, SynchronizedGroup &group) {
        if (group.thread_rank() == 0) {
#ifdef RS_DEBUG_HASHMAP
            printf(
                "Table %d Bucket %d begin locking by thread %d\n",
                (int)is_second_table,
                (int)bucket_id,
                (int)blockDim.x * blockIdx.x + threadIdx.x
            );
#endif
            tables[is_second_table].bucket_locks[bucket_id].lock();
#ifdef RS_DEBUG_HASHMAP
            printf(
                "Table %d Bucket %d locked by thread %d\n",
                (int)is_second_table,
                (int)bucket_id,
                (int)blockDim.x * blockIdx.x + threadIdx.x
            );
#endif
        }
        // Important to sync after lock, make sure all group are ready to perform actions
        group.sync();
    }

    template <typename SynchronizedGroup>
    HOST_DEVICE_INLINE_QUALIFIER void
    _unlockBucket(uint32_t is_second_table, uint32_t bucket_id, SynchronizedGroup &group) {
        // Important to sync before unlock, make sure all writes by the group are finished
        group.sync();
        if (group.thread_rank() == 0) {
            tables[is_second_table].bucket_locks[bucket_id].unlock();
#ifdef RS_DEBUG_HASHMAP
            printf(
                "Table %d Bucket %d unlocked by thread %d\n",
                (int)is_second_table,
                (int)bucket_id,
                (int)blockDim.x * blockIdx.x + threadIdx.x
            );
#endif
        }
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_HASHMAP_H
