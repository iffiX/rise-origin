// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SPARSE_H
#define RS_SPARSE_H

#include <refl.hpp>

#include "rise_origin/device/rs_compute.h"
#include "rise_origin/device/rs_static_construct.h"
#include "rise_origin/utils/rs_coord.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_array.h"
#include "rs_hashmap.h"

namespace rise {

namespace device {

template <RSize XSize_ = 0, RSize YSize_ = 0, RSize ZSize_ = 0>
struct RS_Sparse3DSize {
    static constexpr RSize XSize = XSize_;
    static constexpr RSize YSize = YSize_;
    static constexpr RSize ZSize = ZSize_;
};

/// Level descriptors.
template <RSize MaxEntryNum_ = 0, RSize BucketSize_ = 16>
struct RS_Sparse3DHashMapLevel {
    static constexpr RSize BucketSize = BucketSize_;
    static constexpr RSize MaxEntryNum = MaxEntryNum_;
    static_assert(isPowerOfTwo(BucketSize_), "BucketSize must be power of two");
    static constexpr bool isEmpty() { return MaxEntryNum_ == 0; }
};

template <RSize XSize = 0, RSize YSize = 0, RSize ZSize = 0>
struct RS_Sparse3DArrayLevel {
    using Size = RS_Sparse3DSize<XSize, YSize, ZSize>;
    static_assert(XSize == 0 || isPowerOfTwo(XSize), "XSize must be zero or power of two");
    static_assert(YSize == 0 || isPowerOfTwo(YSize), "YSize must be zero or power of two");
    static_assert(ZSize == 0 || isPowerOfTwo(ZSize), "ZSize must be zero or power of two");
    static constexpr bool isEmpty() { return XSize == 0 || YSize == 0 || ZSize == 0; }
};

namespace sparse_helper {
/// @brief Helper to create a type list of level configurations.
template <typename... LevelConfigs>
using LevelConfigList = refl::util::type_list<LevelConfigs...>;

// Helper to identify level types.
template <typename T>
struct is_hash_map_level : std::false_type {};

template <RSize MaxEntryNum, RSize BucketSize>
struct is_hash_map_level<RS_Sparse3DHashMapLevel<MaxEntryNum, BucketSize>> : std::true_type {};

template <typename T>
struct is_array_level : std::false_type {};

template <RSize XSize, RSize YSize, RSize ZSize>
struct is_array_level<RS_Sparse3DArrayLevel<XSize, YSize, ZSize>> : std::true_type {};

/****************************************************************/
RS_ENUM(NextLevelDataFlag, int, (IS_EMPTY, 0), (IS_INDEX, 0b1), (IS_VALUE, 0b10));

///@brief Next level data either stores an index to the
/// next level in the next mem pool, or a value.
template <typename ValueType>
struct NextLevelData {
    int flags = 0;
    RS_SpinLock lock{};
    union Data {
        RIndex index = RS_NULL_INDEX;
        ValueType value;
    } data;

    NextLevelData() = default;
    HOST_DEVICE_INLINE_QUALIFIER NextLevelData& operator=(const NextLevelData& other) {
        lock.lock();
        flags = other.flags;
        data = other.data;
        lock.unlock();
        return *this;
    }

    HOST_DEVICE_INLINE_QUALIFIER void setValue(const ValueType& value) {
        lock.lock();
        flags = NextLevelDataFlag::IS_VALUE;
        data.value = value;
        lock.unlock();
    }

    HOST_DEVICE_INLINE_QUALIFIER void setIndex(RIndex index) {
        lock.lock();
        flags = NextLevelDataFlag::IS_INDEX;
        data.index = index;
        lock.unlock();
    }

    HOST_DEVICE_INLINE_QUALIFIER void setValueNoLock(const ValueType& value) {
        flags = NextLevelDataFlag::IS_VALUE;
        data.value = value;
    }

    HOST_DEVICE_INLINE_QUALIFIER ValueType* getValue() {
        if (flags == NextLevelDataFlag::IS_VALUE) {
            return &data.value;
        }
        return nullptr;
    }
    HOST_DEVICE_INLINE_QUALIFIER RIndex getIndex() {
        if (flags == NextLevelDataFlag::IS_INDEX) {
            return data.index;
        }
        return 0;
    }
};

/****************************************************************/
///@brief Data structure type for root level.
template <typename ValueType, typename LevelConfig>
struct RootLevelData;

///@brief Specialization for hash map as the root level.
template <typename ValueType, RSize MaxEntryNum, RSize BucketSize>
struct RootLevelData<ValueType, RS_Sparse3DHashMapLevel<MaxEntryNum, BucketSize>> {
    struct Key {
        RIndex x = 0, y = 0, z = 0;
        Key() = default;
        HOST_DEVICE_INLINE_QUALIFIER explicit Key(const RCoord3& coord)
            : x(coord.x - RS_NumTraits<RSignedIndex>::lowest()),
              y(coord.y - RS_NumTraits<RSignedIndex>::lowest()),
              z(coord.z - RS_NumTraits<RSignedIndex>::lowest()) {}
        HOST_DEVICE_INLINE_QUALIFIER Key(RIndex x, RIndex y, RIndex z) : x(x), y(y), z(z) {}
        Key& operator=(const Key& k) = default;

        // Used to support +1 operation in hashmap
        HOST_DEVICE_INLINE_QUALIFIER Key operator+(const RIndex& val) const {
            Key result;
            result.x = x + val;
            result.y = y + (result.x < x ? 1 : 0);
            result.z = z + (result.y < y ? 1 : 0);
            return result;
        }

        HOST_DEVICE_INLINE_QUALIFIER bool operator==(const Key& k) const {
            return x == k.x && y == k.y && z == k.z;
        }
    };

    template <typename InputType, typename OutputType, RIndex HashType>
    struct KeyHasher {
        static_assert(std::is_same_v<InputType, Key>, "InputType must be the Key type.");
        static_assert(std::is_same_v<OutputType, uint32_t>, "OutputType must be uint32_t.");
        static_assert(HashType == 0 || HashType == 1, "HashType must be 0 or 1.");

        HOST_DEVICE_INLINE_QUALIFIER static bool isValidKey(InputType key) {
            return key.x < RS_NumTraits<RIndex>::highest() - 1 &&
                   key.y < RS_NumTraits<RIndex>::highest() - 1 &&
                   key.z < RS_NumTraits<RIndex>::highest() - 1;
        }

        HOST_DEVICE_INLINE_QUALIFIER static bool isNullKey(InputType key) {
            return key.x == 0 && key.y == 0 && key.z == 0;
        }

        HOST_DEVICE_INLINE_QUALIFIER static void setNullKey(InputType& key) { key = {0, 0, 0}; }

        HOST_DEVICE_INLINE_QUALIFIER OutputType operator()(InputType key) const {
            OutputType result;
            if constexpr (HashType == 0) {
                // From VDB paper
                result = key.x * 73856093 ^ key.y * 19349663 ^ key.z * 83492791;
            } else if constexpr (HashType == 1) {
                // Alternative hash function generated with test/generate_hash_constants.py
                result = key.x * 59735267 ^ key.y * 72177407 ^ key.z * 62296957;
            }
            return result;
        }
    };

    struct DataConstructor {
        using DataType =
            RS_HashStableFlatMap<Key, NextLevelData<ValueType>, uint32_t, KeyHasher, BucketSize>;

        static DataType construct(RS_MemoryProvider* provider) {
            return DataType(MaxEntryNum, provider);
        }
        static void free(DataType& data) { data.free(); }
    };
};

///@brief Specialization for array as the first level.
template <typename ValueType, RSize XSize, RSize YSize, RSize ZSize>
struct RootLevelData<ValueType, RS_Sparse3DArrayLevel<XSize, YSize, ZSize>> {
    struct DataConstructor {
        using DataType = RS_Array<NextLevelData<ValueType>, 3, false>;

        static DataType construct(RS_MemoryProvider* provider) {
            return DataType({XSize, YSize, ZSize}, provider);
        }

        static void free(DataType& data) { data.free(); }
    };
};

///@brief Data structure type for descendant levels.
template <typename ValueType, typename LevelConfig>
struct DescendantLevelData;

// Specialization for Array Level
template <typename ValueType, RSize XSize, RSize YSize, RSize ZSize>
struct DescendantLevelData<ValueType, RS_Sparse3DArrayLevel<XSize, YSize, ZSize>> {
    using DataType =
        RS_StaticColumnMajorTwoAlignedArray<NextLevelData<ValueType>, XSize, YSize, ZSize>;
};

/****************************************************************/
// Helper to compute prefix sums
template <typename ProcessedSizeList, typename PrefixMultiplySize, typename LevelConfigList>
struct PrefixMultiplySizeHelper;

// Specialization for non-empty RemainingSeq
template <
    typename ProcessedSizeList,
    typename PrefixMultiplySize,
    typename LevelConfig,
    typename... Rest>
struct PrefixMultiplySizeHelper<
    ProcessedSizeList,
    PrefixMultiplySize,
    refl::util::type_list<LevelConfig, Rest...>> {
    using MultipliedSize = RS_Sparse3DSize<
        PrefixMultiplySize::XSize * LevelConfig::Size::XSize,
        PrefixMultiplySize::YSize * LevelConfig::Size::YSize,
        PrefixMultiplySize::ZSize * LevelConfig::Size::ZSize>;
    using Result = typename PrefixMultiplySizeHelper<
        refl::trait::append_t<MultipliedSize, ProcessedSizeList>,
        MultipliedSize,
        refl::util::type_list<Rest...>>::Result;
};

// Base case for empty RemainingSeq
template <typename ProcessedSizeList, typename PrefixMultiplySize>
struct PrefixMultiplySizeHelper<ProcessedSizeList, PrefixMultiplySize, refl::util::type_list<>> {
    using Result = ProcessedSizeList;
};

/****************************************************************/
// Helper to enforce constraints on all levels
template <RSize CurrentLevel, typename LevelConfigList>
struct ValidateLevelConfigs;

template <RSize CurrentLevel, typename FirstLevelConfig, typename... Rest>
struct ValidateLevelConfigs<CurrentLevel, refl::util::type_list<FirstLevelConfig, Rest...>> {
    static_assert(
        is_hash_map_level<FirstLevelConfig>::value || is_array_level<FirstLevelConfig>::value,
        "Level must be a hash map level, or an array level"
    );
    // Ensure hash levels do not appear after the first level
    static_assert(
        !(CurrentLevel >= 1 && is_hash_map_level<FirstLevelConfig>::value),
        "Hash map level can only be used at the first level"
    );
    static_assert(
        !(CurrentLevel >= 1 && is_array_level<FirstLevelConfig>::value &&
          FirstLevelConfig::isEmpty()),
        "Empty array level can only be used at the first level"
    );
    ValidateLevelConfigs<CurrentLevel + 1, refl::util::type_list<Rest...>> _{};
};

template <RSize CurrentLevel>
struct ValidateLevelConfigs<CurrentLevel, refl::util::type_list<>> {};

/****************************************************************/
// Helper function to call a function on each element of the tuple
template <typename Func, typename Tuple, std::size_t... Indexes>
void for_each_in_tuple_impl(Tuple& tuple, Func&& func, std::index_sequence<Indexes...>) {
    (func(rs_device_std::get<Indexes>(tuple)), ...);
}

// Main function to iterate over the tuple
template <typename Func, typename... Args>
void for_each_in_tuple(rs_device_std::tuple<Args...>& tuple, Func&& func) {
    for_each_in_tuple_impl(tuple, std::forward<Func>(func), std::index_sequence_for<Args...>{});
}

// Const version
template <typename Func, typename Tuple, std::size_t... Indexes>
void for_each_in_tuple_impl(const Tuple& tuple, Func&& func, std::index_sequence<Indexes...>) {
    (func(rs_device_std::get<Indexes>(tuple)), ...);
}

template <typename Func, typename... Args>
void for_each_in_tuple(const rs_device_std::tuple<Args...>& tuple, Func&& func) {
    for_each_in_tuple_impl(tuple, std::forward<Func>(func), std::index_sequence_for<Args...>{});
}

// Host device, and const version
template <typename Func, typename Tuple, std::size_t... Indexes>
HOST_DEVICE_INLINE_QUALIFIER void
for_each_in_tuple_impl_hd(const Tuple& tuple, Func&& func, std::index_sequence<Indexes...>) {
    (func(rs_device_std::get<Indexes>(tuple)), ...);
}

template <typename Func, typename... Args>
HOST_DEVICE_INLINE_QUALIFIER void
for_each_in_tuple_hd(const rs_device_std::tuple<Args...>& tuple, Func&& func) {
    for_each_in_tuple_impl_hd(tuple, std::forward<Func>(func), std::index_sequence_for<Args...>{});
}

// Helper function to call a function on each element of the tuple and the list
template <typename Func, typename List, typename Tuple, std::size_t... Indexes>
void for_each_in_zip_list_tuple_impl(
    List list,
    Tuple& tuple,
    Func&& func,
    std::index_sequence<Indexes...>
) {
    (func(refl::trait::get_t<Indexes, List>{}, rs_device_std::get<Indexes>(tuple)), ...);
}

// Main function to iterate over the tuple and list
template <typename Func, typename List, typename... Args>
void for_each_in_zip_list_tuple(List list, rs_device_std::tuple<Args...>& tuple, Func&& func) {
    for_each_in_zip_list_tuple_impl(
        list, tuple, std::forward<Func>(func), std::index_sequence_for<Args...>{}
    );
}

// Helper function to call a function on each element of the two tuples
template <typename Func, typename Tuple, std::size_t... Indexes>
void for_each_in_zip_tuples_impl(
    Tuple& tuple1,
    Tuple& tuple2,
    Func&& func,
    std::index_sequence<Indexes...>
) {
    (func(rs_device_std::get<Indexes>(tuple1), rs_device_std::get<Indexes>(tuple2)), ...);
}

// Main function to iterate over two tuples
template <typename Func, typename... Args>
void for_each_in_zip_tuples(
    rs_device_std::tuple<Args...>& tuple1,
    rs_device_std::tuple<Args...>& tuple2,
    Func&& func
) {
    for_each_in_zip_tuples_impl(
        tuple1, tuple2, std::forward<Func>(func), std::index_sequence_for<Args...>{}
    );
}

template <typename Func, typename Tuple, std::size_t... Indexes>
void for_each_in_zip_tuples_impl(
    Tuple& tuple1,
    const Tuple& tuple2,
    Func&& func,
    std::index_sequence<Indexes...>
) {
    (func(rs_device_std::get<Indexes>(tuple1), rs_device_std::get<Indexes>(tuple2)), ...);
}

// const version
template <typename Func, typename... Args>
void for_each_in_zip_tuples(
    rs_device_std::tuple<Args...>& tuple1,
    const rs_device_std::tuple<Args...>& tuple2,
    Func&& func
) {
    for_each_in_zip_tuples_impl(
        tuple1, tuple2, std::forward<Func>(func), std::index_sequence_for<Args...>{}
    );
}

}  // namespace sparse_helper

template <typename MemArrayPool>
class RSCF_ClearMemArrayPool : public RS_ComputeFunction<RSCF_ClearMemArrayPool<MemArrayPool>> {
   public:
    using BaseType = RS_ComputeFunction<RSCF_ClearMemArrayPool>;
    using BaseType::index1D;

    HOST_DEVICE_QUALIFIER static void run(MemArrayPool mem_array_pool) {
        if (mem_array_pool.chunk_mask.isOn(index1D())) {
            memset(
                &mem_array_pool.chunks[index1D()],
                0,
                std::remove_reference_t<MemArrayPool>::ChunkBytes
            );
        }
    }
};

/**
 * @brief Sparse 3D data structure. Based on the OpenVDB paper
 * https://dl.acm.org/doi/10.1145/2487228.2487235.
 *
 * Example usage:
 *
 *  RS_Sparse3D<
 *        int,
 *       RS_Sparse3DHashMapLevel<1024>,
 *       RS_Sparse3DArrayLevel<4, 4, 4>,
 *       RS_Sparse3DArrayLevel<8, 8, 8>>
 *       sparse(host_provider);
 * @note Accessor must be called with clear(), or acquired anew after calling resize() or
 * resizeMemPool() as the cached pointers will be invalid.
 * @tparam ValueType The value type of the data.
 * @tparam LevelConfigs The configuration of each level.
 */
template <typename ValueType, typename... LevelConfigs>
class RS_Sparse3DMap {
   public:
    using NextLevelDataType = sparse_helper::NextLevelData<ValueType>;
    using LevelConfigList = refl::util::type_list<LevelConfigs...>;
    using RootLevelConfig = typename refl::trait::get<0, LevelConfigList>::type;
    using DescendantLevelConfigList = typename refl::trait::skip<1, LevelConfigList>::type;
    // Number of levels
    static constexpr size_t LevelNum = sizeof...(LevelConfigs);

    static_assert(sizeof...(LevelConfigs) > 0, "At least one level must be specified");
    sparse_helper::ValidateLevelConfigs<0, LevelConfigList> _{};

    // Root level data constructor and type
    using RootLevelDataConstructor =
        typename sparse_helper::RootLevelData<ValueType, RootLevelConfig>::DataConstructor;
    using RootLevelDataType = typename RootLevelDataConstructor::DataType;

    // Descendant level data type list
    template <typename LevelConfig>
    struct GetDescendantLevelDataType {
        using type = typename sparse_helper::DescendantLevelData<ValueType, LevelConfig>::DataType;
    };
    using DescendantLevelDataTypeList =
        typename refl::trait::map<GetDescendantLevelDataType, DescendantLevelConfigList>::type;

    // Helper to compute descendant level sizes
    template <typename DescendantLevelConfigList>
    using GetDescendantSizeList =
        refl::trait::reverse_t<typename sparse_helper::PrefixMultiplySizeHelper<
            refl::util::type_list<>,
            RS_Sparse3DSize<1, 1, 1>,
            refl::trait::reverse_t<DescendantLevelConfigList>>::Result>;

    using DescendantLevelSizeList = GetDescendantSizeList<DescendantLevelConfigList>;
    // Next level sizes, the number is one more level than descendant levels since we
    // append (1, 1, 1) to the end, the total number is equal to LevelNum
    using NextLevelSizeList =
        refl::trait::append_t<RS_Sparse3DSize<1, 1, 1>, DescendantLevelSizeList>;

    // Descendant mem_pool storage helper
    template <typename>
    struct AsSparseDescendantMemPoolTuple;

    template <template <typename...> typename T2, typename... Ts>
    struct AsSparseDescendantMemPoolTuple<T2<Ts...>> {
        using type = rs_device_std::tuple<RS_MemArrayPool<sizeof(Ts)>...>;
    };

    template <typename T2>
    using AsSparseDescendantMemPoolTupleType = typename AsSparseDescendantMemPoolTuple<T2>::type;

    // Tuple of descendant mem pools
    using DescendantLevelDataMemPoolTuple =
        AsSparseDescendantMemPoolTupleType<DescendantLevelDataTypeList>;

    // Accessor class to cache value pointers for each level
    class RS_Sparse3DMapAccessor {
       public:
        RCoord3 coord{RS_NULL_SIGNED_INDEX, RS_NULL_SIGNED_INDEX, RS_NULL_SIGNED_INDEX};
        RIndex level = RS_NULL_INDEX;

        // (LevelNum - 1) descendant level array pointer + 1 last pointer as the pointer to value
        rs_device_std::array<NextLevelDataType*, LevelNum> descendant_level_pointers;

        HOST_DEVICE_INLINE_QUALIFIER RS_Sparse3DMapAccessor() {
            descendant_level_pointers.fill(nullptr);
        };

        HOST_DEVICE_INLINE_QUALIFIER void clear() {
            coord = RCoord3(RS_NULL_SIGNED_INDEX, RS_NULL_SIGNED_INDEX, RS_NULL_SIGNED_INDEX);
            level = RS_NULL_INDEX;
            descendant_level_pointers.fill(nullptr);
        }

        HOST_DEVICE_INLINE_QUALIFIER void clearAfterLastValidLevel() {
            if (level == RS_NULL_INDEX)
                clear();
            else
                for (RIndex i = level + 1; i < LevelNum; ++i)
                    descendant_level_pointers[i] = nullptr;
        }

        HOST_DEVICE_INLINE_QUALIFIER bool isNull() { return level == RS_NULL_INDEX; }

        HOST_DEVICE_INLINE_QUALIFIER NextLevelDataType* getLast() {
            if (level == RS_NULL_INDEX) return nullptr;
            return descendant_level_pointers[level];
        }
    };

    /*****************************************************************
     * Data
     *****************************************************************/
    // Bound coordinate of the all positive quadrant
    RCoord3 pos_bound{0, 0, 0};
    // Root level data
    RootLevelDataType root{};
    DescendantLevelDataMemPoolTuple descendant_mem_pools{};

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_Sparse3DMap() = default;

    /**
     * @brief Construct a new sparce 3d map.
     * @param provider The memory provider to use.
     */
    explicit RS_Sparse3DMap(RS_MemoryProvider* provider) { init(provider); }

    /**
     * @brief Initialize the sparce 3d map.
     * @param provider The memory provider to use.
     */
    void init(RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        pos_bound = RCoord3::zero();
        // construct parameters are provided by level configs
        root = RootLevelDataConstructor::construct(provider);
        sparse_helper::for_each_in_tuple(descendant_mem_pools, [&provider](auto& pool) {
            pool.init(provider);
        });
        _updateBound();
        _clear<true>(true);
    }

    HOST_DEVICE_INLINE_QUALIFIER RS_Sparse3DMap(const RS_Sparse3DMap& other)
        : pos_bound(other.pos_bound),
          root(other.root),
          descendant_mem_pools(other.descendant_mem_pools) {}

    /**
     * @brief Frees the memory used by the sparce 3d map.
     */
    void free() {
        pos_bound = RCoord3::zero();
        RootLevelDataConstructor::free(root);
        sparse_helper::for_each_in_tuple(descendant_mem_pools, [](auto& pool) { pool.free(); });
    }

    /**
     * @brief Resizes the root hashmap of the sparce 3d map.
     * @param max_entry_num The maximum number of entries in the hashmap, should be power of two,
     * otherwise it will be aligned to the next power of two.
     * @param copy Whether to copy the existing root hashmap data to the new hashmap.
     */
    template <typename RC = RootLevelConfig>
    std::enable_if_t<sparse_helper::is_hash_map_level<RC>::value, void>
    resize(RSize max_entry_num, bool copy = true) {
        RS_DATA_EXPECT_INIT(root.provider() != nullptr);

        root.resize(max_entry_num, copy);
        _updateBound();
    }

    /**
     * @brief Resizes the root array of the sparce 3d map.
     * @note Currently it doesn't provide spatial consistency and querying original data will fail.
     * @param new_shape The new shape of the root array, each dimension should be a power of two.
     * @param copy Whether to copy the existing root array data to the new array.
     */
    template <typename RC = RootLevelConfig>
    std::enable_if_t<sparse_helper::is_array_level<RC>::value, void>
    resize(std::array<RSize, 3> new_shape, bool copy = true) {
        RS_DATA_EXPECT_INIT(root.provider() != nullptr);

        if ((new_shape[0] > 0 && !isPowerOfTwo(new_shape[0])) ||
            (new_shape[1] > 0 && !isPowerOfTwo(new_shape[1])) ||
            (new_shape[2] > 0 && !isPowerOfTwo(new_shape[2]))) {
            RS_THROW(RS_InvalidArgumentException("Root shape of array must be powers of 2"));
        }
        root.resize(new_shape, copy);
        _updateBound();
    }

    /**
     * @brief Resizes the memory pool corresponding to the descendant level of the sparse data
     * structure.
     * @param level The level of the descendant memory pool to resize. Should be from 1 to level num
     * - 1.
     * @param mem_pool_size The new size of the memory pool.
     */
    void resizeMemPool(RIndex level, RSize mem_pool_size, bool copy = true) {
        RS_DATA_EXPECT_INIT(root.provider() != nullptr);

        if (level < 1 || level >= LevelNum) {
            RS_THROW(RS_InvalidArgumentException(
                "Resized mem pool level must be between 1 and level num - 1"
            ));
        }
        loop<1, LevelNum>([&](auto i_const) {
            constexpr RIndex i = i_const.value;
            if (i == level) {
                auto& mem_pool = rs_device_std::get<i - 1>(descendant_mem_pools);
                if (copy) {
                    RSize old_mem_size = mem_pool.chunks.data.size();
                    mem_pool.resize(mem_pool_size);
                    RSize new_mem_size = mem_pool.chunks.data.size();
                    mem_pool.chunks.data.set(0, new_mem_size - old_mem_size, old_mem_size, sync);
                } else {
                    mem_pool.resize(mem_pool_size, false);
                }
            }
        });
    }

    /**
     * @brief Clears all used entries in the sparce 3d map.
     */
    void clear(bool sync = true) {
        RS_DATA_EXPECT_INIT(root.provider() != nullptr);

        _clear<true>(sync);
    }

    /**
     * @brief Copies the sparce 3d map, resize as needed.
     * @param sparse The sparce 3d map to copy from.
     */
    void copy(const RS_Sparse3DMap& sparse) {
        RS_DATA_EXPECT_INIT(root.provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(sparse.root.provider() != nullptr);

        pos_bound = sparse.pos_bound;
        root.copy(sparse.root);
        sparse_helper::for_each_in_zip_tuples(
            descendant_mem_pools, sparse.descendant_mem_pools, [](auto& pool, auto& other_pool) {
                pool.copy(other_pool);
            }
        );
    }

    /**
     * @brief Moves data to the specified MemoryProvider and returns a new sparce 3d map.
     */
    RS_Sparse3DMap move(RS_MemoryProvider* provider) const {
        RS_DATA_EXPECT_INIT(root.provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        RS_Sparse3DMap new_sparse;
        new_sparse.pos_bound = pos_bound;
        new_sparse.root = root.move(provider);
        sparse_helper::for_each_in_zip_tuples(
            new_sparse.descendant_mem_pools,
            descendant_mem_pools,
            [&provider](auto& pool, auto& other_pool) { pool = other_pool.move(provider); }
        );
        return new_sparse;
    }

    /// @brief Returns the memory provider of the sparce 3d map.
    [[nodiscard]] RS_MemoryProvider* provider() const { return root.provider(); }

    /// @brief Returns the device of the sparce 3d map.
    [[nodiscard]] RS_DeviceLocation device() const { return root.device(); }

    /// @brief Returns the base size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const { return sizeof(*this); }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const {
        RSize extra_bytes = 0;
        extra_bytes += root.extraBytes();
        sparse_helper::for_each_in_tuple_hd(descendant_mem_pools, [&extra_bytes](auto& pool) {
            extra_bytes += pool.extraBytes();
        });
        return extra_bytes;
    }

    /// @brief Returns the maximum valid coordinate of the sparce 3d map.
    HOST_DEVICE_INLINE_QUALIFIER RCoord3 maxCoord() const { return pos_bound - 1; }

    /// @brief Returns the minimum valid coordinate of the sparce 3d map.
    HOST_DEVICE_INLINE_QUALIFIER RCoord3 minCoord() const { return -pos_bound; }

    /// @brief Returns the size of the sparce 3d map at the specified level.
    /// @param level The level of the sparce 3d map.
    /// @return The size of the sparce 3d map at the specified level. when level = 0,
    /// returns the max bound size of root, when level = LevelNum - 1, returns the
    /// size of the last level, when level = LevelNum, returns 1 for each dimension.
    HOST_DEVICE_INLINE_QUALIFIER RCoord3 getLevelSize(RIndex level) const {
        assert(level <= LevelNum && "level must be less or equal to LevelNum");
        RCoord3 size = RCoord3::zero();
        switchOver<0, LevelNum + 1>(level, [&](auto l_const) {
            size = getLevelSize<l_const.value>();
        });
        return size;
    }

    /// @brief Returns the size of the sparce 3d map at the specified level.
    /// @tparam Level The level of the sparce 3d map.
    /// @return The size of the sparce 3d map at the specified level. when Level = 0,
    /// returns the max bound size of root, when Level = LevelNum - 1, returns the
    /// size of the last level, when Level = LevelNum, returns 1 for each dimension.
    template <RIndex Level>
    HOST_DEVICE_INLINE_QUALIFIER RCoord3 getLevelSize() const {
        static_assert(Level <= LevelNum, "Level must be less or equal to LevelNum");
        if constexpr (Level == 0 && sparse_helper::is_hash_map_level<RootLevelConfig>::value) {
            return RCoord3::upperBound();
        } else if constexpr (Level == 0 &&
                             !sparse_helper::is_hash_map_level<RootLevelConfig>::value) {
            return pos_bound << 1;
        } else {
            return RCoord3(
                refl::trait::get_t<Level - 1, NextLevelSizeList>::XSize,
                refl::trait::get_t<Level - 1, NextLevelSizeList>::YSize,
                refl::trait::get_t<Level - 1, NextLevelSizeList>::ZSize
            );
        }
    }

    /// @brief Returns the region bound of the sparce 3d map at the specified level.
    /// @tparam Level The level of the sparce 3d map.
    /// @param position The position of the sparce 3d map.
    /// @param min_bound The minimum bound of the sparce 3d map.
    /// @param max_bound The maximum bound of the sparce 3d map.
    template <RIndex Level>
    HOST_DEVICE_INLINE_QUALIFIER void
    getLevelRegionBound(const RCoord3& position, RCoord3& min_bound, RCoord3& max_bound) const {
        static_assert(Level <= LevelNum, "Level must be less or equal to LevelNum");
        if constexpr (Level == 0) {
            min_bound = RCoord3::lowerBound();
            max_bound = RCoord3::upperBound();
        } else {
            min_bound = position & ~(getLevelSize<Level>() - 1);
            max_bound = min_bound + getLevelSize<Level>();
        }
    }

    /// @brief Returns whether the coordinate is in the bound of the sparce 3d map.
    HOST_DEVICE_INLINE_QUALIFIER bool isInBound(const RCoord3& coord) const {
        return (coord.x >= -pos_bound[0] && coord.x < pos_bound[0]) &&
               (coord.y >= -pos_bound[1] && coord.y < pos_bound[1]) &&
               (coord.z >= -pos_bound[2] && coord.z < pos_bound[2]);
    }

    /// @brief Returns the accessor of the sparce 3d map.
    HOST_DEVICE_INLINE_QUALIFIER RS_Sparse3DMapAccessor getAccessor() {
        return RS_Sparse3DMapAccessor();
    }

    /**
     * @brief Returns whether the level at the specified coordinate is empty.
     * @tparam MaxLevel The maximum level to check. Valid range is [0, LevelNum - 1].
     * @tparam CheckBound Whether to check the bound of the sparce 3d map.
     * @param coord The coordinate of the level to check.
     * @return True if the level is empty, false otherwise.
     */
    template <RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool isLevelEmpty(const RCoord3& coord) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return nullptr;
        }
        auto accessor = getAccessor();
        // Accessor is null, we can skip coordinate checking
        return _search<MaxLevel, true>(coord, {}, accessor);
    }

    /**
     * @brief Returns whether the level at the specified coordinate is empty.
     * @tparam MaxLevel The maximum level to check. Valid range is [0, LevelNum - 1].
     * @tparam CheckBound Whether to check the bound of the sparce 3d map.
     * @param coord The coordinate of the level to check.
     * @param accessor The accessor of the sparce 3d map.
     * @return True if the level is empty, false otherwise.
     */
    template <RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool
    isLevelEmpty(const RCoord3& coord, RS_Sparse3DMapAccessor& accessor) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return nullptr;
        }
        accessor.level = RS_NULL_INDEX;
        return _search<MaxLevel, false>(coord, accessor.coord, accessor);
    }

    /**
     * @brief Inserts a value into the sparce 3d map.
     * @tparam MaxLevel The maximum level to insert the value.
     * @tparam CheckBound Whether to check the bound of the sparce 3d map.
     * @param coord The coordinate of the value to insert.
     * @param value The value to insert.
     * @return True if the value is inserted successfully, false otherwise.
     */
    template <RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool insert(const RCoord3& coord, const ValueType& value) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        auto accessor = getAccessor();
        // Accessor is null, we can skip coordinate checking
        bool result = _activate<true, MaxLevel, true>(coord, {}, accessor);
        if (result) accessor.getLast()->setValue(value);
        return result;
    }

    /**
     * @brief Inserts a value into the sparce 3d map with a cached accessor.
     * @tparam MaxLevel The maximum level to insert the value.
     * @tparam CheckBound Whether to check the bound of the sparce 3d map.
     * @param coord The coordinate of the value to insert.
     * @param value The value to insert.
     * @param accessor A cached accessor of the sparce 3d map.
     * @return True if the value is inserted successfully, false otherwise.
     */
    template <RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool
    insert(const RCoord3& coord, const ValueType& value, RS_Sparse3DMapAccessor& accessor) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        accessor.level = RS_NULL_INDEX;
        bool result = _activate<true, MaxLevel, false>(coord, accessor.coord, accessor);
        if (result) accessor.getLast()->setValue(value);
        return result;
    }

    /**
     * @brief Finds a value in the sparce 3d map.
     * @tparam MaxLevel The maximum level to find the value.
     * @tparam CheckBound Whether to check the bound of the sparce 3d map.
     * @param coord The coordinate of the value to find.
     * @return A pointer to the value if found, nullptr otherwise.
     */
    template <RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER ValueType* find(const RCoord3& coord) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return nullptr;
        }
        auto accessor = getAccessor();
        // Accessor is null, we can skip coordinate checking
        bool result = _search<MaxLevel, true>(coord, {}, accessor);
        if (result) return accessor.getLast()->getValue();
        return nullptr;
    }

    /**
     * @brief Finds a value in the sparce 3d map with a cached accessor.
     * @tparam MaxLevel The maximum level to find the value.
     * @tparam CheckBound Whether to check the bound of the sparce 3d map.
     * @param coord The coordinate of the value to find.
     * @param accessor A cached accessor of the sparce 3d map.
     * @return A pointer to the value if found, nullptr otherwise.
     */
    template <RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER ValueType*
    find(const RCoord3& coord, RS_Sparse3DMapAccessor& accessor) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return nullptr;
        }
        accessor.level = RS_NULL_INDEX;
        bool result = _search<MaxLevel, false>(coord, accessor.coord, accessor);
        if (result) return accessor.getLast()->getValue();
        return nullptr;
    }

    /**
     * @brief Activates a value in the sparce 3d map.
     * @tparam OverwriteValue Whether to overwrite the value if the value is already activated, in
     * a level that precedes the specified MaxLevel.
     * @tparam MaxLevel The maximum level to activate the value.
     * @tparam CheckBound Whether to check the bound of the sparce 3d map.
     * @param coord The coordinate of the value to activate.
     * @param accessor The accessor to use for activating the value.
     * @return True if the value is activated successfully, false otherwise. On success, accessor
     * will be inited with the path to the activated value.
     */
    template <bool OverwriteValue = true, RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool
    activate(const RCoord3& coord, RS_Sparse3DMapAccessor& accessor) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        accessor.level = RS_NULL_INDEX;
        bool result = _activate<OverwriteValue, MaxLevel>(coord, accessor.coord, accessor);
        accessor.coord = coord;
        return result;
    }

    /**
     * @brief Accesses a value in the sparce 3d map.
     * @tparam MaxLevel The maximum level to access the value.
     * @tparam CheckBound Whether to check the bound of the sparce 3d map.
     * @param coord The coordinate of the value to access.
     * @param accessor The accessor to use for accessing the value.
     * @return True if the value is accessed successfully, false otherwise. On success, accessor
     * will be inited with the path to the accessed value.
     */
    template <RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool
    access(const RCoord3& coord, RS_Sparse3DMapAccessor& accessor) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        accessor.level = RS_NULL_INDEX;
        bool result = _access<MaxLevel>(coord, accessor.coord, accessor);
        accessor.coord = coord;
        return result;
    }

    /**
     * @brief Finds a value in the sparce 3d map.
     * @tparam MaxLevel The maximum level to search for the value.
     * @tparam CheckBound Whether to check the bound of the sparce 3d map.
     * @param coord The coordinate of the value to search.
     * @param accessor The accessor to use for searching the value.
     * @return True if the value is found, false otherwise. On success, accessor will be inited
     * with the path to the found value.
     */
    template <RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool
    search(const RCoord3& coord, RS_Sparse3DMapAccessor& accessor) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        accessor.level = RS_NULL_INDEX;
        bool result = _search<MaxLevel>(coord, accessor.coord, accessor);
        accessor.coord = coord;
        return result;
    }

   private:
    /// @brief Clears the sparce 3d map.
    template <bool IsInitial>
    void _clear(bool sync) {
        if constexpr (sparse_helper::is_hash_map_level<RootLevelConfig>::value) {
            root.clear();
        } else {
            root.data.set(0, root.data.size(), 0, false);
        }
        castProviderPtrAndCall(provider(), [&](auto p_ptr) {
            if constexpr (IsInitial) {
                sparse_helper::for_each_in_tuple(descendant_mem_pools, [sync](auto& pool) {
                    pool.chunk_mask.fill(false, sync);
                    pool.chunks.data.set(0, pool.chunks.data.size(), 0, sync);
                });
            } else {
                sparse_helper::for_each_in_tuple(
                    descendant_mem_pools, [this, p_ptr, sync](auto& pool) {
                        p_ptr->template run<RSCF_ClearMemArrayPool<decltype(pool)>>(
                            {pool.size(), RS_AUTO_BLOCK_SIZE, sync}, pool
                        );
                        pool.chunk_mask.fill(false, sync);
                    }
                );
            }
        });
    }

    /// @brief Updates the bound of the sparce 3d map.
    void _updateBound() {
        if constexpr (!sparse_helper::is_hash_map_level<RootLevelConfig>::value) {
            if constexpr (LevelNum == 1) {
                pos_bound = RCoord3(root.shape[0], root.shape[1], root.shape[2]) >> 1;
            } else {
                pos_bound = RCoord3(
                                root.shape[0] * refl::trait::get_t<0, NextLevelSizeList>::XSize,
                                root.shape[1] * refl::trait::get_t<0, NextLevelSizeList>::YSize,
                                root.shape[2] * refl::trait::get_t<0, NextLevelSizeList>::ZSize
                            ) >>
                            1;
            }
        } else {
            pos_bound = RCoord3::upperBound();
        }
    }

    /**
     * @brief Activate level pointed by coord, till MaxLevel specified.
     * @return True if activated (including activation blocked by a value).
     * False if allocation failed / blocked by an intermediate value (if OverwriteValue = false).
     * @tparam OverwriteValue If true, intermediate levels initialized as value will be
     * overwritten if accessed level is a recursive child of the intermediate level.
     */
    template <bool OverwriteValue, RIndex MaxLevel, bool IsCoordDifferent = false, RIndex Level = 0>
    HOST_DEVICE_INLINE_QUALIFIER bool _activate(
        const RCoord3& level_relative_coord,
        const RCoord3& accessor_level_relative_coord,
        RS_Sparse3DMapAccessor& accessor
    ) {
        static_assert(MaxLevel < LevelNum, "MaxLevel must be less than LevelNum");

        // Note: next level  size list exclude the root level, so no need to increase to Level + 1
        static constexpr RSize NextLevelXSize = refl::trait::get_t<Level, NextLevelSizeList>::XSize;
        static constexpr RSize NextLevelYSize = refl::trait::get_t<Level, NextLevelSizeList>::YSize;
        static constexpr RSize NextLevelZSize = refl::trait::get_t<Level, NextLevelSizeList>::ZSize;
        static constexpr RSize NextLevelXLog2 = logTwoOf(NextLevelXSize);
        static constexpr RSize NextLevelYLog2 = logTwoOf(NextLevelYSize);
        static constexpr RSize NextLevelZLog2 = logTwoOf(NextLevelZSize);

        RCoord3 current_value_coord =
            level_relative_coord >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
        RCoord3 next_relative_coord =
            level_relative_coord &
            RCoord3(NextLevelXSize - 1, NextLevelYSize - 1, NextLevelZSize - 1);

        NextLevelDataType* pointer;

        if constexpr (!IsCoordDifferent && Level < MaxLevel) {
            RCoord3 acc_current_value_coord =
                accessor_level_relative_coord >>
                RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
            RCoord3 acc_next_relative_coord =
                accessor_level_relative_coord &
                RCoord3(NextLevelXSize - 1, NextLevelYSize - 1, NextLevelZSize - 1);

            pointer = accessor.descendant_level_pointers[Level];

            if (pointer != nullptr && current_value_coord == acc_current_value_coord &&
                pointer->flags == sparse_helper::NextLevelDataFlag::IS_INDEX) {
                // Accessor is still the same for this level, keep level pointers the same
                accessor.level = Level;
                return _activate<OverwriteValue, MaxLevel, false, Level + 1>(
                    next_relative_coord, acc_next_relative_coord, accessor
                );
            }
        }

        // Accessor is different for this level, or the pointer is pointing to
        // a value and not an index to mempool in next level, update level pointers
        if constexpr (sparse_helper::is_hash_map_level<
                          refl::trait::get_t<Level, LevelConfigList>>::value) {
            typename RootLevelDataType::KeyType key{current_value_coord};
            bool exists = false;

            // Hashmap must be root level, so no branch for other levels
            // Note: must use insert to either find existing entry, or insert a new entry
            // because we need to guarantee that memory writes to hashmap are finished
            pointer = root.insertUnique(key, {}, exists, /* overwrite */ false);
            if (pointer == nullptr) {
                accessor.clearAfterLastValidLevel();
                return false;
            }
        } else {
            // is array
            // For level 0, input is global coordinate, transform to relative coordinate
            // do it here to prevent overflow
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                pointer = &root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(descendant_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantLevelDataTypeList>*>(
                        mem_pool.getChunkPtr(
                            accessor.descendant_level_pointers[Level - 1]->data.index
                        )
                    );
                pointer = &array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }
        }
        if constexpr (Level < MaxLevel) {
            pointer->lock.lock();
            if (pointer->flags != sparse_helper::NextLevelDataFlag::IS_INDEX) {
                if (pointer->flags == sparse_helper::NextLevelDataFlag::IS_EMPTY) {
                    if (!_initNextLevelDataAsIndex<Level + 1>(
                            pointer,
                            current_value_coord.x * 73856093 ^ current_value_coord.y * 19349663 ^
                                current_value_coord.z * 83492791
                        )) {
                        pointer->lock.unlock();
                        accessor.clearAfterLastValidLevel();
                        return false;
                    }
                } else if (pointer->flags == sparse_helper::NextLevelDataFlag::IS_VALUE) {
                    if constexpr (OverwriteValue) {
                        ValueType value = pointer->data.value;
                        if (!_initNextLevelDataAsIndex<Level + 1>(
                                pointer,
                                current_value_coord.x * 73856093 ^
                                    current_value_coord.y * 19349663 ^
                                    current_value_coord.z * 83492791
                            )) {
                            pointer->lock.unlock();
                            accessor.clearAfterLastValidLevel();
                            return false;
                        }
                        _fillNextLevelDataWithValue<Level + 1>(pointer, value);
                    } else {
                        pointer->lock.unlock();
                        accessor.clearAfterLastValidLevel();
                        return false;
                    }
                }
            }
            pointer->lock.unlock();
            accessor.level = Level;
            accessor.descendant_level_pointers[Level] = pointer;

            return _activate<OverwriteValue, MaxLevel, true, Level + 1>(
                next_relative_coord, {}, accessor
            );
        } else {
            // Level == MaxLevel
            accessor.level = Level;
            accessor.descendant_level_pointers[Level] = pointer;
            return true;
        }
    }

    /**
     * @brief Access by coord till MaxLevel specified.
     * @return True if target level can be accessed. False if target level is not allocated.
     */
    template <RIndex MaxLevel, bool IsCoordDifferent = false, RIndex Level = 0>
    HOST_DEVICE_INLINE_QUALIFIER bool _access(
        const RCoord3& level_relative_coord,
        const RCoord3& accessor_level_relative_coord,
        RS_Sparse3DMapAccessor& accessor
    ) {
        static_assert(MaxLevel < LevelNum, "MaxLevel must be less than LevelNum");

        // Note: next level  size list exclude the root level, so no need to increase to Level + 1
        static constexpr RSize NextLevelXSize = refl::trait::get_t<Level, NextLevelSizeList>::XSize;
        static constexpr RSize NextLevelYSize = refl::trait::get_t<Level, NextLevelSizeList>::YSize;
        static constexpr RSize NextLevelZSize = refl::trait::get_t<Level, NextLevelSizeList>::ZSize;
        static constexpr RSize NextLevelXLog2 = logTwoOf(NextLevelXSize);
        static constexpr RSize NextLevelYLog2 = logTwoOf(NextLevelYSize);
        static constexpr RSize NextLevelZLog2 = logTwoOf(NextLevelZSize);

        RCoord3 current_value_coord =
            level_relative_coord >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
        RCoord3 next_relative_coord =
            level_relative_coord &
            RCoord3(NextLevelXSize - 1, NextLevelYSize - 1, NextLevelZSize - 1);

        NextLevelDataType* pointer;

        if constexpr (!IsCoordDifferent && Level < MaxLevel) {
            RCoord3 acc_current_value_coord =
                accessor_level_relative_coord >>
                RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
            RCoord3 acc_next_relative_coord =
                accessor_level_relative_coord &
                RCoord3(NextLevelXSize - 1, NextLevelYSize - 1, NextLevelZSize - 1);

            pointer = accessor.descendant_level_pointers[Level];

            if (pointer != nullptr && current_value_coord == acc_current_value_coord &&
                pointer->flags == sparse_helper::NextLevelDataFlag::IS_INDEX) {
                // Accessor is still the same for this level, keep level pointers the same
                accessor.level = Level;
                return _access<MaxLevel, false, Level + 1>(
                    next_relative_coord, acc_next_relative_coord, accessor
                );
            }
        }

        // Accessor is different for this level, or the pointer is pointing to
        // a value and not an index to mempool in next level, update level pointers
        if constexpr (sparse_helper::is_hash_map_level<
                          refl::trait::get_t<Level, LevelConfigList>>::value) {
            typename RootLevelDataType::KeyType key{current_value_coord};
            // Hashmap must be root level, so no branch for other levels
            pointer = root.find(key);
            if (pointer == nullptr) {
                accessor.clearAfterLastValidLevel();
                return false;
            }

        } else {
            // is array
            // For level 0, input is global coordinate, transform to relative coordinate
            // do it here to prevent overflow
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                pointer = &root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(descendant_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantLevelDataTypeList>*>(
                        mem_pool.getChunkPtr(
                            accessor.descendant_level_pointers[Level - 1]->data.index
                        )
                    );
                pointer = &array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }
        }

        if constexpr (Level < MaxLevel) {
            if (pointer->flags == sparse_helper::NextLevelDataFlag::IS_EMPTY ||
                pointer->flags == sparse_helper::NextLevelDataFlag::IS_VALUE) {
                accessor.clearAfterLastValidLevel();
                return false;
            }
            accessor.level = Level;
            accessor.descendant_level_pointers[Level] = pointer;

            return _access<MaxLevel, true, Level + 1>(next_relative_coord, {}, accessor);
        } else {
            // Level == MaxLevel
            accessor.level = Level;
            accessor.descendant_level_pointers[Level] = pointer;
            return true;
        }
    }

    /**
     * @brief Search by coord till MaxLevel specified.
     * @return True if found (including searching an intermediate level value).
     * False if not found.
     */
    template <RIndex MaxLevel, bool IsCoordDifferent = false, RIndex Level = 0>
    HOST_DEVICE_INLINE_QUALIFIER bool _search(
        const RCoord3& level_relative_coord,
        const RCoord3& accessor_level_relative_coord,
        RS_Sparse3DMapAccessor& accessor
    ) {
        static_assert(MaxLevel < LevelNum, "MaxLevel must be less than LevelNum");

        // Note: next level  size list exclude the root level, so no need to increase to Level + 1
        static constexpr RSize NextLevelXSize = refl::trait::get_t<Level, NextLevelSizeList>::XSize;
        static constexpr RSize NextLevelYSize = refl::trait::get_t<Level, NextLevelSizeList>::YSize;
        static constexpr RSize NextLevelZSize = refl::trait::get_t<Level, NextLevelSizeList>::ZSize;
        static constexpr RSize NextLevelXLog2 = logTwoOf(NextLevelXSize);
        static constexpr RSize NextLevelYLog2 = logTwoOf(NextLevelYSize);
        static constexpr RSize NextLevelZLog2 = logTwoOf(NextLevelZSize);

        RCoord3 current_value_coord =
            level_relative_coord >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
        RCoord3 next_relative_coord =
            level_relative_coord &
            RCoord3(NextLevelXSize - 1, NextLevelYSize - 1, NextLevelZSize - 1);

        NextLevelDataType* pointer;
        if constexpr (!IsCoordDifferent && Level < MaxLevel) {
            RCoord3 acc_current_value_coord =
                accessor_level_relative_coord >>
                RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
            RCoord3 acc_next_relative_coord =
                accessor_level_relative_coord &
                RCoord3(NextLevelXSize - 1, NextLevelYSize - 1, NextLevelZSize - 1);

            pointer = accessor.descendant_level_pointers[Level];

            if (pointer != nullptr && current_value_coord == acc_current_value_coord &&
                pointer->flags == sparse_helper::NextLevelDataFlag::IS_INDEX) {
                // Accessor is still the same for this level, keep level pointers the same
                accessor.level = Level;
                return _search<MaxLevel, false, Level + 1>(
                    next_relative_coord, acc_next_relative_coord, accessor
                );
            }
        }

        // Accessor is different for this level, or the pointer is pointing to
        // a value and not an index to mempool in next level, update level pointers
        if constexpr (sparse_helper::is_hash_map_level<
                          refl::trait::get_t<Level, LevelConfigList>>::value) {
            typename RootLevelDataType::KeyType key{current_value_coord};
            // Hashmap must be root level, so no branch for other levels
            pointer = root.find(key);
            if (pointer == nullptr) {
                accessor.clearAfterLastValidLevel();
                return false;
            }

        } else {
            // is array
            // For level 0, input is global coordinate, transform to relative coordinate
            // do it here to prevent overflow
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                pointer = &root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(descendant_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantLevelDataTypeList>*>(
                        mem_pool.getChunkPtr(
                            accessor.descendant_level_pointers[Level - 1]->data.index
                        )
                    );
                pointer = &array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }
        }

        if constexpr (Level < MaxLevel) {
            if (pointer->flags == sparse_helper::NextLevelDataFlag::IS_EMPTY) {
                accessor.clearAfterLastValidLevel();
                return false;
            }
            accessor.level = Level;
            accessor.descendant_level_pointers[Level] = pointer;
            // Stop search here
            if (pointer->flags == sparse_helper::NextLevelDataFlag::IS_VALUE) {
                return true;
            }
            return _search<MaxLevel, true, Level + 1>(next_relative_coord, {}, accessor);
        } else {
            // (accessible, but we need to check whether a value exists)
            // Level == MaxLevel
            accessor.level = Level;
            accessor.descendant_level_pointers[Level] = pointer;
            return pointer->flags == sparse_helper::NextLevelDataFlag::IS_VALUE;
        }
    }

    template <RIndex NextLevel>
    HOST_DEVICE_INLINE_QUALIFIER bool
    _initNextLevelDataAsIndex(NextLevelDataType* pointer, RIndex acquire_seed) {
        RIndex acquired_chunk_index =
            rs_device_std::get<NextLevel - 1>(descendant_mem_pools).acquire(acquire_seed);
        if (acquired_chunk_index != RS_NULL_INDEX) {
            pointer->flags = sparse_helper::NextLevelDataFlag::IS_INDEX;
            pointer->data.index = acquired_chunk_index;
            return true;
        }
        return false;
    }

    template <RIndex NextLevel>
    HOST_DEVICE_INLINE_QUALIFIER void
    _fillNextLevelDataWithValue(NextLevelDataType* pointer, const ValueType& value) {
        auto array_ptr =
            static_cast<refl::trait::get_t<NextLevel - 1, DescendantLevelDataTypeList>*>(
                rs_device_std::get<NextLevel - 1>(descendant_mem_pools)
                    .getChunkPtr(pointer->data.index)
            );
        for (RSize i = 0;
             i < refl::trait::get_t<NextLevel - 1, DescendantLevelDataTypeList>::ElemNum;
             i++) {
            auto next_level_data_ptr = &array_ptr->get(i);
            next_level_data_ptr->flags = sparse_helper::NextLevelDataFlag::IS_VALUE;
            next_level_data_ptr->data.value = value;
        }
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_SPARSE_H
