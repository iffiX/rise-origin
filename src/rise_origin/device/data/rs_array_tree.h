// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_ARRAY_TREE_H
#define RS_ARRAY_TREE_H

#include <refl.hpp>

#include "rise_origin/device/rs_dev_util.h"
#include "rise_origin/device/rs_static_construct.h"
#include "rise_origin/utils/math/rs_coord.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_array.h"
#include "rs_bitmask.h"
#include "rs_data_util.h"
#include "rs_mempool.h"

namespace rise {

namespace device {

template <RSize XSize_ = 0, RSize YSize_ = 0, RSize ZSize_ = 0>
struct RS_ArrayTree3DSize {
    static constexpr RSize XSize = XSize_;
    static constexpr RSize YSize = YSize_;
    static constexpr RSize ZSize = ZSize_;
};

template <RSize XSize = 0, RSize YSize = 0, RSize ZSize = 0>
struct RS_ArrayTree3DLevel {
    using Size = RS_ArrayTree3DSize<XSize, YSize, ZSize>;
    static_assert(XSize == 0 || isPowerOfTwo(XSize), "XSize must be zero or power of two");
    static_assert(YSize == 0 || isPowerOfTwo(YSize), "YSize must be zero or power of two");
    static_assert(ZSize == 0 || isPowerOfTwo(ZSize), "ZSize must be zero or power of two");
    static constexpr bool isEmpty() { return XSize == 0 || YSize == 0 || ZSize == 0; }
};

namespace array_tree_helper {
/// @brief Helper to create a type list of level configurations.
template <typename... LevelConfigs>
using LevelConfigList = refl::util::type_list<LevelConfigs...>;

// Helper to identify level types.
template <typename T>
struct is_array_level : std::false_type {};

template <RSize XSize, RSize YSize, RSize ZSize>
struct is_array_level<RS_ArrayTree3DLevel<XSize, YSize, ZSize>> : std::true_type {};

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
    using MultipliedSize = RS_ArrayTree3DSize<
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
    static_assert(is_array_level<FirstLevelConfig>::value, "Level must be an array level config");
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

}  // namespace array_tree_helper

/**
 *
 */
template <typename ValueType, typename... LevelConfigs>
class RS_ArrayTree3DMap {
   public:
    using LevelConfigList = refl::util::type_list<LevelConfigs...>;
    using RootLevelConfig = typename refl::trait::get<0, LevelConfigList>::type;
    using DescendantLevelConfigList = typename refl::trait::skip<1, LevelConfigList>::type;
    // Number of levels
    static constexpr RSize LevelNum = sizeof...(LevelConfigs);

    static_assert(sizeof...(LevelConfigs) > 1, "At least two levels must be specified");
    array_tree_helper::ValidateLevelConfigs<0, LevelConfigList> _{};

    // Helper to compute internal level sizes
    template <typename DescendantLevelConfigList>
    using GetDescendantSizeList =
        refl::trait::reverse_t<typename array_tree_helper::PrefixMultiplySizeHelper<
            refl::util::type_list<>,
            RS_ArrayTree3DSize<1, 1, 1>,
            refl::trait::reverse_t<DescendantLevelConfigList>>::Result>;

    using DescendantLevelSizeList = GetDescendantSizeList<DescendantLevelConfigList>;

    // Next level sizes, the number is one more level than internal levels since we
    // append (1, 1, 1) to the end, the total number is equal to LevelNum
    using NextLevelSizeList =
        refl::trait::append_t<RS_ArrayTree3DSize<1, 1, 1>, DescendantLevelSizeList>;

    template <typename LevelConfig>
    struct DescendantNonLeafArray3D {
        static constexpr RSize XSize = LevelConfig::Size::XSize;
        static constexpr RSize YSize = LevelConfig::Size::YSize;
        static constexpr RSize ZSize = LevelConfig::Size::ZSize;

        // The indices array is used to store claimer thread indices in the claim() pass
        // It is then reused to store next mempool indices in the activate() pass
        using ArrayType = RS_StaticColumnMajorTwoAlignedArray<RIndex, XSize, YSize, ZSize>;
    };

    template <typename LevelConfig>
    struct LeafArray3D {
        static constexpr RSize XSize = LevelConfig::Size::XSize;
        static constexpr RSize YSize = LevelConfig::Size::YSize;
        static constexpr RSize ZSize = LevelConfig::Size::ZSize;

        struct ArrayType {
            static constexpr RSize ElemNum = XSize * YSize * ZSize;
            RS_StaticColumnMajorTwoAlignedArray<bool, XSize, YSize, ZSize> is_inited;
            RS_StaticColumnMajorTwoAlignedArray<ValueType, XSize, YSize, ZSize> values;
        };
    };

    template <size_t Index, typename LevelConfig>
    struct DescendantArrayType {
        using type = std::conditional_t<
            Index == LevelNum - 2,
            typename LeafArray3D<LevelConfig>::ArrayType,
            typename DescendantNonLeafArray3D<LevelConfig>::ArrayType>;
    };

    template <size_t Index, typename LevelConfig>
    struct DescendantMemPoolType {
        using type = std::conditional_t<
            Index == LevelNum - 2,
            RS_MemArrayPool<sizeof(typename LeafArray3D<LevelConfig>::ArrayType)>,
            RS_MemArrayPool<sizeof(typename DescendantNonLeafArray3D<LevelConfig>::ArrayType)>>;
    };

    template <typename>
    struct AsArrayTreeDescendantTuple;

    template <template <typename...> typename T2, typename... Ts>
    struct AsArrayTreeDescendantTuple<T2<Ts...>> {
        using type = rs_device_std::tuple<Ts...>;
    };

    template <typename T2>
    using AsArrayTreeDescendantTupleType = typename AsArrayTreeDescendantTuple<T2>::type;

    using DescendantArrayTypeList =
        refl::trait::map_with_index_t<DescendantArrayType, DescendantLevelConfigList>;

    using DescendantMemPoolTypeList =
        refl::trait::map_with_index_t<DescendantMemPoolType, DescendantLevelConfigList>;

    // Accessor class to cache value pointers for each level
    class RS_ArrayTree3DMapAccessor {
       public:
        RCoord3 coord{RS_NULL_SIGNED_INDEX, RS_NULL_SIGNED_INDEX, RS_NULL_SIGNED_INDEX};
        RIndex level = RS_NULL_INDEX;
        rs_device_std::array<RIndex, LevelNum - 1> descendant_level_indices;
        bool* is_inited_pointer = nullptr;
        ValueType* value_pointer = nullptr;

        HOST_DEVICE_INLINE_QUALIFIER RS_ArrayTree3DMapAccessor() {
            static constexpr RIndex null_index = RS_NULL_INDEX;
            descendant_level_indices.fill(null_index);
        };

        HOST_DEVICE_INLINE_QUALIFIER void clear() {
            static constexpr RIndex null_index = RS_NULL_INDEX;
            coord = RCoord3(RS_NULL_SIGNED_INDEX, RS_NULL_SIGNED_INDEX, RS_NULL_SIGNED_INDEX);
            level = RS_NULL_INDEX;
            descendant_level_indices.fill(null_index);
            is_inited_pointer = nullptr;
            value_pointer = nullptr;
        }

        HOST_DEVICE_INLINE_QUALIFIER void clearAfterLastValidLevel() {
            if (level == RS_NULL_INDEX)
                clear();
            else {
                for (RIndex i = level + 1; i < LevelNum - 1; ++i)
                    descendant_level_indices[i] = RS_NULL_INDEX;
                if (level != LevelNum - 1) {
                    is_inited_pointer = nullptr;
                    value_pointer = nullptr;
                }
            }
        }

        HOST_DEVICE_INLINE_QUALIFIER bool isNull() { return level == RS_NULL_INDEX; }

        HOST_DEVICE_INLINE_QUALIFIER bool isComplete() { return level == LevelNum - 1; }

        HOST_DEVICE_INLINE_QUALIFIER bool empty() { return !isComplete() || !*is_inited_pointer; }

        HOST_DEVICE_INLINE_QUALIFIER void set(const ValueType& value) {
            if (isComplete()) {
                *is_inited_pointer = true;
                *value_pointer = value;
            }
        }

        HOST_DEVICE_INLINE_QUALIFIER ValueType* get() {
            if (level == RS_NULL_INDEX) return nullptr;
            return value_pointer;
        }
    };

    /*****************************************************************
     * Data
     *****************************************************************/
    // Bound coordinate of the all positive quadrant
    RCoord3 pos_bound{0, 0, 0};

    // Root level data
    RS_Array<RIndex, 3, false> root;
    AsArrayTreeDescendantTupleType<DescendantMemPoolTypeList> internal_mem_pools;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_ArrayTree3DMap() = default;

    /**
     * @brief Construct a new array tree 3d map.
     * @param provider The memory provider to use.
     */
    explicit RS_ArrayTree3DMap(RS_MemoryProvider* provider) { init(provider); }

    /**
     * @brief Initialize the array tree 3d map.
     * @param provider The memory provider to use.
     */
    void init(RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        pos_bound = RCoord3::zero();
        // construct parameters are provided by level configs
        root.init(
            {RootLevelConfig::Size::XSize,
             RootLevelConfig::Size::YSize,
             RootLevelConfig::Size::ZSize},
            provider
        );
        array_tree_helper::for_each_in_tuple(internal_mem_pools, [&provider](auto& level) {
            level.init(provider);
        });
        _updateBound();
        clear();
    }

    HOST_DEVICE_INLINE_QUALIFIER RS_ArrayTree3DMap(const RS_ArrayTree3DMap& other)
        : pos_bound(other.pos_bound),
          root(other.root),
          internal_mem_pools(other.internal_mem_pools) {}

    /**
     * @brief Frees the memory used by the array tree 3d map.
     */
    void free() {
        pos_bound = RCoord3::zero();
        root.free();
        array_tree_helper::for_each_in_tuple(internal_mem_pools, [](auto& pool) { pool.free(); });
    }

    /**
     * @brief Resizes the root array of the array tree 3d map.
     * @note Currently it doesn't provide spatial consistency and querying original data will fail.
     * @param new_shape The new shape of the root array, each dimension should be a power of two.
     * @param copy Whether to copy the existing root array data to the new array.
     */
    void resize(std::array<RSize, 3> new_shape, bool copy = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        if ((new_shape[0] > 0 && !isPowerOfTwo(new_shape[0])) ||
            (new_shape[1] > 0 && !isPowerOfTwo(new_shape[1])) ||
            (new_shape[2] > 0 && !isPowerOfTwo(new_shape[2]))) {
            RS_THROW(RS_InvalidArgumentException("Root shape of array must be powers of 2"));
        }
        root.resize(new_shape, copy);
        _updateBound();
    }

    /**
     * @brief Resizes the memory pool corresponding to the internal level of the array tree 3d map.
     * @param level The level of the internal memory pool to resize. Should be from 1 to level num
     * - 1.
     * @param mem_pool_size The new size of the memory pool.
     */
    void resizeMemPool(RIndex level, RSize mem_pool_size, bool copy = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        if (level < 1 || level >= LevelNum) {
            RS_THROW(RS_InvalidArgumentException(
                "Resized mem pool level must be between 1 and level num - 1"
            ));
        }
        loop<1, LevelNum>([&](auto i_const) {
            constexpr RIndex i = i_const.value;
            if (i == level) {
                auto& mem_pool = rs_device_std::get<i - 1>(internal_mem_pools);
                mem_pool.resize(mem_pool_size, copy);
            }
        });
    }

    /**
     * @brief Clears all used entries in the array tree 3d map.
     */
    void clear(bool sync = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        root.fill(RS_NULL_INDEX, sync);
        array_tree_helper::for_each_in_tuple(internal_mem_pools, [&](auto& pool) {
            pool.clear(sync);
        });
    }

    /**
     * @brief Copies the array tree 3d map, resize as needed.
     * @param array_tree The array tree 3d map to copy from.
     */
    void copy(const RS_ArrayTree3DMap& array_tree) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(array_tree.provider() != nullptr);

        pos_bound = array_tree.pos_bound;
        root.copy(array_tree.root);
        array_tree_helper::for_each_in_zip_tuples(
            internal_mem_pools, array_tree.internal_mem_pools, [](auto& pool, auto& other_pool) {
                pool.copy(other_pool);
            }
        );
    }

    /**
     * @brief Moves data to the specified MemoryProvider and returns a new array tree 3d map.
     */
    RS_ArrayTree3DMap move(RS_MemoryProvider* provider) const {
        RS_DATA_EXPECT_INIT(this->provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        RS_ArrayTree3DMap new_array_tree;
        new_array_tree.pos_bound = pos_bound;
        new_array_tree.root = root.move(provider);
        array_tree_helper::for_each_in_zip_tuples(
            new_array_tree.internal_mem_pools,
            internal_mem_pools,
            [&provider](auto& pool, auto& other_pool) { pool = other_pool.move(provider); }
        );
        return new_array_tree;
    }

    [[nodiscard]] RS_MemoryProvider* provider() const { return root.provider(); }

    [[nodiscard]] RS_DeviceLocation device() const { return root.device(); }

    /// @brief Returns the base size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RBigSize baseBytes() const { return sizeof(*this); }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RBigSize extraBytes() const {
        RBigSize extra_bytes = 0;
        extra_bytes += root.extraBytes();
        array_tree_helper::for_each_in_tuple_hd(internal_mem_pools, [&extra_bytes](auto& pool) {
            extra_bytes += pool.extraBytes();
        });
        return extra_bytes;
    }

    /// @brief Returns the maximum valid coordinate of the array tree 3d map.
    HOST_DEVICE_INLINE_QUALIFIER RCoord3 maxCoord() const { return pos_bound - 1; }

    /// @brief Returns the minimum valid coordinate of the array tree 3d map.
    HOST_DEVICE_INLINE_QUALIFIER RCoord3 minCoord() const { return -pos_bound; }

    /// @brief Returns the size of the array tree 3d map at the specified level.
    /// @param level The level of the array tree 3d map.
    /// @return The size of the array tree 3d map at the specified level. when level = 0,
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

    /// @brief Returns the size of the array tree 3d map at the specified level.
    /// @tparam Level The level of the array tree 3d map.
    /// @return The size of the array tree 3d map at the specified level. when Level = 0,
    /// returns the max bound size of root, when Level = LevelNum - 1, returns the
    /// size of the last level, when Level = LevelNum, returns 1 for each dimension.
    template <RIndex Level>
    HOST_DEVICE_INLINE_QUALIFIER RCoord3 getLevelSize() const {
        static_assert(Level <= LevelNum, "Level must be less or equal to LevelNum");
        if constexpr (Level == 0) {
            return pos_bound << 1;
        } else {
            return RCoord3(
                refl::trait::get_t<Level - 1, NextLevelSizeList>::XSize,
                refl::trait::get_t<Level - 1, NextLevelSizeList>::YSize,
                refl::trait::get_t<Level - 1, NextLevelSizeList>::ZSize
            );
        }
    }

    /// @brief Returns the region bound of the array tree 3d map at the specified level.
    /// @tparam Level The level of the array tree 3d map.
    /// @param position The position of the array tree 3d map.
    /// @param min_bound The minimum bound of the array tree 3d map.
    /// @param max_bound The maximum bound of the array tree 3d map.
    template <RIndex Level>
    HOST_DEVICE_INLINE_QUALIFIER void
    getLevelRegionBound(const RCoord3& position, RCoord3& min_bound, RCoord3& max_bound) const {
        static_assert(Level <= LevelNum, "Level must be less or equal to LevelNum");
        if constexpr (Level == 0) {
            min_bound = -pos_bound;
            max_bound = pos_bound;
        } else {
            min_bound = position & ~(getLevelSize<Level>() - 1);
            max_bound = min_bound + getLevelSize<Level>();
        }
    }

    /// @brief Returns whether the coordinate is in the bound of the array tree 3d map.
    HOST_DEVICE_INLINE_QUALIFIER bool isInBound(const RCoord3& coord) const {
        return (coord.x >= -pos_bound[0] && coord.x < pos_bound[0]) &&
               (coord.y >= -pos_bound[1] && coord.y < pos_bound[1]) &&
               (coord.z >= -pos_bound[2] && coord.z < pos_bound[2]);
    }

    /// @brief Returns the accessor of the array tree 3d map.
    HOST_DEVICE_INLINE_QUALIFIER RS_ArrayTree3DMapAccessor getAccessor() {
        return RS_ArrayTree3DMapAccessor();
    }

    /**
     * @brief Claims a level at the specified coordinate, writes the thread index to
     * the level. The thread that claims the level will be responsible for the memory
     * allocation of the level.
     *
     * @tparam MaxLevel The maximum level to claim. Valid range is [0, LevelNum - 2].
     * @tparam CheckBound Whether to check the bound of the array tree 3d map.
     * @param coord The coordinate of the level to claim.
     * @param thread_index The thread index to claim the level.
     * @return True if the level can be claimed (the previous level is allocated),
     * false otherwise.
     */
    template <RIndex MaxLevel, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool claim(const RCoord3& coord, RIndex thread_index) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        return _claim<MaxLevel>(coord, thread_index);
    }

    /**
     * @brief Allocates a level at the specified coordinate, the thread with the same index
     * as the claimer will be responsible for the memory allocation of the level.
     *
     * @tparam MaxLevel The maximum level to allocate. Valid range is [0, LevelNum - 2].
     * @tparam CheckBound Whether to check the bound of the array tree 3d map.
     * @param coord The coordinate of the level to allocate.
     * @param thread_index The thread index to allocate the level.
     * @return True if the level can be allocated (the previous level is allocated),
     * false otherwise.
     */
    template <RIndex MaxLevel, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool allocate(const RCoord3& coord, RIndex thread_index) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        return _allocate<MaxLevel>(coord, thread_index);
    }

    /**
     * @brief Returns whether the level at the specified coordinate is empty.
     * @tparam MaxLevel The maximum level to check. Valid range is [0, LevelNum - 1].
     * @tparam CheckBound Whether to check the bound of the array tree 3d map.
     * @param coord The coordinate of the level to check.
     * @return True if the level is empty, false otherwise.
     */
    template <RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool isLevelEmpty(const RCoord3& coord) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        auto accessor = getAccessor();
        return _isLevelEmpty<MaxLevel, true>(coord, {}, accessor);
    }

    /**
     * @brief Returns whether the level at the specified coordinate is empty.
     * @tparam MaxLevel The maximum level to check. Valid range is [0, LevelNum - 1].
     * @tparam CheckBound Whether to check the bound of the array tree 3d map.
     * @param coord The coordinate of the level to check.
     * @param accessor The accessor of the array tree 3d map.
     * @return True if the level is empty, false otherwise.
     */
    template <RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool
    isLevelEmpty(const RCoord3& coord, RS_ArrayTree3DMapAccessor& accessor) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        accessor.level = RS_NULL_INDEX;
        if (!_isLevelEmpty<MaxLevel, false>(coord, accessor.coord, accessor)) {
            accessor.coord = coord;
            return false;
        }
        return true;
    }

    /**
     * @brief Inserts a value at the last level of the array tree 3d map.
     * @tparam CheckBound Whether to check the bound of the array tree 3d map.
     * @param coord The coordinate of the value to insert.
     * @param value The value to insert.
     * @return True if the value is inserted successfully, false otherwise.
     */
    template <bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool insert(const RCoord3& coord, const ValueType& value) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        auto accessor = getAccessor();
        if (_access<true>(coord, {}, accessor)) {
            // No need to update coord of accessor, just set value
            accessor.set(value);
            return true;
        }
        return false;
    }

    /**
     * @brief Inserts a value at the last level of the array tree 3d map with a cached accessor.
     * @tparam CheckBound Whether to check the bound of the array tree 3d map.
     * @param coord The coordinate of the value to insert.
     * @param value The value to insert.
     * @param accessor The accessor of the array tree 3d map.
     * @return True if the value is inserted successfully, false otherwise.
     */
    template <bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool
    insert(const RCoord3& coord, const ValueType& value, RS_ArrayTree3DMapAccessor& accessor) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        accessor.level = RS_NULL_INDEX;
        if (_access<false>(coord, accessor.coord, accessor)) {
            accessor.coord = coord;
            accessor.set(value);
            return true;
        }
        return false;
    }

    /**
     * @brief Finds a value in the last level of the array tree 3d map.
     * @tparam CheckBound Whether to check the bound of the array tree 3d map.
     * @param coord The coordinate of the value to find.
     * @return The value at the specified coordinate, nullptr if the value is not found.
     */
    template <bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER ValueType* find(const RCoord3& coord) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return nullptr;
        }
        auto accessor = getAccessor();
        if (_access<true>(coord, {}, accessor) && !accessor.empty()) {
            return accessor.get();
        }
        return nullptr;
    }

    /**
     * @brief Finds a value in the last level of the array tree 3d map with a cached accessor.
     * @tparam CheckBound Whether to check the bound of the array tree 3d map.
     * @param coord The coordinate of the value to find.
     * @param accessor The accessor of the array tree 3d map.
     * @return The value at the specified coordinate, nullptr if the value is not found.
     */
    template <bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER ValueType*
    find(const RCoord3& coord, RS_ArrayTree3DMapAccessor& accessor) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return nullptr;
        }
        accessor.level = RS_NULL_INDEX;
        if (_access(coord, accessor.coord, accessor)) {
            accessor.coord = coord;
            if (!accessor.empty()) return accessor.get();
            return nullptr;
        }
        return nullptr;
    }

   private:
    /// @brief Updates the bound of the array tree 3d map.
    void _updateBound() {
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
    }

    /**
     * @brief Claims a level at the specified coordinate, writes the thread index to
     * the level. The thread that claims the level will be responsible for the memory
     * allocation of the level.
     *
     * @tparam MaxLevel The maximum level to claim.
     * @tparam Level The current level to claim.
     * @param level_relative_coord The coordinate of the level to claim.
     * @param thread_index The thread index to claim the level.
     * @param parent_mempool_index The memory pool page index of the parent level.
     * @return True if the level can be claimed (the previous level is allocated),
     * false otherwise.
     */
    template <RIndex MaxLevel, RIndex Level = 0>
    HOST_DEVICE_INLINE_QUALIFIER bool _claim(
        const RCoord3& level_relative_coord,
        RIndex thread_index,
        RIndex parent_mempool_index = RS_NULL_INDEX
    ) {
        static_assert(MaxLevel < LevelNum - 1, "MaxLevel of _claim must be less than LevelNum - 1");

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

        // For level 0, input is global coordinate, transform to relative coordinate
        // do it here to prevent overflow
        if constexpr (Level == MaxLevel) {
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }] = thread_index;
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                        mem_pool.getChunkPtr(parent_mempool_index)
                    );
                array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                ) = thread_index;
            }
            return true;
        } else {
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                parent_mempool_index = root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                        mem_pool.getChunkPtr(parent_mempool_index)
                    );
                parent_mempool_index = array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }
            if (parent_mempool_index == RS_NULL_INDEX) return false;
            return _claim<MaxLevel, Level + 1>(
                next_relative_coord, thread_index, parent_mempool_index
            );
        }
    }

    /**
     * @brief Allocates a level at the specified coordinate, the thread with the same index
     * as the claimer will be responsible for the memory allocation of the level.
     *
     * @tparam MaxLevel The maximum level to allocate.
     * @tparam Level The current level to allocate.
     * @param level_relative_coord The coordinate of the level to allocate.
     * @param thread_index The thread index to allocate the level.
     * @param parent_mempool_index The memory pool page index of the parent level.
     * @return True if the level can be allocated (the previous level is allocated),
     * false otherwise.
     */
    template <RIndex MaxLevel, RIndex Level = 0>
    HOST_DEVICE_INLINE_QUALIFIER bool _allocate(
        const RCoord3& level_relative_coord,
        RIndex thread_index,
        RIndex parent_mempool_index = RS_NULL_INDEX
    ) {
        static_assert(
            MaxLevel < LevelNum - 1, "MaxLevel of _allocate must be less than LevelNum - 1"
        );

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

        // For level 0, input is global coordinate, transform to relative coordinate
        // do it here to prevent overflow
        if constexpr (Level == MaxLevel) {
            RIndex* next_index;
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                next_index = &root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];

            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                        mem_pool.getChunkPtr(parent_mempool_index)
                    );
                next_index = &array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }
            if (*next_index == thread_index) {
                _initNextLevel<Level>(
                    *next_index,
                    current_value_coord.x * 73856093 ^ current_value_coord.y * 19349663 ^
                        current_value_coord.z * 83492791
                );
            }
            return true;
        } else {
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                parent_mempool_index = root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                        mem_pool.getChunkPtr(parent_mempool_index)
                    );
                parent_mempool_index = array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }
            if (parent_mempool_index == RS_NULL_INDEX) return false;
            return _allocate<MaxLevel, Level + 1>(
                next_relative_coord, thread_index, parent_mempool_index
            );
        }
    }

    /**
     * @brief Returns whether the level at the specified coordinate is empty.
     * @tparam MaxLevel The maximum level to check.
     * @tparam IsCoordDifferent Whether the coordinate is different from the accessor coordinate.
     * @tparam Level The current level to check.
     * @param level_relative_coord The coordinate of the level to check.
     * @param accessor_level_relative_coord The coordinate of the accessor level.
     * @param accessor The accessor of the array tree 3d map.
     * @return True if the level is empty, false otherwise.
     */
    template <RIndex MaxLevel, bool IsCoordDifferent = false, RIndex Level = 0>
    HOST_DEVICE_INLINE_QUALIFIER bool _isLevelEmpty(
        const RCoord3& level_relative_coord,
        const RCoord3& accessor_level_relative_coord,
        RS_ArrayTree3DMapAccessor& accessor
    ) {
        static_assert(MaxLevel < LevelNum, "MaxLevel of _isLevelEmpty must be less than LevelNum");

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

        if constexpr (!IsCoordDifferent && Level < LevelNum - 1) {
            RCoord3 acc_current_value_index =
                accessor_level_relative_coord >>
                RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
            RCoord3 acc_next_relative_coord =
                accessor_level_relative_coord &
                RCoord3(NextLevelXSize - 1, NextLevelYSize - 1, NextLevelZSize - 1);
            if (accessor.descendant_level_indices[Level] != RS_NULL_INDEX &&
                current_value_coord == acc_current_value_index) {
                return _isLevelEmpty<MaxLevel, false, Level + 1>(
                    next_relative_coord, acc_next_relative_coord, accessor
                );
            }
        }

        if constexpr (Level == LevelNum - 1) {
            auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
            auto array_ptr = static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                mem_pool.getChunkPtr(accessor.descendant_level_indices[Level - 1])
            );
            accessor.level = Level;
            accessor.is_inited_pointer = &array_ptr->is_inited[{
                static_cast<RSize>(current_value_coord[0]),
                static_cast<RSize>(current_value_coord[1]),
                static_cast<RSize>(current_value_coord[2])
            }];
            accessor.value_pointer = &array_ptr->values[{
                static_cast<RSize>(current_value_coord[0]),
                static_cast<RSize>(current_value_coord[1]),
                static_cast<RSize>(current_value_coord[2])
            }];
            return !*accessor.is_inited_pointer;
        } else {
            RIndex current_index;
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                current_index = root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                        mem_pool.getChunkPtr(accessor.descendant_level_indices[Level - 1])
                    );
                current_index = array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }

            if (current_index == RS_NULL_INDEX) {
                accessor.clearAfterLastValidLevel();
                return true;
            }

            accessor.level = Level;
            accessor.descendant_level_indices[Level] = current_index;

            if constexpr (Level < MaxLevel) {
                return _isLevelEmpty<MaxLevel, true, Level + 1>(next_relative_coord, {}, accessor);
            } else {
                return false;
            }
        }
    }

    /**
     * @brief Accesses the value of the last level at the specified coordinate.
     * @tparam IsCoordDifferent Whether the coordinate is different from the accessor coordinate.
     * @tparam Level The current level to access.
     * @param level_relative_coord The coordinate of the level to access.
     * @param accessor_level_relative_coord The coordinate of the accessor level.
     * @param accessor The accessor of the array tree 3d map.
     * @return True if the value can be accessed, false otherwise.
     */
    template <bool IsCoordDifferent = false, RIndex Level = 0>
    HOST_DEVICE_INLINE_QUALIFIER bool _access(
        const RCoord3& level_relative_coord,
        const RCoord3& accessor_level_relative_coord,
        RS_ArrayTree3DMapAccessor& accessor
    ) {
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

        if constexpr (!IsCoordDifferent && Level < LevelNum - 1) {
            RCoord3 acc_current_value_index =
                accessor_level_relative_coord >>
                RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
            RCoord3 acc_next_relative_coord =
                accessor_level_relative_coord &
                RCoord3(NextLevelXSize - 1, NextLevelYSize - 1, NextLevelZSize - 1);
            if (accessor.descendant_level_indices[Level] != RS_NULL_INDEX &&
                current_value_coord == acc_current_value_index) {
                return _access<false, Level + 1>(
                    next_relative_coord, acc_next_relative_coord, accessor
                );
            }
        }

        if constexpr (Level == LevelNum - 1) {
            auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
            auto array_ptr = static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                mem_pool.getChunkPtr(accessor.descendant_level_indices[Level - 1])
            );
            accessor.level = Level;
            accessor.is_inited_pointer = &array_ptr->is_inited[{
                static_cast<RSize>(current_value_coord[0]),
                static_cast<RSize>(current_value_coord[1]),
                static_cast<RSize>(current_value_coord[2])
            }];
            accessor.value_pointer = &array_ptr->values[{
                static_cast<RSize>(current_value_coord[0]),
                static_cast<RSize>(current_value_coord[1]),
                static_cast<RSize>(current_value_coord[2])
            }];
            return true;
        } else {
            RIndex current_index;
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                current_index = root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                        mem_pool.getChunkPtr(accessor.descendant_level_indices[Level - 1])
                    );
                current_index = array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }

            if (current_index == RS_NULL_INDEX) {
                accessor.clearAfterLastValidLevel();
                return false;
            }

            accessor.level = Level;
            accessor.descendant_level_indices[Level] = current_index;

            return _access<true, Level + 1>(next_relative_coord, {}, accessor);
        }
    }

    template <RIndex Level>
    HOST_DEVICE_INLINE_QUALIFIER bool _initNextLevel(RIndex& next_index, RIndex acquire_seed) {
        next_index = rs_device_std::get<Level>(internal_mem_pools).acquire(acquire_seed);
        if (next_index != RS_NULL_INDEX) {
            auto& array = *static_cast<refl::trait::get_t<Level, DescendantArrayTypeList>*>(
                rs_device_std::get<Level>(internal_mem_pools).getChunkPtr(next_index)
            );
            if constexpr (Level == LevelNum - 2) {
                for (RSize i = 0; i < refl::trait::get_t<Level, DescendantArrayTypeList>::ElemNum;
                     i++) {
                    array.is_inited[i] = false;
                }
            } else {
                for (RSize i = 0; i < refl::trait::get_t<Level, DescendantArrayTypeList>::ElemNum;
                     i++) {
                    array[i] = RS_NULL_INDEX;
                }
            }

            return true;
        }
        return false;
    }
};

template <typename ValueType, RSize ValueNum_, typename... LevelConfigs>
class RS_ArrayTree3DMultiMap {
   public:
    using LevelConfigList = refl::util::type_list<LevelConfigs...>;
    using RootLevelConfig = typename refl::trait::get<0, LevelConfigList>::type;
    using DescendantLevelConfigList = typename refl::trait::skip<1, LevelConfigList>::type;
    // Number of levels
    static constexpr RSize LevelNum = sizeof...(LevelConfigs);
    static constexpr RSize ValueNum = ValueNum_;

    static_assert(sizeof...(LevelConfigs) > 1, "At least two levels must be specified");
    array_tree_helper::ValidateLevelConfigs<0, LevelConfigList> _{};

    // Helper to compute internal level sizes
    template <typename DescendantLevelConfigList>
    using GetDescendantSizeList =
        refl::trait::reverse_t<typename array_tree_helper::PrefixMultiplySizeHelper<
            refl::util::type_list<>,
            RS_ArrayTree3DSize<1, 1, 1>,
            refl::trait::reverse_t<DescendantLevelConfigList>>::Result>;

    using DescendantLevelSizeList = GetDescendantSizeList<DescendantLevelConfigList>;

    // Next level sizes, the number is one more level than internal levels since we
    // append (1, 1, 1) to the end, the total number is equal to LevelNum
    using NextLevelSizeList =
        refl::trait::append_t<RS_ArrayTree3DSize<1, 1, 1>, DescendantLevelSizeList>;

    template <typename LevelConfig>
    struct DescendantNonLeafArray3D {
        static constexpr RSize XSize = LevelConfig::Size::XSize;
        static constexpr RSize YSize = LevelConfig::Size::YSize;
        static constexpr RSize ZSize = LevelConfig::Size::ZSize;

        // The indices array is used to store claimer thread indices in the claim() pass
        // It is then reused to store next mempool indices in the activate() pass
        using ArrayType = RS_StaticColumnMajorTwoAlignedArray<RIndex, XSize, YSize, ZSize>;
    };

    template <typename LevelConfig>
    struct LeafArray3D {
        static constexpr RSize XSize = LevelConfig::Size::XSize;
        static constexpr RSize YSize = LevelConfig::Size::YSize;
        static constexpr RSize ZSize = LevelConfig::Size::ZSize;

        struct ArrayType {
            static constexpr RSize ElemNum = XSize * YSize * ZSize;
            RS_StaticColumnMajorTwoAlignedArray<RSize, XSize, YSize, ZSize> value_num;
            RS_StaticColumnMajorTwoAlignedArray<ValueType, ValueNum, XSize, YSize, ZSize> values;
        };
    };

    template <size_t Index, typename LevelConfig>
    struct DescendantArrayType {
        using type = std::conditional_t<
            Index == LevelNum - 2,
            typename LeafArray3D<LevelConfig>::ArrayType,
            typename DescendantNonLeafArray3D<LevelConfig>::ArrayType>;
    };

    template <size_t Index, typename LevelConfig>
    struct DescendantMemPoolType {
        using type = std::conditional_t<
            Index == LevelNum - 2,
            RS_MemArrayPool<sizeof(typename LeafArray3D<LevelConfig>::ArrayType)>,
            RS_MemArrayPool<sizeof(typename DescendantNonLeafArray3D<LevelConfig>::ArrayType)>>;
    };

    template <typename>
    struct AsArrayTreeDescendantTuple;

    template <template <typename...> typename T2, typename... Ts>
    struct AsArrayTreeDescendantTuple<T2<Ts...>> {
        using type = rs_device_std::tuple<Ts...>;
    };

    template <typename T2>
    using AsArrayTreeDescendantTupleType = typename AsArrayTreeDescendantTuple<T2>::type;

    using DescendantArrayTypeList =
        refl::trait::map_with_index_t<DescendantArrayType, DescendantLevelConfigList>;

    using DescendantMemPoolTypeList =
        refl::trait::map_with_index_t<DescendantMemPoolType, DescendantLevelConfigList>;

    // Accessor class to cache value pointers for each level
    class RS_ArrayTree3DMultiMapAccessor {
       public:
        RCoord3 coord{RS_NULL_SIGNED_INDEX, RS_NULL_SIGNED_INDEX, RS_NULL_SIGNED_INDEX};
        RIndex level = RS_NULL_INDEX;
        rs_device_std::array<RIndex, LevelNum - 1> descendant_level_indices;
        RSize* value_num_pointer = nullptr;
        ValueType* value_start_pointer = nullptr;

        HOST_DEVICE_INLINE_QUALIFIER RS_ArrayTree3DMultiMapAccessor() {
            static constexpr RIndex null_index = RS_NULL_INDEX;
            descendant_level_indices.fill(null_index);
        };

        HOST_DEVICE_INLINE_QUALIFIER void clear() {
            static constexpr RIndex null_index = RS_NULL_INDEX;
            coord = RCoord3(RS_NULL_SIGNED_INDEX, RS_NULL_SIGNED_INDEX, RS_NULL_SIGNED_INDEX);
            level = RS_NULL_INDEX;
            descendant_level_indices.fill(null_index);
            value_num_pointer = nullptr;
            value_start_pointer = nullptr;
        }

        HOST_DEVICE_INLINE_QUALIFIER void clearAfterLastValidLevel() {
            if (level == RS_NULL_INDEX)
                clear();
            else {
                for (RIndex i = level + 1; i < LevelNum - 1; ++i)
                    descendant_level_indices[i] = RS_NULL_INDEX;
                if (level != LevelNum - 1) {
                    value_num_pointer = nullptr;
                    value_start_pointer = nullptr;
                }
            }
        }

        HOST_DEVICE_INLINE_QUALIFIER bool isNull() { return level == RS_NULL_INDEX; }

        HOST_DEVICE_INLINE_QUALIFIER bool isComplete() { return level == LevelNum - 1; }

        HOST_DEVICE_INLINE_QUALIFIER bool empty() {
            return !isComplete() || *value_num_pointer == 0;
        }

        HOST_DEVICE_INLINE_QUALIFIER RSize size() {
            if (isComplete()) {
                return *value_num_pointer;
            }
            return 0;
        }

        HOST_DEVICE_INLINE_QUALIFIER bool insert(const ValueType& value) {
            if (!isComplete()) return false;
            while (true) {
                RIndex old_index = *value_num_pointer;
                if (old_index >= ValueNum) return false;  // Stack is full.

                // Attempt to increment the counter atomically.
                if (deviceAtomicCAS(value_num_pointer, old_index, old_index + 1) == old_index) {
                    value_start_pointer[old_index] = value;
                    return true;
                }
                // CAS failed due to concurrent modification; retry.
            }
        }

        HOST_DEVICE_INLINE_QUALIFIER void pop() {
            if (!isComplete()) return;
            while (true) {
                RIndex old_index = *value_num_pointer;
                if (old_index == 0) return;  // Stack is empty.

                // Attempt to decrement the counter atomically.
                if (deviceAtomicCAS(value_num_pointer, old_index, old_index - 1) == old_index) {
                    return;
                }
                // CAS failed; retry.
            }
        }

        HOST_DEVICE_INLINE_QUALIFIER ValueType* get(RIndex index) {
            assert(level != RS_NULL_INDEX && index < *value_num_pointer);
            return value_start_pointer + index;
        }
    };

    /*****************************************************************
     * Data
     *****************************************************************/
    // Bound coordinate of the all positive quadrant
    RCoord3 pos_bound{0, 0, 0};

    // Root level data
    RS_Array<RIndex, 3, false> root;
    AsArrayTreeDescendantTupleType<DescendantMemPoolTypeList> internal_mem_pools;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_ArrayTree3DMultiMap() = default;

    /**
     * @brief Construct a new array tree 3d multi map.
     * @param provider The memory provider to use.
     */
    explicit RS_ArrayTree3DMultiMap(RS_MemoryProvider* provider) { init(provider); }

    /**
     * @brief Initialize the array tree 3d multi map.
     * @param provider The memory provider to use.
     */
    void init(RS_MemoryProvider* provider) {
        RS_DATA_EXPECT_INIT(provider != nullptr);
        // construct parameters are provided by level configs
        pos_bound = RCoord3::zero();
        root.init(
            {RootLevelConfig::Size::XSize,
             RootLevelConfig::Size::YSize,
             RootLevelConfig::Size::ZSize},
            provider
        );
        array_tree_helper::for_each_in_tuple(internal_mem_pools, [&provider](auto& level) {
            level.init(provider);
        });
        _updateBound();
        clear();
    }

    HOST_DEVICE_INLINE_QUALIFIER RS_ArrayTree3DMultiMap(const RS_ArrayTree3DMultiMap& other)
        : pos_bound(other.pos_bound),
          root(other.root),
          internal_mem_pools(other.internal_mem_pools) {}

    /**
     * @brief Frees the memory used by the array tree 3d multi map.
     */
    void free() {
        pos_bound = RCoord3::zero();
        root.free();
        array_tree_helper::for_each_in_tuple(internal_mem_pools, [](auto& pool) { pool.free(); });
    }

    /**
     * @brief Resizes the root array of the array tree 3d multi map.
     * @note Currently it doesn't provide spatial consistency and querying original data will fail.
     * @param new_shape The new shape of the root array, each dimension should be a power of two.
     * @param copy Whether to copy the existing root array data to the new array.
     */
    void resize(std::array<RSize, 3> new_shape, bool copy = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        if ((new_shape[0] > 0 && !isPowerOfTwo(new_shape[0])) ||
            (new_shape[1] > 0 && !isPowerOfTwo(new_shape[1])) ||
            (new_shape[2] > 0 && !isPowerOfTwo(new_shape[2]))) {
            RS_THROW(RS_InvalidArgumentException("Root shape of array must be powers of 2"));
        }
        root.resize(new_shape, copy);
        _updateBound();
    }

    /**
     * @brief Resizes the memory pool corresponding to the internal level of the array tree 3d multi
     * map.
     * @param level The level of the internal memory pool to resize. Should be from 1 to level num
     * - 1.
     * @param mem_pool_size The new size of the memory pool.
     */
    void resizeMemPool(RIndex level, RSize mem_pool_size, bool copy = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        if (level < 1 || level >= LevelNum) {
            RS_THROW(RS_InvalidArgumentException(
                "Resized mem pool level must be between 1 and level num - 1"
            ));
        }
        loop<1, LevelNum>([&](auto i_const) {
            constexpr RIndex i = i_const.value;
            if (i == level) {
                auto& mem_pool = rs_device_std::get<i - 1>(internal_mem_pools);
                mem_pool.resize(mem_pool_size, copy);
            }
        });
    }

    /**
     * @brief Clears all used entries in the array tree 3d multi map.
     */
    void clear(bool sync = true) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);

        root.fill(RS_NULL_INDEX, sync);
        array_tree_helper::for_each_in_tuple(internal_mem_pools, [&](auto& pool) {
            pool.clear(sync);
        });
    }

    /**
     * @brief Copies the array tree 3d multi map, resize as needed.
     * @param array_tree The array tree 3d multi map to copy from.
     */
    void copy(const RS_ArrayTree3DMultiMap& array_tree) {
        RS_DATA_EXPECT_INIT(provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(array_tree.provider() != nullptr);

        pos_bound = array_tree.pos_bound;
        root.copy(array_tree.root);
        array_tree_helper::for_each_in_zip_tuples(
            internal_mem_pools, array_tree.internal_mem_pools, [](auto& pool, auto& other_pool) {
                pool.copy(other_pool);
            }
        );
    }

    /**
     * @brief Moves data to the specified MemoryProvider and returns a new array tree 3d multi map.
     */
    RS_ArrayTree3DMultiMap move(RS_MemoryProvider* provider) const {
        RS_DATA_EXPECT_INIT(this->provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        RS_ArrayTree3DMultiMap new_array_tree;
        new_array_tree.pos_bound = pos_bound;
        new_array_tree.root = root.move(provider);
        array_tree_helper::for_each_in_zip_tuples(
            new_array_tree.internal_mem_pools,
            internal_mem_pools,
            [&provider](auto& pool, auto& other_pool) { pool = other_pool.move(provider); }
        );
        return new_array_tree;
    }

    [[nodiscard]] RS_MemoryProvider* provider() const { return root.provider(); }

    [[nodiscard]] RS_DeviceLocation device() const { return root.device(); }

    /// @brief Returns the base size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RBigSize baseBytes() const { return sizeof(*this); }

    /// @brief Returns the extra size in bytes.
    HOST_DEVICE_INLINE_QUALIFIER RBigSize extraBytes() const {
        RBigSize extra_bytes = 0;
        extra_bytes += root.extraBytes();
        array_tree_helper::for_each_in_tuple_hd(internal_mem_pools, [&extra_bytes](auto& pool) {
            extra_bytes += pool.extraBytes();
        });
        return extra_bytes;
    }

    /// @brief Returns the maximum valid coordinate of the array tree 3d multi map.
    HOST_DEVICE_INLINE_QUALIFIER RCoord3 maxCoord() const { return pos_bound - 1; }

    /// @brief Returns the minimum valid coordinate of the array tree 3d multi map.
    HOST_DEVICE_INLINE_QUALIFIER RCoord3 minCoord() const { return -pos_bound; }

    /// @brief Returns the size of the array tree 3d multi map at the specified level.
    /// @param level The level of the array tree 3d multi map.
    /// @return The size of the array tree 3d multi map at the specified level. when level = 0,
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

    /// @brief Returns the size of the array tree 3d multi map at the specified level.
    /// @tparam Level The level of the array tree 3d multi map.
    /// @return The size of the array tree 3d multi map at the specified level. when Level = 0,
    /// returns the max bound size of root, when Level = LevelNum - 1, returns the
    /// size of the last level, when Level = LevelNum, returns 1 for each dimension.
    template <RIndex Level>
    HOST_DEVICE_INLINE_QUALIFIER RCoord3 getLevelSize() const {
        static_assert(Level <= LevelNum, "Level must be less or equal to LevelNum");
        if constexpr (Level == 0) {
            return pos_bound << 1;
        } else {
            return RCoord3(
                refl::trait::get_t<Level - 1, NextLevelSizeList>::XSize,
                refl::trait::get_t<Level - 1, NextLevelSizeList>::YSize,
                refl::trait::get_t<Level - 1, NextLevelSizeList>::ZSize
            );
        }
    }

    /// @brief Returns the region bound of the array tree 3d multi map at the specified level.
    /// @tparam Level The level of the array tree 3d multi map.
    /// @param position The position of the array tree 3d multi map.
    /// @param min_bound The minimum bound of the array tree 3d multi map.
    /// @param max_bound The maximum bound of the array tree 3d multi map.
    template <RIndex Level>
    HOST_DEVICE_INLINE_QUALIFIER void
    getLevelRegionBound(const RCoord3& position, RCoord3& min_bound, RCoord3& max_bound) const {
        static_assert(Level <= LevelNum, "Level must be less or equal to LevelNum");
        if constexpr (Level == 0) {
            min_bound = -pos_bound;
            max_bound = pos_bound;
        } else {
            min_bound = position & ~(getLevelSize<Level>() - 1);
            max_bound = min_bound + getLevelSize<Level>();
        }
    }

    /// @brief Returns whether the coordinate is in the bound of the array tree 3d multi map.
    HOST_DEVICE_INLINE_QUALIFIER bool isInBound(const RCoord3& coord) const {
        return (coord.x >= -pos_bound[0] && coord.x < pos_bound[0]) &&
               (coord.y >= -pos_bound[1] && coord.y < pos_bound[1]) &&
               (coord.z >= -pos_bound[2] && coord.z < pos_bound[2]);
    }

    /// @brief Returns the accessor of the array tree 3d multi map.
    HOST_DEVICE_INLINE_QUALIFIER RS_ArrayTree3DMultiMapAccessor getAccessor() {
        return RS_ArrayTree3DMultiMapAccessor();
    }

    /**
     * @brief Claims a level at the specified coordinate, writes the thread index to
     * the level. The thread that claims the level will be responsible for the memory
     * allocation of the level.
     *
     * @tparam MaxLevel The maximum level to claim. Valid range is [0, LevelNum - 2].
     * @tparam CheckBound Whether to check the bound of the array tree 3d multi map.
     * @param coord The coordinate of the level to claim.
     * @param thread_index The thread index to claim the level.
     * @return True if the level can be claimed (the previous level is allocated),
     * false otherwise.
     */
    template <RIndex MaxLevel, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool claim(const RCoord3& coord, RIndex thread_index) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        return _claim<MaxLevel>(coord, thread_index);
    }

    /**
     * @brief Allocates a level at the specified coordinate, the thread with the same index
     * as the claimer will be responsible for the memory allocation of the level.
     *
     * @tparam MaxLevel The maximum level to allocate. Valid range is [0, LevelNum - 2].
     * @tparam CheckBound Whether to check the bound of the array tree 3d multi map.
     * @param coord The coordinate of the level to allocate.
     * @param thread_index The thread index to allocate the level.
     * @return True if the level can be allocated (the previous level is allocated),
     * false otherwise.
     */
    template <RIndex MaxLevel, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool allocate(const RCoord3& coord, RIndex thread_index) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        return _allocate<MaxLevel>(coord, thread_index);
    }

    /**
     * @brief Returns whether the level at the specified coordinate is empty.
     * @tparam MaxLevel The maximum level to check. Valid range is [0, LevelNum - 1].
     * @tparam CheckBound Whether to check the bound of the array tree 3d multi map.
     * @param coord The coordinate of the level to check.
     * @return True if the level is empty, false otherwise.
     */
    template <RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool isLevelEmpty(const RCoord3& coord) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        auto accessor = getAccessor();
        return _isLevelEmpty<MaxLevel, true>(coord, {}, accessor);
    }

    /**
     * @brief Returns whether the level at the specified coordinate is empty.
     * @tparam MaxLevel The maximum level to check. Valid range is [0, LevelNum - 1].
     * @tparam CheckBound Whether to check the bound of the array tree 3d multi map.
     * @param coord The coordinate of the level to check.
     * @param accessor The accessor of the array tree 3d multi map.
     * @return True if the level is empty, false otherwise.
     */
    template <RIndex MaxLevel = LevelNum - 1, bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool
    isLevelEmpty(const RCoord3& coord, RS_ArrayTree3DMultiMapAccessor& accessor) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        accessor.level = RS_NULL_INDEX;
        if (!_isLevelEmpty<MaxLevel, false>(coord, accessor.coord, accessor)) {
            accessor.coord = coord;
            return false;
        }
        return true;
    }

    /**
     * @brief Inserts a value at the last level of the array tree 3d multi map.
     * @tparam CheckBound Whether to check the bound of the array tree 3d multi map.
     * @param coord The coordinate of the value to insert.
     * @param value The value to insert.
     * @return True if the value is inserted successfully, false otherwise.
     */
    template <bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool insert(const RCoord3& coord, const ValueType& value) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        auto accessor = getAccessor();
        if (_access<true>(coord, {}, accessor)) {
            // No need to update coord of accessor, just insert value
            return accessor.insert(value);
        }
        return false;
    }

    /**
     * @brief Inserts a value at the last level of the array tree 3d multi map with a cached
     * accessor.
     * @tparam CheckBound Whether to check the bound of the array tree 3d multi map.
     * @param coord The coordinate of the value to insert.
     * @param value The value to insert.
     * @param accessor The accessor of the array tree 3d multi map.
     * @return True if the value is inserted successfully, false otherwise.
     */
    template <bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER bool
    insert(const RCoord3& coord, const ValueType& value, RS_ArrayTree3DMultiMapAccessor& accessor) {
        if constexpr (CheckBound) {
            // Check bound
            if (!isInBound(coord)) return false;
        }
        accessor.level = RS_NULL_INDEX;
        if (_access<false>(coord, accessor.coord, accessor)) {
            accessor.coord = coord;
            return accessor.insert(value);
        }
        return false;
    }

    /**
     * @brief Finds a value in the last level of the array tree 3d multi map. If there exists
     * at least one value, return pointer to the first value, which can be used to access the
     * remaining values.
     * @tparam CheckBound Whether to check the bound of the array tree 3d multi map.
     * @param coord The coordinate of the value to find.
     * @param value_num The number of values at the specified coordinate.
     * @return The value at the specified coordinate, nullptr if the value is not found.
     */
    template <bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER ValueType* find(const RCoord3& coord, RSize& value_num) {
        if constexpr (CheckBound) {
            // Check bound
            value_num = 0;
            if (!isInBound(coord)) return nullptr;
        }
        auto accessor = getAccessor();
        if (_access<true>(coord, {}, accessor) && !accessor.empty()) {
            value_num = accessor.size();
            return accessor.get(0);
        }
        value_num = 0;
        return nullptr;
    }

    /**
     * @brief Finds a value in the last level of the array tree 3d multi map with a cached
     * accessor. If there exists at least one value, return pointer to the first value, which
     * can be used to access the remaining values.
     * @tparam CheckBound Whether to check the bound of the array tree 3d multi map.
     * @param coord The coordinate of the value to find.
     * @param value_num The number of values at the specified coordinate.
     * @param accessor The accessor of the array tree 3d multi map.
     * @return The value at the specified coordinate, nullptr if the value is not found.
     */
    template <bool CheckBound = true>
    HOST_DEVICE_INLINE_QUALIFIER ValueType*
    find(const RCoord3& coord, RSize& value_num, RS_ArrayTree3DMultiMapAccessor& accessor) {
        if constexpr (CheckBound) {
            // Check bound
            value_num = 0;
            if (!isInBound(coord)) return nullptr;
        }
        accessor.level = RS_NULL_INDEX;
        if (_access(coord, accessor.coord, accessor)) {
            accessor.coord = coord;
            if (!accessor.empty()) {
                value_num = accessor.size();
                return accessor.get(0);
            }
            value_num = 0;
            return nullptr;
        }
        value_num = 0;
        return nullptr;
    }

   private:
    /// @brief Updates the bound of the array tree 3d multi map.
    void _updateBound() {
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
    }

    /**
     * @brief Claims a level at the specified coordinate, writes the thread index to
     * the level. The thread that claims the level will be responsible for the memory
     * allocation of the level.
     *
     * @tparam MaxLevel The maximum level to claim.
     * @tparam Level The current level to claim.
     * @param level_relative_coord The coordinate of the level to claim.
     * @param thread_index The thread index to claim the level.
     * @param parent_mempool_index The memory pool page index of the parent level.
     * @return True if the level can be claimed (the previous level is allocated),
     * false otherwise.
     */
    template <RIndex MaxLevel, RIndex Level = 0>
    HOST_DEVICE_INLINE_QUALIFIER bool _claim(
        const RCoord3& level_relative_coord,
        RIndex thread_index,
        RIndex parent_mempool_index = RS_NULL_INDEX
    ) {
        static_assert(MaxLevel < LevelNum - 1, "MaxLevel of _claim must be less than LevelNum - 1");

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

        // For level 0, input is global coordinate, transform to relative coordinate
        // do it here to prevent overflow
        if constexpr (Level == MaxLevel) {
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }] = thread_index;
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                        mem_pool.getChunkPtr(parent_mempool_index)
                    );
                array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                ) = thread_index;
            }
            return true;
        } else {
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                parent_mempool_index = root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                        mem_pool.getChunkPtr(parent_mempool_index)
                    );
                parent_mempool_index = array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }
            if (parent_mempool_index == RS_NULL_INDEX) return false;
            return _claim<MaxLevel, Level + 1>(
                next_relative_coord, thread_index, parent_mempool_index
            );
        }
    }

    /**
     * @brief Allocates a level at the specified coordinate, the thread with the same index
     * as the claimer will be responsible for the memory allocation of the level.
     *
     * @tparam MaxLevel The maximum level to allocate.
     * @tparam Level The current level to allocate.
     * @param level_relative_coord The coordinate of the level to allocate.
     * @param thread_index The thread index to allocate the level.
     * @param parent_mempool_index The memory pool page index of the parent level.
     * @return True if the level can be allocated (the previous level is allocated),
     * false otherwise.
     */
    template <RIndex MaxLevel, RIndex Level = 0>
    HOST_DEVICE_INLINE_QUALIFIER bool _allocate(
        const RCoord3& level_relative_coord,
        RIndex thread_index,
        RIndex parent_mempool_index = RS_NULL_INDEX
    ) {
        static_assert(
            MaxLevel < LevelNum - 1, "MaxLevel of _allocate must be less than LevelNum - 1"
        );

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

        // For level 0, input is global coordinate, transform to relative coordinate
        // do it here to prevent overflow
        if constexpr (Level == MaxLevel) {
            RIndex* next_index;
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                next_index = &root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];

            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                        mem_pool.getChunkPtr(parent_mempool_index)
                    );
                next_index = &array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }
            if (*next_index == thread_index) {
                _initNextLevel<Level>(
                    *next_index,
                    current_value_coord.x * 73856093 ^ current_value_coord.y * 19349663 ^
                        current_value_coord.z * 83492791
                );
            }
            return true;
        } else {
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                parent_mempool_index = root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                        mem_pool.getChunkPtr(parent_mempool_index)
                    );
                parent_mempool_index = array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }
            if (parent_mempool_index == RS_NULL_INDEX) return false;
            return _allocate<MaxLevel, Level + 1>(
                next_relative_coord, thread_index, parent_mempool_index
            );
        }
    }

    /**
     * @brief Returns whether the level at the specified coordinate is empty.
     * @tparam MaxLevel The maximum level to check.
     * @tparam IsCoordDifferent Whether the coordinate is different from the accessor coordinate.
     * @tparam Level The current level to check.
     * @param level_relative_coord The coordinate of the level to check.
     * @param accessor_level_relative_coord The coordinate of the accessor level.
     * @param accessor The accessor of the array tree 3d map.
     * @return True if the level is empty, false otherwise.
     */
    template <RIndex MaxLevel, bool IsCoordDifferent = false, RIndex Level = 0>
    HOST_DEVICE_INLINE_QUALIFIER bool _isLevelEmpty(
        const RCoord3& level_relative_coord,
        const RCoord3& accessor_level_relative_coord,
        RS_ArrayTree3DMultiMapAccessor& accessor
    ) {
        static_assert(MaxLevel < LevelNum, "MaxLevel of _isLevelEmpty must be less than LevelNum");

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

        if constexpr (!IsCoordDifferent && Level < LevelNum - 1) {
            RCoord3 acc_current_value_index =
                accessor_level_relative_coord >>
                RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
            RCoord3 acc_next_relative_coord =
                accessor_level_relative_coord &
                RCoord3(NextLevelXSize - 1, NextLevelYSize - 1, NextLevelZSize - 1);
            if (accessor.descendant_level_indices[Level] != RS_NULL_INDEX &&
                current_value_coord == acc_current_value_index) {
                return _isLevelEmpty<MaxLevel, false, Level + 1>(
                    next_relative_coord, acc_next_relative_coord, accessor
                );
            }
        }

        if constexpr (Level == LevelNum - 1) {
            auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
            auto array_ptr = static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                mem_pool.getChunkPtr(accessor.descendant_level_indices[Level - 1])
            );
            accessor.level = Level;
            accessor.value_num_pointer = &array_ptr->value_num[{
                static_cast<RSize>(current_value_coord[0]),
                static_cast<RSize>(current_value_coord[1]),
                static_cast<RSize>(current_value_coord[2])
            }];
            accessor.value_pointer = &array_ptr->values[{
                static_cast<RSize>(current_value_coord[0]),
                static_cast<RSize>(current_value_coord[1]),
                static_cast<RSize>(current_value_coord[2])
            }];
            return *accessor.value_num > 0;
        } else {
            RIndex current_index;
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                current_index = root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                        mem_pool.getChunkPtr(accessor.descendant_level_indices[Level - 1])
                    );
                current_index = array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }

            if (current_index == RS_NULL_INDEX) {
                accessor.clearAfterLastValidLevel();
                return true;
            }

            accessor.level = Level;
            accessor.descendant_level_indices[Level] = current_index;

            if constexpr (Level < MaxLevel) {
                return _isLevelEmpty<MaxLevel, true, Level + 1>(next_relative_coord, {}, accessor);
            } else {
                return false;
            }
        }
    }

    /**
     * @brief Accesses the value of the last level at the specified coordinate.
     * @tparam IsCoordDifferent Whether the coordinate is different from the accessor coordinate.
     * @tparam Level The current level to access.
     * @param level_relative_coord The coordinate of the level to access.
     * @param accessor_level_relative_coord The coordinate of the accessor level.
     * @param accessor The accessor of the array tree 3d multi map.
     * @return True if the value can be accessed, false otherwise.
     */
    template <bool IsCoordDifferent = false, RIndex Level = 0>
    HOST_DEVICE_INLINE_QUALIFIER bool _access(
        const RCoord3& level_relative_coord,
        const RCoord3& accessor_level_relative_coord,
        RS_ArrayTree3DMultiMapAccessor& accessor
    ) {
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

        if constexpr (!IsCoordDifferent && Level < LevelNum - 1) {
            RCoord3 acc_current_value_index =
                accessor_level_relative_coord >>
                RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
            RCoord3 acc_next_relative_coord =
                accessor_level_relative_coord &
                RCoord3(NextLevelXSize - 1, NextLevelYSize - 1, NextLevelZSize - 1);
            if (accessor.descendant_level_indices[Level] != RS_NULL_INDEX &&
                current_value_coord == acc_current_value_index) {
                return _access<false, Level + 1>(
                    next_relative_coord, acc_next_relative_coord, accessor
                );
            }
        }

        if constexpr (Level == LevelNum - 1) {
            auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
            auto array_ptr = static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                mem_pool.getChunkPtr(accessor.descendant_level_indices[Level - 1])
            );
            accessor.level = Level;
            accessor.value_num_pointer = &array_ptr->value_num[{
                static_cast<RSize>(current_value_coord[0]),
                static_cast<RSize>(current_value_coord[1]),
                static_cast<RSize>(current_value_coord[2])
            }];
            accessor.value_start_pointer = &array_ptr->values[{
                static_cast<RSize>(0),
                static_cast<RSize>(current_value_coord[0]),
                static_cast<RSize>(current_value_coord[1]),
                static_cast<RSize>(current_value_coord[2])
            }];
            return true;
        } else {
            RIndex current_index;
            if constexpr (Level == 0) {
                current_value_coord +=
                    pos_bound >> RCoord3(NextLevelXLog2, NextLevelYLog2, NextLevelZLog2);
                current_index = root[{
                    static_cast<RSize>(current_value_coord[0]),
                    static_cast<RSize>(current_value_coord[1]),
                    static_cast<RSize>(current_value_coord[2])
                }];
            } else {
                auto& mem_pool = rs_device_std::get<Level - 1>(internal_mem_pools);
                auto array_ptr =
                    static_cast<refl::trait::get_t<Level - 1, DescendantArrayTypeList>*>(
                        mem_pool.getChunkPtr(accessor.descendant_level_indices[Level - 1])
                    );
                current_index = array_ptr->get(
                    {static_cast<RSize>(current_value_coord[0]),
                     static_cast<RSize>(current_value_coord[1]),
                     static_cast<RSize>(current_value_coord[2])}
                );
            }

            if (current_index == RS_NULL_INDEX) {
                accessor.clearAfterLastValidLevel();
                return false;
            }

            accessor.level = Level;
            accessor.descendant_level_indices[Level] = current_index;

            return _access<true, Level + 1>(next_relative_coord, {}, accessor);
        }
    }

    template <RIndex Level>
    HOST_DEVICE_INLINE_QUALIFIER bool _initNextLevel(RIndex& next_index, RIndex acquire_seed) {
        next_index = rs_device_std::get<Level>(internal_mem_pools).acquire(acquire_seed);
        if (next_index != RS_NULL_INDEX) {
            auto& array = *static_cast<refl::trait::get_t<Level, DescendantArrayTypeList>*>(
                rs_device_std::get<Level>(internal_mem_pools).getChunkPtr(next_index)
            );
            if constexpr (Level == LevelNum - 2) {
                for (RSize i = 0; i < refl::trait::get_t<Level, DescendantArrayTypeList>::ElemNum;
                     i++) {
                    array.value_num[i] = 0;
                }
            } else {
                for (RSize i = 0; i < refl::trait::get_t<Level, DescendantArrayTypeList>::ElemNum;
                     i++) {
                    array[i] = RS_NULL_INDEX;
                }
            }

            return true;
        }
        return false;
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_ARRAY_TREE_H
