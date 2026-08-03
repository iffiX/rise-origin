// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SOV_H
#define RS_SOV_H

#include <assert.h>

#include <refl.hpp>
#include <vector>

#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/utils/rs_conf.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_data_util.h"
#include "rs_soa.h"

#ifndef RS_DEBUG_SOV
#define SOV_DEBUG_INPUT
#define SOV_DEBUG_OUTPUT_AND_ASSERT(index, max_size) assert(index < max_size);
#define SOV_DEBUG_FORWARD
#else
#define SOV_DEBUG_INPUT , const char *caller
#define SOV_DEBUG_OUTPUT_AND_ASSERT(index, max_size)                            \
    if ((index) >= (max_size)) {                                                \
        printf("Boundary exceeded, Caller: %s, index: %lu\n", caller, (index)); \
    }                                                                           \
    assert((index) < (max_size));
#define SOV_DEBUG_FORWARD , ""
#endif

namespace rise {

namespace device {

template <typename T>
class RS_StructOfVectors;

namespace sov_helper {

constexpr RSize SOV_ALIGN_SIZE = RS_DEVICE_MEMORY_ALIGN_SIZE;

/**
 * The type the member returns when invoked.
 * In case of future updates when there exists function descriptors
 * among all descriptors, use the first one. The second one is just used to make
 * CLion IDE happy.
 */
// template<typename Member>
// using underlying_type = refl::trait::remove_qualifiers_t<decltype(
// Member()(std::declval<const typename Member::declaring_type &>()))>;

template <typename Member>
using custom_underlying_type = refl::trait::remove_qualifiers_t<typename Member::value_type>;

/**
 * Member lists of fields in SoV
 */

template <typename T>
constexpr auto filter_field_members =
    filter(refl::member_list<T>{}, [](auto member) { return is_field(member); });

/**
 * Check if a reflected field member is marked with protected_field_marker.
 * Protected fields are stored as T* (not decomposed into sub-member arrays).
 */
template <typename FieldMember>
constexpr bool is_protected_field_v =
    refl::descriptor::has_attribute<refl::attr::protected_field_marker>(FieldMember{});

template <typename T>
using field_member_list = std::remove_const_t<decltype(filter_field_members<T>)>;

template <typename T>
constexpr auto filter_unreadable_field_members = filter(refl::member_list<T>{}, [](auto member) {
    return is_field(member) && !is_readable(member);
});

template <typename T>
using unreadable_field_member_list = decltype(filter_unreadable_field_members<T>);

// These helper functions needs to call private methods of the SoV,
// Therefore they can only be encapsulated in a friend class
struct PrivateHelper {
    // Provide unified access to regular fields and nested fields
    // For regular fields, T is a pointer type
    // For nested fields, T is RS_StructOfVectors<SomeNestedField>
    template <typename T>
    static std::enable_if_t<std::is_class_v<T>, RBigSize>
    _getRequiredBytes(T &t, RSize num, RBigSize prev_last_member_end) {
        return t._getRequiredBytes(num, prev_last_member_end);
    }

    template <typename T>
    static std::enable_if_t<!std::is_class_v<T>, RBigSize>
    _getRequiredBytes(T &t, RSize num, RBigSize prev_last_member_end) {
        using element_type = std::remove_extent_t<std::remove_pointer_t<T>>;
        return RS_ALIGN(prev_last_member_end + sizeof(element_type) * num, SOV_ALIGN_SIZE);
    }

    template <typename T>
    static std::enable_if_t<std::is_class_v<T>, RBigSize>
    _getOffsets(T &t, RSize num, std::vector<RBigSize> &offsets, RBigSize prev_last_member_end) {
        return t._getOffsets(num, offsets, prev_last_member_end);
    }

    template <typename T>
    static std::enable_if_t<!std::is_class_v<T>, RBigSize>
    _getOffsets(T &t, RSize num, std::vector<RBigSize> &offsets, RBigSize prev_last_member_end) {
        using element_type = std::remove_extent_t<std::remove_pointer_t<T>>;
        offsets.push_back(prev_last_member_end);
        RBigSize last_member_end =
            RS_ALIGN(prev_last_member_end + sizeof(element_type) * num, SOV_ALIGN_SIZE);
        return last_member_end;
    }

    template <typename T>
    HOST_DEVICE_QUALIFIER static std::enable_if_t<std::is_class_v<T>, void> _initMember(T &t) {
        t.elem_num = 0;
        t.used_elem_num = 0;
        t.data = {};
        t._initMember();
    }

    template <typename T>
    HOST_DEVICE_QUALIFIER static std::enable_if_t<!std::is_class_v<T>, void> _initMember(T &t) {
        t = nullptr;
    }

    template <typename T>
    static std::enable_if_t<std::is_class_v<T>, RBigSize> _updateMember(
        T &t,
        RS_MemoryPtr data,
        RSize num,
        RSize used_num,
        RBigSize prev_last_member_end
    ) {
        t.elem_num = num;
        t.used_elem_num = used_num;
        return t._updateMember(data, num, used_num, prev_last_member_end);
    }

    template <typename T>
    static std::enable_if_t<!std::is_class_v<T>, RBigSize> _updateMember(
        T &t,
        RS_MemoryPtr data,
        RSize num,
        RSize /*used_num*/,
        RBigSize prev_last_member_end
    ) {
        using element_type = std::remove_extent_t<std::remove_pointer_t<T>>;
        if (num == 0) {
            t = nullptr;
            return 0;
        } else {
            t = reinterpret_cast<T>((data + prev_last_member_end).ptr());
            return RS_ALIGN(prev_last_member_end + sizeof(element_type) * num, SOV_ALIGN_SIZE);
        }
    }

    template <typename T>
    static std::enable_if_t<std::is_class_v<T>, std::pair<RBigSize, RBigSize>> _copyMember(
        T &t,
        RS_ConstMemoryPtr old_data,
        RSize old_num,
        RSize old_offset,
        RS_MemoryPtr new_data,
        RSize new_num,
        RSize new_offset,
        RSize copy_num,
        RBigSize old_prev_last_member_end,
        RBigSize new_prev_last_member_end
    ) {
        return t._copyMember(
            old_data,
            old_num,
            old_offset,
            new_data,
            new_num,
            new_offset,
            copy_num,
            old_prev_last_member_end,
            new_prev_last_member_end
        );
    }

    template <typename T>
    static std::enable_if_t<!std::is_class_v<T>, std::pair<RBigSize, RBigSize>> _copyMember(
        T &t,
        RS_ConstMemoryPtr old_data,
        RSize old_num,
        RSize old_offset,
        RS_MemoryPtr new_data,
        RSize new_num,
        RSize new_offset,
        RSize copy_num,
        RBigSize old_prev_last_member_end,
        RBigSize new_prev_last_member_end
    ) {
        using element_type = std::remove_extent_t<std::remove_pointer_t<T>>;
        if (copy_num > 0) {
            (new_data + new_prev_last_member_end + sizeof(element_type) * new_offset)
                .copyFrom(
                    old_data + old_prev_last_member_end + sizeof(element_type) * old_offset,
                    sizeof(element_type) * copy_num,
                    false
                );
        }
        RBigSize o_p_e =
            RS_ALIGN(old_prev_last_member_end + sizeof(element_type) * old_num, SOV_ALIGN_SIZE);
        RBigSize n_p_e =
            RS_ALIGN(new_prev_last_member_end + sizeof(element_type) * new_num, SOV_ALIGN_SIZE);
        return std::make_pair(o_p_e, n_p_e);
    }

    template <typename T, typename T2>
    static typename std::enable_if<std::is_class_v<T>, RIndex>::type _transverseMemberElementGet(
        T &t,
        RS_MemoryPtr data,
        RSize num,
        T2 &element,
        RIndex element_index,
        const std::vector<RBigSize> &offsets,
        RIndex member_absolute_index
    ) {
        return t._transverseMemberElementGet(
            data, num, element, element_index, offsets, member_absolute_index
        );
    }

    template <typename T, typename T2>
    static typename std::enable_if<!std::is_class_v<T>, RIndex>::type _transverseMemberElementGet(
        T & /*t*/,
        RS_MemoryPtr data,
        RSize /*num*/,
        T2 &element,
        RIndex element_index,
        const std::vector<RBigSize> &offsets,
        RIndex member_absolute_index
    ) {
        auto ptr = reinterpret_cast<T>((data + offsets[member_absolute_index]).ptr());
        element = ptr[element_index];
        member_absolute_index++;
        return member_absolute_index;
    }

    template <typename T, typename T2>
    static typename std::enable_if<std::is_class_v<T>, RIndex>::type _transverseMemberElementSet(
        T &t,
        RS_MemoryPtr data,
        RSize num,
        T2 &element,
        RIndex element_index,
        const std::vector<RBigSize> &offsets,
        RIndex member_absolute_index
    ) {
        return t._transverseMemberElementSet(
            data, num, element, element_index, offsets, member_absolute_index
        );
    }

    template <typename T, typename T2>
    static typename std::enable_if<!std::is_class_v<T>, RIndex>::type _transverseMemberElementSet(
        T & /*t*/,
        RS_MemoryPtr data,
        RSize /*num*/,
        T2 &element,
        RIndex element_index,
        const std::vector<RBigSize> &offsets,
        RIndex member_absolute_index
    ) {
        auto ptr = reinterpret_cast<T>((data + offsets[member_absolute_index]).ptr());
        ptr[element_index] = element;
        member_absolute_index++;
        return member_absolute_index;
    }
};

template <typename T, typename T2>
HOST_DEVICE_QUALIFIER static std::enable_if_t<std::is_array_v<T>> assign(T &t, T2 value) {
    constexpr RSize length = sizeof(T) / sizeof(std::remove_extent_t<T>);
    for (RSize i = 0; i < length; i++) t[i] = value[i];
}

template <typename T, typename T2>
HOST_DEVICE_QUALIFIER static std::enable_if_t<!std::is_array_v<T>> assign(T &t, T2 &value) {
    t = value;
}

template <typename T, RSize N>
HOST_DEVICE_INLINE_QUALIFIER decltype(auto) getSub(T &sov, RSize index) {
    return sov.template get<N>(index SOV_DEBUG_FORWARD);
}

template <typename T, RSize N, RSize N1, RSize... Ns>
HOST_DEVICE_INLINE_QUALIFIER decltype(auto) getSub(T &sov, RSize index) {
    return getSub<std::remove_reference_t<decltype(rs_device_std::get<N>(sov.storage))>, N1, Ns...>(
        rs_device_std::get<N>(sov.storage), index
    );
}

template <typename T, typename T2, RSize N>
HOST_DEVICE_INLINE_QUALIFIER void setSub(T &sov, RSize index, const T2 &value) {
    sov.template set<N>(index, value SOV_DEBUG_FORWARD);
}

template <typename T, typename T2, RSize N, RSize N1, RSize... Ns>
HOST_DEVICE_INLINE_QUALIFIER void setSub(T &sov, RSize index, const T2 &value) {
    setSub<std::remove_reference_t<decltype(rs_device_std::get<N>(sov.storage))>, T2, N1, Ns...>(
        rs_device_std::get<N>(sov.storage), index, value
    );
}

}  // namespace sov_helper

/**
 * @brief RS_StructOfVectors is a template class for representing structure of vectors on the GPU.
 * @tparam T The type of structure to be converted
 *
 * Example usage:
 *
 * struct Vector {
 *    int x;
 *    float y;
 *    double z;
 * };
 * struct A {
 *    int m;
 *    Vector v;
 * };
 * struct B {
 *    A struct_a;
 *    int array[10];
 * };
 *
 * REFL_AUTO(
 *   type(Vector),
 *   field(x),
 *   field(y),
 *   field(z)
 * )
 *
 * REFL_AUTO(
 *   type(A),
 *   field(m),
 *   field(v)
 * )
 *
 * REFL_AUTO(
 *   type(B),
 *   field(struct_a),
 *   field(array)
 * )
 *
 * And suppose we pass RS_StructOfVectors<B> to a global kernel:
 * __global__ kernel(RS_StructOfVectors<B> sov) {
 *     sov.get(0); // Returns first structure B
 *     sov.get<Pr(B, struct_a)>(1); // Returns struct_a in second B
 *     auto array_accessor = sov.get<Pr(B, array)>(1);
 *     array_accessor[0]; // Access first element of array in second B
 *     sov.set<Pr(B, struct_a), Pr(A, v), Pr(Vector, x)>(1); // Nested access
 * }
 */
template <typename T>
class RS_StructOfVectors {
   public:
    using ObjectType = T;

    using members = refl::member_list<T>;
    using field_members = sov_helper::field_member_list<std::remove_cv_t<T>>;
    using unreadable_field_members = sov_helper::unreadable_field_member_list<std::remove_cv_t<T>>;
    static_assert(field_members::size > 0, "Type has no fields!");
    static_assert(unreadable_field_members::size == 0, "Type has unreadable fields!");

   private:
    template <typename FieldMember>
    struct make_storage {
        using underlying = sov_helper::custom_underlying_type<FieldMember>;
        using type = typename std::conditional<
            sov_helper::is_protected_field_v<FieldMember>,
            underlying *,
            typename std::conditional<
                std::is_class_v<underlying>,
                RS_StructOfVectors<underlying>,
                underlying *>::type>::type;
    };

    // Helper: true if the Nth member is a class type that should be decomposed (not protected)
    template <RSize N>
    static constexpr bool is_nested_sov_v =
        std::is_class_v<sov_helper::custom_underlying_type<refl::trait::get_t<N, members>>> &&
        !sov_helper::is_protected_field_v<refl::trait::get_t<N, members>>;

    // Helper: true if the Nth member is a scalar or protected class (stored as T*)
    template <RSize N>
    static constexpr bool is_direct_field_v =
        !std::is_class_v<sov_helper::custom_underlying_type<refl::trait::get_t<N, members>>> ||
        sov_helper::is_protected_field_v<refl::trait::get_t<N, members>>;

    template <typename>
    struct as_sov_tuple;

    template <template <typename...> typename T2, typename... Ts>
    struct as_sov_tuple<T2<Ts...>> {
        using type = rs_device_std::tuple<Ts...>;
    };

    template <typename T2>
    using as_tuple_t = typename as_sov_tuple<T2>::type;

   public:
    /*****************************************************************
     * Data
     *****************************************************************/
    // The root memory region
    // For sub sov, data is always null
    RS_MemoryPtr data;

    // Allocated capacity (column length) for every field column.
    RSize elem_num = 0;

    // Logical number of stored elements (in [0, elem_num]).
    RSize used_elem_num = 0;

    // Storage is a std::tuple containing device side memory pointers to
    // individual arrays Each array contains a property of the virtual
    // "Structure". Each pointer points to a aligned segment in storage_root.
    as_tuple_t<refl::trait::map_t<make_storage, field_members>> storage;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_StructOfVectors() = default;

    /**
     * @brief Construct a new RS_StructOfVectors object with the given provider.
     */
    explicit RS_StructOfVectors(RS_MemoryProvider *provider) { init(provider); }

    /**
     * @brief Construct a new RS_StructOfVectors object with the given initial capacity and
     * provider.
     * @param init_capacity The initial capacity of the structure of vectors.
     * @param provider The memory provider.
     */
    RS_StructOfVectors(RSize init_capacity, RS_MemoryProvider *provider) {
        init(init_capacity, provider);
    }

    /**
     * @brief Initialize an empty structure of vectors with the given provider.
     * @param provider The memory provider.
     */
    void init(RS_MemoryProvider *provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        data = {nullptr, 0, 0, provider->device, provider};
        elem_num = 0;
        used_elem_num = 0;
        _initMember();
    }

    /**
     * @brief Initialize the structure of vectors with the given initial capacity and provider.
     * @param init_capacity The initial capacity of the structure of vectors.
     * @param provider The memory provider.
     */
    void init(RSize init_capacity, RS_MemoryProvider *provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        data = {nullptr, 0, 0, provider->device, provider};
        elem_num = 0;
        used_elem_num = 0;
        _initMember();
        resize(init_capacity, false);
    }

    /**
     * @brief Copy constructor.
     * @warning It is a shallow copy which only copies pointer and size.
     */
    RS_StructOfVectors(const RS_StructOfVectors &other) = default;

    /**
     * @brief Frees the GPU memory associated with the SoV.
     * @warning It will free the data, all related SoVs will be affected.
     */
    void free() {
        data.free();
        elem_num = 0;
        used_elem_num = 0;
        _updateMember(data, 0, 0);
    }

    /**
     * @brief Resize the structure of vectors to new capacity.
     * By default, resize will copy data, but remember copying each segment of member data
     * to their new location is expensive.
     * If copy, when resized smaller, used_elem_num is clamped to the new capacity and the
     * excessive data is truncated.
     *
     * @warning It will free the data, all related arrays will be affected.
     *
     * @note Stream will be synchronized at the end of the function.
     */
    void resize(RSize new_capacity, bool copy = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);

        if (new_capacity == elem_num) return;

        const RSize new_used = RS_MIN(used_elem_num, new_capacity);

        if (elem_num != 0 && new_capacity != 0 && copy) {
            RBigSize required_bytes = _getRequiredBytes(new_capacity);
            auto new_data = data.provider()->allocate(required_bytes, false);
            if (new_used > 0) {
                _copyMember(data, elem_num, 0, new_data, new_capacity, 0, new_used);
            }
            data.free(false);
            data = new_data;
            elem_num = new_capacity;
            used_elem_num = new_used;
            _updateMember(data, new_capacity, new_used);
        } else {
            RS_MemoryProvider *provider = data.provider();
            if (data != nullptr) data.free(false);
            RBigSize required_bytes = _getRequiredBytes(new_capacity);
            data = (new_capacity == 0) ? RS_MemoryPtr{nullptr, 0, 0, provider->device, provider}
                                       : provider->allocate(required_bytes, false);
            elem_num = new_capacity;
            used_elem_num = new_used;
            _updateMember(data, new_capacity, new_used);
        }
        data.provider()->sync();
    }

    /**
     * @brief Fill the struct of vectors with the given value.
     *
     * @details Writes value into every slot of the allocated capacity across all columns
     * and sets used_elem_num to elem_num.
     *
     * @param value Value to fill with.
     */
    void fill(const T &value, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);

        if (elem_num > 0) {
            std::vector<RBigSize> offsets;
            _getOffsets(elem_num, offsets);
            auto tmp_data = new RByte[data.size()];
            for (RIndex i = 0; i < elem_num; i++) {
                _transverseMemberElementSet({tmp_data, data.size()}, elem_num, value, i, offsets);
            }
            data.copyFrom(RS_ConstMemoryPtr(tmp_data, data.size()), data.size(), sync);
            delete[] tmp_data;
            used_elem_num = elem_num;
            _updateMember(data, elem_num, used_elem_num);
        }
    }

    /**
     * @brief Fill a slice of the struct of vectors with the given value.
     *
     * @details Updates used_elem_num to max(used_elem_num, offset + num).
     *
     * @param value Value to fill with.
     * @param num The number of elements to be filled.
     * @param offset The starting offset of the slice in the structure of vectors.
     */
    void fillSlice(const T &value, RSize num, RSize offset, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_BOUND(
            offset + num <= elem_num, "offset={}, num={}, elem_num={}", offset, num, elem_num
        );

        if (num > 0) {
            std::vector<RBigSize> offsets;
            _getOffsets(num, offsets);
            RBigSize required_size = _getRequiredBytes(num);
            auto tmp_data = new RByte[required_size];
            for (RIndex i = 0; i < num; i++) {
                _transverseMemberElementSet({tmp_data, required_size}, num, value, i, offsets);
            }
            _copyMember({tmp_data, required_size}, num, 0, data, elem_num, offset, num);
            delete[] tmp_data;
            if (sync) data.provider()->sync();
            used_elem_num = RS_MAX(used_elem_num, offset + num);
            _updateMember(data, elem_num, used_elem_num);
        }
    }

    /**
     * @brief Write host data to the device side of the struct of vectors.
     *
     * @details Fills the entire allocated capacity and sets used_elem_num to elem_num.
     *
     * @param input The host data to be written; must hold at least elem_num elements.
     */
    void write(const T *input, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            input != nullptr || elem_num == 0,
            "When writing to a non-empty SOV, input pointer must not be null."
        );

        if (elem_num > 0) {
            std::vector<RBigSize> offsets;
            _getOffsets(elem_num, offsets);
            auto tmp_data = new RByte[data.size()];
            for (RIndex i = 0; i < elem_num; i++) {
                _transverseMemberElementSet(
                    {tmp_data, data.size()}, elem_num, input[i], i, offsets
                );
            }
            data.copyFrom(RS_ConstMemoryPtr(tmp_data, data.size()), data.size(), sync);
            delete[] tmp_data;
            used_elem_num = elem_num;
            _updateMember(data, elem_num, used_elem_num);
        }
    }

    /**
     * @brief Write a slice of host data to the device side of the struct of vectors.
     *
     * @details Updates used_elem_num to max(used_elem_num, offset + num).
     *
     * @param input The host data to be written.
     * @param num The number of elements to be written.
     * @param offset The starting offset of the slice in the structure of vectors.
     */
    void writeSlice(const T *input, RSize num, RSize offset, bool sync = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            input != nullptr || num == 0,
            "When writing to a non-empty slice of a SOV, input pointer must not be null."
        );
        RS_DATA_EXPECT_BOUND(
            offset + num <= elem_num, "offset={}, num={}, elem_num={}", offset, num, elem_num
        );

        if (num > 0) {
            std::vector<RBigSize> offsets;
            _getOffsets(num, offsets);
            RBigSize required_size = _getRequiredBytes(num);
            auto tmp_data = new RByte[required_size];
            for (RIndex i = 0; i < num; i++) {
                _transverseMemberElementSet({tmp_data, required_size}, num, input[i], i, offsets);
            }
            _copyMember({tmp_data, required_size}, num, 0, data, elem_num, offset, num);
            delete[] tmp_data;
            if (sync) data.provider()->sync();
            if (offset + num > used_elem_num) {
                used_elem_num = offset + num;
                _updateMember(data, elem_num, used_elem_num);
            }
        }
    }

    /**
     * @brief Copies data from another RS_StructOfVectors. Resize as needed.
     *
     * @details Grows capacity to fit other.used_elem_num if necessary, then copies
     * other.used_elem_num elements per column.
     */
    void copy(const RS_StructOfVectors &other) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(other.data.provider() != nullptr);

        if (data == other.data) return;
        if (other.used_elem_num > elem_num) {
            resize(other.used_elem_num, false);
        }
        used_elem_num = other.used_elem_num;
        _updateMember(data, elem_num, used_elem_num);
        if (used_elem_num > 0) {
            _copyMember(other.data, other.elem_num, 0, data, elem_num, 0, used_elem_num);
            data.provider()->sync();
        }
    }

    /**
     * @brief Copies the used range from device to host.
     *
     * @param output Buffer to store at least used_elem_num elements.
     */
    void read(T *output, bool sync = true) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            output != nullptr || used_elem_num == 0,
            "When reading from a non-empty SOV, output pointer must not be null."
        );

        if (used_elem_num > 0) {
            readSlice(output, used_elem_num, 0, sync);
        }
    }

    /**
     * @brief Read a slice of device data of the struct of vectors to the host side.
     *
     * @param output The host data to write to.
     * @param num The number of elements to be read.
     * @param offset The starting offset of the slice in the structure of vectors.
     */
    void readSlice(T *output, RSize num, RSize offset, bool sync = true) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            output != nullptr || num == 0,
            "When reading a non-empty slice of a SOV, output pointer must not be null."
        );
        RS_DATA_EXPECT_BOUND(
            offset + num <= used_elem_num,
            "offset={}, num={}, used_elem_num={}",
            offset,
            num,
            used_elem_num
        );

        if (num > 0) {
            RBigSize required_size = const_cast<RS_StructOfVectors *>(this)->_getRequiredBytes(num);
            auto tmp_data = new RByte[required_size];
            const_cast<RS_StructOfVectors *>(this)->_copyMember(
                data, elem_num, offset, {tmp_data, required_size}, num, 0, num
            );
            std::vector<RBigSize> offsets;
            const_cast<RS_StructOfVectors *>(this)->_getOffsets(num, offsets);
            if (sync) data.provider()->sync();
            for (RIndex i = 0; i < num; i++) {
                const_cast<RS_StructOfVectors *>(this)->_transverseMemberElementGet(
                    {tmp_data, required_size}, num, output[i], i, offsets
                );
            }
            delete[] tmp_data;
        }
    }

    /// @brief Moves data to specified provider and returns a new SoV
    RS_StructOfVectors move(RS_MemoryProvider *provider) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        if (elem_num == 0) return RS_StructOfVectors(provider);
        RS_StructOfVectors new_sov(elem_num, provider);
        new_sov.copy(*this);
        return new_sov;
    }

    /**
     * @brief Inserts a value at a given index, shifting tail elements one slot
     *        further into each column.
     *
     * @param value The value to be inserted.
     * @param index The index at which to insert the value.
     */
    void insert(T value, RIndex index) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_BOUND(
            index <= used_elem_num, "index={}, used_elem_num={}", index, used_elem_num
        );

        if (used_elem_num == elem_num) {
            RSize new_cap = (elem_num == 0) ? 1 : (elem_num * 2);
            resize(new_cap, true);
        }

        RSize tail = used_elem_num - index;
        if (tail > 0) {
            RBigSize scratch_bytes = _getRequiredBytes(tail);
            auto scratch = data.provider()->allocate(scratch_bytes, false);
            _copyMember(data, elem_num, index, scratch, tail, 0, tail);
            _copyMember(scratch, tail, 0, data, elem_num, index + 1, tail);
            scratch.free(false);
        }

        // Scatter `value` into slot `index` across every column via a host temp.
        std::vector<RBigSize> offsets;
        _getOffsets(1, offsets);
        RBigSize required_size = _getRequiredBytes(1);
        auto tmp_data = new RByte[required_size];
        _transverseMemberElementSet({tmp_data, required_size}, 1, value, 0, offsets);
        _copyMember({tmp_data, required_size}, 1, 0, data, elem_num, index, 1);
        delete[] tmp_data;

        used_elem_num++;
        _updateMember(data, elem_num, used_elem_num);
        data.provider()->sync();
    }

    /**
     * @brief Appends a value at the end of the used elements.
     *
     * @param value The value to be appended.
     */
    void append(T value) { insert(value, used_elem_num); }

    void erase(RIndex index) { eraseRange(index, index + 1); }

    void clear() {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        used_elem_num = 0;
        _updateMember(data, elem_num, used_elem_num);
    }

    void occupy(RSize new_used_elem_num) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_BOUND(
            new_used_elem_num <= elem_num,
            "new_used_elem_num={}, elem_num={}",
            new_used_elem_num,
            elem_num
        );
        used_elem_num = new_used_elem_num;
        _updateMember(data, elem_num, used_elem_num);
    }

    /**
     * @brief Erases elements in the range [begin, end), shifting tail elements
     *        down to fill the gap in each column.
     *
     * @param begin The starting index (inclusive).
     * @param end The ending index (exclusive).
     */
    void eraseRange(RIndex begin, RIndex end) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_BOUND(end <= used_elem_num, "end={}, used_elem_num={}", end, used_elem_num);

        if (begin >= end) {
            return;
        }
        RSize erase_num = end - begin;
        RSize tail = used_elem_num - end;
        if (tail > 0) {
            RBigSize scratch_bytes = _getRequiredBytes(tail);
            auto scratch = data.provider()->allocate(scratch_bytes, false);
            _copyMember(data, elem_num, end, scratch, tail, 0, tail);
            _copyMember(scratch, tail, 0, data, elem_num, begin, tail);
            scratch.free(false);
        }
        used_elem_num -= erase_num;
        _updateMember(data, elem_num, used_elem_num);
        data.provider()->sync();
    }

    [[nodiscard]] RS_MemoryProvider *provider() const { return data.provider(); }

    [[nodiscard]] RS_DeviceLocation device() const { return data.device(); }

    /**
     * G1
     * Access the reference of the Nth (non-structure, non-array) field of the index-th
     * element: Eg: sov.get<0>(100) = 0
     */
    template <
        RSize N,
        std::enable_if_t<
            is_direct_field_v<N> && !std::is_array_v<sov_helper::custom_underlying_type<
                                        refl::trait::get_t<N, RS_StructOfVectors<T>::members>>>,
            bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER auto &get(RSize index SOV_DEBUG_INPUT) {
        SOV_DEBUG_OUTPUT_AND_ASSERT(index, used_elem_num)
        return rs_device_std::get<N>(storage)[index];
    }

    /**
     * G1.1
     * Access the Nth array field of the index-th element using an accessor:
     * Eg: sov.get<0>(100)[3] = 0
     */
    template <
        RSize N,
        std::enable_if_t<
            std::is_array_v<sov_helper::custom_underlying_type<refl::trait::get_t<N, members>>>,
            bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER auto get(RSize index SOV_DEBUG_INPUT) {
        SOV_DEBUG_OUTPUT_AND_ASSERT(index, used_elem_num)
        using type = sov_helper::custom_underlying_type<refl::trait::get_t<N, members>>;
        using element_type = std::remove_extent_t<type>;
        constexpr RSize length = sizeof(type) / sizeof(element_type);
        // Use base type pointer to access values
        return RS_ArrayAccessor(
            reinterpret_cast<element_type *>(rs_device_std::get<N>(storage)),
            elem_num,
            index,
            length
        );
    }

    /**
     * G2
     * Access the const reference of the Nth (non-structure) field of the index-th
     * element: Eg: x = sov.get<0>(100)
     */
    template <
        RSize N,
        std::enable_if_t<
            is_direct_field_v<N> && !std::is_array_v<sov_helper::custom_underlying_type<
                                        refl::trait::get_t<N, members>>>,
            bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER const auto &get(RSize index SOV_DEBUG_INPUT) const {
        SOV_DEBUG_OUTPUT_AND_ASSERT(index, used_elem_num)
        return rs_device_std::get<N>(storage)[index];
    }

    /**
     * G2.1
     * Access the Nth array field of the index-th element using a const accessor:
     * Eg: x = sov.get<0>(100)[3]
     */
    template <
        RSize N,
        std::enable_if_t<
            std::is_array_v<sov_helper::custom_underlying_type<refl::trait::get_t<N, members>>>,
            bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER auto get(RSize index SOV_DEBUG_INPUT) const {
        SOV_DEBUG_OUTPUT_AND_ASSERT(index, used_elem_num)
        using type = sov_helper::custom_underlying_type<refl::trait::get_t<N, members>>;
        using element_type = std::remove_extent_t<type>;
        constexpr RSize length = sizeof(type) / sizeof(element_type);
        // Use base type pointer to access values
        return RS_ConstArrayAccessor(
            reinterpret_cast<element_type *>(rs_device_std::get<N>(storage)),
            elem_num,
            index,
            length
        );
    }

    /**
     * G3
     * Access a copy of the Nth (structure) field of the index-th element
     * Eg:
     * struct SecondLevel;
     * struct FirstLevel { float w; SecondLevel sec;}
     * struct SecondLevel { float x, y, z; }
     * sov.get<1>(100).x                     // x is the field of a copy of sub field sec.
     *
     * Note: the method calls get(index) internally, which access every single element of
     * the second level structure in an aligned manner and copy them back.
     *
     * Note: returned value is const to prevent unintentional modify, since this would
     * have no effect
     */

    template <RSize N, std::enable_if_t<is_nested_sov_v<N>, bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER const auto get(RSize index SOV_DEBUG_INPUT) const {
        SOV_DEBUG_OUTPUT_AND_ASSERT(index, used_elem_num)
        // last get is a recursive call of G5
        return rs_device_std::move(rs_device_std::get<N>(storage).get(index SOV_DEBUG_FORWARD));
    }

    /**
     * G4
     * Access the sub fields of the index-th element
     * Eg:
     * struct SecondLevel;
     * struct FirstLevel { float w; SecondLevel sec;}
     * struct SecondLevel { float x, y, z; }
     * sov.get<1, 2>(100)                     // A reference to sec.x of the 100th
     * element.
     *
     */
    template <RSize N, RSize N1, RSize... Ns, std::enable_if_t<is_nested_sov_v<N>, bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER decltype(auto) get(RSize index SOV_DEBUG_INPUT) {
        SOV_DEBUG_OUTPUT_AND_ASSERT(index, used_elem_num)
        return sov_helper::getSub<decltype(*this), N, N1, Ns...>(*this, index);
    }

    /**
     * G5
     * Access a copy of the index-th element
     * Eg:
     * sov.get(100)
     */
    HOST_DEVICE_INLINE_QUALIFIER T get(RSize index SOV_DEBUG_INPUT) const {
        SOV_DEBUG_OUTPUT_AND_ASSERT(index, used_elem_num)
        T t{};
        for_each(field_members{}, [&](auto member) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            // recursive call of G1.1, G2, G3
            sov_helper::assign(member(t), get<i>(index SOV_DEBUG_FORWARD));
        });
        return rs_device_std::move(t);
    }

    /**
     * S1
     * Set the value of the Nth (non-structure, non-array) field of the index-th element:
     * Eg: sov.set<0>(100, 0)
     * Note: You can also use sov.get<0>(100) = 0
     */
    template <
        RSize N,
        typename T2,
        std::enable_if_t<
            is_direct_field_v<N> && !std::is_array_v<sov_helper::custom_underlying_type<
                                        refl::trait::get_t<N, members>>>,
            bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER void set(RSize index, const T2 &value SOV_DEBUG_INPUT) {
        SOV_DEBUG_OUTPUT_AND_ASSERT(index, used_elem_num)
        rs_device_std::get<N>(storage)[index] = value;
    }

    /**
     * S1.1
     * Set Nth array field of the index-th element using an accessor:
     * Eg: sov.set<0>(100, a-pointer-to-array or array)
     */
    template <
        RSize N,
        typename T2,
        std::enable_if_t<
            std::is_array_v<sov_helper::custom_underlying_type<refl::trait::get_t<N, members>>>,
            bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER void set(RSize index, const T2 &value SOV_DEBUG_INPUT) {
        SOV_DEBUG_OUTPUT_AND_ASSERT(index, used_elem_num)
        using type = sov_helper::custom_underlying_type<refl::trait::get_t<N, members>>;
        using element_type = std::remove_extent_t<type>;
        constexpr RSize length = sizeof(type) / sizeof(element_type);

        // Use base type pointer to access values
        auto acc = RS_ArrayAccessor(
            reinterpret_cast<element_type *>(rs_device_std::get<N>(storage)),
            elem_num,
            index,
            length
        );
        for (RSize i = 0; i < length; i++) acc[i] = value[i];
    }

    /**
     * No corresponding S2
     */

    /**
     * S3
     * Set the value of the Nth (structure) field of the index-th element
     * Eg:
     * struct SecondLevel;
     * struct FirstLevel { float w; SecondLevel sec;}
     * struct SecondLevel { float x, y, z; }
     * sov.set<1>(100, {10, 10, 10})
     * Note: since the mirrored get function returns a copy, you cannot
     * use the get function to set values of structure fields.
     */

    template <RSize N, typename T2, std::enable_if_t<is_nested_sov_v<N>, bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER void set(RSize index, const T2 &value SOV_DEBUG_INPUT) {
        SOV_DEBUG_OUTPUT_AND_ASSERT(index, used_elem_num)
        // last set is a recursive call of S5
        rs_device_std::get<N>(storage).set(index, value SOV_DEBUG_FORWARD);
    }

    /**
     * S4
     * Set the sub fields of the index-th element
     * Eg:
     * struct SecondLevel;
     * struct FirstLevel { float w; SecondLevel sec;}
     * struct SecondLevel { float x, y, z; }
     * sov.set<1, 2>(100, 0)                     // Set sec.x of the 100th element.
     * Note: you can also use sov.get<1, 2>(100) = 0
     * Note:
     */
    template <
        RSize N,
        RSize N1,
        RSize... Ns,
        typename T2,
        std::enable_if_t<is_nested_sov_v<N>, bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER void set(RSize index, const T2 &value SOV_DEBUG_INPUT) {
        SOV_DEBUG_OUTPUT_AND_ASSERT(index, used_elem_num)
        sov_helper::setSub<decltype(*this), T2, N, N1, Ns...>(*this, index, value);
    }

    /**
     * S5
     * Set the value of the index-th element
     * Eg:
     * sov.set(100, {...})
     */
    HOST_DEVICE_INLINE_QUALIFIER void set(RSize index, const T &value SOV_DEBUG_INPUT) {
        SOV_DEBUG_OUTPUT_AND_ASSERT(index, used_elem_num)
        for_each(field_members{}, [&](auto member) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            if constexpr (std::is_class_v<sov_helper::custom_underlying_type<decltype(member)>> &&
                          !sov_helper::is_protected_field_v<decltype(member)>) {
                // recursively call S3
                rs_device_std::get<i>(storage).set(index, member(value));
            } else if constexpr (std::is_array_v<
                                     sov_helper::custom_underlying_type<decltype(member)>>) {
                // recursively call S1.1
                set<i>(index, member(value));
            } else {
                // scalar or protected field
                rs_device_std::get<i>(storage)[index] = member(value);
            }
        });
    }

    /**
     * S5.1
     * Set the value of the index-th element, if the value is constructible
     * from the given value type T2.
     * Eg: sov.set(100, {...})
     */
    template <typename T2>
    HOST_DEVICE_INLINE_QUALIFIER void set(RSize index, const T2 &value SOV_DEBUG_INPUT) {
        static_assert(
            std::is_constructible<T, T2>::value,
            "To use set with T2, Type T must be constructible from type T2."
        );
        set(index, T(value));
    }

    HOST_DEVICE_INLINE_QUALIFIER constexpr RSize size() const { return used_elem_num; }

    /// @brief Returns the allocated capacity per column.
    HOST_DEVICE_INLINE_QUALIFIER constexpr RSize capacity() const { return elem_num; }

    HOST_DEVICE_INLINE_QUALIFIER constexpr bool empty() const { return used_elem_num == 0; }

    HOST_DEVICE_INLINE_QUALIFIER RBigSize baseBytes() const { return sizeof(*this); }

    HOST_DEVICE_INLINE_QUALIFIER RBigSize extraBytes() const { return data.size(); }

    template <typename Member>
    auto &operator[](Member) {
        constexpr auto i = refl::trait::index_of_v<Member, field_members>;
        return rs_device_std::get<i>(storage);
    }

   private:
    friend struct sov_helper::PrivateHelper;

    /**
     * @brief Calculates the total size of memory required by the struct of vectors.
     * @param num The number of elements in the struct of vectors.
     * @param prev_last_member_end The ending size of the previous member.
     * @return The ending size of the last member.
     */
    RBigSize _getRequiredBytes(RSize num, RBigSize prev_last_member_end = 0) {
        RBigSize last_member_end = prev_last_member_end;
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = sov_helper::custom_underlying_type<decltype(member)>;
            if constexpr (!std::is_array_v<type>) {
                last_member_end = sov_helper::PrivateHelper::_getRequiredBytes(
                    rs_device_std::get<i>(storage), num, last_member_end
                );
            } else {
                // Flattens arrays, interleave elements as A1, B1, C1, A2, B2, C2, ...
                // A, B, C are arrays and 1, 2 are sub-indices
                constexpr RSize length = sizeof(type) / sizeof(std::remove_extent_t<type>);
                last_member_end = sov_helper::PrivateHelper::_getRequiredBytes(
                    rs_device_std::get<i>(storage), num * length, last_member_end
                );
            }
        });
        return last_member_end;
    }

    /**
     * @brief Calculates the offset of each member in the struct of vectors.
     * @param num The number of elements in the struct of vectors.
     * @param offsets The vector of offset for each member.
     * @param prev_last_member_end The ending size of the previous member.
     * @return The ending size of the last member.
     */
    RBigSize
    _getOffsets(RSize num, std::vector<RBigSize> &offsets, RBigSize prev_last_member_end = 0) {
        RBigSize last_member_end = prev_last_member_end;
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = sov_helper::custom_underlying_type<decltype(member)>;
            if constexpr (!std::is_array_v<type>) {
                last_member_end = sov_helper::PrivateHelper::_getOffsets(
                    rs_device_std::get<i>(storage), num, offsets, last_member_end
                );
            } else {
                // Flattens arrays, interleave elements as A1, B1, C1, A2, B2, C2, ...
                // A, B, C are arrays and 1, 2 are sub-indices
                constexpr RSize length = sizeof(type) / sizeof(std::remove_extent_t<type>);
                last_member_end = sov_helper::PrivateHelper::_getOffsets(
                    rs_device_std::get<i>(storage), num * length, offsets, last_member_end
                );
            }
        });
        return last_member_end;
    }
    /**
     * @brief Init member pointers with nullptr, and member structure of vectors.
     */
    HOST_DEVICE_QUALIFIER void _initMember() {
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            sov_helper::PrivateHelper::_initMember(rs_device_std::get<i>(storage));
        });
    }

    /**
     * @brief Update member pointers from storage root pointer and the number of elements.
     * And recursively update pointers in member structure of vectors.
     * @param data The root memory region, which is the starting address of the whole struct
     * of vectors.
     * @param num The number of elements in the current struct of vectors.
     * @param used_num The logical used count to propagate to nested sub-vectors.
     * @param prev_last_member_end The total ending size up until the previous member.
     * @return The ending size of the current struct of vectors.
     */
    RBigSize
    _updateMember(RS_MemoryPtr data, RSize num, RSize used_num, RBigSize prev_last_member_end = 0) {
        RBigSize last_member_end = prev_last_member_end;
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = sov_helper::custom_underlying_type<decltype(member)>;
            if constexpr (!std::is_array_v<type>) {
                last_member_end = sov_helper::PrivateHelper::_updateMember(
                    rs_device_std::get<i>(storage), data, num, used_num, last_member_end
                );
            } else {
                // Flattens arrays, interleave elements as A1, B1, C1, A2, B2, C2, ...
                // A, B, C are arrays and 1, 2 are sub-indices
                constexpr RSize length = sizeof(type) / sizeof(std::remove_extent_t<type>);
                last_member_end = sov_helper::PrivateHelper::_updateMember(
                    rs_device_std::get<i>(storage),
                    data,
                    num * length,
                    used_num * length,
                    last_member_end
                );
            }
        });
        return last_member_end;
    }

    /**
     * @brief Copy a portion of the struct of vectors from the old storage root to the new storage
     * root. The storage root could be a host or a device memory address.
     * @param old_data The old storage root pointer.
     * @param old_num The number of elements in the old storage root.
     * @param old_offset The starting offset of the copy in the old storage root.
     * @param new_data The new storage root pointer.
     * @param new_num The number of elements in the new storage root.
     * @param new_offset The starting offset of the copy in the new storage root.
     * @param copy_num The number of elements to copy.
     * @param old_prev_last_member_end The ending size of the previous member in the old storage
     * root.
     * @param new_prev_last_member_end The ending size of the previous member in the new storage
     * root.
     * @return A pair of the ending size of the last member in the old storage root and the new
     * storage root.
     */
    std::pair<RBigSize, RBigSize> _copyMember(
        RS_ConstMemoryPtr old_data,
        RSize old_num,
        RSize old_offset,
        RS_MemoryPtr new_data,
        RSize new_num,
        RSize new_offset,
        RSize copy_num,
        RBigSize old_prev_last_member_end = 0,
        RBigSize new_prev_last_member_end = 0
    ) {
        RBigSize old_last_member_end = old_prev_last_member_end;
        RBigSize new_last_member_end = new_prev_last_member_end;
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = sov_helper::custom_underlying_type<decltype(member)>;
            if constexpr (!std::is_array_v<type>) {
                auto result = sov_helper::PrivateHelper::_copyMember(
                    rs_device_std::get<i>(storage),
                    old_data,
                    old_num,
                    old_offset,
                    new_data,
                    new_num,
                    new_offset,
                    copy_num,
                    old_last_member_end,
                    new_last_member_end
                );
                old_last_member_end = result.first;
                new_last_member_end = result.second;
            } else {
                // Flattens arrays, interleave elements as A1, B1, C1, A2, B2, C2, ...
                // A, B, C are arrays and 1, 2 are sub-indices
                constexpr RSize element_size = sizeof(std::remove_extent_t<type>);
                constexpr RSize length = sizeof(type) / element_size;

                auto old_ptr = old_data + old_last_member_end;
                auto new_ptr = new_data + new_last_member_end;
                for (RSize j = 0; j < length; j++) {
                    if (copy_num > 0) {
                        (new_ptr + element_size * new_offset)
                            .copyFrom(
                                old_ptr + element_size * old_offset, element_size * copy_num, false
                            );
                    }
                    old_ptr += element_size * old_num;
                    new_ptr += element_size * new_num;
                }

                old_last_member_end = RS_ALIGN(
                    old_last_member_end + element_size * length * old_num,
                    sov_helper::SOV_ALIGN_SIZE
                );
                new_last_member_end = RS_ALIGN(
                    new_last_member_end + element_size * length * new_num,
                    sov_helper::SOV_ALIGN_SIZE
                );
            }
        });
        return std::make_pair(old_last_member_end, new_last_member_end);
    }

    template <typename T2>
    RIndex _transverseMemberElementGet(
        RS_MemoryPtr data,
        RSize num,
        T2 &element,
        RIndex element_index,
        const std::vector<RBigSize> &offsets,
        RIndex member_absolute_index = 0
    ) {
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = sov_helper::custom_underlying_type<decltype(member)>;
            if constexpr (!std::is_array_v<type>) {
                member_absolute_index = sov_helper::PrivateHelper::_transverseMemberElementGet(
                    rs_device_std::get<i>(storage),
                    data,
                    num,
                    member(element),
                    element_index,
                    offsets,
                    member_absolute_index
                );
            } else {
                // Flattens arrays, interleave elements as A1, B1, C1, A2, B2, C2, ...
                // A, B, C are arrays and 1, 2 are sub-indices
                using element_type = std::remove_extent_t<type>;
                constexpr RSize element_size = sizeof(element_type);
                constexpr RSize length = sizeof(type) / element_size;
                auto ptr =
                    reinterpret_cast<element_type *>((data + offsets[member_absolute_index]).ptr());
                for (RSize j = 0; j < length; j++) {
                    member(element)[j] = (ptr + num * j)[element_index];
                }
                member_absolute_index++;
            }
        });
        return member_absolute_index;
    }

    template <typename T2>
    RIndex _transverseMemberElementSet(
        RS_MemoryPtr data,
        RSize num,
        const T2 &element,
        RIndex element_index,
        const std::vector<RBigSize> &offsets,
        RIndex member_absolute_index = 0
    ) {
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = sov_helper::custom_underlying_type<decltype(member)>;
            if constexpr (!std::is_array_v<type>) {
                auto field_value = member(element);
                member_absolute_index = sov_helper::PrivateHelper::_transverseMemberElementSet(
                    rs_device_std::get<i>(storage),
                    data,
                    num,
                    field_value,
                    element_index,
                    offsets,
                    member_absolute_index
                );
            } else {
                // Flattens arrays, interleave elements as A1, B1, C1, A2, B2, C2, ...
                // A, B, C are arrays and 1, 2 are sub-indices
                using element_type = std::remove_extent_t<type>;
                constexpr RSize element_size = sizeof(element_type);
                constexpr RSize length = sizeof(type) / element_size;
                auto ptr =
                    reinterpret_cast<element_type *>((data + offsets[member_absolute_index]).ptr());
                for (RSize j = 0; j < length; j++) {
                    (ptr + num * j)[element_index] = member(element)[j];
                }
                member_absolute_index++;
            }
        });
        return member_absolute_index;
    }
};

template <RSize N, typename T>
auto &getArrayPointerOfProperty(RS_StructOfVectors<T> &sov) {
    using member = refl::trait::get_t<N, typename RS_StructOfVectors<T>::members>;
    return sov[member{}];
}

template <RSize N, typename T>
const auto &getArrayPointerOfProperty(const RS_StructOfVectors<T> &sov) {
    using member = refl::trait::get_t<N, typename RS_StructOfVectors<T>::members>;
    return sov[member{}];
}

}  // namespace device

}  // namespace rise

#endif  // RS_SOV_H
