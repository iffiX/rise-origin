// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SOA_H
#define RS_SOA_H

#include <assert.h>

#include <refl.hpp>
#include <vector>

#include "rise_origin/device/platform/rs_platform.h"
#include "rise_origin/utils/rs_conf.h"
#include "rise_origin/utils/rs_def.h"
#include "rs_data_util.h"

#ifndef RS_DEBUG_SOA
#define DEBUG_INPUT
#define DEBUG_OUTPUT_AND_ASSERT(index, max_size) assert(index < max_size);
#define DEBUG_FORWARD
#else
#define DEBUG_INPUT , const char *caller
#define DEBUG_OUTPUT_AND_ASSERT(index, max_size)                                \
    if ((index) >= (max_size)) {                                                \
        printf("Boundary exceeded, Caller: %s, index: %lu\n", caller, (index)); \
    }                                                                           \
    assert((index) < (max_size));
#define DEBUG_FORWARD , ""
#endif
#define Pr(struct_name, property) \
    refl_impl::metadata::type_info__<struct_name>::field##_##property##_index

namespace rise {

namespace device {

namespace soa_helper {

constexpr RSize SOA_ALIGN_SIZE = RS_MEMORY_ALIGN_SIZE;

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
 * Member lists of fields in SoA
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

// These helper functions needs to call private methods of the SoA,
// Therefore they can only be encapsulated in a friend class
struct PrivateHelper {
    // Provide unified access to regular fields and nested fields
    // For regular fields, T is a pointer type
    // For nested fields, T is RS_StructOfArrays<SomeNestedField>
    template <typename T>
    static std::enable_if_t<std::is_class_v<T>, RSize>
    _getRequiredBytes(T &t, RSize num, RSize prev_last_member_end) {
        return t._getRequiredBytes(num, prev_last_member_end);
    }

    template <typename T>
    static std::enable_if_t<!std::is_class_v<T>, RSize>
    _getRequiredBytes(T &t, RSize num, RSize prev_last_member_end) {
        using element_type = std::remove_extent_t<std::remove_pointer_t<T>>;
        return RS_ALIGN(prev_last_member_end + sizeof(element_type) * num, SOA_ALIGN_SIZE);
    }

    template <typename T>
    static std::enable_if_t<std::is_class_v<T>, RSize>
    _getOffsets(T &t, RSize num, std::vector<RSize> &offsets, RSize prev_last_member_end) {
        return t._getOffsets(num, offsets, prev_last_member_end);
    }

    template <typename T>
    static std::enable_if_t<!std::is_class_v<T>, RSize>
    _getOffsets(T &t, RSize num, std::vector<RSize> &offsets, RSize prev_last_member_end) {
        using element_type = std::remove_extent_t<std::remove_pointer_t<T>>;
        offsets.push_back(prev_last_member_end);
        RSize last_member_end =
            RS_ALIGN(prev_last_member_end + sizeof(element_type) * num, SOA_ALIGN_SIZE);
        return last_member_end;
    }

    template <typename T>
    HOST_DEVICE_QUALIFIER static std::enable_if_t<std::is_class_v<T>, void> _initMember(T &t) {
        t.elem_num = 0;
        t.data = {};
        t._initMember();
    }

    template <typename T>
    HOST_DEVICE_QUALIFIER static std::enable_if_t<!std::is_class_v<T>, void> _initMember(T &t) {
        t = nullptr;
    }

    template <typename T>
    static std::enable_if_t<std::is_class_v<T>, RSize>
    _updateMember(T &t, RS_MemoryPtr data, RSize num, RSize prev_last_member_end) {
        t.elem_num = num;
        return t._updateMember(data, num, prev_last_member_end);
    }

    template <typename T>
    static std::enable_if_t<!std::is_class_v<T>, RSize>
    _updateMember(T &t, RS_MemoryPtr data, RSize num, RSize prev_last_member_end) {
        using element_type = std::remove_extent_t<std::remove_pointer_t<T>>;
        if (num == 0) {
            t = nullptr;
            return 0;
        } else {
            t = reinterpret_cast<T>((data + prev_last_member_end).ptr());
            return RS_ALIGN(prev_last_member_end + sizeof(element_type) * num, SOA_ALIGN_SIZE);
        }
    }

    template <typename T>
    static std::enable_if_t<std::is_class_v<T>, std::pair<RSize, RSize>> _copyMember(
        T &t,
        RS_ConstMemoryPtr old_data,
        RSize old_num,
        RSize old_offset,
        RS_MemoryPtr new_data,
        RSize new_num,
        RSize new_offset,
        RSize copy_num,
        RSize old_prev_last_member_end,
        RSize new_prev_last_member_end
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
    static std::enable_if_t<!std::is_class_v<T>, std::pair<RSize, RSize>> _copyMember(
        T &t,
        RS_ConstMemoryPtr old_data,
        RSize old_num,
        RSize old_offset,
        RS_MemoryPtr new_data,
        RSize new_num,
        RSize new_offset,
        RSize copy_num,
        RSize old_prev_last_member_end,
        RSize new_prev_last_member_end
    ) {
        using element_type = std::remove_extent_t<std::remove_pointer_t<T>>;
        (new_data + new_prev_last_member_end + sizeof(element_type) * new_offset)
            .copyFrom(
                old_data + old_prev_last_member_end + sizeof(element_type) * old_offset,
                sizeof(element_type) * copy_num,
                false
            );
        RSize o_p_e =
            RS_ALIGN(old_prev_last_member_end + sizeof(element_type) * old_num, SOA_ALIGN_SIZE);
        RSize n_p_e =
            RS_ALIGN(new_prev_last_member_end + sizeof(element_type) * new_num, SOA_ALIGN_SIZE);
        return std::make_pair(o_p_e, n_p_e);
    }

    template <typename T, typename T2>
    static typename std::enable_if<std::is_class_v<T>, RIndex>::type _transverseMemberElementGet(
        T &t,
        RS_MemoryPtr data,
        RSize num,
        T2 &element,
        RIndex element_index,
        const std::vector<RSize> &offsets,
        RIndex member_absolute_index
    ) {
        return t._transverseMemberElementGet(
            data, num, element, element_index, offsets, member_absolute_index
        );
    }

    template <typename T, typename T2>
    static typename std::enable_if<!std::is_class_v<T>, RIndex>::type _transverseMemberElementGet(
        T &t,
        RS_MemoryPtr data,
        RSize num,
        T2 &element,
        RIndex element_index,
        const std::vector<RSize> &offsets,
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
        const std::vector<RSize> &offsets,
        RIndex member_absolute_index
    ) {
        return t._transverseMemberElementSet(
            data, num, element, element_index, offsets, member_absolute_index
        );
    }

    template <typename T, typename T2>
    static typename std::enable_if<!std::is_class_v<T>, RIndex>::type _transverseMemberElementSet(
        T &t,
        RS_MemoryPtr data,
        RSize num,
        T2 &element,
        RIndex element_index,
        const std::vector<RSize> &offsets,
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
    // T2 is a const array accessor instance
    constexpr RSize length = sizeof(T) / sizeof(std::remove_extent_t<T>);
    for (RSize i = 0; i < length; i++) t[i] = value[i];
}

template <typename T, typename T2>
HOST_DEVICE_QUALIFIER static std::enable_if_t<!std::is_array_v<T>> assign(T &t, T2 &value) {
    t = value;
}

template <typename T, RSize N>
HOST_DEVICE_INLINE_QUALIFIER decltype(auto) getSub(T &soa, RSize index) {
    return soa.template get<N>(index DEBUG_FORWARD);
}

template <typename T, RSize N, RSize N1, RSize... Ns>
HOST_DEVICE_INLINE_QUALIFIER decltype(auto) getSub(T &soa, RSize index) {
    return getSub<std::remove_reference_t<decltype(rs_device_std::get<N>(soa.storage))>, N1, Ns...>(
        rs_device_std::get<N>(soa.storage), index
    );
}

template <typename T, typename T2, RSize N>
HOST_DEVICE_INLINE_QUALIFIER void setSub(T &soa, RSize index, const T2 &value) {
    soa.template set<N>(index, value DEBUG_FORWARD);
}

template <typename T, typename T2, RSize N, RSize N1, RSize... Ns>
HOST_DEVICE_INLINE_QUALIFIER void setSub(T &soa, RSize index, const T2 &value) {
    setSub<std::remove_reference_t<decltype(rs_device_std::get<N>(soa.storage))>, T2, N1, Ns...>(
        rs_device_std::get<N>(soa.storage), index, value
    );
}
}  // namespace soa_helper

template <typename T>
struct RS_ArrayAccessor {
    T *start_address;
    RSize offset;
    RSize stride;
    RSize max_length;
    HOST_DEVICE_INLINE_QUALIFIER
    RS_ArrayAccessor(T *start_address, RSize stride, RSize offset, RSize max_length)
        : start_address(start_address), stride(stride), offset(offset), max_length(max_length) {};
    HOST_DEVICE_INLINE_QUALIFIER T &operator[](RSize index) {
        assert(index < max_length);
        return *(start_address + index * stride + offset);
    }
};

template <typename T>
struct RS_ConstArrayAccessor {
    const T *start_address;
    RSize offset;
    RSize stride;
    RSize max_length;
    HOST_DEVICE_INLINE_QUALIFIER
    RS_ConstArrayAccessor(const T *start_address, RSize stride, RSize offset, RSize max_length)
        : start_address(start_address), stride(stride), offset(offset), max_length(max_length) {};
    HOST_DEVICE_INLINE_QUALIFIER const T &operator[](RSize index) {
        assert(index < max_length);
        return *(start_address + index * stride + offset);
    }
};

/**
 * @brief RS_StructOfArrays is a template class for representing structure of arrays on the GPU.
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
 * And suppose we pass RS_StructOfArrays<B> to a global kernel:
 * __global__ kernel(RS_StructOfArrays<B> soa) {
 *     soa.get(0); // Returns first structure B
 *     soa.get<Pr(B, struct_a)>(1); // Returns struct_a in second B
 *     auto array_accessor = soa.get<Pr(B, array)>(1);
 *     array_accessor[0]; // Access first element of array in second B
 *     soa.set<Pr(B, struct_a), Pr(A, v), Pr(Vector, x)>(1); // Nested access
 * }
 */
template <typename T>
class RS_StructOfArrays {
   public:
    using ObjectType = T;

   private:
    using members = refl::member_list<T>;
    using field_members = soa_helper::field_member_list<std::remove_cv_t<T>>;
    using unreadable_field_members = soa_helper::unreadable_field_member_list<std::remove_cv_t<T>>;
    static_assert(field_members::size > 0, "Type has no fields!");
    static_assert(unreadable_field_members::size == 0, "Type has unreadable fields!");

    template <typename FieldMember>
    struct make_storage {
        using underlying = soa_helper::custom_underlying_type<FieldMember>;
        using type = typename std::conditional<
            soa_helper::is_protected_field_v<FieldMember>,
            underlying *,
            typename std::conditional<
                std::is_class_v<underlying>,
                RS_StructOfArrays<underlying>,
                underlying *>::type>::type;
    };

    // Helper: true if the Nth member is a class type that should be decomposed (not protected)
    template <RSize N>
    static constexpr bool is_nested_soa_v =
        std::is_class_v<soa_helper::custom_underlying_type<refl::trait::get_t<N, members>>> &&
        !soa_helper::is_protected_field_v<refl::trait::get_t<N, members>>;

    // Helper: true if the Nth member is a scalar or protected class (stored as T*)
    template <RSize N>
    static constexpr bool is_direct_field_v =
        !std::is_class_v<soa_helper::custom_underlying_type<refl::trait::get_t<N, members>>> ||
        soa_helper::is_protected_field_v<refl::trait::get_t<N, members>>;

    template <typename>
    struct as_soa_tuple;

    template <template <typename...> typename T2, typename... Ts>
    struct as_soa_tuple<T2<Ts...>> {
        using type = rs_device_std::tuple<Ts...>;
    };

    template <typename T2>
    using as_tuple_t = typename as_soa_tuple<T2>::type;

   public:
    /*****************************************************************
     * Data
     *****************************************************************/
    // The root memory region
    // For sub soa, data is always null
    RS_MemoryPtr data;

    // Number of stored virtual structures
    RSize elem_num = 0;

    // Storage is a std::tuple containing device side memory pointers to
    // individual arrays Each array contains a property of the virtual
    // "Structure". Each pointer points to a aligned segment in storage_root.
    as_tuple_t<refl::trait::map_t<make_storage, field_members>> storage;

    /*****************************************************************
     * Methods
     *****************************************************************/
    RS_StructOfArrays() { _initMember(); }

    /**
     * @brief Construct a new RS_StructOfArrays object with the given provider.
     */
    explicit RS_StructOfArrays(RS_MemoryProvider *provider) { init(provider); }

    /**
     * @brief Construct a new RS_StructOfArrays object with the given initial size and provider.
     * @param init_size The initial size of the structure of arrays.
     * @param provider The memory provider.
     */
    RS_StructOfArrays(RSize init_size, RS_MemoryProvider *provider) { init(init_size, provider); }

    /**
     * @brief Initialize an empty structure of array with the given provider.
     * @param provider The memory provider.
     */
    void init(RS_MemoryProvider *provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        data = {nullptr, 0, 0, provider->device, provider};
        elem_num = 0;
        _initMember();
    }

    /**
     * @brief Initialize the structure of arrays with the given initial size and provider.
     * @param init_size The initial size of the structure of arrays.
     * @param provider The memory provider.
     */
    void init(RSize init_size, RS_MemoryProvider *provider) {
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        data = {nullptr, 0, 0, provider->device, provider};
        elem_num = 0;
        _initMember();
        resize(init_size, false);
    }

    /**
     * @brief Copy constructor.
     * @warning It is a shallow copy which only copies pointer and size.
     */
    RS_StructOfArrays(const RS_StructOfArrays &other) = default;

    /**
     * @brief Frees the GPU memory associated with the SoA.
     * @warning It will free the data, all related SoAs will be affected.
     */
    void free() {
        data.free();
        elem_num = 0;
        _updateMember(data, 0);
    }

    /**
     * @brief Resize the structure of arrays to new size
     * By default, resize will copy data, but remember copying each segment of member data
     * to their new location is expensive.
     * If copy, when resized smaller, the excessive data will be truncated.
     *
     * @warning It will free the data, all related arrays will be affected.
     *
     * @note Stream will be synchronized at the end of the function.
     */
    void resize(RSize new_size, bool copy = true) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);

        // Check if the new size is the same as the old size
        if (new_size == elem_num) return;

        if (elem_num != 0 && new_size != 0 && copy) {
            RSize required_bytes = _getRequiredBytes(new_size);
            auto new_data = data.provider()->allocate(required_bytes, false);
            _copyMember(data, elem_num, 0, new_data, new_size, 0, RS_MIN(elem_num, new_size));
            data.free(false);
            data = new_data;
            elem_num = new_size;
            _updateMember(data, new_size);
        } else {
            RS_MemoryProvider *provider = data.provider();
            // Free old immediately to reduce memory footprint
            if (data != nullptr) data.free(false);
            RSize required_bytes = _getRequiredBytes(new_size);
            data = provider->allocate(required_bytes, false);
            elem_num = new_size;
            _updateMember(data, new_size);
        }
        data.provider()->sync();
    }

    /**
     * @brief Fill the struct of arrays with the given value.
     *
     * @param input Value to fill with.
     */
    void fill(const T &input) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);

        if (elem_num > 0) {
            auto tmp = new T[elem_num];
            for (RSize i = 0; i < elem_num; i++) tmp[i] = input;
            write(tmp);
            delete[] tmp;
        }
    }

    /**
     * @brief Fill a slice of the struct of arrays with the given value.
     *
     * @param input Value to fill with.
     * @param num The number of elements to be filled.
     * @param offset The starting offset of the slice in the structure of arrays.
     */
    void fillSlice(const T &input, RSize num, RSize offset) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_BOUND(
            num + offset <= elem_num, "num={}, offset={}, elem_num={}", num, offset, elem_num
        );

        if (num > 0) {
            std::vector<RSize> offsets;
            _getOffsets(num, offsets);
            RSize required_size = _getRequiredBytes(num);
            auto tmp_data = new std::byte[required_size];
            for (RIndex i = 0; i < num; i++) {
                _transverseMemberElementSet({tmp_data, required_size}, num, input, i, offsets);
            }
            _copyMember({tmp_data, required_size}, num, 0, data, elem_num, offset, num);
            delete[] tmp_data;
        }
    }

    /**
     * @brief Write host data to the device side of the struct of arrays.
     *
     * @param input The host data to be written.
     * @param num The number of elements to be written, by default (num = 0),
     * it assumes the input has the same number of elements as the SoA.
     */
    void write(const T *input, RSize num = 0) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            input != nullptr || num == 0,
            "When writing to a non-empty SOA, input pointer must not be null."
        );

        if (num == 0) num = elem_num;
        if (num > elem_num) {
            RS_THROW(RS_OutOfBoundsException(
                "Trying to write {} elements starting from 0 to a SOA of size {}", num, elem_num
            ));
        }
        if (num > 0) {
            std::vector<RSize> offsets;
            _getOffsets(elem_num, offsets);
            auto tmp_data = new std::byte[data.size()];
            for (RIndex i = 0; i < num; i++) {
                _transverseMemberElementSet(
                    {tmp_data, data.size()}, elem_num, input[i], i, offsets
                );
            }
            // We can just perform copy instead of _copyMember for better efficiency
            data.copyFrom(RS_ConstMemoryPtr(tmp_data, data.size()), data.size());
            delete[] tmp_data;
        }
    }

    /**
     * @brief Write a slice of host data to the device side of the struct of arrays.
     *
     * @param input The host data to be written.
     * @param num The number of elements to be written.
     * @param offset The starting offset of the slice in the structure of arrays.
     */
    void writeSlice(const T *input, RSize num, RSize offset) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            input != nullptr || num == 0,
            "When writing to a non-empty slice of a SOA, input pointer must not be null."
        );
        RS_DATA_EXPECT_BOUND(
            num + offset <= elem_num, "num={}, offset={}, elem_num={}", num, offset, elem_num
        );

        if (num > 0) {
            std::vector<RSize> offsets;
            _getOffsets(num, offsets);
            RSize required_size = _getRequiredBytes(num);
            auto tmp_data = new std::byte[required_size];
            for (RIndex i = 0; i < num; i++) {
                _transverseMemberElementSet({tmp_data, required_size}, num, input[i], i, offsets);
            }
            _copyMember({tmp_data, required_size}, num, 0, data, elem_num, offset, num);
            delete[] tmp_data;
        }
    }

    /**
     * @brief Copies data from another RS_StructOfArrays. Resize as needed.
     */
    void copy(const RS_StructOfArrays &soa) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_COPY_SRC_INIT(soa.data.provider() != nullptr);

        if (data == soa.data) return;
        if (elem_num != soa.elem_num) resize(soa.elem_num, false);
        if (elem_num > 0) _copyMember(soa.data, elem_num, 0, data, elem_num, 0, elem_num);
    }

    /**
     * @brief Copies data from device to host.
     *
     * @param output The address to store the data
     */
    void read(T *output) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            output != nullptr || elem_num == 0,
            "When reading from a non-empty SOA, output pointer must not be null."
        );

        if (elem_num > 0) {
            std::vector<RSize> offsets;
            _getOffsets(elem_num, offsets);
            auto tmp_data = new std::byte[data.size()];
            data.copyTo(RS_MemoryPtr(tmp_data, data.size()), data.size());
            for (RIndex i = 0; i < elem_num; i++) {
                _transverseMemberElementGet(
                    {tmp_data, data.size()}, elem_num, output[i], i, offsets
                );
            }
            delete[] tmp_data;
        }
    }

    /**
     * @brief Read a slice of device data of the struct of arrays to the host side.
     *
     * @param output The host data to write to.
     * @param num The number of elements to be read.
     * @param offset The starting offset of the slice in the structure of arrays.
     */
    void readSlice(T *output, RSize num, RSize offset) {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_ASSERT_ARGUMENT(
            output != nullptr || num == 0,
            "When reading a non-empty slice of a SOA, output pointer must not be null."
        );
        RS_DATA_EXPECT_BOUND(
            num + offset <= elem_num, "num={}, offset={}, elem_num={}", num, offset, elem_num
        );
        if (num > 0) {
            RSize required_size = _getRequiredBytes(num);
            auto tmp_data = new std::byte[required_size];
            _copyMember(data, elem_num, offset, {tmp_data, required_size}, num, 0, num);
            std::vector<RSize> offsets;
            _getOffsets(num, offsets);
            for (RIndex i = 0; i < num; i++) {
                _transverseMemberElementGet({tmp_data, required_size}, num, output[i], i, offsets);
            }
            delete[] tmp_data;
        }
    }

    /// @brief Moves data to specified provider and returns a new SoA
    RS_StructOfArrays move(RS_MemoryProvider *provider) const {
        RS_DATA_EXPECT_INIT(data.provider() != nullptr);
        RS_DATA_EXPECT_PROVIDER_NOT_NULL(provider);

        if (elem_num == 0) return RS_StructOfArrays(provider);
        RS_StructOfArrays new_soa(elem_num, provider);
        new_soa.copy(*this);
        return new_soa;
    }

    [[nodiscard]] RS_MemoryProvider *provider() const { return data.provider(); }

    [[nodiscard]] RS_DeviceLocation device() const { return data.device(); }

    /**
     * G1
     * Access the reference of the Nth (non-structure, non-array) field of the index-th
     * element: Eg: soa.get<0>(100) = 0
     */
    template <
        RSize N,
        std::enable_if_t<
            is_direct_field_v<N> && !std::is_array_v<soa_helper::custom_underlying_type<
                                        refl::trait::get_t<N, RS_StructOfArrays<T>::members>>>,
            bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER auto &get(RSize index DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, elem_num)
        assert(index < elem_num);
        return rs_device_std::get<N>(storage)[index];
    }

    /**
     * G1.1
     * Access the Nth array field of the index-th element using an accessor:
     * Eg: soa.get<0>(100)[3] = 0
     */
    template <
        RSize N,
        std::enable_if_t<
            std::is_array_v<soa_helper::custom_underlying_type<refl::trait::get_t<N, members>>>,
            bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER auto get(RSize index DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, elem_num)
        using type = soa_helper::custom_underlying_type<refl::trait::get_t<N, members>>;
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
     * element: Eg: x = soa.get<0>(100)
     */
    template <
        RSize N,
        std::enable_if_t<
            is_direct_field_v<N> && !std::is_array_v<soa_helper::custom_underlying_type<
                                        refl::trait::get_t<N, members>>>,
            bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER const auto &get(RSize index DEBUG_INPUT) const {
        DEBUG_OUTPUT_AND_ASSERT(index, elem_num)
        return rs_device_std::get<N>(storage)[index];
    }

    /**
     * G2.1
     * Access the Nth array field of the index-th element using a const accessor:
     * Eg: x = soa.get<0>(100)[3]
     */
    template <
        RSize N,
        std::enable_if_t<
            std::is_array_v<soa_helper::custom_underlying_type<refl::trait::get_t<N, members>>>,
            bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER auto get(RSize index DEBUG_INPUT) const {
        DEBUG_OUTPUT_AND_ASSERT(index, elem_num)
        using type = soa_helper::custom_underlying_type<refl::trait::get_t<N, members>>;
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
     * soa.get<1>(100).x                     // x is the field of a copy of sub field sec.
     *
     * Note: the method calls get(index) internally, which access every single element of
     * the second level structure in an aligned manner and copy them back.
     *
     * Note: returned value is const to prevent unintentional modify, since this would
     * have no effect
     */

    template <RSize N, std::enable_if_t<is_nested_soa_v<N>, bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER const auto get(RSize index DEBUG_INPUT) const {
        DEBUG_OUTPUT_AND_ASSERT(index, elem_num)
        // last get is a recursive call of G5
        return rs_device_std::move(rs_device_std::get<N>(storage).get(index DEBUG_FORWARD));
    }

    /**
     * G4
     * Access the sub fields of the index-th element
     * Eg:
     * struct SecondLevel;
     * struct FirstLevel { float w; SecondLevel sec;}
     * struct SecondLevel { float x, y, z; }
     * soa.get<1, 2>(100)                     // A reference to sec.x of the 100th
     * element.
     *
     */
    template <RSize N, RSize N1, RSize... Ns, std::enable_if_t<is_nested_soa_v<N>, bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER decltype(auto) get(RSize index DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, elem_num)
        return soa_helper::getSub<decltype(*this), N, N1, Ns...>(*this, index);
    }

    /**
     * G5
     * Access a copy of the index-th element
     * Eg:
     * soa.get(100)
     */
    HOST_DEVICE_INLINE_QUALIFIER T get(RSize index DEBUG_INPUT) const {
        DEBUG_OUTPUT_AND_ASSERT(index, elem_num)
        T t{};
        for_each(field_members{}, [&](auto member) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            // recursive call of G1.1, G2, G3
            soa_helper::assign(member(t), get<i>(index DEBUG_FORWARD));
        });
        return rs_device_std::move(t);
    }

    /**
     * S1
     * Set the value of the Nth (non-structure, non-array) field of the index-th element:
     * Eg: soa.set<0>(100, 0)
     * Note: You can also use soa.get<0>(100) = 0
     */
    template <
        RSize N,
        typename T2,
        std::enable_if_t<
            is_direct_field_v<N> && !std::is_array_v<soa_helper::custom_underlying_type<
                                        refl::trait::get_t<N, members>>>,
            bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER void set(RSize index, const T2 &value DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, elem_num)
        rs_device_std::get<N>(storage)[index] = value;
    }

    /**
     * S1.1
     * Set Nth array field of the index-th element using an accessor:
     * Eg: soa.set<0>(100, a-pointer-to-array or array)
     */
    template <
        RSize N,
        typename T2,
        std::enable_if_t<
            std::is_array_v<soa_helper::custom_underlying_type<refl::trait::get_t<N, members>>>,
            bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER void set(RSize index, const T2 &value DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, elem_num)
        using type = soa_helper::custom_underlying_type<refl::trait::get_t<N, members>>;
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
     * soa.set<1>(100, {10, 10, 10})
     * Note: since the mirrored get function returns a copy, you cannot
     * use the get function to set values of structure fields.
     */

    template <RSize N, typename T2, std::enable_if_t<is_nested_soa_v<N>, bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER void set(RSize index, const T2 &value DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, elem_num)
        // last set is a recursive call of S5
        rs_device_std::get<N>(storage).set(index, value DEBUG_FORWARD);
    }

    /**
     * S4
     * Set the sub fields of the index-th element
     * Eg:
     * struct SecondLevel;
     * struct FirstLevel { float w; SecondLevel sec;}
     * struct SecondLevel { float x, y, z; }
     * soa.set<1, 2>(100, 0)                     // Set sec.x of the 100th element.
     * Note: you can also use soa.get<1, 2>(100) = 0
     * Note:
     */
    template <
        RSize N,
        RSize N1,
        RSize... Ns,
        typename T2,
        std::enable_if_t<is_nested_soa_v<N>, bool> = true>
    HOST_DEVICE_INLINE_QUALIFIER void set(RSize index, const T2 &value DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, elem_num)
        soa_helper::setSub<decltype(*this), T2, N, N1, Ns...>(*this, index, value);
    }

    /**
     * S5
     * Set the value of the index-th element
     * Eg:
     * soa.set(100, {...})
     */
    HOST_DEVICE_INLINE_QUALIFIER void set(RSize index, const T &value DEBUG_INPUT) {
        DEBUG_OUTPUT_AND_ASSERT(index, elem_num)
        for_each(field_members{}, [&](auto member) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            if constexpr (std::is_class_v<soa_helper::custom_underlying_type<decltype(member)>> &&
                          !soa_helper::is_protected_field_v<decltype(member)>) {
                // recursively call S3
                rs_device_std::get<i>(storage).set(index, member(value));
            } else if constexpr (std::is_array_v<
                                     soa_helper::custom_underlying_type<decltype(member)>>) {
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
     * Eg: soa.set(100, {...})
     */
    template <typename T2>
    HOST_DEVICE_INLINE_QUALIFIER void set(RSize index, const T2 &value DEBUG_INPUT) {
        static_assert(
            std::is_constructible<T, T2>::value,
            "To use set with T2, Type T must be constructible from type T2."
        );
        set(index, T(value));
    }

    HOST_DEVICE_INLINE_QUALIFIER RSize size() const { return elem_num; }

    HOST_DEVICE_INLINE_QUALIFIER constexpr bool empty() const { return elem_num == 0; }

    HOST_DEVICE_INLINE_QUALIFIER RSize baseBytes() const { return sizeof(*this); }

    HOST_DEVICE_INLINE_QUALIFIER RSize extraBytes() const { return data.size(); }

    template <typename Member>
    auto &operator[](Member) {
        constexpr auto i = refl::trait::index_of_v<Member, field_members>;
        return rs_device_std::get<i>(storage);
    }

   private:
    friend struct soa_helper::PrivateHelper;

    /**
     * @brief Calculates the total size of memory required by the struct of arrays.
     * @param num The number of elements in the struct of arrays.
     * @param prev_last_member_end The ending size of the previous member.
     * @return The ending size of the last member.
     */
    RSize _getRequiredBytes(RSize num, RSize prev_last_member_end = 0) {
        RSize last_member_end = prev_last_member_end;
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = soa_helper::custom_underlying_type<decltype(member)>;
            if constexpr (!std::is_array_v<type>) {
                last_member_end = soa_helper::PrivateHelper::_getRequiredBytes(
                    rs_device_std::get<i>(storage), num, last_member_end
                );
            } else {
                // Flattens arrays, interleave elements as A1, B1, C1, A2, B2, C2, ...
                // A, B, C are arrays and 1, 2 are sub-indices
                constexpr RSize length = sizeof(type) / sizeof(std::remove_extent_t<type>);
                last_member_end = soa_helper::PrivateHelper::_getRequiredBytes(
                    rs_device_std::get<i>(storage), num * length, last_member_end
                );
            }
        });
        return last_member_end;
    }

    /**
     * @brief Calculates the offset of each member in the struct of arrays.
     * @param num The number of elements in the struct of arrays.
     * @param offsets The vector of offset for each member.
     * @param prev_last_member_end The ending size of the previous member.
     * @return The ending size of the last member.
     */
    RSize _getOffsets(RSize num, std::vector<RSize> &offsets, RSize prev_last_member_end = 0) {
        RSize last_member_end = prev_last_member_end;
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = soa_helper::custom_underlying_type<decltype(member)>;
            if constexpr (!std::is_array_v<type>) {
                last_member_end = soa_helper::PrivateHelper::_getOffsets(
                    rs_device_std::get<i>(storage), num, offsets, last_member_end
                );
            } else {
                // Flattens arrays, interleave elements as A1, B1, C1, A2, B2, C2, ...
                // A, B, C are arrays and 1, 2 are sub-indices
                constexpr RSize length = sizeof(type) / sizeof(std::remove_extent_t<type>);
                last_member_end = soa_helper::PrivateHelper::_getOffsets(
                    rs_device_std::get<i>(storage), num * length, offsets, last_member_end
                );
            }
        });
        return last_member_end;
    }
    /**
     * @brief Init member pointers with nullptr, and member structure of arrays.
     */
    HOST_DEVICE_QUALIFIER void _initMember() {
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            soa_helper::PrivateHelper::_initMember(rs_device_std::get<i>(storage));
        });
    }

    /**
     * @brief Update member pointers from storage root pointer and the number of elements.
     * And recursively update pointers in member structure of arrays.
     * @param data The root memory region, which is the starting address of the whole struct
     * of arrays.
     * @param num The number of elements in the current struct of array.
     * @param prev_last_member_end The total ending size up until the previous member.
     * @return The ending size of the current struct of array.
     */
    RSize _updateMember(RS_MemoryPtr data, RSize num, RSize prev_last_member_end = 0) {
        RSize last_member_end = prev_last_member_end;
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = soa_helper::custom_underlying_type<decltype(member)>;
            if constexpr (!std::is_array_v<type>) {
                last_member_end = soa_helper::PrivateHelper::_updateMember(
                    rs_device_std::get<i>(storage), data, num, last_member_end
                );
            } else {
                // Flattens arrays, interleave elements as A1, B1, C1, A2, B2, C2, ...
                // A, B, C are arrays and 1, 2 are sub-indices
                constexpr RSize length = sizeof(type) / sizeof(std::remove_extent_t<type>);
                last_member_end = soa_helper::PrivateHelper::_updateMember(
                    rs_device_std::get<i>(storage), data, num * length, last_member_end
                );
            }
        });
        return last_member_end;
    }

    /**
     * @brief Copy a portion of the struct of arrays from the old storage root to the new storage
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
    std::pair<RSize, RSize> _copyMember(
        RS_ConstMemoryPtr old_data,
        RSize old_num,
        RSize old_offset,
        RS_MemoryPtr new_data,
        RSize new_num,
        RSize new_offset,
        RSize copy_num,
        RSize old_prev_last_member_end = 0,
        RSize new_prev_last_member_end = 0
    ) {
        RSize old_last_member_end = old_prev_last_member_end;
        RSize new_last_member_end = new_prev_last_member_end;
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = soa_helper::custom_underlying_type<decltype(member)>;
            if constexpr (!std::is_array_v<type>) {
                auto result = soa_helper::PrivateHelper::_copyMember(
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
                    (new_ptr + element_size * new_offset)
                        .copyFrom(
                            old_ptr + element_size * old_offset, element_size * copy_num, false
                        );
                    old_ptr += element_size * old_num;
                    new_ptr += element_size * new_num;
                }

                old_last_member_end = RS_ALIGN(
                    old_last_member_end + element_size * length * old_num,
                    soa_helper::SOA_ALIGN_SIZE
                );
                new_last_member_end = RS_ALIGN(
                    new_last_member_end + element_size * length * new_num,
                    soa_helper::SOA_ALIGN_SIZE
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
        const std::vector<RSize> &offsets,
        RIndex member_absolute_index = 0
    ) {
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = soa_helper::custom_underlying_type<decltype(member)>;
            if constexpr (!std::is_array_v<type>) {
                member_absolute_index = soa_helper::PrivateHelper::_transverseMemberElementGet(
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
        T2 &element,
        RIndex element_index,
        const std::vector<RSize> &offsets,
        RIndex member_absolute_index = 0
    ) {
        for_each(field_members{}, [&](auto member, RSize index) {
            constexpr auto i = refl::trait::index_of_v<decltype(member), field_members>;
            using type = soa_helper::custom_underlying_type<decltype(member)>;
            if constexpr (!std::is_array_v<type>) {
                member_absolute_index = soa_helper::PrivateHelper::_transverseMemberElementSet(
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
                    (ptr + num * j)[element_index] = member(element)[j];
                }
                member_absolute_index++;
            }
        });
        return member_absolute_index;
    }
};

template <RSize N, typename T>
auto &getArrayPointerOfProperty(RS_StructOfArrays<T> &soa) {
    using member = refl::trait::get_t<N, typename RS_StructOfArrays<T>::members>;
    return soa[member{}];
}

template <RSize N, typename T>
const auto &getArrayPointerOfProperty(const RS_StructOfArrays<T> &soa) {
    using member = refl::trait::get_t<N, typename RS_StructOfArrays<T>::members>;
    return soa[member{}];
}

}  // namespace device

}  // namespace rise

#endif  // RS_SOA_H