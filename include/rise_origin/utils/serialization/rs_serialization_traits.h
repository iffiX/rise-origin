// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SERIALIZATION_TYPE_TRAITS_H
#define RS_SERIALIZATION_TYPE_TRAITS_H

#include <string_view>
#include <type_traits>
#include <vector>

#include "rise_origin/utils/data/rs_data_traits.h"
#include "rise_origin/utils/data/rs_vector_view.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/string/rs_string_traits.h"

namespace rise {

/*****************************************************************
 * Forward Declarations
 *****************************************************************/
template <typename T>
struct RS_Serialize;

/*****************************************************************
 * Serialization Traits
 *****************************************************************/
/**
 * @brief Checks if a type is serializable as a binary
 *
 * @tparam T The type to check
 * @return True if the type is serializable as a binary, false otherwise
 */
template <typename T>
struct RS_IsSerializableBinary {
    static const bool value = false;
};

template <>
struct RS_IsSerializableBinary<RS_VectorView<RByte>> {
    static const bool value = true;
};

template <>
struct RS_IsSerializableBinary<RS_VectorView<char>> {
    static const bool value = true;
};

template <>
struct RS_IsSerializableBinary<RS_BytesView> {
    static const bool value = true;
};

template <>
struct RS_IsSerializableBinary<RS_Bytes> {
    static const bool value = true;
};

template <>
struct RS_IsSerializableBinary<std::vector<RByte>> {
    static const bool value = true;
};

template <>
struct RS_IsSerializableBinary<std::vector<char>> {
    static const bool value = true;
};

/**
 * @brief Checks if a class is defined with a serialize method template
 *
 * @code
 * class MyClass {
 *   public:
 *     int value;
 *     std::string name;
 *     MyClass(int v, std::string n) : value(v), name(n) {}
 *
 *     template <typename Serializer>
 *     void serialize(Serializer& s) {
 *         s("value", value);
 *         s("name", name);
 *     }
 * };
 *
 * @tparam T The type to check
 * @return True if the class is defined with a serialize method, false otherwise
 */
template <typename T>
struct RS_IsSerializableClass {
   private:
    struct FakeSerializer {
        template <typename Value>
        void operator()(const std::string_view& key, Value& value) {}
    };
    // Test, the class should be serializable when we pass a serializer reference
    template <typename U>
    static auto test(int)
        -> decltype(std::declval<U>().serialize(std::declval<FakeSerializer&>()), std::true_type{});

    template <typename>
    static std::false_type test(...);

   public:
    static const bool value = std::is_class_v<T> && decltype(test<T>(0))::value;
};

/**
 * @brief Checks if a type is externally serializable with a defined RS_Serialize<T> specialization
 *
 * @tparam T The type to check
 * @return True if the type is externally serializable with a defined RS_Serialize<T>
 * specialization, false otherwise
 */
template <typename T>
struct RS_IsExternallySerializableClass {
   private:
    struct FakeSerializer {
        template <typename Value>
        void operator()(const std::string_view& key, Value& value) {}
    };

    // Test if RS_Serialize<T> exists and has the operator() method
    template <typename U>
    static auto test(
        int
    ) -> decltype(RS_Serialize<U>{}(std::declval<const U&>(), std::declval<FakeSerializer&>()), std::true_type{});

    template <typename>
    static std::false_type test(...);

   public:
    static const bool value = decltype(test<T>(0))::value;
};

/**
 * @brief Checks if a type is serializable
 *
 * A type is serializable if it satisfies any of the following conditions:
 * - It is a primitive type (int8_t, int16_t, ..., float, double, bool, std::nullptr_t)
 * - It is an enum or RS_NoneType
 * - It is a string-like type (RS_IsStringLike)
 * - It is a container (RS_IsContainerLike)
 * - It has a serialize method template (RS_IsSerializableClass)
 * - It has an RS_Serialize<T> specialization (RS_IsExternallySerializableClass)
 *
 * @tparam T The type to check
 * @return True if the type is serializable, false otherwise
 */
template <typename T>
struct RS_IsSerializable {
   public:
    // Check for primitive types with explicit packer specializations
    static constexpr bool is_primitive_serializable =
        std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> ||
        std::is_same_v<T, int64_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
        std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, float> ||
        std::is_same_v<T, double> || std::is_same_v<T, bool> || std::is_same_v<T, std::nullptr_t>;

    // Check for complex types supported by generic packer
    static constexpr bool is_complex_serializable =
        std::is_enum_v<T> || std::is_same_v<T, RS_NoneType> || RS_IsStringLike<T>::value ||
        RS_IsContainerLike<T>::value || RS_IsSerializableClass<T>::value ||
        RS_IsExternallySerializableClass<T>::value;

    static const bool value = is_primitive_serializable || is_complex_serializable;
};

/**
 * @brief Checks if a type is deserializable
 *
 * A type is deserializable if it satisfies any of the following conditions:
 * - It is a primitive type (int8_t, int16_t, ..., float, double, bool, std::nullptr_t)
 * - It is an enum or RS_NoneType
 * - It is an owning string-like type (RS_IsStringLike, excluding views and char pointers/arrays)
 * - It is an owning container (RS_IsContainerLike, excluding RS_IsContainerViewLike)
 * - It has a serialize method template (RS_IsSerializableClass)
 * - It has an RS_Serialize<T> specialization (RS_IsExternallySerializableClass)
 *
 * @note RS_IsDeserializable is always a strict subset of RS_IsSerializable. View types
 * (RS_IsContainerViewLike, RS_IsStdStringView, char pointers and arrays) are
 * serializable but not deserializable, as they are non-owning and cannot be written into.
 *
 * @tparam T The type to check
 * @return True if the type is deserializable, false otherwise
 */
template <typename T>
struct RS_IsDeserializable {
   public:
    // Check for primitive types with explicit unpacker specializations
    static constexpr bool is_primitive_deserializable =
        std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t> ||
        std::is_same_v<T, int64_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
        std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, float> ||
        std::is_same_v<T, double> || std::is_same_v<T, bool> || std::is_same_v<T, std::nullptr_t>;

    // Views are not deserializable (they are non-owning and cannot be written into)
    static constexpr bool is_non_owning_type =
        RS_IsContainerViewLike<T>::value || RS_IsStringViewLike<T>::value ||
        RS_IsCharPointer<T>::value || RS_IsCharArray<T>::value;

    // Check for complex types supported by generic unpacker (excluding views)
    static constexpr bool is_complex_deserializable =
        !is_non_owning_type &&
        (std::is_enum_v<T> || std::is_same_v<T, RS_NoneType> || RS_IsStringLike<T>::value ||
         RS_IsContainerLike<T>::value || RS_IsSerializableClass<T>::value ||
         RS_IsExternallySerializableClass<T>::value);

    static const bool value = is_primitive_deserializable || is_complex_deserializable;
};

}  // namespace rise

#endif  // RS_SERIALIZATION_TYPE_TRAITS_H
