// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_STRING_TYPE_TRAITS_H
#define RS_STRING_TYPE_TRAITS_H

#include <string>
#include <string_view>
#include <type_traits>

#include "rise_origin/utils/rs_def.h"

namespace rise {

/*****************************************************************
 * Raw Character Traits
 *****************************************************************/
/**
 * @brief Checks if a type is a raw character
 *
 * @tparam T The type to check
 * @return True if the type is a raw character, false otherwise
 */
template <typename T>
struct RS_IsRawChar {
    using RealCharType = std::remove_cv_t<T>;
    static constexpr bool value =
        std::is_same_v<RealCharType, char> || std::is_same_v<RealCharType, wchar_t> ||
        std::is_same_v<RealCharType, char16_t> || std::is_same_v<RealCharType, char32_t>;
};

/**
 * @brief Checks if a type is a character pointer
 *
 * @tparam T The type to check
 * @return True if the type is a character pointer, false otherwise
 */
template <typename T>
struct RS_IsCharPointer {
    using RealType = std::remove_cv_t<T>;
    static constexpr bool value =
        std::is_pointer_v<RealType> &&
        RS_IsRawChar<std::remove_cv_t<std::remove_pointer_t<RealType>>>::value;
};

/**
 * @brief Checks if a type is a character array
 *
 * @tparam T The type to check
 * @return True if the type is a character array, false otherwise
 */
template <typename T>
struct RS_IsCharArray {
    using RealType = std::remove_cv_t<T>;
    static constexpr bool value =
        std::is_array_v<RealType> &&
        RS_IsRawChar<std::remove_cv_t<std::remove_extent_t<RealType>>>::value;
};

/*****************************************************************
 * Standard String Traits
 *****************************************************************/
/**
 * @brief Checks if a type is a string specialized from std::basic_string
 *
 * @tparam T The type to check
 * @return True if the type is a string specialized from std::basic_string, false otherwise
 */
template <typename T>
struct RS_IsStdString : std::false_type {};

template <typename CharType>
struct RS_IsStdString<std::basic_string<CharType>> : std::true_type {};

/**
 * @brief Checks if a type is a string view specialized from std::basic_string_view
 *
 * @tparam T The type to check
 * @return True if the type is a string view specialized from std::basic_string_view, false
 * otherwise
 */
template <typename T>
struct RS_IsStdStringView : std::false_type {};

template <typename CharType>
struct RS_IsStdStringView<std::basic_string_view<CharType>> : std::true_type {};

/*****************************************************************
 * Abstract String Traits
 *****************************************************************/
/**
 * @brief Checks if a type is string like
 *
 * @tparam T The type to check
 * @return True if the type is string like, false otherwise
 */
template <typename T>
struct RS_IsStringLike {
    static constexpr bool value = RS_IsCharPointer<T>::value || RS_IsCharArray<T>::value ||
                                  RS_IsStdString<T>::value || RS_IsStdStringView<T>::value;
};

/**
 * @brief Checks if a type is string view like
 *
 * @tparam T The type to check
 * @return True if the type is string view like, false otherwise
 */
template <typename T>
struct RS_IsStringViewLike {
    static constexpr bool value = RS_IsStdStringView<T>::value;
};

/**
 * @brief Get a std::string_view over a string-like value.
 * @param str The string-like value.
 * @return A std::string_view over the value.
 */
template <typename StringType>
[[nodiscard]] std::string_view toStringView(const StringType& str) {
    return std::string_view(str.data(), str.size());
}

[[nodiscard]] inline std::string_view toStringView(const char* str) {
    return std::string_view(str);
}

}  // namespace rise

#endif  // RS_STRING_TYPE_TRAITS_H
