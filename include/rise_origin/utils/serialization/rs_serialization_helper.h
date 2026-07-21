// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SERIALIZATION_HELPER_H
#define RS_SERIALIZATION_HELPER_H

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "rise_origin/utils/rs_system.h"

// Compiler byte-swap intrinsic detection
#if defined(RS_COMPILER_IS_GCC) || defined(RS_COMPILER_IS_CLANG)
#define RS_SH_HAS_BUILTIN_BSWAP
#elif defined(RS_COMPILER_IS_MSVC)
#define RS_SH_HAS_MSVC_BYTESWAP
#include <cstdlib>
#endif

namespace rise {

/*****************************************************************
 * serialization_helper
 *****************************************************************/
namespace serialization_helper {

/// @brief Cast @p value to uint8_t.
template <typename T>
constexpr uint8_t castU8(T value) {
    return static_cast<uint8_t>(value);
}

/// @brief Cast @p value to uint16_t.
template <typename T>
constexpr uint16_t castU16(T value) {
    return static_cast<uint16_t>(value);
}

/// @brief Cast @p value to uint32_t.
template <typename T>
constexpr uint32_t castU32(T value) {
    return static_cast<uint32_t>(value);
}

/// @brief Cast @p value to uint64_t.
template <typename T>
constexpr uint64_t castU64(T value) {
    return static_cast<uint64_t>(value);
}

/// @brief Cast @p value to int8_t.
template <typename T>
constexpr int8_t castI8(T value) {
    return static_cast<int8_t>(value);
}

/// @brief Cast @p value to int16_t.
template <typename T>
constexpr int16_t castI16(T value) {
    return static_cast<int16_t>(value);
}

/// @brief Cast @p value to int32_t.
template <typename T>
constexpr int32_t castI32(T value) {
    return static_cast<int32_t>(value);
}

/// @brief Cast @p value to int64_t.
template <typename T>
constexpr int64_t castI64(T value) {
    return static_cast<int64_t>(value);
}

/// @brief Cast @p value to float.
template <typename T>
constexpr float castF(T value) {
    return static_cast<float>(value);
}

/// @brief Cast @p value to double.
template <typename T>
constexpr double castD(T value) {
    return static_cast<double>(value);
}

/**
 * @brief Reinterpret the bit pattern of @p from as type @p To.
 *
 * @details Uses std::bit_cast when available (C++20), falls back to a
 * union-based approach for scalars or memcpy for other trivially-copyable types.
 */
template <typename To, typename From>
constexpr To bitcast(const From& from) noexcept {
    static_assert(sizeof(To) == sizeof(From), "bitcast requires same-sized types");
    static_assert(std::is_trivially_copyable_v<From>, "bitcast requires trivially copyable source");
    static_assert(std::is_trivially_copyable_v<To>, "bitcast requires trivially copyable target");
#if __cpp_lib_bit_cast >= 201806L
    return std::bit_cast<To>(from);
#else
    if constexpr (std::is_scalar_v<From> && std::is_scalar_v<To>) {
        union {
            From f;
            To t;
        } u;
        u.f = from;
        return u.t;
    } else {
        To to;
        std::memcpy(&to, &from, sizeof(To));
        return to;
    }
#endif
}

/// @brief Byte-swap a 16-bit value using compiler intrinsics when available.
inline uint16_t bswap16(uint16_t value) {
#if defined(RS_SH_HAS_BUILTIN_BSWAP)
    return __builtin_bswap16(value);
#elif defined(RS_SH_HAS_MSVC_BYTESWAP)
    return _byteswap_ushort(value);
#else
    return static_cast<uint16_t>((value >> 8) | (value << 8));
#endif
}

/// @brief Byte-swap a 32-bit value using compiler intrinsics when available.
inline uint32_t bswap32(uint32_t value) {
#if defined(RS_SH_HAS_BUILTIN_BSWAP)
    return __builtin_bswap32(value);
#elif defined(RS_SH_HAS_MSVC_BYTESWAP)
    return _byteswap_ulong(value);
#else
    return ((value & 0x000000FFU) << 24) | ((value & 0x0000FF00U) << 8) |
           ((value & 0x00FF0000U) >> 8) | ((value & 0xFF000000U) >> 24);
#endif
}

/// @brief Byte-swap a 64-bit value using compiler intrinsics when available.
inline uint64_t bswap64(uint64_t value) {
#if defined(RS_SH_HAS_BUILTIN_BSWAP)
    return __builtin_bswap64(value);
#elif defined(RS_SH_HAS_MSVC_BYTESWAP)
    return _byteswap_uint64(value);
#else
    return ((value & 0x00000000000000FFULL) << 56) | ((value & 0x000000000000FF00ULL) << 40) |
           ((value & 0x0000000000FF0000ULL) << 24) | ((value & 0x00000000FF000000ULL) << 8) |
           ((value & 0x000000FF00000000ULL) >> 8) | ((value & 0x0000FF0000000000ULL) >> 24) |
           ((value & 0x00FF000000000000ULL) >> 40) | ((value & 0xFF00000000000000ULL) >> 56);
#endif
}

}  // namespace serialization_helper

}  // namespace rise

#endif  // RS_SERIALIZATION_HELPER_H
