// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CONSTEXPR_H
#define RS_CONSTEXPR_H

#include "rs_def.h"

#ifndef FOR_ANY_TIME
#define FOR_ANY_TIME
#endif

namespace rise {

/**
 * @brief Checks if a value is a power of two.
 * @note Can be used at compile time.
 *
 * @param x The value to check.
 * @return true if the value is a power of two, false otherwise.
 */
template <typename T>
constexpr bool isPowerOfTwo(T x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

/**
 * @brief Returns the log base 2 of a value.
 * @note Can be used at compile time.
 *
 * @param value The value to find the log base 2 of.
 * @return The log base 2 of the given value.
 */
template <typename T>
constexpr int logTwoOf(T value) {
    if (value == 0) {
        return -1;  // Special case for log2(0)
    }

    int result = 0;
    while (value >>= 1) {
        ++result;
    }
    return result;
}

/**
 * @brief Returns the minimum power of two greater than or equal to the given value.
 * @note Can be used at compile time.
 *
 * @param size The value to find the minimum power of two for.
 * @param max The maximum value allowed.
 * @return The minimum power of two greater than or equal to the given value.
 */
constexpr RSize min2PowerAbove(RSize size, RSize max) {
    RSize min = 1;
    while (min < size && min < max) {
        min <<= 1;
    }
    return min;
}

/**
 * @brief Returns the next power of two greater than or equal to the given value.
 * @note Can be used at compile time.
 *
 * @param x The value to find the next power of two for.
 * @return The next power of two greater than or equal to the given value.
 */
constexpr uint32_t nextPowerOfTwo(uint32_t x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

/**
 * @brief Returns the next power of two greater than or equal to the given value.
 * @note Can be used at compile time.
 *
 * @param x The value to find the next power of two for.
 * @return The next power of two greater than or equal to the given value.
 */
constexpr uint64_t nextPowerOfTwo(uint64_t x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x++;
    return x;
}

/**
 * @brief Exchanges the values of two variables.
 * @note Can be used at compile time.
 *
 * @param a The first variable.
 * @param b The second variable.
 */
template <typename T>
constexpr void exchange(T& a, T& b) {
    T tmp = a;
    a = b;
    b = tmp;
}

/**
 * @brief Compares two strings for less than.
 */
constexpr bool isStringLess(const char* lhs, const char* rhs) noexcept {
    return *lhs && *rhs && *lhs == *rhs ? isStringLess(lhs + 1, rhs + 1) : *lhs < *rhs;
}

/**
 * @brief Compares two strings for equality.
 */
constexpr bool isStringEqual(const char* lhs, const char* rhs) noexcept {
    return *lhs == *rhs && (*lhs == '\0' || isStringEqual(lhs + 1, rhs + 1));
}

/**
 * @brief Applies a function to each index in a compile time range.
 */
template <RSize Start, RSize Stop, RSize Step = 1, typename Func>
constexpr void loop(Func&& func) {
    static_assert(Step > 0, "Step must be greater than 0");
    if constexpr (Start < Stop) {
        func(std::integral_constant<RIndex, Start>{});
        loop<Start + Step, Stop, Step>(std::forward<Func>(func));
    }
}

/**
 * @brief Applies a function to each index in a compile-time sequence.
 */
template <RIndex... Indices, typename Func>
FOR_ANY_TIME constexpr void forEach(Func&& func) {
    (func(Indices), ...);
}

/**
 * @brief Checks if all indices in a compile-time range satisfy a condition.
 */
template <RSize Start, RSize Stop, RSize Step = 1, typename Func>
FOR_ANY_TIME constexpr bool allOf(Func&& func) {
    static_assert(Step > 0, "Step must be greater than 0");
    if constexpr (Start < Stop) {
        return func(std::integral_constant<RIndex, Start>{}) &&
               allOf<Start + Step, Stop, Step>(std::forward<Func>(func));
    }
    return true;
}

/**
 * @brief Checks if any index in a compile-time range satisfies a condition.
 */
template <RSize Start, RSize Stop, RSize Step = 1, typename Func>
FOR_ANY_TIME constexpr bool anyOf(Func&& func) {
    static_assert(Step > 0, "Step must be greater than 0");
    if constexpr (Start < Stop) {
        return func(std::integral_constant<RIndex, Start>{}) ||
               anyOf<Start + Step, Stop, Step>(std::forward<Func>(func));
    }
    return false;
}

/**
 * @brief Switches over a runtime index to call a function corresponding to the matching
 * compile-time range.
 */
template <RSize Start, RSize Stop, RSize Step = 1, typename Func>
FOR_ANY_TIME constexpr void switchOver(RSize index, Func&& func) {
    static_assert(Step > 0, "Step must be greater than 0");
    if constexpr (Start < Stop) {
        if (index == Start)
            func(std::integral_constant<RIndex, Start>{});
        else
            switchOver<Start + Step, Stop, Step>(index, std::forward<Func>(func));
    }
}

/**
 * @brief Switches over a runtime index to call a function corresponding to the matching
 * compile-time index in a pack.
 */
template <RIndex... Indices, typename Func>
FOR_ANY_TIME constexpr void switchEach(RIndex index, Func&& func) {
    bool is_matched =
        ((Indices == index ? (func(std::integral_constant<RIndex, Indices>{}), true) : false) ||
         ...);
    (void)is_matched;
}

}  // namespace rise

#endif  // RS_CONSTEXPR_H
