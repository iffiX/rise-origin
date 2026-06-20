// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_NUM_TRAITS_H
#define RS_NUM_TRAITS_H

#include <limits>
#include <type_traits>

#include "rise_origin/utils/rs_def.h"

namespace rise {

/**
 * @brief Numerical traits template providing type-specific constants and meta-information
 * @tparam T Scalar type for which to provide numerical traits
 */
template <typename T>
struct RS_NumTraits {
    /**
     * @brief Returns true if the type is an integer type
     * @return True if the type is an integer type
     */
    HOST_DEVICE_INLINE_QUALIFIER static constexpr bool isInteger() { return std::is_integral_v<T>; }

    /**
     * @brief Returns true if the type is a floating point type
     * @return True if the type is a floating point type
     */
    HOST_DEVICE_INLINE_QUALIFIER static constexpr bool isFloatingPoint() {
        return std::is_floating_point_v<T>;
    }

    /**
     * @brief Returns machine epsilon for the type
     * @return Difference between 1.0 and next representable value for floats,
     *         0 for integer types
     */
    HOST_DEVICE_INLINE_QUALIFIER static constexpr T epsilon() {
        return std::numeric_limits<T>::epsilon();
    }

    /**
     * @brief Returns number of decimal digits guaranteed to be preserved
     * @return Number of base-10 digits representable without precision loss
     */
    HOST_DEVICE_INLINE_QUALIFIER static constexpr int digits10() {
        return std::numeric_limits<T>::digits10;
    }

    /**
     * @brief Returns number of radix digits in mantissa
     * @return Number of significant digits in base-radix representation
     */
    HOST_DEVICE_INLINE_QUALIFIER static constexpr int digits() {
        return std::numeric_limits<T>::digits;
    }

    /**
     * @brief Returns minimum negative exponent for floating types
     * @return Minimum exponent value (base-radix) for normalized numbers
     */
    HOST_DEVICE_INLINE_QUALIFIER static constexpr int minExponent() {
        return std::numeric_limits<T>::min_exponent;
    }

    /**
     * @brief Returns maximum positive exponent for floating types
     * @return Maximum exponent value (base-radix) for representable numbers
     */
    HOST_DEVICE_INLINE_QUALIFIER static constexpr int maxExponent() {
        return std::numeric_limits<T>::max_exponent;
    }

    /**
     * @brief Returns maximum finite representable value
     * @return Largest finite value of type T
     */
    HOST_DEVICE_INLINE_QUALIFIER static constexpr T highest() {
        return std::numeric_limits<T>::max();
    }

    /**
     * @brief Returns minimum finite representable value
     * @return Smallest (most negative) finite value of type T
     */
    HOST_DEVICE_INLINE_QUALIFIER static constexpr T lowest() {
        return std::numeric_limits<T>::lowest();
    }

    /**
     * @brief Returns positive infinity representation
     * @return Infinity value for floating types, max for integers
     */
    HOST_DEVICE_INLINE_QUALIFIER static constexpr T infinity() {
        return std::numeric_limits<T>::infinity();
    }

    /**
     * @brief Returns quiet NaN representation
     * @return Quiet NaN for floating types, zero for integers
     */
    HOST_DEVICE_INLINE_QUALIFIER static constexpr T nan() {
        return std::numeric_limits<T>::quiet_NaN();
    }
};

}  // namespace rise

#endif  // RS_NUM_TRAITS_H
