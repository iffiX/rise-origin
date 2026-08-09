// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_MORTON_H
#define RS_MORTON_H

#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace device {

/**
 * @brief Morton (Z-order) code conversion between a grid coordinate and a code.
 *
 * Interleaving the three coordinates leaves spatially close cells numerically
 * close, so sorting by code groups them together. Holds no state, so every
 * member is static.
 */
template <typename CodeType>
class RS_MortonCode {
   public:
    /*****************************************************************
     * Data
     *****************************************************************/
    /// Bits each coordinate gets: 10 for a 32 bit code, 21 for a 64 bit one.
    static constexpr RIndex CoordinateBits = sizeof(CodeType) == 4 ? 10 : 21;
    /// Largest coordinate the code can hold on one axis.
    static constexpr CodeType CoordinateMax = (CodeType(1) << CoordinateBits) - 1;

    /*****************************************************************
     * Methods
     *****************************************************************/
    /**
     * @brief Interleave three coordinates into one code.
     *
     * Coordinates above CoordinateMax saturate at the upper grid bound.
     */
    HOST_DEVICE_INLINE_QUALIFIER static CodeType toMortonCode(CodeType x, CodeType y, CodeType z) {
        return _leftShift3(x) | (_leftShift3(y) << 1) | (_leftShift3(z) << 2);
    }

    /**
     * @brief Split one code back into the three coordinates it was built from.
     */
    HOST_DEVICE_INLINE_QUALIFIER static void
    toCoordinate(CodeType code, CodeType& x_out, CodeType& y_out, CodeType& z_out) {
        x_out = _compactShift3(code);
        y_out = _compactShift3(code >> 1);
        z_out = _compactShift3(code >> 2);
    }

   private:
    /*****************************************************************
     * Methods
     *****************************************************************/
    /// @brief Spread the low bits of x so two zero bits sit between each.
    HOST_DEVICE_INLINE_QUALIFIER static CodeType _leftShift3(CodeType x) {
        if constexpr (sizeof(CodeType) == 4)
            return _leftShift3For32(x);
        else
            return _leftShift3For64(x);
    }

    /// @brief Keep every third bit of x and pack them back down.
    HOST_DEVICE_INLINE_QUALIFIER static CodeType _compactShift3(CodeType x) {
        if constexpr (sizeof(CodeType) == 4)
            return _compactShift3For32(x);
        else
            return _compactShift3For64(x);
    }

    HOST_DEVICE_INLINE_QUALIFIER static uint32_t _leftShift3For32(uint32_t x) {
        // Clamp to lower 10 bits; out-of-range coordinates saturate at the upper grid bound.
        x = RS_MIN(x, (uint32_t)0x3ff);

        /* Below code is like this, each time, you shift a group of code
         * to the left, until you shift all bit to their corresponding place.
         * x = (x | (x << 16)) & 0b00000011000000000000000011111111;
         * x = (x | (x << 8)) & 0b00000011000000001111000000001111;
         * x = (x | (x << 4)) & 0b00000011000011000011000011000011;
         * x = (x | (x << 2)) & 0b00001001001001001001001001001001;
         */

        x = (x | (x << 16)) & 0x30000ff;
        x = (x | (x << 8)) & 0x300f00f;
        x = (x | (x << 4)) & 0x30c30c3;
        x = (x | (x << 2)) & 0x9249249;

        return x;
    }

    HOST_DEVICE_INLINE_QUALIFIER static uint64_t _leftShift3For64(uint64_t x) {
        // Clamp to lower 21 bits; out-of-range coordinates saturate at the upper grid bound.
        x = RS_MIN(x, (uint64_t)0x1fffffULL);

        x = (x | (x << 32)) & 0x1F00000000FFFFULL;
        x = (x | (x << 16)) & 0x1F0000FF0000FFULL;
        x = (x | (x << 8)) & 0x100F00F00F00F00FULL;
        x = (x | (x << 4)) & 0x10C30C30C30C30C3ULL;
        x = (x | (x << 2)) & 0x1249249249249249ULL;

        return x;
    }

    HOST_DEVICE_INLINE_QUALIFIER static uint32_t _compactShift3For32(uint32_t x) {
        // The masks of _leftShift3For32 in reverse, gathering every third bit back down.
        x &= 0x9249249;
        x = (x | (x >> 2)) & 0x30c30c3;
        x = (x | (x >> 4)) & 0x300f00f;
        x = (x | (x >> 8)) & 0x30000ff;
        x = (x | (x >> 16)) & 0x3ff;

        return x;
    }

    HOST_DEVICE_INLINE_QUALIFIER static uint64_t _compactShift3For64(uint64_t x) {
        // The masks of _leftShift3For64 in reverse, gathering every third bit back down.
        x &= 0x1249249249249249ULL;
        x = (x | (x >> 2)) & 0x10C30C30C30C30C3ULL;
        x = (x | (x >> 4)) & 0x100F00F00F00F00FULL;
        x = (x | (x >> 8)) & 0x1F0000FF0000FFULL;
        x = (x | (x >> 16)) & 0x1F00000000FFFFULL;
        x = (x | (x >> 32)) & 0x1fffffULL;

        return x;
    }
};

}  // namespace device

}  // namespace rise

#endif  // RS_MORTON_H
