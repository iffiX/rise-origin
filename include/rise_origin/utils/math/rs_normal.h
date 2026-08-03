// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_NORMAL_H
#define RS_NORMAL_H

#include <cstdint>
#include <refl.hpp>

#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

/**
 * @brief Unit vectors stored in memory, octahedral encoded.
 *
 * The direction is projected onto a regular octahedron, its faces are unfolded
 * into the unit square, and the two square coordinates are quantized to 16 bits
 * each. Worst case angular error is about 0.004 degrees.
 */
class RS_NormalOctMem {
   public:
    /// @brief Internal storage.
    uint32_t data;

    /// @brief Default constructor.
    RS_NormalOctMem() = default;

    HOST_DEVICE_INLINE_QUALIFIER static RS_NormalOctMem fromData(uint32_t data_0) {
        RS_NormalOctMem normal;
        normal.data = data_0;
        return normal;
    }

    HOST_DEVICE_INLINE_QUALIFIER static RS_NormalOctMem zero() {
        RS_NormalOctMem normal;
        normal.data = 0;
        return normal;
    }

    /// @brief Constructor from RS_Vec3D<T>.
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_NormalOctMem(const RS_Vec3D<T> &v) {
        set(v.x, v.y, v.z);
    }

    /// @brief Assignment operator from RS_Vec3D<T>.
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER RS_NormalOctMem &operator=(const RS_Vec3D<T> &v) {
        set(v.x, v.y, v.z);
        return *this;
    }

    /// @brief Convert to RS_Vec3D<T>.
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toVec3D() const {
        RS_Vec3D<T> v;
        get(v.x, v.y, v.z);
        return v;
    }

    /// @brief Convert to RS_Vec3D<T> without normalization.
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toVec3DUnnormalized() const {
        RS_Vec3D<T> v;
        getUnnormalized(v.x, v.y, v.z);
        return v;
    }

    /// @brief Helper function to set data from x, y, z.
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER void set(const T &x, const T &y, const T &z) {
        float fx = static_cast<float>(x);
        float fy = static_cast<float>(y);
        float fz = static_cast<float>(z);
        float sum = fabsf(fx) + fabsf(fy) + fabsf(fz);
        if (sum <= 0.0f) {
            data = 0;
            return;
        }
        float px = fx / sum;
        float py = fy / sum;
        if (fz < 0.0f) {
            float folded_x = (1.0f - fabsf(py)) * (px >= 0.0f ? 1.0f : -1.0f);
            float folded_y = (1.0f - fabsf(px)) * (py >= 0.0f ? 1.0f : -1.0f);
            px = folded_x;
            py = folded_y;
        }
        data = _quantize(px) | (static_cast<uint32_t>(_quantize(py)) << 16);
    }

    /// @brief Helper function to get data into x, y, z.
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER void get(T &x, T &y, T &z) const {
        float px, py, pz;
        getUnnormalized(px, py, pz);
        float scale = 1.0f / sqrtf(px * px + py * py + pz * pz);
        x = static_cast<T>(px * scale);
        y = static_cast<T>(py * scale);
        z = static_cast<T>(pz * scale);
    }

    /// @brief Helper function to get raw data into x, y, z without normalization.
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER void getUnnormalized(T &x, T &y, T &z) const {
        float px = _dequantize(static_cast<unsigned short>(data & 0xFFFFu));
        float py = _dequantize(static_cast<unsigned short>(data >> 16));
        float pz = 1.0f - fabsf(px) - fabsf(py);
        if (pz < 0.0f) {
            float folded_x = (1.0f - fabsf(py)) * (px >= 0.0f ? 1.0f : -1.0f);
            float folded_y = (1.0f - fabsf(px)) * (py >= 0.0f ? 1.0f : -1.0f);
            px = folded_x;
            py = folded_y;
        }
        x = static_cast<T>(px);
        y = static_cast<T>(py);
        z = static_cast<T>(pz);
    }

   private:
    /// @brief Maps a square coordinate in [-1, 1] onto a signed 16 bit code.
    HOST_DEVICE_INLINE_QUALIFIER static unsigned short _quantize(float value) {
        float clamped = value < -1.0f ? -1.0f : (value > 1.0f ? 1.0f : value);
        float scaled = clamped * 32767.0f;
        int rounded = static_cast<int>(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
        return static_cast<unsigned short>(static_cast<short>(rounded));
    }

    /// @brief Maps a signed 16 bit code back onto a square coordinate in [-1, 1].
    HOST_DEVICE_INLINE_QUALIFIER static float _dequantize(unsigned short code) {
        return static_cast<float>(static_cast<short>(code)) / 32767.0f;
    }
};

}  // namespace rise

REFL_AUTO(type(rise::RS_NormalOctMem), field(data))

#endif  // RS_NORMAL_H
