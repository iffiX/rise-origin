// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_COLOR_H
#define RS_COLOR_H

#include <refl.hpp>

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/math/rs_float.h"
#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

template <typename T>
class RS_Color3D;

template <typename T>
class RS_Color4D;

/**
 * @brief Template structure representing a 3-channel (RGB) color.
 */
template <typename T>
class RS_Color3D {
   public:
    static_assert(
        std::is_same_v<T, RByte> || std::is_same_v<T, float>,
        "Color channel type must be RByte or float"
    );

    T r, g, b;

    /// Default constructor
    RS_Color3D() = default;

    /// Constructor with specified individual channels
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D(T r, T g, T b) : r(r), g(g), b(b) {};

    /// Copy constructor
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D(const RS_Color3D<U> &c) {
        if constexpr (std::is_same_v<T, RByte> && std::is_same_v<U, float>) {
            r = static_cast<T>(c.r * 255.0f);
            g = static_cast<T>(c.g * 255.0f);
            b = static_cast<T>(c.b * 255.0f);
        } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, RByte>) {
            r = static_cast<T>(c.r / 255.0f);
            g = static_cast<T>(c.g / 255.0f);
            b = static_cast<T>(c.b / 255.0f);
        } else {
            r = c.r;
            g = c.g;
            b = c.b;
        }
    }

    /// Constructor from RS_Vec3D
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D(const RS_Vec3D<U> &v) : r(v.x), g(v.y), b(v.z) {
        static_assert(
            std::is_floating_point_v<U>, "Vector assigned to color must be floating point"
        );
    }

    /// Returns a color 3d of all zeros
    HOST_DEVICE_INLINE_QUALIFIER static RS_Color3D zero() { return RS_Color3D(0, 0, 0); }

    /// Cast operator for type conversion
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER explicit operator RS_Color3D<U>() const {
        if constexpr (std::is_same_v<T, RByte> && std::is_same_v<U, float>) {
            return RS_Color3D<U>(r / 255.0f, g / 255.0f, b / 255.0f);
        } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, RByte>) {
            return RS_Color3D<U>(r * 255.0f, g * 255.0f, b * 255.0f);
        } else {
            return RS_Color3D<U>(r, g, b);
        }
    }

    /// Assign operator
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D operator=(const RS_Color3D<U> &v) {
        if constexpr (std::is_same_v<T, RByte> && std::is_same_v<U, float>) {
            r = static_cast<T>(v.r * 255.0f);
            g = static_cast<T>(v.g * 255.0f);
            b = static_cast<T>(v.b * 255.0f);
        } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, RByte>) {
            r = static_cast<T>(v.r / 255.0f);
            g = static_cast<T>(v.g / 255.0f);
            b = static_cast<T>(v.b / 255.0f);
        } else {
            r = v.r;
            g = v.g;
            b = v.b;
        }
        return *this;
    }

    /// Assign operator from color 4d
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D operator=(const RS_Color4D<T> &v) {
        *this = v.rgb();
        return *this;
    }

    /// Assign operator from vector 3d
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D operator=(const RS_Vec3D<U> &v) {
        static_assert(
            std::is_floating_point_v<U>, "Vector assigned to color must be floating point"
        );
        r = static_cast<T>(v.x);
        g = static_cast<T>(v.y);
        b = static_cast<T>(v.z);
        return *this;
    }

    /// Addition operator for color 3d
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D operator+(const RS_Color3D &c) const {
        return RS_Color3D(r + c.r, g + c.g, b + c.b);
    }

    /// Subtraction operator for color 3d
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D operator-(const RS_Color3D &c) const {
        return RS_Color3D(r - c.r, g - c.g, b - c.b);
    }

    /// Unary negation operator for color 3d
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D operator-() const { return RS_Color3D(-r, -g, -b); }

    /// Multiplication operator for constant for color 3d
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D operator*(T f) const {
        return RS_Color3D(f * r, f * g, f * b);
    }

    /// Multiplication operator for constant as left hand side for color 3d
    HOST_DEVICE_INLINE_QUALIFIER friend RS_Color3D operator*(T f, const RS_Color3D &v) {
        return v * f;
    }

    /// Multiplication operator (dot multiplication) for color 3d
    HOST_DEVICE_INLINE_QUALIFIER T operator*(const RS_Color3D &c) const {
        return r * c.r + g * c.g + b * c.b;
    }

    /// Division operator for constant for color 3d
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D operator/(T f) const {
        if constexpr (std::is_integral_v<T>) {
            return {r / f, g / f, b / f};
        } else {
            T f_inv = T(1) / f;
            return {r * f_inv, g * f_inv, b * f_inv};
        }
    }

    /// Addition and set operator for color 3d
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D &operator+=(const RS_Color3D &c) {
        r += c.r;
        g += c.g;
        b += c.b;
        return *this;
    }

    /// Subtraction and set operator for color 3d
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D &operator-=(const RS_Color3D &c) {
        r -= c.r;
        g -= c.g;
        b -= c.b;
        return *this;
    }

    /// Multiplication and set operator for constant for color 3d
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D &operator*=(T f) {
        r *= f;
        g *= f;
        b *= f;
        return *this;
    }

    /// Division and set operator for constant for color 3d
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D &operator/=(T f) {
        if constexpr (std::is_integral_v<T>) {
            r /= f;
            g /= f;
            b /= f;
        } else {
            T f_inv = T(1) / f;
            r *= f_inv;
            g *= f_inv;
            b *= f_inv;
        }
        return *this;
    }

    /// Is equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_Color3D &c) const {
        return r == c.r && g == c.g && b == c.b;
    }

    /// Is not equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(const RS_Color3D &c) const {
        return !(*this == c);
    }

    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D<T> clip() const {
        if constexpr (std::is_same_v<T, RByte>) {
            return RS_Color3D<T>(RS_BOUND(r, 0, 255), RS_BOUND(g, 0, 255), RS_BOUND(b, 0, 255));
        } else {
            return RS_Color3D<T>(
                RS_BOUND(r, 0.0f, 1.0f), RS_BOUND(g, 0.0f, 1.0f), RS_BOUND(b, 0.0f, 1.0f)
            );
        }
    }

    /// Index operator
    HOST_DEVICE_INLINE_QUALIFIER T &operator[](int index) {
        return *(reinterpret_cast<T *>(this) + index);
    }

    /// Const index operator
    HOST_DEVICE_INLINE_QUALIFIER const T &operator[](int index) const {
        return *(reinterpret_cast<const T *>(this) + index);
    }

    /// CIE 1931 Y luminance of the RGB channels under the sRGB / Rec. 709
    /// primaries
    HOST_DEVICE_INLINE_QUALIFIER T luminance() const {
        if constexpr (std::is_same_v<T, RByte>) {
            const float y = 0.212671f * static_cast<float>(r) + 0.715160f * static_cast<float>(g) +
                            0.072169f * static_cast<float>(b);
            return static_cast<T>(y + 0.5f);
        } else {
            return T(0.212671) * r + T(0.715160) * g + T(0.072169) * b;
        }
    }

    /// from sRGB to linear RGB (gamma encoding)
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D<T> srgbToLinear() const {
        using RS_FI = RS_FloatIntrinsics<T>;
        if constexpr (std::is_same_v<T, RByte>) {
            return RS_Color3D<T>(
                static_cast<T>(RS_FI::rs_pow(static_cast<float>(r) / 255.0f, 2.2f) * 255.0f),
                static_cast<T>(RS_FI::rs_pow(static_cast<float>(g) / 255.0f, 2.2f) * 255.0f),
                static_cast<T>(RS_FI::rs_pow(static_cast<float>(b) / 255.0f, 2.2f) * 255.0f)
            );
        } else {
            return RS_Color3D<T>(
                RS_FI::rs_pow(r, 2.2f), RS_FI::rs_pow(g, 2.2f), RS_FI::rs_pow(b, 2.2f)
            );
        }
    }

    /// from linear RGB to sRGB (gamma decoding)
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D<T> linearToSrgb() const {
        using RS_FI = RS_FloatIntrinsics<T>;
        if constexpr (std::is_same_v<T, RByte>) {
            return RS_Color3D<T>(
                static_cast<T>(RS_FI::rs_pow(static_cast<float>(r) / 255.0f, 1.0f / 2.2f) * 255.0f),
                static_cast<T>(RS_FI::rs_pow(static_cast<float>(g) / 255.0f, 1.0f / 2.2f) * 255.0f),
                static_cast<T>(RS_FI::rs_pow(static_cast<float>(b) / 255.0f, 1.0f / 2.2f) * 255.0f)
            );
        } else {
            return RS_Color3D<T>(
                RS_FI::rs_pow(r, 1.0f / 2.2f),
                RS_FI::rs_pow(g, 1.0f / 2.2f),
                RS_FI::rs_pow(b, 1.0f / 2.2f)
            );
        }
    }

    /// Converts to vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toVec3D() const { return {r, g, b}; }

    /// Converts to 4-channel color
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D<T> toColor4D() const {
        if constexpr (std::is_same_v<T, RByte>)
            return {r, g, b, 255};
        else
            return {r, g, b, 1.0f};
    }

    /// Generic serialization
    template <class Serializer>
    void serialize(Serializer &s) {
        s("r", r);
        s("g", g);
        s("b", b);
    }
};

/**
 * @brief Template structure representing a 4-channel (RGBA) color.
 */
template <typename T>
class RS_Color4D {
   public:
    static_assert(
        std::is_same_v<T, RByte> || std::is_same_v<T, float>,
        "Color channel type must be RByte or float"
    );

    T r, g, b, a;

    /// Default constructor
    RS_Color4D() = default;

    /// Constructor with specified individual channels
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D(T r, T g, T b, T a) : r(r), g(g), b(b), a(a) {}

    /// Copy constructor
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D(const RS_Color4D<U> &c) {
        if constexpr (std::is_same_v<T, RByte> && std::is_same_v<U, float>) {
            r = static_cast<T>(c.r * 255.0f);
            g = static_cast<T>(c.g * 255.0f);
            b = static_cast<T>(c.b * 255.0f);
            a = static_cast<T>(c.a * 255.0f);
        } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, RByte>) {
            r = static_cast<T>(c.r / 255.0f);
            g = static_cast<T>(c.g / 255.0f);
            b = static_cast<T>(c.b / 255.0f);
            a = static_cast<T>(c.a / 255.0f);
        } else {
            r = c.r;
            g = c.g;
            b = c.b;
            a = c.a;
        }
    }

    /// Constructor from RS_Vec4D
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D(const RS_Vec4D<U> &v) : r(v.x), g(v.y), b(v.z), a(v.w) {
        static_assert(
            std::is_floating_point_v<U>, "Vector assigned to color must be floating point"
        );
    }

    /// Returns a color 4D of all zeros
    HOST_DEVICE_INLINE_QUALIFIER static RS_Color4D zero() { return RS_Color4D(0, 0, 0, 0); }

    /// Cast operator for type conversion
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER explicit operator RS_Color4D<U>() const {
        if constexpr (std::is_same_v<T, RByte> && std::is_same_v<U, float>) {
            return RS_Color4D<U>(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
        } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, RByte>) {
            return RS_Color4D<U>(r * 255.0f, g * 255.0f, b * 255.0f, a * 255.0f);
        } else {
            return RS_Color4D<U>(r, g, b, a);
        }
    }

    /// Assign operator
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D operator=(const RS_Color4D<U> &v) {
        if constexpr (std::is_same_v<T, RByte> && std::is_same_v<U, float>) {
            r = static_cast<T>(v.r * 255.0f);
            g = static_cast<T>(v.g * 255.0f);
            b = static_cast<T>(v.b * 255.0f);
            a = static_cast<T>(v.a * 255.0f);
        } else if constexpr (std::is_same_v<T, float> && std::is_same_v<U, RByte>) {
            r = static_cast<T>(v.r / 255.0f);
            g = static_cast<T>(v.g / 255.0f);
            b = static_cast<T>(v.b / 255.0f);
            a = static_cast<T>(v.a / 255.0f);
        } else {
            r = v.r;
            g = v.g;
            b = v.b;
            a = v.a;
        }
        return *this;
    }

    /// Assign operator from color 3D
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D operator=(const RS_Color3D<U> &c) {
        *this = c.rgba();
        return *this;
    }

    /// Assign operator from vector 4d
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D operator=(const RS_Vec4D<U> &v) {
        static_assert(
            std::is_floating_point_v<U>, "Vector assigned to color must be floating point"
        );
        r = static_cast<T>(v.x);
        g = static_cast<T>(v.y);
        b = static_cast<T>(v.z);
        a = static_cast<T>(v.w);
        return *this;
    }

    /// Addition operator for color 4D
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D operator+(const RS_Color4D &c) const {
        return RS_Color4D(r + c.r, g + c.g, b + c.b, a + c.a);
    }

    /// Subtraction operator for color 4D
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D operator-(const RS_Color4D &c) const {
        return RS_Color4D(r - c.r, g - c.g, b - c.b, a - c.a);
    }

    /// Unary negation operator for color 4D
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D operator-() const { return RS_Color4D(-r, -g, -b, -a); }

    /// Multiplication operator for constant for color 4D
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D operator*(T f) const {
        return RS_Color4D(f * r, f * g, f * b, f * a);
    }

    /// Multiplication operator for constant as left hand side for color 4D
    HOST_DEVICE_INLINE_QUALIFIER friend RS_Color4D operator*(T f, const RS_Color4D &v) {
        return v * f;
    }

    /// Multiplication operator (dot multiplication) for color 4D
    HOST_DEVICE_INLINE_QUALIFIER T operator*(const RS_Color4D &c) const {
        return r * c.r + g * c.g + b * c.b + a * c.a;
    }

    /// Division operator for constant for color 4D
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D operator/(T f) const {
        if constexpr (std::is_integral_v<T>) {
            return {r / f, g / f, b / f, a / f};
        } else {
            T f_inv = T(1) / f;
            return {r * f_inv, g * f_inv, b * f_inv, a * f_inv};
        }
    }

    /// Addition and set operator for color 4D
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D &operator+=(const RS_Color4D &c) {
        r += c.r;
        g += c.g;
        b += c.b;
        a += c.a;
        return *this;
    }

    /// Subtraction and set operator for color 4D
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D &operator-=(const RS_Color4D &c) {
        r -= c.r;
        g -= c.g;
        b -= c.b;
        a -= c.a;
        return *this;
    }

    /// Multiplication and set operator for constant for color 4D
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D &operator*=(T f) {
        r *= f;
        g *= f;
        b *= f;
        a *= f;
        return *this;
    }

    /// Division and set operator for constant for color 4D
    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D &operator/=(T f) {
        if constexpr (std::is_integral_v<T>) {
            r /= f;
            g /= f;
            b /= f;
            a /= f;
        } else {
            T f_inv = T(1) / f;
            r *= f_inv;
            g *= f_inv;
            b *= f_inv;
            a *= f_inv;
        }
        return *this;
    }

    /// Is equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_Color4D &c) const {
        return r == c.r && g == c.g && b == c.b && a == c.a;
    }

    /// Is not equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(const RS_Color4D &c) const {
        return !(*this == c);
    }

    HOST_DEVICE_INLINE_QUALIFIER RS_Color4D<T> clip() const {
        if constexpr (std::is_same_v<T, RByte>) {
            return RS_Color4D<T>(
                RS_BOUND(r, 0, 255), RS_BOUND(g, 0, 255), RS_BOUND(b, 0, 255), RS_BOUND(a, 0, 255)
            );
        } else {
            return RS_Color4D<T>(
                RS_BOUND(r, 0.0f, 1.0f),
                RS_BOUND(g, 0.0f, 1.0f),
                RS_BOUND(b, 0.0f, 1.0f),
                RS_BOUND(a, 0.0f, 1.0f)
            );
        }
    }

    /// Index operator
    HOST_DEVICE_INLINE_QUALIFIER T &operator[](int index) {
        return *(reinterpret_cast<T *>(this) + index);
    }

    /// Const index operator
    HOST_DEVICE_INLINE_QUALIFIER const T &operator[](int index) const {
        return *(reinterpret_cast<const T *>(this) + index);
    }

    /// Converts to vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D<T> toVec4D() const { return {r, g, b, a}; }

    /// Returns the RGB components as a 3D color
    HOST_DEVICE_INLINE_QUALIFIER RS_Color3D<T> toColor3D() const { return {r, g, b}; }

    /// CIE 1931 Y luminance of the RGB channels under the sRGB / Rec. 709
    /// primaries; alpha is ignored. See ``RS_Color3D::luminance``.
    HOST_DEVICE_INLINE_QUALIFIER T luminance() const {
        if constexpr (std::is_same_v<T, RByte>) {
            const float y = 0.212671f * static_cast<float>(r) + 0.715160f * static_cast<float>(g) +
                            0.072169f * static_cast<float>(b);
            return static_cast<T>(y + 0.5f);
        } else {
            return T(0.212671) * r + T(0.715160) * g + T(0.072169) * b;
        }
    }

    /// Generic serialization
    template <class Serializer>
    void serialize(Serializer &s) {
        s("r", r);
        s("g", g);
        s("b", b);
        s("a", a);
    }
};

using RColor3b = RS_Color3D<RByte>;
using RColor3f = RS_Color3D<float>;
using RColor4b = RS_Color4D<RByte>;
using RColor4f = RS_Color4D<float>;

}  // namespace rise

REFL_AUTO(type(rise::RColor3b), field(r), field(g), field(b))
REFL_AUTO(type(rise::RColor3f), field(r), field(g), field(b))
REFL_AUTO(type(rise::RColor4b), field(r), field(g), field(b), field(a))
REFL_AUTO(type(rise::RColor4f), field(r), field(g), field(b), field(a))
#endif  // RS_COLOR_H
