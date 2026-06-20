// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_COORD_H
#define RS_COORD_H

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/meta/rs_num_traits.h"
#include "rise_origin/utils/math/rs_vec.h"

namespace rise {
/**
 * @brief Template of generic 3D coordinates. The template parameter must be
 * an integer type.
 *
 * @Note The values are not initialized to make it compatible with plain old
 * data structure (POD).
 *
 * @note Partially referenced NanoVDB https://developer.nvidia.com/nanovdb
 */
template <typename T>
class RS_Coord3D {
   public:
    T x;  //!< The current X coordinate.
    T y;  //!< The current Y coordinate.
    T z;  //!< The current Z coordinate.

    static_assert(std::is_integral_v<T>, "T must be an integer type.");

    // Constructors
    /// Constructor which initializes x, y, z to 0
    RS_Coord3D() = default;

    /// Constructor with specified individual values
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D(T x, T y, T z) : x(x), y(y), z(z) {}

    /// Copy constructor
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D(const RS_Coord3D &c) : x(c.x), y(c.y), z(c.z) {}

    /// Constructor from RS_Vec3D
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D(const RS_Vec3D<U> &v) {
        using FI = RS_FloatIntrinsics<U>;
        x = static_cast<T>(FI::rs_round_down(v.x));
        y = static_cast<T>(FI::rs_round_down(v.y));
        z = static_cast<T>(FI::rs_round_down(v.z));
    }

    /// @brief Returns a coordinate with all zeros
    HOST_DEVICE_INLINE_QUALIFIER static RS_Coord3D zero() { return RS_Coord3D(0, 0, 0); }

    /// @brief Returns a coordinate with all ones
    HOST_DEVICE_INLINE_QUALIFIER static RS_Coord3D one() { return RS_Coord3D(1, 1, 1); }

    /// @brief Returns a coordinate filled with the specified value
    HOST_DEVICE_INLINE_QUALIFIER static RS_Coord3D full(T value) {
        return RS_Coord3D(value, value, value);
    }

    /// @brief Returns max coordinate
    HOST_DEVICE_INLINE_QUALIFIER static RS_Coord3D upperBound() {
        return RS_Coord3D(
            RS_NumTraits<T>::highest(), RS_NumTraits<T>::highest(), RS_NumTraits<T>::highest()
        );
    }

    /// @brief Returns min coordinate
    HOST_DEVICE_INLINE_QUALIFIER static RS_Coord3D lowerBound() {
        return RS_Coord3D(
            RS_NumTraits<T>::lowest(), RS_NumTraits<T>::lowest(), RS_NumTraits<T>::lowest()
        );
    }

    /// @brief Print coordinate content
    HOST_DEVICE_INLINE_QUALIFIER void print(const char *title = "Coord3D") const {
        printf("%s x:%d, y:%d, z:%d\n", title, x, y, z);
    }

    /// @brief Cast operator for type conversion
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER explicit operator RS_Coord3D<U>() const {
        return RS_Coord3D<U>(x, y, z);
    }

    /// @brief Assign operator
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator=(const RS_Coord3D<U> &c) {
        x = c.x;
        y = c.y;
        z = c.z;
        return *this;
    }

    /// @brief Assign operator from vector 3d
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator=(const RS_Vec3D<U> &v) {
        x = static_cast<T>(v.x);
        y = static_cast<T>(v.y);
        z = static_cast<T>(v.z);
    }

    /// @brief Addition operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator+(const RS_Coord3D &c) const {
        return RS_Coord3D(x + c.x, y + c.y, z + c.z);
    }

    /// @brief Subtraction operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator-(const RS_Coord3D &c) const {
        return RS_Coord3D(x - c.x, y - c.y, z - c.z);
    }

    /// @brief Multiplication operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator*(const RS_Coord3D &c) const {
        return RS_Coord3D(x * c.x, y * c.y, z * c.z);
    }

    /// @brief Division operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator/(const RS_Coord3D &c) const {
        return RS_Coord3D(x / c.x, y / c.y, z / c.z);
    }

    /// @brief Modulo operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator%(const RS_Coord3D &c) const {
        return RS_Coord3D(x % c.x, y % c.y, z % c.z);
    }

    /// @brief Bitwise and operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator&(const RS_Coord3D &c) const {
        return RS_Coord3D(x & c.x, y & c.y, z & c.z);
    }

    /// @brief Bitwise or operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator|(const RS_Coord3D &c) const {
        return RS_Coord3D(x | c.x, y | c.y, z | c.z);
    }

    /// @brief Bitwise xor operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator^(const RS_Coord3D &c) const {
        return RS_Coord3D(x ^ c.x, y ^ c.y, z ^ c.z);
    }

    /// @brief Bitwise right shift
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator>>(const RS_Coord3D &c) const {
        return RS_Coord3D(x >> c.x, y >> c.y, z >> c.z);
    }

    /// @brief Bitwise left shift
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator<<(const RS_Coord3D &c) const {
        return RS_Coord3D(x << c.x, y << c.y, z << c.z);
    }

    /// @brief Unary negation
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator-() const { return RS_Coord3D(-x, -y, -z); }

    /// @brief Unary bitwise complement
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator~() const { return RS_Coord3D(~x, ~y, ~z); }

    /// @brief Addition
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator+(T n) const {
        return RS_Coord3D(x + n, y + n, z + n);
    }

    /// @brief Subtraction
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator-(T n) const {
        return RS_Coord3D(x - n, y - n, z - n);
    }

    /// @brief Multiplication
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator*(T n) const {
        return RS_Coord3D(x * n, y * n, z * n);
    }

    /// @brief Integer division
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator/(T n) const {
        return RS_Coord3D(x / n, y / n, z / n);
    }

    /// @brief Integer modulus
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator%(T n) const {
        return RS_Coord3D(x % n, y % n, z % n);
    }

    /// @brief Bitwise and operator
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator&(T n) const {
        return RS_Coord3D(x & n, y & n, z & n);
    }

    /// @brief Bitwise or operator
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator|(T n) const {
        return RS_Coord3D(x | n, y | n, z | n);
    }

    /// @brief Bitwise right shift operator
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator>>(T n) const {
        return RS_Coord3D(x >> n, y >> n, z >> n);
    }

    /// @brief Bitwise left shift operator
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D operator<<(T n) const {
        return RS_Coord3D(x << n, y << n, z << n);
    }

    /// @brief Addition and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator+=(const RS_Coord3D &c) {
        x += c.x;
        y += c.y;
        z += c.z;
        return *this;
    }

    /// @brief Subtraction and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator-=(const RS_Coord3D &c) {
        x -= c.x;
        y -= c.y;
        z -= c.z;
        return *this;
    }

    /// @brief Multiplication and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator*=(const RS_Coord3D &c) {
        x *= c.x;
        y *= c.y;
        z *= c.z;
        return *this;
    }

    /// @brief Division and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator/=(const RS_Coord3D &c) {
        x /= c.x;
        y /= c.y;
        z /= c.z;
        return *this;
    }

    /// @brief Modulus and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator%=(const RS_Coord3D &c) {
        x %= c.x;
        y %= c.y;
        z %= c.z;
        return *this;
    }

    /// @brief Bitwise and and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator&=(const RS_Coord3D &c) {
        x &= c.x;
        y &= c.y;
        z &= c.z;
        return *this;
    }

    /// @brief Bitwise or and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator|=(const RS_Coord3D &c) {
        x |= c.x;
        y |= c.y;
        z |= c.z;
        return *this;
    }

    /// @brief Bitwise xor and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator^=(const RS_Coord3D &c) {
        x ^= c.x;
        y ^= c.y;
        z ^= c.z;
        return *this;
    }

    /// @brief Bitwise right shift and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator>>=(const RS_Coord3D &c) {
        x >>= c.x;
        y >>= c.y;
        z >>= c.z;
        return *this;
    }

    /// @brief Bitwise left shift and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator<<=(const RS_Coord3D &c) {
        x <<= c.x;
        y <<= c.y;
        z <<= c.z;
        return *this;
    }

    /// @brief Addition and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator+=(T n) {
        x += n;
        y += n;
        z += n;
        return *this;
    }

    /// @brief Subtraction and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator-=(T n) {
        x -= n;
        y -= n;
        z -= n;
        return *this;
    }

    /// @brief Multiplication and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator*=(T n) {
        x *= n;
        y *= n;
        z *= n;
        return *this;
    }

    /// @brief Division and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator/=(T n) {
        x /= n;
        y /= n;
        z /= n;
        return *this;
    }

    /// @brief Modulus and set operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator%=(T n) {
        x %= n;
        y %= n;
        z %= n;
        return *this;
    }

    /// @brief Bitwise and operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator&=(T n) {
        x &= n;
        y &= n;
        z &= n;
        return *this;
    }

    /// @brief Bitwise or operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator|=(T n) {
        x |= n;
        y |= n;
        z |= n;
        return *this;
    }

    /// @brief Bitwise xor operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator^=(T n) {
        x ^= n;
        y ^= n;
        z ^= n;
        return *this;
    }

    /// @brief Bitwise left shift operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator<<=(T n) {
        x <<= n;
        y <<= n;
        z <<= n;
        return *this;
    }

    /// @brief Bitwise right shift operator for coordinate
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D &operator>>=(T n) {
        x >>= n;
        y >>= n;
        z >>= n;
        return *this;
    }

    /// @brief Is equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_Coord3D &c) const {
        return x == c.x && y == c.y && z == c.z;
    }

    /// @brief Is not equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(const RS_Coord3D &c) const {
        return x != c.x || y != c.y || z != c.z;
    }

    /// @brief Return true if this coord is lexicographically less than the given coord.
    HOST_DEVICE_INLINE_QUALIFIER bool operator<(const RS_Coord3D &c) const {
        return x < c.x   ? true
               : x > c.x ? false
               : y < c.y ? true
               : y > c.y ? false
                         : z < c.z;
    }

    /// @brief Return true if this Coord is lexicographically less or equal to the given Coord.
    HOST_DEVICE_INLINE_QUALIFIER bool operator<=(const RS_Coord3D &c) const {
        return x < c.x   ? true
               : x > c.x ? false
               : y < c.y ? true
               : y > c.y ? false
                         : z <= c.z;
    }

    /// @brief Return true if this Coord is lexicographically greater than the given Coord.
    HOST_DEVICE_INLINE_QUALIFIER bool operator>(const RS_Coord3D &c) const {
        return x > c.x   ? true
               : x < c.x ? false
               : y > c.y ? true
               : y < c.y ? false
                         : z > c.z;
    }

    /// @brief Return true if this Coord is lexicographically greater or equal to the given Coord.
    HOST_DEVICE_INLINE_QUALIFIER bool operator>=(const RS_Coord3D &c) const {
        return x > c.x   ? true
               : x < c.x ? false
               : y > c.y ? true
               : y < c.y ? false
                         : z >= c.z;
    }

    /// @brief Index operator
    HOST_DEVICE_INLINE_QUALIFIER T &operator[](int index) {
        return *(reinterpret_cast<T *>(this) + index);
    }

    /// @brief Const index operator
    HOST_DEVICE_INLINE_QUALIFIER const T &operator[](int index) const {
        return *(reinterpret_cast<const T *>(this) + index);
    }

    /// @brief Clip the value of the coordinate between low and high
    HOST_DEVICE_INLINE_QUALIFIER const RS_Coord3D &clip(T low, T high) {
        x = RS_MIN(RS_MAX(x, low), high);
        y = RS_MIN(RS_MAX(y, low), high);
        z = RS_MIN(RS_MAX(z, low), high);
        return *this;
    }

    /// @brief Returns the product of x, y, and z
    HOST_DEVICE_INLINE_QUALIFIER T prod() const { return x * y * z; }

    /// @brief Align the value to multiples of n
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D align(T n) {
        return RS_Coord3D(RS_ALIGN(x, n), RS_ALIGN(y, n), RS_ALIGN(z, n));
    }

    /// @brief Align the value to multiples of c
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D align(const RS_Coord3D &c) {
        return RS_Coord3D(RS_ALIGN(x, c.x), RS_ALIGN(y, c.y), RS_ALIGN(z, c.z));
    }

    /// @brief Ceil division
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D ceilDiv(T n) {
        return RS_Coord3D(RS_CEIL_DIV(x, n), RS_CEIL_DIV(y, n), RS_CEIL_DIV(z, n));
    }

    /// @brief Ceil division
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D ceilDiv(const RS_Coord3D &c) {
        return RS_Coord3D(RS_CEIL_DIV(x, c.x), RS_CEIL_DIV(y, c.y), RS_CEIL_DIV(z, c.z));
    }

    /// @brief Floor division
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D floorDiv(T n) {
        return RS_Coord3D(RS_FLOOR_DIV(x, n), RS_FLOOR_DIV(y, n), RS_FLOOR_DIV(z, n));
    }

    /// @brief Floor division
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D floorDiv(const RS_Coord3D &c) {
        return RS_Coord3D(RS_FLOOR_DIV(x, c.x), RS_FLOOR_DIV(y, c.y), RS_FLOOR_DIV(z, c.z));
    }

    /**
     * @brief Returns a coordinate populated by the minimum x, y, and z value of this coordinate and
     * the specified coordinate "v"
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D min(const RS_Coord3D &c) const {
        return RS_Coord3D(x < c.x ? x : c.x, y < c.y ? y : c.y, z < c.z ? z : c.z);
    }

    /**
     * @brief Returns a coordinate populated by the maximum x, y, and z value of this coordinate and
     * the specified coordinate "v"
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Coord3D max(const RS_Coord3D &c) const {
        return RS_Coord3D(x > c.x ? x : c.x, y > c.y ? y : c.y, z > c.z ? z : c.z);
    }

    /// @brief Returns the smallest of x, y, or z of this coordinate
    HOST_DEVICE_INLINE_QUALIFIER T min() const {
        T min_xy = (x < y ? x : y);
        return (z < min_xy ? z : min_xy);
    }

    /// @brief Returns the index of the smallest x, y, or z of this coordinate
    HOST_DEVICE_INLINE_QUALIFIER int argmin() const { return x < y && x < z ? 0 : y < z ? 1 : 2; }

    /// @brief Returns the largest of x, y, or z of this coordinate
    HOST_DEVICE_INLINE_QUALIFIER T max() const {
        T max_xy = (x > y ? x : y);
        return (z > max_xy ? z : max_xy);
    }

    /// @brief Returns the index of the largest x, y, or z of this coordinate
    HOST_DEVICE_INLINE_QUALIFIER int argmax() const { return x > y && x > z ? 0 : y > z ? 1 : 2; }

    /// Converts to vector 3d
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toVec3D() const { return RS_Vec3D<T>(x, y, z); }

    /// @brief Serialize the coordinate
    template <class Serializer>
    void serialize(Serializer &s) {
        s("x", x);
        s("y", y);
        s("z", z);
    }
};

using RCoord3 = RS_Coord3D<RSignedIndex>;

}  // namespace rise

REFL_AUTO(type(rise::RCoord3), field(x), field(y), field(z))

#endif  // RS_COORD_H
