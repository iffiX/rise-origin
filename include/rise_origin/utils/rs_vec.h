// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_VEC3D_H
#define RS_VEC3D_H

#include <cmath>
#include <refl.hpp>

#include "rs_def.h"
#include "rs_float.h"
#include "rs_num_traits.h"

namespace rise {

/**
 * Collection of commonly used vector types
 */

/**
 * @brief Template of generic 2D vectors. The template parameter is assumed to be
 * either float or double depending on the desired numerical resolution.
 * @note For integer, certain operations like length() will cast into float/double
 * depending on integer type.
 *
 * @note The values are not initialized to make it compatible with plain old
 * data structure (POD).
 */
template <typename T>
struct RS_Vec2D {
    using ElementType = T;

    T x;  ///< The current X value.
    T y;  ///< The current Y value.

    /// Default constructor
    RS_Vec2D() = default;

    /// Constructor with specified individual values
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D(T x, T y) : x(x), y(y) {}

    /// Copy constructor
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D(const RS_Vec2D<U> &v) : x(v.x), y(v.y) {}

    /// Returns a vector of all zeros
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec2D zero() { return RS_Vec2D(0, 0); }

    /// Returns a vector of all ones
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec2D one() { return RS_Vec2D(1, 1); }

    /// Returns a vector filled with the specified value
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec2D full(T value) { return RS_Vec2D(value, value); }

    /// Returns a vector of highest possible values
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec2D highest() {
        return RS_Vec2D(RS_NumTraits<T>::highest(), RS_NumTraits<T>::highest());
    }

    /// Returns a vector of lowest possible values
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec2D lowest() {
        return RS_Vec2D(RS_NumTraits<T>::lowest(), RS_NumTraits<T>::lowest());
    }

    /// Print vector content
    HOST_DEVICE_INLINE_QUALIFIER void print(const char *title = "Vec2D") const {
        if constexpr (std::is_integral_v<T>) {
            printf("%s x:%d, y:%d\n", title, x, y);
        } else {
            printf("%s x:%e, y:%e\n", title, x, y);
        }
    }

    /// Returns true if all values are valid numbers
    HOST_DEVICE_INLINE_QUALIFIER bool isValid() const {
        using RS_FI = RS_FloatIntrinsics<T>;
        return !RS_FI::rs_isnan(x) && RS_FI::rs_isfinite(x) && !RS_FI::rs_isnan(y) &&
               RS_FI::rs_isfinite(y);
    }

    /// Cast operator for type conversion
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER explicit operator RS_Vec2D<U>() const {
        return RS_Vec2D<U>(x, y);
    }

    /// Assign operator
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D operator=(const RS_Vec2D<U> &v) {
        x = v.x;
        y = v.y;
        return *this;
    }

    /// Addition operator for vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D operator+(const RS_Vec2D &v) const {
        return RS_Vec2D(x + v.x, y + v.y);
    }

    /// Subtraction operator for vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D operator-(const RS_Vec2D &v) const {
        return RS_Vec2D(x - v.x, y - v.y);
    }

    /// Unary negation operator
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D operator-() const { return RS_Vec2D(-x, -y); }

    /// Multiplication operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D operator*(T f) const { return RS_Vec2D(f * x, f * y); }

    /// Multiplication operator for constant as left hand side
    HOST_DEVICE_INLINE_QUALIFIER friend RS_Vec2D operator*(T f, const RS_Vec2D &v) { return v * f; }

    /// Multiplication operator (dot multiplication) for vector
    HOST_DEVICE_INLINE_QUALIFIER T operator*(const RS_Vec2D &v) const { return x * v.x + y * v.y; }

    /// Division operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D operator/(T f) const {
        if constexpr (std::is_integral_v<T>) {
            return {x / f, y / f};
        } else {
            T f_inv = T(1) / f;
            return {x * f_inv, y * f_inv};
        }
    }

    /// Addition and set operator for vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D &operator+=(const RS_Vec2D &v) {
        x += v.x;
        y += v.y;
        return *this;
    }

    /// Subtraction and set operator for vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D &operator-=(const RS_Vec2D &v) {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    /// Multiplication and set operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D &operator*=(T f) {
        x *= f;
        y *= f;
        return *this;
    }

    /// Division and set operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D &operator/=(T f) {
        if constexpr (std::is_integral_v<T>) {
            x /= f;
            y /= f;
        } else {
            T f_inv = T(1) / f;
            x *= f_inv;
            y *= f_inv;
        }
        return *this;
    }

    /// Is equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_Vec2D &v) const {
        return x == v.x && y == v.y;
    }

    /// Is not equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(const RS_Vec2D &v) const {
        return x != v.x || y != v.y;
    }

    /// Index operator
    HOST_DEVICE_INLINE_QUALIFIER T &operator[](int index) {
        return *(reinterpret_cast<T *>(this) + index);
    }

    /// Const index operator
    HOST_DEVICE_INLINE_QUALIFIER const T &operator[](int index) const {
        return *(reinterpret_cast<const T *>(this) + index);
    }

    /**
     * @brief Returns a vector with each element rounded down to the nearest integer
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D floor() const {
        return RS_Vec2D(RS_FloatIntrinsics<T>::rs_floor(x), RS_FloatIntrinsics<T>::rs_floor(y));
    }

    /**
     * @brief Returns a vector with each element rounded up to the nearest integer
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D ceil() const {
        return RS_Vec2D(RS_FloatIntrinsics<T>::rs_ceil(x), RS_FloatIntrinsics<T>::rs_ceil(y));
    }

    /**
     * @brief Returns a vector with each element rounded to the nearest integer
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D round() const {
        return RS_Vec2D(RS_FloatIntrinsics<T>::rs_round(x), RS_FloatIntrinsics<T>::rs_round(y));
    }

    /**
     * @brief Returns a vector by clipping this vector between low and high
     *
     * @param low The lower bound
     * @param high The upper bound
     * @return The clipped vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D clip(T low, T high) const {
        return RS_Vec2D(RS_MIN(RS_MAX(x, low), high), RS_MIN(RS_MAX(y, low), high));
    }

    /**
     * @brief Returns the product of x and y
     */
    HOST_DEVICE_INLINE_QUALIFIER T prod() const { return x * y; }

    /**
     * @brief Returns true if the vector is bounded between low and high
     *
     * @param low The lower bound
     * @param high The upper bound
     * @return True if the vector is bounded between low and high
     */
    HOST_DEVICE_INLINE_QUALIFIER bool isInBound(T low, T high) const {
        return x >= low && x <= high && y >= low && y <= high;
    }

    /**
     * @brief Returns true if the vector is bounded between low and high
     *
     * @param low The lower bound
     * @param high The upper bound
     * @return True if the vector is bounded between low and high
     */
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER bool isInBound(RS_Vec2D<U> low, RS_Vec2D<U> high) const {
        return x >= low.x && x <= high.x && y >= low.y && y <= high.y;
    }

    /**
     * @brief Returns the mean value of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T mean() const { return sum() / T(2.0); }

    /**
     * @brief Returns the summed value of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T sum() const { return x + y; }

    /**
     * @brief Returns the dot product of this vector dotted with the provided vector v
     *
     * @param v The provided vector
     * @return The dot product of this vector dotted with the provided vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER T dot(const RS_Vec2D &v) const { return x * v.x + y * v.y; }

    /**
     * @brief Returns the absolute value of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D abs() const {
        return RS_Vec2D(x >= 0 ? x : -x, y >= 0 ? y : -y);
    }

    /**
     * @brief Returns the reciprocal of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D reciprocal() const {
        return RS_Vec2D(T(1.0) / x, T(1.0) / y);
    }

    /**
     * @brief Returns the reciprocal of this vector using fast division
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D reciprocalFast() const {
        return RS_Vec2D(
            RS_FloatIntrinsics<T>::rs_divide_fast(1.0, x),
            RS_FloatIntrinsics<T>::rs_divide_fast(1.0, y)
        );
    }

    /**
     * @brief Returns a normalized version of this vector, if the vector is a zero vector,
     * it will return a zero vector.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D normalized() const {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            FloatType fx = static_cast<FloatType>(x);
            FloatType fy = static_cast<FloatType>(y);
            FloatType l = RS_FloatIntrinsics<FloatType>::rs_sqrt(fx * fx + fy * fy);
            return l > 0 ? RS_Vec2D(static_cast<T>(fx / l), static_cast<T>(fy / l))
                         : RS_Vec2D::zero();
        } else {
            T l = RS_FloatIntrinsics<T>::rs_sqrt(x * x + y * y);
            return l > 0 ? (*this) / l : RS_Vec2D::zero();
        }
    }

    /**
     * @brief Returns the length (magnitude) of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T length() const {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            FloatType fx = static_cast<FloatType>(x);
            FloatType fy = static_cast<FloatType>(y);
            return static_cast<T>(RS_FloatIntrinsics<FloatType>::rs_sqrt(fx * fx + fy * fy));
        } else {
            return RS_FloatIntrinsics<T>::rs_sqrt(x * x + y * y);
        }
    }

    /**
     * @brief Returns the length (magnitude) squared of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T length2() const { return x * x + y * y; }

    /**
     * @brief Returns a vector populated by the minimum x and y of this vector and
     * the provided vector v
     *
     * @param v The provided vector
     * @return The vector with the minimum x and y of this vector and the provided vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D min(const RS_Vec2D &v) const {
        return RS_Vec2D(x < v.x ? x : v.x, y < v.y ? y : v.y);
    }

    /**
     * @brief Returns a vector populated by the maximum x and y of this vector and
     * the provided vector v
     *
     * @param v The provided vector
     * @return The vector with the maximum x and y of this vector and the provided vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D max(const RS_Vec2D &v) const {
        return RS_Vec2D(x > v.x ? x : v.x, y > v.y ? y : v.y);
    }

    /**
     * @brief Returns the smallest of x or y of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T min() const { return (x < y ? x : y); }

    /**
     * @brief Returns the index of the smallest of x or y
     */
    HOST_DEVICE_INLINE_QUALIFIER RIndex argmin() const { return x < y ? 0 : 1; }

    /**
     * @brief Returns the largest of x or y of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T max() const { return (x > y ? x : y); }

    /**
     * @brief Returns the index of the largest of x or y
     */
    HOST_DEVICE_INLINE_QUALIFIER RIndex argmax() const { return x > y ? 0 : 1; }

    /**
     * @brief Returns a element-wise multiplication of this vector and vector v
     *
     * @param v The provided vector
     * @return The element-wise multiplication of this vector and vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D elemMul(const RS_Vec2D &v) const {
        return RS_Vec2D(x * v.x, y * v.y);
    }

    /**
     * @brief Returns a element-wise division of this vector and vector v
     *
     * @param v The provided vector
     * @return The element-wise division of this vector and vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D elemDiv(const RS_Vec2D &v) const {
        return RS_Vec2D(x / v.x, y / v.y);
    }

    /**
     * @brief Returns a element-wise division of this vector and vector v using fast
     * division
     *
     * @param v The provided vector
     * @return The element-wise division of this vector and vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D elemDivFast(const RS_Vec2D &v) const {
        return RS_Vec2D(
            RS_FloatIntrinsics<T>::rs_divide_fast(v.x, x),
            RS_FloatIntrinsics<T>::rs_divide_fast(v.y, y)
        );
    }

    /**
     * @brief Returns the euclidian distance between this vector and the provided vector v
     *
     * @param v The provided vector
     * @return The euclidian distance
     */
    HOST_DEVICE_INLINE_QUALIFIER T dist(const RS_Vec2D &v) const {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            return static_cast<T>(RS_FloatIntrinsics<FloatType>::rs_sqrt(dist2(v)));
        } else {
            return RS_FloatIntrinsics<T>::rs_sqrt(dist2(v));
        }
    }

    /**
     * @brief Returns the squared euclidian distance between this vector and the provided vector
     * v
     *
     * @param v The provided vector
     * @return The squared euclidian distance
     */
    HOST_DEVICE_INLINE_QUALIFIER T dist2(const RS_Vec2D &v) const {
        return (v.x - x) * (v.x - x) + (v.y - y) * (v.y - y);
    }

    /**
     * @brief Returns the projected vector on the provided vector v (standard dot projection)
     *
     * @param v The provided vector
     * @return The projected vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec2D project(const RS_Vec2D &v) const {
        return (dot(v) / v.length2()) * v;
    }

    /**
     * @brief Generic serialization
     *
     * @param s The serializer
     */
    template <class Serializer>
    void serialize(Serializer &s) {
        s("x", x);
        s("y", y);
    }
};

using RVec2f = RS_Vec2D<float>;
using RVec2df = RS_Vec2D<double>;
using RVec2rf = RS_Vec2D<RFloat>;
using RVec2i = RS_Vec2D<int>;
using RVec2ui = RS_Vec2D<unsigned int>;

template <typename T>
class RS_Mat3D;

template <RSE_FloatType Type>
class RS_Vec3DMem;

/**
 * @brief Template of generic 3D vectors. The template parameter is assumed to be
 * either float or double depending on the desired numerical resolution, integer
 * is allowed but is not recommended.
 *
 * @Note The values are not initialized to make it compatible with plain old
 * data structure (POD).
 */
template <typename T>
class RS_Vec3D {
   public:
    using ElementType = T;

    T x;  ///< The current X value.
    T y;  ///< The current Y value.
    T z;  ///< The current Z value.

    // Constructors
    /// Default constructor
    RS_Vec3D() = default;

    /// Constructor with specified individual values
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D(T x, T y, T z) : x(x), y(y), z(z) {}

    /// Copy constructor
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D(const RS_Vec3D<U> &v) : x(v.x), y(v.y), z(v.z) {}

    /// Copy constructor for memory format
    template <RSE_FloatType Type>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D(const RS_Vec3DMem<Type> &v) {
        v.get(x, y, z);
    }

    /// Returns a vector of all zeros
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec3D zero() { return RS_Vec3D(0, 0, 0); }

    /// Returns a vector of all ones
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec3D one() { return RS_Vec3D(1, 1, 1); }

    /// Returns a vector filled with the specified value
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec3D full(T value) {
        return RS_Vec3D(value, value, value);
    }

    /// Returns a vector of highest possible values
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec3D highest() {
        return RS_Vec3D(
            RS_NumTraits<T>::highest(), RS_NumTraits<T>::highest(), RS_NumTraits<T>::highest()
        );
    }

    /// Returns a vector of lowest possible values
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec3D lowest() {
        return RS_Vec3D(
            RS_NumTraits<T>::lowest(), RS_NumTraits<T>::lowest(), RS_NumTraits<T>::lowest()
        );
    }

    /// Print vector content
    HOST_DEVICE_INLINE_QUALIFIER void print(const char *title = "Vec3D") const {
        if constexpr (std::is_integral_v<T>) {
            printf("%s x:%d, y:%d, z:%d\n", title, x, y, z);
        } else {
            printf("%s x:%e, y:%e, z:%e\n", title, x, y, z);
        }
    }

    /// Returns true if all values are valid numbers
    HOST_DEVICE_INLINE_QUALIFIER bool isValid() const {
        using RS_FI = RS_FloatIntrinsics<T>;
        return !RS_FI::rs_isnan(x) && RS_FI::rs_isfinite(x) && !RS_FI::rs_isnan(y) &&
               RS_FI::rs_isfinite(y) && !RS_FI::rs_isnan(z) && RS_FI::rs_isfinite(z);
    }

    /// Cast operator for type conversion
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER explicit operator RS_Vec3D<U>() const {
        return RS_Vec3D<U>(x, y, z);
    }

    /// Assign operator
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D operator=(const RS_Vec3D<U> &v) {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    /// Assign operator from memory format
    template <RSE_FloatType Type>
    HOST_DEVICE_QUALIFIER RS_Vec3D operator=(const RS_Vec3DMem<Type> &v) {
        v.get(x, y, z);
        return *this;
    }

    /// Addition operator for vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D operator+(const RS_Vec3D &v) const {
        return RS_Vec3D(x + v.x, y + v.y, z + v.z);
    }

    /// Subtraction operator for vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D operator-(const RS_Vec3D &v) const {
        return RS_Vec3D(x - v.x, y - v.y, z - v.z);
    }

    /// Unary negation operator
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D operator-() const { return RS_Vec3D(-x, -y, -z); }

    /// Multiplication operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D operator*(T f) const {
        return RS_Vec3D(f * x, f * y, f * z);
    }

    /// Multiplication operator for constant as left hand side
    HOST_DEVICE_INLINE_QUALIFIER friend RS_Vec3D operator*(T f, const RS_Vec3D &v) { return v * f; }

    /// Multiplication operator (dot multiplication) for vector
    HOST_DEVICE_INLINE_QUALIFIER T operator*(const RS_Vec3D &v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    /// Division operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D operator/(T f) const {
        if constexpr (std::is_integral_v<T>) {
            return {x / f, y / f, z / f};
        } else {
            T f_inv = T(1) / f;
            return {x * f_inv, y * f_inv, z * f_inv};
        }
    }

    /// Addition and set operator for vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D &operator+=(const RS_Vec3D &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    /// Subtraction and set operator for vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D &operator-=(const RS_Vec3D &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    /// Multiplication and set operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D &operator*=(T f) {
        x *= f;
        y *= f;
        z *= f;
        return *this;
    }

    /// Division and set operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D &operator/=(T f) {
        if constexpr (std::is_integral_v<T>) {
            x /= f;
            y /= f;
            z /= f;
        } else {
            T f_inv = T(1) / f;
            x *= f_inv;
            y *= f_inv;
            z *= f_inv;
        }
        return *this;
    }

    /// Is equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_Vec3D &v) const {
        return x == v.x && y == v.y && z == v.z;
    }

    /// Is not equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(const RS_Vec3D &v) const {
        return x != v.x || y != v.y || z != v.z;
    }

    /// Index operator
    HOST_DEVICE_INLINE_QUALIFIER T &operator[](int index) {
        return *(reinterpret_cast<T *>(this) + index);
    }

    /// Const index operator
    HOST_DEVICE_INLINE_QUALIFIER const T &operator[](int index) const {
        return *(reinterpret_cast<T *>(this) + index);
    }

    /**
     * @brief Returns a vector with each element rounded down to the nearest integer
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D floor() const {
        return RS_Vec3D(
            RS_FloatIntrinsics<T>::rs_floor(x),
            RS_FloatIntrinsics<T>::rs_floor(y),
            RS_FloatIntrinsics<T>::rs_floor(z)
        );
    }

    /**
     * @brief Returns a vector with each element rounded up to the nearest integer
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D ceil() const {
        return RS_Vec3D(
            RS_FloatIntrinsics<T>::rs_ceil(x),
            RS_FloatIntrinsics<T>::rs_ceil(y),
            RS_FloatIntrinsics<T>::rs_ceil(z)
        );
    }

    /**
     * @brief Returns a vector with each element rounded to the nearest integer
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D round() const {
        return RS_Vec3D(
            RS_FloatIntrinsics<T>::rs_round(x),
            RS_FloatIntrinsics<T>::rs_round(y),
            RS_FloatIntrinsics<T>::rs_round(z)
        );
    }

    /**
     * @brief Returns a vector by clipping this vector between low and high
     *
     * @param low The lower bound
     * @param high The upper bound
     * @return The clipped vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D clip(T low, T high) {
        return RS_Vec3D(
            RS_MIN(RS_MAX(x, low), high), RS_MIN(RS_MAX(y, low), high), RS_MIN(RS_MAX(z, low), high)
        );
    }

    /**
     * @brief Returns the product of x, y, and z
     */
    HOST_DEVICE_INLINE_QUALIFIER T prod() const { return x * y * z; }

    /**
     * @brief Returns true if the vector is bounded between low and high
     *
     * @param low The lower bound
     * @param high The upper bound
     * @return True if the vector is bounded between low and high
     */
    HOST_DEVICE_INLINE_QUALIFIER bool isInBound(T low, T high) const {
        return x >= low && x <= high && y >= low && y <= high && z >= low && z <= high;
    }

    /**
     * @brief Returns true if the vector is bounded between low and high
     *
     * @param low The lower bound
     * @param high The upper bound
     * @return True if the vector is bounded between low and high
     */

    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER bool isInBound(RS_Vec3D<U> low, RS_Vec3D<U> high) const {
        return x >= low.x && x <= high.x && y >= low.y && y <= high.y && z >= low.z && z <= high.z;
    }

    /**
     * @brief Returns a vector by rotating this vector about an axis vector
     * defined by v an angle "a" in radians, see
     * http://www.cprogramming.com/tutorial/3d/rotation.html
     *
     * @param v The axis vector
     * @param a The angle in radians
     * @return The rotated vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D rotate(const RS_Vec3D &v, T a) {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            return RS_Vec3D(RS_Vec3D<FloatType>(*this).rotate(v, a));
        } else {
            T ca = RS_FloatIntrinsics<T>::rs_cos(a);
            T sa = RS_FloatIntrinsics<T>::rs_sin(a);
            T t = T(1) - ca;
            T new_x = (v.x * v.x * t + ca) * x + (v.x * v.y * t - v.z * sa) * y +
                      (v.z * v.x * t + v.y * sa) * z;
            T new_y = (v.x * v.y * t + v.z * sa) * x + (v.y * v.y * t + ca) * y +
                      (v.y * v.z * t - v.x * sa) * z;
            T new_z = (v.z * v.x * t - v.y * sa) * x + (v.y * v.z * t + v.x * sa) * y +
                      (v.z * v.z * t + ca) * z;
            return RS_Vec3D(new_x, new_y, new_z);
        }
    }

    /**
     * @brief Returns a vector by rotating this vector about the X axis a radians
     *
     * @param a The angle in radians
     * @return The rotated vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D rotateX(T a) {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            return RS_Vec3D(RS_Vec3D<FloatType>(*this).rotateX(a));
        } else {
            T sin_a, cos_a;
            RS_FloatIntrinsics<T>::rs_sincos(a, &sin_a, &cos_a);
            T yt = y * cos_a + z * sin_a;
            T zt = -y * sin_a + z * cos_a;
            return RS_Vec3D(x, yt, zt);
        }
    }

    /**
     * @brief Returns a vector by rotating this vector about the Y axis a radians
     *
     * @param a The angle in radians
     * @return The rotated vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D rotateY(T a) {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            return RS_Vec3D(RS_Vec3D<FloatType>(*this).rotateY(a));
        } else {
            T sin_a, cos_a;
            RS_FloatIntrinsics<T>::rs_sincos(a, &sin_a, &cos_a);
            T xt = x * cos_a + z * sin_a;
            T zt = -x * sin_a + z * cos_a;
            return RS_Vec3D(xt, y, zt);
        }
    }

    /**
     * @brief Returns a vector by rotating this vector about the Z axis a radians
     *
     * @param a The angle in radians
     * @return The rotated vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D rotateZ(T a) {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            return RS_Vec3D(RS_Vec3D<FloatType>(*this).rotateZ(a));
        } else {
            T sin_a, cos_a;
            RS_FloatIntrinsics<T>::rs_sincos(a, &sin_a, &cos_a);
            T xt = x * cos_a - y * sin_a;
            T yt = x * sin_a + y * cos_a;
            return RS_Vec3D(xt, yt, z);
        }
    }

    /**
     * @brief Returns the mean value of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T mean() const { return sum() / T(3.0); }

    /**
     * @brief Returns the the summed value of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T sum() const { return x + y + z; }

    /**
     * @brief Returns the cross product of this vector crossed by the provided vector v
     *
     * @param v The provided vector
     * @return The cross product
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D cross(const RS_Vec3D &v) const {
        return RS_Vec3D(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    /**
     * @brief Returns the dot product of this vector dotted with the provided vector v
     *
     * @param v The provided vector
     * @return The dot product
     */
    HOST_DEVICE_INLINE_QUALIFIER T dot(const RS_Vec3D &v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    /**
     * @brief Returns the absolute value of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D abs() const {
        return RS_Vec3D(x >= 0 ? x : -x, y >= 0 ? y : -y, z >= 0 ? z : -z);
    }

    /**
     * @brief Returns the reciprocal of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D reciprocal() const {
        return RS_Vec3D(T(1.0) / x, T(1.0) / y, T(1.0) / z);
    }

    /**
     * @brief Returns the reciprocal of this vector using fast division
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D reciprocalFast() const {
        return RS_Vec3D(
            RS_FloatIntrinsics<T>::rs_divide_fast(1.0, x),
            RS_FloatIntrinsics<T>::rs_divide_fast(1.0, y),
            RS_FloatIntrinsics<T>::rs_divide_fast(1.0, z)
        );
    }

    /**
     * @brief Returns the normalized vector, if the vector is a zero vector,
     * it will return a zero vector.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D normalized() const {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            FloatType fx = static_cast<FloatType>(x);
            FloatType fy = static_cast<FloatType>(y);
            FloatType fz = static_cast<FloatType>(z);
            FloatType l = RS_FloatIntrinsics<FloatType>::rs_sqrt(fx * fx + fy * fy + fz * fz);
            return l > 0
                       ? RS_Vec3D(
                             static_cast<T>(fx / l), static_cast<T>(fy / l), static_cast<T>(fz / l)
                         )
                       : RS_Vec3D::zero();
        } else {
            T l = RS_FloatIntrinsics<T>::rs_sqrt(x * x + y * y + z * z);
            return l > 0 ? (*this) / l : RS_Vec3D::zero();
        }
    }

    /**
     * @brief Returns the length (magnitude) of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T length() const {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            FloatType fx = static_cast<FloatType>(x);
            FloatType fy = static_cast<FloatType>(y);
            FloatType fz = static_cast<FloatType>(z);
            return static_cast<T>(
                RS_FloatIntrinsics<FloatType>::rs_sqrt(fx * fx + fy * fy + fz * fz)
            );
        } else {
            return RS_FloatIntrinsics<T>::rs_sqrt(x * x + y * y + z * z);
        }
    }

    /**
     * @brief Returns the length (magnitude) squared of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T length2() const { return x * x + y * y + z * z; }

    /**
     * @brief Returns a vector populated by the minimum x, y, and z value of this vector and
     * the provided vector v
     *
     * @param v The provided vector
     * @return The vector with the minimum x, y, and z value of this vector and the provided vector
     * v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D min(const RS_Vec3D &v) const {
        return RS_Vec3D(x < v.x ? x : v.x, y < v.y ? y : v.y, z < v.z ? z : v.z);
    }

    /**
     * @brief Returns a vector populated by the maximum x, y, and z value of this vector and
     * the provided vector v
     *
     * @param v The provided vector
     * @return The vector with the maximum x, y, and z value of this vector and the provided vector
     * v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D max(const RS_Vec3D &v) const {
        return RS_Vec3D(x > v.x ? x : v.x, y > v.y ? y : v.y, z > v.z ? z : v.z);
    }

    /**
     * @brief Returns the smallest of x, y, or z of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T min() const {
        T min_xy = (x < y ? x : y);
        return (z < min_xy ? z : min_xy);
    }

    /**
     * @brief Returns the index of the smallest of x, y, or z of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RIndex argmin() const {
        T min_xy = (x < y ? x : y);
        return (z < min_xy ? 2 : (x < y ? 0 : 1));
    }

    /**
     * @brief Returns the largest of x, y, or z of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T max() const {
        T max_xy = (x > y ? x : y);
        return (z > max_xy ? z : max_xy);
    }

    /**
     * @brief Returns the index of the largest of x, y, or z of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RIndex argmax() const {
        T max_xy = (x > y ? x : y);
        return (z > max_xy ? 2 : (x > y ? 0 : 1));
    }

    /**
     * @brief Returns a element-wise multiplication of this vector and vector v
     *
     * @param v The provided vector
     * @return The element-wise multiplication of this vector and vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D elemMul(const RS_Vec3D &v) const {
        return RS_Vec3D(x * v.x, y * v.y, z * v.z);
    }

    /**
     * @brief Returns a element-wise division of this vector and vector v
     *
     * @param v The provided vector
     * @return The element-wise division of this vector and vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D elemDiv(const RS_Vec3D &v) const {
        return RS_Vec3D(x / v.x, y / v.y, z / v.z);
    }

    /**
     * @brief Returns a element-wise division of this vector and vector v using fast
     * division
     *
     * @param v The provided vector
     * @return The element-wise division of this vector and vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D elemDivFast(const RS_Vec3D &v) const {
        return RS_Vec3D(
            RS_FloatIntrinsics<T>::rs_divide_fast(v.x, x),
            RS_FloatIntrinsics<T>::rs_divide_fast(v.y, y),
            RS_FloatIntrinsics<T>::rs_divide_fast(v.z, z)
        );
    }

    /**
     * @brief Returns the euclidian distance between this vector and the specified
     * vector v
     *
     * @param v The provided vector
     * @return The euclidian distance
     */
    HOST_DEVICE_INLINE_QUALIFIER T dist(const RS_Vec3D &v) const {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            return static_cast<T>(RS_FloatIntrinsics<FloatType>::rs_sqrt(dist2(v)));
        } else {
            return RS_FloatIntrinsics<T>::rs_sqrt(dist2(v));
        }
    }

    /**
     * @brief Returns the squared euclidian distance between this vector and the specified
     * vector v
     *
     * @param v The provided vector
     * @return The squared euclidian distance
     */
    HOST_DEVICE_INLINE_QUALIFIER T dist2(const RS_Vec3D &v) const {
        return (v.x - x) * (v.x - x) + (v.y - y) * (v.y - y) + (v.z - z) * (v.z - z);
    }

    /**
     * @brief Returns the projected vector on the provided vector v
     *
     * @param v The provided vector
     * @return The projected vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D project(const RS_Vec3D &v) const {
        // use length2 to avoid extra sqrt
        return (dot(v) / v.length2()) * v;
    }

    /**
     * @brief Generates and returns an arbitrary vector that is normal to this one.
     * See https://blog.selfshadow.com/2011/10/17/perp-vectors/
     * Hughes-Möller perpendicular vector generation
     *
     * @note If the vector is a zero vector, it will return a zero vector.
     * @return The arbitrary normal vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D arbitraryNormal() const {
        RS_Vec3D n = normalized();
        RS_Vec3D v(0, 0, 0);
        if (fabs(n.x) <= fabs(n.y) && fabs(n.x) <= fabs(n.z)) {
            v = RS_Vec3D(0, -n.z, n.y);
        } else if (fabs(n.y) <= fabs(n.x) && fabs(n.y) <= fabs(n.z)) {
            v = RS_Vec3D(-n.z, 0, n.x);
        } else {
            v = RS_Vec3D(-n.y, n.x, 0);
        }
        return v.normalized();
    }

    /**
     * @brief Converts this vector v to a cross product operator matrix v, so that
     * V * w = v x w for any vector "w".
     *
     * @return The cross product operator matrix
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D<T> toCrossMatrix() const {
        return RS_Mat3D<T>(
            0,
            -z,
            y,  //
            z,
            0,
            -x,  //
            -y,
            x,
            0
        );
    }

    /**
     * @brief Generic serialization
     *
     * @param s The serializer
     */
    template <class Serializer>
    void serialize(Serializer &s) {
        s("x", x);
        s("y", y);
        s("z", z);
    }
};

// Helper struct to determine data array based on Type
template <RSE_FloatType Type>
struct RS_Vec3DMemDataType;

template <>
struct RS_Vec3DMemDataType<RSE_FLOAT_64> {
    using DataType = double[3];
    using ElementType = double;
};

template <>
struct RS_Vec3DMemDataType<RSE_FLOAT_32> {
    using DataType = float[3];
    using ElementType = float;
};

/**
 * @brief Template of 3D vectors stored in memory.
 */
template <RSE_FloatType Type>
class RS_Vec3DMem {
   public:
    using DataType = typename RS_Vec3DMemDataType<Type>::DataType;
    using ElementType = typename RS_Vec3DMemDataType<Type>::ElementType;
    DataType data;  ///< Internal storage.

    /// Default constructor
    RS_Vec3DMem() = default;

    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec3DMem zero() {
        RS_Vec3DMem vec;
        vec.data[0] = 0;
        vec.data[1] = 0;
        if constexpr (Type == RSE_FLOAT_32 || Type == RSE_FLOAT_64) {
            vec.data[2] = 0;
        }
        return vec;
    }

    /// Constructor taking x, y, z as input
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3DMem(T x, T y, T z) {
        set(x, y, z);
    }

    /// Constructor from RS_Vec3D<T>
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_Vec3DMem(const RS_Vec3D<T> &v) {
        set(v.x, v.y, v.z);
    }

    /// Assignment operator from RS_Vec3D<T>
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3DMem &operator=(const RS_Vec3D<T> &v) {
        set(v.x, v.y, v.z);
        return *this;
    }

    /// Assignment operator from RS_Vec3DMem
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3DMem &operator=(const RS_Vec3DMem &v) {
        data[0] = v.data[0];
        data[1] = v.data[1];
        if constexpr (Type == RSE_FLOAT_32 || Type == RSE_FLOAT_64) {
            data[2] = v.data[2];
        }
        return *this;
    }

    /// Convert to RS_Vec3D<T>
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toVec3D() const {
        RS_Vec3D<T> v;
        get(v.x, v.y, v.z);
        return v;
    }

    /// Helper function to set data from x, y, z
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER void set(const T &x, const T &y, const T &z) {
        if constexpr (Type == RSE_FLOAT_32 || Type == RSE_FLOAT_64) {
            data[0] = static_cast<T>(x);
            data[1] = static_cast<T>(y);
            data[2] = static_cast<T>(z);
        }
    }

    /// Helper function to get data into RS_Vec3D<T>
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER void get(T &x, T &y, T &z) const {
        if constexpr (Type == RSE_FLOAT_32 || Type == RSE_FLOAT_64) {
            x = static_cast<T>(data[0]);
            y = static_cast<T>(data[1]);
            z = static_cast<T>(data[2]);
        }
    }
};

using RVec3f = RS_Vec3D<float>;
using RVec3df = RS_Vec3D<double>;
using RVec3rf = RS_Vec3D<RFloat>;
using RVec3i = RS_Vec3D<int>;
using RVec3ui = RS_Vec3D<unsigned int>;

/**
 * @brief Template of generic 4D vectors. The template parameter is assumed to be
 * either float or double depending on the desired numerical resolution, integer
 * is allowed but is not recommended.
 *
 * @note The values are not initialized to make it compatible with plain old
 * data structure (POD).
 */
template <typename T>
struct RS_Vec4D {
    using ElementType = T;

    T x;  ///< The current X value.
    T y;  ///< The current Y value.
    T z;  ///< The current Z value.
    T w;  ///< The current W value.

    // Constructors
    /// Default constructor
    RS_Vec4D() = default;

    /// Constructor with specified individual values
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}

    /// Copy constructor
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D(const RS_Vec4D<U> &v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

    /// Returns a vector of all zeros
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec4D zero() { return RS_Vec4D(0, 0, 0, 0); }

    /// Returns a vector of all ones
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec4D one() { return RS_Vec4D(1, 1, 1, 1); }

    /// Returns a vector filled with the specified value
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec4D full(T value) {
        return RS_Vec4D(value, value, value, value);
    }

    /// Returns a vector of highest possible values
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec4D highest() {
        return RS_Vec4D(
            RS_NumTraits<T>::highest(),
            RS_NumTraits<T>::highest(),
            RS_NumTraits<T>::highest(),
            RS_NumTraits<T>::highest()
        );
    }

    /// Returns a vector of lowest possible values
    HOST_DEVICE_INLINE_QUALIFIER static RS_Vec4D lowest() {
        return RS_Vec4D(
            RS_NumTraits<T>::lowest(),
            RS_NumTraits<T>::lowest(),
            RS_NumTraits<T>::lowest(),
            RS_NumTraits<T>::lowest()
        );
    }

    /// Print vector content
    HOST_DEVICE_INLINE_QUALIFIER void print(const char *title = "Vec4D") const {
        if constexpr (std::is_integral_v<T>) {
            printf("%s x:%d, y:%d, z:%d, w:%d\n", title, x, y, z, w);
        } else {
            printf("%s x:%e, y:%e, z:%e, w:%e\n", title, x, y, z, w);
        }
    }

    /// Returns true if all values are valid numbers
    HOST_DEVICE_INLINE_QUALIFIER bool isValid() const {
        using RS_FI = RS_FloatIntrinsics<T>;
        return !RS_FI::rs_isnan(x) && RS_FI::rs_isfinite(x) && !RS_FI::rs_isnan(y) &&
               RS_FI::rs_isfinite(y) && !RS_FI::rs_isnan(z) && RS_FI::rs_isfinite(z) &&
               !RS_FI::rs_isnan(w) && RS_FI::rs_isfinite(w);
    }

    /// Cast operator for type conversion
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER explicit operator RS_Vec4D<U>() const {
        return RS_Vec4D<U>(x, y, z, w);
    }

    /// Assign operator
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D operator=(const RS_Vec4D<U> &v) {
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
        return *this;
    }

    /// Addition operator for vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D operator+(const RS_Vec4D &v) const {
        return RS_Vec4D(x + v.x, y + v.y, z + v.z, w + v.w);
    }

    /// Subtraction operator for vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D operator-(const RS_Vec4D &v) const {
        return RS_Vec4D(x - v.x, y - v.y, z - v.z, w - v.w);
    }

    /// Unary negation operator
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D operator-() const { return RS_Vec4D(-x, -y, -z, -w); }

    /// Multiplication operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D operator*(T f) const {
        return RS_Vec4D(f * x, f * y, f * z, f * w);
    }

    /// Multiplication operator for constant as left hand side
    HOST_DEVICE_INLINE_QUALIFIER friend RS_Vec4D operator*(T f, const RS_Vec4D &v) { return v * f; }

    /// Multiplication operator (dot multiplication) for vector
    HOST_DEVICE_INLINE_QUALIFIER T operator*(const RS_Vec4D &v) const {
        return x * v.x + y * v.y + z * v.z + w * v.w;
    }

    /// Division operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D operator/(T f) const {
        if constexpr (std::is_integral_v<T>) {
            return {x / f, y / f, z / f, w / f};
        } else {
            T f_inv = T(1) / f;
            return {x * f_inv, y * f_inv, z * f_inv, w * f_inv};
        }
    }

    /// Addition and set operator for vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D &operator+=(const RS_Vec4D &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }

    /// Subtraction and set operator for vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D &operator-=(const RS_Vec4D &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }

    /// Multiplication and set operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D &operator*=(T f) {
        x *= f;
        y *= f;
        z *= f;
        w *= f;
        return *this;
    }

    /// Division and set operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D &operator/=(T f) {
        if constexpr (std::is_integral_v<T>) {
            x /= f;
            y /= f;
            z /= f;
            w /= f;
        } else {
            T f_inv = T(1) / f;
            x *= f_inv;
            y *= f_inv;
            z *= f_inv;
            w *= f_inv;
        }
        return *this;
    }

    /// Is equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_Vec4D &v) const {
        return x == v.x && y == v.y && z == v.z && w == v.w;
    }

    /// Is not equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(const RS_Vec4D &v) const {
        return x != v.x || y != v.y || z != v.z || w != v.w;
    }

    /// Index operator
    HOST_DEVICE_INLINE_QUALIFIER T &operator[](int index) {
        return *(reinterpret_cast<T *>(this) + index);
    }

    /// Const index operator
    HOST_DEVICE_INLINE_QUALIFIER const T &operator[](int index) const {
        return *(reinterpret_cast<const T *>(this) + index);
    }

    /**
     * @brief Returns a vector with each element rounded down to the nearest integer
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D floor() const {
        return RS_Vec4D(
            RS_FloatIntrinsics<T>::rs_floor(x),
            RS_FloatIntrinsics<T>::rs_floor(y),
            RS_FloatIntrinsics<T>::rs_floor(z),
            RS_FloatIntrinsics<T>::rs_floor(w)
        );
    }

    /**
     * @brief Returns a vector with each element rounded up to the nearest integer
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D ceil() const {
        return RS_Vec4D(
            RS_FloatIntrinsics<T>::rs_ceil(x),
            RS_FloatIntrinsics<T>::rs_ceil(y),
            RS_FloatIntrinsics<T>::rs_ceil(z),
            RS_FloatIntrinsics<T>::rs_ceil(w)
        );
    }

    /**
     * @brief Returns a vector with each element rounded to the nearest integer
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D round() const {
        return RS_Vec4D(
            RS_FloatIntrinsics<T>::rs_round(x),
            RS_FloatIntrinsics<T>::rs_round(y),
            RS_FloatIntrinsics<T>::rs_round(z),
            RS_FloatIntrinsics<T>::rs_round(w)
        );
    }

    /**
     * @brief Returns a vector by clipping this vector between low and high
     *
     * @param low The lower bound
     * @param high The upper bound
     * @return The clipped vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D clip(T low, T high) const {
        return RS_Vec4D(
            RS_MIN(RS_MAX(x, low), high),
            RS_MIN(RS_MAX(y, low), high),
            RS_MIN(RS_MAX(z, low), high),
            RS_MIN(RS_MAX(w, low), high)
        );
    }

    /**
     * @brief Returns the product of x, y, z, and w
     */
    HOST_DEVICE_INLINE_QUALIFIER T prod() const { return x * y * z * w; }

    /**
     * @brief Returns true if the vector is bounded between low and high
     *
     * @param low The lower bound
     * @param high The upper bound
     * @return True if the vector is bounded between low and high
     */
    HOST_DEVICE_INLINE_QUALIFIER bool isInBound(T low, T high) const {
        return x >= low && x <= high && y >= low && y <= high && z >= low && z <= high &&
               w >= low && w <= high;
    }

    /**
     * @brief Returns true if the vector is bounded between low and high
     *
     * @param low The lower bound
     * @param high The upper bound
     * @return True if the vector is bounded between low and high
     */
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER bool isInBound(RS_Vec4D<U> low, RS_Vec4D<U> high) const {
        return x >= low.x && x <= high.x && y >= low.y && y <= high.y && z >= low.z &&
               z <= high.z && w >= low.w && w <= high.w;
    }

    /**
     * @brief Returns the mean value of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T mean() const { return sum() / T(4.0); }

    /**
     * @brief Returns the summed value of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T sum() const { return x + y + z + w; }

    /**
     * @brief Returns the dot product of this vector dotted with the provided vector v
     *
     * @param v The provided vector
     * @return The dot product
     */
    HOST_DEVICE_INLINE_QUALIFIER T dot(const RS_Vec4D &v) const {
        return x * v.x + y * v.y + z * v.z + w * v.w;
    }

    /**
     * @brief Returns the absolute value of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D abs() const {
        return RS_Vec4D(x >= 0 ? x : -x, y >= 0 ? y : -y, z >= 0 ? z : -z, w >= 0 ? w : -w);
    }

    /**
     * @brief Returns the reciprocal of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D reciprocal() const {
        return RS_Vec4D(T(1.0) / x, T(1.0) / y, T(1.0) / z, T(1.0) / w);
    }

    /**
     * @brief Returns the reciprocal of this vector using fast division
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D reciprocalFast() const {
        return RS_Vec4D(
            RS_FloatIntrinsics<T>::rs_divide_fast(1.0, x),
            RS_FloatIntrinsics<T>::rs_divide_fast(1.0, y),
            RS_FloatIntrinsics<T>::rs_divide_fast(1.0, z),
            RS_FloatIntrinsics<T>::rs_divide_fast(1.0, w)
        );
    }

    /**
     * @brief Returns the normalized vector, if the vector is a zero vector,
     * it will return a zero vector.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D normalized() const {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            FloatType fx = static_cast<FloatType>(x);
            FloatType fy = static_cast<FloatType>(y);
            FloatType fz = static_cast<FloatType>(z);
            FloatType fw = static_cast<FloatType>(w);
            FloatType l =
                RS_FloatIntrinsics<FloatType>::rs_sqrt(fx * fx + fy * fy + fz * fz + fw * fw);
            return l > 0 ? RS_Vec4D(
                               static_cast<T>(fx / l),
                               static_cast<T>(fy / l),
                               static_cast<T>(fz / l),
                               static_cast<T>(fw / l)
                           )
                         : RS_Vec4D::zero();
        } else {
            T l = RS_FloatIntrinsics<T>::rs_sqrt(x * x + y * y + z * z + w * w);
            return l > 0 ? (*this) / l : RS_Vec4D::zero();
        }
    }

    /**
     * @brief Returns the length (magnitude) of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T length() const {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            FloatType fx = static_cast<FloatType>(x);
            FloatType fy = static_cast<FloatType>(y);
            FloatType fz = static_cast<FloatType>(z);
            FloatType fw = static_cast<FloatType>(w);
            return static_cast<T>(
                RS_FloatIntrinsics<FloatType>::rs_sqrt(fx * fx + fy * fy + fz * fz + fw * fw)
            );
        } else {
            return RS_FloatIntrinsics<T>::rs_sqrt(x * x + y * y + z * z + w * w);
        }
    }

    /**
     * @brief Returns the length (magnitude) squared of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T length2() const { return x * x + y * y + z * z + w * w; }

    /**
     * @brief Returns a vector populated by the minimum x, y, z, and w of this vector and
     * the provided vector v
     *
     * @param v The provided vector
     * @return The vector with the minimum x, y, z, and w of this vector and the provided vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D min(const RS_Vec4D &v) const {
        return RS_Vec4D(x < v.x ? x : v.x, y < v.y ? y : v.y, z < v.z ? z : v.z, w < v.w ? w : v.w);
    }

    /**
     * @brief Returns a vector populated by the maximum x, y, z, and w of this vector and
     * the provided vector v
     *
     * @param v The provided vector
     * @return The vector with the maximum x, y, z, and w of this vector and the provided vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D max(const RS_Vec4D &v) const {
        return RS_Vec4D(x > v.x ? x : v.x, y > v.y ? y : v.y, z > v.z ? z : v.z, w > v.w ? w : v.w);
    }

    /**
     * @brief Returns the smallest of x, y, z, or w of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T min() const {
        T min_xy = (x < y ? x : y);
        T min_zw = (z < w ? z : w);
        return (min_xy < min_zw ? min_xy : min_zw);
    }

    /**
     * @brief Returns the index of the smallest of x, y, z, or w
     */
    HOST_DEVICE_INLINE_QUALIFIER RIndex argmin() const {
        T min_xy, min_zw;
        RIndex argmin_xy, argmin_zw;
        if (x < y) {
            min_xy = x;
            argmin_xy = 0;
        } else {
            min_xy = y;
            argmin_xy = 1;
        }
        if (z < w) {
            min_zw = z;
            argmin_zw = 2;
        } else {
            min_zw = w;
            argmin_zw = 3;
        }
        return (min_xy < min_zw ? argmin_xy : argmin_zw);
    }

    /**
     * @brief Returns the largest of x, y, z, or w of this vector
     */
    HOST_DEVICE_INLINE_QUALIFIER T max() const {
        T max_xy = (x > y ? x : y);
        T max_zw = (z > w ? z : w);
        return (max_xy > max_zw ? max_xy : max_zw);
    }

    /**
     * @brief Returns the index of the largest of x, y, z, or w
     */
    HOST_DEVICE_INLINE_QUALIFIER RIndex argmax() const {
        T max_xy, max_zw;
        RIndex argmax_xy, argmax_zw;
        if (x > y) {
            max_xy = x;
            argmax_xy = 0;
        } else {
            max_xy = y;
            argmax_xy = 1;
        }
        if (z > w) {
            max_zw = z;
            argmax_zw = 2;
        } else {
            max_zw = w;
            argmax_zw = 3;
        }
        return (max_xy > max_zw ? argmax_xy : argmax_zw);
    }

    /**
     * @brief Returns a element-wise multiplication of this vector and vector v
     *
     * @param v The provided vector
     * @return The element-wise multiplication of this vector and vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D elemMul(const RS_Vec4D &v) const {
        return RS_Vec4D(x * v.x, y * v.y, z * v.z, w * v.w);
    }

    /**
     * @brief Returns a element-wise division of this vector and vector v
     *
     * @param v The provided vector
     * @return The element-wise division of this vector and vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D elemDiv(const RS_Vec4D &v) const {
        return RS_Vec4D(x / v.x, y / v.y, z / v.z, w / v.w);
    }

    /**
     * @brief Returns a element-wise division of this vector and vector v using fast
     * division
     *
     * @param v The provided vector
     * @return The element-wise division of this vector and vector v
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D elemDivFast(const RS_Vec4D &v) const {
        return RS_Vec4D(
            RS_FloatIntrinsics<T>::rs_divide_fast(v.x, x),
            RS_FloatIntrinsics<T>::rs_divide_fast(v.y, y),
            RS_FloatIntrinsics<T>::rs_divide_fast(v.z, z),
            RS_FloatIntrinsics<T>::rs_divide_fast(v.w, w)
        );
    }

    /**
     * @brief Returns the euclidian distance between this vector and the provided vector v
     *
     * @param v The provided vector
     * @return The euclidian distance
     */
    HOST_DEVICE_INLINE_QUALIFIER T dist(const RS_Vec4D &v) const {
        if constexpr (std::is_integral_v<T>) {
            using FloatType = std::conditional_t<
                std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>,
                double,
                float>;
            return static_cast<T>(RS_FloatIntrinsics<FloatType>::rs_sqrt(dist2(v)));
        } else {
            return RS_FloatIntrinsics<T>::rs_sqrt(dist2(v));
        }
    }

    /**
     * @brief Returns the squared euclidian distance between this vector and the specified
     * vector v
     *
     * @param v The provided vector
     * @return The squared euclidian distance
     */
    HOST_DEVICE_INLINE_QUALIFIER T dist2(const RS_Vec4D &v) const {
        return (v.x - x) * (v.x - x) + (v.y - y) * (v.y - y) + (v.z - z) * (v.z - z) +
               (v.w - w) * (v.w - w);
    }

    /**
     * @brief Returns the projected vector on the provided vector v
     *
     * @param v The provided vector
     * @return The projected vector
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec4D project(const RS_Vec4D &v) const {
        // use length2 to avoid extra sqrt
        return (dot(v) / v.length2()) * v;
    }

    /**
     * @brief Generic serialization
     *
     * @param s The serializer
     */
    template <class Serializer>
    void serialize(Serializer &s) {
        s("x", x);
        s("y", y);
        s("z", z);
        s("w", w);
    }
};

using RVec4f = RS_Vec4D<float>;
using RVec4df = RS_Vec4D<double>;
using RVec4rf = RS_Vec4D<RFloat>;
using RVec4i = RS_Vec4D<int>;
using RVec4ui = RS_Vec4D<unsigned int>;

}  // namespace rise

REFL_AUTO(type(rise::RVec2f), field(x), field(y))
REFL_AUTO(type(rise::RVec2df), field(x), field(y))
REFL_AUTO(type(rise::RVec2i), field(x), field(y))

REFL_AUTO(type(rise::RVec3f), field(x), field(y), field(z))
REFL_AUTO(type(rise::RVec3df), field(x), field(y), field(z))
REFL_AUTO(type(rise::RVec3i), field(x), field(y), field(z))

REFL_AUTO(type(rise::RVec4f), field(x), field(y), field(z), field(w))
REFL_AUTO(type(rise::RVec4df), field(x), field(y), field(z), field(w))
REFL_AUTO(type(rise::RVec4i), field(x), field(y), field(z), field(w))

REFL_AUTO(type(rise::RS_Vec3DMem<rise::RSE_FLOAT_32>), field(data))
REFL_AUTO(type(rise::RS_Vec3DMem<rise::RSE_FLOAT_64>), field(data))

#endif  // RS_VEC3D_H
