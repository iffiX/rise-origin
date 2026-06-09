// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_BOUND3D_H
#define RS_BOUND3D_H

#include "rs_def.h"
#include "rs_vec.h"

namespace rise {

/**
 * @brief Template of 3D bounds (axis-aligned bounding box). The template parameter is assumed to be
 * either float or double depending on the desired numerical resolution.
 *
 * @note The values are not initialized to make it compatible with plain old
 * data structure (POD).
 */
template <typename T>
class RS_Bound3D {
   public:
    using ElementType = T;

    T min_x;  ///< The minimum X coordinate.
    T min_y;  ///< The minimum Y coordinate.
    T min_z;  ///< The minimum Z coordinate.
    T max_x;  ///< The maximum X coordinate.
    T max_y;  ///< The maximum Y coordinate.
    T max_z;  ///< The maximum Z coordinate.

    // Constructors
    /// Default constructor
    RS_Bound3D() = default;

    /// Constructor with specified individual values
    HOST_DEVICE_INLINE_QUALIFIER RS_Bound3D(T min_x, T min_y, T min_z, T max_x, T max_y, T max_z)
        : min_x(min_x), min_y(min_y), min_z(min_z), max_x(max_x), max_y(max_y), max_z(max_z) {}

    /// Constructor with min and max vectors
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER
    RS_Bound3D(const RS_Vec3D<U>& min_bound, const RS_Vec3D<U>& max_bound)
        : min_x(min_bound.x),
          min_y(min_bound.y),
          min_z(min_bound.z),
          max_x(max_bound.x),
          max_y(max_bound.y),
          max_z(max_bound.z) {}

    /// Constructor with a single point (creates a bound with zero volume)
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_Bound3D(const RS_Vec3D<U>& point)
        : min_x(point.x),
          min_y(point.y),
          min_z(point.z),
          max_x(point.x),
          max_y(point.y),
          max_z(point.z) {}

    /// Copy constructor
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Bound3D(const RS_Bound3D<U>& bound)
        : min_x(bound.min_x),
          min_y(bound.min_y),
          min_z(bound.min_z),
          max_x(bound.max_x),
          max_y(bound.max_y),
          max_z(bound.max_z) {}

    /// Returns a zero bound (with all dimensions set to zero)
    HOST_DEVICE_INLINE_QUALIFIER static RS_Bound3D zero() { return RS_Bound3D(0, 0, 0, 0, 0, 0); }

    /// Returns a bound representing the lowest possible values
    HOST_DEVICE_INLINE_QUALIFIER static RS_Bound3D lowest() {
        return RS_Bound3D(
            RS_NumTraits<T>::highest(),  // min_x
            RS_NumTraits<T>::highest(),  // min_y
            RS_NumTraits<T>::highest(),  // min_z
            RS_NumTraits<T>::lowest(),   // max_x
            RS_NumTraits<T>::lowest(),   // max_y
            RS_NumTraits<T>::lowest()    // max_z
        );
    }

    /// Returns a bound representing the highest possible values
    HOST_DEVICE_INLINE_QUALIFIER static RS_Bound3D highest() {
        return RS_Bound3D(
            RS_NumTraits<T>::lowest(),   // min_x
            RS_NumTraits<T>::lowest(),   // min_y
            RS_NumTraits<T>::lowest(),   // min_z
            RS_NumTraits<T>::highest(),  // max_x
            RS_NumTraits<T>::highest(),  // max_y
            RS_NumTraits<T>::highest()   // max_z
        );
    }

    /// Print bound content
    HOST_DEVICE_INLINE_QUALIFIER void print(const char* title = "Bound3D") const {
        if constexpr (std::is_integral_v<T>) {
            printf(
                "%s min:(%d, %d, %d), max:(%d, %d, %d)\n",
                title,
                min_x,
                min_y,
                min_z,
                max_x,
                max_y,
                max_z
            );
        } else {
            printf(
                "%s min:(%e, %e, %e), max:(%e, %e, %e)\n",
                title,
                min_x,
                min_y,
                min_z,
                max_x,
                max_y,
                max_z
            );
        }
    }

    /// Returns whether the bound is valid (min <= max for all dimensions)
    HOST_DEVICE_INLINE_QUALIFIER bool isValid() const {
        return min_x <= max_x && min_y <= max_y && min_z <= max_z;
    }

    /// Shift the bound by the specified vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Bound3D<T> shift(const RS_Vec3D<T>& vector) {
        return RS_Bound3D<T>(
            min_x + vector.x,
            min_y + vector.y,
            min_z + vector.z,
            max_x + vector.x,
            max_y + vector.y,
            max_z + vector.z
        );
    }

    /// Extend the bound to include the specified point
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Bound3D<T> extend(const RS_Vec3D<U>& point) {
        return RS_Bound3D<T>(
            RS_MIN(min_x, static_cast<T>(point.x)),
            RS_MIN(min_y, static_cast<T>(point.y)),
            RS_MIN(min_z, static_cast<T>(point.z)),
            RS_MAX(max_x, static_cast<T>(point.x)),
            RS_MAX(max_y, static_cast<T>(point.y)),
            RS_MAX(max_z, static_cast<T>(point.z))
        );
    }

    /// Expand the bound by a fixed amount in all directions
    HOST_DEVICE_INLINE_QUALIFIER RS_Bound3D<T> expand(T amount) {
        return RS_Bound3D<T>(
            min_x - amount,
            min_y - amount,
            min_z - amount,
            max_x + amount,
            max_y + amount,
            max_z + amount
        );
    }

    /// Expand the bound by different amounts in different directions
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Bound3D<T> expand(const RS_Vec3D<U>& amount) {
        return RS_Bound3D<T>(
            min_x - static_cast<T>(amount.x),
            min_y - static_cast<T>(amount.y),
            min_z - static_cast<T>(amount.z),
            max_x + static_cast<T>(amount.x),
            max_y + static_cast<T>(amount.y),
            max_z + static_cast<T>(amount.z)
        );
    }

    /// Returns true if this bound equals another bound
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_Bound3D<U>& bound) const {
        return min_x == bound.min_x && min_y == bound.min_y && min_z == bound.min_z &&
               max_x == bound.max_x && max_y == bound.max_y && max_z == bound.max_z;
    }

    /// Returns true if this bound does not equal another bound
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(const RS_Bound3D<U>& bound) const {
        return !(*this == bound);
    }

    /// Returns the minimum bound as a vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> minBound() const {
        return RS_Vec3D<T>(min_x, min_y, min_z);
    }

    /// Returns the maximum bound as a vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> maxBound() const {
        return RS_Vec3D<T>(max_x, max_y, max_z);
    }

    /// Returns the center of the bound
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> center() const {
        return RS_Vec3D<T>(
            (min_x + max_x) * T(0.5), (min_y + max_y) * T(0.5), (min_z + max_z) * T(0.5)
        );
    }

    /// Returns the size of the bound (max - min)
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> size() const {
        return RS_Vec3D<T>(max_x - min_x, max_y - min_y, max_z - min_z);
    }

    /// Returns the volume of the bound
    HOST_DEVICE_INLINE_QUALIFIER T volume() const {
        RS_Vec3D<T> s = size();
        return s.x * s.y * s.z;
    }

    /// Returns whether the bound contains the specified point
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER bool isContaining(const RS_Vec3D<U>& point) const {
        return point.x >= min_x && point.x <= max_x && point.y >= min_y && point.y <= max_y &&
               point.z >= min_z && point.z <= max_z;
    }

    /// Returns whether the bound contains the specified bound
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER bool isContaining(const RS_Bound3D<U>& bound) const {
        return bound.min_x >= min_x && bound.max_x <= max_x && bound.min_y >= min_y &&
               bound.max_y <= max_y && bound.min_z >= min_z && bound.max_z <= max_z;
    }

    /// Returns whether the bound intersects with the specified bound
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER bool isIntersecting(const RS_Bound3D<U>& bound) const {
        return min_x <= bound.max_x && max_x >= bound.min_x && min_y <= bound.max_y &&
               max_y >= bound.min_y && min_z <= bound.max_z && max_z >= bound.min_z;
    }

    /// Returns the closest point on the bound to the specified point
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> closestPoint(const RS_Vec3D<U>& point) const {
        return RS_Vec3D<T>(
            RS_MIN(RS_MAX(static_cast<T>(point.x), min_x), max_x),
            RS_MIN(RS_MAX(static_cast<T>(point.y), min_y), max_y),
            RS_MIN(RS_MAX(static_cast<T>(point.z), min_z), max_z)
        );
    }

    /// Return the union of this bound and another bound
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Bound3D unionWith(const RS_Bound3D<U>& bound) const {
        return RS_Bound3D(
            RS_MIN(min_x, static_cast<T>(bound.min_x)),
            RS_MIN(min_y, static_cast<T>(bound.min_y)),
            RS_MIN(min_z, static_cast<T>(bound.min_z)),
            RS_MAX(max_x, static_cast<T>(bound.max_x)),
            RS_MAX(max_y, static_cast<T>(bound.max_y)),
            RS_MAX(max_z, static_cast<T>(bound.max_z))
        );
    }

    /// Return the intersection of this bound and another bound
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Bound3D intersectionWith(const RS_Bound3D<U>& bound) const {
        return RS_Bound3D(
            RS_MAX(min_x, static_cast<T>(bound.min_x)),
            RS_MAX(min_y, static_cast<T>(bound.min_y)),
            RS_MAX(min_z, static_cast<T>(bound.min_z)),
            RS_MIN(max_x, static_cast<T>(bound.max_x)),
            RS_MIN(max_y, static_cast<T>(bound.max_y)),
            RS_MIN(max_z, static_cast<T>(bound.max_z))
        );
    }

    /// Generic serialization
    template <class Serializer>
    void serialize(Serializer& s) {
        s("min_x", min_x);
        s("min_y", min_y);
        s("min_z", min_z);
        s("max_x", max_x);
        s("max_y", max_y);
        s("max_z", max_z);
    }
};

using RBound3f = RS_Bound3D<float>;
using RBound3df = RS_Bound3D<double>;
using RBound3rf = RS_Bound3D<RFloat>;
using RBound3i = RS_Bound3D<int>;
using RBound3ui = RS_Bound3D<unsigned int>;
using RBound3ri = RS_Bound3D<RSignedIndex>;
using RBound3rui = RS_Bound3D<RIndex>;

}  // namespace rise

REFL_AUTO(
    type(rise::RBound3f),
    field(min_x),
    field(min_y),
    field(min_z),
    field(max_x),
    field(max_y),
    field(max_z)
)
REFL_AUTO(
    type(rise::RBound3df),
    field(min_x),
    field(min_y),
    field(min_z),
    field(max_x),
    field(max_y),
    field(max_z)
)
REFL_AUTO(
    type(rise::RBound3i),
    field(min_x),
    field(min_y),
    field(min_z),
    field(max_x),
    field(max_y),
    field(max_z)
)

#endif  // RS_BOUND3D_H