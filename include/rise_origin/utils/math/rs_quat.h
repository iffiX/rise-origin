// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_QUAT3D_H
#define RS_QUAT3D_H

//
// Learning resource: refer to https://eater.net/quaternions/video/intro , The
// visualization and tutorial on Quaternions.
//

#include <cmath>
#include <refl.hpp>

#include "rise_origin/utils/math/rs_float.h"
#include "rise_origin/utils/math/rs_vec.h"

#define RS_QUAT_DBL_EPSILONx24 5.328e-15  // DBL_EPSILON*24

// The following values are calculated based on: MAX_ERROR_PERCENT 1e-4
// Anything less than this angle can be considered 0
#define RS_QUAT_DISCARD_ANGLE_RAD 1e-7

// Angles less than this get small angle approximations. To get:
// Root solve atan(t)/t-1+MAX_ERROR_PERCENT. From:
// MAX_ERROR_PERCENT = (t-atan(t))/t
#define RS_QUAT_SMALL_ANGLE_RAD 1.732e-2

// quaternion W value corresponding to a SMALL_ANGLE_RAD. To
// calculate, cos(SMALL_ANGLE_RAD*0.5).
#define RS_QUAT_SMALL_ANGLE_W 0.9999625

#define RS_QUAT_DISCARD_SMALL_MAG 1e-6

// SquareLength threshhold for when we can use square root
// optimization for acos. From SquareLength = 1-w*w. (calculate
// according to 1.0-W_THRESH_ACOS2SQRT*W_THRESH_ACOS2SQRT
#define RS_QUAT_SLTHRESH_ACOS2SQRT 2.4e-3

// quaternion properties (for reference)
// 1) To rotate a vector V, form a quaternion with w = 0; To rotate by
// Quaternion Q, do Q*V*Q.Conjugate() and trim off the w component.
// 2) To do multiple rotations: To Rotate by Q1 THEN Q2,
// Q2*Q1*V*Q1.Conjugate*Q2.Conjugate(), or make a Qtot = Q2*Q1 and do
// Qtot*V*Qtot.Conjucate()
// 3) Q1*Q1.Conjugate - Identity 4) To do a reverse rotation Q1,
// just do Q1.conjugate*V*Q1
// http://www.cprogramming.com/tutorial/3d/quaternions.html

namespace rise {

template <typename T>
class RS_Mat3D;

template <RSE_FloatType Type>
class RS_Quat3DMem;

/**
 * @brief Template of generic 3D quaternions The template parameter is assumed to be
 * either float or double depending on the desired numerical resolution, integer
 * is allowed but is not recommended.
 *
 * @Note The values are not initialized to make it compatible with plain old
 * data structure (POD).
 */
template <typename T>
class RS_Quat3D {
   public:
    using ElementType = T;

    T x;  //!< The current X value.
    T y;  //!< The current Y value.
    T z;  //!< The current Z value.
    T w;  //!< The current W value.

    // constructors
    /// Constructor.
    RS_Quat3D() = default;

    /// Constructor with specified individual values.
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}

    /// Copy constructor
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D(const RS_Quat3D<U> &q) {
        w = q.w;
        x = q.x;
        y = q.y;
        z = q.z;
    }

    /// Copy constructor for memory format
    template <RSE_FloatType Type>
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D(const RS_Quat3DMem<Type> &q) {
        q.get(x, y, z, w);
    }

    /// Returns the identity quaternion
    HOST_DEVICE_INLINE_QUALIFIER static RS_Quat3D unit() { return RS_Quat3D(0, 0, 0, 1); }

    /**
     * @brief Constructs this quaternion from an angle in radians and a unit axis.
     * @param angle An angle in radians.
     * @param axis A normalize rotation axis.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D(T angle, const RS_Vec3D<T> &axis) {
        T a = angle * T(0.5);
        T s, c;
        RS_FloatIntrinsics<T>::rs_sincos(a, &s, &c);
        w = c;
        x = axis.x * s;
        y = axis.y * s;
        z = axis.z * s;
    }

    /**
     * @brief Constructs this quaternion to represent the rotation from two vectors.
     * The vectors need not be normalized and are not modified.
     * @param from A vector representing a pre-rotation orientation.
     * @param to A vector representing a post-rotation orientation.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D(const RS_Vec3D<T> &from, const RS_Vec3D<T> &to) {
        // angle between vectors.
        // from A.B=|A||B|cos(theta)

        T lenFrom = from.length2();
        T lenTo = to.length2();
        if (lenFrom == 0 || lenTo == 0) {
            *this = RS_Quat3D(0, 0, 0, 1);
            return;
        }
        T theta = RS_FloatIntrinsics<T>::rs_acos(
            from.dot(to) / RS_FloatIntrinsics<T>::rs_sqrt(lenFrom * lenTo)
        );

        // very small angle, return no rotation
        if (theta <= 0) {
            *this = RS_Quat3D(0, 0, 0, 1);
            return;
        }
        RS_Vec3D<T> axis = from.cross(to);  // axis of rotation
        axis = axis.normalized();
        // 180 degree rotation (180 degree rot about axis ax, ay, az is
        // Quat(ax,ay,az, 0))
        if (theta > RS_PI - RS_QUAT_DISCARD_ANGLE_RAD) {
            x = axis.x;
            y = axis.y;
            z = axis.z;
            w = 0;
            return;
        }
        // otherwise for the quaternion from angle-axis.
        *this = RS_Quat3D(theta, axis);
    }

    /// Print quaternion content
    HOST_DEVICE_INLINE_QUALIFIER void print(const char *title = "Quat3D") const {
        printf("%s x:%e, y:%e, z:%e, w:%e\n", title, x, y, z, w);
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
    HOST_DEVICE_INLINE_QUALIFIER explicit operator RS_Quat3D<U>() const {
        return RS_Quat3D<U>(x, y, z, w);
    }

    /// Assign operator
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D operator=(const RS_Quat3D<U> &q) {
        w = q.w;
        x = q.x;
        y = q.y;
        z = q.z;
        return *this;
    }

    /// Assign operator for memory format
    template <RSE_FloatType Type>
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D operator=(const RS_Quat3DMem<Type> &q) {
        q.get(x, y, z, w);
        return *this;
    }

    /// Addition operator for quaternion
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D operator+(const RS_Quat3D &q) {
        return RS_Quat3D(x + q.x, y + q.y, z + q.z, w + q.w);
    }

    /// Subtraction operator for quaternion
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D operator-(const RS_Quat3D &q) {
        return RS_Quat3D(x - q.x, y - q.y, z - q.z, w - q.w);
    }

    /// Multiplication operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D operator*(T f) const {
        return RS_Quat3D(f * x, f * y, f * z, f * w);
    }

    /// Multiplication operator for constant as left hand side
    HOST_DEVICE_INLINE_QUALIFIER friend RS_Quat3D operator*(T f, const RS_Quat3D &q) {
        return q * f;
    }

    /// Multiplication operator for quaternion
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D operator*(const RS_Quat3D &q) const {
        return RS_Quat3D(
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w,
            w * q.w - x * q.x - y * q.y - z * q.z
        );
    }

    /// Addition and set operator for quaternion
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D &operator+=(const RS_Quat3D &q) {
        w += q.w;
        x += q.x;
        y += q.y;
        z += q.z;
        return *this;
    }

    /// Subtraction and set operator for quaternion
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D &operator-=(const RS_Quat3D &q) {
        w -= q.w;
        x -= q.x;
        y -= q.y;
        z -= q.z;
        return *this;
    }

    /// Is equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_Quat3D &q) const {
        return (w == q.w && x == q.x && y == q.y && z == q.z);
    }

    /// Is not equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(const RS_Quat3D &q) const {
        return (w != q.w || x != q.x || y != q.y || z != q.z);
    }

    /// Explicit casting to a vector. Throws away w and copies x, y, z directly.
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toVec() const { return RS_Vec3D<T>(x, y, z); }

    // utilities
    /// Returns the length (magnitude) of the quaternion.
    HOST_DEVICE_INLINE_QUALIFIER T length() const {
        return RS_FloatIntrinsics<T>::rs_sqrt(length2());
    }

    /// Returns the length (magnitude) squared of the quaternion.
    HOST_DEVICE_INLINE_QUALIFIER T length2() const { return (w * w + x * x + y * y + z * z); }

    /// @brief Returns the normalized quaternion.
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D normalized() const {
        T l = length();
        if (l == 0) {
            return RS_Quat3D(0, 0, 0, 1);
        } else {
            T li = T(1) / l;
            return RS_Quat3D(x * li, y * li, z * li, w * li);
        }
    }

    /// Returns the inverse of this quaternion.
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D inverse() const {
        T n = w * w + x * x + y * y + z * z;
        return n == 0 ? RS_Quat3D(0, 0, 0, 1) : RS_Quat3D(-x / n, -y / n, -z / n, w / n);
    }

    /// Returns the conjugate of this quaternion.
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D conjugate() const { return RS_Quat3D(-x, -y, -z, w); }

    /**
     * @brief Spherical linear interpolation from this to `to` by fraction `t` in [0, 1].
     * @details Takes the shortest arc and returns a unit quaternion; falls back to a
     * normalized lerp when the two orientations are nearly parallel.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D slerp(const RS_Quat3D &to, T t) const {
        T cos_theta = x * to.x + y * to.y + z * to.z + w * to.w;
        RS_Quat3D end = to;
        if (cos_theta < T(0)) {
            end = RS_Quat3D(-to.x, -to.y, -to.z, -to.w);
            cos_theta = -cos_theta;
        }
        if (cos_theta > T(1) - T(1e-6)) {
            return RS_Quat3D(
                       x + t * (end.x - x),
                       y + t * (end.y - y),
                       z + t * (end.z - z),
                       w + t * (end.w - w)
            )
                .normalized();
        }
        T theta = RS_FloatIntrinsics<T>::rs_acos(cos_theta);
        T sin_theta = RS_FloatIntrinsics<T>::rs_sqrt(T(1) - cos_theta * cos_theta);
        T a = RS_FloatIntrinsics<T>::rs_sin((T(1) - t) * theta) / sin_theta;
        T b = RS_FloatIntrinsics<T>::rs_sin(t * theta) / sin_theta;
        return RS_Quat3D(
            a * x + b * end.x, a * y + b * end.y, a * z + b * end.z, a * w + b * end.w
        );
    }

    // angle and/or axis calculations
    /// Returns the angular rotation of this quaternion in radians.
    HOST_DEVICE_INLINE_QUALIFIER T angle() const {
        return T(2) * RS_FloatIntrinsics<T>::rs_acos(w > 1 ? 1 : w);
    }

    /// Returns the angular rotation of this quaternion in degrees.
    HOST_DEVICE_INLINE_QUALIFIER T angleDegrees() const { return angle() * T(57.29577951308232); }

    /// Returns true if the angular rotation of this quaternion is likely to
    /// be considered negligible.
    HOST_DEVICE_INLINE_QUALIFIER bool isNegligibleAngle() const {
        return T(2) * RS_FloatIntrinsics<T>::rs_acos(w) < RS_QUAT_DISCARD_ANGLE_RAD;
    }

    /// Returns true if the angular rotation of this quaternion is small
    /// enough to be a good candidate for small angle approximations.
    HOST_DEVICE_INLINE_QUALIFIER bool isSmallAngle() const { return w > RS_QUAT_SMALL_ANGLE_W; }

    /// Returns the normalized axis of rotation of this quaternion
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> axis() const {
        T square_length = T(1) - w * w;  // because x*x + y*y + z*z + w*w = 1.0,
                                         // but more susceptible to w noise
        if (square_length <= RS_QUAT_DISCARD_SMALL_MAG) {
            return RS_Vec3D<T>(1, 0, 0);
        } else {
            return RS_Vec3D<T>(x, y, z) / RS_FloatIntrinsics<T>::rs_sqrt(square_length);
        }
    }

    /// @brief Returns the un-normalized axis of rotation of this quaternion
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> axisUnNormalized() const {
        return RS_Vec3D<T>(x, y, z);
    }

    /**
     * @brief Returns the angle and a normalize axis that represent this quaternion's rotation.
     * @param angle (Output) The rotation angle in radians.
     * @param axis (Output) The rotation axis in normalized vector form.
     */
    HOST_DEVICE_INLINE_QUALIFIER void angleAxis(T &angle, RS_Vec3D<T> &axis) const {
        angleAxisUnNormalized(angle, axis);
        axis = axis.normalized();
    }

    /**
     * @brief Returns the angle and an un-normalized axis that represent this quaternion's rotation.
     * @param angle (Output) The rotation angle in radians.
     * @param axis (Output) The rotation axis in un-normalized vector form.
     */
    HOST_DEVICE_INLINE_QUALIFIER void angleAxisUnNormalized(T &angle, RS_Vec3D<T> &axis) const {
        if (w >= T(1)) {
            angle = 0;
            axis = RS_Vec3D<T>(1, 0, 0);
            return;
        }
        angle = T(2) * RS_FloatIntrinsics<T>::rs_acos(w > T(1) ? T(1) : w);
        axis = RS_Vec3D<T>(x, y, z);
    }

    /**
     * @brief Returns the rotation matrix corresponding to the quaternion
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D<T> toRotationMatrix() const {
        return RS_Mat3D<T>(
            1 - 2 * y * y - 2 * z * z,  //
            2 * x * y - 2 * w * z,      //
            2 * x * z + 2 * w * y,      //
            2 * x * y + 2 * w * z,      //
            1 - 2 * x * x - 2 * z * z,  //
            2 * y * z - 2 * w * x,      //
            2 * x * z - 2 * w * y,      //
            2 * y * z + 2 * w * x,      //
            1 - 2 * x * x - 2 * y * y
        );
    }

    /**
     * @brief Returns a rotation vector representing this quaternion rotation.
     * @note See
     * http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toRotationVector() const {
        T ratio;
        if (w < T(1) - RS_QUAT_DISCARD_ANGLE_RAD && w > T(-1) + RS_QUAT_DISCARD_ANGLE_RAD) {
            // because x*x + y*y + z*z + w*w = 1.0,
            // but more susceptible to w noise
            T square_length = T(1) - w * w;
            ratio = T(2) * RS_FloatIntrinsics<T>::rs_acos(w) /
                    RS_FloatIntrinsics<T>::rs_sqrt(square_length);
            return RS_Vec3D<T>(x * ratio, y * ratio, z * ratio);
        } else
            return {0, 0, 0};
    }

    /**
     * @brief Constructs a quaternion from a rotation matrix and sets the components of *this.
     * @param mat_in The input rotation matrix.
     * @return A reference to *this for chaining.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D &fromRotationMatrix(const RS_Mat3D<T> &mat_in) {
        T trace = mat_in(0, 0) + mat_in(1, 1) + mat_in(2, 2);  // Sum of diagonal elements

        if (trace > T(0)) {
            T s = RS_FloatIntrinsics<T>::rs_sqrt(trace + T(1)) * T(2);  // 4 * w
            w = T(0.25) * s;
            x = (mat_in(2, 1) - mat_in(1, 2)) / s;
            y = (mat_in(0, 2) - mat_in(2, 0)) / s;
            z = (mat_in(1, 0) - mat_in(0, 1)) / s;
        } else if (mat_in(0, 0) > mat_in(1, 1) && mat_in(0, 0) > mat_in(2, 2)) {
            T s =
                RS_FloatIntrinsics<T>::rs_sqrt(T(1) + mat_in(0, 0) - mat_in(1, 1) - mat_in(2, 2)) *
                T(2);  // 4 * x
            w = (mat_in(2, 1) - mat_in(1, 2)) / s;
            x = T(0.25) * s;
            y = (mat_in(0, 1) + mat_in(1, 0)) / s;
            z = (mat_in(0, 2) + mat_in(2, 0)) / s;
        } else if (mat_in(1, 1) > mat_in(2, 2)) {
            T s =
                RS_FloatIntrinsics<T>::rs_sqrt(T(1) + mat_in(1, 1) - mat_in(0, 0) - mat_in(2, 2)) *
                T(2);  // 4 * y
            w = (mat_in(0, 2) - mat_in(2, 0)) / s;
            x = (mat_in(0, 1) + mat_in(1, 0)) / s;
            y = T(0.25) * s;
            z = (mat_in(1, 2) + mat_in(2, 1)) / s;
        } else {
            T s =
                RS_FloatIntrinsics<T>::rs_sqrt(T(1) + mat_in(2, 2) - mat_in(0, 0) - mat_in(1, 1)) *
                T(2);  // 4 * z
            w = (mat_in(1, 0) - mat_in(0, 1)) / s;
            x = (mat_in(0, 2) + mat_in(2, 0)) / s;
            y = (mat_in(1, 2) + mat_in(2, 1)) / s;
            z = T(0.25) * s;
        }

        return *this;
    }

    /**
     * @brief Overwrites this quaternion with values from the specified rotation vector.
     * @note See http://physicsforgames.blogspot.com/2010/02/quaternions.html.
     * @param vec_in A rotation vector to calculate this quaternion from.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D fromRotationVector(const RS_Vec3D<T> &vec_in) {
        RS_Vec3D<T> theta = vec_in / T(2);
        T s, theta_mag2 = theta.length2();
        T theta_mag = RS_FloatIntrinsics<T>::rs_sqrt(theta_mag2);
        if (theta_mag < RS_QUAT_DISCARD_SMALL_MAG) {
            w = 1;
            x = 0;
            y = 0;
            z = 0;
            return *this;
        } else {
            RS_FloatIntrinsics<T>::rs_sincos(theta_mag, &s, &w);
            s /= theta_mag;
            x = theta.x * s;
            y = theta.y * s;
            z = theta.z * s;
            return *this;
        }
    }

    /**
     * @brief Overwrites this quaternion with the calculated rotation that would
     *        transform the specified rotate_from vector to point in the positive
     *        X direction.
     * @param rotate_from An arbitrary direction vector. Does not need to be normalized.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D fromAngleToPosX(const RS_Vec3D<T> &rotate_from) {
        RS_Vec3D<T> rot_from_norm = rotate_from.normalized();

        // because rot_from_norm is normalized, 1,0,0 is
        // normalized, and A.B = |A||B|cos(theta) = cos(theta)
        T theta = RS_FloatIntrinsics<T>::rs_acos(rot_from_norm.x);
        if (theta < RS_QUAT_DISCARD_ANGLE_RAD) {
            // 0-degree rotation
            x = 0;
            y = 0;
            z = 0;
            w = 1;
        } else if (theta > RS_PI - RS_QUAT_DISCARD_ANGLE_RAD) {
            // 180-degree rotation (about y-axis, since the vector must be
            // pointing in -x direction
            x = 0;
            y = 1;
            z = 0;
            w = 0;
        } else {
            T axis_mag = RS_FloatIntrinsics<T>::rs_sqrt(
                rot_from_norm.z * rot_from_norm.z + rot_from_norm.y * rot_from_norm.y
            );
            if (axis_mag < RS_QUAT_DISCARD_SMALL_MAG) {
                x = 0;
                y = 0;
                z = 0;
                w = 1;
            } else {
                T axis_mag_inv = T(1) / axis_mag;
                // Here theta is the angle, axis is rot_from_norm.cross(RS_Vec3D(1,0,0))
                // = RS_Vec3D(0, rot_from_norm.z/axis_mag_inv,
                // -rot_from_norm.y/axis_mag_inv), which is still normalized. (super
                // rolled together)
                T a = T(0.5) * theta;
                T s;
                RS_FloatIntrinsics<T>::rs_sincos(a, &s, &w);
                x = 0;
                y = rot_from_norm.z * axis_mag_inv * s;
                z = -rot_from_norm.y * axis_mag_inv * s;  // angle axis function, reduced
            }
        }
        return *this;
    }

    /**
     * @brief Rotates the specified vector by this quaternion.
     * @param v The vector to transform.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> rotateVec3D(const RS_Vec3D<T> &v) const {
        T fx = v.x, fy = v.y, fz = v.z;
        T tw = fx * x + fy * y + fz * z;
        T tx = fx * w - fy * z + fz * y;
        T ty = fx * z + fy * w - fz * x;
        T tz = -fx * y + fy * x + fz * w;
        return RS_Vec3D<T>(
            w * tx + x * tw + y * tz - z * ty,
            w * ty - x * tz + y * tw + z * tx,
            w * tz + x * ty - y * tx + z * tw
        );
    }

    /**
     * @brief Rotates the specified vector by the inverse of this quaternion.
     * @param v The vector to transform.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> rotateVec3DInv(const RS_Vec3D<T> &v) const {
        T fx = v.x, fy = v.y, fz = v.z;
        T tw = x * fx + y * fy + z * fz;
        T tx = w * fx - y * fz + z * fy;
        T ty = w * fy + x * fz - z * fx;
        T tz = w * fz - x * fy + y * fx;
        return RS_Vec3D<T>(
            tw * x + tx * w + ty * z - tz * y,
            tw * y - tx * z + ty * w + tz * x,
            tw * z + tx * y - ty * x + tz * w
        );
    }

    /**
     * @brief Returns orthonormal basis vector X (1, 0, 0) rotated by this quaternion.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toOrthoAxisX() const {
        // (1−2(y^2+z^2),2(xy+zw),2(xz−yw))
        return RS_Vec3D<T>(
            T(1) - T(2) * (y * y + z * z), T(2) * (x * y + w * z), T(2) * (x * z - w * y)
        );
    }

    /**
     * @brief Returns orthonormal basis vector Y (0, 1, 0) rotated by this quaternion.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toOrthoAxisY() const {
        // (2(xy−zw),1−2(x^2+z^2),2(yz+wx))
        return RS_Vec3D<T>(
            T(2) * (x * y - w * z), T(1) - T(2) * (x * x + z * z), T(2) * (y * z + w * x)
        );
    }

    /**
     * @brief Returns orthonormal basis vector Z (0, 0, 1) rotated by this quaternion.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toOrthoAxisZ() const {
        // (2(xz+yw),2(yz-xw),1−2(x^2+y^2))
        return RS_Vec3D<T>(
            T(2) * (x * z + y * w), T(2) * (y * z - x * w), T(1) - T(2) * (x * x + y * y)
        );
    }

    /**
     * @brief Serializes this quaternion.
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

// Helper struct to determine data array based on Type
template <RSE_FloatType Type>
struct RS_Quat3DMemDataType;

template <>
struct RS_Quat3DMemDataType<RSE_FLOAT_16> {
    using DataType = unsigned short[4];
    using ElementType = unsigned short;
};

template <>
struct RS_Quat3DMemDataType<RSE_FLOAT_32> {
    using DataType = float[4];
    using ElementType = float;
};

template <>
struct RS_Quat3DMemDataType<RSE_FLOAT_64> {
    using DataType = double[4];
    using ElementType = double;
};

/**
 * @brief Template of 3D quaternions stored in memory.
 * It is always normalized on construction or assignment.
 */
template <RSE_FloatType Type>
class RS_Quat3DMem {
   public:
    using DataType = typename RS_Quat3DMemDataType<Type>::DataType;
    using ElementType = typename RS_Quat3DMemDataType<Type>::ElementType;
    DataType data;  //!< Internal storage.

    //! Default constructor
    RS_Quat3DMem() = default;

    HOST_DEVICE_INLINE_QUALIFIER static RS_Quat3DMem
    fromData(ElementType data_0, ElementType data_1, ElementType data_2, ElementType data_3) {
        RS_Quat3DMem quat;
        quat.data[0] = data_0;
        quat.data[1] = data_1;
        quat.data[2] = data_2;
        quat.data[3] = data_3;
        return quat;
    }

    //! Returns a unit quaternion
    HOST_DEVICE_INLINE_QUALIFIER static RS_Quat3DMem unit() {
        RS_Quat3DMem quat;
        quat.set(0.0f, 0.0f, 0.0f, 1.0f);
        return quat;
    }

    //! Constructor taking x, y, z, w as input
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3DMem(T x, T y, T z, T w) {
        set(x, y, z, w);
    }

    //! Constructor from RS_Quat3D<T>
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_Quat3DMem(const RS_Quat3D<T> &q) {
        set(q.x, q.y, q.z, q.w);
    }

    //! Assignment operator from RS_Quat3D<T>
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3DMem &operator=(const RS_Quat3D<T> &q) {
        set(q.x, q.y, q.z, q.w);
        return *this;
    }

    //! Convert to RS_Quat3D<T>
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER RS_Quat3D<T> toQuat3D() const {
        RS_Quat3D<T> q;
        get(q.x, q.y, q.z, q.w);
        return q;
    }

    //! Helper function to set data from x, y, z, w
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER void set(const T &x, const T &y, const T &z, const T &w) {
        T length = RS_FloatIntrinsics<T>::rs_sqrt(w * w + x * x + y * y + z * z);

        T nx = static_cast<T>(0);
        T ny = static_cast<T>(0);
        T nz = static_cast<T>(0);
        T nw = static_cast<T>(1);
        if (length != 0) {
            nx = x / length;
            ny = y / length;
            nz = z / length;
            nw = w / length;
        }

        if constexpr (Type == RSE_FLOAT_16) {
            data[0] = RS_FloatConvert::floatToHalf(static_cast<float>(nx));
            data[1] = RS_FloatConvert::floatToHalf(static_cast<float>(ny));
            data[2] = RS_FloatConvert::floatToHalf(static_cast<float>(nz));
            data[3] = RS_FloatConvert::floatToHalf(static_cast<float>(nw));
        } else {
            data[0] = static_cast<ElementType>(nx);
            data[1] = static_cast<ElementType>(ny);
            data[2] = static_cast<ElementType>(nz);
            data[3] = static_cast<ElementType>(nw);
        }
    }

    //! Helper function to get data into RS_Quat3D<T>
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER void get(T &x, T &y, T &z, T &w) const {
        if constexpr (Type == RSE_FLOAT_16) {
            x = static_cast<T>(RS_FloatConvert::halfToFloat(data[0]));
            y = static_cast<T>(RS_FloatConvert::halfToFloat(data[1]));
            z = static_cast<T>(RS_FloatConvert::halfToFloat(data[2]));
            w = static_cast<T>(RS_FloatConvert::halfToFloat(data[3]));
        } else {
            x = static_cast<T>(data[0]);
            y = static_cast<T>(data[1]);
            z = static_cast<T>(data[2]);
            w = static_cast<T>(data[3]);
        }
    }
};

using RQuat3f = RS_Quat3D<float>;
using RQuat3df = RS_Quat3D<double>;
using RQuat3rf = RS_Quat3D<RFloat>;

}  // namespace rise

REFL_AUTO(type(rise::RQuat3f), field(x), field(y), field(z), field(w))
REFL_AUTO(type(rise::RQuat3df), field(x), field(y), field(z), field(w))

REFL_AUTO(type(rise::RS_Quat3DMem<rise::RSE_FLOAT_16>), field(data))
REFL_AUTO(type(rise::RS_Quat3DMem<rise::RSE_FLOAT_32>), field(data))
REFL_AUTO(type(rise::RS_Quat3DMem<rise::RSE_FLOAT_64>), field(data))

#endif  // RS_QUAT3D_H
