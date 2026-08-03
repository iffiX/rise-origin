// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_MAT3D_H
#define RS_MAT3D_H

#include <cassert>
#include <cmath>
#include <refl.hpp>

#include "rise_origin/utils/math/rs_float.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

template <typename T>
struct RS_Mat3DRowAccessor {
    T *row;
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_Mat3DRowAccessor(T *row) : row(row){};
    HOST_DEVICE_INLINE_QUALIFIER T &operator[](size_t col) {
        assert(col < 3);
        return *(row + col);
    }
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toVec() {
        return RS_Vec3D<T>(*row, *(row + 1), *(row + 2));
    }
};

template <typename T>
struct RS_ConstMat3DRowAccessor {
    const T *row;
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_ConstMat3DRowAccessor(const T *row) : row(row){};
    HOST_DEVICE_INLINE_QUALIFIER const T &operator[](size_t col) {
        assert(col < 3);
        return *(row + col);
    }
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toVec() {
        return RS_Vec3D<T>(*row, *(row + 1), *(row + 2));
    }
};

template <typename T>
struct RS_Mat3DColAccessor {
    T *col;
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_Mat3DColAccessor(T *col) : col(col){};
    HOST_DEVICE_INLINE_QUALIFIER T &operator[](size_t row) {
        assert(row < 3);
        return *(col + row * 3);
    }
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toVec() {
        return RS_Vec3D<T>(*col, *(col + 3), *(col + 6));
    }
};

template <typename T>
struct RS_ConstMat3DColAccessor {
    const T *col;
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_ConstMat3DColAccessor(const T *col) : col(col){};
    HOST_DEVICE_INLINE_QUALIFIER const T &operator[](size_t row) {
        assert(row < 3);
        return *(col + row * 3);
    }
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> toVec() {
        return RS_Vec3D<T>(*col, *(col + 3), *(col + 6));
    }
};

template <RSE_FloatType Type>
class RS_Mat3DMem;

/**
 * @brief Template of generic 3D matrices The template parameter is assumed to be
 * either float or double depending on the desired numerical resolution, integer
 * is allowed but is not recommended.
 *
 * @Note The values are not initialized to make it compatible with plain old
 * data structure (POD).
 */
template <typename T>
class RS_Mat3D {
   public:
    using ElementType = T;

    T data[9];

   public:
    /// Constructor which initializes values to 0
    RS_Mat3D() = default;

    /// Constructor with specified individual values
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D(
        T m_00,
        T m_01,
        T m_02,  //
        T m_10,
        T m_11,
        T m_12,  //
        T m_20,
        T m_21,
        T m_22
    ) {
        data[0] = m_00;
        data[1] = m_01;
        data[2] = m_02;
        data[3] = m_10;
        data[4] = m_11;
        data[5] = m_12;
        data[6] = m_20;
        data[7] = m_21;
        data[8] = m_22;
    }

    /// Constructor from 3 row vectors
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER
    RS_Mat3D(const RS_Vec3D<U> &v_0, const RS_Vec3D<U> &v_1, const RS_Vec3D<U> &v_2) {
        data[0] = v_0.x;
        data[1] = v_0.y;
        data[2] = v_0.z;
        data[3] = v_1.x;
        data[4] = v_1.y;
        data[5] = v_1.z;
        data[6] = v_2.x;
        data[7] = v_2.y;
        data[8] = v_2.z;
    }

    /// Create a diagonal matrix using 3 elements
    HOST_DEVICE_INLINE_QUALIFIER static RS_Mat3D diagonal(T m_00, T m_11, T m_12) {
        return RS_Mat3D(m_00, 0, 0, 0, m_11, 0, 0, 0, m_12);
    }

    /// Create a zero matrix
    HOST_DEVICE_INLINE_QUALIFIER static RS_Mat3D zero() {
        return RS_Mat3D(0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    /// Create an identity matrix
    HOST_DEVICE_INLINE_QUALIFIER static RS_Mat3D eye() {
        return RS_Mat3D(1, 0, 0, 0, 1, 0, 0, 0, 1);
    }

    /// Create a matrix filled with specified value
    HOST_DEVICE_INLINE_QUALIFIER static RS_Mat3D full(T value) {
        return RS_Mat3D(value, value, value, value, value, value, value, value, value);
    }

    /// Copy constructor
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D(const RS_Mat3D &m) {
        data[0] = m.data[0];
        data[1] = m.data[1];
        data[2] = m.data[2];
        data[3] = m.data[3];
        data[4] = m.data[4];
        data[5] = m.data[5];
        data[6] = m.data[6];
        data[7] = m.data[7];
        data[8] = m.data[8];
    }

    template <RSE_FloatType Type>
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D(const RS_Mat3DMem<Type> &m) {
        m.get(*this);
    }

    template <RSE_FloatType Type>
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D(
        const RS_Vec3DMem<Type> &v_0,
        const RS_Vec3DMem<Type> &v_1,
        const RS_Vec3DMem<Type> &v_2
    ) {
        v_0.get(data[0], data[1], data[2]);
        v_1.get(data[3], data[4], data[5]);
        v_2.get(data[6], data[7], data[8]);
    }

    /// Print matrix content
    HOST_DEVICE_INLINE_QUALIFIER void print(const char *title = "Mat3D") const {
        printf(
            "%s m_00:%e, m_01:%e, m_02:%e,"
            "m_10:%e, m_11:%e, m_12:%e,"
            "m_20:%e, m_21:%e, m_22:%e\n",
            title,  //
            data[0],
            data[1],
            data[2],  //
            data[3],
            data[4],
            data[5],  //
            data[6],
            data[7],
            data[8]
        );
    }

    /// Returns true if all values are valid numbers
    HOST_DEVICE_INLINE_QUALIFIER bool isValid() const {
        using RS_FI = RS_FloatIntrinsics<T>;
        return !RS_FI::rs_isnan(data[0]) && RS_FI::rs_isfinite(data[0]) &&  //
               !RS_FI::rs_isnan(data[1]) && RS_FI::rs_isfinite(data[1]) &&  //
               !RS_FI::rs_isnan(data[2]) && RS_FI::rs_isfinite(data[2]) &&  //
               !RS_FI::rs_isnan(data[3]) && RS_FI::rs_isfinite(data[3]) &&  //
               !RS_FI::rs_isnan(data[4]) && RS_FI::rs_isfinite(data[4]) &&  //
               !RS_FI::rs_isnan(data[5]) && RS_FI::rs_isfinite(data[5]) &&  //
               !RS_FI::rs_isnan(data[6]) && RS_FI::rs_isfinite(data[6]) &&  //
               !RS_FI::rs_isnan(data[7]) && RS_FI::rs_isfinite(data[7]) &&  //
               !RS_FI::rs_isnan(data[8]) && RS_FI::rs_isfinite(data[8]);
    }

    /// Cast operator for type conversion
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER explicit operator RS_Mat3D<U>() const {
        return RS_Mat3D<U>(
            data[0],
            data[1],
            data[2],  //
            data[3],
            data[4],
            data[5],  //
            data[6],
            data[7],
            data[8]
        );
    }

    /// Assign operator
    template <typename U>
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D &operator=(const RS_Mat3D<U> &m) {
        data[0] = m.data[0];
        data[1] = m.data[1];
        data[2] = m.data[2];
        data[3] = m.data[3];
        data[4] = m.data[4];
        data[5] = m.data[5];
        data[6] = m.data[6];
        data[7] = m.data[7];
        data[8] = m.data[8];
        return *this;
    }

    /// Addition operator for matrix
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D operator+(const RS_Mat3D &m) const {
        return RS_Mat3D(
            data[0] + m.data[0],
            data[1] + m.data[1],
            data[2] + m.data[2],  //
            data[3] + m.data[3],
            data[4] + m.data[4],
            data[5] + m.data[5],  //
            data[6] + m.data[6],
            data[7] + m.data[7],
            data[8] + m.data[8]
        );
    }

    /// Subtraction operator for matrix
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D operator-(const RS_Mat3D &m) const {
        return RS_Mat3D(
            data[0] - m.data[0],
            data[1] - m.data[1],
            data[2] - m.data[2],  //
            data[3] - m.data[3],
            data[4] - m.data[4],
            data[5] - m.data[5],  //
            data[6] - m.data[6],
            data[7] - m.data[7],
            data[8] - m.data[8]
        );
    }

    /// Unary negation operator
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D operator-() const {
        return RS_Mat3D(
            -data[0],
            -data[1],
            -data[2],  //
            -data[3],
            -data[4],
            -data[5],  //
            -data[6],
            -data[7],
            -data[8]
        );
    }

    /// Multiplication operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D operator*(T f) const {
        return RS_Mat3D(
            data[0] * f,
            data[1] * f,
            data[2] * f,  //
            data[3] * f,
            data[4] * f,
            data[5] * f,  //
            data[6] * f,
            data[7] * f,
            data[8] * f
        );
    }

    /// Multiplication operator for constant as left hand side
    HOST_DEVICE_INLINE_QUALIFIER friend RS_Mat3D operator*(T f, const RS_Mat3D &m) {
        return RS_Mat3D(
            m.data[0] * f,
            m.data[1] * f,
            m.data[2] * f,  //
            m.data[3] * f,
            m.data[4] * f,
            m.data[5] * f,  //
            m.data[6] * f,
            m.data[7] * f,
            m.data[8] * f
        );
    }

    /// Multiplication operator for matrix multiples vector,
    /// treats vector as 3x1 column vector
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> operator*(const RS_Vec3D<T> &v) const {
        return RS_Vec3D<T>(
            data[0] * v.x + data[1] * v.y + data[2] * v.z,
            data[3] * v.x + data[4] * v.y + data[5] * v.z,
            data[6] * v.x + data[7] * v.y + data[8] * v.z
        );
    }

    /// Multiplication operator for vector multiplies matrix,
    /// treats vector as 1x3 row vector
    HOST_DEVICE_INLINE_QUALIFIER friend RS_Vec3D<T>
    operator*(const RS_Vec3D<T> &v, const RS_Mat3D &m) {
        return RS_Vec3D<T>(
            m.data[0] * v.x + m.data[3] * v.y + m.data[6] * v.z,
            m.data[1] * v.x + m.data[4] * v.y + m.data[7] * v.z,
            m.data[2] * v.x + m.data[5] * v.y + m.data[8] * v.z
        );
    }

    /// Multiplication operator for matrix
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D operator*(const RS_Mat3D &m) const {
        return RS_Mat3D(
            data[0] * m.data[0] + data[1] * m.data[3] + data[2] * m.data[6],
            data[0] * m.data[1] + data[1] * m.data[4] + data[2] * m.data[7],
            data[0] * m.data[2] + data[1] * m.data[5] + data[2] * m.data[8],
            data[3] * m.data[0] + data[4] * m.data[3] + data[5] * m.data[6],
            data[3] * m.data[1] + data[4] * m.data[4] + data[5] * m.data[7],
            data[3] * m.data[2] + data[4] * m.data[5] + data[5] * m.data[8],
            data[6] * m.data[0] + data[7] * m.data[3] + data[8] * m.data[6],
            data[6] * m.data[1] + data[7] * m.data[4] + data[8] * m.data[7],
            data[6] * m.data[2] + data[7] * m.data[5] + data[8] * m.data[8]
        );
    }

    /// Division operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D operator/(T f) const {
        return RS_Mat3D(
            data[0] / f,
            data[1] / f,
            data[2] / f,  //
            data[3] / f,
            data[4] / f,
            data[5] / f,  //
            data[6] / f,
            data[7] / f,
            data[8] / f
        );
    }

    /// Addition and set operator for matrix
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D &operator+=(const RS_Mat3D &m) {
        data[0] += m.data[0];
        data[1] += m.data[1];
        data[2] += m.data[2];
        data[3] += m.data[3];
        data[4] += m.data[4];
        data[5] += m.data[5];
        data[6] += m.data[6];
        data[7] += m.data[7];
        data[8] += m.data[8];
        return *this;
    }

    /// Subtraction and set operator for matrix
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D &operator-=(const RS_Mat3D &m) {
        data[0] -= m.data[0];
        data[1] -= m.data[1];
        data[2] -= m.data[2];
        data[3] -= m.data[3];
        data[4] -= m.data[4];
        data[5] -= m.data[5];
        data[6] -= m.data[6];
        data[7] -= m.data[7];
        data[8] -= m.data[8];
        return *this;
    }

    // Note: multiplication and set operator is only provided
    // for constant and not other types to avoid complication
    /// Multiplication and set operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D &operator*=(T f) {
        data[0] *= f;
        data[1] *= f;
        data[2] *= f;
        data[3] *= f;
        data[4] *= f;
        data[5] *= f;
        data[6] *= f;
        data[7] *= f;
        data[8] *= f;
        return *this;
    }

    /// Division and set operator for constant
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D &operator/=(T f) {
        data[0] /= f;
        data[1] /= f;
        data[2] /= f;
        data[3] /= f;
        data[4] /= f;
        data[5] /= f;
        data[6] /= f;
        data[7] /= f;
        data[8] /= f;
        return *this;
    }

    /// Is equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_Mat3D &m) const {
        return data[0] == m.data[0] && data[1] == m.data[1] && data[2] == m.data[2] &&
               data[3] == m.data[3] && data[4] == m.data[4] && data[5] == m.data[5] &&
               data[6] == m.data[6] && data[7] == m.data[7] && data[8] == m.data[8];
    }

    /// Is not equal operator
    HOST_DEVICE_INLINE_QUALIFIER bool operator!=(const RS_Mat3D &m) const { return !(*this == m); }

    /// Row index operator
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3DRowAccessor<T> operator[](int row) {
        assert(row < 3);
        return RS_Mat3DRowAccessor<T>(data + 3 * row);
    }

    /// Const row index operator
    HOST_DEVICE_INLINE_QUALIFIER RS_ConstMat3DRowAccessor<T> operator[](int row) const {
        assert(row < 3);
        return RS_ConstMat3DRowAccessor<T>(data + 3 * row);
    }

    /// Row accessor
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3DRowAccessor<T> row(int row) {
        assert(row < 3);
        return RS_Mat3DRowAccessor<T>(data + 3 * row);
    }

    /// Column accessor
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3DColAccessor<T> col(int col) {
        assert(col < 3);
        return RS_Mat3DColAccessor<T>(data + col);
    }

    /// Const row accessor
    HOST_DEVICE_INLINE_QUALIFIER RS_ConstMat3DRowAccessor<T> row(int row) const {
        assert(row < 3);
        return RS_ConstMat3DRowAccessor<T>(data + 3 * row);
    }

    /// Const column accessor
    HOST_DEVICE_INLINE_QUALIFIER RS_ConstMat3DColAccessor<T> col(int col) const {
        assert(col < 3);
        return RS_ConstMat3DColAccessor<T>(data + col);
    }

    /// Element accessor
    HOST_DEVICE_INLINE_QUALIFIER T operator()(int row, int col) const {
        return data[row * 3 + col];
    }

    /// Returns the transpose of the 3x3 matrix
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D transpose() const {
        return RS_Mat3D(
            data[0],
            data[3],
            data[6],  //
            data[1],
            data[4],
            data[7],  //
            data[2],
            data[5],
            data[8]
        );
    }

    /// Returns the determinant of the 3x3 matrix
    HOST_DEVICE_INLINE_QUALIFIER T determinant() const {
        T determinant = 0;
        determinant += data[0] * (data[4] * data[8] - data[5] * data[7]);
        determinant -= data[1] * (data[3] * data[8] - data[5] * data[6]);
        determinant += data[2] * (data[3] * data[7] - data[4] * data[6]);
        return determinant;
    }

    /// Returns the adjoint of the 3x3 matrix
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D adjoint() const {
        return RS_Mat3D(
            (data[4] * data[8] - data[5] * data[7]),
            -(data[1] * data[8] - data[2] * data[7]),
            (data[1] * data[5] - data[2] * data[4]),
            -(data[3] * data[8] - data[5] * data[6]),
            (data[0] * data[8] - data[2] * data[6]),
            -(data[0] * data[5] - data[2] * data[3]),
            (data[3] * data[7] - data[4] * data[6]),
            -(data[0] * data[7] - data[1] * data[6]),
            (data[0] * data[4] - data[1] * data[3])
        );
    }

    /**
     * @brief Returns the inverse of the 3x3 matrix. If no inverse exists,
     * the matrix is invalid (det = 0).
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D inverse() const {
        T det = determinant();
        return adjoint() / det;
    }

    /**
     * @brief Compute the eigenvalues of a symmetric 3x3 matrix.
     *
     * @note The method is based on the closed form analysis of the
     * characteristic polynomial. See:
     * https://en.wikipedia.org/wiki/Characteristic_polynomial
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> symmetricEigenValues() const {
        using RS_FI = RS_FloatIntrinsics<T>;

        T m_00 = data[0], m_01 = data[1], m_02 = data[2];
        T m_11 = data[4], m_12 = data[5];
        T m_22 = data[8];

        T trace = m_00 + m_11 + m_22;
        T q = trace * T(1.0 / 3.0);

        // Check for diagonal matrix (or nearly so)
        T p1 = m_01 * m_01 + m_02 * m_02 + m_12 * m_12;
        if (RS_ABS(p1) < RS_EPS) {
            return RS_Vec3D<T>(m_00, m_11, m_22);
        }

        T p2 =
            (m_00 - q) * (m_00 - q) + (m_11 - q) * (m_11 - q) + (m_22 - q) * (m_22 - q) + T(2) * p1;
        T p = RS_FI::rs_sqrt(p2 * T(1.0 / 6.0));

        // Form the matrix B = (1/p)*(A - q*I)
        T B_00 = (m_00 - q) / p;
        T B_11 = (m_11 - q) / p;
        T B_22 = (m_22 - q) / p;
        T B_01 = m_01 / p;
        T B_02 = m_02 / p;
        T B_12 = m_12 / p;

        // Compute r = det(B) / 2
        T r = (B_00 * B_11 * B_22 + T(2) * B_01 * B_02 * B_12 - B_00 * B_12 * B_12 -
               B_11 * B_02 * B_02 - B_22 * B_01 * B_01) *
              T(0.5);

        // Clamp r to the range [-1,1]
        if (r < -T(1)) r = -T(1);
        if (r > T(1)) r = T(1);

        // Compute angle phi
        T phi = RS_FI::rs_acos(r) * T(1.0 / 3.0);

        T eig1 = q + T(2) * p * RS_FI::rs_cos(phi);
        T eig3 = q + T(2) * p * RS_FI::rs_cos(phi + (RS_2PI * T(1.0 / 3.0)));
        T eig2 = T(3) * q - eig1 - eig3;

        return RS_Vec3D<T>(eig1, eig2, eig3);
    }

    /**
     * @brief Compute the eigenvector of the symmetric 3x3 matrix for a given eigenvalue.
     *
     * @note The returned vector must be checked with isValid()
     * @param eigen_value The eigenvalue to compute the eigenvector for.
     * @return The normalized eigenvector. If the eigenvector is undefined, the returned
     * vector is invalid.
     */
    HOST_DEVICE_INLINE_QUALIFIER RS_Vec3D<T> symmetricEigenVector(T eigen_value) const {
        using RS_FI = RS_FloatIntrinsics<T>;

        T m_00 = data[0], m_01 = data[1], m_02 = data[2];
        T m_11 = data[4], m_12 = data[5];
        T m_22 = data[8];

        // Use the first two rows of matrix (A - \lambda * I) to form two vectors:
        RS_Vec3D<T> r0(m_00 - eigen_value, m_01, m_02);
        RS_Vec3D<T> r1(m_01, m_11 - eigen_value, m_12);

        RS_Vec3D<T> e_vec = r0.cross(r1);

        T norm = e_vec.length();

        if (norm < RS_EPS) {
            // Try with rows 0 and 2 if the first cross product is nearly zero.
            RS_Vec3D<T> r2(m_02, m_12, m_22 - eigen_value);
            e_vec = r0.cross(r2);
            norm = e_vec.length();
            return e_vec / norm;
        }

        return e_vec / norm;
    }

    /**
     * @brief Generic serialization
     *
     * @param s The serializer
     */
    template <typename Serializer>
    void serialize(Serializer &s) {
        s("0", data[0]);
        s("1", data[1]);
        s("2", data[2]);
        s("3", data[3]);
        s("4", data[4]);
        s("5", data[5]);
        s("6", data[6]);
        s("7", data[7]);
        s("8", data[8]);
    }
};

// Helper struct to determine data array based on Type
template <RSE_FloatType Type>
struct RS_Mat3DMemDataType;

template <>
struct RS_Mat3DMemDataType<RSE_FLOAT_16> {
    using DataType = unsigned short[9];
    using ElementType = unsigned short;
};

template <>
struct RS_Mat3DMemDataType<RSE_FLOAT_32> {
    using DataType = float[9];
    using ElementType = float;
};

template <>
struct RS_Mat3DMemDataType<RSE_FLOAT_64> {
    using DataType = double[9];
    using ElementType = double;
};

/**
 * @brief Template of 3D matrices stored in memory.
 */
template <RSE_FloatType Type>
class RS_Mat3DMem {
   public:
    using DataType = typename RS_Mat3DMemDataType<Type>::DataType;
    using ElementType = typename RS_Mat3DMemDataType<Type>::ElementType;
    DataType data;  //!< Internal storage.

    //! Default constructor
    RS_Mat3DMem() = default;

    HOST_DEVICE_INLINE_QUALIFIER static RS_Mat3DMem zero() {
        RS_Mat3DMem vec;
        vec.data[0] = 0;
        vec.data[1] = 0;
        vec.data[2] = 0;
        vec.data[3] = 0;
        vec.data[4] = 0;
        vec.data[5] = 0;
        vec.data[6] = 0;
        vec.data[7] = 0;
        vec.data[8] = 0;
        return vec;
    }

    //! Constructor taking a RS_Mat3D<T> as input
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_Mat3DMem(const RS_Mat3D<T> &m) {
        set(m);
    }

    //! Assignment operator from RS_Mat3D<T>
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3DMem &operator=(const RS_Mat3D<T> &m) {
        set(m);
        return *this;
    }

    //! Convert to RS_Mat3D<T>
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER RS_Mat3D<T> toMat3D() const {
        RS_Mat3D<T> m;
        get(m);
        return m;
    }

    //! Helper function to set data from x, y, z
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER void set(const RS_Mat3D<T> &m) {
        if constexpr (Type == RSE_FLOAT_16) {
            for (int i = 0; i < 9; i++)
                data[i] = RS_FloatConvert::floatToHalf(static_cast<float>(m.data[i]));
        } else {
            for (int i = 0; i < 9; i++) data[i] = static_cast<ElementType>(m.data[i]);
        }
    }

    //! Helper function to get data into RS_Mat3D<T>
    template <typename T>
    HOST_DEVICE_INLINE_QUALIFIER void get(RS_Mat3D<T> &m) const {
        if constexpr (Type == RSE_FLOAT_16) {
            for (int i = 0; i < 9; i++)
                m.data[i] = static_cast<T>(RS_FloatConvert::halfToFloat(data[i]));
        } else {
            for (int i = 0; i < 9; i++) m.data[i] = static_cast<T>(data[i]);
        }
    }
};

using RMat3f = RS_Mat3D<float>;
using RMat3df = RS_Mat3D<double>;
using RMat3rf = RS_Mat3D<RFloat>;

}  // namespace rise

REFL_AUTO(type(rise::RMat3f), field(data))
REFL_AUTO(type(rise::RMat3df), field(data))

REFL_AUTO(type(rise::RS_Mat3DMem<rise::RSE_FLOAT_16>), field(data))
REFL_AUTO(type(rise::RS_Mat3DMem<rise::RSE_FLOAT_32>), field(data))
REFL_AUTO(type(rise::RS_Mat3DMem<rise::RSE_FLOAT_64>), field(data))

#endif  // RS_MAT3D_H
