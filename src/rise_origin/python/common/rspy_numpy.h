// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_NUMPY_H
#define RSPY_NUMPY_H

#include <cstring>
#include <vector>

#include "common/rspy_pybind.h"
#include "common/rspy_traits.h"

/**
 * @brief Return a read-only numpy view that copies the buffer, for record snapshots.
 * @param ptr The buffer start.
 * @param size The element count.
 */
template <typename T>
inline py::array_t<T> rspyRoArray1D(T *ptr, size_t size) {
    return py::array_t<T>({size}, {sizeof(T)}, ptr);
}

/**
 * @brief Return a read-only rank two numpy view that copies the buffer.
 * @param ptr The buffer start.
 * @param n The row count.
 * @param dim The column count.
 */
template <typename T>
inline py::array_t<T> rspyRoArray2D(T *ptr, size_t n, size_t dim) {
    return py::array_t<T>({n, dim}, {dim * sizeof(T), sizeof(T)}, ptr);
}

/**
 * @brief Return a writable non-owning numpy view of the buffer, its lifetime owned by the producer.
 * @param ptr The buffer start.
 * @param size The element count.
 */
template <typename T>
inline py::array_t<T> rspyRwArray1D(T *ptr, size_t size) {
    py::capsule view_owner(ptr, [](void *) {});
    return py::array_t<T>({size}, {sizeof(T)}, ptr, view_owner);
}

/**
 * @brief Return a writable non-owning rank two numpy view of the buffer.
 * @param ptr The buffer start.
 * @param n The row count.
 * @param dim The column count.
 */
template <typename T>
inline py::array_t<T> rspyRwArray2D(T *ptr, size_t n, size_t dim) {
    py::capsule view_owner(ptr, [](void *) {});
    return py::array_t<T>({n, dim}, {dim * sizeof(T), sizeof(T)}, ptr, view_owner);
}

/**
 * @brief Bind a flat std::vector member as a zero-copy writable numpy view, replaced by memcpy on
 * set. The view pins the owning RSC Python object so the buffer stays valid.
 * @param cls The class to add the property to.
 * @param name The property name.
 * @param member Pointer to the vector member.
 */
template <typename PyClass, typename C, typename E>
void rspyBindArray(PyClass &cls, const char *name, std::vector<E> C::*member) {
    using U = typename RSPY_NumpyElement<E>::type;
    cls.def_property(
        name,
        [member](py::object self) {
            std::vector<E> &v = self.cast<C &>().*member;
            return py::array_t<U>(
                {static_cast<py::ssize_t>(v.size())},
                {static_cast<py::ssize_t>(sizeof(E))},
                reinterpret_cast<U *>(v.data()),
                self
            );
        },
        [member](C &c, py::array_t<U, py::array::c_style | py::array::forcecast> arr) {
            std::vector<E> &v = c.*member;
            v.resize(static_cast<std::size_t>(arr.size()));
            std::memcpy(v.data(), arr.data(), v.size() * sizeof(E));
        }
    );
}

#endif  // RSPY_NUMPY_H
