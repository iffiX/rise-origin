// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_RSC_VEC_H
#define RSPY_RSC_VEC_H

#include <algorithm>
#include <cstddef>

#include "common/rspy_pybind.h"
#include "common/rspy_traits.h"
#include "common/rspy_variant.h"

/**
 * @brief Register an RSCVec specialization as an opaque, write through, list like Python type.
 * Variant elements unwrap to their held alternative; concrete RSC elements share by object
 * identity.
 * @param scope The scope to register the type into.
 * @param name The Python type name.
 */
template <typename T>
py::class_<rise::rsc::RSC_Base::RSCVec<T>> rspyBindRSCVec(py::handle scope, const char *name) {
    using Vec = rise::rsc::RSC_Base::RSCVec<T>;
    using Ref = rise::rsc::RSC_Base::RSCRef<T>;
    // Variant elements store by value, concrete RSC elements store the wrapper's shared pointer.
    auto push = [](Vec &v, const py::object &value) {
        if constexpr (RSPY_IsVariant<T>::value) {
            if (py::isinstance<T>(value)) {
                v.push_back(value.cast<T>());
            } else {
                T held;
                rspyVariantSet(held, value);
                v.push_back(held);
            }
        } else {
            v.push_back(Ref(value.cast<std::shared_ptr<T>>()));
        }
    };
    auto set = [](Vec &v, std::size_t i, const py::object &value) {
        if constexpr (RSPY_IsVariant<T>::value) {
            if (py::isinstance<T>(value)) {
                v[i] = value.cast<T>();
            } else {
                T held;
                rspyVariantSet(held, value);
                v[i] = held;
            }
        } else {
            v.setRef(i, Ref(value.cast<std::shared_ptr<T>>()));
        }
    };
    auto read = [](py::handle parent, Vec &v, std::size_t i) -> py::object {
        if constexpr (RSPY_IsVariant<T>::value) {
            return rspyVariantGet(v[i]);
        } else {
            return py::cast(v.ref(i).shared());
        }
    };
    auto norm = [](py::ssize_t i, std::size_t n) -> std::size_t {
        if (i < 0) {
            i += static_cast<py::ssize_t>(n);
        }
        if (i < 0 || static_cast<std::size_t>(i) >= n) {
            throw py::index_error();
        }
        return static_cast<std::size_t>(i);
    };
    auto insertAt = [](Vec &v, std::size_t i, const py::object &value) {
        if constexpr (RSPY_IsVariant<T>::value) {
            if (py::isinstance<T>(value)) {
                v.insert(i, value.cast<T>());
            } else {
                T held;
                rspyVariantSet(held, value);
                v.insert(i, held);
            }
        } else {
            v.insertRef(i, Ref(value.cast<std::shared_ptr<T>>()));
        }
    };

    py::class_<Vec> cls(scope, name);
    cls.def(py::init<>());
    cls.def(py::init([push](const py::iterable &items) {
        Vec v;
        for (py::handle item : items) {
            push(v, py::reinterpret_borrow<py::object>(item));
        }
        return v;
    }));
    cls.def("__len__", [](const Vec &v) { return v.size(); });
    cls.def("__getitem__", [read, norm](py::object self, py::ssize_t i) {
        Vec &v = self.cast<Vec &>();
        return read(self, v, norm(i, v.size()));
    });
    cls.def("__setitem__", [set, norm](Vec &v, py::ssize_t i, const py::object &value) {
        set(v, norm(i, v.size()), value);
    });
    cls.def("__delitem__", [norm](Vec &v, py::ssize_t i) { v.erase(norm(i, v.size())); });
    cls.def("__iter__", [read](py::object self) {
        Vec &v = self.cast<Vec &>();
        py::list out;
        for (std::size_t i = 0; i < v.size(); i++) {
            out.append(read(self, v, i));
        }
        return py::iter(out);
    });
    cls.def("append", [push](Vec &v, const py::object &value) { push(v, value); });
    cls.def("extend", [push](Vec &v, const py::iterable &items) {
        for (py::handle item : items) {
            push(v, py::reinterpret_borrow<py::object>(item));
        }
    });
    cls.def("insert", [insertAt](Vec &v, py::ssize_t i, const py::object &value) {
        insertAt(v, i < 0 ? 0 : std::min<std::size_t>(i, v.size()), value);
    });
    cls.def("clear", [](Vec &v) { v.clear(); });
    py::implicitly_convertible<py::iterable, Vec>();
    return cls;
}

#endif  // RSPY_RSC_VEC_H
