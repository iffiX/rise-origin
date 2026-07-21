// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_VECTOR_VIEW_H
#define RSPY_VECTOR_VIEW_H

#include <algorithm>
#include <cstddef>
#include <vector>

#include "common/rspy_pybind.h"
#include "common/rspy_traits.h"

/**
 * @brief Type-erased operations over one element type, so one Python view class serves every
 * vector.
 */
struct RSPY_VectorViewOps {
    std::size_t (*size)(void *);
    py::object (*get)(const py::object &, void *, std::size_t);
    void (*set)(void *, std::size_t, const py::object &);
    void (*insert)(void *, std::size_t, const py::object &);
    void (*erase)(void *, std::size_t);
    void (*clear)(void *);
};

/**
 * @brief A live write through list view over a std::vector member, pinned to its owning object. A
 * nested element yields a recursive subview over the inner vector, sharing the same owner.
 */
struct RSPY_VectorView {
    py::object owner;
    void *target;
    const RSPY_VectorViewOps *ops;
};

/**
 * @brief Build a std::vector<E> from a Python iterable, recursing into nested vectors.
 * @param items The Python iterable of elements.
 */
template <typename E>
std::vector<E> rspyBuildVector(const py::handle &items) {
    std::vector<E> out;
    for (py::handle item : items) {
        if constexpr (RSPY_IsVector<E>::value) {
            out.push_back(rspyBuildVector<typename E::value_type>(item));
        } else {
            out.push_back(py::reinterpret_borrow<py::object>(item).cast<E>());
        }
    }
    return out;
}

/**
 * @brief The operations table for a vector of element type E, built once per E.
 */
template <typename E>
const RSPY_VectorViewOps &rspyGetVectorViewOps() {
    static const RSPY_VectorViewOps ops = {
        [](void *t) { return static_cast<std::vector<E> *>(t)->size(); },
        [](const py::object &owner, void *t, std::size_t i) -> py::object {
            auto &v = *static_cast<std::vector<E> *>(t);
            if constexpr (RSPY_IsVector<E>::value) {
                return py::cast(
                    RSPY_VectorView{owner, &v[i], &rspyGetVectorViewOps<typename E::value_type>()}
                );
            } else if constexpr (std::is_same_v<E, std::string>) {
                return py::cast(v[i]);
            } else {
                return py::cast(&v[i], py::return_value_policy::reference_internal, owner);
            }
        },
        [](void *t, std::size_t i, const py::object &value) {
            auto &v = *static_cast<std::vector<E> *>(t);
            if constexpr (RSPY_IsVector<E>::value) {
                v[i] = rspyBuildVector<typename E::value_type>(value);
            } else {
                v[i] = value.cast<E>();
            }
        },
        [](void *t, std::size_t i, const py::object &value) {
            auto &v = *static_cast<std::vector<E> *>(t);
            if constexpr (RSPY_IsVector<E>::value) {
                v.insert(v.begin() + i, rspyBuildVector<typename E::value_type>(value));
            } else {
                v.insert(v.begin() + i, value.cast<E>());
            }
        },
        [](void *t, std::size_t i) {
            auto &v = *static_cast<std::vector<E> *>(t);
            v.erase(v.begin() + i);
        },
        [](void *t) { static_cast<std::vector<E> *>(t)->clear(); },
    };
    return ops;
}

/**
 * @brief Register the single vector view class. Called once during module init.
 * @param m The module to register into.
 */
inline void rspyRegisterVectorView(py::module &m) {
    auto norm = [](py::ssize_t i, std::size_t n) -> std::size_t {
        if (i < 0) i += static_cast<py::ssize_t>(n);
        if (i < 0 || static_cast<std::size_t>(i) >= n) throw py::index_error();
        return static_cast<std::size_t>(i);
    };
    auto to_list = [](RSPY_VectorView &self) {
        py::list out;
        std::size_t n = self.ops->size(self.target);
        for (std::size_t i = 0; i < n; i++) out.append(self.ops->get(self.owner, self.target, i));
        return out;
    };
    py::class_<RSPY_VectorView>(m, "RSVectorView")
        .def("__len__", [](RSPY_VectorView &self) { return self.ops->size(self.target); })
        .def(
            "__getitem__",
            [norm](RSPY_VectorView &self, py::ssize_t i) {
                return self.ops->get(self.owner, self.target, norm(i, self.ops->size(self.target)));
            }
        )
        .def(
            "__setitem__",
            [norm](RSPY_VectorView &self, py::ssize_t i, const py::object &value) {
                self.ops->set(self.target, norm(i, self.ops->size(self.target)), value);
            }
        )
        .def(
            "__delitem__",
            [norm](RSPY_VectorView &self, py::ssize_t i) {
                self.ops->erase(self.target, norm(i, self.ops->size(self.target)));
            }
        )
        .def("__iter__", [to_list](RSPY_VectorView &self) { return py::iter(to_list(self)); })
        .def(
            "append",
            [](RSPY_VectorView &self, const py::object &value) {
                self.ops->insert(self.target, self.ops->size(self.target), value);
            }
        )
        .def(
            "extend",
            [](RSPY_VectorView &self, const py::iterable &items) {
                for (py::handle item : items) {
                    self.ops->insert(
                        self.target,
                        self.ops->size(self.target),
                        py::reinterpret_borrow<py::object>(item)
                    );
                }
            }
        )
        .def(
            "insert",
            [](RSPY_VectorView &self, py::ssize_t i, const py::object &value) {
                std::size_t n = self.ops->size(self.target);
                self.ops->insert(self.target, i < 0 ? 0 : std::min<std::size_t>(i, n), value);
            }
        )
        .def("clear", [](RSPY_VectorView &self) { self.ops->clear(self.target); })
        .def(
            "__eq__",
            [to_list](RSPY_VectorView &self, const py::object &other) {
                return to_list(self).equal(other);
            }
        )
        .def("__repr__", [to_list](RSPY_VectorView &self) { return py::repr(to_list(self)); });
}

/**
 * @brief Bind a std::vector member as a live write through view: the getter returns a view pinned
 * to the owner, the setter rebuilds the vector from an iterable. For value elements, not RSC
 * vectors.
 * @param cls The class to add the property to.
 * @param name The property name.
 * @param member Pointer to the vector member.
 */
template <typename PyClass, typename C, typename E>
void rspyBindVector(PyClass &cls, const char *name, std::vector<E> C::*member) {
    cls.def_property(
        name,
        [member](py::object self) {
            std::vector<E> &v = self.cast<C &>().*member;
            return RSPY_VectorView{self, &v, &rspyGetVectorViewOps<E>()};
        },
        [member](C &c, const py::handle &items) { c.*member = rspyBuildVector<E>(items); }
    );
}

#endif  // RSPY_VECTOR_VIEW_H
