// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_INIT_COMMON_H
#define RSPY_INIT_COMMON_H

#include <pybind11/chrono.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rise_origin/core/rs_world_callback.h"
#include "rise_origin/core/rs_world_controller.h"
#include "rise_origin/core/rs_world_def.h"
#include "rise_origin/core/rs_world_handle.h"
#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/core/rs_world_session.h"
#include "rise_origin/core/rs_world_session_component.h"
#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/rise_origin.h"
#include "rise_origin/rsc/rs_rsc_file.h"
#include "rise_origin/rsc/rs_rsc_validator.h"
#include "rise_origin/utils/memory/rs_device.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/serialization/rs_mp_serialization.h"
#include "rise_origin/utils/string/rs_string.h"

PYBIND11_MAKE_OPAQUE(std::vector<int16_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint16_t>);
PYBIND11_MAKE_OPAQUE(std::vector<int32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<int64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<char>);
PYBIND11_MAKE_OPAQUE(std::vector<unsigned char>);

PYBIND11_MAKE_OPAQUE(std::vector<bool>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);

PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
PYBIND11_MAKE_OPAQUE(std::vector<RS_MathExprToken>);

namespace py = pybind11;
using namespace rise;
using namespace rise::rsc;

// Read-only numpy array helpers: return a copy of the underlying buffer.
// Used for record snapshots that should not be mutated by Python callers.
template <typename T>
inline py::array_t<T> rspyRoArray1D(T *ptr, size_t size) {
    return py::array_t<T>({size}, {sizeof(T)}, ptr);
}

template <typename T>
inline py::array_t<T> rspyRoArray2D(T *ptr, size_t n, size_t dim) {
    return py::array_t<T>({n, dim}, {dim * sizeof(T), sizeof(T)}, ptr);
}

// Read-write numpy array helpers: return a non-owning writable view of the
// underlying buffer through a no-op capsule base. The buffer's lifetime is
// owned by the manager that produced it.
template <typename T>
inline py::array_t<T> rspyRwArray1D(T *ptr, size_t size) {
    py::capsule view_owner(ptr, [](void *) {});
    return py::array_t<T>({size}, {sizeof(T)}, ptr, view_owner);
}

template <typename T>
inline py::array_t<T> rspyRwArray2D(T *ptr, size_t n, size_t dim) {
    py::capsule view_owner(ptr, [](void *) {});
    return py::array_t<T>({n, dim}, {dim * sizeof(T), sizeof(T)}, ptr, view_owner);
}

// True when T is an RS_Variant specialization, used to route variant element conversion.
template <typename T>
struct rspy_is_variant : std::false_type {};
template <typename... A>
struct rspy_is_variant<RS_Variant<A...>> : std::true_type {};

// The integral element type a per-voxel array exposes to numpy: enums expose their underlying type.
template <typename E, bool = std::is_enum_v<E>>
struct rspy_numpy_element {
    using type = E;
};
template <typename E>
struct rspy_numpy_element<E, true> {
    using type = std::underlying_type_t<E>;
};

// Serialization helper functions
template <typename T>
py::bytes rspySerializeRSC(const T &rsc) {
    RS_MessagePackSerializer s;
    s.serialize(rsc);
    auto data = s.moveData();
    return py::bytes(reinterpret_cast<const char *>(data.data()), data.size());
}

template <typename T>
T rspyDeserializeRSC(const py::bytes &pybuf) {
    char *buffer;
    Py_ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(pybuf.ptr(), &buffer, &length)) {
        throw std::runtime_error("Failed to extract bytes contents");
    }

    T rsc;
    RS_MessagePackDeserializer d;
    d.setData(
        std::vector<uint8_t>(
            reinterpret_cast<const uint8_t *>(buffer),
            reinterpret_cast<const uint8_t *>(buffer) + length
        )
    );
    d.deserialize(rsc);
    return rsc;
}

// Variant-property helpers: expose an RS_Variant<Alternatives...> field as a
// Python attribute whose get returns the currently held alternative by reference
// and whose set dispatches on the concrete Python type.
template <typename... Alternatives>
py::object rspyVariantGet(RS_Variant<Alternatives...> &v) {
    if (v.index() == RS_NULL_INDEX) {
        return py::none();
    }
    py::object result = py::none();
    v.visit([&](auto &held) { result = py::cast(&held, py::return_value_policy::reference); });
    return result;
}

template <typename Alt, typename... Alternatives>
bool rspyVariantTryAssign(RS_Variant<Alternatives...> &v, py::object arg) {
    if (py::isinstance<Alt>(arg)) {
        v = arg.cast<Alt>();
        return true;
    }
    return false;
}

template <typename... Alternatives>
void rspyVariantSet(RS_Variant<Alternatives...> &v, py::object arg) {
    bool ok = false;
    // Fold-expression: try each alternative type in order.
    ((ok = ok || rspyVariantTryAssign<Alternatives, Alternatives...>(v, arg)), ...);
    if (!ok) {
        throw std::runtime_error("Unsupported type assigned to variant field");
    }
}

// Register each concrete alternative as implicitly convertible to its variant, so a builder can be
// passed wherever the variant is expected, such as handle.add_entity(StructureVoxel(...)).
template <typename... Alternatives>
void rspyRegisterVariantConversions(const RS_Variant<Alternatives...> *) {
    (py::implicitly_convertible<Alternatives, RS_Variant<Alternatives...>>(), ...);
}

// Optional-property helpers: expose an RS_Optional<T> field as a Python attribute
// that reads None when empty and accepts None to clear it.
template <typename T>
py::object rspyOptionalGet(const RS_Optional<T> &o) {
    return o.empty() ? py::none() : py::cast(o.get());
}

template <typename T>
void rspyOptionalSet(RS_Optional<T> &o, py::object arg) {
    if (arg.is_none()) {
        o.reset();
    } else {
        o = arg.cast<T>();
    }
}

// Expose a flat std::vector member as a zero-copy writable numpy view (get) and a memcpy replace
// (set). The view pins the owning RSC Python object so the buffer stays valid.
template <typename PyClass, typename C, typename E>
void rspyBindArray(PyClass &cls, const char *name, std::vector<E> C::*member) {
    using U = typename rspy_numpy_element<E>::type;
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

// Register an RSCVec specialization as an opaque, list like Python type. Reads unwrap a variant
// element to its held alternative or return a live reference into the vector so mutation writes
// through; writes accept either a built variant or any of its alternatives. A def_readwrite member
// of this type then gives append in place through its getter and whole list assignment through the
// iterable conversion registered here.
template <typename T>
py::class_<rise::rsc::RSC_Base::RSCVec<T>> rspyBindRSCVec(py::handle scope, const char *name) {
    using Vec = rise::rsc::RSC_Base::RSCVec<T>;
    auto make = [](const py::object &value) -> T {
        if constexpr (rspy_is_variant<T>::value) {
            if (py::isinstance<T>(value)) {
                return value.cast<T>();
            }
            T held;
            rspyVariantSet(held, value);
            return held;
        } else {
            return value.cast<T>();
        }
    };
    auto read = [](py::handle parent, Vec &v, std::size_t i) -> py::object {
        if constexpr (rspy_is_variant<T>::value) {
            return rspyVariantGet(v[i]);
        } else {
            return py::cast(&v[i], py::return_value_policy::reference_internal, parent);
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

    py::class_<Vec> cls(scope, name);
    cls.def(py::init<>());
    cls.def(py::init([make](const py::iterable &items) {
        Vec v;
        for (py::handle item : items) {
            v.push_back(make(py::reinterpret_borrow<py::object>(item)));
        }
        return v;
    }));
    cls.def("__len__", [](const Vec &v) { return v.size(); });
    cls.def("__getitem__", [read, norm](py::object self, py::ssize_t i) {
        Vec &v = self.cast<Vec &>();
        return read(self, v, norm(i, v.size()));
    });
    cls.def("__setitem__", [make, norm](Vec &v, py::ssize_t i, const py::object &value) {
        v[norm(i, v.size())] = make(value);
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
    cls.def("append", [make](Vec &v, const py::object &value) { v.push_back(make(value)); });
    cls.def("extend", [make](Vec &v, const py::iterable &items) {
        for (py::handle item : items) {
            v.push_back(make(py::reinterpret_borrow<py::object>(item)));
        }
    });
    cls.def("insert", [make](Vec &v, py::ssize_t i, const py::object &value) {
        v.insert(i < 0 ? 0 : std::min<std::size_t>(i, v.size()), make(value));
    });
    cls.def("clear", [](Vec &v) { v.clear(); });
    py::implicitly_convertible<py::iterable, Vec>();
    return cls;
}

// Create a py::class_ for an RSC type and attach pickle then copy support. For abstract bases and
// non RSC types the extra methods are skipped, so it is safe to use for every binding.
template <typename T, typename... Options>
py::class_<T, Options...> rspyRSC(py::module &m, const char *name) {
    py::class_<T, Options...> cls(m, name);
    if constexpr (!std::is_abstract_v<T>) {
        // Add pickle methods.
        cls.def(
            py::pickle(
                [](const T &self) { return rspySerializeRSC(self); },
                [](py::bytes buffer) { return rspyDeserializeRSC<T>(buffer); }
            )
        );
        if constexpr (std::is_base_of_v<RSC_Base, T>) {
            // Copy methods.
            cls.def(
                "shallow_copy",
                [](py::object self, const py::kwargs &overrides) {
                    py::object copy = py::cast(T(self.cast<T &>()));
                    // Apply overrides.
                    for (auto item : overrides) {
                        std::string key = item.first.cast<std::string>();
                        if (!py::hasattr(copy, key.c_str())) {
                            throw py::type_error("unknown field '" + key + "'");
                        }
                        py::setattr(
                            copy, key.c_str(), py::reinterpret_borrow<py::object>(item.second)
                        );
                    }
                    return copy;
                },
                R"doc(
            Copy this characterization while sharing its sub RSC children.

            The copy and the original point at the same child nodes, so mutating a shared child
            shows in both. Pass field=value keyword arguments to override fields on the copy.
        )doc"
            );
            cls.def(
                "deep_copy",
                [](py::object self, const py::kwargs &overrides) {
                    auto deep = self.cast<T &>().deepCopy();
                    py::object copy = py::cast(T(*std::static_pointer_cast<T>(deep.shared())));
                    // Apply overrides.
                    for (auto item : overrides) {
                        std::string key = item.first.cast<std::string>();
                        if (!py::hasattr(copy, key.c_str())) {
                            throw py::type_error("unknown field '" + key + "'");
                        }
                        py::setattr(
                            copy, key.c_str(), py::reinterpret_borrow<py::object>(item.second)
                        );
                    }
                    return copy;
                },
                R"doc(
            Copy this characterization and all its sub RSC children independently.

            A node shared by several parents is copied once, so the sharing structure is preserved.
            Pass field=value keyword arguments to override fields on the copy.
        )doc"
            );
            cls.def(
                "save",
                [](const T &self, const std::string &path) { RS_RSCFile::save(self, path); },
                py::arg("path"),
                R"doc(
            Save to a durable file, the format chosen by the path extension.

            A .rsct path writes readable JSON text; a .rscb path writes compact binary. The two
            formats are interchangeable and carry the same data.
        )doc"
            );
            cls.def_static(
                "load",
                [](const std::string &path) { return RS_RSCFile::load<T>(path); },
                py::arg("path"),
                R"doc(
            Load from a durable file, the format chosen by the path extension.

            Reads a .rsct (JSON text) or .rscb (binary) file and verifies it holds this type.
        )doc"
            );
        }
    }
    return cls;
}

namespace pybind11 {

namespace detail {

// Bind an embedded sub RSC held by RSCRef as the referenced object, or None when null. Reading
// returns a reference into the owner; writing copies the assigned value into a fresh shared node.
template <typename T>
struct type_caster<rise::rsc::RSC_Base::RSCRef<T>> {
    using RefType = rise::rsc::RSC_Base::RSCRef<T>;
    PYBIND11_TYPE_CASTER(RefType, const_name("Optional[") + make_caster<T>::name + const_name("]"));

    bool load(handle src, bool convert) {
        if (src.is_none()) {
            value = RefType();
            return true;
        }
        make_caster<T> caster;
        if (!caster.load(src, convert)) {
            return false;
        }
        value = RefType(cast_op<T &>(caster));
        return true;
    }

    static handle cast(const RefType &src, return_value_policy, handle parent) {
        if (src.isNull()) {
            return none().release();
        }
        return pybind11::cast(src.get(), return_value_policy::reference_internal, parent).release();
    }
};

}  // namespace detail

}  // namespace pybind11

#endif  // RSPY_INIT_COMMON_H
