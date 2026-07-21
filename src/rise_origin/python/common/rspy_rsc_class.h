// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_RSC_CLASS_H
#define RSPY_RSC_CLASS_H

#include <memory>
#include <string>

#include "common/rspy_pybind.h"
#include "common/rspy_rsc_serialize.h"

/**
 * @brief Create a py::class_ for an RSC type with a shared holder, attaching pickle, copy, and file
 * methods. Abstract bases and non RSC types skip the extra methods, so it is safe for every
 * binding.
 * @param m The module to register into.
 * @param name The Python type name.
 */
template <typename T, typename... Options>
py::class_<T, std::shared_ptr<T>, Options...> rspyRSC(py::module &m, const char *name) {
    py::class_<T, std::shared_ptr<T>, Options...> cls(m, name);
    if constexpr (!std::is_abstract_v<T>) {
        cls.def(
            py::pickle(
                [](const T &self) { return rspySerializeRSC(self); },
                [](const py::object &buffer) { return rspyDeserializeRSC<T>(buffer); }
            )
        );
        // Protocol 5 carries the archive bytes as a zero-copy PickleBuffer, so pickling a large
        // scene skips the extra bytes copy and can go out of band; older protocols keep the plain
        // bytes state. Both reconstruct through the __setstate__ defined above.
        cls.def(
            "__reduce_ex__",
            [](py::object self, int protocol) {
                py::object state = protocol >= 5 ? rspySerializeRSCZeroCopy(self.cast<const T &>())
                                                 : self.attr("__getstate__")();
                return py::make_tuple(
                    py::module_::import("copyreg").attr("__newobj__"),
                    py::make_tuple(self.attr("__class__")),
                    state
                );
            },
            py::arg("protocol")
        );
        if constexpr (std::is_base_of_v<RSC_Base, T>) {
            cls.def(
                "shallow_copy",
                [](py::object self, const py::kwargs &overrides) {
                    py::object copy = py::cast(T(self.cast<T &>()));
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
                [](const T &self, const std::string &path, bool is_compressing) {
                    RS_RSCFile::save(self, path, is_compressing);
                },
                py::arg("path"),
                py::arg("is_compressing") = true,
                R"doc(
            Save to a durable file, the format chosen by the path extension.

            A .rsct path writes readable JSON text; a .rscb path writes compact binary. The two
            formats are interchangeable and carry the same data.

            Args:
                path: The destination path, .rsct for text or .rscb for binary.
                is_compressing: Whether the binary format compresses its fields, ignored for text.
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

/**
 * @brief Bind an embedded RSCRef as its referenced object or None. Assigning stores the wrapper's
 * shared pointer, so object identity means sharing, and reading returns that same wrapper.
 */
template <typename T>
struct type_caster<rise::rsc::RSC_Base::RSCRef<T>> {
    using RefType = rise::rsc::RSC_Base::RSCRef<T>;
    PYBIND11_TYPE_CASTER(RefType, const_name("Optional[") + make_caster<T>::name + const_name("]"));

    bool load(handle src, bool convert) {
        if (src.is_none()) {
            value = RefType();
            return true;
        }
        make_caster<std::shared_ptr<T>> caster;
        if (!caster.load(src, convert)) {
            return false;
        }
        value = RefType(cast_op<std::shared_ptr<T>>(caster));
        return true;
    }

    static handle cast(const RefType &src, return_value_policy, handle) {
        if (src.isNull()) {
            return none().release();
        }
        return pybind11::cast(src.shared()).release();
    }
};

}  // namespace detail

}  // namespace pybind11

#endif  // RSPY_RSC_CLASS_H
