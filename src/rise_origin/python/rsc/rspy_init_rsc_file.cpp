// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rsc/rspy_init_rsc_file.h"

void initRSPYRSCFile(py::module &m) {
    py::class_<RS_RSCFile>(
        m,
        "RS_RSCFile",
        "Durable save and load of any RSC node, in the text (.rsct) or binary (.rscb) format."
    )
        .def_static(
            "save",
            &RS_RSCFile::saveAny,
            py::arg("node"),
            py::arg("path"),
            py::arg("is_compressing") = true,
            R"doc(
            Save any RSC node to a durable file, the format chosen by the path extension.

            A .rsct path writes readable JSON text; a .rscb path writes compact binary. The node's
            concrete type is taken from the object itself, so the caller need not name it.

            Args:
                node: The RSC node to save.
                path: The destination path, .rsct for text or .rscb for binary.
                is_compressing: Whether the binary format compresses its fields, ignored for text.
        )doc"
        )
        .def_static(
            "load",
            [](const std::string &path) -> py::object {
                RSC_Base::RSCRef<RSC_Base> ref = RS_RSCFile::loadAny(path);
                if (ref.isNull()) {
                    return py::none();
                }
                // Return the concrete bound type and pin the owning shared node to the returned
                // wrapper so it outlives this call. reference_internal is unfit here because a
                // standalone load has no owner object to pin against.
                std::shared_ptr<RSC_Base> node = ref.shared();
                py::object obj = py::cast(node.get(), py::return_value_policy::reference);
                py::capsule keeper(new std::shared_ptr<RSC_Base>(node), [](void *p) {
                    delete static_cast<std::shared_ptr<RSC_Base> *>(p);
                });
                py::detail::keep_alive_impl(obj, keeper);
                return obj;
            },
            py::arg("path"),
            R"doc(
            Load any RSC node from a durable file, the format chosen by the path extension.

            Reads a .rsct (JSON text) or .rscb (binary) file and returns the node as its concrete
            bound RSC type, dispatching the root type by the name stored in the file.

            Raises:
                RS_DataException: When a path cannot be opened or parsed.
                RS_InvalidCharacterizationException: When the file is not a readable RSC file.
        )doc"
        )
        .def_static(
            "current_version",
            &RS_RSCFile::currentVersion,
            "The RSC schema version this build writes."
        )
        .def_static("engine_version", &RS_RSCFile::engineVersion, "The engine build version.");
}
