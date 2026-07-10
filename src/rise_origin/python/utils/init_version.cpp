// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "utils/init_version.h"

#include <pybind11/operators.h>

#include "rise_origin/utils/version/rs_version.h"

void initRSPYVersion(py::module &m) {
    py::class_<RS_Version>(
        m, "RS_Version", "A semantic version with major, minor, and patch parts."
    )
        .def(py::init<>())
        .def(
            py::init<RIndex, RIndex, RIndex>(),
            py::arg("major"),
            py::arg("minor") = 0,
            py::arg("patch") = 0
        )
        .def_readwrite("major", &RS_Version::major)
        .def_readwrite("minor", &RS_Version::minor)
        .def_readwrite("patch", &RS_Version::patch)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self <= py::self)
        .def(py::self > py::self)
        .def(py::self >= py::self)
        .def("to_string", &RS_Version::toString)
        .def(
            "from_string",
            [](RS_Version &self, const std::string &version_str) { self.fromString(version_str); },
            py::arg("version_str"),
            "Parse major.minor.patch into this version, raising RS_DataException on a bad string."
        )
        .def(
            "matches",
            &RS_Version::matches,
            py::arg("specifier"),
            "Whether this version satisfies a specifier such as '>=1.2', '~=1.4', or '^2.0'."
        )
        .def("is_compatible_with", &RS_Version::isCompatibleWith, py::arg("other"))
        .def("next_major", &RS_Version::nextMajor)
        .def("next_minor", &RS_Version::nextMinor)
        .def("next_patch", &RS_Version::nextPatch)
        .def("is_pre_release", &RS_Version::isPreRelease)
        .def("is_stable", &RS_Version::isStable)
        .def("__str__", &RS_Version::toString)
        .def(
            "__repr__",
            [](const RS_Version &self) { return "RS_Version('" + self.toString() + "')"; }
        )
        .def("__hash__", [](const RS_Version &self) {
            return py::hash(py::make_tuple(self.major, self.minor, self.patch));
        });
}
