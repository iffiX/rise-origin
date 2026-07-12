// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "utils/init_version.h"

#include <pybind11/operators.h>

#include "rise_origin/utils/version/rs_version.h"

void initRSPYVersion(py::module &m) {
    py::class_<RS_Version>(
        m, "RS_Version", "A semantic version with major, minor, and patch parts."
    )
        .def(py::init<>(), "Create version 0.0.0.")
        .def(
            py::init<RIndex, RIndex, RIndex>(),
            py::arg("major"),
            py::arg("minor") = 0,
            py::arg("patch") = 0,
            "Construct a version from its components."
        )
        .def_readwrite("major", &RS_Version::major, "Major version number (breaking changes).")
        .def_readwrite(
            "minor", &RS_Version::minor, "Minor version number (backwards-compatible features)."
        )
        .def_readwrite(
            "patch", &RS_Version::patch, "Patch version number (backwards-compatible fixes)."
        )
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self <= py::self)
        .def(py::self > py::self)
        .def(py::self >= py::self)
        .def("to_string", &RS_Version::toString, "Return a string representation of the version.")
        .def(
            "from_string",
            [](RS_Version &self, const std::string &version_str) { self.fromString(version_str); },
            py::arg("version_str"),
            "Parse a version string into this version, raising RS_DataException on a bad string."
        )
        .def(
            "is_matching",
            &RS_Version::isMatching,
            py::arg("specifier"),
            "Check if this version matches a version specifier such as '>=1.2', '~=1.4', or '^2.0'."
        )
        .def(
            "is_compatible_with",
            &RS_Version::isCompatibleWith,
            py::arg("other"),
            "Check if this version is compatible with another version."
        )
        .def(
            "next_major",
            &RS_Version::nextMajor,
            "Return the next major version (increments major, resets minor and patch to 0)."
        )
        .def(
            "next_minor",
            &RS_Version::nextMinor,
            "Return the next minor version (increments minor, resets patch to 0)."
        )
        .def(
            "next_patch",
            &RS_Version::nextPatch,
            "Return the next patch version (increments patch)."
        )
        .def(
            "is_pre_release",
            &RS_Version::isPreRelease,
            "Check if this is a pre-release version (0.x.y)."
        )
        .def(
            "is_stable",
            &RS_Version::isStable,
            "Check if this is a stable release version (x.y.z where x > 0)."
        )
        .def("__str__", &RS_Version::toString)
        .def(
            "__repr__",
            [](const RS_Version &self) { return "RS_Version('" + self.toString() + "')"; }
        )
        .def("__hash__", [](const RS_Version &self) {
            return py::hash(py::make_tuple(self.major, self.minor, self.patch));
        });
}
