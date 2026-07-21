// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rsc/rspy_init_world.h"

void initRSPYWorld(py::module &m) {
    /***************************************************************
     * World (top level)
     ***************************************************************/
    // Variant aliases exposed as Python types so that users can construct them
    // from any of their alternatives and inspect the held value via `value`.
    rspyRSC<RSC_ResourceVariant>(m, "RSC_ResourceVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_ResourceVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_ResourceVariant &self) { return rspyVariantGet(self); },
            [](RSC_ResourceVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        );

    rspyRSC<RSC_EssenceVariant>(m, "RSC_EssenceVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_EssenceVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_EssenceVariant &self) { return rspyVariantGet(self); },
            [](RSC_EssenceVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        );

    rspyRSC<RSC_EntityVariant>(m, "RSC_EntityVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_EntityVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_EntityVariant &self) { return rspyVariantGet(self); },
            [](RSC_EntityVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        );

    rspyRSC<RSC_RecordVariant>(m, "RSC_RecordVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_RecordVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_RecordVariant &self) { return rspyVariantGet(self); },
            [](RSC_RecordVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        );

    rspyRSC<RSC_SignalVariant>(m, "RSC_SignalVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_SignalVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_SignalVariant &self) { return rspyVariantGet(self); },
            [](RSC_SignalVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        );

    rspyRSC<RSC_RecorderVariant>(m, "RSC_RecorderVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_RecorderVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_RecorderVariant &self) { return rspyVariantGet(self); },
            [](RSC_RecorderVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        );

    rspyRSC<RSC_ControllerVariant>(m, "RSC_ControllerVariant")
        .def(py::init<>())
        .def(py::init([](py::object arg) {
            RSC_ControllerVariant v;
            rspyVariantSet(v, arg);
            return v;
        }))
        .def_property(
            "value",
            [](RSC_ControllerVariant &self) { return rspyVariantGet(self); },
            [](RSC_ControllerVariant &self, py::object arg) { rspyVariantSet(self, arg); }
        );

    rspyRegisterVariantConversions(static_cast<const RSC_ResourceVariant *>(nullptr));
    rspyRegisterVariantConversions(static_cast<const RSC_EssenceVariant *>(nullptr));
    rspyRegisterVariantConversions(static_cast<const RSC_EntityVariant *>(nullptr));
    rspyRegisterVariantConversions(static_cast<const RSC_RecordVariant *>(nullptr));
    rspyRegisterVariantConversions(static_cast<const RSC_SignalVariant *>(nullptr));
    rspyRegisterVariantConversions(static_cast<const RSC_RecorderVariant *>(nullptr));
    rspyRegisterVariantConversions(static_cast<const RSC_ControllerVariant *>(nullptr));

    rspyRSC<RSC_World, RSC_Base>(m, "RSC_World")
        .def(py::init<>())
        .def_readwrite("name", &RSC_World::name)
        .def_readwrite("simulation", &RSC_World::simulation)
        .def_readwrite("resources", &RSC_World::resources)
        .def_readwrite("essences", &RSC_World::essences)
        .def_readwrite("entities", &RSC_World::entities)
        .def_readwrite("records", &RSC_World::records)
        .def_readwrite("signals", &RSC_World::signals)
        .def_readwrite("recorders", &RSC_World::recorders)
        .def_readwrite("controllers", &RSC_World::controllers)
        .def(
            "assemble_from",
            [](RSC_World &self, py::iterable paths) {
                std::vector<std::string> cpp_paths;
                for (auto item : paths) {
                    cpp_paths.push_back(py::cast<std::string>(item));
                }
                self.assembleFrom(cpp_paths);
            },
            py::arg("paths"),
            R"doc(
            Assemble this world from saved RSC files (.rsct or .rscb), read in order.

            Each file holds either a single component (resource, essence, entity, record, signal,
            recorder, controller), which appends to its section, or a world, which appends its
            components and replaces the simulation.

            Args:
                paths: A list or other iterable of RSC file paths to load.

            Raises:
                RS_DataException: When a path cannot be opened or parsed.
        )doc"
        );

    py::class_<RS_RSCValidator>(m, "RS_RSCValidator")
        .def(py::init<>())
        .def(
            "validate",
            &RS_RSCValidator::validate,
            py::arg("rsc"),
            R"doc(
            Validates a characterization and raises on the first failed check.

            Args:
                rsc: Any RSC_Base subclass, such as an RSC_World, to validate.

            Raises:
                RS_InvalidCharacterizationException: When a check fails.
        )doc"
        );
}
