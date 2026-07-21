// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rsc/rspy_init_rsc_base.h"

void initRSPYRSCBase(py::module &m) {
    /***************************************************************
     * Exceptions
     ***************************************************************/
    auto data_exception = py::register_exception<RS_DataException>(m, "RS_DataException");
    py::register_exception<RS_InvalidCharacterizationException>(
        m, "RS_InvalidCharacterizationException", data_exception.ptr()
    );

    /***************************************************************
     * Bases
     ***************************************************************/
    // Note: all RSC_Base classes are consolidated here. They must be bound
    // before any derived class references them.
    rspyRSC<RSC_Base>(m, "RSC_Base");

    // Opaque list like containers for every RSCVec member, registered before any member uses them.
    rspyBindRSCVec<RSC_StructureVoxelBody>(m, "RSCVecStructureVoxelBody");
    rspyBindRSCVec<RSC_StructureVoxelConstraint>(m, "RSCVecStructureVoxelConstraint");
    rspyBindRSCVec<RSC_EntityVariant>(m, "RSCVecEntity");
    rspyBindRSCVec<RSC_EssenceVariant>(m, "RSCVecEssence");
    rspyBindRSCVec<RSC_ResourceVariant>(m, "RSCVecResource");
    rspyBindRSCVec<RSC_RecordVariant>(m, "RSCVecRecord");
    rspyBindRSCVec<RSC_SignalVariant>(m, "RSCVecSignal");
    rspyBindRSCVec<RSC_ControllerVariant>(m, "RSCVecController");
    rspyBindRSCVec<RSC_RecorderVariant>(m, "RSCVecRecorder");

    rspyRSC<RSC_ResourceBase, RSC_Base>(m, "RSC_ResourceBase")
        .def_readwrite("name", &RSC_ResourceBase::name);

    rspyRSC<RSC_EssenceBase, RSC_Base>(m, "RSC_EssenceBase")
        .def_readwrite("name", &RSC_EssenceBase::name);

    rspyRSC<RSC_EntityBase, RSC_Base>(m, "RSC_EntityBase")
        .def_readwrite("name", &RSC_EntityBase::name);

    rspyRSC<RSC_RecordBase, RSC_Base>(m, "RSC_RecordBase")
        .def_readwrite("name", &RSC_RecordBase::name)
        .def_readwrite("target_controller_reference", &RSC_RecordBase::target_controller_reference)
        .def_readwrite("target_recorder_reference", &RSC_RecordBase::target_recorder_reference);

    rspyRSC<RSC_SignalBase, RSC_Base>(m, "RSC_SignalBase")
        .def_readwrite("name", &RSC_SignalBase::name)
        .def_readwrite("source_controller_reference", &RSC_SignalBase::source_controller_reference);

    rspyRSC<RSC_RecorderBase, RSC_Base>(m, "RSC_RecorderBase")
        .def_readwrite("name", &RSC_RecorderBase::name);

    rspyRSC<RSC_ControllerBase, RSC_Base>(m, "RSC_ControllerBase")
        .def_readwrite("name", &RSC_ControllerBase::name);
}
