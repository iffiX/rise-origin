// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rsc/rspy_init_recorder.h"

void initRSPYRecorder(py::module &m) {
    /***************************************************************
     * Recorder
     ***************************************************************/
    rspyRSC<RSC_RecorderHDF5, RSC_RecorderBase>(m, "RSC_RecorderHDF5")
        .def(py::init<>())
        .def_readwrite("path", &RSC_RecorderHDF5::path);
}
