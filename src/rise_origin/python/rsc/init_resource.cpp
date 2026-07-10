// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rsc/init_resource.h"

void initRSPYResource(py::module &m) {
    /***************************************************************
     * Resources
     ***************************************************************/
    rspyRSC<RSC_TextureBase, RSC_ResourceBase>(m, "RSC_TextureBase");

    rspyRSC<RSC_TextureConstant, RSC_TextureBase>(m, "RSC_TextureConstant")
        .def(py::init<>())
        .def_readwrite("value", &RSC_TextureConstant::value);

    rspyRSC<RSC_TextureGrid, RSC_TextureBase>(m, "RSC_TextureGrid")
        .def(py::init<>())
        .def_readwrite("fill_color", &RSC_TextureGrid::fill_color)
        .def_readwrite("line_color", &RSC_TextureGrid::line_color)
        .def_readwrite("is_meters", &RSC_TextureGrid::is_meters)
        .def_readwrite("x_spacing", &RSC_TextureGrid::x_spacing)
        .def_readwrite("y_spacing", &RSC_TextureGrid::y_spacing)
        .def_readwrite("line_width", &RSC_TextureGrid::line_width);
}
