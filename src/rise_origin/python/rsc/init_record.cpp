// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rsc/init_record.h"

void initRSPYRecord(py::module &m) {
    /***************************************************************
     * Record
     ***************************************************************/
    rspyRSC<RSC_IntersectRegionCuboid, RSC_RecordBase>(m, "RSC_IntersectRegionCuboid")
        .def(py::init<>())
        .def_readwrite("position", &RSC_IntersectRegionCuboid::position)
        .def_readwrite("orientation", &RSC_IntersectRegionCuboid::orientation)
        .def_readwrite("size", &RSC_IntersectRegionCuboid::size);

    rspyRSC<RSC_IntersectRegionCuboidRigidBodyLinked, RSC_RecordBase>(
        m, "RSC_IntersectRegionCuboidRigidBodyLinked"
    )
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_IntersectRegionCuboidRigidBodyLinked::structure_name)
        .def_readwrite("body_sid", &RSC_IntersectRegionCuboidRigidBodyLinked::body_sid)
        .def_readwrite("segment_bid", &RSC_IntersectRegionCuboidRigidBodyLinked::segment_bid)
        .def_readwrite(
            "relative_anchor", &RSC_IntersectRegionCuboidRigidBodyLinked::relative_anchor
        )
        .def_readwrite(
            "relative_orientation", &RSC_IntersectRegionCuboidRigidBodyLinked::relative_orientation
        )
        .def_readwrite("size", &RSC_IntersectRegionCuboidRigidBodyLinked::size);

    rspyRSC<RSC_IntersectRegionSphere, RSC_RecordBase>(m, "RSC_IntersectRegionSphere")
        .def(py::init<>())
        .def_readwrite("position", &RSC_IntersectRegionSphere::position)
        .def_readwrite("radius", &RSC_IntersectRegionSphere::radius);

    rspyRSC<RSC_IntersectRegionSphereRigidBodyLinked, RSC_RecordBase>(
        m, "RSC_IntersectRegionSphereRigidBodyLinked"
    )
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_IntersectRegionSphereRigidBodyLinked::structure_name)
        .def_readwrite("body_sid", &RSC_IntersectRegionSphereRigidBodyLinked::body_sid)
        .def_readwrite("segment_bid", &RSC_IntersectRegionSphereRigidBodyLinked::segment_bid)
        .def_readwrite(
            "relative_anchor", &RSC_IntersectRegionSphereRigidBodyLinked::relative_anchor
        )
        .def_readwrite("radius", &RSC_IntersectRegionSphereRigidBodyLinked::radius);
}
