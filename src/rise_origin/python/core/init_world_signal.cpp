// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "core/init_world_signal.h"

void initRSPYWorldSignal(py::module &m) {
    /*****************************************************************
     * Signals
     *****************************************************************/

    auto structure_voxel_signal =
        py::class_<RSPY_WorldStructureVoxelSignal>(m, "RS_WorldStructureVoxelSignal");
    py::class_<RSPY_WorldStructureVoxelSignal::Expansion>(structure_voxel_signal, "Expansion")
        .def("size", &RSPY_WorldStructureVoxelSignal::Expansion::size)
        .def("expansion", &RSPY_WorldStructureVoxelSignal::Expansion::expansion);
    py::class_<RSPY_WorldStructureVoxelSignal::Hinge>(structure_voxel_signal, "Hinge")
        .def("size", &RSPY_WorldStructureVoxelSignal::Hinge::size)
        .def("hinge", &RSPY_WorldStructureVoxelSignal::Hinge::hinge);
    py::class_<RSPY_WorldStructureVoxelSignal::Prismatic>(structure_voxel_signal, "Prismatic")
        .def("size", &RSPY_WorldStructureVoxelSignal::Prismatic::size)
        .def("prismatic", &RSPY_WorldStructureVoxelSignal::Prismatic::prismatic);
    structure_voxel_signal
        .def("expansion_signals", &RSPY_WorldStructureVoxelSignal::expansion_signals)
        .def("hinge_signals", &RSPY_WorldStructureVoxelSignal::hinge_signals)
        .def("prismatic_signals", &RSPY_WorldStructureVoxelSignal::prismatic_signals);

    auto camera_perspective_signal =
        py::class_<RSPY_WorldCameraPerspectiveSignal>(m, "RS_WorldCameraPerspectiveSignal");
    py::class_<RSPY_WorldCameraPerspectiveSignal::Position>(camera_perspective_signal, "Position")
        .def("size", &RSPY_WorldCameraPerspectiveSignal::Position::size)
        .def("position", &RSPY_WorldCameraPerspectiveSignal::Position::position);
    py::class_<RSPY_WorldCameraPerspectiveSignal::Orientation>(
        camera_perspective_signal, "Orientation"
    )
        .def("size", &RSPY_WorldCameraPerspectiveSignal::Orientation::size)
        .def("orientation", &RSPY_WorldCameraPerspectiveSignal::Orientation::orientation);
    py::class_<RSPY_WorldCameraPerspectiveSignal::Optics>(camera_perspective_signal, "Optics")
        .def("size", &RSPY_WorldCameraPerspectiveSignal::Optics::size)
        .def("focus_distance", &RSPY_WorldCameraPerspectiveSignal::Optics::focus_distance)
        .def("field_of_view", &RSPY_WorldCameraPerspectiveSignal::Optics::field_of_view);
    camera_perspective_signal
        .def("position_signals", &RSPY_WorldCameraPerspectiveSignal::position_signals)
        .def("orientation_signals", &RSPY_WorldCameraPerspectiveSignal::orientation_signals)
        .def("optics_signals", &RSPY_WorldCameraPerspectiveSignal::optics_signals);

    auto camera_stereo_signal =
        py::class_<RSPY_WorldCameraStereoSignal>(m, "RS_WorldCameraStereoSignal");
    py::class_<RSPY_WorldCameraStereoSignal::Position>(camera_stereo_signal, "Position")
        .def("size", &RSPY_WorldCameraStereoSignal::Position::size)
        .def("position", &RSPY_WorldCameraStereoSignal::Position::position_array);
    py::class_<RSPY_WorldCameraStereoSignal::Orientation>(camera_stereo_signal, "Orientation")
        .def("size", &RSPY_WorldCameraStereoSignal::Orientation::size)
        .def("orientation", &RSPY_WorldCameraStereoSignal::Orientation::orientation_array);
    py::class_<RSPY_WorldCameraStereoSignal::Optics>(camera_stereo_signal, "Optics")
        .def("size", &RSPY_WorldCameraStereoSignal::Optics::size)
        .def("field_of_view", &RSPY_WorldCameraStereoSignal::Optics::field_of_view);
    camera_stereo_signal
        .def("left_position_signals", &RSPY_WorldCameraStereoSignal::left_position_signals)
        .def("right_position_signals", &RSPY_WorldCameraStereoSignal::right_position_signals)
        .def("left_orientation_signals", &RSPY_WorldCameraStereoSignal::left_orientation_signals)
        .def("right_orientation_signals", &RSPY_WorldCameraStereoSignal::right_orientation_signals)
        .def("left_optics_signals", &RSPY_WorldCameraStereoSignal::left_optics_signals)
        .def("right_optics_signals", &RSPY_WorldCameraStereoSignal::right_optics_signals);

    auto ray_sensor_signal = py::class_<RSPY_WorldRaySensorSignal>(m, "RS_WorldRaySensorSignal");
    py::class_<RSPY_WorldRaySensorSignal::Position>(ray_sensor_signal, "Position")
        .def("size", &RSPY_WorldRaySensorSignal::Position::size)
        .def("position", &RSPY_WorldRaySensorSignal::Position::position);
    py::class_<RSPY_WorldRaySensorSignal::Orientation>(ray_sensor_signal, "Orientation")
        .def("size", &RSPY_WorldRaySensorSignal::Orientation::size)
        .def("orientation", &RSPY_WorldRaySensorSignal::Orientation::orientation);
    ray_sensor_signal.def("position_signals", &RSPY_WorldRaySensorSignal::position_signals)
        .def("orientation_signals", &RSPY_WorldRaySensorSignal::orientation_signals);

    auto force_signal = py::class_<RSPY_WorldForceSignal>(m, "RS_WorldForceSignal");
    py::class_<RSPY_WorldForceSignal::Force>(force_signal, "Force")
        .def("size", &RSPY_WorldForceSignal::Force::size)
        .def("force", &RSPY_WorldForceSignal::Force::force);
    force_signal.def("force_signals", &RSPY_WorldForceSignal::force_signals);

    auto torque_signal = py::class_<RSPY_WorldTorqueSignal>(m, "RS_WorldTorqueSignal");
    py::class_<RSPY_WorldTorqueSignal::Torque>(torque_signal, "Torque")
        .def("size", &RSPY_WorldTorqueSignal::Torque::size)
        .def("torque", &RSPY_WorldTorqueSignal::Torque::torque);
    torque_signal.def("torque_signals", &RSPY_WorldTorqueSignal::torque_signals);

    auto transform_signal = py::class_<RSPY_WorldTransformSignal>(m, "RS_WorldTransformSignal");
    py::class_<RSPY_WorldTransformSignal::Translation>(transform_signal, "Translation")
        .def("size", &RSPY_WorldTransformSignal::Translation::size)
        .def("translation", &RSPY_WorldTransformSignal::Translation::translation);
    py::class_<RSPY_WorldTransformSignal::Rotation>(transform_signal, "Rotation")
        .def("size", &RSPY_WorldTransformSignal::Rotation::size)
        .def("rotation", &RSPY_WorldTransformSignal::Rotation::rotation)
        .def("pivot", &RSPY_WorldTransformSignal::Rotation::pivot);
    transform_signal.def("translation_signals", &RSPY_WorldTransformSignal::translation_signals);
    transform_signal.def("rotation_signals", &RSPY_WorldTransformSignal::rotation_signals);
}
