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
        .def(
            "size",
            &RSPY_WorldStructureVoxelSignal::Expansion::size,
            "Number of expansion signal entries."
        )
        .def(
            "expansion",
            &RSPY_WorldStructureVoxelSignal::Expansion::expansion,
            "Scalar expansion signal per signal entry."
        );
    py::class_<RSPY_WorldStructureVoxelSignal::Hinge>(structure_voxel_signal, "Hinge")
        .def(
            "size", &RSPY_WorldStructureVoxelSignal::Hinge::size, "Number of hinge signal entries."
        )
        .def(
            "hinge",
            &RSPY_WorldStructureVoxelSignal::Hinge::hinge,
            "Scalar hinge signal per signal entry."
        );
    py::class_<RSPY_WorldStructureVoxelSignal::Prismatic>(structure_voxel_signal, "Prismatic")
        .def(
            "size",
            &RSPY_WorldStructureVoxelSignal::Prismatic::size,
            "Number of prismatic signal entries."
        )
        .def(
            "prismatic",
            &RSPY_WorldStructureVoxelSignal::Prismatic::prismatic,
            "Scalar prismatic signal per signal entry."
        );
    structure_voxel_signal
        .def(
            "expansion_signals",
            &RSPY_WorldStructureVoxelSignal::expansion_signals,
            "Expansion signals, one entry per signal index."
        )
        .def(
            "hinge_signals",
            &RSPY_WorldStructureVoxelSignal::hinge_signals,
            "Hinge signals, one entry per signal index."
        )
        .def(
            "prismatic_signals",
            &RSPY_WorldStructureVoxelSignal::prismatic_signals,
            "Prismatic signals, one entry per signal index."
        );

    auto camera_perspective_signal =
        py::class_<RSPY_WorldCameraPerspectiveSignal>(m, "RS_WorldCameraPerspectiveSignal");
    py::class_<RSPY_WorldCameraPerspectiveSignal::Position>(camera_perspective_signal, "Position")
        .def(
            "size",
            &RSPY_WorldCameraPerspectiveSignal::Position::size,
            "Number of position signal entries."
        )
        .def(
            "position",
            &RSPY_WorldCameraPerspectiveSignal::Position::position,
            "Camera position per signal entry."
        );
    py::class_<RSPY_WorldCameraPerspectiveSignal::Orientation>(
        camera_perspective_signal, "Orientation"
    )
        .def(
            "size",
            &RSPY_WorldCameraPerspectiveSignal::Orientation::size,
            "Number of orientation signal entries."
        )
        .def(
            "orientation",
            &RSPY_WorldCameraPerspectiveSignal::Orientation::orientation,
            "Camera orientation per signal entry."
        );
    py::class_<RSPY_WorldCameraPerspectiveSignal::Optics>(camera_perspective_signal, "Optics")
        .def(
            "size",
            &RSPY_WorldCameraPerspectiveSignal::Optics::size,
            "Number of optics signal entries."
        )
        .def(
            "focus_distance",
            &RSPY_WorldCameraPerspectiveSignal::Optics::focus_distance,
            "Scalar focus distance per signal entry."
        )
        .def(
            "field_of_view",
            &RSPY_WorldCameraPerspectiveSignal::Optics::field_of_view,
            "Scalar field of view per signal entry."
        );
    camera_perspective_signal
        .def(
            "position_signals",
            &RSPY_WorldCameraPerspectiveSignal::position_signals,
            "Camera position signals."
        )
        .def(
            "orientation_signals",
            &RSPY_WorldCameraPerspectiveSignal::orientation_signals,
            "Camera orientation signals."
        )
        .def(
            "optics_signals",
            &RSPY_WorldCameraPerspectiveSignal::optics_signals,
            "Camera optics signals (focus distance and field of view)."
        );

    auto camera_stereo_signal =
        py::class_<RSPY_WorldCameraStereoSignal>(m, "RS_WorldCameraStereoSignal");
    py::class_<RSPY_WorldCameraStereoSignal::Position>(camera_stereo_signal, "Position")
        .def(
            "size",
            &RSPY_WorldCameraStereoSignal::Position::size,
            "Number of position signal entries."
        )
        .def(
            "position",
            &RSPY_WorldCameraStereoSignal::Position::position_array,
            "Camera position per signal entry."
        );
    py::class_<RSPY_WorldCameraStereoSignal::Orientation>(camera_stereo_signal, "Orientation")
        .def(
            "size",
            &RSPY_WorldCameraStereoSignal::Orientation::size,
            "Number of orientation signal entries."
        )
        .def(
            "orientation",
            &RSPY_WorldCameraStereoSignal::Orientation::orientation_array,
            "Camera orientation per signal entry."
        );
    py::class_<RSPY_WorldCameraStereoSignal::Optics>(camera_stereo_signal, "Optics")
        .def(
            "size", &RSPY_WorldCameraStereoSignal::Optics::size, "Number of optics signal entries."
        )
        .def(
            "field_of_view",
            &RSPY_WorldCameraStereoSignal::Optics::field_of_view,
            "Scalar field of view per signal entry."
        );
    camera_stereo_signal
        .def(
            "left_position_signals",
            &RSPY_WorldCameraStereoSignal::left_position_signals,
            "Left eye position signals."
        )
        .def(
            "right_position_signals",
            &RSPY_WorldCameraStereoSignal::right_position_signals,
            "Right eye position signals."
        )
        .def(
            "left_orientation_signals",
            &RSPY_WorldCameraStereoSignal::left_orientation_signals,
            "Left eye orientation signals."
        )
        .def(
            "right_orientation_signals",
            &RSPY_WorldCameraStereoSignal::right_orientation_signals,
            "Right eye orientation signals."
        )
        .def(
            "left_optics_signals",
            &RSPY_WorldCameraStereoSignal::left_optics_signals,
            "Left eye frustum signals; the four field of view entries are ordered left, right, up, "
            "down."
        )
        .def(
            "right_optics_signals",
            &RSPY_WorldCameraStereoSignal::right_optics_signals,
            "Right eye frustum signals; the four field of view entries are ordered left, right, "
            "up, "
            "down."
        );

    auto ray_sensor_signal = py::class_<RSPY_WorldRaySensorSignal>(m, "RS_WorldRaySensorSignal");
    py::class_<RSPY_WorldRaySensorSignal::Position>(ray_sensor_signal, "Position")
        .def(
            "size", &RSPY_WorldRaySensorSignal::Position::size, "Number of position signal entries."
        )
        .def(
            "position",
            &RSPY_WorldRaySensorSignal::Position::position,
            "Sensor position per signal entry."
        );
    py::class_<RSPY_WorldRaySensorSignal::Orientation>(ray_sensor_signal, "Orientation")
        .def(
            "size",
            &RSPY_WorldRaySensorSignal::Orientation::size,
            "Number of orientation signal entries."
        )
        .def(
            "orientation",
            &RSPY_WorldRaySensorSignal::Orientation::orientation,
            "Sensor orientation per signal entry."
        );
    ray_sensor_signal
        .def(
            "position_signals",
            &RSPY_WorldRaySensorSignal::position_signals,
            "Sensor position signals."
        )
        .def(
            "orientation_signals",
            &RSPY_WorldRaySensorSignal::orientation_signals,
            "Sensor orientation signals."
        );

    auto force_signal = py::class_<RSPY_WorldForceSignal>(m, "RS_WorldForceSignal");
    py::class_<RSPY_WorldForceSignal::Force>(force_signal, "Force")
        .def("size", &RSPY_WorldForceSignal::Force::size, "Number of force signal entries.")
        .def("force", &RSPY_WorldForceSignal::Force::force, "3D force vector per signal entry.");
    force_signal.def(
        "force_signals",
        &RSPY_WorldForceSignal::force_signals,
        "Force signals, one 3D vector per signal index."
    );

    auto torque_signal = py::class_<RSPY_WorldTorqueSignal>(m, "RS_WorldTorqueSignal");
    py::class_<RSPY_WorldTorqueSignal::Torque>(torque_signal, "Torque")
        .def("size", &RSPY_WorldTorqueSignal::Torque::size, "Number of torque signal entries.")
        .def(
            "torque", &RSPY_WorldTorqueSignal::Torque::torque, "3D torque vector per signal entry."
        );
    torque_signal.def(
        "torque_signals",
        &RSPY_WorldTorqueSignal::torque_signals,
        "Torque signals, one 3D vector per signal index."
    );

    auto transform_signal = py::class_<RSPY_WorldTransformSignal>(m, "RS_WorldTransformSignal");
    py::class_<RSPY_WorldTransformSignal::Translation>(transform_signal, "Translation")
        .def(
            "size",
            &RSPY_WorldTransformSignal::Translation::size,
            "Number of translation signal entries."
        )
        .def(
            "translation",
            &RSPY_WorldTransformSignal::Translation::translation,
            "3D translation vector per signal entry."
        );
    py::class_<RSPY_WorldTransformSignal::Rotation>(transform_signal, "Rotation")
        .def(
            "size", &RSPY_WorldTransformSignal::Rotation::size, "Number of rotation signal entries."
        )
        .def(
            "rotation",
            &RSPY_WorldTransformSignal::Rotation::rotation,
            "4D rotation quaternion per signal entry."
        )
        .def(
            "pivot",
            &RSPY_WorldTransformSignal::Rotation::pivot,
            "3D pivot position the rotation turns about per signal entry."
        );
    transform_signal.def(
        "translation_signals",
        &RSPY_WorldTransformSignal::translation_signals,
        "Translation signals, one 3D vector per signal index."
    );
    transform_signal.def(
        "rotation_signals",
        &RSPY_WorldTransformSignal::rotation_signals,
        "Rotation signals, one quaternion and pivot per signal index."
    );
}
