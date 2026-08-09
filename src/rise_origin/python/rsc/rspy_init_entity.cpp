// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rsc/rspy_init_entity.h"

void initRSPYEntity(py::module &m) {
    /***************************************************************
     * Entities
     ***************************************************************/

    // --- Cosmic background ---
    rspyRSC<RSC_CosmicBackgroundBase, RSC_EntityBase>(m, "RSC_CosmicBackgroundBase");

    rspyRSC<RSC_CosmicBackgroundUniform, RSC_CosmicBackgroundBase>(m, "RSC_CosmicBackgroundUniform")
        .def(py::init<>())
        .def_readwrite("texture_reference", &RSC_CosmicBackgroundUniform::texture_reference)
        .def_readwrite("radiance", &RSC_CosmicBackgroundUniform::radiance)
        .def_readwrite("rotation_rad", &RSC_CosmicBackgroundUniform::rotation_rad);

    rspyRSC<RSC_CosmicBackgroundGradient, RSC_CosmicBackgroundBase>(
        m, "RSC_CosmicBackgroundGradient"
    )
        .def(py::init<>())
        .def_readwrite("zenith_color", &RSC_CosmicBackgroundGradient::zenith_color)
        .def_readwrite("horizon_color", &RSC_CosmicBackgroundGradient::horizon_color)
        .def_readwrite("sun_direction", &RSC_CosmicBackgroundGradient::sun_direction)
        .def_readwrite("sun_color", &RSC_CosmicBackgroundGradient::sun_color)
        .def_readwrite("sun_irradiance", &RSC_CosmicBackgroundGradient::sun_irradiance)
        .def_readwrite("sun_angular_radius", &RSC_CosmicBackgroundGradient::sun_angular_radius);

    rspyRSC<RSC_CosmicBackgroundHosekWilkie, RSC_CosmicBackgroundBase>(
        m, "RSC_CosmicBackgroundHosekWilkie"
    )
        .def(py::init<>())
        .def_readwrite("sun_direction", &RSC_CosmicBackgroundHosekWilkie::sun_direction)
        .def_readwrite("sun_color", &RSC_CosmicBackgroundHosekWilkie::sun_color)
        .def_readwrite("sun_irradiance", &RSC_CosmicBackgroundHosekWilkie::sun_irradiance)
        .def_readwrite("sun_angular_radius", &RSC_CosmicBackgroundHosekWilkie::sun_angular_radius)
        .def_readwrite("turbidity", &RSC_CosmicBackgroundHosekWilkie::turbidity)
        .def_readwrite("ground_color", &RSC_CosmicBackgroundHosekWilkie::ground_color)
        .def_readwrite("radiance_scale", &RSC_CosmicBackgroundHosekWilkie::radiance_scale);

    // --- Floor ---
    rspyRSC<RSC_FloorBase, RSC_EntityBase>(m, "RSC_FloorBase")
        .def_readwrite("material_reference", &RSC_FloorBase::material_reference);

    rspyRSC<RSC_FloorFlat, RSC_FloorBase>(m, "RSC_FloorFlat")
        .def(py::init<>())
        .def_readwrite("height", &RSC_FloorFlat::height)
        .def_readwrite("uv_scale", &RSC_FloorFlat::uv_scale);

    auto cls_floor_array = rspyRSC<RSC_FloorArray, RSC_FloorBase>(m, "RSC_FloorArray");
    cls_floor_array.def(py::init<>())
        .def_readwrite("x_size", &RSC_FloorArray::x_size)
        .def_readwrite("y_size", &RSC_FloorArray::y_size)
        .def_readwrite("x_values", &RSC_FloorArray::x_values)
        .def_readwrite("y_values", &RSC_FloorArray::y_values)
        .def_readwrite("is_x_repeat", &RSC_FloorArray::is_x_repeat)
        .def_readwrite("is_y_repeat", &RSC_FloorArray::is_y_repeat);
    rspyBindArray(cls_floor_array, "height", &RSC_FloorArray::height);

    // --- Structure (voxel) ---
    py::enum_<RSE_StructureExpansionControlMode>(m, "RSE_StructureExpansionControlMode")
        .value(
            "RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION",
            RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION
        )
        .value(
            "RSE_STRUCTURE_EXPANSION_CONTROL_MODE_RATE", RSE_STRUCTURE_EXPANSION_CONTROL_MODE_RATE
        )
        .export_values();

    py::enum_<RSE_StructureVoxelConstraintType>(m, "RSE_StructureVoxelConstraintType")
        .value(
            "RSE_STRUCTURE_VOXEL_BALL_AND_SOCKET_JOINT", RSE_STRUCTURE_VOXEL_BALL_AND_SOCKET_JOINT
        )
        .value("RSE_STRUCTURE_VOXEL_FIXED_JOINT", RSE_STRUCTURE_VOXEL_FIXED_JOINT)
        .value("RSE_STRUCTURE_VOXEL_HINGE_JOINT", RSE_STRUCTURE_VOXEL_HINGE_JOINT)
        .value("RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT", RSE_STRUCTURE_VOXEL_PRISMATIC_JOINT)
        .value("RSE_STRUCTURE_VOXEL_NULL_JOINT", RSE_STRUCTURE_VOXEL_NULL_JOINT)
        .export_values();

    py::enum_<RSE_StructureVoxelHingeControlMode>(m, "RSE_StructureVoxelHingeControlMode")
        .value(
            "RSE_STRUCTURE_VOXEL_HINGE_CONTROL_BASIC_POSITION",
            RSE_STRUCTURE_VOXEL_HINGE_CONTROL_BASIC_POSITION
        )
        .value(
            "RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION",
            RSE_STRUCTURE_VOXEL_HINGE_CONTROL_PD_POSITION
        )
        .value("RSE_STRUCTURE_VOXEL_HINGE_CONTROL_TORQUE", RSE_STRUCTURE_VOXEL_HINGE_CONTROL_TORQUE)
        .value(
            "RSE_STRUCTURE_VOXEL_HINGE_CONTROL_VELOCITY", RSE_STRUCTURE_VOXEL_HINGE_CONTROL_VELOCITY
        )
        .export_values();

    py::enum_<RSE_StructureVoxelPrismaticControlMode>(m, "RSE_StructureVoxelPrismaticControlMode")
        .value(
            "RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_BASIC_POSITION",
            RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_BASIC_POSITION
        )
        .value(
            "RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_PD_POSITION",
            RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_PD_POSITION
        )
        .value(
            "RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_FORCE",
            RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_FORCE
        )
        .value(
            "RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_VELOCITY",
            RSE_STRUCTURE_VOXEL_PRISMATIC_CONTROL_VELOCITY
        )
        .export_values();

    py::enum_<RSE_StructureVoxelRecordVoxelRange>(m, "RSE_StructureVoxelRecordVoxelRange")
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_NONE",
            RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_NONE
        )
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_ALL", RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_ALL
        )
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_RANDOM",
            RSE_STRUCTURE_VOXEL_RECORD_VOXEL_RANGE_RANDOM
        )
        .export_values();

    py::enum_<RSE_StructureVoxelRecordLinkRange>(m, "RSE_StructureVoxelRecordLinkRange")
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_NONE", RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_NONE
        )
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_ALL", RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_ALL
        )
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_RANDOM",
            RSE_STRUCTURE_VOXEL_RECORD_LINK_RANGE_RANDOM
        )
        .export_values();

    py::enum_<RSE_StructureVoxelRecordRigidBodyRange>(m, "RSE_StructureVoxelRecordRigidBodyRange")
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_NONE",
            RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_NONE
        )
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_ALL",
            RSE_STRUCTURE_VOXEL_RECORD_RIGID_BODY_RANGE_ALL
        )
        .export_values();

    py::enum_<RSE_StructureVoxelRecordJointRange>(m, "RSE_StructureVoxelRecordJointRange")
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_NONE",
            RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_NONE
        )
        .value(
            "RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_ALL", RSE_STRUCTURE_VOXEL_RECORD_JOINT_RANGE_ALL
        )
        .export_values();

    auto cls_body = rspyRSC<RSC_StructureVoxelBody, RSC_Base>(m, "RSC_StructureVoxelBody");
    cls_body.def(py::init<>())
        .def_readwrite(
            "relative_origin_position", &RSC_StructureVoxelBody::relative_origin_position
        )
        .def_readwrite("relative_orientation", &RSC_StructureVoxelBody::relative_orientation)
        .def_readwrite("x_voxels", &RSC_StructureVoxelBody::x_voxels)
        .def_readwrite("y_voxels", &RSC_StructureVoxelBody::y_voxels)
        .def_readwrite("z_voxels", &RSC_StructureVoxelBody::z_voxels)
        .def_readwrite("voxel_size", &RSC_StructureVoxelBody::voxel_size);
    rspyBindArray(
        cls_body, "material_reference_sid", &RSC_StructureVoxelBody::material_reference_sid
    );
    rspyBindArray(
        cls_body,
        "material_optical_reference_mid",
        &RSC_StructureVoxelBody::material_optical_reference_mid
    );
    rspyBindArray(cls_body, "segment_bid", &RSC_StructureVoxelBody::segment_bid);
    rspyBindArray(cls_body, "expansion_signal_sid", &RSC_StructureVoxelBody::expansion_signal_sid);
    rspyBindArray(
        cls_body, "expansion_control_mode", &RSC_StructureVoxelBody::expansion_control_mode
    );
    rspyBindArray(cls_body, "link", &RSC_StructureVoxelBody::link);

    rspyRSC<RSC_StructureVoxelConstraint, RSC_Base>(m, "RSC_StructureVoxelConstraint")
        .def(py::init<>())
        .def_readwrite("type", &RSC_StructureVoxelConstraint::type)
        .def_readwrite("is_collision_enabled", &RSC_StructureVoxelConstraint::is_collision_enabled)
        .def_readwrite("a_body_sid", &RSC_StructureVoxelConstraint::a_body_sid)
        .def_readwrite("a_segment_bid", &RSC_StructureVoxelConstraint::a_segment_bid)
        .def_readwrite("b_body_sid", &RSC_StructureVoxelConstraint::b_body_sid)
        .def_readwrite("b_segment_bid", &RSC_StructureVoxelConstraint::b_segment_bid)
        .def_readwrite("relative_anchor", &RSC_StructureVoxelConstraint::relative_anchor)
        .def_readwrite("hinge_control_mode", &RSC_StructureVoxelConstraint::hinge_control_mode)
        .def_readwrite("hinge_relative_axis", &RSC_StructureVoxelConstraint::hinge_relative_axis)
        .def_readwrite("hinge_min", &RSC_StructureVoxelConstraint::hinge_min)
        .def_readwrite("hinge_max", &RSC_StructureVoxelConstraint::hinge_max)
        .def_readwrite(
            "hinge_position_control_kp", &RSC_StructureVoxelConstraint::hinge_position_control_kp
        )
        .def_readwrite(
            "hinge_position_control_kd", &RSC_StructureVoxelConstraint::hinge_position_control_kd
        )
        .def_readwrite("hinge_max_speed", &RSC_StructureVoxelConstraint::hinge_max_speed)
        .def_readwrite("hinge_max_torque", &RSC_StructureVoxelConstraint::hinge_max_torque)
        .def_readwrite("hinge_signal_sid", &RSC_StructureVoxelConstraint::hinge_signal_sid)
        .def_readwrite(
            "prismatic_control_mode", &RSC_StructureVoxelConstraint::prismatic_control_mode
        )
        .def_readwrite(
            "prismatic_relative_axis", &RSC_StructureVoxelConstraint::prismatic_relative_axis
        )
        .def_readwrite("prismatic_min", &RSC_StructureVoxelConstraint::prismatic_min)
        .def_readwrite("prismatic_max", &RSC_StructureVoxelConstraint::prismatic_max)
        .def_readwrite(
            "prismatic_position_control_kp",
            &RSC_StructureVoxelConstraint::prismatic_position_control_kp
        )
        .def_readwrite(
            "prismatic_position_control_kd",
            &RSC_StructureVoxelConstraint::prismatic_position_control_kd
        )
        .def_readwrite("prismatic_max_speed", &RSC_StructureVoxelConstraint::prismatic_max_speed)
        .def_readwrite("prismatic_max_force", &RSC_StructureVoxelConstraint::prismatic_max_force)
        .def_readwrite("prismatic_signal_sid", &RSC_StructureVoxelConstraint::prismatic_signal_sid);

    rspyRSC<RSC_StructureVoxelRecord, RSC_RecordBase>(m, "RSC_StructureVoxelRecord")
        .def(py::init<>())
        .def_readwrite("voxel_record_range", &RSC_StructureVoxelRecord::voxel_record_range)
        .def_readwrite("link_record_range", &RSC_StructureVoxelRecord::link_record_range)
        .def_readwrite(
            "rigid_body_record_range", &RSC_StructureVoxelRecord::rigid_body_record_range
        )
        .def_readwrite("joint_record_range", &RSC_StructureVoxelRecord::joint_record_range)
        .def_readwrite("voxel_sample_num", &RSC_StructureVoxelRecord::voxel_sample_num)
        .def_readwrite("link_sample_num", &RSC_StructureVoxelRecord::link_sample_num);

    rspyRSC<RSC_StructureVoxelSignal, RSC_SignalBase>(m, "RSC_StructureVoxelSignal")
        .def(py::init<>())
        .def_readwrite("expansion_signal_num", &RSC_StructureVoxelSignal::expansion_signal_num)
        .def_readwrite("hinge_signal_num", &RSC_StructureVoxelSignal::hinge_signal_num)
        .def_readwrite("prismatic_signal_num", &RSC_StructureVoxelSignal::prismatic_signal_num);

    auto structure_base = rspyRSC<RSC_StructureBase, RSC_EntityBase>(m, "RSC_StructureBase");
    structure_base.def_readwrite("origin_position", &RSC_StructureBase::origin_position)
        .def_readwrite("orientation", &RSC_StructureBase::orientation);
    rspyBindVector(structure_base, "material_references", &RSC_StructureBase::material_references);
    rspyBindVector(
        structure_base,
        "material_optical_references",
        &RSC_StructureBase::material_optical_references
    );

    rspyRSC<RSC_StructureVoxel, RSC_StructureBase>(m, "RSC_StructureVoxel")
        .def(py::init<>())
        .def_readwrite("is_fixed", &RSC_StructureVoxel::is_fixed)
        .def_readwrite("bodies", &RSC_StructureVoxel::bodies)
        .def_readwrite("constraints", &RSC_StructureVoxel::constraints)
        .def_readwrite("record", &RSC_StructureVoxel::record)
        .def_readwrite("signal", &RSC_StructureVoxel::signal);

    // --- Camera ---
    rspyRSC<RSC_CameraPerspectiveRecord, RSC_RecordBase>(m, "RSC_CameraPerspectiveRecord")
        .def(py::init<>())
        .def_readwrite("has_rgb", &RSC_CameraPerspectiveRecord::has_rgb)
        .def_readwrite("has_depth", &RSC_CameraPerspectiveRecord::has_depth)
        .def_readwrite("has_normal", &RSC_CameraPerspectiveRecord::has_normal)
        .def_readwrite("has_motion", &RSC_CameraPerspectiveRecord::has_motion)
        .def_readwrite("has_entity_info", &RSC_CameraPerspectiveRecord::has_entity_info)
        .def_readwrite("is_depth_hyperbolic", &RSC_CameraPerspectiveRecord::is_depth_hyperbolic)
        .def_readwrite("hyperbolic_near_depth", &RSC_CameraPerspectiveRecord::hyperbolic_near_depth)
        .def_readwrite("hyperbolic_far_depth", &RSC_CameraPerspectiveRecord::hyperbolic_far_depth);

    rspyRSC<RSC_CameraPerspectiveSignal, RSC_SignalBase>(m, "RSC_CameraPerspectiveSignal")
        .def(py::init<>())
        .def_readwrite("position_signal_num", &RSC_CameraPerspectiveSignal::position_signal_num)
        .def_readwrite(
            "orientation_signal_num", &RSC_CameraPerspectiveSignal::orientation_signal_num
        )
        .def_readwrite("optics_signal_num", &RSC_CameraPerspectiveSignal::optics_signal_num);

    rspyRSC<RSC_CameraBase, RSC_EntityBase>(m, "RSC_CameraBase")
        .def_readwrite("saturation_radiance", &RSC_CameraBase::saturation_radiance);

    rspyRSC<RSC_CameraPerspectiveBase, RSC_CameraBase>(m, "RSC_CameraPerspectiveBase")
        .def_readwrite("image_width", &RSC_CameraPerspectiveBase::image_width)
        .def_readwrite("image_height", &RSC_CameraPerspectiveBase::image_height)
        .def_readwrite("samples_per_pixel", &RSC_CameraPerspectiveBase::samples_per_pixel)
        .def_readwrite("max_depth", &RSC_CameraPerspectiveBase::max_depth)
        .def_readwrite("field_of_view", &RSC_CameraPerspectiveBase::field_of_view)
        .def_readwrite("defocus_angle", &RSC_CameraPerspectiveBase::defocus_angle)
        .def_readwrite("focus_distance", &RSC_CameraPerspectiveBase::focus_distance)
        .def_readwrite("record", &RSC_CameraPerspectiveBase::record)
        .def_readwrite("signal", &RSC_CameraPerspectiveBase::signal);

    rspyRSC<RSC_CameraPerspectiveFree, RSC_CameraPerspectiveBase>(m, "RSC_CameraPerspectiveFree")
        .def(py::init<>())
        .def_readwrite("origin_position", &RSC_CameraPerspectiveFree::origin_position)
        .def_readwrite("orientation", &RSC_CameraPerspectiveFree::orientation);

    rspyRSC<RSC_CameraPerspectiveRigidBodyLinked, RSC_CameraPerspectiveBase>(
        m, "RSC_CameraPerspectiveRigidBodyLinked"
    )
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_CameraPerspectiveRigidBodyLinked::structure_name)
        .def_readwrite("body_sid", &RSC_CameraPerspectiveRigidBodyLinked::body_sid)
        .def_readwrite("segment_bid", &RSC_CameraPerspectiveRigidBodyLinked::segment_bid)
        .def_readwrite("relative_anchor", &RSC_CameraPerspectiveRigidBodyLinked::relative_anchor)
        .def_readwrite(
            "relative_orientation", &RSC_CameraPerspectiveRigidBodyLinked::relative_orientation
        );

    rspyRSC<RSC_CameraStereoRecord, RSC_RecordBase>(m, "RSC_CameraStereoRecord")
        .def(py::init<>())
        .def_readwrite("has_rgb", &RSC_CameraStereoRecord::has_rgb)
        .def_readwrite("has_depth", &RSC_CameraStereoRecord::has_depth)
        .def_readwrite("has_normal", &RSC_CameraStereoRecord::has_normal)
        .def_readwrite("has_motion", &RSC_CameraStereoRecord::has_motion)
        .def_readwrite("has_entity_info", &RSC_CameraStereoRecord::has_entity_info)
        .def_readwrite("is_depth_hyperbolic", &RSC_CameraStereoRecord::is_depth_hyperbolic)
        .def_readwrite("hyperbolic_near_depth", &RSC_CameraStereoRecord::hyperbolic_near_depth)
        .def_readwrite("hyperbolic_far_depth", &RSC_CameraStereoRecord::hyperbolic_far_depth);

    rspyRSC<RSC_CameraStereoSignal, RSC_SignalBase>(m, "RSC_CameraStereoSignal").def(py::init<>());

    rspyRSC<RSC_CameraStereoBase, RSC_CameraBase>(m, "RSC_CameraStereoBase")
        .def_readwrite("image_width", &RSC_CameraStereoBase::image_width)
        .def_readwrite("image_height", &RSC_CameraStereoBase::image_height)
        .def_readwrite("samples_per_pixel", &RSC_CameraStereoBase::samples_per_pixel)
        .def_readwrite("max_depth", &RSC_CameraStereoBase::max_depth)
        .def_readwrite("interpupillary_distance", &RSC_CameraStereoBase::interpupillary_distance)
        .def_readwrite("convergence_distance", &RSC_CameraStereoBase::convergence_distance)
        .def_readwrite("angle_left", &RSC_CameraStereoBase::angle_left)
        .def_readwrite("angle_right", &RSC_CameraStereoBase::angle_right)
        .def_readwrite("angle_up", &RSC_CameraStereoBase::angle_up)
        .def_readwrite("angle_down", &RSC_CameraStereoBase::angle_down)
        .def_readwrite("record", &RSC_CameraStereoBase::record)
        .def_readwrite("signal", &RSC_CameraStereoBase::signal);

    rspyRSC<RSC_CameraStereoFree, RSC_CameraStereoBase>(m, "RSC_CameraStereoFree")
        .def(py::init<>())
        .def_readwrite("origin_position", &RSC_CameraStereoFree::origin_position)
        .def_readwrite("orientation", &RSC_CameraStereoFree::orientation);

    rspyRSC<RSC_CameraStereoRigidBodyLinked, RSC_CameraStereoBase>(
        m, "RSC_CameraStereoRigidBodyLinked"
    )
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_CameraStereoRigidBodyLinked::structure_name)
        .def_readwrite("body_sid", &RSC_CameraStereoRigidBodyLinked::body_sid)
        .def_readwrite("segment_bid", &RSC_CameraStereoRigidBodyLinked::segment_bid)
        .def_readwrite("relative_anchor", &RSC_CameraStereoRigidBodyLinked::relative_anchor)
        .def_readwrite(
            "relative_orientation", &RSC_CameraStereoRigidBodyLinked::relative_orientation
        );

    rspyRSC<RSC_RaySensorRecord, RSC_RecordBase>(m, "RSC_RaySensorRecord")
        .def(py::init<>())
        .def_readwrite("has_hit", &RSC_RaySensorRecord::has_hit)
        .def_readwrite("has_entity_info", &RSC_RaySensorRecord::has_entity_info)
        .def_readwrite("has_entity_detail", &RSC_RaySensorRecord::has_entity_detail)
        .def_readwrite("has_material", &RSC_RaySensorRecord::has_material)
        .def_readwrite("has_normal", &RSC_RaySensorRecord::has_normal)
        .def_readwrite("has_velocity", &RSC_RaySensorRecord::has_velocity);

    rspyRSC<RSC_RaySensorSignal, RSC_SignalBase>(m, "RSC_RaySensorSignal")
        .def(py::init<>())
        .def_readwrite("position_signal_num", &RSC_RaySensorSignal::position_signal_num)
        .def_readwrite("orientation_signal_num", &RSC_RaySensorSignal::orientation_signal_num);

    auto ray_sensor_base = rspyRSC<RSC_RaySensorBase, RSC_EntityBase>(m, "RSC_RaySensorBase");
    ray_sensor_base.def_readwrite("max_range", &RSC_RaySensorBase::max_range)
        .def_readwrite("record", &RSC_RaySensorBase::record)
        .def_readwrite("signal", &RSC_RaySensorBase::signal);
    rspyBindVector(ray_sensor_base, "ray_directions", &RSC_RaySensorBase::ray_directions);

    rspyRSC<RSC_RaySensorFree, RSC_RaySensorBase>(m, "RSC_RaySensorFree")
        .def(py::init<>())
        .def_readwrite("origin_position", &RSC_RaySensorFree::origin_position)
        .def_readwrite("orientation", &RSC_RaySensorFree::orientation);

    rspyRSC<RSC_RaySensorRigidBodyLinked, RSC_RaySensorBase>(m, "RSC_RaySensorRigidBodyLinked")
        .def(py::init<>())
        .def_readwrite("structure_name", &RSC_RaySensorRigidBodyLinked::structure_name)
        .def_readwrite("body_sid", &RSC_RaySensorRigidBodyLinked::body_sid)
        .def_readwrite("segment_bid", &RSC_RaySensorRigidBodyLinked::segment_bid)
        .def_readwrite("relative_anchor", &RSC_RaySensorRigidBodyLinked::relative_anchor)
        .def_readwrite("relative_orientation", &RSC_RaySensorRigidBodyLinked::relative_orientation);
}
