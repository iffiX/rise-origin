// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "core/rspy_init_world_record.h"

void initRSPYWorldRecord(py::module &m) {
    /*****************************************************************
     * Records
     *****************************************************************/

    auto structure_voxel_record =
        py::class_<RSPY_WorldStructureVoxelRecord>(m, "RS_WorldStructureVoxelRecord")
            .def("step", &RSPY_WorldStructureVoxelRecord::step, "Simulation step of this record.")
            .def("time", &RSPY_WorldStructureVoxelRecord::time, "Simulation time of this record.")
            .def(
                "structure_gid",
                &RSPY_WorldStructureVoxelRecord::structure_gid,
                "Structure global id."
            )
            .def_property_readonly(
                "voxels",
                &RSPY_WorldStructureVoxelRecord::voxels,
                "Voxel records array view of this structure."
            )
            .def_property_readonly(
                "links",
                &RSPY_WorldStructureVoxelRecord::links,
                "Link records array view of this structure."
            )
            .def_property_readonly(
                "rigid_bodies",
                &RSPY_WorldStructureVoxelRecord::rigid_bodies,
                "Rigid body records array view of this structure."
            )
            .def_property_readonly(
                "joints",
                &RSPY_WorldStructureVoxelRecord::joints,
                "Joint records array view of this structure."
            );

    py::class_<RSPY_WorldStructureVoxelRecord::Voxel>(structure_voxel_record, "Voxel")
        .def(
            "size",
            &RSPY_WorldStructureVoxelRecord::Voxel::size,
            "Number of voxels in this array view."
        )
        .def(
            "body_sid",
            &RSPY_WorldStructureVoxelRecord::Voxel::body_sid,
            "Body index in the structure."
        )
        .def(
            "voxel_sid",
            &RSPY_WorldStructureVoxelRecord::Voxel::voxel_sid,
            "Voxel index in the structure."
        )
        .def(
            "segment_bid",
            &RSPY_WorldStructureVoxelRecord::Voxel::segment_bid,
            "Body segment id in the body."
        )
        .def(
            "index_x",
            &RSPY_WorldStructureVoxelRecord::Voxel::index_x,
            "X index of this voxel in the body."
        )
        .def(
            "index_y",
            &RSPY_WorldStructureVoxelRecord::Voxel::index_y,
            "Y index of this voxel in the body."
        )
        .def(
            "index_z",
            &RSPY_WorldStructureVoxelRecord::Voxel::index_z,
            "Z index of this voxel in the body."
        )
        .def(
            "connectivity",
            &RSPY_WorldStructureVoxelRecord::Voxel::connectivity,
            "Current connectivity, lowest bit is first edge and highest bit is last edge."
        )
        .def(
            "is_surface",
            &RSPY_WorldStructureVoxelRecord::Voxel::is_surface,
            "Whether this voxel is a surface voxel."
        )
        .def(
            "is_rigid",
            &RSPY_WorldStructureVoxelRecord::Voxel::is_rigid,
            "Whether this voxel is a rigid voxel."
        )
        .def("material", &RSPY_WorldStructureVoxelRecord::Voxel::material, "Material.")
        .def("expansion", &RSPY_WorldStructureVoxelRecord::Voxel::expansion, "Expansion.")
        .def(
            "orientation",
            &RSPY_WorldStructureVoxelRecord::Voxel::orientation,
            "Orientation quaternion."
        )
        .def("position", &RSPY_WorldStructureVoxelRecord::Voxel::position, "CoM position.")
        .def(
            "ppp_offset",
            &RSPY_WorldStructureVoxelRecord::Voxel::ppp_offset,
            "PPP corner (upper-right-back) offset in local coordinate system (LCS)."
        )
        .def(
            "nnn_offset",
            &RSPY_WorldStructureVoxelRecord::Voxel::nnn_offset,
            "NNN corner (lower-left-front) offset in local coordinate system (LCS)."
        )
        .def(
            "linear_velocity",
            &RSPY_WorldStructureVoxelRecord::Voxel::linear_velocity,
            "Linear velocity."
        )
        .def(
            "angular_velocity",
            &RSPY_WorldStructureVoxelRecord::Voxel::angular_velocity,
            "Angular velocity."
        )
        .def(
            "poissons_strain",
            &RSPY_WorldStructureVoxelRecord::Voxel::poissons_strain,
            "Poissons strain."
        );

    py::class_<RSPY_WorldStructureVoxelRecord::Link>(structure_voxel_record, "Link")
        .def(
            "size",
            &RSPY_WorldStructureVoxelRecord::Link::size,
            "Number of links in this array view."
        )
        .def(
            "voxel_neg_sid",
            &RSPY_WorldStructureVoxelRecord::Link::voxel_neg_sid,
            "Negative voxel index in the structure."
        )
        .def(
            "voxel_pos_sid",
            &RSPY_WorldStructureVoxelRecord::Link::voxel_pos_sid,
            "Positive voxel index in the structure."
        )
        .def(
            "is_failed",
            &RSPY_WorldStructureVoxelRecord::Link::is_failed,
            "Whether the link is failed."
        )
        .def("strain", &RSPY_WorldStructureVoxelRecord::Link::strain, "Axial strain.")
        .def("stress", &RSPY_WorldStructureVoxelRecord::Link::stress, "Axial stress.");

    py::class_<RSPY_WorldStructureVoxelRecord::RigidBody>(structure_voxel_record, "RigidBody")
        .def(
            "size",
            &RSPY_WorldStructureVoxelRecord::RigidBody::size,
            "Number of rigid bodies in this array view."
        )
        .def(
            "rigid_body_sid",
            &RSPY_WorldStructureVoxelRecord::RigidBody::rigid_body_sid,
            "Rigid body index in the structure."
        )
        .def(
            "body_sid",
            &RSPY_WorldStructureVoxelRecord::RigidBody::body_sid,
            "Body index in the structure."
        )
        .def(
            "segment_bid",
            &RSPY_WorldStructureVoxelRecord::RigidBody::segment_bid,
            "Body segment id in the body."
        )
        .def("mass", &RSPY_WorldStructureVoxelRecord::RigidBody::mass, "Body mass.")
        .def(
            "orientation",
            &RSPY_WorldStructureVoxelRecord::RigidBody::orientation,
            "Orientation quaternion."
        )
        .def("com", &RSPY_WorldStructureVoxelRecord::RigidBody::com, "CoM position.")
        .def(
            "linear_velocity",
            &RSPY_WorldStructureVoxelRecord::RigidBody::linear_velocity,
            "Linear velocity."
        )
        .def(
            "angular_velocity",
            &RSPY_WorldStructureVoxelRecord::RigidBody::angular_velocity,
            "Angular velocity."
        );

    py::class_<RSPY_WorldStructureVoxelRecord::Joint>(structure_voxel_record, "Joint")
        .def(
            "size",
            &RSPY_WorldStructureVoxelRecord::Joint::size,
            "Number of joints in this array view."
        )
        .def(
            "rigid_body_a_sid",
            &RSPY_WorldStructureVoxelRecord::Joint::rigid_body_a_sid,
            "Rigid body A structure index."
        )
        .def(
            "rigid_body_b_sid",
            &RSPY_WorldStructureVoxelRecord::Joint::rigid_body_b_sid,
            "Rigid body B structure index."
        )
        .def("type", &RSPY_WorldStructureVoxelRecord::Joint::type, "Record type.")
        .def(
            "position",
            &RSPY_WorldStructureVoxelRecord::Joint::position,
            "Position (anchor midpoint)."
        )
        .def(
            "axis",
            &RSPY_WorldStructureVoxelRecord::Joint::axis,
            "Axis (hinge: rotation axis, prismatic: slide axis)."
        )
        .def(
            "hinge_signal_sid",
            &RSPY_WorldStructureVoxelRecord::Joint::hinge_signal_sid,
            "Rotation angle signal id (only valid for hinge joint)."
        )
        .def(
            "hinge_min",
            &RSPY_WorldStructureVoxelRecord::Joint::hinge_min,
            "Minimum angle (unit: rad, only valid for hinge joint)."
        )
        .def(
            "hinge_max",
            &RSPY_WorldStructureVoxelRecord::Joint::hinge_max,
            "Maximum angle (unit: rad, only valid for hinge joint)."
        )
        .def(
            "hinge_angle",
            &RSPY_WorldStructureVoxelRecord::Joint::hinge_angle,
            "Current angle (unit: rad, only valid for hinge joint)."
        )
        .def(
            "hinge_torque",
            &RSPY_WorldStructureVoxelRecord::Joint::hinge_torque,
            "Current torque (unit: N-m, only valid for hinge joint)."
        )
        .def(
            "prismatic_signal_sid",
            &RSPY_WorldStructureVoxelRecord::Joint::prismatic_signal_sid,
            "Translation rate signal id (only valid for prismatic joint)."
        )
        .def(
            "prismatic_min",
            &RSPY_WorldStructureVoxelRecord::Joint::prismatic_min,
            "Minimum distance (unit: m, only valid for prismatic joint)."
        )
        .def(
            "prismatic_max",
            &RSPY_WorldStructureVoxelRecord::Joint::prismatic_max,
            "Maximum distance (unit: m, only valid for prismatic joint)."
        )
        .def(
            "prismatic_distance",
            &RSPY_WorldStructureVoxelRecord::Joint::prismatic_distance,
            "Current distance (unit: m, only valid for prismatic joint)."
        )
        .def(
            "prismatic_force",
            &RSPY_WorldStructureVoxelRecord::Joint::prismatic_force,
            "Current force (unit: N, only valid for prismatic joint)."
        );

    auto camera_perspective_record =
        py::class_<RSPY_WorldCameraPerspectiveRecord>(m, "RS_WorldCameraPerspectiveRecord")
            .def(
                "step", &RSPY_WorldCameraPerspectiveRecord::step, "Simulation step of this record."
            )
            .def(
                "time", &RSPY_WorldCameraPerspectiveRecord::time, "Simulation time of this record."
            )
            .def("camera_gid", &RSPY_WorldCameraPerspectiveRecord::camera_gid, "Camera global id.")
            .def(
                "structure_gid",
                &RSPY_WorldCameraPerspectiveRecord::structure_gid,
                "Linked structure gid."
            )
            .def(
                "rigid_body_sid",
                &RSPY_WorldCameraPerspectiveRecord::rigid_body_sid,
                "Linked rigid body sid."
            )
            .def("width", &RSPY_WorldCameraPerspectiveRecord::width, "Image width in pixels.")
            .def("height", &RSPY_WorldCameraPerspectiveRecord::height, "Image height in pixels.")
            .def(
                "is_valid",
                &RSPY_WorldCameraPerspectiveRecord::is_valid,
                "Whether this record is valid."
            )
            .def_property_readonly(
                "rgb", &RSPY_WorldCameraPerspectiveRecord::rgb, "RGB pixel records of this camera."
            )
            .def_property_readonly(
                "depth",
                &RSPY_WorldCameraPerspectiveRecord::depth,
                "Depth pixel records of this camera."
            )
            .def_property_readonly(
                "normal",
                &RSPY_WorldCameraPerspectiveRecord::normal,
                "Normal pixel records of this camera."
            )
            .def_property_readonly(
                "motion",
                &RSPY_WorldCameraPerspectiveRecord::motion,
                "Motion pixel records of this camera."
            )
            .def_property_readonly(
                "entity_info",
                &RSPY_WorldCameraPerspectiveRecord::entity_info,
                "Entity info pixel records of this camera."
            )
            .def_property_readonly(
                "state",
                &RSPY_WorldCameraPerspectiveRecord::state,
                "Camera state records of this camera."
            );

    py::class_<RSPY_WorldCameraPerspectiveRecord::RGB>(camera_perspective_record, "RGB")
        .def("width", &RSPY_WorldCameraPerspectiveRecord::RGB::width, "Image width in pixels.")
        .def("height", &RSPY_WorldCameraPerspectiveRecord::RGB::height, "Image height in pixels.")
        .def(
            "pixel",
            &RSPY_WorldCameraPerspectiveRecord::RGB::pixel,
            "Pixel data as a [height, width, 3] uint8 ndarray."
        );

    py::class_<RSPY_WorldCameraPerspectiveRecord::Depth>(camera_perspective_record, "Depth")
        .def("width", &RSPY_WorldCameraPerspectiveRecord::Depth::width, "Image width in pixels.")
        .def("height", &RSPY_WorldCameraPerspectiveRecord::Depth::height, "Image height in pixels.")
        .def(
            "pixel",
            &RSPY_WorldCameraPerspectiveRecord::Depth::pixel,
            "Pixel data as a [height, width] float32 ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldCameraPerspectiveRecord::Normal>(camera_perspective_record, "Normal")
        .def("width", &RSPY_WorldCameraPerspectiveRecord::Normal::width, "Image width in pixels.")
        .def(
            "height", &RSPY_WorldCameraPerspectiveRecord::Normal::height, "Image height in pixels."
        )
        .def(
            "pixel",
            &RSPY_WorldCameraPerspectiveRecord::Normal::pixel,
            "Pixel data as a [height, width, 3] float32 ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldCameraPerspectiveRecord::Motion>(camera_perspective_record, "Motion")
        .def("width", &RSPY_WorldCameraPerspectiveRecord::Motion::width, "Image width in pixels.")
        .def(
            "height", &RSPY_WorldCameraPerspectiveRecord::Motion::height, "Image height in pixels."
        )
        .def(
            "pixel",
            &RSPY_WorldCameraPerspectiveRecord::Motion::pixel,
            "Pixel data as a [height, width, 2] float32 ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldCameraPerspectiveRecord::EntityInfo>(
        camera_perspective_record, "EntityInfo"
    )
        .def(
            "width", &RSPY_WorldCameraPerspectiveRecord::EntityInfo::width, "Image width in pixels."
        )
        .def(
            "height",
            &RSPY_WorldCameraPerspectiveRecord::EntityInfo::height,
            "Image height in pixels."
        )
        .def(
            "entity_type",
            &RSPY_WorldCameraPerspectiveRecord::EntityInfo::entity_type,
            "Entity type ids as a [height, width] uint32 ndarray, empty when disabled."
        )
        .def(
            "entity_gid",
            &RSPY_WorldCameraPerspectiveRecord::EntityInfo::entity_gid,
            "Entity global ids as a [height, width] uint32 ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldCameraPerspectiveRecord::State>(camera_perspective_record, "State")
        .def(
            "size",
            &RSPY_WorldCameraPerspectiveRecord::State::size,
            "Number of elements in this array view."
        )
        .def("position", &RSPY_WorldCameraPerspectiveRecord::State::position, "Camera position.")
        .def(
            "orientation",
            &RSPY_WorldCameraPerspectiveRecord::State::orientation,
            "Camera orientation."
        )
        .def(
            "focus_distance",
            &RSPY_WorldCameraPerspectiveRecord::State::focus_distance,
            "Camera focus distance."
        )
        .def(
            "field_of_view",
            &RSPY_WorldCameraPerspectiveRecord::State::field_of_view,
            "Camera field of view."
        );

    auto camera_stereo_record =
        py::class_<RSPY_WorldCameraStereoRecord>(m, "RS_WorldCameraStereoRecord")
            .def("step", &RSPY_WorldCameraStereoRecord::step, "Simulation step of this record.")
            .def("time", &RSPY_WorldCameraStereoRecord::time, "Simulation time of this record.")
            .def("camera_gid", &RSPY_WorldCameraStereoRecord::camera_gid, "Camera global id.")
            .def(
                "structure_gid",
                &RSPY_WorldCameraStereoRecord::structure_gid,
                "Linked structure gid."
            )
            .def(
                "rigid_body_sid",
                &RSPY_WorldCameraStereoRecord::rigid_body_sid,
                "Linked rigid body sid."
            )
            .def("width", &RSPY_WorldCameraStereoRecord::width, "Per-eye image width.")
            .def("height", &RSPY_WorldCameraStereoRecord::height, "Per-eye image height.")
            .def(
                "is_valid", &RSPY_WorldCameraStereoRecord::is_valid, "Whether this record is valid."
            )
            .def_property_readonly(
                "left_rgb",
                &RSPY_WorldCameraStereoRecord::left_rgb,
                "Left eye RGB pixel records of this camera."
            )
            .def_property_readonly(
                "right_rgb",
                &RSPY_WorldCameraStereoRecord::right_rgb,
                "Right eye RGB pixel records of this camera."
            )
            .def_property_readonly(
                "left_depth",
                &RSPY_WorldCameraStereoRecord::left_depth,
                "Left eye depth pixel records of this camera."
            )
            .def_property_readonly(
                "right_depth",
                &RSPY_WorldCameraStereoRecord::right_depth,
                "Right eye depth pixel records of this camera."
            )
            .def_property_readonly(
                "left_normal",
                &RSPY_WorldCameraStereoRecord::left_normal,
                "Left eye normal pixel records of this camera."
            )
            .def_property_readonly(
                "right_normal",
                &RSPY_WorldCameraStereoRecord::right_normal,
                "Right eye normal pixel records of this camera."
            )
            .def_property_readonly(
                "left_motion",
                &RSPY_WorldCameraStereoRecord::left_motion,
                "Left eye motion pixel records of this camera."
            )
            .def_property_readonly(
                "right_motion",
                &RSPY_WorldCameraStereoRecord::right_motion,
                "Right eye motion pixel records of this camera."
            )
            .def_property_readonly(
                "left_entity_info",
                &RSPY_WorldCameraStereoRecord::left_entity_info,
                "Left eye entity info pixel records of this camera."
            )
            .def_property_readonly(
                "right_entity_info",
                &RSPY_WorldCameraStereoRecord::right_entity_info,
                "Right eye entity info pixel records of this camera."
            )
            .def_property_readonly(
                "left_camera_state",
                &RSPY_WorldCameraStereoRecord::left_camera_state,
                "Left eye camera state records of this camera."
            )
            .def_property_readonly(
                "right_camera_state",
                &RSPY_WorldCameraStereoRecord::right_camera_state,
                "Right eye camera state records of this camera."
            );

    py::class_<RSPY_WorldCameraStereoRecord::RGB>(camera_stereo_record, "RGB")
        .def("width", &RSPY_WorldCameraStereoRecord::RGB::width, "Image width in pixels.")
        .def("height", &RSPY_WorldCameraStereoRecord::RGB::height, "Image height in pixels.")
        .def(
            "pixel",
            &RSPY_WorldCameraStereoRecord::RGB::pixel,
            "Pixel data as a [height, width, 3] uint8 ndarray."
        );

    py::class_<RSPY_WorldCameraStereoRecord::Depth>(camera_stereo_record, "Depth")
        .def("width", &RSPY_WorldCameraStereoRecord::Depth::width, "Image width in pixels.")
        .def("height", &RSPY_WorldCameraStereoRecord::Depth::height, "Image height in pixels.")
        .def(
            "pixel",
            &RSPY_WorldCameraStereoRecord::Depth::pixel,
            "Pixel data as a [height, width] float32 ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldCameraStereoRecord::Normal>(camera_stereo_record, "Normal")
        .def("width", &RSPY_WorldCameraStereoRecord::Normal::width, "Image width in pixels.")
        .def("height", &RSPY_WorldCameraStereoRecord::Normal::height, "Image height in pixels.")
        .def(
            "pixel",
            &RSPY_WorldCameraStereoRecord::Normal::pixel,
            "Pixel data as a [height, width, 3] float32 ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldCameraStereoRecord::Motion>(camera_stereo_record, "Motion")
        .def("width", &RSPY_WorldCameraStereoRecord::Motion::width, "Image width in pixels.")
        .def("height", &RSPY_WorldCameraStereoRecord::Motion::height, "Image height in pixels.")
        .def(
            "pixel",
            &RSPY_WorldCameraStereoRecord::Motion::pixel,
            "Pixel data as a [height, width, 2] float32 ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldCameraStereoRecord::EntityInfo>(camera_stereo_record, "EntityInfo")
        .def("width", &RSPY_WorldCameraStereoRecord::EntityInfo::width, "Image width in pixels.")
        .def("height", &RSPY_WorldCameraStereoRecord::EntityInfo::height, "Image height in pixels.")
        .def(
            "entity_type",
            &RSPY_WorldCameraStereoRecord::EntityInfo::entity_type,
            "Entity type ids as a [height, width] uint32 ndarray, empty when disabled."
        )
        .def(
            "entity_gid",
            &RSPY_WorldCameraStereoRecord::EntityInfo::entity_gid,
            "Entity global ids as a [height, width] uint32 ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldCameraStereoRecord::State>(camera_stereo_record, "State")
        .def(
            "size",
            &RSPY_WorldCameraStereoRecord::State::size,
            "Number of elements in this array view."
        )
        .def("position", &RSPY_WorldCameraStereoRecord::State::position, "Camera position.")
        .def(
            "orientation", &RSPY_WorldCameraStereoRecord::State::orientation, "Camera orientation."
        )
        .def(
            "focus_distance",
            &RSPY_WorldCameraStereoRecord::State::focus_distance,
            "Camera focus distance."
        )
        .def(
            "field_of_view",
            &RSPY_WorldCameraStereoRecord::State::field_of_view,
            "Camera field of view."
        );

    auto ray_sensor_record =
        py::class_<RSPY_WorldRaySensorRecord>(m, "RS_WorldRaySensorRecord")
            .def("step", &RSPY_WorldRaySensorRecord::step, "Simulation step of this record.")
            .def("time", &RSPY_WorldRaySensorRecord::time, "Simulation time of this record.")
            .def("sensor_gid", &RSPY_WorldRaySensorRecord::sensor_gid, "Ray sensor global id.")
            .def(
                "structure_gid", &RSPY_WorldRaySensorRecord::structure_gid, "Linked structure gid."
            )
            .def(
                "rigid_body_sid",
                &RSPY_WorldRaySensorRecord::rigid_body_sid,
                "Linked rigid body sid."
            )
            .def("ray_num", &RSPY_WorldRaySensorRecord::ray_num, "Number of rays.")
            .def("is_valid", &RSPY_WorldRaySensorRecord::is_valid, "Whether this record is valid.")
            .def_property_readonly(
                "hit", &RSPY_WorldRaySensorRecord::hit, "Hit records of this ray sensor."
            )
            .def_property_readonly(
                "entity_info",
                &RSPY_WorldRaySensorRecord::entity_info,
                "Entity info records of this ray sensor."
            )
            .def_property_readonly(
                "entity_detail",
                &RSPY_WorldRaySensorRecord::entity_detail,
                "Entity detail records of this ray sensor."
            )
            .def_property_readonly(
                "material",
                &RSPY_WorldRaySensorRecord::material,
                "Material records of this ray sensor."
            )
            .def_property_readonly(
                "normal", &RSPY_WorldRaySensorRecord::normal, "Normal records of this ray sensor."
            )
            .def_property_readonly(
                "velocity",
                &RSPY_WorldRaySensorRecord::velocity,
                "Velocity records of this ray sensor."
            )
            .def_property_readonly(
                "state",
                &RSPY_WorldRaySensorRecord::state,
                "Sensor state records of this ray sensor."
            );

    py::class_<RSPY_WorldRaySensorRecord::Hit>(ray_sensor_record, "Hit")
        .def("size", &RSPY_WorldRaySensorRecord::Hit::size, "Number of rays in this array view.")
        .def(
            "is_hit",
            &RSPY_WorldRaySensorRecord::Hit::is_hit,
            "Hit flag per ray as a [ray_num] uint8 ndarray, empty when disabled."
        )
        .def(
            "distance",
            &RSPY_WorldRaySensorRecord::Hit::distance,
            "Hit distance per ray as a [ray_num] float ndarray, empty when disabled."
        )
        .def(
            "hit_position",
            &RSPY_WorldRaySensorRecord::Hit::hit_position,
            "Hit position per ray as a [ray_num, 3] float ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldRaySensorRecord::EntityInfo>(ray_sensor_record, "EntityInfo")
        .def(
            "size",
            &RSPY_WorldRaySensorRecord::EntityInfo::size,
            "Number of rays in this array view."
        )
        .def(
            "entity_gid",
            &RSPY_WorldRaySensorRecord::EntityInfo::entity_gid,
            "Hit entity global id per ray as a [ray_num] uint32 ndarray, empty when disabled."
        )
        .def(
            "entity_type",
            &RSPY_WorldRaySensorRecord::EntityInfo::entity_type,
            "Hit entity type per ray as a [ray_num] uint32 ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldRaySensorRecord::EntityDetail>(ray_sensor_record, "EntityDetail")
        .def(
            "size",
            &RSPY_WorldRaySensorRecord::EntityDetail::size,
            "Number of rays in this array view."
        )
        .def(
            "body_sid",
            &RSPY_WorldRaySensorRecord::EntityDetail::body_sid,
            "Hit body id per ray as a [ray_num] uint32 ndarray, empty when disabled."
        )
        .def(
            "segment_bid",
            &RSPY_WorldRaySensorRecord::EntityDetail::segment_bid,
            "Hit segment id per ray as a [ray_num] uint32 ndarray, empty when disabled."
        )
        .def(
            "voxel_sid",
            &RSPY_WorldRaySensorRecord::EntityDetail::voxel_sid,
            "Hit voxel id per ray as a [ray_num] uint32 ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldRaySensorRecord::Material>(ray_sensor_record, "Material")
        .def(
            "size", &RSPY_WorldRaySensorRecord::Material::size, "Number of rays in this array view."
        )
        .def(
            "material_id",
            &RSPY_WorldRaySensorRecord::Material::material_id,
            "Hit material id per ray as a [ray_num] uint32 ndarray, empty when disabled."
        )
        .def(
            "emission",
            &RSPY_WorldRaySensorRecord::Material::emission,
            "Hit material emission per ray as a [ray_num] float ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldRaySensorRecord::Normal>(ray_sensor_record, "Normal")
        .def("size", &RSPY_WorldRaySensorRecord::Normal::size, "Number of rays in this array view.")
        .def(
            "normal",
            &RSPY_WorldRaySensorRecord::Normal::normal,
            "Hit normal per ray as a [ray_num, 3] float ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldRaySensorRecord::Velocity>(ray_sensor_record, "Velocity")
        .def(
            "size", &RSPY_WorldRaySensorRecord::Velocity::size, "Number of rays in this array view."
        )
        .def(
            "v_hit",
            &RSPY_WorldRaySensorRecord::Velocity::v_hit,
            "Hit world velocity per ray as a [ray_num, 3] float ndarray, empty when disabled."
        )
        .def(
            "v_rel",
            &RSPY_WorldRaySensorRecord::Velocity::v_rel,
            "Relative velocity per ray as a [ray_num, 3] float ndarray, empty when disabled."
        )
        .def(
            "v_radial",
            &RSPY_WorldRaySensorRecord::Velocity::v_radial,
            "Radial velocity per ray as a [ray_num] float ndarray, empty when disabled."
        );

    py::class_<RSPY_WorldRaySensorRecord::State>(ray_sensor_record, "State")
        .def(
            "size",
            &RSPY_WorldRaySensorRecord::State::size,
            "Number of elements in this array view."
        )
        .def("position", &RSPY_WorldRaySensorRecord::State::position, "Sensor origin position.")
        .def("orientation", &RSPY_WorldRaySensorRecord::State::orientation, "Sensor orientation.");

    auto intersect_region_record =
        py::class_<RSPY_WorldIntersectRegionRecord>(m, "RS_WorldIntersectRegionRecord")
            .def("step", &RSPY_WorldIntersectRegionRecord::step, "Simulation step of this record.")
            .def("time", &RSPY_WorldIntersectRegionRecord::time, "Simulation time of this record.")
            .def(
                "record_gid",
                &RSPY_WorldIntersectRegionRecord::record_gid,
                "Region record global id."
            )
            .def_property_readonly(
                "memberships",
                &RSPY_WorldIntersectRegionRecord::memberships,
                "Membership records of this region."
            );

    py::class_<RSPY_WorldIntersectRegionRecord::Membership>(intersect_region_record, "Membership")
        .def(
            "size",
            &RSPY_WorldIntersectRegionRecord::Membership::size,
            "Number of memberships in this array view."
        )
        .def(
            "structure_gid",
            &RSPY_WorldIntersectRegionRecord::Membership::structure_gid,
            "Global id of a structure intersecting the region."
        )
        .def(
            "closest_relative_position",
            &RSPY_WorldIntersectRegionRecord::Membership::closest_relative_position,
            "The intersecting structure's closest voxel to the region center, as a world coord "
            "offset."
        );
}
