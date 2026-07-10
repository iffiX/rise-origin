// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "core/init_world_record.h"

void initRSPYWorldRecord(py::module &m) {
    /*****************************************************************
     * Records
     *****************************************************************/

    auto structure_voxel_record =
        py::class_<RSPY_WorldStructureVoxelRecord>(m, "RS_WorldStructureVoxelRecord")
            .def("step", &RSPY_WorldStructureVoxelRecord::step)
            .def("time", &RSPY_WorldStructureVoxelRecord::time)
            .def("structure_gid", &RSPY_WorldStructureVoxelRecord::structure_gid)
            .def_property_readonly("voxels", &RSPY_WorldStructureVoxelRecord::voxels)
            .def_property_readonly("links", &RSPY_WorldStructureVoxelRecord::links)
            .def_property_readonly("rigid_bodies", &RSPY_WorldStructureVoxelRecord::rigid_bodies)
            .def_property_readonly("joints", &RSPY_WorldStructureVoxelRecord::joints);

    py::class_<RSPY_WorldStructureVoxelRecord::Voxel>(structure_voxel_record, "Voxel")
        .def("size", &RSPY_WorldStructureVoxelRecord::Voxel::size)
        .def("body_sid", &RSPY_WorldStructureVoxelRecord::Voxel::body_sid)
        .def("voxel_sid", &RSPY_WorldStructureVoxelRecord::Voxel::voxel_sid)
        .def("segment_bid", &RSPY_WorldStructureVoxelRecord::Voxel::segment_bid)
        .def("index_x", &RSPY_WorldStructureVoxelRecord::Voxel::index_x)
        .def("index_y", &RSPY_WorldStructureVoxelRecord::Voxel::index_y)
        .def("index_z", &RSPY_WorldStructureVoxelRecord::Voxel::index_z)
        .def("connectivity", &RSPY_WorldStructureVoxelRecord::Voxel::connectivity)
        .def("is_surface", &RSPY_WorldStructureVoxelRecord::Voxel::is_surface)
        .def("is_rigid", &RSPY_WorldStructureVoxelRecord::Voxel::is_rigid)
        .def("material", &RSPY_WorldStructureVoxelRecord::Voxel::material)
        .def("expansion", &RSPY_WorldStructureVoxelRecord::Voxel::expansion)
        .def("orientation", &RSPY_WorldStructureVoxelRecord::Voxel::orientation)
        .def("position", &RSPY_WorldStructureVoxelRecord::Voxel::position)
        .def("ppp_offset", &RSPY_WorldStructureVoxelRecord::Voxel::ppp_offset)
        .def("nnn_offset", &RSPY_WorldStructureVoxelRecord::Voxel::nnn_offset)
        .def("linear_velocity", &RSPY_WorldStructureVoxelRecord::Voxel::linear_velocity)
        .def("angular_velocity", &RSPY_WorldStructureVoxelRecord::Voxel::angular_velocity)
        .def("poissons_strain", &RSPY_WorldStructureVoxelRecord::Voxel::poissons_strain);

    py::class_<RSPY_WorldStructureVoxelRecord::Link>(structure_voxel_record, "Link")
        .def("size", &RSPY_WorldStructureVoxelRecord::Link::size)
        .def("voxel_neg_sid", &RSPY_WorldStructureVoxelRecord::Link::voxel_neg_sid)
        .def("voxel_pos_sid", &RSPY_WorldStructureVoxelRecord::Link::voxel_pos_sid)
        .def("is_failed", &RSPY_WorldStructureVoxelRecord::Link::is_failed)
        .def("strain", &RSPY_WorldStructureVoxelRecord::Link::strain)
        .def("stress", &RSPY_WorldStructureVoxelRecord::Link::stress);

    py::class_<RSPY_WorldStructureVoxelRecord::RigidBody>(structure_voxel_record, "RigidBody")
        .def("size", &RSPY_WorldStructureVoxelRecord::RigidBody::size)
        .def("rigid_body_sid", &RSPY_WorldStructureVoxelRecord::RigidBody::rigid_body_sid)
        .def("body_sid", &RSPY_WorldStructureVoxelRecord::RigidBody::body_sid)
        .def("segment_bid", &RSPY_WorldStructureVoxelRecord::RigidBody::segment_bid)
        .def("mass", &RSPY_WorldStructureVoxelRecord::RigidBody::mass)
        .def("orientation", &RSPY_WorldStructureVoxelRecord::RigidBody::orientation)
        .def("com", &RSPY_WorldStructureVoxelRecord::RigidBody::com)
        .def("linear_velocity", &RSPY_WorldStructureVoxelRecord::RigidBody::linear_velocity)
        .def("angular_velocity", &RSPY_WorldStructureVoxelRecord::RigidBody::angular_velocity);

    py::class_<RSPY_WorldStructureVoxelRecord::Joint>(structure_voxel_record, "Joint")
        .def("size", &RSPY_WorldStructureVoxelRecord::Joint::size)
        .def("rigid_body_a_sid", &RSPY_WorldStructureVoxelRecord::Joint::rigid_body_a_sid)
        .def("rigid_body_b_sid", &RSPY_WorldStructureVoxelRecord::Joint::rigid_body_b_sid)
        .def("type", &RSPY_WorldStructureVoxelRecord::Joint::type)
        .def("position", &RSPY_WorldStructureVoxelRecord::Joint::position)
        .def("axis", &RSPY_WorldStructureVoxelRecord::Joint::axis)
        .def("hinge_signal_sid", &RSPY_WorldStructureVoxelRecord::Joint::hinge_signal_sid)
        .def("hinge_min", &RSPY_WorldStructureVoxelRecord::Joint::hinge_min)
        .def("hinge_max", &RSPY_WorldStructureVoxelRecord::Joint::hinge_max)
        .def("hinge_angle", &RSPY_WorldStructureVoxelRecord::Joint::hinge_angle)
        .def("hinge_torque", &RSPY_WorldStructureVoxelRecord::Joint::hinge_torque)
        .def("prismatic_signal_sid", &RSPY_WorldStructureVoxelRecord::Joint::prismatic_signal_sid)
        .def("prismatic_min", &RSPY_WorldStructureVoxelRecord::Joint::prismatic_min)
        .def("prismatic_max", &RSPY_WorldStructureVoxelRecord::Joint::prismatic_max)
        .def("prismatic_distance", &RSPY_WorldStructureVoxelRecord::Joint::prismatic_distance)
        .def("prismatic_force", &RSPY_WorldStructureVoxelRecord::Joint::prismatic_force);

    auto camera_perspective_record =
        py::class_<RSPY_WorldCameraPerspectiveRecord>(m, "RS_WorldCameraPerspectiveRecord")
            .def("step", &RSPY_WorldCameraPerspectiveRecord::step)
            .def("time", &RSPY_WorldCameraPerspectiveRecord::time)
            .def("camera_gid", &RSPY_WorldCameraPerspectiveRecord::camera_gid)
            .def("structure_gid", &RSPY_WorldCameraPerspectiveRecord::structure_gid)
            .def("rigid_body_sid", &RSPY_WorldCameraPerspectiveRecord::rigid_body_sid)
            .def("width", &RSPY_WorldCameraPerspectiveRecord::width)
            .def("height", &RSPY_WorldCameraPerspectiveRecord::height)
            .def("is_valid", &RSPY_WorldCameraPerspectiveRecord::is_valid)
            .def_property_readonly("rgb", &RSPY_WorldCameraPerspectiveRecord::rgb)
            .def_property_readonly("depth", &RSPY_WorldCameraPerspectiveRecord::depth)
            .def_property_readonly("normal", &RSPY_WorldCameraPerspectiveRecord::normal)
            .def_property_readonly("motion", &RSPY_WorldCameraPerspectiveRecord::motion)
            .def_property_readonly("entity_info", &RSPY_WorldCameraPerspectiveRecord::entity_info)
            .def_property_readonly("state", &RSPY_WorldCameraPerspectiveRecord::state);

    py::class_<RSPY_WorldCameraPerspectiveRecord::RGB>(camera_perspective_record, "RGB")
        .def("width", &RSPY_WorldCameraPerspectiveRecord::RGB::width)
        .def("height", &RSPY_WorldCameraPerspectiveRecord::RGB::height)
        .def("pixel", &RSPY_WorldCameraPerspectiveRecord::RGB::pixel);

    py::class_<RSPY_WorldCameraPerspectiveRecord::Depth>(camera_perspective_record, "Depth")
        .def("width", &RSPY_WorldCameraPerspectiveRecord::Depth::width)
        .def("height", &RSPY_WorldCameraPerspectiveRecord::Depth::height)
        .def("pixel", &RSPY_WorldCameraPerspectiveRecord::Depth::pixel);

    py::class_<RSPY_WorldCameraPerspectiveRecord::Normal>(camera_perspective_record, "Normal")
        .def("width", &RSPY_WorldCameraPerspectiveRecord::Normal::width)
        .def("height", &RSPY_WorldCameraPerspectiveRecord::Normal::height)
        .def("pixel", &RSPY_WorldCameraPerspectiveRecord::Normal::pixel);

    py::class_<RSPY_WorldCameraPerspectiveRecord::Motion>(camera_perspective_record, "Motion")
        .def("width", &RSPY_WorldCameraPerspectiveRecord::Motion::width)
        .def("height", &RSPY_WorldCameraPerspectiveRecord::Motion::height)
        .def("pixel", &RSPY_WorldCameraPerspectiveRecord::Motion::pixel);

    py::class_<RSPY_WorldCameraPerspectiveRecord::EntityInfo>(
        camera_perspective_record, "EntityInfo"
    )
        .def("width", &RSPY_WorldCameraPerspectiveRecord::EntityInfo::width)
        .def("height", &RSPY_WorldCameraPerspectiveRecord::EntityInfo::height)
        .def("entity_type", &RSPY_WorldCameraPerspectiveRecord::EntityInfo::entity_type)
        .def("entity_gid", &RSPY_WorldCameraPerspectiveRecord::EntityInfo::entity_gid);

    py::class_<RSPY_WorldCameraPerspectiveRecord::State>(camera_perspective_record, "State")
        .def("size", &RSPY_WorldCameraPerspectiveRecord::State::size)
        .def("position", &RSPY_WorldCameraPerspectiveRecord::State::position)
        .def("orientation", &RSPY_WorldCameraPerspectiveRecord::State::orientation)
        .def("focus_distance", &RSPY_WorldCameraPerspectiveRecord::State::focus_distance)
        .def("field_of_view", &RSPY_WorldCameraPerspectiveRecord::State::field_of_view);

    auto camera_stereo_record =
        py::class_<RSPY_WorldCameraStereoRecord>(m, "RS_WorldCameraStereoRecord")
            .def("step", &RSPY_WorldCameraStereoRecord::step)
            .def("time", &RSPY_WorldCameraStereoRecord::time)
            .def("camera_gid", &RSPY_WorldCameraStereoRecord::camera_gid)
            .def("structure_gid", &RSPY_WorldCameraStereoRecord::structure_gid)
            .def("rigid_body_sid", &RSPY_WorldCameraStereoRecord::rigid_body_sid)
            .def("width", &RSPY_WorldCameraStereoRecord::width)
            .def("height", &RSPY_WorldCameraStereoRecord::height)
            .def("is_valid", &RSPY_WorldCameraStereoRecord::is_valid)
            .def_property_readonly("left_rgb", &RSPY_WorldCameraStereoRecord::left_rgb)
            .def_property_readonly("right_rgb", &RSPY_WorldCameraStereoRecord::right_rgb)
            .def_property_readonly("left_depth", &RSPY_WorldCameraStereoRecord::left_depth)
            .def_property_readonly("right_depth", &RSPY_WorldCameraStereoRecord::right_depth)
            .def_property_readonly("left_normal", &RSPY_WorldCameraStereoRecord::left_normal)
            .def_property_readonly("right_normal", &RSPY_WorldCameraStereoRecord::right_normal)
            .def_property_readonly("left_motion", &RSPY_WorldCameraStereoRecord::left_motion)
            .def_property_readonly("right_motion", &RSPY_WorldCameraStereoRecord::right_motion)
            .def_property_readonly(
                "left_entity_info", &RSPY_WorldCameraStereoRecord::left_entity_info
            )
            .def_property_readonly(
                "right_entity_info", &RSPY_WorldCameraStereoRecord::right_entity_info
            )
            .def_property_readonly(
                "left_camera_state", &RSPY_WorldCameraStereoRecord::left_camera_state
            )
            .def_property_readonly(
                "right_camera_state", &RSPY_WorldCameraStereoRecord::right_camera_state
            );

    py::class_<RSPY_WorldCameraStereoRecord::RGB>(camera_stereo_record, "RGB")
        .def("width", &RSPY_WorldCameraStereoRecord::RGB::width)
        .def("height", &RSPY_WorldCameraStereoRecord::RGB::height)
        .def("pixel", &RSPY_WorldCameraStereoRecord::RGB::pixel);

    py::class_<RSPY_WorldCameraStereoRecord::Depth>(camera_stereo_record, "Depth")
        .def("width", &RSPY_WorldCameraStereoRecord::Depth::width)
        .def("height", &RSPY_WorldCameraStereoRecord::Depth::height)
        .def("pixel", &RSPY_WorldCameraStereoRecord::Depth::pixel);

    py::class_<RSPY_WorldCameraStereoRecord::Normal>(camera_stereo_record, "Normal")
        .def("width", &RSPY_WorldCameraStereoRecord::Normal::width)
        .def("height", &RSPY_WorldCameraStereoRecord::Normal::height)
        .def("pixel", &RSPY_WorldCameraStereoRecord::Normal::pixel);

    py::class_<RSPY_WorldCameraStereoRecord::Motion>(camera_stereo_record, "Motion")
        .def("width", &RSPY_WorldCameraStereoRecord::Motion::width)
        .def("height", &RSPY_WorldCameraStereoRecord::Motion::height)
        .def("pixel", &RSPY_WorldCameraStereoRecord::Motion::pixel);

    py::class_<RSPY_WorldCameraStereoRecord::EntityInfo>(camera_stereo_record, "EntityInfo")
        .def("width", &RSPY_WorldCameraStereoRecord::EntityInfo::width)
        .def("height", &RSPY_WorldCameraStereoRecord::EntityInfo::height)
        .def("entity_type", &RSPY_WorldCameraStereoRecord::EntityInfo::entity_type)
        .def("entity_gid", &RSPY_WorldCameraStereoRecord::EntityInfo::entity_gid);

    py::class_<RSPY_WorldCameraStereoRecord::State>(camera_stereo_record, "State")
        .def("size", &RSPY_WorldCameraStereoRecord::State::size)
        .def("position", &RSPY_WorldCameraStereoRecord::State::position)
        .def("orientation", &RSPY_WorldCameraStereoRecord::State::orientation)
        .def("focus_distance", &RSPY_WorldCameraStereoRecord::State::focus_distance)
        .def("field_of_view", &RSPY_WorldCameraStereoRecord::State::field_of_view);

    auto ray_sensor_record =
        py::class_<RSPY_WorldRaySensorRecord>(m, "RS_WorldRaySensorRecord")
            .def("step", &RSPY_WorldRaySensorRecord::step)
            .def("time", &RSPY_WorldRaySensorRecord::time)
            .def("sensor_gid", &RSPY_WorldRaySensorRecord::sensor_gid)
            .def("structure_gid", &RSPY_WorldRaySensorRecord::structure_gid)
            .def("rigid_body_sid", &RSPY_WorldRaySensorRecord::rigid_body_sid)
            .def("ray_num", &RSPY_WorldRaySensorRecord::ray_num)
            .def("is_valid", &RSPY_WorldRaySensorRecord::is_valid)
            .def_property_readonly("hit", &RSPY_WorldRaySensorRecord::hit)
            .def_property_readonly("entity_info", &RSPY_WorldRaySensorRecord::entity_info)
            .def_property_readonly("entity_detail", &RSPY_WorldRaySensorRecord::entity_detail)
            .def_property_readonly("material", &RSPY_WorldRaySensorRecord::material)
            .def_property_readonly("normal", &RSPY_WorldRaySensorRecord::normal)
            .def_property_readonly("velocity", &RSPY_WorldRaySensorRecord::velocity)
            .def_property_readonly("state", &RSPY_WorldRaySensorRecord::state);

    py::class_<RSPY_WorldRaySensorRecord::Hit>(ray_sensor_record, "Hit")
        .def("size", &RSPY_WorldRaySensorRecord::Hit::size)
        .def("is_hit", &RSPY_WorldRaySensorRecord::Hit::is_hit)
        .def("distance", &RSPY_WorldRaySensorRecord::Hit::distance)
        .def("hit_position", &RSPY_WorldRaySensorRecord::Hit::hit_position);

    py::class_<RSPY_WorldRaySensorRecord::EntityInfo>(ray_sensor_record, "EntityInfo")
        .def("size", &RSPY_WorldRaySensorRecord::EntityInfo::size)
        .def("entity_gid", &RSPY_WorldRaySensorRecord::EntityInfo::entity_gid)
        .def("entity_type", &RSPY_WorldRaySensorRecord::EntityInfo::entity_type);

    py::class_<RSPY_WorldRaySensorRecord::EntityDetail>(ray_sensor_record, "EntityDetail")
        .def("size", &RSPY_WorldRaySensorRecord::EntityDetail::size)
        .def("body_sid", &RSPY_WorldRaySensorRecord::EntityDetail::body_sid)
        .def("segment_bid", &RSPY_WorldRaySensorRecord::EntityDetail::segment_bid)
        .def("voxel_sid", &RSPY_WorldRaySensorRecord::EntityDetail::voxel_sid);

    py::class_<RSPY_WorldRaySensorRecord::Material>(ray_sensor_record, "Material")
        .def("size", &RSPY_WorldRaySensorRecord::Material::size)
        .def("material_id", &RSPY_WorldRaySensorRecord::Material::material_id)
        .def("emission", &RSPY_WorldRaySensorRecord::Material::emission);

    py::class_<RSPY_WorldRaySensorRecord::Normal>(ray_sensor_record, "Normal")
        .def("size", &RSPY_WorldRaySensorRecord::Normal::size)
        .def("normal", &RSPY_WorldRaySensorRecord::Normal::normal);

    py::class_<RSPY_WorldRaySensorRecord::Velocity>(ray_sensor_record, "Velocity")
        .def("size", &RSPY_WorldRaySensorRecord::Velocity::size)
        .def("v_hit", &RSPY_WorldRaySensorRecord::Velocity::v_hit)
        .def("v_rel", &RSPY_WorldRaySensorRecord::Velocity::v_rel)
        .def("v_radial", &RSPY_WorldRaySensorRecord::Velocity::v_radial);

    py::class_<RSPY_WorldRaySensorRecord::State>(ray_sensor_record, "State")
        .def("size", &RSPY_WorldRaySensorRecord::State::size)
        .def("position", &RSPY_WorldRaySensorRecord::State::position)
        .def("orientation", &RSPY_WorldRaySensorRecord::State::orientation);

    auto intersect_region_record =
        py::class_<RSPY_WorldIntersectRegionRecord>(m, "RS_WorldIntersectRegionRecord")
            .def("step", &RSPY_WorldIntersectRegionRecord::step)
            .def("time", &RSPY_WorldIntersectRegionRecord::time)
            .def("record_gid", &RSPY_WorldIntersectRegionRecord::record_gid)
            .def_property_readonly("memberships", &RSPY_WorldIntersectRegionRecord::memberships);

    py::class_<RSPY_WorldIntersectRegionRecord::Membership>(intersect_region_record, "Membership")
        .def("size", &RSPY_WorldIntersectRegionRecord::Membership::size)
        .def("structure_gid", &RSPY_WorldIntersectRegionRecord::Membership::structure_gid)
        .def(
            "closest_relative_position",
            &RSPY_WorldIntersectRegionRecord::Membership::closest_relative_position
        );
}
