// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rs_simulation_hdf5_recorder.h"

#if !H5_VERSION_GE(1, 10, 1)
#error "Minimum HDF5 version required is 1.10.1"
#endif
using namespace std;
namespace HF = HighFive;

namespace rise {

RS_SimulationHDF5Recorder::RS_SimulationHDF5Recorder(const string &name, const string &file_path)
    : _name(name) {
    HF::FileAccessProps file_access_props;
    file_access_props.add(HF::FileVersionBounds(H5F_LIBVER_V110, H5F_LIBVER_LATEST));
    _hdf5_file = make_unique<HF::File>(file_path, HF::File::Overwrite, file_access_props);
    // Entity-embedded records live under /entity/<name>; future free-floating
    // record entities would live under /record/<name>.
    _entity_root_group = make_unique<HF::Group>(_hdf5_file->createGroup("/entity"));
    H5Fstart_swmr_write(_hdf5_file->getId());
}

string RS_SimulationHDF5Recorder::getName() const { return _name; }

void RS_SimulationHDF5Recorder::start() {}

void RS_SimulationHDF5Recorder::syncStructureVoxelRecords(
    const std::string &name,
    const std::vector<RS_WorldStructureVoxelRecord> &records
) {
    StructureRecordData &d = _createOrGetStructureRecordData(
        name,
        /*has_voxel=*/true,
        /*has_link=*/true,
        /*has_rigid_body=*/true,
        /*has_joint=*/true
    );

    for (auto &record : records) {
        _appendToGroupNd(d.time_group, "step", record.step, 1, 1);
        _appendToGroupNd(d.time_group, "time_point", record.time, 1, 1);

        _writeStructureVoxelFrame(d, record.voxels);
        _writeStructureLinkFrame(d, record.links);
        _writeStructureRigidBodyFrame(d, record.rigid_bodies);
        _writeStructureJointFrame(d, record.joints);
    }
}

void RS_SimulationHDF5Recorder::syncCameraPerspectiveRecords(
    const std::string &name,
    const std::vector<RS_WorldCameraPerspectiveRecord> &records
) {
    if (records.empty()) return;

    // A disabled subrecord carries object_num 0, so its presence is read from the first frame.
    bool has_rgb = records[0].rgb.object_num > 0;
    bool has_depth = records[0].depth.object_num > 0;
    bool has_normal = records[0].normal.object_num > 0;
    bool has_motion = records[0].motion.object_num > 0;
    bool has_entity_info = records[0].entity_info.object_num > 0;

    CameraRecordData &d = _createOrGetCameraRecordData(
        name,
        records[0].width,
        records[0].height,
        has_rgb,
        has_depth,
        has_normal,
        has_motion,
        has_entity_info,
        /*has_camera_state=*/true
    );

    for (auto &record : records) {
        _appendToGroupNd(d.time_group, "step", record.step, 1, 1);
        _appendToGroupNd(d.time_group, "time_point", record.time, 1, 1);
        if (has_rgb) _writeCameraRgbFrame(d, record.rgb);
        if (has_depth) _writeCameraDepthFrame(d, record.depth);
        if (has_normal) _writeCameraNormalFrame(d, record.normal);
        if (has_motion) _writeCameraMotionFrame(d, record.motion);
        if (has_entity_info) _writeCameraEntityInfoFrame(d, record.entity_info);
        _writeCameraStateFrame(d, record.camera_state);
    }
}

void RS_SimulationHDF5Recorder::syncRaySensorRecords(
    const std::string &name,
    const std::vector<RS_WorldRaySensorRecord> &records
) {
    if (records.empty()) return;

    // A disabled subrecord carries object_num 0, so its presence is read from the first frame.
    bool has_hit = records[0].hit.object_num > 0;
    bool has_entity_info = records[0].entity_info.object_num > 0;
    bool has_entity_detail = records[0].entity_detail.object_num > 0;
    bool has_material = records[0].material.object_num > 0;
    bool has_normal = records[0].normal.object_num > 0;
    bool has_velocity = records[0].velocity.object_num > 0;

    RaySensorRecordData &d = _createOrGetRaySensorRecordData(
        name,
        records[0].ray_num,
        has_hit,
        has_entity_info,
        has_entity_detail,
        has_material,
        has_normal,
        has_velocity
    );

    for (auto &record : records) {
        _appendToGroupNd(d.time_group, "step", record.step, 1, 1);
        _appendToGroupNd(d.time_group, "time_point", record.time, 1, 1);
        if (has_hit) _writeRaySensorHitFrame(d, record.hit);
        if (has_entity_info) _writeRaySensorEntityInfoFrame(d, record.entity_info);
        if (has_entity_detail) _writeRaySensorEntityDetailFrame(d, record.entity_detail);
        if (has_material) _writeRaySensorMaterialFrame(d, record.material);
        if (has_normal) _writeRaySensorNormalFrame(d, record.normal);
        if (has_velocity) _writeRaySensorVelocityFrame(d, record.velocity);
        _writeRaySensorStateFrame(d, record.sensor_state);
    }
}

void RS_SimulationHDF5Recorder::finish() {
    _hdf5_file->flush();
    H5Fclose(_hdf5_file->getId());
}

RS_SimulationHDF5Recorder::StructureRecordData &
RS_SimulationHDF5Recorder::_createOrGetStructureRecordData(
    const std::string &structure_name,
    bool has_voxel,
    bool has_link,
    bool has_rigid_body,
    bool has_joint
) {
    if (_structure_record_data.find(structure_name) != _structure_record_data.end()) {
        return _structure_record_data[structure_name];
    }
    StructureRecordData d;
    d.root_group = make_shared<HF::Group>(_entity_root_group->createGroup(structure_name));

    auto &root_group = d.root_group;
    d.time_group = make_shared<HF::Group>(root_group->createGroup("frames/time"));
    if (has_voxel) {
        d.voxels_group = make_shared<HF::Group>(root_group->createGroup("frames/voxels"));
    }
    if (has_link) {
        d.links_group = make_shared<HF::Group>(root_group->createGroup("frames/links"));
    }
    if (has_rigid_body) {
        d.rigid_bodies_group =
            make_shared<HF::Group>(root_group->createGroup("frames/rigid_bodies"));
    }
    if (has_joint) {
        d.joints_group = make_shared<HF::Group>(root_group->createGroup("frames/joints"));
    }

    HF::DataSpace space_1d({0}, {HF::DataSpace::UNLIMITED});
    HF::DataSpace space_3d({0, 3}, {HF::DataSpace::UNLIMITED, 3});
    HF::DataSpace space_4d({0, 4}, {HF::DataSpace::UNLIMITED, 4});
    HF::DataSetCreateProps create_props_1d, create_props_3d, create_props_4d;
    create_props_1d.add(HF::Chunking(chunk_size));
    create_props_3d.add(HF::Chunking(chunk_size, 3));
    create_props_4d.add(HF::Chunking(chunk_size, 4));

    d.time_group->createDataSet<RBigIndex>("step", space_1d, create_props_1d);
    d.time_group->createDataSet<RBigFloat>("time_point", space_1d, create_props_1d);

    if (has_voxel) {
        d.voxels_group->createDataSet<RBigSize>("voxel_num", space_1d, create_props_1d);
        d.voxels_group->createDataSet<RBigSize>("voxel_offset", space_1d, create_props_1d);
        d.voxels_group->createDataSet<RIndex>("body_sid", space_1d, create_props_1d);
        d.voxels_group->createDataSet<RIndex>("voxel_sid", space_1d, create_props_1d);
        d.voxels_group->createDataSet<RIndex>("segment_bid", space_1d, create_props_1d);
        d.voxels_group->createDataSet<RSignedIndex>("index_xyz", space_3d, create_props_3d);
        d.voxels_group->createDataSet<uint8_t>("connectivity", space_1d, create_props_1d);
        d.voxels_group->createDataSet<bool>("is_surface", space_1d, create_props_1d);
        d.voxels_group->createDataSet<bool>("is_rigid", space_1d, create_props_1d);
        d.voxels_group->createDataSet<RIndex>("material", space_1d, create_props_1d);
        d.voxels_group->createDataSet<RFloat>("expansion", space_1d, create_props_1d);
        d.voxels_group->createDataSet<RFloat>("orientation", space_4d, create_props_4d);
        d.voxels_group->createDataSet<RFloat>("position", space_3d, create_props_3d);
        d.voxels_group->createDataSet<RFloat>("ppp_offset", space_3d, create_props_3d);
        d.voxels_group->createDataSet<RFloat>("nnn_offset", space_3d, create_props_3d);
        d.voxels_group->createDataSet<RFloat>("linear_velocity", space_3d, create_props_3d);
        d.voxels_group->createDataSet<RFloat>("angular_velocity", space_3d, create_props_3d);
        d.voxels_group->createDataSet<RFloat>("poissons_strain", space_3d, create_props_3d);
    }

    if (has_link) {
        d.links_group->createDataSet<RBigSize>("link_num", space_1d, create_props_1d);
        d.links_group->createDataSet<RBigSize>("link_offset", space_1d, create_props_1d);
        d.links_group->createDataSet<RIndex>("voxel_neg_sid", space_1d, create_props_1d);
        d.links_group->createDataSet<RIndex>("voxel_pos_sid", space_1d, create_props_1d);
        d.links_group->createDataSet<bool>("is_failed", space_1d, create_props_1d);
        d.links_group->createDataSet<RFloat>("strain", space_1d, create_props_1d);
        d.links_group->createDataSet<RFloat>("stress", space_1d, create_props_1d);
    }

    if (has_rigid_body) {
        d.rigid_bodies_group->createDataSet<RBigSize>("rigid_body_num", space_1d, create_props_1d);
        d.rigid_bodies_group->createDataSet<RBigSize>("rigid_body_offset", space_1d, create_props_1d);
        d.rigid_bodies_group->createDataSet<RIndex>("rigid_body_sid", space_1d, create_props_1d);
        d.rigid_bodies_group->createDataSet<RIndex>("body_sid", space_1d, create_props_1d);
        d.rigid_bodies_group->createDataSet<RIndex>("segment_bid", space_1d, create_props_1d);
        d.rigid_bodies_group->createDataSet<RFloat>("mass", space_1d, create_props_1d);
        d.rigid_bodies_group->createDataSet<RFloat>("orientation", space_4d, create_props_4d);
        d.rigid_bodies_group->createDataSet<RFloat>("com", space_3d, create_props_3d);
        d.rigid_bodies_group->createDataSet<RFloat>("linear_velocity", space_3d, create_props_3d);
        d.rigid_bodies_group->createDataSet<RFloat>("angular_velocity", space_3d, create_props_3d);
    }

    if (has_joint) {
        d.joints_group->createDataSet<RBigSize>("joint_num", space_1d, create_props_1d);
        d.joints_group->createDataSet<RBigSize>("joint_offset", space_1d, create_props_1d);
        d.joints_group->createDataSet<RIndex>("rigid_body_a_sid", space_1d, create_props_1d);
        d.joints_group->createDataSet<RIndex>("rigid_body_b_sid", space_1d, create_props_1d);
        d.joints_group->createDataSet<int>("type", space_1d, create_props_1d);
        d.joints_group->createDataSet<RFloat>("position", space_3d, create_props_3d);
        d.joints_group->createDataSet<RFloat>("axis", space_3d, create_props_3d);
        // Hinge fields
        d.joints_group->createDataSet<RIndex>("hinge_signal_sid", space_1d, create_props_1d);
        d.joints_group->createDataSet<RFloat>("hinge_min", space_1d, create_props_1d);
        d.joints_group->createDataSet<RFloat>("hinge_max", space_1d, create_props_1d);
        d.joints_group->createDataSet<RFloat>("hinge_angle", space_1d, create_props_1d);
        d.joints_group->createDataSet<RFloat>("hinge_torque", space_1d, create_props_1d);
        // Prismatic fields
        d.joints_group->createDataSet<RIndex>("prismatic_signal_sid", space_1d, create_props_1d);
        d.joints_group->createDataSet<RFloat>("prismatic_min", space_1d, create_props_1d);
        d.joints_group->createDataSet<RFloat>("prismatic_max", space_1d, create_props_1d);
        d.joints_group->createDataSet<RFloat>("prismatic_distance", space_1d, create_props_1d);
        d.joints_group->createDataSet<RFloat>("prismatic_force", space_1d, create_props_1d);
    }

    _structure_record_data[structure_name] = d;
    return _structure_record_data[structure_name];
}

RS_SimulationHDF5Recorder::CameraRecordData &
RS_SimulationHDF5Recorder::_createOrGetCameraRecordData(
    const std::string &camera_name,
    RSize width,
    RSize height,
    bool has_rgb,
    bool has_depth,
    bool has_normal,
    bool has_motion,
    bool has_entity_info,
    bool has_camera_state
) {
    if (_camera_record_data.find(camera_name) != _camera_record_data.end()) {
        return _camera_record_data[camera_name];
    }
    CameraRecordData d;
    d.width = width;
    d.height = height;
    d.root_group = make_shared<HF::Group>(_entity_root_group->createGroup(camera_name));
    auto &root_group = d.root_group;

    d.time_group = make_shared<HF::Group>(root_group->createGroup("frames/time"));
    if (has_rgb) {
        d.rgb_group = make_shared<HF::Group>(root_group->createGroup("frames/rgb"));
    }
    if (has_depth) {
        d.depth_group = make_shared<HF::Group>(root_group->createGroup("frames/depth"));
    }
    if (has_normal) {
        d.normal_group = make_shared<HF::Group>(root_group->createGroup("frames/normal"));
    }
    if (has_motion) {
        d.motion_group = make_shared<HF::Group>(root_group->createGroup("frames/motion"));
    }
    if (has_entity_info) {
        d.entity_info_group =
            make_shared<HF::Group>(root_group->createGroup("frames/entity_info"));
    }
    if (has_camera_state) {
        d.camera_state_group =
            make_shared<HF::Group>(root_group->createGroup("frames/camera_state"));
    }

    HF::DataSpace space_1d({0}, {HF::DataSpace::UNLIMITED});
    HF::DataSpace space_3d({0, 3}, {HF::DataSpace::UNLIMITED, 3});
    HF::DataSpace space_4d({0, 4}, {HF::DataSpace::UNLIMITED, 4});
    HF::DataSpace space_rgb({0, height, width, 3}, {HF::DataSpace::UNLIMITED, height, width, 3});
    HF::DataSpace space_depth({0, height, width}, {HF::DataSpace::UNLIMITED, height, width});
    HF::DataSpace space_normal({0, height, width, 3}, {HF::DataSpace::UNLIMITED, height, width, 3});
    HF::DataSpace space_motion({0, height, width, 2}, {HF::DataSpace::UNLIMITED, height, width, 2});

    HF::DataSetCreateProps create_props_1d, create_props_3d, create_props_4d;
    HF::DataSetCreateProps create_props_rgb, create_props_depth, create_props_normal,
        create_props_motion;
    create_props_1d.add(HF::Chunking(chunk_size));
    create_props_3d.add(HF::Chunking(chunk_size, 3));
    create_props_4d.add(HF::Chunking(chunk_size, 4));
    create_props_rgb.add(HF::Chunking(1, height, width, 3));
    create_props_depth.add(HF::Chunking(1, height, width));
    create_props_normal.add(HF::Chunking(1, height, width, 3));
    create_props_motion.add(HF::Chunking(1, height, width, 2));

    d.time_group->createDataSet<RBigIndex>("step", space_1d, create_props_1d);
    d.time_group->createDataSet<RBigFloat>("time_point", space_1d, create_props_1d);

    if (has_rgb) {
        d.rgb_group->createDataSet<uint8_t>("pixel", space_rgb, create_props_rgb);
    }

    if (has_depth) {
        d.depth_group->createDataSet<float>("pixel", space_depth, create_props_depth);
    }

    if (has_normal) {
        d.normal_group->createDataSet<float>("pixel", space_normal, create_props_normal);
    }

    if (has_motion) {
        d.motion_group->createDataSet<float>("pixel", space_motion, create_props_motion);
    }

    if (has_entity_info) {
        d.entity_info_group->createDataSet<RIndex>("entity_type", space_depth, create_props_depth);
        d.entity_info_group->createDataSet<RIndex>("entity_gid", space_depth, create_props_depth);
    }

    if (has_camera_state) {
        d.camera_state_group->createDataSet<RFloat>("position", space_3d, create_props_3d);
        d.camera_state_group->createDataSet<RFloat>("orientation", space_4d, create_props_4d);
        d.camera_state_group->createDataSet<RFloat>("focus_distance", space_1d, create_props_1d);
        d.camera_state_group->createDataSet<RFloat>("field_of_view", space_1d, create_props_1d);
    }

    _camera_record_data[camera_name] = d;
    return _camera_record_data[camera_name];
}

RS_SimulationHDF5Recorder::RaySensorRecordData &
RS_SimulationHDF5Recorder::_createOrGetRaySensorRecordData(
    const std::string &sensor_name,
    RSize ray_num,
    bool has_hit,
    bool has_entity_info,
    bool has_entity_detail,
    bool has_material,
    bool has_normal,
    bool has_velocity
) {
    if (_ray_sensor_record_data.find(sensor_name) != _ray_sensor_record_data.end()) {
        return _ray_sensor_record_data[sensor_name];
    }
    RaySensorRecordData d;
    d.ray_num = ray_num;
    d.root_group = make_shared<HF::Group>(_entity_root_group->createGroup(sensor_name));
    auto &root_group = d.root_group;

    // Frames are stored flat: each frame appends ray_num rows. ray_num is constant, so it is
    // written once as a root attribute to reshape the flat data back to per-frame.
    root_group->createAttribute<RSize>("ray_num", HF::DataSpace::From(ray_num)).write(ray_num);

    d.time_group = make_shared<HF::Group>(root_group->createGroup("frames/time"));
    if (has_hit) {
        d.hit_group = make_shared<HF::Group>(root_group->createGroup("frames/hit"));
    }
    if (has_entity_info) {
        d.entity_info_group =
            make_shared<HF::Group>(root_group->createGroup("frames/entity_info"));
    }
    if (has_entity_detail) {
        d.entity_detail_group =
            make_shared<HF::Group>(root_group->createGroup("frames/entity_detail"));
    }
    if (has_material) {
        d.material_group = make_shared<HF::Group>(root_group->createGroup("frames/material"));
    }
    if (has_normal) {
        d.normal_group = make_shared<HF::Group>(root_group->createGroup("frames/normal"));
    }
    if (has_velocity) {
        d.velocity_group = make_shared<HF::Group>(root_group->createGroup("frames/velocity"));
    }
    d.sensor_state_group =
        make_shared<HF::Group>(root_group->createGroup("frames/sensor_state"));

    HF::DataSpace space_1d({0}, {HF::DataSpace::UNLIMITED});
    HF::DataSpace space_3d({0, 3}, {HF::DataSpace::UNLIMITED, 3});
    HF::DataSpace space_4d({0, 4}, {HF::DataSpace::UNLIMITED, 4});
    HF::DataSetCreateProps create_props_1d, create_props_3d, create_props_4d;
    create_props_1d.add(HF::Chunking(chunk_size));
    create_props_3d.add(HF::Chunking(chunk_size, 3));
    create_props_4d.add(HF::Chunking(chunk_size, 4));

    d.time_group->createDataSet<RBigIndex>("step", space_1d, create_props_1d);
    d.time_group->createDataSet<RBigFloat>("time_point", space_1d, create_props_1d);

    if (has_hit) {
        d.hit_group->createDataSet<uint8_t>("is_hit", space_1d, create_props_1d);
        d.hit_group->createDataSet<RFloat>("distance", space_1d, create_props_1d);
        d.hit_group->createDataSet<RFloat>("hit_position", space_3d, create_props_3d);
    }
    if (has_entity_info) {
        d.entity_info_group->createDataSet<RIndex>("entity_gid", space_1d, create_props_1d);
        d.entity_info_group->createDataSet<RIndex>("entity_type", space_1d, create_props_1d);
    }
    if (has_entity_detail) {
        d.entity_detail_group->createDataSet<RIndex>("body_sid", space_1d, create_props_1d);
        d.entity_detail_group->createDataSet<RIndex>("segment_bid", space_1d, create_props_1d);
        d.entity_detail_group->createDataSet<RIndex>("voxel_sid", space_1d, create_props_1d);
    }
    if (has_material) {
        d.material_group->createDataSet<RIndex>("material_id", space_1d, create_props_1d);
        d.material_group->createDataSet<RFloat>("emission", space_1d, create_props_1d);
    }
    if (has_normal) {
        d.normal_group->createDataSet<RFloat>("normal", space_3d, create_props_3d);
    }
    if (has_velocity) {
        d.velocity_group->createDataSet<RFloat>("v_hit", space_3d, create_props_3d);
        d.velocity_group->createDataSet<RFloat>("v_rel", space_3d, create_props_3d);
        d.velocity_group->createDataSet<RFloat>("v_radial", space_1d, create_props_1d);
    }
    d.sensor_state_group->createDataSet<RFloat>("position", space_3d, create_props_3d);
    d.sensor_state_group->createDataSet<RFloat>("orientation", space_4d, create_props_4d);

    _ray_sensor_record_data[sensor_name] = d;
    return _ray_sensor_record_data[sensor_name];
}

void RS_SimulationHDF5Recorder::_writeStructureVoxelFrame(
    StructureRecordData &d,
    const RS_WorldStructureRecordBase::Voxel &v
) {
    RBigSize v_num = v.object_num;
    auto v_body_sid_buf = make_unique<RIndex[]>(v_num);
    auto v_voxel_sid_buf = make_unique<RIndex[]>(v_num);
    auto v_segment_bid_buf = make_unique<RIndex[]>(v_num);
    auto v_index_xyz_buf = make_unique<RSignedIndex[]>(v_num * 3);
    auto v_connectivity_buf = make_unique<uint8_t[]>(v_num);
    auto v_is_surface_buf = make_unique<bool[]>(v_num);
    auto v_is_rigid_buf = make_unique<bool[]>(v_num);
    auto v_material_buf = make_unique<RIndex[]>(v_num);
    auto v_expansion_buf = make_unique<RFloat[]>(v_num);
    auto v_orientation_buf = make_unique<RFloat[]>(v_num * 4);
    auto v_position_buf = make_unique<RFloat[]>(v_num * 3);
    auto v_ppp_offset_buf = make_unique<RFloat[]>(v_num * 3);
    auto v_nnn_offset_buf = make_unique<RFloat[]>(v_num * 3);
    auto v_linear_velocity_buf = make_unique<RFloat[]>(v_num * 3);
    auto v_angular_velocity_buf = make_unique<RFloat[]>(v_num * 3);
    auto v_poissons_strain_buf = make_unique<RFloat[]>(v_num * 3);

    _pool
        .submitLoop(
            static_cast<RIndex>(0),
            v_num,
            [&v,
             &v_voxel_sid_buf,
             &v_body_sid_buf,
             &v_segment_bid_buf,
             &v_index_xyz_buf,
             &v_connectivity_buf,
             &v_is_surface_buf,
             &v_is_rigid_buf,
             &v_material_buf,
             &v_expansion_buf,
             &v_orientation_buf,
             &v_position_buf,
             &v_ppp_offset_buf,
             &v_nnn_offset_buf,
             &v_linear_velocity_buf,
             &v_angular_velocity_buf,
             &v_poissons_strain_buf](const RIndex i) {
                v_body_sid_buf[i] = v.body_sid[i];
                v_voxel_sid_buf[i] = v.voxel_sid[i];
                v_segment_bid_buf[i] = v.segment_bid[i];
                v_index_xyz_buf[i * 3] = v.index_x[i];
                v_index_xyz_buf[i * 3 + 1] = v.index_y[i];
                v_index_xyz_buf[i * 3 + 2] = v.index_z[i];
                v_connectivity_buf[i] = v.connectivity[i];
                v_is_surface_buf[i] = v.is_surface[i];
                v_is_rigid_buf[i] = v.is_rigid[i];
                v_material_buf[i] = v.material[i];
                v_expansion_buf[i] = v.expansion[i];
                v_orientation_buf[i * 4] = v.orientation[i].x;
                v_orientation_buf[i * 4 + 1] = v.orientation[i].y;
                v_orientation_buf[i * 4 + 2] = v.orientation[i].z;
                v_orientation_buf[i * 4 + 3] = v.orientation[i].w;
                v_position_buf[i * 3] = v.position[i].x;
                v_position_buf[i * 3 + 1] = v.position[i].y;
                v_position_buf[i * 3 + 2] = v.position[i].z;
                v_ppp_offset_buf[i * 3] = v.ppp_offset[i].x;
                v_ppp_offset_buf[i * 3 + 1] = v.ppp_offset[i].y;
                v_ppp_offset_buf[i * 3 + 2] = v.ppp_offset[i].z;
                v_nnn_offset_buf[i * 3] = v.nnn_offset[i].x;
                v_nnn_offset_buf[i * 3 + 1] = v.nnn_offset[i].y;
                v_nnn_offset_buf[i * 3 + 2] = v.nnn_offset[i].z;
                v_linear_velocity_buf[i * 3] = v.linear_velocity[i].x;
                v_linear_velocity_buf[i * 3 + 1] = v.linear_velocity[i].y;
                v_linear_velocity_buf[i * 3 + 2] = v.linear_velocity[i].z;
                v_angular_velocity_buf[i * 3] = v.angular_velocity[i].x;
                v_angular_velocity_buf[i * 3 + 1] = v.angular_velocity[i].y;
                v_angular_velocity_buf[i * 3 + 2] = v.angular_velocity[i].z;
                v_poissons_strain_buf[i * 3] = v.poissons_strain[i].x;
                v_poissons_strain_buf[i * 3 + 1] = v.poissons_strain[i].y;
                v_poissons_strain_buf[i * 3 + 2] = v.poissons_strain[i].z;
            }
        )
        .wait();

    _appendToGroupNd(d.voxels_group, "voxel_num", &v_num, 1, 1);
    _appendToGroupNd(d.voxels_group, "voxel_offset", &d.voxel_offset, 1, 1);
    _appendToGroupNd(d.voxels_group, "body_sid", v_body_sid_buf.get(), v_num, 1);
    _appendToGroupNd(d.voxels_group, "voxel_sid", v_voxel_sid_buf.get(), v_num, 1);
    _appendToGroupNd(d.voxels_group, "segment_bid", v_segment_bid_buf.get(), v_num, 1);
    _appendToGroupNd(d.voxels_group, "index_xyz", v_index_xyz_buf.get(), v_num, 3);
    _appendToGroupNd(d.voxels_group, "connectivity", v_connectivity_buf.get(), v_num, 1);
    _appendToGroupNd(d.voxels_group, "is_surface", v_is_surface_buf.get(), v_num, 1);
    _appendToGroupNd(d.voxels_group, "is_rigid", v_is_rigid_buf.get(), v_num, 1);
    _appendToGroupNd(d.voxels_group, "material", v_material_buf.get(), v_num, 1);
    _appendToGroupNd(d.voxels_group, "expansion", v_expansion_buf.get(), v_num, 1);
    _appendToGroupNd(d.voxels_group, "orientation", v_orientation_buf.get(), v_num, 4);
    _appendToGroupNd(d.voxels_group, "position", v_position_buf.get(), v_num, 3);
    _appendToGroupNd(d.voxels_group, "ppp_offset", v_ppp_offset_buf.get(), v_num, 3);
    _appendToGroupNd(d.voxels_group, "nnn_offset", v_nnn_offset_buf.get(), v_num, 3);
    _appendToGroupNd(d.voxels_group, "linear_velocity", v_linear_velocity_buf.get(), v_num, 3);
    _appendToGroupNd(d.voxels_group, "angular_velocity", v_angular_velocity_buf.get(), v_num, 3);
    _appendToGroupNd(d.voxels_group, "poissons_strain", v_poissons_strain_buf.get(), v_num, 3);
    d.voxel_offset += v_num;
}

void RS_SimulationHDF5Recorder::_writeStructureLinkFrame(
    StructureRecordData &d,
    const RS_WorldStructureRecordBase::Link &l
) {
    RBigSize l_num = l.object_num;
    auto l_voxel_neg_sid_buf = make_unique<RIndex[]>(l_num);
    auto l_voxel_pos_sid_buf = make_unique<RIndex[]>(l_num);
    auto l_is_failed_buf = make_unique<bool[]>(l_num);
    auto l_strain_buf = make_unique<RFloat[]>(l_num);
    auto l_stress_buf = make_unique<RFloat[]>(l_num);

    _pool
        .submitLoop(
            static_cast<RIndex>(0),
            l_num,
            [&l,
             &l_voxel_neg_sid_buf,
             &l_voxel_pos_sid_buf,
             &l_is_failed_buf,
             &l_strain_buf,
             &l_stress_buf](const RIndex i) {
                l_voxel_neg_sid_buf[i] = l.voxel_neg_sid[i];
                l_voxel_pos_sid_buf[i] = l.voxel_pos_sid[i];
                l_is_failed_buf[i] = l.is_failed[i];
                l_strain_buf[i] = l.strain[i];
                l_stress_buf[i] = l.stress[i];
            }
        )
        .wait();

    _appendToGroupNd(d.links_group, "link_num", &l_num, 1, 1);
    _appendToGroupNd(d.links_group, "link_offset", &d.link_offset, 1, 1);
    _appendToGroupNd(d.links_group, "voxel_neg_sid", l_voxel_neg_sid_buf.get(), l_num, 1);
    _appendToGroupNd(d.links_group, "voxel_pos_sid", l_voxel_pos_sid_buf.get(), l_num, 1);
    _appendToGroupNd(d.links_group, "is_failed", l_is_failed_buf.get(), l_num, 1);
    _appendToGroupNd(d.links_group, "strain", l_strain_buf.get(), l_num, 1);
    _appendToGroupNd(d.links_group, "stress", l_stress_buf.get(), l_num, 1);
    d.link_offset += l_num;
}

void RS_SimulationHDF5Recorder::_writeStructureRigidBodyFrame(
    StructureRecordData &d,
    const RS_WorldStructureRecordBase::RigidBody &rb
) {
    RBigSize rb_num = rb.object_num;
    auto rb_rigid_body_sid_buf = make_unique<RIndex[]>(rb_num);
    auto rb_body_sid_buf = make_unique<RIndex[]>(rb_num);
    auto rb_segment_bid_buf = make_unique<RIndex[]>(rb_num);
    auto rb_mass_buf = make_unique<RFloat[]>(rb_num);
    auto rb_orientation_buf = make_unique<RFloat[]>(rb_num * 4);
    auto rb_com_buf = make_unique<RFloat[]>(rb_num * 3);
    auto rb_linear_velocity_buf = make_unique<RFloat[]>(rb_num * 3);
    auto rb_angular_velocity_buf = make_unique<RFloat[]>(rb_num * 3);

    _pool
        .submitLoop(
            static_cast<RIndex>(0),
            rb_num,
            [&rb,
             &rb_rigid_body_sid_buf,
             &rb_body_sid_buf,
             &rb_segment_bid_buf,
             &rb_mass_buf,
             &rb_orientation_buf,
             &rb_com_buf,
             &rb_linear_velocity_buf,
             &rb_angular_velocity_buf](const RIndex i) {
                rb_rigid_body_sid_buf[i] = rb.rigid_body_sid[i];
                rb_body_sid_buf[i] = rb.body_sid[i];
                rb_segment_bid_buf[i] = rb.segment_bid[i];
                rb_mass_buf[i] = rb.mass[i];
                rb_orientation_buf[i * 4] = rb.orientation[i].x;
                rb_orientation_buf[i * 4 + 1] = rb.orientation[i].y;
                rb_orientation_buf[i * 4 + 2] = rb.orientation[i].z;
                rb_orientation_buf[i * 4 + 3] = rb.orientation[i].w;
                rb_com_buf[i * 3] = rb.com[i].x;
                rb_com_buf[i * 3 + 1] = rb.com[i].y;
                rb_com_buf[i * 3 + 2] = rb.com[i].z;
                rb_linear_velocity_buf[i * 3] = rb.linear_velocity[i].x;
                rb_linear_velocity_buf[i * 3 + 1] = rb.linear_velocity[i].y;
                rb_linear_velocity_buf[i * 3 + 2] = rb.linear_velocity[i].z;
                rb_angular_velocity_buf[i * 3] = rb.angular_velocity[i].x;
                rb_angular_velocity_buf[i * 3 + 1] = rb.angular_velocity[i].y;
                rb_angular_velocity_buf[i * 3 + 2] = rb.angular_velocity[i].z;
            }
        )
        .wait();

    _appendToGroupNd(d.rigid_bodies_group, "rigid_body_num", &rb_num, 1, 1);
    _appendToGroupNd(d.rigid_bodies_group, "rigid_body_offset", &d.rigid_body_offset, 1, 1);
    _appendToGroupNd(d.rigid_bodies_group, "rigid_body_sid", rb_rigid_body_sid_buf.get(), rb_num, 1);
    _appendToGroupNd(d.rigid_bodies_group, "body_sid", rb_body_sid_buf.get(), rb_num, 1);
    _appendToGroupNd(d.rigid_bodies_group, "segment_bid", rb_segment_bid_buf.get(), rb_num, 1);
    _appendToGroupNd(d.rigid_bodies_group, "mass", rb_mass_buf.get(), rb_num, 1);
    _appendToGroupNd(d.rigid_bodies_group, "orientation", rb_orientation_buf.get(), rb_num, 4);
    _appendToGroupNd(d.rigid_bodies_group, "com", rb_com_buf.get(), rb_num, 3);
    _appendToGroupNd(
        d.rigid_bodies_group, "linear_velocity", rb_linear_velocity_buf.get(), rb_num, 3
    );
    _appendToGroupNd(
        d.rigid_bodies_group, "angular_velocity", rb_angular_velocity_buf.get(), rb_num, 3
    );
    d.rigid_body_offset += rb_num;
}

void RS_SimulationHDF5Recorder::_writeStructureJointFrame(
    StructureRecordData &d,
    const RS_WorldStructureRecordBase::Joint &j
) {
    RBigSize j_num = j.object_num;
    auto j_rigid_body_a_sid_buf = make_unique<RIndex[]>(j_num);
    auto j_rigid_body_b_sid_buf = make_unique<RIndex[]>(j_num);
    auto j_type_buf = make_unique<int[]>(j_num);
    auto j_position_buf = make_unique<RFloat[]>(j_num * 3);
    auto j_axis_buf = make_unique<RFloat[]>(j_num * 3);
    // Hinge fields
    auto j_hinge_signal_sid_buf = make_unique<RIndex[]>(j_num);
    auto j_hinge_min_buf = make_unique<RFloat[]>(j_num);
    auto j_hinge_max_buf = make_unique<RFloat[]>(j_num);
    auto j_hinge_angle_buf = make_unique<RFloat[]>(j_num);
    auto j_hinge_torque_buf = make_unique<RFloat[]>(j_num);
    // Prismatic fields
    auto j_prismatic_signal_sid_buf = make_unique<RIndex[]>(j_num);
    auto j_prismatic_min_buf = make_unique<RFloat[]>(j_num);
    auto j_prismatic_max_buf = make_unique<RFloat[]>(j_num);
    auto j_prismatic_distance_buf = make_unique<RFloat[]>(j_num);
    auto j_prismatic_force_buf = make_unique<RFloat[]>(j_num);

    _pool
        .submitLoop(
            static_cast<RIndex>(0),
            j_num,
            [&j,
             &j_rigid_body_a_sid_buf,
             &j_rigid_body_b_sid_buf,
             &j_type_buf,
             &j_position_buf,
             &j_axis_buf,
             &j_hinge_signal_sid_buf,
             &j_hinge_min_buf,
             &j_hinge_max_buf,
             &j_hinge_angle_buf,
             &j_hinge_torque_buf,
             &j_prismatic_signal_sid_buf,
             &j_prismatic_min_buf,
             &j_prismatic_max_buf,
             &j_prismatic_distance_buf,
             &j_prismatic_force_buf](const RIndex i) {
                j_rigid_body_a_sid_buf[i] = j.rigid_body_a_sid[i];
                j_rigid_body_b_sid_buf[i] = j.rigid_body_b_sid[i];
                j_type_buf[i] = static_cast<int>(j.type[i]);
                j_position_buf[i * 3] = j.position[i].x;
                j_position_buf[i * 3 + 1] = j.position[i].y;
                j_position_buf[i * 3 + 2] = j.position[i].z;
                j_axis_buf[i * 3] = j.axis[i].x;
                j_axis_buf[i * 3 + 1] = j.axis[i].y;
                j_axis_buf[i * 3 + 2] = j.axis[i].z;
                // Hinge
                j_hinge_signal_sid_buf[i] = j.hinge_signal_sid[i];
                j_hinge_min_buf[i] = j.hinge_min[i];
                j_hinge_max_buf[i] = j.hinge_max[i];
                j_hinge_angle_buf[i] = j.hinge_angle[i];
                j_hinge_torque_buf[i] = j.hinge_torque[i];
                // Prismatic
                j_prismatic_signal_sid_buf[i] = j.prismatic_signal_sid[i];
                j_prismatic_min_buf[i] = j.prismatic_min[i];
                j_prismatic_max_buf[i] = j.prismatic_max[i];
                j_prismatic_distance_buf[i] = j.prismatic_distance[i];
                j_prismatic_force_buf[i] = j.prismatic_force[i];
            }
        )
        .wait();

    _appendToGroupNd(d.joints_group, "joint_num", &j_num, 1, 1);
    _appendToGroupNd(d.joints_group, "joint_offset", &d.joint_offset, 1, 1);
    _appendToGroupNd(d.joints_group, "rigid_body_a_sid", j_rigid_body_a_sid_buf.get(), j_num, 1);
    _appendToGroupNd(d.joints_group, "rigid_body_b_sid", j_rigid_body_b_sid_buf.get(), j_num, 1);
    _appendToGroupNd(d.joints_group, "type", j_type_buf.get(), j_num, 1);
    _appendToGroupNd(d.joints_group, "position", j_position_buf.get(), j_num, 3);
    _appendToGroupNd(d.joints_group, "axis", j_axis_buf.get(), j_num, 3);
    // Hinge
    _appendToGroupNd(d.joints_group, "hinge_signal_sid", j_hinge_signal_sid_buf.get(), j_num, 1);
    _appendToGroupNd(d.joints_group, "hinge_min", j_hinge_min_buf.get(), j_num, 1);
    _appendToGroupNd(d.joints_group, "hinge_max", j_hinge_max_buf.get(), j_num, 1);
    _appendToGroupNd(d.joints_group, "hinge_angle", j_hinge_angle_buf.get(), j_num, 1);
    _appendToGroupNd(d.joints_group, "hinge_torque", j_hinge_torque_buf.get(), j_num, 1);
    // Prismatic
    _appendToGroupNd(
        d.joints_group, "prismatic_signal_sid", j_prismatic_signal_sid_buf.get(), j_num, 1
    );
    _appendToGroupNd(d.joints_group, "prismatic_min", j_prismatic_min_buf.get(), j_num, 1);
    _appendToGroupNd(d.joints_group, "prismatic_max", j_prismatic_max_buf.get(), j_num, 1);
    _appendToGroupNd(d.joints_group, "prismatic_distance", j_prismatic_distance_buf.get(), j_num, 1);
    _appendToGroupNd(d.joints_group, "prismatic_force", j_prismatic_force_buf.get(), j_num, 1);
    d.joint_offset += j_num;
}

void RS_SimulationHDF5Recorder::_writeCameraRgbFrame(
    CameraRecordData &d,
    const RS_WorldCameraRecordBase::RGB &rgb
) {
    HF::DataSetAccessProps access_props;
    auto dataset = d.rgb_group->getDataSet("pixel", access_props);
    auto dimensions = dataset.getDimensions();
    size_t offset = dimensions[0];
    dimensions[0] += 1;
    dataset.resize(dimensions);
    dataset.select({offset, 0, 0, 0}, {1, d.height, d.width, 3})
        .write_raw(reinterpret_cast<uint8_t *>(rgb.pixel));
}

void RS_SimulationHDF5Recorder::_writeCameraDepthFrame(
    CameraRecordData &d,
    const RS_WorldCameraRecordBase::Depth &depth
) {
    HF::DataSetAccessProps access_props;
    auto dataset = d.depth_group->getDataSet("pixel", access_props);
    auto dimensions = dataset.getDimensions();
    size_t offset = dimensions[0];
    dimensions[0] += 1;
    dataset.resize(dimensions);
    dataset.select({offset, 0, 0}, {1, d.height, d.width})
        .write_raw(reinterpret_cast<float *>(depth.pixel));
}

void RS_SimulationHDF5Recorder::_writeCameraNormalFrame(
    CameraRecordData &d,
    const RS_WorldCameraRecordBase::Normal &normal
) {
    HF::DataSetAccessProps access_props;
    auto dataset = d.normal_group->getDataSet("pixel", access_props);
    auto dimensions = dataset.getDimensions();
    size_t offset = dimensions[0];
    dimensions[0] += 1;
    dataset.resize(dimensions);
    dataset.select({offset, 0, 0, 0}, {1, d.height, d.width, 3})
        .write_raw(reinterpret_cast<float *>(normal.pixel));
}

void RS_SimulationHDF5Recorder::_writeCameraMotionFrame(
    CameraRecordData &d,
    const RS_WorldCameraRecordBase::Motion &motion
) {
    HF::DataSetAccessProps access_props;
    auto dataset = d.motion_group->getDataSet("pixel", access_props);
    auto dimensions = dataset.getDimensions();
    size_t offset = dimensions[0];
    dimensions[0] += 1;
    dataset.resize(dimensions);
    dataset.select({offset, 0, 0, 0}, {1, d.height, d.width, 2})
        .write_raw(reinterpret_cast<float *>(motion.pixel));
}

void RS_SimulationHDF5Recorder::_writeCameraEntityInfoFrame(
    CameraRecordData &d,
    const RS_WorldCameraRecordBase::EntityInfo &entity_info
) {
    HF::DataSetAccessProps access_props;

    auto entity_type_dataset = d.entity_info_group->getDataSet("entity_type", access_props);
    auto entity_type_dimensions = entity_type_dataset.getDimensions();
    size_t offset = entity_type_dimensions[0];
    entity_type_dimensions[0] += 1;
    entity_type_dataset.resize(entity_type_dimensions);
    entity_type_dataset.select({offset, 0, 0}, {1, d.height, d.width})
        .write_raw(reinterpret_cast<RIndex *>(entity_info.entity_type));

    auto entity_gid_dataset = d.entity_info_group->getDataSet("entity_gid", access_props);
    auto entity_gid_dimensions = entity_gid_dataset.getDimensions();
    entity_gid_dimensions[0] += 1;
    entity_gid_dataset.resize(entity_gid_dimensions);
    entity_gid_dataset.select({offset, 0, 0}, {1, d.height, d.width})
        .write_raw(entity_info.entity_gid);
}

void RS_SimulationHDF5Recorder::_writeCameraStateFrame(
    CameraRecordData &d,
    const RS_WorldCameraRecordBase::State &camera_state
) {
    HF::DataSetAccessProps access_props;

    // Position (3D vector)
    auto position_dataset = d.camera_state_group->getDataSet("position", access_props);
    auto position_dimensions = position_dataset.getDimensions();
    size_t offset = position_dimensions[0];
    position_dimensions[0] += 1;
    position_dataset.resize(position_dimensions);

    RFloat position_buf[3] = {
        camera_state.position->x, camera_state.position->y, camera_state.position->z
    };
    position_dataset.select({offset, 0}, {1, 3}).write_raw(position_buf);

    // Orientation (quaternion - 4D vector)
    auto orientation_dataset = d.camera_state_group->getDataSet("orientation", access_props);
    auto orientation_dimensions = orientation_dataset.getDimensions();
    orientation_dimensions[0] += 1;
    orientation_dataset.resize(orientation_dimensions);

    RFloat orientation_buf[4] = {
        camera_state.orientation->x,
        camera_state.orientation->y,
        camera_state.orientation->z,
        camera_state.orientation->w
    };
    orientation_dataset.select({offset, 0}, {1, 4}).write_raw(orientation_buf);

    // Focus distance (scalar)
    auto focus_dataset = d.camera_state_group->getDataSet("focus_distance", access_props);
    auto focus_dimensions = focus_dataset.getDimensions();
    focus_dimensions[0] += 1;
    focus_dataset.resize(focus_dimensions);
    focus_dataset.select({offset}, {1}).write_raw(camera_state.focus_distance);

    // Field of view (scalar)
    auto fov_dataset = d.camera_state_group->getDataSet("field_of_view", access_props);
    auto fov_dimensions = fov_dataset.getDimensions();
    fov_dimensions[0] += 1;
    fov_dataset.resize(fov_dimensions);
    fov_dataset.select({offset}, {1}).write_raw(camera_state.field_of_view);
}

void RS_SimulationHDF5Recorder::_writeRaySensorHitFrame(
    RaySensorRecordData &d,
    const RS_WorldRaySensorRecordBase::Hit &hit
) {
    RBigSize rn = d.ray_num;
    _appendToGroupNd(d.hit_group, "is_hit", hit.is_hit, rn, 1);
    _appendToGroupNd(d.hit_group, "distance", hit.distance, rn, 1);
    _appendToGroupNd(d.hit_group, "hit_position", reinterpret_cast<RFloat *>(hit.hit_position), rn, 3);
}

void RS_SimulationHDF5Recorder::_writeRaySensorEntityInfoFrame(
    RaySensorRecordData &d,
    const RS_WorldRaySensorRecordBase::EntityInfo &entity_info
) {
    RBigSize rn = d.ray_num;
    _appendToGroupNd(d.entity_info_group, "entity_gid", entity_info.entity_gid, rn, 1);
    _appendToGroupNd(
        d.entity_info_group, "entity_type", reinterpret_cast<RIndex *>(entity_info.entity_type),
        rn, 1
    );
}

void RS_SimulationHDF5Recorder::_writeRaySensorEntityDetailFrame(
    RaySensorRecordData &d,
    const RS_WorldRaySensorRecordBase::EntityDetail &entity_detail
) {
    RBigSize rn = d.ray_num;
    _appendToGroupNd(d.entity_detail_group, "body_sid", entity_detail.body_sid, rn, 1);
    _appendToGroupNd(d.entity_detail_group, "segment_bid", entity_detail.segment_bid, rn, 1);
    _appendToGroupNd(d.entity_detail_group, "voxel_sid", entity_detail.voxel_sid, rn, 1);
}

void RS_SimulationHDF5Recorder::_writeRaySensorMaterialFrame(
    RaySensorRecordData &d,
    const RS_WorldRaySensorRecordBase::Material &material
) {
    RBigSize rn = d.ray_num;
    _appendToGroupNd(d.material_group, "material_id", material.material_id, rn, 1);
    _appendToGroupNd(d.material_group, "emission", material.emission, rn, 1);
}

void RS_SimulationHDF5Recorder::_writeRaySensorNormalFrame(
    RaySensorRecordData &d,
    const RS_WorldRaySensorRecordBase::Normal &normal
) {
    _appendToGroupNd(d.normal_group, "normal", reinterpret_cast<RFloat *>(normal.normal), d.ray_num, 3);
}

void RS_SimulationHDF5Recorder::_writeRaySensorVelocityFrame(
    RaySensorRecordData &d,
    const RS_WorldRaySensorRecordBase::Velocity &velocity
) {
    RBigSize rn = d.ray_num;
    _appendToGroupNd(d.velocity_group, "v_hit", reinterpret_cast<RFloat *>(velocity.v_hit), rn, 3);
    _appendToGroupNd(d.velocity_group, "v_rel", reinterpret_cast<RFloat *>(velocity.v_rel), rn, 3);
    _appendToGroupNd(d.velocity_group, "v_radial", velocity.v_radial, rn, 1);
}

void RS_SimulationHDF5Recorder::_writeRaySensorStateFrame(
    RaySensorRecordData &d,
    const RS_WorldRaySensorRecordBase::State &sensor_state
) {
    RFloat position_buf[3] = {
        sensor_state.position->x, sensor_state.position->y, sensor_state.position->z
    };
    _appendToGroupNd(d.sensor_state_group, "position", position_buf, 1, 3);
    RFloat orientation_buf[4] = {
        sensor_state.orientation->x,
        sensor_state.orientation->y,
        sensor_state.orientation->z,
        sensor_state.orientation->w
    };
    _appendToGroupNd(d.sensor_state_group, "orientation", orientation_buf, 1, 4);
}

template <typename T>
void RS_SimulationHDF5Recorder::_appendToGroupNd(
    shared_ptr<HighFive::Group> &group,
    const std::string &dataset_name,
    T *data,
    size_t num,
    size_t dim
) {
    HF::DataSetAccessProps access_props;
    access_props.add(HighFive::Caching(256, 256 * 4 * 4 * chunk_size));
    auto dataset = group->getDataSet(dataset_name, access_props);
    auto dimensions = dataset.getDimensions();
    size_t offset = dimensions[0];
    dimensions[0] += num;
    dataset.resize(dimensions);
    if (dim > 1)
        dataset.select({offset, 0}, {num, dim}).write_raw(data);
    else
        dataset.select({offset}, {num}).write_raw(data);
}

}  // namespace rise
