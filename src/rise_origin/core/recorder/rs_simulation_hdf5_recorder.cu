// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include "rs_simulation_hdf5_recorder.h"

#if !H5_VERSION_GE(1, 10, 1)
#error "Minimum HDF5 version required is 1.10.1"
#endif
using namespace std;
namespace HF = HighFive;

namespace rise {
template <typename T, typename F>
std::vector<RIndex>
RS_SimulationHDF5Recorder::_filter(const T &records, RSize record_num, F condition) {
    // For below 4M entries, use 4 cores to parallel process
    RSize block_num = RS_CEIL_DIV(record_num, 32768);
    RS_ThreadLoopBlocks blocks(0, record_num, block_num);
    std::vector<std::future<std::vector<RIndex>>> results;
    results.reserve(block_num);
    for (RIndex block = 0; block < block_num; ++block) {
        RSize start = blocks.start(block);
        RSize end = blocks.end(block);
        if (end > start) {
            results.push_back(_pool.submitTask([records, condition, start, end]() {
                std::vector<RIndex> true_indices;
                for (RIndex i = start; i < end; i++) {
                    if (condition(records, i)) true_indices.push_back(i);
                }
                return true_indices;
            }));
        }
    }
    std::vector<RIndex> true_indices;
    for (auto &result : results) {
        auto indices = result.get();
        true_indices.insert(true_indices.end(), indices.begin(), indices.end());
    }
    return true_indices;
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

void RS_SimulationHDF5Recorder::_appendCameraFrame(
    std::shared_ptr<HighFive::Group> &group,
    const std::string &dataset_name,
    RColor3b *pixels,
    RSize width,
    RSize height
) {
    HF::DataSetAccessProps access_props;
    auto dataset = group->getDataSet(dataset_name, access_props);
    auto dimensions = dataset.getDimensions();
    size_t offset = dimensions[0];
    dimensions[0] += 1;
    dataset.resize(dimensions);
    dataset.select({offset, 0, 0, 0}, {1, height, width, 3})
        .write_raw(reinterpret_cast<uint8_t *>(pixels));
}

void RS_SimulationHDF5Recorder::_appendCameraState(
    std::shared_ptr<HighFive::Group> &group,
    const RS_SimulationCameraStateRecord &camera_state
) {
    HF::DataSetAccessProps access_props;

    // Position (3D vector)
    auto position_dataset = group->getDataSet("position", access_props);
    auto position_dimensions = position_dataset.getDimensions();
    size_t offset = position_dimensions[0];
    position_dimensions[0] += 1;
    position_dataset.resize(position_dimensions);

    RFloat position_buf[3] = {
        camera_state.position->x, camera_state.position->y, camera_state.position->z
    };
    position_dataset.select({offset, 0}, {1, 3}).write_raw(position_buf);

    // Orientation (quaternion - 4D vector)
    auto orientation_dataset = group->getDataSet("orientation", access_props);
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
    auto focus_dataset = group->getDataSet("focus_distance", access_props);
    auto focus_dimensions = focus_dataset.getDimensions();
    focus_dimensions[0] += 1;
    focus_dataset.resize(focus_dimensions);
    focus_dataset.select({offset}, {1}).write_raw(camera_state.focus_distance);

    // Field of view (scalar)
    auto fov_dataset = group->getDataSet("field_of_view", access_props);
    auto fov_dimensions = fov_dataset.getDimensions();
    fov_dimensions[0] += 1;
    fov_dataset.resize(fov_dimensions);
    fov_dataset.select({offset}, {1}).write_raw(camera_state.field_of_view);
}

RS_SimulationHDF5Recorder::RS_SimulationHDF5Recorder(const string &file_path) {
    HF::FileAccessProps file_access_props;
    file_access_props.add(HF::FileVersionBounds(H5F_LIBVER_V110, H5F_LIBVER_LATEST));
    _hdf5_file = make_unique<HF::File>(file_path, HF::File::Overwrite, file_access_props);
    _structure_record_root_group =
        make_unique<HF::Group>(_hdf5_file->createGroup("/structure_record"));
    _camera_record_root_group = make_unique<HF::Group>(_hdf5_file->createGroup("/camera_record"));
    H5Fstart_swmr_write(_hdf5_file->getId());
}

void RS_SimulationHDF5Recorder::start() {}

void RS_SimulationHDF5Recorder::syncStructureRecords(
    const std::string &name,
    const std::vector<RS_SimulationStructureRecord> &records
) {
    StructureRecordData &d = _createOrGetStructureRecordData(name);

    for (auto &record : records) {
        _appendToGroupNd(d.time_group, "step", record.step, 1, 1);
        _appendToGroupNd(d.time_group, "time_point", record.time, 1, 1);

        // Process voxels (only voxels with is_surface = true)
        auto surface_voxel_indices = _filter(
            record.voxels,
            record.voxels.object_num,
            [](const RS_SimulationVoxelRecord &v, size_t i) { return v.is_surface[i]; }
        );

        RBigSize v_num = surface_voxel_indices.size();
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
                [&record,
                 &surface_voxel_indices,
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
                    RIndex voxel_index = surface_voxel_indices[i];
                    v_body_sid_buf[i] = record.voxels.body_sid[voxel_index];
                    v_voxel_sid_buf[i] = record.voxels.voxel_sid[voxel_index];
                    v_segment_bid_buf[i] = record.voxels.segment_bid[voxel_index];
                    v_index_xyz_buf[i * 3] = record.voxels.index_x[voxel_index];
                    v_index_xyz_buf[i * 3 + 1] = record.voxels.index_y[voxel_index];
                    v_index_xyz_buf[i * 3 + 2] = record.voxels.index_z[voxel_index];
                    v_connectivity_buf[i] = record.voxels.connectivity[voxel_index];
                    v_is_surface_buf[i] = record.voxels.is_surface[voxel_index];
                    v_is_rigid_buf[i] = record.voxels.is_rigid[voxel_index];
                    v_material_buf[i] = record.voxels.material[voxel_index];
                    v_expansion_buf[i] = record.voxels.expansion[voxel_index];
                    v_orientation_buf[i * 4] = record.voxels.orientation[voxel_index].x;
                    v_orientation_buf[i * 4 + 1] = record.voxels.orientation[voxel_index].y;
                    v_orientation_buf[i * 4 + 2] = record.voxels.orientation[voxel_index].z;
                    v_orientation_buf[i * 4 + 3] = record.voxels.orientation[voxel_index].w;
                    v_position_buf[i * 3] = record.voxels.position[voxel_index].x;
                    v_position_buf[i * 3 + 1] = record.voxels.position[voxel_index].y;
                    v_position_buf[i * 3 + 2] = record.voxels.position[voxel_index].z;
                    v_ppp_offset_buf[i * 3] = record.voxels.ppp_offset[voxel_index].x;
                    v_ppp_offset_buf[i * 3 + 1] = record.voxels.ppp_offset[voxel_index].y;
                    v_ppp_offset_buf[i * 3 + 2] = record.voxels.ppp_offset[voxel_index].z;
                    v_nnn_offset_buf[i * 3] = record.voxels.nnn_offset[voxel_index].x;
                    v_nnn_offset_buf[i * 3 + 1] = record.voxels.nnn_offset[voxel_index].y;
                    v_nnn_offset_buf[i * 3 + 2] = record.voxels.nnn_offset[voxel_index].z;
                    v_linear_velocity_buf[i * 3] = record.voxels.linear_velocity[voxel_index].x;
                    v_linear_velocity_buf[i * 3 + 1] = record.voxels.linear_velocity[voxel_index].y;
                    v_linear_velocity_buf[i * 3 + 2] = record.voxels.linear_velocity[voxel_index].z;
                    v_angular_velocity_buf[i * 3] = record.voxels.angular_velocity[voxel_index].x;
                    v_angular_velocity_buf[i * 3 + 1] =
                        record.voxels.angular_velocity[voxel_index].y;
                    v_angular_velocity_buf[i * 3 + 2] =
                        record.voxels.angular_velocity[voxel_index].z;
                    v_poissons_strain_buf[i * 3] = record.voxels.poissons_strain[voxel_index].x;
                    v_poissons_strain_buf[i * 3 + 1] = record.voxels.poissons_strain[voxel_index].y;
                    v_poissons_strain_buf[i * 3 + 2] = record.voxels.poissons_strain[voxel_index].z;
                }
            )
            .wait();

        _appendToGroupNd(d.voxel_group, "voxel_num", &v_num, 1, 1);
        _appendToGroupNd(d.voxel_group, "voxel_offset", &d.voxel_offset, 1, 1);
        _appendToGroupNd(d.voxel_group, "body_sid", v_body_sid_buf.get(), v_num, 1);
        _appendToGroupNd(d.voxel_group, "voxel_sid", v_voxel_sid_buf.get(), v_num, 1);
        _appendToGroupNd(d.voxel_group, "segment_bid", v_segment_bid_buf.get(), v_num, 1);
        _appendToGroupNd(d.voxel_group, "index_xyz", v_index_xyz_buf.get(), v_num, 3);
        _appendToGroupNd(d.voxel_group, "connectivity", v_connectivity_buf.get(), v_num, 1);
        _appendToGroupNd(d.voxel_group, "is_surface", v_is_surface_buf.get(), v_num, 1);
        _appendToGroupNd(d.voxel_group, "is_rigid", v_is_rigid_buf.get(), v_num, 1);
        _appendToGroupNd(d.voxel_group, "material", v_material_buf.get(), v_num, 1);
        _appendToGroupNd(d.voxel_group, "expansion", v_expansion_buf.get(), v_num, 1);
        _appendToGroupNd(d.voxel_group, "orientation", v_orientation_buf.get(), v_num, 4);
        _appendToGroupNd(d.voxel_group, "position", v_position_buf.get(), v_num, 3);
        _appendToGroupNd(d.voxel_group, "ppp_offset", v_ppp_offset_buf.get(), v_num, 3);
        _appendToGroupNd(d.voxel_group, "nnn_offset", v_nnn_offset_buf.get(), v_num, 3);
        _appendToGroupNd(d.voxel_group, "linear_velocity", v_linear_velocity_buf.get(), v_num, 3);
        _appendToGroupNd(d.voxel_group, "angular_velocity", v_angular_velocity_buf.get(), v_num, 3);
        _appendToGroupNd(d.voxel_group, "poissons_strain", v_poissons_strain_buf.get(), v_num, 3);
        d.voxel_offset += v_num;

        // Process links (only links with is_tendon = true)
        auto tendon_link_indices = _filter(
            record.links, record.links.object_num, [](const RS_SimulationLinkRecord &l, size_t i) {
                return l.is_tendon[i];
            }
        );

        RBigSize l_num = tendon_link_indices.size();
        auto l_voxel_neg_sid_buf = std::make_unique<RIndex[]>(l_num);
        auto l_voxel_pos_sid_buf = std::make_unique<RIndex[]>(l_num);
        auto l_is_tendon_buf = std::make_unique<bool[]>(l_num);
        auto l_is_failed_buf = std::make_unique<bool[]>(l_num);
        auto l_strain_buf = std::make_unique<RFloat[]>(l_num);
        auto l_stress_buf = std::make_unique<RFloat[]>(l_num);

        _pool
            .submitLoop(
                static_cast<RIndex>(0),
                l_num,
                [&record,
                 &tendon_link_indices,
                 &l_voxel_neg_sid_buf,
                 &l_voxel_pos_sid_buf,
                 &l_is_tendon_buf,
                 &l_is_failed_buf,
                 &l_strain_buf,
                 &l_stress_buf](const RIndex i) {
                    RIndex link_index = tendon_link_indices[i];
                    l_voxel_neg_sid_buf[i] = record.links.voxel_neg_sid[link_index];
                    l_voxel_pos_sid_buf[i] = record.links.voxel_pos_sid[link_index];
                    l_is_tendon_buf[i] = record.links.is_tendon[link_index];
                    l_is_failed_buf[i] = record.links.is_failed[link_index];
                    l_strain_buf[i] = record.links.strain[link_index];
                    l_stress_buf[i] = record.links.stress[link_index];
                }
            )
            .wait();

        _appendToGroupNd(d.link_group, "link_num", &l_num, 1, 1);
        _appendToGroupNd(d.link_group, "link_offset", &d.link_offset, 1, 1);
        _appendToGroupNd(d.link_group, "voxel_neg_sid", l_voxel_neg_sid_buf.get(), l_num, 1);
        _appendToGroupNd(d.link_group, "voxel_pos_sid", l_voxel_pos_sid_buf.get(), l_num, 1);
        _appendToGroupNd(d.link_group, "is_tendon", l_is_tendon_buf.get(), l_num, 1);
        _appendToGroupNd(d.link_group, "is_failed", l_is_failed_buf.get(), l_num, 1);
        _appendToGroupNd(d.link_group, "strain", l_strain_buf.get(), l_num, 1);
        _appendToGroupNd(d.link_group, "stress", l_stress_buf.get(), l_num, 1);
        d.link_offset += l_num;

        // Process rigid bodies
        RBigSize rb_num = record.rigid_bodies.object_num;
        unique_ptr<RIndex[]> rb_rigid_body_sid_buf = make_unique<RIndex[]>(rb_num);
        unique_ptr<RIndex[]> rb_body_sid_buf = make_unique<RIndex[]>(rb_num);
        unique_ptr<RIndex[]> rb_segment_bid_buf = make_unique<RIndex[]>(rb_num);
        unique_ptr<RFloat[]> rb_mass_buf = make_unique<RFloat[]>(rb_num);
        unique_ptr<RFloat[]> rb_orientation_buf = make_unique<RFloat[]>(rb_num * 4);
        unique_ptr<RFloat[]> rb_com_buf = make_unique<RFloat[]>(rb_num * 3);
        unique_ptr<RFloat[]> rb_linear_velocity_buf = make_unique<RFloat[]>(rb_num * 3);
        unique_ptr<RFloat[]> rb_angular_velocity_buf = make_unique<RFloat[]>(rb_num * 3);

        _pool
            .submitLoop(
                static_cast<RIndex>(0),
                rb_num,
                [&record,
                 &rb_rigid_body_sid_buf,
                 &rb_body_sid_buf,
                 &rb_segment_bid_buf,
                 &rb_mass_buf,
                 &rb_orientation_buf,
                 &rb_com_buf,
                 &rb_linear_velocity_buf,
                 &rb_angular_velocity_buf](const RIndex i) {
                    rb_rigid_body_sid_buf[i] = record.rigid_bodies.rigid_body_sid[i];
                    rb_body_sid_buf[i] = record.rigid_bodies.body_sid[i];
                    rb_segment_bid_buf[i] = record.rigid_bodies.segment_bid[i];
                    rb_mass_buf[i] = record.rigid_bodies.mass[i];
                    rb_orientation_buf[i * 4] = record.rigid_bodies.orientation[i].x;
                    rb_orientation_buf[i * 4 + 1] = record.rigid_bodies.orientation[i].y;
                    rb_orientation_buf[i * 4 + 2] = record.rigid_bodies.orientation[i].z;
                    rb_orientation_buf[i * 4 + 3] = record.rigid_bodies.orientation[i].w;
                    rb_com_buf[i * 3] = record.rigid_bodies.com[i].x;
                    rb_com_buf[i * 3 + 1] = record.rigid_bodies.com[i].y;
                    rb_com_buf[i * 3 + 2] = record.rigid_bodies.com[i].z;
                    rb_linear_velocity_buf[i * 3] = record.rigid_bodies.linear_velocity[i].x;
                    rb_linear_velocity_buf[i * 3 + 1] = record.rigid_bodies.linear_velocity[i].y;
                    rb_linear_velocity_buf[i * 3 + 2] = record.rigid_bodies.linear_velocity[i].z;
                    rb_angular_velocity_buf[i * 3] = record.rigid_bodies.angular_velocity[i].x;
                    rb_angular_velocity_buf[i * 3 + 1] = record.rigid_bodies.angular_velocity[i].y;
                    rb_angular_velocity_buf[i * 3 + 2] = record.rigid_bodies.angular_velocity[i].z;
                }
            )
            .wait();

        _appendToGroupNd(d.rigid_body_group, "rigid_body_num", &rb_num, 1, 1);
        _appendToGroupNd(d.rigid_body_group, "rigid_body_offset", &d.rigid_body_offset, 1, 1);
        _appendToGroupNd(
            d.rigid_body_group, "rigid_body_sid", rb_rigid_body_sid_buf.get(), rb_num, 1
        );
        _appendToGroupNd(d.rigid_body_group, "body_sid", rb_body_sid_buf.get(), rb_num, 1);
        _appendToGroupNd(d.rigid_body_group, "segment_bid", rb_segment_bid_buf.get(), rb_num, 1);
        _appendToGroupNd(d.rigid_body_group, "mass", rb_mass_buf.get(), rb_num, 1);
        _appendToGroupNd(d.rigid_body_group, "orientation", rb_orientation_buf.get(), rb_num, 4);
        _appendToGroupNd(d.rigid_body_group, "com", rb_com_buf.get(), rb_num, 3);
        _appendToGroupNd(
            d.rigid_body_group, "linear_velocity", rb_linear_velocity_buf.get(), rb_num, 3
        );
        _appendToGroupNd(
            d.rigid_body_group, "angular_velocity", rb_angular_velocity_buf.get(), rb_num, 3
        );
        d.rigid_body_offset += rb_num;

        // Process joints
        RBigSize j_num = record.joints.object_num;
        auto j_rigid_body_a_sid_buf = std::make_unique<RIndex[]>(j_num);
        auto j_rigid_body_b_sid_buf = std::make_unique<RIndex[]>(j_num);
        auto j_type_buf = std::make_unique<int[]>(j_num);
        auto j_position_buf = std::make_unique<RFloat[]>(j_num * 3);
        auto j_axis_buf = std::make_unique<RFloat[]>(j_num * 3);
        // Hinge fields
        auto j_hinge_signal_sid_buf = std::make_unique<RIndex[]>(j_num);
        auto j_hinge_min_buf = std::make_unique<RFloat[]>(j_num);
        auto j_hinge_max_buf = std::make_unique<RFloat[]>(j_num);
        auto j_hinge_angle_buf = std::make_unique<RFloat[]>(j_num);
        auto j_hinge_torque_buf = std::make_unique<RFloat[]>(j_num);
        // Prismatic fields
        auto j_prismatic_signal_sid_buf = std::make_unique<RIndex[]>(j_num);
        auto j_prismatic_min_buf = std::make_unique<RFloat[]>(j_num);
        auto j_prismatic_max_buf = std::make_unique<RFloat[]>(j_num);
        auto j_prismatic_distance_buf = std::make_unique<RFloat[]>(j_num);
        auto j_prismatic_force_buf = std::make_unique<RFloat[]>(j_num);

        _pool
            .submitLoop(
                static_cast<RIndex>(0),
                j_num,
                [&record,
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
                    j_rigid_body_a_sid_buf[i] = record.joints.rigid_body_a_sid[i];
                    j_rigid_body_b_sid_buf[i] = record.joints.rigid_body_b_sid[i];
                    j_type_buf[i] = record.joints.type[i];
                    j_position_buf[i * 3] = record.joints.position[i].x;
                    j_position_buf[i * 3 + 1] = record.joints.position[i].y;
                    j_position_buf[i * 3 + 2] = record.joints.position[i].z;
                    j_axis_buf[i * 3] = record.joints.axis[i].x;
                    j_axis_buf[i * 3 + 1] = record.joints.axis[i].y;
                    j_axis_buf[i * 3 + 2] = record.joints.axis[i].z;
                    // Hinge
                    j_hinge_signal_sid_buf[i] =
                        record.joints.hinge_signal_sid[i];
                    j_hinge_min_buf[i] = record.joints.hinge_min[i];
                    j_hinge_max_buf[i] = record.joints.hinge_max[i];
                    j_hinge_angle_buf[i] = record.joints.hinge_angle[i];
                    j_hinge_torque_buf[i] = record.joints.hinge_torque[i];
                    // Prismatic
                    j_prismatic_signal_sid_buf[i] =
                        record.joints.prismatic_signal_sid[i];
                    j_prismatic_min_buf[i] = record.joints.prismatic_min[i];
                    j_prismatic_max_buf[i] = record.joints.prismatic_max[i];
                    j_prismatic_distance_buf[i] = record.joints.prismatic_distance[i];
                    j_prismatic_force_buf[i] = record.joints.prismatic_force[i];
                }
            )
            .wait();

        _appendToGroupNd(d.joint_group, "joint_num", &j_num, 1, 1);
        _appendToGroupNd(d.joint_group, "joint_offset", &d.joint_offset, 1, 1);
        _appendToGroupNd(d.joint_group, "rigid_body_a_sid", j_rigid_body_a_sid_buf.get(), j_num, 1);
        _appendToGroupNd(d.joint_group, "rigid_body_b_sid", j_rigid_body_b_sid_buf.get(), j_num, 1);
        _appendToGroupNd(d.joint_group, "type", j_type_buf.get(), j_num, 1);
        _appendToGroupNd(d.joint_group, "position", j_position_buf.get(), j_num, 3);
        _appendToGroupNd(d.joint_group, "axis", j_axis_buf.get(), j_num, 3);
        // Hinge
        _appendToGroupNd(
            d.joint_group,
            "hinge_signal_sid",
            j_hinge_signal_sid_buf.get(),
            j_num,
            1
        );
        _appendToGroupNd(d.joint_group, "hinge_min", j_hinge_min_buf.get(), j_num, 1);
        _appendToGroupNd(d.joint_group, "hinge_max", j_hinge_max_buf.get(), j_num, 1);
        _appendToGroupNd(d.joint_group, "hinge_angle", j_hinge_angle_buf.get(), j_num, 1);
        _appendToGroupNd(d.joint_group, "hinge_torque", j_hinge_torque_buf.get(), j_num, 1);
        // Prismatic
        _appendToGroupNd(
            d.joint_group,
            "prismatic_signal_sid",
            j_prismatic_signal_sid_buf.get(),
            j_num,
            1
        );
        _appendToGroupNd(d.joint_group, "prismatic_min", j_prismatic_min_buf.get(), j_num, 1);
        _appendToGroupNd(d.joint_group, "prismatic_max", j_prismatic_max_buf.get(), j_num, 1);
        _appendToGroupNd(
            d.joint_group, "prismatic_distance", j_prismatic_distance_buf.get(), j_num, 1
        );
        _appendToGroupNd(d.joint_group, "prismatic_force", j_prismatic_force_buf.get(), j_num, 1);
        d.joint_offset += j_num;
    }
}

void RS_SimulationHDF5Recorder::syncCameraRecords(
    const std::string &name,
    const std::vector<RS_SimulationCameraRecord> &records
) {
    if (!records.empty()) {
        CameraRecordData &d =
            _createOrGetCameraRecordData(name, records[0].width, records[0].height);

        for (auto &record : records) {
            _appendToGroupNd(d.time_group, "step", record.step, 1, 1);
            _appendToGroupNd(d.time_group, "time_point", record.time, 1, 1);
            _appendCameraFrame(d.rgb_group, "pixel", record.rgb.pixel, d.width, d.height);
            _appendCameraState(d.camera_state_group, record.camera_state);
        }
    }
}

RS_SimulationHDF5Recorder::StructureRecordData &
RS_SimulationHDF5Recorder::_createOrGetStructureRecordData(const std::string &structure_name) {
    if (_structure_record_data.find(structure_name) != _structure_record_data.end()) {
        return _structure_record_data[structure_name];
    }
    StructureRecordData d;
    d.root_group =
        make_shared<HF::Group>(_structure_record_root_group->createGroup(structure_name));

    auto &root_group = d.root_group;
    d.time_group = make_shared<HF::Group>(root_group->createGroup("frames/time"));
    d.voxel_group = make_shared<HF::Group>(root_group->createGroup("frames/voxel"));
    d.link_group = make_shared<HF::Group>(root_group->createGroup("frames/link"));
    d.rigid_body_group = make_shared<HF::Group>(root_group->createGroup("frames/rigid_body"));
    d.joint_group = make_shared<HF::Group>(root_group->createGroup("frames/joint"));

    HF::DataSpace space_1d({0}, {HF::DataSpace::UNLIMITED});
    HF::DataSpace space_2d({0, 2}, {HF::DataSpace::UNLIMITED, 2});
    HF::DataSpace space_3d({0, 3}, {HF::DataSpace::UNLIMITED, 3});
    HF::DataSpace space_4d({0, 4}, {HF::DataSpace::UNLIMITED, 4});
    HF::DataSetCreateProps create_props_1d, create_props_2d, create_props_3d, create_props_4d;
    create_props_1d.add(HF::Chunking(chunk_size));
    create_props_2d.add(HF::Chunking(chunk_size, 2));
    create_props_3d.add(HF::Chunking(chunk_size, 3));
    create_props_4d.add(HF::Chunking(chunk_size, 4));

    d.time_group->createDataSet<RBigIndex>("step", space_1d, create_props_1d);
    d.time_group->createDataSet<RBigFloat>("time_point", space_1d, create_props_1d);

    d.voxel_group->createDataSet<RBigSize>("voxel_num", space_1d, create_props_1d);
    d.voxel_group->createDataSet<RBigSize>("voxel_offset", space_1d, create_props_1d);
    d.voxel_group->createDataSet<RIndex>("body_sid", space_1d, create_props_1d);
    d.voxel_group->createDataSet<RIndex>("voxel_sid", space_1d, create_props_1d);
    d.voxel_group->createDataSet<RIndex>("segment_bid", space_1d, create_props_1d);
    d.voxel_group->createDataSet<RSignedIndex>("index_xyz", space_3d, create_props_3d);
    d.voxel_group->createDataSet<uint8_t>("connectivity", space_1d, create_props_1d);
    d.voxel_group->createDataSet<bool>("is_surface", space_1d, create_props_1d);
    d.voxel_group->createDataSet<bool>("is_rigid", space_1d, create_props_1d);
    d.voxel_group->createDataSet<RIndex>("material", space_1d, create_props_1d);
    d.voxel_group->createDataSet<RFloat>("expansion", space_1d, create_props_1d);
    d.voxel_group->createDataSet<RFloat>("orientation", space_4d, create_props_4d);
    d.voxel_group->createDataSet<RFloat>("position", space_3d, create_props_3d);
    d.voxel_group->createDataSet<RFloat>("ppp_offset", space_3d, create_props_3d);
    d.voxel_group->createDataSet<RFloat>("nnn_offset", space_3d, create_props_3d);
    d.voxel_group->createDataSet<RFloat>("linear_velocity", space_3d, create_props_3d);
    d.voxel_group->createDataSet<RFloat>("angular_velocity", space_3d, create_props_3d);
    d.voxel_group->createDataSet<RFloat>("poissons_strain", space_3d, create_props_3d);

    d.link_group->createDataSet<RBigSize>("link_num", space_1d, create_props_1d);
    d.link_group->createDataSet<RBigSize>("link_offset", space_1d, create_props_1d);
    d.link_group->createDataSet<RIndex>("voxel_neg_sid", space_1d, create_props_1d);
    d.link_group->createDataSet<RIndex>("voxel_pos_sid", space_1d, create_props_1d);
    d.link_group->createDataSet<bool>("is_tendon", space_1d, create_props_1d);
    d.link_group->createDataSet<bool>("is_failed", space_1d, create_props_1d);
    d.link_group->createDataSet<RFloat>("strain", space_1d, create_props_1d);
    d.link_group->createDataSet<RFloat>("stress", space_1d, create_props_1d);

    d.rigid_body_group->createDataSet<RBigSize>("rigid_body_num", space_1d, create_props_1d);
    d.rigid_body_group->createDataSet<RBigSize>("rigid_body_offset", space_1d, create_props_1d);
    d.rigid_body_group->createDataSet<RIndex>("rigid_body_sid", space_1d, create_props_1d);
    d.rigid_body_group->createDataSet<RIndex>("body_sid", space_1d, create_props_1d);
    d.rigid_body_group->createDataSet<RIndex>("segment_bid", space_1d, create_props_1d);
    d.rigid_body_group->createDataSet<RFloat>("mass", space_1d, create_props_1d);
    d.rigid_body_group->createDataSet<RFloat>("orientation", space_4d, create_props_4d);
    d.rigid_body_group->createDataSet<RFloat>("com", space_3d, create_props_3d);
    d.rigid_body_group->createDataSet<RFloat>("linear_velocity", space_3d, create_props_3d);
    d.rigid_body_group->createDataSet<RFloat>("angular_velocity", space_3d, create_props_3d);

    d.joint_group->createDataSet<RBigSize>("joint_num", space_1d, create_props_1d);
    d.joint_group->createDataSet<RBigSize>("joint_offset", space_1d, create_props_1d);
    d.joint_group->createDataSet<RIndex>("rigid_body_a_sid", space_1d, create_props_1d);
    d.joint_group->createDataSet<RIndex>("rigid_body_b_sid", space_1d, create_props_1d);
    d.joint_group->createDataSet<RIndex>(
        "hinge_signal_sid", space_1d, create_props_1d
    );
    d.joint_group->createDataSet<int>("type", space_1d, create_props_1d);
    d.joint_group->createDataSet<RFloat>("position", space_3d, create_props_3d);
    d.joint_group->createDataSet<RFloat>("axis", space_3d, create_props_3d);
    d.joint_group->createDataSet<RFloat>("hinge_min", space_1d, create_props_1d);
    d.joint_group->createDataSet<RFloat>("hinge_max", space_1d, create_props_1d);
    d.joint_group->createDataSet<RFloat>("hinge_angle", space_1d, create_props_1d);
    d.joint_group->createDataSet<RFloat>("hinge_torque", space_1d, create_props_1d);
    // Prismatic fields
    d.joint_group->createDataSet<RIndex>(
        "prismatic_signal_sid", space_1d, create_props_1d
    );
    d.joint_group->createDataSet<RFloat>("prismatic_min", space_1d, create_props_1d);
    d.joint_group->createDataSet<RFloat>("prismatic_max", space_1d, create_props_1d);
    d.joint_group->createDataSet<RFloat>("prismatic_distance", space_1d, create_props_1d);
    d.joint_group->createDataSet<RFloat>("prismatic_force", space_1d, create_props_1d);
    _structure_record_data[structure_name] = d;
    return _structure_record_data[structure_name];
}

RS_SimulationHDF5Recorder::CameraRecordData &
RS_SimulationHDF5Recorder::_createOrGetCameraRecordData(
    const std::string &camera_name,
    RSize width,
    RSize height
) {
    if (_camera_record_data.find(camera_name) != _camera_record_data.end()) {
        return _camera_record_data[camera_name];
    }
    CameraRecordData d;
    d.width = width;
    d.height = height;
    d.root_group = make_shared<HF::Group>(_camera_record_root_group->createGroup(camera_name));
    auto &root_group = d.root_group;

    d.time_group = make_shared<HF::Group>(root_group->createGroup("frames/time"));
    d.rgb_group = make_shared<HF::Group>(root_group->createGroup("frames/rgb"));
    // Create new group for camera state
    d.camera_state_group = make_shared<HF::Group>(root_group->createGroup("frames/camera_state"));

    HF::DataSpace space_1d({0}, {HF::DataSpace::UNLIMITED});
    HF::DataSpace space_3d({0, 3}, {HF::DataSpace::UNLIMITED, 3});
    HF::DataSpace space_4d({0, 4}, {HF::DataSpace::UNLIMITED, 4});
    HF::DataSpace space_cam({0, height, width, 3}, {HF::DataSpace::UNLIMITED, height, width, 3});

    HF::DataSetCreateProps create_props_1d, create_props_3d, create_props_4d, create_props_cam;
    create_props_1d.add(HF::Chunking(chunk_size));
    create_props_3d.add(HF::Chunking(chunk_size, 3));
    create_props_4d.add(HF::Chunking(chunk_size, 4));
    create_props_cam.add(HF::Chunking(1, height, width, 3));

    d.time_group->createDataSet<RBigIndex>("step", space_1d, create_props_1d);
    d.time_group->createDataSet<RBigFloat>("time_point", space_1d, create_props_1d);

    d.rgb_group->createDataSet<uint8_t>("pixel", space_cam, create_props_cam);

    // Create datasets for camera state
    d.camera_state_group->createDataSet<RFloat>("position", space_3d, create_props_3d);
    d.camera_state_group->createDataSet<RFloat>("orientation", space_4d, create_props_4d);
    d.camera_state_group->createDataSet<RFloat>("focus_distance", space_1d, create_props_1d);
    d.camera_state_group->createDataSet<RFloat>("field_of_view", space_1d, create_props_1d);

    _camera_record_data[camera_name] = d;
    return _camera_record_data[camera_name];
}

void RS_SimulationHDF5Recorder::finish() {
    _hdf5_file->flush();
    H5Fclose(_hdf5_file->getId());
}

}  // namespace rise