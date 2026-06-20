// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_HDF5_RECORDER_H
#define RS_SIMULATION_HDF5_RECORDER_H

#include <highfive/highfive.hpp>
#include <vector>

#include "rise_origin/utils/concurrency/rs_thread_pool.h"
#include "rs_simulation_recorder.h"

namespace rise {

class RS_SimulationHDF5Recorder : public RS_SimulationRecorder {
   public:
    RS_SimulationHDF5Recorder(const std::string& name, const std::string& file_path);

    std::string getName() const override;

    void start() override;

    void syncStructureVoxelRecords(
        const std::string& name,
        const std::vector<RS_WorldStructureVoxelRecord>& records
    ) override;

    void syncCameraPerspectiveRecords(
        const std::string& name,
        const std::vector<RS_WorldCameraPerspectiveRecord>& records
    ) override;

    void finish() override;

   private:
    /// Per-entity HDF5 state for one structure record (voxel structure).
    struct StructureRecordData {
        RBigSize voxel_offset = 0;
        RBigSize link_offset = 0;
        RBigSize rigid_body_offset = 0;
        RBigSize joint_offset = 0;

        std::shared_ptr<HighFive::Group> root_group;
        std::shared_ptr<HighFive::Group> time_group;
        std::shared_ptr<HighFive::Group> voxel_group;
        std::shared_ptr<HighFive::Group> link_group;
        std::shared_ptr<HighFive::Group> rigid_body_group;
        std::shared_ptr<HighFive::Group> joint_group;
    };

    struct CameraRecordData {
        RSize width = 0;
        RSize height = 0;

        std::shared_ptr<HighFive::Group> root_group;
        std::shared_ptr<HighFive::Group> time_group;
        std::shared_ptr<HighFive::Group> rgb_group;
        std::shared_ptr<HighFive::Group> camera_state_group;
    };

    const int chunk_size = 1024;
    std::string _name;
    RS_ThreadPool _pool{1};
    HighFive::DataSetAccessProps _dataset_access_props;
    std::unique_ptr<HighFive::File> _hdf5_file;

    /// Note: Root group for entity-embedded records: /entity/<name>/...
    /// Future free-floating record entities would live under /record/<name>.
    std::unique_ptr<HighFive::Group> _entity_root_group;

    /// Entity records.
    std::unordered_map<std::string, StructureRecordData> _structure_record_data;
    std::unordered_map<std::string, CameraRecordData> _camera_record_data;

    StructureRecordData& _createOrGetStructureRecordData(
        const std::string& structure_name,
        bool has_voxel,
        bool has_link,
        bool has_rigid_body,
        bool has_joint
    );

    CameraRecordData& _createOrGetCameraRecordData(
        const std::string& camera_name,
        RSize width,
        RSize height,
        bool has_rgb,
        bool has_camera_state
    );

    void
    _writeStructureVoxelFrame(StructureRecordData& d, const RS_WorldStructureRecordBase::Voxel& v);

    void
    _writeStructureLinkFrame(StructureRecordData& d, const RS_WorldStructureRecordBase::Link& l);

    void _writeStructureRigidBodyFrame(
        StructureRecordData& d,
        const RS_WorldStructureRecordBase::RigidBody& rb
    );

    void
    _writeStructureJointFrame(StructureRecordData& d, const RS_WorldStructureRecordBase::Joint& j);

    void _writeCameraRgbFrame(CameraRecordData& d, const RS_WorldCameraRecordBase::RGB& rgb);

    void _writeCameraStateFrame(
        CameraRecordData& d,
        const RS_WorldCameraRecordBase::State& camera_state
    );

    template <typename T>
    void _appendToGroupNd(
        std::shared_ptr<HighFive::Group>& group,
        const std::string& dataset_name,
        T* data,
        size_t num,
        size_t dim
    );
};

}  // namespace rise

#endif  // RS_SIMULATION_HDF5_RECORDER_H
