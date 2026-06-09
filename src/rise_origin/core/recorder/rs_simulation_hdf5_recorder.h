// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_HDF5_RECORDER_H
#define RS_SIMULATION_HDF5_RECORDER_H

#include <highfive/highfive.hpp>
#include <vector>

#include "rise_origin/utils/rs_thread_pool.h"
#include "rs_simulation_recorder.h"

namespace rise {

class RS_SimulationHDF5Recorder : public RS_SimulationRecorder {
   public:
    explicit RS_SimulationHDF5Recorder(const std::string& file_path);
    void start() override;

    void syncStructureRecords(
        const std::string& name,
        const std::vector<RS_SimulationStructureRecord>& records
    ) override;

    void syncCameraRecords(
        const std::string& name,
        const std::vector<RS_SimulationCameraRecord>& records
    ) override;

    void finish() override;

   private:
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
        std::shared_ptr<HighFive::Group> camera_state_group;  // Added camera state group
    };

    const int chunk_size = 1024;
    RS_ThreadPool _pool{1};
    HighFive::DataSetAccessProps _dataset_access_props;
    std::unique_ptr<HighFive::File> _hdf5_file;

    std::unique_ptr<HighFive::Group> _structure_record_root_group;
    std::unordered_map<std::string, StructureRecordData> _structure_record_data;

    std::unique_ptr<HighFive::Group> _camera_record_root_group;
    std::unordered_map<std::string, CameraRecordData> _camera_record_data;

    StructureRecordData& _createOrGetStructureRecordData(const std::string& structure_name);

    CameraRecordData&
    _createOrGetCameraRecordData(const std::string& camera_name, RSize width, RSize height);

    template <typename T, typename F>
    std::vector<RIndex> _filter(const T& records, RSize record_num, F condition);

    template <typename T>
    void _appendToGroupNd(
        std::shared_ptr<HighFive::Group>& group,
        const std::string& dataset_name,
        T* data,
        size_t num,
        size_t dim
    );

    void _appendCameraFrame(
        std::shared_ptr<HighFive::Group>& group,
        const std::string& dataset_name,
        RColor3b* pixels,
        RSize width,
        RSize height
    );

    // Added method to append camera state data
    void _appendCameraState(
        std::shared_ptr<HighFive::Group>& group,
        const RS_SimulationCameraStateRecord& camera_state
    );
};

}  // namespace rise

#endif  // RS_SIMULATION_HDF5_RECORDER_H