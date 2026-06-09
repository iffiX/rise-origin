// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef INIT_CORE_H
#define INIT_CORE_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

#include "rise_origin/core/rs_simulation_controller.h"
#include "rise_origin/core/rs_simulation_def.h"
#include "rise_origin/core/rs_simulation_handle.h"
#include "rise_origin/core/rs_simulation_record.h"
#include "rise_origin/core/rs_simulation_signal.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_device.h"

namespace py = pybind11;
using namespace rise;

class RSPY_DeviceLocation {
   public:
    RS_DeviceLocation loc;

    // Default constructor -> "cpu"
    RSPY_DeviceLocation() {
        loc.type = RSE_DEVICE_TYPE_CPU;
        loc.id = 0;
    }

    // Construct from string: "cpu" or "cuda:<id>"
    explicit RSPY_DeviceLocation(const std::string &device_str) {
        if (device_str == "cpu") {
            loc.type = RSE_DEVICE_TYPE_CPU;
            loc.id = 0;
        } else if (startsWith(device_str, "cuda:")) {
            loc.type = RSE_DEVICE_TYPE_CUDA;
            if (!isNumber(device_str.substr(5))) {
                throw RS_InvalidArgumentException("Invalid device str: {}", device_str);
            }
            loc.id = std::stoi(device_str.substr(5));
        } else {
            throw RS_InvalidArgumentException("Invalid device str: {}", device_str);
        }
    }

    explicit RSPY_DeviceLocation(const RS_DeviceLocation &d) : loc(d) {}

    RSPY_DeviceLocation(RSE_DeviceType t, RIndex i) {
        loc.type = t;
        loc.id = i;
    }

    [[nodiscard]] RSE_DeviceType type() const { return loc.type; }
    [[nodiscard]] RIndex id() const { return loc.id; }
    [[nodiscard]] std::string str() const {
        if (loc.type == RSE_DEVICE_TYPE_CPU) return "cpu";
        if (loc.type == RSE_DEVICE_TYPE_CUDA) return "cuda:" + std::to_string(loc.id);
        return "unknown";
    }
};

class RSPY_SimulationStructureSignal {
   public:
    RS_SimulationStructureSignal signal;
    explicit RSPY_SimulationStructureSignal(const RS_SimulationStructureSignal &sig)
        : signal(sig) {}

    py::object expansion_signals() {
        if (!signal.expansion_signals || signal.expansion_signal_num == 0) return py::none();
        return py::memoryview::from_buffer(
            signal.expansion_signals, {signal.expansion_signal_num}, {sizeof(RFloat)}
        );
    }

    py::object hinge_signals() {
        if (!signal.hinge_signals || signal.hinge_signal_num == 0) return py::none();
        return py::memoryview::from_buffer(
            signal.hinge_signals, {signal.hinge_signal_num}, {sizeof(RFloat)}
        );
    }

    py::object prismatic_signals() {
        if (!signal.prismatic_signals || signal.prismatic_signal_num == 0) return py::none();
        return py::memoryview::from_buffer(
            signal.prismatic_signals, {signal.prismatic_signal_num}, {sizeof(RFloat)}
        );
    }
};

class RSPY_SimulationCameraSignal {
   public:
    RS_SimulationCameraSignal signal;
    explicit RSPY_SimulationCameraSignal(const RS_SimulationCameraSignal &sig) : signal(sig) {}

    py::object position_signals() {
        if (!signal.position_signals || signal.position_num == 0) return py::none();
        return py::memoryview::from_buffer(
            signal.position_signals, {signal.position_num}, {sizeof(RFloat)}
        );
    }

    py::object orientation_signals() {
        if (!signal.orientation_signals || signal.orientation_num == 0) return py::none();
        return py::memoryview::from_buffer(
            signal.orientation_signals, {signal.orientation_num}, {sizeof(RFloat)}
        );
    }

    py::object focus_distance_signals() {
        if (!signal.focus_distance_signals || signal.focus_distance_num == 0) return py::none();
        return py::memoryview::from_buffer(
            signal.focus_distance_signals, {signal.focus_distance_num}, {sizeof(RFloat)}
        );
    }

    py::object field_of_view_signals() {
        if (!signal.field_of_view_signals || signal.field_of_view_num == 0) return py::none();
        return py::memoryview::from_buffer(
            signal.field_of_view_signals, {signal.field_of_view_num}, {sizeof(RFloat)}
        );
    }
};

class RSPY_SimulationForceSignal {
   public:
    RS_SimulationForceSignal signal;
    explicit RSPY_SimulationForceSignal(const RS_SimulationForceSignal &sig) : signal(sig) {}

    py::object force_signals() {
        if (!signal.force_signals) return py::none();
        // Force signal is a single 3D vector (3 RFloat values)
        return py::memoryview::from_buffer(
            signal.force_signals, {static_cast<size_t>(3)}, {sizeof(RFloat)}
        );
    }
};

class RSPY_SimulationTorqueSignal {
   public:
    RS_SimulationTorqueSignal signal;
    explicit RSPY_SimulationTorqueSignal(const RS_SimulationTorqueSignal &sig) : signal(sig) {}

    py::object torque_signals() {
        if (!signal.torque_signals) return py::none();
        // Torque signal is a single 3D vector (3 RFloat values)
        return py::memoryview::from_buffer(
            signal.torque_signals, {static_cast<size_t>(3)}, {sizeof(RFloat)}
        );
    }
};

class RSPY_SimulationStructureRecord {
   public:
    RS_SimulationStructureRecord record;

    explicit RSPY_SimulationStructureRecord(const RS_SimulationStructureRecord &rec)
        : record(rec) {}

    // 1D array helper (for scalar fields)
    template <typename T>
    py::array_t<T> array1D(T *ptr, size_t size) const {
        return py::array_t<T>(
            {size},       // shape
            {sizeof(T)},  // stride
            ptr
        );
    }

    // 2D float array helper (for vectors / quaternions)
    template <typename T>
    py::array_t<T> array2D(T *ptr, size_t N, size_t dim) const {
        // shape: [N, dim]
        // strides in bytes: row stride = dim*sizeof(T), col stride = sizeof(T)
        return py::array_t<T>({N, dim}, {dim * sizeof(T), sizeof(T)}, ptr);
    }

    [[nodiscard]] RBigIndex step() const { return record.getStep(); }
    [[nodiscard]] RBigFloat time() const { return record.getTime(); }

    [[nodiscard]] RBigSize voxel_size() const { return record.voxels.object_num; }
    [[nodiscard]] RBigSize link_size() const { return record.links.object_num; }
    [[nodiscard]] RBigSize rigid_body_size() const { return record.rigid_bodies.object_num; }
    [[nodiscard]] RBigSize joint_size() const { return record.joints.object_num; }

    // ------------------ Voxel ------------------
    [[nodiscard]] py::array_t<RIndex> voxel_body_sid() const {
        return array1D(record.voxels.body_sid, voxel_size());
    }

    [[nodiscard]] py::array_t<RIndex> voxel_voxel_sid() const {
        return array1D(record.voxels.voxel_sid, voxel_size());
    }

    [[nodiscard]] py::array_t<RIndex> voxel_segment_bid() const {
        return array1D(record.voxels.segment_bid, voxel_size());
    }

    [[nodiscard]] py::array_t<RSignedIndex> voxel_index_x() const {
        return array1D(record.voxels.index_x, voxel_size());
    }

    [[nodiscard]] py::array_t<RSignedIndex> voxel_index_y() const {
        return array1D(record.voxels.index_y, voxel_size());
    }

    [[nodiscard]] py::array_t<RSignedIndex> voxel_index_z() const {
        return array1D(record.voxels.index_z, voxel_size());
    }

    [[nodiscard]] py::array_t<uint8_t> voxel_connectivity() const {
        return array1D(record.voxels.connectivity, voxel_size());
    }

    [[nodiscard]] py::array_t<bool> voxel_is_surface() const {
        return array1D(record.voxels.is_surface, voxel_size());
    }

    [[nodiscard]] py::array_t<bool> voxel_is_rigid() const {
        return array1D(record.voxels.is_rigid, voxel_size());
    }

    [[nodiscard]] py::array_t<RIndex> voxel_material() const {
        return array1D(record.voxels.material, voxel_size());
    }

    [[nodiscard]] py::array_t<RFloat> voxel_expansion() const {
        return array1D(record.voxels.expansion, voxel_size());
    }

    // Quaternion: shape [N, 4]
    [[nodiscard]] py::array_t<RFloat> voxel_orientation() const {
        size_t N = voxel_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.voxels.orientation);
        return array2D(fptr, N, 4);
    }

    // Vector: shape [N, 3]
    [[nodiscard]] py::array_t<RFloat> voxel_position() const {
        size_t N = voxel_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.voxels.position);
        return array2D(fptr, N, 3);
    }

    [[nodiscard]] py::array_t<RFloat> voxel_ppp_offset() const {
        size_t N = voxel_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.voxels.ppp_offset);
        return array2D(fptr, N, 3);
    }

    [[nodiscard]] py::array_t<RFloat> voxel_nnn_offset() const {
        size_t N = voxel_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.voxels.nnn_offset);
        return array2D(fptr, N, 3);
    }

    [[nodiscard]] py::array_t<RFloat> voxel_linear_velocity() const {
        size_t N = voxel_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.voxels.linear_velocity);
        return array2D(fptr, N, 3);
    }

    [[nodiscard]] py::array_t<RFloat> voxel_angular_velocity() const {
        size_t N = voxel_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.voxels.angular_velocity);
        return array2D(fptr, N, 3);
    }

    [[nodiscard]] py::array_t<RFloat> voxel_poissons_strain() const {
        size_t N = voxel_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.voxels.poissons_strain);
        return array2D(fptr, N, 3);
    }

    // ------------------ Link ------------------
    [[nodiscard]] py::array_t<RIndex> link_voxel_neg_sid() const {
        return array1D(record.links.voxel_neg_sid, link_size());
    }

    [[nodiscard]] py::array_t<RIndex> link_voxel_pos_sid() const {
        return array1D(record.links.voxel_pos_sid, link_size());
    }

    [[nodiscard]] py::array_t<bool> link_is_tendon() const {
        return array1D(record.links.is_tendon, link_size());
    }

    [[nodiscard]] py::array_t<bool> link_is_failed() const {
        return array1D(record.links.is_failed, link_size());
    }

    [[nodiscard]] py::array_t<RFloat> link_strain() const {
        return array1D(record.links.strain, link_size());
    }

    [[nodiscard]] py::array_t<RFloat> link_stress() const {
        return array1D(record.links.stress, link_size());
    }

    // ------------------ Rigid Body ------------------
    [[nodiscard]] py::array_t<RIndex> rigid_body_rigid_body_sid() const {
        return array1D(record.rigid_bodies.rigid_body_sid, rigid_body_size());
    }

    [[nodiscard]] py::array_t<RIndex> rigid_body_body_sid() const {
        return array1D(record.rigid_bodies.body_sid, rigid_body_size());
    }

    [[nodiscard]] py::array_t<RIndex> rigid_body_segment_bid() const {
        return array1D(record.rigid_bodies.segment_bid, rigid_body_size());
    }

    [[nodiscard]] py::array_t<RFloat> rigid_body_mass() const {
        return array1D(record.rigid_bodies.mass, rigid_body_size());
    }

    // Quaternion: [N, 4]
    [[nodiscard]] py::array_t<RFloat> rigid_body_orientation() const {
        size_t N = rigid_body_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.rigid_bodies.orientation);
        return array2D(fptr, N, 4);
    }

    // Vector: [N, 3]
    [[nodiscard]] py::array_t<RFloat> rigid_body_com() const {
        size_t N = rigid_body_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.rigid_bodies.com);
        return array2D(fptr, N, 3);
    }

    [[nodiscard]] py::array_t<RFloat> rigid_body_linear_velocity() const {
        size_t N = rigid_body_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.rigid_bodies.linear_velocity);
        return array2D(fptr, N, 3);
    }

    [[nodiscard]] py::array_t<RFloat> rigid_body_angular_velocity() const {
        size_t N = rigid_body_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.rigid_bodies.angular_velocity);
        return array2D(fptr, N, 3);
    }

    // ------------------ Joint ------------------
    [[nodiscard]] py::array_t<RIndex> joint_rigid_body_a_sid() const {
        return array1D(record.joints.rigid_body_a_sid, joint_size());
    }

    [[nodiscard]] py::array_t<RIndex> joint_rigid_body_b_sid() const {
        return array1D(record.joints.rigid_body_b_sid, joint_size());
    }

    [[nodiscard]] py::array_t<RIndex> joint_hinge_signal_sid() const {
        return array1D(record.joints.hinge_signal_sid, joint_size());
    }

    [[nodiscard]] py::array_t<RSE_SimulationJointRecordType> joint_type() const {
        return array1D(record.joints.type, joint_size());
    }

    // Vector: [N, 3]
    [[nodiscard]] py::array_t<RFloat> joint_position() const {
        size_t N = joint_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.joints.position);
        return array2D(fptr, N, 3);
    }

    [[nodiscard]] py::array_t<RFloat> joint_axis() const {
        size_t N = joint_size();
        auto *fptr = reinterpret_cast<RFloat *>(record.joints.axis);
        return array2D(fptr, N, 3);
    }

    // 1D float arrays
    [[nodiscard]] py::array_t<RFloat> joint_hinge_min() const {
        return array1D(record.joints.hinge_min, joint_size());
    }

    [[nodiscard]] py::array_t<RFloat> joint_hinge_max() const {
        return array1D(record.joints.hinge_max, joint_size());
    }

    [[nodiscard]] py::array_t<RFloat> joint_hinge_angle() const {
        return array1D(record.joints.hinge_angle, joint_size());
    }

    [[nodiscard]] py::array_t<RFloat> joint_hinge_torque() const {
        return array1D(record.joints.hinge_torque, joint_size());
    }

    [[nodiscard]] py::array_t<RIndex> joint_prismatic_signal_sid() const {
        return array1D(record.joints.prismatic_signal_sid, joint_size());
    }

    [[nodiscard]] py::array_t<RFloat> joint_prismatic_min() const {
        return array1D(record.joints.prismatic_min, joint_size());
    }

    [[nodiscard]] py::array_t<RFloat> joint_prismatic_max() const {
        return array1D(record.joints.prismatic_max, joint_size());
    }

    [[nodiscard]] py::array_t<RFloat> joint_prismatic_distance() const {
        return array1D(record.joints.prismatic_distance, joint_size());
    }

    [[nodiscard]] py::array_t<RFloat> joint_prismatic_force() const {
        return array1D(record.joints.prismatic_force, joint_size());
    }
};

class RSPY_SimulationCameraRecord {
   public:
    RS_SimulationCameraRecord record;

    explicit RSPY_SimulationCameraRecord(const RS_SimulationCameraRecord &rec) : record(rec) {}

    // Helper method to create 3D NumPy array from pixel data
    static py::array_t<uint8_t> array3D(RColor3b *ptr, size_t width, size_t height) {
        // Reinterpret RColor3b as array of uint8_t with shape [height, width, 3]
        return py::array_t(
            {height, width, static_cast<size_t>(3)},                              // shape
            {width * 3 * sizeof(uint8_t), 3 * sizeof(uint8_t), sizeof(uint8_t)},  // strides
            reinterpret_cast<uint8_t *>(ptr)                                      // data pointer
        );
    }

    [[nodiscard]] RBigIndex step() const { return record.getStep(); }
    [[nodiscard]] RBigFloat time() const { return record.getTime(); }
    [[nodiscard]] RIndex camera_gid() const { return record.camera_gid; }
    [[nodiscard]] RIndex structure_gid() const { return record.structure_gid; }
    [[nodiscard]] RIndex rigid_body_sid() const { return record.rigid_body_sid; }
    [[nodiscard]] RSize width() const { return record.width; }
    [[nodiscard]] RSize height() const { return record.height; }
    [[nodiscard]] bool is_valid() const { return record.isValid(); }

    // 1D array helper (for scalar fields)
    template <typename T>
    py::array_t<T> array1D(T *ptr, size_t size) const {
        return py::array_t<T>(
            {size},       // shape
            {sizeof(T)},  // stride
            ptr
        );
    }

    // Get pixel data as a 3D NumPy array (height x width x 3)
    [[nodiscard]] py::array_t<uint8_t> rgb() const {
        if (record.width == 0 || record.height == 0 || !record.isValid()) {
            return py::array_t<uint8_t>(0);
        }
        return array3D(record.rgb.pixel, record.width, record.height);
    }

    // Camera position: [3]
    [[nodiscard]] py::array_t<RFloat> position() const {
        return array1D(reinterpret_cast<RFloat *>(record.camera_state.position), 3);
    }

    // Camera orientation: [4]
    [[nodiscard]] py::array_t<RFloat> orientation() const {
        return array1D(reinterpret_cast<RFloat *>(record.camera_state.orientation), 4);
    }

    // Camera focus distance: [1]
    [[nodiscard]] py::array_t<RFloat> focus_distance() const {
        return array1D(record.camera_state.focus_distance, 1);
    }

    // Camera field of view: [1]
    [[nodiscard]] py::array_t<RFloat> field_of_view() const {
        return array1D(record.camera_state.field_of_view, 1);
    }
};

class RSPY_SimulationStructureController {
   public:
    RS_SimulationStructureController controller;

    explicit RSPY_SimulationStructureController(const RS_SimulationStructureController &ctrl)
        : controller(ctrl) {}

    [[nodiscard]] std::string name() const { return controller.name; }

    [[nodiscard]] RSPY_SimulationStructureRecord structure_record() const {
        return RSPY_SimulationStructureRecord(controller.structure_record);
    }

    [[nodiscard]] RSPY_SimulationStructureSignal signal() const {
        return RSPY_SimulationStructureSignal(controller.signal);
    }

    // Return camera records if needed
    [[nodiscard]] py::list camera_records() const {
        py::list result;
        for (const auto &rec : controller.camera_records) {
            result.append(py::cast(RSPY_SimulationCameraRecord(rec)));
        }
        return result;
    }
};

class RSPY_SimulationCameraController {
   public:
    RS_SimulationCameraController controller;

    explicit RSPY_SimulationCameraController(const RS_SimulationCameraController &ctrl)
        : controller(ctrl) {}

    [[nodiscard]] std::string name() const { return controller.name; }

    // Return wrapped camera record instead of raw record
    [[nodiscard]] RSPY_SimulationCameraRecord camera_record() const {
        return RSPY_SimulationCameraRecord(controller.camera_record);
    }

    [[nodiscard]] RSPY_SimulationCameraSignal signal() const {
        return RSPY_SimulationCameraSignal(controller.signal);
    }
};

class RSPY_SimulationForceController {
   public:
    RS_SimulationForceController controller;

    explicit RSPY_SimulationForceController(const RS_SimulationForceController &ctrl)
        : controller(ctrl) {}

    [[nodiscard]] std::string name() const { return controller.name; }

    [[nodiscard]] RSPY_SimulationForceSignal signal() const {
        return RSPY_SimulationForceSignal(controller.signal);
    }
};

class RSPY_SimulationTorqueController {
   public:
    RS_SimulationTorqueController controller;

    explicit RSPY_SimulationTorqueController(const RS_SimulationTorqueController &ctrl)
        : controller(ctrl) {}

    [[nodiscard]] std::string name() const { return controller.name; }

    [[nodiscard]] RSPY_SimulationTorqueSignal signal() const {
        return RSPY_SimulationTorqueSignal(controller.signal);
    }
};

class RSPY_SimulationHandle {
   public:
    std::shared_ptr<RS_SimulationHandle> handle;
    explicit RSPY_SimulationHandle(std::shared_ptr<RS_SimulationHandle> &handle)
        : handle(handle) {}

    [[nodiscard]] bool is_pending() const { return handle->isPending(); }
    [[nodiscard]] bool is_running() const { return handle->isRunning(); }
    [[nodiscard]] bool is_paused() const { return handle->isPaused(); }
    [[nodiscard]] bool is_end() const { return handle->isEnd(); }
    void pause() const { handle->pause(); }
    void resume() const { handle->resume(); }
    void terminate() const { handle->terminate(); }

    [[nodiscard]] RSE_SimulationStatus get_status() const { return handle->getStatus(); }

    bool wait_for_running() const {
        // Make sure to release the GIL before current thread waits on mutex
        py::gil_scoped_release release;
        return handle->waitForRunning();
    }

    void wait_for_end() const {
        // Make sure to release the GIL before current thread waits on mutex
        py::gil_scoped_release release;
        handle->waitForEnd();
    }

    void begin_modification_transaction() const { handle->beginModificationTransaction(); }
    void end_modification_transaction() const { handle->endModificationTransaction(); }

    void set_simulation_config(const RS_SimulationConfig &config) const {
        handle->setSimulationConfig(config);
    }

    void set_world(const RS_WorldConfig &config) const { handle->setWorld(config); }
    void set_floor(const RS_FloorConfig &config) const { handle->setFloor(config); }

    void add_material(const RS_MaterialConfig &config) const { handle->addMaterial(config); }
    void add_record(const RS_RecordConfig &config) const { handle->addRecord(config); }
    void add_signal(const RS_SignalConfig &config) const { handle->addSignal(config); }
    void add_structure(const RS_StructureConfig &config) const { handle->addStructure(config); }
    void add_camera(const RS_CameraConfig &config) const { handle->addCamera(config); }

    void update_material(const RS_MaterialConfig &config) const { handle->updateMaterial(config); }

    void delete_floor() const { handle->deleteFloor(); }
    void delete_material(RIndex id) const { handle->deleteMaterial(id); }
    void delete_record(RIndex id) const { handle->deleteRecord(id); }
    void delete_signal(RIndex id) const { handle->deleteSignal(id); }
    void delete_structure(RIndex id) const { handle->deleteStructure(id); }
    void delete_camera(RIndex id) const { handle->deleteCamera(id); }

    [[nodiscard]] RIndex find_material(const std::string &name) const {
        return handle->findMaterial(name);
    }
    [[nodiscard]] RIndex find_record(const std::string &name) const {
        return handle->findRecord(name);
    }
    [[nodiscard]] RIndex find_signal(const std::string &name) const {
        return handle->findSignal(name);
    }
    [[nodiscard]] RIndex find_structure(const std::string &name) const {
        return handle->findStructure(name);
    }
    [[nodiscard]] RIndex find_camera(const std::string &name) const {
        return handle->findCamera(name);
    }

    [[nodiscard]] RS_Config get_config() const { return handle->getConfig(); }
    [[nodiscard]] RS_SimulationConfig get_simulation_config() const {
        return handle->getSimulationConfig();
    }
    [[nodiscard]] RS_FloorConfig get_floor_config() const { return handle->getFloorConfig(); }
    [[nodiscard]] std::vector<RS_MaterialConfig> get_material_configs() const {
        return handle->getMaterialConfigs();
    }
    [[nodiscard]] std::vector<RS_RecordConfig> get_record_configs() const {
        return handle->getRecordConfigs();
    }
    [[nodiscard]] std::vector<RS_SignalConfig> get_signal_configs() const {
        return handle->getSignalConfigs();
    }
    [[nodiscard]] std::vector<RS_StructureConfig> get_structure_configs() const {
        return handle->getStructureConfigs();
    }
    [[nodiscard]] std::vector<RS_CameraConfig> get_camera_configs() const {
        return handle->getCameraConfigs();
    }
};

inline void initRSPYCore(py::module &m) {
    // Device types enum
    py::enum_<RSE_DeviceType>(m, "RSE_DeviceType")
        .value("RSE_DEVICE_TYPE_UNKNOWN", RSE_DEVICE_TYPE_UNKNOWN)
        .value("RSE_DEVICE_TYPE_CPU", RSE_DEVICE_TYPE_CPU)
        .value("RSE_DEVICE_TYPE_CUDA", RSE_DEVICE_TYPE_CUDA)
        .export_values();

    py::enum_<RSE_SimulationStatus>(m, "RSE_SimulationStatus")
        .value("RSE_SIMULATION_END_SUCCESS", RSE_SIMULATION_END_SUCCESS)
        .value("RSE_SIMULATION_END_TERMINATED", RSE_SIMULATION_END_TERMINATED)
        .value(
            "RSE_SIMULATION_END_FAILED_WITH_INVALID_CONFIG",
            RSE_SIMULATION_END_FAILED_WITH_INVALID_CONFIG
        )
        .value(
            "RSE_SIMULATION_END_FAILED_WITH_SIMULATION_EXCEPTION",
            RSE_SIMULATION_END_FAILED_WITH_SIMULATION_EXCEPTION
        )
        .value(
            "RSE_SIMULATION_END_FAILED_WITH_SIMULATION_MANAGER_EXCEPTION",
            RSE_SIMULATION_END_FAILED_WITH_SIMULATION_MANAGER_EXCEPTION
        )
        .value("RSE_SIMULATION_UNINITIALIZED", RSE_SIMULATION_UNINITIALIZED)
        .value("RSE_SIMULATION_INITIALIZING", RSE_SIMULATION_INITIALIZING)
        .value("RSE_SIMULATION_RUNNING", RSE_SIMULATION_RUNNING)
        .value("RSE_SIMULATION_PAUSED", RSE_SIMULATION_PAUSED)
        .value("RSE_SIMULATION_TERMINATING", RSE_SIMULATION_TERMINATING)
        .value("RSE_SIMULATION_MAX_NORMAL_STATE", RSE_SIMULATION_MAX_NORMAL_STATE)
        .export_values();

    // Simulation controller type enum
    py::enum_<RSE_SimulationControllerType>(m, "RSE_SimulationControllerType")
        .value("RSE_SIMULATION_STRUCTURE_CONTROLLER", RSE_SIMULATION_STRUCTURE_CONTROLLER)
        .value("RSE_SIMULATION_CAMERA_CONTROLLER", RSE_SIMULATION_CAMERA_CONTROLLER)
        .value("RSE_SIMULATION_FORCE_CONTROLLER", RSE_SIMULATION_FORCE_CONTROLLER)
        .value("RSE_SIMULATION_TORQUE_CONTROLLER", RSE_SIMULATION_TORQUE_CONTROLLER)
        .export_values();

    // Device location wrapper
    py::class_<RSPY_DeviceLocation>(m, "RS_DeviceLocation")
        // Expose multiple constructors:
        .def(py::init<>())  // Defaults to CPU
        .def(py::init<const std::string &>())
        .def(py::init<RSE_DeviceType, RIndex>())
        .def(py::init<const RS_DeviceLocation &>())

        // Expose device attributes
        .def("type", &RSPY_DeviceLocation::type)
        .def("id", &RSPY_DeviceLocation::id)
        .def("__str__", &RSPY_DeviceLocation::str);

    // Simulation handle wrapper
    py::class_<RSPY_SimulationHandle>(m, "RS_SimulationHandle")
        .def("is_pending", &RSPY_SimulationHandle::is_pending)
        .def("is_running", &RSPY_SimulationHandle::is_running)
        .def("is_paused", &RSPY_SimulationHandle::is_paused)
        .def("is_end", &RSPY_SimulationHandle::is_end)
        .def("pause", &RSPY_SimulationHandle::pause)
        .def("resume", &RSPY_SimulationHandle::resume)
        .def("terminate", &RSPY_SimulationHandle::terminate)
        .def("get_status", &RSPY_SimulationHandle::get_status)
        .def("wait_for_running", &RSPY_SimulationHandle::wait_for_running)
        .def("wait_for_end", &RSPY_SimulationHandle::wait_for_end)
        .def(
            "begin_modification_transaction",
            &RSPY_SimulationHandle::begin_modification_transaction
        )
        .def("end_modification_transaction", &RSPY_SimulationHandle::end_modification_transaction)
        .def("set_simulation_config", &RSPY_SimulationHandle::set_simulation_config)
        .def("set_world", &RSPY_SimulationHandle::set_world)
        .def("set_floor", &RSPY_SimulationHandle::set_floor)
        .def("add_material", &RSPY_SimulationHandle::add_material)
        .def("add_record", &RSPY_SimulationHandle::add_record)
        .def("add_signal", &RSPY_SimulationHandle::add_signal)
        .def("add_structure", &RSPY_SimulationHandle::add_structure)
        .def("add_camera", &RSPY_SimulationHandle::add_camera)
        .def("update_material", &RSPY_SimulationHandle::update_material)
        .def("delete_floor", &RSPY_SimulationHandle::delete_floor)
        .def("delete_material", &RSPY_SimulationHandle::delete_material)
        .def("delete_record", &RSPY_SimulationHandle::delete_record)
        .def("delete_signal", &RSPY_SimulationHandle::delete_signal)
        .def("delete_structure", &RSPY_SimulationHandle::delete_structure)
        .def("delete_camera", &RSPY_SimulationHandle::delete_camera)
        .def("find_material", &RSPY_SimulationHandle::find_material)
        .def("find_record", &RSPY_SimulationHandle::find_record)
        .def("find_signal", &RSPY_SimulationHandle::find_signal)
        .def("find_structure", &RSPY_SimulationHandle::find_structure)
        .def("find_camera", &RSPY_SimulationHandle::find_camera)
        .def("get_config", &RSPY_SimulationHandle::get_config)
        .def("get_simulation_config", &RSPY_SimulationHandle::get_simulation_config)
        .def("get_floor_config", &RSPY_SimulationHandle::get_floor_config)
        .def("get_material_configs", &RSPY_SimulationHandle::get_material_configs)
        .def("get_record_configs", &RSPY_SimulationHandle::get_record_configs)
        .def("get_signal_configs", &RSPY_SimulationHandle::get_signal_configs)
        .def("get_structure_configs", &RSPY_SimulationHandle::get_structure_configs)
        .def("get_camera_configs", &RSPY_SimulationHandle::get_camera_configs);

    // Signal wrappers
    py::class_<RSPY_SimulationStructureSignal>(m, "RS_SimulationStructureSignal")
        .def("expansion_signals", &RSPY_SimulationStructureSignal::expansion_signals)
        .def("hinge_signals", &RSPY_SimulationStructureSignal::hinge_signals)
        .def(
            "prismatic_signals", &RSPY_SimulationStructureSignal::prismatic_signals
        );

    py::class_<RSPY_SimulationCameraSignal>(m, "RS_SimulationCameraSignal")
        .def("position_signals", &RSPY_SimulationCameraSignal::position_signals)
        .def("orientation_signals", &RSPY_SimulationCameraSignal::orientation_signals)
        .def("focus_distance_signals", &RSPY_SimulationCameraSignal::focus_distance_signals)
        .def("field_of_view_signals", &RSPY_SimulationCameraSignal::field_of_view_signals);

    py::class_<RSPY_SimulationForceSignal>(m, "RS_SimulationForceSignal")
        .def("force_signals", &RSPY_SimulationForceSignal::force_signals);

    py::class_<RSPY_SimulationTorqueSignal>(m, "RS_SimulationTorqueSignal")
        .def("torque_signals", &RSPY_SimulationTorqueSignal::torque_signals);

    // Record wrapper
    py::class_<RSPY_SimulationStructureRecord>(m, "RS_SimulationStructureRecord")
        .def("step", &RSPY_SimulationStructureRecord::step)
        .def("time", &RSPY_SimulationStructureRecord::time)
        .def("voxel_size", &RSPY_SimulationStructureRecord::voxel_size)
        .def("link_size", &RSPY_SimulationStructureRecord::link_size)
        .def("rigid_body_size", &RSPY_SimulationStructureRecord::rigid_body_size)
        .def("joint_size", &RSPY_SimulationStructureRecord::joint_size)
        .def("voxel_body_sid", &RSPY_SimulationStructureRecord::voxel_body_sid)
        .def("voxel_voxel_sid", &RSPY_SimulationStructureRecord::voxel_voxel_sid)
        .def("voxel_segment_bid", &RSPY_SimulationStructureRecord::voxel_segment_bid)
        .def("voxel_index_x", &RSPY_SimulationStructureRecord::voxel_index_x)
        .def("voxel_index_y", &RSPY_SimulationStructureRecord::voxel_index_y)
        .def("voxel_index_z", &RSPY_SimulationStructureRecord::voxel_index_z)
        .def("voxel_connectivity", &RSPY_SimulationStructureRecord::voxel_connectivity)
        .def("voxel_is_surface", &RSPY_SimulationStructureRecord::voxel_is_surface)
        .def("voxel_is_rigid", &RSPY_SimulationStructureRecord::voxel_is_rigid)
        .def("voxel_material", &RSPY_SimulationStructureRecord::voxel_material)
        .def("voxel_expansion", &RSPY_SimulationStructureRecord::voxel_expansion)
        .def("voxel_orientation", &RSPY_SimulationStructureRecord::voxel_orientation)
        .def("voxel_position", &RSPY_SimulationStructureRecord::voxel_position)
        .def("voxel_ppp_offset", &RSPY_SimulationStructureRecord::voxel_ppp_offset)
        .def("voxel_nnn_offset", &RSPY_SimulationStructureRecord::voxel_nnn_offset)
        .def("voxel_linear_velocity", &RSPY_SimulationStructureRecord::voxel_linear_velocity)
        .def("voxel_angular_velocity", &RSPY_SimulationStructureRecord::voxel_angular_velocity)
        .def("voxel_poissons_strain", &RSPY_SimulationStructureRecord::voxel_poissons_strain)
        .def("link_voxel_neg_sid", &RSPY_SimulationStructureRecord::link_voxel_neg_sid)
        .def("link_voxel_pos_sid", &RSPY_SimulationStructureRecord::link_voxel_pos_sid)
        .def("link_is_tendon", &RSPY_SimulationStructureRecord::link_is_tendon)
        .def("link_is_failed", &RSPY_SimulationStructureRecord::link_is_failed)
        .def("link_strain", &RSPY_SimulationStructureRecord::link_strain)
        .def("link_stress", &RSPY_SimulationStructureRecord::link_stress)
        .def(
            "rigid_body_rigid_body_sid", &RSPY_SimulationStructureRecord::rigid_body_rigid_body_sid
        )
        .def("rigid_body_body_sid", &RSPY_SimulationStructureRecord::rigid_body_body_sid)
        .def("rigid_body_segment_bid", &RSPY_SimulationStructureRecord::rigid_body_segment_bid)
        .def("rigid_body_mass", &RSPY_SimulationStructureRecord::rigid_body_mass)
        .def("rigid_body_orientation", &RSPY_SimulationStructureRecord::rigid_body_orientation)
        .def("rigid_body_com", &RSPY_SimulationStructureRecord::rigid_body_com)
        .def(
            "rigid_body_linear_velocity",
            &RSPY_SimulationStructureRecord::rigid_body_linear_velocity
        )
        .def(
            "rigid_body_angular_velocity",
            &RSPY_SimulationStructureRecord::rigid_body_angular_velocity
        )
        .def("joint_rigid_body_a_sid", &RSPY_SimulationStructureRecord::joint_rigid_body_a_sid)
        .def("joint_rigid_body_b_sid", &RSPY_SimulationStructureRecord::joint_rigid_body_b_sid)
        .def(
            "joint_hinge_signal_sid",
            &RSPY_SimulationStructureRecord::joint_hinge_signal_sid
        )
        .def("joint_type", &RSPY_SimulationStructureRecord::joint_type)
        .def("joint_position", &RSPY_SimulationStructureRecord::joint_position)
        .def("joint_axis", &RSPY_SimulationStructureRecord::joint_axis)
        .def("joint_hinge_min", &RSPY_SimulationStructureRecord::joint_hinge_min)
        .def("joint_hinge_max", &RSPY_SimulationStructureRecord::joint_hinge_max)
        .def("joint_hinge_angle", &RSPY_SimulationStructureRecord::joint_hinge_angle)
        .def("joint_hinge_torque", &RSPY_SimulationStructureRecord::joint_hinge_torque)
        .def(
            "joint_prismatic_signal_sid",
            &RSPY_SimulationStructureRecord::joint_prismatic_signal_sid
        )
        .def("joint_prismatic_min", &RSPY_SimulationStructureRecord::joint_prismatic_min)
        .def("joint_prismatic_max", &RSPY_SimulationStructureRecord::joint_prismatic_max)
        .def("joint_prismatic_distance", &RSPY_SimulationStructureRecord::joint_prismatic_distance)
        .def("joint_prismatic_force", &RSPY_SimulationStructureRecord::joint_prismatic_force);

    py::class_<RSPY_SimulationCameraRecord>(m, "RS_SimulationCameraRecord")
        .def("step", &RSPY_SimulationCameraRecord::step)
        .def("time", &RSPY_SimulationCameraRecord::time)
        .def("camera_gid", &RSPY_SimulationCameraRecord::camera_gid)
        .def("structure_gid", &RSPY_SimulationCameraRecord::structure_gid)
        .def("rigid_body_sid", &RSPY_SimulationCameraRecord::rigid_body_sid)
        .def("width", &RSPY_SimulationCameraRecord::width)
        .def("height", &RSPY_SimulationCameraRecord::height)
        .def("is_valid", &RSPY_SimulationCameraRecord::is_valid)
        .def("rgb", &RSPY_SimulationCameraRecord::rgb)
        .def("position", &RSPY_SimulationCameraRecord::position)
        .def("orientation", &RSPY_SimulationCameraRecord::orientation)
        .def("field_of_view", &RSPY_SimulationCameraRecord::field_of_view)
        .def("focus_distance", &RSPY_SimulationCameraRecord::focus_distance);

    // Controller wrappers
    py::class_<RSPY_SimulationStructureController>(m, "RS_SimulationStructureController")
        .def("name", &RSPY_SimulationStructureController::name)
        .def("structure_record", &RSPY_SimulationStructureController::structure_record)
        .def("signal", &RSPY_SimulationStructureController::signal)
        .def("camera_records", &RSPY_SimulationStructureController::camera_records);

    py::class_<RSPY_SimulationCameraController>(m, "RS_SimulationCameraController")
        .def("name", &RSPY_SimulationCameraController::name)
        .def("camera_record", &RSPY_SimulationCameraController::camera_record)
        .def("signal", &RSPY_SimulationCameraController::signal);

    py::class_<RSPY_SimulationForceController>(m, "RS_SimulationForceController")
        .def("name", &RSPY_SimulationForceController::name)
        .def("signal", &RSPY_SimulationForceController::signal);

    py::class_<RSPY_SimulationTorqueController>(m, "RS_SimulationTorqueController")
        .def("name", &RSPY_SimulationTorqueController::name)
        .def("signal", &RSPY_SimulationTorqueController::signal);
}

#endif  // INIT_CORE_H