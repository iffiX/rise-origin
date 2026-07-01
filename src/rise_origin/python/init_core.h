// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef INIT_CORE_H
#define INIT_CORE_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rise_origin/core/rs_world_callback.h"
#include "rise_origin/core/rs_world_controller.h"
#include "rise_origin/core/rs_world_def.h"
#include "rise_origin/core/rs_world_handle.h"
#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/core/rs_world_session.h"
#include "rise_origin/core/rs_world_session_component.h"
#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/utils/memory/rs_device.h"
#include "rise_origin/utils/rs_def.h"

namespace py = pybind11;
using namespace rise;

// Read-only numpy array helpers: return a copy of the underlying buffer.
// Used for record snapshots that should not be mutated by Python callers.
template <typename T>
inline py::array_t<T> rspyRoArray1D(T *ptr, size_t size) {
    return py::array_t<T>({size}, {sizeof(T)}, ptr);
}

template <typename T>
inline py::array_t<T> rspyRoArray2D(T *ptr, size_t n, size_t dim) {
    return py::array_t<T>({n, dim}, {dim * sizeof(T), sizeof(T)}, ptr);
}

// Read-write numpy array helpers: return a non-owning writable view of the
// underlying buffer through a no-op capsule base. The buffer's lifetime is
// owned by the manager that produced it.
template <typename T>
inline py::array_t<T> rspyRwArray1D(T *ptr, size_t size) {
    py::capsule view_owner(ptr, [](void *) {});
    return py::array_t<T>({size}, {sizeof(T)}, ptr, view_owner);
}

template <typename T>
inline py::array_t<T> rspyRwArray2D(T *ptr, size_t n, size_t dim) {
    py::capsule view_owner(ptr, [](void *) {});
    return py::array_t<T>({n, dim}, {dim * sizeof(T), sizeof(T)}, ptr, view_owner);
}

class RSPY_DeviceLocation {
   public:
    RS_DeviceLocation loc;

    RSPY_DeviceLocation() {
        loc.type = RSE_DEVICE_TYPE_CPU;
        loc.id = 0;
    }

    /// Construct from "cpu" or "cuda:<id>".
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

/*****************************************************************
 * Record wrappers.
 *****************************************************************/

class RSPY_WorldStructureVoxelRecord {
   public:
    RS_WorldStructureVoxelRecord record;
    explicit RSPY_WorldStructureVoxelRecord(const RS_WorldStructureVoxelRecord &rec)
        : record(rec) {}

    class Voxel {
       public:
        RS_WorldStructureVoxelRecord record;
        explicit Voxel(const RS_WorldStructureVoxelRecord &rec) : record(rec) {}

        [[nodiscard]] RBigSize size() const { return record.voxels.object_num; }

        [[nodiscard]] py::array_t<RIndex> body_sid() const {
            return rspyRoArray1D(record.voxels.body_sid, size());
        }
        [[nodiscard]] py::array_t<RIndex> voxel_sid() const {
            return rspyRoArray1D(record.voxels.voxel_sid, size());
        }
        [[nodiscard]] py::array_t<RIndex> segment_bid() const {
            return rspyRoArray1D(record.voxels.segment_bid, size());
        }
        [[nodiscard]] py::array_t<RSignedIndex> index_x() const {
            return rspyRoArray1D(record.voxels.index_x, size());
        }
        [[nodiscard]] py::array_t<RSignedIndex> index_y() const {
            return rspyRoArray1D(record.voxels.index_y, size());
        }
        [[nodiscard]] py::array_t<RSignedIndex> index_z() const {
            return rspyRoArray1D(record.voxels.index_z, size());
        }
        [[nodiscard]] py::array_t<uint8_t> connectivity() const {
            return rspyRoArray1D(record.voxels.connectivity, size());
        }
        [[nodiscard]] py::array_t<bool> is_surface() const {
            return rspyRoArray1D(record.voxels.is_surface, size());
        }
        [[nodiscard]] py::array_t<bool> is_rigid() const {
            return rspyRoArray1D(record.voxels.is_rigid, size());
        }
        [[nodiscard]] py::array_t<RIndex> material() const {
            return rspyRoArray1D(record.voxels.material, size());
        }
        [[nodiscard]] py::array_t<RFloat> expansion() const {
            return rspyRoArray1D(record.voxels.expansion, size());
        }
        [[nodiscard]] py::array_t<RFloat> orientation() const {
            return rspyRoArray2D(reinterpret_cast<RFloat *>(record.voxels.orientation), size(), 4);
        }
        [[nodiscard]] py::array_t<RFloat> position() const {
            return rspyRoArray2D(reinterpret_cast<RFloat *>(record.voxels.position), size(), 3);
        }
        [[nodiscard]] py::array_t<RFloat> ppp_offset() const {
            return rspyRoArray2D(reinterpret_cast<RFloat *>(record.voxels.ppp_offset), size(), 3);
        }
        [[nodiscard]] py::array_t<RFloat> nnn_offset() const {
            return rspyRoArray2D(reinterpret_cast<RFloat *>(record.voxels.nnn_offset), size(), 3);
        }
        [[nodiscard]] py::array_t<RFloat> linear_velocity() const {
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.voxels.linear_velocity), size(), 3
            );
        }
        [[nodiscard]] py::array_t<RFloat> angular_velocity() const {
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.voxels.angular_velocity), size(), 3
            );
        }
        [[nodiscard]] py::array_t<RFloat> poissons_strain() const {
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.voxels.poissons_strain), size(), 3
            );
        }
    };

    class Link {
       public:
        RS_WorldStructureVoxelRecord record;
        explicit Link(const RS_WorldStructureVoxelRecord &rec) : record(rec) {}

        [[nodiscard]] RBigSize size() const { return record.links.object_num; }

        [[nodiscard]] py::array_t<RIndex> voxel_neg_sid() const {
            return rspyRoArray1D(record.links.voxel_neg_sid, size());
        }
        [[nodiscard]] py::array_t<RIndex> voxel_pos_sid() const {
            return rspyRoArray1D(record.links.voxel_pos_sid, size());
        }
        [[nodiscard]] py::array_t<bool> is_failed() const {
            return rspyRoArray1D(record.links.is_failed, size());
        }
        [[nodiscard]] py::array_t<RFloat> strain() const {
            return rspyRoArray1D(record.links.strain, size());
        }
        [[nodiscard]] py::array_t<RFloat> stress() const {
            return rspyRoArray1D(record.links.stress, size());
        }
    };

    class RigidBody {
       public:
        RS_WorldStructureVoxelRecord record;
        explicit RigidBody(const RS_WorldStructureVoxelRecord &rec) : record(rec) {}

        [[nodiscard]] RBigSize size() const { return record.rigid_bodies.object_num; }

        [[nodiscard]] py::array_t<RIndex> rigid_body_sid() const {
            return rspyRoArray1D(record.rigid_bodies.rigid_body_sid, size());
        }
        [[nodiscard]] py::array_t<RIndex> body_sid() const {
            return rspyRoArray1D(record.rigid_bodies.body_sid, size());
        }
        [[nodiscard]] py::array_t<RIndex> segment_bid() const {
            return rspyRoArray1D(record.rigid_bodies.segment_bid, size());
        }
        [[nodiscard]] py::array_t<RFloat> mass() const {
            return rspyRoArray1D(record.rigid_bodies.mass, size());
        }
        [[nodiscard]] py::array_t<RFloat> orientation() const {
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.rigid_bodies.orientation), size(), 4
            );
        }
        [[nodiscard]] py::array_t<RFloat> com() const {
            return rspyRoArray2D(reinterpret_cast<RFloat *>(record.rigid_bodies.com), size(), 3);
        }
        [[nodiscard]] py::array_t<RFloat> linear_velocity() const {
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.rigid_bodies.linear_velocity), size(), 3
            );
        }
        [[nodiscard]] py::array_t<RFloat> angular_velocity() const {
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.rigid_bodies.angular_velocity), size(), 3
            );
        }
    };

    class Joint {
       public:
        RS_WorldStructureVoxelRecord record;
        explicit Joint(const RS_WorldStructureVoxelRecord &rec) : record(rec) {}

        [[nodiscard]] RBigSize size() const { return record.joints.object_num; }

        [[nodiscard]] py::array_t<RIndex> rigid_body_a_sid() const {
            return rspyRoArray1D(record.joints.rigid_body_a_sid, size());
        }
        [[nodiscard]] py::array_t<RIndex> rigid_body_b_sid() const {
            return rspyRoArray1D(record.joints.rigid_body_b_sid, size());
        }
        [[nodiscard]] py::array_t<RS_WorldStructureRecordBase::JointType> type() const {
            return rspyRoArray1D(record.joints.type, size());
        }
        [[nodiscard]] py::array_t<RFloat> position() const {
            return rspyRoArray2D(reinterpret_cast<RFloat *>(record.joints.position), size(), 3);
        }
        [[nodiscard]] py::array_t<RFloat> axis() const {
            return rspyRoArray2D(reinterpret_cast<RFloat *>(record.joints.axis), size(), 3);
        }
        [[nodiscard]] py::array_t<RIndex> hinge_signal_sid() const {
            return rspyRoArray1D(record.joints.hinge_signal_sid, size());
        }
        [[nodiscard]] py::array_t<RFloat> hinge_min() const {
            return rspyRoArray1D(record.joints.hinge_min, size());
        }
        [[nodiscard]] py::array_t<RFloat> hinge_max() const {
            return rspyRoArray1D(record.joints.hinge_max, size());
        }
        [[nodiscard]] py::array_t<RFloat> hinge_angle() const {
            return rspyRoArray1D(record.joints.hinge_angle, size());
        }
        [[nodiscard]] py::array_t<RFloat> hinge_torque() const {
            return rspyRoArray1D(record.joints.hinge_torque, size());
        }
        [[nodiscard]] py::array_t<RIndex> prismatic_signal_sid() const {
            return rspyRoArray1D(record.joints.prismatic_signal_sid, size());
        }
        [[nodiscard]] py::array_t<RFloat> prismatic_min() const {
            return rspyRoArray1D(record.joints.prismatic_min, size());
        }
        [[nodiscard]] py::array_t<RFloat> prismatic_max() const {
            return rspyRoArray1D(record.joints.prismatic_max, size());
        }
        [[nodiscard]] py::array_t<RFloat> prismatic_distance() const {
            return rspyRoArray1D(record.joints.prismatic_distance, size());
        }
        [[nodiscard]] py::array_t<RFloat> prismatic_force() const {
            return rspyRoArray1D(record.joints.prismatic_force, size());
        }
    };

    [[nodiscard]] RBigIndex step() const { return record.getStep(); }
    [[nodiscard]] RBigFloat time() const { return record.getTime(); }
    [[nodiscard]] RIndex structure_gid() const { return record.structure_gid; }

    [[nodiscard]] Voxel voxels() const { return Voxel(record); }
    [[nodiscard]] Link links() const { return Link(record); }
    [[nodiscard]] RigidBody rigid_bodies() const { return RigidBody(record); }
    [[nodiscard]] Joint joints() const { return Joint(record); }
};

class RSPY_WorldCameraPerspectiveRecord {
   public:
    RS_WorldCameraPerspectiveRecord record;
    explicit RSPY_WorldCameraPerspectiveRecord(const RS_WorldCameraPerspectiveRecord &rec)
        : record(rec) {}

    class RGB {
       public:
        RS_WorldCameraPerspectiveRecord record;
        explicit RGB(const RS_WorldCameraPerspectiveRecord &rec) : record(rec) {}

        [[nodiscard]] RSize width() const { return record.width; }
        [[nodiscard]] RSize height() const { return record.height; }

        /// Pixel data as a [height, width, 3] uint8 ndarray.
        [[nodiscard]] py::array_t<uint8_t> pixel() const {
            if (record.width == 0 || record.height == 0 || !record.isValid()) {
                return py::array_t<uint8_t>(0);
            }
            const size_t w = record.width;
            const size_t h = record.height;
            return py::array_t<uint8_t>(
                {h, w, static_cast<size_t>(3)},
                {w * 3 * sizeof(uint8_t), 3 * sizeof(uint8_t), sizeof(uint8_t)},
                reinterpret_cast<uint8_t *>(record.rgb.pixel)
            );
        }
    };

    class Depth {
       public:
        RS_WorldCameraPerspectiveRecord record;
        explicit Depth(const RS_WorldCameraPerspectiveRecord &rec) : record(rec) {}

        [[nodiscard]] RSize width() const { return record.width; }
        [[nodiscard]] RSize height() const { return record.height; }

        /// Pixel data as a [height, width] float32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<float> pixel() const {
            if (record.depth.object_num == 0 || !record.isValid()) {
                return py::array_t<float>(0);
            }
            const size_t w = record.width;
            const size_t h = record.height;
            return py::array_t<float>({h, w}, {w * sizeof(float), sizeof(float)}, record.depth.pixel);
        }
    };

    class Normal {
       public:
        RS_WorldCameraPerspectiveRecord record;
        explicit Normal(const RS_WorldCameraPerspectiveRecord &rec) : record(rec) {}

        [[nodiscard]] RSize width() const { return record.width; }
        [[nodiscard]] RSize height() const { return record.height; }

        /// Pixel data as a [height, width, 3] float32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<float> pixel() const {
            if (record.normal.object_num == 0 || !record.isValid()) {
                return py::array_t<float>(0);
            }
            const size_t w = record.width;
            const size_t h = record.height;
            return py::array_t<float>(
                {h, w, static_cast<size_t>(3)},
                {w * 3 * sizeof(float), 3 * sizeof(float), sizeof(float)},
                reinterpret_cast<float *>(record.normal.pixel)
            );
        }
    };

    class Motion {
       public:
        RS_WorldCameraPerspectiveRecord record;
        explicit Motion(const RS_WorldCameraPerspectiveRecord &rec) : record(rec) {}

        [[nodiscard]] RSize width() const { return record.width; }
        [[nodiscard]] RSize height() const { return record.height; }

        /// Pixel data as a [height, width, 2] float32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<float> pixel() const {
            if (record.motion.object_num == 0 || !record.isValid()) {
                return py::array_t<float>(0);
            }
            const size_t w = record.width;
            const size_t h = record.height;
            return py::array_t<float>(
                {h, w, static_cast<size_t>(2)},
                {w * 2 * sizeof(float), 2 * sizeof(float), sizeof(float)},
                reinterpret_cast<float *>(record.motion.pixel)
            );
        }
    };

    class EntityInfo {
       public:
        RS_WorldCameraPerspectiveRecord record;
        explicit EntityInfo(const RS_WorldCameraPerspectiveRecord &rec) : record(rec) {}

        [[nodiscard]] RSize width() const { return record.width; }
        [[nodiscard]] RSize height() const { return record.height; }

        /// Entity type ids as a [height, width] uint32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<uint32_t> entity_type() const {
            if (record.entity_info.object_num == 0 || !record.isValid()) {
                return py::array_t<uint32_t>(0);
            }
            const size_t w = record.width;
            const size_t h = record.height;
            return py::array_t<uint32_t>(
                {h, w},
                {w * sizeof(uint32_t), sizeof(uint32_t)},
                reinterpret_cast<uint32_t *>(record.entity_info.entity_type)
            );
        }

        /// Entity global ids as a [height, width] uint32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<uint32_t> entity_gid() const {
            if (record.entity_info.object_num == 0 || !record.isValid()) {
                return py::array_t<uint32_t>(0);
            }
            const size_t w = record.width;
            const size_t h = record.height;
            return py::array_t<uint32_t>(
                {h, w},
                {w * sizeof(uint32_t), sizeof(uint32_t)},
                reinterpret_cast<uint32_t *>(record.entity_info.entity_gid)
            );
        }
    };

    class State {
       public:
        RS_WorldCameraPerspectiveRecord record;
        explicit State(const RS_WorldCameraPerspectiveRecord &rec) : record(rec) {}

        [[nodiscard]] RBigSize size() const { return record.camera_state.object_num; }

        [[nodiscard]] py::array_t<RFloat> position() const {
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.camera_state.position), size(), 3
            );
        }
        [[nodiscard]] py::array_t<RFloat> orientation() const {
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.camera_state.orientation), size(), 4
            );
        }
        [[nodiscard]] py::array_t<RFloat> focus_distance() const {
            return rspyRoArray1D(record.camera_state.focus_distance, size());
        }
        [[nodiscard]] py::array_t<RFloat> field_of_view() const {
            return rspyRoArray1D(record.camera_state.field_of_view, size());
        }
    };

    [[nodiscard]] RBigIndex step() const { return record.getStep(); }
    [[nodiscard]] RBigFloat time() const { return record.getTime(); }
    [[nodiscard]] RIndex camera_gid() const { return record.camera_gid; }
    [[nodiscard]] RIndex structure_gid() const { return record.structure_gid; }
    [[nodiscard]] RIndex rigid_body_sid() const { return record.rigid_body_sid; }
    [[nodiscard]] RSize width() const { return record.width; }
    [[nodiscard]] RSize height() const { return record.height; }
    [[nodiscard]] bool is_valid() const { return record.isValid(); }

    [[nodiscard]] RGB rgb() const { return RGB(record); }
    [[nodiscard]] Depth depth() const { return Depth(record); }
    [[nodiscard]] Normal normal() const { return Normal(record); }
    [[nodiscard]] Motion motion() const { return Motion(record); }
    [[nodiscard]] EntityInfo entity_info() const { return EntityInfo(record); }
    [[nodiscard]] State state() const { return State(record); }
};

class RSPY_WorldCameraStereoRecord {
   public:
    RS_WorldCameraStereoRecord record;
    explicit RSPY_WorldCameraStereoRecord(const RS_WorldCameraStereoRecord &rec) : record(rec) {}

    class RGB {
       public:
        RS_WorldCameraRecordBase::RGB rgb;
        RSize w, h;
        bool valid;
        RGB(const RS_WorldCameraRecordBase::RGB &r, RSize w, RSize h, bool v)
            : rgb(r), w(w), h(h), valid(v) {}

        [[nodiscard]] RSize width() const { return w; }
        [[nodiscard]] RSize height() const { return h; }

        /// Pixel data as a [height, width, 3] uint8 ndarray.
        [[nodiscard]] py::array_t<uint8_t> pixel() const {
            if (w == 0 || h == 0 || !valid) return py::array_t<uint8_t>(0);
            return py::array_t<uint8_t>(
                {static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(3)},
                {w * 3 * sizeof(uint8_t), 3 * sizeof(uint8_t), sizeof(uint8_t)},
                reinterpret_cast<uint8_t *>(rgb.pixel)
            );
        }
    };

    class Depth {
       public:
        RS_WorldCameraRecordBase::Depth depth;
        RSize w, h;
        bool valid;
        Depth(const RS_WorldCameraRecordBase::Depth &d, RSize w, RSize h, bool v)
            : depth(d), w(w), h(h), valid(v) {}

        [[nodiscard]] RSize width() const { return w; }
        [[nodiscard]] RSize height() const { return h; }

        /// Pixel data as a [height, width] float32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<float> pixel() const {
            if (depth.object_num == 0 || !valid) return py::array_t<float>(0);
            return py::array_t<float>(
                {static_cast<size_t>(h), static_cast<size_t>(w)},
                {w * sizeof(float), sizeof(float)},
                depth.pixel
            );
        }
    };

    class Normal {
       public:
        RS_WorldCameraRecordBase::Normal normal;
        RSize w, h;
        bool valid;
        Normal(const RS_WorldCameraRecordBase::Normal &n, RSize w, RSize h, bool v)
            : normal(n), w(w), h(h), valid(v) {}

        [[nodiscard]] RSize width() const { return w; }
        [[nodiscard]] RSize height() const { return h; }

        /// Pixel data as a [height, width, 3] float32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<float> pixel() const {
            if (normal.object_num == 0 || !valid) return py::array_t<float>(0);
            return py::array_t<float>(
                {static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(3)},
                {w * 3 * sizeof(float), 3 * sizeof(float), sizeof(float)},
                reinterpret_cast<float *>(normal.pixel)
            );
        }
    };

    class Motion {
       public:
        RS_WorldCameraRecordBase::Motion motion;
        RSize w, h;
        bool valid;
        Motion(const RS_WorldCameraRecordBase::Motion &m, RSize w, RSize h, bool v)
            : motion(m), w(w), h(h), valid(v) {}

        [[nodiscard]] RSize width() const { return w; }
        [[nodiscard]] RSize height() const { return h; }

        /// Pixel data as a [height, width, 2] float32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<float> pixel() const {
            if (motion.object_num == 0 || !valid) return py::array_t<float>(0);
            return py::array_t<float>(
                {static_cast<size_t>(h), static_cast<size_t>(w), static_cast<size_t>(2)},
                {w * 2 * sizeof(float), 2 * sizeof(float), sizeof(float)},
                reinterpret_cast<float *>(motion.pixel)
            );
        }
    };

    class EntityInfo {
       public:
        RS_WorldCameraRecordBase::EntityInfo entity_info;
        RSize w, h;
        bool valid;
        EntityInfo(const RS_WorldCameraRecordBase::EntityInfo &e, RSize w, RSize h, bool v)
            : entity_info(e), w(w), h(h), valid(v) {}

        [[nodiscard]] RSize width() const { return w; }
        [[nodiscard]] RSize height() const { return h; }

        /// Entity type ids as a [height, width] uint32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<uint32_t> entity_type() const {
            if (entity_info.object_num == 0 || !valid) return py::array_t<uint32_t>(0);
            return py::array_t<uint32_t>(
                {static_cast<size_t>(h), static_cast<size_t>(w)},
                {w * sizeof(uint32_t), sizeof(uint32_t)},
                reinterpret_cast<uint32_t *>(entity_info.entity_type)
            );
        }

        /// Entity global ids as a [height, width] uint32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<uint32_t> entity_gid() const {
            if (entity_info.object_num == 0 || !valid) return py::array_t<uint32_t>(0);
            return py::array_t<uint32_t>(
                {static_cast<size_t>(h), static_cast<size_t>(w)},
                {w * sizeof(uint32_t), sizeof(uint32_t)},
                reinterpret_cast<uint32_t *>(entity_info.entity_gid)
            );
        }
    };

    class State {
       public:
        RS_WorldCameraRecordBase::State state;
        explicit State(const RS_WorldCameraRecordBase::State &s) : state(s) {}

        [[nodiscard]] RBigSize size() const { return state.object_num; }

        [[nodiscard]] py::array_t<RFloat> position() const {
            return rspyRoArray2D(reinterpret_cast<RFloat *>(state.position), size(), 3);
        }
        [[nodiscard]] py::array_t<RFloat> orientation() const {
            return rspyRoArray2D(reinterpret_cast<RFloat *>(state.orientation), size(), 4);
        }
        [[nodiscard]] py::array_t<RFloat> focus_distance() const {
            return rspyRoArray1D(state.focus_distance, size());
        }
        [[nodiscard]] py::array_t<RFloat> field_of_view() const {
            return rspyRoArray1D(state.field_of_view, size());
        }
    };

    [[nodiscard]] RBigIndex step() const { return record.getStep(); }
    [[nodiscard]] RBigFloat time() const { return record.getTime(); }
    [[nodiscard]] RIndex camera_gid() const { return record.camera_gid; }
    [[nodiscard]] RIndex structure_gid() const { return record.structure_gid; }
    [[nodiscard]] RIndex rigid_body_sid() const { return record.rigid_body_sid; }
    [[nodiscard]] RSize width() const { return record.width; }
    [[nodiscard]] RSize height() const { return record.height; }
    [[nodiscard]] bool is_valid() const { return record.isValid(); }

    [[nodiscard]] RGB left_rgb() const {
        return RGB(record.left_rgb, record.width, record.height, record.isValid());
    }
    [[nodiscard]] RGB right_rgb() const {
        return RGB(record.right_rgb, record.width, record.height, record.isValid());
    }
    [[nodiscard]] Depth left_depth() const {
        return Depth(record.left_depth, record.width, record.height, record.isValid());
    }
    [[nodiscard]] Depth right_depth() const {
        return Depth(record.right_depth, record.width, record.height, record.isValid());
    }
    [[nodiscard]] Normal left_normal() const {
        return Normal(record.left_normal, record.width, record.height, record.isValid());
    }
    [[nodiscard]] Normal right_normal() const {
        return Normal(record.right_normal, record.width, record.height, record.isValid());
    }
    [[nodiscard]] Motion left_motion() const {
        return Motion(record.left_motion, record.width, record.height, record.isValid());
    }
    [[nodiscard]] Motion right_motion() const {
        return Motion(record.right_motion, record.width, record.height, record.isValid());
    }
    [[nodiscard]] EntityInfo left_entity_info() const {
        return EntityInfo(record.left_entity_info, record.width, record.height, record.isValid());
    }
    [[nodiscard]] EntityInfo right_entity_info() const {
        return EntityInfo(record.right_entity_info, record.width, record.height, record.isValid());
    }
    [[nodiscard]] State left_camera_state() const { return State(record.left_camera_state); }
    [[nodiscard]] State right_camera_state() const { return State(record.right_camera_state); }
};

class RSPY_WorldIntersectRegionRecord {
   public:
    RS_WorldIntersectRegionRecord record;
    explicit RSPY_WorldIntersectRegionRecord(const RS_WorldIntersectRegionRecord &rec)
        : record(rec) {}

    class Membership {
       public:
        RS_WorldIntersectRegionRecord record;
        explicit Membership(const RS_WorldIntersectRegionRecord &rec) : record(rec) {}

        [[nodiscard]] RBigSize size() const { return record.memberships.object_num; }

        [[nodiscard]] py::array_t<RIndex> structure_gid() const {
            return rspyRoArray1D(record.memberships.structure_gid, size());
        }
    };

    [[nodiscard]] RBigIndex step() const { return record.getStep(); }
    [[nodiscard]] RBigFloat time() const { return record.getTime(); }
    [[nodiscard]] RIndex record_gid() const { return record.record_gid; }

    [[nodiscard]] Membership memberships() const { return Membership(record); }
};

/*****************************************************************
 * Signal wrappers.
 *****************************************************************/

class RSPY_WorldStructureVoxelSignal {
   public:
    RS_WorldStructureVoxelSignal signal;
    explicit RSPY_WorldStructureVoxelSignal(const RS_WorldStructureVoxelSignal &sig)
        : signal(sig) {}

    class Expansion {
       public:
        RS_WorldStructureVoxelSignal signal;
        explicit Expansion(const RS_WorldStructureVoxelSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.expansion_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> expansion() const {
            return rspyRwArray1D(signal.expansion_signals.expansion, size());
        }
    };

    class Hinge {
       public:
        RS_WorldStructureVoxelSignal signal;
        explicit Hinge(const RS_WorldStructureVoxelSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.hinge_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> hinge() const {
            return rspyRwArray1D(signal.hinge_signals.hinge, size());
        }
    };

    class Prismatic {
       public:
        RS_WorldStructureVoxelSignal signal;
        explicit Prismatic(const RS_WorldStructureVoxelSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.prismatic_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> prismatic() const {
            return rspyRwArray1D(signal.prismatic_signals.prismatic, size());
        }
    };

    [[nodiscard]] Expansion expansion_signals() const { return Expansion(signal); }
    [[nodiscard]] Hinge hinge_signals() const { return Hinge(signal); }
    [[nodiscard]] Prismatic prismatic_signals() const { return Prismatic(signal); }
};

class RSPY_WorldCameraPerspectiveSignal {
   public:
    RS_WorldCameraPerspectiveSignal signal;
    explicit RSPY_WorldCameraPerspectiveSignal(const RS_WorldCameraPerspectiveSignal &sig)
        : signal(sig) {}

    class Position {
       public:
        RS_WorldCameraPerspectiveSignal signal;
        explicit Position(const RS_WorldCameraPerspectiveSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.position_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> position() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.position_signals.position), size(), 3
            );
        }
    };

    class Orientation {
       public:
        RS_WorldCameraPerspectiveSignal signal;
        explicit Orientation(const RS_WorldCameraPerspectiveSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.orientation_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> orientation() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.orientation_signals.orientation), size(), 4
            );
        }
    };

    class Optics {
       public:
        RS_WorldCameraPerspectiveSignal signal;
        explicit Optics(const RS_WorldCameraPerspectiveSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.optics_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> focus_distance() const {
            return rspyRwArray1D(signal.optics_signals.focus_distance, size());
        }
        [[nodiscard]] py::array_t<RFloat> field_of_view() const {
            return rspyRwArray1D(signal.optics_signals.field_of_view, size());
        }
    };

    [[nodiscard]] Position position_signals() const { return Position(signal); }
    [[nodiscard]] Orientation orientation_signals() const { return Orientation(signal); }
    [[nodiscard]] Optics optics_signals() const { return Optics(signal); }
};

class RSPY_WorldCameraStereoSignal {
   public:
    RS_WorldCameraStereoSignal signal;
    explicit RSPY_WorldCameraStereoSignal(const RS_WorldCameraStereoSignal &sig) : signal(sig) {}

    class Position {
       public:
        RS_WorldCameraSignalBase::Position position;
        explicit Position(const RS_WorldCameraSignalBase::Position &p) : position(p) {}

        [[nodiscard]] RBigSize size() const { return position.object_num; }

        [[nodiscard]] py::array_t<RFloat> position_array() const {
            return rspyRwArray2D(reinterpret_cast<RFloat *>(position.position), size(), 3);
        }
    };

    class Orientation {
       public:
        RS_WorldCameraSignalBase::Orientation orientation;
        explicit Orientation(const RS_WorldCameraSignalBase::Orientation &o) : orientation(o) {}

        [[nodiscard]] RBigSize size() const { return orientation.object_num; }

        [[nodiscard]] py::array_t<RFloat> orientation_array() const {
            return rspyRwArray2D(reinterpret_cast<RFloat *>(orientation.orientation), size(), 4);
        }
    };

    class Optics {
       public:
        RS_WorldCameraSignalBase::Optics optics;
        explicit Optics(const RS_WorldCameraSignalBase::Optics &o) : optics(o) {}

        [[nodiscard]] RBigSize size() const { return optics.object_num; }

        [[nodiscard]] py::array_t<RFloat> field_of_view() const {
            return rspyRwArray1D(optics.field_of_view, size());
        }
    };

    [[nodiscard]] Position left_position_signals() const {
        return Position(signal.left_position_signals);
    }
    [[nodiscard]] Position right_position_signals() const {
        return Position(signal.right_position_signals);
    }
    [[nodiscard]] Orientation left_orientation_signals() const {
        return Orientation(signal.left_orientation_signals);
    }
    [[nodiscard]] Orientation right_orientation_signals() const {
        return Orientation(signal.right_orientation_signals);
    }
    [[nodiscard]] Optics left_optics_signals() const {
        return Optics(signal.left_optics_signals);
    }
    [[nodiscard]] Optics right_optics_signals() const {
        return Optics(signal.right_optics_signals);
    }
};

class RSPY_WorldForceSignal {
   public:
    RS_WorldForceSignal signal;
    explicit RSPY_WorldForceSignal(const RS_WorldForceSignal &sig) : signal(sig) {}

    class Force {
       public:
        RS_WorldForceSignal signal;
        explicit Force(const RS_WorldForceSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.force_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> force() const {
            return rspyRwArray2D(reinterpret_cast<RFloat *>(signal.force_signals.force), size(), 3);
        }
    };

    [[nodiscard]] Force force_signals() const { return Force(signal); }
};

class RSPY_WorldTorqueSignal {
   public:
    RS_WorldTorqueSignal signal;
    explicit RSPY_WorldTorqueSignal(const RS_WorldTorqueSignal &sig) : signal(sig) {}

    class Torque {
       public:
        RS_WorldTorqueSignal signal;
        explicit Torque(const RS_WorldTorqueSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.torque_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> torque() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.torque_signals.torque), size(), 3
            );
        }
    };

    [[nodiscard]] Torque torque_signals() const { return Torque(signal); }
};

class RSPY_WorldTransformSignal {
   public:
    RS_WorldTransformSignal signal;
    explicit RSPY_WorldTransformSignal(const RS_WorldTransformSignal &sig) : signal(sig) {}

    class Translation {
       public:
        RS_WorldTransformSignal signal;
        explicit Translation(const RS_WorldTransformSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.translation_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> translation() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.translation_signals.translation), size(), 3
            );
        }
    };

    class Rotation {
       public:
        RS_WorldTransformSignal signal;
        explicit Rotation(const RS_WorldTransformSignal &sig) : signal(sig) {}

        [[nodiscard]] RBigSize size() const { return signal.rotation_signals.object_num; }

        [[nodiscard]] py::array_t<RFloat> rotation() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.rotation_signals.rotation), size(), 4
            );
        }

        [[nodiscard]] py::array_t<RFloat> pivot() const {
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.rotation_signals.pivot), size(), 3
            );
        }
    };

    [[nodiscard]] Translation translation_signals() const { return Translation(signal); }

    [[nodiscard]] Rotation rotation_signals() const { return Rotation(signal); }
};

/*****************************************************************
 * Controller wrapper.
 *****************************************************************/

class RSPY_WorldController {
   public:
    RS_WorldController controller;
    explicit RSPY_WorldController(const RS_WorldController &ctrl) : controller(ctrl) {}

    [[nodiscard]] std::string name() const { return controller.name; }

    [[nodiscard]] py::list records() const {
        py::list result;
        for (const auto &record : controller.records) {
            record.visit([&](const auto &held) {
                using T = std::decay_t<decltype(held)>;
                if constexpr (std::is_same_v<T, RS_WorldStructureVoxelRecord>) {
                    result.append(py::cast(RSPY_WorldStructureVoxelRecord(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldCameraPerspectiveRecord>) {
                    result.append(py::cast(RSPY_WorldCameraPerspectiveRecord(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldCameraStereoRecord>) {
                    result.append(py::cast(RSPY_WorldCameraStereoRecord(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldIntersectRegionRecord>) {
                    result.append(py::cast(RSPY_WorldIntersectRegionRecord(held)));
                }
            });
        }
        return result;
    }

    [[nodiscard]] py::list signals() const {
        py::list result;
        for (const auto &signal : controller.signals) {
            signal.visit([&](const auto &held) {
                using T = std::decay_t<decltype(held)>;
                if constexpr (std::is_same_v<T, RS_WorldStructureVoxelSignal>) {
                    result.append(py::cast(RSPY_WorldStructureVoxelSignal(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldCameraPerspectiveSignal>) {
                    result.append(py::cast(RSPY_WorldCameraPerspectiveSignal(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldCameraStereoSignal>) {
                    result.append(py::cast(RSPY_WorldCameraStereoSignal(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldForceSignal>) {
                    result.append(py::cast(RSPY_WorldForceSignal(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldTorqueSignal>) {
                    result.append(py::cast(RSPY_WorldTorqueSignal(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldTransformSignal>) {
                    result.append(py::cast(RSPY_WorldTransformSignal(held)));
                }
            });
        }
        return result;
    }
};

/*****************************************************************
 * World handle wrapper.
 *****************************************************************/

class RSPY_WorldHandle {
   public:
    std::shared_ptr<RS_WorldHandle> handle;
    explicit RSPY_WorldHandle(std::shared_ptr<RS_WorldHandle> handle) : handle(std::move(handle)) {}

    // Lifecycle
    [[nodiscard]] bool is_pending() const { return handle->isPending(); }
    [[nodiscard]] bool is_running() const { return handle->isRunning(); }
    [[nodiscard]] bool is_paused() const { return handle->isPaused(); }
    [[nodiscard]] bool is_end() const { return handle->isEnd(); }
    void pause() const { handle->pause(); }
    void resume() const { handle->resume(); }
    void terminate() const { handle->terminate(); }
    [[nodiscard]] RSE_WorldStatus get_status() const { return handle->getStatus(); }

    bool wait_for_running() const {
        // Release the GIL before blocking on the simulation thread.
        py::gil_scoped_release release;
        return handle->waitForRunning();
    }

    void wait_for_end() const {
        py::gil_scoped_release release;
        handle->waitForEnd();
    }

    [[nodiscard]] bool has_message() const { return handle->hasMessage(); }

    [[nodiscard]] std::vector<RS_WorldMessage> get_messages() const {
        return handle->getMessages();
    }

    RS_WorldMessage wait_for_message(
        RIndex message_id = RS_NULL_INDEX,
        RSE_WorldMessageLevel level = RSE_WORLD_MESSAGE_ANY_LEVEL
    ) const {
        py::gil_scoped_release release;
        return handle->waitForMessage(message_id, level);
    }

    void begin_modification_transaction() const { handle->beginModificationTransaction(); }
    void end_modification_transaction() const { handle->endModificationTransaction(); }

    [[nodiscard]] rsc::RSC_World get_world() const { return handle->getWorld(); }

    RIndex set_simulation(const rsc::RSC_Simulation &simulation) const {
        return handle->setSimulation(simulation);
    }

    // Resource
    RIndex add_resource(const rsc::RSC_ResourceVariant &resource) const {
        return handle->addResource(resource);
    }
    RIndex remove_resource(const std::string &name) const { return handle->removeResource(name); }
    [[nodiscard]] rsc::RSC_ResourceVariant find_resource(const std::string &name) const {
        return handle->findResource(name);
    }

    // Essence
    RIndex add_essence(const rsc::RSC_EssenceVariant &essence) const {
        return handle->addEssence(essence);
    }
    RIndex update_essence(const std::string &name, const rsc::RSC_EssenceVariant &essence) const {
        return handle->updateEssence(name, essence);
    }
    RIndex remove_essence(const std::string &name) const { return handle->removeEssence(name); }
    [[nodiscard]] rsc::RSC_EssenceVariant find_essence(const std::string &name) const {
        return handle->findEssence(name);
    }

    // Entity
    RIndex add_entity(const rsc::RSC_EntityVariant &entity) const {
        return handle->addEntity(entity);
    }
    RIndex remove_entity(const std::string &name) const { return handle->removeEntity(name); }
    RIndex freeze_entity(const std::string &name) const { return handle->freezeEntity(name); }
    RIndex unfreeze_entity(const std::string &name) const { return handle->unfreezeEntity(name); }
    [[nodiscard]] rsc::RSC_EntityVariant find_entity(const std::string &name) const {
        return handle->findEntity(name);
    }
    [[nodiscard]] bool is_entity_frozen(const std::string &name) const {
        return handle->isEntityFrozen(name);
    }

    // Record
    RIndex add_record(const rsc::RSC_RecordVariant &record) const {
        return handle->addRecord(record);
    }
    RIndex remove_record(const std::string &name) const { return handle->removeRecord(name); }
    [[nodiscard]] rsc::RSC_RecordVariant find_record(const std::string &name) const {
        return handle->findRecord(name);
    }

    // Signal
    RIndex add_signal(const rsc::RSC_SignalVariant &signal) const {
        return handle->addSignal(signal);
    }
    RIndex remove_signal(const std::string &name) const { return handle->removeSignal(name); }
    [[nodiscard]] rsc::RSC_SignalVariant find_signal(const std::string &name) const {
        return handle->findSignal(name);
    }

    // Recorder
    RIndex add_recorder(const rsc::RSC_RecorderVariant &recorder) const {
        return handle->addRecorder(recorder);
    }
    RIndex remove_recorder(const std::string &name) const { return handle->removeRecorder(name); }
    [[nodiscard]] rsc::RSC_RecorderVariant find_recorder(const std::string &name) const {
        return handle->findRecorder(name);
    }

    // Controller
    RIndex add_controller(const rsc::RSC_ControllerVariant &controller) const {
        return handle->addController(controller);
    }
    RIndex remove_controller(const std::string &name) const {
        return handle->removeController(name);
    }
    [[nodiscard]] rsc::RSC_ControllerVariant find_controller(const std::string &name) const {
        return handle->findController(name);
    }
};

/*****************************************************************
 * World callback wrapper.
 *****************************************************************/

class RSPY_WorldCallback {
   public:
    std::shared_ptr<RS_WorldCallback> callback;
    explicit RSPY_WorldCallback(std::shared_ptr<RS_WorldCallback> callback)
        : callback(std::move(callback)) {}

    [[nodiscard]] std::string get_world_name() const { return callback->getWorldName(); }

    void set_callback(py::function py_callback) {
        // The engine drops the last reference to the wrapping function on its own thread, so the
        // captured Python object must be released under the GIL; a custom deleter does that. The same
        // holder backs get_callback, so it stays the function Python set.
        _py_callback = std::shared_ptr<py::object>(
            new py::object(std::move(py_callback)),
            [](py::object *object) {
                py::gil_scoped_acquire gil;
                delete object;
            }
        );
        std::shared_ptr<py::object> py_object = _py_callback;
        callback->setCallback(
            [py_object](RBigFloat time, std::vector<RS_WorldController> controllers) {
                py::gil_scoped_acquire gil;
                py::list controllers_list;
                for (auto &controller : controllers) {
                    controllers_list.append(RSPY_WorldController(controller));
                }
                try {
                    py::cast<py::function>(*py_object)(time, controllers_list);
                } catch (const py::error_already_set &error) {
                    py::print("Exception in world callback:", error.what());
                }
            }
        );
    }

    void reset_callback() {
        _py_callback.reset();
        callback->resetCallback();
    }

    [[nodiscard]] py::object get_callback() const {
        return _py_callback ? *_py_callback : py::none();
    }

   private:
    std::shared_ptr<py::object> _py_callback;
};

inline void initRSPYCore(py::module &m) {
    /*****************************************************************
     * Enums and device location
     *****************************************************************/

    py::enum_<RSE_DeviceType>(m, "RSE_DeviceType")
        .value("RSE_DEVICE_TYPE_UNKNOWN", RSE_DEVICE_TYPE_UNKNOWN)
        .value("RSE_DEVICE_TYPE_CPU", RSE_DEVICE_TYPE_CPU)
        .value("RSE_DEVICE_TYPE_CUDA", RSE_DEVICE_TYPE_CUDA)
        .export_values();

    py::enum_<RSE_WorldStatus>(m, "RSE_WorldStatus")
        .value("RSE_WORLD_END_SUCCESS", RSE_WORLD_END_SUCCESS)
        .value("RSE_WORLD_END_TERMINATED", RSE_WORLD_END_TERMINATED)
        .value("RSE_WORLD_END_FAILED_WITH_INVALID_CONFIG", RSE_WORLD_END_FAILED_WITH_INVALID_CONFIG)
        .value(
            "RSE_WORLD_END_FAILED_WITH_SIMULATION_EXCEPTION",
            RSE_WORLD_END_FAILED_WITH_SIMULATION_EXCEPTION
        )
        .value(
            "RSE_WORLD_END_FAILED_WITH_SIMULATION_MANAGER_EXCEPTION",
            RSE_WORLD_END_FAILED_WITH_SIMULATION_MANAGER_EXCEPTION
        )
        .value("RSE_WORLD_UNINITIALIZED", RSE_WORLD_UNINITIALIZED)
        .value("RSE_WORLD_INITIALIZING", RSE_WORLD_INITIALIZING)
        .value("RSE_WORLD_RUNNING", RSE_WORLD_RUNNING)
        .value("RSE_WORLD_PAUSED", RSE_WORLD_PAUSED)
        .value("RSE_WORLD_TERMINATING", RSE_WORLD_TERMINATING)
        .value("RSE_WORLD_MAX_NORMAL_STATE", RSE_WORLD_MAX_NORMAL_STATE)
        .export_values();

    py::class_<RSPY_DeviceLocation>(m, "RS_DeviceLocation")
        .def(py::init<>())
        .def(py::init<const std::string &>())
        .def(py::init<RSE_DeviceType, RIndex>())
        .def(py::init<const RS_DeviceLocation &>())
        .def("type", &RSPY_DeviceLocation::type)
        .def("id", &RSPY_DeviceLocation::id)
        .def("__str__", &RSPY_DeviceLocation::str);

    /*****************************************************************
     * World messages
     *****************************************************************/
    py::enum_<RSE_WorldMessageLevel>(m, "RSE_WorldMessageLevel")
        .value("RSE_WORLD_MESSAGE_ANY_LEVEL", RSE_WORLD_MESSAGE_ANY_LEVEL)
        .value("RSE_WORLD_MESSAGE_INFO", RSE_WORLD_MESSAGE_INFO)
        .value("RSE_WORLD_MESSAGE_WARNING", RSE_WORLD_MESSAGE_WARNING)
        .value("RSE_WORLD_MESSAGE_ERROR", RSE_WORLD_MESSAGE_ERROR)
        .export_values();

    py::enum_<RSE_WorldMessageCreatedType>(m, "RSE_WorldMessageCreatedType")
        .value("RSE_WORLD_MESSAGE_CREATED_TYPE_NONE", RSE_WORLD_MESSAGE_CREATED_TYPE_NONE)
        .value("RSE_WORLD_MESSAGE_CREATED_TYPE_ENTITY", RSE_WORLD_MESSAGE_CREATED_TYPE_ENTITY)
        .value("RSE_WORLD_MESSAGE_CREATED_TYPE_RESOURCE", RSE_WORLD_MESSAGE_CREATED_TYPE_RESOURCE)
        .value("RSE_WORLD_MESSAGE_CREATED_TYPE_ESSENCE", RSE_WORLD_MESSAGE_CREATED_TYPE_ESSENCE)
        .value("RSE_WORLD_MESSAGE_CREATED_TYPE_RECORD", RSE_WORLD_MESSAGE_CREATED_TYPE_RECORD)
        .value("RSE_WORLD_MESSAGE_CREATED_TYPE_SIGNAL", RSE_WORLD_MESSAGE_CREATED_TYPE_SIGNAL)
        .export_values();

    py::class_<RS_WorldMessageText>(m, "RS_WorldMessageText")
        .def(py::init<>())
        .def_readwrite("text", &RS_WorldMessageText::text);

    py::class_<RS_WorldMessageCreated>(m, "RS_WorldMessageCreated")
        .def(py::init<>())
        .def_readwrite("type", &RS_WorldMessageCreated::type)
        .def_readwrite("gid", &RS_WorldMessageCreated::gid)
        .def_readwrite("name", &RS_WorldMessageCreated::name);

    py::class_<RS_WorldMessageData>(m, "RS_WorldMessageData")
        .def(py::init<>())
        .def_property_readonly("value", [](const RS_WorldMessageData &self) -> py::object {
            if (self.index() == RS_NULL_INDEX) return py::none();
            py::object result = py::none();
            self.visit([&](auto &held) { result = py::cast(held); });
            return result;
        });

    py::class_<RS_WorldMessage>(m, "RS_WorldMessage")
        .def(py::init<>())
        .def_readonly("id", &RS_WorldMessage::id)
        .def_readonly("level", &RS_WorldMessage::level)
        .def_readonly("data", &RS_WorldMessage::data);

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

    py::class_<RSPY_WorldCameraPerspectiveRecord::EntityInfo>(camera_perspective_record, "EntityInfo")
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

    auto intersect_region_record =
        py::class_<RSPY_WorldIntersectRegionRecord>(m, "RS_WorldIntersectRegionRecord")
            .def("step", &RSPY_WorldIntersectRegionRecord::step)
            .def("time", &RSPY_WorldIntersectRegionRecord::time)
            .def("record_gid", &RSPY_WorldIntersectRegionRecord::record_gid)
            .def_property_readonly("memberships", &RSPY_WorldIntersectRegionRecord::memberships);

    py::class_<RSPY_WorldIntersectRegionRecord::Membership>(intersect_region_record, "Membership")
        .def("size", &RSPY_WorldIntersectRegionRecord::Membership::size)
        .def("structure_gid", &RSPY_WorldIntersectRegionRecord::Membership::structure_gid);

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

    /*****************************************************************
     * Controller
     *****************************************************************/

    py::class_<RSPY_WorldController>(m, "RS_WorldController")
        .def("name", &RSPY_WorldController::name)
        .def("records", &RSPY_WorldController::records)
        .def("signals", &RSPY_WorldController::signals);

    /*****************************************************************
     * World handle
     *****************************************************************/

    py::class_<RSPY_WorldHandle>(m, "RS_WorldHandle")
        .def(
            "is_pending",
            &RSPY_WorldHandle::is_pending,
            R"doc(
            Reports whether the world is still uninitialized or initializing.

            Returns:
                True when the world has not yet started running.
        )doc"
        )
        .def(
            "is_running",
            &RSPY_WorldHandle::is_running,
            R"doc(
            Reports whether the world is currently running.

            Returns:
                True when the world is in the running state.
        )doc"
        )
        .def(
            "is_paused",
            &RSPY_WorldHandle::is_paused,
            R"doc(
            Reports whether the world is currently paused.

            Returns:
                True when the world is in the paused state.
        )doc"
        )
        .def(
            "is_end",
            &RSPY_WorldHandle::is_end,
            R"doc(
            Reports whether the world has ended.

            Returns:
                True when the world has reached any end state, whether clean or failed.
        )doc"
        )
        .def(
            "pause",
            &RSPY_WorldHandle::pause,
            R"doc(
            Pauses the running world.

            Blocks until the world is running, then requests a pause. Does nothing if the
            world ends before it starts running.
        )doc"
        )
        .def(
            "resume",
            &RSPY_WorldHandle::resume,
            R"doc(
            Resumes the paused world.

            Blocks until the world is running, then requests a resume. Does nothing if the
            world ends before it starts running.
        )doc"
        )
        .def(
            "terminate",
            &RSPY_WorldHandle::terminate,
            R"doc(
            Terminates the running world.

            Blocks until the world is running, then requests termination. Does nothing if
            the world ends before it starts running.
        )doc"
        )
        .def(
            "get_status",
            &RSPY_WorldHandle::get_status,
            R"doc(
            Reads the current world status.

            Returns:
                The current RSE_WorldStatus.
        )doc"
        )
        .def(
            "wait_for_running",
            &RSPY_WorldHandle::wait_for_running,
            R"doc(
            Blocks until the world starts running or ends.

            Releases the GIL while waiting.

            Returns:
                True if the world reached the running state, False if it ended first.
        )doc"
        )
        .def(
            "wait_for_end",
            &RSPY_WorldHandle::wait_for_end,
            R"doc(
            Blocks until the world ends.

            Releases the GIL while waiting.
        )doc"
        )
        .def(
            "has_message",
            &RSPY_WorldHandle::has_message,
            R"doc(
            Reports whether one or more world messages are waiting to be drained.

            Returns:
                True when get_messages() would return at least one message.
        )doc"
        )
        .def(
            "get_messages",
            &RSPY_WorldHandle::get_messages,
            R"doc(
            Drains all currently queued world messages.

            Messages are returned in posting order and removed from the queue atomically.
            Once drained, a message cannot satisfy a later wait_for_message() call.

            Returns:
                The queued RS_WorldMessage objects.
        )doc"
        )
        .def(
            "wait_for_message",
            &RSPY_WorldHandle::wait_for_message,
            py::arg("message_id") = RS_NULL_INDEX,
            py::arg("level") = RSE_WORLD_MESSAGE_ANY_LEVEL,
            R"doc(
            Blocks until a matching world message arrives.

            Releases the GIL while waiting.
            Messages drained by get_messages() are no longer available to this call.

            Args:
                message_id: The id returned by a mutator to wait on its outcome. Defaults to
                            RS_NULL_INDEX, which waits for the next message posted after this
                            call regardless of id.
                level: The minimum severity to match. Defaults to RSE_WORLD_MESSAGE_ANY_LEVEL,
                       which matches any severity.

            Returns:
                The matching RS_WorldMessage.
        )doc"
        )
        .def(
            "begin_modification_transaction",
            &RSPY_WorldHandle::begin_modification_transaction,
            R"doc(
            Opens a modification transaction so several edits apply on the same step.

            Blocks until the world is running. Does nothing if the world ends before it
            starts running.
        )doc"
        )
        .def(
            "end_modification_transaction",
            &RSPY_WorldHandle::end_modification_transaction,
            R"doc(
            Closes the open modification transaction and applies its batched edits.

            Blocks until the world is running. Does nothing if the world ends before it
            starts running.
        )doc"
        )
        .def(
            "get_world",
            &RSPY_WorldHandle::get_world,
            R"doc(
            Snapshots the current world characterization.

            Blocks until the world is running, then reads back the live world including its
            active recorders and controllers.

            Returns:
                The current RSC_World, or an empty RSC_World if the world ended first.
        )doc"
        )
        .def(
            "set_simulation",
            &RSPY_WorldHandle::set_simulation,
            py::arg("simulation"),
            R"doc(
            Replaces the simulation parameters of the running world.

            Args:
                simulation: The RSC_Simulation to apply.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "add_resource",
            &RSPY_WorldHandle::add_resource,
            py::arg("resource"),
            R"doc(
            Validates and adds a resource to the running world.

            Args:
                resource: The RSC_ResourceVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the resource fails validation.
        )doc"
        )
        .def(
            "remove_resource",
            &RSPY_WorldHandle::remove_resource,
            py::arg("name"),
            R"doc(
            Removes the named resource from the running world.

            Args:
                name: The name of the resource to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_resource",
            &RSPY_WorldHandle::find_resource,
            py::arg("name"),
            R"doc(
            Looks up the named resource in the running world.

            Args:
                name: The name of the resource to find.

            Returns:
                The matching RSC_ResourceVariant, or an empty variant if the world is not
                running.
        )doc"
        )
        .def(
            "add_essence",
            &RSPY_WorldHandle::add_essence,
            py::arg("essence"),
            R"doc(
            Validates and adds an essence to the running world.

            Args:
                essence: The RSC_EssenceVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the essence fails validation.
        )doc"
        )
        .def(
            "update_essence",
            &RSPY_WorldHandle::update_essence,
            py::arg("name"),
            py::arg("essence"),
            R"doc(
            Validates and updates the named essence in the running world.

            Args:
                name: The name of the essence to update.
                essence: The RSC_EssenceVariant holding the new values.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the essence fails validation.
        )doc"
        )
        .def(
            "remove_essence",
            &RSPY_WorldHandle::remove_essence,
            py::arg("name"),
            R"doc(
            Removes the named essence from the running world.

            Args:
                name: The name of the essence to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_essence",
            &RSPY_WorldHandle::find_essence,
            py::arg("name"),
            R"doc(
            Looks up the named essence in the running world.

            Args:
                name: The name of the essence to find.

            Returns:
                The matching RSC_EssenceVariant, or an empty variant if the world is not
                running.
        )doc"
        )
        .def(
            "add_entity",
            &RSPY_WorldHandle::add_entity,
            py::arg("entity"),
            R"doc(
            Validates and adds an entity to the running world.

            Args:
                entity: The RSC_EntityVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the entity fails validation.
        )doc"
        )
        .def(
            "remove_entity",
            &RSPY_WorldHandle::remove_entity,
            py::arg("name"),
            R"doc(
            Removes the named entity from the running world.

            Args:
                name: The name of the entity to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "freeze_entity",
            &RSPY_WorldHandle::freeze_entity,
            py::arg("name"),
            R"doc(
            Freezes the named entity so it stops advancing.

            Args:
                name: The name of the entity to freeze.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "unfreeze_entity",
            &RSPY_WorldHandle::unfreeze_entity,
            py::arg("name"),
            R"doc(
            Unfreezes the named entity so it resumes advancing.

            Args:
                name: The name of the entity to unfreeze.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_entity",
            &RSPY_WorldHandle::find_entity,
            py::arg("name"),
            R"doc(
            Looks up the named entity in the running world.

            Args:
                name: The name of the entity to find.

            Returns:
                The matching RSC_EntityVariant, or an empty variant if the world is not
                running.
        )doc"
        )
        .def(
            "is_entity_frozen",
            &RSPY_WorldHandle::is_entity_frozen,
            py::arg("name"),
            R"doc(
            Reports whether the named entity is frozen.

            Args:
                name: The name of the entity to query.

            Returns:
                True if the entity is frozen, False if it is not or the world is not running.
        )doc"
        )
        .def(
            "add_record",
            &RSPY_WorldHandle::add_record,
            py::arg("record"),
            R"doc(
            Validates and adds a record to the running world.

            Args:
                record: The RSC_RecordVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the record fails validation.
        )doc"
        )
        .def(
            "remove_record",
            &RSPY_WorldHandle::remove_record,
            py::arg("name"),
            R"doc(
            Removes the named record from the running world.

            Args:
                name: The name of the record to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_record",
            &RSPY_WorldHandle::find_record,
            py::arg("name"),
            R"doc(
            Looks up the named record in the running world.

            Args:
                name: The name of the record to find.

            Returns:
                The matching RSC_RecordVariant, or an empty variant if the world is not running.
        )doc"
        )
        .def(
            "add_signal",
            &RSPY_WorldHandle::add_signal,
            py::arg("signal"),
            R"doc(
            Validates and adds a signal to the running world.

            Args:
                signal: The RSC_SignalVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the signal fails validation.
        )doc"
        )
        .def(
            "remove_signal",
            &RSPY_WorldHandle::remove_signal,
            py::arg("name"),
            R"doc(
            Removes the named signal from the running world.

            Args:
                name: The name of the signal to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_signal",
            &RSPY_WorldHandle::find_signal,
            py::arg("name"),
            R"doc(
            Looks up the named signal in the running world.

            Args:
                name: The name of the signal to find.

            Returns:
                The matching RSC_SignalVariant, or an empty variant if the world is not running.
        )doc"
        )
        .def(
            "add_recorder",
            &RSPY_WorldHandle::add_recorder,
            py::arg("recorder"),
            R"doc(
            Validates and adds a recorder to the running world.

            Args:
                recorder: The RSC_RecorderVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the recorder fails validation.
        )doc"
        )
        .def(
            "remove_recorder",
            &RSPY_WorldHandle::remove_recorder,
            py::arg("name"),
            R"doc(
            Removes the named recorder from the running world.

            Args:
                name: The name of the recorder to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_recorder",
            &RSPY_WorldHandle::find_recorder,
            py::arg("name"),
            R"doc(
            Looks up the named recorder in the running world.

            Args:
                name: The name of the recorder to find.

            Returns:
                The matching RSC_RecorderVariant, or an empty variant if the world is not
                running.
        )doc"
        )
        .def(
            "add_controller",
            &RSPY_WorldHandle::add_controller,
            py::arg("controller"),
            R"doc(
            Validates and adds a controller to the running world.

            Args:
                controller: The RSC_ControllerVariant to add.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.

            Raises:
                RS_InvalidCharacterizationException: When the controller fails validation.
        )doc"
        )
        .def(
            "remove_controller",
            &RSPY_WorldHandle::remove_controller,
            py::arg("name"),
            R"doc(
            Removes the named controller from the running world.

            Args:
                name: The name of the controller to remove.

            Returns:
                The message id for this mutation, or RS_NULL_INDEX if the world is not running.
        )doc"
        )
        .def(
            "find_controller",
            &RSPY_WorldHandle::find_controller,
            py::arg("name"),
            R"doc(
            Looks up the named controller in the running world.

            Args:
                name: The name of the controller to find.

            Returns:
                The matching RSC_ControllerVariant, or an empty variant if the world is not
                running.
        )doc"
        );

    /*****************************************************************
     * World callback
     *****************************************************************/

    py::class_<RSPY_WorldCallback>(m, "RS_WorldCallback")
        .def(
            "get_world_name",
            &RSPY_WorldCallback::get_world_name,
            R"doc(
            Returns the name of the world this callback is bound to.
        )doc"
        )
        .def(
            "set_callback",
            &RSPY_WorldCallback::set_callback,
            py::arg("callback"),
            R"doc(
            Sets the function the engine invokes each control step.

            Args:
                callback: A callable invoked as callback(time, controllers).
        )doc"
        )
        .def(
            "reset_callback",
            &RSPY_WorldCallback::reset_callback,
            R"doc(
            Clears the callback function so the engine invokes nothing.
        )doc"
        )
        .def(
            "get_callback",
            &RSPY_WorldCallback::get_callback,
            R"doc(
            Returns the currently set callback function, or None if none is set.
        )doc"
        );

    /*****************************************************************
     * World session components
     *****************************************************************/

    py::class_<RS_WorldSessionComponent, std::shared_ptr<RS_WorldSessionComponent>>(
        m, "RS_WorldSessionComponent"
    );

    py::class_<
        RS_WorldSessionComponentGeneric,
        RS_WorldSessionComponent,
        std::shared_ptr<RS_WorldSessionComponentGeneric>>(m, "RS_WorldSessionComponentGeneric")
        .def(
            py::init<>(),
            R"doc(
            A component that runs a user callback over the controllers in its scope each step.
        )doc"
        )
        .def(
            "get_handle",
            [](RS_WorldSessionComponentGeneric &component) {
                return RSPY_WorldHandle(component.getHandle());
            },
            R"doc(
            This component's scoped world handle.
        )doc"
        )
        .def(
            "set_callback",
            [](RS_WorldSessionComponentGeneric &component, py::function py_callback) {
                // The callback fires from a pool thread, so its Python object is held under a deleter
                // that takes the GIL, and each call acquires the GIL before invoking it.
                std::shared_ptr<py::object> py_object(
                    new py::object(std::move(py_callback)),
                    [](py::object *object) {
                        py::gil_scoped_acquire gil;
                        delete object;
                    }
                );
                component.setCallback(
                    [py_object](
                        RBigFloat time,
                        std::vector<std::reference_wrapper<RS_WorldController>> controllers
                    ) {
                        py::gil_scoped_acquire gil;
                        py::list controllers_list;
                        for (auto &controller : controllers) {
                            controllers_list.append(RSPY_WorldController(controller.get()));
                        }
                        try {
                            py::cast<py::function>(*py_object)(time, controllers_list);
                        } catch (const py::error_already_set &error) {
                            py::print("Exception in session callback:", error.what());
                        }
                    }
                );
            },
            py::arg("callback"),
            R"doc(
            Set the per-step callback, invoked as callback(time, controllers).
        )doc"
        );

    py::class_<
        RS_WorldSessionComponentCamera,
        RS_WorldSessionComponent,
        std::shared_ptr<RS_WorldSessionComponentCamera>>(m, "RS_WorldSessionComponentCamera")
        .def(
            py::init([](std::string name,
                        py::object callback,
                        RVec3rf position,
                        RVec3rf focus,
                        RSize width,
                        RSize height,
                        RSize samples_per_pixel,
                        RSize max_depth,
                        RFloat field_of_view,
                        RFloat defocus_angle,
                        RFloat focus_distance) {
                RS_WorldSessionComponentCamera::Callback cpp_callback;
                if (!callback.is_none()) {
                    std::shared_ptr<py::object> py_object(
                        new py::object(std::move(callback)),
                        [](py::object *object) {
                            py::gil_scoped_acquire gil;
                            delete object;
                        }
                    );
                    cpp_callback = [py_object](RBigFloat time) {
                        py::gil_scoped_acquire gil;
                        try {
                            py::cast<py::function>(*py_object)(time);
                        } catch (const py::error_already_set &error) {
                            py::print("Exception in camera callback:", error.what());
                        }
                    };
                }
                return std::make_shared<RS_WorldSessionComponentCamera>(
                    std::move(name),
                    std::move(cpp_callback),
                    position,
                    focus,
                    width,
                    height,
                    samples_per_pixel,
                    max_depth,
                    field_of_view,
                    defocus_angle,
                    focus_distance
                );
            }),
            py::arg("name"),
            py::arg("callback") = py::none(),
            py::arg("position") = RVec3rf(0.0, 0.0, 0.4),
            py::arg("focus") = RVec3rf(0.0, 0.0, 0.15),
            py::arg("width") = 1280,
            py::arg("height") = 720,
            py::arg("samples_per_pixel") = 4,
            py::arg("max_depth") = 3,
            py::arg("field_of_view") = 60.0,
            py::arg("defocus_angle") = 0.0,
            py::arg("focus_distance") = 1.0,
            R"doc(
            Orbit-camera component: injects one free perspective camera and drives its pose.
        )doc"
        )
        .def(
            "move",
            &RS_WorldSessionComponentCamera::move,
            py::arg("orbit_pitch"),
            py::arg("orbit_yaw"),
            py::arg("pan_x"),
            py::arg("pan_y"),
            py::arg("zoom"),
            R"doc(
            Accumulate a mouse gesture; applied on the next control step.
        )doc"
        )
        .def(
            "frame",
            [](const RS_WorldSessionComponentCamera &camera) -> py::object {
                RSize width = 0;
                RSize height = 0;
                std::shared_ptr<const RColor3b[]> rgb = camera.frame(width, height);
                if (!rgb || width == 0 || height == 0) return py::none();
                return py::array_t<uint8_t>(
                    {static_cast<size_t>(height), static_cast<size_t>(width), size_t(3)},
                    reinterpret_cast<const uint8_t *>(rgb.get())
                );
            },
            R"doc(
            The latest RGB frame as a copied (height, width, 3) uint8 array, or None if empty.
        )doc"
        );

#ifdef RISE_ORIGIN_BUILD_CORE_WITH_VR
    py::class_<
        RS_WorldSessionComponentVR,
        RS_WorldSessionComponent,
        std::shared_ptr<RS_WorldSessionComponentVR>>(m, "RS_WorldSessionComponentVR")
        .def(
            py::init([](std::string name,
                        RSize samples_per_pixel,
                        RSize max_depth,
                        RFloat near_depth,
                        RFloat far_depth,
                        RFloat move_speed,
                        RFloat fly_speed,
                        RFloat turn_speed,
                        RFloat stick_dead_zone,
                        RFloat boost,
                        RFloat hud_follow_tau) {
                return std::make_shared<RS_WorldSessionComponentVR>(
                    std::move(name),
                    samples_per_pixel,
                    max_depth,
                    near_depth,
                    far_depth,
                    move_speed,
                    fly_speed,
                    turn_speed,
                    stick_dead_zone,
                    boost,
                    hud_follow_tau
                );
            }),
            py::arg("name"),
            py::arg("samples_per_pixel") = 1,
            py::arg("max_depth") = 3,
            py::arg("near_depth") = 0.05,
            py::arg("far_depth") = 100.0,
            py::arg("move_speed") = 2.0,
            py::arg("fly_speed") = 1.5,
            py::arg("turn_speed") = 90.0,
            py::arg("stick_dead_zone") = 0.2,
            py::arg("boost") = 3.0,
            py::arg("hud_follow_tau") = 0.25,
            R"doc(
            Stereo VR component: injects one stereo camera and presents it to an OpenXR headset.
        )doc"
        )
        .def(
            "setPushForce",
            &RS_WorldSessionComponentVR::setPushForce,
            py::arg("push_force"),
            R"doc(
            Set the newtons applied along the aim ray at a full trigger pull.
        )doc"
        );
#endif  // RISE_ORIGIN_BUILD_CORE_WITH_VR

    /*****************************************************************
     * World session
     *****************************************************************/

    py::class_<RS_WorldSession>(m, "RS_WorldSession")
        .def(
            py::init([](RSPY_WorldHandle &handle, RSPY_WorldCallback &callback) {
                return std::make_unique<RS_WorldSession>(handle.handle, callback.callback);
            }),
            py::arg("handle"),
            py::arg("callback"),
            R"doc(
            Build a session over a running world's handle and callback.
        )doc"
        )
        .def(
            "has_component",
            &RS_WorldSession::hasComponent,
            py::arg("name"),
            R"doc(
            Whether a component is registered under the given name.
        )doc"
        )
        .def(
            "add_component",
            &RS_WorldSession::addComponent,
            py::arg("name"),
            py::arg("component"),
            R"doc(
            Attach a component under a name, giving it its own scoped access.
        )doc"
        )
        .def(
            "remove_component",
            &RS_WorldSession::removeComponent,
            py::arg("name"),
            R"doc(
            Detach and remove the named component.
        )doc"
        )
        .def(
            "get_core_component",
            &RS_WorldSession::getCoreComponent,
            R"doc(
            The Core component, which owns the initial world and carries the user callback.
        )doc"
        )
        .def(
            "get_handle",
            [](RS_WorldSession &session) { return RSPY_WorldHandle(session.getHandle()); },
            R"doc(
            A fresh scoped handle to drive lifecycle (terminate, wait_for_end, ...).
        )doc"
        )
        .def(
            "run",
            &RS_WorldSession::run,
            py::arg("is_blocking") = true,
            py::call_guard<py::gil_scoped_release>(),
            R"doc(
            When blocking, wait until the world ends and then detach all components; otherwise
            return at once and leave teardown to the destructor.
        )doc"
        );
}

#endif  // INIT_CORE_H
