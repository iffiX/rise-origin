// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RSPY_CORE_INIT_WORLD_RECORD_H
#define RSPY_CORE_INIT_WORLD_RECORD_H

#include "init_common.h"

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
            return py::array_t<float>(
                {h, w}, {w * sizeof(float), sizeof(float)}, record.depth.pixel
            );
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

class RSPY_WorldRaySensorRecord {
   public:
    RS_WorldRaySensorRecord record;
    explicit RSPY_WorldRaySensorRecord(const RS_WorldRaySensorRecord &rec) : record(rec) {}

    class Hit {
       public:
        RS_WorldRaySensorRecord record;
        explicit Hit(const RS_WorldRaySensorRecord &rec) : record(rec) {}

        [[nodiscard]] RSize size() const { return record.ray_num; }

        /// Hit flag per ray as a [ray_num] uint8 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<uint8_t> is_hit() const {
            if (record.hit.object_num == 0 || !record.isValid()) return py::array_t<uint8_t>(0);
            return rspyRoArray1D(reinterpret_cast<uint8_t *>(record.hit.is_hit), record.ray_num);
        }

        /// Hit distance per ray as a [ray_num] float ndarray, empty when disabled.
        [[nodiscard]] py::array_t<RFloat> distance() const {
            if (record.hit.object_num == 0 || !record.isValid()) return py::array_t<RFloat>(0);
            return rspyRoArray1D(record.hit.distance, record.ray_num);
        }

        /// Hit position per ray as a [ray_num, 3] float ndarray, empty when disabled.
        [[nodiscard]] py::array_t<RFloat> hit_position() const {
            if (record.hit.object_num == 0 || !record.isValid()) return py::array_t<RFloat>(0);
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.hit.hit_position), record.ray_num, 3
            );
        }
    };

    class EntityInfo {
       public:
        RS_WorldRaySensorRecord record;
        explicit EntityInfo(const RS_WorldRaySensorRecord &rec) : record(rec) {}

        [[nodiscard]] RSize size() const { return record.ray_num; }

        /// Hit entity global id per ray as a [ray_num] uint32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<uint32_t> entity_gid() const {
            if (record.entity_info.object_num == 0 || !record.isValid())
                return py::array_t<uint32_t>(0);
            return rspyRoArray1D(
                reinterpret_cast<uint32_t *>(record.entity_info.entity_gid), record.ray_num
            );
        }

        /// Hit entity type per ray as a [ray_num] uint32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<uint32_t> entity_type() const {
            if (record.entity_info.object_num == 0 || !record.isValid())
                return py::array_t<uint32_t>(0);
            return rspyRoArray1D(
                reinterpret_cast<uint32_t *>(record.entity_info.entity_type), record.ray_num
            );
        }
    };

    class EntityDetail {
       public:
        RS_WorldRaySensorRecord record;
        explicit EntityDetail(const RS_WorldRaySensorRecord &rec) : record(rec) {}

        [[nodiscard]] RSize size() const { return record.ray_num; }

        /// Hit body id per ray as a [ray_num] uint32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<uint32_t> body_sid() const {
            if (record.entity_detail.object_num == 0 || !record.isValid())
                return py::array_t<uint32_t>(0);
            return rspyRoArray1D(
                reinterpret_cast<uint32_t *>(record.entity_detail.body_sid), record.ray_num
            );
        }

        /// Hit segment id per ray as a [ray_num] uint32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<uint32_t> segment_bid() const {
            if (record.entity_detail.object_num == 0 || !record.isValid())
                return py::array_t<uint32_t>(0);
            return rspyRoArray1D(
                reinterpret_cast<uint32_t *>(record.entity_detail.segment_bid), record.ray_num
            );
        }

        /// Hit voxel id per ray as a [ray_num] uint32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<uint32_t> voxel_sid() const {
            if (record.entity_detail.object_num == 0 || !record.isValid())
                return py::array_t<uint32_t>(0);
            return rspyRoArray1D(
                reinterpret_cast<uint32_t *>(record.entity_detail.voxel_sid), record.ray_num
            );
        }
    };

    class Material {
       public:
        RS_WorldRaySensorRecord record;
        explicit Material(const RS_WorldRaySensorRecord &rec) : record(rec) {}

        [[nodiscard]] RSize size() const { return record.ray_num; }

        /// Hit material id per ray as a [ray_num] uint32 ndarray, empty when disabled.
        [[nodiscard]] py::array_t<uint32_t> material_id() const {
            if (record.material.object_num == 0 || !record.isValid())
                return py::array_t<uint32_t>(0);
            return rspyRoArray1D(
                reinterpret_cast<uint32_t *>(record.material.material_id), record.ray_num
            );
        }

        /// Hit material emission per ray as a [ray_num] float ndarray, empty when disabled.
        [[nodiscard]] py::array_t<RFloat> emission() const {
            if (record.material.object_num == 0 || !record.isValid()) return py::array_t<RFloat>(0);
            return rspyRoArray1D(record.material.emission, record.ray_num);
        }
    };

    class Normal {
       public:
        RS_WorldRaySensorRecord record;
        explicit Normal(const RS_WorldRaySensorRecord &rec) : record(rec) {}

        [[nodiscard]] RSize size() const { return record.ray_num; }

        /// Hit normal per ray as a [ray_num, 3] float ndarray, empty when disabled.
        [[nodiscard]] py::array_t<RFloat> normal() const {
            if (record.normal.object_num == 0 || !record.isValid()) return py::array_t<RFloat>(0);
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.normal.normal), record.ray_num, 3
            );
        }
    };

    class Velocity {
       public:
        RS_WorldRaySensorRecord record;
        explicit Velocity(const RS_WorldRaySensorRecord &rec) : record(rec) {}

        [[nodiscard]] RSize size() const { return record.ray_num; }

        /// Hit world velocity per ray as a [ray_num, 3] float ndarray, empty when disabled.
        [[nodiscard]] py::array_t<RFloat> v_hit() const {
            if (record.velocity.object_num == 0 || !record.isValid()) return py::array_t<RFloat>(0);
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.velocity.v_hit), record.ray_num, 3
            );
        }

        /// Relative velocity per ray as a [ray_num, 3] float ndarray, empty when disabled.
        [[nodiscard]] py::array_t<RFloat> v_rel() const {
            if (record.velocity.object_num == 0 || !record.isValid()) return py::array_t<RFloat>(0);
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.velocity.v_rel), record.ray_num, 3
            );
        }

        /// Radial velocity per ray as a [ray_num] float ndarray, empty when disabled.
        [[nodiscard]] py::array_t<RFloat> v_radial() const {
            if (record.velocity.object_num == 0 || !record.isValid()) return py::array_t<RFloat>(0);
            return rspyRoArray1D(record.velocity.v_radial, record.ray_num);
        }
    };

    class State {
       public:
        RS_WorldRaySensorRecord record;
        explicit State(const RS_WorldRaySensorRecord &rec) : record(rec) {}

        [[nodiscard]] RBigSize size() const { return record.sensor_state.object_num; }

        [[nodiscard]] py::array_t<RFloat> position() const {
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.sensor_state.position), size(), 3
            );
        }
        [[nodiscard]] py::array_t<RFloat> orientation() const {
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.sensor_state.orientation), size(), 4
            );
        }
    };

    [[nodiscard]] RBigIndex step() const { return record.getStep(); }
    [[nodiscard]] RBigFloat time() const { return record.getTime(); }
    [[nodiscard]] RIndex sensor_gid() const { return record.sensor_gid; }
    [[nodiscard]] RIndex structure_gid() const { return record.structure_gid; }
    [[nodiscard]] RIndex rigid_body_sid() const { return record.rigid_body_sid; }
    [[nodiscard]] RSize ray_num() const { return record.ray_num; }
    [[nodiscard]] bool is_valid() const { return record.isValid(); }

    [[nodiscard]] Hit hit() const { return Hit(record); }
    [[nodiscard]] EntityInfo entity_info() const { return EntityInfo(record); }
    [[nodiscard]] EntityDetail entity_detail() const { return EntityDetail(record); }
    [[nodiscard]] Material material() const { return Material(record); }
    [[nodiscard]] Normal normal() const { return Normal(record); }
    [[nodiscard]] Velocity velocity() const { return Velocity(record); }
    [[nodiscard]] State state() const { return State(record); }
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

        [[nodiscard]] py::array_t<RFloat> closest_relative_position() const {
            return rspyRoArray2D(
                reinterpret_cast<RFloat *>(record.memberships.closest_relative_position), size(), 3
            );
        }
    };

    [[nodiscard]] RBigIndex step() const { return record.getStep(); }
    [[nodiscard]] RBigFloat time() const { return record.getTime(); }
    [[nodiscard]] RIndex record_gid() const { return record.record_gid; }

    [[nodiscard]] Membership memberships() const { return Membership(record); }
};

/// Bind the runtime world record snapshot types.
void initRSPYWorldRecord(py::module &m);

#endif  // RSPY_CORE_INIT_WORLD_RECORD_H
