// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef INIT_CORE_H
#define INIT_CORE_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

#include "rise_origin/core/rs_world_controller.h"
#include "rise_origin/core/rs_world_def.h"
#include "rise_origin/core/rs_world_handle.h"
#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/memory/rs_device.h"

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
        [[nodiscard]] py::array_t<uint8_t> pixels() const {
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
            return rspyRwArray2D(
                reinterpret_cast<RFloat *>(signal.force_signals.force), size(), 3
            );
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
                } else if constexpr (std::is_same_v<T, RS_WorldForceSignal>) {
                    result.append(py::cast(RSPY_WorldForceSignal(held)));
                } else if constexpr (std::is_same_v<T, RS_WorldTorqueSignal>) {
                    result.append(py::cast(RSPY_WorldTorqueSignal(held)));
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
    explicit RSPY_WorldHandle(std::shared_ptr<RS_WorldHandle> handle)
        : handle(std::move(handle)) {}

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
        .value(
            "RSE_WORLD_END_FAILED_WITH_INVALID_CONFIG",
            RSE_WORLD_END_FAILED_WITH_INVALID_CONFIG
        )
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
        .def(
            "prismatic_signal_sid",
            &RSPY_WorldStructureVoxelRecord::Joint::prismatic_signal_sid
        )
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
            .def_property_readonly("state", &RSPY_WorldCameraPerspectiveRecord::state);

    py::class_<RSPY_WorldCameraPerspectiveRecord::RGB>(camera_perspective_record, "RGB")
        .def("width", &RSPY_WorldCameraPerspectiveRecord::RGB::width)
        .def("height", &RSPY_WorldCameraPerspectiveRecord::RGB::height)
        .def("pixels", &RSPY_WorldCameraPerspectiveRecord::RGB::pixels);

    py::class_<RSPY_WorldCameraPerspectiveRecord::State>(camera_perspective_record, "State")
        .def("size", &RSPY_WorldCameraPerspectiveRecord::State::size)
        .def("position", &RSPY_WorldCameraPerspectiveRecord::State::position)
        .def("orientation", &RSPY_WorldCameraPerspectiveRecord::State::orientation)
        .def("focus_distance", &RSPY_WorldCameraPerspectiveRecord::State::focus_distance)
        .def("field_of_view", &RSPY_WorldCameraPerspectiveRecord::State::field_of_view);

    auto intersect_region_record =
        py::class_<RSPY_WorldIntersectRegionRecord>(m, "RS_WorldIntersectRegionRecord")
            .def("step", &RSPY_WorldIntersectRegionRecord::step)
            .def("time", &RSPY_WorldIntersectRegionRecord::time)
            .def("record_gid", &RSPY_WorldIntersectRegionRecord::record_gid)
            .def_property_readonly(
                "memberships", &RSPY_WorldIntersectRegionRecord::memberships
            );

    py::class_<RSPY_WorldIntersectRegionRecord::Membership>(
        intersect_region_record, "Membership"
    )
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
        .def("is_pending", &RSPY_WorldHandle::is_pending)
        .def("is_running", &RSPY_WorldHandle::is_running)
        .def("is_paused", &RSPY_WorldHandle::is_paused)
        .def("is_end", &RSPY_WorldHandle::is_end)
        .def("pause", &RSPY_WorldHandle::pause)
        .def("resume", &RSPY_WorldHandle::resume)
        .def("terminate", &RSPY_WorldHandle::terminate)
        .def("get_status", &RSPY_WorldHandle::get_status)
        .def("wait_for_running", &RSPY_WorldHandle::wait_for_running)
        .def("wait_for_end", &RSPY_WorldHandle::wait_for_end)
        .def(
            "wait_for_message",
            &RSPY_WorldHandle::wait_for_message,
            py::arg("message_id") = RS_NULL_INDEX,
            py::arg("level") = RSE_WORLD_MESSAGE_ANY_LEVEL
        )
        .def(
            "begin_modification_transaction",
            &RSPY_WorldHandle::begin_modification_transaction
        )
        .def("end_modification_transaction", &RSPY_WorldHandle::end_modification_transaction)
        .def("get_world", &RSPY_WorldHandle::get_world)
        .def("set_simulation", &RSPY_WorldHandle::set_simulation)
        .def("add_resource", &RSPY_WorldHandle::add_resource)
        .def("remove_resource", &RSPY_WorldHandle::remove_resource)
        .def("find_resource", &RSPY_WorldHandle::find_resource)
        .def("add_essence", &RSPY_WorldHandle::add_essence)
        .def("update_essence", &RSPY_WorldHandle::update_essence)
        .def("remove_essence", &RSPY_WorldHandle::remove_essence)
        .def("find_essence", &RSPY_WorldHandle::find_essence)
        .def("add_entity", &RSPY_WorldHandle::add_entity)
        .def("remove_entity", &RSPY_WorldHandle::remove_entity)
        .def("freeze_entity", &RSPY_WorldHandle::freeze_entity)
        .def("unfreeze_entity", &RSPY_WorldHandle::unfreeze_entity)
        .def("find_entity", &RSPY_WorldHandle::find_entity)
        .def("is_entity_frozen", &RSPY_WorldHandle::is_entity_frozen)
        .def("add_record", &RSPY_WorldHandle::add_record)
        .def("remove_record", &RSPY_WorldHandle::remove_record)
        .def("find_record", &RSPY_WorldHandle::find_record)
        .def("add_signal", &RSPY_WorldHandle::add_signal)
        .def("remove_signal", &RSPY_WorldHandle::remove_signal)
        .def("find_signal", &RSPY_WorldHandle::find_signal)
        .def("add_recorder", &RSPY_WorldHandle::add_recorder)
        .def("remove_recorder", &RSPY_WorldHandle::remove_recorder)
        .def("find_recorder", &RSPY_WorldHandle::find_recorder)
        .def("add_controller", &RSPY_WorldHandle::add_controller)
        .def("remove_controller", &RSPY_WorldHandle::remove_controller)
        .def("find_controller", &RSPY_WorldHandle::find_controller);
}

#endif  // INIT_CORE_H
