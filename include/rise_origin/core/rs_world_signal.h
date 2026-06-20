// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_SIGNAL_H
#define RS_WORLD_SIGNAL_H

#include "rise_origin/utils/rs_conf.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/memory/rs_device.h"
#include "rise_origin/utils/math/rs_quat.h"
#include "rise_origin/utils/math/rs_vec.h"

namespace rise {

namespace world_signal_helper {

static constexpr RBigSize SIM_SIGNAL_ALIGN_SIZE = RS_DEVICE_MEMORY_ALIGN_SIZE;

template <typename SizeType>
HOST_DEVICE_INLINE_QUALIFIER RBigSize align(SizeType needed_size) {
    return RS_ALIGN(needed_size, SIM_SIGNAL_ALIGN_SIZE);
}

template <typename T1, typename T2, typename SizeType>
HOST_DEVICE_INLINE_QUALIFIER void setPtr(T1*& current, T2* previous_start, SizeType previous_size) {
    current = reinterpret_cast<T1*>(
        reinterpret_cast<std::byte*>(previous_start) +
        RS_ALIGN(previous_size, SIM_SIGNAL_ALIGN_SIZE)
    );
}

}  // namespace world_signal_helper

class RS_WorldObjectSignal {
   public:
    void* root = nullptr;

    RBigSize object_num = 0;

    RS_WorldObjectSignal() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
        RBigSize object_num
    ) {
        return 0;
    }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
        this->object_num = object_num;
        this->root = memory;
        return memory;
    }
};

class RS_WorldRootSignal {
   public:
    void* root = nullptr;

    /// Device location of the root signal
    RS_DeviceLocation device = {};

    /// Whether the root signal is valid
    bool is_valid = false;

    RS_WorldRootSignal() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER bool isValid() const { return is_valid; }

    [[nodiscard]] static HOST_DEVICE_INLINE_QUALIFIER RBigSize getRequiredMemorySize() { return 0; }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory) {
        root = memory;
        is_valid = true;
        return memory;
    }
};

class RS_WorldStructureSignalBase : public RS_WorldRootSignal {
   public:
    /**
     * Expansion signals applied to voxels of a structure.
     * One scalar per signal index; voxels select an entry through their
     * own per-voxel expansion_signal_sid.
     */
    struct Expansion : public RS_WorldObjectSignal {
        /// Scalar expansion signal per signal entry
        RFloat* expansion = nullptr;

        Expansion() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_signal_helper;
            return align(object_num * sizeof(RFloat));  // expansion
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_signal_helper;

            RS_WorldObjectSignal::initFromMemory(memory, object_num);

            setPtr(expansion, memory, 0);

            void* end;
            setPtr(end, expansion, object_num * sizeof(RFloat));
            return end;
        }
    };

    /**
     * Hinge signals applied to hinge motor joints of a structure.
     * One scalar per signal index; hinge joints select an entry through
     * their own per-joint hinge_signal_sid.
     */
    struct Hinge : public RS_WorldObjectSignal {
        /// Scalar hinge signal per signal entry
        RFloat* hinge = nullptr;

        Hinge() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_signal_helper;
            return align(object_num * sizeof(RFloat));  // hinge
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_signal_helper;

            RS_WorldObjectSignal::initFromMemory(memory, object_num);

            setPtr(hinge, memory, 0);

            void* end;
            setPtr(end, hinge, object_num * sizeof(RFloat));
            return end;
        }
    };

    /**
     * Prismatic signals applied to prismatic motor joints of a structure.
     * One scalar per signal index; prismatic joints select an entry
     * through their own per-joint prismatic_signal_sid.
     */
    struct Prismatic : public RS_WorldObjectSignal {
        /// Scalar prismatic signal per signal entry
        RFloat* prismatic = nullptr;

        Prismatic() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_signal_helper;
            return align(object_num * sizeof(RFloat));  // prismatic
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_signal_helper;

            RS_WorldObjectSignal::initFromMemory(memory, object_num);

            setPtr(prismatic, memory, 0);

            void* end;
            setPtr(end, prismatic, object_num * sizeof(RFloat));
            return end;
        }
    };
};

class RS_WorldStructureVoxelSignal : public RS_WorldStructureSignalBase {
   public:
    /// Structure global id
    RIndex structure_gid = RS_NULL_INDEX;

    /// Expansion signals, one entry per signal index
    Expansion expansion_signals;

    /// Hinge signals, one entry per signal index
    Hinge hinge_signals;

    /// Prismatic signals, one entry per signal index
    Prismatic prismatic_signals;

    RS_WorldStructureVoxelSignal() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
        RBigSize expansion_num,
        RBigSize hinge_num,
        RBigSize prismatic_num
    ) {
        RBigSize base_size = RS_WorldRootSignal::getRequiredMemorySize();
        RBigSize expansion_size = Expansion::getRequiredMemorySize(expansion_num);
        RBigSize hinge_size = Hinge::getRequiredMemorySize(hinge_num);
        RBigSize prismatic_size =
            Prismatic::getRequiredMemorySize(prismatic_num);

        RBigSize total_size =
            base_size + expansion_size + hinge_size + prismatic_size;
#ifdef RS_DEBUG_ENABLE_WORLD_SIGNAL
        printf(
            "Required memory sizes for voxel structure signal:\n  Base: %u\n  Expansion: %u\n "
            " Hinge: %u\n  Prismatic: %u\nTotal required size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(expansion_size),
            static_cast<unsigned int>(hinge_size),
            static_cast<unsigned int>(prismatic_size),
            static_cast<unsigned int>(total_size)
        );
#endif

        return total_size;
    }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(
        void* memory,
        RBigSize expansion_num,
        RBigSize hinge_num,
        RBigSize prismatic_num
    ) {
        is_valid = true;

#ifdef RS_DEBUG_ENABLE_WORLD_SIGNAL
        void* start_memory = memory;
        void* base_start = memory;
        memory = RS_WorldRootSignal::initFromMemory(memory);
        RBigSize base_size = static_cast<char*>(memory) - static_cast<char*>(base_start);

        void* expansion_start = memory;
        memory = expansion_signals.initFromMemory(memory, expansion_num);
        RBigSize expansion_size =
            static_cast<char*>(memory) - static_cast<char*>(expansion_start);

        void* hinge_start = memory;
        memory = hinge_signals.initFromMemory(memory, hinge_num);
        RBigSize hinge_size =
            static_cast<char*>(memory) - static_cast<char*>(hinge_start);

        void* prismatic_start = memory;
        memory = prismatic_signals.initFromMemory(memory, prismatic_num);
        RBigSize prismatic_size =
            static_cast<char*>(memory) - static_cast<char*>(prismatic_start);

        RBigSize total_size = static_cast<char*>(memory) - static_cast<char*>(start_memory);

        printf(
            "Occupied memory sizes for voxel structure signal:\n  Base: %u\n  Expansion: %u\n "
            " Hinge: %u\n  Prismatic: %u\nTotal occupied size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(expansion_size),
            static_cast<unsigned int>(hinge_size),
            static_cast<unsigned int>(prismatic_size),
            static_cast<unsigned int>(total_size)
        );
#else
        memory = RS_WorldRootSignal::initFromMemory(memory);
        memory = expansion_signals.initFromMemory(memory, expansion_num);
        memory = hinge_signals.initFromMemory(memory, hinge_num);
        memory = prismatic_signals.initFromMemory(memory, prismatic_num);
#endif

        return memory;
    }
};

class RS_WorldCameraSignalBase : public RS_WorldRootSignal {
   public:
    /**
     * Camera position signals (3D vector position).
     */
    struct Position : public RS_WorldObjectSignal {
        /// Camera position per signal entry
        RVec3rf* position = nullptr;

        Position() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num = 1
        ) {
            using namespace world_signal_helper;
            return align(object_num * sizeof(RVec3rf));  // position
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num = 1) {
            using namespace world_signal_helper;

            RS_WorldObjectSignal::initFromMemory(memory, object_num);

            setPtr(position, memory, 0);

            void* end;
            setPtr(end, position, object_num * sizeof(RVec3rf));
            return end;
        }
    };

    /**
     * Camera orientation signals (4D quaternion).
     */
    struct Orientation : public RS_WorldObjectSignal {
        /// Camera orientation per signal entry
        RQuat3rf* orientation = nullptr;

        Orientation() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num = 1
        ) {
            using namespace world_signal_helper;
            return align(object_num * sizeof(RQuat3rf));  // orientation
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num = 1) {
            using namespace world_signal_helper;

            RS_WorldObjectSignal::initFromMemory(memory, object_num);

            setPtr(orientation, memory, 0);

            void* end;
            setPtr(end, orientation, object_num * sizeof(RQuat3rf));
            return end;
        }
    };

    /**
     * Camera optics signals: scalar focus distance and field of view.
     */
    struct Optics : public RS_WorldObjectSignal {
        /// Scalar focus distance per signal entry
        RFloat* focus_distance = nullptr;

        /// Scalar field of view per signal entry
        RFloat* field_of_view = nullptr;

        Optics() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num = 1
        ) {
            using namespace world_signal_helper;
            return align(object_num * sizeof(RFloat)) +  // focus_distance
                   align(object_num * sizeof(RFloat));   // field_of_view
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num = 1) {
            using namespace world_signal_helper;

            RS_WorldObjectSignal::initFromMemory(memory, object_num);

            setPtr(focus_distance, memory, 0);
            setPtr(field_of_view, focus_distance, object_num * sizeof(RFloat));

            void* end;
            setPtr(end, field_of_view, object_num * sizeof(RFloat));
            return end;
        }
    };
};

class RS_WorldCameraPerspectiveSignal : public RS_WorldCameraSignalBase {
   public:
    /// Camera global id
    RIndex camera_gid = RS_NULL_INDEX;

    /// Camera position signals
    Position position_signals;

    /// Camera orientation signals
    Orientation orientation_signals;

    /// Camera optics signals (focus distance and field of view)
    Optics optics_signals;

    RS_WorldCameraPerspectiveSignal() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
        RBigSize position_num = 1,
        RBigSize orientation_num = 1,
        RBigSize optics_num = 1
    ) {
        RBigSize base_size = RS_WorldRootSignal::getRequiredMemorySize();
        RBigSize position_size = Position::getRequiredMemorySize(position_num);
        RBigSize orientation_size = Orientation::getRequiredMemorySize(orientation_num);
        RBigSize optics_size = Optics::getRequiredMemorySize(optics_num);

        RBigSize total_size = base_size + position_size + orientation_size + optics_size;
#ifdef RS_DEBUG_ENABLE_WORLD_SIGNAL
        printf(
            "Required memory sizes for perspective camera signal:\n  Base: %u\n  Position: %u\n  "
            "Orientation: %u\n  Optics: %u\nTotal required size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(position_size),
            static_cast<unsigned int>(orientation_size),
            static_cast<unsigned int>(optics_size),
            static_cast<unsigned int>(total_size)
        );
#endif

        return total_size;
    }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(
        void* memory,
        RBigSize position_num = 1,
        RBigSize orientation_num = 1,
        RBigSize optics_num = 1
    ) {
        is_valid = true;

#ifdef RS_DEBUG_ENABLE_WORLD_SIGNAL
        void* start_memory = memory;
        void* base_start = memory;
        memory = RS_WorldRootSignal::initFromMemory(memory);
        RBigSize base_size = static_cast<char*>(memory) - static_cast<char*>(base_start);

        void* position_start = memory;
        memory = position_signals.initFromMemory(memory, position_num);
        RBigSize position_size = static_cast<char*>(memory) - static_cast<char*>(position_start);

        void* orientation_start = memory;
        memory = orientation_signals.initFromMemory(memory, orientation_num);
        RBigSize orientation_size =
            static_cast<char*>(memory) - static_cast<char*>(orientation_start);

        void* optics_start = memory;
        memory = optics_signals.initFromMemory(memory, optics_num);
        RBigSize optics_size = static_cast<char*>(memory) - static_cast<char*>(optics_start);

        RBigSize total_size = static_cast<char*>(memory) - static_cast<char*>(start_memory);

        printf(
            "Occupied memory sizes for perspective camera signal:\n  Base: %u\n  Position: %u\n  "
            "Orientation: %u\n  Optics: %u\nTotal occupied size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(position_size),
            static_cast<unsigned int>(orientation_size),
            static_cast<unsigned int>(optics_size),
            static_cast<unsigned int>(total_size)
        );
#else
        memory = RS_WorldRootSignal::initFromMemory(memory);
        memory = position_signals.initFromMemory(memory, position_num);
        memory = orientation_signals.initFromMemory(memory, orientation_num);
        memory = optics_signals.initFromMemory(memory, optics_num);
#endif

        return memory;
    }
};

class RS_WorldForceSignalBase : public RS_WorldRootSignal {
   public:
    /**
     * Force signals applied to a target (rigid body or soft voxel range).
     * One 3D force vector per signal entry; force memories select an entry
     * through their own per-object force_signal handle (currently always 0,
     * i.e. one entry per controller).
     */
    struct Force : public RS_WorldObjectSignal {
        /// 3D force vector per signal entry
        RVec3rf* force = nullptr;

        Force() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_signal_helper;
            return align(object_num * sizeof(RVec3rf));  // force
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_signal_helper;

            RS_WorldObjectSignal::initFromMemory(memory, object_num);

            setPtr(force, memory, 0);

            void* end;
            setPtr(end, force, object_num * sizeof(RVec3rf));
            return end;
        }
    };
};

class RS_WorldForceSignal : public RS_WorldForceSignalBase {
   public:
    /// Structure global id
    RIndex structure_gid = RS_NULL_INDEX;

    /// Force signals, one 3D vector per signal index
    Force force_signals;

    RS_WorldForceSignal() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
        RBigSize force_num = 1
    ) {
        RBigSize base_size = RS_WorldRootSignal::getRequiredMemorySize();
        RBigSize force_size = Force::getRequiredMemorySize(force_num);
        RBigSize total_size = base_size + force_size;
#ifdef RS_DEBUG_ENABLE_WORLD_SIGNAL
        printf(
            "Required memory sizes for force signal:\n  Base: %u\n  Force: %u\nTotal required "
            "size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(force_size),
            static_cast<unsigned int>(total_size)
        );
#endif

        return total_size;
    }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize force_num = 1) {
        is_valid = true;

#ifdef RS_DEBUG_ENABLE_WORLD_SIGNAL
        void* start_memory = memory;
        void* base_start = memory;
        memory = RS_WorldRootSignal::initFromMemory(memory);
        RBigSize base_size = static_cast<char*>(memory) - static_cast<char*>(base_start);

        void* force_start = memory;
        memory = force_signals.initFromMemory(memory, force_num);
        RBigSize force_size = static_cast<char*>(memory) - static_cast<char*>(force_start);

        RBigSize total_size = static_cast<char*>(memory) - static_cast<char*>(start_memory);

        printf(
            "Occupied memory sizes for force signal:\n  Base: %u\n  Force: %u\nTotal occupied "
            "size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(force_size),
            static_cast<unsigned int>(total_size)
        );
#else
        memory = RS_WorldRootSignal::initFromMemory(memory);
        memory = force_signals.initFromMemory(memory, force_num);
#endif

        return memory;
    }
};

class RS_WorldTorqueSignalBase : public RS_WorldRootSignal {
   public:
    /**
     * Torque signals applied to a rigid body target.
     * One 3D torque vector per signal entry; torque memories select an entry
     * through their own per-object torque_signal handle (currently always 0,
     * i.e. one entry per controller).
     */
    struct Torque : public RS_WorldObjectSignal {
        /// 3D torque vector per signal entry
        RVec3rf* torque = nullptr;

        Torque() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_signal_helper;
            return align(object_num * sizeof(RVec3rf));  // torque
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_signal_helper;

            RS_WorldObjectSignal::initFromMemory(memory, object_num);

            setPtr(torque, memory, 0);

            void* end;
            setPtr(end, torque, object_num * sizeof(RVec3rf));
            return end;
        }
    };
};

class RS_WorldTorqueSignal : public RS_WorldTorqueSignalBase {
   public:
    /// Structure global id
    RIndex structure_gid = RS_NULL_INDEX;

    /// Torque signals, one 3D vector per signal index
    Torque torque_signals;

    RS_WorldTorqueSignal() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
        RBigSize torque_num = 1
    ) {
        RBigSize base_size = RS_WorldRootSignal::getRequiredMemorySize();
        RBigSize torque_size = Torque::getRequiredMemorySize(torque_num);
        RBigSize total_size = base_size + torque_size;
#ifdef RS_DEBUG_ENABLE_WORLD_SIGNAL
        printf(
            "Required memory sizes for torque signal:\n  Base: %u\n  Torque: %u\nTotal required "
            "size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(torque_size),
            static_cast<unsigned int>(total_size)
        );
#endif

        return total_size;
    }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize torque_num = 1) {
        is_valid = true;

#ifdef RS_DEBUG_ENABLE_WORLD_SIGNAL
        void* start_memory = memory;
        void* base_start = memory;
        memory = RS_WorldRootSignal::initFromMemory(memory);
        RBigSize base_size = static_cast<char*>(memory) - static_cast<char*>(base_start);

        void* torque_start = memory;
        memory = torque_signals.initFromMemory(memory, torque_num);
        RBigSize torque_size = static_cast<char*>(memory) - static_cast<char*>(torque_start);

        RBigSize total_size = static_cast<char*>(memory) - static_cast<char*>(start_memory);

        printf(
            "Occupied memory sizes for torque signal:\n  Base: %u\n  Torque: %u\nTotal occupied "
            "size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(torque_size),
            static_cast<unsigned int>(total_size)
        );
#else
        memory = RS_WorldRootSignal::initFromMemory(memory);
        memory = torque_signals.initFromMemory(memory, torque_num);
#endif

        return memory;
    }
};

}  // namespace rise

#endif  // RS_WORLD_SIGNAL_H
