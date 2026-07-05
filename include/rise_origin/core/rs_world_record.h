// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_WORLD_RECORD_H
#define RS_WORLD_RECORD_H
#include <cassert>

#include "rise_origin/core/rs_world_def.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/math/rs_quat.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/memory/rs_device.h"
#include "rise_origin/utils/rs_conf.h"
#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_exception.h"

/**
 * @note Naming rule for object level member instances under each root:
 * A member is plural when each row is an independently addressable object indexed
 * by its own id, such as voxels, links, rigid_bodies, joints, and memberships. A
 * member is singular when it is one dense buffer or one value, such as rgb, depth,
 * normal, motion, entity_info, and camera_state. Subrecord types and inner column
 * pointers are always singular.
 */

namespace rise {

namespace world_record_helper {

static constexpr RBigSize SIM_RECORD_ALIGN_SIZE = RS_DEVICE_MEMORY_ALIGN_SIZE;

template <typename SizeType>
HOST_DEVICE_INLINE_QUALIFIER RBigSize align(SizeType needed_size) {
    return RS_ALIGN(needed_size, SIM_RECORD_ALIGN_SIZE);
}

template <typename T1, typename T2, typename SizeType>
HOST_DEVICE_INLINE_QUALIFIER void setPtr(T1*& current, T2* previous_start, SizeType previous_size) {
    current = reinterpret_cast<T1*>(
        reinterpret_cast<RByte*>(previous_start) + RS_ALIGN(previous_size, SIM_RECORD_ALIGN_SIZE)
    );
}

}  // namespace world_record_helper

class RS_WorldObjectRecord {
   public:
    void* root = nullptr;

    RBigSize object_num = 0;

    RS_WorldObjectRecord() = default;

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

class RS_WorldRootRecord {
   public:
    void* root = nullptr;

    /// Step of the root record
    RBigIndex* step = nullptr;

    /// Time of the root record
    RBigFloat* time = nullptr;

    /// Device location of the root record
    RS_DeviceLocation device = {};

    /// Whether the root record is valid
    bool is_valid = false;

    RS_WorldRootRecord() = default;

    HOST_DEVICE_INLINE_QUALIFIER RS_WorldRootRecord(
        void* root,
        RBigIndex* step,
        RBigFloat* time,
        const RS_DeviceLocation& device,
        bool is_valid
    )
        : step(step), time(time), device(device), is_valid(is_valid) {}

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER bool isValid() const { return is_valid; }

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER RBigIndex getStep() const {
        return step == nullptr ? 0 : *step;
    }

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER RBigFloat getTime() const {
        return time == nullptr ? 0 : *time;
    }

    [[nodiscard]] static HOST_DEVICE_INLINE_QUALIFIER RBigSize getRequiredMemorySize() {
        return sizeof(RBigIndex) + sizeof(RBigFloat);
    }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory) {
        step = reinterpret_cast<RBigIndex*>(memory);
        time = reinterpret_cast<RBigFloat*>(step + 1);
        is_valid = true;
        return time + 1;
    }
};

/**
 * Design Notes:
 * All RIndex indices are non-decreasing indicies. As simulation proceeds, indices
 * of deleted objects are kept empty and new objects gets assigned with larger
 * indices.
 */

class RS_WorldStructureRecordBase : public RS_WorldRootRecord {
   public:
    /**
     * Voxels:
     * To uniquely identify a voxel in any time of a simulation, we can define it by
     * (structure, body, voxel).
     * index_x, index_y, index_z are signed numbers, we place origin of body at (0, 0, 0)
     * since growth may push body into negative spaces we need signed numbers.
     */
    struct Voxel : public RS_WorldObjectRecord {
        /// Body index in the structure
        RIndex* body_sid = nullptr;

        /// Voxel Index in the structure
        RIndex* voxel_sid = nullptr;

        /// Body segment id in the body
        RIndex* segment_bid = nullptr;

        /// X, Y, Z index of this voxel in the body.
        RSignedIndex *index_x = nullptr, *index_y = nullptr, *index_z = nullptr;

        /// Current connectivity, lowest bit is first edge, and highest bit is last edge
        uint8_t* connectivity = nullptr;

        /// Whether this voxel is a surface voxel
        bool* is_surface = nullptr;

        /// Whether this voxel is a rigid voxel
        bool* is_rigid = nullptr;

        /// Material
        RIndex* material = nullptr;

        /// Expansion
        RFloat* expansion = nullptr;

        /// Orientation quaternion
        RQuat3rf* orientation = nullptr;

        /// CoM position
        RVec3rf* position = nullptr;

        /// PPP corner (upper-right-back) offset in local coordinate system (LCS)
        RVec3rf* ppp_offset = nullptr;

        /// NNN corner (lower-left-front) offset in local coordinate system (LCS)
        RVec3rf* nnn_offset = nullptr;

        /// Linear velocity
        RVec3rf* linear_velocity = nullptr;

        /// Angular velocity
        RVec3rf* angular_velocity = nullptr;

        /// Poissons strain
        RVec3rf* poissons_strain = nullptr;

        Voxel() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;

            return align(object_num * sizeof(RIndex)) +            // body_sid
                   align(object_num * sizeof(RIndex)) +            // voxel_sid
                   align(object_num * sizeof(RIndex)) +            // segment_bid
                   align(object_num * sizeof(RSignedIndex)) * 3 +  // index_x, index_y, index_z
                   align(object_num * sizeof(uint8_t)) +           // connectivity
                   align(object_num * sizeof(bool)) +              // is_surface
                   align(object_num * sizeof(bool)) +              // is_rigid
                   align(object_num * sizeof(RIndex)) +            // material
                   align(object_num * sizeof(RFloat)) +            // expansion
                   align(object_num * sizeof(RQuat3rf)) +          // orientation
                   align(object_num * sizeof(RVec3rf)) +           // position
                   align(object_num * sizeof(RVec3rf)) +           // ppp_offset
                   align(object_num * sizeof(RVec3rf)) +           // nnn_offset
                   align(object_num * sizeof(RVec3rf)) +           // linear_velocity
                   align(object_num * sizeof(RVec3rf)) +           // angular_velocity
                   align(object_num * sizeof(RVec3rf));            // poissons_strain
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;

            RS_WorldObjectRecord::initFromMemory(memory, object_num);

            // Allocate memory for each pointer
            setPtr(body_sid, memory, 0);
            setPtr(voxel_sid, body_sid, object_num * sizeof(RIndex));
            setPtr(segment_bid, voxel_sid, object_num * sizeof(RIndex));
            setPtr(index_x, segment_bid, object_num * sizeof(RIndex));
            setPtr(index_y, index_x, object_num * sizeof(RSignedIndex));
            setPtr(index_z, index_y, object_num * sizeof(RSignedIndex));
            setPtr(connectivity, index_z, object_num * sizeof(RSignedIndex));
            setPtr(is_surface, connectivity, object_num * sizeof(uint8_t));
            setPtr(is_rigid, is_surface, object_num * sizeof(bool));
            setPtr(material, is_rigid, object_num * sizeof(bool));
            setPtr(expansion, material, object_num * sizeof(RIndex));
            setPtr(orientation, expansion, object_num * sizeof(RFloat));
            setPtr(position, orientation, object_num * sizeof(RQuat3rf));
            setPtr(ppp_offset, position, object_num * sizeof(RVec3rf));
            setPtr(nnn_offset, ppp_offset, object_num * sizeof(RVec3rf));
            setPtr(linear_velocity, nnn_offset, object_num * sizeof(RVec3rf));
            setPtr(angular_velocity, linear_velocity, object_num * sizeof(RVec3rf));
            setPtr(poissons_strain, angular_velocity, object_num * sizeof(RVec3rf));

            // Return the updated memory pointer
            void* end;
            setPtr(end, poissons_strain, object_num * sizeof(RVec3rf));
            return end;
        }
    };

    /**
     * Links:
     * To uniquely identify a link in any time of a simulation, we can define it by
     * (structure, pos_body, pos_voxel, neg_body, neg_voxel).
     */
    struct Link : public RS_WorldObjectRecord {
        /// Negative voxel index in the structure
        RIndex* voxel_neg_sid = nullptr;

        /// Positive voxel index in the structure
        RIndex* voxel_pos_sid = nullptr;

        /// Whether the link is failed
        bool* is_failed = nullptr;

        /// Axial strain
        RFloat* strain = nullptr;

        /// Axial stress
        RFloat* stress = nullptr;

        Link() = default;

        HOST_DEVICE_INLINE_QUALIFIER Link(
            RIndex* voxel_neg_sid,
            RIndex* voxel_pos_sid,
            bool* is_failed,
            RFloat* strain,
            RFloat* stress
        )
            : voxel_neg_sid(voxel_neg_sid),
              voxel_pos_sid(voxel_pos_sid),
              is_failed(is_failed),
              strain(strain),
              stress(stress) {}

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RIndex)) +  // voxel_neg_sid
                   align(object_num * sizeof(RIndex)) +  // voxel_pos_sid
                   align(object_num * sizeof(bool)) +    // is_failed
                   align(object_num * sizeof(RFloat)) +  // strain
                   align(object_num * sizeof(RFloat));   // stress
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;

            RS_WorldObjectRecord::initFromMemory(memory, object_num);

            // Allocate memory for each pointer
            setPtr(voxel_neg_sid, memory, 0);
            setPtr(voxel_pos_sid, voxel_neg_sid, object_num * sizeof(RIndex));
            setPtr(is_failed, voxel_pos_sid, object_num * sizeof(RIndex));
            setPtr(strain, is_failed, object_num * sizeof(bool));
            setPtr(stress, strain, object_num * sizeof(RFloat));

            // Return the updated memory pointer
            void* end;
            setPtr(end, stress, object_num * sizeof(RFloat));
            return end;
        }
    };

    /**
     * Rigid bodies:
     * To uniquely identify a rigid body in any time of a simulation, we can define it by
     * (structure, rigid_body).
     */
    struct RigidBody : public RS_WorldObjectRecord {
        /// Rigid Body index in the structure
        RIndex* rigid_body_sid = nullptr;

        /// Body index in the structure
        RIndex* body_sid = nullptr;

        /// Body segment id in the body
        RIndex* segment_bid = nullptr;

        /// Body mass
        RFloat* mass = nullptr;

        /// Orientation quaternion
        RQuat3rf* orientation = nullptr;

        /// CoM position
        RVec3rf* com = nullptr;

        /// Linear velocity
        RVec3rf* linear_velocity = nullptr;

        /// Angular velocity
        RVec3rf* angular_velocity = nullptr;

        RigidBody() = default;

        HOST_DEVICE_INLINE_QUALIFIER RigidBody(
            RIndex* rigid_body_sid,
            RIndex* body_sid,
            RIndex* segment_bid,
            RFloat* mass,
            RQuat3rf* orientation,
            RVec3rf* com,
            RVec3rf* linear_velocity,
            RVec3rf* angular_velocity
        )
            : rigid_body_sid(rigid_body_sid),
              body_sid(body_sid),
              segment_bid(segment_bid),
              mass(mass),
              orientation(orientation),
              com(com),
              linear_velocity(linear_velocity),
              angular_velocity(angular_velocity) {}

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RIndex)) +    // rigid_body_sid
                   align(object_num * sizeof(RIndex)) +    // body_sid
                   align(object_num * sizeof(RIndex)) +    // segment_bid
                   align(object_num * sizeof(RFloat)) +    // mass
                   align(object_num * sizeof(RQuat3rf)) +  // orientation
                   align(object_num * sizeof(RVec3rf)) +   // com (Center of Mass)
                   align(object_num * sizeof(RVec3rf)) +   // linear_velocity
                   align(object_num * sizeof(RVec3rf));    // angular_velocity
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;

            RS_WorldObjectRecord::initFromMemory(memory, object_num);

            // Allocate memory for each pointer
            setPtr(rigid_body_sid, memory, 0);
            setPtr(body_sid, rigid_body_sid, object_num * sizeof(RIndex));
            setPtr(segment_bid, body_sid, object_num * sizeof(RIndex));
            setPtr(mass, segment_bid, object_num * sizeof(RIndex));
            setPtr(orientation, mass, object_num * sizeof(RFloat));
            setPtr(com, orientation, object_num * sizeof(RQuat3rf));
            setPtr(linear_velocity, com, object_num * sizeof(RVec3rf));
            setPtr(angular_velocity, linear_velocity, object_num * sizeof(RVec3rf));

            // Return the updated memory pointer
            void* end;
            setPtr(end, angular_velocity, object_num * sizeof(RVec3rf));
            return end;
        }
    };

    enum class JointType : RIndex {
        NULL_JOINT = 0,
        BALL_AND_SOCKET = 1,
        FIXED = 2,
        HINGE = 3,
        PRISMATIC = 4,
    };

    /**
     * Joints:
     * To uniquely identify a joint in any time of a simulation, we can define it by
     * (structure, rigid_body_a, rigid_body_b).
     */
    struct Joint : public RS_WorldObjectRecord {
        /// Rigid body A structure index
        RIndex* rigid_body_a_sid = nullptr;

        /// Rigid body B structure index
        RIndex* rigid_body_b_sid = nullptr;

        /// Record type
        JointType* type = nullptr;

        /// Position (anchor midpoint)
        RVec3rf* position = nullptr;

        /// Axis (hinge: rotation axis, prismatic: slide axis)
        RVec3rf* axis = nullptr;

        // Hinge joint fields
        /// Rotation angle signal id (only valid for hinge joint)
        RIndex* hinge_signal_sid = nullptr;

        /// Minimum angle (unit: rad, only valid for hinge joint)
        RFloat* hinge_min = nullptr;

        /// Maximum angle (unit: rad, only valid for hinge joint)
        RFloat* hinge_max = nullptr;

        /// Current angle (unit: rad, only valid for hinge joint)
        RFloat* hinge_angle = nullptr;

        /// Current torque (unit: N-m, only valid for hinge joint)
        RFloat* hinge_torque = nullptr;

        // Prismatic joint fields
        /// Translation rate signal id (only valid for prismatic joint)
        RIndex* prismatic_signal_sid = nullptr;

        /// Minimum distance (unit: m, only valid for prismatic joint)
        RFloat* prismatic_min = nullptr;

        /// Maximum distance (unit: m, only valid for prismatic joint)
        RFloat* prismatic_max = nullptr;

        /// Current distance (unit: m, only valid for prismatic joint)
        RFloat* prismatic_distance = nullptr;

        /// Current force (unit: N, only valid for prismatic joint)
        RFloat* prismatic_force = nullptr;

        Joint() = default;

        HOST_DEVICE_INLINE_QUALIFIER Joint(
            RIndex* rigid_body_a_sid,
            RIndex* rigid_body_b_sid,
            JointType* type,
            RVec3rf* position,
            RVec3rf* axis,
            RIndex* hinge_signal_sid,
            RFloat* hinge_min,
            RFloat* hinge_max,
            RFloat* hinge_angle,
            RFloat* hinge_torque,
            RIndex* prismatic_signal_sid,
            RFloat* prismatic_min,
            RFloat* prismatic_max,
            RFloat* prismatic_distance,
            RFloat* prismatic_force
        )
            : rigid_body_a_sid(rigid_body_a_sid),
              rigid_body_b_sid(rigid_body_b_sid),
              type(type),
              position(position),
              axis(axis),
              hinge_signal_sid(hinge_signal_sid),
              hinge_min(hinge_min),
              hinge_max(hinge_max),
              hinge_angle(hinge_angle),
              hinge_torque(hinge_torque),
              prismatic_signal_sid(prismatic_signal_sid),
              prismatic_min(prismatic_min),
              prismatic_max(prismatic_max),
              prismatic_distance(prismatic_distance),
              prismatic_force(prismatic_force) {}

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RIndex)) +     // rigid_body_a_sid
                   align(object_num * sizeof(RIndex)) +     // rigid_body_b_sid
                   align(object_num * sizeof(JointType)) +  // type
                   align(object_num * sizeof(RVec3rf)) +    // position
                   align(object_num * sizeof(RVec3rf)) +    // axis
                   align(object_num * sizeof(RIndex)) +     // hinge_signal_sid
                   align(object_num * sizeof(RFloat)) +     // hinge_min
                   align(object_num * sizeof(RFloat)) +     // hinge_max
                   align(object_num * sizeof(RFloat)) +     // hinge_angle
                   align(object_num * sizeof(RFloat)) +     // hinge_torque
                   align(object_num * sizeof(RIndex)) +     // prismatic_signal_sid
                   align(object_num * sizeof(RFloat)) +     // prismatic_min
                   align(object_num * sizeof(RFloat)) +     // prismatic_max
                   align(object_num * sizeof(RFloat)) +     // prismatic_distance
                   align(object_num * sizeof(RFloat));      // prismatic_force
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;

            RS_WorldObjectRecord::initFromMemory(memory, object_num);

            // Allocate memory for each pointer
            setPtr(rigid_body_a_sid, memory, 0);
            setPtr(rigid_body_b_sid, rigid_body_a_sid, object_num * sizeof(RIndex));
            setPtr(type, rigid_body_b_sid, object_num * sizeof(RIndex));
            setPtr(position, type, object_num * sizeof(JointType));
            setPtr(axis, position, object_num * sizeof(RVec3rf));
            // Hinge fields
            setPtr(hinge_signal_sid, axis, object_num * sizeof(RVec3rf));
            setPtr(hinge_min, hinge_signal_sid, object_num * sizeof(RIndex));
            setPtr(hinge_max, hinge_min, object_num * sizeof(RFloat));
            setPtr(hinge_angle, hinge_max, object_num * sizeof(RFloat));
            setPtr(hinge_torque, hinge_angle, object_num * sizeof(RFloat));
            // Prismatic fields
            setPtr(prismatic_signal_sid, hinge_torque, object_num * sizeof(RFloat));
            setPtr(prismatic_min, prismatic_signal_sid, object_num * sizeof(RIndex));
            setPtr(prismatic_max, prismatic_min, object_num * sizeof(RFloat));
            setPtr(prismatic_distance, prismatic_max, object_num * sizeof(RFloat));
            setPtr(prismatic_force, prismatic_distance, object_num * sizeof(RFloat));

            // Return the updated memory pointer
            void* end;
            setPtr(end, prismatic_force, object_num * sizeof(RFloat));
            return end;
        }
    };
};

class RS_WorldStructureVoxelRecord : public RS_WorldStructureRecordBase {
   public:
    /// Structure global id
    RIndex structure_gid = RS_NULL_INDEX;

    /// Voxel records, the size of each frame is recorded voxel num.
    Voxel voxels;

    /// Link records, the size of each frame is recorded link num.
    Link links;

    /// Rigid body records, the size of each frame is recorded rigid body num
    RigidBody rigid_bodies;

    /// Joint records, the size of each frame is the number of ball & hinge & fixed joints
    Joint joints;

    RS_WorldStructureVoxelRecord() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
        RBigSize voxel_object_num,
        RBigSize link_object_num,
        RBigSize rigid_body_object_num,
        RBigSize joint_object_num
    ) {
        RBigSize base_size = RS_WorldRootRecord::getRequiredMemorySize();
        RBigSize voxel_size = Voxel::getRequiredMemorySize(voxel_object_num);
        RBigSize link_size = Link::getRequiredMemorySize(link_object_num);
        RBigSize rigid_body_size = RigidBody::getRequiredMemorySize(rigid_body_object_num);
        RBigSize joint_size = Joint::getRequiredMemorySize(joint_object_num);

        RBigSize total_size = base_size + voxel_size + link_size + rigid_body_size + joint_size;
#ifdef RS_DEBUG_ENABLE_WORLD_RECORD
        printf(
            "Required memory sizes:\n  Base: %u\n  Voxel: %u\n  Link: %u\n  Rigid Body: %u\n  "
            "Joint: %u\nTotal required size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(voxel_size),
            static_cast<unsigned int>(link_size),
            static_cast<unsigned int>(rigid_body_size),
            static_cast<unsigned int>(joint_size),
            static_cast<unsigned int>(total_size)
        );
#endif

        return total_size;
    }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(
        void* memory,
        RBigSize voxel_object_num,
        RBigSize link_object_num,
        RBigSize rigid_body_object_num,
        RBigSize joint_object_num
    ) {
        is_valid = true;

#ifdef RS_DEBUG_ENABLE_WORLD_RECORD
        void* start_memory = memory;
        void* base_start = memory;
        memory = RS_WorldRootRecord::initFromMemory(memory);
        RBigSize base_size = static_cast<char*>(memory) - static_cast<char*>(base_start);

        void* voxel_start = memory;
        memory = voxels.initFromMemory(memory, voxel_object_num);
        RBigSize voxel_size = static_cast<char*>(memory) - static_cast<char*>(voxel_start);

        void* link_start = memory;
        memory = links.initFromMemory(memory, link_object_num);
        RBigSize link_size = static_cast<char*>(memory) - static_cast<char*>(link_start);

        void* rigid_body_start = memory;
        memory = rigid_bodies.initFromMemory(memory, rigid_body_object_num);
        RBigSize rigid_body_size =
            static_cast<char*>(memory) - static_cast<char*>(rigid_body_start);

        void* joint_start = memory;
        memory = joints.initFromMemory(memory, joint_object_num);
        RBigSize joint_size = static_cast<char*>(memory) - static_cast<char*>(joint_start);

        RBigSize total_size = static_cast<char*>(memory) - static_cast<char*>(start_memory);

        printf(
            "Occupied memory sizes:\n  Base: %u\n  Voxel: %u\n  Link: %u\n  Rigid Body: %u\n  "
            "Joint: %u\nTotal occupied size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(voxel_size),
            static_cast<unsigned int>(link_size),
            static_cast<unsigned int>(rigid_body_size),
            static_cast<unsigned int>(joint_size),
            static_cast<unsigned int>(total_size)
        );
#else
        memory = RS_WorldRootRecord::initFromMemory(memory);
        memory = voxels.initFromMemory(memory, voxel_object_num);
        memory = links.initFromMemory(memory, link_object_num);
        memory = rigid_bodies.initFromMemory(memory, rigid_body_object_num);
        memory = joints.initFromMemory(memory, joint_object_num);
#endif

        return memory;
    }
};

class RS_WorldCameraRecordBase : public RS_WorldRootRecord {
   public:
    struct RGB : public RS_WorldObjectRecord {
        /// Value of each pixel
        RColor3b* pixel = nullptr;

        RGB() = default;

        HOST_DEVICE_INLINE_QUALIFIER
        explicit RGB(RColor3b* pixel) : pixel(pixel) {}

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RColor3b));  // pixel
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;

            // Call base class initFromMemory method
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);

            // Allocate memory for each pointer
            setPtr(pixel, memory, 0);

            // Return the updated memory pointer
            void* end;
            setPtr(end, pixel, object_num * sizeof(RColor3b));
            return end;
        }
    };

    struct Depth : public RS_WorldObjectRecord {
        /// Perpendicular eye-space depth of each pixel in meters
        float* pixel = nullptr;

        Depth() = default;

        HOST_DEVICE_INLINE_QUALIFIER
        explicit Depth(float* pixel) : pixel(pixel) {}

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(float));  // pixel
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);
            setPtr(pixel, memory, 0);
            void* end;
            setPtr(end, pixel, object_num * sizeof(float));
            return end;
        }
    };

    struct Normal : public RS_WorldObjectRecord {
        /// World-space unit surface normal of each pixel
        RVec3f* pixel = nullptr;

        Normal() = default;

        HOST_DEVICE_INLINE_QUALIFIER
        explicit Normal(RVec3f* pixel) : pixel(pixel) {}

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RVec3f));  // pixel
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);
            setPtr(pixel, memory, 0);
            void* end;
            setPtr(end, pixel, object_num * sizeof(RVec3f));
            return end;
        }
    };

    struct Motion : public RS_WorldObjectRecord {
        /// Screen-space motion of each pixel normalized by image size
        RVec2f* pixel = nullptr;

        Motion() = default;

        HOST_DEVICE_INLINE_QUALIFIER
        explicit Motion(RVec2f* pixel) : pixel(pixel) {}

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RVec2f));  // pixel
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);
            setPtr(pixel, memory, 0);
            void* end;
            setPtr(end, pixel, object_num * sizeof(RVec2f));
            return end;
        }
    };

    struct EntityInfo : public RS_WorldObjectRecord {
        /// Entity type of the hit object at each pixel
        RSE_WorldEntityType* entity_type = nullptr;

        /// Global id of the hit entity at each pixel
        RIndex* entity_gid = nullptr;

        EntityInfo() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RSE_WorldEntityType)) +  // entity_type
                   align(object_num * sizeof(RIndex));                // entity_gid
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);
            setPtr(entity_type, memory, 0);
            setPtr(entity_gid, entity_type, object_num * sizeof(RSE_WorldEntityType));
            void* end;
            setPtr(end, entity_gid, object_num * sizeof(RIndex));
            return end;
        }
    };

    struct State : public RS_WorldObjectRecord {
        RVec3rf* position = nullptr;

        RQuat3rf* orientation = nullptr;

        RFloat* focus_distance = nullptr;

        RFloat* field_of_view = nullptr;

        State() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num = 1
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RVec3rf)) +   // position
                   align(object_num * sizeof(RQuat3rf)) +  // orientation
                   align(object_num * sizeof(RFloat)) +    // focus_distance
                   align(object_num * sizeof(RFloat));     // field_of_view
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num = 1) {
            using namespace world_record_helper;

            // Call base class initFromMemory method
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);

            // Allocate memory for each pointer
            setPtr(position, memory, 0);
            setPtr(orientation, position, object_num * sizeof(RVec3rf));
            setPtr(focus_distance, orientation, object_num * sizeof(RQuat3rf));
            setPtr(field_of_view, focus_distance, object_num * sizeof(RFloat));

            // Return the updated memory pointer
            void* end;
            setPtr(end, field_of_view, object_num * sizeof(RFloat));
            return end;
        }
    };
};

class RS_WorldCameraPerspectiveRecord : public RS_WorldCameraRecordBase {
   public:
    /// Camera global id
    RIndex camera_gid = RS_NULL_INDEX;

    /// Linked structure gid
    RIndex structure_gid = RS_NULL_INDEX;

    /// Linked rigid body sid
    RIndex rigid_body_sid = RS_NULL_INDEX;

    RSize width = 0;

    RSize height = 0;

    /// Pixel records, the size of each frame is camera resolution (width * height)
    RGB rgb;

    /// Depth records, the size of each frame is camera resolution (width * height)
    Depth depth;

    /// Normal records, the size of each frame is camera resolution (width * height)
    Normal normal;

    /// Motion records, the size of each frame is camera resolution (width * height)
    Motion motion;

    /// Entity info records, the size of each frame is camera resolution (width * height)
    EntityInfo entity_info;

    /// Camera state records, the size of each frame is 1
    State camera_state;

    RS_WorldCameraPerspectiveRecord() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
        RBigSize rgb_object_num,
        RBigSize depth_object_num,
        RBigSize normal_object_num,
        RBigSize motion_object_num,
        RBigSize entity_info_object_num,
        RBigSize camera_state_object_num = 1
    ) {
        RBigSize base_size = RS_WorldRootRecord::getRequiredMemorySize();
        RBigSize rgb_size = RGB::getRequiredMemorySize(rgb_object_num);
        RBigSize depth_size = Depth::getRequiredMemorySize(depth_object_num);
        RBigSize normal_size = Normal::getRequiredMemorySize(normal_object_num);
        RBigSize motion_size = Motion::getRequiredMemorySize(motion_object_num);
        RBigSize entity_info_size = EntityInfo::getRequiredMemorySize(entity_info_object_num);
        RBigSize camera_state_size = State::getRequiredMemorySize(camera_state_object_num);
        RBigSize total_size = base_size + rgb_size + depth_size + normal_size + motion_size +
                              entity_info_size + camera_state_size;

#ifdef RS_DEBUG_ENABLE_WORLD_RECORD
        printf(
            "Required memory sizes:\n  Base: %u\n  RGB: %u\n  Depth: %u\n  Normal: %u\n  "
            "Motion: %u\n  Entity Info: %u\n  Camera State: %u\nTotal required size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(rgb_size),
            static_cast<unsigned int>(depth_size),
            static_cast<unsigned int>(normal_size),
            static_cast<unsigned int>(motion_size),
            static_cast<unsigned int>(entity_info_size),
            static_cast<unsigned int>(camera_state_size),
            static_cast<unsigned int>(total_size)
        );
#endif

        return total_size;
    }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(
        void* memory,
        RBigSize rgb_object_num,
        RBigSize depth_object_num,
        RBigSize normal_object_num,
        RBigSize motion_object_num,
        RBigSize entity_info_object_num,
        RBigSize camera_state_object_num = 1
    ) {
        is_valid = true;

#ifdef RS_DEBUG_ENABLE_WORLD_RECORD
        void* start_memory = memory;
        void* base_start = memory;
        memory = RS_WorldRootRecord::initFromMemory(memory);
        RBigSize base_size = static_cast<char*>(memory) - static_cast<char*>(base_start);

        void* rgb_start = memory;
        memory = rgb.initFromMemory(memory, rgb_object_num);
        RBigSize rgb_size = static_cast<char*>(memory) - static_cast<char*>(rgb_start);

        void* depth_start = memory;
        memory = depth.initFromMemory(memory, depth_object_num);
        RBigSize depth_size = static_cast<char*>(memory) - static_cast<char*>(depth_start);

        void* normal_start = memory;
        memory = normal.initFromMemory(memory, normal_object_num);
        RBigSize normal_size = static_cast<char*>(memory) - static_cast<char*>(normal_start);

        void* motion_start = memory;
        memory = motion.initFromMemory(memory, motion_object_num);
        RBigSize motion_size = static_cast<char*>(memory) - static_cast<char*>(motion_start);

        void* entity_info_start = memory;
        memory = entity_info.initFromMemory(memory, entity_info_object_num);
        RBigSize entity_info_size =
            static_cast<char*>(memory) - static_cast<char*>(entity_info_start);

        void* camera_state_start = memory;
        memory = camera_state.initFromMemory(memory, camera_state_object_num);
        RBigSize camera_state_size =
            static_cast<char*>(memory) - static_cast<char*>(camera_state_start);

        RBigSize total_size = static_cast<char*>(memory) - static_cast<char*>(start_memory);

        printf(
            "Occupied memory sizes:\n  Base: %u\n  RGB: %u\n  Depth: %u\n  Normal: %u\n  "
            "Motion: %u\n  Entity Info: %u\n  Camera State: %u\nTotal occupied size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(rgb_size),
            static_cast<unsigned int>(depth_size),
            static_cast<unsigned int>(normal_size),
            static_cast<unsigned int>(motion_size),
            static_cast<unsigned int>(entity_info_size),
            static_cast<unsigned int>(camera_state_size),
            static_cast<unsigned int>(total_size)
        );
#else
        memory = RS_WorldRootRecord::initFromMemory(memory);
        memory = rgb.initFromMemory(memory, rgb_object_num);
        memory = depth.initFromMemory(memory, depth_object_num);
        memory = normal.initFromMemory(memory, normal_object_num);
        memory = motion.initFromMemory(memory, motion_object_num);
        memory = entity_info.initFromMemory(memory, entity_info_object_num);
        memory = camera_state.initFromMemory(memory, camera_state_object_num);
#endif

        return memory;
    }
};

class RS_WorldCameraStereoRecord : public RS_WorldCameraRecordBase {
   public:
    /// Camera global id
    RIndex camera_gid = RS_NULL_INDEX;

    /// Linked structure gid
    RIndex structure_gid = RS_NULL_INDEX;

    /// Linked rigid body sid
    RIndex rigid_body_sid = RS_NULL_INDEX;

    /// Per-eye image width
    RSize width = 0;

    /// Per-eye image height
    RSize height = 0;

    /// Left eye pixel records, the size of each frame is camera resolution (width * height)
    RGB left_rgb;

    /// Right eye pixel records, the size of each frame is camera resolution (width * height)
    RGB right_rgb;

    /// Left eye depth records, the size of each frame is camera resolution (width * height)
    Depth left_depth;

    /// Right eye depth records, the size of each frame is camera resolution (width * height)
    Depth right_depth;

    /// Left eye normal records, the size of each frame is camera resolution (width * height)
    Normal left_normal;

    /// Right eye normal records, the size of each frame is camera resolution (width * height)
    Normal right_normal;

    /// Left eye motion records, the size of each frame is camera resolution (width * height)
    Motion left_motion;

    /// Right eye motion records, the size of each frame is camera resolution (width * height)
    Motion right_motion;

    /// Left eye entity info records, the size of each frame is camera resolution (width * height)
    EntityInfo left_entity_info;

    /// Right eye entity info records, the size of each frame is camera resolution (width * height)
    EntityInfo right_entity_info;

    /// Left eye camera state records, the size of each frame is 1
    State left_camera_state;

    /// Right eye camera state records, the size of each frame is 1
    State right_camera_state;

    RS_WorldCameraStereoRecord() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
        RBigSize rgb_object_num,
        RBigSize depth_object_num,
        RBigSize normal_object_num,
        RBigSize motion_object_num,
        RBigSize entity_info_object_num,
        RBigSize camera_state_object_num = 1
    ) {
        RBigSize base_size = RS_WorldRootRecord::getRequiredMemorySize();
        RBigSize rgb_size = RGB::getRequiredMemorySize(rgb_object_num);
        RBigSize depth_size = Depth::getRequiredMemorySize(depth_object_num);
        RBigSize normal_size = Normal::getRequiredMemorySize(normal_object_num);
        RBigSize motion_size = Motion::getRequiredMemorySize(motion_object_num);
        RBigSize entity_info_size = EntityInfo::getRequiredMemorySize(entity_info_object_num);
        RBigSize camera_state_size = State::getRequiredMemorySize(camera_state_object_num);
        // Two eyes: every subrecord, including camera state, is stored once per eye.
        return base_size + 2 * (rgb_size + depth_size + normal_size + motion_size +
                                entity_info_size + camera_state_size);
    }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(
        void* memory,
        RBigSize rgb_object_num,
        RBigSize depth_object_num,
        RBigSize normal_object_num,
        RBigSize motion_object_num,
        RBigSize entity_info_object_num,
        RBigSize camera_state_object_num = 1
    ) {
        is_valid = true;

        memory = RS_WorldRootRecord::initFromMemory(memory);
        memory = left_rgb.initFromMemory(memory, rgb_object_num);
        memory = right_rgb.initFromMemory(memory, rgb_object_num);
        memory = left_depth.initFromMemory(memory, depth_object_num);
        memory = right_depth.initFromMemory(memory, depth_object_num);
        memory = left_normal.initFromMemory(memory, normal_object_num);
        memory = right_normal.initFromMemory(memory, normal_object_num);
        memory = left_motion.initFromMemory(memory, motion_object_num);
        memory = right_motion.initFromMemory(memory, motion_object_num);
        memory = left_entity_info.initFromMemory(memory, entity_info_object_num);
        memory = right_entity_info.initFromMemory(memory, entity_info_object_num);
        memory = left_camera_state.initFromMemory(memory, camera_state_object_num);
        memory = right_camera_state.initFromMemory(memory, camera_state_object_num);

        return memory;
    }
};

class RS_WorldRaySensorRecordBase : public RS_WorldRootRecord {
   public:
    struct Hit : public RS_WorldObjectRecord {
        /// Whether the ray hits anything within range
        RByte* is_hit = nullptr;

        /// Distance to the hit along the ray in meters
        RFloat* distance = nullptr;

        /// World position of the hit
        RVec3rf* hit_position = nullptr;

        Hit() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RByte)) +   // is_hit
                   align(object_num * sizeof(RFloat)) +  // distance
                   align(object_num * sizeof(RVec3rf));  // hit_position
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);
            setPtr(is_hit, memory, 0);
            setPtr(distance, is_hit, object_num * sizeof(RByte));
            setPtr(hit_position, distance, object_num * sizeof(RFloat));
            void* end;
            setPtr(end, hit_position, object_num * sizeof(RVec3rf));
            return end;
        }
    };

    struct EntityInfo : public RS_WorldObjectRecord {
        /// Global id of the hit entity
        RIndex* entity_gid = nullptr;

        /// Type of the hit entity
        RSE_WorldEntityType* entity_type = nullptr;

        EntityInfo() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RIndex)) +              // entity_gid
                   align(object_num * sizeof(RSE_WorldEntityType));  // entity_type
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);
            setPtr(entity_gid, memory, 0);
            setPtr(entity_type, entity_gid, object_num * sizeof(RIndex));
            void* end;
            setPtr(end, entity_type, object_num * sizeof(RSE_WorldEntityType));
            return end;
        }
    };

    struct EntityDetail : public RS_WorldObjectRecord {
        /// Body index of the hit voxel in the structure
        RIndex* body_sid = nullptr;

        /// Segment id of the hit voxel in the body
        RIndex* segment_bid = nullptr;

        /// Voxel index of the hit voxel in the structure
        RIndex* voxel_sid = nullptr;

        EntityDetail() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RIndex)) +  // body_sid
                   align(object_num * sizeof(RIndex)) +  // segment_bid
                   align(object_num * sizeof(RIndex));   // voxel_sid
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);
            setPtr(body_sid, memory, 0);
            setPtr(segment_bid, body_sid, object_num * sizeof(RIndex));
            setPtr(voxel_sid, segment_bid, object_num * sizeof(RIndex));
            void* end;
            setPtr(end, voxel_sid, object_num * sizeof(RIndex));
            return end;
        }
    };

    struct Material : public RS_WorldObjectRecord {
        /// Material id of the hit surface
        RIndex* material_id = nullptr;

        /// Self-emitted intensity of the hit material
        RFloat* emission = nullptr;

        Material() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RIndex)) +  // material_id
                   align(object_num * sizeof(RFloat));   // emission
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);
            setPtr(material_id, memory, 0);
            setPtr(emission, material_id, object_num * sizeof(RIndex));
            void* end;
            setPtr(end, emission, object_num * sizeof(RFloat));
            return end;
        }
    };

    struct Normal : public RS_WorldObjectRecord {
        /// World-space unit surface normal at the hit
        RVec3rf* normal = nullptr;

        Normal() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RVec3rf));  // normal
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);
            setPtr(normal, memory, 0);
            void* end;
            setPtr(end, normal, object_num * sizeof(RVec3rf));
            return end;
        }
    };

    struct Velocity : public RS_WorldObjectRecord {
        /// World velocity of the hit surface
        RVec3rf* v_hit = nullptr;

        /// Velocity of the hit surface relative to the sensor
        RVec3rf* v_rel = nullptr;

        /// Component of the relative velocity along the ray
        RFloat* v_radial = nullptr;

        Velocity() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RVec3rf)) +  // v_hit
                   align(object_num * sizeof(RVec3rf)) +  // v_rel
                   align(object_num * sizeof(RFloat));    // v_radial
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);
            setPtr(v_hit, memory, 0);
            setPtr(v_rel, v_hit, object_num * sizeof(RVec3rf));
            setPtr(v_radial, v_rel, object_num * sizeof(RVec3rf));
            void* end;
            setPtr(end, v_radial, object_num * sizeof(RFloat));
            return end;
        }
    };

    struct State : public RS_WorldObjectRecord {
        /// Sensor origin position
        RVec3rf* position = nullptr;

        /// Sensor orientation
        RQuat3rf* orientation = nullptr;

        State() = default;

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num = 1
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RVec3rf)) +  // position
                   align(object_num * sizeof(RQuat3rf));  // orientation
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num = 1) {
            using namespace world_record_helper;
            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);
            setPtr(position, memory, 0);
            setPtr(orientation, position, object_num * sizeof(RVec3rf));
            void* end;
            setPtr(end, orientation, object_num * sizeof(RQuat3rf));
            return end;
        }
    };
};

class RS_WorldRaySensorRecord : public RS_WorldRaySensorRecordBase {
   public:
    /// Ray sensor global id
    RIndex sensor_gid = RS_NULL_INDEX;

    /// Linked structure gid
    RIndex structure_gid = RS_NULL_INDEX;

    /// Linked rigid body sid
    RIndex rigid_body_sid = RS_NULL_INDEX;

    /// Number of rays
    RSize ray_num = 0;

    /// Hit records, the size of each frame is ray num
    Hit hit;

    /// Entity info records, the size of each frame is ray num
    EntityInfo entity_info;

    /// Entity detail records, the size of each frame is ray num
    EntityDetail entity_detail;

    /// Material records, the size of each frame is ray num
    Material material;

    /// Normal records, the size of each frame is ray num
    Normal normal;

    /// Velocity records, the size of each frame is ray num
    Velocity velocity;

    /// Sensor state records, the size of each frame is 1
    State sensor_state;

    RS_WorldRaySensorRecord() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
        RBigSize hit_object_num,
        RBigSize entity_info_object_num,
        RBigSize entity_detail_object_num,
        RBigSize material_object_num,
        RBigSize normal_object_num,
        RBigSize velocity_object_num,
        RBigSize sensor_state_object_num = 1
    ) {
        return RS_WorldRootRecord::getRequiredMemorySize() +
               Hit::getRequiredMemorySize(hit_object_num) +
               EntityInfo::getRequiredMemorySize(entity_info_object_num) +
               EntityDetail::getRequiredMemorySize(entity_detail_object_num) +
               Material::getRequiredMemorySize(material_object_num) +
               Normal::getRequiredMemorySize(normal_object_num) +
               Velocity::getRequiredMemorySize(velocity_object_num) +
               State::getRequiredMemorySize(sensor_state_object_num);
    }

    HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(
        void* memory,
        RBigSize hit_object_num,
        RBigSize entity_info_object_num,
        RBigSize entity_detail_object_num,
        RBigSize material_object_num,
        RBigSize normal_object_num,
        RBigSize velocity_object_num,
        RBigSize sensor_state_object_num = 1
    ) {
        is_valid = true;
        memory = RS_WorldRootRecord::initFromMemory(memory);
        memory = hit.initFromMemory(memory, hit_object_num);
        memory = entity_info.initFromMemory(memory, entity_info_object_num);
        memory = entity_detail.initFromMemory(memory, entity_detail_object_num);
        memory = material.initFromMemory(memory, material_object_num);
        memory = normal.initFromMemory(memory, normal_object_num);
        memory = velocity.initFromMemory(memory, velocity_object_num);
        memory = sensor_state.initFromMemory(memory, sensor_state_object_num);
        return memory;
    }
};

class RS_WorldIntersectRegionRecordBase : public RS_WorldRootRecord {
   public:
    /**
     * Membership:
     * The set of structures that currently have at least one voxel inside the region this step,
     * one structure global id per recorded object.
     */
    struct Membership : public RS_WorldObjectRecord {
        /// Global id of a structure intersecting the region.
        RIndex* structure_gid = nullptr;

        /// The intersecting structure's closest voxel to the region center, as a world coord offset.
        RVec3rf* closest_relative_position = nullptr;

        Membership() = default;

        HOST_DEVICE_INLINE_QUALIFIER
        Membership(RIndex* structure_gid, RVec3rf* closest_relative_position)
            : structure_gid(structure_gid), closest_relative_position(closest_relative_position) {}

        [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
            RBigSize object_num
        ) {
            using namespace world_record_helper;
            return align(object_num * sizeof(RIndex)) +    // structure_gid
                   align(object_num * sizeof(RVec3rf));    // closest_relative_position
        }

        HOST_DEVICE_INLINE_QUALIFIER void* initFromMemory(void* memory, RBigSize object_num) {
            using namespace world_record_helper;

            memory = RS_WorldObjectRecord::initFromMemory(memory, object_num);

            // Allocate memory for each pointer
            setPtr(structure_gid, memory, 0);
            setPtr(closest_relative_position, structure_gid, object_num * sizeof(RIndex));

            // Return the updated memory pointer
            void* end;
            setPtr(end, closest_relative_position, object_num * sizeof(RVec3rf));
            return end;
        }
    };
};

class RS_WorldIntersectRegionRecord : public RS_WorldIntersectRegionRecordBase {
   public:
    /// Region record global id
    RIndex record_gid = RS_NULL_INDEX;

    /// Membership records, the size of each frame is the number of intersecting structures.
    Membership memberships;

    RS_WorldIntersectRegionRecord() = default;

    [[nodiscard]] HOST_DEVICE_INLINE_QUALIFIER static RBigSize getRequiredMemorySize(
        RBigSize membership_object_num
    ) {
        RBigSize base_size = RS_WorldRootRecord::getRequiredMemorySize();
        RBigSize membership_size = Membership::getRequiredMemorySize(membership_object_num);

        RBigSize total_size = base_size + membership_size;
#ifdef RS_DEBUG_ENABLE_WORLD_RECORD
        printf(
            "Required memory sizes:\n  Base: %u\n  Membership: %u\nTotal required size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(membership_size),
            static_cast<unsigned int>(total_size)
        );
#endif

        return total_size;
    }

    HOST_DEVICE_INLINE_QUALIFIER void*
    initFromMemory(void* memory, RBigSize membership_object_num) {
        is_valid = true;

#ifdef RS_DEBUG_ENABLE_WORLD_RECORD
        void* start_memory = memory;
        void* base_start = memory;
        memory = RS_WorldRootRecord::initFromMemory(memory);
        RBigSize base_size = static_cast<char*>(memory) - static_cast<char*>(base_start);

        void* membership_start = memory;
        memory = memberships.initFromMemory(memory, membership_object_num);
        RBigSize membership_size =
            static_cast<char*>(memory) - static_cast<char*>(membership_start);

        RBigSize total_size = static_cast<char*>(memory) - static_cast<char*>(start_memory);

        printf(
            "Occupied memory sizes:\n  Base: %u\n  Membership: %u\nTotal occupied size: %u\n",
            static_cast<unsigned int>(base_size),
            static_cast<unsigned int>(membership_size),
            static_cast<unsigned int>(total_size)
        );
#else
        memory = RS_WorldRootRecord::initFromMemory(memory);
        memory = memberships.initFromMemory(memory, membership_object_num);
#endif

        return memory;
    }
};

}  // namespace rise

#endif  // RS_WORLD_RECORD_H
