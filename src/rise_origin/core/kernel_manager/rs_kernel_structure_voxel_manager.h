// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_STRUCTURE_VOXEL_MANAGER_H
#define RS_KERNEL_STRUCTURE_VOXEL_MANAGER_H

#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rise_origin/core/kernel_manager/rs_kernel_gid_registration.h"
#include "rise_origin/core/kernel_manager/rs_kernel_material_manager.h"
#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/data/rs_bitmask.h"
#include "rise_origin/device/data/rs_buffer.h"
#include "rise_origin/device/data/rs_sparse.h"
#include "rise_origin/logic/rigid/rs_constraint_m.h"
#include "rise_origin/logic/rigid/rs_rigid_body_m.h"
#include "rise_origin/logic/rigid/rs_rigid_body_surface_voxel_m.h"
#include "rise_origin/logic/soft/rs_link_m.h"
#include "rise_origin/logic/soft/rs_link_material_m.h"
#include "rise_origin/logic/soft/rs_voxel_m.h"
#include "rise_origin/logic/soft/rs_voxel_material_m.h"
#include "rise_origin/rsc/entity/rsc_structure.h"
#include "rise_origin/utils/data/rs_nary.h"
#include "rise_origin/utils/math/rs_bound.h"

namespace rise {

class RS_KernelManager;

/**
 * @brief Material physical kind resolved from the material's physical-property
 * essence. Used internally by the structure manager pipeline and queried by
 * sibling kernel data managers via getSegmentPhysicalType.
 */
RS_ENUM(
    RSE_StructureVoxelMaterialType,
    unsigned int,
    (RSE_STRUCTURE_VOXEL_MATERIAL_SOFT, 0),
    (RSE_STRUCTURE_VOXEL_MATERIAL_FIXED, 1),
    (RSE_STRUCTURE_VOXEL_MATERIAL_RIGID, 2),
    (RSE_STRUCTURE_VOXEL_MATERIAL_RIGID_FIXED, 3),
    (RSE_STRUCTURE_VOXEL_MATERIAL_UNKNOWN, 4)
)

/** @brief Coordinate key for the voxel hashmap. */
struct RS_VoxelKey {
    RIndex x = 0, y = 0, z = 0;
    RS_VoxelKey() = default;
    HOST_DEVICE_INLINE_QUALIFIER explicit RS_VoxelKey(const RCoord3 &coord)
        : x(coord.x - RS_NumTraits<RSignedIndex>::lowest()),
          y(coord.y - RS_NumTraits<RSignedIndex>::lowest()),
          z(coord.z - RS_NumTraits<RSignedIndex>::lowest()) {}
    HOST_DEVICE_INLINE_QUALIFIER RS_VoxelKey(RIndex x, RIndex y, RIndex z) : x(x), y(y), z(z) {}
    RS_VoxelKey &operator=(const RS_VoxelKey &k) = default;

    // Used to support the hashmap's internal +1 (the all-zero key is the empty sentinel).
    HOST_DEVICE_INLINE_QUALIFIER RS_VoxelKey operator+(const RIndex &val) const {
        RS_VoxelKey result;
        result.x = x + val;
        result.y = y + (result.x < x ? 1 : 0);
        result.z = z + (result.y < y ? 1 : 0);
        return result;
    }

    HOST_DEVICE_INLINE_QUALIFIER bool operator==(const RS_VoxelKey &k) const {
        return x == k.x && y == k.y && z == k.z;
    }
};

/** @brief Hasher and null-key policy for RS_VoxelKey. */
template <typename InputType, typename OutputType, RIndex HashType>
struct RS_VoxelKeyHasher {
    static_assert(std::is_same_v<InputType, RS_VoxelKey>, "InputType must be RS_VoxelKey.");
    static_assert(std::is_same_v<OutputType, uint32_t>, "OutputType must be uint32_t.");
    static_assert(HashType == 0 || HashType == 1, "HashType must be 0 or 1.");

    HOST_DEVICE_INLINE_QUALIFIER static bool isValidKey(InputType key) {
        return key.x < RS_NumTraits<RIndex>::highest() - 1 &&
               key.y < RS_NumTraits<RIndex>::highest() - 1 &&
               key.z < RS_NumTraits<RIndex>::highest() - 1;
    }

    HOST_DEVICE_INLINE_QUALIFIER static bool isNullKey(InputType key) {
        return key.x == 0 && key.y == 0 && key.z == 0;
    }

    HOST_DEVICE_INLINE_QUALIFIER static void setNullKey(InputType &key) { key = {0, 0, 0}; }

    HOST_DEVICE_INLINE_QUALIFIER OutputType operator()(InputType key) const {
        OutputType result;
        if constexpr (HashType == 0) {
            // From VDB paper
            result = key.x * 73856093 ^ key.y * 19349663 ^ key.z * 83492791;
        } else if constexpr (HashType == 1) {
            // Alternative hash function generated with test/generate_hash_constants.py
            result = key.x * 59735267 ^ key.y * 72177407 ^ key.z * 62296957;
        }
        return result;
    }
};

/**
 * @brief Manager for a single structure made of voxels.
 */
class RS_KernelStructureVoxelManager {
   public:
    using SupportedRSC = RS_Variant<rsc::RSC_StructureVoxel>;

    /**
     * @brief Bind to the kernel manager, resolve and reference dependent materials, and build the
     *        voxel/link/rigid-body topology host side.
     */
    RS_KernelStructureVoxelManager(
        RS_KernelManager *kernel_manager,
        const SupportedRSC &structure_rsc
    );

    /**
     * @brief Commit the built structure to the kernel context.
     */
    void commit();

    /**
     * @brief Decommit the structure from the kernel context.
     */
    void decommit();

    /**
     * @brief Stop simulating the structure while keeping it collidable and rendered.
     * @details Forces every voxel and rigid body to is_static, moving them into the static
     * subset, and deactivates the link, rigid-body, constraint, signal and record layouts.
     * Idempotent, and a no-op for a permanently fixed structure.
     */
    void freeze();

    /**
     * @brief Resume simulating a frozen structure.
     * @details Restores each voxel and rigid body to is_static = is_fixed and reactivates the
     * compute layouts that were not permanently fixed. Idempotent.
     */
    void unfreeze();

    /**
     * @brief Whether the structure is currently frozen.
     */
    [[nodiscard]] bool isFrozen() const;

    /**
     * @brief Increment use_count (mark this structure as not-removable).
     */
    void reference();

    /**
     * @brief Decrement use_count.
     */
    void dereference();

    /**
     * @brief Get the use count of this structure.
     */
    RSize getUseCount() const;

    /**
     * @brief Stage the latest voxel/link/rigid-body/joint device records
     *        into the recorder double-buffer; non-blocking.
     */
    void syncRecorderRecords();

    /**
     * @brief Block until the most recent syncRecorderRecords() copy has
     *        completed.
     */
    void waitForRecorderRecordsSync();

    /**
     * @brief Pull the staged recorder records (per structure) from the
     *        host buffer.
     */
    std::vector<RS_WorldStructureVoxelRecord> getRecorderRecords();

    /**
     * @brief True iff this structure emits controller signals (expansion /
     *        rotation / translation rate slots).
     */
    bool hasControllerSignal();

    /**
     * @brief Push the host-side control signals (expansion / rotation /
     *        translation rates) to the device.
     */
    void syncControllerSignal();

    /**
     * @brief Read the staged controller signal (host snapshot).
     */
    RS_WorldStructureVoxelSignal getControllerSignal();

    /**
     * @brief True iff this structure exposes a live controller record
     *        (state readout).
     */
    bool hasControllerRecord();

    /**
     * @brief Stage the latest controller record (structure state) into
     *        the controller buffer.
     */
    void syncControllerRecord();

    /**
     * @brief Read the staged controller record (host snapshot).
     */
    RS_WorldStructureVoxelRecord getControllerRecord();

    /**
     * @brief Read-only access to the active structure characterization.
     */
    const SupportedRSC &getRSC() const;

    /**
     * @brief Set of material gids this structure currently holds a reference
     *        on (one entry per distinct dependent material).
     */
    const std::unordered_set<RIndex> &getDependentMaterialGID() const;

    /**
     * @brief Structure entity name from the active characterization.
     */
    std::string getEntityName() const;

    /**
     * @brief Name of the recorder record for this structure.
     */
    std::string getRecordName() const;

    /**
     * @brief Name of the controller signal for this structure.
     */
    std::string getSignalName() const;

    /**
     * @brief Name of the controller this structure's record is sent to
     *        (record.target_controller_reference).
     */
    std::string getRecordControllerName() const;

    /**
     * @brief Name of the recorder this structure's record is sent to
     *        (record.target_recorder_reference).
     */
    std::string getRecordRecorderName() const;

    /**
     * @brief Name of the controller that drives this structure's signal
     *        (signal.source_controller_reference).
     */
    std::string getSignalControllerName() const;

    /**
     * @brief Global ID of the structure entity.
     */
    RIndex getEntityGid() const;

    /**
     * @brief Global ID of the recorder record entry.
     */
    RIndex getRecordGid() const;

    /**
     * @brief Global ID of the controller signal entry.
     */
    RIndex getSignalGid() const;

    /**
     * @brief Number of simulated voxels (soft + rigid surface) in this structure.
     */
    RSize getVoxelNum() const;

    /**
     * @brief Number of emissive voxels in this structure.
     */
    RSize getEmissiveVoxelNum() const;

    /**
     * @brief Number of links in this structure.
     */
    RSize getLinkNum() const;

    /**
     * @brief Number of rigid bodies in this structure.
     */
    RSize getRigidBodyNum() const;

    /**
     * @brief Number of constraint joints in this structure.
     */
    RSize getJointNum() const;

    /**
     * @brief Number of motorized joints in this structure.
     */
    RSize getMotorNum() const;

    /**
     * @brief Number of constraint joint entries (alias for joint count, used
     *        by kernel sizing).
     */
    RSize getConstraintJointNum() const;

    /**
     * @brief Number of distinct colors produced by graph-coloring the
     *        constraint joints (for parallel solver passes).
     */
    RSize getConstraintJointColorNum() const;

    /**
     * @brief Number of expansion-rate signal slots emitted by this structure.
     */
    RSize getExpansionSignalNum() const;

    /**
     * @brief Number of hinge rotation-angle signal slots emitted by this structure.
     */
    RSize getHingeSignalNum() const;

    /**
     * @brief Number of prismatic translation-rate signal slots emitted by this structure.
     */
    RSize getPrismaticSignalNum() const;

    /**
     * @brief Resolve the rigid-body SID for a (body, segment) pair, or
     *        RS_NULL_INDEX if the segment is not rigid.
     */
    RIndex findRigidBodySID(RIndex body_sid, RIndex segment_bid) const;

    /**
     * @brief Compute a body-local position from a rigid-body-local anchor,
     *        composing through the rigid body's origin transform.
     */
    RVec3rf getRigidBodyPositionFromRelativeAnchor(
        RIndex body_sid,
        RIndex rigid_body_sid,
        const RVec3rf &relative_anchor
    ) const;

    /**
     * @brief Physical kind (soft / fixed / rigid) for the segment's material,
     *        resolved at init from the unified material manager.
     */
    RSE_StructureVoxelMaterialType
    getSegmentPhysicalType(RIndex body_sid, RIndex segment_bid) const;

    /**
     * @brief Count voxels of a given segment whose grid indices fall inside
     *        the given AABB range.
     */
    RSize getSegmentVoxelNumInRange(
        RIndex body_sid,
        RIndex segment_bid,
        const RBound3ri &voxel_range
    ) const;

    /**
     * @brief Whether the structure can ever move a voxel: it is not fixed and holds a soft or
     *        rigid segment. Reflects inherent mobility, so freezing does not change it.
     */
    [[nodiscard]] bool hasDynamicVoxels() const;

    /**
     * @brief Whether the structure currently has any static voxels: it is frozen, permanently
     *        fixed, or holds a fixed or rigid-fixed segment.
     */
    [[nodiscard]] bool hasStaticVoxels() const;

    /**
     * @brief Set the build cache byte budget, evicting down to it.
     */
    static void setBuildCacheByteSize(RBigSize byte_size);

    /**
     * @brief Drop every cached body build, keeping the byte budget.
     */
    static void clearBuildCache();

   private:
    /**
     * @brief Scalar material data resolved once at init via the material manager's
     * query interface. Fields populated depend on the material's physical kind;
     * unused fields keep their preset defaults.
     */
    struct MaterialResolvedData {
        // Global scope ID of the material
        RIndex material_gid = RS_NULL_INDEX;

        // Material name (the structure's material reference)
        std::string name;

        // Material physical type
        RSE_StructureVoxelMaterialType phys_type = RSE_STRUCTURE_VOXEL_MATERIAL_UNKNOWN;

        // Soft / rigid density (unit: kg/m^3)
        RFloat density = 0;

        // Soft
        RFloat min_expansion = -1;
        RFloat max_expansion = 1;
        RFloat elastic_modulus = 0;
        RFloat fail_stress = RS_MAX_FLOAT;
        RFloat poissons_ratio = 0;

        // Atlas (prescribed rigid-body mass/com/inertia, overriding voxel-summed values)
        bool is_atlas = false;
        RFloat mass = 0;
        RVec3rf com_position = RVec3rf::zero();
        RMat3rf inertia = RMat3rf::zero();

        // Optical
        bool is_transparent = false;
        bool is_emissive = false;
    };

    /**
     * Note for developer:
     *
     * For all sids, we have to gurantee that they are continuous for correct dependency resolution.
     *
     * For example, when we add voxels in _addBody(), we don't add rigid body voxels that is not on
     * the surface of the rigid body. Leaving a gap in the sid sequence. when we resolve link
     * dependency, the link will not be able to find the indices of the corresponding voxels
     * correctly if sid is not continuous.
     */

    class VoxelData {
       public:
        // Temporary index in _addBody()
        RIndex tmp_index;

        // Structure scope ID of the voxel
        RIndex sid;

        // Structure scope ID of the body containing the voxel
        RIndex body_sid;

        // Body scope ID of the segment containing the voxel
        RIndex segment_bid;

        // Structure scope ID of the rigid body containing the voxel
        RIndex rigid_body_sid;

        // Index of the voxel in the body
        // Use signed index instead of unsigned index to support modifications like growth
        RSignedIndex index_x, index_y, index_z;

        // Index of this voxel's data within the body's per-voxel arrays
        RIndex array_index;

        // Structure scope material reference index (index into _mat_resolved_data)
        RIndex material_sid;

        // Material scope material optical reference index (index into the voxel material's
        // material_optical_references list)
        RIndex material_optical_reference_mid;

        // Material physical type
        RSE_StructureVoxelMaterialType material_type;

        // Structure scope ID of the voxel material (index into _voxel_materials)
        RIndex voxel_material_sid;

        // Structure scope ID of the expansion signal
        RIndex expansion_signal_sid;

        // Expansion control mode for this voxel
        rsc::RSE_StructureExpansionControlMode expansion_control_mode;

        // Global scope position
        RVec3rf g_position;

        // Global scope orientation
        RQuat3rf g_orientation;

        // Body scope position
        RVec3rf b_position;

        // Is a boundary surface voxel of the corresponding segment
        bool is_surface;

        // Has at least one face open to air or a transparent boundary, so it can be ray-traced.
        bool is_render_active;

        // Has at least one face open to air, so it can contact another body.
        bool is_contact_active;

        // Has at least one soft neighbor, so it is a link endpoint (rigid/fixed anchor).
        bool has_soft_neighbor;

        // Packed per-face RSE_VoxelFaceRole, positioned in the face_role bitfield.
        RState role_state;

        // Connected link structure scope IDs
        RIndex link_sid[6];

        // Connected neighbor voxel structure scope IDs
        RIndex neighbor_voxel_sid[6];

        // Default constructor (doesn't do anything)
        VoxelData() {}

        // Static method to get preset values
        static VoxelData preset() {
            VoxelData data;
            data.tmp_index = RS_NULL_INDEX;
            data.sid = RS_NULL_INDEX;
            data.body_sid = RS_NULL_INDEX;
            data.segment_bid = RS_NULL_INDEX;
            data.rigid_body_sid = RS_NULL_INDEX;
            data.index_x = 0;
            data.index_y = 0;
            data.index_z = 0;
            data.array_index = RS_NULL_INDEX;
            data.material_sid = RS_NULL_INDEX;
            data.material_optical_reference_mid = RS_NULL_INDEX;
            data.material_type = RSE_STRUCTURE_VOXEL_MATERIAL_UNKNOWN;
            data.expansion_signal_sid = RS_NULL_INDEX;
            data.expansion_control_mode = rsc::RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION;
            data.g_position = RVec3rf::zero();
            data.g_orientation = RQuat3rf::unit();
            data.b_position = RVec3rf::zero();
            data.is_surface = false;
            data.is_render_active = false;
            data.is_contact_active = false;
            data.has_soft_neighbor = false;
            data.role_state = 0;

            for (int i = 0; i < 6; i++) {
                data.link_sid[i] = RS_NULL_INDEX;
                data.neighbor_voxel_sid[i] = RS_NULL_INDEX;
            }

            return data;
        }
    };

    class LinkData {
       public:
        // Structure scope ID of the link
        RIndex sid;

        // Structure scope ID of the body containing the link (only valid for normal links)
        RIndex body_sid;

        // Structure scope ID of the negative voxel
        RIndex voxel_neg_sid;

        // Structure scope ID of the positive voxel
        RIndex voxel_pos_sid;

        // Structure scope material reference index of the negative voxel (index into
        // _mat_resolved_data)
        RIndex voxel_neg_material_sid;

        // Structure scope material reference index of the positive voxel (index into
        // _mat_resolved_data)
        RIndex voxel_pos_material_sid;

        // Structure scope ID of the link material (index into _link_materials)
        RIndex link_material_sid;

        logic::RSE_LinkAxis axis;

        // Default constructor (doesn't do anything to speed up vector initialization)
        LinkData() {}

        // Static method to get preset values
        static LinkData preset() {
            LinkData data;
            data.sid = RS_NULL_INDEX;
            data.body_sid = RS_NULL_INDEX;
            data.voxel_neg_sid = RS_NULL_INDEX;
            data.voxel_pos_sid = RS_NULL_INDEX;
            data.voxel_neg_material_sid = RS_NULL_INDEX;
            data.voxel_pos_material_sid = RS_NULL_INDEX;
            data.link_material_sid = RS_NULL_INDEX;
            data.axis = logic::RSE_NO_AXIS;
            return data;
        }
    };

    class VoxelMaterialData {
       public:
        // Structure scope material reference index (index into _mat_resolved_data)
        RIndex material_sid;

        // Body SID (used as dedup key since voxel_size varies per body)
        RIndex body_sid;

        // Voxel size for this material instance
        RFloat voxel_size;

        // Default constructor (doesn't do anything to speed up vector initialization)
        VoxelMaterialData() {}

        // Static method to get preset values
        static VoxelMaterialData preset() {
            VoxelMaterialData data;
            data.material_sid = RS_NULL_INDEX;
            data.body_sid = RS_NULL_INDEX;
            data.voxel_size = 0;
            return data;
        }
    };

    class LinkMaterialData {
       public:
        // Negative voxel material reference index (index into _mat_resolved_data)
        RIndex neg_material_sid;

        // Positive voxel material reference index (index into _mat_resolved_data)
        RIndex pos_material_sid;

        // Body SID (used as dedup key since voxel_size varies per body)
        RIndex body_sid;

        // Voxel size for this material instance
        RFloat voxel_size;

        // Default constructor (doesn't do anything to speed up vector initialization)
        LinkMaterialData() {}

        // Static method to get preset values
        static LinkMaterialData preset() {
            LinkMaterialData data;
            data.neg_material_sid = RS_NULL_INDEX;
            data.pos_material_sid = RS_NULL_INDEX;
            data.body_sid = RS_NULL_INDEX;
            data.voxel_size = 0;
            return data;
        }
    };

    class RigidBodyData {
       public:
        // Structure scope ID of the rigid body
        RIndex sid;

        // Structure scope ID of the body containing the rigid body
        RIndex body_sid;

        // Body scope ID of the segment containing the rigid body
        RIndex segment_bid;

        // Global scope origin position
        // Note: origin is the voxel with minimum x, y, z index
        RVec3rf origin_g_position;

        // Global scope orientation
        RQuat3rf g_orientation;

        // Body scope origin position
        RVec3rf origin_b_position;

        // Total mass of the rigid body
        RFloat mass = 0;

        // Inverse mass; zero for fixed (immovable) rigid bodies
        RFloat mass_inverse = 0;

        // Center of mass, relative to the rigid body origin
        RVec3rf com_rb_position = RVec3rf::zero();

        // Inverse of the local moment of inertia tensor about the center of mass
        RMat3rf local_moment_inertia_inverse = RMat3rf::zero();

        // Bitmask of rigid bodies (by sid) this body must not collide with: its own bit plus
        // every body joined to it by a constraint. Split into two words on copy to
        // RS_RigidBodyMem::connected_body_mask_0 and connected_body_mask_1.
        device::RS_StaticBitMask<128, uint64_t> connected_body_mask;

        // Contained voxel temporary indices in _addBody()
        std::vector<RIndex> contained_voxel_tmp_indices;

        // Contained material reference index (index into _mat_resolved_data) for each internal
        // voxel
        std::vector<RIndex> contained_material_sid;

        // Contained material voxel positions relative to the origin, in its own coordinate system
        std::vector<RVec3rf> contained_material_rb_positions;

        // Start index of the rigid body surface voxels in _rigid_body_surface_voxels
        RIndex rigid_body_surface_voxel_start = RS_NULL_INDEX;

        // End index of the rigid body surface voxels in _rigid_body_surface_voxels
        RIndex rigid_body_surface_voxel_end = RS_NULL_INDEX;

        // Default constructor (initializes vectors)
        RigidBodyData()
            : contained_voxel_tmp_indices(),
              contained_material_sid(),
              contained_material_rb_positions() {}

        // Static method to get preset values
        static RigidBodyData preset() {
            RigidBodyData data;
            data.sid = RS_NULL_INDEX;
            data.body_sid = RS_NULL_INDEX;
            data.segment_bid = RS_NULL_INDEX;
            data.origin_g_position = RVec3rf::zero();
            data.g_orientation = RQuat3rf::unit();
            data.origin_b_position = RVec3rf::zero();
            data.mass = 0;
            data.mass_inverse = 0;
            data.com_rb_position = RVec3rf::zero();
            data.local_moment_inertia_inverse = RMat3rf::zero();
            // Vectors are already initialized by the constructor
            return data;
        }
    };

    class RigidBodySurfaceVoxelData {
       public:
        // Structure scope ID of the voxel
        RIndex voxel_sid;

        // Structure scope ID of the rigid body containing the voxel
        RIndex rigid_body_sid;

        // Voxel position relative to rigid body origin
        RVec3rf voxel_rb_position;

        // Default constructor (doesn't do anything to speed up vector initialization)
        RigidBodySurfaceVoxelData() {}

        // Static method to get preset values
        static RigidBodySurfaceVoxelData preset() {
            RigidBodySurfaceVoxelData data;
            data.voxel_sid = RS_NULL_INDEX;
            data.rigid_body_sid = RS_NULL_INDEX;
            data.voxel_rb_position = RVec3rf::zero();
            return data;
        }
    };

    class ConstraintJointData {
       public:
        logic::RSE_ConstraintJointType type;

        // Structure scope ID of the constraint
        RIndex sid;

        // Structure scope ID of the joint
        RIndex joint_sid;

        // If the constraint is a hinge joint with motor enabled,
        // this is the structure scope ID of the motor
        RIndex motor_sid;

        // Structure scope ID of the rigid body a connected by the constraint
        RIndex rigid_body_a_sid;

        // Structure scope ID of the rigid body b connected by the constraint
        RIndex rigid_body_b_sid;

        // Rigid body scope anchor position of anchor A, relative to the origin of rigid body A
        RVec3rf rigid_body_a_anchor_rb_position;

        // Rigid body scope anchor position of anchor B, relative to the origin of rigid body B
        RVec3rf rigid_body_b_anchor_rb_position;

        // Control mode of the hinge joint.
        logic::RSE_ConstraintJointHingeControlMode hinge_control_mode;

        // Rigid body scope axis of the hinge joint, relative to the rigid body A
        RVec3rf hinge_a_rb_axis;

        // Rigid body scope axis of the hinge joint, relative to the rigid body B
        RVec3rf hinge_b_rb_orthogonal_axis_1;

        RVec3rf hinge_b_rb_orthogonal_axis_2;

        // Minimum angle allowed by hinge, relative to initial two rigid body positions
        RFloat hinge_min;

        // Maximum angle allowed by hinge, relative to initial two rigid body positions
        RFloat hinge_max;

        // Proportional gain of the position control of the hinge joint.
        RFloat hinge_position_control_kp;

        // Derivative gain of the position control of the hinge joint.
        RFloat hinge_position_control_kd;

        // Maximum speed of the hinge joint.
        RFloat hinge_max_speed;

        // Maximum motor torque of hinge
        RFloat hinge_max_torque;

        // Structure scope ID of the hinge rotation angle signal
        RIndex hinge_signal_sid;

        // Control mode of the prismatic joint.
        logic::RSE_ConstraintJointPrismaticControlMode prismatic_control_mode;

        // Rigid body scope axis of the prismatic joint, relative to the rigid body A
        RVec3rf prismatic_a_rb_axis;

        // Rigid body scope orthogonal axes for perpendicular translation, relative to rigid body B
        RVec3rf prismatic_b_rb_orthogonal_axis_1;

        RVec3rf prismatic_b_rb_orthogonal_axis_2;

        // Minimum translation allowed by prismatic joint
        RFloat prismatic_min;

        // Maximum translation allowed by prismatic joint
        RFloat prismatic_max;

        // Proportional gain of the position control of the prismatic joint.
        RFloat prismatic_position_control_kp;

        // Derivative gain of the position control of the prismatic joint.
        RFloat prismatic_position_control_kd;

        // Maximum speed of the prismatic joint.
        RFloat prismatic_max_speed;

        // Maximum motor force of prismatic joint
        RFloat prismatic_max_force;

        // Structure scope ID of the prismatic translation rate signal
        RIndex prismatic_signal_sid;

        RIndex color;

        // Default constructor (doesn't do anything to speed up vector initialization)
        ConstraintJointData() {}

        // Static method to get preset values
        static ConstraintJointData preset() {
            ConstraintJointData data;
            data.sid = RS_NULL_INDEX;
            data.joint_sid = RS_NULL_INDEX;
            data.motor_sid = RS_NULL_INDEX;
            data.rigid_body_a_sid = RS_NULL_INDEX;
            data.rigid_body_b_sid = RS_NULL_INDEX;
            data.rigid_body_a_anchor_rb_position = RVec3rf::zero();
            data.rigid_body_b_anchor_rb_position = RVec3rf::zero();
            data.hinge_control_mode = logic::RSE_CONSTRAINT_JOINT_HINGE_CONTROL_BASIC_POSITION;
            data.hinge_a_rb_axis = RVec3rf::zero();
            data.hinge_b_rb_orthogonal_axis_1 = RVec3rf::zero();
            data.hinge_b_rb_orthogonal_axis_2 = RVec3rf::zero();
            data.hinge_min = -RS_PI;
            data.hinge_max = RS_PI;
            data.hinge_position_control_kp = 0;
            data.hinge_position_control_kd = 0;
            data.hinge_max_speed = 0;
            data.hinge_max_torque = 0;
            data.hinge_signal_sid = RS_NULL_INDEX;
            data.prismatic_control_mode =
                logic::RSE_CONSTRAINT_JOINT_PRISMATIC_CONTROL_BASIC_POSITION;
            data.prismatic_a_rb_axis = RVec3rf::zero();
            data.prismatic_min = 0;
            data.prismatic_max = 0;
            data.prismatic_position_control_kp = 0;
            data.prismatic_position_control_kd = 0;
            data.prismatic_max_speed = 0;
            data.prismatic_max_force = 0;
            data.prismatic_signal_sid = RS_NULL_INDEX;
            data.color = RS_NULL_INDEX;
            return data;
        }
    };

    struct Cuboid {
        RIndex x_start, x_end;  // [x_start, x_end)
        RIndex y_start, y_end;  // [y_start, y_end)
        RIndex z_start, z_end;  // [z_start, z_end)
    };

    // Single stable flat hashmap mapping a voxel coordinate directly to its temporary index. Each
    // voxel is one hashmap entry, built at a 50% load factor.
    using VoxelHashmap =
        device::RS_HashStableFlatMap<RS_VoxelKey, RIndex, uint32_t, RS_VoxelKeyHasher, 8>;

    // Lightweight per-voxel record kept after commit for getSegmentVoxelNumInRange.
    struct VoxelOccupancyData {
        RIndex body_sid;
        RIndex segment_bid;
        RIndex rigid_body_sid;
        RSignedIndex index_x, index_y, index_z;
    };

    struct BuildCachedBody {
        // Weak handle to the source body, so a hit on a reused key address is rejected as stale
        std::weak_ptr<const rsc::RSC_StructureVoxelBody> source_body_rsc;

        // Pre-filter temporary voxels as built, with body-local sids and a stale pose
        std::vector<VoxelData> all_voxels;

        // This body's links as built, with body-local voxel sids
        std::vector<LinkData> links;

        // Per segment, which material references it contains; geometry-only, reused on any hit
        std::map<RIndex, std::vector<bool>> segment_material_flags;

        // Resolved material references used to produce the build, compared on a hit
        std::vector<MaterialResolvedData> materials;

        // Scoped provider that owns the hashmap memory
        std::unique_ptr<RS_MemoryScopedProvider> provider;

        // Coordinate hashmap over all_voxels, reused read-only on a hit
        mutable VoxelHashmap hashmap;

        // True when the cached build was produced for a fixed structure
        bool is_fixed = false;

        // Voxel, link and rigid body sid offsets at cache time, so reuse rebases by a delta
        RIndex base_voxel_sid = RS_NULL_INDEX;
        RIndex base_link_sid = RS_NULL_INDEX;
        RIndex base_rigid_body_sid = RS_NULL_INDEX;

        // Total retained bytes, for byte-budget accounting
        RBigSize byte_size = 0;
    };

    // LRU of per-body builds, shared by every manager instance.
    struct BuildCache {
        std::unordered_map<
            const rsc::RSC_StructureVoxelBody *,
            std::pair<
                std::shared_ptr<const BuildCachedBody>,
                std::list<const rsc::RSC_StructureVoxelBody *>::iterator>>
            map;
        std::list<const rsc::RSC_StructureVoxelBody *> lru;
        RBigSize budget = 0;
        RBigSize total = 0;
    };

    template <RSE_FloatType FloatMemAccuracy>
    class KernelStructureData {
       public:
        device::RS_Array<logic::RS_VoxelMem<FloatMemAccuracy>, 1> voxels;
        device::RS_Array<logic::RS_LinkMem<FloatMemAccuracy>, 1> links;
        device::RS_Array<logic::RS_VoxelMaterialMem<FloatMemAccuracy>, 1> voxel_materials;
        device::RS_Array<logic::RS_LinkMaterialMem<FloatMemAccuracy>, 1> link_materials;
        device::RS_Array<logic::RS_RigidBodyMem<FloatMemAccuracy>, 1> rigid_bodies;
        device::RS_Array<logic::RS_RigidBodySurfaceVoxelMem<FloatMemAccuracy>, 1>
            rigid_body_surface_voxels;
        device::RS_Array<logic::RS_ConstraintJointMem<FloatMemAccuracy>, 1> constraint_joints;
        device::RS_Array<RS_FloatMem<FloatMemAccuracy>, 1> rigid_body_mass;
        device::RS_Array<RS_FloatMem<FloatMemAccuracy>, 1> rigid_body_mass_inverses;
        device::RS_Array<RS_Vec3DMem<FloatMemAccuracy>, 1> rigid_body_com_rb_positions;
        device::RS_Array<RS_Mat3DMem<FloatMemAccuracy>, 1> rigid_body_local_moment_inertia_inverses;
    };

    RS_KernelManager *_kernel_manager = nullptr;
    std::unique_ptr<RS_MemoryScopedProvider> _scoped_host_provider;
    std::unique_ptr<RS_MemoryScopedProvider> _scoped_device_provider;

    RS_KernelGidRegistration _entity_registration;
    RS_KernelGidRegistration _record_registration;
    RS_KernelGidRegistration _signal_registration;
    std::vector<RS_KernelMaterialReference> _material_references;

    /// Cached gids and immediate lookup sets.
    RIndex _structure_gid = RS_NULL_INDEX;
    RIndex _record_gid = RS_NULL_INDEX;
    RIndex _signal_gid = RS_NULL_INDEX;
    std::unordered_set<RIndex> _dependent_material_gid_set;

    bool _is_kernel_data_inited = false;

    SupportedRSC _rsc;

    /// True while the structure is frozen, so it stays collidable and rendered but stops
    /// simulating.
    bool _is_frozen = false;

    /// True when the structure defines any constraint, so the connected-body collision mask is
    /// used.
    bool _is_constrained = false;

    /// Live use_count from external consumers (force / camera RBLinked / etc.).
    RSize _use_count = 0;

    /// Counters
    RSize _body_num = 0;
    RSize _voxel_num = 0;
    RSize _emissive_voxel_num = 0;
    RSize _link_num = 0;
    RSize _rigid_body_num = 0;
    RSize _rigid_body_surface_voxel_num = 0;
    RSize _joint_num = 0;
    RSize _constraint_joint_num = 0;
    RSize _motor_num = 0;
    RSize _constraint_joint_color_num = 0;

    /// Largest authored segment id over all bodies, for contact filter key field check
    RIndex _max_segment_bid = 0;

    /// Metrics
    /// Maps body_sid -> body_size
    std::unordered_map<RIndex, RSize> _body_size;

    /// Maps (body_sid, segment_bid) -> physical type derived from material characterization
    std::map<RS_OrderedPair<RIndex, RIndex>, RSE_StructureVoxelMaterialType> _segment_phys_type_map;

    /// Resolved per-material data, indexed by material_sid (the structure's material reference
    /// index, i.e. the value stored in body voxels' material_reference_sid).
    std::vector<MaterialResolvedData> _mat_resolved_data;

    /// For maps with complex keys, tree based map is way more efficient
    /// Maps (body_sid, segment_bid) -> rigid_body_sid
    std::map<RS_OrderedPair<RIndex, RIndex>, RIndex> _rigid_body_sid_map;

    /// Structure data
    std::vector<VoxelData> _voxels;
    std::vector<LinkData> _links;
    std::vector<RigidBodyData> _rigid_bodies;
    std::vector<RigidBodySurfaceVoxelData> _rigid_body_surface_voxels;
    std::vector<ConstraintJointData> _constraint_joints;
    std::vector<VoxelMaterialData> _voxel_materials;
    std::vector<LinkMaterialData> _link_materials;

    /// Lightweight voxel records kept after commit, replacing the freed _voxels array
    std::vector<VoxelOccupancyData> _voxel_occupancy;

    /// Rigid body origins kept after commit, replacing the freed _rigid_bodies array
    std::vector<RVec3rf> _rigid_body_origin_b_position;

    /// Signal data
    RSize _expansion_signal_num = 0;
    RSize _hinge_signal_num = 0;
    RSize _prismatic_signal_num = 0;

    RHandle _structure_signal = RS_NULL_HANDLE;
    RS_MemoryPtr _host_signal_buffer;
    RS_MemoryPtr _device_signal_buffer;
    RS_WorldStructureVoxelSignal _host_signal;
    RS_WorldStructureVoxelSignal _device_signal;

    /// Record data
    RSize _voxel_record_num = 0;
    RSize _link_record_num = 0;
    RSize _rigid_body_record_num = 0;
    RSize _joint_record_num = 0;

    device::RS_DoubleBuffer _recorder_buffer;
    device::RS_DirectBuffer _controller_buffer;

    device::RS_Array<RIndex, 1> _voxel_record_sid;
    device::RS_Array<RHandle, 1> _voxel_record_handles;
    device::RS_Array<RIndex, 1> _link_record_sid;
    device::RS_Array<RHandle, 1> _link_record_handles;
    device::RS_Array<RIndex, 1> _rigid_body_record_sid;
    device::RS_Array<RHandle, 1> _rigid_body_record_handles;
    device::RS_Array<RIndex, 1> _constraint_joint_record_sid;
    device::RS_Array<RHandle, 1> _constraint_joint_record_handles;

    /// Build cache shared by every manager, with its own lock
    static BuildCache _build_cache;
    static std::mutex _build_cache_mutex;

    /**
     * @brief Concrete view of this manager's variant characterization.
     */
    const rsc::RSC_StructureVoxel &_structure() const { return _rsc.as<rsc::RSC_StructureVoxel>(); }
    rsc::RSC_StructureVoxel &_structure() { return _rsc.as<rsc::RSC_StructureVoxel>(); }

    /**
     * @brief Resolve each dependent material's physical kind and data.
     */
    void _resolveMaterials();

    /**
     * @brief Whether resolved materials match a cached build.
     */
    bool _isMaterialCoherent(const std::vector<MaterialResolvedData> &cached_materials) const;

    /**
     * @brief Build one body's voxels, links and rigid bodies.
     */
    void _addBody(const std::shared_ptr<rsc::RSC_StructureVoxelBody> &body_ref);

    /**
     * @brief Add one constraint joint between two rigid bodies.
     */
    void _addConstraint(const rsc::RSC_StructureVoxelConstraint &constraint_rsc);

    /**
     * @brief Build the voxel and link material tables.
     */
    void _addMaterials();

    /**
     * @brief Allocate the controller signal slots for this structure.
     */
    void _addSignals();

    /**
     * @brief Make a constraint joint from a constraint characterization.
     */
    ConstraintJointData _createBasicConstraintJoint(
        const rsc::RSC_StructureVoxelConstraint &constraint_rsc,
        logic::RSE_ConstraintJointType constraint_type,
        RIndex joint_sid
    );

    /**
     * @brief Graph-color the constraint joints for parallel solving.
     */
    void _colorConstraintJoints();

    /**
     * @brief Check the counts that gate fixed-size kernel buffers before they are indexed.
     */
    void _precheckStructureLimits() const;

    /**
     * @brief Check the structure stays within kernel size limits.
     */
    void _postcheckStructureLimits() const;

    /**
     * @brief Check every rigid body has a valid inertia.
     */
    void _checkRigidBodyInertia() const;

    /**
     * @brief Gather all host build data into device arrays.
     */
    template <RSE_FloatType FloatMemAccuracy>
    KernelStructureData<FloatMemAccuracy> _getAll() const;

    /**
     * @brief Keep lightweight query forms and free the heavy build arrays.
     */
    void _cleanupGeneratedData();

    /**
     * @brief Push the built structure data to the device.
     */
    void _initKernelData();

    /**
     * @brief Free the structure data from the device.
     */
    void _freeKernelData();

    /**
     * @brief Push the recorder record layouts to the device.
     */
    void _initKernelRecordData();

    /**
     * @brief Free the recorder record layouts from the device.
     */
    void _freeKernelRecordData();

    /**
     * @brief Push the controller signal layouts to the device.
     */
    void _initKernelSignalData();

    /**
     * @brief Free the controller signal layouts from the device.
     */
    void _freeKernelSignalData();

    /**
     * @brief Register this structure's emissive voxel handles.
     */
    void _initKernelEmissiveVoxelHandlesData(RSize voxel_offset);

    /**
     * @brief Unregister this structure's emissive voxel handles.
     */
    void _freeKernelEmissiveVoxelHandlesData();

    /**
     * @brief Toggle simulation while keeping the structure rendered.
     */
    void _freezeKernelData(bool is_freeze);

    /**
     * @brief Convert an RSC hinge control mode to the joint mode.
     */
    static logic::RSE_ConstraintJointHingeControlMode _convertToConstraintJointHingeControlMode(
        rsc::RSE_StructureVoxelHingeControlMode mode
    );

    /**
     * @brief Convert an RSC prismatic control mode to the joint mode.
     */
    static logic::RSE_ConstraintJointPrismaticControlMode
    _convertToConstraintJointPrismaticControlMode(rsc::RSE_StructureVoxelPrismaticControlMode mode);

    /**
     * @brief Pick spaced-out structure scope ids for a name.
     */
    static std::vector<RIndex>
    _selectSID(const std::string &structure_name, RSize select_num, RSize range);

    /**
     * @brief Partition a body's voxel grid into cuboids.
     */
    static std::vector<Cuboid> _partitionSpace(RSize x_size, RSize y_size, RSize z_size);

    /**
     * @brief Compute the structure's world space bounding box.
     */
    static RBound3f _computeWorldBound(const rsc::RSC_StructureVoxel &structure_rsc);

    /**
     * @brief Look up a body's cached build and get a persistent shared ptr reference.
     */
    static std::shared_ptr<const BuildCachedBody> _tryGetCachedBody(
        const rsc::RSC_StructureVoxelBody *key
    );

    /**
     * @brief Cache a body build, evicting down to budget.
     */
    static void _putCachedBody(const rsc::RSC_StructureVoxelBody *key, BuildCachedBody entry);
};

/**
 * @brief References a structure on construction and dereferences it on destruction.
 */
class RS_KernelStructureVoxelReference {
   public:
    RS_KernelStructureVoxelReference() = default;
    RS_KernelStructureVoxelReference(RS_KernelManager *kernel_manager, std::string name);
    ~RS_KernelStructureVoxelReference();

    RS_KernelStructureVoxelReference(RS_KernelStructureVoxelReference &&other) noexcept;
    RS_KernelStructureVoxelReference &operator=(RS_KernelStructureVoxelReference &&other) noexcept;
    RS_KernelStructureVoxelReference(const RS_KernelStructureVoxelReference &) = delete;
    RS_KernelStructureVoxelReference &operator=(const RS_KernelStructureVoxelReference &) = delete;

    [[nodiscard]] const std::string &name() const { return _name; }
    [[nodiscard]] RIndex gid() const { return _gid; }

   private:
    void _release();

    RS_KernelManager *_kernel_manager = nullptr;
    std::string _name;
    RIndex _gid = RS_NULL_INDEX;
};

}  // namespace rise

#endif  // RS_KERNEL_STRUCTURE_VOXEL_MANAGER_H
