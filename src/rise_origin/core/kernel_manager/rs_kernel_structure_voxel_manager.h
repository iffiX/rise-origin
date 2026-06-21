// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_STRUCTURE_VOXEL_MANAGER_H
#define RS_KERNEL_STRUCTURE_VOXEL_MANAGER_H

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
#include "rise_origin/utils/math/rs_bound.h"
#include "rise_origin/utils/data/rs_nary.h"

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

class RS_KernelStructureVoxelManager {
   public:
    using SupportedRSC = rsc::RSC_StructureVoxel;

    /**
     * @brief Bind to the kernel manager, resolve and reference dependent materials, and build the
     *        voxel/link/rigid-body topology host side.
     */
    RS_KernelStructureVoxelManager(RS_KernelManager *kernel_manager, const SupportedRSC &structure_rsc);

    ~RS_KernelStructureVoxelManager();

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
     *        (signal.source_controller_name).
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

   private:
    /**
     * @brief Scalar material data resolved once at init via the material manager's
     * query interface. Fields populated depend on the material's physical kind;
     * unused fields keep their preset defaults.
     */
    struct MaterialResolvedData {
        // Soft / rigid density (unit: kg/m^3)
        RFloat density = 0;

        // Soft
        RFloat min_expansion = -1;
        RFloat max_expansion = 1;
        RFloat elastic_modulus = 0;
        RFloat fail_stress = RS_MAX_FLOAT;
        RFloat poissons_ratio = 0;

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

        // Global scope ID of the material
        RIndex material_gid;

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
            data.material_gid = RS_NULL_INDEX;
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

        // Global scope ID of the negative voxel material
        RIndex voxel_neg_material_gid;

        // Global scope ID of the positive voxel material
        RIndex voxel_pos_material_gid;

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
            data.voxel_neg_material_gid = RS_NULL_INDEX;
            data.voxel_pos_material_gid = RS_NULL_INDEX;
            data.link_material_sid = RS_NULL_INDEX;
            data.axis = logic::RSE_NO_AXIS;
            return data;
        }
    };

    class VoxelMaterialData {
       public:
        // Global scope ID of the material
        RIndex gid;

        // Body SID (used as dedup key since voxel_size varies per body)
        RIndex body_sid;

        // Voxel size for this material instance
        RFloat voxel_size;

        // Default constructor (doesn't do anything to speed up vector initialization)
        VoxelMaterialData() {}

        // Static method to get preset values
        static VoxelMaterialData preset() {
            VoxelMaterialData data;
            data.gid = RS_NULL_INDEX;
            data.body_sid = RS_NULL_INDEX;
            data.voxel_size = 0;
            return data;
        }
    };

    class LinkMaterialData {
       public:
        // Negative voxel material global scope ID
        RIndex neg_gid;

        // Positive voxel material global scope ID
        RIndex pos_gid;

        // Body SID (used as dedup key since voxel_size varies per body)
        RIndex body_sid;

        // Voxel size for this material instance
        RFloat voxel_size;

        // Default constructor (doesn't do anything to speed up vector initialization)
        LinkMaterialData() {}

        // Static method to get preset values
        static LinkMaterialData preset() {
            LinkMaterialData data;
            data.neg_gid = RS_NULL_INDEX;
            data.pos_gid = RS_NULL_INDEX;
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

        // Contained material global scope IDs for each internal voxel
        std::vector<RIndex> contained_material_gid;

        // Contained material voxel positions relative to the origin, in its own coordinate system
        std::vector<RVec3rf> contained_material_rb_positions;

        // Start index of the rigid body surface voxels in _rigid_body_surface_voxels
        RIndex rigid_body_surface_voxel_start = RS_NULL_INDEX;

        // End index of the rigid body surface voxels in _rigid_body_surface_voxels
        RIndex rigid_body_surface_voxel_end = RS_NULL_INDEX;

        // Default constructor (initializes vectors)
        RigidBodyData()
            : contained_voxel_tmp_indices(),
              contained_material_gid(),
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

    using VoxelVDB = device::RS_Sparse3DMap<
        RIndex,
        device::RS_Sparse3DHashMapLevel<1024, 8>,
        device::RS_Sparse3DArrayLevel<8, 8, 8>,
        device::RS_Sparse3DArrayLevel<8, 8, 8>,
        device::RS_Sparse3DArrayLevel<32, 32, 32>>;

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
    std::unique_ptr<RS_MemoryScopedProvider> _scoped_provider;

    RS_KernelGidRegistration _entity_registration;
    RS_KernelGidRegistration _record_registration;
    RS_KernelGidRegistration _signal_registration;
    std::vector<RS_KernelMaterialReference> _material_references;

    /// Cached gids and immediate lookup sets.
    RIndex _structure_gid = RS_NULL_INDEX;
    RIndex _record_gid = RS_NULL_INDEX;
    RIndex _signal_gid = RS_NULL_INDEX;
    std::vector<RIndex> _dependent_material_gid;
    std::unordered_set<RIndex> _dependent_material_gid_set;

    bool _is_kernel_data_inited = false;

    SupportedRSC _rsc;

    /// True while the structure is frozen, so it stays collidable and rendered but stops simulating.
    bool _is_frozen = false;

    /// True when the structure defines any constraint, so the connected-body collision mask is used.
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

    /// Metrics
    /// Maps body_sid -> body_size
    std::unordered_map<RIndex, RSize> _body_size;

    /// Maps (body_sid, segment_bid) -> physical type derived from material characterization
    std::map<RS_OrderedPair<RIndex, RIndex>, RSE_StructureVoxelMaterialType> _segment_phys_type_map;

    /// Maps material_gid -> physical type
    std::unordered_map<RIndex, RSE_StructureVoxelMaterialType> _mat_phys_type_map;

    /// Maps material_gid -> material_name
    std::unordered_map<RIndex, std::string> _mat_gid_to_name;

    /// Maps material_gid -> resolved per-material properties
    std::unordered_map<RIndex, MaterialResolvedData> _mat_props;

    /// For maps with complex keys, tree based map is way more efficient
    /// Maps (body_sid, segment_bid) -> rigid_body_sid
    std::map<RS_OrderedPair<RIndex, RIndex>, RIndex> _rigid_body_sid_map;

    /// Maps (body_sid) -> VDB(x, y, z) -> voxel_tmp_index
    std::unordered_map<RIndex, VoxelVDB> _voxel_tmp_index_vdb_map;

    /// Structure data
    std::vector<VoxelData> _voxels;
    std::vector<LinkData> _links;
    std::vector<RigidBodyData> _rigid_bodies;
    std::vector<RigidBodySurfaceVoxelData> _rigid_body_surface_voxels;
    std::vector<ConstraintJointData> _constraint_joints;
    std::vector<VoxelMaterialData> _voxel_materials;
    std::vector<LinkMaterialData> _link_materials;
    std::vector<std::pair<RIndex, RIndex>> _fixed_connected_rigid_body_sid;

    /// Stores all voxels, indexable by tmp_index
    std::unordered_map<RIndex, std::vector<VoxelData>> _all_voxels;

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

    /// Resolve every dependent material's physical kind and scalar data via the unified
    /// material manager's query interface. Populates `_mat_gid_to_name`,
    /// `_mat_phys_type_map`, and `_mat_props`.
    void _resolveMaterials();

    void _addBody(const rsc::RSC_StructureVoxelBody &body_rsc);

    void _addConstraint(const rsc::RSC_StructureVoxelConstraint &constraint_rsc);

    void _addMaterials();

    void _addSignals();

    ConstraintJointData _createBasicConstraintJoint(
        const rsc::RSC_StructureVoxelConstraint &constraint_rsc,
        logic::RSE_ConstraintJointType constraint_type,
        RIndex joint_sid
    );

    void _colorConstraintJoints();

    void _checkStructureLimits() const;

    void _checkRigidBodyInertia() const;

    template <RSE_FloatType FloatMemAccuracy>
    KernelStructureData<FloatMemAccuracy> _getAll() const;

    void _initKernelData();

    void _freeKernelData();

    void _initKernelRecordData();

    void _freeKernelRecordData();

    void _initKernelSignalData();

    void _freeKernelSignalData();

    /// Set this structure's entry in `host_data.structure_emissive_voxel_handles_map`
    /// to the voxel handles of its emissive voxels, then rebuild
    /// `ctx.structure_emissive_voxel_handles` by concatenating and sorting all
    /// map entries.
    void _initKernelEmissiveVoxelHandlesData(RSize voxel_offset);

    /// Erase this structure's entry from
    /// `host_data.structure_emissive_voxel_handles_map`, then rebuild
    /// `ctx.structure_emissive_voxel_handles` by concatenating and sorting the
    /// remaining map entries.
    void _freeKernelEmissiveVoxelHandlesData();

    /// Shared body of freeze()/unfreeze(): flip every voxel and rigid body to is_static (or
    /// restore to is_fixed) and toggle the compute, signal and record layouts. The voxel and
    /// voxel material layouts stay active so the structure remains collidable and rendered.
    void _freezeKernelData(bool is_freeze);

    static logic::RSE_ConstraintJointHingeControlMode _convertToConstraintJointHingeControlMode(
        rsc::RSE_StructureVoxelHingeControlMode mode
    );

    static logic::RSE_ConstraintJointPrismaticControlMode
    _convertToConstraintJointPrismaticControlMode(rsc::RSE_StructureVoxelPrismaticControlMode mode);

    static std::vector<RIndex>
    _selectSID(const std::string &structure_name, RSize select_num, RSize range);

    static std::vector<Cuboid> _partitionSpace(RSize x_size, RSize y_size, RSize z_size);

    static RBound3f _computeWorldBound(const SupportedRSC &structure_rsc);
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
