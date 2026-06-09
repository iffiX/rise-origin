// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_KERNEL_STRUCTURE_MANAGER_H
#define RS_KERNEL_STRUCTURE_MANAGER_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/logic/rigid/rs_rigid_body_m.h"
#include "rise_origin/logic/rigid/rs_rigid_body_surface_voxel_m.h"
#include "rise_origin/logic/soft/rs_link_m.h"
#include "rise_origin/logic/soft/rs_link_material_m.h"
#include "rise_origin/logic/soft/rs_voxel_m.h"
#include "rise_origin/logic/soft/rs_voxel_material_m.h"
#include "rise_origin/rsc/rs_structure_config.h"
#include "rise_origin/utils/rs_bound.h"
#include "rise_origin/utils/rs_tuple_hash.h"

namespace rise {

class RS_KernelManager;

class RS_KernelStructureManager {
   public:
    RS_KernelStructureManager();

    void init(RS_KernelManager *kernel_manager, const RS_StructureConfig &config);

    void free();

    void updateMaterials();

    bool hasControllerSignal();

    void syncControllerSignal();

    RS_SimulationStructureSignal getControllerSignal();

    const RS_StructureConfig &getConfig() const;

    const std::unordered_set<RIndex> &getDependentMaterialGID() const;

    std::string getName() const;

    RSize getVoxelNum() const;

    RSize getEmissiveVoxelNum() const;

    RSize getLinkNum() const;

    RSize getRigidBodyNum() const;

    RSize getJointNum() const;

    RSize getMotorNum() const;

    RSize getConstraintJointNum() const;

    RSize getConstraintJointColorNum() const;

    RSize getExpansionSignalNum() const;

    RSize getHingeSignalNum() const;

    RSize getPrismaticSignalNum() const;

    RIndex findRigidBodySID(RIndex body_sid, RIndex segment_bid) const;

    RVec3rf getRigidBodyPositionFromRelativeAnchor(
        RIndex body_sid,
        RIndex rigid_body_sid,
        const RVec3rf &relative_anchor
    ) const;

    bool getSegmentType(RIndex body_sid, RIndex segment_bid) const;

    RSize getSegmentVoxelNumInRange(
        RIndex body_sid,
        RIndex segment_bid,
        const RBound3ri &voxel_range
    ) const;

   private:
    /// Per-material properties resolved once from the material config.
    struct MaterialResolvedData {
        bool is_fixed;
        bool is_rigid;
        bool is_transparent;
        bool is_emissive;
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

        // Index of the actual voxel data in body config array
        RIndex config_index;

        // Global scope ID of the material
        RIndex material_gid;

        // Structure scope ID of the expansion signal
        RIndex expansion_signal_sid;

        // Expansion control mode for this voxel
        RSE_StructureExpansionControlMode expansion_control_mode;

        // Global scope position
        RVec3rf g_position;

        // Global scope orientation
        RQuat3rf g_orientation;

        // Body scope position
        RVec3rf b_position;

        // Is a surface voxel of the corresponding segment
        bool is_surface;

        // Is a fixed voxel, position and orientation won't be updated
        bool is_fixed;

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
            data.config_index = RS_NULL_INDEX;
            data.material_gid = RS_NULL_INDEX;
            data.expansion_signal_sid = RS_NULL_INDEX;
            data.expansion_control_mode = RSE_STRUCTURE_EXPANSION_CONTROL_MODE_BASIC_POSITION;
            data.g_position = RVec3rf::zero();
            data.g_orientation = RQuat3rf::unit();
            data.b_position = RVec3rf::zero();
            data.is_surface = false;
            data.role_state = 0;
            data.is_fixed = false;

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

        // Global scope ID of the material (only valid for tendon links)
        RIndex material_gid;

        // Global scope ID of the negative voxel material
        RIndex voxel_neg_material_gid;

        // Global scope ID of the positive voxel material
        RIndex voxel_pos_material_gid;

        RSE_LinkAxis axis;

        bool is_tendon;

        // Default constructor (doesn't do anything to speed up vector initialization)
        LinkData() {}

        // Static method to get preset values
        static LinkData preset() {
            LinkData data;
            data.sid = RS_NULL_INDEX;
            data.body_sid = RS_NULL_INDEX;
            data.voxel_neg_sid = RS_NULL_INDEX;
            data.voxel_pos_sid = RS_NULL_INDEX;
            data.material_gid = RS_NULL_INDEX;
            data.voxel_neg_material_gid = RS_NULL_INDEX;
            data.voxel_pos_material_gid = RS_NULL_INDEX;
            data.axis = RSE_NO_AXIS;
            data.is_tendon = false;
            return data;
        }
    };

    class VoxelMaterialData {
       public:
        // Global scope ID of the material
        RIndex gid;

        // Default constructor (doesn't do anything to speed up vector initialization)
        VoxelMaterialData() {}

        // Static method to get preset values
        static VoxelMaterialData preset() {
            VoxelMaterialData data;
            data.gid = RS_NULL_INDEX;
            return data;
        }
    };

    class LinkMaterialData {
       public:
        // Negative voxel material global scope ID
        RIndex neg_gid;

        // Positive voxel material global scope ID
        RIndex pos_gid;

        // Default constructor (doesn't do anything to speed up vector initialization)
        LinkMaterialData() {}

        // Static method to get preset values
        static LinkMaterialData preset() {
            LinkMaterialData data;
            data.neg_gid = RS_NULL_INDEX;
            data.pos_gid = RS_NULL_INDEX;
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

        // Bitmask of rigid bodies (by sid) this body must not collide with: its own bit plus
        // every body joined to it by a constraint. Copied to RS_RigidBodyMem::connected_body_mask.
        RBigIndex connected_body_mask = 0;

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
        int hinge_control_mode;

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
        int prismatic_control_mode;

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
            data.hinge_control_mode = RSE_STRUCTURE_HINGE_JOINT_CONTROL_MODE_BASIC_POSITION;
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
            data.prismatic_control_mode = 0;
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
    };

    template <RSE_FloatType FloatMemAccuracy>
    class KernelStructureRigidBodyUpdateData {
       public:
        device::RS_Array<RS_FloatMem<FloatMemAccuracy>, 1> mass;
        device::RS_Array<RS_FloatMem<FloatMemAccuracy>, 1> mass_inverses;
        device::RS_Array<RS_Vec3DMem<FloatMemAccuracy>, 1> com_rb_positions;
        device::RS_Array<RS_Mat3DMem<FloatMemAccuracy>, 1> local_moment_inertia_inverses;
    };

    RS_KernelManager *_kernel_manager = nullptr;

    RIndex _structure_gid = RS_NULL_INDEX;
    RS_StructureConfig _config;
    std::vector<RIndex> _dependent_material_gid;
    std::unordered_set<RIndex> _dependent_material_gid_set;

    bool _is_kernel_data_inited = false;

    /// Counters
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

    /// Maps (body_sid, segment_bid) -> is_rigid (derived from material config)
    std::map<RS_OrderedPair<RIndex, RIndex>, bool> _segment_type_map;

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

    // For device side, the signals directly point to the
    // corresponding signal array region in context
    RSize _device_expansion_signal_offset = 0;
    RSize _device_hinge_signal_offset = 0;
    RSize _device_prismatic_signal_offset = 0;
    RS_MemoryPtr _host_expansion_signal_buffer;
    RS_MemoryPtr _host_hinge_signal_buffer;
    RS_MemoryPtr _host_prismatic_signal_buffer;

    void _addBody(const RS_StructureBodyConfig &body_config);

    void _addConstraint(const RS_StructureConstraintConfig &constraint_config);

    void _addTendon(const RS_StructureTendonConfig &tendon_config);

    void _addMaterials();

    void _addSignals();

    ConstraintJointData _createBasicConstraintJoint(
        const RS_StructureConstraintConfig &constraint_config,
        logic::RSE_ConstraintJointType constraint_type,
        RIndex joint_sid
    );

    void _colorConstraintJoints();

    template <RSE_FloatType FloatMemAccuracy>
    KernelStructureData<FloatMemAccuracy> _getAll() const;

    template <RSE_FloatType FloatMemAccuracy>
    KernelStructureData<FloatMemAccuracy> _getMaterials() const;

    template <RSE_FloatType FloatMemAccuracy>
    KernelStructureRigidBodyUpdateData<FloatMemAccuracy> _getRigidBodyUpdate() const;

    void _initKernelData();

    void _freeKernelData();

    void _updateMaterialKernelData();

    /// Set this structure's entry in `host_data.structure_emissive_voxel_handles_map`
    /// to the voxel handles of its emissive voxels, then rebuild
    /// `ctx.structure_emissive_voxel_handles` by concatenating and sorting all
    /// map entries.
    void _initEmissiveVoxelHandlesKernelData(RSize voxel_offset);

    /// Erase this structure's entry from
    /// `host_data.structure_emissive_voxel_handles_map`, then rebuild
    /// `ctx.structure_emissive_voxel_handles` by concatenating and sorting the
    /// remaining map entries.
    void _freeEmissiveVoxelHandlesKernelData();

    static std::vector<Cuboid> _partitionSpace(RSize x_size, RSize y_size, RSize z_size);

    static RBound3f _computeConfigWorldBound(const RS_StructureConfig &config);
};

}  // namespace rise

#endif  // RS_KERNEL_STRUCTURE_MANAGER_H
