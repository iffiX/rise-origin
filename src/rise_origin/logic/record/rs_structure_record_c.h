// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_STRUCTURE_RECORD_C_H
#define RS_STRUCTURE_RECORD_C_H

#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/logic/soft/rs_link_c.h"
#include "rise_origin/logic/soft/rs_link_m.h"
#include "rise_origin/logic/soft/rs_voxel_c.h"
#include "rise_origin/logic/soft/rs_voxel_m.h"

namespace rise {

namespace logic {

class RSCF_StoreStructureVoxelRecord
    : public device::RS_ComputeFunction<RSCF_StoreStructureVoxelRecord> {
   public:
    using VoxelCompute = RS_VoxelCompute<RS_DEFAULT_FLOAT_TYPE>;

    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_WorldStructureVoxelRecord &record, RHandle voxel, RIndex record_index) {
        if (k != nullptr && voxel != RS_NULL_HANDLE && record.is_valid) {
            auto &ctx = k->ctx;
            auto &r = record.voxels;

            RS_VoxelStateUnion voxel_state_union{V_G(state)};
            RHandle voxel_material = V_G(voxel_material);
            RVec3rf linear_velocity = VoxelCompute::linearVelocity(
                V_G(linear_momentum).template toVec3D<RFloat>(), VM_G(mass_inverse)
            );
            RVec3rf angular_velocity =
                VoxelCompute::angularVelocity(V_G(angular_momentum), VM_G(moment_inertia_inverse));

            auto links = V_G(links);
            uint8_t connectivity = (links[0] != RS_NULL_HANDLE) & 1;
            connectivity += ((links[1] != RS_NULL_HANDLE) & 1) << 1;
            connectivity += ((links[2] != RS_NULL_HANDLE) & 1) << 2;
            connectivity += ((links[3] != RS_NULL_HANDLE) & 1) << 3;
            connectivity += ((links[4] != RS_NULL_HANDLE) & 1) << 4;
            connectivity += ((links[5] != RS_NULL_HANDLE) & 1) << 5;

            r.body_sid[record_index] = V_G(body_sid);
            r.voxel_sid[record_index] = V_G(sid);
            r.segment_bid[record_index] = V_G(segment_bid);
            r.index_x[record_index] = V_G(index_x);
            r.index_y[record_index] = V_G(index_y);
            r.index_z[record_index] = V_G(index_z);
            r.connectivity[record_index] = connectivity;
            r.is_surface[record_index] = voxel_state_union.voxel_state.is_surface;
            r.is_rigid[record_index] = voxel_state_union.voxel_state.is_rigid;
            r.material[record_index] = VM_G(material_gid);
            r.expansion[record_index] = V_G(expansion);
            r.orientation[record_index] = V_G(orientation).template toQuat3D<RFloat>();
            r.position[record_index] = V_G(position).template toVec3D<RFloat>();
            r.ppp_offset[record_index] = V_G(ppp_offset).template toVec3D<RFloat>();
            r.nnn_offset[record_index] = V_G(nnn_offset).template toVec3D<RFloat>();
            r.linear_velocity[record_index] = linear_velocity;
            r.angular_velocity[record_index] = angular_velocity;
            r.poissons_strain[record_index] = V_G(poissons_strain).template toVec3D<RFloat>();
        }
    }
};

class RSCF_StoreStructureLinkRecord
    : public device::RS_ComputeFunction<RSCF_StoreStructureLinkRecord> {
   public:
    using LinkCompute = RS_LinkCompute<RS_DEFAULT_FLOAT_TYPE>;

    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_WorldStructureVoxelRecord &record, RHandle link, RIndex record_index) {
        if (k != nullptr && link != RS_NULL_HANDLE && record.is_valid) {
            auto &ctx = k->ctx;
            auto &r = record.links;

            RS_LinkStateUnion link_state_union{L_G(state)};
            r.voxel_neg_sid[record_index] = L_G(voxel_neg_sid);
            r.voxel_pos_sid[record_index] = L_G(voxel_pos_sid);
            r.is_failed[record_index] = link_state_union.link_state.is_failed;

            r.strain[record_index] = L_G(strain);
            r.stress[record_index] = L_G(stress);
        }
    }
};

class RSCF_StoreStructureRigidBodyRecord
    : public device::RS_ComputeFunction<RSCF_StoreStructureRigidBodyRecord> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        const Kernel *k,
        RS_WorldStructureVoxelRecord &record,
        RHandle rigid_body,
        RIndex record_index
    ) {
        if (k != nullptr && rigid_body != RS_NULL_HANDLE && record.is_valid) {
            auto &ctx = k->ctx;
            auto &r = record.rigid_bodies;

            RFloat mass = RB_G(mass);
            RMat3rf moment_inertia_inverse = RB_G(moment_inertia_inverse);
            RVec3rf angular_momentum = RB_G(angular_momentum);
            RVec3rf linear_momentum = RB_G(linear_momentum);

            r.rigid_body_sid[record_index] = RB_G(sid);
            r.body_sid[record_index] = RB_G(body_sid);
            r.segment_bid[record_index] = RB_G(segment_bid);
            r.mass[record_index] = mass;
            r.orientation[record_index] = RB_G(orientation).template toQuat3D<RFloat>();
            r.com[record_index] = RB_G(com);
            r.linear_velocity[record_index] = linear_momentum / mass;
            r.angular_velocity[record_index] = moment_inertia_inverse * angular_momentum;
        }
    }
};

class RSCF_StoreStructureJointRecord
    : public device::RS_ComputeFunction<RSCF_StoreStructureJointRecord> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        const Kernel *k,
        RS_WorldStructureVoxelRecord &record,
        RHandle constraint_joint,
        RIndex record_index
    ) {
        if (k != nullptr && constraint_joint != RS_NULL_HANDLE && record.is_valid) {
            auto &ctx = k->ctx;
            auto &r = record.joints;

            RS_ConstraintJointStateUnion state_union{CJ_G(state)};
            RSE_ConstraintJointType type = state_union.constraint_joint_state.type;
            RIndex joint_sid = CJ_G(joint_sid);

            if (type == RSE_BALL_AND_SOCKET_JOINT_TRANSLATION) {
                r.rigid_body_a_sid[joint_sid] = CJ_G(rigid_body_a_sid);
                r.rigid_body_b_sid[joint_sid] = CJ_G(rigid_body_b_sid);
                r.type[joint_sid] = RS_WorldStructureRecordBase::JointType::BALL_AND_SOCKET;
                r.position[joint_sid] = CJ_G(position);
                r.axis[joint_sid] = RVec3rf::zero();
            } else if (type == RSE_FIXED_JOINT_TRANSLATION) {
                r.rigid_body_a_sid[joint_sid] = CJ_G(rigid_body_a_sid);
                r.rigid_body_b_sid[joint_sid] = CJ_G(rigid_body_b_sid);
                r.type[joint_sid] = RS_WorldStructureRecordBase::JointType::FIXED;
                r.position[joint_sid] = CJ_G(position);
                r.axis[joint_sid] = RVec3rf::zero();
            } else if (type == RSE_HINGE_JOINT_TRANSLATION) {
                r.rigid_body_a_sid[joint_sid] = CJ_G(rigid_body_a_sid);
                r.rigid_body_b_sid[joint_sid] = CJ_G(rigid_body_b_sid);
                r.hinge_signal_sid[joint_sid] = CJ_G(hinge_signal_sid);
                r.type[joint_sid] = RS_WorldStructureRecordBase::JointType::HINGE;
                r.position[joint_sid] = CJ_G(position);
            } else if (type == RSE_HINGE_JOINT_ROTATION) {
                r.axis[joint_sid] = CJ_G(axis);
            } else if (type == RSE_HINGE_JOINT_LIMIT) {
                r.hinge_min[joint_sid] = CJ_G(hinge_min);
                r.hinge_max[joint_sid] = CJ_G(hinge_max);
                r.hinge_angle[joint_sid] = CJ_G(angle);
            } else if (type == RSE_HINGE_JOINT_MOTOR) {
                RFloat dt = k->getDeviceTimeSinceLastOperation(
                    RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_CONSTRAINT_AND_CONTACT
                );
                r.hinge_torque[joint_sid] = dt > 0 ? CJ_G(hinge_motor_impulse) / dt : RFLOAT(0);
            } else if (type == RSE_PRISMATIC_JOINT_PERPENDICULAR_TRANSLATION) {
                r.rigid_body_a_sid[joint_sid] = CJ_G(rigid_body_a_sid);
                r.rigid_body_b_sid[joint_sid] = CJ_G(rigid_body_b_sid);
                r.prismatic_signal_sid[joint_sid] = CJ_G(prismatic_signal_sid);
                r.type[joint_sid] = RS_WorldStructureRecordBase::JointType::PRISMATIC;
                r.position[joint_sid] = CJ_G(position);
                // Compute slide axis for recording
                RQuat3rf a_orientation(RB_G(CJ_G(rigid_body_a), orientation));
                r.axis[joint_sid] =
                    a_orientation.rotateVec3D(RVec3rf(CJ_G(prismatic_a_rb_axis))).normalized();
            } else if (type == RSE_PRISMATIC_JOINT_ROTATION) {
                // axis already set by perpendicular translation branch
            } else if (type == RSE_PRISMATIC_JOINT_LIMIT) {
                r.prismatic_min[joint_sid] = CJ_G(prismatic_min);
                r.prismatic_max[joint_sid] = CJ_G(prismatic_max);
                r.prismatic_distance[joint_sid] = CJ_G(prismatic_distance);
            } else if (type == RSE_PRISMATIC_JOINT_MOTOR) {
                RFloat dt = k->getDeviceTimeSinceLastOperation(
                    RSE_KERNEL_UPDATE_OPERATION_RIGID_BODY_CONSTRAINT_AND_CONTACT
                );
                r.prismatic_force[joint_sid] = dt > 0 ? CJ_G(prismatic_motor_impulse) / dt
                                                      : RFLOAT(0);
            }
        }
    }
};

class RSCF_UpdateStructureRootRecord
    : public device::RS_ComputeFunction<RSCF_UpdateStructureRootRecord> {
   public:
    template <typename Kernel, typename Buffer, typename... Sizes>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, Buffer &buffer, RS_WorldStructureVoxelRecord &record, Sizes... record_sizes) {
        RBigSize record_data_size = RS_WorldStructureVoxelRecord::getRequiredMemorySize(record_sizes...);
        RS_MemoryPtr chunk = buffer.pushDevice(record_data_size);
        if (chunk == nullptr) {
            record.is_valid = false;
        } else {
            record.is_valid = true;
            record.initFromMemory(chunk.ptr(), record_sizes...);
            *record.time = k->getDeviceTime();
            *record.step = k->getDeviceStep();
        }
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_STRUCTURE_RECORD_C_H
