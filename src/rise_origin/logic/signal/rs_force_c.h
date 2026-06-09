// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_FORCE_C_H
#define RS_FORCE_C_H

#include "rise_origin/logic/rigid/rs_rigid_body_c.h"
#include "rise_origin/logic/signal/rs_force_m.h"
#include "rise_origin/logic/soft/rs_voxel_c.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

template <RSE_FloatType FloatComputeAccuracy>
class RS_ForceCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using FI = RS_FloatIntrinsics<FloatC>;

    /**
     * @brief Compute function to update rigid force memory application time.
     *
     * Updates the last_force_signal_apply_time for a rigid force memory entry.
     */
    class RSCF_UpdateRigidForceTime : public device::RS_ComputeFunction<RSCF_UpdateRigidForceTime> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_UpdateRigidForceTime>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k) {
            if (k != nullptr) {
                auto &ctx = k->ctx;
                RIndex i = BaseType::index1D();

                if (i < ctx.force_rigid.size()) {
                    auto &force_mem = ctx.force_rigid[i];
                    RBigFloat current_time = k->getDeviceTime();
                    force_mem.last_force_signal_apply_time = current_time;
                }
            }
        }
    };

    /**
     * @brief Compute function to update soft force memory application time.
     *
     * Updates the last_force_signal_apply_time for a soft force memory entry.
     */
    class RSCF_UpdateSoftForceTime : public device::RS_ComputeFunction<RSCF_UpdateSoftForceTime> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_UpdateSoftForceTime>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k) {
            if (k != nullptr) {
                auto &ctx = k->ctx;
                RIndex i = BaseType::index1D();

                if (i < ctx.force_soft.size()) {
                    auto &force_mem = ctx.force_soft[i];
                    RBigFloat current_time = k->getDeviceTime();
                    force_mem.last_force_signal_apply_time = current_time;
                }
            }
        }
    };

    /**
     * @brief Compute function to apply force signals to rigid bodies.
     *
     * Iterates through all rigid force memory entries and applies impulses
     * to matching rigid bodies based on structure_gid and rigid_body_sid filters.
     */
    class RSCF_ApplyRigidForceSignal
        : public device::RS_ComputeFunction<RSCF_ApplyRigidForceSignal> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_ApplyRigidForceSignal>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle rigid_body) {
            if (k != nullptr && rigid_body != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;

                // Fixed rigid body is an immovable anchor, skip
                if (RS_RigidBodyStateUnion{RB_G(state)}.rigid_body_state.is_fixed) return;

                RBigFloat current_time = k->getDeviceTime();

                // Get rigid body structure gid and sid
                RIndex rb_structure_gid = RB_G(structure_gid);
                RIndex rb_sid = RB_G(sid);

                // Iterate through all rigid force memories
                RSize force_count = ctx.force_rigid.size();
                for (RIndex i = 0; i < force_count; i++) {
                    auto &force_mem = ctx.force_rigid[i];

                    // Check if this force applies to this rigid body
                    if (force_mem.structure_gid == rb_structure_gid &&
                        force_mem.rigid_body_sid == rb_sid) {
                        // Calculate time delta since last application
                        RBigFloat dt = current_time - force_mem.last_force_signal_apply_time;
                        if (dt <= 0.0) continue;  // Skip if already applied at this time

                        // Read force signal (3 RFloat values: x, y, z)
                        RHandle signal_handle = force_mem.force_signal;
                        FloatC force_x =
                            static_cast<FloatC>(ctx.force_signals[signal_handle * 3 + 0]);
                        FloatC force_y =
                            static_cast<FloatC>(ctx.force_signals[signal_handle * 3 + 1]);
                        FloatC force_z =
                            static_cast<FloatC>(ctx.force_signals[signal_handle * 3 + 2]);

                        // Compute impulse = force * unit_portion * dt
                        FloatC portion = static_cast<FloatC>(force_mem.unit_portion);
                        FloatC dt_f = static_cast<FloatC>(dt);
                        Vec3C impulse = Vec3C(force_x, force_y, force_z) * portion * dt_f;

                        // Apply impulse to linear momentum
                        Vec3C linear_momentum = RB_G(linear_momentum);
                        linear_momentum += impulse;
                        RB_S(linear_momentum, linear_momentum);
                    }
                }
            }
        }
    };

    /**
     * @brief Compute function to apply force signals to soft voxels.
     *
     * Iterates through all soft force memory entries and applies impulses
     * to matching voxels based on structure_gid, voxel_range, body_sid, and segment_bid filters.
     */
    class RSCF_ApplySoftForceSignal : public device::RS_ComputeFunction<RSCF_ApplySoftForceSignal> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_ApplySoftForceSignal>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle voxel) {
            if (k != nullptr && voxel != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;

                RBigFloat current_time = k->getDeviceTime();

                // Get voxel attributes
                RIndex v_structure_gid = V_G(structure_gid);
                RIndex v_body_sid = V_G(body_sid);
                RIndex v_segment_bid = V_G(segment_bid);
                RSignedIndex v_index_x = V_G(index_x);
                RSignedIndex v_index_y = V_G(index_y);
                RSignedIndex v_index_z = V_G(index_z);

                // Check if this is a soft voxel (not part of a rigid body)
                RIndex v_rigid_body_sid = V_G(rigid_body_sid);
                if (v_rigid_body_sid != RS_NULL_INDEX) {
                    return;  // Skip rigid voxels
                }

                // Iterate through all soft force memories
                RSize force_count = ctx.force_soft.size();
                for (RIndex i = 0; i < force_count; i++) {
                    auto &force_mem = ctx.force_soft[i];

                    // Check if this force applies to this voxel
                    if (force_mem.structure_gid != v_structure_gid) continue;
                    if (force_mem.body_sid != v_body_sid) continue;
                    if (force_mem.segment_bid != v_segment_bid) continue;

                    // Check if voxel is within range
                    if (v_index_x < force_mem.voxel_range.min_x ||
                        v_index_x > force_mem.voxel_range.max_x)
                        continue;
                    if (v_index_y < force_mem.voxel_range.min_y ||
                        v_index_y > force_mem.voxel_range.max_y)
                        continue;
                    if (v_index_z < force_mem.voxel_range.min_z ||
                        v_index_z > force_mem.voxel_range.max_z)
                        continue;

                    // Calculate time delta since last application
                    RBigFloat dt = current_time - force_mem.last_force_signal_apply_time;
                    if (dt <= 0.0) continue;  // Skip if already applied at this time

                    // Read force signal (3 RFloat values: x, y, z)
                    RHandle signal_handle = force_mem.force_signal;
                    FloatC force_x = static_cast<FloatC>(ctx.force_signals[signal_handle * 3 + 0]);
                    FloatC force_y = static_cast<FloatC>(ctx.force_signals[signal_handle * 3 + 1]);
                    FloatC force_z = static_cast<FloatC>(ctx.force_signals[signal_handle * 3 + 2]);

                    // Compute impulse = force * unit_portion * dt
                    FloatC portion = static_cast<FloatC>(force_mem.unit_portion);
                    FloatC dt_f = static_cast<FloatC>(dt);
                    Vec3C impulse = Vec3C(force_x, force_y, force_z) * portion * dt_f;

                    // Apply impulse to linear momentum
                    Vec3C linear_momentum = V_G(linear_momentum);
                    linear_momentum += impulse;
                    V_S(linear_momentum, linear_momentum);
                }
            }
        }
    };
};

}  // namespace logic

}  // namespace rise

#endif  // RS_FORCE_C_H
