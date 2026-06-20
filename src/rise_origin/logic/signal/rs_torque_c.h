// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_TORQUE_C_H
#define RS_TORQUE_C_H

#include "rise_origin/logic/rigid/rs_rigid_body_c.h"
#include "rise_origin/logic/signal/rs_torque_m.h"
#include "rise_origin/utils/rs_def.h"

namespace rise {

namespace logic {

template <RSE_FloatType FloatComputeAccuracy>
class RS_TorqueCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using FI = RS_FloatIntrinsics<FloatC>;

    /**
     * @brief Compute function to update rigid torque memory application time.
     *
     * Updates the last_torque_signal_apply_time for a rigid torque memory entry.
     */
    class RSCF_UpdateRigidTorqueTime
        : public device::RS_ComputeFunction<RSCF_UpdateRigidTorqueTime> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_UpdateRigidTorqueTime>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k) {
            if (k != nullptr) {
                auto &ctx = k->ctx;
                RIndex i = BaseType::index1D();

                if (i < ctx.torque_rigid.size()) {
                    auto &torque_mem = ctx.torque_rigid[i];
                    RBigFloat current_time = k->getDeviceTime();
                    torque_mem.last_torque_signal_apply_time = current_time;
                }
            }
        }
    };

    /**
     * @brief Compute function to apply torque signals to rigid bodies.
     *
     * Iterates through all rigid torque memory entries and applies angular
     * impulses to matching rigid bodies based on structure_gid and
     * rigid_body_sid filters.
     */
    class RSCF_ApplyRigidTorqueSignal
        : public device::RS_ComputeFunction<RSCF_ApplyRigidTorqueSignal> {
       public:
        using BaseType = device::RS_ComputeFunction<RSCF_ApplyRigidTorqueSignal>;

        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel *k, RHandle rigid_body) {
            if (k != nullptr && rigid_body != RS_NULL_HANDLE) {
                auto &ctx = k->ctx;

                // Immobile body from fixed material or freeze takes no applied torque, skip
                if (RS_RigidBodyStateUnion{RB_G(state)}.rigid_body_state.is_static) return;

                RBigFloat current_time = k->getDeviceTime();

                // Get rigid body structure gid and sid
                RIndex rb_structure_gid = RB_G(structure_gid);
                RIndex rb_sid = RB_G(sid);

                // Iterate through all rigid torque memories
                RSize torque_count = ctx.torque_rigid.size();
                for (RIndex i = 0; i < torque_count; i++) {
                    auto &torque_mem = ctx.torque_rigid[i];

                    // Check if this torque applies to this rigid body
                    if (torque_mem.structure_gid == rb_structure_gid &&
                        torque_mem.rigid_body_sid == rb_sid) {
                        // Calculate time delta since last application
                        RBigFloat dt = current_time - torque_mem.last_torque_signal_apply_time;
                        if (dt <= 0.0) continue;  // Skip if already applied at this time

                        // Read torque signal (one 3D vector entry).
                        RHandle signal_handle = torque_mem.torque_signal;
                        Vec3C torque =
                            Vec3C(ctx.torque_signals[signal_handle].torque_signals.torque[0]);

                        // Compute angular impulse = torque * dt
                        FloatC dt_f = static_cast<FloatC>(dt);
                        Vec3C angular_impulse = torque * dt_f;

                        // Apply angular impulse to angular momentum
                        Vec3C angular_momentum = RB_G(angular_momentum);
                        angular_momentum += angular_impulse;
                        RB_S(angular_momentum, angular_momentum);
                    }
                }
            }
        }
    };
};

}  // namespace logic

}  // namespace rise

#endif  // RS_TORQUE_C_H
