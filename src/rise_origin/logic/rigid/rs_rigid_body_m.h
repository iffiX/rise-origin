// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RIGID_BODY_M_H
#define RS_RIGID_BODY_M_H

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_float.h"
#include "rise_origin/utils/rs_mat.h"
#include "rise_origin/utils/rs_quat.h"
#include "rise_origin/utils/rs_vec.h"

namespace rise {

namespace logic {

union RS_RigidBodyStateUnion {
    RState rigid_body_state_packed;
    struct RS_RigidBodyState {
        /// Whether this rigid body is fixed in place
        RState is_fixed : 1;
    } rigid_body_state;
};

/**
 * Memory layout for a rigid body object.
 *
 * \tparam FloatMemAccuracy One of RSE_FloatType (RSE_FLOAT_32, RSE_FLOAT_64)
 */
template <RSE_FloatType FloatMemAccuracy>
class RS_RigidBodyMem {
   public:
    using FloatM = RS_FloatMem<FloatMemAccuracy>;
    using Vec3M = RS_Vec3DMem<RS_FloatCapAccuracy<FloatMemAccuracy, RSE_FLOAT_32>::Type>;
    using Quat3M = RS_Quat3DMem<RS_FloatCapAccuracy<FloatMemAccuracy, RSE_FLOAT_32>::Type>;
    using Mat3M = RS_Mat3DMem<RS_FloatCapAccuracy<FloatMemAccuracy, RSE_FLOAT_32>::Type>;

    /// Plain vector for fields written by deviceAtomicAdd in the contact solver
    using Vec3Atomic = RS_Vec3D<typename Vec3M::ElementType>;

    /*****************************************************************
     * Pre-set attributes
     *****************************************************************/
    /// Structure gid of this rigid body
    RIndex structure_gid = RS_NULL_INDEX;

    /// Structure (internal) ID
    RIndex sid = RS_NULL_INDEX;

    /// Body index as specified in config
    RIndex body_sid = RS_NULL_INDEX;

    /// Body segment ID
    RIndex segment_bid = RS_NULL_INDEX;

    /// Initial orientation in global coords
    Quat3M initial_orientation = Quat3M::unit();

    /// Bitmask of rigid bodies by sid) that this body must not collide with
    RBigIndex connected_body_mask = 0;

    /*****************************************************************
     * Post-set attributes
     *****************************************************************/
    /// Packed rigid body state
    RState state = 0;

    /// Mass, in kg
    FloatM mass = 0;

    /// Inverse of the mass (1/mass)
    FloatM mass_inverse = 0;

    /// Inverse of the local moment inertia (in com coords)
    Mat3M local_moment_inertia_inverse = Mat3M::zero();

    /// Inverse of the moment inertia (in global coords)
    Mat3M moment_inertia_inverse = Mat3M::zero();

    /// Center of mass in the rigid body origin coords
    Vec3M com_rb_position = Vec3M::zero();

    /// Center of mass in global coords
    Vec3Atomic com = Vec3Atomic::zero();

    /// Origin in global coords
    Vec3M origin = Vec3M::zero();

    /// Angular orientation in global coords
    Quat3M orientation = Quat3M::unit();

    /// Current linear momentum
    Vec3Atomic linear_momentum = Vec3Atomic::zero();

    /// Current angular momentum
    Vec3Atomic angular_momentum = Vec3Atomic::zero();

    /// Active contact count from the previous rigid step
    RSize contact_count_prev_step = 0;

    /// Active contact count this rigid step
    RSize contact_count_now_step = 0;

    /// Snapshot of com at the start of the contact position-correction loop
    Vec3M com_at_position_correction_start = Vec3M::zero();

    /// Joules of coupling work delivered by the soft since the last reset
    FloatM cumulative_coupling_work = 0;
};

}  // namespace logic

}  // namespace rise

#define REFLECT_RIGID_BODY_MEM(rigid_body_mem_type) \
    REFL_AUTO(                                      \
        type(rigid_body_mem_type),                  \
        field(structure_gid),                       \
        field(sid),                                 \
        field(body_sid),                            \
        field(segment_bid),                         \
        field(initial_orientation),                 \
        field(connected_body_mask),                 \
        field(state),                               \
        field(mass),                                \
        field(mass_inverse),                        \
        field(local_moment_inertia_inverse),        \
        field(moment_inertia_inverse),              \
        field(com_rb_position),                     \
        protected_field(com),                       \
        field(origin),                              \
        field(orientation),                         \
        protected_field(linear_momentum),           \
        protected_field(angular_momentum),          \
        field(contact_count_prev_step),             \
        field(contact_count_now_step),              \
        field(com_at_position_correction_start),    \
        field(cumulative_coupling_work)             \
    )

// Instantiate reflection for specific floating accuracies
REFLECT_RIGID_BODY_MEM(rise::logic::RS_RigidBodyMem<rise::RSE_FLOAT_32>)
REFLECT_RIGID_BODY_MEM(rise::logic::RS_RigidBodyMem<rise::RSE_FLOAT_64>)

#endif  // RS_RIGID_BODY_M_H
