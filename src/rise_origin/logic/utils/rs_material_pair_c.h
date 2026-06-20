// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_MATERIAL_PAIR_C_H
#define RS_MATERIAL_PAIR_C_H

#include "rise_origin/device/data/rs_array.h"
#include "rise_origin/device/rs_compute.h"

namespace rise {

namespace logic {

/**
 * @brief Set friction coefficients for a material pair in the 2D lookup tables.
 *
 * @details Writes symmetric entries: table[gid_a][gid_b] = table[gid_b][gid_a] = value.
 *          Launched with 1 thread.
 */
class RSCF_SetMaterialFriction : public device::RS_ComputeFunction<RSCF_SetMaterialFriction> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        device::RS_Array<float, 2> friction_static_table,
        device::RS_Array<float, 2> friction_dynamic_table,
        RIndex gid_a,
        RIndex gid_b,
        float u_static,
        float u_dynamic
    ) {
        friction_static_table[{(RSize)gid_a, (RSize)gid_b}] = u_static;
        friction_static_table[{(RSize)gid_b, (RSize)gid_a}] = u_static;
        friction_dynamic_table[{(RSize)gid_a, (RSize)gid_b}] = u_dynamic;
        friction_dynamic_table[{(RSize)gid_b, (RSize)gid_a}] = u_dynamic;
    }
};

/**
 * @brief Set restitution coefficient for a material pair in the 2D lookup table.
 *
 * @details Writes symmetric entries: table[gid_a][gid_b] = table[gid_b][gid_a] = value.
 *          Launched with 1 thread.
 */
class RSCF_SetMaterialRestitution : public device::RS_ComputeFunction<RSCF_SetMaterialRestitution> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        device::RS_Array<float, 2> restitution_table,
        RIndex gid_a,
        RIndex gid_b,
        float restitution
    ) {
        restitution_table[{(RSize)gid_a, (RSize)gid_b}] = restitution;
        restitution_table[{(RSize)gid_b, (RSize)gid_a}] = restitution;
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_MATERIAL_PAIR_C_H
