// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_BASIC_RECORD_C_H
#define RS_BASIC_RECORD_C_H

#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/logic/soft/rs_link_c.h"
#include "rise_origin/logic/soft/rs_link_m.h"
#include "rise_origin/logic/soft/rs_voxel_c.h"
#include "rise_origin/logic/soft/rs_voxel_m.h"
#include "rise_origin/logic/vision/rs_camera_m.h"

namespace rise {

namespace logic {

class RSCF_ResolveRecordDependency
    : public device::RS_ComputeFunction<RSCF_ResolveRecordDependency> {
   public:
    HOST_DEVICE_INLINE_QUALIFIER static void run(
        RSize src_data_segment_offset,
        device::RS_Array<RIndex, 1> sid_in,
        device::RS_Array<RHandle, 1> handles_out
    ) {
        RIndex i = index1D();
        handles_out[i] = sid_in[i] + src_data_segment_offset;
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_BASIC_RECORD_C_H
