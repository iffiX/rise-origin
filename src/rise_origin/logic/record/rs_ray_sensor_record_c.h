// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_RAY_SENSOR_RECORD_C_H
#define RS_RAY_SENSOR_RECORD_C_H

#include "rise_origin/device/data/rs_value.h"
#include "rise_origin/device/rs_compute.h"
#include "rise_origin/logic/ray/rs_ray_sensor_m.h"
#include "rise_origin/logic/record/rs_ray_sensor_record_m.h"

namespace rise {

namespace logic {

class RSCF_StoreRaySensorStateRecord
    : public device::RS_ComputeFunction<RSCF_StoreRaySensorStateRecord> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, RS_RaySensorRecordMem &record, RHandle ray_sensor, RIndex record_index) {
        if (k != nullptr && ray_sensor != RS_NULL_HANDLE && record.is_valid) {
            auto &ctx = k->ctx;
            auto &r = record.sensor_state;
            r.position[record_index] = RAY_SENSOR_G(position);
            r.orientation[record_index] = RAY_SENSOR_G(orientation);
        }
    }
};

class RSCF_UpdateRaySensorRootRecord
    : public device::RS_ComputeFunction<RSCF_UpdateRaySensorRootRecord> {
   public:
    template <typename Kernel, typename Buffer, typename... Sizes>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(const Kernel *k, Buffer &buffer, RS_RaySensorRecordMem &record, Sizes... record_sizes) {
        RBigSize record_data_size = record.getRequiredMemorySize(record_sizes...);
        RS_MemoryPtr chunk = buffer.pushDevice(record_data_size);
        if (chunk == nullptr) {
            record.is_valid = false;
        } else {
            record.initFromMemory(chunk.ptr(), record_sizes...);
            *record.time = k->getDeviceTime();
            *record.step = k->getDeviceStep();
        }
    }
};

}  // namespace logic

}  // namespace rise

#endif  // RS_RAY_SENSOR_RECORD_C_H
