// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_SIMULATION_SIGNAL_H
#define RS_SIMULATION_SIGNAL_H

#include "rise_origin/utils/rs_def.h"
#include "rise_origin/utils/rs_device.h"

namespace rise {

/**
 * @param expansion_signals Pointer to the array of expansion signals applied to voxels of every
 * structure. Interpretation depends on per-body expansion_control_mode.
 * @param hinge_signals Pointer to the array of hinge signals applied to hinge joints of every
 * structure. Interpretation depends on per-constraint hinge_control_mode.
 * @param prismatic_signals Pointer to the array of prismatic signals applied to prismatic joints
 * of every structure. Interpretation depends on per-constraint prismatic_control_mode.
 * @param expansion_signal_num Expansion signal number of every structure.
 * @param hinge_signal_num Hinge signal number of every structure.
 * @param prismatic_signal_num Prismatic signal number of every structure.
 */
class RS_SimulationStructureSignal {
   public:
    RS_DeviceLocation device = {RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX};

    RIndex structure_gid = RS_NULL_INDEX;
    RFloat *expansion_signals = nullptr;
    RFloat *hinge_signals = nullptr;
    RFloat *prismatic_signals = nullptr;
    RSize expansion_signal_num = 0;
    RSize hinge_signal_num = 0;
    RSize prismatic_signal_num = 0;

    RS_SimulationStructureSignal() = default;
    RS_SimulationStructureSignal(
        const RS_DeviceLocation &device,
        RIndex structure_gid,
        RFloat *expansion_signals,
        RFloat *hinge_signals,
        RFloat *prismatic_signals,
        RSize expansion_signal_num,
        RSize hinge_signal_num,
        RSize prismatic_signal_num
    )
        : device(device),
          structure_gid(structure_gid),
          expansion_signals(expansion_signals),
          hinge_signals(hinge_signals),
          prismatic_signals(prismatic_signals),
          expansion_signal_num(expansion_signal_num),
          hinge_signal_num(hinge_signal_num),
          prismatic_signal_num(prismatic_signal_num) {}
};

/**
 * @param position_signals Pointer to the array of position signals applied to cameras. (3d vector
 * position)
 * @param orientation_signals Pointer to the array of orientation signals applied to cameras. (4d
 * vector orientation)
 * @param focus_distance_signals Pointer to the array of focus distance signals applied to cameras.
 * (1d vector focus distance)
 * @param field_of_view_signals Pointer to the array of field of view signals applied to cameras.
 * (1d vector field of view)
 */
class RS_SimulationCameraSignal {
   public:
    RS_DeviceLocation device = {RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX};

    RIndex camera_gid = RS_NULL_INDEX;

    RFloat *position_signals = nullptr;
    RFloat *orientation_signals = nullptr;
    RFloat *focus_distance_signals = nullptr;
    RFloat *field_of_view_signals = nullptr;

    RSize position_num = 0;
    RSize orientation_num = 0;
    RSize focus_distance_num = 0;
    RSize field_of_view_num = 0;

    RS_SimulationCameraSignal() = default;
    RS_SimulationCameraSignal(
        const RS_DeviceLocation &device,
        RIndex camera_gid,
        RFloat *position_signals,
        RFloat *orientation_signals,
        RFloat *focus_distance_signals,
        RFloat *field_of_view_signals,
        RSize position_num,
        RSize orientation_num,
        RSize focus_distance_num,
        RSize field_of_view_num
    )
        : device(device),
          camera_gid(camera_gid),
          position_signals(position_signals),
          orientation_signals(orientation_signals),
          focus_distance_signals(focus_distance_signals),
          field_of_view_signals(field_of_view_signals),
          position_num(position_num),
          orientation_num(orientation_num),
          focus_distance_num(focus_distance_num),
          field_of_view_num(field_of_view_num) {}
};

/**
 * @param force_signals Pointer to the array of force signals (3D vector force, size 3 * RFloat).
 */
class RS_SimulationForceSignal {
   public:
    RS_DeviceLocation device = {RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX};

    RIndex structure_gid = RS_NULL_INDEX;

    RFloat *force_signals = nullptr;

    RS_SimulationForceSignal() = default;
    RS_SimulationForceSignal(
        const RS_DeviceLocation &device,
        RIndex structure_gid,
        RFloat *force_signals
    )
        : device(device), structure_gid(structure_gid), force_signals(force_signals) {}
};

/**
 * @param torque_signals Pointer to the array of torque signals (3D vector torque, size 3 * RFloat).
 */
class RS_SimulationTorqueSignal {
   public:
    RS_DeviceLocation device = {RSE_DEVICE_TYPE_UNKNOWN, RS_NULL_INDEX};

    RIndex structure_gid = RS_NULL_INDEX;

    RFloat *torque_signals = nullptr;

    RS_SimulationTorqueSignal() = default;
    RS_SimulationTorqueSignal(
        const RS_DeviceLocation &device,
        RIndex structure_gid,
        RFloat *torque_signals
    )
        : device(device), structure_gid(structure_gid), torque_signals(torque_signals) {}
};

}  // namespace rise

#endif  // RS_SIMULATION_SIGNAL_H
