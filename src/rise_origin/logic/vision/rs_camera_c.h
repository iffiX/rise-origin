// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#ifndef RS_CAMERA_C_H
#define RS_CAMERA_C_H

#include "rise_origin/device/rs_compute.h"
#include "rise_origin/logic/signal/rs_camera_signal_m.h"
#include "rise_origin/utils/math/rs_color.h"
#include "rise_origin/utils/math/rs_vec.h"
#include "rise_origin/utils/rs_conf.h"

namespace rise {

namespace logic {

class RSCF_ResolveRigidBodyLinkedPerspectiveCameraDependency
    : public device::RS_ComputeFunction<RSCF_ResolveRigidBodyLinkedPerspectiveCameraDependency> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(Kernel* k, RHandle camera, RSize rigid_body_segment_offset) {
        auto& ctx = k->ctx;
        RIndex rigid_body_sid = CA_G(rigid_body_sid);
        RHandle rigid_body = rigid_body_sid + rigid_body_segment_offset;

        CA_S(rigid_body, rigid_body);

        // Use double for universal precision
        RVec3df origin = RB_G(origin);
        RQuat3df orientation =
            RQuat3df(RB_G(orientation)) * RQuat3df(CA_G(rigid_body_anchor_rb_orientation));
        RVec3df position =
            origin + orientation.rotateVec3D(RVec3df(CA_G(rigid_body_anchor_rb_position)));
        CA_S(position, position);
        CA_S(orientation, orientation);
    }
};

class RSCF_ResolveRigidBodyLinkedStereoCameraDependency
    : public device::RS_ComputeFunction<RSCF_ResolveRigidBodyLinkedStereoCameraDependency> {
   public:
    template <typename Kernel>
    HOST_DEVICE_INLINE_QUALIFIER static void
    run(Kernel* k, RHandle camera, RSize rigid_body_segment_offset) {
        auto& ctx = k->ctx;
        RIndex rigid_body_sid = CA_G(rigid_body_sid);
        RHandle rigid_body = rigid_body_sid + rigid_body_segment_offset;

        CA_S(rigid_body, rigid_body);

        // Use double for universal precision
        RVec3df origin = RB_G(origin);
        RQuat3df orientation =
            RQuat3df(RB_G(orientation)) * RQuat3df(CA_G(rigid_body_anchor_rb_orientation));
        RVec3df position =
            origin + orientation.rotateVec3D(RVec3df(CA_G(rigid_body_anchor_rb_position)));

        // Offset this eye along the right axis by half the interpupillary distance.
        RVec3df unit_right = -orientation.toOrthoAxisY();
        double half_ipd = double(CA_G(interpupillary_distance)) / 2.0;
        double eye_sign = (CA_G(sub_camera_index) == 0) ? -1.0 : 1.0;
        position = position + unit_right * (eye_sign * half_ipd);

        CA_S(position, position);
        CA_S(orientation, orientation);
    }
};

template <RSE_FloatType FloatComputeAccuracy = RSE_FLOAT_32>
class RS_CameraCompute {
   public:
    using FloatC = typename RS_FloatDataType<FloatComputeAccuracy>::DataType;
    using Vec3C = RS_Vec3D<FloatC>;
    using Quat3C = RS_Quat3D<FloatC>;

    class RSCF_UpdateCamera : public device::RS_ComputeFunction<RSCF_UpdateCamera> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera) {
            using RS_FI = RS_FloatIntrinsics<FloatC>;
            if (k != nullptr && camera != RS_NULL_HANDLE) {
                auto& ctx = k->ctx;

                RS_CameraStateUnion state_union{CA_G(state)};
                RSE_CameraType type = state_union.camera_state.type;
                bool is_stereo = type == RSE_CAMERA_FREE_STEREO ||
                                 type == RSE_CAMERA_RIGID_BODY_LINKED_STEREO;
                bool is_rigid = type == RSE_CAMERA_RIGID_BODY_LINKED_PERSPECTIVE ||
                                type == RSE_CAMERA_RIGID_BODY_LINKED_STEREO;

                CA_S(prev_position, CA_G(position));
                CA_S(prev_orientation, CA_G(orientation));
                CA_S(frame_count, CA_G(frame_count) + RBigSize(1));

                if (is_rigid) {
                    RHandle rigid_body = CA_G(rigid_body);

                    Quat3C rigid_orientation = RB_G(orientation);
                    Quat3C orientation =
                        rigid_orientation * Quat3C(CA_G(rigid_body_anchor_rb_orientation));
                    Vec3C position =
                        Vec3C(RB_G(origin)) +
                        rigid_orientation.rotateVec3D(Vec3C(CA_G(rigid_body_anchor_rb_position)));

                    // For a stereo eye, offset along the right axis by half the eye separation.
                    if (is_stereo) {
                        Vec3C unit_right = -orientation.toOrthoAxisY();
                        FloatC half_ipd = FloatC(CA_G(interpupillary_distance)) / FloatC(2.0);
                        FloatC eye_sign = (CA_G(sub_camera_index) == 0) ? FloatC(-1.0)
                                                                        : FloatC(1.0);
                        position = position + unit_right * (eye_sign * half_ipd);
                    }
                    CA_S(position, position);
                    CA_S(orientation, orientation);
                } else {
                    RHandle camera_signal_handle = CA_G(camera_signal_handle);
                    if (camera_signal_handle != RS_NULL_HANDLE) {
                        auto& signal = ctx.camera_signals[camera_signal_handle];

                        if (is_stereo) {
                            // Both eyes share one stereo signal; pick this eye's arm.
                            Vec3C position;
                            Quat3C orientation;
                            const RFloat* field_of_view;
                            if (CA_G(sub_camera_index) == 0) {
                                position = Vec3C(signal.stereo.left_position_signals.position[0]);
                                orientation =
                                    Quat3C(signal.stereo.left_orientation_signals.orientation[0]);
                                field_of_view = signal.stereo.left_optics_signals.field_of_view;
                            } else {
                                position = Vec3C(signal.stereo.right_position_signals.position[0]);
                                orientation =
                                    Quat3C(signal.stereo.right_orientation_signals.orientation[0]);
                                field_of_view = signal.stereo.right_optics_signals.field_of_view;
                            }
                            orientation = orientation.normalized();
                            CA_S(position, position);
                            CA_S(orientation, orientation);
                            // Four frustum angles, ordered left, right, up, down.
                            CA_S(angle_left, field_of_view[0]);
                            CA_S(angle_right, field_of_view[1]);
                            CA_S(angle_up, field_of_view[2]);
                            CA_S(angle_down, field_of_view[3]);
                        } else {
                            Vec3C position(signal.perspective.position_signals.position[0]);
                            Quat3C orientation(
                                signal.perspective.orientation_signals.orientation[0]
                            );
                            orientation = orientation.normalized();

                            FloatC focus_distance =
                                signal.perspective.optics_signals.focus_distance[0];
                            FloatC field_of_view =
                                signal.perspective.optics_signals.field_of_view[0];

                            CA_S(orientation, orientation);
                            CA_S(position, position);
                            CA_S(focus_distance, focus_distance);
                            CA_S(field_of_view, field_of_view);
                            CA_S(
                                defocus_radius,
                                focus_distance *
                                    RS_FI::rs_tan(
                                        static_cast<FloatC>(CA_G(defocus_angle) * FloatC(RS_PI)) /
                                        FloatC(180.0) / FloatC(2.0)
                                    )
                            );
                            FloatC theta = field_of_view * FloatC(RS_PI) / FloatC(180.0);
                            FloatC h = RS_FI::rs_tan(theta / FloatC(2.0));

                            FloatC viewport_height = FloatC(2.0) * h * focus_distance;
                            FloatC viewport_width = viewport_height / CA_G(aspect_ratio);

                            CA_S(
                                pixel_height,
                                viewport_height / static_cast<FloatC>(CA_G(image_height))
                            );
                            CA_S(
                                pixel_width, viewport_width / static_cast<FloatC>(CA_G(image_width))
                            );
                        }
                        RBigFloat current_time = k->getDeviceTime();
                        CA_S(last_update_time, current_time);
                    }
                }
            }
        }
    };

    class RSCF_ClearCameraCanvas : public device::RS_ComputeFunction<RSCF_ClearCameraCanvas> {
       public:
        template <typename Kernel>
        HOST_DEVICE_INLINE_QUALIFIER static void run(Kernel* k, RHandle camera, RHandle pixel) {
            if (k != nullptr && camera != RS_NULL_HANDLE) {
                auto& ctx = k->ctx;
                ctx.camera_current_rgb_canvas[camera][pixel] = RVec3f::zero();
            }
        }
    };
};

}  // namespace logic

}  // namespace rise

#endif  // RS_CAMERA_C_H
