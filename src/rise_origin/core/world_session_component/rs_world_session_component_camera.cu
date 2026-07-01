// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Muhan Li (iffi/iffiX)

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <utility>

#include "rise_origin/core/rs_world_record.h"
#include "rise_origin/core/rs_world_session_component.h"
#include "rise_origin/core/rs_world_signal.h"
#include "rise_origin/rsc/controller/rsc_controller.h"
#include "rise_origin/rsc/entity/rsc_camera.h"
#include "rise_origin/utils/math/rs_mat.h"

namespace rise {
RS_WorldSessionComponentCamera::RS_WorldSessionComponentCamera(
    std::string name,
    Callback callback,
    RVec3rf position,
    RVec3rf focus,
    RSize width,
    RSize height,
    RSize samples_per_pixel,
    RSize max_depth,
    RFloat field_of_view,
    RFloat defocus_angle,
    RFloat focus_distance
)
    : _name(std::move(name)),
      _controller_name(_name + "#controller"),
      _callback(std::move(callback)),
      _width(width),
      _height(height),
      _samples_per_pixel(samples_per_pixel),
      _max_depth(max_depth),
      _field_of_view(field_of_view),
      _defocus_angle(defocus_angle),
      _focus_distance(focus_distance),
      _position(position),
      _focus(focus),
      _orientation(_lookAt(position, focus)),
      _distance((focus - position).length()) {
}

void RS_WorldSessionComponentCamera::onAttach(RS_WorldSessionAccess access) {
    _access = std::move(access);
    _access->handle()->addEntity(_makeCamera());
    _access->handle()->addController(_makeController());
    RBigSize count = static_cast<RBigSize>(_width) * _height;
    _frames[0] = std::shared_ptr<RColor3b[]>(new RColor3b[count]());
    _frames[1] = std::shared_ptr<RColor3b[]>(new RColor3b[count]());
}

void RS_WorldSessionComponentCamera::onCallback(RBigFloat time, RS_ThreadPool& pool) {
    RIndex pixel_num = static_cast<RIndex>(static_cast<RBigSize>(_width) * _height);
    auto copy = std::make_shared<RS_MultiFuture<void>>(pool.submitBlocks(
        0,
        pixel_num,
        [this](RIndex start, RIndex end) { _copyFrame(start, end); }
    ));
    pool.detachTask([this] {
        _applyPending();
        _writeCameraSignal();
    });
    pool.detachTask([this, copy, time] {
        copy->wait();
        _publishFrame();
        if (_callback) _callback(time);
    });
}

void RS_WorldSessionComponentCamera::onDetach() {
}

void RS_WorldSessionComponentCamera::move(
    RFloat orbit_pitch,
    RFloat orbit_yaw,
    RFloat pan_x,
    RFloat pan_y,
    RFloat zoom
) {
    std::scoped_lock lock(_pending_mutex);
    _pending_orbit_pitch += orbit_pitch;
    _pending_orbit_yaw += orbit_yaw;
    _pending_pan_x += pan_x;
    _pending_pan_y += pan_y;
    _pending_zoom += zoom;
}

std::shared_ptr<const RColor3b[]>
RS_WorldSessionComponentCamera::frame(RSize& width, RSize& height) const {
    std::scoped_lock lock(_frame_mutex);
    width = _width;
    height = _height;
    return _frames[_front];
}

void RS_WorldSessionComponentCamera::_copyFrame(RIndex pixel_start, RIndex pixel_end) {
    RS_WorldController& controller = _access->controller(_controller_name);
    for (auto& record_variant : controller.records) {
        record_variant.visit([&](auto& record) {
            using Record = std::decay_t<decltype(record)>;
            if constexpr (std::is_same_v<Record, RS_WorldCameraPerspectiveRecord>) {
                if (!record.isValid() || record.width != _width || record.height != _height) return;
                std::memcpy(
                    _frames[1 - _front].get() + pixel_start,
                    record.rgb.pixel + pixel_start,
                    static_cast<RBigSize>(pixel_end - pixel_start) * sizeof(RColor3b)
                );
            }
        });
    }
}

void RS_WorldSessionComponentCamera::_publishFrame() {
    std::scoped_lock lock(_frame_mutex);
    _front = 1 - _front;
    // If the GUI still holds the buffer now rotating to the back, give the back slot a fresh buffer
    // so the next copy never overwrites a frame being read.
    if (_frames[1 - _front].use_count() != 1) {
        RBigSize count = static_cast<RBigSize>(_width) * _height;
        _frames[1 - _front] = std::shared_ptr<RColor3b[]>(new RColor3b[count]());
    }
}

void RS_WorldSessionComponentCamera::_applyPending() {
    std::scoped_lock lock(_pending_mutex);
    _applyZoom();
    _applyPan();
    _applyOrbit();
}

void RS_WorldSessionComponentCamera::_applyZoom() {
    if (_pending_zoom == 0) return;
    _distance *= 1.0f - _pending_zoom * 0.1f;
    _distance = std::clamp(_distance, RFloat(0.05), RFloat(100.0));
    _pending_zoom = 0;
}

void RS_WorldSessionComponentCamera::_applyPan() {
    if (_pending_pan_x == 0 && _pending_pan_y == 0) return;
    RVec3rf right = _orientation.toOrthoAxisY();
    RVec3rf up = _orientation.toOrthoAxisZ();
    RFloat pan_speed = _distance * 0.001f;
    _focus = _focus + right * (_pending_pan_x * pan_speed) + up * (_pending_pan_y * pan_speed);
    _pending_pan_x = 0;
    _pending_pan_y = 0;
}

void RS_WorldSessionComponentCamera::_applyOrbit() {
    if (_pending_orbit_pitch == 0 && _pending_orbit_yaw == 0) {
        // No orbit this step: keep the camera on its forward ray at the current distance.
        _position = _focus - _orientation.toOrthoAxisX() * _distance;
        return;
    }
    RVec3rf cam_to_target = _focus - _position;
    RVec3rf right_axis = _orientation.toOrthoAxisY();
    RQuat3rf pitch(_pending_orbit_pitch, right_axis);
    RQuat3rf yaw(_pending_orbit_yaw, RVec3rf(0, 0, 1));
    RVec3rf rotated = yaw.rotateVec3D(pitch.rotateVec3D(cam_to_target));
    _position = _focus - rotated;
    _orientation = _lookAt(_position, _focus);
    _pending_orbit_pitch = 0;
    _pending_orbit_yaw = 0;
}

void RS_WorldSessionComponentCamera::_writeCameraSignal() {
    RS_WorldController& controller = _access->controller(_controller_name);
    for (auto& signal_variant : controller.signals) {
        signal_variant.visit([&](auto& signal) {
            using Signal = std::decay_t<decltype(signal)>;
            if constexpr (std::is_same_v<Signal, RS_WorldCameraPerspectiveSignal>) {
                if (signal.position_signals.object_num > 0 && signal.position_signals.position) {
                    signal.position_signals.position[0] = _position;
                }
                if (signal.orientation_signals.object_num > 0 &&
                    signal.orientation_signals.orientation) {
                    signal.orientation_signals.orientation[0] = _orientation;
                }
            }
        });
    }
}

RQuat3rf RS_WorldSessionComponentCamera::_lookAt(const RVec3rf& eye, const RVec3rf& at) {
    // Aim a +X-forward / +Z-up camera at the target; columns are the rotated basis.
    RVec3rf forward = (at - eye).normalized();
    RVec3rf global_up(0, 0, 1);
    RVec3rf right = global_up.cross(forward).normalized();
    RVec3rf local_up = forward.cross(right);
    RMat3rf rotation(
        forward.x,
        right.x,
        local_up.x,
        //
        forward.y,
        right.y,
        local_up.y,
        //
        forward.z,
        right.z,
        local_up.z
    );
    RQuat3rf orientation;
    orientation.fromRotationMatrix(rotation);
    return orientation;
}

rsc::RSC_EntityVariant RS_WorldSessionComponentCamera::_makeCamera() const {
    rsc::RSC_CameraPerspectiveFree camera;
    camera.name = _name;
    camera.origin_position = _position;
    camera.orientation = _orientation;
    camera.image_width = _width;
    camera.image_height = _height;
    camera.samples_per_pixel = _samples_per_pixel;
    camera.max_depth = _max_depth;
    camera.field_of_view = _field_of_view;
    camera.defocus_angle = _defocus_angle;
    camera.focus_distance = _focus_distance;
    camera.record.has_rgb = true;
    camera.record.target_controller_reference = _controller_name;
    camera.signal.source_controller_reference = _controller_name;
    return rsc::RSC_EntityVariant(camera);
}

rsc::RSC_ControllerVariant RS_WorldSessionComponentCamera::_makeController() const {
    rsc::RSC_ControllerCallback controller;
    controller.name = _controller_name;
    return rsc::RSC_ControllerVariant(controller);
}
} // namespace rise