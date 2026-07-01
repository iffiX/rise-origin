# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Render a stereo camera and show both eyes while orbiting a cube.

What you'll learn:

* Add a stereo camera. A stereo camera renders two eyes in one entity; each eye
  is a full perspective sub-camera with its own pose and an asymmetric frustum.
* Drive the per-eye head pose. The stereo signal carries a left and a right eye
  pose; the control loop offsets the two eyes by the interpupillary distance and
  writes them every step, so the rig orbits the cube on its own.
* Read both eyes from the record. The stereo record exposes left_/right_ buffers
  (rgb, depth, ...) that come back as plain numpy arrays.

A Qt window shows the left and right eye RGB and depth in a live 2x2 grid. The
two RGB views differ by parallax: the cube shifts horizontally between eyes.

Run it::

    python examples/basic/camera_stereo.py
"""

import math
import threading

import numpy as np

import rise_origin as rs
from rise_origin import rsc

CUBE_CENTER = (0.0, 0.0, 0.2)  # world point the rig orbits and looks at
ORBIT_RADIUS = 0.6  # rig distance from the cube center in meters
ORBIT_HEIGHT = 0.25  # rig height above the cube center in meters
ORBIT_SPEED = 0.6  # orbit angular speed in radians per second
INTERPUPILLARY_DISTANCE = 0.063  # distance between the two eyes in meters
IMAGE_WIDTH = 240
IMAGE_HEIGHT = 240


def main():
    """Build the orbit world and show the live left/right eye grid."""
    world = _build_world()
    frame = _SharedFrame()

    origin = rs.RiseOrigin("cpu")
    handle, callback = origin.hello(world)
    callback.set_callback(_make_control(frame))
    _run_window(handle, frame)


def _build_world():
    """A red cube on a floor under a blue sky, viewed by a stereo camera."""
    resources = [
        rsc.resource.TextureConstant("sky", value=(0.45, 0.62, 0.95)),
        rsc.resource.TextureConstant("cube_albedo", value=(0.85, 0.2, 0.2)),
        rsc.resource.TextureConstant("cube_rm", value=(0.6, 0.0, 0.0)),
        rsc.resource.TextureConstant("floor_albedo", value=(0.6, 0.6, 0.6)),
        rsc.resource.TextureConstant("floor_rm", value=(0.9, 0.0, 0.0)),
    ]
    essences = [
        rsc.essence.Material("cube"),
        rsc.essence.MaterialPhysicalPropertyRigid("cube#phys", material_reference="cube", density=1000.0),
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "cube#optic",
            material_reference="cube",
            albedo_texture_reference="cube_albedo",
            roughness_metalness_texture_reference="cube_rm",
        ),
        rsc.essence.Material("floor"),
        rsc.essence.MaterialPhysicalPropertyFixed("floor#phys", material_reference="floor"),
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "floor#optic",
            material_reference="floor",
            albedo_texture_reference="floor_albedo",
            roughness_metalness_texture_reference="floor_rm",
        ),
    ]
    entities = [
        rsc.entity.CosmicBackgroundUniform(texture_reference="sky", intensity=1.0),
        rsc.entity.FloorFlat("floor", material_reference="floor"),
        rsc.entity.StructureVoxel(
            "cube",
            origin_position=CUBE_CENTER,
            material_references=["cube"],
            bodies=[rsc.entity.StructureVoxelBody.box(6, 6, 6, voxel_size=0.03)],
        ),
        rsc.entity.CameraStereoFree(
            "cam",
            origin_position=(-ORBIT_RADIUS, 0.0, CUBE_CENTER[2] + ORBIT_HEIGHT),
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            samples_per_pixel=4,
            max_depth=3,
            interpupillary_distance=INTERPUPILLARY_DISTANCE,
            record=rsc.entity.CameraStereoRecord(
                has_rgb=True,
                has_depth=True,
                is_depth_hyperbolic=True,
                hyperbolic_near_depth=0.3,
                hyperbolic_far_depth=2.0,
                target_controller_reference="cam_ctrl",
            ),
            signal=rsc.entity.CameraStereoSignal(source_controller_reference="cam_ctrl"),
        ),
    ]

    simulation = rsc.simulation.Simulation(runtime=rsc.simulation.Runtime(log_level="warn"))
    simulation.control.frequency = 30.0
    simulation.record.frequency = 30.0

    return rsc.World(
        "camera_stereo",
        resources=resources,
        essences=essences,
        entities=entities,
        controllers=[rsc.controller.ControllerCallback("cam_ctrl")],
        simulation=simulation,
    )


def _make_control(frame):
    """Build the per-step callback that orbits the rig and grabs both eyes."""

    def control(time, controllers):
        angle = time * ORBIT_SPEED
        head = np.array(
            [
                CUBE_CENTER[0] + ORBIT_RADIUS * math.cos(angle),
                CUBE_CENTER[1] + ORBIT_RADIUS * math.sin(angle),
                CUBE_CENTER[2] + ORBIT_HEIGHT,
            ]
        )
        quaternion, right_axis = _look_at(head, np.asarray(CUBE_CENTER, dtype=np.float64))
        half = 0.5 * INTERPUPILLARY_DISTANCE
        left_eye = head - right_axis * half
        right_eye = head + right_axis * half
        for controller in controllers:
            if controller.name() != "cam_ctrl":
                continue
            _drive_camera(controller, left_eye, right_eye, quaternion)
            _capture(controller, frame)

    return control


def _drive_camera(controller, left_eye, right_eye, quaternion):
    """Write the per-eye orbit pose into the stereo camera's signal."""
    for signal in controller.signals():
        if not isinstance(signal, rs.RS_WorldCameraStereoSignal):
            continue
        left_position = signal.left_position_signals().position()
        right_position = signal.right_position_signals().position()
        left_orientation = signal.left_orientation_signals().orientation()
        right_orientation = signal.right_orientation_signals().orientation()
        if len(left_position) > 0:
            left_position[0, :] = left_eye
        if len(right_position) > 0:
            right_position[0, :] = right_eye
        if len(left_orientation) > 0:
            left_orientation[0, :] = quaternion
        if len(right_orientation) > 0:
            right_orientation[0, :] = quaternion


def _capture(controller, frame):
    """Copy both eyes' RGB and depth buffers into the shared frame."""
    for record in controller.records():
        if not isinstance(record, rs.RS_WorldCameraStereoRecord):
            continue
        if not record.is_valid() or record.width() == 0:
            continue
        frame.set(
            left_rgb=np.array(record.left_rgb.pixel(), copy=True),
            right_rgb=np.array(record.right_rgb.pixel(), copy=True),
            left_depth=np.array(record.left_depth.pixel(), copy=True),
            right_depth=np.array(record.right_depth.pixel(), copy=True),
        )


class _SharedFrame:
    """A latest-wins handoff of eye buffers from the engine thread to the window."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = None

    def set(self, **buffers):
        with self._lock:
            self._data = buffers

    def get(self):
        with self._lock:
            return self._data


def _run_window(handle, frame):
    """Open the Qt grid window and pump it until closed (blocking)."""
    from PySide6 import QtCore, QtGui, QtWidgets

    panels = [
        ("Left RGB", lambda data: _rgb_to_image(data["left_rgb"])),
        ("Right RGB", lambda data: _rgb_to_image(data["right_rgb"])),
        ("Left Depth", lambda data: _depth_to_image(data["left_depth"])),
        ("Right Depth", lambda data: _depth_to_image(data["right_depth"])),
    ]
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    label_height = 18

    class _GridWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Stereo Camera")
            self.resize(columns * IMAGE_WIDTH, rows * (IMAGE_HEIGHT + label_height))
            self._images = [None] * len(panels)
            timer = QtCore.QTimer(self)
            timer.timeout.connect(self._refresh)
            timer.start(33)

        def _refresh(self):
            data = frame.get()
            if data is not None:
                for index, (_, convert) in enumerate(panels):
                    image = np.ascontiguousarray(convert(data))
                    height, width, _ = image.shape
                    self._images[index] = QtGui.QImage(
                        image.data, width, height, 3 * width, QtGui.QImage.Format.Format_RGB888
                    ).copy()
            self.update()
            if handle.is_end():
                self.close()

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.fillRect(self.rect(), QtGui.QColor(20, 20, 24))
            cell_width = self.width() / columns
            cell_height = self.height() / rows
            for index, (title, _) in enumerate(panels):
                row, column = divmod(index, columns)
                x = column * cell_width
                y = row * cell_height
                image = self._images[index]
                if image is not None:
                    target = QtCore.QRectF(
                        x, y + label_height, cell_width, cell_height - label_height
                    )
                    painter.drawImage(target, image)
                painter.setPen(QtGui.QColor(230, 230, 230))
                painter.drawText(
                    QtCore.QRectF(x, y, cell_width, label_height),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    title,
                )
            painter.end()

        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self.close()

        def closeEvent(self, event):
            handle.terminate()
            super().closeEvent(event)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _GridWindow()
    window.show()
    app.exec()
    handle.terminate()


def _rgb_to_image(rgb):
    """The RGB image as-is, or gray when empty."""
    if rgb.size == 0:
        return _gray_panel()
    return rgb


def _depth_to_image(depth):
    """Hyperbolic depth as grayscale: near is bright, far is dark, sky is black."""
    if depth.size == 0:
        return _gray_panel()
    background = depth >= 0.999
    brightness = (1.0 - np.clip(depth, 0.0, 1.0)) * 255.0
    value = brightness.astype(np.uint8)
    image = np.stack([value, value, value], axis=-1)
    image[background] = 0
    return image


def _gray_panel():
    """A neutral placeholder for a buffer that has not arrived yet."""
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 60, dtype=np.uint8)


def _look_at(eye, center, global_up=(0.0, 0.0, 1.0)):
    """Quaternion (x, y, z, w) and right axis aiming a +X-forward, +Z-up camera at center."""
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    global_up = np.asarray(global_up, dtype=np.float64)
    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(global_up, forward)
    right = right / np.linalg.norm(right)
    local_up = np.cross(forward, right)
    matrix = np.column_stack((forward, right, local_up))
    return _matrix_to_quaternion(matrix), right


def _matrix_to_quaternion(matrix):
    """Convert a rotation matrix (columns are the rotated basis) to (x, y, z, w)."""
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    return [x, y, z, w]


if __name__ == "__main__":
    main()
