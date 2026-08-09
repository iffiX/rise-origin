# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Visualize every camera AOV at once while orbiting a cube.

What you'll learn:

* Turn on the camera's extra output channels. A camera record can carry, beyond
  the RGB image, a per-pixel depth, world-space normal, screen-space motion, and
  entity-info buffer; this example enables all of them.
* Read the buffers from the per-step record. Each control step delivers the
  camera record to a controller, and the buffers come back as plain numpy arrays.
* Drive the camera from the control loop. The orbit pose is written into the
  camera signal every step, so the camera circles the cube on its own.

A Qt window shows the five outputs in a live grid as the camera orbits.

Run it::

    python examples/basic/camera.py
"""

import math
import threading

import numpy as np

import rise_origin as rs
from rise_origin import rsc

CUBE_CENTER = (0.0, 0.0, 0.2)  # world point the camera orbits and looks at
ORBIT_RADIUS = 0.6  # camera distance from the cube center in meters
ORBIT_HEIGHT = 0.25  # camera height above the cube center in meters
ORBIT_SPEED = 0.6  # orbit angular speed in radians per second
IMAGE_WIDTH = 240
IMAGE_HEIGHT = 240


def main():
    """Build the orbit world and show the live AOV grid."""
    world = _build_world()
    frame = _SharedFrame()

    origin = rs.RiseOrigin("cpu")
    handle, callback = origin.hello(world)
    callback.set_callback(_make_control(frame))
    _run_window(handle, frame)


def _build_world():
    """A red cube on a floor under a blue sky, viewed by an all-AOV camera."""
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
        rsc.entity.CosmicBackgroundUniform(texture_reference="sky", radiance=1.0),
        rsc.entity.FloorFlat("floor", material_reference="floor"),
        rsc.entity.StructureVoxel(
            "cube",
            origin_position=CUBE_CENTER,
            material_references=["cube"],
            bodies=[rsc.entity.StructureVoxelBody.box(6, 6, 6, voxel_size=0.03)],
        ),
        rsc.entity.CameraPerspectiveFree(
            "cam",
            origin_position=(-ORBIT_RADIUS, 0.0, CUBE_CENTER[2] + ORBIT_HEIGHT),
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            samples_per_pixel=4,
            max_depth=3,
            focus_distance=1.0,
            record=rsc.entity.CameraPerspectiveRecord(
                has_rgb=True,
                has_depth=True,
                has_normal=True,
                has_motion=True,
                has_entity_info=True,
                is_depth_hyperbolic=True,
                hyperbolic_near_depth=0.3,
                hyperbolic_far_depth=2.0,
                target_controller_reference="cam_ctrl",
            ),
            signal=rsc.entity.CameraPerspectiveSignal(source_controller_reference="cam_ctrl"),
        ),
    ]

    simulation = rsc.simulation.Simulation(runtime=rsc.simulation.Runtime(log_level="warn"))
    simulation.control.frequency = 30.0
    simulation.record.frequency = 30.0

    return rsc.World(
        "camera_aov",
        resources=resources,
        essences=essences,
        entities=entities,
        controllers=[rsc.controller.ControllerCallback("cam_ctrl")],
        simulation=simulation,
    )


def _make_control(frame):
    """Build the per-step callback that orbits the camera and grabs its AOVs."""

    def control(time, controllers):
        angle = time * ORBIT_SPEED
        eye = (
            CUBE_CENTER[0] + ORBIT_RADIUS * math.cos(angle),
            CUBE_CENTER[1] + ORBIT_RADIUS * math.sin(angle),
            CUBE_CENTER[2] + ORBIT_HEIGHT,
        )
        quaternion = _look_at_quaternion(eye, CUBE_CENTER)
        for controller in controllers:
            if controller.name() != "cam_ctrl":
                continue
            _drive_camera(controller, eye, quaternion)
            _capture(controller, frame)

    return control


def _drive_camera(controller, position, quaternion):
    """Write the orbit pose into the camera's position and orientation signals."""
    for signal in controller.signals():
        if isinstance(signal, rs.RS_WorldCameraPerspectiveSignal):
            positions = signal.position_signals().position()
            orientations = signal.orientation_signals().orientation()
            if len(positions) > 0:
                positions[0, :] = position
            if len(orientations) > 0:
                orientations[0, :] = quaternion


def _capture(controller, frame):
    """Copy the camera record's AOV buffers into the shared frame."""
    for record in controller.records():
        if not isinstance(record, rs.RS_WorldCameraPerspectiveRecord):
            continue
        if not record.is_valid() or record.width() == 0:
            continue
        frame.set(
            rgb=np.array(record.rgb.pixel(), copy=True),
            depth=np.array(record.depth.pixel(), copy=True),
            normal=np.array(record.normal.pixel(), copy=True),
            motion=np.array(record.motion.pixel(), copy=True),
            entity=np.array(record.entity_info.entity_type(), copy=True),
        )


class _SharedFrame:
    """A latest-wins handoff of AOV buffers from the engine thread to the window."""

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
        ("RGB", _rgb_to_image),
        ("Depth", _depth_to_image),
        ("Normal", _normal_to_image),
        ("Motion", _motion_to_image),
        ("Entity", _entity_to_image),
    ]
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    label_height = 18

    class _GridWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Camera AOVs")
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


def _rgb_to_image(data):
    """The RGB image as-is, or gray when empty."""
    rgb = data["rgb"]
    if rgb.size == 0:
        return _gray_panel()
    return rgb


def _depth_to_image(data):
    """Hyperbolic depth as grayscale: near is bright, far is dark, sky is black.

    The camera record stores hyperbolic device depth in [0, 1], which packs most
    of its range into the near scene around the camera, so the cube separates
    cleanly from the floor it sits on. The far plane (and sky) saturates to 1.0
    and is drawn black.
    """
    depth = data["depth"]
    if depth.size == 0:
        return _gray_panel()
    background = depth >= 0.999
    brightness = (1.0 - np.clip(depth, 0.0, 1.0)) * 255.0
    value = brightness.astype(np.uint8)
    image = np.stack([value, value, value], axis=-1)
    image[background] = 0
    return image


def _normal_to_image(data):
    """World-space normals mapped from [-1, 1] to [0, 255] per channel."""
    normal = data["normal"]
    if normal.size == 0:
        return _gray_panel()
    return np.clip((normal * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)


def _motion_to_image(data):
    """Screen-space motion: red is horizontal, green is vertical, amplified."""
    motion = data["motion"]
    if motion.size == 0:
        return _gray_panel()
    amplified = motion * (10.0 * IMAGE_WIDTH) + 0.5
    image = np.full((motion.shape[0], motion.shape[1], 3), 128, dtype=np.uint8)
    image[..., 0] = np.clip(amplified[..., 0] * 255.0, 0.0, 255.0).astype(np.uint8)
    image[..., 1] = np.clip(amplified[..., 1] * 255.0, 0.0, 255.0).astype(np.uint8)
    return image


def _entity_to_image(data):
    """Entity type per pixel: sky, floor, and voxel each get a flat color."""
    entity = data["entity"]
    if entity.size == 0:
        return _gray_panel()
    palette = np.array(
        [[40, 45, 80], [110, 110, 110], [210, 80, 80]], dtype=np.uint8
    )  # cosmic background, floor, structure voxel
    return palette[np.clip(entity, 0, len(palette) - 1)]


def _gray_panel():
    """A neutral placeholder for a buffer that has not arrived yet."""
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 60, dtype=np.uint8)


def _look_at_quaternion(eye, center, global_up=(0.0, 0.0, 1.0)):
    """Quaternion (x, y, z, w) aiming a +X-forward, +Z-up camera at center."""
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    global_up = np.asarray(global_up, dtype=np.float64)
    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(global_up, forward)
    right = right / np.linalg.norm(right)
    local_up = np.cross(forward, right)
    matrix = np.column_stack((forward, right, local_up))
    return _matrix_to_quaternion(matrix)


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
