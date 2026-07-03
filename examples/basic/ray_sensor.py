# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Use a ray sensor as a low-resolution camera and visualize every channel.

What you'll learn:

* Lay out ray directions as a camera-like frustum. The sensor casts one ray per
  cell of a ``width x height`` grid spanning a field of view, so its closest-hit
  output reads back as an image.
* Turn on the sensor's per-ray channels. Beyond the hit flag and distance, the
  record can carry the world-space normal, the hit entity type, and the radial
  velocity of the hit surface relative to the sensor.
* Drive the sensor from the control loop. The orbit pose is written into the
  sensor signal every step, so the sensor circles the cube and keeps aiming at
  it. Its own motion makes the radial velocity channel light up.

A Qt window shows the four channels in a live grid as the sensor orbits.

Run it::

    python examples/basic/ray_sensor.py
"""

import math
import threading

import numpy as np

import rise_origin as rs
from rise_origin import rsc

CUBE_CENTER = (0.0, 0.0, 0.2)  # world point the sensor orbits and looks at
ORBIT_RADIUS = 0.6  # sensor distance from the cube center in meters
ORBIT_HEIGHT = 0.25  # sensor height above the cube center in meters
ORBIT_SPEED = 0.6  # orbit angular speed in radians per second
GRID_WIDTH = 240  # rays across
GRID_HEIGHT = 240  # rays down
FIELD_OF_VIEW = 60.0  # vertical field of view in degrees
MAX_RANGE = 10.0  # farthest a ray reports a hit, in meters


def main():
    """Build the orbit world and show the live channel grid."""
    world = _build_world()
    frame = _SharedFrame()

    origin = rs.RiseOrigin("cuda:0")
    handle, callback = origin.hello(world)
    callback.set_callback(_make_control(frame))
    _run_window(handle, frame)


def _build_world():
    """A red cube on a floor under a blue sky, probed by a frustum ray sensor."""
    resources = [
        rsc.resource.TextureConstant("sky", value=(0.45, 0.62, 0.95)),
        rsc.resource.TextureConstant("cube_albedo", value=(0.85, 0.2, 0.2)),
        rsc.resource.TextureConstant("cube_rm", value=(0.6, 0.0, 0.0)),
        rsc.resource.TextureConstant("floor_albedo", value=(0.6, 0.6, 0.6)),
        rsc.resource.TextureConstant("floor_rm", value=(0.9, 0.0, 0.0)),
    ]
    essences = [
        rsc.essence.Material("cube"),
        rsc.essence.MaterialPhysicalPropertyFixed("cube#phys", material_reference="cube"),
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
        rsc.entity.RaySensorFree(
            "probe",
            ray_directions=_make_ray_grid(GRID_WIDTH, GRID_HEIGHT, FIELD_OF_VIEW),
            origin_position=(-ORBIT_RADIUS, 0.0, CUBE_CENTER[2] + ORBIT_HEIGHT),
            max_range=MAX_RANGE,
            record=rsc.entity.RaySensorRecord(
                has_hit=True,
                has_normal=True,
                has_entity_info=True,
                has_velocity=True,
                target_controller_reference="probe_ctrl",
            ),
            signal=rsc.entity.RaySensorSignal(source_controller_reference="probe_ctrl"),
        ),
    ]

    simulation = rsc.simulation.Simulation(runtime=rsc.simulation.Runtime(log_level="warn"))
    simulation.control.frequency = 30.0

    return rsc.World(
        "ray_sensor_channels",
        resources=resources,
        essences=essences,
        entities=entities,
        controllers=[rsc.controller.ControllerCallback("probe_ctrl")],
        simulation=simulation,
    )


def _make_control(frame):
    """Build the per-step callback that orbits the sensor and grabs its channels."""

    def control(time, controllers):
        angle = time * ORBIT_SPEED
        eye = (
            CUBE_CENTER[0] + ORBIT_RADIUS * math.cos(angle),
            CUBE_CENTER[1] + ORBIT_RADIUS * math.sin(angle),
            CUBE_CENTER[2] + ORBIT_HEIGHT,
        )
        quaternion = _look_at_quaternion(eye, CUBE_CENTER)
        for controller in controllers:
            if controller.name() != "probe_ctrl":
                continue
            _drive_sensor(controller, eye, quaternion)
            _capture(controller, frame)

    return control


def _drive_sensor(controller, position, quaternion):
    """Write the orbit pose into the sensor's position and orientation signals."""
    for signal in controller.signals():
        if isinstance(signal, rs.RS_WorldRaySensorSignal):
            positions = signal.position_signals().position()
            orientations = signal.orientation_signals().orientation()
            if len(positions) > 0:
                positions[0, :] = position
            if len(orientations) > 0:
                orientations[0, :] = quaternion


def _capture(controller, frame):
    """Copy the sensor record's per-ray channels into the shared frame."""
    for record in controller.records():
        if not isinstance(record, rs.RS_WorldRaySensorRecord):
            continue
        if not record.is_valid() or record.ray_num() == 0:
            continue
        shape = (GRID_HEIGHT, GRID_WIDTH)
        frame.set(
            is_hit=np.array(record.hit.is_hit(), copy=True).reshape(shape),
            distance=np.array(record.hit.distance(), copy=True).reshape(shape),
            normal=np.array(record.normal.normal(), copy=True).reshape(GRID_HEIGHT, GRID_WIDTH, 3),
            entity=np.array(record.entity_info.entity_type(), copy=True).reshape(shape),
            radial=np.array(record.velocity.v_radial(), copy=True).reshape(shape),
        )


class _SharedFrame:
    """A latest-wins handoff of channel buffers from the engine thread to the window."""

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
        ("Distance", _distance_to_image),
        ("Normal", _normal_to_image),
        ("Entity", _entity_to_image),
        ("Radial velocity", _radial_to_image),
    ]
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    label_height = 18
    cell_pixels = 256

    class _GridWindow(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Ray sensor channels")
            self.resize(columns * cell_pixels, rows * (cell_pixels + label_height))
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


def _distance_to_image(data):
    """Hit distance as grayscale: near is bright, far is dark, a miss is black."""
    distance = data["distance"]
    is_hit = data["is_hit"]
    if distance.size == 0:
        return _gray_panel()
    brightness = (1.0 - np.clip(distance / MAX_RANGE, 0.0, 1.0)) * 255.0
    value = brightness.astype(np.uint8)
    image = np.stack([value, value, value], axis=-1)
    image[is_hit == 0] = 0
    return image


def _normal_to_image(data):
    """World-space normals mapped from [-1, 1] to [0, 255] per channel."""
    normal = data["normal"]
    is_hit = data["is_hit"]
    if normal.size == 0:
        return _gray_panel()
    image = np.clip((normal * 0.5 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
    image[is_hit == 0] = 0
    return image


def _entity_to_image(data):
    """Entity type per ray: sky/miss, floor, and voxel each get a flat color."""
    entity = data["entity"]
    if entity.size == 0:
        return _gray_panel()
    palette = np.array(
        [[40, 45, 80], [110, 110, 110], [210, 80, 80]], dtype=np.uint8
    )  # cosmic background, floor, structure voxel
    return palette[np.clip(entity, 0, len(palette) - 1)]


def _radial_to_image(data):
    """Radial velocity by Doppler convention: blue approaching, red receding.

    ``v_radial`` is negative when the hit surface closes on the sensor
    (approaching, a blue shift) and positive when it opens up (receding, a red
    shift).
    """
    radial = data["radial"]
    is_hit = data["is_hit"]
    if radial.size == 0:
        return _gray_panel()
    scaled = np.clip(radial / 0.15, -1.0, 1.0)
    image = np.zeros((radial.shape[0], radial.shape[1], 3), dtype=np.uint8)
    image[..., 0] = (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    image[..., 2] = (np.clip(-scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    image[is_hit == 0] = 0
    return image


def _gray_panel():
    """A neutral placeholder for a buffer that has not arrived yet."""
    return np.full((GRID_HEIGHT, GRID_WIDTH, 3), 60, dtype=np.uint8)


def _make_ray_grid(width, height, field_of_view):
    """Ray directions for a ``width x height`` frustum in the sensor's local frame.

    The sensor's local forward is +X, right is +Y, up is +Z. One ray points
    through the center of each grid cell, spanning ``field_of_view`` vertically.
    """
    fov = math.radians(field_of_view)
    half_up = math.tan(fov / 2.0)
    half_right = half_up * (width / height)
    directions = []
    for i in range(height):
        up = (0.5 - (i + 0.5) / height) * 2.0 * half_up
        for j in range(width):
            right = ((j + 0.5) / width - 0.5) * 2.0 * half_right
            directions.append((1.0, right, up))
    return directions


def _look_at_quaternion(eye, center, global_up=(0.0, 0.0, 1.0)):
    """Quaternion (x, y, z, w) aiming a +X-forward, +Z-up sensor at center."""
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
