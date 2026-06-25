# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Interactive Qt viewer and headless recorder for rise-origin worlds.

:class:`Viewer` is your eyes on a running world. It injects and owns one
free-perspective camera, streams that camera's ray-traced frames into a window,
and gives you a mouse-driven orbit/pan/zoom camera. Your whole control loop (the
``callback``) stays free to actuate your structures.

The viewer attaches to an :class:`~rise_origin.runtime.instance.Instance`. Pass
nothing to spin up a default local instance, a ``(device, world)`` tuple to run
a specific world on a device, or your own instance.

Example::

    from rise_origin import rsc
    from rise_origin.runtime.viewer import Viewer

    world = rsc.World(name="warehouse", entities=[...])

    def control(world_name, time, controllers):
        ...  # actuate your structures

    viewer = Viewer("overview", instance=("cuda:0", world), callback=control)
    viewer.run_interactive()                       # live window, runs until closed
    # or, headless:
    viewer.run_record("demo.mp4", duration=8.0); viewer.wait()

Reserved interaction: ``Esc`` closes; ``R`` re-frames. Left-drag orbits,
right-drag pans, scroll/middle-drag zooms.
"""

import threading
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import numpy as np
from pyquaternion import Quaternion

import rise_origin as rs
from rise_origin import rsc
from rise_origin.runtime.instance import Instance, InstanceLocal

if TYPE_CHECKING:
    from PySide6 import QtGui

__all__ = ["Viewer", "OrbitCamera"]

#: A compute device: an ``RS_DeviceLocation`` or a string like ``"cuda:0"``.
Device = Union["rs.RS_DeviceLocation", str]

#: How a viewer is pointed at a world: nothing (default local instance), a
#: ``(device, world)`` to launch locally, or a constructed :class:`Instance`.
InstanceSpec = Union[None, Tuple[Optional[Device], "rs.RSC_World"], Instance]


def _look_at_quaternion(eye, center, global_up=(0.0, 0.0, 1.0)) -> list:
    """Quaternion (``[x, y, z, w]``) aiming a +X-forward / +Z-up camera at center."""
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    global_up = np.asarray(global_up, dtype=np.float64)
    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(global_up, forward)
    right = right / np.linalg.norm(right)
    local_up = np.cross(forward, right)
    rotation_matrix = np.column_stack((forward, right, local_up))
    q = Quaternion(matrix=rotation_matrix)
    return [q.x, q.y, q.z, q.w]


def _rotate(quat_xyzw, vector) -> np.ndarray:
    """Rotate a 3-vector by a quaternion given in ``[x, y, z, w]`` order."""
    q = Quaternion(w=quat_xyzw[3], x=quat_xyzw[0], y=quat_xyzw[1], z=quat_xyzw[2])
    return np.asarray(q.rotate(vector), dtype=np.float64)


class OrbitCamera:
    """Mouse-driven orbit camera: position, focus point, and orbit distance.

    Holds the camera pose and accumulates pending mouse deltas, applied once per
    control step to produce an absolute position/orientation for the free camera.
    """

    def __init__(
        self,
        position: Tuple[float, float, float] = (-0.6, 0.0, 0.4),
        focus: Tuple[float, float, float] = (0.0, 0.0, 0.15),
    ) -> None:
        """Initialize the orbit camera looking from ``position`` at ``focus``."""
        self.position = list(position)
        self.focus = list(focus)
        self.quaternion = _look_at_quaternion(self.position, self.focus)
        self.distance = float(
            np.linalg.norm(np.array(self.focus) - np.array(self.position))
        )
        self.pending_orbit_pitch = 0.0
        self.pending_orbit_yaw = 0.0
        self.pending_pan_x = 0.0
        self.pending_pan_y = 0.0
        self.pending_zoom = 0.0

    def fit_to_bounds(self, bounds_min, bounds_max) -> None:
        """Place the camera on a diagonal so the given AABB fills the view."""
        bounds_min = np.asarray(bounds_min, dtype=np.float64)
        bounds_max = np.asarray(bounds_max, dtype=np.float64)
        center = (bounds_min + bounds_max) / 2.0
        max_size = max(float(np.max(bounds_max - bounds_min)), 0.1)
        offset = max_size * 1.4
        self.focus = center.tolist()
        self.position = (center + offset).tolist()
        self.quaternion = _look_at_quaternion(self.position, self.focus)
        self.distance = float(np.linalg.norm(center + offset - center))

    def apply_pending(self) -> None:
        """Consume accumulated mouse deltas and recompute the camera pose."""
        if self.pending_zoom != 0.0:
            self.distance *= 1.0 - self.pending_zoom * 0.1
            self.distance = float(np.clip(self.distance, 0.05, 100.0))
            self.pending_zoom = 0.0

        if self.pending_pan_x != 0.0 or self.pending_pan_y != 0.0:
            right = _rotate(self.quaternion, [0, 1, 0])
            up = _rotate(self.quaternion, [0, 0, 1])
            pan_speed = self.distance * 0.001
            offset = (
                right * self.pending_pan_x * pan_speed
                + up * self.pending_pan_y * pan_speed
            )
            self.focus = (np.array(self.focus) + offset).tolist()
            self.pending_pan_x = 0.0
            self.pending_pan_y = 0.0

        if self.pending_orbit_pitch != 0.0 or self.pending_orbit_yaw != 0.0:
            cam_to_target = np.array(self.focus) - np.array(self.position)
            right_axis = _rotate(self.quaternion, [0, 1, 0])
            pitch = Quaternion(axis=right_axis, angle=self.pending_orbit_pitch)
            yaw = Quaternion(axis=[0, 0, 1], angle=self.pending_orbit_yaw)
            rotated = yaw.rotate(pitch.rotate(cam_to_target))
            self.position = (np.array(self.focus) - rotated).tolist()
            self.quaternion = _look_at_quaternion(self.position, self.focus)
            self.pending_orbit_pitch = 0.0
            self.pending_orbit_yaw = 0.0
        else:
            forward = _rotate(self.quaternion, [1, 0, 0])
            self.position = (np.array(self.focus) - forward * self.distance).tolist()


class Viewer:
    """Watch a rise-origin world in an interactive window or record it.

    The viewer injects and owns one free-perspective camera; you keep full
    control of your structures through ``callback``. The same class records to
    mp4 and grabs stills headlessly (no window).
    """

    _MOUSE_SENSITIVITY = 0.005

    def __init__(
        self,
        name: str = "rise-origin-viewer",
        instance: InstanceSpec = None,
        callback: Optional[Callable] = None,
        hud: Optional[Callable[["QtGui.QPainter", "Viewer"], None]] = None,
        *,
        title: str = "Rise Origin Viewer",
        width: int = 1280,
        height: int = 720,
        samples_per_pixel: int = 4,
        max_depth: int = 3,
        field_of_view: float = 60.0,
        camera_position: Tuple[float, float, float] = (-0.6, 0.0, 0.4),
        camera_focus: Tuple[float, float, float] = (0.0, 0.0, 0.15),
    ) -> None:
        """Configure a viewer over a local or remote instance.

        Args:
            name: Unique viewer identity. Used to name the injected camera and
                its controller, so it must not collide with your own entities.
            instance: What to view. ``None`` spins up a default-device
                :class:`InstanceLocal` over an empty ``RSC_World``; a
                ``(device, world)`` tuple launches ``world`` on ``device``; or
                pass a constructed :class:`Instance` (remote attach is planned).
            callback: ``callback(world_name, time, controllers)`` invoked each
                control step to actuate your structures, in local or remote mode.
                Called before the viewer drives its own camera.
            hud: ``hud(painter, viewer)`` invoked each repaint to draw an overlay
                on top of the live feed. ``painter`` is a ``PySide6.QtGui.QPainter``
                already aimed at the window; ``viewer`` is this :class:`Viewer`
                (read ``viewer.time``, ``viewer.handle``, etc. to decide what to
                draw). Interactive mode only.
            title: Window title.
            width: Frame / window width in pixels.
            height: Frame / window height in pixels.
            samples_per_pixel: Ray-tracing samples per pixel; higher is cleaner
                but slower.
            max_depth: Maximum ray bounces.
            field_of_view: Vertical field of view in degrees.
            camera_position: Initial camera position in world meters.
            camera_focus: Initial point the camera looks at / orbits.
        """
        self.time: float = 0.0
        self.frame_index: int = 0

        self._name = name
        self._controller_name = f"{name}#controller"
        self._user_callback = callback
        self._hud = hud
        self._title = title
        self._width = width
        self._height = height
        self._samples_per_pixel = samples_per_pixel
        self._max_depth = max_depth
        self._field_of_view = field_of_view

        self._instance, self._world = self._resolve_instance(instance)
        self._camera = OrbitCamera(camera_position, camera_focus)
        self._keys: set = set()

        self._is_camera_injected = False
        self._is_stop_requested = False

        # Latest frame, shared between the step-callback thread (writer) and the
        # Qt GUI thread (reader). Guarded because they run concurrently.
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # Headless recording state.
        self._writer = None
        self._is_collecting_frames = False
        self._video_fps: int = 0
        self._video_frame_count: int = 0

    @property
    def handle(self) -> Optional["rs.RS_WorldHandle"]:
        """The running ``RS_WorldHandle``, or ``None`` before a run starts."""
        return (
            self._instance.handle if isinstance(self._instance, InstanceLocal) else None
        )

    def is_key_down(self, key: "QtCore.Qt.Key") -> bool:
        """Whether a key is currently held, for reading input from a hud callback."""
        return int(key) in self._keys

    def run_interactive(self, *, display_fps: int = 60) -> None:
        """Open a live window and run until it is closed (blocking).

        The world runs detached on a background thread; this call owns the main
        thread for the Qt event loop and returns only when the window closes.
        Control the world from your ``callback`` or :attr:`handle`.

        Args:
            display_fps: Window refresh rate. Independent of the simulation
                control frequency.
        """
        from PySide6 import QtCore, QtGui, QtWidgets

        self._inject_camera()
        self._ensure_frequencies()

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = self._make_window(QtCore, QtGui, QtWidgets, display_fps)
        window.show()
        self._instance.hello(self._world, self._interactive_step)
        app.exec()
        self._is_stop_requested = True
        if (
            isinstance(self._instance, InstanceLocal)
            and self._instance.handle is not None
        ):
            self._instance.handle.terminate()

    def run_record(self, path: str, duration: float, *, video_fps: int = 60) -> None:
        """Start a headless run streaming frames to an mp4 (non-blocking).

        Frames are written as rendered; the file is finalized by :meth:`wait`.

        Args:
            path: Output mp4 path.
            duration: Simulation time in seconds to record.
            video_fps: Output video frame rate. Independent of the simulation
                control frequency: the writer time-locks each captured frame to
                ``sim_time`` and emits as many video frames as needed so the
                video plays back in real time regardless of control rate.
        """
        import imageio

        self._video_fps = video_fps
        self._is_collecting_frames = True
        self._writer = imageio.get_writer(path, fps=video_fps, macro_block_size=8)
        self._start_headless(duration)

    def run_headless(self, duration: Optional[float] = None) -> None:
        """Start a headless run with no window and no file output (non-blocking).

        The current frame is refreshed each step so a ``callback`` can call
        :meth:`snapshot` on demand. Call :meth:`wait` to block until it ends.
        """
        self._is_collecting_frames = False
        self._start_headless(duration)

    def snapshot(self, path: str) -> None:
        """Save the most recent rendered frame to an image file."""
        import imageio

        with self._frame_lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            raise RuntimeError("no frame captured yet; start a run first")
        imageio.imwrite(path, frame)

    def wait(self) -> None:
        """Block until the running world ends, then finalize resources."""
        handle = self.handle
        if handle is None:
            return
        try:
            handle.wait_for_end()
        except KeyboardInterrupt:
            handle.terminate()
        finally:
            self._is_stop_requested = True
            if self._writer is not None:
                self._writer.close()
                self._writer = None
                self._is_collecting_frames = False

    def _is_world_ended(self) -> bool:
        """Whether the attached world has run to its end."""
        handle = self.handle
        return handle is not None and handle.is_end()

    def _resolve_instance(self, instance: InstanceSpec):
        """Turn the ``instance`` spec into an (instance, world) pair."""
        if instance is None:
            return InstanceLocal(), rs.RSC_World()
        if isinstance(instance, tuple):
            device, world = instance
            return InstanceLocal(device), world
        if isinstance(instance, Instance):
            raise NotImplementedError(
                "attaching to a prebuilt/remote Instance is not yet supported; "
                "pass None or a (device, world) tuple"
            )
        raise TypeError(f"unsupported instance spec: {type(instance)!r}")

    def _start_headless(self, duration: Optional[float]) -> None:
        """Shared launcher for :meth:`run_record` and :meth:`run_headless`."""
        self._inject_camera()
        self._ensure_frequencies()
        self.frame_index = 0
        self._video_frame_count = 0
        if duration is not None:
            self._world.simulation.runtime.max_time = duration
        self._instance.hello(self._world, self._headless_step)

    def _inject_camera(self) -> None:
        """Append the viewer's free camera and its controller to the world (once)."""
        if self._is_camera_injected:
            return
        camera = rsc.entity.CameraPerspectiveFree(
            self._name,
            origin_position=tuple(self._camera.position),
            orientation=tuple(self._camera.quaternion),
            image_width=self._width,
            image_height=self._height,
            samples_per_pixel=self._samples_per_pixel,
            max_depth=self._max_depth,
            field_of_view=self._field_of_view,
            record=rsc.entity.CameraPerspectiveRecord(
                f"{self._name}#record",
                has_rgb=True,
                target_controller_reference=self._controller_name,
            ),
            signal=rsc.entity.CameraPerspectiveSignal(
                f"{self._name}#signal",
                source_controller_name=self._controller_name,
            ),
        )
        self._world.entities.append(rs.RSC_EntityVariant(camera))
        self._world.controllers.append(
            rs.RSC_ControllerVariant(
                rsc.controller.ControllerCallback(self._controller_name)
            )
        )
        self._is_camera_injected = True

    def _ensure_frequencies(self) -> None:
        """Give the simulation sensible control/record rates if unset."""
        simulation = self._world.simulation
        if simulation.control.frequency <= 0:
            simulation.control.frequency = 20.0
        if simulation.record.frequency <= 0:
            simulation.record.frequency = 20.0

    def _interactive_step(self, world_name, sim_time, controllers) -> None:
        """Per-step callback for interactive mode: actuate, then drive the camera."""
        self.time = sim_time
        if self._user_callback is not None:
            self._user_callback(
                world_name, sim_time, self._user_controllers(controllers)
            )
        controller = self._find_controller(controllers)
        if controller is not None:
            self._camera.apply_pending()
            self._drive_camera(
                controller, self._camera.position, self._camera.quaternion
            )
            self._capture(controller)
        if self._is_stop_requested:
            handle = self.handle
            if handle is not None:
                handle.terminate()

    def _headless_step(self, world_name, sim_time, controllers) -> None:
        """Per-step callback for headless modes: actuate, then capture."""
        self.time = sim_time
        if self._user_callback is not None:
            self._user_callback(
                world_name, sim_time, self._user_controllers(controllers)
            )
        controller = self._find_controller(controllers)
        if controller is None:
            return
        self._camera.apply_pending()
        self._drive_camera(controller, self._camera.position, self._camera.quaternion)
        if self._capture(controller):
            self.frame_index += 1
            if self._is_collecting_frames and self._writer is not None:
                # Time-lock: emit as many video frames as sim_time has elapsed,
                # regardless of control frequency.  Uses integer division
                # (count / fps) to avoid floating-point drift across many steps.
                with self._frame_lock:
                    frame = self._frame
                while self._video_frame_count / self._video_fps <= sim_time:
                    self._writer.append_data(frame)
                    self._video_frame_count += 1

    def _capture(self, controller) -> bool:
        """Read the camera RGB into the shared frame buffer; return True if updated."""
        for record in controller.records():
            if isinstance(record, rs.RS_WorldCameraPerspectiveRecord):
                if record.is_valid() and record.width() > 0:
                    with self._frame_lock:
                        self._frame = np.array(record.rgb.pixels(), copy=True)
                    return True
        return False

    def _find_controller(self, controllers):
        """Find the viewer's own camera controller among the step's controllers."""
        for controller in controllers:
            if controller.name() == self._controller_name:
                return controller
        return None

    def _user_controllers(self, controllers):
        """The step's controllers minus the viewer's own injected camera one."""
        return [c for c in controllers if c.name() != self._controller_name]

    def _make_window(self, QtCore, QtGui, QtWidgets, display_fps: int):
        """Build the interactive window (defined here to keep PySide6 lazy)."""
        viewer = self

        class _ViewerWindow(QtWidgets.QWidget):
            def __init__(self) -> None:
                super().__init__()
                self.setWindowTitle(viewer._title)
                self.resize(viewer._width, viewer._height)
                self._qimage = QtGui.QImage()
                self._last_mouse = QtCore.QPoint()
                # Poll the shared frame at the display rate (main-thread repaint).
                self._timer = QtCore.QTimer(self)
                self._timer.timeout.connect(self._refresh)
                self._timer.start(max(1, 1000 // display_fps))

            def _refresh(self) -> None:
                with viewer._frame_lock:
                    frame = None if viewer._frame is None else viewer._frame
                    if frame is not None:
                        height, width, _ = frame.shape
                        contiguous = np.ascontiguousarray(frame)
                        self._qimage = QtGui.QImage(
                            contiguous.data,
                            width,
                            height,
                            3 * width,
                            QtGui.QImage.Format.Format_RGB888,
                        ).copy()
                self.update()
                if viewer._is_world_ended():
                    self.close()

            def paintEvent(self, event) -> None:
                painter = QtGui.QPainter(self)
                if not self._qimage.isNull():
                    painter.drawImage(self.rect(), self._qimage)
                if viewer._hud is not None:
                    viewer._hud(painter, viewer)
                painter.end()

            def mousePressEvent(self, event) -> None:
                self._last_mouse = event.position().toPoint()

            def wheelEvent(self, event) -> None:
                viewer._camera.pending_zoom += event.angleDelta().y() / 120.0

            def mouseMoveEvent(self, event) -> None:
                pos = event.position().toPoint()
                dx = pos.x() - self._last_mouse.x()
                dy = pos.y() - self._last_mouse.y()
                buttons = event.buttons()
                if buttons & QtCore.Qt.MouseButton.LeftButton:
                    viewer._camera.pending_orbit_pitch += dy * viewer._MOUSE_SENSITIVITY
                    viewer._camera.pending_orbit_yaw += -dx * viewer._MOUSE_SENSITIVITY
                elif buttons & QtCore.Qt.MouseButton.RightButton:
                    viewer._camera.pending_pan_x += dx
                    viewer._camera.pending_pan_y += dy
                elif buttons & QtCore.Qt.MouseButton.MiddleButton:
                    viewer._camera.pending_zoom += dy * 0.05
                self._last_mouse = pos

            def keyPressEvent(self, event) -> None:
                if not event.isAutoRepeat():
                    viewer._keys.add(int(event.key()))
                if event.key() == QtCore.Qt.Key.Key_Escape:
                    self.close()
                elif event.key() == QtCore.Qt.Key.Key_R:
                    viewer._camera.__init__(
                        tuple(viewer._camera.position), tuple(viewer._camera.focus)
                    )

            def keyReleaseEvent(self, event) -> None:
                if not event.isAutoRepeat():
                    viewer._keys.discard(int(event.key()))

        return _ViewerWindow()

    @staticmethod
    def _drive_camera(controller, position, quaternion) -> None:
        """Write an absolute pose into the camera's position/orientation signals."""
        for signal in controller.signals():
            if isinstance(signal, rs.RS_WorldCameraPerspectiveSignal):
                positions = signal.position_signals().position()
                orientations = signal.orientation_signals().orientation()
                if len(positions) > 0:
                    positions[0, :] = position
                if len(orientations) > 0:
                    orientations[0, :] = quaternion
