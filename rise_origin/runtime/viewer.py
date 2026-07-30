# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Interactive Qt viewer and headless recorder for rise-origin worlds.

:class:`Viewer` is your eyes on a running world. It builds an
:class:`~rise_origin.RS_WorldSession` and adds a camera component that injects one
free-perspective camera, streams that camera's ray-traced frames into a window, and
gives you a mouse-driven orbit, pan, and zoom camera. Your control loop runs as the
``callback`` on the session's Core component, leaving you free to actuate your
structures.

Pass nothing to spin up a default local instance over an empty world, a
``(device, world)`` tuple to run a specific world on a device, or a ``url`` string to
connect to a remote instance.

Example::

    from rise_origin import rsc
    from rise_origin.runtime.viewer import Viewer

    world = rsc.World(name="your_world", entities=[...])

    def control(time, controllers):
        ...  # actuate your structures

    viewer = Viewer("overview", instance=("cuda:0", world), callback=control)
    viewer.run_interactive()                       # live window, runs until closed
    # or, headless:
    viewer.run_record("demo.mp4", duration=8.0); viewer.wait()

Reserved interaction: ``Esc`` closes. Left-drag orbits, right-drag pans,
scroll and middle-drag zoom.
"""

from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

import numpy as np

import rise_origin as rs

if TYPE_CHECKING:
    from PySide6 import QtGui

__all__ = ["Viewer"]

#: A compute device: an ``RS_DeviceLocation`` or a string like ``"cuda:0"``.
Device = Union["rs.RS_DeviceLocation", str]

#: How a viewer is pointed at a world: nothing for a default local instance over an
#: empty world, a ``(device, world)`` to launch locally, or a ``url`` string to
#: connect to a remote instance.
InstanceSpec = Union[None, str, Tuple[Optional[Device], "rs.RSC_World"]]


class Viewer:
    """Watch a rise-origin world in an interactive window or record it.

    The viewer adds one free-perspective camera component to a session; you keep
    full control of your structures through ``callback`` on the Core component. The
    same class records to mp4 and grabs stills headlessly with no window.
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
        defocus_angle: float = 0.0,
        focus_distance: float = 1.0,
        camera_position: Tuple[float, float, float] = (0.0, 0.0, 0.4),
        camera_focus: Tuple[float, float, float] = (0.0, 0.0, 0.15),
        camera_min_distance: float = 0.05,
        camera_max_distance: float = float("inf"),
        camera_zoom_factor: float = 0.95,
    ) -> None:
        """Configure a viewer over a local or remote instance.

        Args:
            name: Unique viewer identity. Used to name the injected camera and
                its controller, so it must not collide with your own entities.
            instance: What to view. ``None`` spins up a default-device instance
                over an empty ``RSC_World``; a ``(device, world)`` tuple launches
                ``world`` on ``device``; a ``url`` string connects to a remote
                instance.
            callback: ``callback(time, controllers)`` invoked each control step to
                actuate your structures. Runs on the session's Core component over
                the world's controllers.
            hud: ``hud(painter, viewer)`` invoked each repaint to draw an overlay
                on top of the live feed. ``painter`` is a ``PySide6.QtGui.QPainter``
                already aimed at the window; ``viewer`` is this :class:`Viewer`,
                so read ``viewer.time``, ``viewer.handle`` to decide what to draw.
                Interactive mode only.
            title: Window title.
            width: Frame and window width in pixels.
            height: Frame and window height in pixels.
            samples_per_pixel: Ray-tracing samples per pixel; higher is cleaner
                but slower.
            max_depth: Maximum ray bounces.
            field_of_view: Vertical field of view in degrees.
            defocus_angle: Depth-of-field defocus angle in degrees; 0 disables
                depth of field.
            focus_distance: Focus distance in meters; matters only with a
                non-zero defocus_angle.
            camera_position: Initial camera position in world meters.
            camera_focus: Initial point the camera looks at and orbits.
            camera_min_distance: Closest the camera may orbit its focus, in
                meters. The distance is held inside this range every step, so a
                camera built nearer than this is seated before its first frame.
            camera_max_distance: Farthest the camera may orbit its focus, in
                meters. Infinite by default, so a distant focus is left alone.
            camera_zoom_factor: What one wheel notch scales the orbit distance
                by. Below 1 zooms in, and the scaling holds however many notches
                arrive between two control steps.
        """
        self.time: float = 0.0
        self.frame_index: int = 0

        self._name = name
        self._instance = instance
        self._user_callback = callback
        self._hud = hud
        self._title = title
        self._width = width
        self._height = height
        self._camera = rs.RS_WorldSessionComponentCamera(
            name,
            self._on_callback,
            camera_position,
            camera_focus,
            width,
            height,
            samples_per_pixel,
            max_depth,
            field_of_view,
            defocus_angle,
            focus_distance,
            camera_min_distance,
            camera_max_distance,
            camera_zoom_factor,
        )
        self._keys: set = set()
        self._is_stop_requested = False

        self._world = None
        self._engine = None
        self._session = None
        self._handle: Optional["rs.RS_WorldHandle"] = None

        # Headless recording state.
        self._writer = None
        self._is_collecting_frames = False
        self._video_fps: int = 0
        self._video_frame_count: int = 0

    @property
    def handle(self) -> Optional["rs.RS_WorldHandle"]:
        """The running world handle, or ``None`` before a run starts."""
        return self._handle

    def is_key_down(self, key: "QtCore.Qt.Key") -> bool:
        """Whether a key is currently held, for reading input from a hud callback."""
        return int(key) in self._keys

    def run_interactive(self, *, display_fps: int = 60) -> None:
        """Open a live window and run until it is closed (blocking).

        The world runs detached on a background thread; this call owns the main
        thread for the Qt event loop and returns only when the window closes.
        Control the world from your ``callback`` or :attr:`handle`.

        Args:
            display_fps: Window refresh rate. Independent of the control rate.
        """
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = self._make_window(display_fps)
        window.show()
        self._start_session()
        app.exec()
        self._is_stop_requested = True
        if self._handle is not None:
            self._handle.terminate()

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
        self._start_session(duration)

    def run_headless(self, duration: Optional[float] = None) -> None:
        """Start a headless run with no window and no file output (non-blocking).

        The current frame is refreshed each step so a ``callback`` can call
        :meth:`snapshot` on demand. Call :meth:`wait` to block until it ends.
        """
        self._is_collecting_frames = False
        self._start_session(duration)

    def snapshot(self, path: str) -> None:
        """Save the most recent rendered frame to an image file."""
        import imageio

        frame = self._camera.frame()
        if frame is None:
            raise RuntimeError("no frame captured yet; start a run first")
        imageio.imwrite(path, frame)

    def wait(self) -> None:
        """Block until the running world ends, then detach components and finalize resources.

        Runs the session so its components (such as the VR present thread and its OpenXR instance)
        are torn down here, while every library is still loaded, rather than left to interpreter
        finalization where the teardown races the library unload.
        """
        if self._session is None:
            return
        try:
            self._session.run(is_blocking=True)
        except KeyboardInterrupt:
            if self._handle is not None:
                self._handle.terminate()
            self._session.run(is_blocking=True)
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

    def _start_session(self, duration: Optional[float] = None) -> None:
        """Resolve the instance spec, start the world, and build the session."""
        self.frame_index = 0
        self._video_frame_count = 0
        if self._instance is None:
            device, world, url = None, rs.RSC_World(), None
        elif isinstance(self._instance, str):
            device, world, url = None, None, self._instance
        elif isinstance(self._instance, tuple):
            device, world, url = self._instance[0], self._instance[1], None
        else:
            raise TypeError(f"unsupported instance spec: {type(self._instance)!r}")

        self._world = world
        self._engine = rs.RiseOrigin() if device is None else rs.RiseOrigin(device)
        if url is not None:
            handle, callback = self._engine.connect(url)
        else:
            self._ensure_frequencies()
            if duration is not None:
                world.simulation.runtime.max_time = duration
            handle, callback = self._engine.hello(world)

        self._session = rs.RS_WorldSession(handle, callback)
        core = self._session.get_core_component()
        if self._user_callback is not None:
            core.set_callback(self._user_callback)
        self._session.add_component(self._name, self._camera)
        self._handle = core.get_handle()

    def _ensure_frequencies(self) -> None:
        """Give the simulation sensible control/record rates if unset."""
        simulation = self._world.simulation
        if simulation.control.frequency <= 0:
            simulation.control.frequency = 20.0
        if simulation.record.frequency <= 0:
            simulation.record.frequency = 20.0

    def _on_callback(self, time) -> None:
        """Track the world time and, when recording, write time-locked frames."""
        self.time = time
        if self._is_stop_requested and self._handle is not None:
            self._handle.terminate()
        if not self._is_collecting_frames or self._writer is None:
            return
        frame = self._camera.frame()
        if frame is None:
            return
        self.frame_index += 1
        while self._video_frame_count / self._video_fps <= time:
            self._writer.append_data(frame)
            self._video_frame_count += 1

    def _make_window(self, display_fps: int):
        """Build the interactive window, defined here to keep PySide6 lazy."""
        from PySide6 import QtCore, QtGui, QtWidgets

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
                frame = viewer._camera.frame()
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
                viewer._camera.move(0.0, 0.0, 0.0, 0.0, event.angleDelta().y() / 120.0)

            def mouseMoveEvent(self, event) -> None:
                pos = event.position().toPoint()
                dx = pos.x() - self._last_mouse.x()
                dy = pos.y() - self._last_mouse.y()
                buttons = event.buttons()
                if buttons & QtCore.Qt.MouseButton.LeftButton:
                    viewer._camera.move(
                        dy * viewer._MOUSE_SENSITIVITY,
                        -dx * viewer._MOUSE_SENSITIVITY,
                        0.0,
                        0.0,
                        0.0,
                    )
                elif buttons & QtCore.Qt.MouseButton.RightButton:
                    viewer._camera.move(0.0, 0.0, float(dx), float(dy), 0.0)
                elif buttons & QtCore.Qt.MouseButton.MiddleButton:
                    viewer._camera.move(0.0, 0.0, 0.0, 0.0, dy * 0.05)
                self._last_mouse = pos

            def keyPressEvent(self, event) -> None:
                if not event.isAutoRepeat():
                    viewer._keys.add(int(event.key()))
                if event.key() == QtCore.Qt.Key.Key_Escape:
                    self.close()

            def keyReleaseEvent(self, event) -> None:
                if not event.isAutoRepeat():
                    viewer._keys.discard(int(event.key()))

        return _ViewerWindow()
