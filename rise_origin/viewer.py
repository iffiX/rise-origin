# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Interactive pygame viewer and headless recorder for RiseOrigin simulations.

:class:`Viewer` wraps the boilerplate of driving a RiseOrigin simulation with a live
window: it owns a free-perspective camera (named ``"#viewer_main_camera"``),
streams the ray-traced frames to a pygame window, and gives you a mouse-driven
orbit/pan/zoom camera — while leaving the whole keyboard free for controlling
your robot.

You supply a ``callback`` to actuate your structures each control step; the
viewer only takes over its own camera. The same class records to mp4 headlessly
(no window) so examples can produce gallery media.

Example::

    import rise_origin as rs
    from rise_origin.viewer import Viewer

    def control(sim_id, controllers):
        for ctype, c in controllers:
            if ctype == rs.RSE_SimulationControllerType.RSE_SIMULATION_STRUCTURE_CONTROLLER:
                signals = c.signal().hinge_signals()
                if signals is not None:
                    signals[0] = 0.5  # drive joint 0

    rise = rs.RiseOrigin("cuda:0")
    viewer = Viewer(rise, config, callback=control)

    # Pick exactly one run mode (they are alternatives, not run together):
    viewer.run_interactive()                     # live pygame window
    viewer.run_record("demo.mp4", duration=8.0)  # headless, write an mp4
    viewer.run_headless(duration=8.0)            # headless, snapshot() on demand

    # Operate on the simulation while it runs:
    # Use viewer.handle, you can pause/resume/terminate the simulation, and add additional
    # structures or cameras etc or delete them etc.
    viewer.handle.pause()

    # Wait for the simulation to end
    viewer.wait()

Reserved keys: ``Esc`` quits, ``R`` resets the camera to frame the scene. Mouse:
left-drag orbits, right-drag pans, scroll/middle-drag zooms. Every other key and
event is forwarded to your ``on_event`` handler and ``key_bindings``.
"""

import threading
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
from pyquaternion import Quaternion

import rise_origin as rs
from rise_origin.create import CameraCreator, RecordCreator
from rise_origin.create.utils import Vec3

__all__ = ["Viewer", "OrbitCamera"]

#: Name of the free camera the viewer injects and drives. The leading ``#``
#: keeps it from colliding with user-named cameras.
MAIN_CAMERA_NAME = "#viewer_main_camera"

_CONTROLLER = rs.RSE_SimulationControllerType


def _look_at_quaternion(eye, center, global_up=(0.0, 0.0, 1.0)) -> List[float]:
    """Quaternion that points a camera at ``center`` from ``eye``.

    Assumes the camera's local forward axis is +X and local up is +Z.

    Args:
        eye: Camera position ``[x, y, z]``.
        center: Target point to look at ``[x, y, z]``.
        global_up: World up direction.

    Returns:
        Orientation quaternion as ``[x, y, z, w]``.
    """
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

    Holds the camera pose and accumulates pending mouse deltas, which are
    applied once per control step to produce an absolute position/orientation
    for the simulation's free camera.
    """

    def __init__(
        self,
        position: Vec3 = (-0.6, 0.0, 0.4),
        focus: Vec3 = (0.0, 0.0, 0.15),
    ):
        """Initialize the orbit camera looking from ``position`` at ``focus``.

        Args:
            position: Initial camera position in world meters.
            focus: Initial point the camera orbits around and looks at.
        """
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
        """Place the camera on a diagonal so the given AABB fills the view.

        Args:
            bounds_min: Minimum corner ``[x, y, z]`` of the scene bounds.
            bounds_max: Maximum corner ``[x, y, z]`` of the scene bounds.
        """
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
    """Drive a RiseOrigin simulation in an interactive window or record it to mp4.

    The viewer injects and owns one free-perspective camera; you keep full
    control of your structures through ``callback``.
    """

    _MOUSE_SENSITIVITY = 0.005

    def __init__(
        self,
        rise: "rs.RiseOrigin",
        config: "rs.RS_Config",
        callback: Optional[Callable] = None,
        hud: Optional[Callable] = None,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
        samples_per_pixel: int = 1,
        max_depth: int = 4,
        field_of_view: float = 60.0,
        camera_position: Vec3 = (-0.6, 0.0, 0.4),
        camera_focus: Vec3 = (0.0, 0.0, 0.15),
        track_structure: Optional[str] = None,
        record_structures: Optional[List[str]] = None,
        key_bindings: Optional[Dict[int, Callable]] = None,
        on_event: Optional[Callable] = None,
    ):
        """Configure a viewer over a built simulation config.

        Args:
            rise: A :class:`rise_origin.RiseOrigin` instance bound to a device.
            config: The simulation config to run. The viewer appends its own
                camera to it.
            callback: ``callback(sim_id, controllers)`` invoked each control step
                to actuate your structures. Called before the viewer touches its
                camera.
            hud: ``hud(surface, viewer)`` invoked each rendered frame to draw an
                overlay on the pygame surface (interactive mode only).
            width: Window / frame width in pixels (> 0).
            height: Window / frame height in pixels (> 0).
            fps: Window refresh / mp4 playback frame rate.
            samples_per_pixel: Ray-tracing samples per pixel (> 0). Higher is
                cleaner but slower.
            max_depth: Maximum ray bounces (> 0). Higher captures more indirect
                light but is slower.
            field_of_view: Vertical field of view in degrees (> 0).
            camera_position: Initial camera position in world meters.
            camera_focus: Initial point the camera looks at / orbits.
            track_structure: Name of a structure to follow. If given (and that
                structure is found), the camera frames it initially and its
                focus follows that structure's center every step (you keep mouse
                orbit/zoom around it). If ``None`` (default), the camera frames
                the scene once and then holds. The tracked structure is recorded
                automatically (the viewer reads its record to follow it), so it
                need not be listed in ``record_structures``.
            record_structures: Names of the structures to add a structure record
                for, so a ``callback`` can read ``structure_record()`` (positions,
                time, rigid-body state, ...). Recording is expensive, so by
                default (``None``) **no** structure is recorded; list only the
                structures whose records you actually read. Names with an existing
                record config are left as-is; unknown names raise a warning.
            key_bindings: Optional ``{pygame_key: callable(viewer)}`` map invoked
                on key-down for custom controls.
            on_event: Optional ``on_event(event, viewer)`` handler receiving raw
                pygame events for custom interaction.
        """
        self._rise = rise
        self._config = config
        self._user_callback = callback
        self._hud = hud
        self._width = width
        self._height = height
        self._fps = fps
        self._samples_per_pixel = samples_per_pixel
        self._max_depth = max_depth
        self._field_of_view = field_of_view
        self._key_bindings = key_bindings or {}
        self._on_event = on_event

        self._camera = OrbitCamera(camera_position, camera_focus)
        self._track_structure = track_structure
        self._record_structures = record_structures
        self._handle: Optional["rs.RS_SimulationHandle"] = None

        # Latest rendered frame (numpy) plus the pygame surfaces built from it.
        # Surfaces are created in the sim-callback thread and only blitted in the
        # window thread, matching pygame's threading expectations.
        self._frame: Optional[np.ndarray] = None
        self._surface = None
        self._screen = None
        self._pygame = None

        # Focus center the camera orbits. Starts at camera_focus and, when
        # track_structure is set, follows that structure's center each step.
        self._center: np.ndarray = np.asarray(camera_focus, dtype=np.float64)
        self._bounds_min: Optional[np.ndarray] = None
        self._bounds_max: Optional[np.ndarray] = None

        self._stop_requested = False
        self._camera_injected = False
        self._records_ensured = False

        # Run state. Run modes start the simulation and return immediately
        # (detached); call wait() to block and finalize the window / mp4.
        self._window_thread: Optional[threading.Thread] = None
        self._writer = None  # open imageio mp4 writer while recording
        self._frame_index = 0
        self._collect_frames = False
        self._record_orbit = True
        self._record_elevation = 0.0

    ################################################################
    # Public API
    ################################################################

    @property
    def handle(self) -> Optional["rs.RS_SimulationHandle"]:
        """The running :class:`RS_SimulationHandle`, or ``None`` before run."""
        return self._handle

    def run_interactive(self) -> None:
        """Start the simulation with a live pygame window (non-blocking).

        Returns immediately with the simulation running and a window open in a
        background thread. Operate via :attr:`handle`, and call :meth:`wait` to
        block until the simulation ends or the window is closed.
        """
        import pygame

        self._pygame = pygame
        self._inject_camera()
        self._ensure_records()
        self._ensure_frequencies()

        self._window_thread = threading.Thread(
            target=self._window_loop, args=(pygame,), daemon=True
        )
        self._window_thread.start()
        self._config.simulation_config.runtime_save_record = False
        self._handle = self._rise.run_sim(
            self._config, 0, callback=self._interactive_step
        )

    def run_record(
        self,
        path: str,
        duration: float,
        *,
        orbit: bool = True,
        elevation_deg: float = 25.0,
    ) -> None:
        """Start a headless run that streams frames to an mp4 (non-blocking).

        Returns immediately with the simulation running. Frames are written to
        ``path`` as they are rendered; the file is finalized by :meth:`wait`, so
        you must call ``viewer.wait()`` to produce a complete video.

        Args:
            path: Output ``.mp4`` file path.
            duration: Simulated seconds to record.
            orbit: If ``True``, slowly orbit the camera around the scene; if
                ``False``, hold a fixed framing.
            elevation_deg: Camera elevation angle above the horizon, in degrees.
        """
        import imageio

        self._collect_frames = True
        self._writer = imageio.get_writer(path, fps=self._fps, macro_block_size=8)
        self._start_headless(duration, orbit, elevation_deg)

    def run_headless(
        self,
        duration: Optional[float] = None,
        *,
        orbit: bool = True,
        elevation_deg: float = 25.0,
    ) -> None:
        """Start a headless run with no window and no file output (non-blocking).

        Returns immediately with the simulation running. The current frame is
        refreshed each control step, so a ``callback`` can call :meth:`snapshot`
        whenever it likes — e.g. to grab stills at chosen times. Reference the
        viewer from your callback's closure to call ``viewer.snapshot(...)``.
        Call :meth:`wait` to block until the run ends.

        Args:
            duration: Simulated seconds to run, or ``None`` to run until the
                config's stop condition ends the simulation.
            orbit: If ``True``, slowly orbit the camera around the scene; if
                ``False``, hold a fixed framing.
            elevation_deg: Camera elevation angle above the horizon, in degrees.
        """
        self._collect_frames = False
        self._start_headless(duration, orbit, elevation_deg)

    def wait(self) -> None:
        """Block until the running simulation ends, then finalize resources.

        Joins the interactive window thread (if any) and closes the mp4 writer
        (if recording), so the video file is complete on return. Safe to call
        even if no run is active.
        """
        if self._handle is None:
            return
        try:
            self._handle.wait_for_end()
        except KeyboardInterrupt:
            self._handle.terminate()
        finally:
            self._stop_requested = True
            if self._window_thread is not None:
                self._window_thread.join(timeout=1.0)
                self._window_thread = None
            if self._writer is not None:
                self._writer.close()
                self._writer = None
                self._collect_frames = False

    def snapshot(self, path: str) -> None:
        """Save the most recent rendered frame to an image file.

        Args:
            path: Output image path (e.g. ``.png``).

        Raises:
            RuntimeError: If no frame has been rendered yet.
        """
        import imageio

        if self._frame is None:
            raise RuntimeError(
                "no frame captured yet; call run_interactive/run_headless/run_record first"
            )
        imageio.imwrite(path, self._frame)

    def _start_headless(
        self, duration: Optional[float], orbit: bool, elevation_deg: float
    ) -> None:
        """Shared launcher for :meth:`run_record` and :meth:`run_headless`."""
        self._inject_camera()
        self._ensure_records()
        self._ensure_frequencies()

        self._frame_index = 0
        self._record_orbit = orbit
        self._record_elevation = float(np.deg2rad(elevation_deg))

        self._config.simulation_config.runtime_save_record = False
        if duration is not None:
            self._config.simulation_config.runtime_max_time = duration
        self._handle = self._rise.run_sim(
            self._config, 0, callback=self._headless_step
        )

    ################################################################
    # Config setup
    ################################################################

    def _inject_camera(self) -> None:
        """Append the viewer's free camera to the config (once)."""
        if self._camera_injected:
            return
        camera = CameraCreator.free(
            MAIN_CAMERA_NAME,
            origin_position=tuple(self._camera.position),
            orientation=tuple(self._camera.quaternion),
            image_width=self._width,
            image_height=self._height,
            samples_per_pixel=self._samples_per_pixel,
            max_depth=self._max_depth,
            field_of_view=self._field_of_view,
        )
        self._config.camera_configs.append(camera)
        self._camera_injected = True

    def _ensure_frequencies(self) -> None:
        """Give the simulation sensible control/record rates if unset."""
        sim = self._config.simulation_config
        if sim.control_frequency <= 0:
            sim.control_frequency = 20
        if sim.record_frequency <= 0:
            sim.record_frequency = 20

    def _ensure_records(self) -> None:
        """Add structure records for the requested structures (once).

        Recording every structure is wasteful, so the viewer records nothing by
        default. It adds a record only for the structures named in
        ``record_structures`` (whose records a callback reads) plus
        ``track_structure`` (whose record the viewer reads to follow it).
        Structures that already have a record config are left untouched; names
        that match no structure raise a warning.
        """
        if self._records_ensured:
            return
        wanted = set(self._record_structures or [])
        if self._track_structure is not None:
            wanted.add(self._track_structure)
        recorded = {
            record.config.structure_name
            for record in self._config.record_configs
            if record.type == rs.RSE_RecordType.RSE_RECORD_STRUCTURE
        }
        for structure in self._config.structure_configs:
            if structure.name in wanted and structure.name not in recorded:
                self._config.record_configs.append(
                    RecordCreator.structure(
                        f"{structure.name}#viewer_record",
                        structure_name=structure.name,
                    )
                )
                recorded.add(structure.name)
        missing = wanted - recorded
        if missing:
            warnings.warn(
                f"Viewer asked to record unknown structures: {sorted(missing)}",
                stacklevel=2,
            )
        self._records_ensured = True

    ################################################################
    # Bounds tracking
    ################################################################

    def _update_tracking(self, name, structure_record) -> None:
        """Follow ``track_structure``'s center, if this is that structure.

        Updates the orbit focus (and the bounds used to size the record-mode
        orbit) from the named structure's voxel positions each step. No-op when
        no ``track_structure`` was requested or this isn't the tracked one.

        Args:
            name: The structure controller's name.
            structure_record: That structure's record for this step.
        """
        if self._track_structure is None or name != self._track_structure:
            return
        positions = structure_record.voxel_position()
        if positions is None or len(positions) == 0:
            return
        self._bounds_min = np.min(positions, axis=0)
        self._bounds_max = np.max(positions, axis=0)
        self._center = (self._bounds_min + self._bounds_max) / 2.0
        self._camera.focus = self._center.tolist()

    ################################################################
    # Callbacks
    ################################################################

    def _interactive_step(self, sim_id, controllers) -> None:
        """Per-step callback for :meth:`run`: actuate, then drive the camera."""
        if self._user_callback is not None:
            self._user_callback(sim_id, controllers)
        for ctype, controller in controllers:
            if ctype == _CONTROLLER.RSE_SIMULATION_STRUCTURE_CONTROLLER:
                self._update_tracking(controller.name(), controller.structure_record())
            elif ctype == _CONTROLLER.RSE_SIMULATION_CAMERA_CONTROLLER:
                if controller.name() == MAIN_CAMERA_NAME:
                    self._drive_main_camera(controller)
        if self._stop_requested and self._handle is not None:
            self._handle.terminate()

    def _headless_step(self, sim_id, controllers) -> None:
        """Per-step callback for headless modes: actuate, auto-frame, capture.

        Always refreshes the current frame (so :meth:`snapshot` works); only
        appends to the frame buffer when recording an mp4.
        """
        if self._user_callback is not None:
            self._user_callback(sim_id, controllers)
        for ctype, controller in controllers:
            if ctype == _CONTROLLER.RSE_SIMULATION_STRUCTURE_CONTROLLER:
                self._update_tracking(controller.name(), controller.structure_record())
            elif ctype == _CONTROLLER.RSE_SIMULATION_CAMERA_CONTROLLER:
                if controller.name() == MAIN_CAMERA_NAME:
                    self._drive_record_camera(
                        controller,
                        self._frame_index,
                        self._record_orbit,
                        self._record_elevation,
                    )
                    record = controller.camera_record()
                    if record.is_valid():
                        self._frame = np.array(record.rgb(), copy=True)
                        self._frame_index += 1
                        if self._collect_frames and self._writer is not None:
                            self._writer.append_data(self._frame)

    def _drive_main_camera(self, controller) -> None:
        """Apply mouse-driven orbit, push the pose, and build the window surface."""
        record = controller.camera_record()
        if not record.is_valid():
            return
        self._camera.apply_pending()
        self._write_camera_pose(
            controller, self._camera.position, self._camera.quaternion
        )
        self._frame = np.array(record.rgb(), copy=True)
        # Build the pygame surface here, in the callback thread, so the window
        # thread only blits it (creating SDL surfaces cross-thread can crash).
        self._surface = self._pygame.surfarray.make_surface(
            np.transpose(self._frame, (1, 0, 2))
        )

    def _drive_record_camera(self, controller, frame_index, orbit, elevation) -> None:
        """Compute and push an automatic camera pose for headless recording.

        Orbits around the focus center — which follows ``track_structure`` when
        set (see :meth:`_update_tracking`), else the initial ``camera_focus``.
        The orbit radius is sized from the tracked structure's bounds if known,
        otherwise from the initial camera distance.
        """
        center = self._center
        if self._bounds_min is not None:
            radius = max(float(np.max(self._bounds_max - self._bounds_min)), 0.1) * 1.6
        else:
            radius = max(self._camera.distance, 0.3)
        azimuth = frame_index * 0.01 if orbit else 0.7
        eye = center + radius * np.array(
            [
                np.cos(azimuth) * np.cos(elevation),
                np.sin(azimuth) * np.cos(elevation),
                np.sin(elevation),
            ]
        )
        quat = _look_at_quaternion(eye, center)
        self._write_camera_pose(controller, eye.tolist(), quat)

    @staticmethod
    def _write_camera_pose(controller, position, quaternion) -> None:
        """Write an absolute position + orientation to the camera signal arrays."""
        signal = controller.signal()
        position_signals = signal.position_signals()
        orientation_signals = signal.orientation_signals()
        if position_signals is not None and len(position_signals) >= 3:
            position_signals[0] = position[0]
            position_signals[1] = position[1]
            position_signals[2] = position[2]
        if orientation_signals is not None and len(orientation_signals) >= 4:
            orientation_signals[0] = quaternion[0]
            orientation_signals[1] = quaternion[1]
            orientation_signals[2] = quaternion[2]
            orientation_signals[3] = quaternion[3]

    ################################################################
    # Window loop
    ################################################################

    def _window_loop(self, pygame) -> None:
        """Run the pygame event/render loop until the simulation ends."""
        pygame.init()
        self._screen = pygame.display.set_mode((self._width, self._height))
        pygame.display.set_caption("RiseOrigin Viewer")
        clock = pygame.time.Clock()

        last_mouse = (0, 0)
        buttons = {"left": False, "right": False, "middle": False}

        while not self._stop_requested:
            for event in pygame.event.get():
                last_mouse = self._handle_event(pygame, event, buttons, last_mouse)
            self._render_frame(pygame)
            clock.tick(self._fps)

        pygame.quit()
        if self._handle is not None:
            self._handle.terminate()

    def _handle_event(self, pygame, event, buttons, last_mouse):
        """Process one pygame event; returns the updated last-mouse position."""
        if event.type == pygame.QUIT:
            self._stop_requested = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._stop_requested = True
            elif event.key == pygame.K_r and self._bounds_min is not None:
                self._camera.fit_to_bounds(self._bounds_min, self._bounds_max)
            elif event.key in self._key_bindings:
                self._key_bindings[event.key](self)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                buttons["left"] = True
            elif event.button == 2:
                buttons["middle"] = True
            elif event.button == 3:
                buttons["right"] = True
            elif event.button == 4:
                self._camera.pending_zoom += 1.0
            elif event.button == 5:
                self._camera.pending_zoom -= 1.0
            last_mouse = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                buttons["left"] = False
            elif event.button == 2:
                buttons["middle"] = False
            elif event.button == 3:
                buttons["right"] = False
        elif event.type == pygame.MOUSEMOTION:
            pos = pygame.mouse.get_pos()
            dx = pos[0] - last_mouse[0]
            dy = pos[1] - last_mouse[1]
            if buttons["left"]:
                self._camera.pending_orbit_pitch += dy * self._MOUSE_SENSITIVITY
                self._camera.pending_orbit_yaw += -dx * self._MOUSE_SENSITIVITY
            elif buttons["right"]:
                self._camera.pending_pan_x += dx
                self._camera.pending_pan_y += dy
            elif buttons["middle"]:
                self._camera.pending_zoom += dy * 0.05
            last_mouse = pos

        if self._on_event is not None:
            self._on_event(event, self)
        return last_mouse

    def _render_frame(self, pygame) -> None:
        """Blit the latest camera surface and HUD to the window.

        The main surface is created in the callback thread (see
        :meth:`_drive_main_camera`); this only blits it, then calls the hud.
        """
        self._screen.fill((0, 0, 0))
        if self._surface is not None:
            self._screen.blit(self._surface, (0, 0))
        if self._hud is not None and self._surface is not None:
            self._hud(self._screen, self)
        pygame.display.flip()
