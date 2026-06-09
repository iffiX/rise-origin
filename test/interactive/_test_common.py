# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Shared helpers for the interactive test scripts.

Every interactive test builds a scene with ``rise_origin.create`` and shows it
with ``rise_origin.viewer.Viewer``. Each test exposes:

* a keyboard-driven joint-actuation HUD (keys ``1..N`` drive joint N to its
  high signal, ``Shift+1..N`` to its low signal, release returns to neutral);
* a yellow "Controls" status panel listing every joint with its current ``+`` /
  ``-`` / ``0`` indicator;
* an on-board (rigid-body-linked) camera preview in the top-right with a white
  border and a ``Robot Camera`` label, when the scene declares one;
* a bottom-of-frame keybinding hint.

The HUD pulls the orbit-camera state directly off the :class:`Viewer` instance
the user passes into :func:`hud_factory`, so it stays in sync with the
viewer's mouse / WASD controls.
"""

from typing import Iterable, List, Optional

import numpy as np
import pygame

import rise_origin as rs
from rise_origin.create import FloorCreator
from rise_origin.viewer import Viewer

_CONTROLLER = rs.RSE_SimulationControllerType
STRUCTURE = _CONTROLLER.RSE_SIMULATION_STRUCTURE_CONTROLLER
FORCE = _CONTROLLER.RSE_SIMULATION_FORCE_CONTROLLER
TORQUE = _CONTROLLER.RSE_SIMULATION_TORQUE_CONTROLLER
CAMERA = _CONTROLLER.RSE_SIMULATION_CAMERA_CONTROLLER

SKY = (135, 206, 235)
FLOOR = (120, 120, 120)

# Original test_*.py joint signal levels: 0 = full reverse, 0.5 = neutral, 1 = full forward.
SIGNAL_MIN = 0.0
SIGNAL_NEUTRAL = 0.5
SIGNAL_MAX = 1.0

ROBOT_CAMERA_SIZE = 256

_JOINT_KEYS = {
    pygame.K_1: 0,
    pygame.K_2: 1,
    pygame.K_3: 2,
    pygame.K_4: 3,
    pygame.K_5: 4,
    pygame.K_6: 5,
    pygame.K_7: 6,
    pygame.K_8: 7,
    pygame.K_9: 8,
    pygame.K_0: 9,
}


def make_config(
    *,
    gravity: float = -9.81,
    control_hz: float = 20.0,
    record_hz: float = 20.0,
    duration: float = 1.0e7,
    sky: tuple = SKY,
    floor_color: tuple = FLOOR,
) -> rs.RS_Config:
    """Build a world/simulation config with the interactive harness defaults.

    Sets the runtime tuning the interactive harness needs for stable soft-body
    + rigid-body integration (200 Hz internal step, 1-substep soft updates,
    10-substep rigid / contact / signal updates, warm-start damping = 1). Tests
    that need different values can override after the call.
    """
    config = rs.RS_Config()
    config.world_config.grav_acc = gravity
    config.world_config.render_config.type = (
        rs.RSE_WorldRenderType.RSE_WORLD_RENDER_CONSTANT_COLOR
    )
    sky_render = rs.RS_ConstantColorWorldRenderConfig()
    sky_render.color = rs.RColor3b(*sky)
    config.world_config.render_config.config = sky_render

    config.floor_config = FloorCreator.flat(render=floor_color)
    sim = config.simulation_config
    sim.control_frequency = control_hz
    sim.record_frequency = record_hz
    sim.stop_condition.from_string(f"t > {duration}")
    sim.rigidbody_constraint_warm_start_damping_z = 1
    sim.runtime_dt_update_interval = 10
    sim.runtime_rigidbody_update_interval = 10
    sim.runtime_contact_update_interval = 10
    sim.runtime_mechanical_signal_update_interval = 10
    sim.runtime_divergence_check_interval = 100
    return config


def controllers_of(controllers, kind):
    """Yield controllers of the given kind."""
    for ctype, controller in controllers:
        if ctype == kind:
            yield controller


class JointControlState:
    """Tracks signal values for keyboard-actuated joints.

    Each joint has a per-key state of ``+`` (max), ``-`` (min), or ``0`` (neutral),
    plus a numeric signal value forwarded to the simulator each control step. Use
    :meth:`as_hinge` / :meth:`as_basic_prismatic` per joint so hinge target
    angles default to 0 rad (true neutral) while basic-position prismatics get
    the legacy ``0.5`` normalized midpoint.
    """

    def __init__(self, joint_names: List[str]):
        """Capture the per-joint display names; signal defaults start at 0."""
        self._names = list(joint_names)
        self._statuses = ["0"] * len(self._names)
        self._signals = [0.0] * len(self._names)
        # Per-joint signal value used when nothing is held. Default 0.0 (true
        # neutral for hinges); call as_basic_prismatic(i) to shift to 0.5.
        self._neutrals = [0.0] * len(self._names)
        self._highs = [1.0] * len(self._names)
        self._lows = [-1.0] * len(self._names)
        for i in range(len(self._names)):
            self._signals[i] = self._neutrals[i]

    # ----- per-joint configuration -----

    def as_hinge(self, joint_idx: int, hi: float = 1.0, lo: float = -1.0) -> None:
        """Configure a hinge joint (control_mode="basic"): signal is target rad."""
        self._neutrals[joint_idx] = 0.0
        self._highs[joint_idx] = hi
        self._lows[joint_idx] = lo
        if self._statuses[joint_idx] == "0":
            self._signals[joint_idx] = 0.0

    def as_basic_prismatic(self, joint_idx: int) -> None:
        """Configure a prismatic joint in basic-position mode (signal in [0,1])."""
        self._neutrals[joint_idx] = SIGNAL_NEUTRAL
        self._highs[joint_idx] = SIGNAL_MAX
        self._lows[joint_idx] = SIGNAL_MIN
        if self._statuses[joint_idx] == "0":
            self._signals[joint_idx] = SIGNAL_NEUTRAL

    def as_custom(self, joint_idx: int, neutral: float, hi: float, lo: float) -> None:
        """Configure a joint with arbitrary neutral / high / low signal values.

        Use for non-position-control signals (e.g. expansion rate, where the
        idle value is non-zero so the muscle relaxes back to rest).
        """
        self._neutrals[joint_idx] = neutral
        self._highs[joint_idx] = hi
        self._lows[joint_idx] = lo
        if self._statuses[joint_idx] == "0":
            self._signals[joint_idx] = neutral

    # ----- read-only accessors -----

    def names(self) -> List[str]:
        return list(self._names)

    def signal(self, joint_idx: int) -> float:
        return self._signals[joint_idx]

    def status(self, joint_idx: int) -> str:
        return self._statuses[joint_idx]

    def apply_to(self, signals) -> None:
        """Write current per-joint signal values into a runtime signal array."""
        if signals is None:
            return
        for jid, value in enumerate(self._signals):
            if jid < len(signals):
                signals[jid] = float(value)

    # ----- pygame events -----

    def handle_event(self, event) -> bool:
        """Update joint state from a pygame key event. Returns True if handled."""
        if event.type == pygame.KEYDOWN and event.key in _JOINT_KEYS:
            jid = _JOINT_KEYS[event.key]
            if jid >= len(self._names):
                return False
            if event.mod & pygame.KMOD_SHIFT:
                self._signals[jid] = self._lows[jid]
                self._statuses[jid] = "-"
            else:
                self._signals[jid] = self._highs[jid]
                self._statuses[jid] = "+"
            return True
        if event.type == pygame.KEYUP and event.key in _JOINT_KEYS:
            jid = _JOINT_KEYS[event.key]
            if jid >= len(self._names):
                return False
            self._signals[jid] = self._neutrals[jid]
            self._statuses[jid] = "0"
            return True
        return False


class HudState:
    """Per-frame state shared between the control callback and the HUD drawer.

    Stores the latest on-board camera frame (so the HUD can blit it without
    accessing engine threads) and the most recent structure-bounds snapshot for
    the optional ``Robot Bounds`` panel.
    """

    def __init__(self):
        """Initialize with no frame and invalid bounds."""
        self.robot_frame: Optional[np.ndarray] = None
        self.bounds_valid = False
        self.bounds_min = [0.0, 0.0, 0.0]
        self.bounds_max = [0.0, 0.0, 0.0]

    def update_bounds(self, structure_record) -> None:
        """Recompute bounds from a structure record's voxel positions."""
        positions = structure_record.voxel_position()
        if positions is None:
            return
        arr = np.asarray(positions)
        if arr.size == 0:
            return
        self.bounds_min = arr.min(axis=0).tolist()
        self.bounds_max = arr.max(axis=0).tolist()
        self.bounds_valid = True

    def update_robot_frame(self, camera_record) -> None:
        """Cache the latest valid rigid-body-linked camera frame."""
        if camera_record.is_valid():
            self.robot_frame = np.array(camera_record.rgb(), copy=True)


def hud_factory(
    state: HudState,
    controls: JointControlState,
    *,
    robot_camera: bool = False,
    title: Optional[str] = None,
):
    """Build a ``hud(surface, viewer)`` callable for the interactive harness.

    Args:
        state: Per-frame state container populated by the control callback.
        controls: Joint-state tracker that the HUD reads status flags from.
        robot_camera: If ``True``, blit the cached on-board camera frame in the
            top-right corner with a white border and a ``Robot Camera`` label.
        title: Optional override for the controls header.

    Returns:
        A callable suitable for ``Viewer(..., hud=...)``.
    """
    fonts = {}

    def _font(size: int = 20):
        if size not in fonts:
            if not pygame.font.get_init():
                pygame.font.init()
            fonts[size] = pygame.font.SysFont(None, size)
        return fonts[size]

    def hud(screen, viewer) -> None:
        font = _font(20)

        if robot_camera and state.robot_frame is not None:
            arr = state.robot_frame
            cam_surface = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
            cam_surface = pygame.transform.scale(
                cam_surface, (ROBOT_CAMERA_SIZE, ROBOT_CAMERA_SIZE)
            )
            cam_x = screen.get_width() - ROBOT_CAMERA_SIZE - 10
            screen.blit(cam_surface, (cam_x, 10))
            pygame.draw.rect(
                screen,
                (255, 255, 255),
                (cam_x - 2, 8, ROBOT_CAMERA_SIZE + 4, ROBOT_CAMERA_SIZE + 4),
                2,
            )
            screen.blit(
                font.render("Robot Camera", True, (255, 255, 255)),
                (cam_x, ROBOT_CAMERA_SIZE + 20),
            )

        names = controls.names()
        header_text = title or f"Controls (keys 1-{len(names)}):"
        screen.blit(font.render(header_text, True, (255, 255, 0)), (10, 10))
        for i, name in enumerate(names):
            line = font.render(
                f"  {i + 1}. {name}: {controls.status(i)}", True, (255, 255, 255)
            )
            screen.blit(line, (10, 35 + i * 25))

        y_offset = 35 + len(names) * 25 + 20

        cam = viewer._camera  # OrbitCamera; private but stable across the codebase.
        screen.blit(
            font.render(
                f"Focus Point: X={cam.focus[0]:.2f}, Y={cam.focus[1]:.2f}, Z={cam.focus[2]:.2f}",
                True,
                (255, 255, 255),
            ),
            (10, y_offset),
        )
        screen.blit(
            font.render(
                f"Camera Position: X={cam.position[0]:.2f}, Y={cam.position[1]:.2f}, Z={cam.position[2]:.2f}",
                True,
                (255, 255, 255),
            ),
            (10, y_offset + 25),
        )
        screen.blit(
            font.render(f"Distance: {cam.distance:.2f}", True, (255, 255, 255)),
            (10, y_offset + 50),
        )

        if state.bounds_valid:
            screen.blit(
                font.render(
                    "Robot Bounds: Min="
                    f"{state.bounds_min[0]:.2f},{state.bounds_min[1]:.2f},{state.bounds_min[2]:.2f}",
                    True,
                    (255, 255, 255),
                ),
                (10, y_offset + 75),
            )
            screen.blit(
                font.render(
                    "Robot Bounds: Max="
                    f"{state.bounds_max[0]:.2f},{state.bounds_max[1]:.2f},{state.bounds_max[2]:.2f}",
                    True,
                    (255, 255, 255),
                ),
                (10, y_offset + 100),
            )
        else:
            screen.blit(
                font.render("Robot Bounds: Not available", True, (255, 255, 255)),
                (10, y_offset + 75),
            )

        screen.blit(
            font.render(
                "Camera: Left Mouse=Orbit, Right Mouse=Pan, Scroll/Middle=Zoom, "
                "WASD=Move Focus (Camera Dir)",
                True,
                (200, 200, 200),
            ),
            (10, y_offset + 125),
        )
        screen.blit(
            font.render(
                f"Controls: 1-{len(names)}=Actuate (Shift+#=reverse), "
                "T=Reset Camera, ESC=Exit",
                True,
                (200, 200, 200),
            ),
            (10, y_offset + 150),
        )

    return hud


def launch(
    config: rs.RS_Config,
    callback,
    *,
    hud=None,
    on_event=None,
    device: str = "cuda:0",
    **viewer_kwargs,
) -> None:
    """Open an interactive viewer on the given config."""
    rise = rs.RiseOrigin(device)
    viewer = Viewer(
        rise,
        config,
        callback=callback,
        hud=hud,
        on_event=on_event,
        **viewer_kwargs,
    )
    viewer.run_interactive()
    viewer.wait()
