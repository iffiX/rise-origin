# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""VR scene with a desktop-driven ball: one person steers, another watches in VR.

What you'll learn:

* Build a tiny world (grid floor + a single rigid ball).
* Present it to an OpenXR headset through an
  :class:`~rise_origin.RS_WorldSession` and a
  :class:`~rise_origin.RS_WorldSessionComponentVR`, while the same world also
  streams to a desktop :class:`~rise_origin.runtime.viewer.Viewer` window. The
  headset wearer looks around; the person at the desktop keyboard drives the ball.
* Steer the ball with a **force signal** (:class:`rsc.signal.Force`) written each
  control step from the ``control`` callback, with the push direction read from
  the arrow keys through the viewer's ``is_key_down`` in the ``hud`` callback.

The control frequency is raised to the headset display rate: each control step
renders one stereo frame and consumes one head pose, and the runtime reprojects
from the submitted depth between steps.

Requires a core built with ``RISE_ORIGIN_BUILD_CORE_WITH_VR=ON`` and a running
OpenXR runtime with a headset attached.

Run it and steer with the arrow keys::

    python examples/basic/vr2.py
"""

import rise_origin as rs
from rise_origin import RS_WorldForceSignal, rsc

VOXEL = 0.02
BALL_R = 3  # ball radius in voxels -> 0.06 m
DISPLAY_HZ = 30.0  # control rate = headset display rate
PUSH = 8.0  # steering force (N); must beat the ball's weight times friction to move it

FLOOR_FILL = (210 / 255.0, 208 / 255.0, 202 / 255.0)
FLOOR_LINE = (40 / 255.0, 40 / 255.0, 40 / 255.0)
BALL_COLOR = (33 / 255.0, 158 / 255.0, 188 / 255.0)
SKY_COLOR = (170 / 255.0, 200 / 255.0, 235 / 255.0)

CONTROLLER = "vr2#brain"
HINT = "Steer the ball with the UP / DOWN / LEFT / RIGHT arrow keys"

# Current push direction (XY), set by the hud from held keys and read by the control callback.
_push = [0.0, 0.0]


def build() -> "rsc.World":
    """Build the VR scene: a grid floor, one rigid ball, a steering controller, and a sky."""
    resources = [
        rsc.resource.TextureGrid(
            "ground#albedo",
            fill_color=FLOOR_FILL,
            line_color=FLOOR_LINE,
            x_spacing=1,
            y_spacing=1,
            line_width=0.05,
        ),
        rsc.resource.TextureConstant("ground#rm", value=(0.6, 0.0, 0.0)),
        rsc.resource.TextureConstant("ball#albedo", value=BALL_COLOR),
        rsc.resource.TextureConstant("sky#color", value=SKY_COLOR),
    ]

    essences = [
        rsc.essence.Material("ground"),
        rsc.essence.MaterialPhysicalPropertyFixed(
            "ground#phys", material_reference="ground"
        ),
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "ground#optic",
            material_reference="ground",
            albedo_texture_reference="ground#albedo",
            roughness_metalness_texture_reference="ground#rm",
        ),
        rsc.essence.Material("ball"),
        rsc.essence.MaterialPhysicalPropertyRigid(
            "ball#phys",
            material_reference="ball",
            density=1000.0,
        ),
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "ball#optic",
            material_reference="ball",
            albedo_texture_reference="ball#albedo",
        ),
    ]

    for first, second in (("ground", "ball"), ("ball", "ball")):
        pair = f"{first}_{second}"
        essences.append(
            rsc.essence.MaterialPairPropertyCoulombFriction(
                f"{pair}#friction",
                first_material_reference=first,
                second_material_reference=second,
                u_static=1.0,
                u_dynamic=0.85,
            )
        )
        essences.append(
            rsc.essence.MaterialPairPropertyRestitution(
                f"{pair}#restitution",
                first_material_reference=first,
                second_material_reference=second,
                restitution=0.1,
            )
        )

    entities = [
        rsc.entity.CosmicBackgroundUniform("sky", texture_reference="sky#color"),
        rsc.entity.FloorFlat("floor", material_reference="ground"),
        rsc.entity.StructureVoxel(
            "ball",
            origin_position=(0.0, 0.0, 0.02),
            material_references=["ball"],
            bodies=[
                rsc.entity.StructureVoxelBody.sphere(
                    BALL_R,
                    voxel_size=VOXEL,
                    relative_origin_position=(-BALL_R * VOXEL, -BALL_R * VOXEL, 0.0),
                )
            ],
        ),
    ]

    controllers = [rsc.controller.ControllerCallback(CONTROLLER)]
    signals = [
        rsc.signal.Force(
            "push",
            structure_name="ball",
            source_controller_reference=CONTROLLER,
            body_sid=0,
            segment_bid=0,
        )
    ]

    simulation = rsc.simulation.Simulation(
        environment=rsc.simulation.Environment(gravity_acceleration=-9.81),
        runtime=rsc.simulation.Runtime(log_level="debug"),
        control=rsc.simulation.Control(frequency=DISPLAY_HZ),
        vision=rsc.simulation.Vision(
            # restir=rsc.simulation.VisionReSTIR(is_enabled=True),
            denoise=rsc.simulation.VisionDenoiseSVGF(atrous_iterations=2),
        ),
    )
    return rsc.World(
        "vr2",
        simulation=simulation,
        resources=resources,
        essences=essences,
        entities=entities,
        controllers=controllers,
        signals=signals,
    )


def control(time, controllers):
    """Apply the current arrow-key push to the ball each control step.

    Idle push is zero, so the ball stays put until an arrow key is held.
    """
    controller = _controller_by_name(controllers, CONTROLLER)
    if controller is None:
        return
    for signal in controller.signals():
        if isinstance(signal, RS_WorldForceSignal):
            force = (
                signal.force_signals().force()
            )  # shape (1, 3): one rigid-body force vector
            if force.shape[0] >= 1:
                force[0] = (float(_push[0]), float(_push[1]), 0.0)


def hud(painter, viewer):
    """Read the arrow keys into the push vector and draw the steering hint."""
    from PySide6 import QtGui
    from PySide6.QtCore import Qt

    _push[0] = (PUSH if viewer.is_key_down(Qt.Key.Key_Right) else 0.0) - (
        PUSH if viewer.is_key_down(Qt.Key.Key_Left) else 0.0
    )
    _push[1] = (PUSH if viewer.is_key_down(Qt.Key.Key_Up) else 0.0) - (
        PUSH if viewer.is_key_down(Qt.Key.Key_Down) else 0.0
    )

    font = QtGui.QFont()
    font.setPointSize(11)
    painter.setFont(font)
    painter.setPen(QtGui.QColor(40, 40, 40))
    painter.drawText(16, 24, HINT)


def _controller_by_name(controllers, name):
    """Return the controller with the given name from a callback's list, or None."""
    for controller in controllers:
        if controller.name() == name:
            return controller
    return None


def main() -> None:
    """Present the world to the headset and to a simultaneous 720p desktop window.

    The headset gets the stereo camera; a free perspective camera streams to a Qt
    window so the desktop driver can watch and orbit with the mouse. The arrow
    keys steer the ball from that window while the headset wearer looks on. Close
    the window, press ``Esc``, or hit ``Ctrl+C`` in the terminal to stop.
    """
    import signal

    from PySide6 import QtCore, QtWidgets

    from rise_origin.runtime.viewer import Viewer

    class VRViewer(Viewer):
        """The desktop viewer, with the world also presented to an OpenXR headset."""

        def _start_session(self, duration=None) -> None:
            super()._start_session(duration)
            self._session.add_component(
                "vr",
                rs.RS_WorldSessionComponentVR("vr-camera"),
                scope=rs.RSE_WorldSessionScope.RSE_WORLD_SESSION_SCOPE_FULL,
            )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    viewer = VRViewer(
        "preview",
        instance=("cuda:0", build()),
        callback=control,
        hud=hud,
        width=1280,
        height=720,
        camera_position=(-0.6, 0.0, 0.4),
        camera_focus=(0.0, 0.0, 0.06),
    )

    # Qt's event loop swallows Ctrl+C, so route SIGINT to a clean quit and wake the
    # loop periodically to give the Python interrupt handler a chance to run.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    interrupt_timer = QtCore.QTimer()
    interrupt_timer.timeout.connect(lambda: None)
    interrupt_timer.start(200)

    viewer.run_interactive()
    viewer.wait()


if __name__ == "__main__":
    main()
