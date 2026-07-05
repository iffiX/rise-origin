# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Host an MCP server inside a running world so an AI client can drive it.

What you'll learn:

* Build a tiny world (grid floor + one cube material).
* Host it through an :class:`~rise_origin.RS_WorldSessionComponentMCP` attached with
  FULL scope, while a desktop :class:`~rise_origin.runtime.viewer.Viewer` window
  streams the same world so you can watch the AI edit it over MCP.
* Declare a Python authoring tool with ``@mcp.tool`` and serve it next to the
  built-in tools with :meth:`~rise_origin.runtime.mcp.MCPToolRegistry.register_all`.

Run it::

    python examples/basic/mcp.py

Then point a client at it, e.g.::

    claude mcp add --transport http rise-origin http://127.0.0.1:8765/mcp
"""

import rise_origin as rs
from rise_origin import rsc
from rise_origin.runtime.mcp import mcp

VOXEL = 0.02
CUBE = 4  # cube edge in voxels -> 0.08 m

FLOOR_FILL = (210 / 255.0, 208 / 255.0, 202 / 255.0)
FLOOR_LINE = (40 / 255.0, 40 / 255.0, 40 / 255.0)
CUBE_COLOR = (33 / 255.0, 158 / 255.0, 188 / 255.0)
SKY_COLOR = (170 / 255.0, 200 / 255.0, 235 / 255.0)


@mcp.tool(
    "spawn_box",
    "Drop a cube at (x, y).",
    schema={
        "type": "object",
        "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
        "required": ["x", "y"],
    },
)
def spawn_box(handle, x, y):
    """Author a cube through the world handle, the ergonomic bridge over rsc."""
    size = CUBE * VOXEL
    name = f"box_{x}_{y}"
    handle.add_entity(
        rsc.entity.StructureVoxel(
            name,
            origin_position=(x, y, 0.3),
            material_references=["cube"],
            bodies=[
                rsc.entity.StructureVoxelBody.box(
                    CUBE,
                    CUBE,
                    CUBE,
                    voxel_size=VOXEL,
                    relative_origin_position=(-size / 2, -size / 2, 0.0),
                )
            ],
        )
    )
    return {"ok": True, "name": name}


def build() -> "rsc.World":
    """Build the scene: a grid floor, one cube material, and a uniform sky."""
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
        rsc.resource.TextureConstant("cube#albedo", value=CUBE_COLOR),
        rsc.resource.TextureConstant("sky#color", value=SKY_COLOR),
    ]

    essences = [
        rsc.essence.Material("ground"),
        rsc.essence.MaterialPhysicalPropertyFixed("ground#phys", material_reference="ground"),
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "ground#optic",
            material_reference="ground",
            albedo_texture_reference="ground#albedo",
            roughness_metalness_texture_reference="ground#rm",
        ),
        rsc.essence.Material("cube"),
        rsc.essence.MaterialPhysicalPropertyRigid(
            "cube#phys", material_reference="cube", density=1000.0
        ),
        rsc.essence.MaterialOpticalPropertySimpleMicrofacet(
            "cube#optic",
            material_reference="cube",
            albedo_texture_reference="cube#albedo",
        ),
    ]

    for first, second in (("ground", "cube"), ("cube", "cube")):
        pair = f"{first}_{second}"
        essences.append(
            rsc.essence.MaterialPairPropertyCoulombFriction(
                f"{pair}#friction",
                first_material_reference=first,
                second_material_reference=second,
                u_static=1.0,
                u_dynamic=0.8,
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
    ]

    simulation = rsc.simulation.Simulation(
        time=rsc.simulation.Time(is_aligned_to_wallclock=True),
        environment=rsc.simulation.Environment(gravity_acceleration=-9.81),
        runtime=rsc.simulation.Runtime(log_level="info"),
    )
    return rsc.World(
        "mcp",
        simulation=simulation,
        resources=resources,
        essences=essences,
        entities=entities,
    )


def main() -> None:
    """Show the world in a desktop window while the MCP server hosts it.

    A free perspective camera streams to a Qt window so you can watch the world as an
    AI client edits it over MCP. Close the window, press ``Esc``, or hit ``Ctrl+C`` in
    the terminal to stop.
    """
    import signal

    from PySide6 import QtCore, QtWidgets

    from rise_origin.runtime.viewer import Viewer

    class MCPViewer(Viewer):
        """The desktop viewer, with the world also hosting an MCP server."""

        def _start_session(self, duration=None) -> None:
            super()._start_session(duration)
            component = rs.RS_WorldSessionComponentMCP("ai", port=8765)
            self._session.add_component(
                "ai", component, scope=rs.RSE_WorldSessionScope.RSE_WORLD_SESSION_SCOPE_FULL
            )
            mcp.register_all(component)
            print("MCP server on http://127.0.0.1:8765/mcp")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    viewer = MCPViewer(
        "preview",
        instance=("cuda:0", build()),
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
