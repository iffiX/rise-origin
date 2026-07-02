# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Ball maze — roll a ball through a procedural maze to the goal.

What you'll learn:

* Generate a solvable maze with a recursive-backtracker (depth-first) carve and
  emit it as one ``fixed`` structure of a single voxel body that never moves.
* Steer a rigid ball with a **force signal** (:class:`rsc.signal.Force`) written
  each step, wired to the arrow keys through ``viewer.is_key_down``.
* Score the run with an **intersect-region record**
  (:class:`rsc.record.IntersectRegionCuboid`): a world-fixed box over the goal
  reports the gids of the structures inside it, so the moment the ball's gid (read
  from the add-entity message) shows up in the region the referee declares a win.

Run it and steer with the arrow keys::

    python examples/game/maze.py
"""

import os, sys

import numpy as np
from PySide6 import QtGui
from PySide6.QtCore import Qt

from rise_origin import (
    RS_NULL_INDEX,
    RS_WorldForceSignal,
    RS_WorldIntersectRegionRecord,
    RS_WorldTaskDataCreated,
    RSC_EntityVariant,
    RSC_RecordVariant,
    RSC_SignalVariant,
    rsc,
)

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # noqa: E401,E702 — find sibling _common
import _common

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
VOXEL = 0.01
MAZE_W, MAZE_H = 16, 16  # maze size in cells
SEED = 7  # change for a different maze

CELL_V = 12  # cell pitch (voxels); passage width = CELL_V - WALL_T_V
WALL_T_V = 2  # wall thickness (voxels)
WALL_H_V = 12  # wall height (voxels)
HALF_T = WALL_T_V // 2

BALL_R = 3  # ball radius (voxels)
TARGET_V = 4  # goal cube edge (voxels)

PUSH = 2.5  # steering force (N); must beat the ball's weight-times-friction to move it

WALL_COLOR = _common.PALETTE["deep_blue"]
BALL_COLOR = (60, 140, 255)  # glowing blue
TARGET_COLOR = (255, 45, 35)  # glowing red
GLOW = 4.0  # emission intensity of the ball and goal

# Material reference order; per-voxel ids index into it.
WALL_ID, TARGET_ID = 0, 1
BALL_ID = 0

CONTROLLER = "maze#brain"
GOAL_RECORD = "maze#goal"

HINT = "Roll the ball to the red cube  -  steer with the UP / DOWN / LEFT / RIGHT arrow keys"

#: Goal cell centre in voxels; the goal is always the far corner.
GOAL_V = ((MAZE_W - 0.5) * CELL_V, (MAZE_H - 0.5) * CELL_V)
#: Start cell centre in voxels.
START_V = 0.5 * CELL_V

# ---------------------------------------------------------------------------
# Shared run state
# ---------------------------------------------------------------------------
# Current push direction (XY), updated by the hud; shared with the callback.
_push = np.zeros(2, dtype=float)
# Run state shared control -> hud. ``won`` is raised by the referee callback and
# cleared by the hud once it has swapped in the next maze.
_state = {
    "won": False,
    "in_goal": False,
    "populated": False,
    "t": 0.0,
    "level_start": 0.0,
    "seed": SEED,
    "level": 1,
    "flash": 0,
    "ball_gid": RS_NULL_INDEX,
}


# ---------------------------------------------------------------------------
# Scene content
# ---------------------------------------------------------------------------
def generate_maze(w, h, seed):
    """Carve a perfect maze with the recursive-backtracker algorithm.

    Args:
        w: Maze width in cells.
        h: Maze height in cells.
        seed: Seed for the random number generator (reproducible layout).

    Returns:
        A ``(vwall, hwall)`` pair of boolean arrays. ``vwall[i, j]`` is the wall
        on the vertical grid line ``x = i`` beside cell row ``j``; ``hwall[i, j]``
        is the wall on the horizontal grid line ``y = j`` beside cell column
        ``i``. ``True`` means the wall is present.
    """
    rng = np.random.default_rng(seed)
    visited = np.zeros((w, h), dtype=bool)
    vwall = np.ones((w + 1, h), dtype=bool)
    hwall = np.ones((w, h + 1), dtype=bool)

    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        i, j = stack[-1]
        neighbors = []
        if i > 0 and not visited[i - 1, j]:
            neighbors.append((i - 1, j, "W"))
        if i < w - 1 and not visited[i + 1, j]:
            neighbors.append((i + 1, j, "E"))
        if j > 0 and not visited[i, j - 1]:
            neighbors.append((i, j - 1, "S"))
        if j < h - 1 and not visited[i, j + 1]:
            neighbors.append((i, j + 1, "N"))
        if not neighbors:
            stack.pop()
            continue
        ni, nj, direction = neighbors[rng.integers(len(neighbors))]
        if direction == "W":
            vwall[i, j] = False
        elif direction == "E":
            vwall[i + 1, j] = False
        elif direction == "S":
            hwall[i, j] = False
        else:
            hwall[i, j + 1] = False
        visited[ni, nj] = True
        stack.append((ni, nj))
    return vwall, hwall


def _paint(grid, corner, size, material):
    """Stamp a solid block of one material into the maze voxel grid.

    ``corner`` is the block's min corner in structure-local voxels (the same
    coordinates the geometry is laid out in); the grid is offset so its index 0
    sits at voxel ``-HALF_T``.
    """
    x, y, z = corner
    sx, sy, sz = size
    grid[x + HALF_T : x + HALF_T + sx, y + HALF_T : y + HALF_T + sy, z : z + sz] = (
        material
    )


def _make_maze(seed):
    """A fixed maze structure (walls + glowing goal cube).

    Posts sit at every grid intersection; wall segments span the gap between
    posts along each present grid edge, so the pieces tile without overlap. The
    whole maze is painted into one material grid and emitted as a single voxel
    body, so a reset builds it in one shot instead of hundreds of small blocks.
    """
    vwall, hwall = generate_maze(MAZE_W, MAZE_H, seed)
    span = CELL_V - WALL_T_V

    grid = np.full(
        (MAZE_W * CELL_V + WALL_T_V, MAZE_H * CELL_V + WALL_T_V, WALL_H_V),
        RS_NULL_INDEX,
        dtype=np.int64,
    )

    for i in range(MAZE_W + 1):
        for j in range(MAZE_H + 1):
            _paint(
                grid,
                (i * CELL_V - HALF_T, j * CELL_V - HALF_T, 0),
                (WALL_T_V, WALL_T_V, WALL_H_V),
                WALL_ID,
            )
    for i in range(MAZE_W + 1):
        for j in range(MAZE_H):
            if vwall[i, j]:
                _paint(
                    grid,
                    (i * CELL_V - HALF_T, j * CELL_V + HALF_T, 0),
                    (WALL_T_V, span, WALL_H_V),
                    WALL_ID,
                )
    for i in range(MAZE_W):
        for j in range(MAZE_H + 1):
            if hwall[i, j]:
                _paint(
                    grid,
                    (i * CELL_V + HALF_T, j * CELL_V - HALF_T, 0),
                    (span, WALL_T_V, WALL_H_V),
                    WALL_ID,
                )

    # Glowing goal cube parked on the floor in the far corner cell.
    _paint(
        grid,
        (int(GOAL_V[0] - TARGET_V / 2), int(GOAL_V[1] - TARGET_V / 2), 0),
        (TARGET_V, TARGET_V, TARGET_V),
        TARGET_ID,
    )

    body = rsc.entity.StructureVoxelBody.from_grid(
        grid != RS_NULL_INDEX,
        material_grid=grid,
        voxel_size=VOXEL,
        relative_origin_position=(-HALF_T * VOXEL, -HALF_T * VOXEL, 0.0),
    )
    return rsc.entity.StructureVoxel(
        "maze",
        material_references=["wall", "target"],
        is_fixed=True,
        bodies=[body],
    )


def _make_ball():
    """The glowing ball parked in the start cell: a bare rigid sphere."""
    return rsc.entity.StructureVoxel(
        "ball",
        material_references=["ball"],
        bodies=[
            rsc.entity.StructureVoxelBody.sphere(
                BALL_R,
                material_sid=BALL_ID,
                voxel_size=VOXEL,
                relative_origin_position=(
                    (START_V - BALL_R) * VOXEL,
                    (START_V - BALL_R) * VOXEL,
                    0.02,
                ),
            )
        ],
    )


def _generate(seed):
    """All per-level scene content for ``seed``: maze, ball, push, goal.

    A single generator drives both the initial populate and every reset, so the
    first level and the next ones run the exact same code path.
    """
    maze = _make_maze(seed)
    ball = _make_ball()
    push = rsc.signal.Force(
        "push",
        structure_name="ball",
        source_controller_reference=CONTROLLER,
        body_sid=0,
        segment_bid=0,
    )
    # A world-fixed box over the goal cube. The referee watches the region's
    # member gids and wins when the ball's gid shows up among them.
    goal = rsc.record.IntersectRegionCuboid(
        GOAL_RECORD,
        position=(GOAL_V[0] * VOXEL, GOAL_V[1] * VOXEL, BALL_R * VOXEL),
        size=(8 * VOXEL, 8 * VOXEL, WALL_H_V * VOXEL),
        target_controller_reference=CONTROLLER,
    )
    return maze, ball, push, goal


# ---------------------------------------------------------------------------
# World build and level lifecycle
# ---------------------------------------------------------------------------
def build():
    """Build the empty world: materials, floor, sky, and the referee controller.

    The maze, ball, goal, and push signal are added after the simulation starts
    by :func:`_populate`, so the same path that resets the level also seeds it.

    Returns:
        The :class:`rsc.World` ready to hand to a viewer.
    """
    materials = [
        _common.MaterialSpec("wall", color=WALL_COLOR, kind="fixed"),
        _common.MaterialSpec(
            "target",
            color=TARGET_COLOR,
            kind="fixed",
            emission_color=TARGET_COLOR,
            emission_intensity=GLOW,
        ),
        _common.MaterialSpec(
            "ball",
            color=BALL_COLOR,
            kind="rigid",
            emission_color=BALL_COLOR,
            emission_intensity=GLOW,
        ),
    ]

    return _common.make_world(
        "maze",
        materials=materials,
        controllers=[rsc.controller.ControllerCallback(CONTROLLER)],
    )


def _populate(handle, seed):
    """Add a freshly generated level to the running world."""
    maze, ball, push, goal = _generate(seed)
    handle.begin_modification_transaction()
    handle.add_entity(RSC_EntityVariant(maze))
    ball_task = handle.add_entity(RSC_EntityVariant(ball))
    handle.add_signal(RSC_SignalVariant(push))
    handle.add_record(RSC_RecordVariant(goal))
    handle.end_modification_transaction()
    # The world reports the gid it assigned the ball once the transaction commits.
    # The goal region lists that gid among its members exactly while the ball is inside.
    created = ball_task.get().value
    _state["ball_gid"] = (
        created.gid if isinstance(created, RS_WorldTaskDataCreated) else RS_NULL_INDEX
    )


def _depopulate(handle):
    """Remove the current level. The push signal goes first so it never
    dangles over a deleted ball."""
    handle.begin_modification_transaction()
    handle.remove_record(GOAL_RECORD)
    handle.remove_signal("push")
    handle.remove_entity("ball")
    handle.remove_entity("maze")
    handle.end_modification_transaction()


def _reset_maze(viewer):
    """Swap in the next maze (seed + 1): tear the level down and regenerate it.

    The goal region rides with the level, and the start position is fixed, so
    deleting and regenerating everything is the whole reset.
    """
    handle = viewer.handle
    if handle is None:
        return
    _depopulate(handle)
    _state["seed"] += 1
    _state["level"] += 1
    _state["level_start"] = _state["t"]
    _state["flash"] = 90
    _populate(handle, _state["seed"])
    _push[:] = 0.0
    _state["won"] = False
    # Treat the goal as still occupied until the new ball is seen to leave it,
    # so the next win needs a fresh entry.
    _state["in_goal"] = True


# ---------------------------------------------------------------------------
# Runtime callbacks
# ---------------------------------------------------------------------------
def control(time, controllers):
    """Apply the arrow-key push to the ball and watch the goal region.

    Idle force is zero, so the ball stays put until an arrow key is held. The
    ball's gid appearing among the goal region's members means it has arrived.
    """
    controller = _common.controller_by_name(controllers, CONTROLLER)
    if controller is None:
        return

    _state["t"] = time
    for signal in controller.signals():
        if isinstance(signal, RS_WorldForceSignal):
            force = (
                signal.force_signals().force()
            )  # shape (1, 3): one rigid-body force vector
            if force.shape[0] >= 1:
                force[0] = (float(_push[0]), float(_push[1]), 0.0)

    # The goal region lists the gid of every structure inside it; the ball is home
    # the moment its gid appears. Fire on the rising edge only, so a ball lingering
    # in the region (or a reset that has not yet moved it out) does not re-trigger
    # every step.
    inside = False
    ball_gid = _state["ball_gid"]
    for record in controller.records():
        if isinstance(record, RS_WorldIntersectRegionRecord):
            if ball_gid in record.memberships.structure_gid():
                inside = True
    if inside and not _state["in_goal"] and not _state["won"]:
        _state["won"] = True
    _state["in_goal"] = inside


def hud(painter, viewer):
    """Read the arrow keys into the push vector and draw the score banner."""
    if viewer.handle is not None and not _state["populated"]:
        _populate(viewer.handle, _state["seed"])
        _state["populated"] = True

    if _state["won"]:
        _reset_maze(viewer)
    else:
        _push[0] = (PUSH if viewer.is_key_down(Qt.Key.Key_Right) else 0.0) - (
            PUSH if viewer.is_key_down(Qt.Key.Key_Left) else 0.0
        )
        _push[1] = (PUSH if viewer.is_key_down(Qt.Key.Key_Up) else 0.0) - (
            PUSH if viewer.is_key_down(Qt.Key.Key_Down) else 0.0
        )

    hint_font = QtGui.QFont()
    hint_font.setPointSize(11)
    painter.setFont(hint_font)
    painter.setPen(QtGui.QColor(40, 40, 40))
    painter.drawText(16, 24, HINT)

    score_font = QtGui.QFont()
    score_font.setPointSize(16)
    score_font.setBold(True)
    painter.setFont(score_font)
    painter.setPen(QtGui.QColor(30, 30, 30))
    level_time = _state["t"] - _state["level_start"]
    painter.drawText(16, 56, f"Level: {_state['level']}    Time: {level_time:.1f} s")

    if _state["flash"] > 0:
        _state["flash"] -= 1
        win_font = QtGui.QFont()
        win_font.setPointSize(40)
        win_font.setBold(True)
        painter.setFont(win_font)
        metrics = QtGui.QFontMetrics(win_font)
        text = f"LEVEL {_state['level']}!"
        x = (painter.device().width() - metrics.horizontalAdvance(text)) // 2
        y = painter.device().height() // 2
        painter.setPen(QtGui.QColor(255, 200, 60))
        painter.drawText(x, y, text)


def main():
    """Build the world and launch it in the selected run mode."""
    cx = MAZE_W * CELL_V * VOXEL / 2
    cy = MAZE_H * CELL_V * VOXEL / 2
    center = (cx, cy, 0.0)
    # Look down into the maze at a steep angle.
    camera_position = (cx, cy - 0.6, 2.6)
    _common.launch(
        build(),
        control,
        hud=hud,
        camera_position=camera_position,
        camera_focus=center,
    )


if __name__ == "__main__":
    main()
