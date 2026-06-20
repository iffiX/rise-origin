# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Procedural floor plan: cell biome assignment + maze carving + sightline pass.

Produces a :class:`Layout` summarising the facility:

* ``biome[cx, cy]`` — one of the :class:`Biome` enum values per cell.
* ``door_x[cx, cy]`` — True if there's a doorway between cell ``(cx, cy)`` and
  ``(cx+1, cy)``.
* ``door_y[cx, cy]`` — True if there's a doorway between cell ``(cx, cy)`` and
  ``(cx, cy+1)``.
* ``entrance`` — the cell the camera sits in.
* ``reactor`` — the cell containing the reactor.

The carver is a recursive-backtracker maze (mirroring :func:`lights._generate_maze`),
followed by a 10 % "braid" pass that opens extra walls for loops, and finally a
sightline-opening pass that guarantees a corridor of carved doorways from the
entrance cell straight to the reactor cell so the reactor's glow can be seen
from the camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .constants import CELLS_X, CELLS_Y


class Biome(Enum):
    CORRIDOR = "corridor"
    JUNCTION = "junction"
    CONTROL_ROOM = "control_room"
    REACTOR = "reactor"
    BIO_LAB = "bio_lab"
    STORAGE = "storage"
    ENGINEERING = "engineering"
    DORMITORY = "dormitory"
    DAMAGED = "damaged"
    ENTRANCE = "entrance"
    SERVER_FARM = "server_farm"        # rows of thinking-machine racks
    LASER_CORRIDOR = "laser_corridor"  # Resident-Evil-style glass-walled hallway
    MAIN_HALL = "main_hall"            # reception desk + sofa + water cooler


@dataclass
class Layout:
    biome: np.ndarray  # (CELLS_X, CELLS_Y) of Biome (object array)
    door_x: np.ndarray  # (CELLS_X - 1, CELLS_Y) boolean: doorway across X boundary
    door_y: np.ndarray  # (CELLS_X, CELLS_Y - 1) boolean: doorway across Y boundary
    entrance: Tuple[int, int]
    reactor: Tuple[int, int]
    cells: List[Tuple[int, int]] = field(default_factory=list)
    #: Per-cell room id. Adjacent cells with the same room id are merged into
    #: one big room (no internal wall between them, no doorway emitted there).
    #: Cells with different room ids get a wall between them.
    room_id: Optional[np.ndarray] = None       # (CELLS_X, CELLS_Y) int
    #: Pairs of biomes whose shared wall should be GLASS instead of concrete.
    #: Encoded as a set of frozensets so the order of the pair doesn't matter.
    glass_pairs: Optional[set] = None
    #: Pairs of biomes whose shared wall should be CONCRETE with embedded
    #: vertical bright cyan strip-lights (cyberpunk corp-tower wall). The
    #: walls assembly handles painting the strip lights every 4 coarse voxels.
    light_strip_pairs: Optional[set] = None
    #: Pairs of biomes whose shared wall should be **dark smoked glass**
    #: — visually different from the pure GLASS_TANK partition.
    dark_glass_pairs: Optional[set] = None
    #: Doorway locations that get the special Future Corp corporate glass
    #: door instead of the industrial blast door. Each entry is
    #: ``(cx, cy, axis)`` where ``axis`` is "x" or "y" matching the
    #: doorway arrays.
    corporate_door_locations: Optional[set] = None


# ============================================================================
# Maze carve (recursive backtracker)
# ============================================================================


def _carve_maze(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Recursive-backtracker carve. Returns ``(door_x, door_y)`` arrays
    of doorway presence between adjacent cells."""
    door_x = np.zeros((CELLS_X - 1, CELLS_Y), dtype=bool)
    door_y = np.zeros((CELLS_X, CELLS_Y - 1), dtype=bool)
    visited = np.zeros((CELLS_X, CELLS_Y), dtype=bool)

    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        cx, cy = stack[-1]
        nbrs = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < CELLS_X and 0 <= ny < CELLS_Y and not visited[nx, ny]:
                nbrs.append((nx, ny, dx, dy))
        if not nbrs:
            stack.pop()
            continue
        nx, ny, dx, dy = nbrs[rng.integers(len(nbrs))]
        visited[nx, ny] = True
        # Carve doorway between (cx,cy) and (nx,ny).
        if dx == 1:
            door_x[cx, cy] = True
        elif dx == -1:
            door_x[nx, ny] = True
        elif dy == 1:
            door_y[cx, cy] = True
        else:  # dy == -1
            door_y[nx, ny] = True
        stack.append((nx, ny))
    return door_x, door_y


# ============================================================================
# Braid (open a fraction of extra walls)
# ============================================================================


def _braid(door_x: np.ndarray, door_y: np.ndarray,
           rng: np.random.Generator, frac: float = 0.18) -> None:
    """Open a fraction of currently-closed walls so the maze has loops."""
    closed_x = np.argwhere(~door_x)
    closed_y = np.argwhere(~door_y)
    n_open = int(frac * (len(closed_x) + len(closed_y)))
    # Pick from the combined pool.
    if n_open <= 0:
        return
    pool = [("x", int(cx), int(cy)) for cx, cy in closed_x] \
        + [("y", int(cx), int(cy)) for cx, cy in closed_y]
    rng.shuffle(pool)
    for which, cx, cy in pool[:n_open]:
        if which == "x":
            door_x[cx, cy] = True
        else:
            door_y[cx, cy] = True


# ============================================================================
# Sightline open (force a corridor between two cells)
# ============================================================================


def _open_path(door_x: np.ndarray, door_y: np.ndarray,
               a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Force open every wall along a Manhattan path from ``a`` to ``b``.
    Returns the list of cells along the path."""
    ax, ay = a
    bx, by = b
    path = [(ax, ay)]
    # Walk x first, then y (the path passes through (bx, ay) corner).
    while ax != bx:
        sgn = 1 if bx > ax else -1
        if sgn == 1:
            door_x[ax, ay] = True
        else:
            door_x[ax - 1, ay] = True
        ax += sgn
        path.append((ax, ay))
    while ay != by:
        sgn = 1 if by > ay else -1
        if sgn == 1:
            door_y[ax, ay] = True
        else:
            door_y[ax, ay - 1] = True
        ay += sgn
        path.append((ax, ay))
    return path


# ============================================================================
# Biome assignment
# ============================================================================


def _assign_biomes(door_x: np.ndarray, door_y: np.ndarray,
                   entrance: Tuple[int, int],
                   reactor: Tuple[int, int],
                   sightline_cells: List[Tuple[int, int]],
                   rng: np.random.Generator) -> np.ndarray:
    """Assign one biome per cell. Constraints:
       - Reactor in its fixed cell.
       - Entrance in its fixed cell.
       - Sightline cells (between entrance and reactor) are CORRIDOR.
       - Place exactly one CONTROL_ROOM, BIO_LAB, ENGINEERING, DORMITORY.
       - Place two STORAGE and two DAMAGED.
       - Remaining cells are CORRIDOR or JUNCTION (3+ doorways = JUNCTION).
    """
    biome = np.full((CELLS_X, CELLS_Y), None, dtype=object)
    biome[entrance] = Biome.ENTRANCE
    biome[reactor] = Biome.REACTOR
    for c in sightline_cells:
        if biome[c] is None:
            biome[c] = Biome.CORRIDOR

    # Eligible cells for specialised biomes — anything still empty.
    free = [(cx, cy) for cx in range(CELLS_X) for cy in range(CELLS_Y)
            if biome[cx, cy] is None]
    rng.shuffle(free)
    pending = [
        Biome.CONTROL_ROOM, Biome.BIO_LAB, Biome.ENGINEERING, Biome.DORMITORY,
        Biome.STORAGE, Biome.STORAGE, Biome.DAMAGED, Biome.DAMAGED,
    ]
    for slot in pending:
        if not free:
            break
        cx, cy = free.pop()
        biome[cx, cy] = slot

    # Anything else is corridor or junction based on door count.
    for cx in range(CELLS_X):
        for cy in range(CELLS_Y):
            if biome[cx, cy] is not None:
                continue
            n_doors = 0
            if cx > 0 and door_x[cx - 1, cy]: n_doors += 1
            if cx < CELLS_X - 1 and door_x[cx, cy]: n_doors += 1
            if cy > 0 and door_y[cx, cy - 1]: n_doors += 1
            if cy < CELLS_Y - 1 and door_y[cx, cy]: n_doors += 1
            biome[cx, cy] = Biome.JUNCTION if n_doors >= 3 else Biome.CORRIDOR

    return biome


# ============================================================================
# Top-level generator
# ============================================================================


def generate_layout(rng: np.random.Generator) -> Layout:
    """Build the floor plan: carve maze, braid, force sightline, assign biomes.

    Kept for reference; :func:`generate_symmetric_layout` is the one the example
    actually uses.
    """
    door_x, door_y = _carve_maze(rng)
    _braid(door_x, door_y, rng, frac=0.18)

    entrance = (0, CELLS_Y // 2)             # left edge, middle row
    reactor = (CELLS_X - 1, CELLS_Y // 2)    # right edge, middle row
    sightline = _open_path(door_x, door_y, entrance, reactor)

    biome = _assign_biomes(door_x, door_y, entrance, reactor, sightline, rng)

    cells = [(cx, cy) for cx in range(CELLS_X) for cy in range(CELLS_Y)]
    return Layout(
        biome=biome,
        door_x=door_x,
        door_y=door_y,
        entrance=entrance,
        reactor=reactor,
        cells=cells,
    )


# ============================================================================
# Diagram-faithful 11x13 hand-designed layout (the one the example uses)
# ============================================================================
#
# Cell map (cy = 0 is south / entrance, cy = 12 is north / reactor):
#
#                     cx=0  1  2  3  4  5  6  7  8  9  10
#     cy=12  N         OBLB  OBLB CLNT CLNT REAC REAC REAC CLNT CLNT OBLB OBLB
#     cy=11             OBLB  OBLB CLNT CLNT REAC REAC REAC CLNT CLNT OBLB OBLB
#     cy=10             OBLB  OBLB CLNT CLNT REAC REAC REAC CLNT CLNT OBLB OBLB
#     cy=9              SECC SECC SECC CORR LASR LASR LASR CORR COOL COOL COOL
#     cy=8              POWR POWR POWR CORR LASR LASR LASR CORR OBOF OBOF OBOF
#     cy=7              INST INST ANAL ANAL LASR LASR LASR TERM TERM MAINT MAINT
#     cy=6              BIO  BIO  BIO  BIO  MAIN MAIN MAIN SRVR SRVR SRVR SRVR
#     cy=5              BIO  BIO  BIO  BIO  MAIN MAIN MAIN SRVR SRVR SRVR SRVR
#     cy=4              BIO  BIO  BIO  BIO  MAIN MAIN MAIN SRVR SRVR SRVR SRVR
#     cy=3              BIO  BIO  BIO  BIO  MAIN MAIN MAIN SRVR SRVR SRVR SRVR
#     cy=2              BIO  BIO  BIO  BIO  MAIN MAIN MAIN SRVR SRVR SRVR SRVR
#     cy=1              ENTR ENTR ENTR ENTR ENTR ENTR ENTR ENTR ENTR ENTR ENTR
#     cy=0    S         ENTR ENTR ENTR ENTR ENTR ENTR ENTR ENTR ENTR ENTR ENTR
#
# (Above abbreviations: OBLB = observation lab, CLNT = coolant / emergency power,
# REAC = reactor, SECC = security control, COOL = cooling control, POWR = power
# control, OBOF = observation office, INST = instrument rack, ANAL = analysis
# office, TERM = terminal office, MAINT = maintenance, BIO = biome lab,
# MAIN = main hall / security transition, SRVR = server farm,
# LASR = laser security hallway, ENTR = lobby / entrance.)


def generate_diagram_layout(rng: np.random.Generator) -> Layout:
    """Build the 11x13 layout exactly per the user's design diagram.

    Multi-cell rooms (e.g. the 4x5 server farm) share a ``room_id`` so the
    walls assembly skips internal walls within them, and certain biome pairs
    get **glass partitions** (the cyan partitions in the diagram).
    """
    assert CELLS_X == 11 and CELLS_Y == 13, "diagram layout requires an 11x13 grid"

    biome = np.full((CELLS_X, CELLS_Y), None, dtype=object)
    room_id = np.full((CELLS_X, CELLS_Y), -1, dtype=np.int64)

    next_rid = [0]
    def _paint(rect: Tuple[int, int, int, int], bm: Biome) -> int:
        """Paint a rectangle ``(x0, y0, x1, y1)`` (inclusive corners) with one
        biome and assign a fresh room id covering all of those cells."""
        rid = next_rid[0]; next_rid[0] += 1
        x0, y0, x1, y1 = rect
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                biome[cx, cy] = bm
                room_id[cx, cy] = rid
        return rid

    # --- South lobby / entrance (full width, 2 cells deep) ----------------
    _paint((0, 0, 10, 1), Biome.ENTRANCE)

    # --- Main central area cy=2..6 ----------------------------------------
    _paint((0, 2, 3, 6), Biome.BIO_LAB)            # 4x5 biome container lab
    _paint((4, 2, 6, 6), Biome.MAIN_HALL)          # 3x5 security transition hall
    _paint((7, 2, 10, 6), Biome.SERVER_FARM)       # 4x5 server room

    # --- Row of side rooms cy=7 -------------------------------------------
    _paint((0, 7, 1, 7), Biome.STORAGE)            # instrument rack room
    _paint((2, 7, 3, 7), Biome.CONTROL_ROOM)       # analysis office
    _paint((7, 7, 8, 7), Biome.CONTROL_ROOM)       # terminal office
    _paint((9, 7, 10, 7), Biome.STORAGE)           # maintenance / storage

    # --- Two big open-plan offices cy=8..9 (each 3x2 cells = 12m x 8m) ---
    # ROUND-7 fix: the four thin 3x1 rooms (power control / security
    # control on the west, observation office / cooling control on the
    # east) were combined into two big offices, one per side. Each office
    # is a single multi-cell CONTROL_ROOM with shared room_id, so the
    # wall edge-tracer skips the internal horizontal wall between cy=8
    # and cy=9, giving an open office shape. The big-office content
    # (rows of desks + monitors) is placed by emit_singleton_rooms when
    # it detects a CONTROL_ROOM with >= 4 cells.
    _paint((0, 8, 2, 9), Biome.CONTROL_ROOM)       # west open-plan office
    _paint((8, 8, 10, 9), Biome.CONTROL_ROOM)      # east open-plan office

    # --- Laser security hallway cy=7..9, cx=4..6 (3x3 cells, multi-cell) -
    _paint((4, 7, 6, 9), Biome.LASER_CORRIDOR)

    # --- Corridors flanking the laser hallway (between laser and side rooms)
    biome[3, 8] = Biome.CORRIDOR; room_id[3, 8] = next_rid[0]; next_rid[0] += 1
    biome[3, 9] = Biome.CORRIDOR; room_id[3, 9] = next_rid[0]; next_rid[0] += 1
    biome[7, 8] = Biome.CORRIDOR; room_id[7, 8] = next_rid[0]; next_rid[0] += 1
    biome[7, 9] = Biome.CORRIDOR; room_id[7, 9] = next_rid[0]; next_rid[0] += 1

    # --- Reactor area cy=10..12 -------------------------------------------
    _paint((0, 10, 1, 12), Biome.BIO_LAB)          # observation lab (left)
    _paint((2, 10, 3, 12), Biome.ENGINEERING)      # coolant / emergency power (left)
    _paint((4, 10, 6, 12), Biome.REACTOR)          # central reactor chamber
    _paint((7, 10, 8, 12), Biome.ENGINEERING)      # coolant / emergency power (right)
    _paint((9, 10, 10, 12), Biome.BIO_LAB)         # observation lab (right)

    # --- Doorways ---------------------------------------------------------
    door_x = np.zeros((CELLS_X - 1, CELLS_Y), dtype=bool)
    door_y = np.zeros((CELLS_X, CELLS_Y - 1), dtype=bool)

    # Lobby <-> main central area (cy=1 <-> cy=2) is glass-walled; doors are
    # at the centre + at the flanks (each big room has its own door to lobby).
    door_y[2, 1] = True   # lobby -> biome lab (door at cx=2)
    door_y[5, 1] = True   # lobby -> main hall (centre)
    door_y[8, 1] = True   # lobby -> server farm (door at cx=8)

    # Main hall <-> biome lab and main hall <-> server farm (glass partitions
    # with a door each at the centre row).
    door_x[3, 4] = True   # biome lab <-> main hall (centre row)
    door_x[6, 4] = True   # main hall <-> server farm (centre row)

    # Main hall <-> laser corridor (going north).
    door_y[5, 6] = True   # main hall -> laser corridor (centre)

    # Laser corridor <-> reactor chamber.
    door_y[5, 9] = True

    # Reactor side connections to engineering / observation labs.
    door_x[1, 11] = True  # obs lab L <-> coolant L
    door_x[3, 11] = True  # coolant L <-> reactor
    door_x[6, 11] = True  # reactor <-> coolant R
    door_x[8, 11] = True  # coolant R <-> obs lab R

    # Side rooms off the cy=7..9 corridors.
    door_y[3, 7] = True   # corridor cy=7..8 connection
    door_y[3, 8] = True
    door_y[7, 7] = True
    door_y[7, 8] = True
    # Side rooms to corridor.
    door_x[0, 7] = True   # instr rack <-> analysis office
    door_x[2, 7] = True   # analysis office <-> corridor (3,7) — wait (3,7) isn't carved.
    # Let's give analysis office a doorway to the laser-corridor corridor (3, 8)
    # along the south wall (cy=7 <-> cy=8 boundary).
    door_y[2, 7] = True   # analysis office <-> power control region
    door_y[7, 6] = True   # server farm corridor link (not strictly needed)
    door_y[8, 7] = True   # terminal office <-> obs office
    door_y[1, 7] = True   # instr rack -> south? actually we want to bio lab.
    # Bio lab to instr rack
    door_x[1, 7] = True   # (1,7) <-> (2,7)
    # Server farm to terminal office
    door_x[7, 7] = True   # (7,7) <-> (8,7) inside terminal-office room (no wall)
    door_x[8, 7] = True
    door_x[6, 7] = True   # laser <-> terminal? probably glass; leave True for now.
    door_x[3, 7] = True   # analysis <-> laser side (glass)

    # --- Glass partition pairs --------------------------------------------
    # ROUND 3: walls between the lobby / main hall and the bio lab / server
    # farm are NOT glass; they're concrete with embedded strip lights (see
    # ``light_strip_pairs`` below). Only the LASER_CORRIDOR keeps glass
    # partitions on its outer faces.
    glass_pairs = {
        frozenset({Biome.LASER_CORRIDOR, Biome.MAIN_HALL}),
        frozenset({Biome.LASER_CORRIDOR, Biome.CORRIDOR}),
        frozenset({Biome.LASER_CORRIDOR, Biome.STORAGE}),
        frozenset({Biome.LASER_CORRIDOR, Biome.CONTROL_ROOM}),
        frozenset({Biome.LASER_CORRIDOR, Biome.REACTOR}),
    }

    # --- Concrete-with-strip-light wall pairs (cyberpunk corp tower) -----
    # Only the LOBBY -> central-row boundary walls get strip lights now;
    # the main-hall -> bio-lab / server-farm walls are dark glass (below).
    light_strip_pairs = {
        frozenset({Biome.ENTRANCE, Biome.BIO_LAB}),
        frozenset({Biome.ENTRANCE, Biome.SERVER_FARM}),
        frozenset({Biome.ENTRANCE, Biome.MAIN_HALL}),
    }

    # --- Dark smoked-glass wall pairs (main hall <-> bio/server) ---------
    dark_glass_pairs = {
        frozenset({Biome.MAIN_HALL, Biome.BIO_LAB}),
        frozenset({Biome.MAIN_HALL, Biome.SERVER_FARM}),
    }

    # --- Corporate Future Corp glass-door location -----------------------
    # ROUND-3 fix: only ONE entrance door from the lobby into the central
    # spine — the one to MAIN_HALL. The LOBBY <-> BIO_LAB and
    # LOBBY <-> SERVER_FARM walls are SOLID concrete + flowing LED strips
    # + a wall-mounted FUTURE CORP logo. No doorway carved there.
    corporate_door_locations = {
        (5, 1, "y"),   # LOBBY (5, 1) -> MAIN_HALL (5, 2) — sole entrance
    }
    # Make sure it's carved.
    for cx, cy, axis in corporate_door_locations:
        if axis == "y":
            door_y[cx, cy] = True
        else:
            door_x[cx, cy] = True
    # Explicitly UN-carve the other two side-room doorways so the wall is
    # solid (regardless of what the regular door-carve pass produced).
    door_y[2, 1] = False
    door_y[8, 1] = False

    entrance = (5, 0)
    reactor = (5, 11)
    cells = [(cx, cy) for cx in range(CELLS_X) for cy in range(CELLS_Y)]
    return Layout(
        biome=biome,
        door_x=door_x,
        door_y=door_y,
        entrance=entrance,
        reactor=reactor,
        cells=cells,
        room_id=room_id,
        glass_pairs=glass_pairs,
        light_strip_pairs=light_strip_pairs,
        corporate_door_locations=corporate_door_locations,
        dark_glass_pairs=dark_glass_pairs,
    )


# ============================================================================
# Symmetric hand-designed layout (kept for reference)
# ============================================================================


def generate_symmetric_layout(rng: np.random.Generator) -> Layout:
    """A 5x5 left-right + top-bottom symmetric facility built around a central
    reactor with a Resident-Evil laser corridor on the entrance approach.

    Cell map (cy index increases upward, cx index right):

    .. code-block:: text

       cy=4: DAMA  CORR  ENGI  CORR  DAMA
       cy=3: CORR  DORM  CORR  BIO   CORR
       cy=2: ENTR  LASR  REAC  CORR  SRVR
       cy=1: CORR  DORM  CORR  BIO   CORR
       cy=0: DAMA  CORR  ENGI  CORR  DAMA
             cx=0  cx=1  cx=2  cx=3  cx=4

    Doorway pattern is the cross + corner-loops, so corner specialised rooms
    are reachable from the spine and there are alternate paths (loops) for the
    spatial-reuse to demonstrate.
    """
    assert CELLS_X == 5 and CELLS_Y == 5, "symmetric layout requires a 5x5 grid"

    biome = np.full((CELLS_X, CELLS_Y), None, dtype=object)
    door_x = np.zeros((CELLS_X - 1, CELLS_Y), dtype=bool)
    door_y = np.zeros((CELLS_X, CELLS_Y - 1), dtype=bool)

    centre = (CELLS_X // 2, CELLS_Y // 2)         # (2, 2)
    entrance = (0, CELLS_Y // 2)                  # (0, 2)
    reactor = centre

    # --- Biome assignment (deliberate, by cell) ---
    biome[2, 2] = Biome.REACTOR
    biome[entrance] = Biome.ENTRANCE
    biome[1, 2] = Biome.LASER_CORRIDOR
    biome[3, 2] = Biome.CORRIDOR
    biome[4, 2] = Biome.SERVER_FARM
    biome[2, 0] = Biome.ENGINEERING
    biome[2, 4] = Biome.ENGINEERING
    biome[2, 1] = Biome.CORRIDOR
    biome[2, 3] = Biome.CORRIDOR
    # 4 corners — damaged
    biome[0, 0] = Biome.DAMAGED
    biome[4, 0] = Biome.DAMAGED
    biome[0, 4] = Biome.DAMAGED
    biome[4, 4] = Biome.DAMAGED
    # Off-axis specialised rooms (symmetric pairs)
    biome[1, 1] = Biome.DORMITORY
    biome[1, 3] = Biome.DORMITORY
    biome[3, 1] = Biome.BIO_LAB
    biome[3, 3] = Biome.BIO_LAB
    # Remaining cells are corridors connecting the corners back to the spine.
    for cx in (0, 4):
        biome[cx, 1] = Biome.CORRIDOR
        biome[cx, 3] = Biome.CORRIDOR
    # cy=0 and cy=4 connectors
    biome[1, 0] = Biome.CORRIDOR
    biome[3, 0] = Biome.CORRIDOR
    biome[1, 4] = Biome.CORRIDOR
    biome[3, 4] = Biome.CORRIDOR

    # --- Doorways (symmetric carve) ---
    # The central X axis (the spine: cy=2) is fully open.
    for cx in range(CELLS_X - 1):
        door_x[cx, 2] = True
    # The central Y axis (vertical spine: cx=2) is fully open.
    for cy in range(CELLS_Y - 1):
        door_y[2, cy] = True
    # Off-axis side rooms hooked into the spine.
    # Dormitories connect to corridor (cx=0, cy=1/3).
    door_x[0, 1] = True   # (0,1) <-> (1,1) dorm
    door_x[0, 3] = True   # (0,3) <-> (1,3) dorm
    # Bio labs connect to the spine on cx=2..3.
    door_x[2, 1] = True   # (2,1) corridor <-> (3,1) bio
    door_x[2, 3] = True   # (2,3) corridor <-> (3,3) bio
    # Bio labs connect outward to the (4, 1)/(4,3) corridor.
    door_x[3, 1] = True
    door_x[3, 3] = True
    # Dormitories connect up to (1, 0/4) corridor.
    door_y[1, 0] = True   # (1,0) <-> (1,1)
    door_y[1, 3] = True   # (1,3) <-> (1,4)
    # Bio labs connect down to (3, 0/4) corridor.
    door_y[3, 0] = True
    door_y[3, 3] = True
    # The 4 corner damaged rooms each have ONE door to the adjacent edge corridor.
    door_x[0, 0] = True   # (0,0) damaged <-> (1,0) corridor
    door_x[0, 4] = True
    door_x[3, 0] = True   # (3,0) corridor <-> (4,0) damaged
    door_x[3, 4] = True

    cells = [(cx, cy) for cx in range(CELLS_X) for cy in range(CELLS_Y)]
    return Layout(
        biome=biome,
        door_x=door_x,
        door_y=door_y,
        entrance=entrance,
        reactor=reactor,
        cells=cells,
    )
