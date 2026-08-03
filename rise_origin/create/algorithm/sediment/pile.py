# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The rock pile: stones grown into contact over an aperiodic tiling.

Peytavie et al. 2009, *Procedural Generation of Rock Piles using
Aperiodic Tiling*, sections 3, 4 and 6. Section 3 lays a corner cube
grid whose seed points are shared between neighboring cubes, section 4
grows those seeds into stone volumes under an anisotropic distance, and
section 6 keeps only the stones a support shape holds. The wear that
carves the packed volumes apart is :class:`PileErosion` of the erosion
phase, which is where section 5 lives. :class:`SegmentTone` gives each
finished stone its own shade.
"""

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from ...protocol.rng import Rng
from ..field import ValueNoiseField
from .generator import UniformGenerator
from .volume import Volume

if TYPE_CHECKING:
    import torch

__all__ = ["PileGenerator", "SegmentTone"]


@dataclass(frozen=True)
class _PileSeeds:
    """The seeds reachable from one grid, keyed by lattice cell.

    Used only by :class:`PileGenerator`, which reads the lattice fields
    by slicing during growth and the flat rows by seed id afterward. A
    seed id is ``((hx * hy + hy) * hz + hz) * slot_num + slot`` over the
    lattice, which is the order the flat rows come out in.

    Attributes:
        halo_num: (hx, hy, hz) cells of the lattice covering the grid
            and its search reach.
        slot_num: How many seed slots each lattice cell holds.
        seed_num: (hx, hy, hz) int64 torch tensor of seeds per cell.
        point: (hx, hy, hz, slot, 3) float32 torch tensor of seed
            positions in the grid frame (unit: m).
        growth: (hx, hy, hz, slot) float32 torch tensor of the growth
            factor, the paper's per-seed size control.
        coefficient: (hx, hy, hz, slot, 3) float32 torch tensor of the
            per-frame-direction coefficients.
        frame: (hx, hy, hz, slot, 3, 3) float32 torch tensor of the seed
            frames, one unit direction per row.
        draw: (hx, hy, hz, slot) float32 torch tensor of one value in 0
            to 1 per seed, keyed by lattice position so two grids
            overlapping the same seed draw the same value.
    """

    halo_num: Tuple[int, int, int]
    slot_num: int
    seed_num: "torch.Tensor"
    point: "torch.Tensor"
    growth: "torch.Tensor"
    coefficient: "torch.Tensor"
    frame: "torch.Tensor"
    draw: "torch.Tensor"

    def get_seed_total(self) -> int:
        """Return how many seed ids the flat rows span."""
        return int(np.prod(self.halo_num)) * self.slot_num

    def make_flat_rows(self) -> tuple:
        """Return point, growth, coefficient and frame by seed id."""
        total = self.get_seed_total()
        return (
            self.point.reshape(total, 3),
            self.growth.reshape(total),
            self.coefficient.reshape(total, 3),
            self.frame.reshape(total, 3, 3),
        )


class PileGenerator(UniformGenerator):
    """Generates a packed pile of stones filling a chosen shape.

    Section 3 decomposes a corner cube of ``cube_cell_num`` cells per
    side into vertex, edge, face and center regions and builds a seed
    bank holding one entry per region configuration over two corner
    colors: 2 vertex, 12 edge, 48 face and 256 center entries. The four
    region kinds partition the cube and are filled in that order, so
    every cell is seeded exactly once by the earliest kind that reaches
    it, which is the paper's copy-then-fill rule stated as a partition.
    Each entry is keyed by the lattice colors it depends on, so two
    cubes meeting at a corner, an edge or a face read the same entry
    there and their stones match across the join. A lattice corner takes
    its color from a hash of its position, so the tiling never repeats.

    Section 4 grows every seed under the anisotropic distance
    ``d_k(p) = f_k(p) / g_k``, where
    ``f_k(p) = alpha_k ((p - p_k).x)^2 + beta_k ((p - p_k).y)^2
    + gamma_k ((p - p_k).z)^2`` over the seed's own frame; the nearest
    seed under that distance owns the cell.

    The paper's symbols, and the dials that set them:

    ============================== ===============================
    ``stone_size``                 the corner cube cell edge
    ``cube_cell_num``              n, the cells per cube side
    ``seed_num_high``              s, with n_k drawn in ``[1, s]``
    ``size_spread_ratio``          the g_k interval ``[1, g_max]``
    ``alpha``, ``beta``, ``gamma`` alpha_k, beta_k and gamma_k
    ``contact_radius_share``       a, the contact radius
    ============================== ===============================

    Seed points are p_k and contact points c_ij. The paper's blending
    radius b has no dial, since section 5 is stepped rather than ramped;
    :class:`PileErosion` records why.

    The pile is born packed: every cell inside the shape belongs to some
    stone and no air stands between them. Stone boundaries carry
    ``SEAM_MARK`` so wear opens them into gaps, and the cells around
    each contact point carry ``CONTACT_MARK`` so wear leaves them alone
    and the pile keeps resting on itself.

    Named deviations, all recorded here:

    - The bank is keyed and the tiling read from lattice geometry
      instead of from assembled tiles, so the paper's 256 corner cubes
      are built but never materialized as geometry. The seeds are
      identical either way.
    - The nearest seed is searched over the lattice cells within
      ``SEARCH_CELL_NUM`` of the cell being labeled. An exact search
      would need a reach set by the widest ratio the size and shape
      dials allow; past this reach the only cells that change hands are
      boundary cells, and the draft splits every stone by connectivity
      afterward, so a changed owner stays an honest stone.
    - Section 6 measures a stone against the shape by the share of its
      cells inside it, rather than by the paper's anchor points. As in
      the paper the stone is then kept whole, so a pile ends on stone
      surfaces and stands about a stone proud of its shape.

    Note: the paper meshes its rocks by polygonizing the implicit field
    ``r(v) = 2 rho(v) - 1`` over this same grid. :class:`PileErosion`
    keeps ``rho`` as ``1 - wear_progress``, so ``r`` is
    ``1 - 2 * wear_progress`` on solid cells and ``-1`` on empty ones. A
    mesh path would run marching cubes on that field once per
    ``segment_id`` and hand the triangles to ``manifold3d`` for the
    boolean cleanup and the watertightness check; nothing upstream would
    change, because the voxel path already carries the field the paper
    meshes.

    Args:
        grid: The evaluation grid the volume is generated on.
        device: The torch device name the volume lives on.
        rng: The random stream the seed bank draws from.
        material_shares: How the stones divide between the materials, as
            shares in job order; volume material ids run from 1.
        stone_size: The corner cube cell edge (unit: m), which one seed
            per cell grows to about; snapped to a whole number of grid
            cells.
        size_spread_ratio: How widely stone sizes spread, 0 to 1; it
            sets the growth interval to ``[1, 1 + 3 * ratio]``, reaching
            the paper's growth ceiling of 4 at 1.
        seed_num_high: The most seeds one cell holds, the paper's seed
            count per cell; 1 gives one stone per cell and larger values
            mix sizes.
        alpha: The coefficient of ``f_k`` along the seed's first frame
            direction, drawn over an interval around it.
        beta: The same along the second frame direction.
        gamma: The same along the third frame direction. All three at 1
            give the paper's round pebbles of Figure 11, and 5, 5, 1 its
            flat plates of Figure 13.
        cube_cell_num: The cells per corner cube side; 5 or more, since
            a smaller cube leaves no room for the edge, face and center
            regions.
        contact_radius_share: The paper's contact radius, as a share of
            ``stone_size``; wear never touches inside it. Keep it small:
            the paper needs a wide patch to stop a meshed pile parting,
            but voxels of two segments hold on their own, so a wide one
            here only welds the stones into a mass.
        stone_keep_share: The least share of a stone's cells that must
            lie inside the shape for the stone to be kept whole; a low
            share lets the pile stand further proud of its shape.
        shape: The generated shape, as :class:`UniformGenerator` reads
            it; the pile fills whatever the shape holds.
        shape_arguments: The rest of the shape's arguments, passed
            through unchanged.

    Attributes:
        stone_size: The snapped corner cube cell edge (unit: m).
        cell_span: How many grid cells one corner cube cell spans.
    """

    # The paper's two corner colors; a complete corner cube set is
    # 2 ** 8 = 256 cubes, which is what the center bank holds.
    CORNER_COLOR_NUM = 2

    # A cube smaller than this has no room for all four region kinds:
    # the vertex regions and their neighborhoods already claim two cells
    # in from each side, so the middle band would be empty.
    MIN_CUBE_CELL_NUM = 5

    # How far the nearest-seed search reaches, in corner cube cells.
    SEARCH_CELL_NUM = 2

    # The paper's growth ceiling at a full size spread.
    MAX_GROWTH = 4.0

    # How wide the interval one stone's coefficients are drawn over, as
    # a share of what is asked, so a bed is not a bed of matching stones.
    COEFFICIENT_SWAY_RATIO = 0.3

    # How many halvings the contact point search runs; the span between
    # two seeds is under a millionth of a stone by then.
    BISECT_STEP_NUM = 24

    def __init__(
        self,
        grid,
        device: str,
        rng: Rng,
        material_shares: Sequence[float],
        stone_size: float,
        size_spread_ratio: float = 0.3,
        seed_num_high: int = 1,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        cube_cell_num: int = 5,
        contact_radius_share: float = 0.05,
        stone_keep_share: float = 0.7,
        shape: str = "box",
        **shape_arguments,
    ) -> None:
        if int(cube_cell_num) < self.MIN_CUBE_CELL_NUM:
            raise ValueError(
                f"cube_cell_num {cube_cell_num} is under "
                f"{self.MIN_CUBE_CELL_NUM}; a smaller corner cube has no "
                "edge, face and center regions"
            )
        if not len(material_shares):
            raise ValueError("a pile needs at least one material share")
        super().__init__(grid, device, rng, 1, shape, **shape_arguments)
        self.material_shares = tuple(float(share) for share in material_shares)
        self.cell_span = max(1, int(round(float(stone_size) / grid.cell_size)))
        self.stone_size = self.cell_span * grid.cell_size
        self.size_spread_ratio = float(size_spread_ratio)
        self.seed_num_high = max(1, int(seed_num_high))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.cube_cell_num = int(cube_cell_num)
        self.contact_radius_share = float(contact_radius_share)
        self.stone_keep_share = float(stone_keep_share)
        self._color_salt = np.uint64(rng.split("corner_colors").integers(1, 1 << 30))
        self._seed_salt = np.uint64(rng.split("seed_draws").integers(1, 1 << 30))
        self._region_bank = self._make_region_bank()

    def _fill_materials(self, volume: Volume, solid: "torch.Tensor") -> None:
        """Grow the stones, then mark their seams and their contacts.

        Args:
            volume: The volume whose channels fill.
            solid: (X, Y, Z) bool torch tensor of the generated shape.
        """
        import torch

        seeds = self._make_seed_lattice()
        labels = self._make_stone_labels(seeds)
        stone_id, stone_seed = self._make_kept_stones(labels, solid, seeds)
        segment = stone_id[labels]
        # Kept stones are emitted whole, so the pile ends on stone
        # surfaces rather than on the shape's own plane.
        inside = segment >= 0
        volume.segment_id = torch.where(
            inside, segment, torch.full_like(segment, -1)
        ).to(torch.int32)
        material = self._make_stone_materials(seeds, stone_seed)
        volume.material_id = torch.where(
            inside,
            material[torch.clamp(segment, min=0)],
            torch.zeros_like(volume.material_id),
        )
        volume.write_seam_marks(volume.grid.cell_size)
        self._write_contact_marks(volume, seeds, stone_seed, inside)

    # ------------------------------------------------------------------
    # Section 3: the corner cube grid and its seed bank
    # ------------------------------------------------------------------

    def _make_region_bank(self) -> Dict[str, dict]:
        """Return the seed bank, one entry per region configuration.

        Each region kind is stacked into one array per field, indexed by
        the configuration key, so resolving a lattice cell later is a
        plain lookup. Vertex entries are filled first, then edge, face
        and center, which is the paper's order.
        """
        middle_num = self.cube_cell_num - 4
        return {
            "vertex": self._make_kind_seeds(
                [("vertex", color) for color in range(self.CORNER_COLOR_NUM)],
                (4, 4, 4),
            ),
            "edge": self._make_kind_seeds(
                [
                    ("edge", side, low_color, high_color)
                    for side in range(3)
                    for low_color in range(self.CORNER_COLOR_NUM)
                    for high_color in range(self.CORNER_COLOR_NUM)
                ],
                (middle_num, 4, 4),
            ),
            "face": self._make_kind_seeds(
                [
                    ("face", side) + PileGenerator._make_color_row(pattern, 4)
                    for side in range(3)
                    for pattern in range(self.CORNER_COLOR_NUM**4)
                ],
                (4, middle_num, middle_num),
            ),
            "center": self._make_kind_seeds(
                [
                    ("center",) + PileGenerator._make_color_row(pattern, 8)
                    for pattern in range(self.CORNER_COLOR_NUM**8)
                ],
                (middle_num, middle_num, middle_num),
            ),
        }

    def _make_kind_seeds(self, keys: list, cell_num: tuple) -> dict:
        """Return one region kind's entries stacked by key index.

        Args:
            keys: The configuration keys of this kind, in key-index
                order; each one salts its own stream, so the same
                configuration always seeds the same way.
            cell_num: How many cells one entry spans along each of its
                own three directions.
        """
        entries = [self._make_region_seeds(key, cell_num) for key in keys]
        return {
            name: np.stack([entry[name] for entry in entries]) for name in entries[0]
        }

    def _make_region_seeds(self, key: tuple, cell_num: tuple) -> dict:
        """Return one region entry's seeds, drawn once per cell.

        Args:
            key: The region's configuration key.
            cell_num: How many cells the entry spans along each of its
                own three directions.
        """
        stream = self.rng.split("/".join(str(part) for part in key))
        generator = stream.get_numpy_generator()
        slot_num = self.seed_num_high
        shape = tuple(cell_num) + (slot_num,)
        seed_num = generator.integers(1, slot_num + 1, size=cell_num)
        start = generator.integers(0, 1 << 24, size=cell_num)
        point = np.zeros(shape + (3,))
        for slot in range(slot_num):
            for side, base in enumerate((2, 3, 5)):
                point[..., slot, side] = PileGenerator._make_halton(start + slot, base)
        growth = 1.0 + generator.random(shape) * (
            (self.MAX_GROWTH - 1.0) * self.size_spread_ratio
        )
        # The paper reads one coefficient triple per rock family and
        # draws each seed's over an interval around it.
        sway = self.COEFFICIENT_SWAY_RATIO
        coefficient = np.ones(shape + (3,))
        for side, asked in enumerate((self.alpha, self.beta, self.gamma)):
            coefficient[..., side] = asked * (
                1.0 + (generator.random(shape) - 0.5) * sway
            )
        frame = PileGenerator._make_frames(generator.random(shape + (3,)))
        return {
            "seed_num": seed_num.astype(np.int64),
            "point": point,
            "growth": growth,
            "coefficient": coefficient,
            "frame": frame,
        }

    def _make_seed_lattice(self) -> _PileSeeds:
        """Return the seeds of every corner cube cell the grid reaches.

        Each cell of the lattice covering the grid plus its search reach
        is resolved to the bank entry that owns it, so the growth pass
        can read seeds by plain slicing.
        """
        import torch

        reach = self.SEARCH_CELL_NUM
        block_num = self._make_block_num()
        halo_num = tuple(count + 2 * reach for count in block_num)
        lines = [np.arange(-reach, block_num[side] + reach) for side in range(3)]
        block = np.stack(np.meshgrid(*lines, indexing="ij"), axis=-1)
        # The lattice is named from the world, so two grids whose origins
        # differ by whole lattice cells read the same cells there.
        cell = block + self._make_lattice_base()
        cube = np.floor_divide(cell, self.cube_cell_num)
        local = cell - cube * self.cube_cell_num
        is_low = local <= 1
        is_high = local >= self.cube_cell_num - 2
        is_corner = is_low | is_high
        # The corner-band offset runs 0 to 3 across the lattice point, so
        # a cell reads the same offset from either cube meeting there.
        corner_offset = np.where(is_low, local, local - self.cube_cell_num) + 2
        corner_point = np.where(is_high, cube + 1, cube)
        slot_num = self.seed_num_high
        fields = {
            "seed_num": np.zeros(halo_num, dtype=np.int64),
            "point": np.zeros(halo_num + (slot_num, 3)),
            "growth": np.zeros(halo_num + (slot_num,)),
            "coefficient": np.zeros(halo_num + (slot_num, 3)),
            "frame": np.zeros(halo_num + (slot_num, 3, 3)),
        }
        readers = (
            (3, "vertex", self._make_vertex_lookup),
            (2, "edge", self._make_edge_lookup),
            (1, "face", self._make_face_lookup),
            (0, "center", self._make_center_lookup),
        )
        corner_num = is_corner.sum(axis=-1)
        for count, kind, make_lookup in readers:
            picked = corner_num == count
            if not bool(picked.any()):
                continue
            key_index, offset = make_lookup(
                cube[picked],
                local[picked],
                is_corner[picked],
                corner_offset[picked],
                corner_point[picked],
            )
            entry = self._region_bank[kind]
            for name, values in fields.items():
                values[picked] = entry[name][
                    key_index, offset[:, 0], offset[:, 1], offset[:, 2]
                ]
        # A seed sits at its own lattice cell's low corner plus its
        # fraction of one cell, measured in the grid frame.
        low = block * self.stone_size + np.asarray(self.grid.origin)
        point = fields["point"] * self.stone_size + low[:, :, :, None, :]
        draw = PileGenerator._make_seed_draws(cell, slot_num, self._seed_salt)
        return _PileSeeds(
            halo_num=halo_num,
            slot_num=slot_num,
            seed_num=torch.as_tensor(fields["seed_num"], device=self.device),
            point=torch.as_tensor(point, dtype=torch.float32, device=self.device),
            growth=torch.as_tensor(
                fields["growth"], dtype=torch.float32, device=self.device
            ),
            coefficient=torch.as_tensor(
                fields["coefficient"], dtype=torch.float32, device=self.device
            ),
            frame=torch.as_tensor(
                fields["frame"], dtype=torch.float32, device=self.device
            ),
            draw=torch.as_tensor(draw, dtype=torch.float32, device=self.device),
        )

    def _make_vertex_lookup(
        self,
        cube: np.ndarray,
        local: np.ndarray,
        is_corner: np.ndarray,
        corner_offset: np.ndarray,
        corner_point: np.ndarray,
    ) -> tuple:
        """Return the bank key index and entry offset of vertex cells."""
        return self._make_colors(corner_point), corner_offset

    def _make_edge_lookup(
        self,
        cube: np.ndarray,
        local: np.ndarray,
        is_corner: np.ndarray,
        corner_offset: np.ndarray,
        corner_point: np.ndarray,
    ) -> tuple:
        """Return the bank key index and entry offset of edge cells.

        The edge runs along the one side sitting in the middle band, and
        its two lattice ends are read in rising order along that side,
        so the four cubes sharing the edge all name it the same way.
        """
        along = np.argmin(is_corner, axis=-1)
        across = PileGenerator._make_other_sides(along)
        is_along = np.arange(3)[None, :] == along[:, None]
        low_point = np.where(is_along, cube, corner_point)
        high_point = low_point + is_along
        key_index = (
            along * 4 + self._make_colors(low_point) * 2 + self._make_colors(high_point)
        )
        offset = np.stack(
            [
                np.take_along_axis(local, along[:, None], axis=1)[:, 0] - 2,
                np.take_along_axis(corner_offset, across[:, 0:1], axis=1)[:, 0],
                np.take_along_axis(corner_offset, across[:, 1:2], axis=1)[:, 0],
            ],
            axis=1,
        )
        return key_index, offset

    def _make_face_lookup(
        self,
        cube: np.ndarray,
        local: np.ndarray,
        is_corner: np.ndarray,
        corner_offset: np.ndarray,
        corner_point: np.ndarray,
    ) -> tuple:
        """Return the bank key index and entry offset of face cells.

        The face lies across the one side sitting in a corner band, and
        its four lattice corners are read in rising order along the two
        remaining sides, so both cubes sharing it name it the same way.
        """
        across = np.argmax(is_corner, axis=-1)
        along = PileGenerator._make_other_sides(across)
        is_across = np.arange(3)[None, :] == across[:, None]
        base = np.where(is_across, corner_point, cube)
        row = np.arange(base.shape[0])
        key_index = across * 16
        for corner in range(4):
            point = base.copy()
            for place in range(2):
                if (corner >> place) & 1:
                    point[row, along[:, place]] += 1
            key_index = key_index + self._make_colors(point) * (1 << corner)
        offset = np.stack(
            [
                np.take_along_axis(corner_offset, across[:, None], axis=1)[:, 0],
                np.take_along_axis(local, along[:, 0:1], axis=1)[:, 0] - 2,
                np.take_along_axis(local, along[:, 1:2], axis=1)[:, 0] - 2,
            ],
            axis=1,
        )
        return key_index, offset

    def _make_center_lookup(
        self,
        cube: np.ndarray,
        local: np.ndarray,
        is_corner: np.ndarray,
        corner_offset: np.ndarray,
        corner_point: np.ndarray,
    ) -> tuple:
        """Return the bank key index and entry offset of center cells."""
        key_index = np.zeros(cube.shape[0], dtype=np.int64)
        for corner in range(8):
            step = np.array([(corner >> place) & 1 for place in range(3)])
            key_index = key_index + self._make_colors(cube + step) * (1 << corner)
        return key_index, local - 2

    def _make_colors(self, point: np.ndarray) -> np.ndarray:
        """Return the corner color of each lattice point.

        Args:
            point: (n, 3) integer array of lattice points.
        """
        value = PileGenerator._make_point_hash(point) ^ self._color_salt
        value = (value ^ (value >> np.uint64(29))) * np.uint64(1274126177)
        value = value ^ (value >> np.uint64(32))
        return (value % np.uint64(self.CORNER_COLOR_NUM)).astype(np.int64)

    def _make_block_num(self) -> tuple:
        """Return how many lattice cells the grid spans along each side."""
        return tuple(-(-count // self.cell_span) for count in self.grid.cell_num)

    def _make_lattice_base(self) -> np.ndarray:
        """Return the lattice name of the grid's first cell.

        Naming the lattice from the world instead of from the grid is
        what lets two grids share the seeds where they overlap; the
        names line up with the geometry when a grid origin sits a whole
        number of stone sizes from the world origin.
        """
        return np.round(
            np.asarray(self.grid.origin, dtype=np.float64) / self.stone_size
        ).astype(np.int64)

    # ------------------------------------------------------------------
    # Section 4: growing the seeds into stone volumes
    # ------------------------------------------------------------------

    def _make_stone_labels(self, seeds: _PileSeeds) -> "torch.Tensor":
        """Return the seed id owning each grid cell.

        Every cell reads the seeds of the lattice cells within the
        search reach and keeps the nearest one under the anisotropic
        distance. The lattice is aligned to the grid, so the reach comes
        out as plain slicing rather than a per-cell gather.

        Args:
            seeds: The lattice seeds covering the grid.
        """
        import torch

        span = self.cell_span
        reach = self.SEARCH_CELL_NUM
        block_num = self._make_block_num()
        origin = self.grid.origin
        line = (
            torch.arange(span, device=self.device, dtype=torch.float32) + 0.5
        ) * self.grid.cell_size
        cell_lines = []
        for side in range(3):
            view = [1, 1, 1, 1, 1, 1]
            view[2 * side + 1] = span
            block_line = torch.arange(
                block_num[side], device=self.device, dtype=torch.float32
            ) * self.stone_size + float(origin[side])
            view[2 * side] = block_num[side]
            cell_lines.append(
                (block_line.reshape(-1, 1) + line.reshape(1, -1)).reshape(view)
            )
        best = torch.full(
            (block_num[0], span, block_num[1], span, block_num[2], span),
            float("inf"),
            device=self.device,
        )
        label = torch.zeros_like(best, dtype=torch.int64)
        id_step = torch.arange(
            int(np.prod(seeds.halo_num)), device=self.device, dtype=torch.int64
        ).reshape(seeds.halo_num)
        for step_x in range(2 * reach + 1):
            for step_y in range(2 * reach + 1):
                for step_z in range(2 * reach + 1):
                    # The reach is a ball, not a cube: a lattice cell in
                    # the corner sits reach times root three away and its
                    # seeds never win, so skipping those corners leaves
                    # the labels untouched and visits 33 windows in place
                    # of 125.
                    away = (
                        (step_x - reach) ** 2
                        + (step_y - reach) ** 2
                        + (step_z - reach) ** 2
                    )
                    if away > reach**2:
                        continue
                    window = (
                        slice(step_x, step_x + block_num[0]),
                        slice(step_y, step_y + block_num[1]),
                        slice(step_z, step_z + block_num[2]),
                    )
                    self._grow_from_window(
                        seeds, window, cell_lines, id_step, best, label
                    )
        flat = label.reshape(tuple(count * span for count in block_num))
        return flat[
            : self.grid.cell_num[0],
            : self.grid.cell_num[1],
            : self.grid.cell_num[2],
        ].contiguous()

    def _grow_from_window(
        self,
        seeds: _PileSeeds,
        window: tuple,
        cell_lines: list,
        id_step: "torch.Tensor",
        best: "torch.Tensor",
        label: "torch.Tensor",
    ) -> None:
        """Let one lattice window's seeds claim the cells they win.

        Args:
            seeds: The lattice seeds covering the grid.
            window: The lattice slice this window reads.
            cell_lines: The three grid-frame coordinate lines (unit: m),
                each shaped to broadcast along its own side.
            id_step: (hx, hy, hz) int64 seed id base of each cell.
            best: The running least distance, written in place.
            label: The running seed id, written in place.
        """
        import torch

        seed_num = seeds.seed_num[window]
        view = (seed_num.shape[0], 1, seed_num.shape[1], 1, seed_num.shape[2], 1)
        for slot in range(seeds.slot_num):
            has_seed = seed_num > slot
            if not bool(has_seed.any()):
                continue
            point = seeds.point[window][:, :, :, slot]
            growth = seeds.growth[window][:, :, :, slot]
            coefficient = seeds.coefficient[window][:, :, :, slot]
            frame = seeds.frame[window][:, :, :, slot]
            reaches = [
                cell_lines[side] - point[..., side].reshape(view) for side in range(3)
            ]
            distance = None
            for row in range(3):
                along = (
                    reaches[0] * frame[..., row, 0].reshape(view)
                    + reaches[1] * frame[..., row, 1].reshape(view)
                    + reaches[2] * frame[..., row, 2].reshape(view)
                )
                part = coefficient[..., row].reshape(view) * along * along
                distance = part if distance is None else distance + part
            distance = distance / growth.reshape(view)
            distance = torch.where(
                has_seed.reshape(view),
                distance,
                torch.full_like(distance, float("inf")),
            )
            is_better = distance < best
            torch.minimum(best, distance, out=best)
            seed_id = (id_step[window] * seeds.slot_num + slot).reshape(view)
            label.copy_(torch.where(is_better, seed_id, label))

    # ------------------------------------------------------------------
    # Section 6: which stones the shape instantiates, and the
    # marks section 5 wears against
    # ------------------------------------------------------------------

    def _make_kept_stones(
        self, labels: "torch.Tensor", solid: "torch.Tensor", seeds: _PileSeeds
    ) -> tuple:
        """Return each seed's stone id, and the seed ids kept.

        A stone is kept when enough of it lies inside the shape, which
        is the paper's instantiation test; the rest are dropped whole so
        the pile's rim carries whole stones instead of sliced ones.

        Args:
            labels: (X, Y, Z) int64 torch tensor of seed ids.
            solid: (X, Y, Z) bool torch tensor of the generated shape.
            seeds: The lattice seeds covering the grid.

        Returns:
            An int64 tensor mapping every seed id to its stone id, -1
            where the stone is dropped, and an int64 tensor of the seed
            ids kept, in stone id order.
        """
        import torch

        total = seeds.get_seed_total()
        whole_num = torch.bincount(labels.reshape(-1), minlength=total)
        inside_num = torch.bincount(labels[solid], minlength=total)
        is_kept = (inside_num > 0) & (
            inside_num.to(torch.float32)
            >= self.stone_keep_share * whole_num.to(torch.float32)
        )
        stone_seed = torch.nonzero(is_kept).reshape(-1)
        stone_id = torch.full((total,), -1, dtype=torch.int64, device=labels.device)
        stone_id[stone_seed] = torch.arange(
            stone_seed.shape[0], dtype=torch.int64, device=labels.device
        )
        return stone_id, stone_seed

    def _make_stone_materials(
        self, seeds: _PileSeeds, stone_seed: "torch.Tensor"
    ) -> "torch.Tensor":
        """Return the material id of each stone.

        A stone draws its material from its own seed's draw against the
        cumulative material shares, so a bed mixes minerals without any
        stone splitting between two of them.

        Args:
            seeds: The lattice seeds covering the grid.
            stone_seed: The seed id of each stone, in stone id order.
        """
        import torch

        edges = np.cumsum(np.asarray(self.material_shares, dtype=np.float64))
        edges = edges / max(float(edges[-1]), 1.0e-12)
        draw = seeds.draw.reshape(-1)[stone_seed]
        picked = torch.searchsorted(
            torch.as_tensor(edges, dtype=torch.float32, device=self.device),
            draw.contiguous(),
        )
        return torch.clamp(picked + 1, max=len(self.material_shares)).to(torch.uint8)

    def _write_contact_marks(
        self,
        volume: Volume,
        seeds: _PileSeeds,
        stone_seed: "torch.Tensor",
        inside: "torch.Tensor",
    ) -> None:
        """Mark the cells around every contact point between stones.

        A contact point sits where two neighboring stones measure the
        same distance along the line between their seeds, found by
        halving that line. Wear never touches the cells within the
        contact radius of one, so the stones keep resting on each other.

        Args:
            volume: The volume whose marks are written.
            seeds: The lattice seeds covering the grid.
            stone_seed: The seed id of each stone, in stone id order.
            inside: (X, Y, Z) bool torch tensor of the pile's cells.
        """
        import torch

        pairs = PileGenerator._make_touching_pairs(volume.segment_id, inside)
        if pairs.shape[0] == 0:
            return
        point, growth, coefficient, frame = seeds.make_flat_rows()
        first = stone_seed[pairs[:, 0]]
        second = stone_seed[pairs[:, 1]]
        low = torch.zeros(pairs.shape[0], device=self.device)
        high = torch.ones(pairs.shape[0], device=self.device)
        for _step in range(self.BISECT_STEP_NUM):
            middle = (low + high) / 2.0
            place = point[first] + (point[second] - point[first]) * middle[:, None]
            gap = PileGenerator._make_distance(
                place, point[first], frame[first], coefficient[first], growth[first]
            ) - PileGenerator._make_distance(
                place, point[second], frame[second], coefficient[second], growth[second]
            )
            low = torch.where(gap < 0.0, middle, low)
            high = torch.where(gap < 0.0, high, middle)
        share = ((low + high) / 2.0)[:, None]
        touch = point[first] + (point[second] - point[first]) * share
        self._write_contact_balls(volume, touch, inside)

    def _write_contact_balls(
        self, volume: Volume, touch: "torch.Tensor", inside: "torch.Tensor"
    ) -> None:
        """Mark the pile cells within the contact radius of a point.

        Args:
            volume: The volume whose marks are written.
            touch: (n, 3) float32 torch tensor of contact points in the
                grid frame (unit: m).
            inside: (X, Y, Z) bool torch tensor of the pile's cells.
        """
        import torch

        cell_size = volume.grid.cell_size
        counts = volume.grid.cell_num
        origin = torch.as_tensor(
            volume.grid.origin, dtype=torch.float32, device=self.device
        )
        index = torch.floor((touch - origin) / cell_size).to(torch.int64)
        is_held = torch.ones(index.shape[0], dtype=torch.bool, device=self.device)
        for side in range(3):
            is_held &= (index[:, side] >= 0) & (index[:, side] < counts[side])
        index = index[is_held]
        if index.shape[0] == 0:
            return
        seed = torch.zeros(counts, dtype=torch.float32, device=self.device)
        seed[index[:, 0], index[:, 1], index[:, 2]] = 1.0
        radius = max(
            1, int(round(self.contact_radius_share * self.stone_size / cell_size))
        )
        line = torch.arange(
            -radius, radius + 1, device=self.device, dtype=torch.float32
        )
        ball = (
            line[:, None, None] ** 2
            + line[None, :, None] ** 2
            + line[None, None, :] ** 2
            <= float(radius) ** 2
        ).to(torch.float32)
        near = (
            torch.nn.functional.conv3d(
                seed[None, None], ball[None, None], padding=radius
            )[0, 0]
            > 0.5
        ) & inside
        if volume.structure_mark is None:
            volume.structure_mark = torch.zeros_like(volume.material_id)
        volume.structure_mark = torch.where(
            near,
            volume.structure_mark | Volume.CONTACT_MARK,
            volume.structure_mark,
        )

    @staticmethod
    def _make_touching_pairs(
        segment_id: "torch.Tensor", inside: "torch.Tensor"
    ) -> "torch.Tensor":
        """Return every pair of stones sharing a face, lower id first."""
        import torch

        found = []
        for side in range(3):
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[side] = slice(None, -1)
            high[side] = slice(1, None)
            first = segment_id[tuple(low)].to(torch.int64)
            second = segment_id[tuple(high)].to(torch.int64)
            differs = inside[tuple(low)] & inside[tuple(high)] & (first != second)
            found.append(torch.stack([first[differs], second[differs]], dim=1))
        pairs = torch.cat(found, dim=0)
        if pairs.shape[0] == 0:
            return pairs
        ordered = torch.stack([pairs.amin(dim=1), pairs.amax(dim=1)], dim=1)
        return torch.unique(ordered, dim=0)

    @staticmethod
    def _make_distance(
        place: "torch.Tensor",
        point: "torch.Tensor",
        frame: "torch.Tensor",
        coefficient: "torch.Tensor",
        growth: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return the anisotropic distance of points to their seeds."""
        reach = place - point
        along = (frame * reach[:, None, :]).sum(dim=2)
        return (coefficient * along * along).sum(dim=1) / growth

    @staticmethod
    def _make_seed_draws(
        cell: np.ndarray, slot_num: int, salt: np.uint64
    ) -> np.ndarray:
        """Return one value in 0 to 1 per seed of the lattice.

        The value is keyed by the seed's lattice position and slot, so
        two grids overlapping the same seed draw the same value.

        Args:
            cell: (hx, hy, hz, 3) integer lattice positions.
            slot_num: How many seed slots each lattice cell holds.
            salt: The salt this generator's draws hash against.
        """
        base = PileGenerator._make_point_hash(cell.reshape(-1, 3)).reshape(
            cell.shape[:3]
        )
        draw = np.zeros(cell.shape[:3] + (slot_num,))
        for slot in range(slot_num):
            step = np.uint64((slot * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
            value = (base ^ salt) + step
            value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            value = value ^ (value >> np.uint64(27))
            draw[..., slot] = (value % np.uint64(1 << 24)) / float(1 << 24)
        return draw

    @staticmethod
    def _make_point_hash(point: np.ndarray) -> np.ndarray:
        """Return one unsigned hash per integer point row."""
        return (
            point[:, 0].astype(np.uint64) * np.uint64(73856093)
            ^ point[:, 1].astype(np.uint64) * np.uint64(19349663)
            ^ point[:, 2].astype(np.uint64) * np.uint64(83492791)
        )

    @staticmethod
    def _make_other_sides(side: np.ndarray) -> np.ndarray:
        """Return the two remaining sides of each row, in rising order."""
        rows = np.tile(np.arange(3), (side.shape[0], 1))
        return rows[rows != side[:, None]].reshape(side.shape[0], 2)

    @staticmethod
    def _make_halton(index: np.ndarray, base: int) -> np.ndarray:
        """Return the radical inverse of an index row in one base.

        Args:
            index: Integer array of sequence positions.
            base: The prime base of this coordinate.
        """
        value = np.zeros(np.shape(index), dtype=np.float64)
        weight = 1.0
        rest = np.asarray(index, dtype=np.int64) + 1
        while bool((rest > 0).any()):
            weight = weight / base
            value = value + weight * (rest % base)
            rest = rest // base
        return value

    @staticmethod
    def _make_frames(draws: np.ndarray) -> np.ndarray:
        """Return uniformly turned frames from three draws each.

        Args:
            draws: (..., 3) float array of independent uniform draws.
        """
        first = draws[..., 0]
        second = draws[..., 1] * 2.0 * math.pi
        third = draws[..., 2] * 2.0 * math.pi
        root = np.sqrt(1.0 - first)
        grown = np.sqrt(first)
        part_x = root * np.sin(second)
        part_y = root * np.cos(second)
        part_z = grown * np.sin(third)
        part_w = grown * np.cos(third)
        frame = np.empty(draws.shape[:-1] + (3, 3))
        frame[..., 0, 0] = 1.0 - 2.0 * (part_y**2 + part_z**2)
        frame[..., 0, 1] = 2.0 * (part_x * part_y - part_z * part_w)
        frame[..., 0, 2] = 2.0 * (part_x * part_z + part_y * part_w)
        frame[..., 1, 0] = 2.0 * (part_x * part_y + part_z * part_w)
        frame[..., 1, 1] = 1.0 - 2.0 * (part_x**2 + part_z**2)
        frame[..., 1, 2] = 2.0 * (part_y * part_z - part_x * part_w)
        frame[..., 2, 0] = 2.0 * (part_x * part_z - part_y * part_w)
        frame[..., 2, 1] = 2.0 * (part_y * part_z + part_x * part_w)
        frame[..., 2, 2] = 1.0 - 2.0 * (part_x**2 + part_y**2)
        return frame

    @staticmethod
    def _make_color_row(pattern: int, length: int) -> tuple:
        """Return one color per bit of a pattern index."""
        return tuple((pattern >> place) & 1 for place in range(length))


class SegmentTone:
    """Gives every segment of a volume one optical mid of its own.

    Not from a paper. A stone reads as one stone when it carries one
    shade, so the tone is picked per segment and never drifts inside it:
    ``tone_bias`` sets where in the run the whole body settles, a value
    noise field drifts neighborhoods around it, and a per-segment draw
    swings each stone away from its neighbors. Substance mids are left
    alone, since only the shade run varies between stones.
    """

    @staticmethod
    def make_tones(
        volume: Volume,
        tone_rows: Sequence[Sequence[int]],
        tone_bias: float,
        shade_swing_ratio: float,
        drift_feature_size: float,
        rng: Rng,
    ) -> None:
        """Write one optical mid per segment of the volume.

        Args:
            volume: The volume whose optical mids are written; its
                segment ids must already be compact and start at 0.
            tone_rows: One optical mid run per material id, light end to
                dark, with an air row at index 0 included.
            tone_bias: Where in the run the body settles, 0 the light
                end to 1 the dark end.
            shade_swing_ratio: How far one stone swings from its
                neighbors along the run, 0 to 1.
            drift_feature_size: The size of the neighborhoods that share
                a shade (unit: m).
            rng: The random stream the drift and the swing draw from.
        """
        import torch

        if volume.segment_id is None:
            return
        solid = volume.get_solid_cells()
        if not bool(solid.any()):
            return
        segment = volume.segment_id.to(torch.int64)
        stone_num = int(segment[solid].max().item()) + 1
        if stone_num <= 0:
            return
        drift = ValueNoiseField(rng.split("drift"), drift_feature_size)(
            volume.grid, device=str(volume.device)
        ).to(torch.float32)
        flat_segment = segment[solid]
        weight = torch.bincount(flat_segment, minlength=stone_num).to(torch.float32)
        drift_sum = torch.bincount(
            flat_segment, weights=drift[solid], minlength=stone_num
        )
        stone_drift = drift_sum / torch.clamp(weight, min=1.0)
        swing = torch.as_tensor(
            rng.split("swing").get_numpy_generator().random(stone_num),
            dtype=torch.float32,
            device=volume.device,
        )
        place = torch.clamp(
            tone_bias + shade_swing_ratio * (0.5 * stone_drift + (swing - 0.5)),
            0.0,
            1.0,
        )
        picks = torch.zeros(
            (len(tone_rows), stone_num), dtype=torch.int64, device=volume.device
        )
        for material, run in enumerate(tone_rows):
            if not len(run):
                continue
            table = torch.as_tensor(tuple(run), dtype=torch.int64, device=volume.device)
            step = torch.round(place * (len(run) - 1)).to(torch.int64)
            picks[material] = table[torch.clamp(step, 0, len(run) - 1)]
        picked = picks[volume.material_id.to(torch.int64), torch.clamp(segment, min=0)]
        volume.optical_mid = torch.where(
            solid, picked.to(torch.uint8), torch.zeros_like(volume.material_id)
        )
