# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The sediment volume: the one state every phase reads and writes.

Three kinds of per-voxel information stay separate and are never
conflated: what material each point is (``material_id``), which piece
it belongs to (``segment_id``, one segment can span several materials),
and what structural marks it carries (``structure_mark``, where a seam
and a contact are different marks). Channels are allocated by the phase
that writes them, so a volume never pays for information it does not
use.
"""

from typing import Optional, Sequence, Tuple, TYPE_CHECKING

import cc3d
import numpy as np

from ..grid_span import GridSpan

if TYPE_CHECKING:
    import torch

__all__ = ["Volume"]


class Volume:
    """The voxel state of one generation job, on one device.

    Args:
        grid: The evaluation grid the channels are indexed by.
        device: The torch device name the channels live on.

    Attributes:
        SEAM_MARK: The structure-mark bit of a fracture seam voxel.
        CONTACT_MARK: The structure-mark bit of a protected contact
            voxel.
        grid: The evaluation grid.
        device: The torch device of every channel.
        material_id: (X, Y, Z) uint8 torch tensor on ``device``; the
            material of each cell, 0 meaning air; always present.
        segment_id: (X, Y, Z) int32 torch tensor on ``device``; the
            piece each cell belongs to, -1 meaning none; None until a
            fracture phase writes it.
        structure_mark: (X, Y, Z) uint8 torch tensor on ``device``;
            mark bits per cell; None until a fracture phase writes it.
        optical_mid: (X, Y, Z) uint8 torch tensor on ``device``; the
            optical mid within each cell's material; None until a
            generator writes it.
        wear_progress: (X, Y, Z) float32 torch tensor on ``device``;
            accumulated wear toward 1 where a cell erodes; None until
            an erosion phase writes it.
    """

    SEAM_MARK = 1
    CONTACT_MARK = 2

    def __init__(self, grid: GridSpan, device: str) -> None:
        import torch

        self.grid = grid
        self.device = torch.device(device)
        self.material_id = torch.zeros(
            grid.cell_num, dtype=torch.uint8, device=self.device
        )
        self.segment_id: Optional["torch.Tensor"] = None
        self.structure_mark: Optional["torch.Tensor"] = None
        self.optical_mid: Optional["torch.Tensor"] = None
        self.wear_progress: Optional["torch.Tensor"] = None

    @staticmethod
    def make_from_density(
        grid: GridSpan, density: "torch.Tensor", fill_material_id: int, device: str
    ) -> "Volume":
        """Return a volume solid where a density tensor is positive.

        Args:
            grid: The evaluation grid.
            density: (X, Y, Z) float32 torch tensor of density values;
                a cell is solid where its value is positive.
            fill_material_id: The material every solid cell starts as.
            device: The torch device name the channels live on.
        """
        import torch

        volume = Volume(grid, device)
        solid = density.to(volume.device) > 0.0
        volume.material_id = torch.where(
            solid,
            torch.full_like(volume.material_id, fill_material_id),
            volume.material_id,
        )
        return volume

    @staticmethod
    def make_from_height(
        grid: GridSpan, heights: np.ndarray, fill_material_id: int, device: str
    ) -> "Volume":
        """Return a volume solid below a height grid.

        Args:
            grid: The evaluation grid.
            heights: (X, Y) float numpy array or torch tensor of
                surface heights in the job's local frame (unit: m).
            fill_material_id: The material every solid cell starts as.
            device: The torch device name the channels live on.
        """
        import torch

        volume = Volume(grid, device)
        height_line = torch.as_tensor(
            heights, dtype=torch.float32, device=volume.device
        )
        axis_z = grid.make_axes(str(volume.device))[2].to(torch.float32)
        solid = axis_z[None, None, :] < height_line[:, :, None]
        volume.material_id = torch.where(
            solid,
            torch.full_like(volume.material_id, fill_material_id),
            volume.material_id,
        )
        return volume

    def get_solid_cells(self) -> "torch.Tensor":
        """Return the (X, Y, Z) bool torch tensor of non-air cells."""
        return self.material_id != 0

    def write_seam_marks(self, seam_width: float = 0.0) -> None:
        """Mark the seams where two segments meet, at a real width.

        A crack finer than one cell cannot be drawn, so a width under
        the cell size marks nothing and the mass reads uncracked, which
        is how a coarse rock honestly looks.

        Args:
            seam_width: How wide the seams read (unit: m); the boundary
                itself already spans two cells, so a wider seam grows
                out from there.
        """
        import torch

        if self.segment_id is None:
            return
        width_cell_num = float(seam_width) / self.grid.cell_size
        if width_cell_num < 1.0:
            return
        solid = self.get_solid_cells()
        seam = torch.zeros_like(solid)
        for axis in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(None, -1)
            hi[axis] = slice(1, None)
            differs = (
                (self.segment_id[tuple(lo)] != self.segment_id[tuple(hi)])
                & solid[tuple(lo)]
                & solid[tuple(hi)]
                & (self.segment_id[tuple(lo)] >= 0)
                & (self.segment_id[tuple(hi)] >= 0)
            )
            seam[tuple(lo)] |= differs
            seam[tuple(hi)] |= differs
        for _grow in range(max(0, int(round((width_cell_num - 2.0) / 2.0)))):
            seam = (
                torch.nn.functional.max_pool3d(
                    seam.to(torch.float32)[None, None],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )[0, 0]
                > 0.5
            ) & solid
        if self.structure_mark is None:
            self.structure_mark = torch.zeros_like(self.material_id)
        self.structure_mark = torch.where(
            seam,
            self.structure_mark | Volume.SEAM_MARK,
            self.structure_mark,
        )

    def remove_debris(self) -> None:
        """Drop solid cells left with one or no solid neighbor.

        A few passes catch the chains a first removal exposes without
        chewing into sound surface.
        """
        import torch

        kernel = torch.zeros((1, 1, 3, 3, 3), device=self.device)
        for offset in (
            (0, 1, 1),
            (2, 1, 1),
            (1, 0, 1),
            (1, 2, 1),
            (1, 1, 0),
            (1, 1, 2),
        ):
            kernel[0, 0, offset[0], offset[1], offset[2]] = 1.0
        for _pass in range(3):
            solid = self.get_solid_cells()
            neighbor_num = torch.nn.functional.conv3d(
                solid.to(kernel.dtype)[None, None], kernel, padding=1
            )[0, 0]
            loose = solid & (neighbor_num <= 1.5)
            if not bool(loose.any()):
                return
            self.material_id = torch.where(
                loose, torch.zeros_like(self.material_id), self.material_id
            )

    def remove_loose_parts(self) -> None:
        """Keep only the largest connected solid part of the volume.

        One 6-connectivity pass over the occupancy names the parts;
        everything but the largest becomes air, since a disconnected
        part would have fallen.
        """
        import torch

        occupancy = self.get_solid_cells().cpu().numpy()
        if not occupancy.any():
            return
        labels = cc3d.connected_components(occupancy, connectivity=6)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        keep = torch.as_tensor(labels == int(sizes.argmax()), device=self.device)
        self.material_id = torch.where(
            keep, self.material_id, torch.zeros_like(self.material_id)
        )

    def gather(self, rows: Sequence[float]) -> "torch.Tensor":
        """Return one row value per cell through the material channel.

        Args:
            rows: One value per material id, air row at index 0
                included; every material id present must be covered.

        Returns:
            (X, Y, Z) float32 torch tensor on ``device``.
        """
        import torch

        table = torch.as_tensor(tuple(rows), dtype=torch.float32, device=self.device)
        return table[self.material_id.to(torch.int64)]

    def make_coarser(self, coarsen_ratio: int) -> "Volume":
        """Return a new volume with cells ``coarsen_ratio`` times wider.

        Each coarse cell covers a block of ``coarsen_ratio`` cells per
        side: it is solid when at least half the block is, its material
        is the most common solid material of the block, its optical mid
        the most common mid within that material, and its segment the
        most common solid segment. Structure marks and wear progress do
        not carry over.

        Args:
            coarsen_ratio: How many fine cells one coarse cell covers
                per side; every grid count must divide by it.
        """
        import torch

        for count in self.grid.cell_num:
            if count % coarsen_ratio:
                raise ValueError("grid cell counts must divide by the coarsen ratio")
        coarse_grid = GridSpan(
            self.grid.origin,
            self.grid.cell_size * int(coarsen_ratio),
            tuple(count // int(coarsen_ratio) for count in self.grid.cell_num),
        )
        coarse = Volume(coarse_grid, str(self.device))
        solid = self.get_solid_cells()
        solid_share = Volume._make_blocks(solid.to(torch.float32), coarsen_ratio).mean(
            dim=(1, 3, 5)
        )
        occupied = solid_share >= 0.5
        material_grid = Volume._pick_common_value(
            self.material_id, solid, coarsen_ratio
        )
        coarse.material_id = torch.where(
            occupied, material_grid, torch.zeros_like(material_grid)
        ).to(torch.uint8)
        if self.optical_mid is not None:
            winner = (
                coarse.material_id.repeat_interleave(coarsen_ratio, 0)
                .repeat_interleave(coarsen_ratio, 1)
                .repeat_interleave(coarsen_ratio, 2)
            )
            mid_grid = Volume._pick_common_value(
                self.optical_mid, solid & (self.material_id == winner), coarsen_ratio
            )
            coarse.optical_mid = mid_grid.to(torch.uint8)
        if self.segment_id is not None:
            coarse.segment_id = Volume._pick_common_segment(
                self.segment_id, solid, occupied, coarsen_ratio
            )
        return coarse

    @staticmethod
    def _make_blocks(values: "torch.Tensor", ratio: int) -> "torch.Tensor":
        """Return the values reshaped into per-coarse-cell blocks."""
        counts = values.shape
        return values.reshape(
            counts[0] // ratio,
            ratio,
            counts[1] // ratio,
            ratio,
            counts[2] // ratio,
            ratio,
        )

    @staticmethod
    def _pick_common_value(
        values: "torch.Tensor", keep: "torch.Tensor", ratio: int
    ) -> "torch.Tensor":
        """Return each block's most common kept value by counting.

        Counts one candidate value at a time, so it suits channels with
        few distinct values such as materials and optical mids.
        """
        import torch

        weight_blocks = Volume._make_blocks(keep.to(torch.float32), ratio)
        value_blocks = Volume._make_blocks(values, ratio)
        best_count = None
        best_value = None
        for value in torch.unique(values).tolist():
            count = (weight_blocks * (value_blocks == value)).sum(dim=(1, 3, 5))
            if best_count is None:
                best_count = count
                best_value = torch.full_like(count, value, dtype=torch.int64)
            else:
                is_better = count > best_count
                best_count = torch.maximum(best_count, count)
                best_value = torch.where(
                    is_better, torch.full_like(best_value, value), best_value
                )
        return best_value

    @staticmethod
    def _pick_common_segment(
        segment_id: "torch.Tensor",
        solid: "torch.Tensor",
        occupied: "torch.Tensor",
        ratio: int,
    ) -> "torch.Tensor":
        """Return each block's most common solid segment id.

        Non-solid slots take distinct negative sentinels so no air
        value can accumulate a count above one; the block mode is then
        the true solid majority whenever any segment repeats. Blocks
        where every solid segment appears once fall back to the largest
        solid segment id, and unoccupied blocks answer -1.
        """
        import torch

        counts = segment_id.shape
        block_cell_num = ratio**3
        slot_index = torch.arange(
            block_cell_num, device=segment_id.device, dtype=torch.int64
        )
        flat = (
            Volume._make_blocks(segment_id.to(torch.int64), ratio)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(counts[0] // ratio, counts[1] // ratio, counts[2] // ratio, -1)
        )
        flat_solid = (
            Volume._make_blocks(solid, ratio)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(flat.shape)
        )
        sentineled = torch.where(flat_solid, flat, -(slot_index + 2))
        mode = sentineled.mode(dim=-1).values
        fallback = torch.where(flat_solid, flat, torch.full_like(flat, -1)).amax(dim=-1)
        picked = torch.where(mode >= 0, mode, fallback)
        return torch.where(occupied, picked, torch.full_like(picked, -1)).to(
            torch.int32
        )
