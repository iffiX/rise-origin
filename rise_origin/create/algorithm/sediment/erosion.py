# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The erosion phase: material leaves the volume.

:class:`BallErosion` is one law with three weights covering three
sources exactly; :class:`DetailErosion` is our small-scale surface
carving. Structure marks steer every erosion class: seam voxels erode
with a rate scale because seams weather first, and contact voxels are
protected.
"""

import math
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

from ...protocol.rng import Rng
from ..field import CellNoise, FBMNoiseField
from .volume import Volume

if TYPE_CHECKING:
    import torch

__all__ = ["BallErosion", "DetailErosion"]


class BallErosion:
    """Surface wear under one weighted law on the ball measurement.

    The measurement is Jones 2010's bubble (section 4.1): the air share
    inside a ball around each surface voxel. The three weights make the
    law of each source exactly:

    - Bloore 1977 tumbling abrasion, carried to grids by Domokos 2012:
      ``flat_weight=1, round_weight=2b, pit_weight=0`` gives the
      paper's ``v = 1 + 2bH`` with curvature read from the ball; the
      ``cK`` term is dropped as second order for single-stone shape.
    - Jones 2010 weathering (section 4.2): ``flat_weight=0,
      round_weight=2s, pit_weight=2c`` gives the spheroidal plus
      cavernous rates; ``stretch`` is the section 4.5 skewed bubble.
    - Resting pile erosion: ``flat_weight=1, round_weight=0,
      pit_weight=0`` erodes the surface uniformly and skips the ball
      convolution entirely.

    Wear is gated by air access: a field measures how much outside air
    reaches each wall, and the rate scales with it, so a sealed pocket
    never erodes and a hairline seam stays inert past its mouth,
    widening into a funnel only as the mouth opens.

    Durability comes from per-material hardness rows swayed by value
    noise instead of a hand-edited layer graph, a named deviation. A
    second named deviation is resolution only: a ball spanning many
    cells is measured on a pooled grid and the share upsampled back,
    which keeps the stencil a ball and the cost bounded.

    Args:
        rng: The random stream the durability sway draws from.
        step_num: How many synchronized steps the erosion runs.
        ball_radius: The ball radius (unit: m); the feature size the
            measurement reads.
        wear_pace: The wear a fully exposed corner collects per step.
        flat_weight: The uniform term of the law.
        round_weight: The exposed-corner term of the law.
        pit_weight: The sheltered-pocket term of the law.
        hardness_rows: Per material id, how strongly it resists wear;
            air row at index 0 included.
        cavernous_rows: Per material id, how strongly sheltered pockets
            deepen; empty applies the pit term evenly.
        durability_sway_ratio: How strongly the value noise sways
            durability, 0 to 1.
        durability_feature_size: The feature size of the sway
            (unit: m).
        seam_rate_scale: The rate scale of seam-marked voxels; seams
            weather first.
        stretch: None keeps the ball round; a (direction, ratio) pair
            stretches it along the unit direction by the ratio for
            directional weathering.
        air_access_reach: How far outside air carries into a wide
            opening (unit: m).
        air_access_source_share: How open a spot must be to count as
            outside air.
        air_access_stop_share: The air access at or below which wear
            stops.
        air_access_full_share: The air access at or above which wear
            runs at full rate.
    """

    # A ball spanning more cells than this is measured at a pooled
    # resolution and the share upsampled back: the stencil stays a
    # ball, and the air share is a low-pass quantity, so pooling
    # changes it little while the cost drops by the pool ratio cubed.
    MAX_STENCIL_RADIUS_CELL_NUM = 6

    def __init__(
        self,
        rng: Rng,
        step_num: int,
        ball_radius: float,
        wear_pace: float,
        flat_weight: float,
        round_weight: float,
        pit_weight: float,
        hardness_rows: Sequence[float],
        cavernous_rows: Sequence[float] = (),
        durability_sway_ratio: float = 0.3,
        durability_feature_size: float = 0.15,
        seam_rate_scale: float = 4.0,
        stretch: Optional[Tuple[Tuple[float, float, float], float]] = None,
        air_access_reach: float = 0.5,
        air_access_source_share: float = 0.45,
        air_access_stop_share: float = 0.01,
        air_access_full_share: float = 0.18,
    ) -> None:
        self.rng = rng
        self.step_num = int(step_num)
        self.ball_radius = float(ball_radius)
        self.wear_pace = float(wear_pace)
        self.flat_weight = float(flat_weight)
        self.round_weight = float(round_weight)
        self.pit_weight = float(pit_weight)
        self.hardness_rows = tuple(hardness_rows)
        self.cavernous_rows = tuple(cavernous_rows)
        self.durability_sway_ratio = float(durability_sway_ratio)
        self.durability_feature_size = float(durability_feature_size)
        self.seam_rate_scale = float(seam_rate_scale)
        self.stretch = stretch
        self.air_access_reach = float(air_access_reach)
        self.air_access_source_share = float(air_access_source_share)
        self.air_access_stop_share = float(air_access_stop_share)
        self.air_access_full_share = float(air_access_full_share)

    def erode(self, volume: Volume) -> None:
        """Wear the volume for the configured steps.

        Args:
            volume: The volume material erodes from.
        """
        import torch

        durability = self._make_durability(volume)
        needs_ball = self.round_weight != 0.0 or self.pit_weight != 0.0
        pool_ratio = 1
        ball = None
        if needs_ball:
            radius_cell_num = max(
                1, int(round(self.ball_radius / volume.grid.cell_size))
            )
            pool_ratio = max(1, -(-radius_cell_num // self.MAX_STENCIL_RADIUS_CELL_NUM))
            ball = self._make_ball(volume, radius_cell_num, pool_ratio)
        cavernous = volume.gather(self.cavernous_rows) if self.cavernous_rows else None
        if volume.wear_progress is None:
            volume.wear_progress = torch.zeros(
                volume.grid.cell_num, dtype=torch.float32, device=volume.device
            )
        is_seam = None
        is_contact = None
        if volume.structure_mark is not None:
            is_seam = (volume.structure_mark & Volume.SEAM_MARK) != 0
            is_contact = (volume.structure_mark & Volume.CONTACT_MARK) != 0
        for _step in range(self.step_num):
            solid = volume.get_solid_cells()
            air = (~solid).to(torch.float32)
            near_air = torch.nn.functional.max_pool3d(
                air[None, None], kernel_size=3, stride=1, padding=1
            )[0, 0]
            wall_access = torch.nn.functional.max_pool3d(
                self._make_air_access(volume, air, pool_ratio)[None, None],
                kernel_size=3,
                stride=1,
                padding=1,
            )[0, 0]
            erodible = solid & (near_air > 0.5)
            if is_seam is not None:
                erodible = erodible | (
                    solid & is_seam & (wall_access > self.air_access_stop_share)
                )
            if needs_ball:
                air_share = self._make_air_share(air, ball, pool_ratio)
                rate = torch.clamp(
                    self.flat_weight + self.round_weight * (air_share - 0.5),
                    min=0.0,
                )
                pit = self.pit_weight * torch.clamp(0.5 - air_share, min=0.0)
                if cavernous is not None:
                    pit = pit * cavernous
                rate = rate + pit
            else:
                rate = torch.full_like(durability, self.flat_weight)
            rate = rate * self._make_air_access_gate(wall_access)
            rate = rate * self.wear_pace / durability
            if is_seam is not None:
                rate = torch.where(is_seam, rate * self.seam_rate_scale, rate)
            if is_contact is not None:
                rate = torch.where(is_contact, torch.zeros_like(rate), rate)
            volume.wear_progress = volume.wear_progress + torch.where(
                erodible, rate, torch.zeros_like(rate)
            )
            volume.material_id = torch.where(
                erodible & (volume.wear_progress >= 1.0),
                torch.zeros_like(volume.material_id),
                volume.material_id,
            )

    def _make_durability(self, volume: Volume) -> "torch.Tensor":
        """Return per-cell durability: hardness rows swayed by noise.

        The sway is layered noise, so weak and strong patches exist at
        several sizes at once and the worn silhouette turns organic
        instead of evenly rounded.
        """
        import torch

        hardness = volume.gather(self.hardness_rows)
        if self.durability_sway_ratio <= 0.0:
            return torch.clamp(hardness, min=1.0e-3)
        sway = FBMNoiseField(
            self.rng.split("durability"), self.durability_feature_size, layer_num=3
        )(volume.grid, device=str(volume.device))
        return torch.clamp(
            hardness * (1.0 + sway * self.durability_sway_ratio), min=1.0e-3
        )

    def _make_ball(
        self, volume: Volume, radius_cell_num: int, pool_ratio: int
    ) -> "torch.Tensor":
        """Return the normalized ball stencil as a conv kernel.

        Args:
            volume: The volume the stencil erodes.
            radius_cell_num: The ball radius in full-resolution cells.
            pool_ratio: The measurement pooling; the stencil is built
                in pooled cells.
        """
        import torch

        radius = max(1, int(round(radius_cell_num / pool_ratio)))
        if self.stretch is None:
            reach = radius
        else:
            reach = max(radius, int(math.ceil(radius * self.stretch[1])))
        line = torch.arange(-reach, reach + 1, device=volume.device)
        part_x = line[:, None, None].to(torch.float32)
        part_y = line[None, :, None].to(torch.float32)
        part_z = line[None, None, :].to(torch.float32)
        squared = part_x**2 + part_y**2 + part_z**2
        if self.stretch is not None:
            direction, ratio = self.stretch
            along = (
                part_x * float(direction[0])
                + part_y * float(direction[1])
                + part_z * float(direction[2])
            )
            squared = squared - (1.0 - 1.0 / float(ratio) ** 2) * along**2
        inside = (squared <= float(radius) ** 2).to(torch.float32)
        return (inside / inside.sum())[None, None]

    def _make_air_access(
        self, volume: Volume, air: "torch.Tensor", pool_ratio: int
    ) -> "torch.Tensor":
        """Return how much outside air reaches each cell, 0 to 1.

        Open air holds one and rock holds none while the value relaxes
        between them, and each sweep keeps only the air share around a
        cell, so a narrow opening starves within a short distance and a
        wide one carries the whole reach. Everything past the grid is
        open air, so the openness that names the sources reads air
        there and a body filling its grid still takes air access at its
        rim. The measurement is pooled the way the ball share is: the air share
        a cell conducts is a low pass quantity, so pooling changes it
        little while the cost drops by the pool ratio to the fourth.
        """
        import torch

        conduct = air
        if pool_ratio > 1:
            conduct = torch.nn.functional.avg_pool3d(
                air[None, None],
                kernel_size=pool_ratio,
                stride=pool_ratio,
                ceil_mode=True,
                count_include_pad=False,
            )[0, 0]
        openness = torch.nn.functional.avg_pool3d(
            torch.nn.functional.pad(conduct[None, None], (3,) * 6, value=1.0),
            kernel_size=7,
            stride=1,
        )[0, 0]
        source = ((conduct > 0.5) & (openness > self.air_access_source_share)).to(
            torch.float32
        )
        sweep_num = max(
            1,
            int(round(self.air_access_reach / (volume.grid.cell_size * pool_ratio))),
        )
        access = source
        for _sweep in range(sweep_num):
            access = torch.nn.functional.avg_pool3d(
                access[None, None], kernel_size=3, stride=1, padding=1
            )[0, 0]
            access = torch.maximum(access * conduct, source)
        if pool_ratio == 1:
            return access
        return torch.nn.functional.interpolate(
            access[None, None],
            size=air.shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]

    def _make_air_access_gate(self, wall_access: "torch.Tensor") -> "torch.Tensor":
        """Return the wear share the wall air access allows, 0 to 1."""
        import torch

        share = torch.clamp(
            (wall_access - self.air_access_stop_share)
            / max(self.air_access_full_share - self.air_access_stop_share, 1.0e-9),
            0.0,
            1.0,
        )
        return share * share * (3.0 - 2.0 * share)

    @staticmethod
    def _make_air_share(
        air: "torch.Tensor", ball: "torch.Tensor", pool_ratio: int
    ) -> "torch.Tensor":
        """Return the ball air share, measured pooled when the ball is big.

        Args:
            air: (X, Y, Z) float32 air indicator.
            ball: The normalized ball stencil in pooled cells.
            pool_ratio: The measurement pooling; 1 measures at full
                resolution.
        """
        import torch

        padding = (ball.shape[-3] - 1) // 2
        if pool_ratio == 1:
            return torch.nn.functional.conv3d(air[None, None], ball, padding=padding)[
                0, 0
            ]
        pooled = torch.nn.functional.avg_pool3d(
            air[None, None],
            kernel_size=pool_ratio,
            stride=pool_ratio,
            ceil_mode=True,
            count_include_pad=False,
        )
        share = torch.nn.functional.conv3d(pooled, ball, padding=padding)
        return torch.nn.functional.interpolate(
            share,
            size=air.shape,
            mode="trilinear",
            align_corners=False,
        )[0, 0]


class DetailErosion:
    """Small dents bitten from the surface, a rugged look without holes.

    A cell pattern scatters dent seeds in world space, so the same
    stream dents the same places at every resolution and a dent finer
    than one cell skips cleanly. A hashed share of the seeds is active,
    and each active seed near the surface bites a small sphere: the
    sphere meets the surface shell as a round patch wider than it is
    deep, so a dent reads one cell deep and a couple cells wide, never
    a drilled pit. Two guards keep the body sound. A cell only leaves
    within ``depth`` of air, so a bite peels the outer shell only. And
    a cell only leaves where the body is locally thick, judged by a
    morphological opening, so thin walls and rims are never carved and
    no dent can open a hole. A final sweep drops cells left with one or
    no solid neighbor, so no loose specks remain to fall off.

    Args:
        rng: The random stream the dent seeds draw from.
        depth: How deep a dent reaches (unit: m); finer than one cell
            skips cleanly.
        dent_spacing: The spacing of the dent seed pattern (unit: m).
        dent_width: How wide a dent reads on the surface (unit: m).
        dent_share: The share of seeds that bite, 0 none to 1 all.
    """

    # The opening that judges local thickness reaches this many cells
    # past the dent depth, so a peel can never meet a peel from the
    # other side of a wall.
    THICKNESS_MARGIN_CELL_NUM = 2

    def __init__(
        self,
        rng: Rng,
        depth: float,
        dent_spacing: float,
        dent_width: float,
        dent_share: float = 0.5,
    ) -> None:
        self.rng = rng
        self.depth = float(depth)
        self.dent_spacing = float(dent_spacing)
        self.dent_width = float(dent_width)
        self.dent_share = float(dent_share)

    def erode(self, volume: Volume) -> None:
        """Bite the dents into the volume surface.

        Args:
            volume: The volume material erodes from.
        """
        import torch

        cell_size = volume.grid.cell_size
        if self.depth < cell_size:
            return
        depth_num = int(math.ceil(self.depth / cell_size))
        gate_num = depth_num + self.THICKNESS_MARGIN_CELL_NUM
        solid = volume.get_solid_cells()
        air = ~solid
        shell = DetailErosion._grow(air, depth_num) & solid
        core = ~DetailErosion._grow(air, gate_num)
        thick = DetailErosion._grow(core, gate_num)
        candidate = torch.nonzero(shell & thick)
        if candidate.shape[0] == 0:
            return
        axes = volume.grid.make_axes(str(volume.device))
        points = torch.stack(
            [axes[side][candidate[:, side]].to(torch.float32) for side in range(3)],
            dim=1,
        )
        sample = CellNoise(self.rng.split("dent"), self.dent_spacing)(
            points, device=str(volume.device)
        )
        salt = int(self.rng.split("share").integers(0, 2**31))
        is_active = DetailErosion._make_hash01(sample.cell_id, salt) < self.dent_share
        half_width = self.dent_width / 2.0
        bite_radius = (half_width * half_width + self.depth * self.depth) / (
            2.0 * self.depth
        )
        bite = is_active & (sample.nearest_distance * self.dent_spacing < bite_radius)
        hit = candidate[bite]
        volume.material_id[hit[:, 0], hit[:, 1], hit[:, 2]] = 0
        volume.remove_debris()

    @staticmethod
    def _grow(mask: "torch.Tensor", step_num: int) -> "torch.Tensor":
        """Return the bool mask dilated by the step count.

        Args:
            mask: The (X, Y, Z) bool tensor to dilate.
            step_num: How many one-cell dilations to apply.
        """
        import torch

        grown = mask
        for _ in range(step_num):
            grown = (
                torch.nn.functional.max_pool3d(
                    grown.to(torch.float32)[None, None],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )[0, 0]
                > 0.5
            )
        return grown

    @staticmethod
    def _make_hash01(cell_id: "torch.Tensor", salt: int) -> "torch.Tensor":
        """Return a per-cell value in [0, 1) hashed from the cell id.

        Args:
            cell_id: The int64 cell ids to hash.
            salt: The stream-drawn salt mixed into the hash.
        """
        import torch

        mask = 0xFFFFFFFF
        mixed = ((cell_id & mask) * 2654435761) & mask
        mixed = mixed ^ salt
        mixed = mixed ^ (mixed >> 13)
        mixed = (mixed * 0x5BD1E995) & mask
        mixed = mixed ^ (mixed >> 15)
        return (mixed & mask).to(torch.float32) / float(2**32)
