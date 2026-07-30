# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The generator phase: classes that produce the initial volume.

Every generator takes all its arguments through ``__init__`` and its
``generate()`` takes none and returns a new :class:`Volume`.
:class:`UniformGenerator` fills one material into a generated shape;
its subclasses swap the material fill for a spatial layout of matter.
Each layout is its own class because they share nothing beneath the
surface: :class:`StrataGenerator` bands material by height (layered
sedimentary rock), :class:`GrainGenerator` colors isotropic crystals
(salt-and-pepper igneous rock), and :class:`VeinGenerator` threads
filaments through a body (veined metamorphic rock). Nothing here
simulates deposition; we generate convincing structure as a pointwise
function of position controlled by seeded noise.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from ...protocol.rng import Rng
from ..grid_span import GridSpan
from ..field import (
    CellNoise,
    FBMNoiseField,
    RidgedNoiseField,
    TurbulenceNoiseField,
    ValueNoiseField,
)
from .volume import Volume

if TYPE_CHECKING:
    import torch

__all__ = [
    "UniformGenerator",
    "StrataGeneratorLayer",
    "StrataGenerator",
    "GrainGenerator",
    "VeinGenerator",
    "ColumnGenerator",
]


class UniformGenerator:
    """Generates a volume of one material in a chosen shape.

    The ridge shape is the TerraForged terrain-module recipe, read from
    its code and cited to the module (github.com/TerraForged, terrain
    noise presets): the Voronoi cell-edge closeness field is the ridgeline
    mask, domain-warped so ranges meander, multiplied by ridged noise, and
    shaped by a power profile with optional terrace steps.

    Args:
        grid: The evaluation grid the volume is generated on.
        device: The torch device name the volume lives on.
        rng: The random stream every seeded field draws from.
        material_id: The material every solid cell starts as.
        shape: The generated shape: ``box`` (a plain block, meant to be
            fractured later), ``ridge`` (the TerraForged recipe),
            ``height`` (solid below a supplied height grid), or
            ``density`` (solid where a supplied density tensor is
            positive).
        box_size: The box edge lengths, centered in the grid, for the
            box shape (unit: m).
        box_occupancy: (nx, ny, nz) bool numpy array of presence cells
            stretched over the box shape, each dimension divided
            independently; None keeps the full box. The stretch is
            smooth and the threshold is noise-jittered, so mask edges
            come out rounded and slightly ragged at mask-cell scale.
        height_grid: (X, Y) float numpy array or torch tensor of
            surface heights for the height shape (unit: m).
        density_grid: (X, Y, Z) float32 torch tensor for the density
            shape; solid where positive.
        base_height: The ridge shape's valley floor height above the
            grid origin (unit: m).
        ridge_height: The ridge shape's crest height above the valley
            floor (unit: m).
        ridge_feature_size: The spacing of the ridge lines (unit: m).
        warp_size: How far the ridge lines wander sideways (unit: m).
        warp_feature_size: The feature size of the wander (unit: m).
        ridged_layer_num: How many ridged noise layers stack onto the
            ridge lines.
        ridged_gain: The weight each finer ridged layer scales by.
        profile_power: The exponent steepening the ridge profile.
        terrace_step_num: How many terrace steps quantize the ridge
            profile; 0 keeps it smooth.
        terrace_slope_share: How much of the raw profile leaks back
            into each terrace step, 0 flat treads to 1 no steps.

    Attributes:
        grid: The evaluation grid.
        device: The torch device name.
        rng: The random stream.
        material_id: The uniform fill material.
        shape: The generated shape name.
    """

    # How far the occupancy threshold jitters, in presence units where
    # one mask cell spans about one unit of presence slope.
    OCCUPANCY_EDGE_SWING = 0.3

    _SHAPES = ("box", "ridge", "height", "density")

    def __init__(
        self,
        grid: GridSpan,
        device: str,
        rng: Rng,
        material_id: int,
        shape: str,
        box_size: Optional[Tuple[float, float, float]] = None,
        box_occupancy: Optional[np.ndarray] = None,
        height_grid: Optional[np.ndarray] = None,
        density_grid: Optional["torch.Tensor"] = None,
        base_height: float = 0.0,
        ridge_height: float = 1.0,
        ridge_feature_size: float = 100.0,
        warp_size: float = 20.0,
        warp_feature_size: float = 80.0,
        ridged_layer_num: int = 4,
        ridged_gain: float = 0.975,
        profile_power: float = 1.1,
        terrace_step_num: int = 0,
        terrace_slope_share: float = 0.35,
    ) -> None:
        if shape not in self._SHAPES:
            raise ValueError(f"unknown shape {shape!r}; pick one of {self._SHAPES}")
        self.grid = grid
        self.device = device
        self.rng = rng
        self.material_id = int(material_id)
        self.shape = shape
        self.box_size = box_size
        self.box_occupancy = box_occupancy
        self.height_grid = height_grid
        self.density_grid = density_grid
        self.base_height = float(base_height)
        self.ridge_height = float(ridge_height)
        self.ridge_feature_size = float(ridge_feature_size)
        self.warp_size = float(warp_size)
        self.warp_feature_size = float(warp_feature_size)
        self.ridged_layer_num = int(ridged_layer_num)
        self.ridged_gain = float(ridged_gain)
        self.profile_power = float(profile_power)
        self.terrace_step_num = int(terrace_step_num)
        self.terrace_slope_share = float(terrace_slope_share)

    def generate(self) -> Volume:
        """Return the generated volume."""
        import torch

        solid = self._make_solid_cells()
        volume = Volume(self.grid, self.device)
        volume.material_id = torch.where(
            solid,
            torch.full_like(volume.material_id, self.material_id),
            volume.material_id,
        )
        self._fill_materials(volume, solid)
        return volume

    def _fill_materials(self, volume: Volume, solid: "torch.Tensor") -> None:
        """Write the material composition; the subclass hook.

        The uniform fill already happened in :meth:`generate`, so the
        base class writes nothing more.

        Args:
            volume: The volume whose channels fill.
            solid: (X, Y, Z) bool torch tensor of the generated shape.
        """

    def _make_solid_cells(self) -> "torch.Tensor":
        """Return the (X, Y, Z) bool torch tensor of the shape."""
        import torch

        if self.shape == "density":
            return self.density_grid.to(device=self.device) > 0.0
        if self.shape == "box":
            axes = self.grid.make_axes(self.device)
            lo, hi = self.grid.get_extent()
            center = (lo + hi) / 2.0
            inside = []
            for side in range(3):
                line = axes[side].to(torch.float32)
                half = float(self.box_size[side]) / 2.0
                inside.append(torch.abs(line - float(center[side])) <= half)
            box = (
                inside[0][:, None, None]
                & inside[1][None, :, None]
                & inside[2][None, None, :]
            )
            if self.box_occupancy is None:
                return box
            return box & self._make_box_presence()
        if self.shape == "height":
            heights = torch.as_tensor(
                self.height_grid, dtype=torch.float32, device=self.device
            )
        else:
            heights = self._make_ridge_heights()
        axis_z = self.grid.make_axes(self.device)[2].to(torch.float32)
        return axis_z[None, None, :] < heights[:, :, None]

    def _make_box_presence(self) -> "torch.Tensor":
        """Return the (X, Y, Z) bool torch tensor of the stretched mask.

        The presence cells stretch over the box with smooth
        interpolation and the half threshold is jittered by value noise
        with a feature of about one mask cell, so mask edges round and
        roughen instead of following the cell walls.
        """
        import torch
        import torch.nn.functional as functional

        mask = torch.as_tensor(
            np.asarray(self.box_occupancy, dtype=np.float32), device=self.device
        )
        axes = self.grid.make_axes(self.device)
        lo, hi = self.grid.get_extent()
        center = (lo + hi) / 2.0
        reaches = []
        for side in range(3):
            line = axes[side].to(torch.float32)
            half = float(self.box_size[side]) / 2.0
            low = float(center[side]) - half
            share = (line - low) / max(2.0 * half, 1.0e-9)
            reaches.append(share * 2.0 - 1.0)
        cell_num = self.grid.cell_num
        sample = torch.empty(
            (1, cell_num[0], cell_num[1], cell_num[2], 3),
            dtype=torch.float32,
            device=self.device,
        )
        sample[..., 0] = reaches[2][None, None, :]
        sample[..., 1] = reaches[1][None, :, None]
        sample[..., 2] = reaches[0][:, None, None]
        presence = functional.grid_sample(
            mask[None, None],
            sample,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )[0, 0]
        feature = (
            sum(float(self.box_size[side]) / int(mask.shape[side]) for side in range(3))
            / 3.0
        )
        jitter = ValueNoiseField(self.rng.split("occupancy_edge"), feature)(
            self.grid, device=self.device
        )
        return presence + self.OCCUPANCY_EDGE_SWING * jitter > 0.5

    def _make_ridge_heights(self) -> "torch.Tensor":
        """Return the ridge shape's (X, Y) surface heights (unit: m)."""
        import torch

        axes = self.grid.make_axes(self.device)
        axis_x = axes[0].to(torch.float32)[:, None]
        axis_y = axes[1].to(torch.float32)[None, :]
        flat_num = self.grid.cell_num[0] * self.grid.cell_num[1]
        warp_x = FBMNoiseField(self.rng.split("ridge/warp_x"), self.warp_feature_size)
        warp_y = FBMNoiseField(self.rng.split("ridge/warp_y"), self.warp_feature_size)
        points = torch.zeros((flat_num, 3), dtype=torch.float32, device=self.device)
        points[:, 0] = axis_x.expand(-1, self.grid.cell_num[1]).reshape(-1)
        points[:, 1] = axis_y.expand(self.grid.cell_num[0], -1).reshape(-1)
        warped = points.clone()
        warped[:, 0] = (
            points[:, 0] + warp_x(points, device=self.device) * self.warp_size
        )
        warped[:, 1] = (
            points[:, 1] + warp_y(points, device=self.device) * self.warp_size
        )
        edge_sample = CellNoise(self.rng.split("ridge/cells"), self.ridge_feature_size)(
            warped, device=self.device
        )
        edge_closeness = torch.clamp(
            1.2 * (1.0 - (edge_sample.second_distance - edge_sample.nearest_distance)),
            0.0,
            1.0,
        )
        ridged = RidgedNoiseField(
            self.rng.split("ridge/ridged"),
            self.ridge_feature_size / 2.0,
            layer_num=self.ridged_layer_num,
            gain=self.ridged_gain,
        )(points, device=self.device)
        profile = torch.clamp(edge_closeness * ridged, 0.0, 1.0) ** self.profile_power
        if self.terrace_step_num > 0:
            stepped = torch.round(profile * self.terrace_step_num) / float(
                self.terrace_step_num
            )
            profile = stepped + (profile - stepped) * self.terrace_slope_share
        heights = self.base_height + self.ridge_height * profile
        return heights.reshape(self.grid.cell_num[0], self.grid.cell_num[1])


@dataclass(frozen=True)
class StrataGeneratorLayer:
    """One sedimentary layer; used only by :class:`StrataGenerator`.

    A layer's tone drifts smoothly through its optical mid run so
    neighboring cells stay tonal neighbors, the way sedimentary grains
    vary. The sub-pattern fields add finer internal structure, all of
    it layering-flavored: tilted laminae, and a tone grade across the
    layer.

    Attributes:
        material_id: The layer's material.
        thickness: The layer's thickness (unit: m).
        optical_mids: The layer material's tone run, light end to dark.
        tone_bias: Where in the run the layer centers, 0 lightest to 1
            darkest.
        tone_spread: How far the tone drifts around the bias, as a
            share of the run.
        feature_size: The feature size of the tone drift (unit: m).
        warp_size: How far the layer boundary wanders (unit: m).
        warp_feature_size: The feature size of the wander (unit: m).
        lamination_feature_size: The spacing of internal laminae
            (unit: m); 0 leaves the layer unlaminated.
        lamination_strength: How strongly the laminae shift the tone,
            as a share of the run.
        lamination_angle: The tilt of the laminae (unit: radians); 0
            lays them flat, larger tilts them into cross-layers whose
            direction sweeps across the stone.
        grade_strength: A tone ramp from the layer bottom to its top, as
            a share of the run; 0 leaves the layer ungraded.
    """

    material_id: int
    thickness: float
    optical_mids: Tuple[int, ...] = ()
    tone_bias: float = 0.5
    tone_spread: float = 0.25
    feature_size: float = 0.08
    warp_size: float = 0.0
    warp_feature_size: float = 0.5
    lamination_feature_size: float = 0.0
    lamination_strength: float = 0.0
    lamination_angle: float = 0.0
    grade_strength: float = 0.0


class StrataGenerator(UniformGenerator):
    """Bands material by height inside any :class:`UniformGenerator` shape.

    The layers are implicit: boundary fields accumulate bottom-up and
    every cell reads the layer whose band contains its fold-warped
    height, so dipping and folded strata come from warping the read
    coordinate. Within a layer the tone drifts, laminae tilt, and a
    grade ramps, all as pointwise functions of position.

    Args:
        grid: The evaluation grid the volume is generated on.
        device: The torch device name the volume lives on.
        rng: The random stream every seeded field draws from.
        layers: The layers, bottom to top; the last layer extends to the
            grid top.
        shape: The generated shape, as in :class:`UniformGenerator`.
        fold_size: How far the strata bend vertically (unit: m); 0
            keeps them flat. The fold is a coherent vertical shift per
            column, so it waves the whole stack without adding noise
            within a layer.
        fold_feature_size: The horizontal wavelength of the bending
            (unit: m); smaller than the stone gives several visible
            crests, larger reads as a single tilt.
        layer_slope: The (x, y) rise of the layers per meter from the grid
            center; (0, 0) lays them level, larger tilts the whole
            stack.
        **shape_arguments: The shape arguments of
            :class:`UniformGenerator`.
    """

    def __init__(
        self,
        grid: GridSpan,
        device: str,
        rng: Rng,
        layers: Tuple[StrataGeneratorLayer, ...],
        shape: str,
        fold_size: float = 0.0,
        fold_feature_size: float = 8.0,
        layer_slope: Tuple[float, float] = (0.0, 0.0),
        **shape_arguments,
    ) -> None:
        if not layers:
            raise ValueError("strata need at least one layer")
        super().__init__(
            grid,
            device,
            rng,
            material_id=layers[0].material_id,
            shape=shape,
            **shape_arguments,
        )
        self.layers = tuple(layers)
        self.fold_size = float(fold_size)
        self.fold_feature_size = float(fold_feature_size)
        self.layer_slope = (float(layer_slope[0]), float(layer_slope[1]))

    @staticmethod
    def make_layer_slope(direction: Sequence[float]) -> Tuple[float, float]:
        """Return the (x, y) rise per meter a direction vector traces.

        The horizontal part aims the rise and the vertical part sets
        the steepness; a straight-up direction gives no slope.

        Args:
            direction: The (x, y, z) direction the layers rise along.
        """
        reach_x, reach_y, rise = direction
        reach_length_sq = reach_x * reach_x + reach_y * reach_y
        if reach_length_sq < 1.0e-9:
            return (0.0, 0.0)
        return (rise * reach_x / reach_length_sq, rise * reach_y / reach_length_sq)

    @staticmethod
    def make_layer_thicknesses(
        rng: Rng,
        total_thickness: float,
        mean_thickness: float,
        min_thickness: float,
        mean_layers_per_package: float = 2.5,
        package_sample_num: int = 3,
    ) -> Tuple[float, ...]:
        """Return layer thicknesses following a two-scale deposit rhythm.

        Real strata vary at two geological scales: several depositional
        episodes span the rock, and layers within an episode may be
        compressed or dominant. Episodes take Gamma-distributed
        durations and their layers take exponential shares, so the run of
        thicknesses reads as natural laminae through dominant layers
        rather than a regular stack. Thicknesses sum to
        ``total_thickness`` and none fall below ``min_thickness``.

        Args:
            rng: The random stream the durations draw from.
            total_thickness: The total thickness to fill (unit: m).
            mean_thickness: The nominal layer thickness (unit: m).
            min_thickness: The floor a layer keeps after voxelization
                (unit: m).
            mean_layers_per_package: The mean layers one episode holds.
            package_sample_num: How many exponentials each episode
                duration sums, higher makes episodes more even.
        """
        layer_num = max(1, int(round(total_thickness / mean_thickness)))
        floor = min(min_thickness, 0.5 * total_thickness / layer_num)
        distributed = total_thickness - layer_num * floor
        package_num = max(1, int(round(layer_num / mean_layers_per_package)))
        package_size, large_package_num = divmod(layer_num, package_num)
        package_sizes = [
            package_size + int(index < large_package_num)
            for index in range(package_num)
        ]
        package_rng = rng.split("packages")
        package_ratios = [
            sum(
                -math.log1p(-package_rng.split(f"{index}/{sample}").uniform(0.0, 1.0))
                for sample in range(package_sample_num)
            )
            for index in range(package_num)
        ]
        package_ratio_sum = sum(package_ratios)
        thicknesses = []
        for index, size in enumerate(package_sizes):
            region_rng = rng.split(f"regions/{index}")
            region_ratios = [
                -math.log1p(-region_rng.split(str(layer)).uniform(0.0, 1.0))
                for layer in range(size)
            ]
            region_ratio_sum = sum(region_ratios) or 1.0
            package_thickness = distributed * package_ratios[index] / package_ratio_sum
            thicknesses.extend(
                floor + package_thickness * ratio / region_ratio_sum
                for ratio in region_ratios
            )
        return tuple(thicknesses)

    def _fill_materials(self, volume: Volume, solid: "torch.Tensor") -> None:
        """Write the layer bands and their tone structure.

        Args:
            volume: The volume whose channels fill.
            solid: (X, Y, Z) bool torch tensor of the generated shape.
        """
        import torch

        height, boundaries = self._make_heights_and_boundaries(volume)
        band = torch.zeros(self.grid.cell_num, dtype=torch.int64, device=volume.device)
        for top in boundaries:
            band = band + (height >= top[:, :, None]).to(torch.int64)
        material_table = torch.as_tensor(
            tuple(layer.material_id for layer in self.layers),
            dtype=torch.uint8,
            device=volume.device,
        )
        volume.material_id = torch.where(
            solid, material_table[band], torch.zeros_like(volume.material_id)
        )
        bottom = float(self.grid.origin[2])
        optical_mid = torch.zeros_like(volume.material_id)
        for index, layer in enumerate(self.layers):
            inside = solid & (band == index)
            if not bool(inside.any()):
                continue
            layer_bottom = (
                boundaries[index - 1][:, :, None]
                if index > 0
                else torch.full_like(height[:, :, :1], bottom)
            )
            layer_top = (
                boundaries[index][:, :, None]
                if index < len(boundaries)
                else torch.full_like(height[:, :, :1], height.max() + 1.0)
            )
            picks = self._make_layer_mids(
                layer,
                self.rng.split(f"strata/{index}"),
                volume,
                height,
                layer_bottom,
                layer_top,
            )
            optical_mid = torch.where(inside, picks, optical_mid)
        volume.optical_mid = optical_mid

    def _make_heights_and_boundaries(self, volume: Volume):
        """Return the fold-warped height field and the layer boundaries."""
        import torch

        flat_grid = GridSpan(
            self.grid.origin,
            self.grid.cell_size,
            (self.grid.cell_num[0], self.grid.cell_num[1], 1),
        )
        bottom = float(self.grid.origin[2])
        running = None
        boundaries = []
        for index, layer in enumerate(self.layers[:-1]):
            stream = self.rng.split(f"strata/{index}")
            top = (running if running is not None else bottom) + layer.thickness
            if layer.warp_size > 0.0:
                wander = TurbulenceNoiseField(
                    stream.split("boundary"), layer.warp_feature_size
                )(flat_grid, device=self.device)[:, :, 0]
                top = top + (wander - 0.5) * (2.0 * layer.warp_size)
            elif running is None:
                top = torch.full(
                    (self.grid.cell_num[0], self.grid.cell_num[1]),
                    top,
                    dtype=torch.float32,
                    device=volume.device,
                )
            boundaries.append(top)
            running = top
        axes = self.grid.make_axes(self.device)
        axis_z = axes[2].to(torch.float32)
        height = axis_z[None, None, :].expand(
            self.grid.cell_num[0], self.grid.cell_num[1], -1
        )
        if self.layer_slope != (0.0, 0.0):
            lo, hi = self.grid.get_extent()
            axis_x = axes[0].to(torch.float32)
            axis_y = axes[1].to(torch.float32)
            tilt = (axis_x - float(lo[0] + hi[0]) / 2.0)[:, None] * self.layer_slope[
                0
            ] + (axis_y - float(lo[1] + hi[1]) / 2.0)[None, :] * self.layer_slope[1]
            height = height + tilt[:, :, None]
        if self.fold_size > 0.0:
            fold = FBMNoiseField(self.rng.split("strata/fold"), self.fold_feature_size)(
                flat_grid, device=self.device
            )[:, :, 0]
            height = height + fold[:, :, None] * self.fold_size
        return height, boundaries

    def _make_layer_mids(
        self,
        layer: StrataGeneratorLayer,
        stream: Rng,
        volume: Volume,
        height: "torch.Tensor",
        layer_bottom: "torch.Tensor",
        layer_top: "torch.Tensor",
    ) -> "torch.Tensor":
        """Return one layer's optical mid picks over the whole grid."""
        import torch

        run_num = len(layer.optical_mids)
        if run_num == 0:
            return torch.zeros_like(volume.material_id)
        top_index = run_num - 1
        drift = ValueNoiseField(stream.split("drift"), layer.feature_size)(
            self.grid, device=self.device
        )
        reach = (layer.tone_bias + drift * layer.tone_spread) * top_index
        if layer.grade_strength != 0.0:
            span = torch.clamp(layer_top - layer_bottom, min=1.0e-4)
            grade = torch.clamp((height - layer_bottom) / span, 0.0, 1.0)
            reach = reach + layer.grade_strength * (grade - 0.5) * top_index
        if layer.lamination_strength > 0.0 and layer.lamination_feature_size > 0.0:
            reach = reach + layer.lamination_strength * top_index * self._make_laminae(
                layer, stream, height
            )
        picks = torch.clamp(torch.round(reach).to(torch.int64), 0, top_index)
        table = torch.as_tensor(
            layer.optical_mids, dtype=torch.uint8, device=volume.device
        )
        return table[picks]

    def _make_laminae(
        self, layer: StrataGeneratorLayer, stream: Rng, height: "torch.Tensor"
    ) -> "torch.Tensor":
        """Return the tilted laminae oscillation in [-1, 1].

        The laminae run along an axis tilted by ``lamination_angle``;
        the tilt direction sweeps slowly across the stone through a
        coarse noise, so flat laminae read as pinstripes and tilted
        ones read as curving cross-layers.
        """
        import torch

        axes = self.grid.make_axes(self.device)
        axis_x = axes[0].to(torch.float32)[:, None, None]
        axis_y = axes[1].to(torch.float32)[None, :, None]
        if layer.lamination_angle > 0.0:
            flat_grid = GridSpan(
                self.grid.origin,
                self.grid.cell_size,
                (self.grid.cell_num[0], self.grid.cell_num[1], 1),
            )
            direction = (
                ValueNoiseField(
                    stream.split("cross_direction"),
                    max(self.grid.cell_size * 8.0, 8.0 * layer.lamination_feature_size),
                )(flat_grid, device=self.device)
                * math.pi
            )
            lateral = axis_x * torch.cos(direction) + axis_y * torch.sin(direction)
            along = height * math.cos(layer.lamination_angle) + lateral * math.sin(
                layer.lamination_angle
            )
        else:
            along = height
        phase = TurbulenceNoiseField(
            stream.split("lamina_phase"), 6.0 * layer.lamination_feature_size
        )(self.grid, device=self.device)
        return torch.sin(
            2.0 * math.pi * along / layer.lamination_feature_size + 3.0 * phase
        )


class GrainGenerator(UniformGenerator):
    """Colors isotropic crystals inside any :class:`UniformGenerator` shape.

    One material fills the shape; each Voronoi crystal picks a mineral,
    so the surface reads as speckled mineral grains. A coarse domain
    field biases which mineral each region draws, so the same mineral
    clusters into patches the way it does in real igneous rock instead
    of scattering as even confetti. Every mineral is a group of a few
    slight shades, and a fine dither noise picks the shade per cell, so
    a crystal patch varies subtly rather than reading as one flat
    color.

    Args:
        grid: The evaluation grid the volume is generated on.
        device: The torch device name the volume lives on.
        rng: The random stream every seeded field draws from.
        material_id: The single rigid material of the rock.
        grain_mids: The minerals as groups of shade mids, in the mix
            proportion the caller wants (repeat a group to weight it).
        grain_feature_size: The crystal size (unit: m).
        cluster_feature_size: The size of the mineral patches
            (unit: m); larger than the crystal so several crystals
            cluster together.
        cluster_strength: How strongly a region biases toward one
            mineral, 0 even confetti to 1 hard patches.
        dither_feature_size: The size of the within-mineral shade
            dither (unit: m); around one voxel reads as fine grain.
        shape: The generated shape, as in :class:`UniformGenerator`.
        **shape_arguments: The shape arguments of
            :class:`UniformGenerator`.
    """

    def __init__(
        self,
        grid: GridSpan,
        device: str,
        rng: Rng,
        material_id: int,
        grain_mids: Sequence[Sequence[int]],
        grain_feature_size: float,
        cluster_feature_size: float,
        cluster_strength: float,
        dither_feature_size: float,
        shape: str,
        **shape_arguments,
    ) -> None:
        super().__init__(
            grid, device, rng, material_id=material_id, shape=shape, **shape_arguments
        )
        self.grain_mids = tuple(tuple(group) for group in grain_mids)
        self.grain_feature_size = float(grain_feature_size)
        self.cluster_feature_size = float(cluster_feature_size)
        self.cluster_strength = float(cluster_strength)
        self.dither_feature_size = float(dither_feature_size)

    def _fill_materials(self, volume: Volume, solid: "torch.Tensor") -> None:
        """Write the clustered, dithered crystal tones over the shape.

        Args:
            volume: The volume whose channels fill.
            solid: (X, Y, Z) bool torch tensor of the generated shape.
        """
        import torch

        group_num = len(self.grain_mids)
        crystal = CellNoise(self.rng.split("grain/crystal"), self.grain_feature_size)(
            self.grid, device=self.device
        )
        crystal_fraction = (crystal.cell_id % 4096).to(torch.float32) / 4096.0
        domain = CellNoise(self.rng.split("grain/cluster"), self.cluster_feature_size)(
            self.grid, device=self.device
        )
        domain_fraction = (domain.cell_id % 4096).to(torch.float32) / 4096.0
        reach = (
            crystal_fraction * (1.0 - self.cluster_strength)
            + domain_fraction * self.cluster_strength
        ) * group_num
        group = torch.clamp(reach.to(torch.int64), 0, group_num - 1)
        flat = []
        group_start = []
        group_len = []
        for shades in self.grain_mids:
            group_start.append(len(flat))
            group_len.append(len(shades))
            flat.extend(shades)
        flat_table = torch.as_tensor(flat, dtype=torch.uint8, device=volume.device)
        start_table = torch.as_tensor(
            group_start, dtype=torch.int64, device=volume.device
        )
        len_table = torch.as_tensor(group_len, dtype=torch.int64, device=volume.device)
        dither = ValueNoiseField(
            self.rng.split("grain/dither"), self.dither_feature_size
        )(self.grid, device=self.device)
        shade = torch.clamp(
            torch.round(
                (dither * 0.5 + 0.5) * (len_table[group] - 1).to(torch.float32)
            ),
            torch.zeros((), device=volume.device),
            (len_table[group] - 1).to(torch.float32),
        ).to(torch.int64)
        picks = flat_table[start_table[group] + shade]
        volume.optical_mid = torch.where(
            solid, picks, torch.zeros_like(volume.material_id)
        )


class VeinGenerator(UniformGenerator):
    """Threads a branching vein network through a body inside any shape.

    One material fills the shape with one body tone, the way a quarry
    block carries one colour. The veins are the walls of a cellular
    pattern:
    where two feature points sit equally near, a cell boundary runs, so
    thresholding that boundary distance draws the connected branching web
    that a marble crack network makes, not the rounded blobs a ridge
    would. Every wall of the pattern would read as a complete cell
    division, so a random share of the walls is dropped, hashed by the
    pair of cells they part, and the veins break into partial paths. The
    network widens where three cells meet, so the veins thicken at their
    junctions the way real ones do, and a coherent noise varies their
    width so no run stays a single voxel line for long. A wide cell size
    keeps the network sparse and its veins thin, and a coarser one lays
    the few bold veins that run thicker and darker; a domain warp winds
    the veins off the straight cell walls.

    Args:
        grid: The evaluation grid the volume is generated on.
        device: The torch device name the volume lives on.
        rng: The random stream every seeded field draws from.
        material_id: The single rigid material of the rock.
        body_mid: The one tone the whole body takes.
        vein_mids: The vein tones, the thin network first then the tone
            the bold veins take.
        vein_feature_size: The spacing of the vein network (unit: m).
        vein_share: How wide the network veins are, 0 hairline to 1
            broad.
        bold_vein_share: How wide the few bold veins are, 0 for none.
        vein_path_share: What share of the network walls stay veins, 0
            to 1; below 1 the complete cell division breaks into partial
            paths.
        bold_vein_path_share: What share of the bold walls stay veins, 0 to
            1; None matches ``vein_path_share``, lower breaks the bold
            veins more than the network.
        winding: How far the veins wander off the cell walls, 0 straight
            to 1 sinuous.
        shape: The generated shape, as in :class:`UniformGenerator`.
        **shape_arguments: The shape arguments of
            :class:`UniformGenerator`.
    """

    # The bold veins sit on a cell pattern this much coarser than the
    # network, so a few long thick veins cross the fine thin web.
    BOLD_VEIN_FEATURE_RATIO = 1.6

    # A junction where the third cell comes within this boundary distance
    # widens the vein by up to this share of its width, and a coherent
    # noise sways the width by this share so runs are not uniform.
    JUNCTION_REACH = 0.15
    JOINT_WIDTH_BOOST = 1.5
    WIDTH_VARY_RATIO = 0.4

    def __init__(
        self,
        grid: GridSpan,
        device: str,
        rng: Rng,
        material_id: int,
        body_mid: int,
        vein_mids: Sequence[int],
        vein_feature_size: float,
        vein_share: float,
        bold_vein_share: float = 0.0,
        vein_path_share: float = 1.0,
        bold_vein_path_share: Optional[float] = None,
        winding: float = 0.5,
        shape: str = "box",
        **shape_arguments,
    ) -> None:
        super().__init__(
            grid, device, rng, material_id=material_id, shape=shape, **shape_arguments
        )
        self.body_mid = int(body_mid)
        self.vein_mids = tuple(vein_mids)
        self.vein_feature_size = float(vein_feature_size)
        self.vein_share = float(vein_share)
        self.bold_vein_share = float(bold_vein_share)
        self.vein_path_share = float(vein_path_share)
        self.bold_vein_path_share = (
            float(bold_vein_path_share)
            if bold_vein_path_share is not None
            else self.vein_path_share
        )
        self.winding = float(winding)

    def _fill_materials(self, volume: Volume, solid: "torch.Tensor") -> None:
        """Write the clouded body with its vein network.

        Args:
            volume: The volume whose channels fill.
            solid: (X, Y, Z) bool torch tensor of the generated shape.
        """
        import torch

        optical_mid = torch.full_like(volume.material_id, self.body_mid)
        vein_body = self.vein_mids[0]
        bold_vein_mid = self.vein_mids[-1]
        edge, wall, junction = self._make_cells("vein/network", self.vein_feature_size)
        width = self._make_vein_width(self.vein_share, junction)
        is_network = (edge < width) & (wall < self.vein_path_share)
        optical_mid = torch.where(
            is_network, torch.full_like(optical_mid, vein_body), optical_mid
        )
        if self.bold_vein_share > 0.0:
            bold_edge, bold_wall, _bold_junction = self._make_cells(
                "vein/bold", self.vein_feature_size * self.BOLD_VEIN_FEATURE_RATIO
            )
            is_bold = (bold_edge < self.bold_vein_share) & (
                bold_wall < self.bold_vein_path_share
            )
            optical_mid = torch.where(
                is_bold, torch.full_like(optical_mid, bold_vein_mid), optical_mid
            )
        volume.optical_mid = torch.where(
            solid, optical_mid, torch.zeros_like(volume.material_id)
        )

    def _make_vein_width(
        self, base_width: float, junction: "torch.Tensor"
    ) -> "torch.Tensor":
        """Return the per-cell vein width, wider at junctions and swayed.

        Args:
            base_width: The plain network vein width.
            junction: The (X, Y, Z) junction distance, near zero where a
                third cell comes close.
        """
        import torch

        junction = torch.clamp(1.0 - junction / self.JUNCTION_REACH, min=0.0, max=1.0)
        width = base_width * (1.0 + self.JOINT_WIDTH_BOOST * junction)
        factor, grid = self._make_coarse_grid()
        sway = FBMNoiseField(self.rng.split("vein/width"), self.vein_feature_size)(
            grid, device=self.device
        )
        if factor > 1:
            sway = self._upsample(sway, factor)
        return width * (1.0 + self.WIDTH_VARY_RATIO * sway)

    def _make_coarse_grid(self):
        """Return the coarsen factor and the coarser scan grid."""
        factor = self._make_coarsen_factor()
        grid = GridSpan(
            self.grid.origin,
            self.grid.cell_size * factor,
            tuple(count // factor for count in self.grid.cell_num),
        )
        return factor, grid

    def _make_cells(self, path: str, feature_size: float):
        """Return the cell boundary distance and the wall keep hash.

        The cellular scan is the costly step, so it runs on a coarser
        grid and the answers block-upsample back; the veins coarsen to
        the voxel anyway, so nothing visible is lost.

        Args:
            path: The stable stream path of this tier's fields.
            feature_size: The cell spacing of this tier (unit: m).

        Returns:
            The (X, Y, Z) boundary distance near zero on cell walls, the
            (X, Y, Z) per-wall hash in [0, 1) keyed by the pair of cells
            the wall parts, and the (X, Y, Z) junction distance near zero
            where a third cell also comes close.
        """
        import torch

        factor, grid = self._make_coarse_grid()
        lines = grid.make_axes(self.device)
        base = torch.zeros(grid.cell_num, dtype=torch.float32, device=self.device)
        point_x = lines[0].to(torch.float32)[:, None, None] + base
        point_y = lines[1].to(torch.float32)[None, :, None] + base
        point_z = lines[2].to(torch.float32)[None, None, :] + base
        stream = self.rng.split(path)
        warp_size = self.winding * feature_size
        if warp_size > 0.0:
            warp_feature = 3.0 * feature_size
            point_x = (
                point_x + self._warp(stream.split("wx"), warp_feature, grid) * warp_size
            )
            point_y = (
                point_y + self._warp(stream.split("wy"), warp_feature, grid) * warp_size
            )
            point_z = (
                point_z + self._warp(stream.split("wz"), warp_feature, grid) * warp_size
            )
        points = torch.stack(
            (point_x.reshape(-1), point_y.reshape(-1), point_z.reshape(-1)), dim=1
        )
        sample = CellNoise(stream.split("cell"), feature_size)(
            points, device=str(self.device)
        )
        edge = (sample.second_distance - sample.nearest_distance).reshape(grid.cell_num)
        wall = self._make_wall_hash(
            sample.cell_id, sample.second_cell_id, stream.split("keep")
        ).reshape(grid.cell_num)
        junction = (sample.third_distance - sample.nearest_distance).reshape(
            grid.cell_num
        )
        if factor > 1:
            edge = self._upsample(edge, factor)
            wall = self._upsample(wall, factor)
            junction = self._upsample(junction, factor)
        return edge, wall, junction

    def _make_coarsen_factor(self) -> int:
        """Return how much coarser the cellular scan grid runs."""
        for factor in (2,):
            if all(count % factor == 0 for count in self.grid.cell_num):
                return factor
        return 1

    def _warp(self, stream: Rng, feature_size: float, grid: GridSpan) -> "torch.Tensor":
        """Return a coarse (X, Y, Z) offset field in [-1, 1]."""
        return FBMNoiseField(stream, feature_size)(grid, device=self.device)

    @staticmethod
    def _upsample(field: "torch.Tensor", factor: int) -> "torch.Tensor":
        """Return the field with each cell repeated into a factor block."""
        return (
            field.repeat_interleave(factor, 0)
            .repeat_interleave(factor, 1)
            .repeat_interleave(factor, 2)
        )

    @staticmethod
    def _make_wall_hash(
        cell_id: "torch.Tensor", second_cell_id: "torch.Tensor", stream: Rng
    ) -> "torch.Tensor":
        """Return a per-wall hash in [0, 1), one value per cell pair.

        Args:
            cell_id: The nearest cell id per position.
            second_cell_id: The second nearest cell id per position.
            stream: The random stream the salt draws from, so each stone
                breaks different walls.
        """
        import torch

        mask = 0xFFFFFFFF
        salt = int(stream.integers(0, 2**31))
        low = torch.minimum(cell_id, second_cell_id)
        high = torch.maximum(cell_id, second_cell_id)
        mixed = ((low * 73856093) & mask) ^ ((high * 19349663) & mask) ^ salt
        mixed = mixed ^ (mixed >> 13)
        mixed = (mixed * 0x5BD1E995) & mask
        mixed = mixed ^ (mixed >> 15)
        return (mixed & mask).to(torch.float32) / float(2**32)


class ColumnGenerator:
    """Packs a footprint with upright polygonal columns, columnar basalt.

    Every 2D Voronoi cell extrudes one flat-topped column from the
    floor to a height sampled at its own center, so the tops step as
    terraces along the trend; boundary grooves and wavy rings cut the
    crevice shadow between packed columns, and each column tones itself
    off the run with a fine dither around its own mean.

    Args:
        grid: The evaluation grid the volume is generated on; larger
            than the footprint so the leaning columns have room.
        device: The torch device name the volume lives on.
        rng: The random stream every seeded field draws from.
        material_id: The single rigid material of the rock.
        body_mids: The tone run the columns hash their base shade over.
        box_size: The (x, y) footprint the columns fill (unit: m) and
            the z the grid holds; columns are pruned to the footprint.
        column_width: How wide the columns grow (unit: m); the cell
            size of the packing.
        top_height: The envelope height above the floor at the
            footprint center (unit: m).
        top_slope: The (x, y) rise of the envelope per meter from the
            center; tilts the tops high on one side to low on the other.
        top_sway_ratio: How far a column top steps below the trend at
            most, as a share of that column's own trend height; half
            coherent terrace sway, half per-column randomness.
        top_sway_feature_size: The feature size of the coherent sway
            half (unit: m); a few column widths reads as terraces.
        top_rough_size: How far the top faces roughen up and down
            (unit: m), a small coherent noise over the whole floor; 0
            leaves them flat.
        min_column_area: The smallest cross-section a column may keep
            (unit: m^2); a smaller column is pruned whole. 0 keeps all.
        ring_spacing: How far apart the cross-cutting rings sit along a
            column on average (unit: m); every ring steps an uneven
            hashed distance and some drop out, so the rings never read
            as a regular ladder. 0 leaves the faces unsegmented.
        ring_wave_size: How far a ring wanders up and down across a
            column face (unit: m), so the ring reads as a wavy crack
            rather than a flat band; 0 keeps it level.
        bend_amount: How far the column tops lean off vertical
            (unit: m); 0 stands them straight.
        tone_bias: Where the per-column base tones center on the run,
            0 the dark end to 1 the light end.
        tone_swing_ratio: How far the per-column base tones spread over
            the run, 0 one tone to 1 the whole run.
        dither_feature_size: The size of the within-column shade dither
            (unit: m).
    """

    # The fine dither shifts a column's shade by at most one run step
    # around the column's own mean, so a face has visible grain but
    # stays clearly its tone.
    DITHER_SHARE = 0.2

    # A column whose footprint holds less than this share of the median
    # column area is a sliver and is dropped whole.
    SLIVER_AREA_SHARE = 0.3

    # Each ring steps up by a hashed share of the spacing in this
    # range, and this share of rings drop out, so no column carries a
    # regular ladder.
    RING_STEP_LOW_SHARE = 0.6
    RING_STEP_HIGH_SHARE = 1.4
    RING_SKIP_SHARE = 0.25

    # The ring wave draws its wander over a floor patch this many
    # column widths across, so a ring bends over the face rather than
    # jittering cell to cell.
    RING_WAVE_FEATURE_WIDTH_NUM = 1.5

    def __init__(
        self,
        grid: GridSpan,
        device: str,
        rng: Rng,
        material_id: int,
        body_mids: Sequence[int],
        box_size: Tuple[float, float, float],
        column_width: float,
        top_height: float,
        top_slope: Tuple[float, float],
        top_sway_ratio: float,
        top_sway_feature_size: float,
        top_rough_size: float = 0.0,
        min_column_area: float = 0.0,
        ring_spacing: float = 0.0,
        ring_wave_size: float = 0.0,
        bend_amount: float = 0.0,
        tone_bias: float = 0.5,
        tone_swing_ratio: float = 0.6,
        dither_feature_size: float = 0.03,
    ) -> None:
        self.grid = grid
        self.device = device
        self.rng = rng
        self.material_id = int(material_id)
        self.body_mids = tuple(body_mids)
        self.box_size = tuple(float(part) for part in box_size)
        self.column_width = float(column_width)
        self.top_height = float(top_height)
        self.top_slope = (float(top_slope[0]), float(top_slope[1]))
        self.top_sway_ratio = float(top_sway_ratio)
        self.top_sway_feature_size = float(top_sway_feature_size)
        self.top_rough_size = float(top_rough_size)
        self.min_column_area = float(min_column_area)
        self.ring_spacing = float(ring_spacing)
        self.ring_wave_size = float(ring_wave_size)
        self.bend_amount = float(bend_amount)
        self.tone_bias = float(tone_bias)
        self.tone_swing_ratio = float(tone_swing_ratio)
        self.dither_feature_size = float(dither_feature_size)

    def generate(self) -> Volume:
        """Return the pruned, leaning, tone-varied column volume."""
        import torch

        axes = self.grid.make_axes(self.device)
        axis_x = axes[0].to(torch.float32)
        axis_y = axes[1].to(torch.float32)
        axis_z = axes[2].to(torch.float32)
        cell_id = self._make_columns(axis_x, axis_y)
        lo, hi = self.grid.get_extent()
        center = [(float(lo[side]) + float(hi[side])) / 2.0 for side in range(3)]
        floor_z = center[2] - self.box_size[2] / 2.0
        footprint = (torch.abs(axis_x - center[0]) <= self.box_size[0] / 2.0)[
            :, None
        ] & (torch.abs(axis_y - center[1]) <= self.box_size[1] / 2.0)[None, :]
        crossing = torch.isin(cell_id, torch.unique(cell_id[~footprint]))
        column_ids, column_index, column = self._make_column_index(
            cell_id, footprint & ~crossing
        )
        top = self._make_top(axis_x, axis_y, column_ids, column_index, column, center)
        tone_index = self._make_tone_index(cell_id)
        boundary = ColumnGenerator._make_boundary(column_index, column)
        ring_surface = self._make_ring_surface(axis_x, axis_y, column_ids, column_index)
        depth_z = axis_z - floor_z
        offset_x, offset_y = self._make_bend(depth_z, float(top.max()))
        table = torch.as_tensor(self.body_mids, dtype=torch.uint8, device=self.device)
        span = len(self.body_mids) - 1
        dither = ValueNoiseField(self.rng.split("dither"), self.dither_feature_size)(
            self.grid, device=self.device
        )
        volume = Volume(self.grid, self.device)
        material = volume.material_id
        optical = torch.zeros_like(material)
        groove = torch.zeros_like(material, dtype=torch.bool)
        for index in range(axis_z.shape[0]):
            if float(depth_z[index]) < 0.0:
                continue
            leaned_column = self._shift(column, offset_x[index], offset_y[index])
            leaned_top = self._shift(top, offset_x[index], offset_y[index])
            leaned_tone = self._shift(tone_index, offset_x[index], offset_y[index])
            layer = leaned_column & (float(depth_z[index]) < leaned_top)
            material[:, :, index] = torch.where(
                layer, self.material_id, material[:, :, index]
            )
            shade = torch.clamp(
                leaned_tone
                + torch.round(dither[:, :, index] * self.DITHER_SHARE * span).to(
                    torch.int64
                ),
                0,
                span,
            )
            optical[:, :, index] = torch.where(
                layer, table[shade], optical[:, :, index]
            )
            line = boundary
            if ring_surface is not None:
                on_ring = (
                    torch.abs(ring_surface - float(depth_z[index]))
                    < 0.5 * self.grid.cell_size
                ).any(dim=0)
                line = line | (column & on_ring)
            groove[:, :, index] = (
                self._shift(line, offset_x[index], offset_y[index]) & layer
            )
        volume.material_id = material
        volume.optical_mid = optical
        ColumnGenerator._carve_grooves(volume, groove)
        return volume

    def _make_columns(self, axis_x: "torch.Tensor", axis_y: "torch.Tensor"):
        """Return the per-column id over the floor.

        Args:
            axis_x: The x cell centers.
            axis_y: The y cell centers.
        """
        import torch

        grid_x, grid_y = torch.meshgrid(axis_x, axis_y, indexing="ij")
        points = torch.stack(
            (
                grid_x.reshape(-1),
                grid_y.reshape(-1),
                torch.zeros_like(grid_x).reshape(-1),
            ),
            dim=1,
        )
        sample = CellNoise(self.rng.split("column"), self.column_width)(
            points, device=str(self.device)
        )
        shape = (axis_x.shape[0], axis_y.shape[0])
        return sample.cell_id.reshape(shape)

    def _make_column_index(self, cell_id: "torch.Tensor", kept: "torch.Tensor"):
        """Return column ids, compact indices, and the sliver-dropped mask.

        Three gates drop the slivers whole: a column under the minimum
        cross-section goes, a column under a share of the median
        footprint area goes, and a column with no interior cell, one
        whose four floor neighbors are all its own, goes too, so no
        thin spike ever stands.

        Args:
            cell_id: The (X, Y) hashed column id per floor cell.
            kept: The (X, Y) mask of cells kept after the edge pruning.
        """
        import torch

        column_ids, inverse = torch.unique(cell_id, return_inverse=True)
        column_index = inverse.reshape(cell_id.shape)
        column_num = int(column_ids.shape[0])
        area = torch.bincount(column_index[kept], minlength=column_num).to(
            torch.float32
        )
        min_area_cell_num = self.min_column_area / self.grid.cell_size**2
        kept = kept & (area >= min_area_cell_num)[column_index]
        present = area >= max(min_area_cell_num, 1.0)
        if bool(present.any()):
            median_area = torch.median(area[present])
            kept = kept & (area >= self.SLIVER_AREA_SHARE * median_area)[column_index]
        interior = ColumnGenerator._make_interior(kept, column_index)
        interior_num = torch.bincount(column_index[interior], minlength=column_num)
        kept = kept & (interior_num > 0)[column_index]
        return column_ids, column_index, kept

    @staticmethod
    def _make_interior(
        kept: "torch.Tensor", column_index: "torch.Tensor"
    ) -> "torch.Tensor":
        """Return the cells whose four floor neighbors are all their own.

        Args:
            kept: The (X, Y) mask the interior is taken of.
            column_index: The (X, Y) compact column index per floor cell.
        """
        import torch

        interior = torch.zeros_like(kept)
        interior[1:-1, 1:-1] = (
            kept[1:-1, 1:-1]
            & kept[:-2, 1:-1]
            & (column_index[:-2, 1:-1] == column_index[1:-1, 1:-1])
            & kept[2:, 1:-1]
            & (column_index[2:, 1:-1] == column_index[1:-1, 1:-1])
            & kept[1:-1, :-2]
            & (column_index[1:-1, :-2] == column_index[1:-1, 1:-1])
            & kept[1:-1, 2:]
            & (column_index[1:-1, 2:] == column_index[1:-1, 1:-1])
        )
        return interior

    def _make_top(
        self,
        axis_x: "torch.Tensor",
        axis_y: "torch.Tensor",
        column_ids: "torch.Tensor",
        column_index: "torch.Tensor",
        column: "torch.Tensor",
        center: Sequence[float],
    ) -> "torch.Tensor":
        """Return the (X, Y) column-top height above the floor.

        Each column extrudes flat to one height: its trend at its own
        center, dropped by up to the sway ratio of that same trend
        height, half coherent terrace sway and half the column's own
        hash, so tall columns step further and neighbors never repeat.

        Args:
            axis_x: The x cell centers.
            axis_y: The y cell centers.
            column_ids: The hashed id per column, indexed by the compact
                column index.
            column_index: The (X, Y) compact column index per floor cell.
            column: The (X, Y) mask of standing column cells.
            center: The grid center (x, y, z) (unit: m).
        """
        import torch

        column_num = int(column_ids.shape[0])
        grid_x = axis_x[:, None].expand(column_index.shape)
        grid_y = axis_y[None, :].expand(column_index.shape)
        count = (
            torch.bincount(column_index[column], minlength=column_num)
            .to(torch.float32)
            .clamp(min=1.0)
        )
        center_x = (
            torch.zeros(column_num, device=self.device).index_add_(
                0, column_index[column], grid_x[column]
            )
            / count
        )
        center_y = (
            torch.zeros(column_num, device=self.device).index_add_(
                0, column_index[column], grid_y[column]
            )
            / count
        )
        trend = (
            self.top_height
            + (center_x - center[0]) * self.top_slope[0]
            + (center_y - center[1]) * self.top_slope[1]
        )
        points = torch.stack((center_x, center_y, torch.zeros_like(center_x)), dim=1)
        sway = ValueNoiseField(self.rng.split("top"), self.top_sway_feature_size)(
            points, device=str(self.device)
        ).to(torch.float32)
        jitter = self._hash01(column_ids, "height")
        drop_share = self.top_sway_ratio * (0.5 * (0.5 + 0.5 * sway) + 0.5 * jitter)
        height = trend * (1.0 - drop_share)
        height = torch.clamp(height, min=2.0 * self.grid.cell_size)
        top = torch.where(
            column, height[column_index], torch.zeros_like(height[column_index])
        )
        if self.top_rough_size > 0.0:
            floor_points = torch.stack(
                (
                    grid_x.reshape(-1),
                    grid_y.reshape(-1),
                    torch.zeros_like(grid_x).reshape(-1),
                ),
                dim=1,
            )
            rough = (
                ValueNoiseField(self.rng.split("top_rough"), 0.5 * self.column_width)(
                    floor_points, device=str(self.device)
                )
                .to(torch.float32)
                .reshape(top.shape)
            )
            top = torch.where(column, top + rough * self.top_rough_size, top)
        return top

    def _make_ring_surface(
        self,
        axis_x: "torch.Tensor",
        axis_y: "torch.Tensor",
        column_ids: "torch.Tensor",
        column_index: "torch.Tensor",
    ):
        """Return the (R, X, Y) wavy ring-crack heights, None when off.

        Each column climbs its own uneven ladder: every ring steps up
        by a hashed share of the spacing and some rings drop out, so no
        column carries a regular ladder and no two columns line up. Each
        ring then wanders up and down over the face by its own coherent
        noise, so it reads as a wavy crack, not a level band. A dropped
        ring sits far below the floor where no layer can match it.

        Args:
            axis_x: The x cell centers.
            axis_y: The y cell centers.
            column_ids: The hashed id per column, indexed by the compact
                column index.
            column_index: The (X, Y) compact column index per floor cell.
        """
        import torch

        if self.ring_spacing <= 0.0:
            return None
        ring_num = max(
            1,
            int(
                math.ceil(
                    self.box_size[2] / (self.RING_STEP_LOW_SHARE * self.ring_spacing)
                )
            ),
        )
        step_span = self.RING_STEP_HIGH_SHARE - self.RING_STEP_LOW_SHARE
        grid_x = axis_x[:, None].expand(column_index.shape)
        grid_y = axis_y[None, :].expand(column_index.shape)
        surface = []
        climb = self._hash01(column_ids, "ring_start") * self.ring_spacing
        for ring in range(ring_num):
            step_sway = self._hash01(column_ids, f"ring_step_{ring}")
            climb = climb + self.ring_spacing * (
                self.RING_STEP_LOW_SHARE + step_span * step_sway
            )
            is_skipped = (
                self._hash01(column_ids, f"ring_skip_{ring}") < self.RING_SKIP_SHARE
            )
            level = torch.where(is_skipped, torch.full_like(climb, -1.0e6), climb)
            height = level[column_index]
            if self.ring_wave_size > 0.0:
                height = height + self._make_ring_wave(grid_x, grid_y, ring)
            surface.append(height)
        return torch.stack(surface, dim=0)

    def _make_ring_wave(
        self, grid_x: "torch.Tensor", grid_y: "torch.Tensor", ring: int
    ) -> "torch.Tensor":
        """Return the (X, Y) up-down wander of one ring over the face.

        Args:
            grid_x: The (X, Y) x coordinate of each floor cell.
            grid_y: The (X, Y) y coordinate of each floor cell.
            ring: The ring index, salting its own coherent noise.
        """
        import torch

        points = torch.stack(
            (
                grid_x.reshape(-1),
                grid_y.reshape(-1),
                torch.zeros_like(grid_x).reshape(-1),
            ),
            dim=1,
        )
        feature_size = self.RING_WAVE_FEATURE_WIDTH_NUM * self.column_width
        wander = ValueNoiseField(self.rng.split(f"ring_wave_{ring}"), feature_size)(
            points, device=str(self.device)
        ).to(torch.float32)
        return wander.reshape(grid_x.shape) * self.ring_wave_size

    @staticmethod
    def _make_boundary(
        column_index: "torch.Tensor", column: "torch.Tensor"
    ) -> "torch.Tensor":
        """Return the one-cell boundary lines between standing columns.

        Only the lower-index side of each wall is marked, so the line
        stays one cell wide.

        Args:
            column_index: The (X, Y) compact column index per floor cell.
            column: The (X, Y) mask of standing column cells.
        """
        import torch

        boundary = torch.zeros_like(column)
        boundary[:-1, :] |= (
            column[:-1, :]
            & column[1:, :]
            & (column_index[:-1, :] != column_index[1:, :])
        )
        boundary[:, :-1] |= (
            column[:, :-1]
            & column[:, 1:]
            & (column_index[:, :-1] != column_index[:, 1:])
        )
        return boundary

    @staticmethod
    def _carve_grooves(volume: Volume, groove: "torch.Tensor") -> None:
        """Cut the marked groove cells where they touch exposed surface.

        Only marked cells within one cell of air leave, so a groove is
        a one-cell crevice on the tops and walls and the body behind it
        stays packed solid.

        Args:
            volume: The filled column volume the grooves cut into.
            groove: The (X, Y, Z) mask of boundary and ring cells.
        """
        import torch

        air = ~volume.get_solid_cells()
        near_air = (
            torch.nn.functional.max_pool3d(
                air.to(torch.float32)[None, None], kernel_size=3, stride=1, padding=1
            )[0, 0]
            > 0.5
        )
        carve = groove & near_air & ~air
        volume.material_id = torch.where(
            carve, torch.zeros_like(volume.material_id), volume.material_id
        )

    def _make_tone_index(self, cell_id: "torch.Tensor") -> "torch.Tensor":
        """Return the per-column base shade index over the tone run.

        Args:
            cell_id: The (X, Y) column id per floor cell.
        """
        import torch

        span = len(self.body_mids) - 1
        column_tone = self._hash01(cell_id, "tone")
        reach = self.tone_bias + (column_tone - 0.5) * self.tone_swing_ratio
        return torch.clamp(torch.round(reach * span).to(torch.int64), 0, span)

    def _make_bend(self, depth_z: "torch.Tensor", top_height: float):
        """Return the per-layer lean offsets in cells, growing with height.

        Args:
            depth_z: The (Z,) height of each layer above the floor.
            top_height: The tallest column top, the full-lean height.
        """
        import torch

        stream = self.rng.split("bend")
        angle = stream.uniform(0.0, 2.0 * math.pi)
        direction = (math.cos(angle), math.sin(angle))
        top = max(top_height, 1.0e-6)
        fraction = torch.clamp(depth_z / top, min=0.0)
        lean = self.bend_amount * (fraction + 0.3 * torch.sin(fraction * math.pi))
        offset_x = torch.round(lean * direction[0] / self.grid.cell_size).to(
            torch.int64
        )
        offset_y = torch.round(lean * direction[1] / self.grid.cell_size).to(
            torch.int64
        )
        return offset_x, offset_y

    @staticmethod
    def _shift(field: "torch.Tensor", offset_x, offset_y) -> "torch.Tensor":
        """Return the 2D field shifted by the offsets, the gap zero filled.

        Args:
            field: The (X, Y) field to shift.
            offset_x: The x shift in cells.
            offset_y: The y shift in cells.
        """
        import torch

        shift_x = int(offset_x)
        shift_y = int(offset_y)
        if shift_x == 0 and shift_y == 0:
            return field
        out = torch.zeros_like(field)
        size_x, size_y = field.shape
        source_x = slice(max(-shift_x, 0), size_x - max(shift_x, 0))
        target_x = slice(max(shift_x, 0), size_x - max(-shift_x, 0))
        source_y = slice(max(-shift_y, 0), size_y - max(shift_y, 0))
        target_y = slice(max(shift_y, 0), size_y - max(-shift_y, 0))
        out[target_x, target_y] = field[source_x, source_y]
        return out

    def _hash01(self, cell_id: "torch.Tensor", path: str) -> "torch.Tensor":
        """Return a per-column value in [0, 1) hashed from the column id.

        Args:
            cell_id: The (X, Y) column id.
            path: The stream path of this hash, salting it per stone.
        """
        import torch

        mask = 0xFFFFFFFF
        salt = int(self.rng.split(path).integers(0, 2**31))
        mixed = ((cell_id & mask) * 2654435761) & mask
        mixed = mixed ^ salt
        mixed = mixed ^ (mixed >> 13)
        mixed = (mixed * 0x5BD1E995) & mask
        mixed = mixed ^ (mixed >> 15)
        return (mixed & mask).to(torch.float32) / float(2**32)
