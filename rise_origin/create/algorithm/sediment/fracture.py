# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The fracture phase: continuous sections become marked pieces.

A fracture class writes two channels: ``segment_id`` names the piece
each cell belongs to, and ``structure_mark`` seams the boundaries
between pieces so erosion can eat the seams first. Multiple materials
per segment happen naturally because cuts run through generated
strata. :class:`FractureFragment` is the shared output type of every
fracture class in this file.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from ...protocol.rng import Rng
from ..field import CellNoise, FBMNoiseField, ValueNoiseField
from .volume import Volume

if TYPE_CHECKING:
    import torch

__all__ = [
    "FractureFragment",
    "PlaneFracture",
    "SeamFracture",
    "ChunkFracture",
    "PlateFracture",
]


@dataclass(frozen=True)
class FractureFragment:
    """One convex piece described as half spaces.

    Attributes:
        normals: (k, 3) float64 numpy array of unit plane normals
            pointing out of the piece.
        offsets: (k,) float64 numpy array of plane distances from the
            piece center (unit: m).
        corner_radius: How far the sharp junctions between the planes
            round off (unit: m); 0 keeps the plain polytope with its
            knife edges and right-angle corners.
    """

    normals: np.ndarray
    offsets: np.ndarray
    corner_radius: float = 0.0

    def get_distance(
        self, point_x: object, point_y: object, point_z: object, lib: object
    ) -> object:
        """Return the signed distance to the piece, negative inside.

        The plain maximum over the half-space distances leaves a knife
        edge wherever two planes meet, so a positive corner radius takes
        a smooth maximum instead: a single dominant plane is unchanged,
        so flat faces and the piece size hold, and only the junctions
        round inward into fillets.

        Args:
            point_x: The x coordinates (unit: m), broadcastable against
                the other parts, numpy or torch.
            point_y: The y coordinates (unit: m).
            point_z: The z coordinates (unit: m).
            lib: The array library the coordinates live in.
        """
        planes = []
        total = None
        for side in range(self.normals.shape[0]):
            part_x, part_y, part_z = (float(part) for part in self.normals[side])
            plane = (
                point_x * part_x
                + point_y * part_y
                + point_z * part_z
                - float(self.offsets[side])
            )
            planes.append(plane)
            total = plane if total is None else lib.maximum(total, plane)
        if self.corner_radius <= 0.0:
            return total
        sharpness = 1.0 / self.corner_radius
        blend = None
        for plane in planes:
            term = lib.exp((plane - total) * sharpness)
            blend = term if blend is None else blend + term
        return total + lib.log(blend) / sharpness


class PlaneFracture:
    """Pieces as cells of random planes around centers, not from a paper.

    A plane cut leaves flat faces, but a real rock breaks along an
    irregular surface that follows its nodules and fossils, so a
    coherent noise displaces the cut boundary in and out by the surface
    roughness, giving rough faces instead of clean planes.

    Args:
        fragments: The pieces to cut along.
        centers: (k, 3) float64 numpy array of piece center positions
            in the job's local frame (unit: m).
        rng: The random stream the boundary noise draws from; needed
            only when the surface roughness is above zero.
        surface_roughness: How far the cut boundary wanders in and out
            (unit: m); 0 keeps the faces flat.
        roughness_feature_size: The feature size of that wander
            (unit: m); the nodule scale of the rough face.

    Attributes:
        fragments: The pieces to cut along.
        centers: The piece center positions.
    """

    _OFFSET_JITTER = (0.85, 1.1)

    def __init__(
        self,
        fragments: Sequence[FractureFragment],
        centers: np.ndarray,
        rng: Optional[Rng] = None,
        surface_roughness: float = 0.0,
        roughness_feature_size: float = 0.1,
    ) -> None:
        self.fragments = tuple(fragments)
        self.centers = np.asarray(centers, dtype=np.float64)
        self.rng = rng
        self.surface_roughness = float(surface_roughness)
        self.roughness_feature_size = float(roughness_feature_size)

    def fracture(self, volume: Volume) -> None:
        """Cut the volume down to the pieces, clearing the outside.

        Every cell takes the segment of the piece it sits deepest
        inside; cells outside every piece become air.

        Args:
            volume: The volume to cut.
        """
        import torch

        deepest, segment = self._make_segments(volume)
        deepest = self._apply_roughness(volume, deepest)
        inside = deepest < 0.0
        volume.material_id = torch.where(
            inside, volume.material_id, torch.zeros_like(volume.material_id)
        )
        volume.segment_id = torch.where(inside, segment, torch.full_like(segment, -1))

    # The junctions round off over this share of the piece size unless
    # a caller asks for sharper, so a plain plane fracture never leaves
    # a knife edge or a right-angle corner.
    _CORNER_RADIUS_SHARE = 0.03

    @staticmethod
    def make_fragment(
        rng: Rng,
        size: float,
        flatness: float,
        elongation: float,
        plane_num_low: int = 8,
        plane_num_high: int = 14,
        corner_radius_share: float = _CORNER_RADIUS_SHARE,
    ) -> FractureFragment:
        """Return one piece as the cell of random planes.

        Args:
            rng: The random stream the planes are drawn from.
            size: The piece's target diameter (unit: m).
            flatness: How much the piece flattens along z, 0 to 1.
            elongation: How much the piece stretches along x, 0 to 1.
            plane_num_low: The fewest planes a piece is cut by.
            plane_num_high: The most planes a piece is cut by.
            corner_radius_share: How far the plane junctions round off,
                as a share of size; 0 keeps the knife edges.
        """
        generator = rng.split("fracture").get_numpy_generator()
        plane_num = int(generator.integers(plane_num_low, plane_num_high + 1))
        raw = generator.normal(0.0, 1.0, size=(plane_num, 3))
        lengths = np.linalg.norm(raw, axis=1, keepdims=True)
        lengths[lengths < 1.0e-9] = 1.0
        normals = raw / lengths
        reaches = np.array(
            [
                (size / 2.0) * (1.0 + 0.6 * elongation),
                size / 2.0,
                (size / 2.0) * (1.0 - 0.6 * flatness),
            ]
        )
        scaled = normals / reaches[None, :]
        offsets = 1.0 / np.linalg.norm(scaled, axis=1)
        offsets *= generator.uniform(
            PlaneFracture._OFFSET_JITTER[0],
            PlaneFracture._OFFSET_JITTER[1],
            size=offsets.shape,
        )
        return FractureFragment(normals, offsets, corner_radius_share * size)

    @staticmethod
    def make_prism_fragment(
        rng: Rng,
        radius: float,
        height: float,
        side_num: int = 6,
        corner_radius_share: float = _CORNER_RADIUS_SHARE,
    ) -> FractureFragment:
        """Return one piece as a jittered upright prism, columnar basalt.

        The side planes stand evenly around a circle and two caps close
        the top and bottom, so the piece is a regular prism (a hexagon
        at six sides) with the slight jitter that real columns carry.

        Args:
            rng: The random stream the jitter is drawn from.
            radius: The distance from the axis to each side (unit: m).
            height: The length of the prism along z (unit: m).
            side_num: How many sides the prism has, 6 for a hexagon.
            corner_radius_share: How far the edges round off, as a share
                of radius; 0 keeps the sharp column edges.
        """
        generator = rng.split("prism").get_numpy_generator()
        start = float(generator.uniform(0.0, 2.0 * np.pi))
        normals = []
        offsets = []
        for side in range(side_num):
            angle = start + 2.0 * np.pi * side / side_num
            normals.append([np.cos(angle), np.sin(angle), 0.0])
            offsets.append(radius * float(generator.uniform(0.92, 1.04)))
        normals.append([0.0, 0.0, 1.0])
        offsets.append((height / 2.0) * float(generator.uniform(0.92, 1.04)))
        normals.append([0.0, 0.0, -1.0])
        offsets.append((height / 2.0) * float(generator.uniform(0.92, 1.04)))
        return FractureFragment(
            np.asarray(normals, dtype=np.float64),
            np.asarray(offsets, dtype=np.float64),
            corner_radius_share * radius,
        )

    def _make_segments(self, volume: Volume):
        """Return the deepest distance and its segment per cell."""
        import torch

        axes = volume.grid.make_axes(str(volume.device))
        point_x = axes[0].to(torch.float32)[:, None, None]
        point_y = axes[1].to(torch.float32)[None, :, None]
        point_z = axes[2].to(torch.float32)[None, None, :]
        deepest = torch.full(
            volume.grid.cell_num, 1.0e9, dtype=torch.float32, device=volume.device
        )
        segment = torch.full(
            volume.grid.cell_num, -1, dtype=torch.int32, device=volume.device
        )
        for index, fragment in enumerate(self.fragments):
            distance = fragment.get_distance(
                point_x - float(self.centers[index][0]),
                point_y - float(self.centers[index][1]),
                point_z - float(self.centers[index][2]),
                torch,
            )
            is_deeper = distance < deepest
            deepest = torch.minimum(deepest, distance)
            segment = torch.where(is_deeper, torch.full_like(segment, index), segment)
        return deepest, segment

    def _apply_roughness(
        self, volume: Volume, deepest: "torch.Tensor"
    ) -> "torch.Tensor":
        """Return the boundary distance with the rough face noise added.

        Args:
            volume: The volume being cut, for its grid and device.
            deepest: The signed boundary distance per cell.
        """
        if self.surface_roughness <= 0.0:
            return deepest
        noise = FBMNoiseField(
            self.rng.split("roughness"), self.roughness_feature_size, layer_num=3
        )(volume.grid, device=str(volume.device))
        return deepest + noise * self.surface_roughness


class SeamFracture(PlaneFracture):
    """Seams a standing mass into pieces, keeping every solid cell.

    Pieces sit on a lattice of the seam spacing, each jittered off its
    post and grown past its cell so pieces touch, and the boundaries
    between touching pieces become the seams weather opens first. Cells
    the lattice misses stay unfractured mass.

    Args:
        rng: The random stream the lattice draws from.
        box_size: The (x, y, z) box edge lengths the lattice fills
            (unit: m).
        seam_spacing: The nominal piece spacing the mass breaks along
            (unit: m).
        seam_width: How wide the seams read (unit: m); finer than one
            cell marks nothing.
    """

    # Each piece sits this far off its lattice post as a share of the
    # spacing, and grows this much past its cell so pieces touch.
    _JITTER_SHARE = 0.35
    _PIECE_SIZE_SHARE = 1.5

    def __init__(
        self,
        rng: Rng,
        box_size: Tuple[float, float, float],
        seam_spacing: float,
        seam_width: float = 0.0,
    ) -> None:
        counts = tuple(
            max(1, int(round(box_size[side] / seam_spacing))) for side in range(3)
        )
        fragments = []
        centers = []
        for index_x in range(counts[0]):
            for index_y in range(counts[1]):
                for index_z in range(counts[2]):
                    piece_rng = rng.split(f"{index_x}/{index_y}/{index_z}")
                    centers.append(
                        [
                            (index - (count - 1) / 2.0) * seam_spacing
                            + piece_rng.uniform(-self._JITTER_SHARE, self._JITTER_SHARE)
                            * seam_spacing
                            for index, count in zip((index_x, index_y, index_z), counts)
                        ]
                    )
                    fragments.append(
                        PlaneFracture.make_fragment(
                            piece_rng,
                            self._PIECE_SIZE_SHARE * seam_spacing,
                            0.0,
                            0.0,
                        )
                    )
        super().__init__(tuple(fragments), np.asarray(centers))
        self.seam_width = float(seam_width)

    def fracture(self, volume: Volume) -> None:
        """Seam the volume into its pieces, removing no material.

        Solid cells inside a piece take its segment; solid cells
        outside every piece stay segment -1 as unfractured mass.

        Args:
            volume: The filled volume to seam.
        """
        import torch

        deepest, segment = self._make_segments(volume)
        deepest = self._apply_roughness(volume, deepest)
        keeps = volume.get_solid_cells() & (deepest < 0.0)
        volume.segment_id = torch.where(keeps, segment, torch.full_like(segment, -1))
        volume.write_seam_marks(self.seam_width)


class ChunkFracture:
    """Knocks local chunks off a stone, breaking its convexity.

    A single plane cut leaves a convex block, but a real rock breaks
    more than once and loses corners and edges to the later breaks.
    Each chunk removes the solid beyond a random plane through a
    surface point, but only within a local radius, so it knocks off one
    corner and leaves a fresh flat scar instead of shaving the whole
    stone convex. It keeps one body: no segment splits, only material
    leaves, so a chunk is a partial break, not a stone cut in two.

    Args:
        rng: The random stream the chunks are drawn from.
        chunk_num: How many chunks to knock off.
        chunk_radius: The reach of one chunk (unit: m); the size of the
            knocked-off corner.
        radius_spread_ratio: How widely the chunk reach varies around
            the radius, 0 to 1.
        normal_spread_ratio: How far a chunk plane tilts off the
            straight outward direction, 0 to 1.
    """

    def __init__(
        self,
        rng: Rng,
        chunk_num: int,
        chunk_radius: float,
        radius_spread_ratio: float = 0.4,
        normal_spread_ratio: float = 0.5,
    ) -> None:
        self.rng = rng
        self.chunk_num = int(chunk_num)
        self.chunk_radius = float(chunk_radius)
        self.radius_spread_ratio = float(radius_spread_ratio)
        self.normal_spread_ratio = float(normal_spread_ratio)

    def fracture(self, volume: Volume) -> None:
        """Knock the configured chunks off the volume.

        Args:
            volume: The volume to break chunks from.
        """
        import torch

        generator = self.rng.get_torch_generator(str(volume.device))
        radius_cell = self.chunk_radius / volume.grid.cell_size
        lines = [
            torch.arange(count, device=volume.device, dtype=torch.float32)
            for count in volume.grid.cell_num
        ]
        cell_x = lines[0][:, None, None]
        cell_y = lines[1][None, :, None]
        cell_z = lines[2][None, None, :]
        for _index in range(self.chunk_num):
            solid = volume.get_solid_cells()
            occupied = solid.nonzero(as_tuple=False)
            point = self._pick_surface_point(solid, generator)
            if point is None:
                return
            centroid = occupied.to(torch.float32).mean(dim=0)
            normal = self._make_outward_normal(point, centroid, generator)
            spread = float(
                torch.rand(1, generator=generator, device=volume.device) * 2.0 - 1.0
            )
            reach = radius_cell * (1.0 + self.radius_spread_ratio * spread)
            offset_x = cell_x - float(point[0])
            offset_y = cell_y - float(point[1])
            offset_z = cell_z - float(point[2])
            beyond = (
                offset_x * float(normal[0])
                + offset_y * float(normal[1])
                + offset_z * float(normal[2])
            ) > 0.0
            near = offset_x**2 + offset_y**2 + offset_z**2 < reach**2
            chunk = solid & beyond & near
            volume.material_id = torch.where(
                chunk, torch.zeros_like(volume.material_id), volume.material_id
            )
            if volume.segment_id is not None:
                volume.segment_id = torch.where(
                    chunk, torch.full_like(volume.segment_id, -1), volume.segment_id
                )

    def _make_outward_normal(
        self, point: "torch.Tensor", centroid: "torch.Tensor", generator: object
    ) -> "torch.Tensor":
        """Return the mostly outward chunk plane normal, tilted a little.

        Args:
            point: The surface point the chunk plane passes through.
            centroid: The stone's solid centroid.
            generator: The device torch generator the tilt draws from.
        """
        import torch

        direction = point - centroid
        length = torch.linalg.vector_norm(direction)
        if float(length) < 1.0e-6:
            outward = torch.tensor([0.0, 0.0, 1.0], device=point.device)
        else:
            outward = direction / length
        tilt = (
            torch.randn(3, generator=generator, device=point.device)
            * self.normal_spread_ratio
        )
        normal = outward + tilt
        return normal / torch.linalg.vector_norm(normal).clamp(min=1.0e-6)

    @staticmethod
    def _pick_surface_point(solid: "torch.Tensor", generator: object):
        """Return one random boundary cell, or None when none is solid."""
        import torch

        air = (~solid).to(torch.float32)
        near_air = (
            torch.nn.functional.max_pool3d(
                air[None, None], kernel_size=3, stride=1, padding=1
            )[0, 0]
            > 0.5
        )
        boundary = (solid & near_air).nonzero(as_tuple=False)
        if boundary.shape[0] == 0:
            return None
        pick = torch.randint(
            boundary.shape[0], (1,), generator=generator, device=solid.device
        )
        return boundary[int(pick)].to(torch.float32)


class PlateFracture:
    """Splits a volume into thin aligned plates and sheds the weak ones.

    The rock parts into thin plates along its one weak direction, and
    whether a plate falls follows Weibull 1939 ("A Statistical Theory
    of the Strength of Materials"): its fall chance is the weakest-link
    law over its load relative to what still holds it, with held faces
    across the weak planes far stronger than the cleavage glue along
    them. Falls cascade until quiet, and each plate takes its own shade
    of its layer's hue family with a fine streak running along it.

    Args:
        rng: The random stream the plates and shedding draw from.
        break_dir: The (x, y, z) weak direction; plates lie thin along
            it and their long side runs down its steepest slope.
        plate_thickness: How thick a plate grows along the weak
            direction (unit: m).
        plate_width: How wide a plate grows across its long side
            (unit: m).
        plate_length: How long a plate grows down the slope (unit: m).
        hold_strength: The rock's unitless strength scale, the Weibull
            law's characteristic strength: a plate whose load ratio
            reaches it fails more often than not. Lower is more
            weathered rock and sheds more; around 1 is heavily broken
            and around 2.5 barely touched.
        shed_round_num: The most cascade rounds to run; the cascade
            stops earlier on the first quiet round.
        family_runs: The layer hue family mid runs, all one length; a
            plate reads its layer's family from the volume and takes one
            hashed shade of that family's run.
        stain_run: The stain mid run, same length as the families; a
            hashed share of plates take it instead of their layer family.
        stain_share: The share of plates that stain, 0 none to 1 all.
        tone_bias: Where the per-plate shades center on the run, 0 the
            dark end to 1 the light end.
    """

    # The fine streak shifts a plate's shade by at most one run step
    # around the plate's own shade, so a face has combed grain but
    # stays clearly its tone.
    DITHER_SHARE = 0.2

    # How far the per-plate base shades spread around the tone bias
    # over the run, so neighbors differ while the dial keeps its say.
    TONE_SWING = 0.6

    # The streak noise features in plate frame units: long along the
    # plate, shorter across it, tight through its thickness, so the
    # grain reads as combed streaks running down the plates.
    STREAK_LENGTH_SHARE = 1.0
    STREAK_WIDTH_SHARE = 0.3
    STREAK_THICKNESS_SHARE = 0.5

    # The Weibull modulus, the flaw scatter of the material; flaky
    # flaw-rich rock sits at the low end of the rock range, so failure
    # is gradual and scattered rather than a sharp threshold.
    WEIBULL_MODULUS = 6.0

    # Slate's strength across the weak planes is several times its
    # cleavage glue, so a bond aligned with the weak direction keeps
    # only this share of full strength.
    CLEAVAGE_STRENGTH_SHARE = 0.25

    # The three bond channels a contact carries by its direction
    # against world up: a contact below rests in compression, which
    # routes weight but never fails at rock scale, so it carries much
    # and adds no failure area; a side contact keys in shear; a contact
    # above hangs in tension, the weakest channel and where the
    # cleavage glue fails first.
    COMPRESSION_STRENGTH = 50.0
    SHEAR_STRENGTH = 0.6
    TENSION_STRENGTH = 0.35

    # Weathering is its own failure source on exposed glue: an open
    # flat face means the plate's back cleavage bond takes this
    # standing stress, growing toward the crest of the current shape
    # with the power concentrating it in the crest band. Not from the
    # paper.
    WEATHER_STRESS = 1.0
    HEIGHT_LOAD_GAIN = 0.6
    HEIGHT_LOAD_POWER = 4.0

    # A plate cut down to a stub of its nominal volume is pre-cracked
    # by whatever truncated it (a fragment cut, an occupancy rim), so
    # the volume deficit stands as this stress on the remaining body;
    # below the share the deficit reads fully, above it not at all.
    SLIVER_VOLUME_SHARE = 0.3
    SLIVER_STRESS = 1.2

    # How many grid points one noise evaluation slab holds at most.
    SLAB_POINT_NUM = 4_000_000

    def __init__(
        self,
        rng: Rng,
        break_dir: Tuple[float, float, float],
        plate_thickness: float,
        plate_width: float,
        plate_length: float,
        hold_strength: float = 1.6,
        shed_round_num: int = 6,
        family_runs: Sequence[Sequence[int]] = (),
        stain_run: Sequence[int] = (),
        stain_share: float = 0.0,
        tone_bias: float = 0.5,
    ) -> None:
        self.rng = rng
        self.break_dir = tuple(float(part) for part in break_dir)
        self.plate_thickness = float(plate_thickness)
        self.plate_width = float(plate_width)
        self.plate_length = float(plate_length)
        self.hold_strength = float(hold_strength)
        self.shed_round_num = int(shed_round_num)
        self.family_runs = tuple(tuple(run) for run in family_runs)
        self.stain_run = tuple(stain_run)
        self.stain_share = float(stain_share)
        self.tone_bias = float(tone_bias)
        run_lengths = {len(run) for run in self.family_runs}
        if self.stain_run:
            run_lengths.add(len(self.stain_run))
        if len(run_lengths) > 1:
            raise ValueError("plate family runs must share one length")
        # thin_dir: the normalized break_dir; plates stack along it, one
        #   plate ending every plate_thickness.
        # long_dir: world up projected into the weak plane, the plane's
        #   steepest slope line; plates run their plate_length along it,
        #   the way broken slate hangs and slides.
        # wide_dir: the cross of the two, the remaining in-plane side;
        #   plates span their plate_width along it.
        thin_dir, long_dir, wide_dir = PlateFracture._make_frame(self.break_dir)
        self.thin_dir = thin_dir
        self.long_dir = long_dir
        self.wide_dir = wide_dir

    def fracture(self, volume: Volume) -> None:
        """Split the volume into plates and shed the exposed ones.

        Args:
            volume: The volume to break into plates.
        """
        import torch

        partition = self._make_plate_partition(volume)
        plate_ids, inverse = torch.unique(partition, return_inverse=True)
        partition = inverse.reshape(partition.shape)
        self._shed(volume, partition, plate_ids)
        volume.segment_id = torch.where(
            volume.get_solid_cells(),
            partition.to(torch.int32),
            torch.full_like(partition, -1, dtype=torch.int32),
        )

    def make_plate_tones(self, volume: Volume) -> None:
        """Give every plate its own shade of its layer's hue family.

        A plate reads the majority family its layer painted, takes one
        hashed shade of that family's run, or the stain run for a
        hashed share of plates, and a fine streak dither runs along the
        plate direction within it.

        Args:
            volume: The plate-fractured volume whose tones write.
        """
        import torch

        if not self.family_runs or volume.segment_id is None:
            return
        solid = volume.get_solid_cells()
        segment = volume.segment_id.to(torch.int64)
        has_plate = solid & (segment >= 0)
        if not bool(has_plate.any()):
            return
        plate_num = int(segment.max()) + 1
        family_num = len(self.family_runs)
        run_length = len(self.family_runs[0])
        span = run_length - 1
        mid_to_family = torch.full((256,), -1, dtype=torch.int64, device=volume.device)
        for family_index, run in enumerate(self.family_runs):
            for mid in run:
                mid_to_family[int(mid)] = family_index
        voxel_family = mid_to_family[volume.optical_mid.to(torch.int64)]
        counted = has_plate & (voxel_family >= 0)
        pair = segment[counted] * family_num + voxel_family[counted]
        counts = torch.bincount(pair, minlength=plate_num * family_num).reshape(
            plate_num, family_num
        )
        plate_family = counts.argmax(dim=1)
        plate_row = torch.arange(plate_num, device=volume.device)
        shade_reach = (
            self.tone_bias + (self._hash01(plate_row, "shade") - 0.5) * self.TONE_SWING
        )
        plate_shade = torch.clamp(
            torch.round(shade_reach * span).to(torch.int64), 0, span
        )
        table_runs = list(self.family_runs)
        if self.stain_run and self.stain_share > 0.0:
            is_stained = self._hash01(plate_row, "stain") < self.stain_share
            plate_family = torch.where(
                is_stained, torch.full_like(plate_family, len(table_runs)), plate_family
            )
            table_runs.append(self.stain_run)
        table = torch.as_tensor(table_runs, dtype=torch.uint8, device=volume.device)
        streak = self._make_streak(volume)
        anchored = segment.clamp(min=0)
        shade = torch.clamp(
            plate_shade[anchored]
            + torch.round(streak * self.DITHER_SHARE * span).to(torch.int64),
            0,
            span,
        )
        picks = table[plate_family[anchored], shade]
        volume.optical_mid = torch.where(has_plate, picks, volume.optical_mid)

    def _make_plate_partition(self, volume: Volume) -> "torch.Tensor":
        """Return the (X, Y, Z) raw plate id of every cell.

        The cell pattern runs in the plate frame with each side scaled
        by its plate dimension, so every cell comes out as one thin
        plate aligned to the weak direction.

        Args:
            volume: The volume whose grid the partition covers.
        """
        import torch

        pattern = CellNoise(self.rng.split("plate"), 1.0)
        scales = (self.plate_thickness, self.plate_length, self.plate_width)
        size_x, size_y, _size_z = volume.grid.cell_num
        pieces = []
        for points, slab in self._iterate_slab_points(volume, scales):
            sample = pattern(points, device=str(volume.device))
            pieces.append(sample.cell_id.reshape(size_x, size_y, slab[1] - slab[0]))
        return torch.cat(pieces, dim=2)

    def _make_streak(self, volume: Volume) -> "torch.Tensor":
        """Return the (X, Y, Z) combed streak noise in the plate frame.

        Args:
            volume: The volume whose grid the streak covers.
        """
        import torch

        field = ValueNoiseField(self.rng.split("streak"), 1.0)
        scales = (
            self.STREAK_THICKNESS_SHARE * self.plate_thickness,
            self.STREAK_LENGTH_SHARE * self.plate_length,
            self.STREAK_WIDTH_SHARE * self.plate_width,
        )
        size_x, size_y, _size_z = volume.grid.cell_num
        pieces = []
        for points, slab in self._iterate_slab_points(volume, scales):
            value = field(points, device=str(volume.device)).to(torch.float32)
            pieces.append(value.reshape(size_x, size_y, slab[1] - slab[0]))
        return torch.cat(pieces, dim=2)

    def _iterate_slab_points(self, volume: Volume, scales: Tuple[float, float, float]):
        """Yield plate-frame points of the grid, one z slab at a time.

        Args:
            volume: The volume whose grid the points cover.
            scales: The (thin, long, wide) frame divisors (unit: m).
        """
        import torch

        axes = volume.grid.make_axes(str(volume.device))
        axis_x = axes[0].to(torch.float32)
        axis_y = axes[1].to(torch.float32)
        axis_z = axes[2].to(torch.float32)
        floor_num = int(axis_x.shape[0]) * int(axis_y.shape[0])
        slab_num = max(1, self.SLAB_POINT_NUM // floor_num)
        rows = (self.thin_dir, self.long_dir, self.wide_dir)
        for start in range(0, int(axis_z.shape[0]), slab_num):
            slab_z = axis_z[start : start + slab_num]
            grid_x = axis_x[:, None, None]
            grid_y = axis_y[None, :, None]
            grid_z = slab_z[None, None, :]
            parts = []
            for row, scale in zip(rows, scales):
                value = (grid_x * row[0] + grid_y * row[1] + grid_z * row[2]) / scale
                parts.append(
                    value.expand(
                        axis_x.shape[0], axis_y.shape[0], slab_z.shape[0]
                    ).reshape(-1)
                )
            yield torch.stack(parts, dim=1), (start, start + int(slab_z.shape[0]))

    def _shed(
        self,
        volume: Volume,
        partition: "torch.Tensor",
        plate_ids: "torch.Tensor",
    ) -> None:
        """Drop failing plates whole by the Weibull law, until quiet.

        Weight flows like water through the support graph: each plate
        passes what it carries to its lower neighbors, split by bond
        capacity, and every carrying bond takes a stress. A bond then
        fails by Weibull 1939 ("A Statistical Theory of the Strength of
        Materials"), the weakest-link law over its stress and area, and
        a plate falls when its carrying bonds fail. Each fall reroutes
        the flow, so failures cascade toward the shape where every
        plate holds.

        Args:
            volume: The volume material sheds from.
            partition: The (X, Y, Z) compact plate index of every cell.
            plate_ids: The raw hash id per compact plate index.
        """
        import torch

        start_solid = volume.get_solid_cells()
        start_material = volume.material_id
        ground_index = int(start_solid.any(dim=0).any(dim=0).nonzero()[0])
        plate_num = int(partition.max()) + 1
        cell_size = volume.grid.cell_size
        for round_index in range(self.shed_round_num):
            solid = volume.get_solid_cells()
            weight = torch.bincount(partition[solid], minlength=plate_num).to(
                torch.float32
            )
            level = PlateFracture._make_plate_levels(solid, partition)
            edges, spall_open = self._make_contact_edges(solid, partition, ground_index)
            weather = self._make_weather_stress(weight, level)
            chance = self._make_fall_chance(
                edges,
                weight.cpu().numpy(),
                level.cpu().numpy(),
                weather,
                spall_open.cpu().numpy(),
                plate_num,
                cell_size,
            )
            roll = self._hash01(plate_ids, f"shed_{round_index}").cpu().numpy()
            dropped_rows = np.nonzero(roll < chance)[0]
            if dropped_rows.size == 0:
                break
            dropped = torch.as_tensor(dropped_rows, device=partition.device)
            gone = torch.isin(partition, dropped) & solid
            if not bool(gone.any()):
                break
            volume.material_id = torch.where(
                gone, torch.zeros_like(volume.material_id), volume.material_id
            )
            PlateFracture._fill_sealed_air(volume, start_material)

    def _make_weather_stress(
        self, weight: "torch.Tensor", level: "torch.Tensor"
    ) -> np.ndarray:
        """Return each plate's standing weathering stress.

        Weathering attacks exposed glue: the standing stress grows
        toward the crest of the current shape, where rock is least
        confined.

        Args:
            weight: The per-plate solid cell counts.
            level: The per-plate mean height (unit: cells).
        """
        present = weight > 0
        low = level[present].min() if bool(present.any()) else level.min()
        span = (level.max() - low).clamp(min=1.0)
        height_share = ((level - low) / span).clamp(min=0.0)
        stress = self.WEATHER_STRESS * (
            1.0 + self.HEIGHT_LOAD_GAIN * height_share**self.HEIGHT_LOAD_POWER
        )
        return stress.cpu().numpy()

    def _make_fall_chance(
        self,
        edges: dict,
        weight: np.ndarray,
        level: np.ndarray,
        weather: np.ndarray,
        spall_open: np.ndarray,
        plate_num: int,
        cell_size: float,
    ) -> np.ndarray:
        """Return each plate's fall chance from the support graph flow.

        Bands of one plate thickness sweep top to bottom: every plate
        in the band hands its carried weight to its lower neighbors,
        split by bond capacity, and each carrying bond records its
        stress. Only glue area can fail, compression never does; a
        plate with no lower bond hangs its weight on its upward bonds,
        and one with no bonds at all always falls. The fall chance
        composes the Weibull survival of the plate's carrying glue with
        the weathering of its exposed flat face and the pre-crack of a
        plate cut down to a stub of its nominal volume.

        Args:
            edges: The contact edge arrays: ``src``, ``dst``, ``area``,
                ``glue_area``, and ``capacity``; the ground node is the
                index after the last plate.
            weight: The per-plate solid cell counts.
            level: The per-plate mean height (unit: cells); the ground
                node sits below every plate.
            weather: The per-plate standing weathering stress.
            spall_open: The per-plate open faces along the weak
                direction.
            plate_num: How many plates the partition holds.
            cell_size: The volume cell size (unit: m).
        """
        thickness_cell_num = max(self.plate_thickness / cell_size, 1.0)
        reference_area = max(self.plate_width * self.plate_length / cell_size**2, 1.0)
        src = edges["src"]
        dst = edges["dst"]
        glue_area = edges["glue_area"]
        capacity = edges["capacity"]
        rest_capacity = edges["rest_capacity"]
        # An edge carries weight only where real down faces rest on the
        # neighbor, so support follows geometry, not centroid order.
        is_carrying = rest_capacity > 0.0
        node_level = np.concatenate([level, [level.min() - 1.0]])
        demand = weight.astype(np.float64).copy()
        exponent = np.zeros(plate_num, dtype=np.float64)
        band_num = (
            int(
                np.ceil(
                    (node_level[:plate_num].max() - node_level[:plate_num].min())
                    / thickness_cell_num
                )
            )
            + 1
        )
        band = np.minimum(
            (
                (node_level[:plate_num].max() - node_level[:plate_num])
                // thickness_cell_num
            ).astype(np.int64),
            band_num - 1,
        )
        rest_sum = np.zeros(plate_num, dtype=np.float64)
        np.add.at(rest_sum, src[is_carrying], rest_capacity[is_carrying])
        for band_index in range(band_num):
            in_band = is_carrying & (band[src] == band_index)
            if not in_band.any():
                continue
            flow = (
                demand[src[in_band]] * rest_capacity[in_band] / rest_sum[src[in_band]]
            )
            keeps_ground = dst[in_band] < plate_num
            np.add.at(demand, dst[in_band][keeps_ground], flow[keeps_ground])
            stress = (flow / thickness_cell_num) / np.maximum(
                rest_capacity[in_band], 1.0e-6
            )
            np.add.at(
                exponent,
                src[in_band],
                (glue_area[in_band] / reference_area)
                * (stress / self.hold_strength) ** self.WEIBULL_MODULUS,
            )
        # A plate resting on nothing hangs its whole carried weight on
        # its glue bonds; with no bonds at all it always falls.
        hangs = (rest_sum <= 0.0) & (weight > 0)
        if hangs.any():
            hang_capacity = np.zeros(plate_num, dtype=np.float64)
            is_hanging = hangs[src]
            np.add.at(hang_capacity, src[is_hanging], capacity[is_hanging])
            flow = (
                demand[src[is_hanging]]
                * capacity[is_hanging]
                / np.maximum(hang_capacity[src[is_hanging]], 1.0e-6)
            )
            stress = (flow / thickness_cell_num) / np.maximum(
                capacity[is_hanging], 1.0e-6
            )
            np.add.at(
                exponent,
                src[is_hanging],
                (glue_area[is_hanging] / reference_area)
                * (stress / self.hold_strength) ** self.WEIBULL_MODULUS,
            )
            loose = hangs & (hang_capacity <= 0.0)
            exponent[loose] = np.inf
        exponent = (
            exponent
            + (spall_open / reference_area)
            * (weather / self.hold_strength) ** self.WEIBULL_MODULUS
        )
        # A plate cut down to a stub is pre-cracked by whatever
        # truncated it, so the volume deficit stands as a stress on the
        # remaining body and rim slivers shed instead of hanging on.
        nominal_cell_num = max(
            self.plate_thickness * self.plate_width * self.plate_length / cell_size**3,
            1.0,
        )
        sliver_reach = np.clip(
            1.0 - weight / (self.SLIVER_VOLUME_SHARE * nominal_cell_num), 0.0, 1.0
        )
        exponent = exponent + sliver_reach * (
            (self.SLIVER_STRESS / self.hold_strength) ** self.WEIBULL_MODULUS
        )
        chance = 1.0 - np.exp(-exponent)
        chance[weight <= 0] = 0.0
        return chance

    @staticmethod
    def _fill_sealed_air(volume: Volume, start_material: "torch.Tensor") -> None:
        """Refill air pockets sealed inside the solid with their rock.

        One connected-components pass over the air names its regions; a
        region that never reaches the grid border is a sealed pocket,
        and a pocket that got sealed never really emptied, so its cells
        take their material from before the shedding back.

        Args:
            volume: The volume whose sealed pockets refill.
            start_material: The (X, Y, Z) material ids before shedding.
        """
        import cc3d
        import torch

        air = (~volume.get_solid_cells()).cpu().numpy()
        labels = cc3d.connected_components(air, connectivity=6)
        border_labels = np.unique(
            np.concatenate(
                [
                    labels[0, :, :].ravel(),
                    labels[-1, :, :].ravel(),
                    labels[:, 0, :].ravel(),
                    labels[:, -1, :].ravel(),
                    labels[:, :, 0].ravel(),
                    labels[:, :, -1].ravel(),
                ]
            )
        )
        sealed = air & ~np.isin(labels, border_labels)
        if not sealed.any():
            return
        is_sealed = torch.as_tensor(sealed, device=volume.device)
        volume.material_id = torch.where(is_sealed, start_material, volume.material_id)

    def _make_contact_edges(
        self,
        solid: "torch.Tensor",
        partition: "torch.Tensor",
        ground_index: int,
    ):
        """Return the support graph edges and the spall openings.

        Every contact interface between two plates becomes one edge
        carrying its face area and its bond capacity: the contact
        direction against world up picks the channel (compression
        below, shear sideways, tension above) and the cleavage
        alignment scales the glue, so a bond is what its geometry says
        it is. Down faces on the lowest solid level bond to the ground
        node, the index after the last plate. The spall openings count
        each plate's open faces along the weak direction.

        Args:
            solid: The (X, Y, Z) solid cells of the volume.
            partition: The (X, Y, Z) compact plate index of every cell.
            ground_index: The z index of the lowest solid level, where
                the ground carries.
        """
        import torch

        plate_num = int(partition.max()) + 1
        ground_node = plate_num
        spall_open = torch.zeros(plate_num, dtype=torch.float32, device=solid.device)
        faces = (
            ((0, 0, -1), self.COMPRESSION_STRENGTH, False),
            ((0, 0, 1), self.TENSION_STRENGTH, True),
            ((-1, 0, 0), self.SHEAR_STRENGTH, True),
            ((1, 0, 0), self.SHEAR_STRENGTH, True),
            ((0, -1, 0), self.SHEAR_STRENGTH, True),
            ((0, 1, 0), self.SHEAR_STRENGTH, True),
        )
        keys = []
        areas = []
        glue_areas = []
        rest_capacities = []
        capacities = []
        for step, channel, is_glue in faces:
            aligned_reach = (
                step[0] * self.thin_dir[0]
                + step[1] * self.thin_dir[1]
                + step[2] * self.thin_dir[2]
            )
            alignment = abs(aligned_reach)
            strength = channel
            if is_glue:
                strength = channel * (
                    self.CLEAVAGE_STRENGTH_SHARE
                    + (1.0 - self.CLEAVAGE_STRENGTH_SHARE) * (1.0 - alignment)
                )
            neighbor_solid = PlateFracture._shift_mask(solid, step, False)
            neighbor_index = torch.full_like(partition, -1)
            source = tuple(
                slice(max(part, 0), size + min(part, 0))
                for part, size in zip(step, partition.shape)
            )
            target = tuple(
                slice(max(-part, 0), size + min(-part, 0))
                for part, size in zip(step, partition.shape)
            )
            neighbor_index[target] = partition[source]
            contact = solid & neighbor_solid & (neighbor_index != partition)
            if step == (0, 0, -1):
                grounded = solid[:, :, ground_index]
                neighbor_index = neighbor_index.clone()
                neighbor_index[:, :, ground_index] = torch.where(
                    grounded,
                    torch.full_like(neighbor_index[:, :, ground_index], ground_node),
                    neighbor_index[:, :, ground_index],
                )
                contact = contact.clone()
                contact[:, :, ground_index] = grounded
            else:
                # Only faces pointing along the weak direction weather,
                # so break_dir aims the face that spalls and the far
                # side stays quiet.
                open_face = solid & ~neighbor_solid
                if aligned_reach > 0.5:
                    spall_open = spall_open + aligned_reach * torch.bincount(
                        partition[open_face], minlength=plate_num
                    )
            pair_key = partition[contact].to(torch.int64) * (
                ground_node + 1
            ) + neighbor_index[contact].to(torch.int64)
            unique_key, count = torch.unique(pair_key, return_counts=True)
            face_area = count.cpu().numpy().astype(np.float64)
            keys.append(unique_key.cpu().numpy())
            areas.append(face_area)
            glue_areas.append(face_area if is_glue else np.zeros_like(face_area))
            rest_capacities.append(
                np.zeros_like(face_area) if is_glue else face_area * strength
            )
            capacities.append(face_area * strength)
        key_all = np.concatenate(keys)
        area_all = np.concatenate(areas)
        glue_all = np.concatenate(glue_areas)
        rest_all = np.concatenate(rest_capacities)
        capacity_all = np.concatenate(capacities)
        merged_key, inverse = np.unique(key_all, return_inverse=True)
        area = np.zeros(merged_key.shape[0], dtype=np.float64)
        glue_area = np.zeros_like(area)
        rest_capacity = np.zeros_like(area)
        capacity = np.zeros_like(area)
        np.add.at(area, inverse, area_all)
        np.add.at(glue_area, inverse, glue_all)
        np.add.at(rest_capacity, inverse, rest_all)
        np.add.at(capacity, inverse, capacity_all)
        edges = {
            "src": (merged_key // (ground_node + 1)).astype(np.int64),
            "dst": (merged_key % (ground_node + 1)).astype(np.int64),
            "area": area,
            "glue_area": glue_area,
            "rest_capacity": rest_capacity,
            "capacity": capacity,
        }
        return edges, spall_open

    @staticmethod
    def _make_plate_levels(
        solid: "torch.Tensor", partition: "torch.Tensor"
    ) -> "torch.Tensor":
        """Return each plate's mean solid height (unit: cells).

        Args:
            solid: The (X, Y, Z) solid cells of the volume.
            partition: The (X, Y, Z) compact plate index of every cell.
        """
        import torch

        plate_num = int(partition.max()) + 1
        level = torch.arange(
            solid.shape[2], dtype=torch.float32, device=solid.device
        ).expand(solid.shape)
        count = (
            torch.bincount(partition[solid], minlength=plate_num)
            .to(torch.float32)
            .clamp(min=1.0)
        )
        mean_level = (
            torch.zeros(plate_num, dtype=torch.float32, device=solid.device).index_add_(
                0, partition[solid], level[solid]
            )
            / count
        )
        return mean_level

    @staticmethod
    def _shift_mask(
        mask: "torch.Tensor", step: Tuple[int, int, int], fill: bool
    ) -> "torch.Tensor":
        """Return the bool mask shifted by one step, the gap filled.

        Args:
            mask: The (X, Y, Z) bool tensor to shift.
            step: The (x, y, z) single-cell shift.
            fill: The value the vacated border takes.
        """
        import torch

        out = torch.full_like(mask, fill)
        source = tuple(
            slice(max(part, 0), size + min(part, 0))
            for part, size in zip(step, mask.shape)
        )
        target = tuple(
            slice(max(-part, 0), size + min(-part, 0))
            for part, size in zip(step, mask.shape)
        )
        out[target] = mask[source]
        return out

    def _hash01(self, values: "torch.Tensor", path: str) -> "torch.Tensor":
        """Return a value in [0, 1) hashed from each entry.

        Args:
            values: The int64 entries to hash.
            path: The stream path of this hash, salting it per stone.
        """
        import torch

        mask = 0xFFFFFFFF
        salt = int(self.rng.split(path).integers(0, 2**31))
        mixed = ((values & mask) * 2654435761) & mask
        mixed = mixed ^ salt
        mixed = mixed ^ (mixed >> 13)
        mixed = (mixed * 0x5BD1E995) & mask
        mixed = mixed ^ (mixed >> 15)
        return (mixed & mask).to(torch.float32) / float(2**32)

    @staticmethod
    def _make_frame(break_dir: Tuple[float, float, float]):
        """Return the plate frame rows: thin, long, and wide directions.

        The long side runs down the weak plane's steepest slope, so
        plates hang the way gravity-broken slate does.

        Args:
            break_dir: The (x, y, z) weak direction to build from.
        """
        direction = np.asarray(break_dir, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length < 1.0e-9:
            raise ValueError("break_dir must not be the zero vector")
        thin = direction / length
        slope = np.array([0.0, 0.0, 1.0]) - thin * thin[2]
        if float(np.linalg.norm(slope)) < 1.0e-6:
            slope = np.array([1.0, 0.0, 0.0]) - thin * thin[0]
        long_side = slope / np.linalg.norm(slope)
        wide = np.cross(thin, long_side)
        return tuple(thin), tuple(long_side), tuple(wide)
