# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the pile generator and its wear, named after what they pin."""

import cc3d
import numpy as np
import torch

from rise_origin.create.algorithm import (
    GridSpan,
    PileErosion,
    PileGenerator,
    SegmentTone,
    Volume,
)
from rise_origin.create.protocol.rng import Rng

STONE_SIZE = 0.06
CELL_SIZE = 0.01


def make_pile(origin=(0.0, 0.0, 0.0), cell_num=(48, 48, 24), **arguments):
    """Return a small packed pile filling most of its grid."""
    grid = GridSpan(origin, CELL_SIZE, cell_num)
    settings = dict(
        material_shares=(1.0,),
        stone_size=STONE_SIZE,
        size_spread_ratio=0.3,
        seed_num_high=1,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        cube_cell_num=5,
        shape="box",
        box_size=(0.4, 0.4, 0.18),
    )
    settings.update(arguments)
    return PileGenerator(grid, "cpu", Rng(3), **settings)


def make_worn(step_num=12, **arguments):
    """Return a pile worn by the wear the bed kinds run."""
    volume = make_pile(**arguments).generate()
    PileErosion(
        Rng(7),
        step_num=step_num,
        ball_radius=0.03,
        wear_pace=0.25,
        flat_weight=0.4,
        round_weight=1.2,
        hardness_rows=(1.0, 1.0),
    ).erode(volume)
    return volume


def count_welded_faces(volume):
    """Return how many faces still weld two different stones."""
    segment = volume.segment_id
    solid = volume.get_solid_cells()
    total = 0
    for side in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[side] = slice(None, -1)
        high[side] = slice(1, None)
        total += int(
            (
                solid[tuple(low)]
                & solid[tuple(high)]
                & (segment[tuple(low)] != segment[tuple(high)])
            ).sum()
        )
    return total


class TestPileGenerator:
    def test_the_pile_is_born_packed_with_labeled_stones(self):
        volume = make_pile().generate()
        solid = volume.get_solid_cells()
        segment = volume.segment_id
        assert int(solid.sum()) > 0
        # Every solid cell belongs to a stone, and no air stands inside.
        assert not bool((solid & (segment < 0)).any())
        # Measured over the core, since the rim is where whole stones
        # the box only clipped were dropped.
        core = solid.numpy()[16:32, 16:32, 6:12]
        assert core.mean() > 0.98
        assert int(segment[solid].max()) + 1 > 20

    def test_peytavie_region_identity(self):
        # Two grids whose origins differ by whole lattice cells must
        # read the same seeds and cut the same stones where they meet.
        shift_num = 2 * int(round(STONE_SIZE / CELL_SIZE))
        near = make_pile(origin=(0.0, 0.0, 0.0))
        far = make_pile(origin=(shift_num * CELL_SIZE, 0.0, 0.0))
        near_seeds = near._make_seed_lattice()
        far_seeds = far._make_seed_lattice()
        near_labels = near._make_stone_labels(near_seeds)[shift_num:, :, :]
        far_labels = far._make_stone_labels(far_seeds)[:-shift_num, :, :]
        assert near_labels.shape == far_labels.shape
        near_point = near_seeds.point.reshape(-1, 3)[torch.unique(near_labels)]
        far_point = far_seeds.point.reshape(-1, 3)[torch.unique(far_labels)]
        assert near_point.shape == far_point.shape
        assert torch.allclose(near_point, far_point, atol=1.0e-6)
        # The two labelings name the stones differently, so the partition
        # matches when the label pairs form one-to-one rows.
        pairs = torch.unique(
            torch.stack([near_labels.reshape(-1), far_labels.reshape(-1)], dim=1),
            dim=0,
        )
        assert pairs.shape[0] == int(torch.unique(pairs[:, 0]).shape[0])
        assert pairs.shape[0] == int(torch.unique(pairs[:, 1]).shape[0])

    def test_the_bank_holds_one_entry_per_region_configuration(self):
        bank = make_pile()._region_bank
        assert bank["vertex"]["seed_num"].shape[0] == 2
        assert bank["edge"]["seed_num"].shape[0] == 12
        assert bank["face"]["seed_num"].shape[0] == 48
        assert bank["center"]["seed_num"].shape[0] == 256

    def test_every_lattice_cell_carries_a_seed(self):
        # The four region kinds partition the corner cube, so no cell
        # may be left unseeded by the fill order.
        seeds = make_pile()._make_seed_lattice()
        assert bool((seeds.seed_num > 0).all())

    def test_a_cube_smaller_than_its_regions_is_refused(self):
        try:
            make_pile(cube_cell_num=4)
        except ValueError:
            return
        raise AssertionError("a four-cell corner cube should be refused")

    def test_wider_coefficients_stretch_the_stones(self):
        # The paper's flat plates of Figure 13 against its round rocks.
        assert _make_mean_stretch(make_pile(alpha=5.0, beta=5.0).generate()) > (
            _make_mean_stretch(make_pile().generate())
        )

    def test_more_seeds_per_cell_make_more_and_smaller_stones(self):
        few = make_pile(seed_num_high=1).generate()
        many = make_pile(seed_num_high=8).generate()
        assert _count_stones(many) > _count_stones(few)

    def test_material_shares_split_the_stones(self):
        volume = make_pile(material_shares=(0.5, 0.5)).generate()
        solid = volume.get_solid_cells()
        assert set(torch.unique(volume.material_id[solid]).tolist()) == {1, 2}
        # A stone never splits between two materials.
        segment = volume.segment_id
        for stone in torch.unique(segment[solid])[:40].tolist():
            picked = solid & (segment == stone)
            assert len(torch.unique(volume.material_id[picked])) == 1

    def test_stones_outside_the_shape_are_dropped_whole(self):
        # Raising the keep share drops the stones the box only clips.
        loose = make_pile(stone_keep_share=0.1).generate()
        strict = make_pile(stone_keep_share=0.9).generate()
        assert _count_stones(strict) < _count_stones(loose)

    def test_contacts_are_marked_between_touching_stones(self):
        volume = make_pile().generate()
        contact = (volume.structure_mark & Volume.CONTACT_MARK) != 0
        seam = (volume.structure_mark & Volume.SEAM_MARK) != 0
        solid = volume.get_solid_cells()
        assert int(contact.sum()) > 0
        assert int(seam.sum()) > 0
        # Contacts sit on the boundaries between stones, not inside one.
        assert int((contact & seam).sum()) > 0
        assert not bool((contact & ~solid).any())


class TestPileErosion:
    def test_peytavie_gap_opening(self):
        born = make_pile().generate()
        worn = make_worn()
        born_solid = born.get_solid_cells()
        worn_solid = worn.get_solid_cells()
        # The gaps open: most of what welded two stones is now air.
        assert count_welded_faces(worn) < 0.25 * count_welded_faces(born)
        # The pile keeps its stones, each still one piece of real size.
        assert _count_stones(worn) == _count_stones(born)
        shares = _make_main_piece_shares(worn)
        assert float(np.mean(shares)) > 0.9
        # The contacts hold, which is what keeps the pile resting.
        contact = (worn.structure_mark & Volume.CONTACT_MARK) != 0
        assert int((contact & worn_solid).sum()) == int((contact & born_solid).sum())
        # Air stands between the stones where none stood before.
        assert _make_inner_air_share(worn) > _make_inner_air_share(born) + 0.15

    def test_a_stone_rounds_inside_a_solid_pile(self):
        # Not one cell is air, and the second stone's convex edge still
        # wears back while its flat boundary holds. This is what the
        # ball measurement against air cannot do.
        volume = _make_packed_pair()
        PileErosion(
            Rng(3),
            step_num=12,
            ball_radius=0.2,
            wear_pace=0.4,
            flat_weight=0.0,
            round_weight=2.0,
            hardness_rows=(1.0, 1.0),
            durability_sway_ratio=0.0,
        ).erode(volume)
        solid = volume.get_solid_cells()
        assert not bool(solid[12, 12, 12])
        assert bool(solid[12, 18, 12])

    def test_a_neighbor_stone_reads_the_way_air_does(self):
        packed = _make_packed_pair()
        offsets = PileErosion(
            Rng(1), 1, 0.2, 1.0, 1.0, 1.0, (1.0, 1.0)
        )._make_ball_offsets(packed)
        packed_share = PileErosion._make_own_share(
            packed.segment_id, packed.get_solid_cells(), offsets
        )
        alone = _make_packed_pair()
        alone.material_id = torch.where(
            alone.segment_id == 1,
            torch.zeros_like(alone.material_id),
            alone.material_id,
        )
        alone_share = PileErosion._make_own_share(
            alone.segment_id, alone.get_solid_cells(), offsets
        )
        # A cell on the first stone's flat boundary reads the same
        # whether the second stone stands there or open air does.
        assert abs(float(packed_share[11, 18, 12]) - float(alone_share[11, 18, 12])) < (
            1.0e-6
        )

    def test_contacts_hold_against_pile_wear(self):
        volume = make_pile().generate()
        contact = (volume.structure_mark & Volume.CONTACT_MARK) != 0
        PileErosion(
            Rng(7),
            step_num=30,
            ball_radius=0.03,
            wear_pace=0.5,
            flat_weight=1.0,
            round_weight=1.0,
            hardness_rows=(1.0, 1.0),
        ).erode(volume)
        assert bool((volume.get_solid_cells() | ~contact).all())

    def test_more_steps_wear_more(self):
        light = int(make_worn(step_num=6).get_solid_cells().sum())
        heavy = int(make_worn(step_num=14).get_solid_cells().sum())
        assert heavy < light

    def test_a_volume_without_stones_is_refused(self):
        volume = Volume(GridSpan((0.0, 0.0, 0.0), CELL_SIZE, (8, 8, 8)), "cpu")
        volume.material_id[:] = 1
        try:
            PileErosion(Rng(1), 1, 0.02, 1.0, 1.0, 1.0, (1.0, 1.0)).erode(volume)
        except ValueError:
            return
        raise AssertionError("pile wear should refuse a volume with no segments")


class TestSegmentTone:
    def test_one_stone_carries_one_tone(self):
        volume = make_pile().generate()
        SegmentTone.make_tones(volume, ((), tuple(range(6))), 0.5, 0.6, 0.25, Rng(9))
        solid = volume.get_solid_cells()
        segment = volume.segment_id
        for stone in torch.unique(segment[solid])[:60].tolist():
            picked = solid & (segment == stone)
            assert len(torch.unique(volume.optical_mid[picked])) == 1

    def test_the_bias_moves_the_bed_down_the_run(self):
        pale = make_pile().generate()
        dark = make_pile().generate()
        SegmentTone.make_tones(pale, ((), tuple(range(6))), 0.0, 0.2, 0.25, Rng(9))
        SegmentTone.make_tones(dark, ((), tuple(range(6))), 1.0, 0.2, 0.25, Rng(9))
        assert float(dark.optical_mid[dark.get_solid_cells()].float().mean()) > float(
            pale.optical_mid[pale.get_solid_cells()].float().mean()
        )

    def test_the_bed_spreads_over_its_run(self):
        volume = make_pile().generate()
        SegmentTone.make_tones(volume, ((), tuple(range(6))), 0.5, 0.9, 0.25, Rng(9))
        solid = volume.get_solid_cells()
        assert len(torch.unique(volume.optical_mid[solid])) > 2


def _make_packed_pair(cell_num=(24, 24, 24)):
    """Return a wholly filled volume split into two stones.

    Not one cell is air, and the second stone takes the quarter beyond
    (12, 12), so it carries a convex edge running into the first stone
    at (12, 12, z) and flat boundaries away from it.
    """
    grid = GridSpan((0.0, 0.0, 0.0), 0.05, cell_num)
    volume = Volume(grid, "cpu")
    volume.material_id[:] = 1
    segment = torch.zeros(cell_num, dtype=torch.int32)
    segment[12:, 12:, :] = 1
    volume.segment_id = segment
    return volume


def _count_stones(volume):
    """Return how many stones hold cells in the volume."""
    solid = volume.get_solid_cells()
    return int(torch.unique(volume.segment_id[solid]).shape[0])


def _make_main_piece_shares(volume):
    """Return each stone's share of cells left in its largest piece."""
    segment = volume.segment_id.numpy()
    solid = volume.get_solid_cells().numpy()
    shares = []
    for stone in np.unique(segment[solid]):
        labels = cc3d.connected_components(solid & (segment == stone), connectivity=6)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        shares.append(sizes.max() / sizes.sum())
    return shares


def _make_inner_air_share(volume):
    """Return the air share inside the pile's own bounding box."""
    solid = volume.get_solid_cells().numpy()
    cells = np.argwhere(solid)
    low = cells.min(axis=0)
    high = cells.max(axis=0) + 1
    return 1.0 - solid[low[0] : high[0], low[1] : high[1], low[2] : high[2]].mean()


def _make_mean_stretch(volume):
    """Return the mean long-over-short principal span of the stones."""
    segment = volume.segment_id
    solid = volume.get_solid_cells()
    found = []
    for stone in torch.unique(segment[solid])[:120].tolist():
        cells = torch.nonzero(solid & (segment == stone)).to(torch.float64)
        if cells.shape[0] < 30:
            continue
        centered = cells - cells.mean(dim=0)
        spread = torch.linalg.eigvalsh(centered.T @ centered / cells.shape[0]).clamp(
            min=1.0e-9
        )
        found.append(float((spread.max() / spread.min()).sqrt()))
    return float(np.mean(found))
