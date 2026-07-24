# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Durable .rsct and .rscb file format tests: fidelity, interchange, bulk, sharing, memory."""

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import rise_origin as rs
from _serialization_common import body, pool_entry_count, pool_type_names, rich_world, state
from rise_origin import rsc

# The uint32 null sentinel that bulk index arrays use for an absent reference.
_NULL_SID = 0xFFFFFFFF

# The three durable variants under test: text, compressed binary, and uncompressed binary.
_VARIANTS = [
    ("world.rsct", {}),
    ("world.rscb", {"is_compressing": True}),
    ("world.rscb", {"is_compressing": False}),
]


def _save_load(node_type, world, tmp_path, filename, **save_kwargs):
    """Save a world then load it back as node_type from a fresh path."""
    path = os.path.join(tmp_path, filename)
    world.save(path, **save_kwargs)
    return node_type.load(path)


# ======================================================================================
# Round trip fidelity across formats and compression
# ======================================================================================
@pytest.mark.parametrize("filename,save_kwargs", _VARIANTS)
def test_round_trip_is_lossless_across_formats(tmp_path, filename, save_kwargs):
    world = rich_world()
    # The text format is the normalization reference; every variant must load to the same bytes.
    reference = _save_load(rsc.World, world, tmp_path, "reference.rsct")
    loaded = _save_load(rsc.World, world, tmp_path, filename, **save_kwargs)
    assert state(loaded) == state(reference)


def test_round_trip_preserves_field_values(tmp_path):
    world = rich_world()
    loaded = _save_load(rsc.World, world, tmp_path, "world.rscb")

    assert loaded.name == "rich"
    assert [e.name for e in loaded.entities] == ["s1", "s2", "floor", "cam", "stereo", "ray", "sky"]
    assert loaded.simulation.seed == 11
    assert list(loaded.entities[0].material_references) == ["steel"]

    body_a = loaded.entities[0].bodies[0]
    assert body_a.voxel_size == pytest.approx(0.01)
    assert loaded.entities[0].bodies[1].voxel_size == pytest.approx(0.02)

    ray = loaded.entities[5]
    got = [(v.x, v.y, v.z) for v in ray.ray_directions]
    assert got == pytest.approx([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
    assert ray.max_range == pytest.approx(50.0)

    assert list(loaded.entities[2].height) == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert loaded.entities[3].image_width == 64
    assert loaded.simulation.vision.denoise.sigma_color == pytest.approx(2.0)


def test_idempotent_after_first_load(tmp_path):
    world = rich_world()
    once = _save_load(rsc.World, world, tmp_path, "once.rscb")
    twice = _save_load(rsc.World, once, tmp_path, "twice.rscb")
    assert state(twice) == state(once)


# ======================================================================================
# Interchange
# ======================================================================================
def test_formats_are_mutually_interchangeable(tmp_path):
    world = rich_world()
    text = _save_load(rsc.World, world, tmp_path, "a.rsct")
    binary = _save_load(rsc.World, world, tmp_path, "a.rscb")
    plain_binary = _save_load(rsc.World, world, tmp_path, "a_plain.rscb", is_compressing=False)
    assert state(text) == state(binary) == state(plain_binary)

    # A binary save re-saved as text, and the reverse, stay stable.
    text_from_binary = _save_load(rsc.World, binary, tmp_path, "from_binary.rsct")
    binary_from_text = _save_load(rsc.World, text, tmp_path, "from_text.rscb")
    assert state(text_from_binary) == state(binary_from_text) == state(text)


# ======================================================================================
# Bulk array correctness
# ======================================================================================
def test_bulk_arrays_round_trip_element_wise(tmp_path):
    # A body whose index arrays hold null sentinels, an enum vector, and a bitmask vector.
    voxel_count = 8
    source = rsc.entity.StructureVoxelBody(
        x_voxels=2,
        y_voxels=2,
        z_voxels=2,
        voxel_size=0.03,
        material_reference_sid=[0, 1, _NULL_SID, 2, 0, 1, _NULL_SID, 2],
        material_optical_reference_mid=[0, _NULL_SID, 1, 1, 2, 2, 0, _NULL_SID],
        segment_bid=[0, 0, 1, 1, 2, 2, 3, 3],
        expansion_control_mode=[0, 1, 0, 1, 0, 1, 0, 1],
        link=[0b0001, 0b0010, 0b0100, 0b1000, 0, 0b1111, 0b1010, 0b0101],
    )
    assert len(list(source.segment_bid)) == voxel_count
    structure = rsc.entity.StructureVoxel("b", material_references=["m"], bodies=[source])
    world = rsc.World("bulk", essences=[rsc.essence.Material("m")], entities=[structure])

    loaded = _save_load(rsc.World, world, tmp_path, "bulk.rscb")
    got = loaded.entities[0].bodies[0]
    assert list(got.material_reference_sid) == [0, 1, _NULL_SID, 2, 0, 1, _NULL_SID, 2]
    assert list(got.material_optical_reference_mid) == [0, _NULL_SID, 1, 1, 2, 2, 0, _NULL_SID]
    assert list(got.segment_bid) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert list(got.expansion_control_mode) == [0, 1, 0, 1, 0, 1, 0, 1]
    assert list(got.link) == [0b0001, 0b0010, 0b0100, 0b1000, 0, 0b1111, 0b1010, 0b0101]


def test_absent_optional_array_stays_empty(tmp_path):
    source = rsc.entity.StructureVoxelBody(
        x_voxels=1, y_voxels=1, z_voxels=1, voxel_size=0.05, segment_bid=[0]
    )
    structure = rsc.entity.StructureVoxel("b", material_references=["m"], bodies=[source])
    world = rsc.World("opt", essences=[rsc.essence.Material("m")], entities=[structure])
    loaded = _save_load(rsc.World, world, tmp_path, "opt.rscb")
    got = loaded.entities[0].bodies[0]
    assert len(list(got.expansion_signal_sid)) == 0
    assert list(got.segment_bid) == [0]


# ======================================================================================
# Sharing and dedup
# ======================================================================================
def _shared_world(count):
    base = rsc.entity.StructureVoxel("h0", material_references=["m"], bodies=[body(6)])
    entities = [base] + [base.shallow_copy(name=f"h{i}") for i in range(1, count)]
    return rsc.World("shared", essences=[rsc.essence.Material("m")], entities=entities)


def _unique_world(count):
    entities = [
        rsc.entity.StructureVoxel(f"u{i}", material_references=["m"], bodies=[body(6)])
        for i in range(count)
    ]
    return rsc.World("unique", essences=[rsc.essence.Material("m")], entities=entities)


@pytest.mark.parametrize("filename,save_kwargs", _VARIANTS)
def test_shared_body_is_stored_once(tmp_path, filename, save_kwargs):
    count = 8
    shared_path = os.path.join(tmp_path, "shared_" + filename)
    unique_path = os.path.join(tmp_path, "unique_" + filename)
    _shared_world(count).save(shared_path, **save_kwargs)
    _unique_world(count).save(unique_path, **save_kwargs)
    # The shared file stores one body; the unique file stores count bodies, so it is far larger.
    assert os.path.getsize(shared_path) * 2 < os.path.getsize(unique_path)


def test_shared_identity_is_restored(tmp_path):
    count = 8
    loaded = _save_load(rsc.World, _shared_world(count), tmp_path, "shared.rscb")
    unique = _save_load(rsc.World, _unique_world(count), tmp_path, "unique.rscb")
    # A shared load pools the one body once; a unique load pools every body, so its pool is larger.
    assert pool_entry_count(state(loaded)) < pool_entry_count(state(unique))
    # The one shared body appears exactly once in the loaded pool.
    assert pool_type_names(state(loaded)).count("RSC_StructureVoxelBody") == 1


# ======================================================================================
# Backward compatibility
# ======================================================================================
def test_committed_data_files_load():
    env = rsc.World.load("data/env.rsct")
    robot = rsc.World.load("data/robot.rsct")
    assert len(env.entities) == 2
    assert len(robot.entities) == 4


def test_old_style_text_without_shared_region_loads(tmp_path):
    # An older file predates the shared pool region and stored every reference inline. Emulate one by
    # inlining the only shared reference (the simulation) back into the body and dropping the shared
    # region, then confirm the reader still loads it through the inline path.
    import json

    path = os.path.join(tmp_path, "legacy.rsct")
    rsc.World("legacy", entities=[rsc.entity.FloorFlat("ground", height=0.5)]).save(path)
    with open(path) as handle:
        document = json.load(handle)
    marker = document["body"]["simulation"]
    assert "$shared" in marker
    document["body"]["simulation"] = document["shared"][str(marker["$shared"])]["data"]
    document.pop("shared")
    with open(path, "w") as handle:
        json.dump(document, handle)

    loaded = rsc.World.load(path)
    assert loaded.name == "legacy"
    assert loaded.entities[0].name == "ground"


# ======================================================================================
# Compression size
# ======================================================================================
def test_compression_shrinks_bytes(tmp_path):
    world = _shared_world(4)
    compressed = os.path.join(tmp_path, "c.rscb")
    plain = os.path.join(tmp_path, "p.rscb")
    text = os.path.join(tmp_path, "t.rsct")
    world.save(compressed, is_compressing=True)
    world.save(plain, is_compressing=False)
    world.save(text)
    assert os.path.getsize(compressed) < os.path.getsize(plain)
    assert os.path.getsize(plain) <= os.path.getsize(text)


def test_incompressible_body_still_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    values = rng.integers(0, 2**31, size=64, dtype=np.uint32)
    source = rsc.entity.StructureVoxelBody(
        x_voxels=4,
        y_voxels=4,
        z_voxels=4,
        voxel_size=0.01,
        segment_bid=[0] * 64,
        material_reference_sid=[int(v) for v in values],
    )
    structure = rsc.entity.StructureVoxel("r", material_references=["m"], bodies=[source])
    world = rsc.World("rand", essences=[rsc.essence.Material("m")], entities=[structure])
    loaded = _save_load(rsc.World, world, tmp_path, "rand.rscb", is_compressing=True)
    assert list(loaded.entities[0].bodies[0].material_reference_sid) == [int(v) for v in values]


# ======================================================================================
# Save memory bound (the E.2 gate)
# ======================================================================================
_MEMORY_SCRIPT = r"""
import ctypes, gc, os, sys, tempfile
import numpy as np
import rise_origin as rs
from rise_origin import rsc

def peak_rss():
    # VmHWM is the process peak resident set, in kibibytes.
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return 0

voxels = int(sys.argv[1])
compress = sys.argv[2] == "1"

occ = np.ones((voxels, voxels, voxels), dtype=bool)
b = rsc.entity.StructureVoxelBody.from_occupancy(
    occ, material_sid=0, segment_sid=0, voxel_size=0.01
)
count = len(list(b.segment_bid))
# The bulk arrays a full body populates, each a uint32 per voxel.
array_bytes = count * 4 * 3
s = rsc.entity.StructureVoxel("m", material_references=["m"], bodies=[b])
w = rsc.World("mem", essences=[rsc.essence.Material("m")], entities=[s])
del occ
gc.collect()
# Return the build's freed pages to the OS and reset the peak mark, so the measurement below
# captures only the save's transient allocation rather than the larger build high-water.
try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass
try:
    with open("/proc/self/clear_refs", "w") as handle:
        handle.write("5")
except Exception:
    pass

before = peak_rss()
path = os.path.join(tempfile.mkdtemp(), "mem.rscb")
w.save(path, is_compressing=compress)
after = peak_rss()
print(array_bytes, max(0, after - before))
"""


@pytest.mark.perf
@pytest.mark.parametrize("compress", ["1", "0"])
def test_save_peak_memory_within_one_times_arrays(tmp_path, compress):
    voxels = 112
    script = os.path.join(tmp_path, "mem_probe.py")
    with open(script, "w") as handle:
        handle.write(_MEMORY_SCRIPT)
    result = subprocess.run(
        [sys.executable, script, str(voxels), compress],
        capture_output=True,
        text=True,
        check=True,
    )
    array_bytes, delta_bytes = (int(value) for value in result.stdout.split())
    # The save must not hold more than one full copy of the body's array bytes at its peak.
    assert delta_bytes <= array_bytes, f"peak {delta_bytes} exceeded 1x of {array_bytes}"


# ======================================================================================
# Errors and edges
# ======================================================================================
def test_unknown_extension_raises(tmp_path):
    with pytest.raises(Exception):
        rsc.World("x").save(os.path.join(tmp_path, "world.bad"))


def test_load_type_mismatch_raises(tmp_path):
    path = os.path.join(tmp_path, "voxel.rscb")
    structure = rsc.entity.StructureVoxel("only", material_references=["m"], bodies=[body(2)])
    rsc.World("w", essences=[rsc.essence.Material("m")], entities=[structure]).save(path)
    with pytest.raises(Exception):
        rsc.entity.StructureVoxel.load(path)


def test_missing_file_raises():
    with pytest.raises(Exception):
        rsc.World.load("does_not_exist_12345.rscb")


def test_empty_world_round_trips(tmp_path):
    loaded = _save_load(rsc.World, rsc.World("empty"), tmp_path, "empty.rscb")
    assert loaded.name == "empty"
    assert len(loaded.entities) == 0


def test_unicode_name_survives(tmp_path):
    loaded = _save_load(rsc.World, rsc.World("wörld_名前_🌍"), tmp_path, "u.rscb")
    assert loaded.name == "wörld_名前_🌍"


def test_extreme_finite_floats_survive(tmp_path):
    world = rsc.World("floats", entities=[rsc.entity.FloorFlat("f", height=3.4e38)])
    loaded = _save_load(rsc.World, world, tmp_path, "floats.rscb")
    assert loaded.entities[0].height == pytest.approx(3.4e38)
