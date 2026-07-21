# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""In-engine bytes archive tests, driven through pickle (RS_RSCArchive::toBytes/fromBytes)."""

import hashlib
import os
import pickle
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from _serialization_common import body, pool_type_names, rich_world, state
from rise_origin import rsc


def _reload(node):
    """Round trip a node through the bytes archive."""
    return pickle.loads(pickle.dumps(node))


# ======================================================================================
# Fidelity
# ======================================================================================
def test_rich_world_round_trips():
    world = rich_world()
    restored = _reload(world)
    # The archive is deterministic, so a faithful round trip reproduces the exact bytes.
    assert state(restored) == state(world)


def test_every_variant_section_survives():
    world = _reload(rich_world())
    # Each variant section holds its representative concrete member after the round trip.
    assert world.resources[0].name == "t"
    assert world.essences[0].name == "steel"
    assert world.entities[0].name == "s1"
    assert world.records[0].name == "zone"
    assert world.signals[0].name == "push"
    assert world.recorders[0].name == "rec"
    assert world.controllers[0].name == "brain"
    assert world.simulation.vision.denoise.sigma_color == pytest.approx(2.0)


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


def test_shared_body_is_pooled_once():
    count = 8
    shared_bytes = pickle.dumps(_shared_world(count))
    unique_bytes = pickle.dumps(_unique_world(count))
    # The shared archive pools the one body once, so it is far smaller than the unique archive.
    assert len(shared_bytes) * 2 < len(unique_bytes)
    # The restored shared world still pools exactly one body.
    restored = pickle.loads(shared_bytes)
    assert pool_type_names(state(restored)).count("RSC_StructureVoxelBody") == 1


# ======================================================================================
# Determinism and independence
# ======================================================================================
def test_dumps_is_byte_deterministic():
    world = rich_world()
    assert pickle.dumps(world) == pickle.dumps(world)


def test_round_trip_is_stable():
    world = rich_world()
    once = pickle.dumps(world)
    twice = pickle.dumps(pickle.loads(once))
    assert hashlib.sha256(once).hexdigest() == hashlib.sha256(twice).hexdigest()


def test_unpickled_object_is_independent():
    world = rich_world()
    restored = _reload(world)
    restored.entities[0].bodies[0].voxel_size = 0.999
    # The archive produced a deep copy, so the original keeps its value.
    assert world.entities[0].bodies[0].voxel_size == pytest.approx(0.01)


# ======================================================================================
# Degenerate cases
# ======================================================================================
def test_empty_world_round_trips():
    world = rsc.World("empty")
    assert state(_reload(world)) == state(world)


def test_structure_without_bodies_round_trips():
    structure = rsc.entity.StructureVoxel("hollow", material_references=["m"])
    world = rsc.World("w", essences=[rsc.essence.Material("m")], entities=[structure])
    assert state(_reload(world)) == state(world)


def test_simulation_only_world_round_trips():
    world = rsc.World("sim", simulation=rsc.simulation.Simulation(seed=42))
    restored = _reload(world)
    assert restored.simulation.seed == 42
    assert state(restored) == state(world)


# ======================================================================================
# Pickle protocols and zero-copy out-of-band buffers
# ======================================================================================
def test_default_protocol_round_trips():
    # The default protocol (below 5 on 3.10) carries a plain bytes state.
    world = rich_world()
    restored = pickle.loads(pickle.dumps(world))
    assert state(restored) == state(world)


def test_protocol_5_round_trips():
    world = rich_world()
    restored = pickle.loads(pickle.dumps(world, protocol=5))
    assert state(restored) == state(world)


def test_out_of_band_buffers_are_zero_copy():
    world = _shared_world(4)
    captured = []
    data = pickle.dumps(world, protocol=5, buffer_callback=captured.append)
    # The archive bytes leave the main stream and ride out of band, so the stream stays tiny.
    assert len(captured) == 1
    assert len(data) < len(captured[0].raw())
    restored = pickle.loads(data, buffers=[buffer.raw() for buffer in captured])
    assert state(restored) == state(world)


# ======================================================================================
# Speed
# ======================================================================================
@pytest.mark.perf
def test_large_world_archive_speed():
    world = _shared_world(64)
    start = time.perf_counter()
    for _ in range(10):
        pickle.loads(pickle.dumps(world))
    elapsed = time.perf_counter() - start
    # A generous ceiling that only fires on a large regression.
    assert elapsed < 10.0, f"archive round trips took {elapsed:.2f}s"
