# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.world: variant auto-wrapping and assembly."""

import pickle

import rise_origin as rs
from rise_origin import rsc


def test_empty_world_has_default_simulation():
    world = rsc.World("empty")
    assert world.name == "empty"
    assert len(world.resources) == 0
    assert isinstance(world.simulation, rs.RSC_Simulation)


def test_builders_are_wrapped_into_variants():
    world = rsc.World(
        "scene",
        resources=[rsc.resource.TextureConstant("sky", value=(0.5, 0.7, 0.9, 1.0))],
        essences=[rsc.essence.Material("rubber")],
        entities=[rsc.entity.FloorFlat("ground")],
        records=[rsc.record.IntersectRegionSphere("zone", radius=1.0)],
        signals=[rsc.signal.Force("push", structure_name="dog")],
        recorders=[rsc.recorder.RecorderHDF5("rec", path="/tmp/o.h5")],
        controllers=[rsc.controller.ControllerCallback("brain")],
    )
    assert len(world.resources) == 1
    assert isinstance(world.resources[0], rs.RSC_ResourceVariant)
    assert world.resources[0].value.name == "sky"
    assert world.entities[0].value.name == "ground"
    assert world.records[0].value.name == "zone"
    assert world.controllers[0].value.name == "brain"


def test_world_round_trips_through_pickle():
    world = rsc.World(
        "scene",
        entities=[rsc.entity.StructureVoxel("dog")],
        simulation=rsc.simulation.Simulation(seed=3),
    )
    restored = pickle.loads(pickle.dumps(world))
    assert restored.name == "scene"
    assert restored.entities[0].value.name == "dog"
    assert restored.simulation.seed == 3
