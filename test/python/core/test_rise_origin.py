# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for RiseOrigin and RS_WorldHandle.

The tests walk the lifecycle of using the engine, in order:

1. Constructing a RiseOrigin engine.
2. Validating a world, and surfacing validation failures as typed exceptions,
   before anything runs.
3. Running a world: letting it finish on its own and stopping one early.
4. Driving a running world through its handle, where invalid input is rejected.
5. Guarding a world that has ended, where every handle call short-circuits.

The headless engine steps with no real-time pacing, so a world cannot be parked
in the running state from the test thread. Tests that need a finished world give
it a small max_time and wait for it to end; tests that need a live world hand it
the default max_time and stop it with terminate. Every test builds its own world
inline; there are no shared fixtures or helpers by design.
"""

import pytest

import rise_origin as rs
from rise_origin import rsc


# ---------------------------------------------------------------------------
# 1. Constructing the engine
# ---------------------------------------------------------------------------


def test_constructs_engine_with_default_device():
    assert rs.RiseOrigin() is not None


def test_constructs_engine_from_device_string():
    assert rs.RiseOrigin("cpu") is not None


# ---------------------------------------------------------------------------
# 2. Validating a world before it runs
# ---------------------------------------------------------------------------


def test_validator_accepts_a_valid_world():
    world = rsc.World(
        "valid",
        essences=[
            rsc.essence.Material("m"),
            rsc.essence.MaterialPhysicalPropertyRigid(
                "mp", material_reference="m", density=1000.0
            ),
        ],
    )
    rs.RS_RSCValidator().validate(world)  # must not raise


def test_validator_rejects_an_invalid_world():
    world = rsc.World(
        "invalid",
        essences=[
            rsc.essence.Material("m"),
            rsc.essence.MaterialPhysicalPropertyRigid(
                "mp", material_reference="m", density=-1.0
            ),
        ],
    )
    with pytest.raises(rs.RS_InvalidCharacterizationException):
        rs.RS_RSCValidator().validate(world)


def test_invalid_characterization_is_a_data_exception():
    # The two registered exceptions form the same hierarchy as their C++ types,
    # so catching the broad RS_DataException catches either failure.
    assert issubclass(rs.RS_InvalidCharacterizationException, rs.RS_DataException)


# ---------------------------------------------------------------------------
# 3. Running a world and stopping it
# ---------------------------------------------------------------------------


def test_world_runs_to_completion_on_its_own():
    world = rsc.World(
        "completion",
        essences=[
            rsc.essence.Material("m"),
            rsc.essence.MaterialPhysicalPropertyRigid(
                "mp", material_reference="m", density=1000.0
            ),
        ],
        entities=[
            rsc.entity.StructureVoxel(
                "cube",
                origin_position=(0.0, 0.0, 0.2),
                material_references=["m"],
                bodies=[rsc.entity.StructureVoxelBody.box(2, 2, 2, voxel_size=0.01)],
            )
        ],
        simulation=rsc.simulation.Simulation(
            runtime=rsc.simulation.Runtime(max_time=0.05, log_level="warn")
        ),
    )
    origin = rs.RiseOrigin("cpu")
    handle = origin.hello(world)
    handle.wait_for_end()
    assert handle.is_end() is True
    assert handle.is_running() is False
    assert handle.get_status() == rs.RSE_WorldStatus.RSE_WORLD_END_SUCCESS


def test_terminate_ends_a_running_world():
    world = rsc.World(
        "terminate",
        essences=[
            rsc.essence.Material("m"),
            rsc.essence.MaterialPhysicalPropertyRigid(
                "mp", material_reference="m", density=1000.0
            ),
        ],
        entities=[
            rsc.entity.StructureVoxel(
                "cube",
                origin_position=(0.0, 0.0, 0.2),
                material_references=["m"],
                bodies=[rsc.entity.StructureVoxelBody.box(2, 2, 2, voxel_size=0.01)],
            )
        ],
        simulation=rsc.simulation.Simulation(
            runtime=rsc.simulation.Runtime(log_level="warn")
        ),
    )
    origin = rs.RiseOrigin("cpu")
    handle = origin.hello(world)
    assert handle.wait_for_running() is True
    handle.terminate()
    handle.wait_for_end()
    assert handle.is_end() is True
    assert handle.is_running() is False


# ---------------------------------------------------------------------------
# 4. Driving a running world through the handle
# ---------------------------------------------------------------------------


def test_adding_an_invalid_characterization_raises():
    world = rsc.World(
        "invalid_add",
        essences=[
            rsc.essence.Material("m"),
            rsc.essence.MaterialPhysicalPropertyRigid(
                "mp", material_reference="m", density=1000.0
            ),
        ],
        entities=[
            rsc.entity.StructureVoxel(
                "cube",
                origin_position=(0.0, 0.0, 0.2),
                material_references=["m"],
                bodies=[rsc.entity.StructureVoxelBody.box(2, 2, 2, voxel_size=0.01)],
            )
        ],
        simulation=rsc.simulation.Simulation(
            runtime=rsc.simulation.Runtime(log_level="warn")
        ),
    )
    origin = rs.RiseOrigin("cpu")
    handle = origin.hello(world)
    assert handle.wait_for_running() is True
    try:
        # The handle validates a characterization before forwarding it, so an
        # invalid one (a sphere record with non-positive radius) raises here.
        with pytest.raises(rs.RS_InvalidCharacterizationException):
            handle.add_record(
                rs.RSC_RecordVariant(rsc.record.IntersectRegionSphere("bad", radius=-1.0))
            )
    finally:
        handle.terminate()
        handle.wait_for_end()


# ---------------------------------------------------------------------------
# 5. Guarding a world that has ended
# ---------------------------------------------------------------------------


def test_mutators_return_null_after_the_world_ends():
    world = rsc.World(
        "mutators",
        essences=[
            rsc.essence.Material("m"),
            rsc.essence.MaterialPhysicalPropertyRigid(
                "mp", material_reference="m", density=1000.0
            ),
        ],
        entities=[
            rsc.entity.StructureVoxel(
                "cube",
                origin_position=(0.0, 0.0, 0.2),
                material_references=["m"],
                bodies=[rsc.entity.StructureVoxelBody.box(2, 2, 2, voxel_size=0.01)],
            )
        ],
        simulation=rsc.simulation.Simulation(
            runtime=rsc.simulation.Runtime(max_time=0.05, log_level="warn")
        ),
    )
    origin = rs.RiseOrigin("cpu")
    handle = origin.hello(world)
    handle.wait_for_end()
    assert handle.is_end() is True

    null = rs.RS_NULL_INDEX
    assert handle.set_simulation(rsc.simulation.Simulation()) == null

    assert handle.add_resource(rs.RSC_ResourceVariant(rsc.resource.TextureConstant("t"))) == null
    assert handle.remove_resource("t") == null

    essence = rs.RSC_EssenceVariant(rsc.essence.Material("m2"))
    assert handle.add_essence(essence) == null
    assert handle.update_essence("m2", essence) == null
    assert handle.remove_essence("m2") == null

    assert handle.add_entity(rs.RSC_EntityVariant(rsc.entity.FloorFlat("floor"))) == null
    assert handle.remove_entity("cube") == null
    assert handle.freeze_entity("cube") == null
    assert handle.unfreeze_entity("cube") == null
    assert handle.is_entity_frozen("cube") is False

    assert handle.add_record(rs.RSC_RecordVariant(rsc.record.IntersectRegionSphere("z", radius=1.0))) == null
    assert handle.remove_record("z") == null

    signal = rs.RSC_SignalVariant(rsc.signal.Force("f", structure_name="cube"))
    assert handle.add_signal(signal) == null
    assert handle.remove_signal("f") == null

    recorder = rs.RSC_RecorderVariant(rsc.recorder.RecorderHDF5("rec", path="/tmp/guard.h5"))
    assert handle.add_recorder(recorder) == null
    assert handle.remove_recorder("rec") == null

    controller = rs.RSC_ControllerVariant(rsc.controller.ControllerCallback("c"))
    assert handle.add_controller(controller) == null
    assert handle.remove_controller("c") == null


def test_finders_return_an_empty_variant_after_the_world_ends():
    world = rsc.World(
        "finders",
        essences=[
            rsc.essence.Material("m"),
            rsc.essence.MaterialPhysicalPropertyRigid(
                "mp", material_reference="m", density=1000.0
            ),
        ],
        entities=[
            rsc.entity.StructureVoxel(
                "cube",
                origin_position=(0.0, 0.0, 0.2),
                material_references=["m"],
                bodies=[rsc.entity.StructureVoxelBody.box(2, 2, 2, voxel_size=0.01)],
            )
        ],
        simulation=rsc.simulation.Simulation(
            runtime=rsc.simulation.Runtime(max_time=0.05, log_level="warn")
        ),
    )
    origin = rs.RiseOrigin("cpu")
    handle = origin.hello(world)
    handle.wait_for_end()
    assert handle.is_end() is True

    # Each finder takes the not-running path and returns an empty variant of the
    # right kind rather than raising.
    assert isinstance(handle.find_resource("t"), rs.RSC_ResourceVariant)
    assert isinstance(handle.find_essence("m"), rs.RSC_EssenceVariant)
    assert isinstance(handle.find_entity("cube"), rs.RSC_EntityVariant)
    assert isinstance(handle.find_record("z"), rs.RSC_RecordVariant)
    assert isinstance(handle.find_signal("f"), rs.RSC_SignalVariant)
    assert isinstance(handle.find_recorder("rec"), rs.RSC_RecorderVariant)
    assert isinstance(handle.find_controller("c"), rs.RSC_ControllerVariant)


def test_lifecycle_controls_are_noops_after_the_world_ends():
    world = rsc.World(
        "controls",
        essences=[
            rsc.essence.Material("m"),
            rsc.essence.MaterialPhysicalPropertyRigid(
                "mp", material_reference="m", density=1000.0
            ),
        ],
        entities=[
            rsc.entity.StructureVoxel(
                "cube",
                origin_position=(0.0, 0.0, 0.2),
                material_references=["m"],
                bodies=[rsc.entity.StructureVoxelBody.box(2, 2, 2, voxel_size=0.01)],
            )
        ],
        simulation=rsc.simulation.Simulation(
            runtime=rsc.simulation.Runtime(max_time=0.05, log_level="warn")
        ),
    )
    origin = rs.RiseOrigin("cpu")
    handle = origin.hello(world)
    handle.wait_for_end()
    assert handle.is_end() is True

    # Pause, resume, terminate, and the transaction frame all short-circuit on an
    # ended world; they must return without raising and leave it ended.
    handle.pause()
    handle.resume()
    handle.terminate()
    handle.begin_modification_transaction()
    handle.end_modification_transaction()
    assert handle.is_end() is True
