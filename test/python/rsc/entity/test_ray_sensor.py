# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""End-to-end ray sensor test.

A free ray sensor sits at a known pose and casts three rays: one straight ahead
at a fixed cube, one straight down at the floor, and one sideways into empty
space. The world runs headless to a fixed end time; the last delivered record is
asserted for hit flags, distances, hit positions, and entity types.
"""

import numpy as np

import rise_origin as rs
from rise_origin import rsc

SENSOR_ORIGIN = (0.0, 0.0, 0.3)
CUBE_CENTER = (0.6, 0.0, 0.3)
CUBE_HALF_EXTENT = 0.105  # half of the 7-voxel, 0.03 m box
CUBE_ORIGIN = tuple(c - CUBE_HALF_EXTENT for c in CUBE_CENTER)  # box min-corner
FLOOR_HEIGHT = 0.0

RAY_FORWARD = (1.0, 0.0, 0.0)  # hits the cube
RAY_DOWN = (0.0, 0.0, -1.0)  # hits the floor
RAY_SIDE = (0.0, 1.0, 0.0)  # misses everything

MAX_RANGE = 100.0


def test_ray_sensor_reports_hits():
    """A forward ray hits the cube, a down ray hits the floor, a side ray misses."""
    captured = []
    world = _build_world(captured)

    origin = rs.RiseOrigin("cpu")
    handle, callback = origin.hello(world)
    callback.set_callback(_make_capture(captured))
    handle.wait_for_end()

    assert handle.get_status() == rs.RSE_WorldStatus.RSE_WORLD_END_SUCCESS
    assert captured, "no valid ray sensor record was delivered"
    record = captured[-1]

    is_hit = record["is_hit"]
    distance = record["distance"]
    hit_position = record["hit_position"]
    entity_type = record["entity_type"]

    # Ray 0: forward into the cube's near face.
    assert is_hit[0] == 1
    assert entity_type[0] == int(rs.RSE_WorldEntityType.RSE_WORLD_ENTITY_TYPE_STRUCTURE_VOXEL)
    expected_forward = CUBE_CENTER[0] - CUBE_HALF_EXTENT - SENSOR_ORIGIN[0]
    assert abs(distance[0] - expected_forward) < 0.05
    assert np.allclose(hit_position[0], SENSOR_ORIGIN + np.array(RAY_FORWARD) * distance[0], atol=1e-3)

    # Ray 1: straight down onto the floor plane.
    assert is_hit[1] == 1
    assert entity_type[1] == int(rs.RSE_WorldEntityType.RSE_WORLD_ENTITY_TYPE_FLOOR)
    expected_down = SENSOR_ORIGIN[2] - FLOOR_HEIGHT
    assert abs(distance[1] - expected_down) < 0.05

    # Ray 2: sideways into empty space.
    assert is_hit[2] == 0


def _build_world(captured):
    """A fixed cube on a floor, probed by a free ray sensor."""
    essences = [
        rsc.essence.Material("box"),
        rsc.essence.MaterialPhysicalPropertyFixed("box#phys", material_reference="box"),
        rsc.essence.Material("floor"),
        rsc.essence.MaterialPhysicalPropertyFixed("floor#phys", material_reference="floor"),
    ]
    entities = [
        rsc.entity.FloorFlat("floor", material_reference="floor", height=FLOOR_HEIGHT),
        rsc.entity.StructureVoxel(
            "cube",
            origin_position=CUBE_ORIGIN,
            material_references=["box"],
            bodies=[rsc.entity.StructureVoxelBody.box(7, 7, 7, voxel_size=0.03)],
        ),
        rsc.entity.RaySensorFree(
            "probe",
            ray_directions=[RAY_FORWARD, RAY_DOWN, RAY_SIDE],
            origin_position=SENSOR_ORIGIN,
            max_range=MAX_RANGE,
            record=rsc.entity.RaySensorRecord(
                has_hit=True,
                has_entity_info=True,
                has_normal=True,
                target_controller_reference="probe_ctrl",
            ),
            signal=rsc.entity.RaySensorSignal(source_controller_reference="probe_ctrl"),
        ),
    ]
    simulation = rsc.simulation.Simulation(
        runtime=rsc.simulation.Runtime(max_time=0.1, log_level="warn")
    )
    simulation.control.frequency = 30.0
    return rsc.World(
        "ray_sensor",
        essences=essences,
        entities=entities,
        controllers=[rsc.controller.ControllerCallback("probe_ctrl")],
        simulation=simulation,
    )


def _make_capture(captured):
    """Build the per-step callback that copies the ray sensor record out."""

    def control(time, controllers):
        for controller in controllers:
            if controller.name() != "probe_ctrl":
                continue
            for record in controller.records():
                if not isinstance(record, rs.RS_WorldRaySensorRecord):
                    continue
                if not record.is_valid() or record.ray_num() == 0:
                    continue
                captured.append(
                    {
                        "is_hit": np.array(record.hit.is_hit(), copy=True),
                        "distance": np.array(record.hit.distance(), copy=True),
                        "hit_position": np.array(record.hit.hit_position(), copy=True),
                        "entity_type": np.array(record.entity_info.entity_type(), copy=True),
                    }
                )

    return control


if __name__ == "__main__":
    test_ray_sensor_reports_hits()
    print("ray sensor test passed")
