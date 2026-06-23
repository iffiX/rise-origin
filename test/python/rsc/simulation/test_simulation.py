# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.simulation."""

import pickle

import pytest

import rise_origin as rs
from rise_origin.rsc import simulation as sim


def test_environment_field():
    env = sim.Environment(gravity_acceleration=-3.71)
    assert env.gravity_acceleration == pytest.approx(-3.71)


def test_omitted_field_keeps_engine_default():
    # A fresh SoftBody and one built with no kwargs must agree on every field.
    default = rs.RSC_SimulationSoftBody()
    built = sim.SoftBody()
    assert built.global_damping_z == default.global_damping_z


def test_vision_bundles_subconfigs():
    vision = sim.Vision(
        trace=sim.VisionTrace(firefly_cap=10.0),
        restir=sim.VisionReSTIR(is_enabled=True),
        denoise=sim.VisionDenoiseSVGF(atrous_iterations=4),
    )
    assert vision.trace.firefly_cap == pytest.approx(10.0)
    assert vision.restir.is_enabled is True
    assert vision.denoise.atrous_iterations == 4


def test_simulation_composes_and_pickles():
    s = sim.Simulation(
        seed=7,
        environment=sim.Environment(gravity_acceleration=-9.81),
        time=sim.Time(dt=1e-3),
        control=sim.Control(frequency=50.0),
    )
    assert s.seed == 7
    assert s.environment.gravity_acceleration == pytest.approx(-9.81)
    assert s.control.frequency == pytest.approx(50.0)
    restored = pickle.loads(pickle.dumps(s))
    assert restored.seed == 7
