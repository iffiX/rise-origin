# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.signal.mechanical."""

import pickle

import rise_origin as rs
from rise_origin.rsc.signal import Force, Torque


def test_force_fields():
    force = Force("push", structure_name="dog", voxel_range=(0, 1, 0, 1, 0, 1))
    assert force.structure_name == "dog"
    assert force.voxel_range.max_x == 1


def test_torque_fields():
    torque = Torque("twist", structure_name="dog", body_sid=2)
    assert torque.structure_name == "dog"
    assert torque.body_sid == 2


def test_signal_in_variant_pickles():
    variant = rs.RSC_SignalVariant(Force("push", structure_name="dog"))
    restored = pickle.loads(pickle.dumps(variant))
    assert restored.value.name == "push"
