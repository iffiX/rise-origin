# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.controller.callback."""

import pickle

import rise_origin as rs
from rise_origin.rsc.controller import ControllerCallback


def test_controller_name():
    assert ControllerCallback("brain").name == "brain"


def test_controller_in_variant_pickles():
    variant = rs.RSC_ControllerVariant(ControllerCallback("brain"))
    restored = pickle.loads(pickle.dumps(variant))
    assert restored.value.name == "brain"
