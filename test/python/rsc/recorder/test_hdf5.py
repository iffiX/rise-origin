# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.recorder.hdf5."""

import pickle

import rise_origin as rs
from rise_origin.rsc.recorder import RecorderHDF5


def test_recorder_path():
    recorder = RecorderHDF5("rec", path="/tmp/out.h5")
    assert recorder.name == "rec"
    assert recorder.path == "/tmp/out.h5"


def test_recorder_in_variant_pickles():
    variant = rs.RSC_RecorderVariant(RecorderHDF5("rec", path="/tmp/out.h5"))
    restored = pickle.loads(pickle.dumps(variant))
    assert restored.value.path == "/tmp/out.h5"
