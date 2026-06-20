# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Recorder characterizations.

A recorder is an I/O sink that persists the records routed to it (by their
``target_recorder_reference``). Currently an HDF5 file recorder.

Example::

    from rise_origin import rsc

    rec = rsc.recorder.RecorderHDF5("out", path="run.h5")
"""

from rise_origin import RSC_RecorderBase

from rise_origin.rsc.recorder.hdf5 import RecorderHDF5

#: Abstract base of every recorder characterization (carries name).
RecorderBase = RSC_RecorderBase

__all__ = ["RecorderBase", "RecorderHDF5"]
