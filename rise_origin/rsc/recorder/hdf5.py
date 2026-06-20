# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""HDF5 recorder: persists routed records to an HDF5 file."""

from rise_origin import RSC_RecorderHDF5

__all__ = ["RecorderHDF5"]


class RecorderHDF5(RSC_RecorderHDF5):
    """Writes every record routed to it (by ``target_recorder_reference``) to disk.

    Args:
        name: Unique recorder name that records target by reference.
        path: Filesystem path of the output HDF5 file.
    """

    def __init__(self, name: str, *, path: str = "") -> None:
        super().__init__()
        self.name = name
        self.path = path
