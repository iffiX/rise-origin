# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""One-line builders for :class:`RS_RecorderConfig`.

A recorder writes recorded simulation state to disk. The HDF5 recorder streams
records to an ``.h5`` file you can read back with h5py or the bundled record
accessors.

Example::

    from rise_origin.create import RecorderCreator
    config.recorder_configs.append(RecorderCreator.hdf5("out.h5"))
"""

from rise_origin import (
    RS_HDF5RecorderConfig,
    RS_RecorderConfig,
    RSE_RecorderType,
)

__all__ = ["RecorderCreator"]


class RecorderCreator:
    """Factory of :class:`RS_RecorderConfig` presets."""

    @staticmethod
    def hdf5(path: str) -> RS_RecorderConfig:
        """Build an HDF5 recorder that writes records to ``path``.

        Args:
            path: Output ``.h5`` file path (must be non-empty).

        Returns:
            A populated :class:`RS_RecorderConfig`.
        """
        recorder = RS_RecorderConfig()
        recorder.type = RSE_RecorderType.RSE_RECORDER_HDF5

        cfg = RS_HDF5RecorderConfig()
        cfg.path = path
        recorder.config = cfg
        return recorder
