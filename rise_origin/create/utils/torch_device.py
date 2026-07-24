# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The torch device chooser shared by every torch-running algorithm.

Callers forecast their memory need (grid cells times bytes per cell times
working buffers) and get the device that fits: the GPU when its free
memory covers the forecast with a safety margin, the host otherwise.
Callers that can tile pass per-tile bytes.
"""

from ..constants import TORCH_DEVICE_FREE_MEMORY_SHARE

__all__ = ["TorchDevice"]


class TorchDevice:
    """Stateless device choices, used through the class name."""

    @staticmethod
    def pick(bytes_needed: int) -> object:
        """Return the torch device with room for a memory forecast.

        Args:
            bytes_needed: The forecast working-set size in bytes.
        """
        import torch

        if torch.cuda.is_available():
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            if bytes_needed <= free_bytes * TORCH_DEVICE_FREE_MEMORY_SHARE:
                return torch.device("cuda")
        return torch.device("cpu")
