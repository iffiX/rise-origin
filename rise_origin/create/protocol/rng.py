# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The deterministic random stream every module and solver draws from.

All randomness flows from a seed, split by stable string paths: adding a
branch must never reshuffle its siblings. A stream that seeds itself is a
protocol violation; a module builds its own stream from ``settings.seed``.
"""

import hashlib
from typing import Dict, Sequence

import numpy as np

__all__ = ["Rng"]


class Rng:
    """A deterministic random stream split by stable string paths.

    Args:
        seed: The stream family seed.
        path: The stable path of this stream within the family.
    """

    def __init__(self, seed: int, path: str = "") -> None:
        self.seed = int(seed)
        self.path = path
        self._numpy_generator: np.random.Generator = None
        self._torch_generators: Dict[str, object] = {}

    def split(self, path: str) -> "Rng":
        """Return the child stream at a stable sub path.

        Args:
            path: The stable sub path joined onto this stream's path.
        """
        joined = f"{self.path}/{path}" if self.path else path
        return Rng(self.seed, joined)

    def get_seed_value(self) -> int:
        """Return the 64-bit seed value derived from the seed and path."""
        digest = hashlib.sha256(f"{self.seed}:{self.path}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little")

    def get_numpy_generator(self) -> np.random.Generator:
        """Return the numpy generator of this stream, created once."""
        if self._numpy_generator is None:
            self._numpy_generator = np.random.Generator(
                np.random.PCG64(self.get_seed_value())
            )
        return self._numpy_generator

    def get_torch_generator(self, device: str = "cpu") -> object:
        """Return the torch generator of this stream, created once per device.

        Args:
            device: The torch device the generator lives on.
        """
        import torch

        key = str(torch.device(device))
        if key not in self._torch_generators:
            generator = torch.Generator(key)
            generator.manual_seed(self.get_seed_value())
            self._torch_generators[key] = generator
        return self._torch_generators[key]

    def uniform(self, low: float, high: float) -> float:
        """Return one uniform draw in ``[low, high)``.

        Args:
            low: The inclusive lower end of the range.
            high: The exclusive upper end of the range.
        """
        return float(self.get_numpy_generator().uniform(low, high))

    def integers(self, low: int, high: int) -> int:
        """Return one integer draw in ``[low, high)``.

        Args:
            low: The inclusive lower end of the range.
            high: The exclusive upper end of the range.
        """
        return int(self.get_numpy_generator().integers(low, high))

    def normal(self, mean: float = 0.0, sigma: float = 1.0) -> float:
        """Return one normal draw.

        Args:
            mean: The center of the distribution.
            sigma: The standard deviation of the distribution.
        """
        return float(self.get_numpy_generator().normal(mean, sigma))

    def choice(self, options: Sequence) -> object:
        """Return one uniformly chosen element.

        Args:
            options: The elements to choose among.
        """
        return options[self.integers(0, len(options))]
