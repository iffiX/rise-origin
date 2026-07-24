# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the torch device chooser."""

import pytest

torch = pytest.importorskip("torch")

from rise_origin.create.utils import TorchDevice  # noqa: E402


class TestTorchDevice:
    def test_an_oversized_forecast_falls_back_to_the_host(self):
        assert TorchDevice.pick(2**60) == torch.device("cpu")

    def test_a_small_forecast_lands_on_the_best_device(self):
        picked = TorchDevice.pick(1024)
        if torch.cuda.is_available():
            assert picked == torch.device("cuda")
        else:
            assert picked == torch.device("cpu")
