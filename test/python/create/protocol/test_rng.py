# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the deterministic random stream."""

import pytest

from rise_origin import create as cr


class TestRng:
    def test_same_stream_same_numbers(self):
        assert cr.Rng(7, "trunk").uniform(0, 1) == cr.Rng(7, "trunk").uniform(0, 1)

    def test_split_streams_are_independent(self):
        stream = cr.Rng(7)
        first = stream.split("branch/0").uniform(0, 1)
        second = stream.split("branch/1").uniform(0, 1)
        assert first != second

    def test_adding_a_branch_never_reshuffles_siblings(self):
        before = cr.Rng(7).split("branch/0").uniform(0, 1)
        stream = cr.Rng(7)
        stream.split("branch/1").uniform(0, 1)
        after = stream.split("branch/0").uniform(0, 1)
        assert before == after

    def test_choice_stays_in_options(self):
        assert cr.Rng(3).choice(["a", "b", "c"]) in ("a", "b", "c")

    def test_seed_value_is_the_stream_identity(self):
        assert (
            cr.Rng(7, "trunk").get_seed_value() == cr.Rng(7, "trunk").get_seed_value()
        )
        assert cr.Rng(7, "trunk").get_seed_value() != cr.Rng(7, "leaf").get_seed_value()
        assert cr.Rng(7).split("trunk").get_seed_value() == (
            cr.Rng(7, "trunk").get_seed_value()
        )

    def test_numpy_generator_reproduces_per_stream(self):
        first = cr.Rng(7, "bark").get_numpy_generator().uniform(0, 1, 4)
        second = cr.Rng(7, "bark").get_numpy_generator().uniform(0, 1, 4)
        assert (first == second).all()

    def test_torch_generator_reproduces_per_stream(self):
        torch = pytest.importorskip("torch")
        stream = cr.Rng(7, "bark")
        first = torch.rand(4, generator=cr.Rng(7, "bark").get_torch_generator())
        second = torch.rand(4, generator=cr.Rng(7, "bark").get_torch_generator())
        assert torch.equal(first, second)
        assert stream.get_torch_generator() is stream.get_torch_generator()
