# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the module reference: name levels and version matching."""

import pytest

from rise_origin import create as cr


class TestModuleReference:
    def test_from_text_parses_four_levels(self):
        reference = cr.ModuleReference.from_text("rise_origin.nature.tree.maple")
        assert reference.organization == "rise_origin"
        assert reference.area == "nature"
        assert reference.group == "tree"
        assert reference.kind == "maple"
        assert reference.version_text is None

    def test_three_levels_default_the_organization(self):
        reference = cr.ModuleReference.from_text("material.organic.leaf@^1")
        assert reference.get_name() == "rise_origin.material.organic.leaf"
        assert reference.version_text == "^1"

    def test_bad_level_counts_refused(self):
        for text in ("maple", "tree.maple", "a.b.c.d.e", "a..c.d"):
            with pytest.raises(ValueError):
                cr.ModuleReference.from_text(text)

    def test_to_text_round_trips(self):
        for text in (
            "rise_origin.nature.tree.maple",
            "acme.material.custom.glow@>=1.2",
        ):
            assert cr.ModuleReference.from_text(text).to_text() == text

    def test_version_matching_treats_star_and_none_as_any(self):
        version = cr.ModuleReference.parse_version("1.2.3")
        anything = cr.ModuleReference.from_text("a.b.c.d")
        assert anything.is_version_matching(version)
        starred = cr.ModuleReference.from_text("a.b.c.d@*")
        assert starred.is_version_matching(version)
        pinned = cr.ModuleReference.from_text("a.b.c.d@^2")
        assert not pinned.is_version_matching(version)

    def test_parse_version_refuses_specifiers_and_garbage(self):
        with pytest.raises(ValueError):
            cr.ModuleReference.parse_version("^1")
        with pytest.raises(ValueError):
            cr.ModuleReference.parse_version("not-a-version")
