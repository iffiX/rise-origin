# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the identity: hash text, json, and the module reference."""

from rise_origin import create as cr


class TestIdentity:
    def test_hash_text_and_json(self):
        identity = cr.Identity("rise_origin.nature.tree.maple", "1.0.0", '{"seed": 0}')
        assert (
            identity.make_hash_text()
            == 'rise_origin.nature.tree.maple@1.0.0|{"seed": 0}'
        )
        assert cr.Identity.from_json(identity.to_json()) == identity

    def test_reference_carries_the_concrete_version(self):
        identity = cr.Identity("rise_origin.nature.tree.maple", "1.2.3", "{}")
        reference = identity.get_reference()
        assert reference.to_text() == "rise_origin.nature.tree.maple@1.2.3"
        version = cr.ModuleReference.parse_version(reference.version_text)
        assert (version.major, version.minor, version.patch) == (1, 2, 3)
