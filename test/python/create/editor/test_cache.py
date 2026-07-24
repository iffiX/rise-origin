# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the realization cache: tiers, strategies, and eviction races."""

import os

from rise_origin import create as cr
from rise_origin.create.editor import RealizationCache
from rise_origin.rsc import essence


def make_draft_json(name: str = "block", rscs: dict = None) -> dict:
    stored_rscs = None
    if rscs:
        stored_rscs = sorted(rscs)
    return {
        "uid": 1,
        "name": name,
        "pose": cr.Pose().to_json(),
        "boxes": [cr.Box((0, 0, 0), (1, 1, 1)).to_json()],
        "requires": [],
        "tags": [],
        "annotations": {},
        "is_fixed": False,
        "has_floor": False,
        "identity": None,
        "rscs": stored_rscs,
        "children": [],
    }


class TestRealizationCache:
    def test_miss_then_store_then_both_needs(self, tmp_path):
        cache = RealizationCache(str(tmp_path / "cache"))
        identity = cr.Identity("rise_origin.material.mineral.clay", "1.0.0", "{}")
        assert cache.resolve(identity) is None
        rscs = {0: essence.Material("clay")}
        cache.store(identity, make_draft_json(rscs=rscs), rscs, build_ms=1.0)
        declared = cache.resolve(identity, need="json")
        assert declared.rscs is None
        assert declared.draft_json["name"] == "block"
        realized = cache.resolve(identity, need="rscs")
        assert realized.rscs is not None
        assert realized.rscs[0].name == "clay"

    def test_two_connections_share_one_directory(self, tmp_path):
        directory = str(tmp_path / "cache")
        first = RealizationCache(directory)
        second = RealizationCache(directory)
        identity = cr.Identity("rise_origin.material.mineral.clay", "1.0.0", "{}")
        rscs = {0: essence.Material("clay")}
        first.store(identity, make_draft_json(rscs=rscs), rscs, build_ms=1.0)
        hit = second.resolve(identity, need="rscs")
        assert hit is not None and hit.rscs[0].name == "clay"

    def test_prune_flips_the_row_and_readers_fall_back(self, tmp_path):
        directory = str(tmp_path / "cache")
        writer = RealizationCache(directory)
        reader = RealizationCache(directory)
        identity = cr.Identity("rise_origin.material.mineral.clay", "1.0.0", "{}")
        rscs = {0: essence.Material("clay")}
        writer.store(identity, make_draft_json(rscs=rscs), rscs, build_ms=1.0)
        assert writer.prune(target_bytes=0) > 0
        hit = reader.resolve(identity, need="rscs")
        assert hit is not None
        assert hit.rscs is None
        assert reader.resolve(identity, need="json").draft_json["name"] == "block"

    def test_declaration_only_store_needs_no_blobs(self, tmp_path):
        cache = RealizationCache(str(tmp_path / "cache"))
        identity = cr.Identity("rise_origin.nature.tree.maple", "1.0.0", "{}")
        cache.store(identity, make_draft_json(), {}, build_ms=1.0)
        hit = cache.resolve(identity, need="rscs")
        assert hit is not None and hit.rscs == {}
        assert not os.path.exists(os.path.join(cache.directory, "blobs", "anything"))

    def test_local_strategy_uses_wal(self, tmp_path):
        cache = RealizationCache(str(tmp_path / "cache"))
        assert not cache.is_shared
        mode = cache._connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_shared_strategy_locks_per_operation(self, tmp_path):
        directory = str(tmp_path / "cache")
        writer = RealizationCache(directory, is_shared=True)
        reader = RealizationCache(directory, is_shared=True)
        assert writer._connection is None
        identity = cr.Identity("rise_origin.material.mineral.clay", "1.0.0", "{}")
        rscs = {0: essence.Material("clay")}
        writer.store(identity, make_draft_json(rscs=rscs), rscs, build_ms=1.0)
        hit = reader.resolve(identity, need="rscs")
        assert hit is not None and hit.rscs[0].name == "clay"
        assert writer.prune(target_bytes=0) > 0
        late_reader = RealizationCache(directory, is_shared=True)
        assert late_reader.resolve(identity, need="rscs").rscs is None
        assert os.path.exists(os.path.join(directory, "index.lock"))

    def test_network_mount_detection(self, tmp_path):
        mounts = tmp_path / "mounts"
        mounts.write_text("server:/export /net nfs rw 0 0\n/dev/sda1 / ext4 rw 0 0\n")
        assert RealizationCache._find_network_mount("/net/cache", str(mounts)) == "nfs"
        assert RealizationCache._find_network_mount("/home/x", str(mounts)) is None
