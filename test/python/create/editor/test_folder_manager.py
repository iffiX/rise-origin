# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the folder manager: homes, session folders, and the lock."""

import os

import pytest

from rise_origin.create.editor import FolderManager, FolderManagerError


class TestFolderManager:
    def test_cache_home_honors_the_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RISE_CACHE_DIR", str(tmp_path / "cache"))
        home = FolderManager.get_cache_home()
        assert home == str(tmp_path / "cache")
        assert os.path.isdir(home)

    def test_sessions_home_honors_the_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RISE_SESSION_DIR", str(tmp_path / "sessions"))
        home = FolderManager.get_sessions_home()
        assert home == str(tmp_path / "sessions")
        assert os.path.isdir(home)

    def test_session_folders_are_fresh_and_created(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RISE_SESSION_DIR", str(tmp_path / "sessions"))
        first = FolderManager.make_session_folder()
        second = FolderManager.make_session_folder()
        assert first != second
        assert os.path.isdir(first) and os.path.isdir(second)

    def test_cache_paths_sit_under_their_root(self, tmp_path):
        root = str(tmp_path)
        assert FolderManager.get_cache_index_path(root).startswith(root)
        assert FolderManager.get_cache_index_lock_path(root).startswith(root)
        assert FolderManager.get_cache_prune_lock_path(root).startswith(root)
        entry = FolderManager.get_cache_entry_blobs_path(root, "abc")
        assert entry == os.path.join(root, "blobs", "abc")

    def test_session_paths_sit_under_their_root(self, tmp_path):
        root = str(tmp_path)
        assert FolderManager.get_session_journal_path(root) == os.path.join(
            root, "journal.jsonl"
        )
        assert FolderManager.get_session_blobs_path(root) == os.path.join(root, "blobs")

    def test_second_lock_of_one_folder_fails_fast(self, tmp_path):
        session = str(tmp_path / "session")
        lock = FolderManager.lock_session_folder(session)
        with pytest.raises(FolderManagerError):
            FolderManager.lock_session_folder(session)
        lock.release()
        again = FolderManager.lock_session_folder(session)
        again.release()
