# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Every folder and file the create layer puts on disk, named in one place.

The layer keeps two roots. The cache home holds the realization cache shared
by every editor; the sessions home holds one folder per editing session::

    cache home  (RISE_CACHE_DIR, or the platform cache folder)
        index.sqlite                      the realization cache index
        index.lock                        shared-strategy index lock
        prune.lock                        eviction mutex
        blobs/<identity hash>/<n>.rscb    stored rsc files of one entry

    sessions home  (RISE_SESSION_DIR, or the platform data folder)
        <session uuid>/
            journal.jsonl                 the session journal
            session.lock                  the editor's exclusive session lock
            blobs/<n>.rscb                hand-made rscs the journal references

The homes are only the defaults: a cache and a session both accept an
explicit folder, so the sub path methods take the root they resolve under.
"""

import os
import sys
import uuid

from filelock import FileLock, Timeout

from ..constants import SESSION_LOCK_TIMEOUT_S

__all__ = ["FolderManagerError", "FolderManager"]


class FolderManagerError(RuntimeError):
    """A folder that cannot be prepared or locked."""


class FolderManager:
    """The static name table of the create layer's on-disk footprint.

    Cache methods resolve the files of one realization cache folder;
    session methods allocate, lock, and resolve the files of one session
    folder. Nothing here holds state: every method is a pure path rule
    except the two home methods and :meth:`make_session_folder`, which
    also create the folder they name.
    """

    @staticmethod
    def get_cache_home() -> str:
        """Return the default realization cache folder, created when absent.

        ``RISE_CACHE_DIR`` overrides the platform default: the local
        application data folder on Windows, ``~/Library/Caches`` on macOS,
        and the ``XDG_CACHE_HOME`` folder elsewhere.
        """
        override = os.environ.get("RISE_CACHE_DIR")
        if override:
            home = override
        elif os.name == "nt":
            base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            home = os.path.join(base, "rise-origin", "cache")
        elif sys.platform == "darwin":
            home = os.path.expanduser("~/Library/Caches/rise-origin")
        else:
            base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            home = os.path.join(base, "rise-origin")
        os.makedirs(home, exist_ok=True)
        return home

    @staticmethod
    def get_cache_index_path(cache_dir: str) -> str:
        """Return the SQLite index path of a cache folder.

        Args:
            cache_dir: The cache folder to resolve under.
        """
        return os.path.join(cache_dir, "index.sqlite")

    @staticmethod
    def get_cache_index_lock_path(cache_dir: str) -> str:
        """Return the shared-strategy index lock path of a cache folder.

        Args:
            cache_dir: The cache folder to resolve under.
        """
        return os.path.join(cache_dir, "index.lock")

    @staticmethod
    def get_cache_prune_lock_path(cache_dir: str) -> str:
        """Return the eviction mutex path of a cache folder.

        Args:
            cache_dir: The cache folder to resolve under.
        """
        return os.path.join(cache_dir, "prune.lock")

    @staticmethod
    def get_cache_blobs_path(cache_dir: str) -> str:
        """Return the blob root of a cache folder.

        Args:
            cache_dir: The cache folder to resolve under.
        """
        return os.path.join(cache_dir, "blobs")

    @staticmethod
    def get_cache_entry_blobs_path(cache_dir: str, identity_hash: str) -> str:
        """Return the blob folder of one cache entry.

        Args:
            cache_dir: The cache folder to resolve under.
            identity_hash: The identity hash naming the entry's blob folder.
        """
        return os.path.join(cache_dir, "blobs", identity_hash)

    @staticmethod
    def get_sessions_home() -> str:
        """Return the default sessions folder, created when absent.

        ``RISE_SESSION_DIR`` overrides the platform default: the local
        application data folder on Windows, ``~/Library/Application
        Support`` on macOS, and the ``XDG_DATA_HOME`` folder elsewhere.
        """
        override = os.environ.get("RISE_SESSION_DIR")
        if override:
            home = override
        elif os.name == "nt":
            base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            home = os.path.join(base, "rise-origin", "sessions")
        elif sys.platform == "darwin":
            home = os.path.expanduser(
                "~/Library/Application Support/rise-origin/sessions"
            )
        else:
            base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
            home = os.path.join(base, "rise-origin", "sessions")
        os.makedirs(home, exist_ok=True)
        return home

    @staticmethod
    def make_session_folder() -> str:
        """Return a fresh uuid-named session folder under the sessions home."""
        path = os.path.join(FolderManager.get_sessions_home(), uuid.uuid4().hex)
        os.makedirs(path)
        return path

    @staticmethod
    def lock_session_folder(session_dir: str) -> FileLock:
        """Acquire a session folder's exclusive lock and return it held.

        The caller keeps the returned lock for the editor's lifetime; a
        second acquisition of the same folder fails fast with an error
        naming the lock file.

        Args:
            session_dir: The session folder to lock.
        """
        os.makedirs(session_dir, exist_ok=True)
        lock = FileLock(
            os.path.join(session_dir, "session.lock"),
            timeout=SESSION_LOCK_TIMEOUT_S,
        )
        try:
            lock.acquire()
        except Timeout as error:
            raise FolderManagerError(
                f"'{session_dir}' is already held by another editor; waited "
                f"{SESSION_LOCK_TIMEOUT_S:g} s for '{lock.lock_file}'"
            ) from error
        return lock

    @staticmethod
    def get_session_journal_path(session_dir: str) -> str:
        """Return the journal file path of a session folder.

        Args:
            session_dir: The session folder to resolve under.
        """
        return os.path.join(session_dir, "journal.jsonl")

    @staticmethod
    def get_session_blobs_path(session_dir: str) -> str:
        """Return the hand-made rsc blob folder of a session folder.

        Args:
            session_dir: The session folder to resolve under.
        """
        return os.path.join(session_dir, "blobs")
