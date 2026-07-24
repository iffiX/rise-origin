# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Literal, Optional, Set, Tuple

from filelock import FileLock, Timeout

from rise_origin import RS_ENGINE_VERSION, RS_DataException
from rise_origin import rsc as rsc_module

from ..constants import (
    CACHE_DEFAULT_CAP_BYTES,
    CACHE_LOCK_TIMEOUT_S,
    CACHE_MEMORY_LIMIT,
    CACHE_TOUCH_INTERVAL_S,
)
from ..data.identity import Identity
from .folder_manager import FolderManager

__all__ = [
    "RealizationCacheError",
    "RealizationCacheStatus",
    "RealizationCacheHit",
    "RealizationCache",
]

_NETWORK_FILESYSTEMS = {
    "nfs",
    "nfs4",
    "cifs",
    "smbfs",
    "smb3",
    "sshfs",
    "fuse.sshfs",
    "9p",
    "afs",
    "afpfs",
    "webdav",
    "glusterfs",
    "lustre",
}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    identity_hash TEXT PRIMARY KEY,
    module TEXT, version TEXT, settings_json TEXT,
    tier TEXT, size_bytes INTEGER, build_ms REAL,
    blob_path TEXT,
    draft_json TEXT, cover TEXT, annotations TEXT,
    content_hash TEXT, engine_version TEXT, rsc_version TEXT,
    created_at REAL, last_access REAL
)
"""


class RealizationCacheError(RuntimeError):
    """A cache location, content, or lock problem."""


@dataclass(frozen=True)
class RealizationCacheStatus:
    """One snapshot of the cache accounting.

    Attributes:
        directory: The cache root folder.
        entry_count: How many identities the index holds, both tiers.
        payload_bytes: Total bytes of the stored rsc files.
        cap_bytes: The size cap prune evicts the rscs tier down to.
    """

    directory: str
    entry_count: int
    payload_bytes: int
    cap_bytes: int


@dataclass(frozen=True)
class RealizationCacheHit:
    """One resolved cache entry.

    Attributes:
        draft_json: The draft's json tree, shared read-only.
        rscs: Loaded rsc objects by file index; None when only the json was
            asked for or the stored files are gone.
    """

    draft_json: dict
    rscs: Optional[Dict[int, object]]


class RealizationCache:
    """Content-addressed store of realized module draft output.

    What it holds, and why a hit is exact:

    - The key is a Draft's identity: the module at a version plus its
      canonical settings json. A deterministic module always produces the
      same bytes for one identity, so a cache hit is the realization,
      replayed, never an approximation of it. A non-deterministic module's
      entry is a first-build snapshot: valid while it lives, replaced by a
      new equally valid build after eviction, and never what a session's
      reproducibility rests on, because the journal records placed content.
    - Every entry stores the draft json (the tree shape ``Draft.to_json``
      emits) inline in one SQLite index file, and the draft's rsc objects
      as plain ``.rscb`` files in a blob directory beside it. The cache
      never constructs Drafts; callers rebuild them from the json and the
      loaded rscs.
    - An entry sits at one of two tiers: "rscs" (json plus blob files) or
      "json" (json only, kept forever). ``resolve(need="json")`` answers
      box and require questions with zero file IO.

    How concurrent access stays safe:

    - Blob files publish by staging plus atomic rename, so two processes
      storing the same identity at once are harmless: both write identical
      bytes and the last rename wins.
    - Eviction (``prune``) flips an entry's row to the json tier first and
      unlinks its blob files after, so a reader losing the race sees a json
      row and rebuilds through the module. One ``prune.lock`` keeps two
      prunes from running at once; nothing else takes it.
    - The SQLite access strategy picks itself from the directory's
      filesystem at construction, as follows.

    On a local disk:

    - SQLite runs in WAL mode over one persistent connection with a 30
      second busy timeout.
    - Any number of processes on one node (an editor plus a preview
      window's separate engine instance is the common case) read without
      blocking the writer and never see partial commits; writers serialize
      on WAL's own write lock.
    - A killed process leaves a ``-wal`` file the next opener replays.
    - WAL is a journaling mode, not a storage format: the database pages
      are identical and the mode converts in place.

    On a network filesystem, or with ``RISE_CACHE_SHARED=1``, or with
    ``is_shared=True``:

    - WAL's shared memory does not cross NFS clients, so every index
      operation instead runs as: acquire ``index.lock`` (30 second
      timeout), open a fresh connection in PERSIST journal mode, work,
      commit, close, release.
    - Opening after locking and closing before releasing rides NFS
      close-to-open consistency: every node reads the latest committed
      index, writers never interleave, and all nodes serialize per
      operation.
    - A node dying with the lock stalls others until the NFSv4 lease
      expires; after that the timeout raises an error naming the lock file.
    - Working network file locking is required: a ``nolock`` mount loses
      cross-node exclusion and can corrupt the index. A warning says so
      once per directory.
    - One trap needs the caller's caution: a directory that is local disk
      to one node but NFS-exported to others detects as local on the
      owning node, and mixed WAL and shared access can corrupt the index.
      The exporting node must opt in with ``RISE_CACHE_SHARED=1``; no
      client-side code can detect an export.

    Args:
        directory: Cache root; defaults to ``RISE_CACHE_DIR`` or the
            platform cache home under ``rise-origin``.
        cap_bytes: Size cap the rscs tier is pruned to.
        is_shared: Force the shared strategy; None picks by filesystem and
            the ``RISE_CACHE_SHARED`` environment switch.
    """

    _warned_directories: ClassVar[Set[str]] = set()

    def __init__(
        self,
        directory: Optional[str] = None,
        cap_bytes: int = CACHE_DEFAULT_CAP_BYTES,
        is_shared: Optional[bool] = None,
    ) -> None:
        self.directory = os.path.abspath(directory or FolderManager.get_cache_home())
        self.cap_bytes = int(cap_bytes)
        os.makedirs(FolderManager.get_cache_blobs_path(self.directory), exist_ok=True)
        network_type = RealizationCache._find_network_mount(self.directory)
        if is_shared is None:
            is_shared = bool(os.environ.get("RISE_CACHE_SHARED")) or (
                network_type is not None
            )
        self.is_shared = bool(is_shared)
        if network_type is not None and self.directory not in (
            RealizationCache._warned_directories
        ):
            RealizationCache._warned_directories.add(self.directory)
            warnings.warn(
                f"cache directory '{self.directory}' sits on '{network_type}'; "
                "the shared strategy needs working network file locking, such "
                "as NFSv4 or lockd, and a nolock mount can corrupt the index",
                RuntimeWarning,
            )
        self._memory: Dict[str, RealizationCacheHit] = {}
        self._connection: Optional[sqlite3.Connection] = None
        self._lock: Optional[FileLock] = None
        if self.is_shared:
            self._lock = FileLock(
                FolderManager.get_cache_index_lock_path(self.directory),
                timeout=CACHE_LOCK_TIMEOUT_S,
            )
        else:
            self._connection = sqlite3.connect(self._get_index_path())
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                f"PRAGMA busy_timeout={int(CACHE_LOCK_TIMEOUT_S * 1000)}"
            )
        with self._use_connection() as connection:
            connection.execute(_SCHEMA)

    def get_status(self) -> RealizationCacheStatus:
        """Return the current accounting snapshot."""
        with self._use_connection() as connection:
            count, payload_bytes = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM entries"
            ).fetchone()
        return RealizationCacheStatus(
            self.directory, int(count), int(payload_bytes), self.cap_bytes
        )

    def has_entry(self, identity: Identity) -> bool:
        """Return whether the identity is cached at all.

        Args:
            identity: The draft identity to look up.
        """
        identity_hash = RealizationCache._make_identity_hash(identity)
        if identity_hash in self._memory:
            return True
        with self._use_connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM entries WHERE identity_hash = ?", (identity_hash,)
            ).fetchone()
        return row is not None

    def resolve(
        self, identity: Identity, need: Literal["json", "rscs"] = "rscs"
    ) -> Optional[RealizationCacheHit]:
        """Return the cached hit of an identity, or None on a full miss.

        Args:
            identity: The draft identity to resolve.
            need: ``"json"`` returns the draft json with no file IO at
                all; ``"rscs"`` also loads the stored rsc files.

        Returns:
            The cached hit, or None on a full miss. With ``need="rscs"``
            the hit answers with ``rscs=None`` when eviction won the
            race, so the caller rebuilds through the module.
        """
        identity_hash = RealizationCache._make_identity_hash(identity)
        held = self._memory.get(identity_hash)
        if held is not None:
            if need == "json":
                return RealizationCacheHit(held.draft_json, None)
            return held
        with self._use_connection() as connection:
            row = connection.execute(
                "SELECT draft_json, blob_path, last_access FROM entries "
                "WHERE identity_hash = ?",
                (identity_hash,),
            ).fetchone()
            if row is None:
                return None
            draft_json_text, blob_path, last_access = row
            RealizationCache._touch(connection, identity_hash, last_access)
        draft_json = json.loads(draft_json_text)
        if need == "json":
            return RealizationCacheHit(draft_json, None)
        rscs = self._load_rscs(identity_hash, draft_json, blob_path)
        hit = RealizationCacheHit(draft_json, rscs)
        if rscs is not None:
            self._remember(identity_hash, hit)
        return hit

    def store(
        self,
        identity: Identity,
        draft_json: dict,
        rscs: Dict[int, object],
        build_ms: float,
    ) -> None:
        """Store one realized draft json with its rsc objects.

        The rsc files write into a staging directory and move into place
        with one atomic replace; concurrent stores of one identity are
        harmless, deterministic modules writing identical bytes and
        non-deterministic ones letting the last valid build win.

        Args:
            identity: The draft identity to store under.
            draft_json: The draft's json tree.
            rscs: The draft's rsc objects, keyed by file index.
            build_ms: How long the realization took to build, in
                milliseconds.
        """
        identity_hash = RealizationCache._make_identity_hash(identity)
        blob_dir, size_bytes, content_hash = self._save_rscs(identity_hash, rscs)
        now = time.time()
        with self._use_connection() as connection:
            connection.execute(
                "REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identity_hash,
                    identity.module,
                    identity.version,
                    identity.settings_json,
                    "rscs" if blob_dir is not None else "json",
                    size_bytes,
                    float(build_ms),
                    blob_dir,
                    json.dumps(draft_json, sort_keys=True),
                    json.dumps(draft_json.get("boxes", []), sort_keys=True),
                    json.dumps(draft_json.get("annotations", {}), sort_keys=True),
                    content_hash,
                    RS_ENGINE_VERSION,
                    RS_ENGINE_VERSION,
                    now,
                    now,
                ),
            )
        self._remember(identity_hash, RealizationCacheHit(draft_json, dict(rscs)))

    def prune(self, target_bytes: Optional[int] = None) -> int:
        """Evict stored rsc files down to a target, coldest and cheapest first.

        Only the rscs tier is evicted; json rows stay forever. Each eviction
        flips the row to the json tier first and unlinks the blobs after, so
        concurrent readers never see files without a row. One file lock
        keeps two prunes from running at once.

        Args:
            target_bytes: Size to evict the rscs tier down to; None uses
                the configured cap.

        Returns:
            The total bytes freed.
        """
        target = self.cap_bytes if target_bytes is None else int(target_bytes)
        freed = 0
        prune_lock = FileLock(
            FolderManager.get_cache_prune_lock_path(self.directory),
            timeout=CACHE_LOCK_TIMEOUT_S,
        )
        try:
            with prune_lock:
                with self._use_connection() as connection:
                    remaining = int(
                        connection.execute(
                            "SELECT COALESCE(SUM(size_bytes), 0) FROM entries"
                        ).fetchone()[0]
                    )
                    rows = connection.execute(
                        "SELECT identity_hash, blob_path, size_bytes FROM entries "
                        "WHERE tier = 'rscs' "
                        "ORDER BY (last_access + build_ms / 1000.0 * 60.0) ASC"
                    ).fetchall()
                evicted_paths = []
                for identity_hash, blob_path, size_bytes in rows:
                    if remaining <= target:
                        break
                    with self._use_connection() as connection:
                        connection.execute(
                            "UPDATE entries SET tier = 'json', "
                            "blob_path = NULL, size_bytes = 0 "
                            "WHERE identity_hash = ?",
                            (identity_hash,),
                        )
                    self._memory.pop(identity_hash, None)
                    evicted_paths.append(blob_path)
                    remaining -= size_bytes
                    freed += size_bytes
                for blob_path in evicted_paths:
                    if blob_path and os.path.exists(blob_path):
                        try:
                            shutil.rmtree(blob_path)
                        except OSError:
                            pass
                self._remove_unused_blobs()
        except Timeout as error:
            raise RealizationCacheError(
                f"timed out after {CACHE_LOCK_TIMEOUT_S:g} s waiting for "
                f"'{prune_lock.lock_file}'; a dead holder may need cleanup"
            ) from error
        return freed

    def _save_rscs(
        self, identity_hash: str, rscs: Dict[int, object]
    ) -> Tuple[Optional[str], int, str]:
        """Save the rsc objects of one entry as published blob files.

        The files write into a staging directory and move into place with
        one atomic replace. Returns the published blob directory (None when
        nothing was stored), the total size in bytes, and the content hash
        over the saved files.
        """
        blob_dir = FolderManager.get_cache_entry_blobs_path(
            self.directory, identity_hash
        )
        staging = blob_dir + ".tmp"
        if os.path.exists(staging):
            shutil.rmtree(staging)
        os.makedirs(staging)
        digest = hashlib.sha256()
        size_bytes = 0
        for index in sorted(rscs):
            path = os.path.join(staging, f"{index}.rscb")
            rscs[index].save(path)
            with open(path, "rb") as handle:
                data = handle.read()
            digest.update(data)
            size_bytes += len(data)
        if not rscs:
            shutil.rmtree(staging)
            return None, 0, digest.hexdigest()
        if os.path.exists(blob_dir):
            shutil.rmtree(blob_dir)
        os.replace(staging, blob_dir)
        return blob_dir, size_bytes, digest.hexdigest()

    def _load_rscs(
        self, identity_hash: str, draft_json: dict, blob_path: Optional[str]
    ) -> Optional[Dict[int, object]]:
        """Load the stored rscs of a draft json, or None when evicted."""
        stored_rscs = RealizationCache._get_stored_rscs_from_draft(draft_json)
        if not stored_rscs:
            return {}
        if blob_path is None:
            return None
        rscs = {}
        for index in stored_rscs:
            path = os.path.join(blob_path, f"{index}.rscb")
            try:
                rscs[index] = rsc_module.load(path)
            except (FileNotFoundError, RS_DataException):
                with self._use_connection() as connection:
                    row = connection.execute(
                        "SELECT tier FROM entries WHERE identity_hash = ?",
                        (identity_hash,),
                    ).fetchone()
                if row is None or row[0] != "rscs":
                    return None
                raise
        return rscs

    def _remember(self, identity_hash: str, hit: RealizationCacheHit) -> None:
        """Hold a hot hit in memory, bounded."""
        self._memory[identity_hash] = hit
        while len(self._memory) > CACHE_MEMORY_LIMIT:
            self._memory.pop(next(iter(self._memory)))

    def _remove_unused_blobs(self) -> None:
        """Drop blob directories whose rows already left the rscs tier."""
        with self._use_connection() as connection:
            held_hashes = {
                row[0]
                for row in connection.execute(
                    "SELECT identity_hash FROM entries WHERE tier = 'rscs'"
                )
            }
        blobs = FolderManager.get_cache_blobs_path(self.directory)
        for name in os.listdir(blobs):
            if name in held_hashes and not name.endswith(".tmp"):
                continue
            try:
                shutil.rmtree(os.path.join(blobs, name))
            except OSError:
                pass

    def _get_index_path(self) -> str:
        """Return the SQLite index path."""
        return FolderManager.get_cache_index_path(self.directory)

    @contextmanager
    def _use_connection(self):
        """Yield a connection under the strategy's discipline, then commit.

        The local strategy reuses the WAL connection; the shared strategy
        holds the index lock around a fresh PERSIST connection so every
        operation reads the latest committed index over the network.
        """
        if self._connection is not None:
            yield self._connection
            self._connection.commit()
            return
        try:
            with self._lock:
                connection = sqlite3.connect(self._get_index_path())
                try:
                    connection.execute("PRAGMA journal_mode=PERSIST")
                    yield connection
                    connection.commit()
                finally:
                    connection.close()
        except Timeout as error:
            raise RealizationCacheError(
                f"timed out after {CACHE_LOCK_TIMEOUT_S:g} s waiting for "
                f"'{self._lock.lock_file}'; a dead holder may need cleanup"
            ) from error

    @staticmethod
    def _touch(
        connection: sqlite3.Connection, identity_hash: str, last_access: float
    ) -> None:
        """Refresh last access, throttled so hits are rarely writes."""
        now = time.time()
        if now - float(last_access) < CACHE_TOUCH_INTERVAL_S:
            return
        connection.execute(
            "UPDATE entries SET last_access = ? WHERE identity_hash = ?",
            (now, identity_hash),
        )

    @staticmethod
    def _make_identity_hash(identity: Identity) -> str:
        """Return the content address of an identity."""
        return hashlib.sha256(identity.make_hash_text().encode("utf-8")).hexdigest()

    @staticmethod
    def _get_stored_rscs_from_draft(draft_json: dict) -> List[int]:
        """Return every stored rsc file index of a draft json, depth first."""
        stored_rscs = []
        stack = [draft_json]
        while stack:
            current = stack.pop()
            rscs = current.get("rscs")
            if rscs is not None:
                stored_rscs.extend(rscs)
            stack.extend(reversed(current["children"]))
        return stored_rscs

    @staticmethod
    def _find_network_mount(
        directory: str, mounts_path: str = "/proc/mounts"
    ) -> Optional[str]:
        """Return the network filesystem type under a directory, or None.

        Linux reads the mounts table, Windows classifies UNC paths and
        remote drives, and macOS asks statfs for the filesystem name. A
        detection failure answers None, so an undetected share needs the
        ``RISE_CACHE_SHARED`` switch.
        """
        if os.name == "nt":
            return RealizationCache._find_network_mount_nt(directory)
        if sys.platform == "darwin":
            return RealizationCache._find_network_mount_darwin(directory)
        return RealizationCache._find_network_mount_linux(directory, mounts_path)

    @staticmethod
    def _find_network_mount_linux(directory: str, mounts_path: str) -> Optional[str]:
        """Return the network filesystem type from the mounts table, or None."""
        if not os.path.exists(mounts_path):
            return None
        best_point = ""
        best_type = ""
        with open(mounts_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3:
                    continue
                point, fs_type = parts[1], parts[2]
                if directory.startswith(point) and len(point) > len(best_point):
                    best_point = point
                    best_type = fs_type
        return best_type if best_type in _NETWORK_FILESYSTEMS else None

    @staticmethod
    def _find_network_mount_nt(directory: str) -> Optional[str]:
        """Return the Windows network share kind of a directory, or None."""
        try:
            import ctypes

            path = os.path.abspath(directory)
            if path.startswith("\\\\"):
                return "unc"
            drive, _tail = os.path.splitdrive(path)
            if not drive:
                return None
            drive_remote = 4
            if ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\") == drive_remote:
                return "remote drive"
            return None
        except Exception:
            return None

    @staticmethod
    def _find_network_mount_darwin(directory: str) -> Optional[str]:
        """Return the macOS network filesystem name of a directory, or None."""
        try:
            import ctypes
            import ctypes.util

            class StatFs(ctypes.Structure):
                _fields_ = [
                    ("f_bsize", ctypes.c_uint32),
                    ("f_iosize", ctypes.c_int32),
                    ("f_blocks", ctypes.c_uint64),
                    ("f_bfree", ctypes.c_uint64),
                    ("f_bavail", ctypes.c_uint64),
                    ("f_files", ctypes.c_uint64),
                    ("f_ffree", ctypes.c_uint64),
                    ("f_fsid", ctypes.c_uint64),
                    ("f_owner", ctypes.c_uint32),
                    ("f_type", ctypes.c_uint32),
                    ("f_flags", ctypes.c_uint32),
                    ("f_fssubtype", ctypes.c_uint32),
                    ("f_fstypename", ctypes.c_char * 16),
                    ("f_mntonname", ctypes.c_char * 1024),
                    ("f_mntfromname", ctypes.c_char * 1024),
                    ("f_flags_ext", ctypes.c_uint32),
                    ("f_reserved", ctypes.c_uint32 * 7),
                ]

            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            # The INODE64 symbol serves the 64 bit struct this layout matches.
            statfs = getattr(libc, "statfs$INODE64", libc.statfs)
            buffer = StatFs()
            if statfs(directory.encode("utf-8"), ctypes.byref(buffer)) != 0:
                return None
            fs_type = buffer.f_fstypename.decode("utf-8", "ignore")
            return fs_type if fs_type in _NETWORK_FILESYSTEMS else None
        except Exception:
            return None
