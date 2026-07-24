# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The editor package: the working context and its session organs.

The Editor is the door: it owns the locked session folder, the journal,
the scene index, the material table, and a handle on the shared
realization cache; drafts and module bodies see only its access wall. The
organs are pure and testable on their own, and the folder manager names
every location the layer puts on disk.
"""

from .folder_manager import FolderManager, FolderManagerError
from .index import (
    SceneIndex,
    SceneIndexBlock,
    SceneIndexCellHit,
    SceneIndexError,
    SceneIndexOverlapHit,
)
from .journal import EditorJournal, EditorJournalEntry, EditorJournalError
from .cache import (
    RealizationCache,
    RealizationCacheError,
    RealizationCacheHit,
    RealizationCacheStatus,
)
from .material import (
    EditorMaterialDataTable,
    EditorMaterialError,
    EditorMaterialRequireTable,
)
from .editor import Editor, EditorAccess, EditorError

__all__ = [
    "FolderManager",
    "FolderManagerError",
    "SceneIndex",
    "SceneIndexBlock",
    "SceneIndexCellHit",
    "SceneIndexError",
    "SceneIndexOverlapHit",
    "EditorJournal",
    "EditorJournalEntry",
    "EditorJournalError",
    "RealizationCache",
    "RealizationCacheError",
    "RealizationCacheHit",
    "RealizationCacheStatus",
    "EditorMaterialDataTable",
    "EditorMaterialError",
    "EditorMaterialRequireTable",
    "Editor",
    "EditorAccess",
    "EditorError",
]
