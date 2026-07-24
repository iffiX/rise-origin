# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The session journal: an append-only op log the session rebuilds from.

Reading the journal file alone reproduces the output: the head line carries
the root name, the session title, and the editor's declared requires; every
insert and remove line carries its subtree's declared content, with
hand-made rscs saved as numbered sidecar blobs in the session folder and
module-born content rebuilt by identity through the cache. Undo and retract
append control lines and later require declarations append require lines,
keeping the file strictly append-only and every history linear; a session
branches by copying its folder, never inside the file. The journal is
bookkeeping only; the Draft applies ops and inverses.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from rise_origin import rsc as rsc_module

from .folder_manager import FolderManager

__all__ = ["EditorJournalError", "EditorJournalEntry", "EditorJournal"]


class EditorJournalError(RuntimeError):
    """A journal operation that cannot be applied."""


@dataclass
class EditorJournalEntry:
    """One journaled mutation.

    Attributes:
        entry_id: Monotonic id, unique for the session's lifetime.
        op: The mutation kind.
        args: Json-ready description of the applied op, content included.
        inverse: Json-ready description of what undo applies.
        created: Uids of nodes this entry brought into the tree.
        touched: Uids of nodes this entry referenced.
    """

    entry_id: int
    op: Literal["insert", "remove", "move", "mirror"]
    args: dict
    inverse: dict
    created: Tuple[int, ...] = ()
    touched: Tuple[int, ...] = ()


class EditorJournal:
    """Append-only op log with undo, retract, and cold-open reads.

    Args:
        session_dir: The session folder; None keeps the journal in memory
            only.
        head: The head json of a new session: the root name, the session
            title, and the editor's declared requires. Required when
            starting fresh with a session folder; refused when the folder
            already holds a journal.
    """

    def __init__(self, session_dir: Optional[str], head: Optional[dict] = None) -> None:
        self._session_dir = session_dir
        self._entries: List[EditorJournalEntry] = []
        self._next_id = 1
        self._next_rsc_index = 0
        self._memory_rscs: Dict[int, object] = {}
        if session_dir is None:
            return
        os.makedirs(session_dir, exist_ok=True)
        path = self._get_journal_path()
        if os.path.exists(path):
            raise EditorJournalError(
                f"'{path}' already holds a journal; open the session instead"
            )
        if head is None:
            raise EditorJournalError("a new session journal needs a head json")
        self._write_line({"head": head})

    def append(
        self,
        op: Literal["insert", "remove", "move", "mirror"],
        args: dict,
        inverse: dict,
        created: Tuple[int, ...] = (),
        touched: Tuple[int, ...] = (),
    ) -> EditorJournalEntry:
        """Record one applied op and return its entry.

        Args:
            op: The mutation kind.
            args: Json-ready description of the applied op, content included.
            inverse: Json-ready description of what undo applies.
            created: Uids of nodes this entry brought into the tree.
            touched: Uids of nodes this entry referenced.
        """
        entry = EditorJournalEntry(self._next_id, op, args, inverse, created, touched)
        self._next_id += 1
        self._entries.append(entry)
        self._write_line(
            {
                "op": op,
                "id": entry.entry_id,
                "args": args,
                "inverse": inverse,
                "created": list(created),
                "touched": list(touched),
            }
        )
        return entry

    def append_require(self, requires_json: List[dict]) -> None:
        """Record material requires declared after the head was written.

        A require line is a head extension, not an op: reading the journal
        folds it into the head's declared requires, and undo and retract
        never touch it.

        Args:
            requires_json: The material requires to record, json-ready.
        """
        self._write_line({"require": list(requires_json)})

    def undo(self, num: int = 1) -> List[EditorJournalEntry]:
        """Pop the last entries and return them newest first for reversal.

        Args:
            num: How many of the most recent entries to pop.
        """
        if num < 1 or num > len(self._entries):
            raise EditorJournalError(
                f"cannot undo {num} of {len(self._entries)} entries"
            )
        popped = self._entries[-num:][::-1]
        del self._entries[-num:]
        self._write_line({"undo": num})
        return popped

    def retract(self, entry_id: int) -> EditorJournalEntry:
        """Remove one older entry, refusing when later entries depend on it.

        A later entry depends on the retracted one when it references any
        node the retracted entry created.

        Args:
            entry_id: The entry to remove.

        Returns:
            The removed entry.
        """
        position = self._find(entry_id)
        entry = self._entries[position]
        created = set(entry.created)
        dependents = [
            later.entry_id
            for later in self._entries[position + 1 :]
            if created & (set(later.touched) | set(later.created))
        ]
        if dependents:
            raise EditorJournalError(
                f"entry {entry_id} has dependents {dependents}; retract them first"
            )
        del self._entries[position]
        self._write_line({"retract": entry_id})
        return entry

    def get_cursor(self) -> int:
        """Return the current cursor, the number of applied entries."""
        return len(self._entries)

    def get_entries(self) -> List[EditorJournalEntry]:
        """Return the applied entries, oldest first."""
        return list(self._entries)

    def save_rsc(self, rsc: object) -> int:
        """Save one hand-made rsc and return its file index.

        With a session folder the rsc becomes a numbered sidecar blob;
        without one it stays in memory, serving undo within the process.

        Args:
            rsc: The hand-made rsc to save.
        """
        index = self._next_rsc_index
        self._next_rsc_index += 1
        if self._session_dir is None:
            self._memory_rscs[index] = rsc
            return index
        blobs = FolderManager.get_session_blobs_path(self._session_dir)
        os.makedirs(blobs, exist_ok=True)
        rsc.save(os.path.join(blobs, f"{index}.rscb"))
        return index

    def load_rsc(self, index: int) -> object:
        """Load one hand-made rsc back by its file index.

        Args:
            index: The file index the rsc was saved under.
        """
        if self._session_dir is None:
            held = self._memory_rscs.get(index)
            if held is None:
                raise EditorJournalError(f"no rsc {index} in the memory journal")
            return held
        blobs = FolderManager.get_session_blobs_path(self._session_dir)
        return rsc_module.load(os.path.join(blobs, f"{index}.rscb"))

    @staticmethod
    def read(session_dir: str) -> Tuple[dict, List[EditorJournalEntry], int]:
        """Read a session journal file with its control lines folded in.

        Require lines fold into the head's declared requires; undo and
        retract lines fold into the surviving entries.

        Args:
            session_dir: The session folder to read the journal from.

        Returns:
            The head json, the surviving entries oldest first, and the
            largest entry id ever written.
        """
        path = FolderManager.get_session_journal_path(session_dir)
        if not os.path.exists(path):
            raise EditorJournalError(f"no journal at '{path}'")
        head: Optional[dict] = None
        entries: List[EditorJournalEntry] = []
        max_id = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                line_json = json.loads(line)
                if "head" in line_json:
                    head = line_json["head"]
                elif "require" in line_json:
                    if head is None:
                        raise EditorJournalError(f"'{path}' has no head line")
                    head.setdefault("material_requires", []).extend(
                        line_json["require"]
                    )
                elif "undo" in line_json:
                    del entries[-int(line_json["undo"]) :]
                elif "retract" in line_json:
                    wanted = int(line_json["retract"])
                    entries = [e for e in entries if e.entry_id != wanted]
                else:
                    entries.append(
                        EditorJournalEntry(
                            int(line_json["id"]),
                            line_json["op"],
                            line_json["args"],
                            line_json["inverse"],
                            tuple(line_json["created"]),
                            tuple(line_json["touched"]),
                        )
                    )
                    max_id = max(max_id, int(line_json["id"]))
        if head is None:
            raise EditorJournalError(f"'{path}' has no head line")
        return head, entries, max_id

    @staticmethod
    def resume(session_dir: str) -> Tuple["EditorJournal", dict]:
        """Return the journal of an existing session folder and its head.

        The returned journal holds the folded entries and continues
        appending to the same file with fresh entry ids.

        Args:
            session_dir: The existing session folder to resume.
        """
        head, entries, max_id = EditorJournal.read(session_dir)
        journal = EditorJournal(None)
        journal._session_dir = session_dir
        journal._entries = entries
        journal._next_id = max_id + 1
        journal._next_rsc_index = EditorJournal._get_next_rsc_index(session_dir)
        return journal, head

    def _find(self, entry_id: int) -> int:
        """Return the position of an entry id, or raise."""
        for position, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                return position
        raise EditorJournalError(f"no journal entry {entry_id}")

    def _get_journal_path(self) -> str:
        """Return the journal file path of the session folder."""
        return FolderManager.get_session_journal_path(self._session_dir)

    def _write_line(self, line_json: dict) -> None:
        """Append one json line when a session folder is configured."""
        if self._session_dir is None:
            return
        with open(self._get_journal_path(), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line_json, sort_keys=True) + "\n")

    @staticmethod
    def _get_next_rsc_index(session_dir: str) -> int:
        """Return the next free rsc file index of a session's blob folder."""
        blobs = FolderManager.get_session_blobs_path(session_dir)
        if not os.path.isdir(blobs):
            return 0
        highest = -1
        for name in os.listdir(blobs):
            stem, _, _extension = name.partition(".")
            if stem.isdigit():
                highest = max(highest, int(stem))
        return highest + 1
