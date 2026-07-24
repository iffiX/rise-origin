# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the session journal: folding, resume, and rsc blobs."""

import pytest

from rise_origin.create.editor import EditorJournal, EditorJournalError
from rise_origin.rsc import essence


class TestEditorJournal:
    def test_head_and_ops_round_trip(self, tmp_path):
        session = str(tmp_path / "session")
        journal = EditorJournal(
            session, head={"name": "world", "material_requires": []}
        )
        journal.append("insert", {"child": 2}, {"remove": 2}, created=(2,))
        journal.append("move", {"node": 2}, {"node": 2}, touched=(2,))
        head, entries, max_id = EditorJournal.read(session)
        assert head["name"] == "world"
        assert [entry.op for entry in entries] == ["insert", "move"]
        assert max_id == 2

    def test_undo_and_retract_fold_on_read(self, tmp_path):
        session = str(tmp_path / "session")
        journal = EditorJournal(session, head={"name": "world"})
        journal.append("insert", {"child": 2}, {"remove": 2}, created=(2,))
        journal.append("insert", {"child": 3}, {"remove": 3}, created=(3,))
        journal.append("move", {"node": 2}, {"node": 2}, touched=(2,))
        journal.undo(1)
        journal.retract(2)
        _head, entries, _max_id = EditorJournal.read(session)
        assert [entry.entry_id for entry in entries] == [1]

    def test_require_lines_fold_into_the_head(self, tmp_path):
        session = str(tmp_path / "session")
        journal = EditorJournal(
            session, head={"name": "world", "material_requires": [{"name": "a"}]}
        )
        journal.append_require([{"name": "b"}, {"name": "c"}])
        head, _entries, _max_id = EditorJournal.read(session)
        assert head["material_requires"] == [
            {"name": "a"},
            {"name": "b"},
            {"name": "c"},
        ]

    def test_retract_refuses_dependents(self):
        journal = EditorJournal(None)
        journal.append("insert", {"child": 2}, {"remove": 2}, created=(2,))
        journal.append("move", {"node": 2}, {"node": 2}, touched=(2,))
        with pytest.raises(EditorJournalError):
            journal.retract(1)

    def test_resume_continues_entry_ids(self, tmp_path):
        session = str(tmp_path / "session")
        journal = EditorJournal(session, head={"name": "world"})
        journal.append("insert", {"child": 2}, {"remove": 2}, created=(2,))
        resumed, head = EditorJournal.resume(session)
        assert head["name"] == "world"
        entry = resumed.append("move", {"node": 2}, {"node": 2}, touched=(2,))
        assert entry.entry_id == 2

    def test_new_journal_refuses_an_existing_file(self, tmp_path):
        session = str(tmp_path / "session")
        EditorJournal(session, head={"name": "world"})
        with pytest.raises(EditorJournalError):
            EditorJournal(session, head={"name": "world"})

    def test_rscs_round_trip_through_session_blobs(self, tmp_path):
        session = str(tmp_path / "session")
        journal = EditorJournal(session, head={"name": "world"})
        index = journal.save_rsc(essence.Material("clay"))
        assert index == 0
        loaded = journal.load_rsc(index)
        assert loaded.name == "clay"
        resumed, _head = EditorJournal.resume(session)
        assert resumed.save_rsc(essence.Material("sand")) == 1
