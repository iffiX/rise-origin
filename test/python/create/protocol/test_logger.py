# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the create logger and its progress reporting."""

import logging

from rise_origin.create.protocol.logger import Logger


class TestLogger:
    def test_lines_carry_the_logger_name(self, caplog):
        with caplog.at_level(logging.INFO, logger="rise_origin"):
            Logger("rise_origin.nature.tree.willow").info("building")
        assert caplog.records[0].name == "rise_origin.nature.tree.willow"
        assert caplog.records[0].message == "building"

    def test_progress_throttles_and_completes(self, caplog):
        logger = Logger("rise_origin.nature.tree.willow")
        with caplog.at_level(logging.INFO, logger="rise_origin"):
            progress = logger.make_progress("sweeping limbs", total=100)
            for _ in range(100):
                progress.step()
            progress.finish()
        messages = [record.message for record in caplog.records]
        assert len(messages) <= 3
        assert "sweeping limbs 100/100 (100%)" in messages
        assert messages[-1].startswith("sweeping limbs done: 100 in")

    def test_progress_without_total_reports_counts(self, caplog):
        logger = Logger("rise_origin.create.editor")
        with caplog.at_level(logging.INFO, logger="rise_origin"):
            progress = logger.make_progress("loading")
            progress.step(5)
            progress.finish()
        assert caplog.records[-1].message.startswith("loading done: 5 in")
