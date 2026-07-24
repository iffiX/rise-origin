# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the query results: defaults and vector coercion."""

from rise_origin import RVec3rf
from rise_origin import create as cr


class TestResults:
    def test_filters_defaults(self):
        filters = cr.Filters(module="rise_origin.nature.tree.*", is_fixed=True)
        assert filters.tags == ()
        assert filters.under is None

    def test_overlap_coerces_vectors(self):
        overlap = cr.Overlap(None, (0, 0.1, 0), (0, 0.1, 0), ("static", "static"))
        assert isinstance(overlap.depth, RVec3rf)

    def test_placement_report_defaults(self):
        report = cr.PlacementReport(True)
        assert report.violations == ()
