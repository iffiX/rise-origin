# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the path: sampling and the json round trip."""

import numpy as np
import pytest

from rise_origin import create as cr


class TestPath:
    def test_straight_polyline_length(self):
        path = cr.Path(((0, 0, 0), (10, 0, 0)), smooth="none")
        points, lengths = path.resample()
        assert np.isclose(lengths[-1], 10.0)

    def test_smoothed_curve_passes_endpoints(self):
        path = cr.Path(((0, 0, 0), (5, 5, 0), (10, 0, 0)))
        points, lengths = path.resample()
        assert np.allclose(points[0], [0, 0, 0])
        assert np.allclose(points[-1], [10, 0, 0])

    def test_sample_at_station_interpolates(self):
        path = cr.Path(((0, 0, 0), (10, 0, 0)), smooth="none")
        points, lengths = path.resample()
        position, tangent = cr.Path.sample_at_station(points, lengths, 4.0)
        assert np.allclose(position, [4.0, 0.0, 0.0])
        assert np.allclose(tangent, [1.0, 0.0, 0.0])

    def test_single_waypoint_refused(self):
        with pytest.raises(ValueError):
            cr.Path(((0, 0, 0),)).resample()

    def test_json_round_trip(self):
        path = cr.Path(((0, 0, 0), (1, 2, 3)), smooth="none", min_radius=0.5)
        loaded = cr.Path.from_json(path.to_json())
        assert loaded == path
