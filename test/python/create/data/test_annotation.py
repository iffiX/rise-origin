# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the annotation family: json round trips per kind."""

import pytest

from rise_origin import create as cr


class TestAnnotation:
    def test_json_round_trips_every_kind(self):
        annotations = [
            cr.AnnotationPoint((1, 2, 3)),
            cr.AnnotationPoint((1, 2, 3), (0, 0, 0, 1)),
            cr.AnnotationRegion(cr.AABB((0, 0, 0), (1, 1, 1))),
            cr.AnnotationPath(cr.Path(((0, 0, 0), (1, 1, 1)), smooth="none")),
            cr.AnnotationView(cr.Pose((0, 0, 1.7)), fov_deg=45.0),
            cr.AnnotationNote("north side intentionally bare"),
        ]
        for annotation in annotations:
            loaded = type(annotation).from_json(annotation.to_json())
            assert loaded == annotation

    def test_base_json_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            cr.Annotation().to_json()
        with pytest.raises(NotImplementedError):
            cr.Annotation.from_json({"type": "point"})
