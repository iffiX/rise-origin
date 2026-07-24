# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the scene index: both levels, queries, and distance."""

import fcl
import numpy as np
import pytest

from rise_origin import RQuat3rf, RVec3rf
from rise_origin import create as cr
from rise_origin.create.editor import SceneIndex, SceneIndexError
from rise_origin.rsc import entity


def make_quat_about_z(angle_rad: float) -> RQuat3rf:
    return RQuat3rf.unit().from_rotation_vector(RVec3rf(0.0, 0.0, angle_rad))


def make_structure(name: str = "block") -> entity.StructureVoxel:
    body = entity.StructureVoxelBody.box(2, 2, 2, voxel_size=0.1, material_sid=0)
    return entity.StructureVoxel(
        name, material_references=["stone"], bodies=[body], is_fixed=True
    )


class TestSceneIndex:
    def test_insert_twice_is_a_loud_error(self):
        index = SceneIndex()
        index.insert(1, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose())
        with pytest.raises(SceneIndexError):
            index.insert(1, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose())

    def test_query_ray_hits_the_nearest_box(self):
        index = SceneIndex()
        index.insert(1, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose((5, 0, 0)))
        index.insert(2, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose((9, 0, 0)))
        hit = index.query_ray((0.0, 0.5, 0.5), (1.0, 0.0, 0.0))
        assert hit is not None
        uid, t, normal = hit
        assert uid == 1
        assert np.isclose(t, 5.0)
        assert np.allclose(normal, [-1, 0, 0])

    def test_query_ray_respects_include_and_distance(self):
        index = SceneIndex()
        index.insert(1, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose((5, 0, 0)))
        index.insert(2, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose((9, 0, 0)))
        hit = index.query_ray((0.0, 0.5, 0.5), (1.0, 0.0, 0.0), include={2})
        assert hit is not None and hit[0] == 2
        assert index.query_ray((0.0, 0.5, 0.5), (1.0, 0.0, 0.0), distance=2.0) is None

    def test_refit_moves_the_entry(self):
        index = SceneIndex()
        index.insert(1, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose((5, 0, 0)))
        index.refit(1, cr.Pose((9, 0, 0)))
        hit = index.query_ray((0.0, 0.5, 0.5), (1.0, 0.0, 0.0))
        assert np.isclose(hit[1], 9.0)

    def test_remove_clears_the_entry(self):
        index = SceneIndex()
        index.insert(1, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose((5, 0, 0)))
        index.remove(1)
        assert index.query_ray((0.0, 0.5, 0.5), (1.0, 0.0, 0.0)) is None
        with pytest.raises(SceneIndexError):
            index.remove(1)

    def test_query_overlaps_reports_the_pair(self):
        index = SceneIndex()
        index.insert(1, [cr.Box((0, 0, 0), (2, 2, 2))], cr.Pose())
        index.insert(2, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose((10, 0, 0)))
        hits = index.query_overlaps([cr.Box((1, 1, 1), (3, 3, 3))], cr.Pose())
        assert [hit.uid for hit in hits] == [1]
        assert hits[0].own_box_index == 0

    def test_query_nearby_sorts_ascending(self):
        index = SceneIndex()
        index.insert(1, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose((3, 0, 0)))
        index.insert(2, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose((7, 0, 0)))
        results = index.query_nearby(
            [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose(), distance=10.0
        )
        assert [uid for uid, _ in results] == [1, 2]
        assert np.isclose(results[0][1].euclidean, 2.0)
        assert np.isclose(results[0][1].x, 2.0)

    def test_get_bounds_in_root_unions_the_uids(self):
        index = SceneIndex()
        index.insert(1, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose())
        index.insert(2, [cr.Box((0, 0, 0), (1, 1, 1))], cr.Pose((4, 0, 0)))
        lo, hi = index.get_bounds_in_root([1, 2])
        assert np.allclose(lo, [0, 0, 0])
        assert np.allclose(hi, [5, 1, 1])
        assert index.get_bounds_in_root([9]) is None

    def test_compute_distance_matches_brute_force(self):
        index = SceneIndex()
        pose_a = cr.Pose((0, 0, 0), make_quat_about_z(0.3))
        pose_b = cr.Pose((4, 1, 0), make_quat_about_z(-0.5))
        box = cr.Box((0, 0, 0), (1, 2, 1))
        index.insert(1, [box], pose_a)
        index.insert(2, [box], pose_b)
        result = index.compute_distance([1], [2])
        expected = TestSceneIndex._make_exact_distance(box, pose_a, box, pose_b)
        assert np.isclose(result.euclidean, expected, atol=1e-5)

    def test_compute_distance_overlap_is_zero(self):
        index = SceneIndex()
        index.insert(1, [cr.Box((0, 0, 0), (2, 2, 2))], cr.Pose())
        index.insert(2, [cr.Box((0, 0, 0), (2, 2, 2))], cr.Pose((1, 0, 0)))
        assert index.compute_distance([1], [2]).euclidean == 0.0

    def test_realize_answers_cell_hits_through_the_broadphase(self):
        index = SceneIndex()
        structure = make_structure()
        index.insert(1, [cr.Box((0, 0, 0), (0.2, 0.2, 0.2))], cr.Pose((1, 0, 0)))
        index.realize(1, [structure])
        hits = index.get_cell_hits_at_point((1.05, 0.05, 0.05))
        assert len(hits) == 1
        assert hits[0].uid == 1
        assert hits[0].material_reference == "stone"
        assert hits[0].is_fixed_structure
        assert index.get_cell_hits_at_point((5.0, 5.0, 5.0)) == []

    def test_refit_moves_the_blocks_too(self):
        index = SceneIndex()
        index.insert(1, [cr.Box((0, 0, 0), (0.2, 0.2, 0.2))], cr.Pose())
        index.realize(1, [make_structure()])
        index.refit(1, cr.Pose((2, 0, 0)))
        assert index.get_cell_hits_at_point((0.05, 0.05, 0.05)) == []
        assert len(index.get_cell_hits_at_point((2.05, 0.05, 0.05))) == 1

    @staticmethod
    def _make_exact_distance(box_a, pose_a, box_b, pose_b) -> float:
        def make_object(box, pose):
            lo = np.array([box.min.x, box.min.y, box.min.z])
            hi = np.array([box.max.x, box.max.y, box.max.z])
            rotation = np.asarray(
                pose.orientation.to_rotation_matrix().to_list(), dtype=float
            )
            position = pose.relative_position
            center = (lo + hi) * 0.5
            translation = (
                np.array([position.x, position.y, position.z]) + rotation @ center
            )
            geometry = fcl.Box(*(hi - lo))
            return fcl.CollisionObject(geometry, fcl.Transform(rotation, translation))

        return float(
            fcl.distance(
                make_object(box_a, pose_a),
                make_object(box_b, pose_b),
                fcl.DistanceRequest(),
                fcl.DistanceResult(),
            )
        )
