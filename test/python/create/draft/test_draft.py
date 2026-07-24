# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the Draft on detached trees: tree ops, queries, solvers, json."""

import math

import numpy as np
import pytest

from rise_origin import RQuat3rf, RVec3rf
from rise_origin import create as cr
from rise_origin.rsc import entity, essence


class FakeAccess:
    """The minimal access wall a detached tree needs."""

    def __init__(self):
        self._next_uid = 0

    def allocate_uid(self):
        self._next_uid += 1
        return self._next_uid

    def is_session_node(self, node):
        return False


@pytest.fixture
def access():
    return FakeAccess()


def make_quat_about_z(angle_rad: float) -> RQuat3rf:
    return RQuat3rf.unit().from_rotation_vector(RVec3rf(0.0, 0.0, angle_rad))


def make_box_draft(access, name, size=(1.0, 1.0, 1.0), material_class="static"):
    return cr.Draft(access, name, boxes=[cr.Box((0, 0, 0), size, material_class)])


class TestTreeBasics:
    def test_ctor_allocates_uids_through_the_access(self, access):
        first = make_box_draft(access, "a")
        second = make_box_draft(access, "b")
        assert second.uid == first.uid + 1

    def test_has_floor_reads_the_rscs(self, access):
        floor = cr.Draft(access, "ground", rscs=[entity.FloorFlat("floor")])
        assert floor.has_floor
        assert not make_box_draft(access, "crate").has_floor

    def test_insert_stores_the_given_frame_pose(self, access):
        world = cr.Draft(access, "world")
        crate = make_box_draft(access, "crate")
        world.insert(crate, cr.Pose((2, 3, 0)))
        pose = world.get_pose(crate)
        assert np.allclose(
            [pose.relative_position.x, pose.relative_position.y], [2, 3], atol=1e-6
        )

    def test_nested_insert_composes_frames(self, access):
        world = cr.Draft(access, "world")
        desk = make_box_draft(access, "desk")
        lamp = make_box_draft(access, "lamp", size=(0.2, 0.2, 0.4))
        world.insert(desk, cr.Pose((5, 0, 0)))
        world.insert(lamp, cr.Pose((5.5, 0, 1.0)), parent=desk)
        lamp_in_desk = desk.get_pose(lamp)
        assert np.isclose(lamp_in_desk.relative_position.x, 0.5, atol=1e-6)
        desk.relative_pose = cr.Pose((10, 0, 0))
        lamp_in_world = world.get_pose(lamp)
        assert np.isclose(lamp_in_world.relative_position.x, 10.5, atol=1e-6)

    def test_sibling_names_uniquify(self, access):
        world = cr.Draft(access, "world")
        world.insert(make_box_draft(access, "crate"), cr.Pose())
        world.insert(make_box_draft(access, "crate"), cr.Pose((2, 0, 0)))
        assert {child.name for child in world.children} == {"crate", "crate_2"}

    def test_scope_errors(self, access):
        world = cr.Draft(access, "world")
        stray = make_box_draft(access, "stray")
        with pytest.raises(cr.DraftScopeError):
            world.get_pose(stray)
        with pytest.raises(cr.DraftScopeError):
            world.find_node_by_name("nothing")

    def test_placed_and_cross_access_inserts_refused(self, access):
        world = cr.Draft(access, "world")
        crate = make_box_draft(access, "crate")
        world.insert(crate, cr.Pose())
        with pytest.raises(cr.DraftError):
            world.insert(crate, cr.Pose())
        other = make_box_draft(FakeAccess(), "alien")
        with pytest.raises(cr.DraftError):
            world.insert(other, cr.Pose())

    def test_floor_refused_on_detached_trees(self, access):
        world = cr.Draft(access, "world")
        ground = cr.Draft(access, "ground", rscs=[entity.FloorFlat("floor")])
        with pytest.raises(cr.DraftError):
            world.insert(ground, cr.Pose())

    def test_effective_is_fixed_flows_down_only(self, access):
        world = cr.Draft(access, "world")
        shelf = cr.Draft(
            access, "shelf", boxes=[cr.Box((0, 0, 0), (1, 1, 2))], is_fixed=True
        )
        cup = make_box_draft(access, "cup", size=(0.1, 0.1, 0.1))
        world.insert(shelf, cr.Pose())
        world.insert(cup, cr.Pose((0, 0, 2)), parent=shelf)
        assert cup.get_effective_is_fixed()
        assert not world.get_effective_is_fixed()

    def test_supporters_and_dependents(self, access):
        world = cr.Draft(access, "world")
        desk = make_box_draft(access, "desk")
        lamp = make_box_draft(access, "lamp")
        world.insert(desk, cr.Pose())
        world.insert(lamp, cr.Pose(), parent=desk)
        assert world.get_supporters(lamp) == [desk, world]
        assert world.get_dependents(desk) == [lamp]


class TestQueries:
    def test_get_bounds_and_rotation_override(self, access):
        world = cr.Draft(access, "world")
        slab = make_box_draft(access, "slab", size=(4.0, 2.0, 1.0))
        world.insert(slab, cr.Pose((10, 0, 0)))
        bounds = world.get_bounds()
        assert np.allclose(
            [bounds.min.x, bounds.max.x, bounds.max.y], [10, 14, 2], atol=1e-6
        )
        turned = world.get_bounds(slab, rotation=make_quat_about_z(math.pi / 2))
        assert np.isclose(turned.max.y - turned.min.y, 4.0, atol=1e-5)

    def test_floor_sampling_on_the_floor_node_only(self, access):
        ground = cr.Draft(
            access, "ground", rscs=[entity.FloorFlat("floor", height=2.5)]
        )
        sample = ground.get_floor_at(1.0, 1.0)
        assert np.isclose(sample.height, 2.5)
        assert sample.slope_deg == 0.0
        other = make_box_draft(access, "crate")
        assert other.get_floor_at(0.0, 0.0) is None

    def test_cast_ray_and_surface_anchor(self, access):
        world = cr.Draft(access, "world")
        wall = make_box_draft(access, "wall", size=(0.2, 4.0, 3.0))
        world.insert(wall, cr.Pose((5, -2, 0)))
        hit = world.cast_ray((0, 0, 1.5), "+x")
        assert hit is not None and hit.blocker is wall
        assert np.isclose(hit.distance, 5.0, atol=1e-6)
        anchor = world.get_bound_surface_at((0, 0, 1.5), "+x")
        assert np.isclose(anchor.position.x, 5.0, atol=1e-6)
        assert np.allclose(
            [anchor.normal.x, anchor.normal.y, anchor.normal.z], [-1, 0, 0]
        )
        assert world.cast_ray((0, 0, 1.5), "-x") is None

    def test_get_distance_decomposes_by_axis(self, access):
        world = cr.Draft(access, "world")
        a = make_box_draft(access, "a")
        b = make_box_draft(access, "b")
        world.insert(a, cr.Pose())
        world.insert(b, cr.Pose((3, 0, 0)))
        result = world.get_distance(a, b)
        assert np.isclose(result.euclidean, 2.0, atol=1e-5)
        assert np.isclose(result.x, 2.0, atol=1e-5)
        assert result.y == 0.0

    def test_get_distance_excludes_the_nested_side(self, access):
        world = cr.Draft(access, "world")
        desk = make_box_draft(access, "desk", size=(2, 2, 1))
        lamp = make_box_draft(access, "lamp", size=(0.2, 0.2, 0.4))
        world.insert(desk, cr.Pose())
        world.insert(lamp, cr.Pose((0.9, 0.9, 1.0)), parent=desk)
        result = world.get_distance(desk, lamp)
        assert result.euclidean == 0.0

    def test_get_gap_axis_and_sweep_agree(self, access):
        world = cr.Draft(access, "world")
        a = make_box_draft(access, "a")
        b = make_box_draft(access, "b")
        world.insert(a, cr.Pose())
        world.insert(b, cr.Pose((4, 0.5, 0)))
        axis_gap = world.get_gap(a, b, "+x")
        sweep_gap = world.get_gap(a, b, (1.0, 0.0, 0.0))
        assert np.isclose(axis_gap, 3.0, atol=1e-6)
        assert np.isclose(sweep_gap, 3.0, atol=1e-6)
        assert world.get_gap(a, b, "+z") == np.inf

    def test_get_nearby_sorts_filters_and_takes_boxes(self, access):
        world = cr.Draft(access, "world")
        bench = make_box_draft(access, "bench")
        near = make_box_draft(access, "near")
        far = make_box_draft(access, "far")
        world.insert(bench, cr.Pose())
        world.insert(near, cr.Pose((3, 0, 0)))
        world.insert(far, cr.Pose((8, 0, 0)))
        hits = world.get_nearby(bench, distance=10.0)
        assert [hit.node.name for hit in hits] == ["near", "far"]
        filtered = world.get_nearby(
            bench, distance=10.0, filters=cr.Filters(exclude=("near",))
        )
        assert [hit.node.name for hit in filtered] == ["far"]
        boxed = world.get_nearby(cr.AABB((7, 0, 0), (9, 1, 1)), distance=0.5)
        assert [hit.node.name for hit in boxed] == ["far"]

    def test_get_surrounding_from_a_point(self, access):
        world = cr.Draft(access, "world")
        world.insert(make_box_draft(access, "crate"), cr.Pose((2, 0, 0)))
        hits = world.get_surrounding((0.0, 0.5, 0.5), distance=5.0)
        assert [hit.node.name for hit in hits] == ["crate"]

    def test_placement_validity_classification(self, access):
        world = cr.Draft(access, "world")
        wall = make_box_draft(access, "wall", size=(4, 0.3, 3))
        door = make_box_draft(access, "door", size=(1, 0.3, 2))
        crate = make_box_draft(access, "crate", material_class="dynamic")
        world.insert(wall, cr.Pose())
        world.insert(door, cr.Pose((1, 0, 0)))
        world.insert(crate, cr.Pose((2, 0, 0)))
        report = world.check_placement_validity(door)
        assert report.is_ok
        assert len(report.embeddings) == 1
        report = world.check_placement_validity(crate)
        assert not report.is_ok
        assert report.violations[0].other is wall

    def test_touching_faces_are_not_overlaps(self, access):
        world = cr.Draft(access, "world")
        a = make_box_draft(access, "a")
        b = make_box_draft(access, "b")
        world.insert(a, cr.Pose())
        world.insert(b, cr.Pose((1, 0, 0)))
        assert world.check_placement_validity(b).is_ok


class TestSolvers:
    def test_snap_pose_rests_on_top(self, access):
        world = cr.Draft(access, "world")
        desk = make_box_draft(access, "desk", size=(2, 1, 1))
        lamp = make_box_draft(access, "lamp", size=(0.2, 0.2, 0.4))
        world.insert(desk, cr.Pose((5, 0, 0)))
        pose = world.compute_snap_pose(lamp, "-z", (0, 0), desk, "+z", (0, 0))
        lamp_center = np.asarray(
            [pose.relative_position.x, pose.relative_position.y]
        ) + np.array([0.1, 0.1])
        assert np.allclose(lamp_center, [6.0, 0.5], atol=1e-5)
        assert np.isclose(pose.relative_position.z, 1.0, atol=1e-5)

    def test_facing_pose_turns_about_z_only(self, access):
        world = cr.Draft(access, "world")
        bench = make_box_draft(access, "bench")
        fountain = make_box_draft(access, "fountain")
        world.insert(bench, cr.Pose())
        world.insert(fountain, cr.Pose((-0.5, 6, 0)))
        pose = world.compute_facing_pose(bench, "+x", fountain)
        turned = pose.orientation.rotate_vec3d(RVec3rf(1, 0, 0))
        assert np.allclose([turned.x, turned.y, turned.z], [0, 1, 0], atol=1e-5)

    def test_along_poses_ride_the_path(self, access):
        world = cr.Draft(access, "world")
        thing = make_box_draft(access, "post")
        path = cr.Path(((0, 0, 0), (10, 0, 0)), smooth="none")
        poses = world.compute_along_poses(thing, "+x", path, num=3)
        assert len(poses) == 3
        assert np.isclose(poses[1].relative_position.x, 5.0, atol=1e-6)

    def test_grid_poses_stay_inside_the_box(self, access):
        world = cr.Draft(access, "world")
        tree = make_box_draft(access, "tree")
        region = cr.AABB((0, 0, 0), (12, 8, 0))
        poses = world.compute_grid_poses(tree, "+x", region, num_x=3, num_y=2)
        assert len(poses) == 6
        for pose in poses:
            assert 0 <= pose.relative_position.x <= 12
            assert 0 <= pose.relative_position.y <= 8

    def test_ring_poses_face_the_center(self, access):
        world = cr.Draft(access, "world")
        chair = make_box_draft(access, "chair")
        poses = world.compute_ring_poses(chair, "+x", (0, 0, 0), 2.0, num=4)
        assert len(poses) == 4
        first = poses[0]
        facing = first.orientation.rotate_vec3d(RVec3rf(1, 0, 0))
        inward = -np.asarray(
            [first.relative_position.x, first.relative_position.y, 0.0]
        )
        inward /= np.linalg.norm(inward)
        assert np.allclose([facing.x, facing.y, facing.z], inward, atol=1e-5)

    def test_lean_pose_touches_both_planes(self, access):
        world = cr.Draft(access, "world")
        floor = make_box_draft(access, "floor", size=(10, 10, 0.1))
        wall = make_box_draft(access, "wall", size=(10, 0.3, 5))
        ladder = make_box_draft(access, "ladder", size=(0.5, 0.1, 4))
        world.insert(floor, cr.Pose((0, -5, -0.1)))
        world.insert(wall, cr.Pose((0, 5, 0)))
        result = world.compute_lean_pose(
            ladder, "-z", "+y", floor, "+z", wall, "-y", angle=70.0
        )
        base, lean = result.contacts
        assert np.isclose(base.z, 0.0, atol=1e-5)
        assert np.isclose(lean.y, 5.0, atol=1e-5)
        assert result.tilt == 70.0


class TestDraftJson:
    def test_json_round_trip_keeps_declared_content(self, access):
        node = cr.Draft(
            access,
            "crate",
            boxes=[cr.Box((0, 0, 0), (1, 1, 1), "mixed")],
            requires=[
                cr.MaterialRequire(
                    "rise_origin.material.nature.plant", "oak_wood", 0.05
                )
            ],
            is_fixed=True,
            tags=("prop",),
            annotations={"top": cr.AnnotationPoint((0.5, 0.5, 1.0))},
        )
        child = make_box_draft(access, "lid")
        node.insert(child, cr.Pose((0, 0, 1)))
        rscs = {}

        def save_rsc(rsc):
            index = len(rscs)
            rscs[index] = rsc
            return index

        draft_json = node.to_json(save_rsc)
        rebuilt = cr.Draft.from_json(access, draft_json)
        assert rebuilt.name == "crate"
        assert rebuilt.is_fixed
        assert rebuilt.tags == {"prop"}
        assert (
            rebuilt.requires[0].get_world_material_reference()
            == "rise_origin.material.nature.plant:oak_wood"
        )
        assert isinstance(rebuilt.annotations["top"], cr.AnnotationPoint)
        assert rebuilt.children[0].name == "lid"
        assert np.isclose(
            rebuilt.children[0].relative_pose.relative_position.z, 1.0, atol=1e-6
        )

    def test_json_saves_and_reloads_hand_made_rscs(self, access):
        node = cr.Draft(access, "clay", rscs=[essence.Material("clay")])
        rscs = {}

        def save_rsc(rsc):
            index = len(rscs)
            rscs[index] = rsc
            return index

        draft_json = node.to_json(save_rsc)
        assert len(rscs) == 1

        def load_rsc(index):
            return rscs[index]

        rebuilt = cr.Draft.from_json(access, draft_json, load_rsc)
        assert rebuilt.rscs[0].name == "clay"

    def test_module_content_skips_rsc_saving(self, access):
        node = cr.Draft(access, "maple", rscs=[essence.Material("bark")])
        node.identity = cr.Identity("rise_origin.nature.tree.maple", "1.0.0", "{}")

        def save_rsc(_rsc):
            raise AssertionError("module content must not save rscs")

        draft_json = node.to_json(save_rsc, is_identity_content_saved=False)
        assert draft_json["rscs"] == []

    def test_non_deterministic_content_saves_regardless(self, access):
        node = cr.Draft(access, "scree", rscs=[essence.Material("stone")])
        node.identity = cr.Identity("rise_origin.nature.rock.scree", "1.0.0", "{}")
        node.is_deterministic = False
        rscs = {}

        def save_rsc(rsc):
            index = len(rscs)
            rscs[index] = rsc
            return index

        draft_json = node.to_json(save_rsc, is_identity_content_saved=False)
        assert draft_json["rscs"] == [0]
        assert not draft_json["is_deterministic"]

        def load_rsc(index):
            return rscs[index]

        rebuilt = cr.Draft.from_json(access, draft_json, load_rsc)
        assert not rebuilt.is_deterministic
        assert rebuilt.is_realized and rebuilt.rscs[0].name == "stone"

    def test_json_without_the_deterministic_key_defaults_true(self, access):
        node = cr.Draft(access, "old")
        draft_json = node.to_json(None)
        del draft_json["is_deterministic"]
        rebuilt = cr.Draft.from_json(access, draft_json)
        assert rebuilt.is_deterministic

    def test_json_keeps_has_floor_without_loading(self, access):
        ground = cr.Draft(access, "ground", rscs=[entity.FloorFlat("floor")])
        rscs = {}

        def save_rsc(rsc):
            index = len(rscs)
            rscs[index] = rsc
            return index

        draft_json = ground.to_json(save_rsc)
        rebuilt = cr.Draft.from_json(access, draft_json)
        assert rebuilt.has_floor
        assert rebuilt.rscs == []
