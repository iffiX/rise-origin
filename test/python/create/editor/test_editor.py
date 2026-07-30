# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the Editor: sessions, lazy create, compile, and cold opens."""

import contextlib
import http.server
import itertools
import json
import math
import os
import threading

import numpy as np
import pytest

from rise_origin import create as cr
from rise_origin import rsc as rsc_module
from rise_origin.create.constants import SOFT_INTEGRATION_FREQUENCY_MULTIPLIER
from rise_origin.rsc import entity, essence
from rise_origin.rsc.simulation import Simulation, Time

CLAY = "rise_origin.test.material.clay"
FOAM = "rise_origin.test.material.foam"
STEEL = "rise_origin.test.material.steel"
CUBE = "rise_origin.test.block.cube"
DICE = "rise_origin.test.block.dice"

FOAM_MODULUS = 1e6
FOAM_DENSITY = 1e3


@cr.module(name=CLAY, version="1.0.0")
def clay(editor_access, settings):
    """A fixed test clay."""
    return cr.Draft(
        editor_access,
        "clay",
        rscs=[
            essence.Material(f"{CLAY}:clay"),
            essence.MaterialPhysicalPropertyFixed(
                f"{CLAY}:clay#fixed", material_reference=f"{CLAY}:clay"
            ),
        ],
    )


@cr.module(name=FOAM, version="1.0.0")
def foam(editor_access, settings):
    """A soft test foam."""
    return cr.Draft(
        editor_access,
        "foam",
        rscs=[
            essence.Material(f"{FOAM}:foam"),
            essence.MaterialPhysicalPropertySoft(
                f"{FOAM}:foam#soft",
                material_reference=f"{FOAM}:foam",
                elastic_modulus=FOAM_MODULUS,
                density=FOAM_DENSITY,
            ),
        ],
    )


@cr.module(name=STEEL, version="1.0.0")
def steel(editor_access, settings):
    """A rigid test steel."""
    return cr.Draft(
        editor_access,
        "steel",
        rscs=[
            essence.Material(f"{STEEL}:steel"),
            essence.MaterialPhysicalPropertyRigid(
                f"{STEEL}:steel#rigid", material_reference=f"{STEEL}:steel", density=8e3
            ),
        ],
    )


@cr.module(
    name=CUBE,
    version="1.0.0",
    settings=cr.settings(size=cr.Float(0.1, 1.0, default=0.2)),
    dependencies=(CLAY,),
)
def cube(editor_access, settings):
    """A fixed test cube of clay."""
    size = settings.size
    return cr.Draft(
        editor_access,
        "cube",
        boxes=[cr.Box((0, 0, 0), (size, size, size), "static")],
        rscs=[make_structure("cube", f"{CLAY}:clay")],
        requires=[cr.MaterialRequire(f"{CLAY}@^1", "clay", 0.1)],
    )


BRICK = "rise_origin.test.block.brick"


@cr.module(
    name=BRICK,
    version="1.0.0",
    settings=cr.settings(
        voxel_size=cr.VoxelSize(default="10cm"),
        occupancy=cr.Array3D(default=None),
    ),
    dependencies=(CLAY,),
)
def brick(editor_access, settings):
    """A fixed clay block reading decoded rich setting values."""
    size = 2.0 * settings.voxel_size
    return cr.Draft(
        editor_access,
        "brick",
        boxes=[cr.Box((0, 0, 0), (size, size, size), "static")],
        rscs=[make_structure("brick", f"{CLAY}:clay", voxel_size=settings.voxel_size)],
        requires=[cr.MaterialRequire(f"{CLAY}@^1", "clay", settings.voxel_size)],
    )


_dice_rolls = itertools.count()


@cr.module(name=DICE, version="1.0.0", dependencies=(CLAY,), is_deterministic=False)
def dice(editor_access, settings):
    """A clay block whose voxel count rolls anew on every build."""
    roll = next(_dice_rolls)
    body = entity.StructureVoxelBody.box(
        2, 2, 2 + roll % 3, voxel_size=0.1, material_sid=0, segment_sid=0
    )
    return cr.Draft(
        editor_access,
        "dice",
        boxes=[cr.Box((0, 0, 0), (0.2, 0.2, 0.5), "static")],
        rscs=[
            entity.StructureVoxel(
                "dice",
                material_references=[f"{CLAY}:clay"],
                bodies=[body],
                is_fixed=True,
            )
        ],
        requires=[cr.MaterialRequire(f"{CLAY}@^1", "clay", 0.1)],
    )


def make_editor(tmp_path, name="world", **kwargs):
    return cr.Editor(
        name,
        session_dir=str(tmp_path / "session"),
        cache=cr.RealizationCache(str(tmp_path / "cache")),
        **kwargs,
    )


def make_structure(name, material, voxel_size=0.1, is_fixed=True):
    body = entity.StructureVoxelBody.box(
        2, 2, 2, voxel_size=voxel_size, material_sid=0, segment_sid=0
    )
    return entity.StructureVoxel(
        name,
        material_references=[material],
        bodies=[body],
        is_fixed=is_fixed,
    )


def make_block_draft(
    editor, name, module=CLAY, material="clay", is_fixed=True, extra_rscs=()
):
    return cr.Draft(
        editor.root.editor_access,
        name,
        boxes=[cr.Box((0, 0, 0), (0.2, 0.2, 0.2), "static")],
        rscs=[
            make_structure(name, f"{module}:{material}", is_fixed=is_fixed),
            *extra_rscs,
        ],
        requires=[cr.MaterialRequire(f"{module}@^1", material, 0.1)],
    )


def make_canonical_bytes(path):
    canonical = path.replace(".rscb", ".canonical.rscb")
    rsc_module.save(rsc_module.load(path), canonical)
    with open(canonical, "rb") as handle:
        return handle.read()


@contextlib.contextmanager
def run_mcp_stub(queues):
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length))
            name = request["params"]["name"]
            calls.append((name, request["params"]["arguments"]))
            queue = queues.get(name)
            inner = queue.pop(0) if queue else {"ok": True}
            body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(inner)}]
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/mcp", calls
    finally:
        server.shutdown()


class TestEditorSession:
    def test_the_root_is_editor_born_with_uid_zero(self, tmp_path):
        with make_editor(tmp_path, title="a test") as editor:
            assert editor.root.uid == 0
            assert editor.root.name == "world"
            assert editor.title == "a test"

    def test_none_session_dir_allocates_under_the_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RISE_SESSION_DIR", str(tmp_path / "sessions"))
        with cr.Editor(
            "world", cache=cr.RealizationCache(str(tmp_path / "cache"))
        ) as editor:
            assert editor.session_dir.startswith(str(tmp_path / "sessions"))
            assert os.path.isfile(os.path.join(editor.session_dir, "journal.jsonl"))

    def test_a_held_session_folder_refuses_a_second_editor(self, tmp_path):
        editor = make_editor(tmp_path)
        with pytest.raises(cr.FolderManagerError):
            cr.Editor.open(editor.session_dir, cache=editor.cache)
        editor.close()
        reopened = cr.Editor.open(editor.session_dir, cache=editor.cache)
        reopened.close()


class TestEditorCreate:
    def test_a_miss_returns_realized_and_a_hit_declared_only(self, tmp_path):
        with make_editor(tmp_path) as editor:
            first = editor.create(CUBE, {"size": 0.2})
            assert first.rscs and first.identity is not None
            second = editor.create(CUBE, {"size": 0.2})
            assert not second.rscs
            assert second.identity == first.identity
            assert [box.to_json() for box in second.boxes] == [
                box.to_json() for box in first.boxes
            ]

    def test_check_placement_validity_answers_before_insert(self, tmp_path):
        with make_editor(tmp_path) as editor:
            placed = make_block_draft(editor, "placed")
            editor.insert(placed, cr.Pose())
            mover = cr.Draft(
                editor.root.editor_access,
                "mover",
                boxes=[cr.Box((0, 0, 0), (0.2, 0.2, 0.2), "dynamic")],
            )
            bad = editor.check_placement_validity(mover, cr.Pose((0.1, 0.0, 0.0)))
            assert not bad.is_ok and bad.violations
            assert bad.violations[0].other is placed
            good = editor.check_placement_validity(mover, cr.Pose((5.0, 0.0, 0.0)))
            assert good.is_ok and not good.violations
            assert mover.parent is None

    def test_a_static_overlap_reports_a_legal_embedding(self, tmp_path):
        with make_editor(tmp_path) as editor:
            placed = make_block_draft(editor, "placed")
            editor.insert(placed, cr.Pose())
            block = make_block_draft(editor, "block")
            report = editor.check_placement_validity(block, cr.Pose((0.1, 0.0, 0.0)))
            assert report.is_ok and report.embeddings

    def test_is_realized_tracks_the_content_lifecycle(self, tmp_path):
        with make_editor(tmp_path) as editor:
            first = editor.create(CUBE)
            assert first.is_realized
            second = editor.create(CUBE)
            assert not second.is_realized
            editor.insert(second, cr.Pose())
            editor.realize()
            assert second.is_realized
            assert second.rscs

    def test_a_miss_with_a_nested_hit_returns_a_full_tree(self, tmp_path):
        inner = "rise_origin.test.block.inner"
        outer = "rise_origin.test.block.outer"

        @cr.module(name=inner, version="1.0.0")
        def inner_module(editor_access, settings):
            """An inner clay block."""
            return cr.Draft(
                editor_access,
                "inner",
                boxes=[cr.Box((0, 0, 0), (0.2, 0.2, 0.2), "static")],
                rscs=[make_structure("inner", f"{CLAY}:clay")],
                requires=[cr.MaterialRequire(f"{CLAY}@^1", "clay", 0.1)],
            )

        @cr.module(name=outer, version="1.0.0")
        def outer_module(editor_access, settings):
            """An outer group holding the inner module."""
            root = cr.Draft(editor_access, "outer")
            root.insert(editor_access.create(inner), cr.Pose())
            return root

        with make_editor(tmp_path) as editor:
            editor.create(inner)
            placed = editor.create(outer)
            inner_node = placed.find_node_by_name("inner")
            assert inner_node.rscs
            assert placed.is_realized and inner_node.is_realized

    def test_settings_validate_against_the_schema(self, tmp_path):
        with make_editor(tmp_path) as editor:
            with pytest.raises(cr.SettingsError):
                editor.create(CUBE, {"size": 5.0})
            with pytest.raises(cr.SettingsError):
                editor.create(CUBE, {"stranger": 1})

    def test_an_evicted_entry_rebuilds_through_the_body(self, tmp_path):
        with make_editor(tmp_path) as editor:
            block = editor.create(CUBE)
            editor.cache.prune(0)
            editor.insert(block, cr.Pose())
            update = editor.compile()
            names = [
                instruction.rsc.name
                for instruction in update.instructions
                if instruction.op == "add"
            ]
            assert "cube/cube" in names

    def test_an_evicted_entry_with_rich_settings_rebuilds(self, tmp_path):
        with make_editor(tmp_path) as editor:
            block = editor.create(
                BRICK,
                {
                    "voxel_size": "10cm",
                    "occupancy": np.ones((2, 2, 2), dtype=bool),
                },
            )
            editor.cache.prune(0)
            editor.insert(block, cr.Pose())
            update = editor.compile()
            names = [
                instruction.rsc.name
                for instruction in update.instructions
                if instruction.op == "add"
            ]
            assert "brick/brick" in names


class TestEditorNonDeterministic:
    def test_the_inserted_world_carries_the_built_content(self, tmp_path):
        with make_editor(tmp_path) as editor:
            block = editor.create(DICE)
            built_z = block.rscs[0].bodies[0].z_voxels
            editor.insert(block, cr.Pose())
            path = str(tmp_path / "world.rscb")
            editor.save(path)
            world = rsc_module.load(path)
            placed = [held for held in world.entities if held.name == "dice/dice"]
            assert len(placed) == 1
            assert placed[0].bodies[0].z_voxels == built_z

    def test_an_insert_records_content_blobs_in_the_journal(self, tmp_path):
        with make_editor(tmp_path) as editor:
            blobs = cr.FolderManager.get_session_blobs_path(editor.session_dir)

            def count_blobs():
                return len(os.listdir(blobs)) if os.path.isdir(blobs) else 0

            editor.insert(editor.create(DICE), cr.Pose())
            recorded = count_blobs()
            assert recorded > 0
            editor.insert(editor.create(CUBE), cr.Pose((1.0, 0.0, 0.0)))
            assert count_blobs() == recorded

    def test_a_prune_and_cold_open_replays_the_recorded_bytes(self, tmp_path):
        editor = make_editor(tmp_path)
        editor.insert(editor.create(DICE), cr.Pose())
        before = str(tmp_path / "before.rscb")
        editor.save(before)
        editor.cache.prune(0)
        editor.close()
        reopened = cr.Editor.open(editor.session_dir, cache=editor.cache)
        try:
            after = str(tmp_path / "after.rscb")
            reopened.save(after)
            assert make_canonical_bytes(before) == make_canonical_bytes(after)
        finally:
            reopened.close()

    def test_a_lazy_hit_realizes_at_insert(self, tmp_path):
        with make_editor(tmp_path) as editor:
            first = editor.create(DICE)
            second = editor.create(DICE)
            assert not second.is_realized
            editor.insert(second, cr.Pose())
            assert second.is_realized and second.rscs
            assert second.rscs[0].bodies[0].z_voxels == first.rscs[0].bodies[0].z_voxels

    def test_an_undone_removal_survives_a_prune_and_a_cold_open(self, tmp_path):
        editor = make_editor(tmp_path)
        editor.insert(editor.create(DICE), cr.Pose())
        editor.remove("dice")
        editor.undo(1)
        before = str(tmp_path / "before.rscb")
        editor.save(before)
        editor.cache.prune(0)
        editor.close()
        reopened = cr.Editor.open(editor.session_dir, cache=editor.cache)
        try:
            after = str(tmp_path / "after.rscb")
            reopened.save(after)
            assert make_canonical_bytes(before) == make_canonical_bytes(after)
        finally:
            reopened.close()

    def test_a_deterministic_body_refuses_a_non_deterministic_create(self, tmp_path):
        steady = "rise_origin.test.block.steady"

        @cr.module(name=steady, version="1.0.0")
        def steady_module(editor_access, settings):
            """A deterministic group wrongly nesting the dice."""
            root = cr.Draft(editor_access, "steady")
            root.insert(editor_access.create(DICE), cr.Pose())
            return root

        with make_editor(tmp_path) as editor:
            with pytest.raises(cr.ModuleError, match="is_deterministic=False"):
                editor.create(steady)


class TestEditorFreeze:
    def test_requires_refuse_to_extend_when_frozen(self, tmp_path):
        with make_editor(tmp_path) as editor:
            editor.require(cr.MaterialRequire(f"{CLAY}@^1", "clay", 0.1))
            editor.set_simulation_rsc_frozen(True)
            with pytest.raises(cr.EditorMaterialError):
                editor.require(cr.MaterialRequire(f"{FOAM}@^1", "foam", 0.1))
            editor.set_simulation_rsc_frozen(False)
            editor.require(cr.MaterialRequire(f"{FOAM}@^1", "foam", 0.1))

    def test_an_uncovered_require_faults_at_creation(self, tmp_path):
        with make_editor(tmp_path) as editor:
            editor.set_simulation_rsc_frozen(True)
            with pytest.raises(cr.EditorError):
                editor.create(CUBE)

    def test_a_covered_require_creates_after_the_freeze(self, tmp_path):
        with make_editor(tmp_path) as editor:
            editor.require(cr.MaterialRequire(f"{CLAY}@^1", "clay", 0.1))
            editor.set_simulation_rsc_frozen(True)
            block = editor.create(CUBE)
            assert block.identity is not None

    def test_a_soft_envelope_below_the_frozen_step_faults(self, tmp_path):
        soft_block = "rise_origin.test.block.jelly"

        @cr.module(name=soft_block, version="1.0.0")
        def jelly(editor_access, settings):
            """A private soft test block."""
            return cr.Draft(
                editor_access,
                "jelly",
                boxes=[cr.Box((0, 0, 0), (0.1, 0.1, 0.1), "dynamic")],
                rscs=[
                    essence.Material("jelly_private"),
                    essence.MaterialPhysicalPropertySoft(
                        "jelly_private#soft",
                        material_reference="jelly_private",
                        elastic_modulus=1e8,
                        density=1e3,
                    ),
                    make_structure("jelly", "jelly_private", voxel_size=0.01),
                ],
            )

        with make_editor(tmp_path) as editor:
            editor.set_simulation_rsc_frozen(True, base=Simulation(time=Time(dt=0.005)))
            with pytest.raises(cr.EditorError):
                editor.create(soft_block)


class TestEditorSimulation:
    def test_declared_soft_requires_pin_the_step(self, tmp_path):
        with make_editor(tmp_path) as editor:
            editor.require(cr.MaterialRequire(f"{FOAM}@^1", "foam", 0.02))
            simulation = editor.get_simulation_rsc()
            expected = 0.02 / (
                SOFT_INTEGRATION_FREQUENCY_MULTIPLIER
                * math.sqrt(FOAM_MODULUS / FOAM_DENSITY)
            )
            assert simulation.time.dt == pytest.approx(expected, rel=1e-5)
            assert simulation.time.dt_contact_update == simulation.time.dt

    def test_rigid_requires_pin_the_linear_rule(self, tmp_path):
        with make_editor(tmp_path) as editor:
            editor.require(cr.MaterialRequire(f"{STEEL}@^1", "steel", 0.1))
            simulation = editor.get_simulation_rsc()
            assert simulation.time.dt == pytest.approx(0.025, rel=1e-6)

    def test_a_smaller_base_step_wins(self, tmp_path):
        with make_editor(tmp_path) as editor:
            editor.require(cr.MaterialRequire(f"{STEEL}@^1", "steel", 0.1))
            simulation = editor.get_simulation_rsc(base=Simulation(time=Time(dt=1e-5)))
            assert simulation.time.dt == pytest.approx(1e-5, rel=1e-9)

    def test_dt_reads_the_world_pinned_module_version(self, tmp_path):
        pinned = "rise_origin.test.material.pinned"
        for version, modulus in (("1.0.0", 1e6), ("1.2.0", 1e8)):

            @cr.module(name=pinned, version=version)
            def pinned_module(editor_access, settings, held_modulus=modulus):
                """A soft test material with a version-specific stiffness."""
                return cr.Draft(
                    editor_access,
                    "pinned",
                    rscs=[
                        essence.Material(f"{pinned}:goo"),
                        essence.MaterialPhysicalPropertySoft(
                            f"{pinned}:goo#soft",
                            material_reference=f"{pinned}:goo",
                            elastic_modulus=held_modulus,
                            density=1e3,
                        ),
                    ],
                )

        with make_editor(tmp_path) as editor:
            editor.require(
                cr.MaterialRequire(f"{pinned}@^1", "goo", 0.1),
                cr.MaterialRequire(f"{pinned}@1.0.0", "goo", 0.1),
            )
            simulation = editor.get_simulation_rsc()
            expected = 0.1 / (
                SOFT_INTEGRATION_FREQUENCY_MULTIPLIER * math.sqrt(1e6 / 1e3)
            )
            assert simulation.time.dt == pytest.approx(expected, rel=1e-5)

    def test_a_derived_editor_overrides_the_policy(self, tmp_path):
        class PinnedEditor(cr.Editor):
            def get_simulation_rsc(self, base=None, node=None):
                return Simulation(time=Time(dt=0.5))

        editor = PinnedEditor(
            "world",
            session_dir=str(tmp_path / "session"),
            cache=cr.RealizationCache(str(tmp_path / "cache")),
        )
        try:
            assert editor.get_simulation_rsc().time.dt == pytest.approx(0.5)
        finally:
            editor.close()


class TestEditorCompile:
    def test_the_first_compile_orders_materials_before_entities(self, tmp_path):
        with make_editor(tmp_path) as editor:
            editor.insert(editor.create(CUBE), cr.Pose())
            update = editor.compile()
            kinds = [
                (
                    "essence"
                    if isinstance(instruction.rsc, essence.Material)
                    or "#" in getattr(instruction.rsc, "name", "")
                    else "entity"
                )
                for instruction in update.instructions
            ]
            assert kinds == ["essence"] * kinds.count("essence") + [
                "entity"
            ] * kinds.count("entity")
            names = [instruction.rsc.name for instruction in update.instructions]
            assert names[-1] == "cube/cube"

    def test_compiles_are_incremental_against_the_editor_memo(self, tmp_path):
        with make_editor(tmp_path) as editor:
            block = editor.create(CUBE)
            editor.insert(block, cr.Pose())
            editor.compile()
            assert editor.compile().instructions == ()
            editor.move(block, cr.Pose((1.0, 0.0, 0.0)))
            update = editor.compile()
            assert [instruction.op for instruction in update.instructions] == ["move"]
            assert update.instructions[0].entity_name == "cube/cube"

    def test_a_removed_node_compiles_a_delete(self, tmp_path):
        with make_editor(tmp_path) as editor:
            block = editor.create(CUBE)
            editor.insert(block, cr.Pose())
            editor.compile()
            editor.remove(block)
            update = editor.compile()
            assert [instruction.op for instruction in update.instructions] == ["delete"]

    def test_a_fresh_memo_emits_the_full_world(self, tmp_path):
        with make_editor(tmp_path) as editor:
            editor.insert(editor.create(CUBE), cr.Pose())
            editor.compile()
            update = editor.compile(memo=cr.CompileMemo())
            assert any(instruction.op == "add" for instruction in update.instructions)

    def test_static_embeddings_pass_the_gate_and_dynamic_faults(self, tmp_path):
        with make_editor(tmp_path) as editor:
            anchor = make_block_draft(editor, "anchor")
            anchor.boxes = [cr.Box((0, 0, 0), (0.2, 0.2, 0.2), "mixed")]
            guest = make_block_draft(
                editor,
                "guest",
                extra_rscs=[
                    essence.Material(f"{CLAY}:clay"),
                    essence.MaterialPhysicalPropertyFixed(
                        f"{CLAY}:clay#fixed", material_reference=f"{CLAY}:clay"
                    ),
                ],
            )
            editor.insert(anchor, cr.Pose())
            editor.insert(guest, cr.Pose((0.05, 0.05, 0.05)))
            editor.compile()

    def test_a_dynamic_co_occupation_faults_the_gate(self, tmp_path):
        with make_editor(tmp_path) as editor:
            anchor = make_block_draft(editor, "anchor")
            anchor.boxes = [cr.Box((0, 0, 0), (0.2, 0.2, 0.2), "mixed")]
            mover = make_block_draft(
                editor, "mover", module=STEEL, material="steel", is_fixed=False
            )
            editor.insert(anchor, cr.Pose())
            editor.insert(mover, cr.Pose((0.05, 0.05, 0.05)))
            with pytest.raises(cr.EditorError):
                editor.compile()


class TestEditorSessionFiles:
    def test_save_writes_a_loadable_world(self, tmp_path):
        with make_editor(tmp_path) as editor:
            editor.insert(editor.create(CUBE), cr.Pose())
            path = str(tmp_path / "world.rscb")
            editor.save(path)
            world = rsc_module.load(path)
            assert [held.name for held in world.entities] == ["cube/cube"]
            assert f"{CLAY}:clay" in [held.name for held in world.essences]

    def test_a_cold_open_rebuilds_the_same_world(self, tmp_path):
        editor = make_editor(tmp_path)
        first = editor.create(CUBE)
        second = editor.create(CUBE)
        editor.insert(first, cr.Pose())
        editor.insert(second, cr.Pose((1.0, 0.0, 0.0)))
        editor.move(second, cr.Pose((2.0, 0.0, 0.0)))
        recorded_uids = sorted(editor._nodes_by_uid)
        before = str(tmp_path / "before.rscb")
        editor.save(before)
        editor.close()
        reopened = cr.Editor.open(editor.session_dir, cache=editor.cache)
        try:
            assert sorted(reopened._nodes_by_uid) == recorded_uids
            after = str(tmp_path / "after.rscb")
            reopened.save(after)
            assert make_canonical_bytes(before) == make_canonical_bytes(after)
        finally:
            reopened.close()

    def test_an_undone_removal_survives_a_cold_open(self, tmp_path):
        editor = make_editor(tmp_path)
        block = editor.create(CUBE)
        editor.insert(block, cr.Pose())
        editor.remove("cube")
        editor.undo(1)
        editor.move("cube", cr.Pose((1.0, 0.0, 0.0)))
        before = str(tmp_path / "before.rscb")
        editor.save(before)
        editor.close()
        reopened = cr.Editor.open(editor.session_dir, cache=editor.cache)
        try:
            after = str(tmp_path / "after.rscb")
            reopened.save(after)
            assert make_canonical_bytes(before) == make_canonical_bytes(after)
        finally:
            reopened.close()

    def test_requires_declared_late_survive_a_cold_open(self, tmp_path):
        editor = make_editor(tmp_path)
        editor.require(cr.MaterialRequire(f"{CLAY}@^1", "clay", 0.1))
        editor.close()
        reopened = cr.Editor.open(editor.session_dir, cache=editor.cache)
        try:
            names = [
                require.get_world_material_reference()
                for require in reopened._require_table.get_requires()
            ]
            assert names == [f"{CLAY}:clay"]
        finally:
            reopened.close()

    def test_a_fork_deviates_without_touching_the_original(self, tmp_path):
        with make_editor(tmp_path) as editor:
            editor.insert(editor.create(CUBE), cr.Pose())
            fork = editor.fork(title="deviation")
            try:
                assert fork.session_dir != editor.session_dir
                assert fork.title == "deviation"
                assert sorted(fork._nodes_by_uid) == sorted(editor._nodes_by_uid)
                fork.insert(fork.create(CUBE), cr.Pose((1.0, 0.0, 0.0)))
                assert len(fork._journal.get_entries()) == 2
                assert len(editor._journal.get_entries()) == 1
            finally:
                fork.close()


class TestEditorMaterials:
    def test_in_tree_materials_satisfy_without_a_module(self, tmp_path):
        with make_editor(tmp_path) as editor:
            block = make_block_draft(
                editor,
                "block",
                module="rise_origin.test.material.private",
                material="own",
                extra_rscs=[
                    essence.Material("rise_origin.test.material.private:own"),
                    essence.MaterialPhysicalPropertyFixed(
                        "rise_origin.test.material.private:own#fixed",
                        material_reference="rise_origin.test.material.private:own",
                    ),
                ],
            )
            editor.insert(block, cr.Pose())
            update = editor.compile()
            names = [
                instruction.rsc.name
                for instruction in update.instructions
                if instruction.op == "add"
            ]
            assert "rise_origin.test.material.private:own" in names

    def test_missing_materials_list_together(self, tmp_path):
        with make_editor(tmp_path) as editor:
            block = make_block_draft(editor, "block")
            block.requires = [
                cr.MaterialRequire("rise_origin.test.material.unknown_a", "one", 0.1),
                cr.MaterialRequire("rise_origin.test.material.unknown_b", "two", 0.1),
            ]
            editor.insert(block, cr.Pose())
            with pytest.raises(cr.EditorMaterialError) as caught:
                editor.compile()
            assert "unknown_a" in str(caught.value)
            assert "unknown_b" in str(caught.value)

    def test_a_module_must_carry_its_own_material(self, tmp_path):
        impostor = "rise_origin.test.material.impostor"

        @cr.module(name=impostor, version="1.0.0")
        def impostor_module(editor_access, settings):
            """A material module carrying the wrong material."""
            return cr.Draft(
                editor_access, "impostor", rscs=[essence.Material(f"{CLAY}:clay")]
            )

        with make_editor(tmp_path) as editor:
            block = make_block_draft(editor, "block")
            block.requires = [cr.MaterialRequire(impostor, "self", 0.1)]
            editor.insert(block, cr.Pose())
            with pytest.raises(cr.EditorMaterialError):
                editor.compile()

    def test_conflicting_version_texts_have_no_matching_version(self, tmp_path):
        with make_editor(tmp_path) as editor:
            first = make_block_draft(editor, "first")
            second = make_block_draft(editor, "second")
            second.requires = [cr.MaterialRequire(f"{CLAY}@^9", "clay", 0.1)]
            editor.insert(first, cr.Pose())
            editor.insert(second, cr.Pose((1, 0, 0)))
            with pytest.raises(cr.EditorMaterialError):
                editor.compile()

    def test_a_module_builds_once_for_all_its_materials(self, tmp_path):
        pack = "rise_origin.test.material.pack"

        @cr.module(name=pack, version="1.0.0")
        def pack_module(editor_access, settings):
            """A material module releasing two materials."""
            return cr.Draft(
                editor_access,
                "pack",
                rscs=[
                    essence.Material(f"{pack}:soil"),
                    essence.MaterialPhysicalPropertyFixed(
                        f"{pack}:soil#fixed", material_reference=f"{pack}:soil"
                    ),
                    essence.Material(f"{pack}:dust"),
                    essence.MaterialPhysicalPropertyFixed(
                        f"{pack}:dust#fixed", material_reference=f"{pack}:dust"
                    ),
                ],
            )

        with make_editor(tmp_path) as editor:
            block = make_block_draft(editor, "block", module=pack, material="soil")
            editor.insert(block, cr.Pose())
            update = editor.compile()
            names = [
                instruction.rsc.name
                for instruction in update.instructions
                if instruction.op == "add"
            ]
            assert f"{pack}:soil" in names
            assert f"{pack}:dust" in names

    def test_dependencies_emit_pair_modules(self, tmp_path):
        contact = "rise_origin.test.material.contact"
        pad = "rise_origin.test.block.pad"

        @cr.module(name=contact, version="1.0.0")
        def contact_module(editor_access, settings):
            """A pair module between clay and steel."""
            return cr.Draft(
                editor_access,
                "contact",
                rscs=[
                    essence.MaterialPairPropertyRestitution(
                        "clay_steel#restitution",
                        first_material_reference=f"{CLAY}:clay",
                        second_material_reference=f"{STEEL}:steel",
                        restitution=0.5,
                    )
                ],
            )

        @cr.module(name=pad, version="1.0.0", dependencies=(contact,))
        def pad_module(editor_access, settings):
            """A block standing on clay and steel."""
            return cr.Draft(
                editor_access,
                "pad",
                boxes=[cr.Box((0, 0, 0), (0.2, 0.2, 0.2), "static")],
                rscs=[make_structure("pad", f"{CLAY}:clay")],
                requires=[
                    cr.MaterialRequire(f"{CLAY}@^1", "clay", 0.1),
                    cr.MaterialRequire(f"{STEEL}@^1", "steel", 0.1),
                ],
            )

        with make_editor(tmp_path) as editor:
            editor.insert(editor.create(pad), cr.Pose())
            update = editor.compile()
            names = [
                instruction.rsc.name
                for instruction in update.instructions
                if instruction.op == "add"
            ]
            assert "clay_steel#restitution" in names
            assert names.index(f"{CLAY}:clay") < names.index("clay_steel#restitution")


class TestEditorPreview:
    def test_a_headless_preview_runs_the_session(self, tmp_path):
        with make_editor(tmp_path) as editor:
            editor.insert(editor.create(CUBE), cr.Pose())
            editor.preview(is_interactive=False, duration=0.1)


class TestEditorApplyWorldUpdate:
    def test_the_mcp_road_drives_the_built_in_tools(self, tmp_path):
        with make_editor(tmp_path) as editor:
            update = cr.WorldUpdate(
                (
                    cr.WorldUpdateInstruction(op="delete", entity_name="old/old"),
                    cr.WorldUpdateInstruction(
                        op="add", rsc=essence.Material(f"{CLAY}:clay")
                    ),
                    cr.WorldUpdateInstruction(
                        op="add", rsc=make_structure("piece", f"{CLAY}:clay")
                    ),
                    cr.WorldUpdateInstruction(
                        op="move",
                        entity_name="piece/piece",
                        pose=cr.Pose((4.0, 0.0, 0.0)),
                    ),
                )
            )
            queues = {
                "add_entity": [{"id": 7, "status": "pending"}],
                "get_result": [{"ok": True}],
                "find_entity": [
                    {
                        "type": "RSC_StructureVoxel",
                        "data": {
                            "name": "piece",
                            "origin_position": [0.0, 0.0, 0.0],
                            "orientation": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                ],
                "is_entity_frozen": [False],
            }
            with run_mcp_stub(queues) as (address, calls):
                editor.apply_world_update(update, address)
            assert [name for name, _ in calls] == [
                "remove_entity",
                "add_essence",
                "add_entity",
                "get_result",
                "find_entity",
                "is_entity_frozen",
                "remove_entity",
                "add_entity",
            ]
            assert calls[1][1]["rsct"]["type"] == "RSC_Material"
            assert calls[2][1]["rsct"]["type"] == "RSC_StructureVoxel"
            moved = calls[7][1]["rsct"]["data"]
            assert moved["origin_position"] == [4.0, 0.0, 0.0]
            assert moved["orientation"] == [0.0, 0.0, 0.0, 1.0]

    @pytest.mark.skip(
        reason="engine: synchronous handle reads (find_entity) can block "
        "forever on a running world; see rs_world_local_handle.cu findEntity "
        "waitForLive + kernel manager read race"
    )
    def test_the_handle_road_reaches_a_running_world(self, tmp_path):
        import rise_origin as rs

        with make_editor(tmp_path) as editor:
            first = make_block_draft(editor, "first")
            editor.insert(first, cr.Pose())
            editor.compile()
            saved = str(tmp_path / "world.rscb")
            editor.save(saved)
            second = make_block_draft(editor, "second")
            editor.insert(second, cr.Pose((1.0, 0.0, 0.0)))
            editor.move(first, cr.Pose((0.0, 2.0, 0.0)))
            update = editor.compile()
            world = rsc_module.load(saved)
            world.simulation.runtime.max_time = 30.0
            world.simulation.control.frequency = 20.0
            world.simulation.record.frequency = 20.0
            engine = rs.RiseOrigin()
            handle, _callback = engine.hello(world)
            try:
                assert handle.wait_for_running()
                editor.apply_world_update(update, handle)
                added = handle.find_entity("second/second").value
                assert added is not None
                moved = handle.find_entity("first/first").value
                assert float(moved.origin_position.y) == pytest.approx(2.0, abs=1e-5)
            finally:
                handle.terminate()
                handle.wait_for_end()


class TestEditorWall:
    def test_drafts_refuse_to_cross_editors(self, tmp_path):
        with (
            make_editor(tmp_path / "a") as first,
            make_editor(tmp_path / "b") as second,
        ):
            stranger = second.create(CUBE)
            with pytest.raises(cr.EditorError):
                first.insert(stranger, cr.Pose())

    def test_session_trees_refuse_draft_level_mutations(self, tmp_path):
        with make_editor(tmp_path) as editor:
            block = editor.create(CUBE)
            editor.insert(block, cr.Pose())
            with pytest.raises(cr.DraftScopeError):
                editor.root.insert(editor.create(CUBE), cr.Pose((1, 0, 0)))
            with pytest.raises(cr.DraftScopeError):
                block.insert(editor.create(CUBE), cr.Pose())

    def test_the_access_carries_builds_and_gated_reads(self, tmp_path):
        with make_editor(tmp_path) as editor:
            access = editor.root.editor_access
            held = [name for name in dir(access) if not name.startswith("_")]
            assert held == [
                "allocate_uid",
                "compute_distance",
                "create",
                "is_session_node",
                "load_rscs",
                "logger",
                "pool",
                "query_nearby",
                "query_overlaps",
                "query_ray",
            ]

    def test_session_reads_refuse_detached_nodes(self, tmp_path):
        with make_editor(tmp_path) as editor:
            detached = editor.create(CUBE)
            assert not editor.root.editor_access.is_session_node(detached)
            with pytest.raises(cr.EditorError):
                editor.root.editor_access.query_ray(detached, (0, 0, 0), (1, 0, 0))
