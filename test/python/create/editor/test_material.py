# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the material tables: declared requires and gathered data."""

import pytest

from rise_origin.create.data import MaterialRequire
from rise_origin.create.editor import (
    EditorMaterialDataTable,
    EditorMaterialError,
    EditorMaterialRequireTable,
)
from rise_origin.rsc import essence

CLAY = "rise_origin.material.mineral.clay"
SAND = "rise_origin.material.mineral.sand"
PLANT = "rise_origin.material.nature.plant"
MINERAL = "rise_origin.material.nature.mineral"


class FakeResource:
    def __init__(self, name):
        self.name = name

    def save(self, path):
        with open(path, "wb") as handle:
            handle.write(self.name.encode("utf-8"))


class TestEditorMaterialRequireTable:
    def test_requires_keep_declaration_order(self):
        table = EditorMaterialRequireTable([MaterialRequire(PLANT, "leaf", 0.1)])
        table.add_requires([MaterialRequire(MINERAL, "iron", 0.2)])
        names = [
            require.get_world_material_reference() for require in table.get_requires()
        ]
        assert names == [f"{PLANT}:leaf", f"{MINERAL}:iron"]

    def test_one_material_merges_into_one_entry(self):
        table = EditorMaterialRequireTable([MaterialRequire(PLANT, "leaf", 0.1)])
        table.add_requires([MaterialRequire(PLANT, "leaf", (0.1, 0.05))])
        held = table.get_requires()
        assert len(held) == 1
        assert held[0].voxel_size == (0.1, 0.05)

    def test_materials_of_one_module_stay_separate_entries(self):
        table = EditorMaterialRequireTable(
            [
                MaterialRequire(PLANT, "leaf", 0.1),
                MaterialRequire(PLANT, "bark", 0.05),
            ]
        )
        assert len(table.get_requires()) == 2

    def test_different_version_parts_stay_separate_entries(self):
        table = EditorMaterialRequireTable(
            [MaterialRequire(f"{PLANT}@1.0.0", "leaf", 0.1)]
        )
        table.add_requires([MaterialRequire(f"{PLANT}@0.2.0", "leaf", 0.05)])
        assert len(table.get_requires()) == 2
        held = table.get_required_material_name_and_voxel_sizes()
        assert held == {f"{PLANT}:leaf": {0.1, 0.05}}

    def test_add_requires_refuses_when_frozen(self):
        table = EditorMaterialRequireTable()
        table.add_requires([MaterialRequire(PLANT, "leaf", 0.1)])
        table.freeze()
        with pytest.raises(EditorMaterialError):
            table.add_requires([MaterialRequire(MINERAL, "iron", 0.1)])

    def test_has_require_checks_module_material_then_sizes(self):
        table = EditorMaterialRequireTable(
            [MaterialRequire(f"{PLANT}@^1", "leaf", (0.1, 0.05))]
        )
        assert table.has_require(MaterialRequire(PLANT, "leaf", 0.1))
        assert table.has_require(MaterialRequire(PLANT, "leaf", (0.05, 0.1)))
        assert not table.has_require(MaterialRequire(PLANT, "leaf", 0.2))
        assert not table.has_require(MaterialRequire(PLANT, "bark", 0.1))
        assert not table.has_require(MaterialRequire(MINERAL, "iron", 0.1))

    def test_has_require_rounds_across_float32(self):
        table = EditorMaterialRequireTable([MaterialRequire(PLANT, "leaf", 0.1)])
        assert table.has_require(MaterialRequire(PLANT, "leaf", 0.1000000001))


class TestEditorMaterialDataTable:
    def test_identical_content_merges_and_conflicts_refuse(self):
        table = EditorMaterialDataTable()
        table.add_essence(essence.Material(CLAY))
        table.add_essence(essence.Material(CLAY))
        assert [rsc.name for rsc in table.get_essences()] == [CLAY]
        with pytest.raises(EditorMaterialError):
            table.add_essence(
                essence.MaterialPhysicalPropertyFixed(CLAY, material_reference=CLAY)
            )

    def test_has_material_wants_a_material_essence(self):
        table = EditorMaterialDataTable()
        table.add_essence(
            essence.MaterialPhysicalPropertyFixed("fixedness", material_reference=CLAY)
        )
        assert not table.has_material("fixedness")
        table.add_essence(essence.Material(CLAY))
        assert table.has_material(CLAY)

    def test_physical_kinds_record_per_material(self):
        table = EditorMaterialDataTable()
        table.add_essence(essence.Material(CLAY))
        table.add_essence(
            essence.MaterialPhysicalPropertyFixed("fixedness", material_reference=CLAY)
        )
        table.add_essence(
            essence.MaterialPhysicalPropertySoft(
                "softness",
                material_reference=CLAY,
                elastic_modulus=1e5,
                density=1e3,
            )
        )
        assert table.get_material_physical_kinds()[CLAY] == {"fixed", "soft"}

    def test_essences_order_materials_properties_pairs(self):
        table = EditorMaterialDataTable()
        table.add_essence(
            essence.MaterialPairPropertyRestitution(
                "pair",
                first_material_reference=CLAY,
                second_material_reference=SAND,
                restitution=0.5,
            )
        )
        table.add_essence(
            essence.MaterialPhysicalPropertyFixed("fixedness", material_reference=CLAY)
        )
        table.add_essence(essence.Material(CLAY))
        table.add_essence(essence.Material(SAND))
        assert [rsc.name for rsc in table.get_essences()] == [
            CLAY,
            SAND,
            "fixedness",
            "pair",
        ]

    def test_pair_properties_drop_without_both_materials(self):
        table = EditorMaterialDataTable()
        table.add_essence(essence.Material(CLAY))
        table.add_essence(
            essence.MaterialPairPropertyRestitution(
                "pair",
                first_material_reference=CLAY,
                second_material_reference=SAND,
                restitution=0.5,
            )
        )
        assert [rsc.name for rsc in table.get_essences()] == [CLAY]

    def test_resources_keep_registration_order(self):
        table = EditorMaterialDataTable()
        table.add_resource(FakeResource("bark_albedo"))
        table.add_resource(FakeResource("leaf_albedo"))
        assert [rsc.name for rsc in table.get_resources()] == [
            "bark_albedo",
            "leaf_albedo",
        ]
