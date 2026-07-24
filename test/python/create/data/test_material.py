# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the material require: parsing, coverage, and merging."""

import pytest

from rise_origin import create as cr

PLANT = "rise_origin.material.nature.plant"


class TestMaterialRequire:
    def test_texts_parse_and_sizes_store_as_tuples(self):
        require = cr.MaterialRequire(f"{PLANT}@^1", "leaf", 0.1)
        assert require.module_reference.get_name() == PLANT
        assert require.module_reference.version_text == "^1"
        assert require.material_reference == "leaf"
        assert require.voxel_size == (0.1,)
        listed = cr.MaterialRequire(PLANT, "bark", (0.05, 0.1))
        assert listed.voxel_size == (0.05, 0.1)

    def test_world_material_references_join_with_a_colon(self):
        require = cr.MaterialRequire(f"{PLANT}@^1", "leaf", 0.1)
        assert require.get_world_material_reference() == f"{PLANT}:leaf"

    def test_requires_must_name_a_material_and_a_size(self):
        with pytest.raises(ValueError):
            cr.MaterialRequire(PLANT, "", 0.1)
        with pytest.raises(ValueError):
            cr.MaterialRequire(PLANT, "leaf", ())

    def test_version_matching_asks_the_specifier(self):
        require = cr.MaterialRequire(f"{PLANT}@^1", "leaf", 0.1)
        assert require.is_version_matching(cr.ModuleReference.parse_version("1.2.0"))
        assert not require.is_version_matching(
            cr.ModuleReference.parse_version("2.0.0")
        )

    def test_superset_ignores_the_version_part_and_checks_sizes(self):
        wide = cr.MaterialRequire(f"{PLANT}@^1", "leaf", (0.05, 0.1))
        assert wide.is_superset_of(cr.MaterialRequire(PLANT, "leaf", 0.1))
        assert not wide.is_superset_of(cr.MaterialRequire(PLANT, "leaf", 0.2))
        assert not wide.is_superset_of(cr.MaterialRequire(PLANT, "bark", 0.1))

    def test_merge_unions_the_sizes_of_one_material(self):
        first = cr.MaterialRequire(f"{PLANT}@^1", "leaf", (0.05, 0.1))
        second = cr.MaterialRequire(f"{PLANT}@^1", "leaf", (0.1000000001, 0.2))
        merged = first.merge(second)
        assert merged.module_reference == first.module_reference
        assert merged.voxel_size == (0.05, 0.1, 0.2)

    def test_merge_refuses_different_modules_or_materials(self):
        base = cr.MaterialRequire(f"{PLANT}@1.0.0", "leaf", 0.1)
        with pytest.raises(ValueError):
            base.merge(cr.MaterialRequire(f"{PLANT}@0.2.0", "leaf", 0.1))
        with pytest.raises(ValueError):
            base.merge(cr.MaterialRequire(f"{PLANT}@1.0.0", "bark", 0.1))

    def test_round_voxel_size_absorbs_float32(self):
        assert cr.MaterialRequire.round_voxel_size(0.100000001) == 0.1

    def test_json_round_trip(self):
        require = cr.MaterialRequire(f"{PLANT}@^1", "bark", (0.05, 0.1))
        assert cr.MaterialRequire.from_json(require.to_json()) == require
