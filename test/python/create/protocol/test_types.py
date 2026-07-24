# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the settings vocabulary: schema, fields, and canonical json."""

import json

import pytest

from rise_origin import create as cr


class TestSettingsSchema:
    def test_defaults_fill_and_seed_auto_added(self):
        schema = cr.settings(height=cr.Float(2.0, 18.0, default=9.0))
        values = schema.validate(None)
        assert values.height == 9.0
        assert values.seed == 0

    def test_declaring_seed_is_refused(self):
        with pytest.raises(cr.SettingsError):
            cr.settings(seed=cr.Int(0, 10, default=0))

    def test_unknown_names_refused(self):
        schema = cr.settings(height=cr.Float(2.0, 18.0, default=9.0))
        with pytest.raises(cr.SettingsError):
            schema.validate({"width": 1.0})

    def test_range_check(self):
        schema = cr.settings(height=cr.Float(2.0, 18.0, default=9.0))
        with pytest.raises(cr.SettingsError):
            schema.validate({"height": 30.0})

    def test_choice_check(self):
        schema = cr.settings(season=cr.Choice("spring", "summer", default="summer"))
        assert schema.validate({"season": "spring"}).season == "spring"
        with pytest.raises(cr.SettingsError):
            schema.validate({"season": "winter"})

    def test_untyped_field_refused(self):
        with pytest.raises(cr.SettingsError):
            cr.settings(height=9.0)

    def test_to_json_describes_every_field(self):
        schema = cr.settings(height=cr.Float(2.0, 18.0, default=9.0))
        description = schema.to_json()
        assert description["height"]["type"] == "float"
        assert description["seed"]["type"] == "int"


class TestFloatList:
    def test_validates_items_and_returns_tuple(self):
        field = cr.FloatList(0.01, 0.5, default=(0.05, 0.1))
        assert field.validate("voxel_size", [0.02, 0.3]) == (0.02, 0.3)

    def test_item_range_check(self):
        field = cr.FloatList(0.01, 0.5, default=(0.05,))
        with pytest.raises(cr.SettingsError):
            field.validate("voxel_size", [0.9])

    def test_length_pin(self):
        field = cr.FloatList(0.01, 0.5, default=(0.05, 0.1), length=2)
        with pytest.raises(cr.SettingsError):
            field.validate("voxel_size", [0.05])

    def test_default_outside_range_refused(self):
        with pytest.raises(cr.SettingsError):
            cr.FloatList(0.01, 0.5, default=(0.9,))

    def test_scalar_refused(self):
        field = cr.FloatList(0.01, 0.5, default=(0.05,))
        with pytest.raises(cr.SettingsError):
            field.validate("voxel_size", 0.05)


class TestVoxelSize:
    def test_a_choice_answers_meters(self):
        field = cr.VoxelSize(default="2cm", min="5mm", max="10cm")
        assert field.validate("trunk_voxel_size", "5cm") == 0.05
        assert field.default == 0.02
        assert field.options == ("5mm", "1cm", "2cm", "5cm", "10cm")

    def test_off_ladder_and_floats_are_refused(self):
        field = cr.VoxelSize(default="2cm")
        with pytest.raises(cr.SettingsError):
            field.validate("voxel_size", "3cm")
        with pytest.raises(cr.SettingsError):
            field.validate("voxel_size", 0.02)

    def test_outside_the_field_range_is_refused(self):
        field = cr.VoxelSize(default="2cm", min="1cm", max="5cm")
        with pytest.raises(cr.SettingsError):
            field.validate("voxel_size", "1mm")
        with pytest.raises(cr.SettingsError):
            field.validate("voxel_size", "1m")

    def test_bad_declarations_are_refused(self):
        with pytest.raises(cr.SettingsError):
            cr.VoxelSize(default="3cm")
        with pytest.raises(cr.SettingsError):
            cr.VoxelSize(default="1cm", min="5cm", max="2mm")
        with pytest.raises(cr.SettingsError):
            cr.VoxelSize(default="1mm", min="1cm", max="10cm")

    def test_schema_fills_the_default_in_meters(self):
        schema = cr.settings(trunk_voxel_size=cr.VoxelSize(default="2cm"))
        values = schema.validate({"trunk_voxel_size": "5mm"})
        assert values.trunk_voxel_size == 0.005
        assert schema.validate(None).trunk_voxel_size == 0.02

    def test_json_description_lists_the_options(self):
        field = cr.VoxelSize(default="10cm", min="2cm", max="20cm")
        assert field.to_json() == {
            "type": "voxel_size",
            "options": ["2cm", "5cm", "10cm", "20cm"],
            "default": "10cm",
        }

    def test_coarse_sky_scale_steps_resolve(self):
        field = cr.VoxelSize(default="2m", min="1m", max="10m")
        assert field.options == ("1m", "2m", "5m", "10m")
        assert field.default == 2.0
        assert field.validate("cloud_voxel_size", "5m") == 5.0
        assert field.validate("cloud_voxel_size", "10m") == 10.0


class TestSettings:
    def test_canonical_json_is_sorted_and_stable(self):
        schema = cr.settings(
            b=cr.Float(0.0, 1.0, default=0.5),
            a=cr.Int(0, 10, default=3),
        )
        text = schema.validate({"a": 4}).to_json()
        assert text == json.dumps({"a": 4, "b": 0.5, "seed": 0}, sort_keys=True)

    def test_float_list_serializes_as_list(self):
        schema = cr.settings(voxel_size=cr.FloatList(0.01, 0.5, default=(0.05, 0.1)))
        record = json.loads(schema.validate(None).to_json())
        assert record["voxel_size"] == [0.05, 0.1]

    def test_attribute_and_mapping_access(self):
        values = cr.Settings({"height": 9.0})
        assert values.height == 9.0
        assert values["height"] == 9.0
        with pytest.raises(AttributeError):
            _ = values.width
