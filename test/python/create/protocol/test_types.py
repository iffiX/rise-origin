# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests of the settings vocabulary: schema, fields, and canonical json."""

import json

import numpy as np
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

    def test_doc_rides_the_field_description(self):
        schema = cr.settings(
            height=cr.Float(2.0, 18.0, default=9.0, doc="The block height.")
        )
        assert schema.to_json()["height"]["doc"] == "The block height."

    def test_every_field_round_trips_through_identity_json(self):
        schema = cr.settings(
            width=cr.Float(0.1, 5.0, default=1.0),
            count=cr.Int(0, 10, default=3),
            is_fixed=cr.Bool(default=True),
            label=cr.Str(default="plain"),
            season=cr.Choice("spring", "summer", default="summer"),
            aim=cr.Vector(-1.0, 1.0, default=(1.0, 0.0, 0.0)),
            span=cr.Bounds(-5.0, 5.0, default=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))),
            weights=cr.Array1D(min=0.0, max=1.0, default=[0.2, 0.8]),
            heights=cr.Array2D(default=[[0.0, 0.5], [0.25, 1.0]]),
            occupancy=cr.Array3D(default=np.ones((2, 2, 2), dtype=bool)),
            corner=cr.StaticArray1D(shape=(3,), default=[0.1, 0.2, 0.3]),
            voxel_size=cr.VoxelSize(default="2cm"),
        )
        first = schema.validate(None)
        text = first.to_json()
        second = schema.validate(json.loads(text))
        assert second.to_json() == text


class TestArrays:
    def test_bool_cells_accept_arrays_and_lists(self):
        field = cr.Array3D()
        mask = np.zeros((3, 2, 2), dtype=bool)
        mask[0] = True
        stored = field.validate("occupancy", mask)
        assert stored["cell_num"] == [3, 2, 2]
        assert stored["dtype"] == "bool"
        assert np.array_equal(field.make_value(stored), mask)
        listed = field.validate("occupancy", mask.astype(int).tolist())
        assert listed == stored

    def test_bool_cells_refuse_values_beyond_zero_and_one(self):
        field = cr.Array3D()
        with pytest.raises(cr.SettingsError):
            field.validate("occupancy", np.full((2, 2, 2), 2))

    def test_numeric_dtypes_round_trip_exactly(self):
        for dtype, cells in (
            ("uint8", np.arange(12, dtype=np.uint8).reshape(3, 4)),
            ("int32", np.arange(-6, 6, dtype=np.int32).reshape(3, 4)),
            ("float32", np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)),
        ):
            field = cr.Array2D(dtype=dtype)
            stored = field.validate("cells", cells)
            again = field.validate("cells", json.loads(json.dumps(stored)))
            assert again == stored
            assert np.array_equal(field.make_value(stored), cells)

    def test_cell_range_check(self):
        field = cr.Array2D(min=0.0, max=1.0)
        with pytest.raises(cr.SettingsError):
            field.validate("heights", np.full((2, 2), 3.0))

    def test_non_finite_cells_refused(self):
        field = cr.Array2D()
        with pytest.raises(cr.SettingsError):
            field.validate("heights", np.full((2, 2), np.inf))

    def test_cell_cap_check(self):
        field = cr.Array3D(cell_cap=8)
        with pytest.raises(cr.SettingsError):
            field.validate("occupancy", np.ones((3, 3, 3), dtype=bool))

    def test_wrong_rank_refused(self):
        field = cr.Array3D()
        with pytest.raises(cr.SettingsError):
            field.validate("occupancy", np.ones((2, 2), dtype=bool))

    def test_none_needs_a_none_default(self):
        optional = cr.Array3D(default=None)
        assert optional.validate("occupancy", None) is None
        required = cr.Array3D(default=np.ones((2, 2, 2), dtype=bool))
        with pytest.raises(cr.SettingsError):
            required.validate("occupancy", None)

    def test_bool_cells_take_no_range(self):
        with pytest.raises(cr.SettingsError):
            cr.Array3D(min=0.0)

    def test_tampered_json_form_refused(self):
        field = cr.Array2D()
        stored = field.validate("heights", np.zeros((2, 2), dtype=np.float32))
        wrong_dtype = dict(stored, dtype="int32")
        with pytest.raises(cr.SettingsError):
            field.validate("heights", wrong_dtype)
        short_data = dict(stored, cell_num=[2, 3])
        with pytest.raises(cr.SettingsError):
            field.validate("heights", short_data)
        stray_text = dict(stored, data="not base64!")
        with pytest.raises(cr.SettingsError):
            field.validate("heights", stray_text)


class TestStaticArrays:
    def test_the_declared_shape_is_required(self):
        field = cr.StaticArray1D(shape=(3,), dtype="float32")
        stored = field.validate("corner", [0.1, 0.2, 0.3])
        assert stored["cell_num"] == [3]
        with pytest.raises(cr.SettingsError):
            field.validate("corner", [0.1, 0.2])

    def test_shape_must_match_the_rank(self):
        with pytest.raises(cr.SettingsError):
            cr.StaticArray2D(shape=(3,))

    def test_grid_shapes_pin_both_sides(self):
        field = cr.StaticArray2D(shape=(2, 3), dtype="int32")
        stored = field.validate("cells", np.zeros((2, 3), dtype=np.int32))
        assert stored["cell_num"] == [2, 3]
        with pytest.raises(cr.SettingsError):
            field.validate("cells", np.zeros((3, 2), dtype=np.int32))


class TestVoxelSize:
    def test_a_choice_keeps_the_unit_text_and_decodes_meters(self):
        field = cr.VoxelSize(default="2cm", min="5mm", max="10cm")
        assert field.validate("trunk_voxel_size", "5cm") == "5cm"
        assert field.make_value("5cm") == 0.05
        assert field.default == "2cm"
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

    def test_schema_hands_bodies_the_size_in_meters(self):
        schema = cr.settings(trunk_voxel_size=cr.VoxelSize(default="2cm"))
        values = schema.validate({"trunk_voxel_size": "5mm"})
        assert values.trunk_voxel_size == 0.005
        assert schema.validate(None).trunk_voxel_size == 0.02
        assert json.loads(values.to_json())["trunk_voxel_size"] == "5mm"

    def test_json_description_lists_the_options(self):
        field = cr.VoxelSize(default="10cm", min="2cm", max="20cm")
        assert field.to_json() == {
            "type": "voxel_size",
            "options": ["2cm", "5cm", "10cm", "20cm"],
            "default": "10cm",
            "doc": "",
        }

    def test_coarse_sky_scale_steps_resolve(self):
        field = cr.VoxelSize(default="2m", min="1m", max="10m")
        assert field.options == ("1m", "2m", "5m", "10m")
        assert field.default == "2m"
        assert field.make_value(field.validate("cloud_voxel_size", "5m")) == 5.0


class TestSettings:
    def test_canonical_json_is_sorted_and_stable(self):
        schema = cr.settings(
            b=cr.Float(0.0, 1.0, default=0.5),
            a=cr.Int(0, 10, default=3),
        )
        text = schema.validate({"a": 4}).to_json()
        assert text == json.dumps({"a": 4, "b": 0.5, "seed": 0}, sort_keys=True)

    def test_arrays_serialize_as_their_canonical_dict(self):
        schema = cr.settings(occupancy=cr.Array3D(default=None))
        mask = np.ones((2, 2, 2), dtype=bool)
        record = json.loads(schema.validate({"occupancy": mask}).to_json())
        assert record["occupancy"]["cell_num"] == [2, 2, 2]
        assert record["occupancy"]["dtype"] == "bool"

    def test_decode_hooks_hand_bodies_rich_values(self):
        schema = cr.settings(
            voxel_size=cr.VoxelSize(default="2cm"),
            occupancy=cr.Array3D(default=None),
        )
        mask = np.zeros((4, 3, 2), dtype=bool)
        mask[:2] = True
        values = schema.validate({"occupancy": mask})
        assert values.voxel_size == 0.02
        assert values["voxel_size"] == 0.02
        assert values.occupancy.dtype == np.bool_
        assert np.array_equal(values.occupancy, mask)
        assert schema.validate(None).occupancy is None

    def test_attribute_and_mapping_access(self):
        values = cr.Settings({"height": 9.0})
        assert values.height == 9.0
        assert values["height"] == 9.0
        with pytest.raises(AttributeError):
            _ = values.width
