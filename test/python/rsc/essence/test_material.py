# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.essence.material."""

import pytest

from rise_origin.rsc.essence import (
    Material,
    MaterialOpticalPropertySimpleMicrofacet,
    MaterialPairPropertyCoulombFriction,
    MaterialPhysicalPropertyRigid,
    MaterialPhysicalPropertySoft,
)


def test_material_handle():
    assert Material("rubber").name == "rubber"


def test_soft_property_fields():
    prop = MaterialPhysicalPropertySoft(
        "p", material_reference="rubber", elastic_modulus=1.0e6, density=1100.0
    )
    assert prop.material_reference == "rubber"
    assert prop.elastic_modulus == 1.0e6
    assert prop.density == 1100.0


def test_soft_fail_stress_default_is_unbreakable():
    prop = MaterialPhysicalPropertySoft("p", material_reference="m")
    assert prop.fail_stress > 0


def test_rigid_density():
    assert (
        MaterialPhysicalPropertyRigid("p", material_reference="m", density=7800).density
        == 7800
    )


def test_pair_friction_fields():
    friction = MaterialPairPropertyCoulombFriction(
        "mu",
        first_material_reference="a",
        second_material_reference="b",
        u_static=0.9,
        u_dynamic=0.8,
    )
    assert friction.u_static == pytest.approx(0.9)
    assert friction.u_dynamic == pytest.approx(0.8)
    assert friction.second_material_reference == "b"


def test_optical_microfacet_fields():
    optic = MaterialOpticalPropertySimpleMicrofacet(
        "look",
        material_reference="m",
        albedo_texture_reference="tan",
        is_transparent=True,
        ior=1.33,
    )
    assert optic.albedo_texture_reference == "tan"
    assert optic.is_transparent is True
    assert abs(optic.ior - 1.33) < 1e-6
