# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Tests for rsc.entity.cosmic_background."""

from rise_origin.rsc.entity import (
    CosmicBackgroundGradient,
    CosmicBackgroundHosekWilkie,
    CosmicBackgroundUniform,
)


def test_uniform_fields():
    sky = CosmicBackgroundUniform("sky", radiance=2.0, rotation_rad=0.5)
    assert sky.radiance == 2.0
    assert sky.rotation_rad == 0.5


def test_gradient_colors_and_sun():
    sky = CosmicBackgroundGradient(
        "sky", zenith_color=(10, 20, 30), sun_direction=(0.0, 0.0, 1.0)
    )
    assert (sky.zenith_color.r, sky.zenith_color.g, sky.zenith_color.b) == (10, 20, 30)
    assert sky.sun_direction.z == 1.0


def test_hosek_wilkie_turbidity():
    sky = CosmicBackgroundHosekWilkie("sky", turbidity=2.5)
    assert sky.turbidity == 2.5
