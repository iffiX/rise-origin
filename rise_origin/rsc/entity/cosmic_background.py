# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Cosmic-background entities: the environment light seen along escaped rays.

A world has at most one cosmic background. Three models are available: a uniform
color (optionally from an environment texture), an analytic zenith-to-horizon
gradient with a sun disk, and the physically based Hosek-Wilkie daylight sky.
"""

from rise_origin import (
    RSC_CosmicBackgroundGradient,
    RSC_CosmicBackgroundHosekWilkie,
    RSC_CosmicBackgroundUniform,
)

from rise_origin import RColor3b, RVec3rf

__all__ = [
    "CosmicBackgroundUniform",
    "CosmicBackgroundGradient",
    "CosmicBackgroundHosekWilkie",
]


class CosmicBackgroundUniform(RSC_CosmicBackgroundUniform):
    """A constant environment color, optionally sampled from a texture.

    Args:
        name: Unique entity name.
        texture_reference: Environment texture resource name; empty uses a flat
            color of unit value scaled by ``radiance``.
        radiance: Radiance the background emits, scaling the sampled color, in
            W/(m^2*sr).
        rotation_rad: Rotation of the environment about the up axis, in radians.
    """

    def __init__(
        self,
        name: str = "cosmic_background",
        *,
        texture_reference: str = "",
        radiance: float = 1.0,
        rotation_rad: float = 0.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.texture_reference = texture_reference
        self.radiance = radiance
        self.rotation_rad = rotation_rad


class CosmicBackgroundGradient(RSC_CosmicBackgroundGradient):
    """An analytic sky: a zenith-to-horizon color gradient with a sun disk.

    The sky color blends ``zenith_color`` and ``horizon_color`` by the ray's
    vertical angle; a disk of angular radius ``sun_angular_radius`` is painted
    along ``sun_direction``.

    Args:
        name: Unique entity name.
        zenith_color: Sky color at the zenith (byte RGB).
        horizon_color: Sky color at the horizon (byte RGB).
        sun_direction: Direction to the sun ``(x, y, z)`` (need not be unit).
        sun_color: Sun chromaticity (byte RGB).
        sun_irradiance: Irradiance the sun delivers to a surface facing it, in
            W/m^2.
        sun_angular_radius: Angular radius of the sun disk, in radians. Larger
            is a softer shadow.
    """

    def __init__(
        self,
        name: str = "cosmic_background",
        *,
        zenith_color: RColor3b = RColor3b.zero(),
        horizon_color: RColor3b = RColor3b.zero(),
        sun_direction: RVec3rf = RVec3rf(0.3, 0.4, -1.0),
        sun_color: RColor3b = RColor3b.zero(),
        sun_irradiance: float = 1.0,
        sun_angular_radius: float = 0.00465,
    ) -> None:
        super().__init__()
        self.name = name
        self.zenith_color = zenith_color
        self.horizon_color = horizon_color
        self.sun_direction = sun_direction
        self.sun_color = sun_color
        self.sun_irradiance = sun_irradiance
        self.sun_angular_radius = sun_angular_radius


class CosmicBackgroundHosekWilkie(RSC_CosmicBackgroundHosekWilkie):
    """The physically based Hosek-Wilkie analytic daylight sky.

    Args:
        name: Unique entity name.
        sun_direction: Direction to the sun ``(x, y, z)`` (need not be unit).
        sun_color: Sun chromaticity (byte RGB).
        sun_irradiance: Irradiance the sun delivers to a surface facing it, in
            W/m^2.
        sun_angular_radius: Angular radius of the sun disk, in radians. Larger
            is a softer shadow.
        turbidity: Atmospheric turbidity in [1, 10]; 1 is clear, 10 is hazy.
        ground_color: Ground albedo color used by the sky model (byte RGB).
        radiance_scale: Dimensionless scale on the radiance the sky model
            returns.
    """

    def __init__(
        self,
        name: str = "cosmic_background",
        *,
        sun_direction: RVec3rf = RVec3rf(0.3, 0.4, 0.866),
        sun_color: RColor3b = RColor3b.zero(),
        sun_irradiance: float = 1.0,
        sun_angular_radius: float = 0.00465,
        turbidity: float = 3.0,
        ground_color: RColor3b = RColor3b.zero(),
        radiance_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.sun_direction = sun_direction
        self.sun_color = sun_color
        self.sun_irradiance = sun_irradiance
        self.sun_angular_radius = sun_angular_radius
        self.turbidity = turbidity
        self.ground_color = ground_color
        self.radiance_scale = radiance_scale
