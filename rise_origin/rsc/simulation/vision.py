# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Vision (path-tracing) simulation sub-configuration.

Every field defaults to ``None`` and is only written when supplied, so an
omitted argument keeps the engine's built-in default.
"""

from typing import Optional, Union

from rise_origin import (
    RSC_SimulationVision,
    RSC_SimulationVisionDenoiseAtrous,
    RSC_SimulationVisionDenoiseNone,
    RSC_SimulationVisionDenoiseSVGF,
    RSC_SimulationVisionLight,
    RSC_SimulationVisionReSTIR,
    RSC_SimulationVisionTrace,
)

from rise_origin.rsc._utils import _apply

__all__ = [
    "VisionLight",
    "VisionTrace",
    "VisionReSTIR",
    "VisionDenoiseNone",
    "VisionDenoiseAtrous",
    "VisionDenoiseSVGF",
    "Vision",
]


class VisionLight(RSC_SimulationVisionLight):
    """Light selection settings for next event estimation.

    Args:
        light_bin_cell_size: Cell edge of the spatial bin a light pick draws
            from, in meters.
        light_bin_probe_num: Cells a pick draws from, one of ``{1, 7}``: the
            shading point's own cell, or that plus its six face neighbours.
    """

    def __init__(
        self,
        *,
        light_bin_cell_size: Optional[float] = None,
        light_bin_probe_num: Optional[int] = None,
    ) -> None:
        super().__init__()
        _apply(
            self,
            light_bin_cell_size=light_bin_cell_size,
            light_bin_probe_num=light_bin_probe_num,
        )


class VisionTrace(RSC_SimulationVisionTrace):
    """Path-tracer integrator settings.

    Args:
        firefly_cap: Maximum radiance per sample, clamping fireflies.
        russian_roulette_start_depth: Bounce depth at which Russian-roulette
            path termination begins.
    """

    def __init__(
        self,
        *,
        firefly_cap: Optional[float] = None,
        russian_roulette_start_depth: Optional[int] = None,
    ) -> None:
        super().__init__()
        _apply(
            self,
            firefly_cap=firefly_cap,
            russian_roulette_start_depth=russian_roulette_start_depth,
        )


class VisionReSTIR(RSC_SimulationVisionReSTIR):
    """ReSTIR reservoir-resampling settings for direct and global illumination.

    Args:
        is_enabled: Whether ReSTIR resampling is active.
        disocclusion_position_threshold: Position delta (meters) that invalidates
            a temporal reservoir.
        disocclusion_normal_threshold: Normal delta that invalidates a temporal
            reservoir.
        max_specular_chain: Maximum specular-bounce chain length tracked.
        di_m_cap: Direct-illumination reservoir confidence cap.
        di_spatial_neighbors: Direct-illumination spatial-reuse neighbor count.
        di_spatial_radius: Direct-illumination spatial-reuse radius in pixels.
        is_gi_bounce_nee_enabled: Whether global-illumination bounces use
            next-event estimation.
        gi_m_cap: Global-illumination reservoir confidence cap.
        gi_spatial_neighbors: Global-illumination spatial-reuse neighbor count.
        gi_spatial_radius: Global-illumination spatial-reuse radius in pixels.
        gi_validation_period: Steps between global-illumination reservoir
            validations.
    """

    def __init__(
        self,
        *,
        is_enabled: Optional[bool] = None,
        disocclusion_position_threshold: Optional[float] = None,
        disocclusion_normal_threshold: Optional[float] = None,
        max_specular_chain: Optional[int] = None,
        di_m_cap: Optional[float] = None,
        di_spatial_neighbors: Optional[int] = None,
        di_spatial_radius: Optional[float] = None,
        is_gi_bounce_nee_enabled: Optional[bool] = None,
        gi_m_cap: Optional[float] = None,
        gi_spatial_neighbors: Optional[int] = None,
        gi_spatial_radius: Optional[float] = None,
        gi_validation_period: Optional[int] = None,
    ) -> None:
        super().__init__()
        _apply(
            self,
            is_enabled=is_enabled,
            disocclusion_position_threshold=disocclusion_position_threshold,
            disocclusion_normal_threshold=disocclusion_normal_threshold,
            max_specular_chain=max_specular_chain,
            di_m_cap=di_m_cap,
            di_spatial_neighbors=di_spatial_neighbors,
            di_spatial_radius=di_spatial_radius,
            is_gi_bounce_nee_enabled=is_gi_bounce_nee_enabled,
            gi_m_cap=gi_m_cap,
            gi_spatial_neighbors=gi_spatial_neighbors,
            gi_spatial_radius=gi_spatial_radius,
            gi_validation_period=gi_validation_period,
        )


class VisionDenoiseNone(RSC_SimulationVisionDenoiseNone):
    """Disable denoising."""

    def __init__(self) -> None:
        super().__init__()


class VisionDenoiseAtrous(RSC_SimulationVisionDenoiseAtrous):
    """Edge-aware a-trous wavelet denoiser.

    Args:
        atrous_iterations: Number of a-trous wavelet passes.
        atrous_min_passes: Minimum passes applied regardless of convergence.
        sigma_color: Color edge-stopping width.
        sigma_normal: Normal edge-stopping width.
        sigma_depth: Depth edge-stopping width.
    """

    def __init__(
        self,
        *,
        atrous_iterations: Optional[int] = None,
        atrous_min_passes: Optional[int] = None,
        sigma_color: Optional[float] = None,
        sigma_normal: Optional[float] = None,
        sigma_depth: Optional[float] = None,
    ) -> None:
        super().__init__()
        _apply(
            self,
            atrous_iterations=atrous_iterations,
            atrous_min_passes=atrous_min_passes,
            sigma_color=sigma_color,
            sigma_normal=sigma_normal,
            sigma_depth=sigma_depth,
        )


class VisionDenoiseSVGF(RSC_SimulationVisionDenoiseSVGF):
    """Spatiotemporal variance-guided (SVGF) denoiser.

    Args:
        atrous_iterations: Number of a-trous wavelet passes.
        atrous_min_passes: Minimum passes applied regardless of convergence.
        sigma_color: Color edge-stopping width.
        sigma_normal: Normal edge-stopping width.
        sigma_depth: Depth edge-stopping width.
        temporal_alpha: Temporal blend factor for color.
        temporal_alpha_moments: Temporal blend factor for variance moments.
        temporal_min_history: Minimum accumulated frames before full temporal
            weight is used.
    """

    def __init__(
        self,
        *,
        atrous_iterations: Optional[int] = None,
        atrous_min_passes: Optional[int] = None,
        sigma_color: Optional[float] = None,
        sigma_normal: Optional[float] = None,
        sigma_depth: Optional[float] = None,
        temporal_alpha: Optional[float] = None,
        temporal_alpha_moments: Optional[float] = None,
        temporal_min_history: Optional[float] = None,
    ) -> None:
        super().__init__()
        _apply(
            self,
            atrous_iterations=atrous_iterations,
            atrous_min_passes=atrous_min_passes,
            sigma_color=sigma_color,
            sigma_normal=sigma_normal,
            sigma_depth=sigma_depth,
            temporal_alpha=temporal_alpha,
            temporal_alpha_moments=temporal_alpha_moments,
            temporal_min_history=temporal_min_history,
        )


#: A denoiser sub-config: none, a-trous, or SVGF.
DenoiseSpec = Union[VisionDenoiseNone, VisionDenoiseAtrous, VisionDenoiseSVGF]


class Vision(RSC_SimulationVision):
    """Top-level vision config bundling the tracer, ReSTIR, and denoiser.

    Args:
        light: Light selection settings.
        trace: Path-tracer integrator settings.
        restir: ReSTIR resampling settings.
        denoise: The denoiser to use (one of the ``VisionDenoise*`` configs).
    """

    def __init__(
        self,
        *,
        light: Optional[VisionLight] = None,
        trace: Optional[VisionTrace] = None,
        restir: Optional[VisionReSTIR] = None,
        denoise: Optional[DenoiseSpec] = None,
    ) -> None:
        super().__init__()
        _apply(self, light=light, trace=trace, restir=restir, denoise=denoise)
