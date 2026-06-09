# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""One-line builders for :class:`RS_MaterialConfig`.

A material bundles mechanical properties (density, stiffness, friction,
actuation limits) with a render colour. Structures reference materials by
name, so every material needs a unique ``name``.

Instead of the verbose pattern::

    mat = RS_MaterialConfig()
    mat.name = "soft"
    mat.mechanical_config.is_rigid = False
    mat.mechanical_config.density = 1000.0
    mat.mechanical_config.elastic_modulus = 5e5
    ...
    render = RS_OpaqueMaterialRenderConfig()
    render.color = RColor3b(180, 80, 80)
    mat.render_config.type = RSE_MaterialRenderType.RSE_MATERIAL_RENDER_OPAQUE
    mat.render_config.config = render

use a single call::

    from rise_origin.create import MaterialCreator
    soft = MaterialCreator.soft(density=1000.0, elastic_modulus=5e5)
    rigid = MaterialCreator.rigid(render=(80, 80, 180))
    bicep = MaterialCreator.muscle(min_expansion=-0.3, max_expansion=0.3)

All presets return a ready-to-use :class:`RS_MaterialConfig`; append it to
``config.material_configs``.
"""

from typing import Optional

from rise_origin import (
    RColor3b,
    RS_OpaqueMaterialRenderConfig,
    RS_MaterialConfig,
    RSE_MaterialRenderType,
)
from rise_origin.create.utils import Color

__all__ = ["MaterialCreator", "RenderSpec"]

#: How a material surface is rendered. For now only an RGB ``Color`` (a flat
#: constant-color surface) is supported.
RenderSpec = Color


def _apply_render(mat: RS_MaterialConfig, render: RenderSpec) -> None:
    """Populate ``mat.render_config`` from a :data:`RenderSpec` (RGB color)."""
    cfg = RS_OpaqueMaterialRenderConfig()
    cfg.color = RColor3b(*render)
    mat.render_config.type = RSE_MaterialRenderType.RSE_MATERIAL_RENDER_OPAQUE
    mat.render_config.config = cfg


class MaterialCreator:
    """Factory of :class:`RS_MaterialConfig` presets.

    Every method returns a fully-populated config. Defaults sit inside the
    ranges the engine enforces (``density > 0``, ``poissons_ratio`` in
    ``[0, 0.5]``, friction coefficients in ``[0, 1]``,
    ``min_expansion >= -1`` and ``max_expansion >= min_expansion``), so a
    preset called with no arguments is always valid.
    """

    @staticmethod
    def base(
        name: str,
        *,
        is_rigid: bool = False,
        is_fixed: bool = False,
        density: float = 1000.0,
        elastic_modulus: float = 5e5,
        poissons_ratio: float = 0.35,
        u_static: float = 1.0,
        u_dynamic: float = 0.8,
        min_expansion: Optional[float] = None,
        max_expansion: Optional[float] = None,
        fail_stress: Optional[float] = None,
        render: RenderSpec = (180, 180, 180),
    ) -> RS_MaterialConfig:
        """Build a material from explicit mechanical properties.

        Args:
            name: Unique material name; structures reference it by this name.
            is_rigid: If ``True``, voxels of this material form rigid bodies
                that joints can constrain. If ``False``, they deform elastically.
            is_fixed: If ``True``, voxels of this material are pinned in space.
            density: Mass density (kg/m^3). Must be > 0.
            elastic_modulus: Young's modulus (Pa). Must be > 0. Soft tissue is
                ~1e4-1e6; stiff/rigid materials ~1e8-1e9 (simulation is slow!).
            poissons_ratio: Poisson's ratio (dimensionless), in ``[0, 0.5]``.
            u_static: Static friction coefficient, in ``[0, 1]``.
            u_dynamic: Dynamic friction coefficient, in ``[0, 1]``.
            min_expansion: Minimum expansion ratio (>= -1; -1 means -100%
                contraction). ``None`` leaves the engine default.
            max_expansion: Maximum expansion ratio (>= ``min_expansion``).
                ``None`` leaves the engine default.
            fail_stress: Failure stress (N/m^2). ``None`` leaves the engine
                default (effectively unbreakable).
            render: Surface render spec; an RGB color (0-255 per channel) for
                the constant-color renderer. See :data:`RenderSpec`.

        Returns:
            A populated :class:`RS_MaterialConfig`.
        """
        mat = RS_MaterialConfig()
        mat.name = name
        mat.mechanical_config.is_rigid = is_rigid
        mat.mechanical_config.is_fixed = is_fixed
        mat.mechanical_config.density = density
        mat.mechanical_config.elastic_modulus = elastic_modulus
        mat.mechanical_config.poissons_ratio = poissons_ratio
        mat.mechanical_config.u_static = u_static
        mat.mechanical_config.u_dynamic = u_dynamic
        if min_expansion is not None:
            mat.mechanical_config.min_expansion = min_expansion
        if max_expansion is not None:
            mat.mechanical_config.max_expansion = max_expansion
        if fail_stress is not None:
            mat.mechanical_config.fail_stress = fail_stress

        _apply_render(mat, render)
        return mat

    @classmethod
    def soft(
        cls,
        name: str = "soft",
        *,
        elastic_modulus: float = 5e5,
        render: RenderSpec = (180, 80, 80),
        **kwargs,
    ) -> RS_MaterialConfig:
        """Deformable (non-rigid) material. See :meth:`base` for ``kwargs``."""
        return cls.base(
            name,
            is_rigid=False,
            elastic_modulus=elastic_modulus,
            render=render,
            **kwargs,
        )

    @classmethod
    def rigid(
        cls,
        name: str = "rigid",
        *,
        elastic_modulus: float = 5e8,
        render: RenderSpec = (80, 80, 180),
        **kwargs,
    ) -> RS_MaterialConfig:
        """Rigid material; its voxels form joint-constrainable rigid bodies."""
        return cls.base(
            name,
            is_rigid=True,
            elastic_modulus=elastic_modulus,
            render=render,
            **kwargs,
        )

    @classmethod
    def fixed(
        cls,
        name: str = "fixed",
        *,
        render: RenderSpec = (120, 120, 120),
        **kwargs,
    ) -> RS_MaterialConfig:
        """Static material; its voxels are pinned in space (walls, anchors)."""
        return cls.base(name, is_fixed=True, render=render, **kwargs)

    @classmethod
    def muscle(
        cls,
        name: str = "muscle",
        *,
        min_expansion: float = -0.3,
        max_expansion: float = 0.3,
        elastic_modulus: float = 5e5,
        render: RenderSpec = (200, 60, 60),
        **kwargs,
    ) -> RS_MaterialConfig:
        """Soft actuator material that contracts/expands via expansion signals.

        ``min_expansion``/``max_expansion`` bound the actuation range (ratios;
        -0.3 means up to 30% contraction, 0.3 up to 30% expansion).
        """
        return cls.base(
            name,
            is_rigid=False,
            elastic_modulus=elastic_modulus,
            min_expansion=min_expansion,
            max_expansion=max_expansion,
            render=render,
            **kwargs,
        )
