# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The cloud material package: an opaque white and rainy shell materials.

The world ray-traces the light, so the albedos carry no shading. The sky
takes ``cloud``, one opaque near-white for fair-weather puffs, and a set
of rainy shell materials the voxelizer stacks from the skin to the core
of a rain cloud. The shells are separate materials on purpose: the
renderer culls a face between two voxels of the same material, so an
optical reference alone never draws an interior layer (see
tmp-docs/todo.md). Giving each shell its own material makes the shell
boundary a real rendered face, so the outer translucent shells pass light
and the darker opaque core shows through as depth.
"""

from rise_origin import create as cr
from rise_origin.rsc import essence, resource

MODULE = "rise_origin.material.nature.cloud"

CLOUD = f"{MODULE}:cloud"

# The rainy glass refraction index, near air so a ray passes with a soft
# frosted bend.
CLOUD_RAIN_IOR = 1.05

# The rainy shell materials, ordered skin to core, one material per shell
# so the boundary between shells renders. Fields are (name, albedo RGB
# 0-1, roughness, is_transparent, ior). The albedo darkens and the
# roughness climbs inward by a widening step; the two outer shells pass
# light and the core is opaque, so the heavy core reads through the
# translucent edge.
CLOUD_RAIN_LAYERS = (
    ("cloud_rain_skin", (0.78, 0.80, 0.86), 0.6, True, CLOUD_RAIN_IOR),
    ("cloud_rain_middle", (0.68, 0.70, 0.76), 0.8, True, CLOUD_RAIN_IOR),
    ("cloud_rain_core", (0.48, 0.50, 0.56), 1.0, False, 1.0),
)

# The material names and world references, and the per-material optical
# reference lists structures pass; the sweep indexes both by shell.
CLOUD_RAIN_LAYER_NAMES = tuple(name for name, *_ in CLOUD_RAIN_LAYERS)
CLOUD_RAIN_LAYER_REFERENCES = tuple(
    f"{MODULE}:{name}" for name, *_ in CLOUD_RAIN_LAYERS
)
CLOUD_RAIN_LAYER_OPTICAL_REFERENCES = tuple(
    (f"{MODULE}:{name}#optic",) for name, *_ in CLOUD_RAIN_LAYERS
)
CLOUD_OPTICAL_REFERENCES = (f"{CLOUD}#optic",)

# How many shells the voxelizer stacks; white is one flat material, rainy
# three.
CLOUD_LAYER_COUNT = 1
CLOUD_RAIN_LAYER_COUNT = len(CLOUD_RAIN_LAYERS)


@cr.module(
    name=MODULE,
    version="1.0.0",
    tags=("material", "cloud", "sky"),
)
def cloud(editor_access, settings):
    """The opaque white and translucent rainy shell cloud materials.

    Releases ``cloud`` (opaque near-white) and the rainy shell materials
    ``cloud_rain_skin``, ``cloud_rain_middle`` and ``cloud_rain_core``,
    all fixed so they hang in place. A rainy cloud stacks the shells from
    the skin to the core.

    Args:
        editor_access: The build wall handed to the module body.
        settings: The validated material settings.
    """
    rscs = _CloudMaterials.make_material_rscs(CLOUD, (1.0, 1.0, 1.0), 0.0, False, 1.0)
    for name, albedo, roughness, is_transparent, ior in CLOUD_RAIN_LAYERS:
        rscs += _CloudMaterials.make_material_rscs(
            f"{MODULE}:{name}", albedo, roughness, is_transparent, ior
        )
    return cr.Draft(editor_access, "cloud", rscs=rscs)


class _CloudMaterials:
    """The shared emission of one cloud material's resources."""

    @staticmethod
    def make_material_rscs(name, albedo, roughness, is_transparent, ior):
        """Return the resources and essences of one cloud material.

        Args:
            name: The material reference name.
            albedo: The material albedo as RGB in 0-1.
            roughness: The surface roughness in the texture R channel.
            is_transparent: Whether the surface refracts through the
                material.
            ior: The refraction index used when transparent.
        """
        return [
            resource.TextureConstant(f"{name}#rm", value=(roughness, 0.0, 0.0)),
            resource.TextureConstant(f"{name}#albedo", value=albedo),
            essence.Material(name),
            essence.MaterialPhysicalPropertyFixed(
                f"{name}#fixed", material_reference=name
            ),
            essence.MaterialOpticalPropertySimpleMicrofacet(
                f"{name}#optic",
                material_reference=name,
                albedo_texture_reference=f"{name}#albedo",
                roughness_metalness_texture_reference=f"{name}#rm",
                is_transparent=is_transparent,
                ior=ior,
            ),
        ]
