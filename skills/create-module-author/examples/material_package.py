"""A complete material package: one module releasing two materials.

Material modules carry only essences. Structures in other modules point at
these materials by the colon-joined world reference, and declare them with
MaterialRequire. The pair module at the end shows how contact behavior
between two materials ships separately and enters worlds through
dependencies.
"""

from rise_origin import create as cr
from rise_origin.rsc import essence

OAK = "acme.material.wood.oak"


@cr.module(name=OAK, version="1.0.0")
def oak(editor_access, settings):
    """Oak wood: rigid heartwood and a softer bark layer."""
    return cr.Draft(
        editor_access,
        "oak",
        rscs=[
            # One Material essence per released material, named module:material.
            essence.Material(f"{OAK}:heartwood"),
            essence.MaterialPhysicalPropertyRigid(
                f"{OAK}:heartwood#rigid",
                material_reference=f"{OAK}:heartwood",
                density=700.0,
            ),
            essence.Material(f"{OAK}:bark"),
            # Soft properties drive the simulation step:
            # dt = h / (3.5 * sqrt(E / density)) at voxel size h.
            essence.MaterialPhysicalPropertySoft(
                f"{OAK}:bark#soft",
                material_reference=f"{OAK}:bark",
                elastic_modulus=2e6,
                density=500.0,
            ),
        ],
    )


CONTACT = "acme.material.pair.oak_stone"


@cr.module(name=CONTACT, version="1.0.0")
def oak_stone_contact(editor_access, settings):
    """Contact behavior between oak heartwood and a stone material.

    A pair-only module: no Material of its own, so nothing requires it
    directly. Modules whose content relies on this contact declare it in
    their dependencies, and the pair enters any world that holds both
    referenced materials; where one side is absent, it is dropped
    harmlessly.
    """
    return cr.Draft(
        editor_access,
        "oak_stone_contact",
        rscs=[
            essence.MaterialPairPropertyRestitution(
                "oak_stone#restitution",
                first_material_reference=f"{OAK}:heartwood",
                second_material_reference="acme.material.stone.granite:granite",
                restitution=0.3,
            )
        ],
    )
