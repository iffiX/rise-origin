"""A complete generation module: a wooden bench with settings and children.

Shows the full authoring contract: seeded randomness, settings, cover
boxes, voxel structures referencing a released material, requires, child
drafts posed under the root, and a dependency on the material package.
"""

from rise_origin import create as cr
from rise_origin.rsc import entity

OAK = "acme.material.wood.oak"
BENCH = "acme.furniture.seating.bench"

VOXEL_SIZE = 0.02


@cr.module(
    name=BENCH,
    version="1.0.0",
    settings=cr.settings(
        length_m=cr.Float(0.6, 3.0, default=1.5),
        wobble=cr.Float(0.0, 1.0, default=0.2),
    ),
    dependencies=(f"{OAK}@^1",),
)
def bench(editor_access, settings):
    """A plain oak bench; wobble jitters the leg placement."""
    # The only randomness door. Same seed, same bench, forever.
    rng = cr.Rng(settings.seed)
    length = settings.length_m

    seat_cells = max(2, round(length / VOXEL_SIZE))
    seat = entity.StructureVoxel(
        "seat",
        material_references=[f"{OAK}:heartwood"],
        bodies=[
            entity.StructureVoxelBody.box(
                seat_cells,
                round(0.35 / VOXEL_SIZE),
                round(0.05 / VOXEL_SIZE),
                voxel_size=VOXEL_SIZE,
                material_sid=0,
                segment_sid=0,
            )
        ],
        is_fixed=True,
    )
    root = cr.Draft(
        editor_access,
        "bench",
        boxes=[cr.Box((0, 0, 0.4), (length, 0.35, 0.45), "static")],
        rscs=[seat],
        # Every material a structure references is declared here, with the
        # voxel sizes the bodies use. The editor checks this at build.
        requires=[cr.MaterialRequire(f"{OAK}@^1", "heartwood", VOXEL_SIZE)],
    )

    leg_cells = round(0.4 / VOXEL_SIZE)
    for index, along in enumerate((0.05, length - 0.1)):
        jitter = (rng.uniform(-0.02, 0.02) * settings.wobble, 0.0, 0.0)
        leg = cr.Draft(
            editor_access,
            f"leg_{index}",
            boxes=[cr.Box((0, 0, 0), (0.05, 0.35, 0.4), "static")],
            rscs=[
                entity.StructureVoxel(
                    f"leg_{index}",
                    material_references=[f"{OAK}:heartwood"],
                    bodies=[
                        entity.StructureVoxelBody.box(
                            round(0.05 / VOXEL_SIZE),
                            round(0.35 / VOXEL_SIZE),
                            leg_cells,
                            voxel_size=VOXEL_SIZE,
                            material_sid=0,
                            segment_sid=0,
                        )
                    ],
                    is_fixed=True,
                )
            ],
            requires=[cr.MaterialRequire(f"{OAK}@^1", "heartwood", VOXEL_SIZE)],
        )
        # Children pose in the root's frame; the caller only places the root.
        root.insert(leg, cr.Pose((along + jitter[0], 0.0, 0.0)))
    return root
