---
name: create-module-author
description: Author rise-origin create modules — material packages and
  generation modules declared with @cr.module and built through
  editor.create. Use when writing a rise-origin mod, releasing materials,
  adding new object kinds, versioning or deprecating module releases, or
  packaging modules for pip.
---

# Authoring rise-origin create modules

A module is one pure Python function that builds a `Draft`, registered
under a versioned name. Users and agents never call it directly; they call
`editor.create("your.module.name", settings)` and the framework does the
rest: caching, validation, version resolution, and world compilation.

```python
from rise_origin import create as cr
from rise_origin.rsc import entity, essence

@cr.module(
    name="acme.furniture.seating.bench",
    version="1.0.0",
    settings=cr.settings(
        length_m=cr.Float(0.6, 3.0, default=1.5),
    ),
    dependencies=("acme.material.wood.oak@^1",),
)
def bench(editor_access, settings):
    """A plain wooden bench."""
    rng = cr.Rng(settings.seed)
    ...
    return cr.Draft(editor_access, "bench", boxes=..., rscs=..., requires=...)
```

Read `reference.md` in this skill for the full API tables, and
`examples/material_package.py` plus `examples/structure_module.py` for
complete working files.

## 1. Names: the grammar everything meets on

- A module name has four lowercase dotted levels:
  `organization.area.group.kind`, such as `acme.furniture.seating.bench`.
  The organization is YOUR namespace — like a pip package name, it is the
  ownership boundary. Never publish under an organization you do not own;
  `rise_origin.*` is reserved for first-party modules.
- A colon names a material released by a module:
  `acme.material.wood.oak:heartwood`. This exact string is the world
  material reference — voxel structures point at materials with it.
- A hash names a property of a material:
  `acme.material.wood.oak:heartwood#soft`. Property essence NAMES use it;
  their `material_reference` field carries the colon form.

## 2. Identity, cache, and why purity is law

`(module name, version, settings json)` is the identity of every build.
The framework caches builds by identity and relies on this promise:

**The same identity always produces byte-identical content.**

The cache rebuilds evicted content by re-running your body. Session
journals replay identities to restore worlds after a crash. Both silently
trust your function. Therefore, inside the body:

- Draw ALL randomness from `cr.Rng(settings.seed)`. `seed` is a setting
  every module carries automatically — never declare it yourself, never
  use `random`, `numpy.random` global state, or the clock.
- Never read files, the network, environment variables, or module-level
  mutable state.
- You cannot read the scene: the access refuses scene queries for
  detached drafts, so a module output can never depend on what already
  exists in a world. Design shapes that stand on their own; placement is
  the caller's job.

The one sanctioned exception is declared: a module whose content comes
out of a simulation (erosion, weathering, anything the GPU makes
float-order dependent) sets `is_deterministic=False` on the decorator.
Its cache entry then acts as a first-build snapshot (a later rebuild is
a new, equally valid artifact), and the journal records the built
content of every placed instance, so a reopened session still restores
the exact world. Every rule above still applies to such a module — the
flag excuses simulation residue, not sloppy seeding. A deterministic
body may not create a non-deterministic module; that create refuses and
asks you to declare the outer module non-deterministic too.

Cache behavior you get for free: the first `create` of an identity runs
your body once (a fully populated tree comes back); every later `create`
of the same identity returns instantly with boxes and structure only, and
the heavy content loads lazily when the world actually realizes it.

## 3. Declaring a module

Every `@cr.module` field:

- `name` — the four-level name. Registration refuses anything else.
- `version` — concrete semver (`"1.2.0"`). ANY change in output is a new
  version: same identity must mean same bytes forever (same content
  behavior, for a non-deterministic module).
- `settings` — `cr.settings(field=cr.Float(...), ...)`; omit for none.
  The automatic `seed` is always added. See reference.md for field types.
  Every resolution you expose must be a `cr.VoxelSize` field: callers
  pick a unit text from the catalog-wide 1-2-5 ladder (`"5mm"`,
  `"2cm"`, `"10cm"`, ...) and your body receives meters. One ladder for
  every module keeps the whole world on the same physical grain.
- `dependencies` — names of modules whose content must be present in any
  world that uses yours, each with an optional version specifier:
  `("acme.material.wood.oak@^1",)`. Use for material modules your
  structures reference and for pair or optical modules that carry no
  material of their own. Dependencies are presence declarations; they are
  NOT built when your module builds. If your body needs another module's
  draft, call `editor_access.create(...)` inside the body instead.
- `tags` — loose search strings.
- `doc` — usage text for `cr.help_module`; defaults to the docstring.
- `deprecation` — a message that marks this version deprecated. Anyone
  resolving it gets a warning naming your message; resolution itself
  never changes. Deprecate first, remove the old body a few releases
  later, once users had time to move.

Registration is order free and repeat registration of the same name and
version replaces the body, so re-running a notebook cell while iterating
just works.

## 4. Writing the body

Signature is fixed: `fn(editor_access, settings) -> Draft`. Build a
detached tree and return its root.

```python
def bench(editor_access, settings):
    """A plain wooden bench."""
    rng = cr.Rng(settings.seed)
    length = settings.length_m
    root = cr.Draft(
        editor_access,
        "bench",
        boxes=[cr.Box((0, 0, 0), (length, 0.4, 0.45), "static")],
        rscs=[make_seat_structure(length)],
        requires=[
            cr.MaterialRequire("acme.material.wood.oak@^1", "heartwood", 0.02)
        ],
    )
    leg = cr.Draft(editor_access, "leg", boxes=[...], rscs=[...])
    root.insert(leg, cr.Pose((0.05, 0.05, 0.0)))
    return root
```

- `cr.Draft(editor_access, name, *, boxes=(), rscs=(), requires=(),
  is_fixed=False, tags=(), annotations=None)`. Boxes are the authoring
  truth: local-frame cover boxes with a material class — `"static"`
  (never moves), `"dynamic"` (simulated), or `"mixed"` (contains both;
  judged voxel-exactly at compile).
- `rscs` are plain engine objects: entities (`entity.StructureVoxel`),
  essences, resources. No wrapper layer exists.
- Attach children with `root.insert(child, cr.Pose(...), parent=None)` —
  pose in the root draft's frame. Sub-poses compose; the caller places
  only your root.
- Nest other modules with `editor_access.create("their.module@^1",
  {...})` and insert the returned draft. Nested builds cache under their
  own identity; version conflicts across a world surface at compile.
- Pose solvers work on your own detached tree:
  `root.compute_snap_pose(...)`, `compute_along_poses`,
  `compute_grid_poses`, `compute_ring_poses`, `compute_lean_pose`,
  `get_pose`, `get_bounds`. Scene-level queries are refused.
- Voxel structures: `entity.StructureVoxelBody.box(nx, ny, nz,
  voxel_size=..., material_sid=..., segment_sid=...)` builds a body;
  `entity.StructureVoxel(name, material_references=[...], bodies=[...],
  is_fixed=...)` wraps bodies; `material_sid` indexes into
  `material_references`, whose entries are world material references
  (`"acme.material.wood.oak:heartwood"`).

## 5. Requires: declaring the materials your content uses

Every material your structures reference must be declared:

```python
cr.MaterialRequire("acme.material.wood.oak@^1", "heartwood", 0.02)
```

- First argument: the material MODULE reference, version specifier
  behind `@` (`^1`, `>=1.2`, exact `1.2.0`, or none for any).
- Second: the material name inside that module (the part after the
  colon). Exactly one material per require.
- Third: the voxel size or sizes (a float or a tuple) your bodies use
  for this material. Sizes compare rounded to 6 decimals (float32).

Requires ride your draft nodes and replay through the cache, so declare
them on the drafts that use them. At compile, all requires of a world
resolve together: one version per module per world, pinned by
intersecting every specifier, preferring the newest — the pip model. Ship
compatible specifiers (`^1`) rather than exact pins unless you truly need
one version.

## 6. Material packages

A material module is an ordinary module whose draft carries only
essences — no geometry:

```python
@cr.module(name="acme.material.wood.oak", version="1.0.0")
def oak(editor_access, settings):
    """Oak wood materials."""
    mod = "acme.material.wood.oak"
    return cr.Draft(
        editor_access,
        "oak",
        rscs=[
            essence.Material(f"{mod}:heartwood"),
            essence.MaterialPhysicalPropertyRigid(
                f"{mod}:heartwood#rigid",
                material_reference=f"{mod}:heartwood",
                density=700.0,
            ),
        ],
    )
```

- Release several materials from one module; each `Material` essence is
  named `module:material`, each property essence `module:material#prop`
  with `material_reference` pointing at the colon name.
- Physical kinds: `MaterialPhysicalPropertyFixed` (never simulated),
  `...Rigid` (rigid body, `density`), `...Soft` (soft body,
  `elastic_modulus`, `density`). Soft moduli drive the simulation step:
  the stable step is `h / (4.6 * wave_speed)` at voxel size `h`, with
  the pressure wave speed from `E`, `density`, and `poissons_ratio` —
  a very stiff soft material forces tiny steps on every world using it.
- Pair properties (`MaterialPairPropertyRestitution` and friends, with
  `first_material_reference`/`second_material_reference`) may live in the
  same module or in a dedicated pair module that other modules pull in
  via `dependencies`. A pair whose two materials are not both in the
  world is dropped harmlessly.
- One version of a material module exists per world — two mods demanding
  incompatible versions of the same material module cannot share a world,
  by design (one physics per world).
- Texture rides optical looks, not extra materials: one material may
  carry many optical property essences (its looks), and every voxel
  picks a look through the optical mid grid — shades, weathering, and
  seasonal palettes are looks on one material, all sharing its one
  physical identity. The exact rules (look grids, the 1024 cap, one
  optical category per material, one physical kind per segment, soft
  welds to non-soft neighbors, non-soft pairs air-gap) are in
  reference.md under "Multi-material rules inside one structure".

## 7. What the editor will enforce on your module

Design against these instead of debugging them:

- The body must return a detached `Draft` (not placed, not another type).
- Honesty checks at every cache-miss build: every material reference in
  your structures is covered by your requires or by essences you carry
  in-tree; every body voxel size appears among the declared require
  sizes; constraints stay inside one structure.
- Recursive creates that cycle back into your own module are refused.
- In a frozen session (simulation pinned), a create whose requires are
  not covered by the session's declared set, or whose soft materials
  demand a smaller step than the frozen dt, faults immediately at
  `editor.create`.
- At compile, the embedding gate checks voxel co-occupancy: static
  inside static is a legal embedding; anything dynamic co-occupying
  anything faults. Use `"mixed"` boxes where covers legitimately overlap
  both ways and let the voxel gate decide.

## 8. Versioning and retirement

- Old worlds reference your module at exact versions. Keep old version
  bodies registered in your package so those worlds can always rebuild;
  deleting a version orphans every session that used it.
- To retire a version: add `deprecation="use ^2, the crown shape
  changed"` to it first, publish, and remove the body only a few mod
  releases later.
- Never register a name another mod owns, and never patch another mod's
  behavior. To alter another module's output: depend on it, create it
  inside your body, modify the returned draft — the result carries YOUR
  identity — or release your own superseding module.

## 9. Test your module before publishing

Minimum bar, per module:

1. **Purity double-build**: build once, wipe the cache blobs, force a
   rebuild, byte-compare. If this fails, some randomness or state escaped
   the seed. A module declared `is_deterministic=False` replaces the byte
   compare with build-twice-both-valid: each build honest, inside budget,
   and correct against its requires.

   ```python
   with cr.Editor("t", session_dir=..., cache=cr.RealizationCache(...)) as e:
       first = e.create("acme.furniture.seating.bench", {"seed": 7})
       e.cache.prune(0)                      # evict blobs
       e.insert(first, cr.Pose())
       e.realize()                            # rebuilds through the body
       # compare saved bytes of two independent sessions if strictness
       # matters; identical trees are the passing signal
   ```

2. **Preview**: `editor.insert(draft, cr.Pose())` then
   `editor.preview(is_interactive=False, duration=0.5)` runs the real
   engine headless; interactively, `editor.preview()` opens paused for
   inspection.
3. **Settings sweep**: build at min, max, and default of each setting;
   the honesty checks and the engine will catch degenerate shapes.
4. Check `cr.help_module("your.module.name")` reads as a usable manual —
   that text is what agents see.

## 10. Publishing as a pip package

- Ship a normal pip package that imports register your modules: the
  `@cr.module` decorators run at import time, so your package's
  `__init__` should import every module file.
- Declare the engine dependency in YOUR pyproject:
  `rise-origin>=X` — pip enforces engine compatibility; the module
  registry handles only module-to-module versions.
- Document the import (`import acme_furniture`) users need before
  `editor.create` can resolve your names, and list your module names in
  the README. `cr.list_modules()` shows everything registered.
