# create-module-author reference

API tables for module authors. Everything here is importable from
`rise_origin.create` (conventionally `from rise_origin import create as cr`)
unless noted otherwise.

## @cr.module

```python
@cr.module(
    name: str,                      # "organization.area.group.kind"
    version: str,                   # concrete semver, "1.2.0"
    settings: SettingsSchema | None = None,   # cr.settings(...)
    dependencies: tuple[str, ...] = (),       # "name" or "name@specifier"
    tags: tuple[str, ...] = (),
    doc: str | None = None,         # None takes the function docstring
    deprecation: str | None = None, # message warned at resolution
    is_deterministic: bool = True,  # False for simulation-based content
)
def body(editor_access, settings) -> cr.Draft: ...
```

Calling the decorated function directly raises; build through
`editor.create`. The registered spec stays reachable as `body.spec`.

`is_deterministic=False` declares that the same identity may build
different bytes (GPU simulation residue). The cache entry becomes a
first-build snapshot, the journal records placed content so cold opens
restore the exact world, and a deterministic body creating such a module
is refused. The seed rules still apply in full.

## Settings fields (cr.settings keyword values)

| Field | Arguments | Notes |
|---|---|---|
| `cr.Float(min, max, default=...)` | inclusive range | |
| `cr.Int(min, max, default=...)` | inclusive range | `seed` is an automatic Int; never declare it |
| `cr.Bool(default=...)` | | |
| `cr.Str(default=...)` | | |
| `cr.Choice(options, default=...)` | sequence of strings | |
| `cr.FloatList(min, max, default=..., length=None)` | per-item range | `length=None` accepts any count |
| `cr.VoxelSize(default="2cm", min="5mm", max="10cm")` | unit texts from the one ladder | the body reads meters (float); callers pass the text |

Every resolution a module exposes goes through `cr.VoxelSize`. The ladder
is fixed for the whole catalog, in 1-2-5 steps:
`"1mm" "2mm" "5mm" "1cm" "2cm" "5cm" "10cm" "20cm" "50cm" "1m"`.
Choosing guidance: 5mm-1cm handheld and finger detail, 2cm the default
interactable world (grip, climb, stack), 5cm background contact, 10cm and
up volumetric filler and far field. Remember the world step couples to
your choice: rigid content contributes `dt = 0.25 * h_min`, so the finest
voxel anywhere taxes every step of the world.

Inside the body, read values as attributes: `settings.length_m`,
`settings.seed`. Validation happens at `editor.create`; out-of-range or
unknown keys raise `cr.SettingsError` there.

## cr.Rng

`cr.Rng(settings.seed)` — the only sanctioned randomness source. Supports
stable named splits so subtrees draw independent streams; see its
docstrings. Never use `random` or global numpy state.

## cr.Draft (author-facing surface)

```python
cr.Draft(
    editor_access,                  # pass through, always
    name: str,                      # sibling names uniquify on insert
    *,
    boxes: Sequence[cr.Box] = (),   # local-frame cover, the authoring truth
    rscs: Sequence[object] = (),    # entities, essences, resources
    requires: Sequence[cr.MaterialRequire] = (),
    is_fixed: bool = False,         # fixes the whole subtree below
    tags: Sequence[str] = (),
    annotations: Mapping[str, cr.Annotation] | None = None,
)
```

Structure and layout on a detached tree:

- `draft.insert(child, pose, parent=None)` — attach a detached child at a
  pose in `draft`'s frame.
- `draft.get_pose(thing)` / `draft.get_bounds(...)` — frames and bounds
  within the subtree.
- Solvers (all return poses, mutate nothing): `compute_snap_pose`,
  `compute_facing_pose`, `compute_along_poses`, `compute_grid_poses`,
  `compute_ring_poses`, `compute_lean_pose`.
- `draft.find_node_by_name(name)`, `draft.iterate_subtree_nodes()`.

Scene-level queries (`get_nearby`, `cast_ray`, overlap checks against the
session) are refused for detached drafts — module bodies are scene-blind
by design.

## cr.Box

`cr.Box(min_xyz, max_xyz, material_class)` — local-frame axis-aligned
cover. `material_class` is `"static"`, `"dynamic"`, or `"mixed"`.
Static-static overlaps are legal embeddings; anything dynamic overlapping
is a violation; mixed defers to the voxel-exact gate at compile.

## cr.MaterialRequire

```python
cr.MaterialRequire(module_reference, material_reference, voxel_size)
# ("acme.material.wood.oak@^1", "heartwood", 0.02)
# ("acme.material.wood.oak",    "bark",      (0.02, 0.05))
```

- `module_reference`: text or `cr.ModuleReference`; version specifier
  behind `@` (`^1`, `>=1.2`, exact, or absent).
- `material_reference`: one material name (the part after the colon).
- `voxel_size`: float or tuple of floats; compared rounded to 6 decimals.
- World material reference = `f"{module_name}:{material_reference}"` —
  the string structures use in `material_references`.

## Module name grammar

- Module: `organization.area.group.kind` — four lowercase dotted levels.
- Material in a module: `module:material`.
- Property of a material: `module:material#property` (essence names).
- Version part: `name@1.2.0` (concrete) or `name@^1` (specifier); `*` or
  no part accepts everything.

## Engine rsc objects commonly used in bodies

From `from rise_origin.rsc import entity, essence`:

| Object | Key arguments |
|---|---|
| `essence.Material(name)` | name = `"module:material"` |
| `essence.MaterialPhysicalPropertyFixed(name, material_reference=...)` | never simulated |
| `essence.MaterialPhysicalPropertyRigid(name, material_reference=..., density=...)` | rigid body |
| `essence.MaterialPhysicalPropertySoft(name, material_reference=..., elastic_modulus=..., density=...)` | soft body; drives dt |
| `essence.MaterialPairPropertyRestitution(name, first_material_reference=..., second_material_reference=..., restitution=...)` | emitted only when both materials are in the world |
| `entity.StructureVoxelBody.box(nx, ny, nz, voxel_size=..., material_sid=..., segment_sid=...)` | one voxel body |
| `entity.StructureVoxel(name, material_references=[...], bodies=[...], is_fixed=...)` | `material_sid` indexes `material_references` |

Property essence names follow `module:material#kind`; their
`material_reference` carries the colon form without the hash.

## Multi-material rules inside one structure (texture faking)

Verified against the kernel manager. Materials are addressed PER VOXEL:
every voxel's `material_sid` indexes the structure-level
`material_references` list, and `StructureVoxelBody.from_grid(...,
material_grid=...)` assigns a distinct material per voxel. A segment is
the set of voxels in one body sharing a `segment_bid`; it is the physics
unit.

- A segment MUST reference materials of one physical kind (fixed, rigid,
  or soft); mixing kinds in one segment throws at build.
- Atlas segments MUST share identical mass, center of mass, and inertia
  across their materials; plain rigid density is NOT validated but each
  voxel's own density is summed into the body's mass and inertia — give
  every optical variant the same density yourself.
- Different segments of one entity MAY carry different physical kinds:
  a fixed trunk segment and a soft leaf segment coexist in one
  StructureVoxel. A soft segment bonds to its non-soft neighbors
  voxel-by-voxel through links; two non-soft segments touching are an
  air gap unless an explicit constraint joins them, and constraints MUST
  join rigid or rigid-fixed segments only.
- Structure-level `is_fixed=True` makes the WHOLE entity static (no
  rigid bodies, no links); per-material fixed properties make single
  segments static inside an otherwise dynamic entity.

The sanctioned way to give one material several looks: optical
deviations. A structure carries `material_optical_references` — one list
of optical property essence names per material reference — and each
voxel picks a look through the `material_optical_reference_mid_grid` of
`StructureVoxelBody.from_grid` (up to 1024 looks per material). Looks
share the material's one physical identity, so intra-volume look
variation adds no physics cost, and all looks of one material must stay
in one optical category (never mix a transparent look with a solid
one). Each look is a `MaterialOpticalPropertySimpleMicrofacet` essence
plus its `TextureConstant` albedo resource. Reserve genuinely separate
materials for genuinely different physics (wood vs leaf), not for
shades.

## Simulation step rule (why soft moduli matter)

For every soft material at voxel size `h`, the stable step is
`h / (4.6 * wave_speed)`, with the pressure wave speed
`sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * density))`. Rigid content adds
`0.25 * h_min`. The world's dt is the minimum over all of these — a stiff
soft material at a small voxel size makes every world using it slow.

## Discovery and lifecycle helpers

- `cr.list_modules()` — every registered module name, sorted.
- `cr.help_module(name)` — the manual: version, doc, DEPRECATED line when
  set, dependencies, tags, every setting with type, range, default.
- Registering the same name and version again replaces the body (notebook
  reload). Different versions coexist; resolution prefers the newest that
  satisfies every constraint in play.

## Errors you may see while iterating

| Error | Meaning |
|---|---|
| `cr.ModuleError` | protocol break: bad name/version, direct call, cycle, unresolvable dependency graph, unsatisfiable version set |
| `cr.SettingsError` | settings validation failed at create |
| `cr.EditorError` | honesty check failed, frozen-session fault, or another editor rule |
| `cr.EditorMaterialError` | material resolution failed at compile: missing module, missing material in a module, version conflict |
