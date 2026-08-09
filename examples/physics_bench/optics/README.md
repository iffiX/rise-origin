# Optics bench

Tests that measure the renderer against closed-form optics, rather than
against how a scene looks.

Each test lives in its own folder and answers one falsifiable question. The
rule for everything here is that the prediction has to be computable without
running the engine, so a disagreement is attributable rather than arguable.

| test | question | state |
|---|---|---|
| `focal_position` | does a voxel lens focus where optics says it should | measured, −8% |

Readings here prefer an observable that is a position, an angle, a
threshold, or a slope. Those survive an unknown gain and an 8-bit ceiling.
An absolute brightness does not, which is why `focal_position` has to keep
its source dim enough that no pixel clips.

## Why not just look at the eye demo

`examples/vision` is a demo, not an instrument. Its readings, peak over mean
and spread, have no closed form, so there is nothing to falsify against and
a wrong result cannot be told from a right one. Every number here is set
against a prediction that came from somewhere else.

## What the renderer does not model

These are absent by construction, not broken. Nothing here is worth a test,
and a result that seems to show one of them is measuring something else.

| quantity | why there is nothing to measure |
|---|---|
| dispersion | `ior` is one scalar on the optical property, so every channel refracts alike and a prism cannot split light |
| Beer-Lambert absorption | no extinction coefficient anywhere. The "Beer-Lambert" in the scatter docstring is a per-bounce albedo multiply, not a path-length law, so there is no exponential to fit against thickness |
| polarization | not represented, so no Brewster angle and no s/p split |
| diffraction | no wave optics. A lens can never be diffraction limited here and the ground truth for any spot size is purely geometric |

One near-neighbour is present but not worth measuring. Fresnel reflectance
is Schlick's approximation, so a measurement would confirm the
approximation is wired up rather than say anything about the physics. Its
few percent departure from the real Fresnel equations at middle angles is a
model choice already made.

Two further limits are real and do belong in a test.

Every ray is born believing it stands in air. The medium a ray sits in is
one scalar seeded at 1.0 when the ray is generated, and one scalar is
enough for the rest of the path because the voxel being entered always
names the index on the far side, so a boundary between two dense media is
reported by the entered voxel and gets the ratio right. Nothing, though,
tells a ray what it started inside. A camera standing in a dense medium
therefore crosses that medium's outer surface with both sides read as air,
so the ray leaves without bending, loses no light to Fresnel, and can never
trap. The error clears at the first air gap and no earlier, which for an
eye is the whole optical train. `examples/vision` sidesteps it today by
leaving the back cavity empty instead of filling it with vitreous fluid,
which is also why that eye is not yet an eye.

A staircase surface also carries a half voxel error in where the surface
sits, which a smoothed normal fixes the direction of but not the position
of. That is the likely source of the bias below.

---

# focal_position

## What it measures

Where a plano-convex voxel lens brings light to a focus, against an exact
ray trace of the same lens with no voxels in it.

A plano-convex has one curved face and one flat one, so exactly one
staircase surface is under test. The flat face is axis aligned and therefore
represented perfectly, which keeps it out of the measurement.

## How the reading is taken

A small emissive ball stands in front of the lens and one camera stands far
behind it, looking back through the glass at the aperture.

The camera's own rays are gathered by the lens onto a point in front of it,
so the aperture lights up when the ball sits at that point. Sweeping the
ball through that distance and taking the brightest aperture measures where
the lens focuses.

Two choices in that setup are load bearing, and both were arrived at by
getting them wrong first:

* **The camera's rays do the gathering, not the source's.** A source small
  enough to act as a point subtends about 0.025 degrees, while a sensor
  samples a cone tens of degrees wide, so backward tracing essentially never
  finds it. Letting the lens concentrate the camera's rays onto the ball
  turns an impossible search into a bright, high-contrast reading.
* **Emission is kept low enough that no pixel clips.** Every ray that finds
  the ball returns the same radiance, so the reading counts rays that
  arrived. If pixels clip, that count becomes "pixels that saw anything",
  which is *larger* out of focus than in it, and the measurement inverts.

## The prediction

Not the paraxial formula. The lens is f/1.7 with its flat face toward the
near-collimated side, which is the high-aberration orientation, so marginal
rays genuinely cross closer than paraxial ones. The paraxial answer is wrong
by 18% here through no fault of the engine.

`get_exact_conjugate` in `main.py` traces the ideal sphere and plane with
exact Snell, weights each ray by the annulus area it stands for, and finds
the distance capturing the most rays inside the ball radius. Spherical
aberration and the finite aperture are inside that answer instead of left
over as error.

## Reproducing

```
python examples/physics_bench/optics/focal_position/sweep.py
python examples/physics_bench/optics/focal_position/sweep.py --index 2.0
python examples/physics_bench/optics/focal_position/sweep.py --no-smooth
```

The sweep prints a table, fits a parabola through the brightest reading and
its two neighbours, and writes `focal_position.png` showing the flux curve
against both predictions. One reading on its own:

```
OPTICS_SOURCE=180 python examples/physics_bench/optics/focal_position/main.py
```

Knobs are environment variables: `OPTICS_INDEX`, `OPTICS_RADIUS`,
`OPTICS_APERTURE`, `OPTICS_SAMPLES`, `OPTICS_EMISSION`, `OPTICS_SOURCE`,
`OPTICS_SMOOTH`, and `OPTICS_NO_LENS` for the control.

## Result

Peak fitted from sweeps at 15 and 8 voxel steps:

| n | paraxial f | paraxial conjugate | **exact conjugate** | **measured** | error vs exact |
|---|---|---|---|---|---|
| 1.5 | 200.0 | 230.77 | **188.40** | **172.7** | **−8.3%** |
| 2.0 | 100.0 | 107.14 | **83.70** | **76.9** | **−8.1%** |

Two indices, conjugates 2.2x apart, both land within 10% of exact geometric
optics and both miss in the same direction by the same amount. A systematic
bias, not scatter.

The sign says the voxel lens is slightly too strong, focusing nearer than
the ideal one, which is the direction a residual outward tilt in the
smoothed normal or the half voxel position error of the staircase would
push it.

**Control.** With the lens removed the aperture reads 0.35 / 0.32 / 0.23
across the same distances, against 6.25 / 5.64 / 6.05 with it. The lens
gathers about 20x more light and the bare source trend is smooth and
monotone, so the peak is not an artifact of the source's angular size.

## What this does and does not establish

Established: **where** the lens focuses, to within 10%, on two
configurations.

Not established, and not to be read into the number:

* **Sharpness.** At best focus about 47% of aperture rays land within the
  ball radius, against 100% for the ideal lens, so the spot is roughly 1.4x
  wider than it should be. That is consistent with a normal that fixes ray
  direction but not the half voxel error in surface position, and it is an
  estimate derived from the same data rather than a measurement.
* **The normals themselves.** Their angular error has never been measured
  directly. A sphere of known radius has a closed-form normal at every face,
  so that test is cheap and is the missing diagnostic if optical behaviour
  ever stops making sense.
* **Other apertures.** Both configurations are f/1.7. The −8% is likely
  aberration adjacent and should not be carried to a very different f
  number without re-measuring.
