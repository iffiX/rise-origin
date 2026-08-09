# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Sweep the source through focus and fit where the lens actually gathers.

Each reading is taken by running main.py in its own process, since a world
is built once and the source has to move between readings. The readings are
fitted with a parabola through the three points around the brightest, and
set against an exact ray trace of the same lens with no voxels in it.

Run it::

    python examples/physics_bench/optics/focal_position/sweep.py
    python examples/physics_bench/optics/focal_position/sweep.py --index 2.0
    python examples/physics_bench/optics/focal_position/sweep.py --no-smooth
"""

import argparse
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main as bench  # noqa: E402

RESULT_PATTERN = re.compile(r"RESULT source=([\d.]+) flux=([\d.naN]+)")


def take_reading(distance: float, environment: Dict[str, str]) -> Optional[float]:
    """Run one reading of the bench.

    Args:
        distance: Where to stand the source, in voxels.
        environment: Extra environment for the bench process.

    Returns:
        The aperture flux, or ``None`` when the reading did not arrive.
    """
    child = dict(os.environ)
    child.update(environment)
    child["OPTICS_SOURCE"] = f"{distance}"
    done = subprocess.run(
        [sys.executable, os.path.join(HERE, "main.py")],
        capture_output=True,
        text=True,
        env=child,
    )
    for line in done.stdout.splitlines():
        found = RESULT_PATTERN.search(line)
        if found:
            try:
                return float(found.group(2))
            except ValueError:
                return None
    return None


def fit_peak(distances, fluxes) -> float:
    """Fit a parabola through the brightest reading and its two neighbors.

    Args:
        distances: Source distances, in voxels.
        fluxes: Aperture flux at each distance.

    Returns:
        Where the fit peaks, in voxels.
    """
    values = np.asarray(fluxes, dtype=np.float64)
    if not np.isfinite(values).any():
        return float("nan")
    best = int(np.nanargmax(values))
    if best == 0 or best == len(values) - 1:
        return float(distances[best])
    x = np.asarray(distances[best - 1 : best + 2], dtype=np.float64)
    y = values[best - 1 : best + 2]
    a, b, _ = np.polyfit(x, y, 2)
    if a >= 0.0:
        return float(distances[best])
    return float(-b / (2.0 * a))


def make_plot(distances, fluxes, measured, exact, paraxial, path: str) -> None:
    """Draw the sweep against the two predictions.

    Args:
        distances: Source distances, in voxels.
        fluxes: Aperture flux at each distance.
        measured: Fitted peak, in voxels.
        exact: Conjugate from the exact ray trace, in voxels.
        paraxial: Conjugate from the paraxial formula, in voxels.
        path: Where to write the image.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plot
    except ImportError:
        print("matplotlib missing, skipping the plot")
        return

    figure, axes = plot.subplots(figsize=(7.5, 4.5))
    axes.plot(distances, fluxes, "o-", color="#2b7bba", label="aperture flux")
    axes.axvline(measured, color="#2b7bba", linestyle="-", label=f"measured {measured:.1f}")
    axes.axvline(exact, color="#26a269", linestyle="--", label=f"exact trace {exact:.1f}")
    axes.axvline(paraxial, color="#c64600", linestyle=":", label=f"paraxial {paraxial:.1f}")
    axes.set_xlabel("source distance in front of lens (voxels)")
    axes.set_ylabel("aperture flux (lit area)")
    axes.set_title("Where the voxel lens gathers light")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    print(f"plot written to {path}")


def main():
    """Run the sweep, fit the peak, and set it against the predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=float, default=1.5)
    parser.add_argument("--radius", type=float, default=100.0)
    parser.add_argument("--aperture", type=float, default=30.0)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--plot", default=os.path.join(HERE, "focal_position.png"))
    args = parser.parse_args()

    environment = {
        "OPTICS_INDEX": f"{args.index}",
        "OPTICS_RADIUS": f"{args.radius}",
        "OPTICS_APERTURE": f"{args.aperture}",
        "OPTICS_SAMPLES": f"{args.samples}",
        "OPTICS_SMOOTH": "0" if args.no_smooth else "1",
    }
    for key, value in environment.items():
        os.environ[key] = value

    # Reload so the bench's constants pick the swept configuration up.
    import importlib

    importlib.reload(bench)
    exact, share = bench.get_exact_conjugate()
    paraxial = bench.get_paraxial_conjugate()

    # Centre the sweep on the exact answer, since the paraxial one is well
    # off for a fast lens and would put the peak at the edge of the window.
    span = 0.45 * exact
    distances = list(np.linspace(exact - span, exact + span, args.points))

    print(f"lens R={args.radius} n={args.index} semi-aperture={args.aperture}")
    print(f"paraxial f          {bench.get_focal_length():8.2f} vx")
    print(f"paraxial conjugate  {paraxial:8.2f} vx")
    print(f"exact conjugate     {exact:8.2f} vx   (captures {share:.3f} of aperture)")
    print()
    print(f"{'source (vx)':>12} {'flux':>10}")

    fluxes = []
    for distance in distances:
        flux = take_reading(round(distance, 2), environment)
        fluxes.append(float("nan") if flux is None else flux)
        print(f"{distance:12.1f} {fluxes[-1]:10.4f}")

    measured = fit_peak(distances, fluxes)
    print()
    print(f"measured conjugate  {measured:8.2f} vx")
    if np.isfinite(measured):
        print(f"error vs exact      {100.0 * (measured - exact) / exact:+8.1f} %")
        print(f"error vs paraxial   {100.0 * (measured - paraxial) / paraxial:+8.1f} %")
    make_plot(distances, fluxes, measured, exact, paraxial, args.plot)


if __name__ == "__main__":
    main()
