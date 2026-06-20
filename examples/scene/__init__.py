# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Procedural voxel scene example: a research outpost.

Prototype for the future ``fab`` package — built as a small package so each
concept lives in its own file:

* :mod:`constants` — grid sizes for the two voxel resolutions, world extent.
* :mod:`materials` — material name constants, the segment-id map, render helpers.
* :mod:`stamp` — the ``Stamp`` voxel-block abstraction.
* :mod:`world` — the ``World`` wrapper around an ``add_voxel_grid_body`` body.
* :mod:`layout` — procedural floorplan: maze carver + biome assignment.
* :mod:`objects` — the catalog of voxel-object builders (one module per category).
* :mod:`biomes` — per-biome cell furnishers.
* :mod:`lab` — the entry point: ``build()`` + ``main()``.

Run with ``python -m examples.scene.lab`` (from the repo root) or
``python examples/scene/lab.py`` directly.
"""
