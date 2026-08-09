# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Every tuning number of the create layer, in one place.

The simulation numbers mirror the engine defaults and the known-good rigid
contact configuration proven by ``examples/basic/stack.py``.
"""

# The organization every three-level module name defaults to.
DEFAULT_ORGANIZATION = "rise_origin"

# The one voxel size ladder of the create layer: unit text to meters, in
# 1-2-5 steps. Modules expose resolution choices as these options only, so
# the whole catalog picks sizes from the same physical scale. Guidance:
# 5mm-1cm handheld detail, 2cm the default interactable world, 5cm
# background contact, 10cm to 50cm volumetric filler, 1m to 10m sky-scale
# clouds and celestial bodies that fill the far field.
VOXEL_SIZE_SCALE = {
    "1mm": 0.001,
    "2mm": 0.002,
    "5mm": 0.005,
    "1cm": 0.01,
    "2cm": 0.02,
    "5cm": 0.05,
    "10cm": 0.1,
    "20cm": 0.2,
    "50cm": 0.5,
    "1m": 1.0,
    "2m": 2.0,
    "5m": 5.0,
    "10m": 10.0,
    "20m": 20.0,
    "50m": 50.0,
    "100m": 100.0,
}

# The stable soft step is dt = h / (SOFT_INTEGRATION_FREQUENCY_MULTIPLIER *
# wave speed), with the pressure wave speed
# sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * density)). The engine's
# softbody_and_rigidbody task reports its assembled stability limit at
# runtime: measured h / (4.5 * wave speed) for a chunky block and
# h / (5.6 * wave speed) for one-voxel-thick sheets, so 6.0 keeps a
# margin under the worst geometry seen.
SOFT_INTEGRATION_FREQUENCY_MULTIPLIER = 6.0

# The stable rigid constraint step per voxel size: 0.0025 s at 0.01 m,
# scaling linearly with the smallest voxel size.
RIGID_CONSTRAINT_DT_PER_VOXEL_SIZE = 0.25

# The known-good rigid body contact configuration of examples/basic/stack.py.
# The matching time rule is that every sub-update dt equals the base dt.
RIGID_BODY_CONTACT_ITERATIONS = 20
RIGID_BODY_CONTACT_WARM_START_DAMPING_Z = 0.0
RIGID_BODY_CONTACT_POSITION_CORRECTION_ITERATIONS = 3
RIGID_BODY_CONTACT_POSITION_CORRECTION_RATIO = 0.2

# The round cap handed to resolvelib when pinning module versions; the
# library default of 100 starves large worlds, pip itself passes millions.
RESOLVE_MAX_ROUNDS = 200_000

# Voxel sizes compare rounded to this many decimals because the binding
# stores voxel_size as float32.
VOXEL_SIZE_DECIMALS = 6

# Cover corners within this distance of the extremal plane belong to a face.
FACE_PLANE_EPSILON = 1e-6

# Overlaps thinner than this are touching, not penetration; the bound math
# runs on float32 vectors, so touching faces wobble around 1e-7 m.
CONTACT_OVERLAP_EPSILON = 1e-6

# Realization cache accounting defaults; the lock timeout serves both the
# shared index lock and the WAL busy timeout.
CACHE_DEFAULT_CAP_BYTES = 100 * 1024**3
CACHE_TOUCH_INTERVAL_S = 120.0
CACHE_MEMORY_LIMIT = 64
CACHE_LOCK_TIMEOUT_S = 30.0

# A session folder admits one editor at a time; a second open should fail
# fast instead of waiting for the first to leave.
SESSION_LOCK_TIMEOUT_S = 1.0

# The grid ground plane a preview injects when the content has no floor;
# colors are RGB 0-1 floats.
PREVIEW_GROUND_FILL_COLOR = (0.81, 0.80, 0.78)
PREVIEW_GRID_LINE_COLOR = (0.0, 0.0, 0.0)
PREVIEW_GRID_SPACING = 1.0
PREVIEW_GRID_LINE_WIDTH = 0.02

# The Hosek-Wilkie sky a preview injects. The entity defaults both to
# black, which leaves the sun disk dark and the sky unlit, so the preview
# sets a warm-white sun and a neutral ground albedo bounced into the sky.
# Colors are RGB 0-1 floats, converted to byte at the call site.
PREVIEW_SKY_SUN_COLOR = (1.0, 0.97, 0.94)
PREVIEW_SKY_GROUND_COLOR = (0.5, 0.5, 0.5)
PREVIEW_SKY_RADIANCE_SCALE = 1.3

# A torch memory forecast may claim at most this share of the free VRAM;
# anything larger falls back to the host.
TORCH_DEVICE_FREE_MEMORY_SHARE = 0.8
