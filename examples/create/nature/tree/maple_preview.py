# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Preview the maple: an upright summer tree beside a gnarled autumn one.

The preview injects a default Hosek-Wilkie sky and a grid floor, so the
scene needs nothing but the trees. Run it directly for the interactive
viewer (paused on start, close the window to leave); pass ``--headless``
to play a short headless run instead, which is what the automated smoke
uses.

    python examples/create/maple_preview.py [--headless]
"""

import shutil
import sys
import tempfile

from rise_origin import create as cr

# Grow soft canopies that hang on their fixed holders and sway. The
# resolved step drops to the soft stability bound (about 0.1 ms at the
# 2cm canopy), so the simulation plays far slower than wall time.
IS_SOFT = True

# Share of the naturally loose leaf parts kept unattached; soft trees
# shed them once the world runs, and zero sheds nothing at all.
LEAF_FALL = 0.5


def main() -> None:
    is_interactive = "--headless" not in sys.argv
    base = tempfile.mkdtemp(prefix="maple_preview_")
    editor = cr.Editor(
        "maple_preview",
        session_dir=f"{base}/session",
        cache=cr.RealizationCache(f"{base}/cache"),
    )
    try:
        upright = editor.create(
            "rise_origin.nature.tree.maple",
            {
                "season": "summer",
                "gnarl": 0.1,
                "seed": 3,
                "is_leaf_soft": IS_SOFT,
                "leaf_fall": LEAF_FALL,
            },
        )
        gnarled = editor.create(
            "rise_origin.nature.tree.maple",
            {
                "season": "autumn",
                "gnarl": 0.9,
                "trunk_count": 3,
                "height": 5.0,
                "spread": 0.8,
                "seed": 4,
                "is_leaf_soft": IS_SOFT,
                "leaf_fall": LEAF_FALL,
            },
        )
        editor.insert(upright, cr.Pose())
        report = editor.check_placement_validity(gnarled, cr.Pose((9.0, 0.0, 0.0)))
        editor.insert(gnarled, cr.Pose((9.0, 0.0, 0.0)))
        editor.preview(is_interactive=is_interactive, device="cuda:0", duration=1.0)
    finally:
        editor.close()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
