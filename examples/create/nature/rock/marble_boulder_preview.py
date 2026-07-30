# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Preview the marble boulder: a feathery Carrara beside a bold-veined one.

The preview injects a default Hosek-Wilkie sky and a grid floor, so the
scene needs nothing but the stones. Run it directly for the interactive
viewer (paused on start, close the window to leave); pass ``--headless``
to play a short headless run instead, which is what the automated smoke
uses.

    python examples/create/nature/rock/marble_boulder_preview.py [--headless]
"""

import shutil
import sys
import tempfile

from rise_origin import create as cr


def main() -> None:
    is_interactive = "--headless" not in sys.argv
    base = tempfile.mkdtemp(prefix="marble_boulder_preview_")
    editor = cr.Editor(
        "marble_boulder_preview",
        session_dir=f"{base}/session",
        cache=cr.RealizationCache(f"{base}/cache"),
    )
    try:
        feathery = editor.create(
            "rise_origin.nature.rock.marble_boulder",
            {"size": 1.2, "vein_thickness": 0.2, "vein_winding": 0.5, "seed": 3},
        )
        bold = editor.create(
            "rise_origin.nature.rock.marble_boulder",
            {
                "size": 0.9,
                "vein_thickness": 0.8,
                "vein_winding": 0.8,
                "weathering": 0.3,
                "seed": 4,
            },
        )
        editor.insert(feathery, cr.Pose((0.0, 0.0, 0.7)))
        editor.insert(bold, cr.Pose((2.0, 0.0, 0.6)))
        editor.preview(is_interactive=is_interactive, device="cuda:0", duration=1.0)
    finally:
        editor.close()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
