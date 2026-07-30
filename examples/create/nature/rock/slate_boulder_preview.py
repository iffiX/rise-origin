# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Preview the slate boulder: sharp cleaved chunks resting on the floor.

The preview injects a default Hosek-Wilkie sky and a grid floor, so the
scene needs nothing but the chunks. Run it directly for the interactive
viewer (paused on start, close the window to leave); pass ``--headless``
to play a short headless run instead, which is what the automated smoke
uses.

    python examples/create/nature/rock/slate_boulder_preview.py [--headless]
"""

import shutil
import sys
import tempfile

from rise_origin import create as cr


def main() -> None:
    is_interactive = "--headless" not in sys.argv
    base = tempfile.mkdtemp(prefix="slate_boulder_preview_")
    editor = cr.Editor(
        "slate_boulder_preview",
        session_dir=f"{base}/session",
        cache=cr.RealizationCache(f"{base}/cache"),
    )
    try:
        sharp = editor.create(
            "rise_origin.nature.rock.slate_boulder",
            {
                "size": 0.9,
                "raggedness": 0.25,
                "stain": 0.35,
                "seed": 3,
            },
        )
        stepped = editor.create(
            "rise_origin.nature.rock.slate_boulder",
            {
                "size": 0.7,
                "flatness": 0.65,
                "raggedness": 0.7,
                "tone": 0.8,
                "stain": 0.15,
                "seed": 8,
            },
        )
        small = editor.create(
            "rise_origin.nature.rock.slate_boulder",
            {
                "size": 0.4,
                "elongation": 0.5,
                "raggedness": 0.45,
                "tone": 0.2,
                "seed": 11,
            },
        )
        editor.insert(sharp, cr.Pose((0.0, 0.0, 0.6)))
        editor.insert(stepped, cr.Pose((1.4, 0.4, 0.5)))
        editor.insert(small, cr.Pose((-1.1, -0.3, 0.4)))
        editor.preview(is_interactive=is_interactive, device="cuda:0", duration=1.0)
    finally:
        editor.close()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
