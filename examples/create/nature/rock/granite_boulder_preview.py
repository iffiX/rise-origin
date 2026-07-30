# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Preview the granite boulder: a gray worn one beside a fresh pink block.

The preview injects a default Hosek-Wilkie sky and a grid floor, so the
scene needs nothing but the stones. Run it directly for the interactive
viewer (paused on start, close the window to leave); pass ``--headless``
to play a short headless run instead, which is what the automated smoke
uses.

    python examples/create/nature/rock/granite_boulder_preview.py [--headless]
"""

import shutil
import sys
import tempfile

from rise_origin import create as cr


def main() -> None:
    is_interactive = "--headless" not in sys.argv
    base = tempfile.mkdtemp(prefix="granite_boulder_preview_")
    editor = cr.Editor(
        "granite_boulder_preview",
        session_dir=f"{base}/session",
        cache=cr.RealizationCache(f"{base}/cache"),
    )
    try:
        gray = editor.create(
            "rise_origin.nature.rock.granite_boulder",
            {"size": 1.2, "weathering": 0.6, "tone": 0.1, "seed": 7},
        )
        pink = editor.create(
            "rise_origin.nature.rock.granite_boulder",
            {
                "size": 0.9,
                "weathering": 0.1,
                "flatness": 0.3,
                "elongation": 0.3,
                "tone": 0.9,
                "seed": 8,
            },
        )
        editor.insert(gray, cr.Pose((0.0, 0.0, 0.7)))
        editor.insert(pink, cr.Pose((2.0, 0.0, 0.6)))
        editor.preview(is_interactive=is_interactive, device="cuda:0", duration=1.0)
    finally:
        editor.close()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
