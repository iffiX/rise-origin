# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Preview the limestone boulder: a buff massive one beside a pocked gray.

The preview injects a default Hosek-Wilkie sky and a grid floor, so the
scene needs nothing but the stones. Run it directly for the interactive
viewer (paused on start, close the window to leave); pass ``--headless``
to play a short headless run instead, which is what the automated smoke
uses.

    python examples/create/nature/rock/limestone_boulder_preview.py [--headless]
"""

import shutil
import sys
import tempfile

from rise_origin import create as cr


def main() -> None:
    is_interactive = "--headless" not in sys.argv
    base = tempfile.mkdtemp(prefix="limestone_boulder_preview_")
    editor = cr.Editor(
        "limestone_boulder_preview",
        session_dir=f"{base}/session",
        cache=cr.RealizationCache(f"{base}/cache"),
    )
    try:
        buff = editor.create(
            "rise_origin.nature.rock.limestone_boulder",
            {"size": 1.2, "tone": 0.0, "layering": 0.2, "karst": 0.3, "seed": 5},
        )
        gray = editor.create(
            "rise_origin.nature.rock.limestone_boulder",
            {
                "size": 0.9,
                "tone": 0.7,
                "layering": 0.8,
                "karst": 1.0,
                "weathering": 0.6,
                "seed": 6,
            },
        )
        editor.insert(buff, cr.Pose((0.0, 0.0, 0.7)))
        editor.insert(gray, cr.Pose((2.0, 0.0, 0.6)))
        editor.preview(is_interactive=is_interactive, device="cuda:0", duration=1.0)
    finally:
        editor.close()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
