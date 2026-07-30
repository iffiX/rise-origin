# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Preview the basalt columns: a wide colonnade beside a taller narrow one.

The preview injects a default Hosek-Wilkie sky and a grid floor, so the
scene needs nothing but the formations. Run it directly for the
interactive viewer (paused on start, close the window to leave); pass
``--headless`` to play a short headless run instead, which is what the
automated smoke uses.

    python examples/create/nature/rock/basalt_columns_preview.py [--headless]
"""

import shutil
import sys
import tempfile

from rise_origin import create as cr


def main() -> None:
    is_interactive = "--headless" not in sys.argv
    base = tempfile.mkdtemp(prefix="basalt_columns_preview_")
    editor = cr.Editor(
        "basalt_columns_preview",
        session_dir=f"{base}/session",
        cache=cr.RealizationCache(f"{base}/cache"),
    )
    try:
        wide = editor.create(
            "rise_origin.nature.rock.basalt_columns",
            {
                "width": 2.0,
                "depth": 1.4,
                "height": 2.6,
                "tone": 1,
                "column_width": 0.2,
                "top_unevenness": 0.5,
                "top_outline_dir": (1.0, 0.35, 0.5),
                "seed": 5,
            },
        )
        tall = editor.create(
            "rise_origin.nature.rock.basalt_columns",
            {
                "width": 1.2,
                "depth": 1.0,
                "height": 3.4,
                "column_width": 0.2,
                "tone": 0,
                "top_unevenness": 0.3,
                "top_outline_dir": (-1.0, 0.5, 0.7),
                "seed": 6,
            },
        )
        editor.insert(wide, cr.Pose((0.0, 0.0, 1.3)))
        editor.insert(tall, cr.Pose((3.0, 0.0, 1.7)))
        editor.preview(is_interactive=is_interactive, device="cuda:0", duration=1.0)
    finally:
        editor.close()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
