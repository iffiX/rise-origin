# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Preview a stratus field: wide low slabs spread over a patch.

One ``stratus_field`` structure scatters wide low stratus slabs across a
patch of sky, dropped low. Run it directly for the interactive viewer
(paused on start, close the window to leave); pass ``--headless`` to play
a short headless run instead.

    python examples/create/nature/cloud/stratus_field_preview.py [--headless]
"""

import shutil
import sys
import tempfile

from rise_origin import create as cr


def main() -> None:
    is_interactive = "--headless" not in sys.argv
    base = tempfile.mkdtemp(prefix="stratus_field_preview_")
    editor = cr.Editor(
        "stratus_field_preview",
        session_dir=f"{base}/session",
        cache=cr.RealizationCache(f"{base}/cache"),
    )
    try:
        field = editor.create(
            "rise_origin.nature.cloud.stratus_field",
            {
                "seed": 3,
                "size": 1500.0,
                "span": ((-8000.0, -8000.0, -200.0), (8000.0, 8000.0, 200.0)),
                "cloud_num": 20,
                "is_rainy": False,
                "voxel_size": "20m",
            },
        )
        editor.insert(field, cr.Pose((0.0, 0.0, 1000.0)))
        editor.preview(is_interactive=is_interactive, device="cuda:0", duration=1.0)
    finally:
        editor.close()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
