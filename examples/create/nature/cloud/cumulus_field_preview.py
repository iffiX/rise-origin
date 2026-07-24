# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Preview a cumulus field: many small puffy clouds over a wide patch.

One ``cumulus_field`` structure scatters many small opaque white cumulus
across a patch of sky, dropped at cruising altitude. Run it directly for
the interactive viewer (paused on start, close the window to leave); pass
``--headless`` to play a short headless run instead.

    python examples/create/nature/cloud/cumulus_field_preview.py [--headless]
"""

import shutil
import sys
import tempfile

from rise_origin import create as cr


def main() -> None:
    is_interactive = "--headless" not in sys.argv
    base = tempfile.mkdtemp(prefix="cumulus_field_preview_")
    editor = cr.Editor(
        "cumulus_field_preview",
        session_dir=f"{base}/session",
        cache=cr.RealizationCache(f"{base}/cache"),
    )
    try:
        field = editor.create(
            "rise_origin.nature.cloud.cumulus_field",
            {
                "seed": 3,
                "span": ((-8000.0, -8000.0, -400.0), (8000.0, 8000.0, 400.0)),
                "cloud_num": 100,
                "size": 400.0,
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
