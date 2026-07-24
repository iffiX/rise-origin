# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Preview a cumulonimbus field: a few large deep storm heaps.

One ``cumulonimbus_field`` structure scatters a few large deep
cumulonimbus across a wide patch of sky, dropped high. Run it directly
for the interactive viewer (paused on start, close the window to leave);
pass ``--headless`` to play a short headless run instead.

    python examples/create/nature/cloud/cumulonimbus_field_preview.py [--headless]
"""

import shutil
import sys
import tempfile

from rise_origin import create as cr


def main() -> None:
    is_interactive = "--headless" not in sys.argv
    base = tempfile.mkdtemp(prefix="cumulonimbus_field_preview_")
    editor = cr.Editor(
        "cumulonimbus_field_preview",
        session_dir=f"{base}/session",
        cache=cr.RealizationCache(f"{base}/cache"),
    )
    try:
        field = editor.create(
            "rise_origin.nature.cloud.cumulonimbus_field",
            {
                "seed": 3,
                "size": 1500.0,
                "span": ((-9000.0, -9000.0, -400.0), (9000.0, 9000.0, 400.0)),
                "cloud_num": 30,
                "is_rainy": False,
                "voxel_size": "50m",
            },
        )
        editor.insert(field, cr.Pose((0.0, 0.0, 1500.0)))
        editor.preview(is_interactive=is_interactive, device="cuda:0", duration=1.0)
    finally:
        editor.close()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
