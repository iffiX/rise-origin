# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""Preview the dead_snag: a default tree beside a varied one.

Run it directly for the interactive viewer (opens paused, space toggles
the world running); pass ``--headless`` to play a short run instead.

    python examples/create/nature/tree/dead_snag_preview.py [--headless]
"""

import shutil
import sys
import tempfile

from rise_origin import create as cr


def main() -> None:
    is_interactive = "--headless" not in sys.argv
    base = tempfile.mkdtemp(prefix="dead_snag_preview_")
    editor = cr.Editor(
        "dead_snag_preview",
        session_dir=f"{base}/session",
        cache=cr.RealizationCache(f"{base}/cache"),
    )
    try:
        first = editor.create("rise_origin.nature.tree.dead_snag", {"seed": 3})
        second = editor.create(
            "rise_origin.nature.tree.dead_snag", {"decay": 0.9, "seed": 7}
        )
        editor.insert(first, cr.Pose())
        editor.insert(second, cr.Pose((12.0, 0.0, 0.0)))
        editor.preview(is_interactive=is_interactive, device="cuda:0", duration=1.0)
    finally:
        editor.close()
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
