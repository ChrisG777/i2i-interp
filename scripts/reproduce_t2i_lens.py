"""Reproduce the paper-scale T2I Lens sweep on a single GPU.

Runs the i2i->unconditional activation-patching experiment across the five
active task families. The ``style`` cell uses the padding/content
text-subset triplet; every other cell runs the ``all`` mode only. Each cell
is an independent subprocess invocation of
:mod:`experiments.i2i_to_unconditional.i2i_to_unconditional_patch`, so the
GPU is released between cells and ``--skip-if-completed`` makes the script
resumable.

Usage::

    uv run python scripts/reproduce_t2i_lens.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Single block-9 diagonal patch at 1 inference step. Only color uses this
# layout — it's the cheap-and-fast cell that established the lens method.
SOLID_COLOR_FLAGS = [
    "--sweep-mode", "diagonal",
    "--patched-inference-steps", "1",
    "--block-range", "17", "17",
    "--text-token-mode", "all",
    "--categories", "text",
    "--results-subdir", "single9_1step",
    "--skip-if-completed",
]

# MM-block-7 input-to-block0 patch at 4 inference steps. Used by the
# style/humans/add/remove cells. Only style runs the padding/content
# triplet (the others run --text-token-mode all only).
_MM7_BASE = [
    "--sweep-mode", "input_to_block0",
    "--patched-inference-steps", "4",
    "--block-range", "7", "7",
    "--categories", "text",
    "--results-subdir", "mm7_4step",
    "--skip-if-completed",
]
STYLE_FLAGS = [*_MM7_BASE, "--text-token-mode", "all", "padding_only", "content_only"]
MM7_ALL_ONLY_FLAGS = [*_MM7_BASE, "--text-token-mode", "all"]

CELLS: list[tuple[str, list[str], list[str]]] = [
    ("solid_color",       ["--bucket", "solid_color"],       SOLID_COLOR_FLAGS),
    ("style",             ["--bucket", "style"],             STYLE_FLAGS),
    ("dreambench_humans", ["--bucket", "dreambench_humans"], MM7_ALL_ONLY_FLAGS),
    ("add",               ["--bucket", "add"],               MM7_ALL_ONLY_FLAGS),
    ("remove",            ["--bucket", "remove"],            MM7_ALL_ONLY_FLAGS),
]


def _run(cell_name: str, selection: list[str], flags: list[str]) -> None:
    argv = [
        "uv", "run", "python", "-m",
        "experiments.i2i_to_unconditional.i2i_to_unconditional_patch",
        *selection, *flags,
    ]
    print(f"\n=== t2i_lens / {cell_name} ===")
    print(">>> " + " ".join(argv))
    result = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"[FAIL] cell={cell_name} returncode={result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    for cell_name, selection, flags in CELLS:
        _run(cell_name, selection, flags)
    print(f"\nDone. Ran {len(CELLS)} t2i-lens cells.")


if __name__ == "__main__":
    main()
