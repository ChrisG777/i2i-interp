"""Reproduce the paper-scale I2I-to-I2I Patching sweep on a single GPU.

Three pair families, each loaded from a checked-in ``.txt`` file under
``experiments/i2i_to_i2i_patching/pairs/``. Each cell is an independent
subprocess invocation of
:mod:`experiments.i2i_to_i2i_patching.i2i_to_i2i_patch`;
``--skip-if-completed`` makes the script resumable.

Usage::

    uv run python scripts/reproduce_i2i_to_i2i_patching.py
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIRS_DIR = REPO_ROOT / "experiments" / "i2i_to_i2i_patching" / "pairs"

ALL_TEXT_TOKEN_MODES = ("all", "padding_only", "content_only")


@dataclass(frozen=True)
class Cell:
    """One paper-scale i2i->i2i cell. ``block`` is the single block index to
    patch (flat layout, one patched.png per mode)."""

    subdir: str
    pairs_file: str
    block: int
    text_token_modes: tuple[str, ...]


CELLS: list[Cell] = [
    Cell("single9_4step_color",         "single9_4step_color.txt",     17, ALL_TEXT_TOKEN_MODES),
    Cell("mm7_4step_style_to_real",     "mm7_4step_style_to_real.txt",   7, ALL_TEXT_TOKEN_MODES),
    Cell("mm7_4step_dreambench_humans", "mm7_4step_dreambench_humans.txt", 7, ALL_TEXT_TOKEN_MODES),
]


def _run(cell: Cell) -> None:
    pairs_path = PAIRS_DIR / cell.pairs_file
    assert pairs_path.exists(), f"missing pairs file: {pairs_path}"
    argv = [
        "uv", "run", "python", "-m", "experiments.i2i_to_i2i_patching.i2i_to_i2i_patch",
        "--pair-list", str(pairs_path),
        "--block-range", str(cell.block), str(cell.block),
        "--num-inference-steps", "4",
        "--text-token-mode", *cell.text_token_modes,
        "--results-subdir", cell.subdir,
        "--skip-if-completed",
    ]
    print(f"\n=== i2i_to_i2i_patching / {cell.subdir} ===")
    print(">>> " + " ".join(argv))
    result = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"[FAIL] cell={cell.subdir} returncode={result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    for cell in CELLS:
        _run(cell)
    print(f"\nDone. Ran {len(CELLS)} i2i->i2i patching cells.")


if __name__ == "__main__":
    main()
