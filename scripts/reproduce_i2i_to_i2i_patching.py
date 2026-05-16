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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIRS_DIR = REPO_ROOT / "experiments" / "i2i_to_i2i_patching" / "pairs"

ALL_TEXT_TOKEN_MODES = ("all", "padding_only", "content_only")

# (results_subdir, block_range_first, block_range_last, pairs_file)
CELLS: list[tuple[str, str, str, str]] = [
    ("single9_4step_color",        "17", "17", "single9_4step_color.txt"),
    ("mm7_4step_style_to_real",    "7",  "7",  "mm7_4step_style_to_real.txt"),
    ("mm7_4step_dreambench_humans", "7", "7",  "mm7_4step_dreambench_humans.txt"),
]


def _run(name: str, lo: str, hi: str, pairs_file: str) -> None:
    pairs_path = PAIRS_DIR / pairs_file
    assert pairs_path.exists(), f"missing pairs file: {pairs_path}"
    argv = [
        "uv", "run", "python", "-m", "experiments.i2i_to_i2i_patching.i2i_to_i2i_patch",
        "--pair-list", str(pairs_path),
        "--block-range", lo, hi,
        "--num-inference-steps", "4",
        "--text-token-mode", *ALL_TEXT_TOKEN_MODES,
        "--results-subdir", name,
        "--skip-if-completed",
    ]
    print(f"\n=== i2i_to_i2i_patching / {name} ===")
    print(">>> " + " ".join(argv))
    result = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"[FAIL] cell={name} returncode={result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    for name, lo, hi, pairs_file in CELLS:
        _run(name, lo, hi, pairs_file)
    print(f"\nDone. Ran {len(CELLS)} i2i->i2i patching cells.")


if __name__ == "__main__":
    main()
