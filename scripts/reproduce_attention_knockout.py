"""Reproduce the paper-scale Attention Knockout sweep on a single GPU.

Runs the same task families and hyperparameters used in the paper
(``ref->text`` + ``ref->image`` knockouts at 4 inference steps, plus the
text-subset variants), one cell at a time. Each cell calls
:mod:`experiments.attention_knockout.knockout_run` as a subprocess so the
GPU is fully released between cells. ``--skip-if-completed`` makes the
script idempotent — rerun anytime to resume.

Usage::

    uv run python scripts/reproduce_attention_knockout.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

COMMON_FLAGS = [
    "--settings", "ref->text", "ref->image",
                  "ref->text[padding]", "ref->text[content]",
    "--full-ko-only",
    "--num-inference-steps", "4",
    "--results-subdir", "full_ko_4step",
    "--skip-if-completed",
]

CELLS: list[tuple[str, list[str]]] = [
    ("solid_color",       ["--bucket", "solid_color"]),
    ("style",             ["--bucket", "style"]),
    ("dreambench_humans", ["--bucket", "dreambench_humans"]),
]


def _run(cell_name: str, selection: list[str]) -> None:
    argv = [
        "uv", "run", "python", "-m", "experiments.attention_knockout.knockout_run",
        *selection, *COMMON_FLAGS,
    ]
    print(f"\n=== attention_knockout / {cell_name} ===")
    print(">>> " + " ".join(argv))
    result = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"[FAIL] cell={cell_name} returncode={result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    for cell_name, selection in CELLS:
        _run(cell_name, selection)
    print(f"\nDone. Ran {len(CELLS)} knockout cells.")


if __name__ == "__main__":
    main()
