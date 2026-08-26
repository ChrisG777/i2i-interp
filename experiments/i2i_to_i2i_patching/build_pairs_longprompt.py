"""Emit the long-prompt smoke pairs for the padding-token ablation.

Mirrors the pairing logic of ``build_pairs_color.py`` (color) and
``build_pairs_style.py`` (style) but over the ``*_longprompt`` task buckets.
Same cyclic seed shift (source slot ``i`` -> target slot ``(i+1) % NUM_SEEDS``)
so the long-prompt pairs are structurally identical to the known-working short
pairs they mirror — only the instruction length differs.

Writes two pair files into ``experiments/i2i_to_i2i_patching/pairs/``:

* ``single9_4step_color_lp.txt``       — color cross-pairs (block 17)
* ``mm7_4step_style_to_real_lp.txt``   — style->real pairs (block 7)

Usage::

    uv run python -m experiments.i2i_to_i2i_patching.build_pairs_longprompt
"""

from __future__ import annotations

from pathlib import Path

from experiments.common.tasks import load_tasks
from experiments.i2i_to_i2i_patching.build_longprompt_tasks import (
    COLOR_NAMES,
    COLOR_OBJECTS,
    NUM_SEEDS,
    STYLE_SPECS,
)

SHIFT = 1
PAIRS_DIR = Path(__file__).resolve().parent / "pairs"


def build_color_pairs() -> list[tuple[str, str]]:
    have = {t.task_id for t in load_tasks("solid_color_longprompt")}
    pairs: list[tuple[str, str]] = []
    for obj in COLOR_OBJECTS:
        directed = [(a, b) for a in COLOR_NAMES for b in COLOR_NAMES if a != b]
        for idx, (a, b) in enumerate(directed):
            i = idx % NUM_SEEDS
            j = (i + SHIFT) % NUM_SEEDS
            src = f"{a}_{obj}_lp_s{i}"
            tgt = f"{b}_{obj}_lp_s{j}"
            assert src in have, f"missing source task: {src}"
            assert tgt in have, f"missing target task: {tgt}"
            pairs.append((src, tgt))
    return pairs


def build_style_pairs() -> list[tuple[str, str]]:
    have = {t.task_id for t in load_tasks("style_longprompt")}
    pairs: list[tuple[str, str]] = []
    for subject, slug in STYLE_SPECS:
        for i in range(NUM_SEEDS):
            j = (i + SHIFT) % NUM_SEEDS
            src = f"customize_property_style_free_{subject}_{slug}_lp_s{i}"
            tgt = f"manual_free_{subject}_real_{slug}_lp_s{j}"
            assert src in have, f"missing source task: {src}"
            assert tgt in have, f"missing target task: {tgt}"
            pairs.append((src, tgt))
    return pairs


def _write(name: str, pairs: list[tuple[str, str]]) -> None:
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    out = PAIRS_DIR / name
    with open(out, "w") as f:
        for src, tgt in pairs:
            f.write(f"{src}\t{tgt}\n")
    print(f"Wrote {len(pairs)} pairs to {out}")


def main() -> None:
    _write("single9_4step_color_lp.txt", build_color_pairs())
    _write("mm7_4step_style_to_real_lp.txt", build_style_pairs())


if __name__ == "__main__":
    main()
