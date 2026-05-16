"""Emit the 448 color cross-pairs for the i2i->i2i color experiment.

Per the paper plan §1, ``i2i2i / single9_4step_color`` runs every directed
(color_A, color_B) cross-pair within an object (8 objects * 8*7 = 448).

Within each pair, source and target use *different* noise-seed slots via
cyclic shift ``s_i -> s_{(i+1) % NUM_SEEDS}`` (mirroring
``build_pairs_style.py``). The source slot index ``i`` rotates as
``idx % NUM_SEEDS`` across the 56 directed pairs per object, so all 5
materialized seeds get exercised on both source and target sides.

Usage::

    uv run python experiments/i2i_to_i2i_patching/build_pairs_color.py \\
        --out slurm/i2i_to_i2i_patching/single9_4step_color/pairs.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common.tasks import load_tasks

NUM_SEEDS = 5
SHIFT = 1  # cyclic shift between source seed and target seed
COLORS = [
    "solid_blue", "solid_brown", "solid_green", "solid_orange",
    "solid_pink", "solid_purple", "solid_red", "solid_yellow",
]
OBJECTS = ["ball", "car", "chair", "couch", "hat", "mug", "pillow", "vase"]


def build_pairs() -> list[tuple[str, str]]:
    # Validate by loading the solid_color bucket and confirming every expected
    # task_id exists. Caller relies on this check rather than re-parsing jsonl.
    have = {t.task_id for t in load_tasks("solid_color")}
    pairs: list[tuple[str, str]] = []
    for obj in OBJECTS:
        directed = [(a, b) for a in COLORS for b in COLORS if a != b]
        # Within each pair: source seed slot ``i`` rotates as ``idx % NUM_SEEDS``
        # across the 56 directed pairs; target seed slot is ``(i + SHIFT) %
        # NUM_SEEDS`` so source and target use distinct seeds.
        for idx, (a, b) in enumerate(directed):
            i = idx % NUM_SEEDS
            j = (i + SHIFT) % NUM_SEEDS
            src = f"{a}_{obj}_s{i}"
            tgt = f"{b}_{obj}_s{j}"
            assert src in have, f"missing source task: {src}"
            assert tgt in have, f"missing target task: {tgt}"
            pairs.append((src, tgt))
    assert len(pairs) == 448, f"expected 448 pairs, got {len(pairs)}"
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, metavar="PATH")
    args = ap.parse_args()
    pairs = build_pairs()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for src, tgt in pairs:
            f.write(f"{src}\t{tgt}\n")
    print(f"Wrote {len(pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()
