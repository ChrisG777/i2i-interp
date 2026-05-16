"""Emit style->real pairs for the i2i->i2i style-transfer experiment.

``i2i2i / mm7_4step_style_to_real`` runs 5 cyclic-shift pairs per
(subject, prompt): for i in 0..4, pair the style-source seed i with the
real-target seed (i+1)%5. With 18 subjects × 5 prompts × 5 seeds, that's
450 pairs.

Pairing is keyed on (subject, instruction) rather than reconstructed task
IDs, so any tweak to the task_id namespace (e.g. ``_free_`` batch marker)
flows through automatically as long as the customize and manual rows
share their instruction text and real_ref_name root.

Usage::

    uv run python experiments/i2i_to_i2i_patching/build_pairs_style.py \\
        --out slurm/i2i_to_i2i_patching/mm7_4step_style_to_real/pairs.txt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from experiments.common.tasks import load_tasks

NUM_SEEDS = 5
SHIFT = 1  # cyclic shift between source seed and target seed


def _seed_idx(task_id: str) -> int:
    """Trailing _s<i> in the task_id encodes the noise-seed slot."""
    suffix = task_id.rsplit("_s", 1)[-1]
    return int(suffix)


def build_pairs() -> list[tuple[str, str]]:
    style_tasks = load_tasks("style")
    manual = load_tasks("manual")

    # Group style by (subject, instruction). subject lives in real_ref_name
    # (e.g. "bear") and the instruction text disambiguates the prompt. Each
    # group should have exactly NUM_SEEDS entries (one per noise seed).
    style_groups: dict[tuple[str, str], list] = defaultdict(list)
    for t in style_tasks:
        style_groups[(t.real_ref_name, t.instruction)].append(t)

    # Group manual real entries by (subject_real_name, instruction). Keys
    # are <subj>_real (e.g. "bear_real") so we lookup by appending "_real"
    # to the style subject.
    real_groups: dict[tuple[str, str], list] = defaultdict(list)
    for t in manual:
        if not (t.real_ref_name or "").endswith("_real"):
            continue
        real_groups[(t.real_ref_name, t.instruction)].append(t)

    pairs: list[tuple[str, str]] = []
    missing: list[str] = []
    for (subject, instr), src_tasks in sorted(style_groups.items()):
        real_key = (f"{subject}_real", instr)
        tgt_tasks = real_groups.get(real_key, [])
        if len(src_tasks) != NUM_SEEDS or len(tgt_tasks) != NUM_SEEDS:
            missing.append(
                f"  (subject={subject}, instr={instr[:60]!r}): "
                f"style={len(src_tasks)}/{NUM_SEEDS}, "
                f"real={len(tgt_tasks)}/{NUM_SEEDS}"
            )
            continue
        src_by_seed = {_seed_idx(t.task_id): t for t in src_tasks}
        tgt_by_seed = {_seed_idx(t.task_id): t for t in tgt_tasks}
        for i in range(NUM_SEEDS):
            j = (i + SHIFT) % NUM_SEEDS
            pairs.append((src_by_seed[i].task_id, tgt_by_seed[j].task_id))

    if missing:
        msg = "Could not find matching 5-seed groups for some style prompts:\n"
        msg += "\n".join(missing)
        raise SystemExit(msg)
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
