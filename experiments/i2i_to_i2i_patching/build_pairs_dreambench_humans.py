"""Emit the 450 dreambench-humans cross-pairs for the i2i->i2i experiment.

Per the paper plan, ``i2i2i / mm7_4step_dreambench_humans`` runs every
directed (subject_A, subject_B) cross-pair within a shared-prompt slug
(10 active humans * 10*9 = 90 directed pairs per slug, 5 slugs = 450).

Each task in the ``dreambench_humans_shared`` bucket carries metadata
``shared_slug=<one of 5>`` and ``human_idx=<one of 10>``. We group by
``shared_slug`` and emit the directed cross-product of distinct
``human_idx`` values per group.

Cross-subject pairs automatically use different noise seeds (each subject
has its own seed, shared across that subject's 5 prompt variants); the
runner permits mismatched noise_seeds, no relax flag is required.

Usage::

    uv run python experiments/i2i_to_i2i_patching/build_pairs_dreambench_humans.py \\
        --out experiments/i2i_to_i2i_patching/pairs/mm7_4step_dreambench_humans.txt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from experiments.common.tasks import load_tasks

PROMPT_SLUGS = (
    "holding_coffee_kitchen",
    "laughing_dinner_party",
    "reading_park_bench",
    "standing_foggy_beach",
    "walking_city_street",
)
HUMAN_INDICES = (1, 2, 4, 7, 8, 10, 11, 15, 18, 19)


def build_pairs() -> list[tuple[str, str]]:
    tasks = load_tasks("dreambench_humans_shared")
    have = {t.task_id for t in tasks}

    # Group by (shared_slug, human_idx).
    by_slug: dict[str, dict[int, str]] = defaultdict(dict)
    for t in tasks:
        slug = t.metadata["shared_slug"]
        idx = t.metadata["human_idx"]
        assert idx not in by_slug[slug], (
            f"duplicate task for slug={slug} human_idx={idx}: "
            f"{by_slug[slug][idx]} vs {t.task_id}"
        )
        by_slug[slug][idx] = t.task_id

    # Validate that every expected (slug, human_idx) is present.
    expected_slugs = set(PROMPT_SLUGS)
    assert set(by_slug.keys()) == expected_slugs, (
        f"slug mismatch: expected {expected_slugs}, got {set(by_slug.keys())}"
    )
    for slug in PROMPT_SLUGS:
        assert set(by_slug[slug].keys()) == set(HUMAN_INDICES), (
            f"slug {slug!r}: expected human_idx={set(HUMAN_INDICES)}, "
            f"got {set(by_slug[slug].keys())}"
        )

    pairs: list[tuple[str, str]] = []
    for slug in PROMPT_SLUGS:
        for a in HUMAN_INDICES:
            for b in HUMAN_INDICES:
                if a == b:
                    continue
                src = by_slug[slug][a]
                tgt = by_slug[slug][b]
                assert src in have, f"missing source task: {src}"
                assert tgt in have, f"missing target task: {tgt}"
                pairs.append((src, tgt))
    assert len(pairs) == 450, f"expected 450 pairs, got {len(pairs)}"
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
