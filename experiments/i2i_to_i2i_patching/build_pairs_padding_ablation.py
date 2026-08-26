"""Emit the padding-ablation pair files, one per (family, level).

Mirrors the pairing logic of ``build_pairs_longprompt.py`` (cyclic seed shift
``s_i -> s_{(i+1) % NUM_SEEDS}``) over the ``solid_color_padding`` /
``style_padding`` buckets built by ``build_padding_ablation_tasks.py``. Pair
composition is identical across levels — same objects, colors, subjects and
seed slots — so the dose-response is within-pair.

Per level: color = 8 objects x 6 directed (color_a, color_b) = 48 pairs;
style = 10 (subject, prompt_slug) combos x 5 seeds = 50 pairs.

Writes into ``experiments/i2i_to_i2i_patching/pairs/`` (committed):

* ``single9_4step_color_p<NNN>.txt``
* ``mm7_4step_style_to_real_p<NNN>.txt``

Usage::

    uv run python -m experiments.i2i_to_i2i_patching.build_pairs_padding_ablation
"""

from __future__ import annotations

from experiments.common.tasks import load_tasks
from experiments.i2i_to_i2i_patching.build_longprompt_tasks import NUM_SEEDS
from experiments.i2i_to_i2i_patching.build_padding_ablation_tasks import (
    COLOR_NAMES,
    COLOR_OBJECTS,
    PADDING_LEVELS,
    STYLE_SPECS,
    level_slug,
)
from experiments.i2i_to_i2i_patching.build_pairs_longprompt import SHIFT, _write


def build_color_pairs(slug: str) -> list[tuple[str, str]]:
    have = {t.task_id for t in load_tasks("solid_color_padding")}
    pairs: list[tuple[str, str]] = []
    for obj in COLOR_OBJECTS:
        directed = [(a, b) for a in COLOR_NAMES for b in COLOR_NAMES if a != b]
        for idx, (a, b) in enumerate(directed):
            i = idx % NUM_SEEDS
            j = (i + SHIFT) % NUM_SEEDS
            src = f"{a}_{obj}_{slug}_s{i}"
            tgt = f"{b}_{obj}_{slug}_s{j}"
            assert src in have, f"missing source task: {src}"
            assert tgt in have, f"missing target task: {tgt}"
            pairs.append((src, tgt))
    return pairs


def build_style_pairs(slug: str) -> list[tuple[str, str]]:
    have = {t.task_id for t in load_tasks("style_padding")}
    pairs: list[tuple[str, str]] = []
    for subject, pslug in STYLE_SPECS:
        for i in range(NUM_SEEDS):
            j = (i + SHIFT) % NUM_SEEDS
            src = f"customize_property_style_free_{subject}_{pslug}_{slug}_s{i}"
            tgt = f"manual_free_{subject}_real_{pslug}_{slug}_s{j}"
            assert src in have, f"missing source task: {src}"
            assert tgt in have, f"missing target task: {tgt}"
            pairs.append((src, tgt))
    return pairs


def main() -> None:
    for target in PADDING_LEVELS:
        slug = level_slug(target)
        _write(f"single9_4step_color_{slug}.txt", build_color_pairs(slug))
        _write(f"mm7_4step_style_to_real_{slug}.txt", build_style_pairs(slug))


if __name__ == "__main__":
    main()
