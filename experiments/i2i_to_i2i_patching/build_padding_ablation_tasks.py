"""Build the padding-token dose-response task buckets.

The paper's categorical evidence says the *padding positions* of the fixed
512-token Qwen3 text sequence carry the transferred color/style
(``padding_only`` ~85% vs ``content_only`` ~0-3%). This ablation turns that
into a dose-response: hold ``text_seq_len`` fixed at 512 and vary how many of
the 512 slots are padding by varying the *prompt length*. Total joint sequence
length, RoPE range, noise and ref tokens are identical across conditions; only
the content/padding split moves.

Four nested levels (target padding counts): ``p494`` (the original short
instruction), ``p322``, ``p149``, ``p000`` (the full long-prompt filler,
~0 padding). Each level's instruction is a strict word-prefix of the next
longer one — the short instruction verbatim, then a word-calibrated prefix of
the neutral ``COLOR_FILLER`` / ``STYLE_FILLER`` (which never name the actual
color/style). Calibration binary-searches the largest filler word-prefix whose
chat-template token count fits the level's content budget, so realized padding
lands within ``PADDING_TOLERANCE`` of the target.

Composition (same objects/subjects at every level, so the dose-response is
within-pair):

* color — 8 objects x 3 colors x 5 seeds = 120 rows/level (the ``_lp`` recipe
  widened from 2 objects to all 8);
* style — 10 (subject, prompt_slug) combos x {style, manual-real} x 5 seeds
  = 100 rows/level (widened from 2 combos).

Outputs (committed artifacts; regenerate with this script):

* ``data/tasks/solid_color_padding/tasks.jsonl``  (480 rows, all levels)
* ``data/tasks/style_padding/tasks.jsonl``        (400 rows, all levels)

CPU-only: loads just the Qwen3 tokenizer (a few MB; gated repo — authenticate
once via ``huggingface-cli login`` or ``HF_TOKEN``), not the 9B model.

Usage::

    uv run python -m experiments.i2i_to_i2i_patching.build_padding_ablation_tasks
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.common.tasks import TASKS_ROOT, load_tasks
from experiments.i2i_to_i2i_patching.build_longprompt_tasks import (
    COLOR_FILLER,
    NUM_SEEDS,
    STYLE_FILLER,
    _long_instruction,
    _row,
)
from utils.flux2_klein import MODEL_ID, TEXT_SEQ_LEN

# ---------------------------------------------------------------------------
# Levels and composition
# ---------------------------------------------------------------------------

# Target padding-token counts. 494 = the original short color instruction
# (18 content tokens incl. chat-template scaffold); 0 = the full filler.
# 322/149 sit at the color filler's 5- and 9-sentence boundaries, evenly
# spreading the range.
PADDING_LEVELS: tuple[int, ...] = (494, 322, 149, 0)

# Realized padding may differ from the target: word-granularity calibration
# overshoots by at most a few tokens, the style short instructions are 25-32
# tokens (level 494 realizes 480-487), and the longest style fillers leave
# 1-5 padding at level 0.
PADDING_TOLERANCE = 32

# Color: all 8 paper objects (vs the _lp smoke set's 2), 3 colors, 5 seeds.
COLOR_OBJECTS = ("ball", "car", "chair", "couch", "hat", "mug", "pillow", "vase")
COLOR_NAMES = ("solid_red", "solid_green", "solid_blue")

# Style: 10 (subject, prompt_slug) combos — the 2 _lp combos plus 8 more, one
# slug per subject for maximal subject/style diversity. Each contributes a
# style-source row (real_ref_name=<subject>) and a manual real-target row
# (real_ref_name=<subject>_real) sharing the same long instruction.
STYLE_SPECS = (
    ("alice_in_wonderland", "tea_party"),
    ("dog", "frisbee"),
    ("capybara", "hotspring"),
    ("orange_cat", "windowsill_nap"),
    ("woman_runner", "park_stretch"),
    ("bald_man", "coffee_diner"),
    ("elephant", "savanna_sunset"),
    ("fox", "snow"),
    ("mouse", "cheese_wedge"),
    ("sparrow", "branch_perch"),
)


def level_slug(target_padding: int) -> str:
    return f"p{target_padding:03d}"


def _pa_id(task_id: str, slug: str) -> str:
    """Insert the level marker before the trailing ``_s<i>`` seed slot."""
    base, seed = task_id.rsplit("_s", 1)
    return f"{base}_{slug}_s{seed}"


# ---------------------------------------------------------------------------
# Tokenizer-calibrated instruction fitting
# ---------------------------------------------------------------------------


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")


def make_token_counter(tokenizer):
    """Chat-template token count for a prompt — the same path the diffusers
    Flux2 pipeline uses, *without* padding/truncation, so the count is the
    prompt's true content length. Memoized: calibration re-counts shared
    prefixes across levels and seeds."""
    cache: dict[str, int] = {}

    def ntok(prompt: str) -> int:
        if prompt not in cache:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            cache[prompt] = len(tokenizer(text)["input_ids"])
        return cache[prompt]

    return ntok


def fit_instruction(
    short: str, filler: str, target_padding: int, ntok,
) -> tuple[str, int]:
    """Longest instruction ``short [+ filler word-prefix]`` whose chat-template
    token count fits the content budget ``TEXT_SEQ_LEN - target_padding``.

    Word granularity keeps every level a strict prefix of the longer ones and
    lands within a couple of tokens of the budget (for the color filler the
    322/149 budgets fall exactly on sentence boundaries). Returns
    ``(instruction, realized_padding)``; asserts the realized padding is
    within ``PADDING_TOLERANCE`` of the target. The short instruction is
    always kept whole, even when it alone exceeds the budget (style level 494).
    """
    budget = TEXT_SEQ_LEN - target_padding
    words = filler.split()

    def instr(n: int) -> str:
        return short if n == 0 else _long_instruction(short, " ".join(words[:n]))

    # Binary search the largest n with ntok(instr(n)) <= budget (monotone).
    lo, hi = 0, len(words)
    if ntok(instr(0)) > budget:
        hi = 0  # short alone overshoots (style @ p494); tolerance check below
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ntok(instr(mid)) <= budget:
            lo = mid
        else:
            hi = mid - 1
    chosen = instr(lo)
    realized_padding = TEXT_SEQ_LEN - ntok(chosen)
    assert abs(realized_padding - target_padding) <= PADDING_TOLERANCE, (
        f"calibration missed: target padding {target_padding}, realized "
        f"{realized_padding} (content {ntok(chosen)}) for short={short!r}"
    )
    return chosen, realized_padding


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _pa_row(task, instruction: str, slug: str, target: int,
            realized_padding: int, metadata_extra: dict) -> dict:
    md_extra = dict(metadata_extra)
    md_extra.update(
        padding_level=slug,
        target_padding=target,
        actual_padding=realized_padding,
        actual_content=TEXT_SEQ_LEN - realized_padding,
    )
    row = _row(task, instruction, md_extra)
    row["task_id"] = _pa_id(task.task_id, slug)
    row["metadata"]["prompt_variant"] = "padding_ablation"
    return row


def build_color_rows(ntok) -> list[dict]:
    by_id = {t.task_id: t for t in load_tasks("solid_color")}
    lp_by_id = {t.task_id: t for t in load_tasks("solid_color_longprompt")}
    rows: list[dict] = []
    for obj in COLOR_OBJECTS:
        short = f"draw a {obj} in this color"
        for target in PADDING_LEVELS:
            slug = level_slug(target)
            instr, realized = fit_instruction(
                short, COLOR_FILLER.format(object=obj), target, ntok,
            )
            if target == 0:
                # The 0-padding level must reproduce the committed longprompt
                # instruction exactly (for the objects that bucket covers).
                lp = lp_by_id.get(f"solid_red_{obj}_lp_s0")
                assert lp is None or lp.instruction == instr, (
                    f"p000 instruction for {obj!r} diverges from the committed "
                    f"solid_color_longprompt bucket"
                )
            for color in COLOR_NAMES:
                for i in range(NUM_SEEDS):
                    src_id = f"{color}_{obj}_s{i}"
                    assert src_id in by_id, f"missing solid_color task: {src_id}"
                    rows.append(_pa_row(
                        by_id[src_id], instr, slug, target, realized,
                        {"object": obj},
                    ))
    return rows


def build_style_rows(ntok) -> list[dict]:
    style = {t.task_id: t for t in load_tasks("style")}
    manual = {t.task_id: t for t in load_tasks("manual")}
    lp_by_id = {t.task_id: t for t in load_tasks("style_longprompt")}
    rows: list[dict] = []
    for subject, pslug in STYLE_SPECS:
        src0 = style[f"customize_property_style_free_{subject}_{pslug}_s0"]
        subject_kind = src0.metadata["subject_kind"]
        for target in PADDING_LEVELS:
            slug = level_slug(target)
            instr, realized = fit_instruction(
                src0.instruction,
                STYLE_FILLER.format(subject_kind=subject_kind),
                target, ntok,
            )
            if target == 0:
                lp = lp_by_id.get(
                    f"customize_property_style_free_{subject}_{pslug}_lp_s0"
                )
                assert lp is None or lp.instruction == instr, (
                    f"p000 instruction for {subject}/{pslug} diverges from the "
                    f"committed style_longprompt bucket"
                )
            for i in range(NUM_SEEDS):
                src_id = f"customize_property_style_free_{subject}_{pslug}_s{i}"
                tgt_id = f"manual_free_{subject}_real_{pslug}_s{i}"
                assert src_id in style, f"missing style task: {src_id}"
                assert tgt_id in manual, f"missing manual task: {tgt_id}"
                rows.append(_pa_row(style[src_id], instr, slug, target, realized, {}))
                rows.append(_pa_row(manual[tgt_id], instr, slug, target, realized, {}))
    return rows


def _write(bucket: str, rows: list[dict]) -> None:
    out = TASKS_ROOT / bucket / "tasks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} tasks to {out}")


def main() -> None:
    ntok = make_token_counter(_load_tokenizer())
    _write("solid_color_padding", build_color_rows(ntok))
    _write("style_padding", build_style_rows(ntok))


if __name__ == "__main__":
    main()
