# Human validation of the VLM judges

Checks the Claude/GPT judge verdicts against crowdworkers on a reference-diverse
subset of the Attention Knockout and I2I-to-I2I Patching cells, for the three
families whose judge questions are subjective enough to be worth a human check:
**solid color transfer**, **style transfer**, and **human identity transfer**.

**[SETUP.md](SETUP.md) is the operational runbook** — accounts, S3 hosting,
building the six MTurk projects, publishing, reading the results back, and every
gotcha we hit. This file covers the design: what is measured, what is sampled,
and why.

## Build it

```bash
uv run python -m scripts.human_eval.select_subset \
    --base-url https://<your-bucket>.s3.amazonaws.com/images
```

Writes to `results_v4/human_eval/`:

| File | In git |
|---|---|
| `manifest.csv` — one row per item; the join key for everything downstream | yes |
| `mturk_input_<group>.csv` × 6 — HIT input data | yes |
| `hit_<group>.html` × 6 — the matching layouts | yes |
| `images/` — 534 web-sized JPEGs (~26 MB) | no, regenerated |

`<group>` is one of `ko_color`, `ko_style`, `ko_human`, `i2i2i_color`,
`i2i2i_style`, `i2i2i_human` — **six separate MTurk projects**. Splitting by
family is what lets each layout name the one property being judged instead of
hedging across all three.

Three counts that are easy to confuse:

| | count | what it is |
|---|---|---|
| **tasks** | 120 | paper cells — a `task_id` (knockout) or a `<src>__<tgt>` pair |
| **items** | 294 | questions put to a worker — one per task per *arm* |
| **images** | 534 | exported JPEGs — 5 per knockout task, 4 per patching pair, shared across that task's arms |

Each item shows 3 images at once, the same way a judge bundle does.
294 items × 3 workers = **882 judgements**, 17 HITs, **$61.20**.

`--per-hit` is a ceiling, not a fixed size: a HIT can hold at most one item per
distinct cell, so groups whose arms share a small task pool shrink. The emitted
HTML always matches the CSV its group actually used.

## What gets sampled

15 conditions:

| group | tasks | arms | items |
|---|---|---|---|
| `ko_color` | 16 | `ref_to_text`, `ref_to_image`, `check` | 48 |
| `ko_style` | 18 | same | 54 |
| `ko_human` | 20 | same | 60 |
| `i2i2i_color` | 24 pairs | `patched`, `check` | 48 |
| `i2i2i_style` | 18 pairs | same | 36 |
| `i2i2i_human` | 24 pairs | same | 48 |
| `ko_color_pair` | 16 | `pair`, `paircheck` | 22 |
| `ko_style_pair` | 18 | same | 24 |
| `ko_human_pair` | 19 | same | 25 |

The three `_pair` groups are the redesigned `ref→image` arms; see below. They
supersede `ko_*_ref_to_image`, whose rows stay in the manifest so the round-2
results remain analysable.

**Every arm of a group grades the identical task list.** `ko_color_ref_to_text`,
`ko_color_ref_to_image` and `ko_color_check` are the same 16 `task_id`s, so the
arms are a within-task contrast sharing one reference and one baseline image,
not independent samples. That is what makes the `ref→image` result readable at
all — see the note on "Both" below.

Selection is a round-robin over **distinct reference images**, then distinct
prompts, and only then repeated seeds; a random draw of 18 style cells would
miss several of the 18 style references. All 8 colors, 18 style refs and 10
humans are covered in every condition, as both source and target. No condition
repeats a seed.

Seeds are sha256-derived ints, never `hash()` of a string, so the selection
reproduces across machines and processes.

**`ko_human` samples only the explicitly photographic prompts.** Three of the
nine `dreambench_humans` prompts per subject ask for a watercolour, an anime
illustration or a retro drawing, and two more name no medium. Identity is a
photo-to-photo judgement, so the filter keeps instructions starting `A photo` /
`A photograph`, leaving a 40-task universe.

## Question design

The judge prompts in `scripts/judge/bundles.py` are written for a VLM and are
too technical for crowdworkers, so each is reformulated. **Both methods use one
layout and one question per family**, mirroring the judges — each judge also
asks a comparative question against a clean baseline.

A reference on top, two outputs below in randomised left/right order, four
options: **Left / Right / Both / Neither**.

| arm | top | the two below | judge `pass=1` ⟺ worker answers |
|---|---|---|---|
| `ko_*_ref_to_text` | task reference | baseline, `ref_to_text_full_ko` | **baseline** |
| `ko_*_ref_to_image` | task reference | baseline, `ref_to_image_full_ko` | **Both** |
| `ko_*_check` | task reference | baseline, `t2i_clean_4step` | **baseline** *(known answer)* |
| `i2i2i_*_patched` | **source** reference | target baseline, `patched` | **intervention** |
| `i2i2i_*_check` | **source** reference | target baseline, `source_i2i_4step` | **intervention** *(known answer)* |

For patching, the judge's Image 2 (the target reference) is dropped — it exists
only to say what the baseline should look like, and the worker sees the baseline
itself.

**"Both" and "Neither" are load-bearing.** The `ref→image` prediction is that the
property *survives*, i.e. both lower images carry it. A plain Left/Right choice
could only express that as a chance split, which is indistinguishable from both
having *lost* it. With four options every arm gets an item-level mapping.

### Paired foils for `ref→image`

Round 2 put the two `ref→image` numbers at 56–59%, and the disagreements were
not noise: humans and the judge agreed wherever an item was determinate and
split only on where to put a threshold on a continuum. The arm shows the
knockout output beside its own baseline, and when the property survives *both*
carry it, so the worker is asked to grade a matter of degree.

The fix is to make exactly one of the two images carry the property:

| the knockout output | is shown against | so the answer is |
|---|---|---|
| **kept** the property | one that lacks it — a different reference colour, a generation with no reference at all, a different person | the knockout output |
| **lost** the property | one that has it — the task's own baseline | the other one |

Either way the question is a forced binary choice with a determinate answer,
and "Both" / "Neither" disappear.

Which way to build each pair depends on whether the property actually survived,
so the 54 `ref→image` outputs are **hand-labelled**:

```bash
uv run python -m scripts.human_eval.label_ui      # then open the printed URL
```

The label cannot come from Claude. Claude is the judge under test, so a
Claude-derived label would agree with the judge by construction, and every item
Claude got wrong would become a coin flip rather than a visible disagreement —
which raises the measured agreement rate instead of exposing the error. The page
and its ~300 JPEGs are regenerated by the command above and not tracked; the
labels themselves land in `ref_to_image_labels.csv`, which is.

Note what the label does **not** do: it never enters the scoring. The readout is
just *did the worker pick the knockout output* — that is the human's verdict on
whether the property survived, and it is compared directly to the judge's. The
label only decides which foil is shown. Its one failure mode is a mislabel,
which puts two same-status images side by side and reduces that item to a coin
flip; `ref_to_image_labels.csv` records the label next to `vlm_pass` so the
items where the two differ stay visible.

The paired arms are **three separate projects** — `ko_color_pair`,
`ko_style_pair`, `ko_human_pair` — because a two-option layout cannot share a
HIT with the four-option one. The original six are untouched and their round-2
results stand; only `ref→image` is re-collected.

| condition | items | the two below | pass ⟺ worker picks |
|---|---|---|---|
| `ko_*_pair` | 16 / 18 / 19 | knockout output, foil | **the knockout output** |
| `ko_*_paircheck` | 6 each | baseline, non-carrier foil | **the baseline** *(known)* |

`_paircheck` reuses the pairing machinery with an answer fixed by construction:
the task's own baseline against a foil that cannot carry the property. It is
restricted to cells whose labelled foil is a non-carrier — for a cell labelled
"lost" the foil *is* the baseline, which would pit an image against itself.

Two things the paired arms do that the originals do not. **Positions are exactly
balanced** rather than an independent coin per item: on 53 items a coin lands
35/18 often enough to matter, and a side imbalance multiplies straight into any
worker-side position bias. And each item's passing answer is written to the
manifest as `target_role` — every sign error in this pipeline has come from a
condition name whose meaning changed without the name changing, so the analysis
now reads the answer instead of re-deriving it, falling back to the old
name-based rule only for manifests written before the column existed.

Six of the original 54 cells were unlabellable and were replaced from the same
reference image (`label_ui.py` picks them; `replaces` records the substitution).
One human cell — `human_02_man` — lost both its original and its replacement,
so that arm carries 19 rather than 20.

### Attention checks

Both knockout directions are genuine experimental questions — their answers are
the result, so neither can score a worker. Each method therefore carries a third
arm whose correct answer is fixed by construction:

* **knockout** — `t2i_clean_4step.png` is the same prompt generated with **no
  reference at all**, so it cannot carry the reference property and the baseline
  is correct. Verified for color: across the selected tasks the clean
  generation's dominant colour is 97–199 RGB away from its reference swatch,
  never close enough to be ambiguous. For the DreamBench humans this file only
  exists under the T2I-lens results root, so the check arm sources that one slot
  from there (`Condition.slot_base`).
* **patching** — `source_i2i_4step.png` is the source pass's own output, so it
  always carries the source property and the intervention side is correct.
* **paired knockout** (`ko_*_paircheck`): the task's own baseline against a foil
  the hand label marked a non-carrier, so the baseline is correct. Restricted to
  cells whose foil is a non-carrier, since for a cell labelled "lost" the foil
  *is* the baseline and the item would pit an image against itself.

These put their known-good image on **different sides**, so a rule keyed on the
suffix alone would score some of them exactly backwards. Each item therefore
states its own answer in the manifest's `target_role` column and the analysis
reads it, falling back to a per-method rule only for manifests written before
that column existed.

Check rows carry a blank `vlm_pass`: no judge was ever shown those images. The
analyzer recognises them by suffix (`CHECK_SUFFIXES`, currently `_check` and
`_paircheck`); a check arm whose name matches neither is silently treated as a
real item, which disables screening for its whole batch.

### Presentation

Nothing worker-facing says "reference", "baseline", "patched", "knockout" or
"model" — each layout names a concrete thing at a position on the page and asks
about one property. Grep the generated HTML for those words; the count must stay
zero.

Options are native `<input type="radio">`, **not** `crowd-radio-button`: the AWS
docs require crowd-radio siblings to have *different* `name` values for mutual
exclusion, so the natural same-name idiom lets several be selected at once, and
its output shape is a boolean per option rather than a single value.

The true role behind each display slot is recorded in `manifest.csv`
(`role_left`, `role_right`), so left/right bias is measurable rather than
assumed away.

Within a HIT the arms are **balanced** and the question order is **shuffled**.
Balance alone leaves a strict A-B-A-B run, which is learnable: a worker who
picks up the rhythm can settle into an alternating response habit that would
correlate perfectly with condition. No HIT ever repeats an underlying cell.

## Analyse

```bash
uv run python -m scripts.human_eval.analyze_results --results results_v4/human_eval/
```

Every `*.csv` in that directory is read and the ones with no
`Answer.taskAnswers` column are skipped, so the nine committed batch downloads
are picked up and the manifest / HIT inputs / pilot are not.

Reports the **agreement rate** per condition / family / method / overall, always
with numerator and denominator visible, with 95% CIs from a bootstrap that
resamples **workers** (the unit of independence), plus worker screening on the
check arms (`_check` and `_paircheck`, see `CHECK_SUFFIXES`), position-bias and
inter-rater diagnostics, and `disagreements.csv`, every item where humans and
the judge differ, with image names for eyeballing.

Collected results and the agreement tables live in
[`results_v4/human_eval/README.md`](../../results_v4/human_eval/README.md).

No Cohen's kappa. The quantity of interest is the raw rate at which humans and
the judge land on the same verdict on these items, not a coefficient whose
chance baseline depends on a marginal the intervention itself moves.

## Human subjects

Collecting from human participants needs MIT COUHES review — usually exempt, but
the determination is theirs — before the production batch. Sandbox testing does
not; no real workers are involved.
