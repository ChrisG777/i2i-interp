# Attention knockout

Measures the causal role of cross-modal attention in FLUX.2-klein i2i generations by setting the additive attention mask to `-inf` on chosen row/column quadrants, severing information flow from one token category to another on selected transformer blocks. Information flows keys → queries in attention, so a setting `A->B` makes queries in B unable to attend to keys in A. Masks are applied **inside SDPA before softmax** via trivial `attn_processor` subclasses (`KnockoutFlux2AttnProcessor`, `KnockoutFlux2ParallelSelfAttnProcessor`) that forward `attention_mask=self._mask` unchanged — with `_mask=None` they are bytewise identical to the stock diffusers processors (unit-tested with `torch.equal`).

## Knockout settings

Each setting blocks information flow from `sources` to `destination`. Select via `--settings`; the default runs every known setting.

Cross-modal pair knockouts:

| name          | from → to    |
|---------------|--------------|
| `text->ref`   | text → ref   |
| `image->ref`  | image → ref  |
| `ref->text`   | ref → text   |
| `image->text` | image → text |
| `ref->image`  | ref → image  |
| `text->image` | text → image |

Group knockouts:

| name              | from → to          |
|-------------------|--------------------|
| `image+text->ref` | image + text → ref |
| `image+ref->text` | image + ref → text |

Composite (OR-union of two atomic settings in one mask):

| name           | severs                                  |
|----------------|-----------------------------------------|
| `image<->ref`  | both `ref->image` and `image->ref`      |

Text-subset settings (resolved per task via the Qwen3 tokenizer; padding = complement of the instruction content tokens within the 512-token sequence):

| name                        | severs                                                    |
|-----------------------------|-----------------------------------------------------------|
| `text[padding]+ref->image`  | `text[padding]->image` ⊕ `ref->image` in one mask         |
| `ref->text[padding]`        | text-padding queries cannot attend to ref keys            |
| `ref->text[content]`        | text-content queries cannot attend to ref keys            |

The `ref->text[padding]` / `ref->text[content]` pair isolates which slice of the text sequence is the load-bearing target for ref → text influence — any surviving effect must route through the unmasked slice.

Atomic builders in [`masks.py`](masks.py): `build_knockout_mask` (category-level), `build_subset_knockout_mask` (subset-of-category source), `build_destination_subset_knockout_mask` (subset-of-category destination). `combine_masks(*masks)` OR-unions any number of additive masks via elementwise minimum and runs a softmax-row safety check. Adding a new atomic category setting is a one-line change; category-only composites take one entry in `COMPOSITE_KNOCKOUT_SETTINGS`.

## Layer-mode sweep

For each setting we sweep `L` over `ALL_BLOCK_NAMES` (8 MM + 24 single = 32 blocks) in one of four modes via `--layer-mode`:

| Mode         | Blocks masked at step `L` | `L` range  | Notes                                                                          |
|--------------|---------------------------|------------|--------------------------------------------------------------------------------|
| `suffix`     | `[L..32)`                 | `0..31`    | `L=0` masks every block; `L=31` masks only the last                            |
| `prefix`     | `[0..L]` (inclusive)      | `0..31`    | `L=0` masks only block 0; `L=31` masks every block                             |
| `individual` | `{L}`                     | `0..31`    | Mask a single block                                                            |
| `window`     | `[L..L+k)`                | `0..32-k`  | Sliding window of `k` consecutive blocks (`--window-size`, default `3`)        |

`apply_mask_to_layers` explicitly sets `_mask=None` on blocks outside the selected set every iteration — masks never leak. Multiple modes can be passed in one invocation; the model loads once and each mode writes its own subdirectory.

`--all-layers-4step` adds one extra cell per (task, setting) grid: a single 4-step generation with the mask installed on every block. Lands as `<setting>/full_ko_4step.png` and as a trailing cell appended to every layer-mode grid — an asymptote check against the 1-step full-KO bookend.

## Split-schedule knockout

Instead of one mask swept across block ranges, split-schedule knocks out one set of attention pathways on the prefix blocks `[0, cutoff)` and a different set on the suffix blocks `[cutoff, 32)`, in a single generation. This isolates *where in depth* a pathway matters — e.g. block `ref->image` only in the prefix while severing the reference entirely in the suffix.

| flag               | what it does                                                          | required |
|--------------------|------------------------------------------------------------------------|----------|
| `--split-block`    | the cutoff block — first block of the suffix; pass several to sweep multiple cutoffs | yes |
| `--suffix-setting` | attention pathway(s) to knock out on the suffix blocks `[cutoff, 32)`  | yes |
| `--prefix-setting` | attention pathway(s) to knock out on the prefix blocks `[0, cutoff)`   | no — omit to knock out only the suffix |

Passing multiple names to `--prefix-setting` / `--suffix-setting` knocks out all of them together. Split-schedule replaces the per-block layer sweep, so `--settings` / `--layer-mode` / `--window-size` are ignored. It requires `--full-ko-only`, `--num-inference-steps 4`, and `--results-subdir` (flat layout). Settings must be full-category — text-subset settings (`ref->text[content]` etc.) are rejected. `apply_split_mask_to_layers` ([`masks.py`](masks.py)) installs both masks in one pass; each cutoff produces `split_at_<cutoff_block>_full_ko.png`.

## Output layout

Default (legacy) layout — one timestamped dir per run, nested per setting and layer mode:

```
results/attention_knockout/<edit_type>/<task_id>/<run_timestamp>/
    task_metadata.json
    reference.png
    i2i_baseline.png
    t2i_clean.png
    text_padding_indices.json     # only when text[padding]+ref->image is selected
    <setting_name>/
        full_ko.png
        full_ko_4step.png         # only when --all-layers-4step
        <layer_mode>/
            L00.png ... L31.png   # per-L raw cells
            grid.png              # bookends + sweep + (optional) 4-step cell
        window/k=<k>/
            L00.png ... grid.png
```

Flat layout (`--results-subdir NAME`, used by the paper-scale runs and the demo notebook) — no `<edit_type>`, no timestamp; the per-setting subdir is dropped and the setting/cutoff is folded into the filename. Requires `--full-ko-only --num-inference-steps 4`:

```
results_v4/attention_knockout/<NAME>/<task_id>/
    task_metadata.json
    reference.png
    i2i_baseline_4step.png
    t2i_clean_4step.png
    text_padding_indices.json               # only when text[padding]+ref->image is selected
    <setting_name>_full_ko.png              # standard --settings mode, one per setting
    split_at_<cutoff_block>_full_ko.png     # split-schedule mode, one per cutoff
```

Setting names are filesystem-sanitised (`ref->text` → `ref_to_text`, `image<->ref` → `image_bidir_ref`).

## Usage

```bash
# Single task, default prefix sweep, all known settings
uv run python experiments/attention_knockout/knockout_run.py \
    --task-id solid_red_couch

# One task per edit-type, single model load (~20 min on H200)
uv run python experiments/attention_knockout/knockout_run.py \
    --edit-type add remove customize --limit 1 \
    --settings 'ref->image' --layer-mode prefix

# Bidirectional ref<->image; single-quote anything containing > or <-
uv run python experiments/attention_knockout/knockout_run.py \
    --task-id solid_red_couch --settings 'image<->ref'

# Sliding window of 3 consecutive blocks
uv run python experiments/attention_knockout/knockout_run.py \
    --task-id solid_red_couch --layer-mode window --window-size 3

# Trailing 4-step full-KO cell on every grid
uv run python experiments/attention_knockout/knockout_run.py \
    --task-id solid_red_couch --settings 'image->text' --all-layers-4step

# Split schedule: ref->image masked only in the prefix, the reference severed
# entirely in the suffix; sweep two cutoffs in one run
uv run python experiments/attention_knockout/knockout_run.py \
    --task-id solid_red_couch \
    --split-block 'single_transformer_blocks.2' 'single_transformer_blocks.10' \
    --prefix-setting 'ref->image' \
    --suffix-setting 'image<->ref' 'ref->text' 'text->ref' \
    --full-ko-only --num-inference-steps 4 \
    --results-subdir split_schedule
```

For the paper-scale sweep across every task family, run [`scripts/reproduce_attention_knockout.py`](../../scripts/reproduce_attention_knockout.py). See [experiments/README.md](../README.md#run--grade--tabulate) for the run → grade → tabulate workflow that pairs the three `ref->text` siblings with their VLM judges.

## Tests

```bash
uv run python -m pytest experiments/attention_knockout/tests/ -v
```

Covers `KnockoutSetting` validation, mask shape/dtype + quadrant correctness, no fully-masked query rows, `masked_indices` boundary cases, `apply_mask_to_layers` set/clear semantics for every layer mode, the `image<->ref` composite, and stock-vs-knockout processor parity (`torch.equal` with `_mask=None`).
