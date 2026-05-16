# i2i → unconditional activation patching (T2I Lens)

Patches activations from an image-to-image run into an unconditional (empty-prompt) text-to-image run, sweeping one token category at a time across all 32 blocks. The unconditional t2i acts as a neutral "lens" revealing what information the i2i tokens carry after each block.

## Categories

Target is t2i, so `ref` is not patched. Slice boundaries come from the per-task `TokenLayout` (see [Patching framework](../README.md#patching-framework)).

| Category | MM block slice           | Single block slice                  |
|----------|--------------------------|-------------------------------------|
| `image`  | `img_stream` (all noise) | `hidden[:, text_end:image_end, :]`  |
| `text`   | `txt_stream`             | `hidden[:, :text_end, :]`           |

## Sweep flags

| Flag                          | Value                | Behavior                                                                                                                                                  |
|-------------------------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--sweep-mode`                | `diagonal` (default) | Source block K patched into target block K's **output**. Standard lens sweep.                                                                             |
| `--sweep-mode`                | `input_to_block0`    | Source block K patched as the **input** to `transformer_blocks.0` via `transformer.context_embedder`. Text-only; only mode that supports multi-step inference. |
| `--text-token-mode`           | `all` (default)      | Replace the full text slice in one sweep.                                                                                                                 |
| `--text-token-mode`           | `per_content`        | One grid per content token in the instruction; replaces only that single position. Empty-instruction tasks auto-skip.                                     |
| `--text-token-mode`           | `per_position`       | One grid per *every* position 0..511. Always pair with `--block-range` and/or `--position-range`.                                                          |
| `--text-token-mode`           | `content_vs_padding` | Two grids per task: ALL content tokens patched together, and ALL padding tokens patched together.                                                          |
| `--text-token-indices N…`     | —                    | Explicit text-token positions (0..511); overrides `--text-token-mode`.                                                                                     |
| `--block-range FIRST LAST`    | —                    | Limit the block sweep to `[FIRST, LAST]` inclusive.                                                                                                       |
| `--patched-inference-steps N` | —                    | Multi-step patched generation. Source i2i capture stays at one step. Only valid with `--sweep-mode input_to_block0`. Pair with `--block-range 0 15` to keep runtime manageable. |

## Knockout coupling

`--knockout-setting NAME [NAME ...] --knockout-side {source,target,both}` installs an attention-knockout mask alongside the patching sweep:

* `source` — mask installed during i2i capture only.
* `target` — mask installed during t2i sweep only (also writes a no-patch KO-baseline bookend).
* `both` — mask installed in both passes.

Multiple settings can be passed; the runner loops settings around tasks, so all scenes complete for setting A before setting B starts.

## Output layout

```
results_v4/i2i_to_unconditional/<edit_type>/<task_id>/<run_timestamp>/
    task_metadata.json
    reference.png
    source_i2i.png
    unconditional_baseline.png
    t2i_clean.png
    <sweep_mode>/                            # diagonal / input_to_block0
        no_knockout/
            image_tokens/
                patched_mm0_to_mm0.png ... patched_single23_to_single23.png
                grid.png
            text_tokens/                     # --text-token-mode all
                grid.png
            text/                            # per_content / per_position / indices / content_vs_padding
                token_018_add/grid.png       # or all_content/grid.png, all_padding/grid.png
                ...
        knockout_image_to_text_source/
            ...same shape as no_knockout/
        knockout_image_to_text_target/
            unconditional_baseline_with_ko.png
            ...
```

Each grid: `[Source (i2i), Unconditional baseline, t2i clean, <one cell per source block>]`, plus `Source (no KO)` / `KO baseline (...)` cells when knockout flags are active.

## Usage

```bash
# Single task, default diagonal sweep, all text + image categories
uv run python experiments/i2i_to_unconditional/i2i_to_unconditional_patch.py \
    --task-id solid_red_couch

# Per-content-token grids
uv run python experiments/i2i_to_unconditional/i2i_to_unconditional_patch.py \
    --task-id solid_red_couch \
    --categories text --text-token-mode per_content

# Input-to-block-0, 4-step inference, first 16 source blocks
uv run python experiments/i2i_to_unconditional/i2i_to_unconditional_patch.py \
    --task-id solid_red_couch --categories text \
    --sweep-mode input_to_block0 \
    --block-range 0 15 --patched-inference-steps 4

# Source-side knockout coupling
uv run python experiments/i2i_to_unconditional/i2i_to_unconditional_patch.py \
    --task-id solid_red_couch \
    --knockout-setting 'image->text' --knockout-side source

# Whole bucket
uv run python experiments/i2i_to_unconditional/i2i_to_unconditional_patch.py \
    --edit-type customize --limit 50
```

[`dump_content_vs_padding_tokens.py`](dump_content_vs_padding_tokens.py) loads only the Qwen3 tokenizer (no diffusion model) and writes a per-task text report listing `all_content` vs. `all_padding` positions — useful before kicking off a `content_vs_padding` sweep.

For the paper-scale sweep across every task family, run [`scripts/reproduce_t2i_lens.py`](../../scripts/reproduce_t2i_lens.py). See [experiments/README.md](../README.md#run--grade--tabulate) for the run → grade → tabulate workflow.
