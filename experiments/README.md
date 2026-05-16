# Experiments

Three causal-intervention experiments — `attention_knockout`, `i2i_to_unconditional`, `i2i_to_i2i_patching` — share one infrastructure layer (`common/` task loader + runner ABC + CLI; `patching/` activation-patching hooks + sweep). Each experiment is a runner subclass plus a thin entry script.

```
experiments/
├── common/                  # task layer + runner ABC + CLI helpers
├── patching/                # shared activation-patching hooks + sweep
│
├── attention_knockout/      # block cross-modal attention; sweep across blocks
├── i2i_to_unconditional/    # patch i2i activations into empty-prompt t2i
└── i2i_to_i2i_patching/     # patch between two i2i runs varying one modality
```

Each priority experiment has the same two-file surface:

| File | Role |
|---|---|
| `runner.py` | `<Experiment>Runner` subclass of `ExperimentRunner`; per-task pipeline. |
| `<entry>.py` | Thin entry script: parses args via `add_task_selection`, dispatches to the runner. |

## Runner pattern

[`common/runner.py`](common/runner.py) defines the ABC. A subclass implements `run_one(task) -> Path`; the base class provides `run_many`, `task_dir`, `write_task_metadata`, and `reference_image`. Outputs:

```
results/<experiment>/<edit_type>/<task_id>/<run_timestamp>/
    task_metadata.json    # task fields + runner extras + git_sha
    reference.png
    ... per-experiment artifacts ...
results/<experiment>/_runs/<run_timestamp>/run_metadata.json
    # one row per task: task_id, task_dir, plus the launcher's flags
```

## Task selection (CLI)

Every entry script and launcher imports the same group from [`common/cli.py`](common/cli.py):

```
--task-id ID [ID ...]                                      # manual: explicit task IDs
--edit-type {add,remove,customize} [{...} ...]             # batch: full bucket(s)
--limit N                                                  # cap per --edit-type
--bucket NAME                                              # full bucket (e.g. solid_color)
```

Either `--task-id` or `--edit-type` is required (mutually exclusive). The `manual` bucket is unioned into each `--edit-type` and filtered by edit_type — you get every dataset task plus every manual scene of that type.

## Running

Single task, local:

```bash
uv run python experiments/attention_knockout/knockout_run.py \
    --task-id solid_red_couch --settings 'image->text' --layer-mode prefix
```

Quoting reminder: settings like `'ref->text'` must be single-quoted so the shell doesn't redirect on `>`.

Paper-scale reproduction (every task family for one intervention, in one process): use the matching top-level script — [`scripts/reproduce_attention_knockout.py`](../scripts/reproduce_attention_knockout.py), [`scripts/reproduce_t2i_lens.py`](../scripts/reproduce_t2i_lens.py), or [`scripts/reproduce_i2i_to_i2i_patching.py`](../scripts/reproduce_i2i_to_i2i_patching.py). Each iterates over the active task families with the right per-cell hyperparameters and uses `--skip-if-completed` so reruns resume.

## Run → grade → tabulate

Each priority experiment has full / padding-only / content-only sibling variants that share a runner and differ only in CLI flags + the patched/knocked-out cell filename. Sibling VLM judges score them with identical question wording so the CSVs are row-by-row comparable per `entity_id`.

| Experiment | Run flag | Variants |
|---|---|---|
| `attention_knockout` | `--settings` | `'ref->text'`, `'ref->text[padding]'`, `'ref->text[content]'` (plus `'ref->image'`) |
| `i2i_to_i2i_patching` | `--text-token-mode` | `all`, `padding_only`, `content_only` |
| `i2i_to_unconditional` | `--text-token-mode` | `all`, `padding_only`, `content_only` |

1. **Run** via each experiment's entry script (or the matching `scripts/reproduce_<exp>.py` for the full paper-scale sweep) — flags documented in [`attention_knockout/README.md`](attention_knockout/README.md), [`i2i_to_i2i_patching/README.md`](i2i_to_i2i_patching/README.md), [`i2i_to_unconditional/README.md`](i2i_to_unconditional/README.md).
2. **Grade** via `scripts.run_judge`:
   ```bash
   uv run python -m scripts.run_judge --all                              # every group + ungrouped judge
   uv run python -m scripts.run_judge --judge ko_color_ref_to_text_content  # single judge
   uv run python -m scripts.run_judge --write-readme                     # refresh judge index
   ```
3. **Tabulate** by pooling pass rates from `results_v4/vlm_judge/<judge>.csv`.

Judge naming: `<family>_<direction>` (full), `<family>_<direction>_padding` / `_text_padding` (padding-only), `<family>_<direction>_content` / `_text_content` (content-only).

## Paper-scale reproduction

```bash
uv run python scripts/reproduce_attention_knockout.py
uv run python scripts/reproduce_t2i_lens.py
uv run python scripts/reproduce_i2i_to_i2i_patching.py
```

Each script subprocess-invokes the matching experiment's entry-point module with the right per-cell hyperparameters and task family, releases the GPU between cells, and uses `--skip-if-completed` so reruns pick up where the last run left off.

| Family | Tasks | Used by | Source |
|---|---:|---|---|
| `solid_color` | 320 | All 3 experiments | 8 colors × 8 objects × 5 noise seeds (synthetic) |
| `style` | 450 | All 3 experiments | 18 hand-curated style subjects × 5 prompts × 5 seeds (`property_manual`) |
| `dreambench_humans` | 90 | KO + T2I Lens | 10 real-human DreamBench++ subjects × 9 individualized prompts |
| `dreambench_humans_shared` | 50 | I2I-to-I2I Patching | Same 10 real-human subjects × 5 shared prompts (pair source pool) |
| `add` | 789 | T2I Lens | sun397 + property_manual |
| `remove` | 726 | T2I Lens | sun397 |

Active families are mirrored in [`scripts/judge/configs.py::JUDGES`](../scripts/judge/configs.py); verify with `uv run python scripts/v4_status.py --consistency-check`. Output layout:

```
results_v4/
├── attention_knockout/full_ko_4step/<setting>/<task_id>/
├── i2i_to_unconditional/<sweep_mode>/<task_id>/
├── i2i_to_i2i_patching/<pair_family>/<source>__<target>/
└── vlm_judge/<judge_name>.csv          # CHECKED IN: small CSV, useful to diff
```

Generation outputs are gitignored under `/results_v4/*` except the judge CSVs — pass rates in the paper tables are pooled directly from those CSVs.

## Patching framework

[`patching/`](patching/) is the shared hook + sweep utility used by `i2i_to_unconditional/` and `i2i_to_i2i_patching/`. (Attention knockout doesn't use it — that experiment installs additive masks via custom `attn_processor` subclasses instead.) Activation patching is a causal intervention: capture activations from a **source** run, then inject them into a **target** run at specific points to measure what information each component carries.

Per-task token counts come from a `TokenLayout` built by [`utils.flux2_klein.layout_for`](../utils/flux2_klein.py), threaded down by the runner. **MM blocks** return `(txt_stream, img_stream)`:

```
txt_stream:  [text]
img_stream:  [noise | ref]    # i2i (layout.has_ref)
img_stream:  [noise]          # t2i
```

**Single blocks** return a flat sequence `[text | noise | ref]` (or `[text | noise]` for t2i). `ref` tokens are present only in i2i; `ref_seq_len` matches `noise_seq_len` at 1024² and differs at native dataset resolutions.

Module entry points (full APIs in the module docstrings):

| File | Provides |
|---|---|
| [`patching/hooks.py`](patching/hooks.py) | `make_patch_hook(layer, source_act, category, target_layout, text_token_indices=...)`; additive variant `make_add_hook`. |
| [`patching/sweep.py`](patching/sweep.py) | `sweep_and_grid(...)`, `make_patch_pipeline_producer(...)`, `make_input_to_block0_producer(...)`. |
| [`patching/utils.py`](patching/utils.py) | `resolve_block`, `run_pipeline_with_hooks`, `extract_category_acts`, `resolve_content_token_indices`. |

## Adding a new experiment

1. `experiments/<new>/runner.py` — subclass `ExperimentRunner`, set `name` + `results_root`, implement `run_one(task)`.
2. `experiments/<new>/<entry>.py` — `add_task_selection(parser)` then `runner.run_many(resolve_tasks(args))`.
3. (Optional) `scripts/reproduce_<new>.py` — loop over the paper-scale task families with the right per-cell flags, mirroring [`scripts/reproduce_attention_knockout.py`](../scripts/reproduce_attention_knockout.py).

For activation-patching experiments, build a `layout_for(...)` for source and target inside `run_one`, capture via `model.capture_activations()`, and run the target sweep with `make_patch_pipeline_producer` + `sweep_and_grid`.

## Tests

CPU unit tests live under [`common/tests/`](common/tests/) and [`tests/`](../tests/) — task schema, judge registry, pair builders, skip-if-completed. Run with:

```bash
uv run pytest tests/ experiments/common/tests/ -xvs
```

GPU byte-identity verification fixtures were retired in May 2026; capture fresh fixtures off a canonical cluster run if reproducibility checks are needed again.
