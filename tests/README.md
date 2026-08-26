# Repo-level tests

CPU-only, run with:

```bash
uv run pytest tests/ -xvs
```

| File | Covers |
|---|---|
| `test_judge_configs.py` | JUDGES config schema / consistency. |
| `test_judge_groups.py` | Judge grouping / bundle logic. |
| `test_i2i2i_flat_layout.py` | Block naming (`block_suffix` round-trip) and flat-layout cell invariants. |
| `test_longprompt_ablation.py` | Long-prompt ablation task wiring. |
| `test_padding_ablation.py` | Padding-token dose-response buckets, per-level pair builders, judge registry. |
| `test_pair_builders.py` | i2i pair-list builders. |
| `test_skip_if_completed.py` | Runner completion-log skip logic. |
| `test_qwen_layout.py` | Qwen-Image-Edit layout math (`effective_ref_dims`, block naming, `TokenLayout` vision spans, `text[vision]`/`text[prompt]` knockout regions) plus template token spans against the real Qwen2.5-VL processor (skips offline). |

GPU verification tests live per-experiment under `experiments/<name>/tests/`;
the Qwen bring-up micro-tests are `experiments/qwen_port/t1_*.py`.
