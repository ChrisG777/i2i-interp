# Repair by amplification

Inverse of the attention knockout: instead of `-inf` at the (destination,
source) attention entries, add a positive bias `+lam` so those softmax logits
are boosted ("pay more attention") on a chosen set of blocks, reusing the
knockout SDPA-mask processors unchanged. The use case is repair: find tasks
where the unmodified i2i edit fails (color/style not applied), then boost the
binding pathway to fix them.

Directions (destination reads from source; bias at rows = destination
queries, cols = source keys):

| Direction | Effect |
|---|---|
| `text_from_ref` | text tokens attend more to reference tokens (strengthen binding formation; the paper's row) |
| `image_from_text` | noise tokens attend more to text tokens (strengthen binding readout) |
| `image_from_ref` | noise tokens attend more to reference tokens (direct route bypassing the binding) |

`--dest-subset padding|content` restricts the destination rows of
`text_from_ref` to the instruction's padding / content token positions.

## Pipeline

1. `gen_baselines.py`: clean 4-step i2i baselines for a whole bucket, one PNG
   per task (resumable), to find where the unmodified edit fails.
2. Triage: `judge_baselines.py` (VLM judge, strict whole-image style
   criterion).
3. `amplify_run.py`: the intervention sweep over tasks x directions x blocks
   x lambdas. Outputs land flat under `results/repair_amplify/runs/<task_id>/`
   as `amp_<direction>[_<subset>]_<blocks>_lam<x>.png` next to
   `reference.png` and `baseline_4step.png`.
4. `judge_repairs.py`: grade baseline + each repair cell on the same VLM
   question; baseline rows double-check the failure diagnosis.
5. Figures: `headline_figures.py` (before/after rows of reference | baseline
   | repair cells).

Example:

```bash
uv run python -m experiments.repair_amplify.amplify_run \
    --task-id solid_green_mug_s2 solid_red_couch_s1 \
    --blocks single_mm9 --lam 1 2 4 --direction text_from_ref
```
