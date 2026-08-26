# GPT-5.6 re-grade runbook (cluster)

> **Status: complete (2026-07-26).** All 26 paper judges are fully re-graded
> (9,859/9,859 verdicts, 91.1% agreement with Claude) and the CSVs are
> committed. The results are incorporated in the paper as the "Second VLM
> Judge" appendix section, rendered with
> `scripts/build_judge_tables.py --model gpt-5.6-terra [--agreement]`.
> Nothing below needs to run again unless verdicts are regenerated.

Instructions for re-grading the paper's VLM-judge verdicts with GPT-5.6-terra
(OpenAI Responses API) as a double-check on the Claude Opus 4.7 verdicts.
This is meant to run on the cluster checkout at
`/data/scratch/chrisge/i2i-interp`, where the result images live.

Background: `scripts/run_judge.py` is now model-agnostic. `--model
gpt-5.6-terra` writes verdicts to `results_v4/vlm_judge/gpt-5.6-terra/<judge>.csv`,
side by side with the paper's Claude CSVs at `results_v4/vlm_judge/<judge>.csv`.
The Claude CSVs and the paper tables are never touched by any of this.
`--paper` restricts the run to the 26 judges that feed the paper's four judge
tables (`scripts/judge/configs.py::PAPER_JUDGES`).

## IMPORTANT: style-judge wording fix (2026-07-25)

The 8 style judges' question wording was corrected (a mechanical
clipart->style rename had mangled the prompt prose; it now matches the paper
appendix again). If you graded ANY of these judges with GPT before pulling
the commit containing this notice, their verdicts used the old wording:
delete these files (only the ones that exist) and re-run so they are
re-graded with the corrected prompts:

```bash
cd /data/scratch/chrisge/i2i-interp
rm -f results_v4/vlm_judge/gpt-5.6-terra/i2i_unc_style_text_lens.csv \
      results_v4/vlm_judge/gpt-5.6-terra/ko_style_ref_to_text.csv \
      results_v4/vlm_judge/gpt-5.6-terra/ko_style_ref_to_text_padding.csv \
      results_v4/vlm_judge/gpt-5.6-terra/ko_style_ref_to_text_content.csv \
      results_v4/vlm_judge/gpt-5.6-terra/ko_style_ref_to_image.csv \
      results_v4/vlm_judge/gpt-5.6-terra/i2i2i_style.csv \
      results_v4/vlm_judge/gpt-5.6-terra/i2i2i_style_text_padding.csv \
      results_v4/vlm_judge/gpt-5.6-terra/i2i2i_style_text_content.csv
```

The other 18 paper judges' prompts were never affected; their verdicts from
before the fix remain valid and resume normally.

## 0. Prerequisites

```bash
cd /data/scratch/chrisge/i2i-interp
git pull
uv sync          # picks up the new openai dependency
```

`OPENAI_API_KEY` must be present in `.env` at the repo root (or exported).
Verify presence only; never read or print the contents of `.env`:

```bash
uv run python -c "from dotenv import load_dotenv; import os; load_dotenv('.env'); print(bool(os.getenv('OPENAI_API_KEY')))"
```

If that prints `False`, stop and ask Chris to add the key.

## 1. Smoke test (a few cents, ~1 minute)

Grade 2 entities with one ungrouped-style judge call path:

```bash
uv run python -m scripts.run_judge --judge i2i2i_color --model gpt-5.6-terra --limit 2
```

Then 2 entities through the grouped (cache-sharing) path:

```bash
uv run python -m scripts.run_judge --group i2i2i_color --model gpt-5.6-terra --limit 2
```

Note: `--limit N` counts entities per judge (or per group), and in group mode
each entity fans out to all sibling judges, so the second command makes up to
6 calls.

Check the output:

```bash
cat results_v4/vlm_judge/gpt-5.6-terra/i2i2i_color.csv
```

Expected: header `entity_id,pass,reason,model,input_tokens,output_tokens`,
rows with `pass` in {0,1}, `model=gpt-5.6-terra`, nonzero token counts, and
reasons that plausibly describe the images. If `pass` is empty and `reason`
starts with `PARSE_ERROR` or `ERROR`, stop and report the raw reason instead
of proceeding.

Sanity-check agreement on the smoked rows (pure CSV read, no API):

```bash
uv run python scripts/compare_judges.py --model gpt-5.6-terra --judge i2i2i_color
```

## 2. Full paper re-grade

About 9,859 prompts total, roughly $100-150 at Terra list prices, a few hours
at the default concurrency of 10.

Either run it directly (tmux recommended):

```bash
uv run python -m scripts.run_judge --paper --model gpt-5.6-terra
```

Or as a CPU SBATCH job (never on a vision-shared GPU partition):

```bash
mkdir -p logs
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=judge_paper_gpt56
#SBATCH --partition=tig-cpu
#SBATCH --account=csail
#SBATCH --qos=tig-main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/data/scratch/chrisge/i2i-interp/logs/%x-%A.out
#SBATCH --error=/data/scratch/chrisge/i2i-interp/logs/%x-%A.err
set -euo pipefail
cd /data/scratch/chrisge/i2i-interp
uv run python scripts/run_judge.py --paper --model gpt-5.6-terra
EOF
```

The run is resumable and idempotent: verdicts are appended per call, restart
skips everything with a recorded 0/1 verdict, and error rows (empty `pass`)
are retried on the next invocation. If it dies or hits the walltime, just
rerun the same command. The smoke rows from step 1 are kept, not re-graded.

## 3. Verify completeness

Rerun the full command; it should print "nothing to judge (all already
verdicted)" for every judge. Error rows would instead trigger retries; let
them. Quick row-count check:

```bash
uv run python - <<'EOF'
from scripts.judge.configs import PAPER_JUDGES, get
from scripts.judge.csv_io import load_existing_rows
for name in sorted(PAPER_JUDGES):
    cfg = get(name)
    rows = load_existing_rows(cfg.csv_path_for("gpt-5.6-terra"))
    done = sum(1 for r in rows.values() if r.get("pass", "") in ("0", "1"))
    base = len(load_existing_rows(cfg.csv_path))
    flag = "" if done == base else "  <-- INCOMPLETE"
    print(f"{name:<42} gpt={done:>4}  claude={base:>4}{flag}")
EOF
```

Every judge should match the Claude row count (that is what is on disk and
was graded before).

## 4. Report agreement and push the CSVs

```bash
uv run python scripts/compare_judges.py --model gpt-5.6-terra
```

Then commit and push the new CSVs (they are whitelisted in `.gitignore`):

```bash
git add results_v4/vlm_judge/gpt-5.6-terra/
git commit -m "judge: gpt-5.6-terra re-grade of paper judges"
git push
```

Include the compare_judges overall agreement number in the commit message or
report it back to Chris. Do not modify anything under
`results_v4/vlm_judge/*.csv` (the Claude verdicts) and never commit `.env`.
