"""VLM-judge the amplification repairs: for each failing task, grade the
baseline and each candidate repair cell on the same question — is the WHOLE
image rendered in the reference's illustrated style (not a photograph)?

Writes ``results/repair_amplify/judge_repairs.csv`` with one row per
(task, cell). The baseline rows double-check the failure diagnosis (expected
pass=0); repair rows quantify the fix.

Usage::

    uv run python -m experiments.repair_amplify.judge_repairs \\
        --tasks slurm/repair_amplify/failures_style_round2.txt \\
        --cells baseline_4step \\
                amp_text_from_ref_mm7_lam2 \\
                amp_text_from_ref_mm5+mm6+mm7_lam2 \\
                amp_text_from_ref_mm5+mm6+mm7_lam4 \\
                amp_text_from_ref_mm5+mm6+mm7_lam6
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

from scripts.judge.api import call_judge
from utils.vlm import make_vlm

RUNS_ROOT = Path("results/repair_amplify/runs")
OUT_CSV = Path("results/repair_amplify/judge_repairs.csv")

QUESTION = (
    "Image 1 is a stylized illustration reference. Image 2 is a generated "
    "edit that was supposed to render its ENTIRE scene in the same "
    "illustrated / cartoon / stylized rendering as Image 1 (matching its "
    "medium, not its content). Look at the whole of Image 2 - subject AND "
    "background. Reply 1 if the entire image reads as an illustration / "
    "cartoon / stylized rendering rather than a photograph; reply 0 if any "
    "substantial part of Image 2 (subject, people, or background) still "
    "looks photographic."
)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    task_ids = [l.strip() for l in args.tasks.read_text().splitlines() if l.strip()]
    vlm = make_vlm(**({"model": args.model} if args.model else {}))

    rows: list[list] = []
    for task_id in task_ids:
        d = RUNS_ROOT / task_id
        ref = d / "reference.png"
        assert ref.exists(), f"missing {ref}"
        for cell in args.cells:
            img = d / f"{cell}.png"
            if not img.exists():
                print(f"[skip] {task_id}/{cell} (no file)")
                continue
            verdict, reason, in_tok, out_tok = await call_judge(
                vlm,
                ["Image 1 - style reference:", "Image 2 - generated edit:"],
                [ref, img],
                QUESTION,
                cache_prefix_len=1,
                cache_key=f"repair_amplify/{task_id}",
            )
            rows.append([task_id, cell, verdict, reason, vlm.model, in_tok, out_tok])
            print(f"{task_id[-40:]:42s} {cell:44s} -> {verdict} ({reason})")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "cell", "pass", "reason", "model",
                    "input_tokens", "output_tokens"])
        w.writerows(rows)

    print(f"\n[judge_repairs] {len(rows)} verdicts -> {OUT_CSV}")
    by_cell: dict[str, list[int]] = {}
    for _, cell, v, *_ in rows:
        if v is not None:
            by_cell.setdefault(cell, []).append(int(v))
    for cell, vs in by_cell.items():
        print(f"  {cell:46s} {sum(vs)}/{len(vs)}")


if __name__ == "__main__":
    asyncio.run(main())
