"""VLM-judge every baseline in a bucket against the strict whole-image style
criterion (same question as ``judge_repairs.py``): pass=1 iff the ENTIRE
image reads as the reference's illustrated style, 0 if any substantial part
looks photographic.

Tasks are grouped by reference so consecutive calls share the ref-image
prompt prefix (ephemeral cache). Resumable: existing rows in the output CSV
are kept and their tasks skipped.

Usage::

    uv run python -m experiments.repair_amplify.judge_baselines \\
        --bucket style --model claude-sonnet-5
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from pathlib import Path

from experiments.common.tasks import load_tasks
from experiments.repair_amplify.judge_repairs import QUESTION
from scripts.judge.api import call_judge
from utils.vlm import make_vlm

BASELINES_ROOT = Path("results/repair_amplify/baselines")

# Subject-focused re-grade criterion: a stylized/fictional-looking MAIN
# SUBJECT is a pass even on a photographic background; fail only when the
# subject itself reads as a real photographed thing. Motivated by strict-pass
# false positives where the judge conflated "softer rendering than the ref's
# flat cel-shading" with "photographic".
SUBJECT_QUESTION = (
    "Image 1 is a stylized illustration reference. Image 2 is a generated "
    "edit that was supposed to depict the subject from Image 1 in a new "
    "scene, carrying over the reference's stylized rendering. Focus ONLY on "
    "the MAIN SUBJECT of Image 2 (the character/animal/object inherited "
    "from Image 1), not the background. If the main subject is rendered in "
    "a stylized, fictional, illustrated, cartoon, anime, painterly, or "
    "toy-like manner - even loosely, and even if the background looks "
    "photographic - reply 1. Reply 0 ONLY if the main subject itself looks "
    "like a real photographed animal, person, or object with no stylization. "
    "Note: a softer or more detailed illustration style than Image 1 still "
    "counts as stylized (reply 1); do not fail for a style-degree mismatch."
)

QUESTIONS = {"strict": QUESTION, "subject": SUBJECT_QUESTION}


def out_csv_path(bucket: str, question: str = "strict") -> Path:
    suffix = "" if question == "strict" else f"_{question}"
    return BASELINES_ROOT / f"{bucket}_judge_baselines{suffix}.csv"


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--question", default="strict", choices=sorted(QUESTIONS))
    ap.add_argument(
        "--tasks-file", type=Path, default=None,
        help="optional file of task_ids to judge (default: whole bucket)",
    )
    args = ap.parse_args()

    images_dir = BASELINES_ROOT / args.bucket
    refs_dir = images_dir / "_refs"
    out_csv = out_csv_path(args.bucket, args.question)
    question = QUESTIONS[args.question]

    done: dict[str, list] = {}
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                if row["pass"] != "":
                    done[row["task_id"]] = list(row.values())
        print(f"[judge_baselines] resuming: {len(done)} verdicts already on disk")

    tasks = load_tasks(args.bucket)
    if args.tasks_file is not None:
        keep = {l.strip() for l in args.tasks_file.read_text().splitlines() if l.strip()}
        tasks = [t for t in tasks if t.task_id in keep]
        assert len(tasks) == len(keep), "tasks-file contains unknown task_ids"
    # Group by reference so the shared ref-image prefix stays cache-hot.
    tasks.sort(key=lambda t: (t.real_ref_name or "", t.task_id))
    vlm = make_vlm(model=args.model)

    rows: list[list] = list(done.values())
    total_in = total_out = 0
    for task in tasks:
        if task.task_id in done:
            continue
        img = images_dir / f"{task.task_id}.png"
        ref = refs_dir / f"{task.real_ref_name}.png"
        assert img.exists(), f"missing baseline {img}"
        assert ref.exists(), f"missing ref {ref}"
        verdict, reason, in_tok, out_tok = await call_judge(
            vlm,
            ["Image 1 - style reference:", "Image 2 - generated edit:"],
            [ref, img],
            question,
            cache_prefix_len=1,
            cache_key=f"repair_amplify_baselines/{task.real_ref_name}",
        )
        total_in += in_tok
        total_out += out_tok
        rows.append([task.task_id, "baseline_4step", verdict, reason,
                     vlm.model, in_tok, out_tok])
        print(f"{task.task_id[-46:]:48s} -> {verdict} ({reason})", flush=True)
        # Persist incrementally so a crash never loses verdicts.
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["task_id", "cell", "pass", "reason", "model",
                        "input_tokens", "output_tokens"])
            w.writerows(rows)

    n_fail = sum(1 for r in rows if str(r[2]) == "0")
    n_pass = sum(1 for r in rows if str(r[2]) == "1")
    print(f"\n[judge_baselines] {len(rows)} verdicts -> {out_csv}")
    print(f"  pass={n_pass} fail={n_fail} err={len(rows) - n_pass - n_fail}")
    print(f"  tokens this run: in={total_in} out={total_out}")
    print("\nFailing tasks:")
    for r in sorted(rows):
        if str(r[2]) == "0":
            print(f"  {r[0]}")


if __name__ == "__main__":
    asyncio.run(main())
