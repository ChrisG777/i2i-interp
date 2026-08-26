"""Generate clean 4-step i2i baselines for a whole bucket, one PNG per task.

Purpose: find tasks where the *unmodified* FLUX.2-klein i2i edit fails
(color transfer / style transfer not applied), as candidates for the
attention-amplification repair experiment (`amplify_run.py`).

Outputs (flat, resumable — existing files are skipped):

    results/repair_amplify/baselines/<bucket>/<task_id>.png
    results/repair_amplify/baselines/<bucket>/_refs/<ref_label>.png

Usage::

    uv run python -m experiments.repair_amplify.gen_baselines --bucket solid_color
    uv run python -m experiments.repair_amplify.gen_baselines --bucket style
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from experiments.common.baselines import load_or_make_reference
from experiments.common.tasks import load_tasks
from utils.flux2_klein import Flux2KleinModel

OUT_ROOT = Path("results/repair_amplify/baselines")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-inference-steps", type=int, default=4)
    args = ap.parse_args()

    tasks = load_tasks(args.bucket, limit=args.limit)
    assert tasks, f"no tasks in bucket {args.bucket!r}"
    out_dir = OUT_ROOT / args.bucket
    out_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = out_dir / "_refs"
    refs_dir.mkdir(exist_ok=True)

    todo = [t for t in tasks if not (out_dir / f"{t.task_id}.png").exists()]
    print(f"[gen_baselines] bucket={args.bucket}: {len(tasks)} tasks, "
          f"{len(todo)} to generate")
    if not todo:
        return

    model = Flux2KleinModel()
    for i, task in enumerate(todo):
        t0 = time.time()
        ref_img = load_or_make_reference(model, task)
        ref_name = task.real_ref_name or task.task_id
        ref_path = refs_dir / f"{ref_name}.png"
        if not ref_path.exists():
            ref_img.save(ref_path)
        img = model.generate(
            task.instruction,
            seed=task.noise_seed,
            num_inference_steps=args.num_inference_steps,
            image=ref_img,
            height=task.height,
            width=task.width,
        )
        img.save(out_dir / f"{task.task_id}.png")
        print(f"[{i + 1}/{len(todo)}] {task.task_id} ({time.time() - t0:.1f}s)",
              flush=True)


if __name__ == "__main__":
    main()
