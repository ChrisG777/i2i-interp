"""Verify a single-task run against the existing paper-scale output.

Use on the cluster (where ``results_v4/`` is populated) after a paper-scale
sweep has run. Pick any ``task_id`` whose paper output exists, re-run
JUST that task with the same per-cell flags that
``scripts/reproduce_<exp>.py`` would pass for the paper-scale sweep, and
diff the new artifacts against the previously-committed run.

Per-cell flags are imported directly from ``scripts/reproduce_<exp>.py``
(one source of truth) and ``--skip-if-completed`` is stripped so the run
actually fires. ``--task-id <id>`` replaces ``--bucket <name>`` so only
one task runs.

Numerical tolerance: CUDA is not byte-deterministic across GPU
architectures, so we report mean absolute pixel difference and PASS if
it's at or below ``--tolerance`` (default ``1.0`` on a 0-255 scale,
which catches semantic regressions while tolerating minor float drift).
Metadata is compared exactly except for ``run_timestamp`` and
``git_sha``.

Usage::

    # Knockout — pick any task in solid_color / style / dreambench_humans:
    uv run python scripts/verify_against_paper.py \\
        --experiment knockout --task-id solid_red_couch

    # T2I Lens — pick any task in solid_color / style / dreambench_humans / add / remove:
    uv run python scripts/verify_against_paper.py \\
        --experiment t2i_lens --task-id solid_red_couch

Exits 0 if every PNG in the existing run matches its counterpart in the
fresh run within tolerance; exits 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

# Per-experiment config: entry-point module + the matching reproduce_*.py
# from which we lift per-cell flags so there's only one source of truth.
import scripts.reproduce_attention_knockout as repro_ko
import scripts.reproduce_t2i_lens as repro_t2i

KO_CELLS_BY_BUCKET = {name: sel for name, sel in repro_ko.CELLS}
T2I_CELLS_BY_BUCKET = {name: flags for name, sel, flags in repro_t2i.CELLS}

EXPERIMENTS = {
    "knockout": {
        "module": "experiments.attention_knockout.knockout_run",
        "results_root": REPO_ROOT / "results_v4" / "attention_knockout",
        "flags_for_bucket": lambda b: repro_ko.COMMON_FLAGS if b in KO_CELLS_BY_BUCKET else None,
    },
    "t2i_lens": {
        "module": "experiments.i2i_to_unconditional.i2i_to_unconditional_patch",
        "results_root": REPO_ROOT / "results_v4" / "i2i_to_unconditional",
        "flags_for_bucket": lambda b: T2I_CELLS_BY_BUCKET.get(b),
    },
}


def find_task_bucket(task_id: str) -> str:
    from experiments.common.tasks import BUCKETS, load_tasks
    for bucket in BUCKETS:
        if any(t.task_id == task_id for t in load_tasks(bucket)):
            return bucket
    raise SystemExit(f"task_id {task_id!r} not found in any bucket")


def snapshot_run_dirs(results_root: Path, task_id: str) -> set[Path]:
    """All <run_timestamp> dirs under any task dir matching task_id."""
    if not results_root.exists():
        return set()
    out: set[Path] = set()
    for task_dir in results_root.rglob(task_id):
        if not task_dir.is_dir():
            continue
        for run_dir in task_dir.iterdir():
            if run_dir.is_dir():
                out.add(run_dir.resolve())
    return out


def pixel_diff(a: Path, b: Path) -> float:
    arr_a = np.asarray(Image.open(a).convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32)
    assert arr_a.shape == arr_b.shape, f"shape mismatch: {arr_a.shape} vs {arr_b.shape}"
    return float(np.abs(arr_a - arr_b).mean())


def diff_runs(old_dir: Path, new_dir: Path, tolerance: float) -> bool:
    print(f"  old: {old_dir}")
    print(f"  new: {new_dir}")

    old_meta = json.loads((old_dir / "task_metadata.json").read_text())
    new_meta = json.loads((new_dir / "task_metadata.json").read_text())
    for k in ("run_timestamp", "git_sha"):
        old_meta.pop(k, None)
        new_meta.pop(k, None)
    if old_meta == new_meta:
        print("  metadata: match (excluding run_timestamp, git_sha)")
    else:
        print("  metadata: DIFFER")
        for k in sorted(set(old_meta) | set(new_meta)):
            if old_meta.get(k) != new_meta.get(k):
                print(f"    {k!r}: {old_meta.get(k)!r}  vs  {new_meta.get(k)!r}")

    old_pngs = {p.relative_to(old_dir): p for p in old_dir.rglob("*.png")}
    new_pngs = {p.relative_to(new_dir): p for p in new_dir.rglob("*.png")}
    common = sorted(set(old_pngs) & set(new_pngs))
    only_old = sorted(set(old_pngs) - set(new_pngs))
    only_new = sorted(set(new_pngs) - set(old_pngs))
    if only_old:
        print(f"  PNGs only in old run: {[str(p) for p in only_old]}")
    if only_new:
        print(f"  PNGs only in new run: {[str(p) for p in only_new]}")

    print(f"  comparing {len(common)} png(s):")
    max_diff = 0.0
    for rel in common:
        d = pixel_diff(old_pngs[rel], new_pngs[rel])
        max_diff = max(max_diff, d)
        tag = "OK  " if d <= tolerance else "FAIL"
        print(f"    [{tag}] {rel}: mean-abs-diff={d:.4f}/255")
    passed = (max_diff <= tolerance) and not (only_old or only_new)
    print(f"  -> {'PASS' if passed else 'FAIL'} (max pixel diff {max_diff:.4f} / 255)")
    return passed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--tolerance", type=float, default=1.0,
                    help="Max mean-abs pixel diff (0-255 scale) to PASS. "
                         "Default 1.0; lower if you want stricter checks.")
    args = ap.parse_args()

    cfg = EXPERIMENTS[args.experiment]
    bucket = find_task_bucket(args.task_id)
    flags = cfg["flags_for_bucket"](bucket)
    if flags is None:
        raise SystemExit(
            f"task_id {args.task_id!r} is in bucket {bucket!r}, but the "
            f"{args.experiment!r} reproduce script doesn't have a cell for "
            f"that bucket. Pick a task from a bucket that's actually used."
        )

    # Strip --skip-if-completed so the verification run actually fires.
    cell_flags = [f for f in flags if f != "--skip-if-completed"]
    argv = [
        "uv", "run", "python", "-m", cfg["module"],
        "--task-id", args.task_id,
        *cell_flags,
    ]

    print(f"task_id={args.task_id!r} bucket={bucket!r}")
    print(">>> " + " ".join(argv))
    print()

    before = snapshot_run_dirs(cfg["results_root"], args.task_id)
    result = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"verification run failed: returncode={result.returncode}")

    after = snapshot_run_dirs(cfg["results_root"], args.task_id)
    new_run_dirs = sorted(after - before)
    if not new_run_dirs:
        raise SystemExit(
            "No new run subdir was created. The entry point may have a stale "
            "--skip-if-completed semantic, or the task didn't produce output."
        )

    pass_count = 0
    fail_count = 0
    for new_dir in new_run_dirs:
        task_dir = new_dir.parent
        old_dirs = sorted(p for p in task_dir.iterdir()
                          if p.is_dir() and p.resolve() in before)
        if not old_dirs:
            print(f"[skip] {task_dir.relative_to(REPO_ROOT)}: no prior run to compare")
            continue
        old_dir = old_dirs[-1]
        print(f"\n=== {task_dir.relative_to(REPO_ROOT)} ===")
        if diff_runs(old_dir, new_dir, args.tolerance):
            pass_count += 1
        else:
            fail_count += 1

    print()
    print(f"=== summary: {pass_count} pass, {fail_count} fail ===")
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
