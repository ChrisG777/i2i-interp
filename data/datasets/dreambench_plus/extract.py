"""Curate DreamBench++ customize tasks into ``data/tasks/customize/``.

Reads the locally-downloaded dataset at
``data/datasets/dreambench_plus/raw/`` (see ``download.py`` for the layout)
and emits one TaskDefinition per (subject, prompt) pair. DreamBench++ ships
exactly one image per subject (150 total), so the default of one prompt
per subject yields 150 customize tasks.

Variant ``i==0`` keeps the bare task_id (byte-identical to a single-prompt
run); variants ``i>=1`` get a ``_p{i}`` suffix. Pass
``--prompts-per-subject 9`` to materialize the full prompt list per
subject (1,350 tasks total).

``--multi-prompt-categories`` is an optional whitelist: when supplied, only
caption files whose nested-category path starts with one of the listed
prefixes get expanded; everything else falls back to 1 prompt. When empty
(default), all subjects expand to ``--prompts-per-subject``.

This extractor is *additive*: it preserves rows from other ``source`` values
already present in ``customize/tasks.jsonl`` (e.g., ``dreambooth``,
``property_manual``) and only overwrites rows whose ``source ==
"dreambench_plus"``.

.. note::

    Output rows land in ``data/tasks/customize/`` regardless of subject
    kind. The active task buckets that consume these rows are
    ``dreambench_humans`` (10 real-human subjects × 9 individualized
    prompts, used by KO and T2I Lens) and ``dreambench_humans_shared``
    (the same 10 subjects × 5 shared prompts, used as the source pool
    for I2I-to-I2I Patching pairs). Routing requires
    ``metadata.subject_kind`` (real_human vs. not) and
    ``metadata.category`` (live_subject_human_shared vs. not), and this
    extractor does NOT set those fields. If you re-run this script,
    annotate the rows post-hoc and split into the two destination
    JSONLs.

Usage:
    uv run python data/datasets/dreambench_plus/extract.py
    uv run python data/datasets/dreambench_plus/extract.py --dry-run
    uv run python data/datasets/dreambench_plus/extract.py --prompts-per-subject 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.datasets._anchor import anchor_to_reference  # noqa: E402
from data.datasets._image_utils import center_crop_to_multiple  # noqa: E402
from data.tasks._seed import task_seed  # noqa: E402
from experiments.common.tasks import TaskDefinition  # noqa: E402

RAW_ROOT = REPO_ROOT / "data" / "datasets" / "dreambench_plus" / "raw"
OUT_DIR = REPO_ROOT / "data" / "tasks" / "customize"

SOURCE_NAME = "dreambench_plus"


def _read_caption_file(path: Path) -> tuple[str, list[str]]:
    """Returns (subject_string, list_of_prompts)."""
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) >= 2, f"Caption file too short: {path}"
    subject = lines[0]
    prompts = lines[1:]
    return subject, prompts


def _load_existing_other_sources(tasks_path: Path) -> list[dict]:
    """Read existing tasks.jsonl, keep rows whose source != SOURCE_NAME."""
    if not tasks_path.exists():
        return []
    rows: list[dict] = []
    for line in tasks_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("source") != SOURCE_NAME:
            rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--multi-prompt-categories",
        nargs="*",
        default=[],
        metavar="PATH",
        help=(
            "Optional whitelist of nested-category paths (relative to "
            "raw/captions/) — when non-empty, only listed prefixes expand to "
            "--prompts-per-subject; others emit 1. When empty (default), "
            "every subject expands."
        ),
    )
    p.add_argument(
        "--prompts-per-subject",
        type=int,
        default=1,
        help=(
            "Number of prompts to materialize per subject. Default 1 (one "
            "prompt per image → 150 customize tasks). Pass 9 to expand to "
            "the full DreamBench++ prompt list per subject."
        ),
    )
    args = p.parse_args()
    assert args.prompts_per_subject >= 1, "--prompts-per-subject must be >= 1"

    images_root = RAW_ROOT / "images"
    captions_root = RAW_ROOT / "captions"
    assert images_root.is_dir() and captions_root.is_dir(), (
        f"Missing raw DreamBench++ data under {RAW_ROOT}\n"
        f"Run 'uv run python data/datasets/dreambench_plus/download.py' first."
    )

    images_out = OUT_DIR / "images"
    if not args.dry_run:
        images_out.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for caption_file in sorted(captions_root.rglob("*.txt")):
        rel = caption_file.relative_to(captions_root)
        category = rel.parts[0]
        index = rel.stem
        # Some categories (e.g. live_subject/{animal,human}/) are nested.
        # Mirror the full relative path on the image side, swapping suffix.
        image_file = images_root / rel.with_suffix(".jpg")
        if not image_file.exists():
            skipped.append((f"{rel.with_suffix('')}", "missing image"))
            continue
        # Use the full nested path (minus suffix) as the slug suffix so
        # task_ids stay unique across nested categories.
        nested_slug = "_".join(rel.with_suffix("").parts[1:])  # drops top category

        subject, prompts = _read_caption_file(caption_file)
        if not prompts:
            skipped.append((f"{category}/{index}", "no prompts"))
            continue

        nested_dir = "/".join(rel.with_suffix("").parts[:-1])
        if args.multi_prompt_categories:
            matched = any(
                nested_dir == c or nested_dir.startswith(c + "/")
                for c in args.multi_prompt_categories
            )
        else:
            # Empty whitelist: expand every subject.
            matched = True
        n_for_this_subject = (
            min(args.prompts_per_subject, len(prompts)) if matched else 1
        )

        subject_slug = subject.lower().replace(" ", "_")
        base_task_id = f"customize_{SOURCE_NAME}_{category}_{nested_slug}_{subject_slug}"
        dest_image = images_out / f"{base_task_id}.jpg"

        with Image.open(image_file) as im:
            im = im.convert("RGB")
            im = center_crop_to_multiple(im)
            w, h = im.size
            from data.datasets._image_utils import PATCH
            if w < PATCH or h < PATCH:
                skipped.append((base_task_id, f"too small after crop: {w}x{h}"))
                continue
            if not args.dry_run:
                im.save(dest_image, format="JPEG", quality=95)

        rel_image = dest_image.relative_to(REPO_ROOT).as_posix()

        for prompt_idx in range(n_for_this_subject):
            task_id = base_task_id if prompt_idx == 0 else f"{base_task_id}_p{prompt_idx}"
            original_instruction = prompts[prompt_idx]
            instruction = anchor_to_reference(original_instruction, subject, category)
            task = {
                "task_id": task_id,
                "edit_type": "customize",
                "source": SOURCE_NAME,
                "instruction": instruction,
                "source_image_path": rel_image,
                "source_caption": None,
                "ref_seed": None,
                "noise_seed": task_seed(task_id),
                "real_ref_name": None,
                "height": h,
                "width": w,
                "metadata": {
                    "subject": subject,
                    "category": category,
                    "subject_index": index,
                    "all_prompts": prompts,
                    "selected_prompt_idx": prompt_idx,
                    "original_instruction": original_instruction,
                },
            }
            TaskDefinition(
                **{k: v for k, v in task.items() if k != "metadata"},
                metadata=task["metadata"],
            )
            tasks.append(task)

    print(f"\n[{SOURCE_NAME} customize] kept {len(tasks)} subjects (skipped {len(skipped)})")
    for tid, reason in skipped[:5]:
        print(f"  skip {tid}: {reason}")

    if args.dry_run:
        return 0

    out_path = OUT_DIR / "tasks.jsonl"
    other = _load_existing_other_sources(out_path)
    all_rows = other + tasks
    all_rows.sort(key=lambda r: (r["source"], r["task_id"]))
    with open(out_path, "w") as f:
        for t in all_rows:
            f.write(json.dumps(t, sort_keys=True) + "\n")

    log_path = OUT_DIR / f"_extract_log_{SOURCE_NAME}.json"
    with open(log_path, "w") as f:
        json.dump({
            "source": SOURCE_NAME,
            "task_count": len(tasks),
            "skipped": skipped,
            "raw_root_sha256": hashlib.sha256(
                "\n".join(sorted(str(p.relative_to(RAW_ROOT)) for p in RAW_ROOT.rglob("*"))).encode()
            ).hexdigest(),
        }, f, indent=2)
    print(f"  wrote {out_path} ({len(all_rows)} total rows; {len(other)} preserved from other sources)")
    print(f"  wrote {log_path}")
    if tasks:
        sample = tasks[0]
        print(
            f"  manual inspection: open "
            f"{REPO_ROOT / sample['source_image_path']} and "
            f"`grep '\"task_id\": \"{sample['task_id']}\"' {out_path}`"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
