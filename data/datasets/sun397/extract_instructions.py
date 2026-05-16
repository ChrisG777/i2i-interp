"""Author new SUN397 add/remove task instructions via the VLM pipeline.

This is the **task-authoring** path that originally generated the 789 add +
726 remove rows committed under ``data/tasks/{add,remove}/tasks.jsonl``. End
users reproducing the paper do NOT need to run this — the rows already ship
in the repo, and the per-row source JPGs can be cropped without an API key
via ``prepare_images.py``. Run this only to extend the task set with new
SUN397 categories / images. Requires ``ANTHROPIC_API_KEY``.

One walk per invocation: pick ``--num-locations`` distinct categories from
SUN397's 397-class index (even stride by default; seeded random sample if
``--seed`` is set, or all 397 with ``--all-categories``). For each category,
pick ``--num-images-per-category`` JPGs at random (per-category seeded by
``--image-seed``). Each image is sent to the shared VLM pipeline
(``data/datasets/_vlm_tasks.py``) which authors optional add / remove
instructions. Writes 0-2 task rows per image into the standard buckets.

CSAIL-only by default — SUN397 lives on the Torralba NFS mount. Override the
root with ``--root`` if you have a local checkout.

Idempotence: ``_write_bucket`` merges by task_id (existing rows kept; new rows
with matching task_ids replace them; otherwise appended). Re-running with the
same ``--image-seed`` is a no-op for already-extracted task_ids — extraction
short-circuits before billing the VLM. Re-running with a *different* seed
*adds* new rows on top of the old ones; to fully reset, manually drop active
sun397 rows from the JSONL first.

Usage:
    uv run python -m data.datasets.sun397.extract_instructions
    uv run python -m data.datasets.sun397.extract_instructions --dry-run
    uv run python -m data.datasets.sun397.extract_instructions --num-locations 2 --concurrency 2
    uv run python -m data.datasets.sun397.extract_instructions --seed 0
    uv run python -m data.datasets.sun397.extract_instructions \\
        --all-categories --num-images-per-category 2 --image-seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from data.datasets._vlm_tasks import propose_edit_objects, proposal_to_task_rows  # noqa: E402
from data.datasets.sun397.prepare_images import crop_and_save  # noqa: E402
from utils.vlm import DEFAULT_MODEL, make_client  # noqa: E402

DEFAULT_SUN397_ROOT = Path("/data/vision/torralba/datasets/SUN397/SUN397")
TASKS_ROOT = REPO_ROOT / "data" / "tasks"
SOURCE = "sun397"


def _read_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _read_classnames(root: Path) -> list[str]:
    """Return the 397 category paths from ClassName.txt, e.g. '/a/abbey'."""
    classname = root / "ClassName.txt"
    assert classname.exists(), f"Missing {classname}"
    out: list[str] = []
    for line in classname.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(line)
    return out


def _slug(category_path: str) -> str:
    """Map '/a/airport_terminal' -> 'a_airport_terminal' (filesystem-safe)."""
    s = category_path.lstrip("/")
    return s.replace("/", "_").replace(" ", "_").replace("-", "_")


def _pick_categories(all_cats: list[str], n: int, seed: int | None) -> list[str]:
    """Even-stride pick by default; seeded random sample if seed is set."""
    assert 1 <= n <= len(all_cats), (
        f"--num-locations {n} not in [1, {len(all_cats)}]"
    )
    cats = sorted(all_cats)
    if seed is None:
        return [cats[int(i * len(cats) / n)] for i in range(n)]
    return sorted(random.Random(seed).sample(cats, n))


def _first_image(root: Path, category_path: str) -> Path | None:
    """First (alphabetically) JPG under root + category_path."""
    cat_dir = root / category_path.lstrip("/")
    if not cat_dir.is_dir():
        return None
    jpgs = sorted(cat_dir.glob("*.jpg"))
    return jpgs[0] if jpgs else None


def _stable_seed(image_seed: int, category_path: str) -> int:
    """Process-stable seed derived from ``image_seed`` + category. Python's
    builtin ``hash()`` is randomized per process, so we use sha256 instead."""
    h = hashlib.sha256(f"{image_seed}|{category_path}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _pick_n_images(
    root: Path, category_path: str, n: int, *, image_seed: int
) -> list[Path]:
    """Pick ``n`` random JPGs under ``root + category_path``, seeded per-category.

    Returns at most ``n`` paths; fewer if the category has fewer JPGs. Per-
    category seeding (``image_seed`` mixed with the category path via sha256)
    means re-running with the same ``image_seed`` is deterministic across
    processes and changing one category's image set doesn't reshuffle every
    other category's picks.
    """
    cat_dir = root / category_path.lstrip("/")
    if not cat_dir.is_dir():
        return []
    jpgs = sorted(cat_dir.glob("*.jpg"))
    if not jpgs:
        return []
    rng = random.Random(_stable_seed(image_seed, category_path))
    k = min(n, len(jpgs))
    return rng.sample(jpgs, k=k)


def _crop_and_save_per_bucket(
    *,
    src_image: Path,
    image_basename: str,
    buckets: tuple[str, ...],
    dry_run: bool,
) -> tuple[dict[str, str], tuple[int, int]] | None:
    """Crop ``src_image`` once via the shared :func:`crop_and_save` and write
    it into each bucket's ``images/`` dir.

    Returns ``(rel_paths_by_bucket, (w, h))`` where (w, h) are the effective
    dims, or ``None`` if the cropped image is below the patch size.
    """
    dest_paths = [TASKS_ROOT / b / "images" / image_basename for b in buckets]
    result = crop_and_save(src_image=src_image, dest_paths=dest_paths, dry_run=dry_run)
    if result is None:
        return None
    rel_paths = {
        b: dest.relative_to(REPO_ROOT).as_posix()
        for b, dest in zip(buckets, dest_paths)
    }
    return rel_paths, result


def _existing_task_id_stems(buckets: tuple[str, ...]) -> set[str]:
    """Return the set of task_id stems already covered by ``data/tasks/<bucket>/tasks.jsonl``.

    A "stem" is the part of the task_id after the bucket prefix, e.g. for
    ``add_sun397_a_abbey_rowboat`` the stem is ``sun397_a_abbey_rowboat``. We
    use this to short-circuit images whose task_id_stem (per-image, not per-
    object) is already present — but since the stem here is per-image (e.g.
    ``sun397_a_abbey__sun_aa...``), we conservatively skip an image when ANY
    row whose task_id starts with ``<bucket>_<stem>`` already exists.
    """
    stems: set[str] = set()
    for bucket in buckets:
        path = TASKS_ROOT / bucket / "tasks.jsonl"
        if not path.exists():
            continue
        prefix = f"{bucket}_"
        for row in _read_existing_rows(path):
            tid = row.get("task_id", "")
            if tid.startswith(prefix):
                stems.add(tid[len(prefix):])
    return stems


def _stem_already_covered(image_stem: str, existing_stems: set[str]) -> bool:
    """True if any existing task_id stem starts with ``image_stem`` + ``_``.

    Image-level stem: ``sun397_<cat_slug>__<img_stem>``. Task-level stems
    append the per-object slug, e.g. ``sun397_<cat_slug>__<img_stem>_chair``.
    """
    needle = image_stem + "_"
    return any(s.startswith(needle) for s in existing_stems)


async def _vlm_walk(
    *,
    root: Path,
    chosen_cats: list[str],
    images_per_category: int,
    image_seed: int,
    dry_run: bool,
    limit: int | None,
    model: str,
    concurrency: int,
) -> tuple[dict[str, list[dict]], list[dict], list[dict], int]:
    """Returns ({bucket: new_rows}, image_summary_rows, vlm_failures, n_skipped)."""
    new_rows: dict[str, list[dict]] = {"add": [], "remove": []}
    failures: list[dict] = []
    image_summary: list[dict] = []

    write_buckets: tuple[str, ...] = ("add", "remove")
    existing_stems = _existing_task_id_stems(write_buckets)

    pending: list[tuple[str, Path, str, str, dict[str, str], tuple[int, int]]] = []
    threshold_px = int(1.5 * 1024 * 1024)
    n_skipped = 0
    for cat in chosen_cats:
        src_images = _pick_n_images(root, cat, images_per_category, image_seed=image_seed)
        if not src_images:
            failures.append({"category": cat, "attempts": 0, "error": "no jpg in category dir"})
            continue
        cat_slug = _slug(cat)
        for src_image in src_images:
            image_stem = f"sun397_{cat_slug}__{src_image.stem}"
            image_basename = f"{image_stem}.jpg"
            if _stem_already_covered(image_stem, existing_stems):
                n_skipped += 1
                continue
            result = _crop_and_save_per_bucket(
                src_image=src_image,
                image_basename=image_basename,
                buckets=write_buckets,
                dry_run=dry_run,
            )
            if result is None:
                failures.append({
                    "category": cat,
                    "src_filename": src_image.name,
                    "attempts": 0,
                    "error": "image too small after crop",
                })
                continue
            rel_paths, (w, h) = result
            image_summary.append({
                "category": cat,
                "cat_slug": cat_slug,
                "image_stem": image_stem,
                "src_filename": src_image.name,
                "width": w,
                "height": h,
                "pixels": w * h,
                "exceeds_routing_threshold": (w * h) > threshold_px,
            })
            pending.append((cat, src_image, cat_slug, image_stem, rel_paths, (w, h)))

    if limit is not None:
        pending = pending[:limit]

    n_big = sum(1 for s in image_summary if s["exceeds_routing_threshold"])
    print(
        f"\n[sun397] {len(chosen_cats)} categories chosen, "
        f"{len(pending)} images ready to call VLM on, {n_skipped} skipped (already extracted)"
        + (f" (limited to {limit})" if limit is not None else "")
        + f"; {n_big}/{len(image_summary)} exceed 1.5x1024^2 px (routing threshold)."
    )
    if n_big > 0:
        print(
            "  -> use --cluster csail (vision-shared-h200,h100), NOT csail_any "
            "(smaller cards would OOM on the BIG ones)."
        )
        for s in image_summary:
            if s["exceeds_routing_threshold"]:
                print(
                    f"     BIG: {s['category']} {s['src_filename']} "
                    f"{s['width']}x{s['height']} ({s['pixels']} px)"
                )

    if dry_run or not pending:
        return new_rows, image_summary, failures, n_skipped

    client = make_client()
    sem = asyncio.Semaphore(concurrency)

    async def _one(idx: int) -> None:
        cat, src, _cat_slug, image_stem, rel_paths, image_size = pending[idx]
        # Use the bucket-canonical image; "add" if generated, otherwise the first
        # bucket we wrote (always a valid path for the cropped image).
        canonical_bucket = "add" if "add" in rel_paths else next(iter(rel_paths))
        canonical_image_path = REPO_ROOT / rel_paths[canonical_bucket]
        async with sem:
            proposal, error, in_tok, out_tok = await propose_edit_objects(
                client, canonical_image_path, model=model,
            )
        if proposal is None:
            failures.append({
                "category": cat,
                "image_stem": image_stem,
                "src_filename": src.name,
                "attempts": 3,
                "error": error,
                "in_tok": in_tok,
                "out_tok": out_tok,
            })
            return
        rows = proposal_to_task_rows(
            proposal=proposal,
            task_id_stem=image_stem,
            source=SOURCE,
            rel_image_paths=rel_paths,
            image_size=image_size,
            metadata_extra={
                "sun397_category": cat,
                "sun397_src_filename": src.name,
            },
        )
        for row in rows:
            new_rows[row["edit_type"]].append(row)

    await asyncio.gather(*(_one(i) for i in range(len(pending))))
    return new_rows, image_summary, failures, n_skipped


def _write_bucket_merge_by_task_id(bucket: str, *, new_rows: list[dict]) -> int:
    """Merge ``new_rows`` into ``data/tasks/<bucket>/tasks.jsonl`` by task_id.

    Existing rows are preserved; rows in ``new_rows`` whose task_id already
    appears overwrite the matching existing row in place; otherwise they are
    appended in input order. Other sources (and the inactive flag of
    pre-existing rows) are untouched.
    """
    out_path = TASKS_ROOT / bucket / "tasks.jsonl"
    existing = _read_existing_rows(out_path)
    by_id_pos: dict[str, int] = {r["task_id"]: i for i, r in enumerate(existing)}
    merged = list(existing)
    for new in new_rows:
        if new["task_id"] in by_id_pos:
            merged[by_id_pos[new["task_id"]]] = new
        else:
            by_id_pos[new["task_id"]] = len(merged)
            merged.append(new)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in merged:
            f.write(json.dumps(t, sort_keys=True) + "\n")
    return len(merged)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--root", type=Path, default=DEFAULT_SUN397_ROOT,
                   help=f"SUN397 root (default: {DEFAULT_SUN397_ROOT})")
    p.add_argument("--num-locations", type=int, default=20,
                   help="How many distinct categories to sample (default 20). "
                        "Ignored if --all-categories is set.")
    p.add_argument("--all-categories", action="store_true",
                   help="Use all 397 categories. Overrides --num-locations and --seed.")
    p.add_argument("--seed", type=int, default=None,
                   help="If set, switch from even-stride to seeded random "
                        "*category* sample. Independent from --image-seed.")
    p.add_argument("--num-images-per-category", type=int, default=1, metavar="N",
                   help="Number of JPGs to randomly pick from each chosen "
                        "category (default 1).")
    p.add_argument("--image-seed", type=int, default=42,
                   help="Per-category seed for random image selection (default 42). "
                        "Same value -> same picks across runs.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap NEW VLM calls (uncapped by default). Sample-first runs.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    load_dotenv(REPO_ROOT / ".env")

    assert args.root.is_dir(), (
        f"SUN397 root not found: {args.root}\n"
        f"This dataset lives on the CSAIL Torralba NFS mount. Run from CSAIL,\n"
        f"or pass --root pointing at a local checkout."
    )
    assert args.num_images_per_category >= 1, (
        f"--num-images-per-category must be >= 1, got {args.num_images_per_category}"
    )

    all_cats = _read_classnames(args.root)
    if args.all_categories:
        chosen_cats = sorted(all_cats)
        selection_method = "all_categories"
    else:
        chosen_cats = _pick_categories(all_cats, args.num_locations, args.seed)
        selection_method = "seeded_random" if args.seed is not None else "even_stride"
    print(
        f"[sun397] picked {len(chosen_cats)}/{len(all_cats)} categories via "
        f"{selection_method}, {args.num_images_per_category} image(s)/cat "
        f"(image_seed={args.image_seed}):"
    )
    if len(chosen_cats) <= 30:
        for cat in chosen_cats:
            print(f"  {cat}")
    else:
        print(f"  (omitting list -- {len(chosen_cats)} categories)")

    new_rows, image_summary, failures, n_skipped = asyncio.run(_vlm_walk(
        root=args.root,
        chosen_cats=chosen_cats,
        images_per_category=args.num_images_per_category,
        image_seed=args.image_seed,
        dry_run=args.dry_run,
        limit=args.limit,
        model=args.model,
        concurrency=args.concurrency,
    ))

    n_add = len(new_rows["add"])
    n_remove = len(new_rows["remove"])
    print(f"[sun397] new rows: add={n_add}, remove={n_remove}, "
          f"failures={len(failures)}, skipped={n_skipped}")
    for f in failures[:5]:
        print(f"  fail {f.get('category')} ({f.get('src_filename', '?')}): "
              f"{f.get('error')}")

    if args.dry_run:
        return 0

    add_n = _write_bucket_merge_by_task_id("add", new_rows=new_rows["add"])
    remove_n = _write_bucket_merge_by_task_id("remove", new_rows=new_rows["remove"])
    print(f"  wrote add/tasks.jsonl ({add_n} total)")
    print(f"  wrote remove/tasks.jsonl ({remove_n} total)")

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = REPO_ROOT / "data" / "datasets" / "sun397" / f"_extract_log_{ts}.json"
    with open(log_path, "w") as f:
        json.dump({
            "source": SOURCE,
            "root": str(args.root),
            "num_classes": len(all_cats),
            "num_locations_chosen": len(chosen_cats),
            "selection_method": selection_method,
            "seed": args.seed,
            "image_seed": args.image_seed,
            "num_images_per_category": args.num_images_per_category,
            "model": args.model,
            "categories": chosen_cats,
            "image_summary": image_summary,
            "count_add": n_add,
            "count_remove": n_remove,
            "count_skipped": n_skipped,
            "failures": failures,
        }, f, indent=2)
    print(f"  wrote {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
