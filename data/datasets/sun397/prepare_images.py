"""Crop + lanczos-resize SUN397 source JPGs into ``data/tasks/{add,remove}/images/``.

Validates that ``--root`` is a SUN397 checkout (looks for ``ClassName.txt``),
then walks the already-committed ``data/tasks/{add,remove}/tasks.jsonl``
files, takes every row whose ``source == "sun397"``, locates its source JPG
under ``--root``, and applies the per-image pipeline that
:class:`Flux2KleinPipeline` expects: center-crop to a multiple of 16, then
lanczos-downscale to the pipeline's effective ref dims (≤1M px).

SUN397 is research-only and has no scriptable download; grab the tarball from
https://vision.princeton.edu/projects/2010/SUN/ and point ``--root`` at the
extracted directory. On CSAIL the Torralba NFS mount holds it already at the
default path.

This is the **end-user** reproduction path — the 789 add + 726 remove task
instructions already ship in this repo, so no Anthropic API call is needed.
The original VLM-driven authoring path (which generated those rows) lives at
``extract_instructions.py``.

Usage:
    uv run python -m data.datasets.sun397.prepare_images --root <path-to-SUN397>
    uv run python -m data.datasets.sun397.prepare_images --root <path-to-SUN397> --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.datasets._image_utils import PATCH, center_crop_to_multiple  # noqa: E402
from utils.flux2_klein import effective_ref_dims  # noqa: E402

DEFAULT_SUN397_ROOT = Path("/data/vision/torralba/datasets/SUN397/SUN397")
TASKS_ROOT = REPO_ROOT / "data" / "tasks"
SOURCE = "sun397"
BUCKETS = ("add", "remove")


def read_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def crop_and_save(
    *,
    src_image: Path,
    dest_paths: list[Path],
    dry_run: bool,
) -> tuple[int, int] | None:
    """Center-crop ``src_image`` to a multiple of 16, lanczos-downscale to
    Flux2KleinPipeline's effective ref dims (≤1M px), and write the resulting
    JPG to every path in ``dest_paths``.

    Returns the saved (w, h), or ``None`` if the cropped image is below the
    patch size and should be dropped. Existing destination JPGs are not
    overwritten.
    """
    with Image.open(src_image) as im:
        im = im.convert("RGB")
        im = center_crop_to_multiple(im)
        w, h = im.size
        if w < PATCH or h < PATCH:
            return None
        eff_h, eff_w = effective_ref_dims(h, w)
        if (eff_w, eff_h) != (w, h):
            im = im.resize((eff_w, eff_h), Image.LANCZOS)
            w, h = eff_w, eff_h
        for dest in dest_paths:
            if dry_run:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                im.save(dest, format="JPEG", quality=95)
    return (w, h)


def _collect_targets() -> dict[tuple[str, str], list[Path]]:
    """Walk committed sun397 rows; return ``{(category, src_filename): [dest_paths]}``.

    Multiple rows often share the same source image (e.g. the same JPG appears
    in both ``add/`` and ``remove/`` buckets), so we group by source to crop
    each input JPG exactly once.
    """
    targets: dict[tuple[str, str], list[Path]] = {}
    for bucket in BUCKETS:
        for row in read_jsonl_rows(TASKS_ROOT / bucket / "tasks.jsonl"):
            if row.get("source") != SOURCE:
                continue
            meta = row.get("metadata", {})
            cat = meta.get("sun397_category")
            src_filename = meta.get("sun397_src_filename")
            assert cat and src_filename, (
                f"sun397 row {row.get('task_id')!r} missing "
                f"metadata.sun397_category / metadata.sun397_src_filename"
            )
            dest = REPO_ROOT / row["source_image_path"]
            targets.setdefault((cat, src_filename), []).append(dest)
    return targets


def main() -> int:
    p = argparse.ArgumentParser(
        description="Crop SUN397 source JPGs referenced by the committed "
                    "sun397 task rows. No Anthropic API key required.",
    )
    p.add_argument("--root", type=Path, default=DEFAULT_SUN397_ROOT,
                   help=f"SUN397 root directory (default: {DEFAULT_SUN397_ROOT}).")
    p.add_argument("--dry-run", action="store_true",
                   help="List planned crops without writing any JPGs.")
    args = p.parse_args()

    classname = args.root / "ClassName.txt"
    assert args.root.is_dir() and classname.exists(), (
        f"SUN397 not found at {args.root} (missing {classname.name}).\n"
        f"Download from https://vision.princeton.edu/projects/2010/SUN/ and "
        f"pass --root pointing at the extracted directory."
    )
    n_classes = sum(1 for line in classname.read_text().splitlines() if line.strip())
    print(f"[sun397.prepare_images] mounted at {args.root} ({n_classes} categories).")

    targets = _collect_targets()
    print(f"[sun397.prepare_images] {len(targets)} unique source images "
          f"referenced by committed sun397 rows.")

    n_done = n_skipped = n_missing = n_too_small = 0
    for (cat, src_filename), dest_paths in sorted(targets.items()):
        src_image = args.root / cat.lstrip("/") / src_filename
        if not src_image.is_file():
            n_missing += 1
            print(f"  MISSING source: {src_image}")
            continue
        if all(d.exists() for d in dest_paths):
            n_skipped += 1
            continue
        result = crop_and_save(
            src_image=src_image,
            dest_paths=dest_paths,
            dry_run=args.dry_run,
        )
        if result is None:
            n_too_small += 1
            print(f"  TOO SMALL after crop: {src_image}")
            continue
        n_done += 1

    verb = "would crop" if args.dry_run else "cropped"
    print(f"[sun397.prepare_images] {verb} {n_done}, "
          f"skipped {n_skipped} (already present), "
          f"missing source {n_missing}, too-small {n_too_small}.")
    return 1 if n_missing else 0


if __name__ == "__main__":
    sys.exit(main())
