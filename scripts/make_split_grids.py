"""Build a per-task image grid for split-schedule knockout results.

For every task subdirectory under a results subdir (e.g. ``ref_cutoff_sweep``
or ``ref_split_schedule``), lay out the reference image, the i2i/t2i baselines,
and every ``split_at_<block>_full_ko.png`` in cutoff order into one grid,
written as ``split_grid.png`` inside that same subdirectory.

Cutoff cells are ordered the way the network runs — MM blocks
(``transformer_blocks.N``) first, then single-stream blocks
(``single_transformer_blocks.N``) — so the grid reads as a sweep of the
prefix/suffix boundary from early to late. The reference and baselines are
highlighted so they stand apart from the swept cutoffs.

Usage::

    uv run python scripts/make_split_grids.py ref_cutoff_sweep
    uv run python scripts/make_split_grids.py ref_split_schedule ref_suffix_only
    uv run python scripts/make_split_grids.py ref_cutoff_sweep --ncols 9
    uv run python scripts/make_split_grids.py ref_cutoff_sweep \\
        --only solid_green_chair_s0 --circle single_transformer_blocks.9 \\
        --out-name split_grid.pdf
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image

from utils.scoring import create_image_grid

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results_v4" / "attention_knockout"

# split_at_<block>_full_ko.png  ->  capture <block>
_SPLIT_RE = re.compile(r"^split_at_(.+)_full_ko\.png$")


def _block_sort_key(block: str) -> tuple[int, int]:
    """Order block names the way the transformer runs: the MM blocks
    (``transformer_blocks.N``) before the single-stream blocks
    (``single_transformer_blocks.N``), each by numeric index."""
    stream, idx = block.rsplit(".", 1)
    is_single = stream == "single_transformer_blocks"
    return (1 if is_single else 0, int(idx))


def _block_label(block: str) -> str:
    """``single_transformer_blocks.3`` -> ``Single 3``;
    ``transformer_blocks.5`` -> ``MM 5``."""
    stream, idx = block.rsplit(".", 1)
    kind = "Single" if stream == "single_transformer_blocks" else "MM"
    return f"{kind} {idx}"


def _collect_cells(task_dir: Path) -> tuple[list[Path], list[str], int]:
    """Return ``(image_paths, titles, n_prepend)`` for one task dir.

    The prepended cells are the reference and the two baselines (highlighted
    in the grid); the rest are the split cutoffs in run order.
    """
    prepend: list[tuple[Path, str]] = []
    ref = task_dir / "reference.png"
    if ref.exists():
        prepend.append((ref, "Reference"))
    for matches, label in (
        (sorted(task_dir.glob("i2i_baseline_*step.png")), "Clean I2I"),
        (sorted(task_dir.glob("t2i_clean_*step.png")), "Clean T2I"),
    ):
        if matches:
            prepend.append((matches[0], label))

    splits: list[tuple[Path, str]] = []
    for p in task_dir.glob("split_at_*_full_ko.png"):
        m = _SPLIT_RE.match(p.name)
        if m:
            splits.append((p, m.group(1)))
    splits.sort(key=lambda pb: _block_sort_key(pb[1]))

    paths = [p for p, _ in prepend] + [p for p, _ in splits]
    titles = (
        [t for _, t in prepend]
        + [f"cut@{_block_label(b)}" for _, b in splits]
    )
    return paths, titles, len(prepend)


def _suptitle(task_dir: Path) -> str:
    """The edit instruction from ``task_metadata.json``; task id if absent."""
    meta = task_dir / "task_metadata.json"
    if meta.exists():
        try:
            instruction = json.loads(meta.read_text()).get("instruction")
        except (json.JSONDecodeError, OSError):
            instruction = None
        if instruction:
            return f'"{instruction}"'
    return task_dir.name


def make_grid(
    task_dir: Path, ncols: int, out_name: str,
    fontsize: float, cell_size: float, circle: str | None,
) -> bool:
    """Write one grid for ``task_dir``; return True iff a grid was written."""
    paths, titles, n_prepend = _collect_cells(task_dir)
    n_splits = len(paths) - n_prepend
    if n_splits == 0:
        print(f"  {task_dir.name}: no split_at_*.png, skipped")
        return False
    circle_indices = None
    if circle is not None:
        circle_label = f"cut@{_block_label(circle)}"
        assert circle_label in titles, (
            f"{task_dir.name}: --circle block {circle!r} ({circle_label}) "
            f"not among grid cells"
        )
        circle_indices = [titles.index(circle_label)]
    images = [Image.open(p) for p in paths]
    out_path = task_dir / out_name
    create_image_grid(
        images, titles, str(out_path),
        ncols=ncols,
        cell_size=cell_size,
        fontsize=fontsize,
        highlight_indices=list(range(n_prepend)),
        circle_indices=circle_indices,
        suptitle=_suptitle(task_dir),
    )
    for im in images:
        im.close()
    print(
        f"  {task_dir.name}: {n_prepend} baseline + {n_splits} cutoff "
        f"cells -> {out_path.name}"
    )
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "subdirs", nargs="+",
        help="Results subdir name(s) under --results-root, e.g. 'ref_cutoff_sweep'.",
    )
    ap.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help=f"Parent of the subdirs (default: {DEFAULT_RESULTS_ROOT}).",
    )
    ap.add_argument(
        "--ncols", type=int, default=8,
        help="Columns in each grid (default: 8).",
    )
    ap.add_argument(
        "--fontsize", type=float, default=28,
        help="Cell-label and title font size (default: 28).",
    )
    ap.add_argument(
        "--cell-size", type=float, default=4.5,
        help="Per-cell size in inches (default: 4.5).",
    )
    ap.add_argument(
        "--circle", default=None,
        help="Ring the cell for this cutoff block, e.g. "
             "'single_transformer_blocks.9' -> circles the 'cut@Single 9' cell.",
    )
    ap.add_argument(
        "--only", action="append", default=None, metavar="TASK_ID",
        help="Only build grids for these task dir name(s); repeatable.",
    )
    ap.add_argument(
        "--out-name", default="split_grid.png",
        help="Grid filename written into each task dir (default: split_grid.png).",
    )
    args = ap.parse_args()

    total = 0
    for subdir in args.subdirs:
        root = args.results_root / subdir
        if not root.is_dir():
            raise SystemExit(f"not a directory: {root}")
        print(f"[{subdir}]")
        made = 0
        task_dirs = sorted(
            p for p in root.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )
        if args.only:
            wanted = set(args.only)
            task_dirs = [p for p in task_dirs if p.name in wanted]
        for task_dir in task_dirs:
            if make_grid(
                task_dir, args.ncols, args.out_name,
                args.fontsize, args.cell_size, args.circle,
            ):
                made += 1
        print(f"  {made} grid(s) written")
        total += made
    print(f"done: {total} grid(s)")


if __name__ == "__main__":
    main()
