"""Headline before/after repair figures: rows = failing tasks, cols =
[reference | baseline | repair cells], chunked into PNGs of --rows-per-fig
rows each.

Usage::

    uv run python -m experiments.repair_amplify.headline_figures \\
        --tasks slurm/repair_amplify/failures_style_all.txt \\
        --cells "amp_text_from_ref_mm5+mm6+mm7_lam4" \\
                "amp_text_from_ref_mm5+mm6+mm7_lam6" \\
        --out-dir results_v4/repair_amplify_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

RUNS_ROOT = Path("results/repair_amplify/runs")
THUMB = 256
LABEL_H = 16
HEADER_H = 20
PREFIX = "customize_property_style_free_"


def cell_label(cell: str) -> str:
    return cell.removeprefix("amp_").replace("text_from_ref", "text<-ref")


def build_figure(tasks: list[str], cells: list[str], out_path: Path) -> None:
    col_specs = [("reference", "reference.png"),
                 ("baseline (fails)", "baseline_4step.png")] + [
                 (cell_label(c), f"{c}.png") for c in cells]
    W = len(col_specs) * THUMB
    H = HEADER_H + len(tasks) * (THUMB + LABEL_H)
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    for c, (label, _) in enumerate(col_specs):
        draw.text((c * THUMB + 4, 4), label, fill="black")
    for r, task_id in enumerate(tasks):
        y = HEADER_H + r * (THUMB + LABEL_H)
        for c, (_, fname) in enumerate(col_specs):
            p = RUNS_ROOT / task_id / fname
            assert p.exists(), f"missing {p}"
            canvas.paste(Image.open(p).convert("RGB").resize((THUMB, THUMB)),
                         (c * THUMB, y))
        draw.text((4, y + THUMB + 1), task_id.removeprefix(PREFIX), fill="black")
    canvas.save(out_path)
    print(f"[headline_figures] {out_path} ({len(tasks)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rows-per-fig", type=int, default=8)
    args = ap.parse_args()

    tasks = [l.strip() for l in args.tasks.read_text().splitlines() if l.strip()]
    assert tasks
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(tasks), args.rows_per_fig):
        chunk = tasks[i : i + args.rows_per_fig]
        n = i // args.rows_per_fig + 1
        build_figure(chunk, args.cells, args.out_dir / f"repair_figure_{n:02d}.png")


if __name__ == "__main__":
    main()
