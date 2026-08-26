"""Plot the padding-token dose-response: transfer success vs padding count.

Reads the 24 padding-ablation judge CSVs (``i2i2i_{color,style}_p<NNN>[...]``,
see ``scripts/judge/configs.py::PADDING_ABLATION_JUDGES``), pools binary
verdicts per (family, level, text-token mode), and renders one panel per
family with one line per mode: success rate vs *actual mean padding-token
count* (494 -> 0 left to right, i.e. padding decreasing as the prompt grows).

If padding tokens are the carriers, ``padding_only`` should fall monotonically
toward the right; a rising ``content_only`` instead indicates spare capacity
rather than padding positions per se.

Error bars are the 95% Wilson score interval on the pooled binary verdicts
(the paper's error-bar convention).

Usage::

    uv run python scripts/plot_padding_ablation.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.i2i_to_i2i_patching.build_padding_ablation_tasks import (
    PADDING_LEVELS,
    level_slug,
)
from scripts.judge.csv_io import load_existing_rows

REPO_ROOT = Path(__file__).resolve().parents[1]
JUDGE_DIR = REPO_ROOT / "results_v4" / "vlm_judge"
TASKS_ROOT = REPO_ROOT / "data" / "tasks"

# (panel title, judge family, task bucket)
FAMILIES = [
    ("Color transfer (Single-9)", "color", "solid_color_padding"),
    ("Style transfer (MM-7)", "style", "style_padding"),
]

# (legend label, judge-name suffix, color, linestyle, marker)
#
# Palette: Okabe-Ito, checked with the dataviz validator over ALL pairs (every
# series is visible at once, so adjacent-only is not enough). The previous
# tab10 blue/red/green FAILed — #2ca02c vs #d62728 is deltaE 3.9 under
# deuteranopia, i.e. the red and green lines were indistinguishable to a
# red-green colorblind reader. This set's worst pair is 6.9, which sits in the
# band that is legal only with a secondary encoding — hence the per-series
# linestyle and marker below (identity is never carried by hue alone).
#
# instruction_only + filler_only partition content_only, so they are dashed
# and content_only is solid: the visual grouping mirrors the decomposition.
MODES = [
    ("all text tokens", "", "#0072B2", "-", "o"),
    ("padding only", "_text_padding", "#D55E00", "-", "s"),
    ("content only", "_text_content", "#009E73", "-", "^"),
    ("instruction only", "_text_instruction", "#6A3D9A", "--", "v"),
    ("filler only", "_text_filler", "#E69F00", "--", "D"),
]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Point estimate + 95% Wilson score interval for ``k`` successes of ``n``.
    Returns ``(p_hat, lo, hi)`` as fractions; ``(0, 0, 0)`` when ``n == 0``."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, center - half, center + half


def _pass_counts(csv_path: Path) -> tuple[int, int]:
    """(n_pass, n_total) over the last verdict per entity; skips error rows."""
    k = n = 0
    for _entity, row in load_existing_rows(csv_path).items():
        verdict = (row.get("pass") or "").strip()
        if verdict not in ("0", "1"):
            continue
        n += 1
        if verdict == "1":
            k += 1
    return k, n


def _mean_actual_padding(bucket: str) -> dict[str, float]:
    """Level slug -> mean realized padding over the bucket's tasks (recorded
    by the task builder; exact for color, a few tokens off target for style)."""
    sums: dict[str, list[int]] = {}
    with open(TASKS_ROOT / bucket / "tasks.jsonl") as f:
        for line in f:
            md = json.loads(line)["metadata"]
            sums.setdefault(md["padding_level"], []).append(md["actual_padding"])
    return {slug: sum(v) / len(v) for slug, v in sums.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge-dir", type=Path, default=JUDGE_DIR)
    ap.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "paper_images" / "fig_padding_ablation",
    )
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(11, 4.2), sharey=True)

    for ax, (title, family, bucket) in zip(axes, FAMILIES):
        x_by_slug = _mean_actual_padding(bucket)
        panel_n = 0
        for label, suffix, color, ls, marker in MODES:
            xs, ys, los, his, ns = [], [], [], [], []
            for target in PADDING_LEVELS:
                slug = level_slug(target)
                csv_path = args.judge_dir / f"i2i2i_{family}_{slug}{suffix}.csv"
                k, n = _pass_counts(csv_path)
                # Skip levels this mode does not define. instruction_only /
                # filler_only are degenerate at p494 (the prompt IS the
                # instruction, so there is no filler to split off) — plotting
                # them as 0% there would invent a data point.
                if n == 0:
                    continue
                p, lo, hi = _wilson(k, n)
                xs.append(x_by_slug[slug])
                ys.append(100 * p)
                # Clamp: at p==0 or p==1 the Wilson bound equals p only up to
                # floating-point error, and errorbar rejects negative yerr.
                los.append(max(0.0, 100 * (p - lo)))
                his.append(max(0.0, 100 * (hi - p)))
                ns.append(n)
            if not xs:
                continue
            n_pairs = max(ns)
            panel_n = max(panel_n, n_pairs)
            ax.errorbar(
                xs, ys, yerr=[los, his], color=color, marker=marker,
                markersize=5, linestyle=ls, capsize=2, linewidth=1.5,
                label=label,
            )
        # Padding decreasing left -> right (the prompt grows).
        ax.set_xlim(512, -18)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Padding tokens (of 512 text slots)")
        # n lives in the panel title: it differs per family (48 vs 50), so a
        # single shared legend cannot carry it honestly.
        ax.set_title(f"{title} — n={panel_n}/level" if panel_n else title)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Successful transfers (%)")
    # One shared legend below the panels: in-axes placement covered the color
    # panel's 100% lines, and the two panels share the same series.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODES),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("i2i→i2i transfer success vs padding-token count (text_seq_len=512)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / "padding_ablation.png"
    pdf = args.out_dir / "padding_ablation.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
