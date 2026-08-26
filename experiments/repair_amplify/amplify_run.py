"""Repair failing i2i edits by *amplifying* cross-modal attention.

The paper's knockout results show reference content reaches the output
through an implicit text-token binding: text queries absorb reference
content (ref -> text flow, strongest at Single-9 for color and MM-7 for
style), and the noise tokens then read it back from the text band. This
script runs the inverse intervention: instead of ``-inf`` at the
(destination, source) attention entries, add a *positive* bias ``+lam``
so those softmax logits are boosted — "pay more attention" — on a chosen
set of blocks, reusing the knockout SDPA-mask processors unchanged.

Directions (dest reads from src; bias at rows=dest queries, cols=src keys):

* ``text_from_ref``   — text tokens attend more to reference tokens
                        (strengthen binding formation; the paper's row).
* ``image_from_text`` — noise tokens attend more to text tokens
                        (strengthen binding readout).
* ``image_from_ref``  — noise tokens attend more to reference tokens
                        (direct route bypassing the binding).

``--dest-subset padding|content`` restricts the destination rows of
``text_from_ref`` to the instruction's padding / content token positions.

Outputs land flat under ``results/repair_amplify/runs/<task_id>/``:

    reference.png
    baseline_4step.png                       # unmodified i2i (same seed)
    amp_<direction>[_<subset>]_<blocks>_lam<x>.png

Usage::

    uv run python -m experiments.repair_amplify.amplify_run \\
        --task-id solid_green_mug_s2 solid_red_couch_s1 \\
        --blocks single_mm9 --lam 1 2 4 --direction text_from_ref
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.attention_knockout.knockout_processors import (
    install_knockout_processors,
)
from experiments.attention_knockout.masks import clear_all_masks
from experiments.common.baselines import load_or_make_reference
from experiments.common.file_cache import load_or_run
from experiments.common.tasks import get_task
from experiments.patching.utils import resolve_content_token_indices
from utils.flux2_klein import (
    ALL_BLOCK_NAMES,
    Flux2KleinModel,
    block_index_from_suffix,
    block_suffix,
    layout_for,
)
from utils.token_layout import TokenLayout, get_category_slices

OUT_ROOT = Path("results/repair_amplify/runs")

# direction name -> (destination category, source category); the bias sits at
# rows = destination queries, cols = source keys, matching masks.py semantics.
DIRECTIONS: dict[str, tuple[str, str]] = {
    "text_from_ref": ("text", "ref"),
    "image_from_text": ("image", "text"),
    "image_from_ref": ("image", "ref"),
}


def build_amplify_mask(
    layout: TokenLayout,
    direction: str,
    lam: float,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    dest_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Additive ``(1, 1, S, S)`` mask: ``+lam`` at (dest, src), ``0`` elsewhere.

    ``dest_rows`` (1D long tensor, joint-stream coords) overrides the
    destination band — used for the padding/content text subsets. Positive
    entries only, so the knockout builders' NaN-row concern cannot arise.
    """
    assert direction in DIRECTIONS, f"unknown direction {direction!r}"
    assert lam > 0, f"lam must be positive, got {lam}"
    dest_cat, src_cat = DIRECTIONS[direction]
    slices = get_category_slices(layout)
    assert "ref" in slices, "amplification requires an i2i layout with ref tokens"
    mask = torch.zeros(
        (1, 1, layout.total, layout.total), device=device, dtype=torch.float32
    )
    dest_indexer = slices[dest_cat] if dest_rows is None else dest_rows.to(device)
    mask[:, :, dest_indexer, slices[src_cat]] = float(lam)
    return mask.to(dtype)


def apply_mask_to_blocks(
    procs: dict[str, object],
    mask: torch.Tensor | None,
    block_indices: set[int],
) -> None:
    """Install ``mask`` on exactly ``block_indices``; clear every other block."""
    for i, name in enumerate(ALL_BLOCK_NAMES):
        procs[name]._mask = mask if (mask is not None and i in block_indices) else None


def _fmt_lam(lam: float) -> str:
    return f"{lam:g}".replace(".", "p")


def _blocks_tag(suffixes: list[str]) -> str:
    return "+".join(suffixes)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--task-id", nargs="+", required=True)
    ap.add_argument(
        "--blocks", nargs="+", required=True,
        help="block suffixes amplified together, e.g. mm7 single_mm9",
    )
    ap.add_argument("--lam", nargs="+", type=float, required=True)
    ap.add_argument(
        "--direction", nargs="+", default=["text_from_ref"],
        choices=sorted(DIRECTIONS),
    )
    ap.add_argument(
        "--combine", action="store_true",
        help="sum the masks of all --direction values into ONE bias per lam "
             "(destination bands are disjoint, so the sum is a union) instead "
             "of one output per direction; file tag 'combo-<d1>-<d2>'",
    )
    ap.add_argument(
        "--dest-subset", default="all", choices=["all", "padding", "content"],
        help="restrict text_from_ref destination rows to padding/content "
             "text positions (ignored for other directions)",
    )
    ap.add_argument("--num-inference-steps", type=int, default=4)
    args = ap.parse_args()

    block_indices = {block_index_from_suffix(s) for s in args.blocks}
    suffixes = [block_suffix(ALL_BLOCK_NAMES[i]) for i in sorted(block_indices)]
    tag = _blocks_tag(suffixes)

    model = Flux2KleinModel()
    procs, _original = install_knockout_processors(model.transformer)
    mask_dtype = next(model.transformer.parameters()).dtype

    for task_id in args.task_id:
        task = get_task(task_id)
        assert task.noise_seed is not None, f"{task_id}: no noise_seed"
        out_dir = OUT_ROOT / task_id
        out_dir.mkdir(parents=True, exist_ok=True)

        ref_img = load_or_make_reference(model, task)
        if not (out_dir / "reference.png").exists():
            ref_img.save(out_dir / "reference.png")
        ref_w, ref_h = ref_img.size
        layout = layout_for(task.height, task.width, ref_h=ref_h, ref_w=ref_w)

        def gen():
            return model.generate(
                task.instruction,
                seed=task.noise_seed,
                num_inference_steps=args.num_inference_steps,
                image=ref_img,
                height=task.height,
                width=task.width,
            )

        clear_all_masks(procs)
        load_or_run(out_dir / "baseline_4step.png", generate=gen)

        dest_rows: torch.Tensor | None = None
        subset_tag = ""
        if args.dest_subset != "all":
            positions = resolve_content_token_indices(model.pipe, task.instruction)
            content = torch.zeros(layout.text_seq_len, dtype=torch.bool)
            content[[i for i, _ in positions]] = True
            selector = content if args.dest_subset == "content" else ~content
            dest_rows = selector.nonzero(as_tuple=True)[0]
            subset_tag = f"_{args.dest_subset}"

        meta = {
            "task_id": task_id,
            "instruction": task.instruction,
            "noise_seed": task.noise_seed,
            "blocks": suffixes,
            "block_indices": sorted(block_indices),
            "lams": args.lam,
            "directions": args.direction,
            "dest_subset": args.dest_subset,
            "num_inference_steps": args.num_inference_steps,
            "total_seq_len": layout.total,
        }
        with open(out_dir / f"amplify_meta_{tag}.json", "w") as f:
            json.dump(meta, f, indent=2)

        direction_runs: list[tuple[str, list[str]]] = (
            [("combo-" + "-".join(args.direction), list(args.direction))]
            if args.combine
            else [(d, [d]) for d in args.direction]
        )
        for run_name, run_dirs in direction_runs:
            sub = subset_tag if "text_from_ref" in run_dirs else ""
            for lam in args.lam:
                fname = f"amp_{run_name}{sub}_{tag}_lam{_fmt_lam(lam)}.png"
                if (out_dir / fname).exists():
                    print(f"[skip] {task_id}/{fname}")
                    continue
                t0 = time.time()
                mask = sum(
                    build_amplify_mask(
                        layout, d, lam,
                        device=model.device, dtype=mask_dtype,
                        dest_rows=(dest_rows if d == "text_from_ref" else None),
                    )
                    for d in run_dirs
                )
                apply_mask_to_blocks(procs, mask, block_indices)
                img = gen()
                clear_all_masks(procs)
                torch.cuda.empty_cache()
                img.save(out_dir / fname)
                print(f"[amp] {task_id}/{fname} ({time.time() - t0:.1f}s)",
                      flush=True)

    print("[amplify_run] done")


if __name__ == "__main__":
    main()
