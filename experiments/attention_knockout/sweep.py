"""Shared mode x L knockout sweep helper.

Used by the i2i driver (``knockout_run.py``). Encapsulates: install mask on
selected blocks, generate, save per-L image, clear masks, build the comparison
grid.

Callers supply a ``generate_fn`` (closure over their model + prompt + seed and
any other generation kwargs like ``image=`` for i2i) and the prefix images +
labels they want at the left of the grid (e.g. baselines, reference).
"""

from __future__ import annotations

import os
from typing import Callable, Iterable

import torch
from PIL import Image

from experiments.attention_knockout.masks import (
    LayerMode,
    apply_mask_to_layers,
    clear_all_masks,
)
from utils.scoring import create_image_grid

__all__ = ["run_layer_sweep", "describe_selection", "format_sweep_labels"]


def run_layer_sweep(
    *,
    procs: dict[str, object],
    mask: torch.Tensor,
    mode: LayerMode,
    L_range: Iterable[int],
    ordered_block_names: list[str],
    block_labels: list[str],
    window_size: int | None,
    generate_fn: Callable[[], Image.Image],
    prepend_images: list[Image.Image],
    prepend_labels: list[str],
    out_dir: str,
    suptitle: str,
    append_images: list[Image.Image] | None = None,
    append_labels: list[str] | None = None,
    grid_ncols: int = 8,
) -> list[Image.Image]:
    """Install ``mask`` on each ``(mode, L)`` selection, generate, save grid.

    Args:
        procs: per-block knockout processors keyed by block name.
        mask: the additive attention mask to install. Same mask is reused for
            every L; only the *selection* of blocks changes.
        mode: layer-selection mode (``"prefix"``, ``"suffix"``, ``"individual"``,
            ``"window"``).
        L_range: iterable of L values to sweep. Caller chooses the range
            (e.g. ``range(num_blocks)`` for prefix/suffix/individual, or
            ``range(num_blocks - k + 1)`` for window).
        ordered_block_names: block names in order matching ``block_labels``.
        block_labels: human-readable labels per block, used in grid titles.
        window_size: only meaningful when ``mode == "window"``.
        generate_fn: zero-arg callable returning a PIL image. Caller closes
            over the model, prompt, seed, and any other generation kwargs
            (e.g. ``image=ref_img`` for i2i).
        prepend_images: images to prepend to the grid (e.g. ``[ref, baseline]``
            for i2i, ``[baseline]`` for t2i). Highlighted in the grid.
        prepend_labels: labels for the prepended images.
        out_dir: directory where ``L{L:02d}.png`` and ``grid.png`` are saved.
            Created if missing.
        suptitle: grid suptitle.
        append_images / append_labels: extra cells appended after the swept
            cells (e.g. the ``--all-layers-4step`` "4-step full KO" cell).
            Both must be provided together; both must be the same length.
        grid_ncols: columns in the output grid.

    Returns:
        The list of swept images (one per L), in ``L_range`` order.
    """
    if append_images is None:
        append_images = []
    if append_labels is None:
        append_labels = []
    assert len(append_images) == len(append_labels), (
        f"append_images ({len(append_images)}) / append_labels "
        f"({len(append_labels)}) length mismatch"
    )

    os.makedirs(out_dir, exist_ok=True)
    L_list = list(L_range)
    swept: list[Image.Image] = []
    for L in L_list:
        apply_mask_to_layers(
            procs, mode, L, ordered_block_names, mask, window_size=window_size,
        )
        print(f"    [L={L:02d}] {describe_selection(mode, L, block_labels, window_size)}")
        img = generate_fn()
        img.save(os.path.join(out_dir, f"L{L:02d}.png"))
        swept.append(img)
        torch.cuda.empty_cache()
    clear_all_masks(procs)

    sweep_labels = format_sweep_labels(mode, L_list, block_labels, window_size)
    grid_path = os.path.join(out_dir, "grid.png")
    create_image_grid(
        prepend_images + swept + list(append_images),
        prepend_labels + sweep_labels + list(append_labels),
        grid_path,
        ncols=grid_ncols,
        highlight_indices=list(range(len(prepend_images))),
        suptitle=suptitle,
    )
    print(f"    grid -> {grid_path}")
    return swept


def describe_selection(
    mode: LayerMode,
    L: int,
    block_labels: list[str],
    window_size: int | None = None,
) -> str:
    """Short human description of which blocks ``(mode, L)`` selects."""
    label = block_labels[L]
    if mode == "suffix":
        return f"suffix [{label}..end)"
    if mode == "prefix":
        return f"prefix [begin..{label}]"
    if mode == "individual":
        return f"only {label}"
    if mode == "window":
        assert window_size is not None
        end_label = block_labels[L + window_size - 1]
        return f"window [{label}..{end_label}]"
    raise AssertionError(f"Unknown layer mode: {mode!r}")


def format_sweep_labels(
    mode: LayerMode,
    L_list: list[int],
    block_labels: list[str],
    window_size: int | None,
) -> list[str]:
    """Per-cell grid labels for a sweep across ``L_list``."""
    if mode == "window":
        assert window_size is not None
        return [
            f"L{L:02d}-L{L + window_size - 1:02d} "
            f"{block_labels[L]}..{block_labels[L + window_size - 1]}"
            for L in L_list
        ]
    return [f"L{L:02d} {block_labels[L]}" for L in L_list]
