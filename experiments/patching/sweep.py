"""Block sweep + grid creation for activation patching experiments.

Provides ``sweep_and_grid()`` which iterates over blocks, patches one token
category at a time from captured source activations, and produces a comparison
image grid (``grid.png``).

Default sweep is diagonal (``src == dst``). Pass ``fixed_dst_name`` to pin
the destination across iterations (used for input-to-block-0 sweeps, which
hook ``transformer.context_embedder`` via a caller-supplied ``image_producer``).

The per-iteration image production can be swapped out by passing an
``image_producer`` callback; the default closure runs the pipeline with a
patch hook at ``dst_name`` on the block's output.
"""

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from experiments.common.tasks import NUM_INFERENCE_STEPS
from experiments.patching.hooks import (
    make_context_embedder_patch_hook,
    make_patch_hook,
    make_patch_hook_multi_step,
)
from experiments.patching.utils import (
    extract_category_acts,
    extract_category_acts_per_step,
    run_pipeline_with_hooks,
)
from utils.flux2_klein import ALL_BLOCK_LABELS, ALL_BLOCK_NAMES, TokenLayout
from utils.scoring import create_image_grid


# Per-iteration image producer: ``(src_name, dst_name, src_act) -> PIL.Image``.
ImageProducer = Callable[[str, str, torch.Tensor], Image.Image]


def make_patch_pipeline_producer(
    model,
    category: str,
    *,
    layout: TokenLayout,
    target_prompt: str,
    target_seed: int,
    target_h: int,
    target_w: int,
    target_ref_image: Optional[Image.Image] = None,
    callback_on_step_end: Optional[Callable] = None,
    text_token_indices: Optional[Sequence[int]] = None,
) -> ImageProducer:
    """Default producer: patch ``src_act`` at ``dst_name``'s output.

    Reproduces the original inline behavior of ``sweep_and_grid`` as a
    standalone closure. ``text_token_indices`` (text category only) restricts
    the replacement to those positions; ``None`` replaces the full text slice.
    ``layout`` is the *target*-side token layout — its ``has_ref`` selects
    i2i vs t2i hook shape. ``target_h``/``target_w`` are passed explicitly so
    Flux2KleinPipeline doesn't fall back to its 1024² default when ``image=``
    is passed without them.
    """
    def producer(src_name: str, dst_name: str, src_act: torch.Tensor) -> Image.Image:
        hook_fn = make_patch_hook(
            dst_name, src_act, category, layout,
            text_token_indices=text_token_indices,
        )
        generator = torch.Generator(model.device).manual_seed(target_seed)
        pipe_kwargs = dict(
            prompt=target_prompt,
            generator=generator,
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=target_h,
            width=target_w,
        )
        if target_ref_image is not None:
            pipe_kwargs["image"] = target_ref_image
        return run_pipeline_with_hooks(
            model,
            [(dst_name, hook_fn)],
            callback_on_step_end=callback_on_step_end,
            **pipe_kwargs,
        )
    return producer


def make_patch_pipeline_producer_multi_step(
    model,
    category: str,
    *,
    layout: TokenLayout,
    target_prompt: str,
    target_seed: int,
    target_h: int,
    target_w: int,
    target_ref_image: Optional[Image.Image] = None,
    callback_on_step_end: Optional[Callable] = None,
    text_token_indices: Optional[Sequence[int]] = None,
    num_inference_steps: int,
) -> ImageProducer:
    """Multi-step variant of :func:`make_patch_pipeline_producer`.

    ``src_act`` passed to the producer is interpreted as a sequence of per-step
    source activations (length == ``num_inference_steps``). The hook closure
    advances an internal step counter on each call so step ``k`` of the target
    pipeline is patched with ``src_act[k]``.
    """
    assert num_inference_steps >= 1, (
        f"num_inference_steps must be >= 1, got {num_inference_steps}"
    )

    def producer(src_name: str, dst_name: str, src_act) -> Image.Image:
        assert len(src_act) == num_inference_steps, (
            f"Expected {num_inference_steps} per-step source acts at "
            f"{src_name}, got {len(src_act)}"
        )
        hook_fn = make_patch_hook_multi_step(
            dst_name, src_act, category, layout,
            text_token_indices=text_token_indices,
        )
        generator = torch.Generator(model.device).manual_seed(target_seed)
        pipe_kwargs = dict(
            prompt=target_prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            height=target_h,
            width=target_w,
        )
        if target_ref_image is not None:
            pipe_kwargs["image"] = target_ref_image
        return run_pipeline_with_hooks(
            model,
            [(dst_name, hook_fn)],
            callback_on_step_end=callback_on_step_end,
            **pipe_kwargs,
        )
    return producer


def make_input_to_block0_producer(
    model,
    *,
    target_seed: int,
    target_h: int,
    target_w: int,
    text_token_indices: Optional[Sequence[int]] = None,
    callback_on_step_end: Optional[Callable] = None,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
) -> ImageProducer:
    """Producer for input-to-block-0 sweeps.

    Hooks ``transformer.context_embedder`` output (the text stream fed to
    ``transformer_blocks.0`` as ``encoder_hidden_states``). Block 0 runs its
    own attention/FF pass on the patched input; blocks 1..31 see the
    downstream result. Distinct semantics from patching a block's output.

    ``num_inference_steps`` controls how many denoising steps the patched
    t2i runs. ``context_embedder`` is deterministic w.r.t. the prompt, so its
    output is identical every step in normal generation; the hook therefore
    fires every step and replaces with ``src_act`` every step, which is the
    multi-step-consistent behavior (text-stream input to block 0 stays
    constant across steps).

    Ignores ``dst_name`` passed by the sweep loop.
    """
    def producer(src_name: str, dst_name: str, src_act: torch.Tensor) -> Image.Image:
        hook_fn = make_context_embedder_patch_hook(
            src_act, text_token_indices=text_token_indices,
        )
        generator = torch.Generator(model.device).manual_seed(target_seed)
        return run_pipeline_with_hooks(
            model,
            [("context_embedder", hook_fn)],
            callback_on_step_end=callback_on_step_end,
            prompt="",
            generator=generator,
            num_inference_steps=num_inference_steps,
            height=target_h,
            width=target_w,
        )
    return producer


def sweep_and_grid(
    model,
    source_captured: Dict[str, list],
    category: str,
    save_dir: str,
    suptitle: str,
    *,
    source_layout: TokenLayout,
    bookend_images: List[Image.Image],
    bookend_labels: List[str],
    image_producer: ImageProducer,
    cat_subdir: Optional[str] = None,
    ncols: int = 6,
    fixed_dst_name: Optional[str] = None,
    fixed_dst_label: Optional[str] = None,
    block_slice: Optional[slice] = None,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
):
    """Sweep all blocks for one token category, creating a comparison grid.

    Args:
        model: A DiffusionModel instance.
        source_captured: ``{layer_name: [step_outputs...]}`` from
            ``model.capture_activations()``.
        category: Token category to patch (``"image"``, ``"text"``, or
            ``"ref"``).
        save_dir: Root save directory for this experiment.
        suptitle: Grid title (experiment description).
        source_layout: Token layout of the *source* generation; used to
            slice the captured activations into per-category tensors.
        bookend_images / bookend_labels: Images (and labels) to prepend to
            the grid.
        image_producer: Required callback
            ``(src_name, dst_name, src_act) -> Image`` that produces the
            image for each sweep iteration. Callers construct this via
            ``make_patch_pipeline_producer`` / ``make_input_to_block0_producer``
            (or a custom factory such as ``make_early_decode_producer``).
        cat_subdir: Subdirectory under ``save_dir``. Defaults to
            ``f"{category}_tokens"``.
        ncols: Grid columns.
        fixed_dst_name: When set, every sweep iteration targets this single
            destination name instead of the diagonal ``src == dst`` pattern.
        fixed_dst_label: Display label for ``fixed_dst_name``. Defaults to
            a compact block label or the raw name.
        block_slice: If set, sweep only ``ALL_BLOCK_NAMES[block_slice]``
            instead of all 32 blocks.
        num_inference_steps: When 1 (default), each layer's source activation
            is a single tensor (passed to single-step producers). When >1,
            each layer's source activation is a list of per-step tensors
            (passed to multi-step producers like
            :func:`make_patch_pipeline_producer_multi_step`).
    """
    assert image_producer is not None, "image_producer is required"
    cat_dir = os.path.join(save_dir, cat_subdir or f"{category}_tokens")
    os.makedirs(cat_dir, exist_ok=True)

    if num_inference_steps == 1:
        source_acts = extract_category_acts(source_captured, category, source_layout)
    else:
        source_acts = extract_category_acts_per_step(
            source_captured, category, source_layout,
        )

    block_names = ALL_BLOCK_NAMES if block_slice is None else ALL_BLOCK_NAMES[block_slice]
    block_labels = ALL_BLOCK_LABELS if block_slice is None else ALL_BLOCK_LABELS[block_slice]

    if fixed_dst_name is not None:
        dst_label = fixed_dst_label or _block_label_from_name(fixed_dst_name)
        sequence: List[Tuple[str, str, str, str]] = [
            (src_name, src_label, fixed_dst_name, dst_label)
            for src_name, src_label in zip(block_names, block_labels)
        ]
    else:
        sequence = [
            (name, label, name, label)
            for name, label in zip(block_names, block_labels)
        ]

    patched_images: List[Image.Image] = []
    for src_name, src_label, dst_name, dst_label in sequence:
        print(f"    [{src_label}->{dst_label}]")
        patched_img = image_producer(src_name, dst_name, source_acts[src_name])
        src_suffix = _block_suffix(src_name)
        dst_suffix = _block_suffix(dst_name)
        patched_img.save(
            os.path.join(cat_dir, f"patched_{src_suffix}_to_{dst_suffix}.png"),
        )
        patched_images.append(patched_img)
        torch.cuda.empty_cache()

    grid_images = list(bookend_images) + patched_images
    grid_labels = list(bookend_labels) + [
        src_label for _, src_label, _, _ in sequence
    ]
    grid_path = os.path.join(cat_dir, "grid.png")
    create_image_grid(
        grid_images, grid_labels, grid_path, ncols=ncols,
        suptitle=f"{suptitle}\nPatched category: {category} tokens",
    )
    print(f"    Saved {os.path.relpath(grid_path, save_dir)}")


def _block_suffix(block_name: str) -> str:
    """Convert a block name to a compact filename suffix."""
    return block_name.replace(
        "transformer_blocks.", "mm",
    ).replace(
        "single_transformer_blocks.", "single",
    )


def _block_label_from_name(block_name: str) -> str:
    """Human-readable label for a block name (used when fixed_dst is set)."""
    if block_name == "context_embedder":
        return "\u2192block 0 input"
    if block_name.startswith("transformer_blocks."):
        return f"MM {block_name.split('.')[-1]}"
    if block_name.startswith("single_transformer_blocks."):
        return f"Single {block_name.split('.')[-1]}"
    return block_name
