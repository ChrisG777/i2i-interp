"""Unified hook factories for activation patching experiments.

Supports patching any token category (image, text, ref) into either i2i or
t2i target runs. Per-task token counts come from a :class:`TokenLayout` that
the caller threads through.

Token layouts (driven by ``layout.text_seq_len``/``noise_seq_len``/
``ref_seq_len``):

* MM blocks return ``(txt_stream, img_stream)``:
    - i2i: ``img_stream = [noise | ref]``.
    - t2i: ``img_stream = [noise]`` (``layout.has_ref`` is False).
* Single blocks return ``[text | noise | ref]`` (i2i) or ``[text | noise]``
  (t2i), possibly wrapped in a tuple.
"""

from typing import Dict, Optional, Sequence

import torch

from utils.flux2_klein import TokenLayout


def make_mm_patch_hook(
    name: str,
    source_act: torch.Tensor,
    category: str,
    layout: TokenLayout,
    text_token_indices: Optional[Sequence[int]] = None,
):
    """Hook for MM blocks: replace one token category in (txt, img) output.

    Args:
        name: Block name (for error messages).
        source_act: Source activation tensor to patch in.
        category: Which token category to patch (``"image"``, ``"text"``,
            or ``"ref"``).
        layout: Per-task token layout. ``layout.has_ref`` selects the i2i
            (txt, [noise|ref]) shape vs. the t2i (txt, [noise]) shape.
        text_token_indices: When ``category == "text"`` and this is provided,
            replace only those text-token positions instead of the full
            text-token slice. Must be ``None`` for non-text categories.
    """
    assert category in ("image", "text", "ref")
    if text_token_indices is not None:
        assert category == "text", (
            f"text_token_indices only valid for category='text', got {category!r}"
        )
    if category == "ref":
        assert layout.has_ref, (
            f"Cannot patch ref tokens into t2i target at {name} "
            f"(layout.ref_seq_len=0)"
        )
    noise_len = layout.noise_seq_len
    has_ref = layout.has_ref

    def hook(module, input, output):
        txt_stream, img_stream = output

        if category == "text":
            src = source_act.to(dtype=txt_stream.dtype, device=txt_stream.device)
            if text_token_indices is None:
                return (src, img_stream)
            patched = txt_stream.clone()
            patched[:, text_token_indices, :] = src[:, text_token_indices, :]
            return (patched, img_stream)

        if category == "image":
            src = source_act.to(dtype=img_stream.dtype, device=img_stream.device)
            if has_ref:
                patched = img_stream.clone()
                patched[:, :noise_len, :] = src
                return (txt_stream, patched)
            assert img_stream.shape[1] == noise_len, (
                f"Expected {noise_len} image tokens in t2i at {name}, "
                f"got {img_stream.shape[1]}"
            )
            return (txt_stream, src)

        # category == "ref"
        patched = img_stream.clone()
        src = source_act.to(dtype=img_stream.dtype, device=img_stream.device)
        patched[:, noise_len:, :] = src
        return (txt_stream, patched)

    return hook


def make_single_patch_hook(
    name: str,
    source_act: torch.Tensor,
    category: str,
    layout: TokenLayout,
    text_token_indices: Optional[Sequence[int]] = None,
):
    """Hook for single blocks: replace one token category in [text|noise|ref].

    Args:
        name: Block name (for error messages).
        source_act: Source activation tensor to patch in.
        category: Which token category to patch (``"image"``, ``"text"``,
            or ``"ref"``).
        layout: Per-task token layout supplying the slice boundaries.
        text_token_indices: When ``category == "text"`` and this is provided,
            replace only those text-token positions instead of the full
            text-token slice. Must be ``None`` for non-text categories.
    """
    assert category in ("image", "text", "ref")
    if text_token_indices is not None:
        assert category == "text", (
            f"text_token_indices only valid for category='text', got {category!r}"
        )
    if category == "ref":
        assert layout.has_ref, (
            f"Cannot patch ref tokens into t2i target at {name} "
            f"(layout.ref_seq_len=0)"
        )
    text_end = layout.text_seq_len
    image_end = text_end + layout.noise_seq_len
    expected_t2i_len = layout.total_t2i

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output

        if not layout.has_ref:
            assert hidden.shape[1] == expected_t2i_len, (
                f"Expected {expected_t2i_len} tokens in t2i at "
                f"{name}, got {hidden.shape[1]}"
            )

        patched = hidden.clone()
        src = source_act.to(dtype=hidden.dtype, device=hidden.device)

        if category == "image":
            patched[:, text_end:image_end, :] = src
        elif category == "text":
            if text_token_indices is None:
                patched[:, :text_end, :] = src
            else:
                patched[:, text_token_indices, :] = src[:, text_token_indices, :]
        else:  # category == "ref"
            patched[:, image_end:, :] = src

        return (patched,) if isinstance(output, tuple) else patched

    return hook


def make_patch_hook(
    layer_name: str,
    source_act: torch.Tensor,
    category: str,
    layout: TokenLayout,
    text_token_indices: Optional[Sequence[int]] = None,
):
    """Dispatch to MM or single hook factory based on layer name prefix."""
    if layer_name.startswith("transformer_blocks."):
        return make_mm_patch_hook(
            layer_name, source_act, category, layout,
            text_token_indices=text_token_indices,
        )
    return make_single_patch_hook(
        layer_name, source_act, category, layout,
        text_token_indices=text_token_indices,
    )


def make_patch_hook_multi_step(
    layer_name: str,
    source_acts_per_step: Sequence[torch.Tensor],
    category: str,
    layout: TokenLayout,
    text_token_indices: Optional[Sequence[int]] = None,
):
    """Multi-step variant of :func:`make_patch_hook`.

    Holds one single-step patch hook per denoising step (built up-front from
    ``source_acts_per_step``) and dispatches by an internal step counter that
    advances on every call. Use when the target pipeline runs ``N`` inference
    steps and you want to patch ``source_acts_per_step[k]`` at step ``k``.

    Asserts the hook fires no more than ``len(source_acts_per_step)`` times
    so a step-count mismatch fails fast rather than silently reusing the
    last activation.
    """
    per_step_hooks = [
        make_patch_hook(
            layer_name, src_act, category, layout,
            text_token_indices=text_token_indices,
        )
        for src_act in source_acts_per_step
    ]
    n = len(per_step_hooks)
    assert n >= 1, f"source_acts_per_step is empty for {layer_name}"
    counter = [0]

    def hook(module, input, output):
        assert counter[0] < n, (
            f"hook on {layer_name} fired {counter[0] + 1} times but only {n} "
            f"per-step source activations were provided"
        )
        out = per_step_hooks[counter[0]](module, input, output)
        counter[0] += 1
        return out

    return hook


def make_context_embedder_patch_hook(
    source_act: torch.Tensor,
    text_token_indices: Optional[Sequence[int]] = None,
):
    """Forward hook on ``transformer.context_embedder``: replace (part of) its
    output text stream.

    ``context_embedder`` is the ``nn.Linear`` that produces the 512-token text
    stream fed as ``encoder_hidden_states`` to ``transformer_blocks.0``. Its
    output flows unmodified into block 0 (no intervening layernorm/projection;
    RoPE is computed on token IDs, not embeddings). Replacing the output here
    puts ``source_act`` upstream of block 0's own attention/FF computation,
    which is what we want for input-to-block-0 sweeps (distinct semantics
    from a forward hook on block 0, which would overwrite its *output*).

    Args:
        source_act: Source text-stream tensor ``[1, TEXT_SEQ_LEN, inner_dim]``.
        text_token_indices: If provided, replace only those text-token
            positions instead of the full 512-token slice.
    """
    def hook(module, input, output):
        src = source_act.to(dtype=output.dtype, device=output.device)
        if text_token_indices is None:
            assert output.shape == src.shape, (
                f"context_embedder output shape {tuple(output.shape)} does "
                f"not match source shape {tuple(src.shape)}"
            )
            return src
        patched = output.clone()
        patched[:, text_token_indices, :] = src[:, text_token_indices, :]
        return patched
    return hook


# ---------------------------------------------------------------------------
# Additive variants: ADD a delta to one token category's slice.
#
# Same dispatch pattern as the replacement hooks above, but the output is
# ``output + delta`` on the category slice rather than ``output := source``.
# ---------------------------------------------------------------------------


def make_mm_add_hook(
    name: str,
    delta: torch.Tensor,
    category: str,
    layout: TokenLayout,
):
    """Hook for MM blocks: add ``delta`` to one token category in the output.

    Same category slicing as ``make_mm_patch_hook``; differs only in that
    ``delta`` is added to the existing tokens rather than replacing them.
    """
    assert category in ("image", "text", "ref")
    if category == "ref":
        assert layout.has_ref, (
            f"Cannot add to ref tokens in t2i target at {name} "
            f"(layout.ref_seq_len=0)"
        )
    noise_len = layout.noise_seq_len
    has_ref = layout.has_ref

    def hook(module, input, output):
        txt_stream, img_stream = output
        d_dtype, d_device = txt_stream.dtype, txt_stream.device

        if category == "text":
            d = delta.to(dtype=d_dtype, device=d_device)
            return (txt_stream + d, img_stream)

        if category == "image":
            d = delta.to(dtype=img_stream.dtype, device=img_stream.device)
            if has_ref:
                patched = img_stream.clone()
                patched[:, :noise_len, :] = patched[:, :noise_len, :] + d
            else:
                assert img_stream.shape[1] == noise_len, (
                    f"Expected {noise_len} image tokens in t2i at {name}, "
                    f"got {img_stream.shape[1]}"
                )
                patched = img_stream + d
            return (txt_stream, patched)

        # category == "ref"
        d = delta.to(dtype=img_stream.dtype, device=img_stream.device)
        patched = img_stream.clone()
        patched[:, noise_len:, :] = patched[:, noise_len:, :] + d
        return (txt_stream, patched)

    return hook


def make_single_add_hook(
    name: str,
    delta: torch.Tensor,
    category: str,
    layout: TokenLayout,
):
    """Hook for single blocks: add ``delta`` to one token category in the output.

    Same category slicing as ``make_single_patch_hook``; differs only in that
    ``delta`` is added to the existing tokens rather than replacing them.
    """
    assert category in ("image", "text", "ref")
    if category == "ref":
        assert layout.has_ref, (
            f"Cannot add to ref tokens in t2i target at {name} "
            f"(layout.ref_seq_len=0)"
        )
    text_end = layout.text_seq_len
    image_end = text_end + layout.noise_seq_len
    expected_t2i_len = layout.total_t2i

    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output

        if not layout.has_ref:
            assert hidden.shape[1] == expected_t2i_len, (
                f"Expected {expected_t2i_len} tokens in t2i at "
                f"{name}, got {hidden.shape[1]}"
            )

        patched = hidden.clone()
        d = delta.to(dtype=hidden.dtype, device=hidden.device)

        if category == "image":
            patched[:, text_end:image_end, :] = patched[:, text_end:image_end, :] + d
        elif category == "text":
            patched[:, :text_end, :] = patched[:, :text_end, :] + d
        else:  # category == "ref"
            patched[:, image_end:, :] = patched[:, image_end:, :] + d

        return (patched,) if is_tuple else patched

    return hook


def make_add_hook(
    layer_name: str,
    delta: torch.Tensor,
    category: str,
    layout: TokenLayout,
):
    """Dispatch to MM or single add-hook factory based on layer name prefix."""
    if layer_name.startswith("transformer_blocks."):
        return make_mm_add_hook(layer_name, delta, category, layout)
    return make_single_add_hook(layer_name, delta, category, layout)
