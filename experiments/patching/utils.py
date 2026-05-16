"""Shared utilities for activation patching experiments.

Provides common helpers for registering forward hooks on transformer blocks,
running the pipeline, and extracting token-category-specific activations
from captured block outputs.
"""

from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image

from utils.flux2_klein import TEXT_SEQ_LEN, TokenLayout


# ---------------------------------------------------------------------------
# Block resolution
# ---------------------------------------------------------------------------


def resolve_block(model, layer_name: str) -> torch.nn.Module:
    """Walk a dotted layer name to get the actual nn.Module.

    Example: ``resolve_block(model, "transformer_blocks.3")`` returns
    ``model.transformer.transformer_blocks[3]``.
    """
    block = model.transformer
    for part in layer_name.split("."):
        block = getattr(block, part) if not part.isdigit() else block[int(part)]
    return block


# ---------------------------------------------------------------------------
# Pipeline execution with hooks
# ---------------------------------------------------------------------------


def run_pipeline_with_hooks(
    model,
    block_hook_pairs: List[Tuple[str, Callable]],
    callback_on_step_end: Optional[Callable] = None,
    **pipe_kwargs,
) -> Image.Image:
    """Run the pipeline with forward hooks registered on transformer blocks.

    Registers all hooks, calls ``model.pipe(**pipe_kwargs)``, removes all
    hooks, and returns the generated image.

    Args:
        model: A DiffusionModel instance (e.g. Flux2KleinModel).
        block_hook_pairs: List of ``(layer_name, hook_fn)`` tuples.  Each
            hook is registered via ``register_forward_hook`` on the resolved
            block.
        callback_on_step_end: Optional pipeline callback applied after each
            scheduler step (used for noise correction in vary-noise setting).
        **pipe_kwargs: Keyword arguments forwarded to ``model.pipe()``
            (must include ``prompt``, ``generator``, ``num_inference_steps``,
            and optionally ``image`` for i2i).
    """
    handles = []
    for layer_name, hook_fn in block_hook_pairs:
        block = resolve_block(model, layer_name)
        handles.append(block.register_forward_hook(hook_fn))

    if callback_on_step_end is not None:
        pipe_kwargs["callback_on_step_end"] = callback_on_step_end
        pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    output = model.pipe(**pipe_kwargs)

    for h in handles:
        h.remove()

    return output.images[0]


# ---------------------------------------------------------------------------
# Token category extraction
# ---------------------------------------------------------------------------


def _slice_category_from_step(
    output,
    category: str,
    layout: TokenLayout,
    layer_name: str,
) -> torch.Tensor:
    """Slice one captured per-step output to the requested token category.

    MM blocks return ``(txt_stream, img_stream)`` where
    ``img_stream = [noise | ref]`` (i2i) or just ``[noise]`` (t2i, ref_seq_len=0).
    Single blocks return ``[text | noise | ref]`` (possibly tuple-wrapped).
    """
    text_end = layout.text_seq_len
    noise_len = layout.noise_seq_len
    flat_image_end = text_end + noise_len

    if layer_name.startswith("transformer_blocks."):
        assert isinstance(output, tuple) and len(output) == 2, (
            f"Expected tuple of (txt, img) for {layer_name}, got {type(output)}"
        )
        txt_stream, img_stream = output
        if category == "image":
            return img_stream[:, :noise_len, :]
        if category == "text":
            return txt_stream
        return img_stream[:, noise_len:, :]  # ref

    hidden = output[0] if isinstance(output, tuple) else output
    if category == "image":
        return hidden[:, text_end:flat_image_end, :]
    if category == "text":
        return hidden[:, :text_end, :]
    return hidden[:, flat_image_end:, :]  # ref


def extract_category_acts(
    captured: Dict[str, list],
    category: str,
    layout: TokenLayout,
) -> Dict[str, torch.Tensor]:
    """Extract a specific token category from single-step captured block outputs.

    ``captured`` is the dict returned by ``model.capture_activations()``,
    mapping layer names to lists of per-step outputs. This function asserts
    exactly one step (1-step generation); use
    :func:`extract_category_acts_per_step` for multi-step captures.

    Args:
        captured: ``{layer_name: [step_outputs...]}`` from capture_activations.
        category: One of ``"image"``, ``"text"``, or ``"ref"``.
        layout: Per-task token layout supplying the slice boundaries.

    Returns:
        ``{layer_name: tensor}`` with the requested token slice from each block.
    """
    assert category in ("image", "text", "ref"), (
        f"Unknown category {category!r}, expected 'image', 'text', or 'ref'"
    )
    if category == "ref":
        assert layout.has_ref, (
            "Cannot extract 'ref' tokens from a t2i layout (ref_seq_len=0)"
        )

    acts: Dict[str, torch.Tensor] = {}
    for layer_name, step_outputs in captured.items():
        assert len(step_outputs) == 1, (
            f"Expected 1 step output for {layer_name}, got {len(step_outputs)}"
        )
        acts[layer_name] = _slice_category_from_step(
            step_outputs[0], category, layout, layer_name,
        )
    return acts


def extract_category_acts_per_step(
    captured: Dict[str, list],
    category: str,
    layout: TokenLayout,
) -> Dict[str, list]:
    """Per-step variant of :func:`extract_category_acts` for multi-step capture.

    Returns ``{layer_name: [step0_tensor, step1_tensor, ...]}`` — one tensor
    per denoising step, in step order. Intended for multi-step patching where
    a per-step source activation is patched into the corresponding target step.
    """
    assert category in ("image", "text", "ref"), (
        f"Unknown category {category!r}, expected 'image', 'text', or 'ref'"
    )
    if category == "ref":
        assert layout.has_ref, (
            "Cannot extract 'ref' tokens from a t2i layout (ref_seq_len=0)"
        )

    acts: Dict[str, list] = {}
    for layer_name, step_outputs in captured.items():
        assert len(step_outputs) >= 1, (
            f"Expected >=1 step output for {layer_name}, got {len(step_outputs)}"
        )
        acts[layer_name] = [
            _slice_category_from_step(out, category, layout, layer_name)
            for out in step_outputs
        ]
    return acts


# ---------------------------------------------------------------------------
# Content-token resolution (for per-text-token patching)
# ---------------------------------------------------------------------------


def resolve_content_token_indices(
    pipe,
    instruction_prompt: str,
) -> List[Tuple[int, str]]:
    """Return ``[(text_token_index, token_string), ...]`` for the instruction's
    content tokens within the 512-token Qwen3 sequence.

    Uses the same chat-template + tokenization path the model sees, and locates
    ``instruction_prompt`` as a contiguous span via ``find_object_text_indices``
    (which is robust to BPE re-merging between isolated and in-context
    tokenization). Returns one ``(index, string)`` tuple per content token in
    the instruction, preserving token order. Fails fast if the instruction is
    empty or cannot be located unambiguously.
    """
    assert instruction_prompt.strip(), (
        f"Instruction prompt must be non-empty to resolve content tokens, "
        f"got {instruction_prompt!r}"
    )
    token_strings = get_token_strings(pipe, instruction_prompt)
    indices = find_object_text_indices(pipe, instruction_prompt, instruction_prompt)
    return [(int(i), token_strings[int(i)]) for i in indices]


def get_token_strings(pipe, prompt: str, max_length: int = TEXT_SEQ_LEN) -> list[str]:
    """Tokenize ``prompt`` and return per-position token strings.

    Non-content positions are marked as ``"<pad>"``. Uses the same chat-template
    path the diffusers Flux2 pipeline uses internally, so the indices match the
    text-token positions in the model's attention.
    """
    messages = [{"role": "user", "content": prompt}]
    text = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = pipe.tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tokens: list[str] = []
    for tok_id, mask in zip(inputs["input_ids"][0], inputs["attention_mask"][0]):
        if mask.item() == 0:
            tokens.append("<pad>")
        else:
            tokens.append(pipe.tokenizer.decode(tok_id.item()))
    return tokens


def find_object_text_indices(
    pipe,
    prompt: str,
    object_phrase: str,
) -> torch.Tensor:
    """Locate the text-token indices of ``object_phrase`` within ``prompt``.

    Tokenises the prompt the same way the model sees it (via the chat-template
    path used by :func:`get_token_strings`) and searches for a contiguous span
    of prompt tokens whose concatenation equals ``object_phrase`` (modulo
    whitespace). Comparing concatenated strings — rather than per-subword —
    makes this robust to BPE position-dependent merges: e.g. standalone
    ``"television"`` splits as ``["te", "levision"]`` while mid-sentence it
    becomes a single ``" television"`` token, and both cases resolve to the
    same span.

    Fails fast if zero matches or more than one match is found.
    """
    prompt_tokens = get_token_strings(pipe, prompt)  # length TEXT_SEQ_LEN

    def _norm(s: str) -> str:
        return s.strip()

    target = _norm(object_phrase).replace(" ", "")
    assert target, f"Empty object phrase {object_phrase!r}"

    matches: list[tuple[int, int]] = []  # (start, length)
    n = len(prompt_tokens)
    for start in range(n):
        if not _norm(prompt_tokens[start]):
            continue
        concat = ""
        for end in range(start, n):
            concat += _norm(prompt_tokens[end]).replace(" ", "")
            if len(concat) > len(target):
                break
            if concat == target:
                matches.append((start, end - start + 1))
                break

    real_tokens = [t for t in prompt_tokens if t != "<pad>"]
    if len(matches) == 0:
        raise ValueError(
            f"Object phrase {object_phrase!r} not found in prompt tokens.\n"
            f"  Target (concatenated, whitespace-stripped): {target!r}\n"
            f"  Prompt tokens (non-pad): {real_tokens}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Object phrase {object_phrase!r} matched {len(matches)} times in "
            f"prompt at spans {matches}; attribution is ambiguous.\n"
            f"  Prompt tokens (non-pad): {real_tokens}"
        )

    start, length = matches[0]
    return torch.arange(start, start + length, dtype=torch.long)
