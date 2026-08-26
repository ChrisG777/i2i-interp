"""Flux 2 Klein model wrapper with activation-capture utilities.

Provides a simple interface for loading the FLUX.2-klein-9B model,
enumerating layers, and generating images while capturing per-block
activations via baukit ``TraceDict``.
"""

import math
import torch
from diffusers import Flux2KleinPipeline
from typing import List, Tuple, Dict
from PIL import Image

from utils.model_base import DiffusionModel
from utils.token_layout import TokenLayout, get_category_slices, seq_len_for_image

# Re-exported for the many existing importers of these names from this module.
__all__ = [
    "TokenLayout", "get_category_slices", "layout_for", "effective_ref_dims",
    "Flux2KleinModel", "ALL_BLOCK_NAMES", "ALL_BLOCK_LABELS", "block_suffix",
    "block_index_from_suffix", "MODEL_ID", "NUM_MM_BLOCKS", "NUM_SINGLE_BLOCKS",
    "TEXT_SEQ_LEN", "NOISE_SEQ_LEN", "TOTAL_SEQ_LEN", "TOTAL_SEQ_LEN_T2I",
    "VAE_PATCH",
]


MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

# Architecture constants (importable by experiment scripts).
NUM_MM_BLOCKS = 8
NUM_SINGLE_BLOCKS = 24

# Sequence-length defaults. ``TEXT_SEQ_LEN`` is the Qwen3 max — a model
# parameter that doesn't change with image size. ``NOISE_SEQ_LEN`` is the
# value at 1024x1024 specifically; production code derives the actual value
# from each task's image size via :func:`layout_for`.
TEXT_SEQ_LEN = 512
NOISE_SEQ_LEN = 4096  # (1024 / vae_scale_factor=8 / patch_size=2)^2 at 1024x1024
# t2i joint-attention sequence at 1024x1024: [text | noise].
TOTAL_SEQ_LEN_T2I = TEXT_SEQ_LEN + NOISE_SEQ_LEN
# i2i joint-attention sequence at 1024x1024: [text | noise | ref]. Ref is the
# same length as noise (one patch grid per image).
TOTAL_SEQ_LEN = TEXT_SEQ_LEN + NOISE_SEQ_LEN + NOISE_SEQ_LEN

# Patch size of the VAE+patchify path: noise_seq_len = (h / VAE_PATCH) * (w / VAE_PATCH).
VAE_PATCH = 16

ALL_BLOCK_NAMES = (
    [f"transformer_blocks.{i}" for i in range(NUM_MM_BLOCKS)]
    + [f"single_transformer_blocks.{i}" for i in range(NUM_SINGLE_BLOCKS)]
)

ALL_BLOCK_LABELS = (
    [f"MM {i}" for i in range(NUM_MM_BLOCKS)]
    + [f"Single {i}" for i in range(NUM_SINGLE_BLOCKS)]
)


def block_suffix(block_name: str) -> str:
    """Compact filename suffix for a block name: ``transformer_blocks.7`` ->
    ``mm7``, ``single_transformer_blocks.9`` -> ``single_mm9``. Single home for
    the convention used in sweep filenames, the i2i / i2i_unc runners, the
    layer-sweep judge, and the layer-sweep plot.

    NB: ``transformer_blocks.`` is a substring of ``single_transformer_blocks.``,
    so the single-block names become ``single_mm<i>`` (not ``single<i>``). This
    is the *established on-disk convention* — persisted i2i_to_unconditional
    artifacts and the hardcoded judge bundles (``patched_single_mm9.png``)
    depend on it, so do not "tidy" it without renaming every produced file."""
    return block_name.replace(
        "transformer_blocks.", "mm",
    ).replace("single_transformer_blocks.", "single")


_SUFFIX_TO_INDEX = {block_suffix(n): i for i, n in enumerate(ALL_BLOCK_NAMES)}


def block_index_from_suffix(suffix: str) -> int:
    """Inverse of :func:`block_suffix`: ``mm7`` -> 7, ``single_mm9`` -> 17 (the
    global sweep index across all 32 blocks)."""
    assert suffix in _SUFFIX_TO_INDEX, (
        f"unknown block suffix {suffix!r}; known: {sorted(_SUFFIX_TO_INDEX)}"
    )
    return _SUFFIX_TO_INDEX[suffix]


def _seq_len_for_image(h: int, w: int) -> int:
    """Token count for an image of size ``(h, w)`` after the VAE+patchify path."""
    return seq_len_for_image(h, w, patch=VAE_PATCH)


# Maximum reference-image area accepted by Flux2KleinPipeline before it
# silently downscales. Mirrors the ``1024 * 1024`` constant used in
# ``Flux2KleinPipeline.__call__`` (process-images step).
_REF_MAX_AREA = 1024 * 1024


def effective_ref_dims(ref_h: int, ref_w: int) -> tuple[int, int]:
    """Apply the pipeline's reference-image preprocessing to ``(h, w)``.

    ``Flux2KleinPipeline`` resizes any reference whose area exceeds
    ``1024 * 1024`` (lanczos, preserving aspect ratio) and then floors both
    dims to a multiple of ``VAE_PATCH``. Layout math has to mirror that, or
    the joint-attention mask we build will not match the actual sequence
    length the model sees.
    """
    assert ref_h > 0 and ref_w > 0, f"ref dims must be positive, got ({ref_h}, {ref_w})"
    if ref_h * ref_w > _REF_MAX_AREA:
        scale = math.sqrt(_REF_MAX_AREA / (ref_w * ref_h))
        # Match Flux2ImageProcessor._resize_to_target_area: int() on (w*scale, h*scale).
        ref_w = int(ref_w * scale)
        ref_h = int(ref_h * scale)
    ref_w = (ref_w // VAE_PATCH) * VAE_PATCH
    ref_h = (ref_h // VAE_PATCH) * VAE_PATCH
    return ref_h, ref_w


def layout_for(
    target_h: int,
    target_w: int,
    *,
    ref_h: int | None = None,
    ref_w: int | None = None,
    text_seq_len: int = TEXT_SEQ_LEN,
) -> TokenLayout:
    """Build a :class:`TokenLayout` from raw image sizes.

    Pass both ``ref_h`` and ``ref_w`` for an i2i layout; pass neither for t2i
    (``ref_seq_len = 0``). Ref dims are normalized via
    :func:`effective_ref_dims` to mirror the pipeline's pre-encode resize.
    """
    noise = _seq_len_for_image(target_h, target_w)
    if ref_h is None and ref_w is None:
        ref = 0
    else:
        assert ref_h is not None and ref_w is not None, (
            "ref_h and ref_w must both be set or both be None"
        )
        eff_h, eff_w = effective_ref_dims(ref_h, ref_w)
        ref = _seq_len_for_image(eff_h, eff_w)
    return TokenLayout(text_seq_len, noise, ref)


class Flux2KleinModel(DiffusionModel):
    """Wrapper for Flux2KleinPipeline with baukit hook-based interventions."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        torch_dtype=torch.bfloat16,
        device: str = "cuda:0",
    ):
        self.pipe = Flux2KleinPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype
        )
        self.pipe.to(device)
        self.device = device
        self.transformer = self.pipe.transformer
        self.pipe.set_progress_bar_config(disable=True)

        # Architecture metadata
        self.name = "flux2_klein"
        self.num_heads = self.transformer.config.num_attention_heads  # 32
        self.head_dim = 128
        self.inner_dim = self.num_heads * self.head_dim  # 4096
        self.text_seq_len = 512
        self.has_bias = False
        self.has_fused_single_qkv = True

    def capture_activations(
        self,
        prompt: str,
        seed: int,
        capture_layers: List[str],
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 1.0,
        captures_to_cpu: bool = False,
        **kwargs,
    ) -> Tuple[Image.Image, Dict[str, list]]:
        """Klein capture: base implementation with this model's legacy
        generation defaults (notably ``guidance_scale=1.0``, which differs
        from ``generate()``'s 0.0 default and is what all existing capture
        call sites rely on)."""
        return super().capture_activations(
            prompt, seed, capture_layers,
            captures_to_cpu=captures_to_cpu,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            **kwargs,
        )
