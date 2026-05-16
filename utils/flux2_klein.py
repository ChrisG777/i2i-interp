"""Flux 2 Klein model wrapper with activation-capture utilities.

Provides a simple interface for loading the FLUX.2-klein-9B model,
enumerating layers, and generating images while capturing per-block
activations via baukit ``TraceDict``.
"""

import math
import torch
from dataclasses import dataclass
from diffusers import Flux2KleinPipeline
from baukit.nethook import TraceDict
from typing import List, Tuple, Dict
from PIL import Image

from utils.model_base import DiffusionModel


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


@dataclass(frozen=True)
class TokenLayout:
    """Per-task sequence-length layout for the joint attention sequence.

    The transformer's joint stream is ``[text | noise | ref]`` (i2i) or
    ``[text | noise]`` (t2i, ``ref_seq_len == 0``). All slicing in the
    knockout/patching stack reads token counts from a layout instance so
    different tasks can run at different resolutions without touching the
    module-level constants.
    """

    text_seq_len: int
    noise_seq_len: int
    ref_seq_len: int  # 0 for t2i

    def __post_init__(self) -> None:
        assert self.text_seq_len > 0, f"text_seq_len must be positive, got {self.text_seq_len}"
        assert self.noise_seq_len > 0, f"noise_seq_len must be positive, got {self.noise_seq_len}"
        assert self.ref_seq_len >= 0, f"ref_seq_len must be >= 0, got {self.ref_seq_len}"

    @property
    def has_ref(self) -> bool:
        return self.ref_seq_len > 0

    @property
    def total_t2i(self) -> int:
        return self.text_seq_len + self.noise_seq_len

    @property
    def total_i2i(self) -> int:
        return self.text_seq_len + self.noise_seq_len + self.ref_seq_len

    @property
    def total(self) -> int:
        """Joint-stream length: includes ref tokens iff ``has_ref``."""
        return self.total_i2i if self.has_ref else self.total_t2i


def _seq_len_for_image(h: int, w: int) -> int:
    """Token count for an image of size ``(h, w)`` after the VAE+patchify path."""
    assert h > 0 and w > 0, f"image size must be positive, got ({h}, {w})"
    assert h % VAE_PATCH == 0 and w % VAE_PATCH == 0, (
        f"image size ({h}, {w}) must be a multiple of VAE_PATCH={VAE_PATCH}"
    )
    return (h // VAE_PATCH) * (w // VAE_PATCH)


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


def get_category_slices(layout: TokenLayout) -> dict[str, slice]:
    """Return slices for text, image (noise), and (for i2i) ref categories.

    Operates on the joint stream ``[text | noise | ref]``. For t2i layouts
    (``ref_seq_len == 0``) the ``"ref"`` key is omitted. Applies to both MM
    and single blocks.
    """
    txt_end = layout.text_seq_len
    img_end = txt_end + layout.noise_seq_len
    if layout.ref_seq_len == 0:
        return {
            "text": slice(0, txt_end),
            "image": slice(txt_end, img_end),
        }
    return {
        "text": slice(0, txt_end),
        "image": slice(txt_end, img_end),
        "ref": slice(img_end, img_end + layout.ref_seq_len),
    }


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
        """Generate an image while capturing activations at specified layers.

        Args:
            captures_to_cpu: If True, move captured tensors to CPU inside the
                hook to conserve GPU memory when capturing many layers.

        Returns:
            Tuple of (image, activations_dict) where activations_dict maps
            layer names to lists of captured outputs (one per forward pass).
        """
        captured: Dict[str, list] = {name: [] for name in capture_layers}

        def capture_fn(output, layer):
            if isinstance(output, tuple):
                tensors = tuple(o.detach().cpu().clone() if captures_to_cpu else o.detach().clone() for o in output)
                captured[layer].append(tensors)
            else:
                t = output.detach().cpu().clone() if captures_to_cpu else output.detach().clone()
                captured[layer].append(t)
            return output

        generator = torch.Generator(self.device).manual_seed(seed)

        with TraceDict(
            self.transformer,
            layers=capture_layers,
            edit_output=capture_fn,
        ):
            output = self.pipe(
                prompt=prompt,
                generator=generator,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                **kwargs,
            )

        return output.images[0], captured
