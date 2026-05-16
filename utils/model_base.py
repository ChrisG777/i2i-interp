"""Abstract base class for diffusion model wrappers.

Subclasses (Flux2KleinModel) supply the architecture metadata and the
``pipe`` / ``transformer`` / ``device`` triple; ``generate()`` is shared.
"""

import torch
from PIL import Image


class DiffusionModel:
    """Unified interface for diffusion model wrappers.

    Subclasses must set the architecture properties and ``pipe`` /
    ``transformer`` / ``device`` in their ``__init__``.  The generation
    methods are shared — only loading differs between models.
    """

    # -- Set by subclass __init__ ------------------------------------------
    pipe: object               # diffusers pipeline (Flux2KleinPipeline, …)
    transformer: torch.nn.Module
    device: str

    # Architecture metadata (read by HookBuilder)
    name: str                  # "flux2_klein"
    num_heads: int             # attention heads per block
    head_dim: int              # dimension per head
    inner_dim: int             # num_heads * head_dim
    text_seq_len: int          # text token count in the fused sequence
    has_bias: bool             # whether Linear layers have bias
    has_fused_single_qkv: bool # True if single blocks use a fused to_qkv_mlp_proj

    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        seed: int,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 0.0,
        **kwargs,
    ) -> Image.Image:
        """Generate a single image with a deterministic seed."""
        generator = torch.Generator(self.device).manual_seed(seed)
        output = self.pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            **kwargs,
        )
        return output.images[0]
